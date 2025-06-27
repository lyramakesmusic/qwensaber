"""
Beat Saber dataset and vocabulary handling.

We use the first 14,522 tokens of Qwen3's vocabulary for our data:
- 0-12,287: Time ticks (48 PPQ resolution)
- 12,288-12,467: Note types (180 combinations)
- 12,468-14,515: Audio tokens (2 x 1,024 Encodec codebooks)
- 14,516: BOS (beginning)
- 14,517: EOS (end)
- 14,518: PAD (padding)
- 14,519-14,521: Structure separators
"""

import torch
import numpy as np
import pickle
import hashlib
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Dict

from audio import create_audio_tokenizer

# Constants
PPQ = 48  # Pulses per quarter note
VOCAB_SIZE = 14522  # Total tokens we use from Qwen3's vocabulary (0-14521)

# Token ranges
TIME_RANGE = (0, 12288)      # 12,288 time ticks
NOTE_RANGE = (12288, 12468)  # 180 note types
AUDIO_RANGE = (12468, 14516) # 2,048 audio tokens (2 codebooks × 1,024 values)

# Special tokens packed right after audio range
BOS = 14516        # Beginning of sequence
EOS = 14517        # End of sequence
PAD = 14518        # Padding
AUDIO_START = 14519  # Start of audio tokens
AUDIO_END = 14520    # End of audio tokens
NOTES_START = 14521  # Start of note tokens


# encoding / decoding data
# -------------------------------------------------------------------

def time_to_token(beat_time):

    """Convert beat time to time token."""

    tick = int(round(beat_time * PPQ))
    return min(tick, TIME_RANGE[1] - 1)

def audio_tokens_to_sequence(audio_tokens):

    """Convert Encodec audio tokens to our token sequence."""

    # audio_tokens = (frames, 2) - frames of 2 codebooks, values 0-1023
    sequence = []
    n_codebooks = audio_tokens.shape[1]
    values_per_codebook = (AUDIO_RANGE[1] - AUDIO_RANGE[0]) // n_codebooks
    
    for frame in audio_tokens:
        for slot, value in enumerate(frame):
            # Scale 0-1023 to 0-(values_per_codebook-1) without losing 1023
            scaled_value = int(round(value * (values_per_codebook - 1) / 1023))
            scaled_value = min(scaled_value, values_per_codebook - 1)  # Safety clamp
            token = AUDIO_RANGE[0] + slot * values_per_codebook + scaled_value
            sequence.append(int(token))
    
    # Returns: List[int] - flattened tokens in range 12468-14515
    return sequence

def create_training_sequence(audio_tokens, notes):

    """Create a training sequence from audio tokens and notes.
    [BOS, AUDIO_START, audio_tokens, AUDIO_END, NOTES_START, time_token, note_token, ... EOS]"""

    # audio_tokens = (frames, 2) - encodec output
    # notes = [(beat_time, note_type), ...] - beat times and note types (0-179)
    
    sequence = [BOS, AUDIO_START]
    
    audio_seq = audio_tokens_to_sequence(audio_tokens)
    sequence.extend(audio_seq)
    
    sequence.extend([AUDIO_END, NOTES_START])

    note_tokens_created = []
    for beat_time, note_type in notes:
        time_token = time_to_token(beat_time)
        note_token = NOTE_RANGE[0] + int(note_type)
        sequence.extend([time_token, note_token])
        note_tokens_created.extend([time_token, note_token])
    
    sequence.append(EOS)
    
    # Returns: List[int] - complete sequence ready for training
    return sequence


# collator
# -------------------------------------------------------------------

def audio_masking_collator(batch):

    """Custom collator for audio masking."""

    # batch = [{"input_ids": tensor, "attention_mask": tensor}, ...]
    
    # Find max length in batch for padding
    max_len = max(item["input_ids"].size(0) for item in batch)
    
    # Prepare padded tensors
    input_ids = []
    attention_masks = []
    labels = []
    
    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len
        
        # Pad input_ids with PAD token
        padded_input = torch.cat([
            item["input_ids"],
            torch.full((pad_len,), PAD, dtype=torch.long)
        ])
        
        # Pad attention_mask with 0s
        padded_mask = torch.cat([
            item["attention_mask"],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        
        # Create labels by cloning padded input
        padded_labels = padded_input.clone()
        
        # Find NOTES_START position
        notes_start_idx = None
        for j, token in enumerate(item["input_ids"]):
            if token == NOTES_START:
                notes_start_idx = j
                break
        
        if notes_start_idx is not None:
            # Mask everything up to and including NOTES_START
            padded_labels[:notes_start_idx + 1] = -100
        else:
            # No NOTES_START found - only keep EOS
            eos_mask = padded_input != EOS
            padded_labels[eos_mask] = -100
        
        # Always mask padding
        padded_labels[seq_len:] = -100
        
        input_ids.append(padded_input)
        attention_masks.append(padded_mask)
        labels.append(padded_labels)
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels),
    }


# dataset class
# -------------------------------------------------------------------

class BeatsDataset(Dataset):

    """
    Wraps a ZippedBeatSaberDataset and handles chunking with caching.
    """

    def __init__(self, base_dataset, chunk_duration=30.0, max_seq_length=32767, cache_dir="/tmp/bs_cache"):

        # base_dataset is a ZippedBeatSaberDataset (directory of zip files passed to it)
        self.base = base_dataset 
        self.chunk_duration = chunk_duration
        self.max_seq_length = max_seq_length
        
        # cache_dir for storing encoded audio
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        print(f"[Cache] Using directory: {self.cache_dir}")
        
        # encodec model singleton
        self.audio_tokenizer = None
    
    def _get_audio_tokenizer(self):
        """Lazy load audio tokenizer to avoid CUDA issues."""
        if self.audio_tokenizer is None:
            from audio import create_audio_tokenizer
            self.audio_tokenizer = create_audio_tokenizer(bandwidth=1.5)
        return self.audio_tokenizer
    
    def _get_cache_key(self, idx):
        """Generate cache key for sample."""
        # Include all parameters that affect output
        device = "cuda" if torch.cuda.is_available() else "cpu"
        key = f"sample_{idx}_dur{self.chunk_duration}_max{self.max_seq_length}_ppq{PPQ}_bw1.5_{device}_v3"
        return hashlib.md5(key.encode()).hexdigest()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        
        # Check cache first
        cache_key = self._get_cache_key(idx)
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    audio_tokens = cached_data['audio_tokens']
                    notes = cached_data['notes']
                    bpm = cached_data['bpm']
                    print(f"[Cache] Loaded sample {idx} from cache")
            except Exception as e:
                print(f"[Cache] Failed to load cache: {e}")
                cache_path.unlink()  # Remove corrupted cache
                # Fall through to regenerate
                audio_tokens = None
        else:
            audio_tokens = None
        
        # If not cached, load and encode
        if audio_tokens is None:
            audio, notes, bpm = self.base[idx]
            
            # Encode audio with encodec (expensive operation)
            # Note: Encodec model runs on GPU as per spec
            print(f"[Cache] Encoding sample {idx}...")
            audio_tokens = self._get_audio_tokenizer().encode(audio)
            
            # Save to cache
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'audio_tokens': audio_tokens,
                        'notes': notes,
                        'bpm': bpm
                    }, f)
            except Exception as e:
                print(f"[Cache] Failed to save: {e}")

        # Now do chunking
        bps = bpm / 60
        fps = 100
        total_beats = len(audio_tokens) / fps * bps

        if self.chunk_duration is None:
            return create_training_sequence(audio_tokens, notes)
        
        # Calculate chunks and note alignment
        # and slice chunks out correctly

        chunk_beats = self.chunk_duration * bps
        # Use deterministic offset based on idx for reproducibility
        offset_ratio = (idx % 10) / 10.0 
        start_beat = offset_ratio * max(0, total_beats - chunk_beats)
        end_beat = start_beat + chunk_beats
        
        # Convert beats to frames: beats -> seconds -> frames
        start_idx = int(round(start_beat * 60 / bpm * fps))
        end_idx = int(round(end_beat * 60 / bpm * fps))
        
        audio_chunk = audio_tokens[start_idx:end_idx]
        notes_chunk = [(t - start_beat, n) for t, n in notes if start_beat <= t < end_beat]

        # audio_chunk = (frames, 2)
        # notes_chunk = [(time, note_type), ...]
        sequence = create_training_sequence(audio_chunk, notes_chunk)
        
        # Check sequence length
        if len(sequence) > self.max_seq_length:
            print(f"[Warning] Sequence too long ({len(sequence)} > {self.max_seq_length}), truncating")
            sequence = sequence[:self.max_seq_length - 1] + [EOS]
        
        # Return dict format expected by HuggingFace trainer
        return {
            "input_ids": torch.tensor(sequence, dtype=torch.long),
            "attention_mask": torch.ones(len(sequence), dtype=torch.long)
        }
