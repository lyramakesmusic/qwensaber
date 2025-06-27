"""
Beat Saber dataset and vocabulary handling.

We use the first 16,570 tokens of Qwen3's vocabulary for our data:
- 0-12,287: Time ticks (48 PPQ resolution)
- 12,288-12,467: Note types (180 combinations)
- 12,468-14,515: Audio tokens (2 × 1,024 Encodec codebooks)
- 16,564: BOS (beginning)
- 16,565: EOS (end)
- 16,566: PAD (padding)
- 16,567-16,569: Structure separators
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
VOCAB_SIZE = 16570  # Total tokens we use from Qwen3's vocabulary

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

# Verify ranges
assert AUDIO_RANGE[1] - AUDIO_RANGE[0] == 2048, "Audio range must fit 2 codebooks × 1,024 values"
assert NOTE_RANGE[1] - NOTE_RANGE[0] == 180, "Note range must fit 9 positions x 2 colors × 10 directions"
assert TIME_RANGE[1] - TIME_RANGE[0] == 12288, "Time range must fit 48 PPQ x 256 beats"
assert NOTES_START < VOCAB_SIZE, "All special tokens must fit in vocabulary"

# Singleton audio tokenizer
_AUDIO_TOKENIZER = None

def get_audio_tokenizer(bandwidth=1.5):
    """Get or create singleton audio tokenizer instance."""
    global _AUDIO_TOKENIZER
    if _AUDIO_TOKENIZER is None:
        _AUDIO_TOKENIZER = create_audio_tokenizer(bandwidth)
    return _AUDIO_TOKENIZER


def encode_note(x, y, color, direction):
    """Encode Beat Saber note position/color/direction to token ID."""
    # Position (3x3 grid) * 20 + color (0/1) * 10 + direction (0-9)
    note_type = (y * 3 + x) * 20 + color * 10 + direction
    return NOTE_RANGE[0] + note_type


def decode_note(token_id):
    """Decode token ID back to note properties."""
    note_type = token_id - NOTE_RANGE[0]
    direction = note_type % 10
    color = (note_type // 10) % 2
    position = note_type // 20
    y = position // 3
    x = position % 3
    return x, y, color, direction


def time_to_token(beat_time):
    """Convert beat time to time token."""
    tick = int(round(beat_time * PPQ))
    return min(tick, TIME_RANGE[1] - 1)  # Clamp to valid range


def audio_tokens_to_sequence(audio_tokens):
    """Convert Encodec audio tokens to our token sequence."""
    # Flatten the codebooks and add offsets
    sequence = []
    n_codebooks = audio_tokens.shape[1]
    values_per_codebook = (AUDIO_RANGE[1] - AUDIO_RANGE[0]) // n_codebooks
    
    for frame in audio_tokens:
        for slot, value in enumerate(frame):
            # Scale value from 0-1023 to fit in our space
            scaled_value = int(value * (values_per_codebook - 1) / 1023)
            token = AUDIO_RANGE[0] + slot * values_per_codebook + scaled_value
            sequence.append(int(token))
            
    return sequence


class CachedDataset(Dataset):
    """Wrapper that caches expensive audio encoding operations."""
    
    def __init__(self, base_dataset, cache_dir="/tmp/bs_cache_unsloth"):
        self.base = base_dataset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        print(f"[Cache] Using directory: {self.cache_dir}")
        
    def _get_cache_key(self, idx):
        """Generate unique key for caching."""
        key = f"{self.base.__class__.__name__}_{len(self.base)}_{idx}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        # Try cache first
        cache_key = self._get_cache_key(idx)
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                cache_path.unlink()  # Remove corrupted cache
        
        # Load and encode
        print(f"[Cache] Encoding sample {idx}...")
        audio, notes, bpm = self.base[idx]
        
        # Encode audio (expensive operation)
        audio_tokens = get_audio_tokenizer().encode(audio)
        result = (audio_tokens, notes, bpm)
        
        # Save to cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"[Cache] Failed to save: {e}")
            
        return result


class ChunkedDataset(Dataset):
    """Takes chunks from songs, aligned to beats. If chunk_duration is None, uses full song."""
    
    def __init__(self, base_dataset, chunk_duration=30.0, max_seq_length=2048):
        self.base = base_dataset
        self.chunk_duration = chunk_duration
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.base)
        
    def __getitem__(self, idx):
        audio_tokens, notes, bpm = self.base[idx]
        
        # Calculate beats per second
        beats_per_sec = bpm / 60.0
        
        # Calculate total beats based on audio tokens
        # Each Encodec frame is ~10ms at 24kHz
        frames_per_sec = 100  # ~100Hz frame rate
        total_seconds = len(audio_tokens) / frames_per_sec
        total_beats = total_seconds * beats_per_sec
        
        # If no chunk duration specified, use full song
        if self.chunk_duration is None:
            audio_chunk = audio_tokens
            notes_chunk = notes
        else:
            # Calculate beats in chunk
            beats_per_chunk = self.chunk_duration * beats_per_sec
            
            # Random starting beat
            if total_beats > beats_per_chunk:
                start_beat = torch.rand(1).item() * (total_beats - beats_per_chunk)
            else:
                start_beat = 0
                
            end_beat = start_beat + beats_per_chunk
            
            # Convert beats to audio token indices
            start_frame = int(start_beat * frames_per_sec / beats_per_sec)
            end_frame = int(end_beat * frames_per_sec / beats_per_sec)
            
            # Extract audio chunk
            audio_chunk = audio_tokens[start_frame:end_frame]
            
            # Filter notes within beat range
            notes_chunk = []
            for beat_time, note_type in notes:
                if start_beat <= beat_time < end_beat:
                    # Adjust time relative to chunk start
                    adjusted_time = beat_time - start_beat
                    notes_chunk.append((adjusted_time, note_type))
                
        # Create training sequence from chunk
        return create_training_sequence(audio_chunk, notes_chunk, self.max_seq_length)


def create_training_sequence(audio_tokens, notes, max_length=2048):
    """Create a training sequence from audio tokens and notes."""
    # Start sequence
    sequence = [BOS, AUDIO_START]
    
    # Add audio tokens (should already be properly chunked by ChunkedDataset)
    audio_seq = audio_tokens_to_sequence(audio_tokens)
    sequence.extend(audio_seq)
    
    # Transition to notes
    sequence.extend([AUDIO_END, NOTES_START])
    
    # Add note pairs (time, type)
    note_tokens_created = []
    for beat_time, note_type in notes:
        time_token = time_to_token(beat_time)
        note_token = NOTE_RANGE[0] + int(note_type)
        sequence.extend([time_token, note_token])
        note_tokens_created.extend([time_token, note_token])
    
    # End sequence
    sequence.append(EOS)
    
    # Sanity check - sequence should be reasonable length now
    if len(sequence) > max_length:
        print(f"[Sequence Debug] WARNING: Sequence too long ({len(sequence)} > {max_length}). Consider shorter chunks!")
        # Emergency truncation only if needed
        sequence = sequence[:max_length-1] + [EOS]
    
    # Don't pad here! Let collator handle padding
    return sequence


def collate_sequences(batch):
    """Collate function for DataLoader."""
    # Find max length in this batch
    max_len = max(len(seq) for seq in batch)
    
    sequences = []
    labels = []
    attention_masks = []
    
    for batch_idx, seq in enumerate(batch):
        # Pad sequence to max_len in batch
        pad_length = max_len - len(seq)
        padded_seq = seq + [PAD] * pad_length
        
        # Create labels: only train on tokens AFTER NOTES_START
        # Audio context should not be predicted, only notes!
        padded_labels = []
        
        # Find where notes start in this sequence
        notes_start_idx = None
        for i, token in enumerate(seq):
            if token == NOTES_START:
                notes_start_idx = i
                break
        
        if notes_start_idx is not None:
            # For causal LM: position i predicts position i+1
            # So label[i] should be seq[i+1]
            for i in range(len(seq)):
                if i < len(seq) - 1:
                    # Can we predict the next token?
                    if i >= notes_start_idx:  # At or after NOTES_START
                        padded_labels.append(seq[i + 1])  # Predict NEXT token
                    else:
                        padded_labels.append(-100)  # Ignore audio predictions
                else:
                    # Last position has no next token
                    padded_labels.append(-100)
                    
            # Debug: show first few label alignments for first batch
            if batch_idx == 0 and notes_start_idx < len(seq) - 5:
                print(f"[Label Debug] First few note predictions:")
                for i in range(notes_start_idx, min(notes_start_idx + 5, len(seq) - 1)):
                    print(f"  Position {i}: token={seq[i]} -> predicts {seq[i+1]} (label={padded_labels[i]})")
        else:
            # Fallback: if no NOTES_START found, ignore everything
            padded_labels = [-100] * len(seq)
            print(f"[Debug] WARNING: No NOTES_START found in sequence {batch_idx}!")
        
        # Add padding labels
        padded_labels += [-100] * pad_length
        
        # Create attention mask
        attention_mask = [1] * len(seq) + [0] * pad_length
        
        sequences.append(padded_seq)
        labels.append(padded_labels)
        attention_masks.append(attention_mask)
    
    # Convert to tensors
    return {
        "input_ids": torch.tensor(sequences, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long)
    }


def tokenize_beat_saber_data(
    audio_codes: np.ndarray,
    notes: List[Dict],
    bpm: float,
    audio_duration: float,
    frames_per_second: float = 75,
    max_sequence_length: int = 32767
) -> np.ndarray:
    """
    Tokenize Beat Saber data into a single sequence.
    """
    # Create flattened audio tokens
    audio_tokens = audio_codes.flatten() + AUDIO_RANGE[0]
    
    # Create note tokens
    note_tokens = np.array([note['note_type'] for note in notes])
    note_tokens = np.clip(note_tokens, NOTE_RANGE[0], NOTE_RANGE[1]-1).astype(int)
    
    # Create attention mask
    attention_mask = np.ones(len(audio_tokens), dtype=bool)
    
    # Create sequence tokens in order: audio, time, notes
    sequence = np.concatenate([
        audio_tokens,
        note_tokens
    ])
    
    return sequence 