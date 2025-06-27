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
from typing import List, Dict, Tuple, Optional, Union

from audio import create_audio_tokenizer

from config import (
    PPQ, VOCAB_SIZE, 
    TIME_TICK_RANGE as TIME_RANGE,
    NOTE_TYPE_RANGE as NOTE_RANGE, 
    AUDIO_TOKEN_RANGE as AUDIO_RANGE,
    SpecialTokens,
    ENCODEC_FPS,
    DEFAULT_CACHE_DIR, get_logger
)

logger = get_logger(__name__)

# Import special tokens for backward compatibility
BOS = SpecialTokens.BOS
EOS = SpecialTokens.EOS
PAD = SpecialTokens.PAD
AUDIO_START = SpecialTokens.AUDIO_START
AUDIO_END = SpecialTokens.AUDIO_END
NOTES_START = SpecialTokens.NOTES_START


# encoding / decoding data
# -------------------------------------------------------------------

def time_to_token(beat_time: float) -> int:
    """Convert beat time to time token."""
    tick = int(round(beat_time * PPQ))
    return min(tick, TIME_RANGE[1] - 1)

def audio_tokens_to_sequence(audio_tokens: np.ndarray) -> List[int]:  # type: ignore[type-arg]
    """Convert Encodec audio tokens to our token sequence.
    
    Args:
        audio_tokens: Tensor[frames, 2] with values 0-1023 from Encodec
        
    Returns:
        List of tokens in range 12468-14515 (length = frames * 2)
    """
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
    
    return sequence

def create_training_sequence(audio_tokens: np.ndarray, notes: List[Tuple[float, int]]) -> List[int]:
    """Create a training sequence from audio tokens and notes.
    
    Format: [BOS, AUDIO_START, audio_tokens, AUDIO_END, NOTES_START, time_token, note_token, ... EOS]
    
    Args:
        audio_tokens: Tensor[frames, 2] - Encodec output tokens
        notes: [(beat_time, note_type), ...] - beat times and note types (0-179)
        
    Returns:
        Complete training sequence as list of token IDs
    """
    
    sequence: List[int] = [BOS, AUDIO_START]
    
    audio_seq = audio_tokens_to_sequence(audio_tokens)
    sequence.extend(audio_seq)  # type: ignore[arg-type]
    
    sequence.extend([AUDIO_END, NOTES_START])  # type: ignore[arg-type]

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

def audio_masking_collator(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collator for audio masking.
    
    Args:
        batch: List of dicts with "input_ids" and "attention_mask" tensors
        
    Returns:
        Dict with batched "input_ids", "attention_mask", and "labels"
    """
    
    # Find max length in batch for padding
    # Handle both tensors and lists
    max_len = max(
        item["input_ids"].size(0) if torch.is_tensor(item["input_ids"]) 
        else len(item["input_ids"]) 
        for item in batch
    )
    
    # Prepare padded tensors
    input_ids = []
    attention_masks = []
    labels = []
    
    for item in batch:
        # Convert to tensors if needed
        item_input_ids = item["input_ids"] if torch.is_tensor(item["input_ids"]) else torch.tensor(item["input_ids"], dtype=torch.long)
        item_attention_mask = item["attention_mask"] if torch.is_tensor(item["attention_mask"]) else torch.tensor(item["attention_mask"], dtype=torch.long)
        
        seq_len = item_input_ids.size(0)
        pad_len = max_len - seq_len
        
        # Pad input_ids with PAD token
        padded_input = torch.cat([
            item_input_ids,
            torch.full((pad_len,), PAD, dtype=torch.long)
        ])
        
        # Pad attention_mask with 0s
        padded_mask = torch.cat([
            item_attention_mask,
            torch.zeros(pad_len, dtype=torch.long)
        ])
        
        # Create labels by cloning padded input
        padded_labels = padded_input.clone()
        
        # Find NOTES_START position
        notes_start_idx = None
        for j, token in enumerate(item_input_ids):
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


# Worker initialization for DataLoader
# -------------------------------------------------------------------

def worker_init_fn(worker_id: int) -> None:
    """
    Initialize DataLoader worker to avoid GPU conflicts.
    
    Each worker should use CPU for encoding to avoid GPU memory conflicts,
    or share the model from main process.
    """
    import os
    # Force workers to use CPU to avoid multiple GPU model copies
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


# dataset class
# -------------------------------------------------------------------

class BeatsDataset(Dataset):

    """
    Wraps a ZippedBeatSaberDataset and handles chunking with caching.
    """

    def __init__(self, base_dataset: Dataset, chunk_duration: float = 30.0, 
                 max_seq_length: int = 32767, cache_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize chunking dataset wrapper.
        
        Args:
            base_dataset: ZippedBeatSaberDataset instance
            chunk_duration: Duration of audio chunks in seconds
            max_seq_length: Maximum sequence length for model
            cache_dir: Directory for caching encoded audio
        """
        self.base = base_dataset 
        self.chunk_duration = chunk_duration
        self.max_seq_length = max_seq_length
        
        # cache_dir for storing encoded audio
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)
        
        # encodec model singleton
        self.audio_tokenizer = None
    
    def _get_audio_tokenizer(self):
        """Lazy load audio tokenizer to avoid CUDA issues."""
        if self.audio_tokenizer is None:
            from audio import create_audio_tokenizer
            self.audio_tokenizer = create_audio_tokenizer(bandwidth=1.5)
        return self.audio_tokenizer
    
    def _get_cache_key(self, idx: int) -> str:
        """Generate cache key for sample."""
        # Hash based on content and version, not device
        # This way cache works across GPU/CPU switches
        zip_path = self.base.zip_files[idx]
        zip_stat = zip_path.stat()
        content_hash = f"{zip_path.name}_{zip_stat.st_size}_{zip_stat.st_mtime}"
        version = "v4"  # Increment when changing encoding logic
        key = f"{content_hash}_ppq{PPQ}_bw1.5_{version}"
        return hashlib.md5(key.encode()).hexdigest()

    def __len__(self) -> int:
        return len(self.base)

    def _load_cache(self, idx: int, cache_key: str) -> Tuple[Optional[np.ndarray], Optional[List[Tuple[float, int]]], Optional[float]]:  # type: ignore[type-arg]
        """Load cached audio tokens, notes, and BPM if available.
        
        Args:
            idx: Sample index
            cache_key: Cache key for this sample
            
        Returns:
            Tuple of (audio_tokens, notes, bpm) or (None, None, None) if cache miss
        """
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_path.exists():
            return None, None, None
            
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data['audio_tokens'], cached_data['notes'], cached_data['bpm']
        except Exception as e:
            logger.warning(f"Failed to load cache, removing corrupted file: {e}")
            cache_path.unlink()  # Remove corrupted cache
            return None, None, None

    def _encode_audio(self, idx: int, cache_key: str) -> Tuple[np.ndarray, List[Tuple[float, int]], float]:  # type: ignore[type-arg]
        """Encode audio with Encodec and save to cache.
        
        Args:
            idx: Sample index
            cache_key: Cache key for saving
            
        Returns:
            Tuple of (audio_tokens, notes, bpm) where audio_tokens shape is (frames, 2)
        """
        audio, notes, bpm = self.base[idx]
        
        # Encode audio with encodec (expensive operation)
        # Note: Encodec model runs on GPU as per spec
        audio_tokens = self._get_audio_tokenizer().encode(audio)
        
        # Save to cache
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'audio_tokens': audio_tokens,
                    'notes': notes,
                    'bpm': bpm
                }, f)
        except Exception as e:
            logger.warning(f"Failed to cache encoded audio: {e}")
            # Continue anyway, will regenerate next time
            
        return audio_tokens, notes, bpm

    def _slice_chunk(self, audio_tokens: np.ndarray, notes: List[Tuple[float, int]], bpm: float, cache_key: str, idx: int) -> Tuple[np.ndarray, List[Tuple[float, int]]]:  # type: ignore[type-arg]
        """Slice audio and notes into training chunks.
        
        Args:
            audio_tokens: Encoded audio tokens, shape (frames, 2)
            notes: List of (beat_time, note_type) tuples
            bpm: Beats per minute
            cache_key: For deterministic offset calculation
            idx: Sample index for offset calculation
            
        Returns:
            Tuple of (audio_chunk, notes_chunk) where:
                audio_chunk shape is (chunk_frames, 2)
                notes_chunk is [(relative_time, note_type), ...]
        """
        if self.chunk_duration is None:
            # Return full audio/notes without chunking
            return audio_tokens, notes
        
        bps = bpm / 60
        fps = ENCODEC_FPS
        total_beats = len(audio_tokens) / fps * bps
        chunk_beats = self.chunk_duration * bps
        
        # Use deterministic hash-based offset for better variance across tracks
        hash_input = f"{cache_key}_{idx}"
        hash_int = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        offset_ratio = (hash_int % 1000) / 1000.0  # More granular than modulo 10
        start_beat = offset_ratio * max(0, total_beats - chunk_beats)
        end_beat = start_beat + chunk_beats
        
        # Convert beats to frames: beats -> seconds -> frames
        start_idx = int(round(start_beat * 60 / bpm * fps))
        end_idx = int(round(end_beat * 60 / bpm * fps))
        
        audio_chunk = audio_tokens[start_idx:end_idx]
        notes_chunk = [(t - start_beat, n) for t, n in notes if start_beat <= t < end_beat]
        
        return audio_chunk, notes_chunk

    def _build_sequence(self, audio_chunk: np.ndarray, notes_chunk: List[Tuple[float, int]], idx: int) -> List[int]:  # type: ignore[type-arg]
        """Build final training sequence with length checking.
        
        Args:
            audio_chunk: Audio tokens, shape (frames, 2)
            notes_chunk: Note list [(time, note_type), ...]
            idx: Sample index for logging
            
        Returns:
            Training sequence as list of token IDs
        """
        sequence = create_training_sequence(audio_chunk, notes_chunk)
        
        # Check sequence length and truncate if needed
        if len(sequence) > self.max_seq_length:
            logger.warning(f"Sequence too long ({len(sequence)} > {self.max_seq_length}), truncating")
            sequence = sequence[:self.max_seq_length - 1] + [EOS]
        
        # Debug: log actual sequence length once in a while
        if idx == 0:  # Only log for first sample to avoid spam
            logger.debug(f"Sample {idx} sequence length: {len(sequence)} tokens (audio_frames: {len(audio_chunk)}, notes: {len(notes_chunk)})")
        
        return sequence

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get training sample by orchestrating cache, encode, slice, and build steps."""
        cache_key = self._get_cache_key(idx)
        
        # Step 1: Try to load from cache
        audio_tokens, notes, bpm = self._load_cache(idx, cache_key)
        
        # Step 2: If cache miss, encode audio
        if audio_tokens is None:
            audio_tokens, notes, bpm = self._encode_audio(idx, cache_key)
        
        # Step 3: Slice into chunks
        audio_chunk, notes_chunk = self._slice_chunk(audio_tokens, notes, bpm, cache_key, idx)
        
        # Step 4: Build final sequence
        sequence = self._build_sequence(audio_chunk, notes_chunk, idx)
        
        # Return dict format expected by HuggingFace trainer
        return {
            "input_ids": torch.tensor(sequence, dtype=torch.long),
            "attention_mask": torch.ones(len(sequence), dtype=torch.long)
        }
