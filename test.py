"""
Comprehensive test suite for Beat Saber AI.

Tests core functionality including:
- Audio tokenization round-trip
- Token space ranges
- Dataset caching invariants  
- Note encoding/decoding
"""

import pytest
import numpy as np
import torch
import tempfile
import json
import zipfile
from pathlib import Path

from audio import create_audio_tokenizer
from data import (
    time_to_token, audio_tokens_to_sequence, create_training_sequence,
    audio_masking_collator, BeatsDataset, 
    TIME_RANGE, NOTE_RANGE, AUDIO_RANGE, PPQ,
    BOS, EOS, PAD, AUDIO_START, AUDIO_END, NOTES_START
)
from maps import encode_note, decode_note, parse_note
from config import SAMPLE_RATE, ENCODEC_FPS
from config import get_logger

logger = get_logger(__name__)


class TestAudioTokenizer:
    """Test audio tokenization functionality."""
    
    def test_encode_decode_roundtrip(self):
        """Test that encode->decode preserves audio quality."""
        tokenizer = create_audio_tokenizer()
        
        # Create test audio - 1 second sine wave
        duration = 1.0
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
        original_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Round trip
        tokens = tokenizer.encode(original_audio)
        decoded_audio = tokenizer.decode(tokens)
        
        # Verify shapes
        assert tokens.shape[1] == 2, "Should have 2 codebooks"
        assert decoded_audio.shape[0] == 1, "Should be mono output"
        
        # Check frame rate (allowing some tolerance)
        actual_fps = tokens.shape[0] / duration
        assert 70 <= actual_fps <= 80, f"Frame rate {actual_fps} outside expected range"
        
        # Check token value ranges
        assert np.all(tokens >= 0), "Tokens should be non-negative"
        assert np.all(tokens <= 1023), "Tokens should be <= 1023"
        
    def test_stereo_to_mono_conversion(self):
        """Test stereo input gets converted to mono correctly."""
        tokenizer = create_audio_tokenizer()
        
        # Create stereo test audio
        duration = 0.5
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
        left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
        stereo_audio = np.stack([left, right])
        
        # Should not raise exception
        tokens = tokenizer.encode(stereo_audio)
        assert tokens.shape[1] == 2, "Should have 2 codebooks"


class TestTokenSpace:
    """Test token space mappings and ranges."""
    
    def test_time_token_ranges(self):
        """Test time token conversion stays in range."""
        # Test boundary cases
        assert time_to_token(0.0) == 0
        assert time_to_token(1.0) == PPQ  # 1 beat = PPQ ticks
        
        # Test large values get clamped
        large_time = 1000.0
        token = time_to_token(large_time)
        assert TIME_RANGE[0] <= token < TIME_RANGE[1]
        
    def test_audio_token_conversion(self):
        """Test audio token sequence conversion."""
        # Create mock audio tokens
        frames = 10
        codebooks = 2
        audio_tokens = np.random.randint(0, 1024, (frames, codebooks))
        
        sequence = audio_tokens_to_sequence(audio_tokens)
        
        # Check length
        assert len(sequence) == frames * codebooks
        
        # Check all tokens in audio range
        for token in sequence:
            assert AUDIO_RANGE[0] <= token < AUDIO_RANGE[1]
            
    def test_training_sequence_format(self):
        """Test complete training sequence structure."""
        # Mock inputs
        audio_tokens = np.random.randint(0, 1024, (5, 2))
        notes = [(1.0, 0), (2.0, 179)]  # Valid note types
        
        sequence = create_training_sequence(audio_tokens, notes)
        
        # Check structure
        assert sequence[0] == BOS
        assert sequence[1] == AUDIO_START
        assert sequence[-1] == EOS
        
        # Find separators
        audio_end_idx = sequence.index(AUDIO_END)
        notes_start_idx = sequence.index(NOTES_START)
        
        assert audio_end_idx < notes_start_idx
        assert notes_start_idx < len(sequence) - 1


class TestNoteEncoding:
    """Test Beat Saber note encoding/decoding."""
    
    def test_note_encode_decode_roundtrip(self):
        """Test all valid notes can be encoded and decoded."""
        for x in range(3):
            for y in range(3):
                for color in range(2):
                    for direction in range(10):
                        # Encode then decode
                        note_type = encode_note(x, y, color, direction)
                        decoded_x, decoded_y, decoded_color, decoded_direction = decode_note(note_type)
                        
                        # Should match original
                        assert decoded_x == x
                        assert decoded_y == y
                        assert decoded_color == color
                        assert decoded_direction == direction
                        
                        # Should be in valid range
                        assert 0 <= note_type < 180
                        
    def test_parse_note_v2_format(self):
        """Test parsing old format notes."""
        v2_note = {
            "_time": 4.5,
            "_lineIndex": 1,
            "_lineLayer": 0,
            "_type": 0,
            "_cutDirection": 3
        }
        
        parsed = parse_note(v2_note)
        assert parsed == (4.5, 1, 0, 0, 3)
        
    def test_parse_note_v3_format(self):
        """Test parsing new format notes."""
        v3_note = {
            "b": 4.5,
            "x": 1,
            "y": 0,
            "c": 0,
            "d": 3,
            "a": 0
        }
        
        parsed = parse_note(v3_note)
        assert parsed == (4.5, 1, 0, 0, 3)
        
    def test_parse_note_bomb_filtering(self):
        """Test that bombs (type 2,3) are filtered out."""
        bomb_note = {
            "_time": 4.5,
            "_lineIndex": 1,
            "_lineLayer": 0,
            "_type": 2,  # Bomb
            "_cutDirection": 3
        }
        
        parsed = parse_note(bomb_note)
        assert parsed is None


class TestDatasetCaching:
    """Test dataset caching behavior."""
    
    def create_mock_zip(self, zip_path: Path):
        """Create a mock Beat Saber map zip for testing."""
        info_data = {
            "_beatsPerMinute": 120.0,
            "_songName": "Test Song"
        }
        
        map_data = {
            "colorNotes": [
                {"b": 1.0, "x": 1, "y": 0, "c": 0, "d": 3, "a": 0},
                {"b": 2.0, "x": 2, "y": 1, "c": 1, "d": 1, "a": 0}
            ]
        }
        
        # Create minimal audio (1 second of silence)
        audio_data = np.zeros(SAMPLE_RATE, dtype=np.float32)
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr('Info.dat', json.dumps(info_data))
            zf.writestr('ExpertPlusStandard.dat', json.dumps(map_data))
            
            # Write audio as raw bytes (mock)
            import io
            import soundfile as sf
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, audio_data, SAMPLE_RATE, format='OGG')
            zf.writestr('song.egg', audio_bytes.getvalue())
    
    def test_cache_key_consistency(self):
        """Test that cache keys are consistent for same content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock dataset
            maps_dir = Path(temp_dir) / "maps"
            maps_dir.mkdir()
            zip_path = maps_dir / "test.zip"
            self.create_mock_zip(zip_path)
            
            from maps import ZippedBeatSaberDataset
            base_dataset = ZippedBeatSaberDataset(maps_dir)
            
            # Create dataset with caching
            cache_dir = Path(temp_dir) / "cache"
            dataset = BeatsDataset(base_dataset, cache_dir=cache_dir)
            
            # Get cache key twice
            key1 = dataset._get_cache_key(0)
            key2 = dataset._get_cache_key(0)
            
            assert key1 == key2, "Cache keys should be consistent"


class TestDataCollator:
    """Test the audio masking data collator."""
    
    def test_audio_masking(self):
        """Test that audio tokens are masked correctly."""
        # Create mock batch
        batch = [
            {
                "input_ids": torch.tensor([BOS, AUDIO_START, 12500, 12501, AUDIO_END, NOTES_START, 100, 12300, EOS]),
                "attention_mask": torch.ones(9)
            }
        ]
        
        result = audio_masking_collator(batch)
        
        # Check structure
        assert "input_ids" in result
        assert "attention_mask" in result  
        assert "labels" in result
        
        labels = result["labels"][0]
        
        # Everything before NOTES_START should be masked (-100)
        notes_start_pos = 5  # Position of NOTES_START
        assert torch.all(labels[:notes_start_pos + 1] == -100)
        
        # Note tokens should not be masked
        assert labels[6] == 100  # Time token
        assert labels[7] == 12300  # Note token
        assert labels[8] == EOS  # EOS token


def test_config_constants():
    """Test that config constants are sensible."""
    from config import PPQ, ENCODEC_FPS, SAMPLE_RATE
    
    assert PPQ > 0, "PPQ should be positive"
    assert ENCODEC_FPS > 0, "FPS should be positive"
    assert SAMPLE_RATE > 0, "Sample rate should be positive"
    
    # Check token ranges don't overlap
    assert TIME_RANGE[1] == NOTE_RANGE[0], "Time and note ranges should be adjacent"
    assert NOTE_RANGE[1] == AUDIO_RANGE[0], "Note and audio ranges should be adjacent"


# Use pytest for test discovery and running
# Run with: pytest test_beat_saber_ai.py -v