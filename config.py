"""
Central configuration for Beat Saber AI project.

All constants, paths, and magic numbers should be defined here.
"""

from enum import IntEnum
from pathlib import Path


# Audio configuration
SAMPLE_RATE = 24000
ENCODEC_BANDWIDTH = 1.5
ENCODEC_FPS = 75  # Encodec frames per second at bandwidth 1.5
CHUNK_DURATION = 30.0  # Default chunk duration in seconds

# Token space configuration
PPQ = 48  # Pulses per quarter note
VOCAB_SIZE = 14522  # Total tokens we use from Qwen3's vocabulary

# Token ranges
TIME_TICK_RANGE = (0, 12288)      # 12,288 time ticks
NOTE_TYPE_RANGE = (12288, 12468)  # 180 note types  
AUDIO_TOKEN_RANGE = (12468, 14516) # 2,048 audio tokens

# Special tokens as IntEnum for type safety
class SpecialTokens(IntEnum):
    BOS = 14516         # Beginning of sequence
    EOS = 14517         # End of sequence  
    PAD = 14518         # Padding
    AUDIO_START = 14519 # Start of audio tokens
    AUDIO_END = 14520   # End of audio tokens
    NOTES_START = 14521 # Start of note tokens

# Model configuration  
MAX_SEQ_LENGTH = 32768

# Training defaults
DEFAULT_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WARMUP_STEPS = 5
DEFAULT_SAVE_STEPS = 100
DEFAULT_MAX_STEPS = 500

# Paths
DEFAULT_CACHE_DIR = Path("/tmp/bs_cache")
DEFAULT_OUTPUT_DIR = Path("./beatsaber_model")

# Beat Saber map constants
GRID_WIDTH = 3  # 0, 1, 2
GRID_HEIGHT = 3  # 0, 1, 2  
NOTE_COLORS = 2  # Red (0), Blue (1)
CUT_DIRECTIONS = 10  # 0-8 + dot (9)

# Logging
def get_logger(name: str):
    """Get a logger instance with consistent formatting."""
    import logging
    import sys
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger 