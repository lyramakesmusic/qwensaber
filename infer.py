"""
Beat Saber map generation from audio using trained model.
"""
import json
from pathlib import Path
import argparse

import numpy as np
import soundfile as sf
import torch
from unsloth import FastLanguageModel
import librosa
from typing import List, Dict, Union, Optional

from data import (
    BOS, EOS, AUDIO_START, AUDIO_END, NOTES_START, PAD,
    TIME_RANGE, NOTE_RANGE, AUDIO_RANGE, PPQ,
    audio_tokens_to_sequence, time_to_token
)
from maps import convert_to_v3_format, decode_note
from audio import create_audio_tokenizer

from config import ENCODEC_FPS, SAMPLE_RATE, ENCODEC_BANDWIDTH, CHUNK_DURATION, get_logger

logger = get_logger(__name__)

# Remove global tokenizer - pass explicitly instead


def prepare_audio(audio_path: str, duration: Optional[float] = None) -> np.ndarray:  # type: ignore[type-arg]
    """Load and prepare audio for generation.
    
    Args:
        audio_path: Path to audio file
        duration: Optional duration limit in seconds
        
    Returns:
        Audio tensor of shape [1, samples] - mono audio at 24kHz ready for Encodec
    """
    # Load audio
    audio, sr = sf.read(audio_path)
    # audio = (samples,) or (samples, 2) - mono or stereo
    
    # Convert to mono if stereo
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    # audio = (samples,) - now guaranteed mono
    
    # Resample if needed (Encodec requirement)
    if sr != SAMPLE_RATE:
        logger.info(f"Resampling audio from {sr}Hz to {SAMPLE_RATE}Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    
    # Trim to duration if specified
    if duration is not None:
        max_samples = int(duration * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            logger.info(f"Trimmed audio to {duration} seconds")
    
    # Reshape to match training format
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)  # [channels, samples]
    
    logger.info(f"Loaded {audio.shape[-1]/sr:.1f} seconds of audio at {sr}Hz")
    # Returns: (1, samples) - mono audio ready for encodec
    return audio


def generate_map(model, audio_tokenizer, audio, bpm, max_new_tokens=500, temperature=0.8, min_p=0.1, duration=None, chunk_duration=CHUNK_DURATION):
    """Generate Beat Saber map from audio with chunking for long songs."""
    # audio = (1, samples) - mono audio
    
    # Encode audio
    logger.info("Encoding audio with Encodec...")
    audio_tokens = audio_tokenizer.encode(audio)
    # audio_tokens = (frames, 2) - encodec output
    
    # Limit frames if duration specified
    if duration is not None:
        max_frames = int(duration * ENCODEC_FPS)
        audio_tokens = audio_tokens[:max_frames]
    
    total_duration = audio_tokens.shape[0] / ENCODEC_FPS
    
    # If audio is longer than chunk_duration, process in chunks
    if total_duration > chunk_duration:
        logger.info(f"Audio is {total_duration:.1f}s, processing in {chunk_duration}s chunks...")
        return _generate_chunked_map(model, audio_tokens, bpm, max_new_tokens, temperature, min_p, chunk_duration)
    else:
        return _generate_single_chunk(model, audio_tokens, bpm, max_new_tokens, temperature, min_p)

def _generate_single_chunk(model, audio_tokens: np.ndarray, bpm: float, max_new_tokens: int, temperature: float, min_p: float) -> List[Dict[str, Union[float, int]]]:
    """Generate map for a single chunk of audio."""
    
    # Create input sequence
    sequence = [BOS, AUDIO_START] + audio_tokens_to_sequence(audio_tokens) + [AUDIO_END, NOTES_START]
    input_ids = torch.tensor([sequence], dtype=torch.long, device=model.device)
    # input_ids = (1, seq_len) - batch of 1 sequence
    
    logger.info(f"Input sequence: {len(sequence)} tokens, generating up to {max_new_tokens} new tokens...")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            min_p=min_p if temperature > 0 else None,  # Only use min_p with sampling
            do_sample=temperature > 0,
            eos_token_id=EOS,
            pad_token_id=EOS,
            use_cache=True,
            return_dict_in_generate=False,
            min_length=len(sequence) + 2  # At least one note pair
        )
    # outputs = (1, total_len) - includes input + generated
    
    # Parse note pairs from generated tokens - more robust parsing
    generated = outputs[0].tolist()[len(sequence):]
    notes = []
    
    # State machine for parsing time/note pairs
    expecting_time = True
    current_time = None
    
    for token in generated:
        if token == EOS:
            break
        
        if token == PAD:  # Skip padding
            continue
            
        if expecting_time:
            # Look for time token
            if TIME_RANGE[0] <= token < TIME_RANGE[1]:
                current_time = float(token / PPQ)
                expecting_time = False
            else:
                logger.debug(f"Unexpected token {token} when expecting time, skipping")
        else:
            # Look for note token
            if NOTE_RANGE[0] <= token < NOTE_RANGE[1]:
                notes.append({
                    "time": current_time,
                    "type": int(token - NOTE_RANGE[0])
                })
                expecting_time = True
            else:
                logger.debug(f"Unexpected token {token} when expecting note, resetting")
                expecting_time = True
    
    logger.info(f"Generated {len(notes)} notes")
    return notes

def _generate_chunked_map(model, audio_tokens: np.ndarray, bpm: float, max_new_tokens: int, temperature: float, min_p: float, chunk_duration: float = CHUNK_DURATION) -> List[Dict[str, Union[float, int]]]:
    """Generate map by processing audio in chunks."""
    chunk_frames = int(chunk_duration * ENCODEC_FPS)
    overlap_frames = int(2.0 * ENCODEC_FPS)  # 2 second overlap
    
    all_notes = []
    current_offset = 0.0
    
    for start_frame in range(0, audio_tokens.shape[0], chunk_frames - overlap_frames):
        end_frame = min(start_frame + chunk_frames, audio_tokens.shape[0])
        chunk_tokens = audio_tokens[start_frame:end_frame]
        
        chunk_start_time = start_frame / ENCODEC_FPS
        
        logger.info(f"Processing chunk {chunk_start_time:.1f}s - {end_frame/ENCODEC_FPS:.1f}s")
        
        # Generate notes for this chunk
        chunk_notes = _generate_single_chunk(model, chunk_tokens, bpm, max_new_tokens, temperature, min_p)
        
        # Adjust note times to global timeline
        for note in chunk_notes:
            adjusted_note = {
                "time": note["time"] + chunk_start_time,
                "type": note["type"]
            }
            
            # Skip notes in overlap region (except first chunk)
            if start_frame > 0 and note["time"] < 2.0:
                continue
                
            all_notes.append(adjusted_note)
    
    # Sort by time and remove duplicates
    all_notes.sort(key=lambda x: x["time"])
    return all_notes


def run_inference(args: argparse.Namespace) -> None:
    """Run inference with given arguments."""
    
    try:
        # Load model
        logger.info(f"Loading model from {args.model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=32768,
            load_in_4bit=True,
            trust_remote_code=True
        )
        FastLanguageModel.for_inference(model)
        logger.info("Model loaded successfully")
        
        # Generate map
        audio = prepare_audio(args.audio_path, args.duration)
        audio_tokenizer = create_audio_tokenizer(bandwidth=ENCODEC_BANDWIDTH)
        notes = generate_map(
            model, audio_tokenizer, audio, args.bpm,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            min_p=args.min_p,
            duration=args.duration,
            chunk_duration=args.chunk_duration
        )
        
        # Save
        v3_map = convert_to_v3_format(notes, args.bpm)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(v3_map, f, indent=2)
        
        logger.info(f"Saved {len(notes)} notes to {output_path}")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Beat Saber maps from audio")
    
    parser.add_argument("model_path", type=str, help="Path to trained model")
    parser.add_argument("audio_path", type=str, help="Path to audio file")
    parser.add_argument("--bpm", type=float, required=True, help="Song BPM")
    parser.add_argument("--output", type=str, default="generated_map.dat", help="Output map file")
    parser.add_argument("--duration", type=float, help="Generate for first N seconds of audio")
    parser.add_argument("--max_tokens", type=int, default=500, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--min_p", type=float, default=0.1, help="Minimum token probability to sample")
    parser.add_argument("--chunk_duration", type=float, default=CHUNK_DURATION, help="Chunk duration for long audio (seconds)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not Path(args.model_path).exists():
        logger.error(f"Model path does not exist: {args.model_path}")
        parser.print_help()
        exit(1)
    
    if not Path(args.audio_path).exists():
        logger.error(f"Audio file does not exist: {args.audio_path}")
        parser.print_help()
        exit(1)
    
    if args.bpm <= 0:
        logger.error("BPM must be positive")
        parser.print_help()
        exit(1)
    
    if args.temperature < 0:
        logger.error("Temperature must be non-negative")
        parser.print_help()
        exit(1)
    
    try:
        run_inference(args)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        exit(1) 