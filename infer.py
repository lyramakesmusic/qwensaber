"""
Beat Saber map generation from audio using trained model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from unsloth import FastLanguageModel
import librosa

from data import (
    BOS, EOS, AUDIO_START, AUDIO_END, NOTES_START, PAD,
    TIME_RANGE, NOTE_RANGE, AUDIO_RANGE, PPQ,
    audio_tokens_to_sequence, time_to_token
)
from maps import convert_to_v3_format, decode_note
from audio import create_audio_tokenizer

# Use same audio tokenizer as training (runs on GPU if available)
AUDIO_TOKENIZER = create_audio_tokenizer(bandwidth=1.5)


def prepare_audio(audio_path, duration=None):
    """Load and prepare audio for generation."""
    # Load audio
    audio, sr = sf.read(audio_path)
    # audio = (samples,) or (samples, 2) - mono or stereo
    
    # Convert to mono if stereo
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    # audio = (samples,) - now guaranteed mono
    
    # Resample to 24kHz if needed (Encodec requirement)
    if sr != 24000:
        print(f"[Audio] Resampling from {sr}Hz to 24000Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        sr = 24000
    
    # Trim to duration if specified
    if duration is not None:
        max_samples = int(duration * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            print(f"[Audio] Trimmed to {duration} seconds")
    
    # Reshape to match training format
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)  # [channels, samples]
    
    print(f"[Audio] Loaded {audio.shape[-1]/sr:.1f} seconds at {sr}Hz")
    # Returns: (1, samples) - mono audio ready for encodec
    return audio


def generate_map(model, audio, bpm, max_new_tokens=500, temperature=0.8, min_p=0.1, duration=None):
    """Generate Beat Saber map from audio."""
    # audio = (1, samples) - mono audio
    
    # Encode audio
    print("[Audio] Encoding with Encodec...")
    audio_tokens = AUDIO_TOKENIZER.encode(audio)
    # audio_tokens = (frames, 2) - encodec output, ~100fps
    
    # Limit frames if duration specified
    if duration is not None:
        max_frames = int(duration * 100)  # 100 fps
        audio_tokens = audio_tokens[:max_frames]
    
    # Create input sequence
    sequence = [BOS, AUDIO_START] + audio_tokens_to_sequence(audio_tokens) + [AUDIO_END, NOTES_START]
    input_ids = torch.tensor([sequence], dtype=torch.long, device=model.device)
    # input_ids = (1, seq_len) - batch of 1 sequence
    
    print(f"[Generate] Input: {len(sequence)} tokens, generating up to {max_new_tokens} new tokens...")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            # min_p=min_p,
            do_sample=True,
            eos_token_id=EOS,
            pad_token_id=EOS,
            use_cache=True,
            return_dict_in_generate=False,
            min_length=len(sequence) + 2  # At least one note pair
        )
    # outputs = (1, total_len) - includes input + generated
    
    # Parse note pairs from generated tokens
    generated = outputs[0].tolist()[len(sequence):]
    # generated = List[int] - only new tokens
    notes = []
    
    i = 0
    while i < len(generated) - 1:
        if generated[i] == EOS:
            break
            
        # Check for time/note pair
        if TIME_RANGE[0] <= generated[i] < TIME_RANGE[1] and NOTE_RANGE[0] <= generated[i + 1] < NOTE_RANGE[1]:
            notes.append({
                "time": float(generated[i] / PPQ),
                "type": int(generated[i + 1] - NOTE_RANGE[0])
            })
            i += 2
        else:
            i += 1
    
    print(f"[Generate] Created {len(notes)} notes")
    # Returns: List[{"time": float, "type": int}] - decoded notes
    return notes


def main():
    parser = argparse.ArgumentParser(description="Generate Beat Saber maps from audio")
    
    parser.add_argument("model_path", type=str, help="Path to trained model")
    parser.add_argument("audio_path", type=str, help="Path to audio file")
    parser.add_argument("--bpm", type=float, required=True, help="Song BPM")
    parser.add_argument("--output", type=str, default="generated_map.dat", help="Output map file")
    parser.add_argument("--duration", type=float, help="Generate for first N seconds of audio")
    parser.add_argument("--max_tokens", type=int, default=500, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--min_p", type=float, default=0.1, help="Minimum token probability to sample")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=32768,
        load_in_4bit=True,
        trust_remote_code=True
    )
    FastLanguageModel.for_inference(model)
    print(f"[Model] Loaded from {args.model_path}")
    
    # Generate map
    audio = prepare_audio(args.audio_path, args.duration)
    notes = generate_map(
        model, audio, args.bpm,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        min_p=args.min_p,
        duration=args.duration
    )
    
    # Save
    v3_map = convert_to_v3_format(notes, args.bpm)
    with open(args.output, 'w') as f:
        json.dump(v3_map, f, indent=2)
    
    print(f"[Done] Saved {len(notes)} notes to {args.output}")


if __name__ == "__main__":
    main() 