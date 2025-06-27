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
    TIME_RANGE, NOTE_RANGE, AUDIO_RANGE, audio_tokens_to_sequence, 
    time_to_token, decode_note, get_audio_tokenizer
)
from maps import convert_to_v3_format

# Use same audio tokenizer as training
AUDIO_TOKENIZER = get_audio_tokenizer(bandwidth=1.5)

def load_model(model_path, max_seq_length=32768):
    """Load finetuned model for inference."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        trust_remote_code=True  # Required for Qwen models
    )
    
    FastLanguageModel.for_inference(model)
    print(f"[Model] Loaded from {model_path}")
    return model, tokenizer


def prepare_audio(audio_path, duration=None):
    """Load and prepare audio for generation."""
    # Load audio
    audio, sr = sf.read(audio_path)
    
    # Convert to mono if stereo
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    
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
    
    # Reshape to match training format - IMPORTANT!
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)  # [channels, samples] to match maps.py
        print(f"[Audio] Reshaped to {audio.shape} to match training format")
    
    print(f"[Audio] Loaded {audio.shape[-1]/sr:.1f} seconds at {sr}Hz")
    return audio


def generate_map(model, tokenizer, audio, bpm, max_new_tokens=500, temperature=0.8, min_p=0.1, duration=None):
    """Generate Beat Saber map from audio."""
    # Encode audio
    print("[Audio] Encoding with Encodec...")
    audio_tokens = AUDIO_TOKENIZER.encode(audio)
    print(f"[Audio] Got {len(audio_tokens)} raw audio frames")
    
    # Calculate how many frames to use based on duration
    frames_per_sec = 100  # Encodec frame rate
    
    # Limit frames if duration specified
    if duration is not None:
        max_frames = int(duration * frames_per_sec)
        if len(audio_tokens) > max_frames:
            print(f"[Audio] Limiting to first {duration:.1f} seconds ({max_frames} frames at {frames_per_sec}Hz)")
            audio_tokens = audio_tokens[:max_frames]
    
    # Create input sequence with audio
    sequence = [BOS, AUDIO_START]
    audio_seq = audio_tokens_to_sequence(audio_tokens)
    sequence.extend(audio_seq)
    sequence.extend([AUDIO_END, NOTES_START])
    
    print(f"\n[Generate] Sequence breakdown:")
    print(f"- Total length: {len(sequence)} tokens")
    print(f"- Audio tokens: {len(audio_seq)} ({len(audio_tokens)} frames x 2 codebooks)")
    print(f"- Audio token range: {AUDIO_RANGE[0]} to {AUDIO_RANGE[1]} ({AUDIO_RANGE[1] - AUDIO_RANGE[0]} values)")
    print(f"- Special tokens: BOS={BOS}, AUDIO_START={AUDIO_START}, AUDIO_END={AUDIO_END}, NOTES_START={NOTES_START}")
    
    # Convert to tensor
    input_ids = torch.tensor([sequence], dtype=torch.long, device=model.device)
    
    print(f"\n[Generate] Generating up to {max_new_tokens} new tokens...")
    
    # Generate
    with torch.no_grad():
        try:
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                min_p=min_p,
                do_sample=True,
                eos_token_id=EOS,
                pad_token_id=EOS,  # Use EOS as PAD for generation
                use_cache=True,
                return_dict_in_generate=False,
                min_new_tokens=2  # Force at least one note pair
            )
            
            # Parse generated tokens
            generated = outputs[0].tolist()
            
            # Debug print generation
            print(f"\n[Generate] Generation results:")
            print(f"- Total tokens: {len(generated)}")
            print(f"- New tokens: {len(generated) - len(sequence)}")
            
            # Start parsing notes immediately after our NOTES_START token
            note_tokens = generated[len(sequence):]  # Skip input sequence
            print(f"\n[Parse] Note token ranges:")
            print(f"- Time tokens: {TIME_RANGE[0]} to {TIME_RANGE[1]} ({TIME_RANGE[1] - TIME_RANGE[0]} values)")
            print(f"- Note tokens: {NOTE_RANGE[0]} to {NOTE_RANGE[1]} ({NOTE_RANGE[1] - NOTE_RANGE[0]} values)")
            
            # Parse note pairs
            all_notes = []
            i = 0
            skipped_tokens = 0  # Counter for non-note tokens
            token_distribution = {}  # Track what tokens we're seeing
            
            while i < len(note_tokens) - 1:
                if note_tokens[i] == EOS:
                    print(f"[Parse] Found EOS at position {i}")
                    break
                    
                # Track token distribution
                token = note_tokens[i]
                if token in token_distribution:
                    token_distribution[token] += 1
                else:
                    token_distribution[token] = 1
                    
                # Check if valid time token
                if TIME_RANGE[0] <= note_tokens[i] < TIME_RANGE[1]:
                    time_tick = note_tokens[i]
                    
                    # Check if next is valid note token
                    if i + 1 < len(note_tokens) and NOTE_RANGE[0] <= note_tokens[i + 1] < NOTE_RANGE[1]:
                        note_type = note_tokens[i + 1] - NOTE_RANGE[0]
                        
                        # Convert tick back to beat time
                        beat_time = time_tick / 48.0  # PPQ = 48
                        
                        all_notes.append({
                            "time": float(beat_time),
                            "type": int(note_type),
                        })
                        
                        i += 2
                        continue
                
                # If we get here, current token wasn't valid time or note token
                skipped_tokens += 1  # Increment counter
                i += 1
                if skipped_tokens <= 10:  # Show first 10 skipped
                    print(f"[Parse] Skipped token at {i-1}: {note_tokens[i-1]} (not in expected ranges)")
            
            # Show token distribution
            print(f"\n[Token Distribution] Top 10 most common tokens:")
            sorted_tokens = sorted(token_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
            for token, count in sorted_tokens:
                token_type = "unknown"
                if TIME_RANGE[0] <= token < TIME_RANGE[1]:
                    token_type = f"time (beat {token/48.0:.2f})"
                elif NOTE_RANGE[0] <= token < NOTE_RANGE[1]:
                    token_type = "note"
                elif token == EOS:
                    token_type = "EOS"
                elif token == PAD:
                    token_type = "PAD"
                print(f"  Token {token}: {count} times ({token_type})")
            
            print(f"\n[Generate] Results:")
            print(f"- Created {len(all_notes)} notes")
            print(f"- Skipped {skipped_tokens} non-note tokens")  # Print skipped tokens count
            if all_notes:
                print(f"- Time range: {all_notes[0]['time']:.2f} to {all_notes[-1]['time']:.2f} beats")
            else:
                print("[Warning] No valid note pairs found in generation")
            return all_notes
            
        except Exception as e:
            print(f"[Error] Generation failed: {str(e)}")
            return []


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
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path)
    
    # Prepare audio
    audio = prepare_audio(args.audio_path, args.duration)
    
    # Generate map
    notes = generate_map(
        model,
        tokenizer,
        audio, 
        args.bpm,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        min_p=args.min_p,
        duration=args.duration  # Pass duration to generate_map
    )
    
    # Convert to v3 format
    v3_map = convert_to_v3_format(notes, args.bpm)
    
    # Save output
    with open(args.output, 'w') as f:
        json.dump(v3_map, f, indent=2)
    
    print(f"[Done] Saved v3 format map to {args.output}")
    
    # Also save debug info
    debug_path = Path(args.output).with_suffix('.debug.txt')
    with open(debug_path, 'w') as f:
        f.write(f"Generated {len(notes)} notes\n")
        f.write(f"BPM: {args.bpm}\n")
        f.write(f"Audio duration: {len(audio)/AUDIO_TOKENIZER.sample_rate:.1f}s\n")
        if notes:
            f.write(f"First note: {notes[0]}\n")
            f.write(f"Last note: {notes[-1]}\n")
            f.write(f"\nV3 format first note:\n")
            f.write(json.dumps(v3_map["colorNotes"][0], indent=2))


if __name__ == "__main__":
    main() 