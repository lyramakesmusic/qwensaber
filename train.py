"""
Beat Saber AI training script using Qwen3-0.6B with Unsloth optimization.
"""

import unsloth  # Must be first for patching
from unsloth import FastLanguageModel

import argparse
from pathlib import Path
import numpy as np

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from data import CachedDataset, ChunkedDataset, collate_sequences, VOCAB_SIZE
from maps import ZippedBeatSaberDataset


def setup_model(model_name="unsloth/Qwen3-0.6B-Base", max_seq_length=32768):
    """Load and setup the model for training."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        full_finetuning=True,
        trust_remote_code=True,
    )
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    print(f"[Model] Loaded {model_name}")
    print(f"[Model] Using first {VOCAB_SIZE} tokens for Beat Saber data")
    print(f"[Model] Gradient checkpointing enabled for memory efficiency")
    
    # Save tokenizer config and files
    tokenizer.save_pretrained("./beatsaber_model")
    print(f"[Model] Saved tokenizer files to ./beatsaber_model")
    
    return model, tokenizer


def prepare_dataset(dataset_dir, max_samples=None, max_length=32768):
    """Load and prepare the Beat Saber dataset."""
    # Load raw data
    raw_dataset = ZippedBeatSaberDataset(dataset_dir)
    
    # Add caching for expensive audio encoding
    cached_dataset = CachedDataset(raw_dataset)
    
    # Create chunked dataset - use 30-second chunks for proper training
    # Each Encodec frame is ~10ms (240 samples at 24kHz), giving 100Hz frame rate
    # 30 seconds = 3000 audio frames + notes = reasonable sequence length
    chunk_duration = 30.0  # 30 seconds
    print(f"[Dataset] Using {chunk_duration}-second audio chunks (max sequence length: {max_length})")
    
    chunked_dataset = ChunkedDataset(cached_dataset, chunk_duration=chunk_duration, max_seq_length=max_length)
    
    # Convert to list of sequences for HuggingFace
    sequences = []
    lengths = []  # Track sequence lengths
    n_samples = len(chunked_dataset) if max_samples is None else min(max_samples, len(chunked_dataset))
    
    print(f"[Dataset] Preparing {n_samples} samples...")
    
    # Debug first sequence to verify structure
    first_sequence_shown = False
    
    for i in range(n_samples):
        sequence = chunked_dataset[i]
        sequences.append(sequence)
        lengths.append(len(sequence))
        
        # Show structure of first sequence only
        if not first_sequence_shown and len(sequence) > 0:
            from data import BOS, AUDIO_START, AUDIO_END, NOTES_START, EOS
            
            notes_start_idx = None
            for idx, token in enumerate(sequence):
                if token == NOTES_START:
                    notes_start_idx = idx
                    break
            
            if notes_start_idx:
                audio_tokens = notes_start_idx - 2  # subtract BOS and AUDIO_START
                note_tokens = len(sequence) - notes_start_idx - 2  # subtract NOTES_START and EOS
            
                print(f"[Dataset] First sequence structure ({len(sequence)} tokens):")
                print(f"  - Audio tokens: {audio_tokens}")
                print(f"  - Note tokens: {note_tokens}")
                print(f"  - NOTES_START at index: {notes_start_idx}")
            first_sequence_shown = True
        
        if (i + 1) % 10 == 0:
            print(f"[Dataset] Processed {i+1}/{n_samples} samples")
    
    # Create HuggingFace dataset
    return Dataset.from_dict({"sequences": sequences})


def train(args):
    """Main training function."""
    # Setup model
    model, tokenizer = setup_model(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length
    )
    
    # Prepare dataset
    dataset = prepare_dataset(
        args.dataset_dir, 
        max_samples=args.max_samples,
        max_length=args.max_seq_length
    )
    
    if len(dataset) == 0:
        print("[Error] No samples found! Check your dataset directory.")
        return
    
    print(f"[Dataset] Ready with {len(dataset)} samples")

    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        data_collator=audio_masking_collator,
        args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            logging_steps=1,
            save_steps=args.save_steps,
            save_total_limit=3,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            bf16=True,
            remove_unused_columns=False,
            dataloader_drop_last=False,
        )
    )

    # Train
    print("[Training] Starting...")
    trainer.train()
    
    # Save final model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)  # Save tokenizer again with final model
    print(f"[Done] Model and tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument("dataset_dir", type=str, help="Directory containing .zip map files")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of training samples")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-0.6B-Base", help="Base model to finetune")
    parser.add_argument("--max_seq_length", type=int, default=32768, help="Maximum sequence length (full context for A100)")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./beatsaber_model", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=500, help="Max training steps (overrides epochs)")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    
    args = parser.parse_args()
    train(args) 