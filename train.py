"""
Beat Saber AI training script using Qwen3-0.6B with Unsloth optimization.
"""

import unsloth  # Must be first for patching
from unsloth import FastLanguageModel

import argparse
from datasets import Dataset as HFDataset
from transformers import TrainingArguments
from trl import SFTTrainer

from data import BeatsDataset, audio_masking_collator, VOCAB_SIZE
from maps import ZippedBeatSaberDataset

import torch


def train(args):
    """Main training function."""
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("[Error] CUDA not available! This model requires GPU for training.")
        return
        
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        full_finetuning=True,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    
    print(f"[Model] Loaded {args.model_name}")
    print(f"[Model] Using first {VOCAB_SIZE} tokens for Beat Saber data")
    
    # Create dataset
    dataset = BeatsDataset(
        ZippedBeatSaberDataset(args.dataset_dir),
        chunk_duration=30.0,
        max_seq_length=args.max_seq_length
    )
    
    # Convert to HuggingFace dataset
    n_samples = min(args.max_samples or len(dataset), len(dataset))
    print(f"[Dataset] Loading {n_samples} samples...")
    
    sequences = [dataset[i] for i in range(n_samples)]
    # sequences = List[Dict[str, tensor]] - tokenized training sequences
    hf_dataset = HFDataset.from_dict({"sequences": sequences})
    
    if not sequences:
        print("[Error] No samples found! Check your dataset directory.")
        return
    
    print(f"[Dataset] Ready with {len(sequences)} samples")

    # Train
    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_dataset,
        data_collator=audio_masking_collator,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            logging_steps=1,
            save_steps=args.save_steps,
            save_total_limit=4,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            bf16=True,
            remove_unused_columns=False,
            dataloader_drop_last=False,
        )
    )

    print("[Training] Starting...")
    trainer.train()
    
    # Save (trainer.save_model saves both model and tokenizer)
    trainer.save_model(args.output_dir)
    print(f"[Done] Model and tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument("dataset_dir", type=str, help="Directory containing .zip map files")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of training samples")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-0.6B-Base", help="Base model to finetune")
    parser.add_argument("--max_seq_length", type=int, default=32768, help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./beatsaber_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=500, help="Max training steps (overrides epochs)")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    
    args = parser.parse_args()
    train(args) 