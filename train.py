"""
Beat Saber AI training script using Qwen3-0.6B with Unsloth optimization.
"""

import argparse

import unsloth  # Must be first for patching
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

from data import BeatsDataset, audio_masking_collator
from maps import ZippedBeatSaberDataset

from config import (
    VOCAB_SIZE, MAX_SEQ_LENGTH,
    DEFAULT_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LEARNING_RATE, DEFAULT_WARMUP_STEPS,
    DEFAULT_SAVE_STEPS, DEFAULT_MAX_STEPS,
    DEFAULT_OUTPUT_DIR, CHUNK_DURATION, get_logger
)

import torch

logger = get_logger(__name__)


def train(args: argparse.Namespace) -> None:
    """Main training function."""
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available! This model requires GPU for training.")
        return
        
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-0.6B-Base-bnb-4bit",
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        full_finetuning=True,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    
    logger.info("Loaded model: unsloth/Qwen3-0.6B-Base-bnb-4bit")
    logger.info(f"Using first {VOCAB_SIZE} tokens for Beat Saber data")
    
    # Create dataset
    dataset = BeatsDataset(
        ZippedBeatSaberDataset(args.dataset_dir),
        chunk_duration=CHUNK_DURATION,
        max_seq_length=args.max_seq_length
    )
    
    # Limit samples if needed
    if args.max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))
    
    logger.info(f"Dataset ready with {len(dataset)} samples")

    # Train (using SFTTrainer like in the working example)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=lambda batch: audio_masking_collator(batch),
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accumulation,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            logging_steps=1,
            save_steps=args.save_steps,
            save_total_limit=4,
            optim=args.optimizer,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.scheduler,
            max_grad_norm=args.max_grad_norm,  # Gradient clipping
            seed=3407,
            bf16=True,
            remove_unused_columns=False,
            dataloader_drop_last=False,
        )
    )

    logger.info("Starting training...")
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error(f"OOM Error! Try reducing --max_seq_length (current: {args.max_seq_length}) or batch size")
            logger.error("Full error: " + str(e))
            raise
        else:
            raise
    
    # Save (trainer.save_model saves both model and tokenizer)
    trainer.save_model(args.output_dir)
    logger.info(f"Model and tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Train Beat Saber AI model")
    
    # Data arguments
    parser.add_argument("dataset_dir", type=str, help="Directory containing .zip map files")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of training samples")
    
    # Model arguments
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH, help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS, help="Max training steps (overrides epochs)")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_WARMUP_STEPS, help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_STEPS, help="Save checkpoint every N steps")
    
    # Advanced training options
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Per-device batch size")
    parser.add_argument("--grad_accumulation", type=int, default=GRADIENT_ACCUMULATION_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--optimizer", type=str, default="adamw_8bit", choices=["adamw_8bit", "adamw_torch", "sgd"], help="Optimizer to use")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--scheduler", type=str, default="linear", choices=["linear", "cosine", "constant"], help="Learning rate scheduler")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not Path(args.dataset_dir).exists():
        logger.error(f"Dataset directory does not exist: {args.dataset_dir}")
        parser.print_help()
        exit(1)
    
    if args.max_seq_length < 512:
        logger.error("max_seq_length must be at least 512")
        parser.print_help()
        exit(1)
    
    if args.batch_size < 1:
        logger.error("batch_size must be at least 1")
        parser.print_help()
        exit(1)
    
    if args.learning_rate <= 0:
        logger.error("learning_rate must be positive")
        parser.print_help()
        exit(1)
    
    try:
        train(args)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        exit(1) 