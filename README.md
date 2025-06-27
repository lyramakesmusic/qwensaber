# Beat Saber AI

Beat Saber map generation from audio based on Qwen-0.6b.

## Features

Creates notes JSON from a given song. Does not create arcs, chains, or lighting. Should handle up to 4 minute songs with no BPM changes.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Unsloth (separate due to custom build)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## Usage

### Training a Model

```bash
python train.py /path/to/beat/saber/maps \
    --max_samples 1000 \
    --output_dir ./my_model \
    --learning_rate 2e-5 \
    --max_steps 500
```

### Generating Maps

```bash
python infer.py ./my_model /path/to/audio.wav \
    --bpm 120 \
    --output generated_map.dat \
    --temperature 0.8 \
    --chunk_duration 30
```

## Project Structure

```
├── train.py           # Training script
├── infer.py           # Inference
├── config.py          # All constants, configuration, and logging
├── data.py            # Dataset, tokenization, and collator
├── maps.py            # Beat Saber map parsing (v2/v3)
├── audio.py           # Audio tokenization (encodec)
└── test.py            # Simple test suite
```

### Training Options

```bash
python train.py maps/ \
    --batch_size 2 \
    --grad_accumulation 4 \
    --optimizer adamw_8bit \
    --scheduler cosine \
    --max_grad_norm 1.0 \
    --weight_decay 0.01
```

### Inference Options

```bash
python infer.py model/ audio.wav \
    --bpm 128 \
    --temperature 0.9 \
    --min_p 0.05 \
    --chunk_duration 45 \
    --max_tokens 1000
```
