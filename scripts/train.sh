#!/usr/bin/env bash
# FADRA training (paper setting: 2x A100, bs 32, lr 1e-4, 10k iters).
# Usage: bash scripts/train.sh <VBFR-JJ-final dir> [num_gpus]
set -e
cd "$(dirname "$0")/.."
DATA=${1:-data/VBFR-JJ-final}
NGPU=${2:-2}

torchrun --nproc_per_node=$NGPU train.py \
    --dataset $DATA \
    --latent_cache_dir $DATA/cache_vae_latents_train \
    --val_samples 5 \
    --control_lora \
    --gradient_checkpointing \
    --input_lr_scale 5 \
    --learning_rate 1e-4 \
    --train_batch_size 32 \
    --lora_rank 128 \
    --shift 1.0 \
    --max_train_steps 10000 \
    --checkpointing_steps 100 \
    --val_steps 50 \
    --val_vis \
    --use_freq_loss \
    --freq_loss_weight 1.0 \
    --residual_head \
    --residual_dit_blocks 1 \
    --wandb_run_name FADRA
    # add --wandb --wandb_project <name> to log to Weights & Biases
