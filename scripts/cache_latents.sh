#!/usr/bin/env bash
# Pre-encode HQ / degraded-LQ VAE latents for training.
# Paper setting: 3 random 21-frame chunks per video, 6 random LQ degradations per chunk.
# Writes <out>/*.pt and <out>/manifest.json (read by train.py).
# Usage: bash scripts/cache_latents.sh <VBFR-JJ-final dir> [num_gpus]
set -e
cd "$(dirname "$0")/.."
DATA=${1:-data/VBFR-JJ-final}

PYTHONPATH=. python tools/cache_vae_latents_from_video.py \
    --input $DATA/train \
    --output_dir $DATA/cache_vae_latents_train \
    --bucket 512 \
    --chunk_frames 21 \
    --drop_incomplete \
    --random_chunks_per_video 3 \
    --lq_variants 6 \
    --chunk_batch 6
