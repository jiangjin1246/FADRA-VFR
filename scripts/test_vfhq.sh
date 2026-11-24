#!/usr/bin/env bash
# Paired evaluation: folder with <clip>_hq.mp4 / <clip>_lq.mp4 pairs.
# Usage: bash scripts/test_vfhq.sh <input dir> <output dir>
set -e
cd "$(dirname "$0")/.."
IN=${1:?input dir}
OUT=${2:-outputs/vfhq}

python inference_paired.py \
    --lora checkpoints/FADRA/wan-lora.safetensors \
    --patch_embedding checkpoints/FADRA/patch_embedding.pt \
    --residual_head_path checkpoints/FADRA/residual_head.pt \
    --control_video_dir $IN \
    --output_dir $OUT \
    --width 512 --height 512 --steps 30 --cfg 0.0
