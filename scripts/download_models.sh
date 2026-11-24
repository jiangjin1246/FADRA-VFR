#!/usr/bin/env bash
# Download the official Wan2.1 weights used by FADRA into ./models (~17 GB).
# Mainland China users: export HF_ENDPOINT=https://hf-mirror.com
set -e
cd "$(dirname "$0")/.."

REPO=Wan-AI/Wan2.1-T2V-1.3B

# DiT backbone
huggingface-cli download $REPO config.json diffusion_pytorch_model.safetensors \
    --local-dir models/$REPO

# VAE
huggingface-cli download $REPO Wan2.1_VAE.pth --local-dir models/vae

# UMT5-XXL text encoder + tokenizer (only needed to compute new prompt embeddings)
huggingface-cli download $REPO models_t5_umt5-xxl-enc-bf16.pth --local-dir models/UMT5
huggingface-cli download $REPO --include "google/*" --local-dir models/UMT5
