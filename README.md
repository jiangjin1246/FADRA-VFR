<div align="center">

# FADRA: Frequency-Aware Diffusion with Residual Adaptation for Video Face Restoration

[Jin Jiang](https://github.com/jiangjin1246), Jia Wang, Panwen Hu, Weiran Zhao, Shengcai Liao

[![arXiv](https://img.shields.io/badge/arXiv-2607.06389-b31b1b.svg)](https://arxiv.org/abs/2607.06389)
[![GitHub](https://img.shields.io/badge/GitHub-FADRA--VFR-black?logo=github)](https://github.com/jiangjin1246/FADRA-VFR)
[![Weights](https://img.shields.io/badge/🤗%20Weights-FADRA-yellow)](https://huggingface.co/JinJiang/FADRA-VFR)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://fadra-vfr.github.io/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-VBFR--JJ-blue)](https://huggingface.co/datasets/JinJiang/VBFR-JJ-final)

</div>

## 📣 Updates

- **2026.11** Training and inference code released.

## 🔎 Overview

FADRA builds on the frozen **Wan2.1-T2V-1.3B** video diffusion transformer and adapts it to video face restoration with three components:

- **LQ Pixel-Alignment Feature Fusion**: the LQ latent is concatenated channel-wise with the noisy latent; the input 3D patch-embedding conv of Wan is expanded with zero-initialized weights.
- **Repeated Residual Adaptation Head (RRAH)**: a lightweight DiT block that takes the predicted velocity and the LQ latent and predicts a residual correction at **every** sampling step.
- **Frequency-Aware Loss (FAL)**: an 8×8 block-DCT loss on the predicted clean latent `z`, weighted by the JPEG luminance quantization table.

LoRA (rank 128) is applied to the self/cross-attention layers of the DiT and to the expanded patch embedding.

| Module | Params |
|---|---|
| Wan2.1-T2V-1.3B (frozen) | 1419.0M |
| LoRA | 94.3M |
| Feature fusion | 0.2M |
| RRAH | 72.3M |

## ⚒️ Installation

```bash
git clone https://github.com/jiangjin1246/FADRA-VFR.git
cd FADRA-VFR

conda create -n fadra python=3.10 -y
conda activate fadra

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# optional, faster attention
# pip install flash-attn --no-build-isolation
```

`ffmpeg` is required for video I/O.

Tested environment: Python 3.10, PyTorch 2.8.0 + CUDA 12.8, NVIDIA A100 80GB. `requirements.txt` pins the package versions we used.

## 📦 Download Weights

### 1. Wan2.1 base models (official, ~17 GB)

FADRA uses the official Wan2.1 weights without modification. Download them into `./models`:

```bash
bash scripts/download_models.sh
```

This fetches:

| File | Source | Local path |
|---|---|---|
| DiT (T2V-1.3B) | [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | `models/Wan-AI/Wan2.1-T2V-1.3B/` |
| VAE | `Wan2.1_VAE.pth` from [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | `models/vae/Wan2.1_VAE.pth` |
| UMT5-XXL text encoder | `models_t5_umt5-xxl-enc-bf16.pth` + `google/umt5-xxl` tokenizer | `models/UMT5/` |

The UMT5 encoder is only needed to (re)compute prompt embeddings; the default face prompt embedding is already provided in `embeddings/face-prompt_wan.safetensors`.

If you are in mainland China, set `export HF_ENDPOINT=https://hf-mirror.com` first.

### 2. FADRA checkpoints

Download from [🤗 JinJiang/FADRA-VFR](https://huggingface.co/JinJiang/FADRA-VFR) into `./checkpoints/FADRA`:

```bash
huggingface-cli download JinJiang/FADRA-VFR --local-dir checkpoints/FADRA
```

### Folder layout

```
FADRA
├── checkpoints/FADRA/
│   ├── wan-lora.safetensors        # LoRA (DiT + expanded patch embedding)
│   ├── patch_embedding.pt          # expanded 3D patch embedding (LQ fusion)
│   └── residual_head.pt            # RRAH
├── embeddings/                     # pre-computed UMT5 prompt embeddings
├── models/                         # Wan2.1 official weights (see above)
│   ├── Wan-AI/Wan2.1-T2V-1.3B/
│   ├── vae/Wan2.1_VAE.pth
│   └── UMT5/
├── train.py
├── inference.py                    # LQ videos -> restored videos
├── inference_paired.py             # paired HQ/LQ videos
├── examples/lq/                    # 7 example inputs from the project page
└── tools/                          # latent caching, parameter counting
```

## 🚀 Inference

### Quick start with the provided examples

`examples/lq/` contains the 7 low-quality inputs shown on our [project page](https://fadra-vfr.github.io/) (3 × VFHQ-test, 2 × CelebV-HQ, 2 × real-world movie clips):

```bash
python inference.py \
    --control_video_dir ./examples/lq \
    --output_dir ./outputs/examples \
    --width 512 --height 512 --steps 30 --cfg 0.0
```

### Your own videos

Put low-quality face videos (`.mp4`) into a folder and pass it with `--control_video_dir`. Checkpoint paths default to `./checkpoints/FADRA/*`; override them with `--lora`, `--patch_embedding` and `--residual_head_path` (the scripts stop with an error if a checkpoint is missing).

For every input `xxx.mp4` the script writes `xxx_pred.mp4` (restored), `xxx_lq.mp4` (resized input) and `xxx_all.mp4` (side-by-side LQ | output). Each clip is restored in a single pass over all of its frames.

Notes:

- **Input faces must be pre-cropped and roughly centered** (square crops, as in VFHQ / CelebV-HQ); videos are resized to `--width` × `--height` (512×512 in the paper). This repository does not include a face detection / alignment step.
- **GPU memory**: all frames of a clip are processed together, so memory grows with clip length. Our experiments (up to 161 frames at 512×512) were run on an A100 80GB; split long videos into shorter clips if you run out of memory.

### Paired HQ/LQ videos

If you have ground-truth videos, `inference_paired.py` takes a folder of `<clip>_hq.mp4` / `<clip>_lq.mp4` pairs and additionally saves the HQ reference next to the result:

```bash
bash scripts/test_vfhq.sh /path/to/pairs ./outputs/pairs
```

## 🏋️ Training

### 1. Data

We train on 15,127 high-quality face clips curated from VFHQ and HDTF, released as [🤗 VBFR-JJ-final](https://huggingface.co/datasets/JinJiang/VBFR-JJ-final):

```bash
mkdir -p data && cd data
huggingface-cli download JinJiang/VBFR-JJ-final --repo-type dataset --local-dir VBFR-JJ-final
cd VBFR-JJ-final && unzip VBFR-JJ-final.zip && cd ../..
```

```
VBFR-JJ-final
├── manifest.json
├── train/*.mp4
├── test/*.mp4
├── text-embedding/face-prompt_wan.safetensors
└── cache_vae_latents_train/          # generated in step 2
    ├── manifest.json
    ├── xxx_chunk0000_hq.pt
    ├── xxx_chunk0000_lqv00.pt
    └── ...
```

### 2. Cache VAE latents

Training reads pre-encoded latents. For every training video we sample 3 random 21-frame chunks and, for each chunk, encode the HQ frames plus 6 randomly degraded LQ versions (bilinear down/up-sampling ×1–8, Gaussian blur k∈[3,25] σ∈[1.5,6], Gaussian noise σ∈[0.01,0.05], JPEG q∈[25,85]; parameters shared across frames of a chunk):

```bash
bash scripts/cache_latents.sh data/VBFR-JJ-final
```

This writes `data/VBFR-JJ-final/cache_vae_latents_train/*.pt` together with a `manifest.json` that `train.py` picks up automatically. The cache takes roughly 500 GB for the full training set; the degradations are random, so the cache will not be bit-identical to ours.

### 3. Train

Default setting in the paper: 2× A100, batch size 32 per GPU, lr 1e-4, 10k iterations.

```bash
bash scripts/train.sh data/VBFR-JJ-final
```

Key flags:

| Flag | Meaning |
|---|---|
| `--control_lora` | enable LQ feature fusion (expanded patch embedding) |
| `--lora_rank 128` | LoRA rank |
| `--residual_head --residual_dit_blocks 1` | enable RRAH with one DiT block |
| `--use_freq_loss --freq_loss_weight 1.0` | enable FAL |

Checkpoints are written to `outputs/<run_name>/checkpoints/` (`wan-lora-*.safetensors`, `patch_embedding-32-*.pt`, `residual_head-*.pt`, and `latest_training_state.pt` for resuming with `--resume_dir`).

## 📊 Results

| Dataset | PSNR↑ | SSIM↑ | LPIPS↓ | IDD↓ | FVD↓ |
|---|---|---|---|---|---|
| VFHQ-test | 29.95 | 0.8576 | 0.2378 | 0.2912 | 38.97 |
| CelebV-HQ | 28.70 | 0.8326 | 0.2421 | 0.3103 | 48.32 |

## 🙏 Acknowledgements

This code is built on [Wan2.1](https://github.com/Wan-Video/Wan2.1) and [WanTraining](https://github.com/spacepxl/WanTraining). We thank the authors for sharing their code.

## 📖 Citation

```bibtex
@article{jiang2026fadra,
  title   = {FADRA: Frequency-Aware Diffusion with Residual Adaptation for Video Face Restoration},
  author  = {Jiang, Jin and Wang, Jia and Hu, Panwen and Zhao, Weiran and Liao, Shengcai},
  journal = {arXiv preprint arXiv:2607.06389},
  year    = {2026}
}
```

## 📜 License

This project is released under the [Apache 2.0 License](LICENSE).
