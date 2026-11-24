import argparse
import json
import os
import random
from glob import glob
from pathlib import Path

import numpy as np
import torch
import decord
from torchvision.transforms import v2
from tqdm import tqdm

from utils.dataset import get_resolution, BUCKETS
from wan.modules.vae import WanVAE
import kornia.augmentation as K

decord.bridge.set_bridge("torch")


# --- LQ synthesis (same as train.py): bilinear down/up-sampling + Gaussian blur, Gaussian noise, JPEG ---
# One set of degradation parameters is sampled per chunk and shared by all frames.
def preprocess_control(control):
    # make sure the input is (B, F, C, H, W)
    if control.shape[0] == 3 and control.shape[1] != 3:
        control = control.movedim(0, 1).unsqueeze(0)  # CFHW -> BFCHW
    height, width = control.shape[-2:]
    kernel_size_ = random.choice(range(3, 27, 2))  # odd kernel size in [3, 25]
    blur_strength = random.uniform(1.5,6)
    resize_ratio   = random.uniform(1.0,8.0)
    down_h = max(1, int(round(height / resize_ratio)))
    down_w = max(1, int(round(width / resize_ratio)))
    noise_sigma    = random.uniform(0.01,0.05)
    random.uniform(20, 35)  # unused (former CRF value); kept so the RNG stream matches the paper code

    degrade = v2.Compose([
        v2.Resize(size=(down_h, down_w)),
        v2.Resize(size=(height, width)),
        v2.GaussianBlur(kernel_size=kernel_size_, sigma=blur_strength),
    ])

    degraded = degrade(control)
    degraded = torch.clamp(torch.nan_to_num(degraded), -1, 1)

    x01 = (degraded * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
    orig_dtype = x01.dtype
    x01 = x01.to(torch.float32)               # kornia does not support bfloat16
    x01 = v2.GaussianNoise(mean=0.0, sigma=noise_sigma, clip=True)(x01)
    # JPEG compression on GPU
    jpeg = K.RandomJPEG(jpeg_quality=(25.0, 85.0), p=1.0, keepdim=True).to(x01.device)
    y01 = jpeg(x01[0])  # (F,3,H,W) float in [0,1]
    degraded = (y01 * 2.0 - 1.0).clamp(-1, 1).to(orig_dtype)  # [0,1] -> [-1,1]

    degraded = torch.clamp(torch.nan_to_num(degraded), -1, 1)

    degraded = degraded.movedim(0,1)   # (F,C,H,W) -> (C,F,H,W)
    return degraded


def center_crop_and_resize(frames_fhwc: torch.Tensor, target_size, device):
    """
    frames_fhwc: [F,H,W,3] uint8/float tensor in [0,255]
    return: CFHW float tensor in [-1,1] on device
    """
    F_, H, W, _ = frames_fhwc.shape
    target_h, target_w = target_size
    pixels_ar = W / H
    target_ar = target_w / target_h
    if pixels_ar > target_ar:
        crop_w = min(int(H * target_ar), W)
        crop_h = H
    elif pixels_ar < target_ar:
        crop_w = W
        crop_h = min(int(W / target_ar), H)
    else:
        crop_w, crop_h = W, H

    
    # convert to expected dtype, resolution, shape, and value range
    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomCrop(size=(crop_h, crop_w)),
        v2.Resize(size=(target_h, target_w)),
    ])
    
    frames = frames_fhwc.movedim(3, 1).unsqueeze(0).contiguous() # FHWC -> FCHW -> BFCHW
    frames = transform(frames) * 2 - 1
    frames = torch.clamp(torch.nan_to_num(frames), min=-1, max=1)
    frames = frames.squeeze(0).permute(1, 0, 2, 3)  # BFCHW -> FCHW -> CFHW
    frames = frames.to(device=device, dtype=torch.bfloat16)
    return frames


def iter_videos(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    exts = [".mp4", ".mkv", ".mov", ".avi", ".webm"]
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(input_path, f"**/*{ext}"), recursive=True))
    return sorted(files)


def save_chunk(base_path, latent, meta, args):
    """
    base_path: full path without extension
    - pt:  torch.save(dict) with the latent and its metadata
    - npy: latent.npy + latent.json (metadata)
    """
    to_save = latent.to(torch.float16) if args.save_fp16 else latent
    if args.save_format == "pt":
        path = base_path + ".pt"
        torch.save({**meta, "latent": to_save.cpu()}, path)
        print(f"[saved] {path}")
    elif args.save_format == "npy":
        npy_path = base_path + ".npy"
        json_path = base_path + ".json"
        np.save(npy_path, to_save.cpu().numpy())
        with open(json_path, "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[saved] {npy_path} (latent) + {json_path} (meta)")
    else:
        raise ValueError(f"Unknown save_format: {args.save_format}")


def process_video(video_path, args, vae, device):
    vr = decord.VideoReader(video_path)
    orig_h, orig_w = vr[0].shape[0], vr[0].shape[1]
    buckets = BUCKETS[args.bucket]
    target_w, target_h = get_resolution(orig_w, orig_h, buckets)
    orig_frames = len(vr)
    stride = args.frame_stride
    seg_len = min(stride * args.chunk_frames, orig_frames)
    if seg_len <= 0:
        return Path(video_path).stem, []

    max_start = max(orig_frames - seg_len, 0)
    start_list = list(range(max_start + 1))
    if args.random_chunks_per_video is not None:
        k = min(args.random_chunks_per_video, len(start_list))
        start_list = random.sample(start_list, k)
        start_list.sort()
    chunk_ids = list(range(len(start_list)))
    manifest_entries = []
    for batch_start in range(0, len(start_list), args.chunk_batch):
        batch_starts = start_list[batch_start:batch_start + args.chunk_batch]
        batch_chunk_ids = chunk_ids[batch_start:batch_start + args.chunk_batch]

        hq_batch = []
        metas_batch = []
        for chunk_id, start_frame in zip(batch_chunk_ids, batch_starts):
            idx = list(range(start_frame, start_frame + seg_len, stride))
            max_frames = ((len(idx) - 1) // 4) * 4 + 1
            idx = idx[:max_frames]
            if len(idx) < args.chunk_frames and args.drop_incomplete:
                continue
            frames = vr.get_batch(idx)  # [F,H,W,3] torch tensor
            hq_pixels = center_crop_and_resize(frames, (target_h, target_w), device)

            base_meta = {
                "frame_indices": idx,
                "orig_video": video_path,
                "orig_size": (orig_w, orig_h),
                "bucket_size": (target_w, target_h),
                "chunk_frames": len(idx),
                "chunk_id": chunk_id,
                "start_frame": start_frame,
                "frame_stride": stride,
            }
            hq_batch.append(hq_pixels)
            metas_batch.append(base_meta)

        if not hq_batch:
            continue

        with torch.inference_mode():
            hq_latents = vae.encode(hq_batch)

        batch_entries = []
        for hq_pixels, hq_latent, base_meta in zip(hq_batch, hq_latents, metas_batch):
            chunk_id = base_meta["chunk_id"]
            hq_base = os.path.join(args.output_dir, f"{Path(video_path).stem}_chunk{chunk_id:04d}_hq")
            save_chunk(hq_base, hq_latent, base_meta, args)
            ext = ".pt" if args.save_format == "pt" else ".npy"
            hq_name = os.path.basename(hq_base + ext)

            batch_entries.append({
                "chunk_id": chunk_id,
                "hq": hq_name,
                "lq": [],
            })

        for vid in range(args.lq_variants):
            lq_batch = []
            lq_meta_batch = []
            for hq_pixels, base_meta in zip(hq_batch, metas_batch):
                lq_batch.append(preprocess_control(hq_pixels))
                lq_meta_batch.append({**base_meta, "lq_variant": vid})

            with torch.inference_mode():
                lq_latents = vae.encode(lq_batch)

            for entry, lq_latent, lq_meta in zip(batch_entries, lq_latents, lq_meta_batch):
                chunk_id = entry["chunk_id"]
                lq_base = os.path.join(args.output_dir, f"{Path(video_path).stem}_chunk{chunk_id:04d}_lqv{vid:02d}")
                save_chunk(lq_base, lq_latent, lq_meta, args)
                lq_name = os.path.basename(lq_base + ext)
                entry["lq"].append(lq_name)

        manifest_entries.extend(batch_entries)

    return Path(video_path).stem, manifest_entries


def main():
    parser = argparse.ArgumentParser(description="Read HD video, degrade to LQ, encode both with VAE, and cache latents.")
    parser.add_argument("--input", required=True, help="Input video file or folder.")
    parser.add_argument("--output_dir", required=True, help="Output folder for the VAE latents and manifest.json.")
    parser.add_argument("--vae_ckpt", default="./models/vae/Wan2.1_VAE.pth", help="Wan VAE checkpoint path.")
    parser.add_argument("--bucket", type=int, default=960, choices=sorted(BUCKETS.keys()), help="Resolution bucket.")
    parser.add_argument("--chunk_frames", type=int, default=21, help="Frames per chunk (4n+1).")
    parser.add_argument("--frame_stride", type=int, default=1, help="Frame sampling stride.")
    parser.add_argument("--drop_incomplete", action="store_true", help="Drop chunks shorter than --chunk_frames.")
    parser.add_argument("--save_fp16", action="store_true", help="Store latents in fp16 to save space.")
    parser.add_argument("--save_format", choices=["pt", "npy"], default="pt", help="Output format: pt (dict) or npy (+ json metadata).")
    parser.add_argument("--lq_variants", type=int, default=1, help="Number of random LQ degradations per HQ chunk.")
    parser.add_argument("--chunk_batch", type=int, default=4, help="Chunks encoded per VAE batch.")
    parser.add_argument("--random_chunks_per_video", type=int, default=None, help="Random chunks per video (default: every possible start frame).")
    parser.add_argument("--device", default=None, help="e.g. cuda:0 (default: first available GPU).")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    vae = WanVAE(vae_pth=args.vae_ckpt, dtype=torch.bfloat16, device=device)
    os.makedirs(args.output_dir, exist_ok=True)

    manifest = {}
    for video in tqdm(iter_videos(args.input), desc="videos"):
        video_key, entries = process_video(video, args, vae, device)
        manifest[video_key] = entries

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[saved] {manifest_path}")


if __name__ == "__main__":
    main()
