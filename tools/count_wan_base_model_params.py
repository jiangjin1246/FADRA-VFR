#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


DEFAULT_MODEL = Path("./models/Wan-AI/Wan2.1-T2V-1.3B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count parameters in the original Wan diffusion model checkpoint."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Local model directory or checkpoint file. Defaults to ./models/Wan-AI/Wan2.1-T2V-1.3B.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=30,
        help="Print largest N tensors.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(model: Path) -> Path:
    model = model.expanduser().resolve()
    if model.is_file():
        return model
    if not model.is_dir():
        raise FileNotFoundError(f"Model path not found: {model}")

    candidates = [
        model / "diffusion_pytorch_model.safetensors",
        model / "diffusion_pytorch_model.bin",
        model / "pytorch_model.bin",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    sharded = sorted(model.glob("diffusion_pytorch_model-*.safetensors"))
    if sharded:
        raise RuntimeError(
            "This model appears to be sharded. Pass each shard explicitly or extend this script for sharded loading."
        )
    raise FileNotFoundError(f"No diffusion checkpoint found in {model}")


def load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(obj).__name__}")
    return {key: value for key, value in obj.items() if torch.is_tensor(value)}


def iter_stats(state_dict: Dict[str, torch.Tensor]) -> Iterable[Tuple[str, Tuple[int, ...], str, int]]:
    for name, tensor in state_dict.items():
        yield name, tuple(tensor.shape), str(tensor.dtype).replace("torch.", ""), tensor.numel()


def fmt_count(value: int) -> str:
    return f"{value:,}"


def fmt_billions(value: int) -> str:
    return f"{value / 1_000_000_000:.3f}B"


def fmt_millions(value: int) -> str:
    return f"{value / 1_000_000:.3f}M"


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.model)
    state_dict = load_state_dict(checkpoint_path)
    stats = list(iter_stats(state_dict))
    total_params = sum(numel for _, _, _, numel in stats)
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())

    by_prefix: Dict[str, int] = {}
    for name, _, _, numel in stats:
        prefix = name.split(".", 1)[0]
        by_prefix[prefix] = by_prefix.get(prefix, 0) + numel

    print(f"checkpoint: {checkpoint_path}")
    print(f"num_tensors: {len(stats)}")
    print(f"total_params: {fmt_count(total_params)} ({fmt_billions(total_params)} / {fmt_millions(total_params)})")
    print(f"checkpoint_tensor_size: {total_bytes / 1024 / 1024 / 1024:.2f} GiB")

    print("\nparams_by_top_prefix:")
    for prefix, numel in sorted(by_prefix.items(), key=lambda item: item[1], reverse=True):
        print(f"  {prefix:24s} {fmt_count(numel):>18s}  {fmt_billions(numel):>10s}")

    print(f"\nlargest_tensors_top{args.topk}:")
    for name, shape, dtype, numel in sorted(stats, key=lambda item: item[-1], reverse=True)[: args.topk]:
        print(f"  {fmt_count(numel):>18s}  {dtype:10s}  {str(shape):26s}  {name}")


if __name__ == "__main__":
    main()
