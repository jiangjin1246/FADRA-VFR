#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch


DEFAULT_RESIDUAL_HEAD = Path("./checkpoints/FADRA/residual_head.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count parameters saved in a residual_head checkpoint."
    )
    parser.add_argument(
        "--residual_head_path",
        type=Path,
        default=DEFAULT_RESIDUAL_HEAD,
        help="Path to residual_head checkpoint.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Print only the largest N tensors by parameter count. Defaults to all tensors.",
    )
    parser.add_argument(
        "--show_all",
        action="store_true",
        help="Print every tensor. This is already the default unless --topk is set.",
    )
    return parser.parse_args()


def unwrap_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "residual_head" in obj:
        obj = obj["residual_head"]

    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict-like checkpoint, got {type(obj).__name__}")

    state_dict = {key: value for key, value in obj.items() if torch.is_tensor(value)}
    if not state_dict:
        raise ValueError("No tensor parameters found in checkpoint.")
    return state_dict


def iter_tensor_stats(state_dict: Dict[str, torch.Tensor]) -> Iterable[Tuple[str, Tuple[int, ...], int, str]]:
    for name, tensor in state_dict.items():
        yield name, tuple(tensor.shape), tensor.numel(), str(tensor.dtype).replace("torch.", "")


def format_count(count: int) -> str:
    return f"{count:,}"


def format_millions(count: int) -> str:
    return f"{count / 1_000_000:.3f}M"


def main() -> None:
    args = parse_args()
    checkpoint_path = args.residual_head_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Residual head checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = unwrap_state_dict(checkpoint)

    tensor_stats = list(iter_tensor_stats(state_dict))
    total_params = sum(numel for _, _, numel, _ in tensor_stats)
    trainable_params = total_params
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())

    print(f"checkpoint: {checkpoint_path}")
    print(f"num_tensors: {len(tensor_stats)}")
    print(f"total_params: {format_count(total_params)} ({format_millions(total_params)})")
    print(f"trainable_params: {format_count(trainable_params)} ({format_millions(trainable_params)})")
    print(f"checkpoint_tensor_size: {total_bytes / 1024 / 1024:.2f} MiB")

    by_prefix: Dict[str, int] = {}
    for name, _, numel, _ in tensor_stats:
        prefix = name.split(".", 1)[0]
        by_prefix[prefix] = by_prefix.get(prefix, 0) + numel

    print("\nparams_by_top_prefix:")
    for prefix, numel in sorted(by_prefix.items(), key=lambda item: item[1], reverse=True):
        print(f"  {prefix:24s} {format_count(numel):>16s}  {format_millions(numel):>10s}")

    print("\nparameter_names:")
    rows = sorted(tensor_stats, key=lambda item: item[2], reverse=True)
    if args.topk is not None and not args.show_all:
        rows = rows[: args.topk]
    for name, shape, numel, dtype in rows:
        print(f"  {format_count(numel):>16s}  {dtype:10s}  {str(shape):24s}  {name}")


if __name__ == "__main__":
    main()
