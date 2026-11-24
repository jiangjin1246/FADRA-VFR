#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import torch


DEFAULT_PATCH_EMBEDDING = Path("./checkpoints/FADRA/patch_embedding.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count extra-channel parameters in a 32-channel Wan patch_embedding checkpoint."
    )
    parser.add_argument(
        "--patch_embedding",
        type=Path,
        default=DEFAULT_PATCH_EMBEDDING,
        help="Path to patch_embedding checkpoint.",
    )
    parser.add_argument(
        "--base_in_channels",
        type=int,
        default=16,
        help="Original Wan patch_embedding input channels. Extra channels are counted after this index.",
    )
    return parser.parse_args()


def fmt_count(value: int) -> str:
    return f"{value:,}"


def main() -> None:
    args = parse_args()
    checkpoint_path = args.patch_embedding.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"patch_embedding checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "weight" not in state:
        raise KeyError(f"Checkpoint has no 'weight' tensor. Keys: {list(state.keys())}")

    weight = state["weight"]
    bias = state.get("bias")
    if not torch.is_tensor(weight):
        raise TypeError("'weight' is not a tensor")
    if weight.ndim < 2:
        raise ValueError(f"Expected Conv weight with at least 2 dims, got shape={tuple(weight.shape)}")

    out_channels = weight.shape[0]
    total_in_channels = weight.shape[1]
    base_in_channels = args.base_in_channels
    if total_in_channels <= base_in_channels:
        raise ValueError(
            f"total_in_channels={total_in_channels} must be larger than base_in_channels={base_in_channels}"
        )

    extra_weight = weight[:, base_in_channels:, ...]
    total_weight_params = weight.numel()
    base_weight_params = weight[:, :base_in_channels, ...].numel()
    extra_weight_params = extra_weight.numel()
    bias_params = bias.numel() if torch.is_tensor(bias) else 0
    total_checkpoint_params = total_weight_params + bias_params

    print(f"checkpoint: {checkpoint_path}")
    print(f"weight_shape: {tuple(weight.shape)}")
    print(f"weight_dtype: {str(weight.dtype).replace('torch.', '')}")
    print(f"out_channels: {out_channels}")
    print(f"total_in_channels: {total_in_channels}")
    print(f"base_in_channels_wan: {base_in_channels}")
    print(f"extra_in_channels_counted: {total_in_channels - base_in_channels}")
    print()
    print(f"total_checkpoint_params_weight_plus_bias: {fmt_count(total_checkpoint_params)}")
    print(f"total_weight_params: {fmt_count(total_weight_params)}")
    print(f"base_wan_weight_params_not_counted: {fmt_count(base_weight_params)}")
    print(f"extra_weight_params_counted: {fmt_count(extra_weight_params)}")
    print(f"bias_params_not_counted: {fmt_count(bias_params)}")
    print()
    print("counted_parameter_slice:")
    print(f"  weight[:, {base_in_channels}:, ...]  shape={tuple(extra_weight.shape)}  params={fmt_count(extra_weight_params)}")


if __name__ == "__main__":
    main()
