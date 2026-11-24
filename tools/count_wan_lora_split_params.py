#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

from safetensors.torch import load_file


DEFAULT_LORA = Path("./checkpoints/FADRA/wan-lora.safetensors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split Wan LoRA params into patch_embedding LoRA and other LoRA params."
    )
    parser.add_argument(
        "--lora",
        type=Path,
        default=DEFAULT_LORA,
        help="Path to wan-lora safetensors checkpoint.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Only print largest N tensors per group. Defaults to all tensors.",
    )
    return parser.parse_args()


def fmt_count(value: int) -> str:
    return f"{value:,}"


def fmt_millions(value: int) -> str:
    return f"{value / 1_000_000:.3f}M"


def group_name(param_name: str) -> str:
    return "patch_embedding_lora" if param_name.startswith("patch_embedding.") else "wan_lora_other"


def iter_stats(state_dict: Dict) -> Iterable[Tuple[str, str, Tuple[int, ...], str, int]]:
    for name, tensor in state_dict.items():
        yield group_name(name), name, tuple(tensor.shape), str(tensor.dtype).replace("torch.", ""), tensor.numel()


def print_group(group: str, rows, topk: int | None) -> None:
    total = sum(row[-1] for row in rows)
    print(f"\n[{group}]")
    print(f"num_tensors: {len(rows)}")
    print(f"params: {fmt_count(total)} ({fmt_millions(total)})")

    rows = sorted(rows, key=lambda row: row[-1], reverse=True)
    if topk is not None:
        rows = rows[:topk]

    print("parameter_names:")
    for _, name, shape, dtype, numel in rows:
        print(f"  {fmt_count(numel):>16s}  {dtype:10s}  {str(shape):24s}  {name}")


def main() -> None:
    args = parse_args()
    lora_path = args.lora.expanduser().resolve()
    if not lora_path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

    state_dict = load_file(str(lora_path), device="cpu")
    stats = list(iter_stats(state_dict))
    total_params = sum(row[-1] for row in stats)
    patch_rows = [row for row in stats if row[0] == "patch_embedding_lora"]
    other_rows = [row for row in stats if row[0] == "wan_lora_other"]
    patch_params = sum(row[-1] for row in patch_rows)
    other_params = sum(row[-1] for row in other_rows)

    print(f"checkpoint: {lora_path}")
    print(f"total_tensors: {len(stats)}")
    print(f"total_lora_params: {fmt_count(total_params)} ({fmt_millions(total_params)})")
    print(f"patch_embedding_lora_params: {fmt_count(patch_params)} ({fmt_millions(patch_params)})")
    print(f"wan_lora_other_params: {fmt_count(other_params)} ({fmt_millions(other_params)})")

    print_group("patch_embedding_lora", patch_rows, args.topk)
    print_group("wan_lora_other", other_rows, args.topk)


if __name__ == "__main__":
    main()
