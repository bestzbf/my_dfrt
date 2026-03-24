#!/usr/bin/env python3
"""Profile PointOdyssey data loading and training-pipeline bottlenecks."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import PointOdysseyDataset, collate_fn


NETWORK_FS_TYPES = {"cifs", "smbfs", "nfs", "nfs4", "fuse.sshfs"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config to seed defaults")

    parser.add_argument("--data-root", type=str, default="/mnt/D4RT/datasets/PointOdyssey")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument("--num-queries", type=int, default=2048)
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument(
        "--patch-provider",
        type=str,
        default="precomputed_highres",
        choices=[
            "auto",
            "sampled_resized",
            "precomputed_resized",
            "sampled_highres",
            "precomputed_highres",
        ],
    )
    parser.add_argument("--query-mode", type=str, default="full")
    parser.add_argument("--t-tgt-eq-t-cam-ratio", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--sample-count", type=int, default=3, help="Number of single-sample profiles to average")
    parser.add_argument("--batch-count", type=int, default=3, help="Number of DataLoader batches to time")
    parser.add_argument("--use-augs", action="store_true", help="Enable training augmentations during profiling")
    parser.add_argument(
        "--disable-motion-boundaries",
        action="store_true",
        help="Disable motion-boundary oversampling during profiling",
    )
    parser.add_argument(
        "--disable-precompute-local-patches",
        action="store_true",
        help="Disable local patch precomputation; only valid for sampled_* providers",
    )
    parser.add_argument(
        "--worker-sweep",
        type=str,
        default="",
        help="Optional comma-separated num_workers sweep, e.g. 0,2,4,8",
    )
    parser.add_argument(
        "--prefetch-sweep",
        type=str,
        default="",
        help="Optional comma-separated prefetch_factor sweep, e.g. 1,2,4",
    )

    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        with open(pre_args.config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        parser.set_defaults(
            **{
                key.replace("-", "_"): value
                for key, value in config.items()
                if any(action.dest == key.replace("-", "_") for action in parser._actions)
            }
        )

    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    if not raw:
        return []
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def resolve_patch_data_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.disable_precompute_local_patches and args.patch_provider in {
        "precomputed_resized",
        "precomputed_highres",
    }:
        raise ValueError(
            f"patch_provider='{args.patch_provider}' requires precomputed local patches, "
            "but --disable-precompute-local-patches was set."
        )
    precompute_local_patches = (
        not args.disable_precompute_local_patches
        and args.patch_provider not in {"sampled_resized", "sampled_highres"}
    )
    return {
        "precompute_local_patches": precompute_local_patches,
        "return_query_video": args.patch_provider == "sampled_highres",
        "local_patch_source": "highres" if args.patch_provider == "precomputed_highres" else "resized",
    }


def build_dataset(args: argparse.Namespace) -> PointOdysseyDataset:
    patch_data = resolve_patch_data_config(args)
    return PointOdysseyDataset(
        dataset_location=args.data_root,
        dset=args.split,
        S=args.num_frames,
        img_size=args.img_size,
        num_queries=args.num_queries,
        patch_size=args.patch_size,
        use_augs=args.use_augs,
        verbose=False,
        sequence_name=args.sequence,
        query_mode=args.query_mode,
        t_tgt_eq_t_cam_ratio=args.t_tgt_eq_t_cam_ratio,
        use_motion_boundaries=not args.disable_motion_boundaries,
        precompute_local_patches=patch_data["precompute_local_patches"],
        return_query_video=patch_data["return_query_video"],
        local_patch_source=patch_data["local_patch_source"],
        return_aux_tensors=False,
    )


def format_seconds(value: float) -> str:
    return f"{value:.3f}s"


def format_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{size:.2f}TB"


def tensor_tree_nbytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_tree_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_tree_nbytes(item) for item in value)
    return 0


def filesystem_info(path: str) -> dict[str, str]:
    target = Path(path).resolve()
    proc = subprocess.run(
        ["df", "-Th", str(target)],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line for line in proc.stdout.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return {"raw": proc.stdout.strip()}
    header = lines[0].split()
    values = lines[1].split()
    row = dict(zip(header, values))
    row["raw"] = lines[1]
    return row


def instrument_dataset_methods(dataset: PointOdysseyDataset, method_names: list[str]):
    stats = defaultdict(float)
    counts = defaultdict(int)
    for name in method_names:
        original = getattr(dataset, name)

        def wrapped(*args, __orig=original, __name=name, **kwargs):
            start = time.perf_counter()
            result = __orig(*args, **kwargs)
            stats[__name] += time.perf_counter() - start
            counts[__name] += 1
            return result

        setattr(dataset, name, wrapped)
    return stats, counts


def profile_single_samples(args: argparse.Namespace) -> tuple[list[float], dict[str, float], dict[str, int], dict[str, Any] | None]:
    dataset = build_dataset(args)
    method_names = [
        "_get_sequence_assets",
        "_load_annotations",
        "_load_rgb",
        "_load_depth",
        "_load_normal_or_mask",
        "_apply_color_aug",
        "_sample_query_data",
        "_compute_boundary_mask",
        "_compute_motion_boundary_mask",
        "extract_patches",
    ]
    stats, counts = instrument_dataset_methods(dataset, method_names)

    sample_times: list[float] = []
    example_sample: dict[str, Any] | None = None
    sample_count = min(max(args.sample_count, 1), len(dataset))
    for index in range(sample_count):
        start = time.perf_counter()
        sample, ok = dataset[index]
        elapsed = time.perf_counter() - start
        sample_times.append(elapsed)
        if not ok:
            raise RuntimeError(f"Dataset sample {index} failed during profiling")
        if example_sample is None:
            example_sample = sample
    return sample_times, dict(stats), dict(counts), example_sample


def build_loader(dataset: PointOdysseyDataset, batch_size: int, num_workers: int, prefetch_factor: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


def profile_dataloader(
    args: argparse.Namespace,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
    prefetch_factor: int | None = None,
    batch_count: int | None = None,
) -> tuple[list[float], dict[str, Any] | None]:
    dataset = build_dataset(args)
    loader = build_loader(
        dataset,
        batch_size=args.batch_size if batch_size is None else batch_size,
        num_workers=args.num_workers if num_workers is None else num_workers,
        prefetch_factor=args.prefetch_factor if prefetch_factor is None else prefetch_factor,
    )
    iterator = iter(loader)
    waits: list[float] = []
    example_batch: dict[str, Any] | None = None
    target_batches = min(max(batch_count or args.batch_count, 1), len(loader))
    for _ in range(target_batches):
        start = time.perf_counter()
        batch = next(iterator)
        waits.append(time.perf_counter() - start)
        if example_batch is None:
            example_batch = batch
    return waits, example_batch


def summarize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            summary[key] = tuple(value.shape)
        elif isinstance(value, dict):
            nested = {}
            for nested_key, nested_value in value.items():
                if torch.is_tensor(nested_value):
                    nested[nested_key] = tuple(nested_value.shape)
            summary[key] = nested
    return summary


def print_sample_profile(
    sample_times: list[float],
    method_stats: dict[str, float],
    method_counts: dict[str, int],
    sample: dict[str, Any] | None,
) -> None:
    print("Single-sample profile:")
    print(f"  samples profiled: {len(sample_times)}")
    print(f"  average total: {format_seconds(sum(sample_times) / max(len(sample_times), 1))}")
    print(f"  min / max: {format_seconds(min(sample_times))} / {format_seconds(max(sample_times))}")
    for name, total in sorted(method_stats.items(), key=lambda item: item[1], reverse=True):
        if method_counts.get(name, 0) == 0:
            continue
        print(
            f"  {name}: avg {format_seconds(total / method_counts[name])} "
            f"over {method_counts[name]} calls"
        )
    if sample is not None:
        print("  example sample tensors:")
        for key, value in sample.items():
            if torch.is_tensor(value):
                print(f"    {key}: {tuple(value.shape)} [{format_bytes(tensor_tree_nbytes(value))}]")
        sample_bytes = tensor_tree_nbytes(sample)
        print(f"  approximate sample tensor bytes: {format_bytes(sample_bytes)}")


def print_dataloader_profile(
    waits: list[float],
    batch: dict[str, Any] | None,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> None:
    print("DataLoader profile:")
    print(f"  batch_size={batch_size}, num_workers={num_workers}, prefetch_factor={prefetch_factor}")
    print(f"  waits: {[round(value, 3) for value in waits]}")
    print(f"  average batch wait: {format_seconds(sum(waits) / max(len(waits), 1))}")
    if batch is not None:
        batch_bytes = tensor_tree_nbytes(batch)
        print(f"  approximate batch tensor bytes: {format_bytes(batch_bytes)}")
        if num_workers > 0:
            in_flight = batch_bytes * num_workers * prefetch_factor
            print(f"  estimated prefetched tensor bytes: {format_bytes(in_flight)}")
        print("  example batch tensors:")
        summary = summarize_batch(batch)
        for key, value in summary.items():
            print(f"    {key}: {value}")


def print_sweeps(args: argparse.Namespace) -> None:
    worker_sweep = parse_int_list(args.worker_sweep)
    if worker_sweep:
        print("Worker sweep (first batch only):")
        for num_workers in worker_sweep:
            waits, _ = profile_dataloader(
                args,
                batch_size=max(1, min(args.batch_size, 2)),
                num_workers=num_workers,
                prefetch_factor=args.prefetch_factor,
                batch_count=1,
            )
            print(f"  num_workers={num_workers}: first_batch_wait={format_seconds(waits[0])}")

    prefetch_sweep = parse_int_list(args.prefetch_sweep)
    if prefetch_sweep:
        print("Prefetch sweep (first batch only):")
        for prefetch_factor in prefetch_sweep:
            waits, _ = profile_dataloader(
                args,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                prefetch_factor=prefetch_factor,
                batch_count=1,
            )
            print(f"  prefetch_factor={prefetch_factor}: first_batch_wait={format_seconds(waits[0])}")


def main() -> None:
    args = parse_args()

    fs_info = filesystem_info(args.data_root)
    fs_type = fs_info.get("Type", "unknown")
    patch_data = resolve_patch_data_config(args)

    print("PointOdyssey Pipeline Profiler")
    print("=" * 60)
    print(f"data_root: {Path(args.data_root).resolve()}")
    print(f"filesystem: {fs_info.get('raw', 'unknown')}")
    print(f"patch_provider: {args.patch_provider}")
    print(f"precompute_local_patches: {patch_data['precompute_local_patches']}")
    print(f"return_query_video: {patch_data['return_query_video']}")
    print(f"local_patch_source: {patch_data['local_patch_source']}")
    print(f"use_motion_boundaries: {not args.disable_motion_boundaries}")
    if fs_type in NETWORK_FS_TYPES:
        print("warning: data_root is on a network filesystem; small-file random IO can dominate training.")
    print("=" * 60)

    sample_times, method_stats, method_counts, example_sample = profile_single_samples(args)
    print_sample_profile(sample_times, method_stats, method_counts, example_sample)
    print()

    waits, example_batch = profile_dataloader(args)
    print_dataloader_profile(
        waits,
        example_batch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    print()

    if method_stats.get("_compute_motion_boundary_mask", 0.0) > 0.0:
        total_motion = method_stats["_compute_motion_boundary_mask"]
        total_sample_query = method_stats.get("_sample_query_data", 0.0)
        if total_sample_query > 0.0:
            ratio = total_motion / total_sample_query
            print(
                "motion-boundary share inside _sample_query_data: "
                f"{ratio * 100.0:.1f}%"
            )

    if example_batch is not None:
        approx_batch_bytes = tensor_tree_nbytes(example_batch)
        if args.num_workers > 0:
            approx_in_flight = approx_batch_bytes * args.num_workers * args.prefetch_factor
            if approx_in_flight >= 8 * (1024 ** 3):
                print(
                    "warning: estimated prefetched tensor bytes are very large; "
                    "high batch_size/num_workers/prefetch_factor can cause host-memory and pin-memory pressure."
                )

    print()
    print_sweeps(args)


if __name__ == "__main__":
    main()
