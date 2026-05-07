#!/usr/bin/env python3
"""Audit normal-supervision consistency on real D4RT validation samples.

Checks:
1. ``mask_normal`` is always a subset of ``mask_3d``.
2. Supervised target normals agree with depth-derived normals at the exact
   supervised query locations.
3. ``D4RTLoss`` behaves correctly on real batches:
   - oracle normals   -> near-zero ``loss_normal``
   - flipped normals  -> near-two ``loss_normal``
   - random normals   -> around-one ``loss_normal``

Usage:
    python md/audit_normal_loss_consistency.py \
        --config configs/mixture_5datasets_blendedmvs.yaml
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.collate import d4rt_collate_fn
from datasets.computer.depth_to_normals import compute_normals
from datasets.factory import create_training_dataset
from losses import D4RTLoss


def _single_dataset_val_config(base_cfg: dict[str, Any], ds_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = {
        "mode": "mixture",
        "datasets": [copy.deepcopy(ds_cfg)],
        "clip_len": base_cfg["clip_len"],
        "img_size": base_cfg["img_size"],
        "num_queries": base_cfg["num_queries"],
        "use_augs": False,
        "boundary_ratio": base_cfg.get("boundary_ratio", 0.3),
        "t_tgt_eq_t_cam_ratio": base_cfg.get("t_tgt_eq_t_cam_ratio", 0.4),
        "seed": base_cfg.get("seed", 42),
        "precompute_patches": base_cfg.get("precompute_patches", True),
        "precompute_from_highres": base_cfg.get("precompute_from_highres", False),
        "allow_track_fallback": base_cfg.get("allow_track_fallback", False),
        "epoch_size": base_cfg.get("epoch_size", 10000),
    }
    if "index_cache_dir" in base_cfg:
        cfg["index_cache_dir"] = base_cfg["index_cache_dir"]
    return cfg


def _visibility_logits_from_targets(target_vis: torch.Tensor) -> torch.Tensor:
    pos = torch.full_like(target_vis, 6.0)
    neg = torch.full_like(target_vis, -6.0)
    return torch.where(target_vis > 0.5, pos, neg).unsqueeze(-1)


def _make_predictions(batch: dict[str, Any], *, normal_mode: str, rng: torch.Generator) -> dict[str, torch.Tensor]:
    target_normal = batch["targets"]["normal"]
    if normal_mode == "oracle":
        pred_normal = target_normal.clone()
    elif normal_mode == "flipped":
        pred_normal = -target_normal
    elif normal_mode == "random":
        pred_normal = torch.randn(
            target_normal.shape,
            generator=rng,
            device=target_normal.device,
            dtype=target_normal.dtype,
        )
    else:
        raise ValueError(f"Unsupported normal_mode={normal_mode!r}")

    pred_normal = torch.nn.functional.normalize(pred_normal, dim=-1)

    return {
        "pos_3d": batch["targets"]["pos_3d"].clone(),
        "pos_2d": batch["targets"]["pos_2d"].clone(),
        "visibility": _visibility_logits_from_targets(batch["targets"]["visibility"]),
        "displacement": batch["targets"]["displacement"].clone(),
        "normal": pred_normal,
        "uncertainty": torch.zeros_like(batch["targets"]["visibility"]).unsqueeze(-1),
    }


def _query_normal_depth_alignment(sample) -> tuple[int, float | None]:
    if sample.normals is None or sample.depths is None:
        return 0, None

    target_mask = sample.targets["mask_normal"].bool().numpy()
    if not target_mask.any():
        return 0, None

    S = int(sample.normals.shape[-1])
    depths = sample.depths.numpy()[:, 0]          # [T, H, W]
    intrinsics = sample.intrinsics.numpy()        # [T, 3, 3]
    depth_normals = np.stack(
        [compute_normals(depths[t], intrinsics[t]) for t in range(depths.shape[0])],
        axis=0,
    )

    uv = sample.targets["pos_2d"].numpy()[target_mask]
    tgt_frames = sample.t_tgt.numpy()[target_mask]
    nx = np.clip(np.round(uv[:, 0] * (S - 1)).astype(np.int32), 0, S - 1)
    ny = np.clip(np.round(uv[:, 1] * (S - 1)).astype(np.int32), 0, S - 1)

    target_normal = sample.targets["normal"].numpy()[target_mask]
    depth_normal = depth_normals[tgt_frames, ny, nx]
    valid = np.isfinite(depth_normal).all(axis=-1) & (np.linalg.norm(depth_normal, axis=-1) > 1e-6)
    if not valid.any():
        return int(target_mask.sum()), None

    dots = np.sum(target_normal[valid] * depth_normal[valid], axis=-1)
    return int(valid.sum()), float(dots.mean())


def _audit_dataset(dataset_name: str, dataset, *, num_samples: int) -> dict[str, Any]:
    loss_fn = D4RTLoss(
        lambda_3d=0.0,
        lambda_raw_3d=0.0,
        lambda_2d=0.0,
        lambda_vis=0.0,
        lambda_disp=0.0,
        lambda_normal=1.0,
        lambda_conf=0.0,
    )
    rng = torch.Generator().manual_seed(0)

    stats: dict[str, Any] = {
        "dataset": dataset_name,
        "samples": 0,
        "samples_with_normal_queries": 0,
        "mask_normal_ratio": [],
        "mask_normal_subset_mask3d": True,
        "query_alignment_count": 0,
        "query_alignment_dot": [],
        "oracle_loss": [],
        "flipped_loss": [],
        "random_loss": [],
        "normal_query_count": 0,
        "normal_compatible_flags": set(),
    }

    for idx in range(num_samples):
        sample = dataset[idx]
        stats["samples"] += 1
        mask_normal = sample.targets["mask_normal"].bool()
        mask_3d = sample.targets["mask_3d"].bool()
        stats["mask_normal_ratio"].append(float(mask_normal.float().mean().item()))
        if bool((mask_normal & ~mask_3d).any().item()):
            stats["mask_normal_subset_mask3d"] = False
        num_normal_queries = int(mask_normal.sum().item())
        stats["normal_query_count"] += num_normal_queries
        stats["normal_compatible_flags"].add(bool(sample.metadata.get("normal_supervision_compatible")))

        valid_count, dot = _query_normal_depth_alignment(sample)
        stats["query_alignment_count"] += valid_count
        if dot is not None:
            stats["query_alignment_dot"].append(dot)

        if num_normal_queries == 0:
            continue

        stats["samples_with_normal_queries"] += 1
        batch = d4rt_collate_fn([sample])
        normalize_groups = batch["dataset_id"] * int(sample.video.shape[0]) + batch["t_cam"]
        for mode, key in (
            ("oracle", "oracle_loss"),
            ("flipped", "flipped_loss"),
            ("random", "random_loss"),
        ):
            predictions = _make_predictions(batch, normal_mode=mode, rng=rng)
            out = loss_fn(predictions, batch["targets"], normalize_groups=normalize_groups)
            stats[key].append(float(out["loss_normal"].item()))

    return stats


def _summarize(stats: dict[str, Any]) -> dict[str, Any]:
    def mean_or_none(values: list[float]) -> float | None:
        return None if not values else float(np.mean(values))

    return {
        "dataset": stats["dataset"],
        "samples": stats["samples"],
        "samples_with_normal_queries": stats["samples_with_normal_queries"],
        "normal_query_count": stats["normal_query_count"],
        "mask_normal_ratio_mean": mean_or_none(stats["mask_normal_ratio"]),
        "mask_normal_subset_mask3d": stats["mask_normal_subset_mask3d"],
        "query_alignment_count": stats["query_alignment_count"],
        "query_alignment_dot_mean": mean_or_none(stats["query_alignment_dot"]),
        "oracle_loss_mean": mean_or_none(stats["oracle_loss"]),
        "flipped_loss_mean": mean_or_none(stats["flipped_loss"]),
        "random_loss_mean": mean_or_none(stats["random_loss"]),
        "normal_compatible_flags": sorted(stats["normal_compatible_flags"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit normal supervision consistency on validation samples.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mixture_5datasets_blendedmvs.yaml",
        help="Mixture config to audit.",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=3,
        help="Number of validation samples to probe for each dataset.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open() as f:
        base_cfg = yaml.safe_load(f)

    summaries = []
    for ds_cfg in base_cfg["datasets"]:
        dataset_name = ds_cfg["name"]
        cfg = _single_dataset_val_config(base_cfg, ds_cfg)
        dataset = create_training_dataset(cfg, split="val")
        stats = _audit_dataset(dataset_name, dataset, num_samples=args.samples_per_dataset)
        summary = _summarize(stats)
        summaries.append(summary)

        print(f"\n[{dataset_name}]")
        for key, value in summary.items():
            if key == "dataset":
                continue
            print(f"  {key}: {value}")

    print("\n[overall]")
    overall = defaultdict(list)
    all_subset = True
    for row in summaries:
        all_subset = all_subset and bool(row["mask_normal_subset_mask3d"])
        for key in ("mask_normal_ratio_mean", "query_alignment_dot_mean", "oracle_loss_mean", "flipped_loss_mean", "random_loss_mean"):
            if row[key] is not None:
                overall[key].append(row[key])
    print(f"  mask_normal_subset_mask3d: {all_subset}")
    for key, values in overall.items():
        print(f"  {key}: {float(np.mean(values)) if values else None}")


if __name__ == "__main__":
    main()
