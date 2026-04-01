"""
Collate function for D4RT training batches.

Handles:
- Stacking video tensors
- Stacking query coordinates and time indices
- Stacking targets and masks
- Collecting metadata

Usage:
    from torch.utils.data import DataLoader
    from datasets.collate import d4rt_collate_fn

    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=d4rt_collate_fn,
    )
"""

from __future__ import annotations

from typing import Any

import torch


def d4rt_collate_fn(batch: list[Any]) -> dict[str, Any]:
    """
    Collate function for D4RT QuerySample batches.

    Args:
        batch: List of QuerySample dataclass instances from MixtureDataset

    Returns:
        Batched dict with stacked tensors
    """
    if len(batch) == 0:
        raise ValueError("Cannot collate empty batch")

    # Stack video tensors [B, S, 3, H, W]
    videos = torch.stack([s.video for s in batch], dim=0)

    # Stack query data [B, Q, ...]
    coords = torch.stack([s.coords for s in batch], dim=0)
    t_src = torch.stack([s.t_src for s in batch], dim=0)
    t_tgt = torch.stack([s.t_tgt for s in batch], dim=0)
    t_cam = torch.stack([s.t_cam for s in batch], dim=0)

    # Stack camera parameters [B, S, ...]
    intrinsics = torch.stack([s.intrinsics for s in batch], dim=0)
    extrinsics = torch.stack([s.extrinsics for s in batch], dim=0)

    # Stack targets (each is [B, Q, ...])
    targets = {
        "pos_2d": torch.stack([s.targets["pos_2d"] for s in batch], dim=0),
        "pos_3d": torch.stack([s.targets["pos_3d"] for s in batch], dim=0),
        "visibility": torch.stack([s.targets["visibility"] for s in batch], dim=0),
        "displacement": torch.stack([s.targets["displacement"] for s in batch], dim=0),
        "normal": torch.stack([s.targets["normal"] for s in batch], dim=0),
        "mask_2d": torch.stack([s.targets["mask_2d"] for s in batch], dim=0),
        "mask_3d": torch.stack([s.targets["mask_3d"] for s in batch], dim=0),
        "mask_vis": torch.stack([s.targets["mask_vis"] for s in batch], dim=0),
        "mask_disp": torch.stack([s.targets["mask_disp"] for s in batch], dim=0),
        "mask_normal": torch.stack([s.targets["mask_normal"] for s in batch], dim=0),
    }

    # Stack local patches [B, Q, 3, P, P] (may be None if precompute_patches=False)
    if batch[0].local_patches is not None:
        local_patches = torch.stack([s.local_patches for s in batch], dim=0)
    else:
        local_patches = None

    # Collect metadata (list of dicts)
    dataset_names = [s.dataset_name for s in batch]
    sequence_names = [s.sequence_name for s in batch]
    metadata = [s.metadata for s in batch]

    return {
        "video": videos,
        "coords": coords,
        "t_src": t_src,
        "t_tgt": t_tgt,
        "t_cam": t_cam,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "targets": targets,
        "local_patches": local_patches,
        "dataset_names": dataset_names,
        "sequence_names": sequence_names,
        "metadata": metadata,
    }
