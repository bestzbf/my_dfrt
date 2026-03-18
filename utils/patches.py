"""Shared local patch extraction helpers."""

import torch
import torch.nn.functional as F


def _build_patch_sampling_grid(
    frames_btchw: torch.Tensor,
    coords: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """Build the normalized sampling grid used by ``grid_sample``.

    This keeps the exact coordinate math of the original implementation so the
    extracted patch values do not change when we swap in a more memory-efficient
    batching strategy.
    """
    _, _, _, height, width = frames_btchw.shape

    half = patch_size // 2
    offsets_1d = torch.arange(-half, half + 1, device=frames_btchw.device, dtype=coords.dtype)
    offsets = torch.stack(torch.meshgrid(offsets_1d, offsets_1d, indexing="xy"), dim=-1)

    coords_pixel = coords.to(dtype=frames_btchw.dtype).clone()
    coords_pixel[..., 0] = coords_pixel[..., 0] * (width - 1)
    coords_pixel[..., 1] = coords_pixel[..., 1] * (height - 1)

    grid = coords_pixel.view(coords.shape[0], coords.shape[1], 1, 1, 2) + offsets.view(1, 1, patch_size, patch_size, 2)
    grid[..., 0] = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
    return grid


def _extract_local_patches_grouped_by_source_frame(
    frames_btchw: torch.Tensor,
    grid: torch.Tensor,
    t_src: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """Extract patches while grouping queries that share the same source frame.

    The previous implementation first materialized ``(B * N, C, H, W)`` by
    copying one full-resolution frame per query before sampling a tiny
    ``patch_size x patch_size`` neighborhood. Grouping queries by ``t_src`` keeps
    the exact same sampling math but avoids that large per-query frame tensor.
    """
    batch_size, num_frames, _, _, _ = frames_btchw.shape
    _, num_queries = t_src.shape

    patches_by_batch = []
    for batch_idx in range(batch_size):
        if num_queries == 0:
            empty = frames_btchw.new_empty((0, frames_btchw.shape[2], patch_size, patch_size))
            patches_by_batch.append(empty)
            continue

        frame_indices = t_src[batch_idx].clamp(0, num_frames - 1)
        batch_frames = frames_btchw[batch_idx]
        batch_grid = grid[batch_idx]

        unique_frames, inverse = torch.unique(frame_indices, sorted=True, return_inverse=True)
        sampled_chunks = []
        sampled_indices = []

        for group_idx in range(unique_frames.numel()):
            query_indices = torch.nonzero(inverse == group_idx, as_tuple=False).flatten()
            source_frame = batch_frames.index_select(0, unique_frames[group_idx:group_idx + 1])
            source_grid = batch_grid.index_select(0, query_indices)

            # Reuse the same source frame for every query in this group rather than
            # materializing one full-resolution frame per query.
            sampled = F.grid_sample(
                source_frame.expand(source_grid.shape[0], -1, -1, -1),
                source_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            sampled_chunks.append(sampled)
            sampled_indices.append(query_indices)

        sampled_indices = torch.cat(sampled_indices, dim=0)
        sampled_patches = torch.cat(sampled_chunks, dim=0)
        restore_order = torch.argsort(sampled_indices)
        patches_by_batch.append(sampled_patches.index_select(0, restore_order))

    return torch.stack(patches_by_batch, dim=0)


def extract_local_patches(
    frames_btchw: torch.Tensor,
    coords: torch.Tensor,
    t_src: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """Extract query-centered RGB patches using the same sampler as decoder inference.

    Args:
        frames_btchw: (B, T, C, H, W) video frames in [0, 1]
        coords: (B, N, 2) normalized coordinates in [0, 1]
        t_src: (B, N) source frame indices
        patch_size: square patch size

    Returns:
        patches: (B, N, C, patch_size, patch_size)
    """
    if frames_btchw.dim() != 5:
        raise ValueError(f"Expected frames_btchw to be 5D, got shape {tuple(frames_btchw.shape)}")
    if coords.dim() != 3 or coords.shape[-1] != 2:
        raise ValueError(f"Expected coords to have shape (B, N, 2), got {tuple(coords.shape)}")
    if t_src.dim() != 2:
        raise ValueError(f"Expected t_src to have shape (B, N), got {tuple(t_src.shape)}")

    batch_size, num_frames = frames_btchw.shape[:2]
    _, num_queries, _ = coords.shape
    if t_src.shape != (batch_size, num_queries):
        raise ValueError(
            "Expected t_src shape to match coords batch/query dimensions, "
            f"got coords={tuple(coords.shape)} and t_src={tuple(t_src.shape)}"
        )

    grid = _build_patch_sampling_grid(frames_btchw, coords, patch_size=patch_size)
    return _extract_local_patches_grouped_by_source_frame(
        frames_btchw=frames_btchw,
        grid=grid,
        t_src=t_src,
        patch_size=patch_size,
    )


def extract_local_patches_with_valid_hw(
    frames_btchw: torch.Tensor,
    coords: torch.Tensor,
    t_src: torch.Tensor,
    patch_size: int,
    valid_hw: torch.Tensor,
) -> torch.Tensor:
    """Extract patches from per-sample valid crop regions inside a padded batch tensor.

    Args:
        frames_btchw: (B, T, C, H_pad, W_pad) padded video frames in [0, 1]
        coords: (B, N, 2) normalized coordinates in canonical crop space [0, 1]
        t_src: (B, N) source frame indices
        patch_size: square patch size
        valid_hw: (B, 2) actual (crop_h, crop_w) before padding

    Returns:
        patches: (B, N, C, patch_size, patch_size)
    """
    if valid_hw.dim() != 2 or valid_hw.shape[-1] != 2:
        raise ValueError(f"Expected valid_hw to have shape (B, 2), got {tuple(valid_hw.shape)}")
    if valid_hw.shape[0] != frames_btchw.shape[0]:
        raise ValueError(
            "Expected valid_hw batch dimension to match frames batch dimension, "
            f"got frames={tuple(frames_btchw.shape)} and valid_hw={tuple(valid_hw.shape)}"
        )

    padded_h, padded_w = frames_btchw.shape[-2:]
    patches = []
    for batch_idx in range(frames_btchw.shape[0]):
        crop_h = int(round(float(valid_hw[batch_idx, 0].item())))
        crop_w = int(round(float(valid_hw[batch_idx, 1].item())))
        crop_h = max(1, min(crop_h, padded_h))
        crop_w = max(1, min(crop_w, padded_w))

        sample_frames = frames_btchw[batch_idx:batch_idx + 1, :, :, :crop_h, :crop_w]
        sample_patches = extract_local_patches(
            sample_frames,
            coords[batch_idx:batch_idx + 1],
            t_src[batch_idx:batch_idx + 1],
            patch_size=patch_size,
        )
        patches.append(sample_patches.squeeze(0))

    return torch.stack(patches, dim=0)
