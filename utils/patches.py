"""Shared local patch extraction helpers."""

import torch
import torch.nn.functional as F

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

    batch_size, num_frames, channels, height, width = frames_btchw.shape
    _, num_queries, _ = coords.shape
    if t_src.shape != (batch_size, num_queries):
        raise ValueError(
            "Expected t_src shape to match coords batch/query dimensions, "
            f"got coords={tuple(coords.shape)} and t_src={tuple(t_src.shape)}"
        )

    query_frames = []
    for batch_idx in range(batch_size):
        frame_indices = t_src[batch_idx].clamp(0, num_frames - 1)
        query_frames.append(frames_btchw[batch_idx, frame_indices])
    query_frames = torch.stack(query_frames, dim=0).reshape(batch_size * num_queries, channels, height, width)

    half = patch_size // 2
    offsets_1d = torch.arange(-half, half + 1, device=frames_btchw.device, dtype=coords.dtype)
    offsets = torch.stack(torch.meshgrid(offsets_1d, offsets_1d, indexing="xy"), dim=-1)

    coords_pixel = coords.to(dtype=frames_btchw.dtype).clone()
    coords_pixel[..., 0] = coords_pixel[..., 0] * (width - 1)
    coords_pixel[..., 1] = coords_pixel[..., 1] * (height - 1)

    grid = coords_pixel.view(batch_size, num_queries, 1, 1, 2) + offsets.view(1, 1, patch_size, patch_size, 2)
    grid[..., 0] = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
    grid = grid.reshape(batch_size * num_queries, patch_size, patch_size, 2)

    patches = F.grid_sample(
        query_frames,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return patches.reshape(batch_size, num_queries, channels, patch_size, patch_size)


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
