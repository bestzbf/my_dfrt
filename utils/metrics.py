"""Evaluation metrics for D4RT."""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def compute_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale_invariant: bool = True,
    shift_invariant: bool = False
) -> dict[str, torch.Tensor]:
    """Compute depth estimation metrics.

    Args:
        pred: Predicted depth maps
        target: Ground truth depth maps
        mask: Optional validity mask
        scale_invariant: Apply scale alignment
        shift_invariant: Apply scale and shift alignment

    Returns:
        Dictionary with metrics:
            - abs_rel: Absolute relative error
            - sq_rel: Squared relative error
            - rmse: Root mean squared error
            - rmse_log: Log RMSE
            - a1, a2, a3: Threshold accuracy metrics
    """
    if mask is None:
        mask = (target > 0) & torch.isfinite(target)

    pred_flat = pred[mask]
    target_flat = target[mask]

    if len(pred_flat) == 0:
        return {k: torch.tensor(float('nan')) for k in
                ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']}

    # Alignment
    if shift_invariant:
        # Scale and shift alignment
        pred_median = pred_flat.median()
        target_median = target_flat.median()

        pred_flat = pred_flat - pred_median
        target_flat = target_flat - target_median

        scale = (target_flat * pred_flat).sum() / (pred_flat ** 2).sum().clamp(min=1e-6)
        pred_flat = pred_flat * scale + target_median

    elif scale_invariant:
        # Scale-only alignment
        scale = (target_flat / pred_flat.clamp(min=1e-6)).median()
        pred_flat = pred_flat * scale

    # Clamp predictions
    pred_flat = pred_flat.clamp(min=1e-6)

    # Compute errors
    abs_diff = torch.abs(pred_flat - target_flat)
    abs_rel = (abs_diff / target_flat).mean()
    sq_rel = ((abs_diff ** 2) / target_flat).mean()
    rmse = torch.sqrt((abs_diff ** 2).mean())

    # Log errors
    log_diff = torch.abs(torch.log(pred_flat) - torch.log(target_flat))
    rmse_log = torch.sqrt((log_diff ** 2).mean())

    # Threshold accuracy (delta < 1.25^k)
    ratio = torch.max(pred_flat / target_flat, target_flat / pred_flat)
    a1 = (ratio < 1.25).float().mean()
    a2 = (ratio < 1.25 ** 2).float().mean()
    a3 = (ratio < 1.25 ** 3).float().mean()

    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3
    }


def compute_pose_metrics(
    pred_poses: torch.Tensor,
    gt_poses: torch.Tensor,
    align: bool = True
) -> dict[str, torch.Tensor]:
    """Compute camera pose estimation metrics.

    Args:
        pred_poses: (N, 4, 4) predicted camera poses
        gt_poses: (N, 4, 4) ground truth camera poses
        align: Whether to apply Sim(3) alignment first

    Returns:
        Dictionary with metrics:
            - ate: Absolute Trajectory Error
            - rpe_trans: Relative Pose Error (translation)
            - rpe_rot: Relative Pose Error (rotation in degrees)
    """
    from .camera import sim3_alignment

    N = pred_poses.shape[0]

    if align and N > 1:
        # Sim(3) alignment — eval-time only (appendix long-video setting).
        # NOT the paper main-text pose solver (which is rigid/SE(3)).
        R, t, s = sim3_alignment(pred_poses, gt_poses)

        # Apply alignment to both rotation and translation
        aligned_poses = pred_poses.clone()

        # Apply rotation alignment: R_aligned = R_pred @ R^T
        aligned_poses[:, :3, :3] = pred_poses[:, :3, :3] @ R.T

        # Apply scale, rotation, and translation to centers
        pred_centers = pred_poses[:, :3, 3]
        aligned_poses[:, :3, 3] = s * (pred_centers @ R.T) + t
    else:
        aligned_poses = pred_poses

    # Absolute Trajectory Error (ATE)
    pred_centers = aligned_poses[:, :3, 3]
    gt_centers = gt_poses[:, :3, 3]
    ate = torch.sqrt(((pred_centers - gt_centers) ** 2).sum(dim=-1)).mean()

    # Relative Pose Error (RPE)
    if N > 1:
        rpe_trans_list = []
        rpe_rot_list = []

        for i in range(N - 1):
            # Relative pose (ground truth)
            gt_rel = torch.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]

            # Relative pose (predicted)
            pred_rel = torch.linalg.inv(aligned_poses[i]) @ aligned_poses[i + 1]

            # Translation error
            trans_error = torch.norm(pred_rel[:3, 3] - gt_rel[:3, 3])
            rpe_trans_list.append(trans_error)

            # Rotation error
            R_error = pred_rel[:3, :3] @ gt_rel[:3, :3].T
            trace = R_error.trace().clamp(-1, 3)
            angle = torch.acos((trace - 1) / 2)
            rpe_rot_list.append(torch.rad2deg(angle))

        rpe_trans = torch.stack(rpe_trans_list).mean()
        rpe_rot = torch.stack(rpe_rot_list).mean()
    else:
        rpe_trans = torch.tensor(0.0)
        rpe_rot = torch.tensor(0.0)

    return {
        'ate': ate,
        'rpe_trans': rpe_trans,
        'rpe_rot': rpe_rot
    }


def compute_tracking_metrics(
    pred_tracks: torch.Tensor,
    gt_tracks: torch.Tensor,
    visibility: torch.Tensor,
    pred_visibility: Optional[torch.Tensor] = None,
    thresholds: Optional[list[float]] = None
) -> dict[str, torch.Tensor]:
    """Compute 3D point tracking metrics.

    WARNING: This is an APPROXIMATE implementation, NOT the full TAPVid-3D protocol.
    - AJ (Average Jaccard): Simplified to threshold-based accuracy, not true IoU
    - OA (Occlusion Accuracy): Only computed if pred_visibility is provided

    For paper reproduction, use official TAPVid-3D evaluation code.

    Args:
        pred_tracks: (N, T, 3) predicted 3D tracks
        gt_tracks: (N, T, 3) ground truth 3D tracks
        visibility: (N, T) ground truth point visibility
        pred_visibility: (N, T) predicted visibility (optional, for OA)
        thresholds: Delta thresholds for APD metric

    Returns:
        Dictionary with metrics:
            - l1: Mean L1 error
            - apd: Average Percent within Delta
            - aj_approx: Approximate Average Jaccard (threshold-based)
            - oa: Occlusion Accuracy (NaN if pred_visibility not provided)
    """
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    N, T, _ = pred_tracks.shape

    # Compute per-point errors
    errors = torch.norm(pred_tracks - gt_tracks, dim=-1)  # (N, T)

    # Apply visibility mask
    visible = visibility > 0.5

    # L1 error (only visible points)
    l1 = errors[visible].mean() if visible.any() else torch.tensor(0.0)

    # Average Percent within Delta (APD)
    apd_scores = []
    for thresh in thresholds:
        within_thresh = (errors < thresh) & visible
        apd = within_thresh.sum().float() / visible.sum().float() if visible.any() else torch.tensor(0.0)
        apd_scores.append(apd)
    apd = torch.stack(apd_scores).mean()

    # Average Jaccard (AJ) - APPROXIMATE version
    # True TAPVid-3D AJ requires IoU of predicted/GT visibility masks + position accuracy
    # This simplified version: threshold-based accuracy on visible points
    pos_accuracy = (errors < 0.1) & visible
    aj_approx = pos_accuracy.sum().float() / max(visible.sum().float(), 1.0)

    # Occlusion Accuracy (OA)
    if pred_visibility is not None:
        pred_vis = pred_visibility > 0.5
        # OA = accuracy of visibility prediction
        vis_correct = (pred_vis == visible).float()
        oa = vis_correct.mean()
    else:
        oa = torch.tensor(float('nan'))

    return {
        'l1': l1,
        'apd': apd,
        'aj_approx': aj_approx,
        'oa': oa
    }


def compute_point_cloud_metrics(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    align: bool = True
) -> dict[str, torch.Tensor]:
    """Compute point cloud reconstruction metrics.

    Paper specification: mean-shift alignment + mean L1 distance.

    Args:
        pred_points: (N, 3) predicted point cloud
        gt_points: (M, 3) ground truth point cloud
        align: Whether to apply mean-shift alignment (paper protocol)

    Returns:
        Dictionary with metrics:
            - l1: Mean L1 distance (after mean-shift alignment)
            - chamfer: Chamfer distance
    """
    if align:
        # Mean-shift alignment (paper protocol)
        pred_mean = pred_points.mean(dim=0)
        gt_mean = gt_points.mean(dim=0)

        pred_aligned = pred_points - pred_mean + gt_mean
    else:
        pred_aligned = pred_points

    # If same number of points, compute paired L1
    if pred_aligned.shape[0] == gt_points.shape[0]:
        l1 = torch.abs(pred_aligned - gt_points).mean()
    else:
        l1 = torch.tensor(float('nan'))

    # Chamfer distance
    # pred -> gt
    dist_pred_to_gt = torch.cdist(pred_aligned, gt_points).min(dim=1)[0]
    # gt -> pred
    dist_gt_to_pred = torch.cdist(gt_points, pred_aligned).min(dim=1)[0]

    chamfer = (dist_pred_to_gt.mean() + dist_gt_to_pred.mean()) / 2

    return {
        'l1': l1,
        'chamfer': chamfer
    }


def compute_pose_auc(
    rotation_errors: torch.Tensor,
    translation_errors: torch.Tensor,
    threshold: float = 30.0
) -> torch.Tensor:
    """Compute Pose AUC at given threshold.

    Args:
        rotation_errors: (N,) rotation errors in degrees
        translation_errors: (N,) translation errors in degrees or cm
        threshold: Maximum error threshold

    Returns:
        AUC value
    """
    # Combined error (max of rotation and translation)
    combined_errors = torch.max(rotation_errors, translation_errors)

    # Sort errors
    sorted_errors, _ = torch.sort(combined_errors)

    # Compute AUC
    n = len(sorted_errors)
    thresholds = torch.linspace(0, threshold, 100, device=sorted_errors.device)

    accuracies = []
    for t in thresholds:
        acc = (sorted_errors < t).float().mean()
        accuracies.append(acc)

    auc = torch.trapezoid(torch.stack(accuracies), thresholds) / threshold

    return auc
