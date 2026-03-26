"""Loss functions for D4RT training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def compute_depth_normalizers(
    points: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    groups: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the mean-depth denominators used by scale-invariant 3D losses."""
    z_coords = points[..., 2]  # (B, N)

    if mask is not None:
        valid_mask = mask.bool()
    else:
        valid_mask = torch.ones_like(z_coords, dtype=torch.bool)

    if groups is not None:
        if groups.shape != z_coords.shape:
            raise ValueError(
                f"Expected groups to match points batch/query shape, got {tuple(groups.shape)} "
                f"for points {tuple(points.shape)}"
            )
        mean_depth_safe = torch.ones_like(z_coords)
        for batch_idx in range(z_coords.shape[0]):
            valid_queries = valid_mask[batch_idx]
            if not valid_queries.any():
                continue
            for group_id in torch.unique(groups[batch_idx, valid_queries]):
                group_mask = valid_queries & (groups[batch_idx] == group_id)
                group_mean = z_coords[batch_idx, group_mask].mean()
                group_mean = torch.nan_to_num(group_mean, nan=1.0, posinf=1.0, neginf=1.0)
                mean_depth_safe[batch_idx, group_mask] = torch.clamp(torch.abs(group_mean), min=eps)
        return mean_depth_safe, valid_mask

    if mask is not None:
        safe_z = torch.where(valid_mask, z_coords, torch.zeros_like(z_coords))
        sum_depth = safe_z.sum(dim=-1, keepdim=True)  # (B, 1)
        valid_count = valid_mask.sum(dim=-1, keepdim=True).to(z_coords.dtype)  # (B, 1)
        mean_depth = sum_depth / (valid_count + eps)
    else:
        mean_depth = z_coords.mean(dim=-1, keepdim=True)  # (B, 1)

    mean_depth = torch.nan_to_num(mean_depth, nan=1.0, posinf=1.0, neginf=1.0)
    mean_depth_safe = torch.clamp(torch.abs(mean_depth), min=eps).expand_as(z_coords)
    return mean_depth_safe, valid_mask


def normalize_points(
    points: torch.Tensor, 
    mask: Optional[torch.Tensor] = None, 
    groups: Optional[torch.Tensor] = None,
    eps: float = 1e-6
) -> torch.Tensor:
    """Normalize points by their mean depth, accounting for invalid points.

    Args:
        points: (B, N, 3) point positions
        mask: (B, N) optional validity mask
        groups: (B, N) optional group ids. When provided, mean-depth normalization is
            computed independently per group inside each batch element.
        eps: Small value for numerical stability

    Returns:
        normalized_points: (B, N, 3) normalized points
    """
    mean_depth_safe, valid_mask = compute_depth_normalizers(
        points,
        mask=mask,
        groups=groups,
        eps=eps,
    )

    if valid_mask is not None:
        points = torch.where(valid_mask.unsqueeze(-1), points, torch.zeros_like(points))

    normalized = points / mean_depth_safe.unsqueeze(-1)
    return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def compute_depth_statistics(
    points: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    groups: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute raw absolute depth and effective normalization depth for logging."""
    mean_depth_safe, valid_mask = compute_depth_normalizers(
        points,
        mask=mask,
        groups=groups,
        eps=eps,
    )
    abs_depth = torch.abs(points[..., 2])
    abs_depth = torch.where(valid_mask, abs_depth, torch.zeros_like(abs_depth))
    mean_depth_safe = torch.where(valid_mask, mean_depth_safe, torch.zeros_like(mean_depth_safe))
    denom = valid_mask.sum().to(points.dtype) + eps
    return abs_depth.sum() / denom, mean_depth_safe.sum() / denom

# # ==========================================
# # 下方代码替换 D4RTLoss.forward 中的 3D Loss 计算部分
# # ==========================================

# # 获取 confidence
# confidence = predictions['confidence'].squeeze(-1)  # (B, N)

# # 1. 带 Mask 的归一化
# pred_norm = normalize_points(predictions['pos_3d'], mask=mask_3d)
# target_norm = normalize_points(targets['pos_3d'], mask=mask_3d)

# # 2. 对数抑制变换 sign(x) * log(1 + |x|)
# pred_log = log_transform(pred_norm)
# target_log = log_transform(target_norm)

# # 3. 计算单个点的 L1 距离 (在特征维度 x,y,z 上取平均)
# point_loss = torch.abs(pred_log - target_log).mean(dim=-1)  # (B, N)

# # 4. 强制进行置信度加权 (对应公式: c * L_3D)
# weighted_loss = confidence * point_loss  # (B, N)

# # 5. Mask 过滤与聚合
# if mask_3d is not None:
#     losses['loss_3d'] = (weighted_loss * mask_3d).sum() / (mask_3d.sum() + 1e-6)
# else:
#     losses['loss_3d'] = weighted_loss.mean()



def log_transform(x: torch.Tensor) -> torch.Tensor:
    """Apply log transform to dampen influence of far points.

    Transform: sign(x) * log(1 + |x|)

    Args:
        x: Input tensor

    Returns:
        Transformed tensor
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def compute_3d_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    groups: Optional[torch.Tensor] = None,
    normalize: bool = True,
    use_log_transform: bool = True
) -> torch.Tensor:
    """Compute L1 loss on 3D positions.

    Args:
        pred: (B, N, 3) predicted 3D positions
        target: (B, N, 3) target 3D positions
        mask: (B, N) optional validity mask
        groups: (B, N) optional group ids used for per-group mean-depth normalization
        normalize: Whether to normalize by mean depth
        use_log_transform: Whether to apply log transform

    Returns:
        loss: Scalar loss value
    """
    if normalize:
        pred = normalize_points(pred, mask=mask, groups=groups)
        target = normalize_points(target, mask=mask, groups=groups)

    if use_log_transform:
        pred = log_transform(pred)
        target = log_transform(target)

    # L1 loss
    loss = torch.abs(pred - target)

    if mask is not None:
        valid_mask = mask.bool().unsqueeze(-1).expand_as(loss)
        loss = torch.where(valid_mask, loss, torch.zeros_like(loss))
        return loss.sum() / (valid_mask.sum() + 1e-6)

    return loss.mean()


def compute_2d_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute L1 loss on 2D reprojection.

    Args:
        pred: (B, N, 2) predicted 2D positions
        target: (B, N, 2) target 2D positions
        mask: (B, N) optional validity mask

    Returns:
        loss: Scalar loss value
    """
    loss = torch.abs(pred - target)

    if mask is not None:
        valid_mask = mask.bool().unsqueeze(-1).expand_as(loss)
        loss = torch.where(valid_mask, loss, torch.zeros_like(loss))
        return loss.sum() / (valid_mask.sum() + 1e-6)

    return loss.mean()


def compute_visibility_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute binary cross-entropy loss for visibility prediction.

    Args:
        pred: (B, N, 1) predicted visibility logits
        target: (B, N) target visibility (0 or 1)
        mask: (B, N) optional validity mask

    Returns:
        loss: Scalar loss value
    """
    pred = pred.squeeze(-1)  # (B, N)

    loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')

    if mask is not None:
        valid_mask = mask.bool()
        loss = torch.where(valid_mask, loss, torch.zeros_like(loss))
        return loss.sum() / (valid_mask.sum() + 1e-6)

    return loss.mean()


def compute_displacement_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute L1 loss on motion displacement.

    Args:
        pred: (B, N, 3) predicted displacement
        target: (B, N, 3) target displacement
        mask: (B, N) optional validity mask

    Returns:
        loss: Scalar loss value
    """
    loss = torch.abs(pred - target)

    if mask is not None:
        valid_mask = mask.bool().unsqueeze(-1).expand_as(loss)
        loss = torch.where(valid_mask, loss, torch.zeros_like(loss))
        return loss.sum() / (valid_mask.sum() + 1e-6)

    return loss.mean()


def compute_normal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute cosine similarity loss for surface normals.

    Args:
        pred: (B, N, 3) predicted normals (should be normalized)
        target: (B, N, 3) target normals (should be normalized)
        mask: (B, N) optional validity mask

    Returns:
        loss: Scalar loss value (1 - cosine_similarity)
    """
    # Normalize to unit vectors
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    # Cosine similarity
    cos_sim = (pred * target).sum(dim=-1)  # (B, N)

    # Loss is 1 - cosine_similarity
    loss = 1.0 - cos_sim

    if mask is not None:
        valid_mask = mask.bool()
        loss = torch.where(valid_mask, loss, torch.zeros_like(loss))
        return loss.sum() / (valid_mask.sum() + 1e-6)

    return loss.mean()


def compute_confidence_loss(
    confidence: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute the confidence penalty term.
    Using Kendall's log-parameterization: the network outputs s = -log(c).
    The penalty term is simply s.

    Args:
        confidence: (B, N, 1) this is actually 's' from the network
        mask: (B, N) optional validity mask

    Returns:
        loss: Scalar loss value
    """
    s = confidence.squeeze(-1)  # (B, N)

    # The penalty is exactly s
    loss = s

    if mask is not None:
        valid_mask = mask.bool()
        loss = torch.where(valid_mask, loss, torch.zeros_like(loss))
        return loss.sum() / (valid_mask.sum() + 1e-6)

    return loss.mean()


def compute_raw_3d_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute raw camera-space 3D errors for logging only."""
    l1 = torch.abs(pred - target).mean(dim=-1)
    euclidean = torch.linalg.norm(pred - target, dim=-1)

    if mask is not None:
        valid_mask = mask.bool()
        l1 = torch.where(valid_mask, l1, torch.zeros_like(l1))
        euclidean = torch.where(valid_mask, euclidean, torch.zeros_like(euclidean))
        denom = valid_mask.sum() + 1e-6
        return l1.sum() / denom, euclidean.sum() / denom

    return l1.mean(), euclidean.mean()


class D4RTLoss(nn.Module):
    """Combined loss function for D4RT training.

    Computes weighted sum of:
    - L_3D: L1 loss on normalized, log-transformed 3D positions (weighted by confidence)
    - L_raw3D: optional L1 loss on raw camera-space 3D positions
    - L_2D: L1 loss on 2D reprojection
    - L_vis: BCE loss for visibility
    - L_disp: L1 loss on motion displacement
    - L_normal: Cosine similarity loss for surface normals
    - L_conf: Confidence penalty -log(c)
    """

    def __init__(
        self,
        lambda_3d: float = 1.0,
        lambda_raw_3d: float = 0.0,
        lambda_2d: float = 0.1,
        lambda_vis: float = 0.1,
        lambda_disp: float = 0.1,
        lambda_normal: float = 0.5,
        lambda_conf: float = 0.2,
        use_confidence_weighting: Optional[bool] = None,
        debug_3d_loss_mode: str = "scale_invariant",
    ):
        super().__init__()
        self.lambda_3d = lambda_3d
        self.lambda_raw_3d = lambda_raw_3d
        self.lambda_2d = lambda_2d
        self.lambda_vis = lambda_vis
        self.lambda_disp = lambda_disp
        self.lambda_normal = lambda_normal
        self.lambda_conf = lambda_conf
        if debug_3d_loss_mode not in {"scale_invariant", "raw_l1"}:
            raise ValueError(
                f"Unsupported debug_3d_loss_mode={debug_3d_loss_mode!r}. "
                "Expected one of {'scale_invariant', 'raw_l1'}."
            )
        self.debug_3d_loss_mode = debug_3d_loss_mode
        self.use_confidence_weighting = (
            lambda_conf > 0.0 if use_confidence_weighting is None else use_confidence_weighting
        )
        self.confidence_weighting_factor = 1.0 if self.use_confidence_weighting else 0.0

    def _reduce_point_loss(
        self,
        point_loss: torch.Tensor,
        uncertainty: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply Kendall confidence weighting exp(-s) * L_3D with gradual ramp-up."""
        if self.use_confidence_weighting:
            # Using Kendall's log-parameterization: uncertainty is 's'
            # Weighted loss = exp(-s) * L_3D
            confidence_weight = torch.exp(-uncertainty)
            # Apply weighting_factor for gradual ramp-up: lerp from 1.0 to exp(-s)
            confidence_weight = 1.0 + self.confidence_weighting_factor * (confidence_weight - 1.0)
            point_loss = confidence_weight * point_loss

        if mask is not None:
            valid_mask = mask.bool()
            point_loss = torch.where(valid_mask, point_loss, torch.zeros_like(point_loss))
            return point_loss.sum() / (valid_mask.sum() + 1e-6)

        return point_loss.mean()

    def set_confidence_schedule(self, lambda_conf: float, weighting_factor: Optional[float] = None) -> None:
        """Set confidence loss weight and weighting factor for gradual ramp-up.

        Args:
            lambda_conf: Weight for confidence penalty loss
            weighting_factor: Factor in [0,1] for gradual confidence weighting ramp-up.
                            0.0 = no weighting (weight=1.0), 1.0 = full Kendall weighting (weight=exp(-s))
        """
        self.lambda_conf = float(lambda_conf)
        if weighting_factor is not None:
            self.confidence_weighting_factor = float(weighting_factor)
        else:
            self.confidence_weighting_factor = 1.0
        self.use_confidence_weighting = True

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        normalize_groups: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            predictions: Dictionary with model outputs:
                - pos_3d: (B, N, 3) predicted 3D positions
                - pos_2d: (B, N, 2) predicted 2D positions
                - visibility: (B, N, 1) predicted visibility logits
                - displacement: (B, N, 3) predicted displacement
                - normal: (B, N, 3) predicted normals
                - confidence: (B, N, 1) predicted positive confidence

            targets: Dictionary with ground truth:
                - pos_3d: (B, N, 3) target 3D positions
                - pos_2d: (B, N, 2) target 2D positions (optional)
                - visibility: (B, N) target visibility
                - displacement: (B, N, 3) target displacement (optional)
                - normal: (B, N, 3) target normals (optional)
                - mask_3d: (B, N) validity mask for 3D loss
                - mask_2d: (B, N) validity mask for 2D loss (optional)
                - mask_vis: (B, N) validity mask for visibility (optional)
                - mask_disp: (B, N) validity mask for displacement (optional)
                - mask_normal: (B, N) validity mask for normals (optional)

        Returns:
            Dictionary with:
                - loss: Total weighted loss
                - loss_3d: 3D position loss
                - loss_2d: 2D position loss
                - loss_vis: Visibility loss
                - loss_disp: Displacement loss
                - loss_normal: Normal loss
                - loss_conf: Confidence penalty
        """
        losses = {}

        # Get masks (default to all valid if not provided)
        mask_3d = targets.get('mask_3d')
        mask_2d = targets.get('mask_2d', mask_3d)
        mask_vis = targets.get('mask_vis', mask_3d)
        mask_disp = targets.get('mask_disp')
        mask_normal = targets.get('mask_normal')

        # 3D position loss. Confidence weighting is only enabled once the
        # confidence penalty itself participates in optimization; otherwise the
        # model can minimize this term by collapsing confidence toward zero.
        uncertainty = predictions['uncertainty'].squeeze(-1)  # (B, N) - Kendall's s

        if self.debug_3d_loss_mode == "raw_l1":
            point_loss = torch.abs(predictions['pos_3d'] - targets['pos_3d']).mean(dim=-1)
            losses['loss_3d'] = self._reduce_point_loss(
                point_loss,
                uncertainty,
                mask=mask_3d,
            )
        elif mask_3d is not None:
            pred_norm = normalize_points(predictions['pos_3d'], mask=mask_3d, groups=normalize_groups)
            target_norm = normalize_points(targets['pos_3d'], mask=mask_3d, groups=normalize_groups)
            pred_log = log_transform(pred_norm)
            target_log = log_transform(target_norm)

            point_loss = torch.abs(pred_log - target_log).mean(dim=-1)  # (B, N)
            losses['loss_3d'] = self._reduce_point_loss(
                point_loss,
                uncertainty,
                mask=mask_3d,
            )
        else:
            losses['loss_3d'] = compute_3d_loss(
                predictions['pos_3d'],
                targets['pos_3d'],
                mask=None,
                groups=normalize_groups,
                normalize=True,
                use_log_transform=True
            )

        losses['loss_raw_3d'] = compute_3d_loss(
            predictions['pos_3d'],
            targets['pos_3d'],
            mask=mask_3d,
            groups=None,
            normalize=False,
            use_log_transform=False,
        )

        # 2D position loss (if target provided)
        if 'pos_2d' in targets:
            losses['loss_2d'] = compute_2d_loss(
                predictions['pos_2d'],
                targets['pos_2d'],
                mask=mask_2d
            )
        else:
            losses['loss_2d'] = torch.tensor(0.0, device=predictions['pos_3d'].device)

        # Visibility loss
        if 'visibility' in targets:
            losses['loss_vis'] = compute_visibility_loss(
                predictions['visibility'],
                targets['visibility'],
                mask=mask_vis
            )
        else:
            losses['loss_vis'] = torch.tensor(0.0, device=predictions['pos_3d'].device)

        # Displacement loss (if target provided)
        if 'displacement' in targets and mask_disp is not None:
            losses['loss_disp'] = compute_displacement_loss(
                predictions['displacement'],
                targets['displacement'],
                mask=mask_disp
            )
        else:
            losses['loss_disp'] = torch.tensor(0.0, device=predictions['pos_3d'].device)

        # Normal loss (if target provided)
        if 'normal' in targets and mask_normal is not None:
            losses['loss_normal'] = compute_normal_loss(
                predictions['normal'],
                targets['normal'],
                mask=mask_normal
            )
        else:
            losses['loss_normal'] = torch.tensor(0.0, device=predictions['pos_3d'].device)

        # Confidence penalty
        losses['loss_conf'] = compute_confidence_loss(
            predictions['uncertainty'],
            mask=mask_3d
        )

        raw_3d_l1, raw_3d_euclidean = compute_raw_3d_metrics(
            predictions['pos_3d'],
            targets['pos_3d'],
            mask=mask_3d,
        )
        losses['metric_raw_3d_l1'] = raw_3d_l1
        losses['metric_raw_3d_euclidean'] = raw_3d_euclidean
        pred_abs_depth_mean, pred_norm_depth_mean = compute_depth_statistics(
            predictions['pos_3d'],
            mask=mask_3d,
            groups=normalize_groups,
        )
        target_abs_depth_mean, target_norm_depth_mean = compute_depth_statistics(
            targets['pos_3d'],
            mask=mask_3d,
            groups=normalize_groups,
        )
        losses['metric_pred_abs_depth_mean'] = pred_abs_depth_mean
        losses['metric_target_abs_depth_mean'] = target_abs_depth_mean
        losses['metric_pred_norm_depth_mean'] = pred_norm_depth_mean
        losses['metric_target_norm_depth_mean'] = target_norm_depth_mean
        if mask_3d is not None:
            valid_mask_3d = mask_3d.bool()
            conf_mean = torch.where(valid_mask_3d, confidence, torch.zeros_like(confidence)).sum()
            conf_mean = conf_mean / (valid_mask_3d.sum() + 1e-6)
        else:
            conf_mean = confidence.mean()
        losses['metric_conf_mean'] = conf_mean

        # Total loss
        losses['loss'] = (
            self.lambda_3d * losses['loss_3d'] +
            self.lambda_raw_3d * losses['loss_raw_3d'] +
            self.lambda_2d * losses['loss_2d'] +
            self.lambda_vis * losses['loss_vis'] +
            self.lambda_disp * losses['loss_disp'] +
            self.lambda_normal * losses['loss_normal'] +
            self.lambda_conf * losses['loss_conf']
        )

        return losses


class DepthLoss(nn.Module):
    """Loss function for depth estimation evaluation."""

    def __init__(self, scale_invariant: bool = True):
        super().__init__()
        self.scale_invariant = scale_invariant

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """Compute depth metrics.

        Args:
            pred: (B, H, W) predicted depth
            target: (B, H, W) target depth
            mask: (B, H, W) validity mask

        Returns:
            Dictionary with metrics
        """
        if mask is None:
            mask = torch.ones_like(pred, dtype=torch.bool)

        # Flatten
        pred_flat = pred[mask]
        target_flat = target[mask]

        if self.scale_invariant:
            # Scale alignment
            scale = (target_flat / (pred_flat + 1e-6)).median()
            pred_flat = pred_flat * scale

        # AbsRel
        abs_rel = torch.abs(pred_flat - target_flat) / (target_flat + 1e-6)
        abs_rel = abs_rel.mean()

        # RMSE
        rmse = torch.sqrt(((pred_flat - target_flat) ** 2).mean())

        # Log RMSE
        log_rmse = torch.sqrt(((torch.log(pred_flat + 1e-6) - torch.log(target_flat + 1e-6)) ** 2).mean())

        return {
            'abs_rel': abs_rel,
            'rmse': rmse,
            'log_rmse': log_rmse
        }
