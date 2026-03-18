#!/usr/bin/env python3
"""Training script for D4RT."""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import PointOdysseyDataset, collate_fn
from losses import D4RTLoss
from models import create_d4rt


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


DEFAULT_DATA_ROOT = "/mnt/D4RT/datasets/PointOdyssey"
DEFAULT_PRETRAINED_WEIGHTS = None
DEFAULT_INPUT_IMAGE_SIZE = 256
LOSS_KEYS = (
    "loss",
    "loss_3d",
    "loss_raw_3d",
    "loss_2d",
    "loss_vis",
    "loss_disp",
    "loss_normal",
    "loss_conf",
)
EXTRA_METRIC_KEYS = (
    "metric_raw_3d_l1",
    "metric_raw_3d_euclidean",
    "metric_conf_mean",
    "metric_pred_abs_depth_mean",
    "metric_target_abs_depth_mean",
    "metric_pred_norm_depth_mean",
    "metric_target_norm_depth_mean",
)
METRIC_KEYS = LOSS_KEYS + EXTRA_METRIC_KEYS


def parse_args():
    parser = argparse.ArgumentParser(description="Train D4RT model")

    # Model
    parser.add_argument(
        "--encoder",
        type=str,
        default="base",
        choices=["base", "large", "huge", "giant"],
        help="Encoder variant",
    )
    parser.add_argument(
        "--decoder-depth",
        type=int,
        default=None,
        help="Number of decoder layers; defaults to the paper setting for the chosen encoder",
    )
    parser.add_argument("--img-size", type=int, default=DEFAULT_INPUT_IMAGE_SIZE, help="Input image size")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=48,
        help="Number of frames per clip",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=9,
        help="Local RGB patch size for queries",
    )
    parser.add_argument(
        "--patch-provider",
        type=str,
        default="auto",
        choices=[
            "auto",
            "sampled_resized",
            "precomputed_resized",
            "sampled_highres",
            "precomputed_highres",
        ],
        help="Query patch provider; current stable baseline is the resized patch route",
    )
    parser.add_argument(
        "--videomae-model",
        type=str,
        default=None,
        help="Optional Hugging Face model id or local path for VideoMAE encoder initialization",
    )
    parser.add_argument(
        "--zero-init-2d-residual-head",
        action="store_true",
        help=(
            "Zero-initialize the decoder 2D residual head so pos_2d starts as an identity mapping "
            "(pos_2d = coords) at initialization."
        ),
    )
    parser.add_argument(
        "--disable-query-patch-embedding",
        action="store_true",
        help="Debug option: zero out the decoder's local patch embedding term when building query tokens.",
    )
    parser.add_argument(
        "--disable-query-timestep-embedding",
        action="store_true",
        help="Debug option: zero out the decoder's source/target/camera timestep embeddings when building query tokens.",
    )
    parser.add_argument(
        "--disable-decoder-cross-attention",
        action="store_true",
        help="Debug option: bypass decoder cross-attention so queries are updated only by the decoder MLP blocks.",
    )
    parser.add_argument(
        "--debug-3d-head-mode",
        type=str,
        default="linear",
        choices=["linear", "mlp256"],
        help="Debug option for the 3D prediction head: keep the default linear head or replace it with a deeper 256-256 MLP.",
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=2048,
        help="Number of queries per batch",
    )
    parser.add_argument(
        "--query-chunk-size",
        type=int,
        default=0,
        help="Optional query chunk size for lower-memory decoding; 0 disables chunking",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Optional LR scheduler total steps; training length still follows epochs",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2500,
        help="Warmup steps",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.03,
        help="Weight decay",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=10.0,
        help="Gradient clipping (L2 norm)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Use automatic mixed precision",
    )

    # Memory optimization
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for faster training",
    )

    # Loss weights
    parser.add_argument("--lambda-3d", type=float, default=1.0)
    parser.add_argument("--lambda-raw-3d", type=float, default=0.0)
    parser.add_argument("--lambda-2d", type=float, default=0.1)
    parser.add_argument("--lambda-vis", type=float, default=0.1)
    parser.add_argument("--lambda-disp", type=float, default=0.1)
    parser.add_argument("--lambda-normal", type=float, default=0.5)
    parser.add_argument("--lambda-conf", type=float, default=0.2)
    parser.add_argument(
        "--conf-ramp-start-step",
        type=int,
        default=0,
        help="Optimizer step to start enabling confidence weighting/loss",
    )
    parser.add_argument(
        "--conf-ramp-steps",
        type=int,
        default=0,
        help="Number of optimizer steps to linearly ramp the confidence penalty to its target",
    )
    parser.add_argument(
        "--conf-weighting-start-step",
        type=int,
        default=-1,
        help="Optimizer step to start using confidence to reweight the 3D loss; defaults to conf-ramp-start-step",
    )
    parser.add_argument(
        "--conf-weighting-ramp-steps",
        type=int,
        default=-1,
        help="Number of optimizer steps to ramp confidence-based 3D weighting; defaults to conf-ramp-steps",
    )
    parser.add_argument(
        "--debug-3d-loss-mode",
        type=str,
        default="scale_invariant",
        choices=["scale_invariant", "raw_l1"],
        help=(
            "Debug-only override for loss_3d. 'scale_invariant' matches the paper-style mean-depth normalized "
            "log-L1 objective; 'raw_l1' uses direct camera-space XYZ L1 to test memorization without scale coupling."
        ),
    )

    # Data
    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root containing train/val/test",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pointodyssey",
        choices=["pointodyssey"],
        help="Dataset type",
    )
    parser.add_argument("--train-split", type=str, default="train", help="Train split")
    parser.add_argument("--val-split", type=str, default="val", help="Validation split")
    parser.add_argument("--test-split", type=str, default="test", help="Test split")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--val-every-epochs",
        type=int,
        default=1,
        help="Run validation every N epochs; 1 validates every epoch",
    )
    parser.add_argument(
        "--disable-precompute-local-patches",
        action="store_true",
        help="Skip CPU-side local patch precomputation and sample resized patches online in the decoder",
    )
    parser.add_argument(
        "--disable-train-augs",
        action="store_true",
        help="Disable train-time data augmentation for overfit/debug runs",
    )
    parser.add_argument(
        "--train-sequence",
        type=str,
        default=None,
        help="Optional sequence name to restrict the train split to a single sequence",
    )
    parser.add_argument(
        "--val-sequence",
        type=str,
        default=None,
        help="Optional sequence name to restrict the val split to a single sequence",
    )
    parser.add_argument(
        "--query-mode",
        type=str,
        default="full",
        choices=["full", "target_cam", "same_frame"],
        help="Query curriculum mode: full D4RT queries, t_cam=t_tgt, or t_src=t_tgt=t_cam",
    )
    parser.add_argument(
        "--t-tgt-eq-t-cam-ratio",
        type=float,
        default=0.4,
        help="For query_mode=full, probability of sampling t_cam=t_tgt",
    )
    parser.add_argument(
        "--disable-motion-boundary-oversampling",
        action="store_true",
        help="Disable motion-boundary hard-query oversampling; the default samples from depth or motion boundaries",
    )
    parser.add_argument(
        "--static-scene-frame-idx",
        type=int,
        default=None,
        help=(
            "Optional local clip frame index for the static-scene degradation test. "
            "When set, every per-frame modality and label in the sampled clip is replaced with this frame."
        ),
    )
    parser.add_argument(
        "--disable-grouped-3d-normalization",
        action="store_true",
        help=(
            "Disable the default per-t_cam grouped mean-depth normalization in loss_3d and "
            "normalize over all valid queries together instead."
        ),
    )
    parser.add_argument(
        "--skip-pointodyssey-sanity",
        action="store_true",
        help="Skip the PointOdyssey geometry sanity gate before training",
    )
    parser.add_argument(
        "--pointodyssey-sanity-split",
        type=str,
        default=None,
        help="Optional split to use for PointOdyssey sanity checks; defaults to sample, then val, then train if available",
    )
    parser.add_argument(
        "--pointodyssey-sanity-sequence",
        type=str,
        default=None,
        help="Optional sequence name to use for PointOdyssey sanity checks",
    )
    parser.add_argument(
        "--pointodyssey-sanity-max-frames-check",
        type=int,
        default=6,
        help="Maximum number of frames to sample inside the PointOdyssey sanity gate",
    )
    parser.add_argument(
        "--pointodyssey-sanity-max-points-per-frame",
        type=int,
        default=1024,
        help="Maximum number of points per frame to use in PointOdyssey sanity checks",
    )

    # Checkpointing and logging
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output dir")
    parser.add_argument(
        "--save-freq",
        type=int,
        default=0,
        help="Unused in epoch mode; kept for compatibility",
    )
    parser.add_argument(
        "--save-epochs",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=0,
        help="Unused in epoch mode; kept for compatibility",
    )
    parser.add_argument(
        "--log-epochs",
        type=int,
        default=1,
        help="Print metrics every N epochs",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from checkpoint_latest.pth if it exists",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        default=DEFAULT_PRETRAINED_WEIGHTS,
        help="Load full model weights before training",
    )
    parser.add_argument(
        "--pretrained-encoder",
        type=str,
        default=None,
        help="Load encoder-only weights before training",
    )

    # Distributed
    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Config
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")

    pre_args, _ = parser.parse_known_args()

    if pre_args.config:
        with open(pre_args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        parser.set_defaults(**{
            key.replace("-", "_"): value
            for key, value in config.items()
            if any(action.dest == key.replace("-", "_") for action in parser._actions)
        })

    args = parser.parse_args()
    return args


def setup_distributed():
    """Setup distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif torch.cuda.is_available():
        rank = 0
        world_size = 1
        local_rank = 0
    else:
        return 0, 1, 0

    torch.cuda.set_device(local_rank)

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    return rank, world_size, local_rank


def resolve_dataset_root(data_root, train_split):
    """Allow passing either the dataset root or the train split path."""
    data_root = os.path.abspath(data_root)
    if os.path.isdir(os.path.join(data_root, train_split)):
        return data_root
    if os.path.basename(data_root) == train_split:
        return os.path.dirname(data_root)
    return data_root


def split_dir_exists(data_root, split):
    return os.path.isdir(os.path.join(data_root, split))


def _sequence_for_split(args, split):
    if split == args.train_split:
        return args.train_sequence
    if split == args.val_split:
        return args.val_sequence if args.val_sequence is not None else args.train_sequence if args.val_split == args.train_split else None
    return None


def build_dataset(args, split, use_augs, verbose, deterministic_sampling):
    precompute_local_patches = (
        not args.disable_precompute_local_patches
        and args.patch_provider not in {"sampled_resized", "sampled_highres"}
    )
    return PointOdysseyDataset(
        dataset_location=args.data_root,
        dset=split,
        S=args.num_frames,
        img_size=args.img_size,
        num_queries=args.num_queries,
        patch_size=args.patch_size,
        use_augs=use_augs,
        deterministic_sampling=deterministic_sampling,
        verbose=verbose,
        sequence_name=_sequence_for_split(args, split),
        query_mode=args.query_mode,
        t_tgt_eq_t_cam_ratio=args.t_tgt_eq_t_cam_ratio,
        use_motion_boundaries=not args.disable_motion_boundary_oversampling,
        precompute_local_patches=precompute_local_patches,
        return_query_video=args.patch_provider == "sampled_highres",
        static_scene_frame_idx=args.static_scene_frame_idx,
    )


def _resolve_pointodyssey_sanity_split(args):
    if args.pointodyssey_sanity_split is not None:
        return args.pointodyssey_sanity_split
    for candidate in ("sample", args.val_split, args.train_split):
        if split_dir_exists(args.data_root, candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find a split for the PointOdyssey sanity gate under {args.data_root}. "
        "Tried sample, val, and train."
    )


def maybe_run_pointodyssey_sanity_gate(args, rank, world_size):
    if args.dataset != "pointodyssey" or args.skip_pointodyssey_sanity:
        return

    sanity_split = _resolve_pointodyssey_sanity_split(args)
    sanity_sequence = args.pointodyssey_sanity_sequence
    passed = True

    if rank == 0:
        sanity_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "scripts", "check_pointodyssey_sanity.py"),
            "--data-root",
            args.data_root,
            "--split",
            sanity_split,
            "--img-size",
            str(args.img_size),
            "--num-frames",
            str(min(args.num_frames, 8)),
            "--num-queries",
            str(min(args.num_queries, 256)),
            "--patch-size",
            str(args.patch_size),
            "--max-frames-check",
            str(args.pointodyssey_sanity_max_frames_check),
            "--max-points-per-frame",
            str(args.pointodyssey_sanity_max_points_per_frame),
            "--seed",
            str(args.seed),
        ]
        if sanity_sequence is not None:
            sanity_cmd.extend(["--sequence", sanity_sequence])

        print(
            f"Running PointOdyssey sanity gate on split={sanity_split}"
            + (f", sequence={sanity_sequence}" if sanity_sequence is not None else "")
        )
        result = subprocess.run(sanity_cmd, check=False)
        passed = result.returncode == 0

    if world_size > 1:
        status = torch.tensor(
            [1 if passed else 0],
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.int32,
        )
        dist.broadcast(status, src=0)
        passed = bool(status.item())

    if not passed:
        raise RuntimeError(
            "PointOdyssey sanity gate failed. Fix dataset geometry/labels before interpreting training results. "
            "Re-run with --skip-pointodyssey-sanity only if you intentionally want to bypass the preflight check."
        )


def create_dataloaders(args, rank, world_size):
    """Create train and validation dataloaders."""
    is_main = rank == 0

    train_dir = os.path.join(args.data_root, args.train_split)
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Train split not found: {train_dir}. "
            f"Expected {args.data_root}/{{{args.train_split},{args.val_split},{args.test_split}}}"
        )

    train_dataset = build_dataset(
        args,
        split=args.train_split,
        use_augs=not args.disable_train_augs,
        verbose=is_main,
        deterministic_sampling=False,
    )

    if len(train_dataset) == 0:
        raise RuntimeError(f"Train dataset is empty: {train_dir}")

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
    else:
        train_sampler = None

    train_samples_per_rank = len(train_sampler) if train_sampler is not None else len(train_dataset)
    train_drop_last = train_samples_per_rank >= args.batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=train_drop_last,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = None
    val_dir = os.path.join(args.data_root, args.val_split)
    if os.path.isdir(val_dir):
        val_dataset = build_dataset(
            args,
            split=args.val_split,
            use_augs=False,
            verbose=is_main,
            deterministic_sampling=True,
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
                persistent_workers=args.num_workers > 0,
            )
        elif is_main:
            print(f"Validation split is empty: {val_dir}")
    elif is_main:
        print(f"Validation split not found, skip validation: {val_dir}")

    return train_loader, train_sampler, val_loader


def create_optimizer_scheduler(model, args, total_steps):
    """Create optimizer and learning rate scheduler."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name.lower() or "ln" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)

        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def prepare_batch(batch, device):
    """Move a batch to device."""
    if batch is None or "video" not in batch:
        return None

    video = batch["video"].to(device, non_blocking=True)
    coords = batch["coords"].to(device, non_blocking=True)
    t_src = batch["t_src"].to(device, non_blocking=True)
    t_tgt = batch["t_tgt"].to(device, non_blocking=True)
    t_cam = batch["t_cam"].to(device, non_blocking=True)
    aspect_ratio = batch["aspect_ratio"].to(device, non_blocking=True)

    targets = {}
    if "targets" in batch:
        for key, value in batch["targets"].items():
            targets[key] = value.to(device, non_blocking=True)

    transform_metadata = None
    if "transform_metadata" in batch:
        transform_metadata = {}
        for key, value in batch["transform_metadata"].items():
            transform_metadata[key] = value.to(device, non_blocking=True)

    local_patches = None
    if "local_patches" in batch:
        local_patches = batch["local_patches"].to(device, non_blocking=True)

    video_query = None
    if "video_query" in batch:
        video_query = batch["video_query"].to(device, non_blocking=True)

    return {
        "video_input": video,
        "video_query": video_query,
        "coords": coords,
        "t_src": t_src,
        "t_tgt": t_tgt,
        "t_cam": t_cam,
        "aspect_ratio": aspect_ratio,
        "local_patches": local_patches,
        "transform_metadata": transform_metadata,
        "targets": targets,
    }


def iter_query_slices(total_queries, chunk_size):
    if chunk_size is None or chunk_size <= 0 or total_queries <= chunk_size:
        yield 0, total_queries
        return

    for start in range(0, total_queries, chunk_size):
        yield start, min(start + chunk_size, total_queries)


def slice_targets(targets, start, end):
    return {key: value[:, start:end] for key, value in targets.items()}


def losses_are_finite(losses):
    for value in losses.values():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            return False
    return True


def get_amp_dtype(device):
    if device.type != "cuda":
        return None
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(args, device):
    if not args.amp or device.type != "cuda":
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=get_amp_dtype(device),
    )


def run_query_chunks(model, prepared, criterion, scaler, args, device, backward=False):
    """Encode once, then decode query chunks to control activation memory."""
    total_queries = prepared["coords"].shape[1]
    core_model = model.module if hasattr(model, "module") else model

    with autocast_context(args, device):
        encoder_features = core_model.encode(
            prepared["video_input"],
            prepared["aspect_ratio"],
        )
    query_frames = prepared["video_query"] if prepared["video_query"] is not None else prepared["video_input"]

    metric_sums = init_metric_sums()
    total_weight = 0.0
    chunk_ranges = list(iter_query_slices(total_queries, args.query_chunk_size))

    for chunk_idx, (start, end) in enumerate(chunk_ranges):
        chunk_weight = (end - start) / max(1, total_queries)
        chunk_targets = slice_targets(prepared["targets"], start, end)
        chunk_local_patches = None
        if prepared["local_patches"] is not None:
            chunk_local_patches = prepared["local_patches"][:, start:end]

        with autocast_context(args, device):
            predictions = core_model.decode(
                encoder_features=encoder_features,
                frames=query_frames,
                coords=prepared["coords"][:, start:end],
                t_src=prepared["t_src"][:, start:end],
                t_tgt=prepared["t_tgt"][:, start:end],
                t_cam=prepared["t_cam"][:, start:end],
                local_patches=chunk_local_patches,
                transform_metadata=prepared["transform_metadata"],
            )
            normalize_groups = None
            if not args.disable_grouped_3d_normalization:
                normalize_groups = prepared["t_cam"][:, start:end]
            losses = criterion(
                predictions,
                chunk_targets,
                normalize_groups=normalize_groups,
            )
        if not losses_are_finite(losses):
            return None

        for key in METRIC_KEYS:
            if key in losses:
                metric_sums[key] += float(losses[key].item()) * chunk_weight
        total_weight += chunk_weight

        if backward:
            loss = losses["loss"] * chunk_weight / args.gradient_accumulation_steps
            retain_graph = chunk_idx < len(chunk_ranges) - 1
            if args.amp and scaler is not None:
                scaler.scale(loss).backward(retain_graph=retain_graph)
            else:
                loss.backward(retain_graph=retain_graph)

        del predictions
        del losses

    if total_weight <= 0:
        return {}

    return {key: value / total_weight for key, value in metric_sums.items()}


def forward_backward_step(model, batch, criterion, scaler, args, device, is_accumulating):
    """Run a single training micro-step."""
    prepared = prepare_batch(batch, device)
    if prepared is None:
        return {}

    sync_context = model.no_sync() if is_accumulating and hasattr(model, "no_sync") else nullcontext()

    with sync_context:
        losses = run_query_chunks(
            model=model,
            prepared=prepared,
            criterion=criterion,
            scaler=scaler,
            args=args,
            device=device,
            backward=True,
        )
        if losses is None:
            print("WARNING: Non-finite loss detected before backward. Skipping step.", flush=True)
            return {}
        loss_value = losses["loss"]
    return losses


def sync_gradients(model):
    """Synchronize manually accumulated gradients before a partial optimizer step."""
    if not dist.is_available() or not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world_size


def optimizer_step(model, optimizer, scheduler, scaler, args):
    """Perform optimizer step with gradient clipping."""
    if args.amp and scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def init_metric_sums():
    return {key: 0.0 for key in METRIC_KEYS}


def update_metric_sums(metric_sums, metrics):
    for key in METRIC_KEYS:
        if key in metrics:
            metric_sums[key] += float(metrics[key])


def reduce_metric_sums(metric_sums, count, device, world_size):
    packed = torch.tensor(
        [metric_sums[key] for key in METRIC_KEYS] + [float(count)],
        device=device,
        dtype=torch.float64,
    )
    if world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)

    total_count = packed[-1].item()
    if total_count <= 0:
        return {}

    return {
        key: packed[idx].item() / total_count
        for idx, key in enumerate(METRIC_KEYS)
    }


def format_metrics(metrics):
    if not metrics:
        return "n/a"
    return " | ".join(
        f"{key}: {value:.4f}"
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    )


def append_metrics_log(output_dir, epoch, step, train_metrics, val_metrics, extra=None):
    record = {
        "epoch": epoch,
        "step": step,
        "train": train_metrics,
        "val": val_metrics,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        record.update(extra)
    log_path = Path(output_dir) / "metrics_history.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    step,
    epoch,
    args,
    output_dir,
    train_metrics=None,
    val_metrics=None,
):
    """Save training checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "train_metrics": train_metrics or {},
        "val_metrics": val_metrics or {},
        "args": vars(args),
    }

    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch:04d}.pth"
    torch.save(checkpoint, checkpoint_path)
    torch.save(checkpoint, output_dir / "checkpoint_latest.pth")

    # Limit the number of checkpoints to 20
    all_checkpoints = sorted(output_dir.glob("checkpoint_epoch_*.pth"))
    if len(all_checkpoints) > 20:
        for ckpt in all_checkpoints[:-20]:
            try:
                ckpt.unlink()
                print(f"Removed old checkpoint: {ckpt}")
            except Exception as e:
                print(f"Error removing checkpoint {ckpt}: {e}")

    return checkpoint_path


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format")


def strip_common_prefix(state_dict, prefix):
    if not state_dict:
        return state_dict
    if all(key.startswith(prefix) for key in state_dict.keys()):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def normalize_state_dict_keys(state_dict):
    state_dict = strip_common_prefix(state_dict, "module.")
    state_dict = strip_common_prefix(state_dict, "model.")
    return state_dict


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """Load full training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = normalize_state_dict_keys(extract_state_dict(checkpoint))

    target_model = model.module if hasattr(model, "module") else model
    target_model.load_state_dict(model_state)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return (
        checkpoint.get("step", 0),
        checkpoint.get("epoch", 0),
        checkpoint.get("val_metrics", {}),
    )


def load_pretrained_weights(checkpoint_path, model, is_main=False):
    """Load model weights only, without optimizer/scheduler state."""
    if not checkpoint_path:
        return False
    if not os.path.exists(checkpoint_path):
        if is_main:
            print(f"Pretrained weights not found, skip loading: {checkpoint_path}")
        return False

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = normalize_state_dict_keys(extract_state_dict(checkpoint))
    target_model = model.module if hasattr(model, "module") else model
    missing, unexpected = target_model.load_state_dict(state_dict, strict=False)

    if is_main:
        print(f"Loaded pretrained weights from {checkpoint_path}")
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
    return True


def load_pretrained_encoder(checkpoint_path, model, is_main=False):
    """Load encoder-only weights."""
    if not checkpoint_path:
        return False
    if not os.path.exists(checkpoint_path):
        if is_main:
            print(f"Pretrained encoder weights not found, skip loading: {checkpoint_path}")
        return False

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = normalize_state_dict_keys(extract_state_dict(checkpoint))
    encoder_state = {
        key[len("encoder."):]: value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        encoder_state = state_dict
    target_model = model.module if hasattr(model, "module") else model
    missing, unexpected = target_model.encoder.load_state_dict(encoder_state, strict=False)

    if is_main:
        print(f"Loaded pretrained encoder from {checkpoint_path}")
        if missing:
            print(f"Missing encoder keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected encoder keys: {len(unexpected)}")
    return True


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def compute_linear_ramp(step, start_step, ramp_steps):
    start_step = max(0, int(start_step))
    ramp_steps = max(0, int(ramp_steps))
    if step < start_step:
        return 0.0
    if ramp_steps == 0:
        return 1.0
    return min(1.0, max(0.0, (step - start_step) / float(ramp_steps)))


def compute_confidence_schedule(step, args):
    target_lambda_conf = float(args.lambda_conf)
    if target_lambda_conf <= 0.0:
        return 0.0, 0.0

    penalty_factor = compute_linear_ramp(step, args.conf_ramp_start_step, args.conf_ramp_steps)
    weighting_start = args.conf_weighting_start_step if args.conf_weighting_start_step >= 0 else args.conf_ramp_start_step
    weighting_ramp = args.conf_weighting_ramp_steps if args.conf_weighting_ramp_steps >= 0 else args.conf_ramp_steps
    weighting_factor = compute_linear_ramp(step, weighting_start, weighting_ramp)
    return target_lambda_conf * penalty_factor, weighting_factor


@torch.no_grad()
def run_validation(model, val_dataloader, criterion, args, device):
    """Run validation loop."""
    was_training = model.training
    model.eval()

    metric_sums = init_metric_sums()
    batch_count = 0

    for batch in val_dataloader:
        prepared = prepare_batch(batch, device)
        if prepared is None:
            continue

        losses = run_query_chunks(
            model=model,
            prepared=prepared,
            criterion=criterion,
            scaler=None,
            args=args,
            device=device,
            backward=False,
        )
        if losses is None:
            print("WARNING: Non-finite validation loss detected. Skipping batch.", flush=True)
            continue
        update_metric_sums(metric_sums, losses)
        batch_count += 1

    if was_training:
        model.train()

    if batch_count == 0:
        return {}

    return {key: metric_sums[key] / batch_count for key in METRIC_KEYS}


def main():
    args = parse_args()

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if args.amp and device.type != "cuda":
        if is_main:
            print("AMP requested without CUDA. Disabling AMP.")
        args.amp = False

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + rank)

    args.data_root = resolve_dataset_root(args.data_root, args.train_split)
    maybe_run_pointodyssey_sanity_gate(args, rank, world_size)
    output_dir = Path(args.output_dir)

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(vars(args), f, default_flow_style=False, sort_keys=True)

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size

    if is_main:
        print("=" * 60)
        print("D4RT Training")
        print("=" * 60)
        print(f"Data root: {args.data_root}")
        print(f"Train split: {args.train_split}")
        print(f"Val split: {args.val_split}")
        print(f"Test split exists: {split_dir_exists(args.data_root, args.test_split)}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Train augmentations: {not args.disable_train_augs}")
        print(f"Query mode: {args.query_mode}")
        print(f"t_tgt=t_cam ratio: {args.t_tgt_eq_t_cam_ratio:.2f}")
        print(f"Motion-boundary oversampling: {not args.disable_motion_boundary_oversampling}")
        print(f"Zero-init 2D residual head: {args.zero_init_2d_residual_head}")
        print(f"Disable query patch embedding: {args.disable_query_patch_embedding}")
        print(f"Disable query timestep embedding: {args.disable_query_timestep_embedding}")
        print(f"Disable decoder cross-attention: {args.disable_decoder_cross_attention}")
        print(f"3D head mode: {args.debug_3d_head_mode}")
        print(f"Grouped 3D normalization: {not args.disable_grouped_3d_normalization}")
        print(f"3D loss mode: {args.debug_3d_loss_mode}")
        print(
            "Static-scene degradation: "
            + (
                f"enabled (frame {args.static_scene_frame_idx})"
                if args.static_scene_frame_idx is not None
                else "disabled"
            )
        )
        print(f"PointOdyssey sanity gate: {not args.skip_pointodyssey_sanity}")
        if args.train_sequence is not None:
            print(f"Train sequence filter: {args.train_sequence}")
        if args.val_sequence is not None or (args.val_split == args.train_split and args.train_sequence is not None):
            print(f"Val sequence filter: {args.val_sequence or args.train_sequence}")
        print(f"AMP: {args.amp}")
        if args.amp and device.type == "cuda":
            print(f"AMP dtype: {get_amp_dtype(device)}")
        print(f"Loss weights: lambda_3d={args.lambda_3d:.3f}, lambda_raw_3d={args.lambda_raw_3d:.3f}, lambda_2d={args.lambda_2d:.3f}, "
              f"lambda_vis={args.lambda_vis:.3f}, lambda_disp={args.lambda_disp:.3f}, lambda_normal={args.lambda_normal:.3f}, lambda_conf={args.lambda_conf:.3f}")
        if args.lambda_conf > 0.0:
            weighting_start = args.conf_weighting_start_step if args.conf_weighting_start_step >= 0 else args.conf_ramp_start_step
            weighting_ramp = args.conf_weighting_ramp_steps if args.conf_weighting_ramp_steps >= 0 else args.conf_ramp_steps
            print(
                f"Confidence schedule: target={args.lambda_conf:.4f}, penalty_start={args.conf_ramp_start_step}, "
                f"penalty_ramp={args.conf_ramp_steps}, weighting_start={weighting_start}, weighting_ramp={weighting_ramp}"
            )
        print(f"Epochs: {args.epochs}")
        print(f"Save every {args.save_epochs} epochs")
        print(f"Validate every {max(1, args.val_every_epochs)} epochs")
        print(
            "Precompute local patches on CPU: "
            f"{not args.disable_precompute_local_patches and args.patch_provider not in {'sampled_resized', 'sampled_highres'}}"
        )
        print("=" * 60)

    model_kwargs = dict(
        variant=args.encoder,
        img_size=args.img_size,
        temporal_size=args.num_frames,
        query_patch_size=args.patch_size,
        disable_query_patch_embedding=args.disable_query_patch_embedding,
        disable_query_timestep_embedding=args.disable_query_timestep_embedding,
        disable_decoder_cross_attention=args.disable_decoder_cross_attention,
        debug_3d_head_mode=args.debug_3d_head_mode,
    )
    if args.decoder_depth is not None:
        model_kwargs["decoder_depth"] = args.decoder_depth
    if args.videomae_model:
        model_kwargs["videomae_model"] = args.videomae_model
    model_kwargs["patch_provider"] = args.patch_provider

    model = create_d4rt(**model_kwargs)
    if args.zero_init_2d_residual_head:
        nn.init.zeros_(model.decoder.head_2d.weight)
        nn.init.zeros_(model.decoder.head_2d.bias)
        if is_main:
            print("Zero-initialized decoder 2D residual head")

    if args.gradient_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
        if is_main:
            print("Gradient checkpointing enabled")

    if args.pretrained_weights and not args.resume:
        load_pretrained_weights(args.pretrained_weights, model, is_main=is_main)
    elif args.pretrained_encoder and not args.resume:
        load_pretrained_encoder(args.pretrained_encoder, model, is_main=is_main)

    model = model.to(device)

    if args.compile and hasattr(torch, "compile"):
        if is_main:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if is_main:
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    criterion = D4RTLoss(
        lambda_3d=args.lambda_3d,
        lambda_raw_3d=args.lambda_raw_3d,
        lambda_2d=args.lambda_2d,
        lambda_vis=args.lambda_vis,
        lambda_disp=args.lambda_disp,
        lambda_normal=args.lambda_normal,
        lambda_conf=args.lambda_conf,
        debug_3d_loss_mode=args.debug_3d_loss_mode,
    )
    current_lambda_conf, current_conf_factor = compute_confidence_schedule(start_step if 'start_step' in locals() else 0, args)
    criterion.set_confidence_schedule(current_lambda_conf, current_conf_factor)

    train_loader, train_sampler, val_loader = create_dataloaders(args, rank, world_size)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps)))
    scheduler_total_steps = args.steps if args.steps > 0 else args.epochs * steps_per_epoch
    max_epochs = args.epochs if args.steps <= 0 else max(1, math.ceil(args.steps / steps_per_epoch))

    if len(train_loader) == 0:
        raise RuntimeError(
            "Train loader is empty. This usually means batch_size is larger than the number of samples available on each rank. "
            f"train_sequences={len(train_loader.dataset)}, batch_size={args.batch_size}, world_size={world_size}."
        )

    if is_main:
        print(f"Train sequences: {len(train_loader.dataset)}")
        print(f"Train batches per epoch: {len(train_loader)}")
        print(f"Val loader ready: {val_loader is not None}")
        print(f"Optimizer steps per epoch: {steps_per_epoch}")
        print(f"Scheduler total steps: {scheduler_total_steps}")
        if args.steps > 0:
            print(f"Training will stop after {args.steps} optimizer steps (~{max_epochs} epochs)")
        if args.batch_size != 1:
            print(
                f"WARNING: Current batch_size={args.batch_size}, but the paper trains with batch_size=1 per device. "
                "This changes steps/epoch and LR schedule."
            )
        if args.warmup_steps >= max(1, scheduler_total_steps // 4):
            print(
                f"WARNING: warmup_steps={args.warmup_steps} is large relative to total_steps={scheduler_total_steps}. "
                "If you are not training for paper-scale step counts, reduce warmup."
            )

    optimizer, scheduler = create_optimizer_scheduler(model, args, scheduler_total_steps)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and get_amp_dtype(device) == torch.float16,
    )

    writer = None
    if is_main and TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(output_dir / "tensorboard")

    start_step = 0
    start_epoch = 0

    if args.auto_resume and (output_dir / "checkpoint_latest.pth").exists():
        args.resume = str(output_dir / "checkpoint_latest.pth")

    if args.resume:
        if is_main:
            print(f"Resuming from {args.resume}")
        start_step, start_epoch, _ = load_checkpoint(
            args.resume,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

    if is_main:
        print(f"Starting training from epoch {start_epoch + 1}, step {start_step}")

    train_start_time = time.time()
    step = start_step
    stop_training = False

    for epoch_idx in range(start_epoch, max_epochs):
        completed_epoch = epoch_idx + 1
        epoch_start_time = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch_idx)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_metric_sums = init_metric_sums()
        local_batch_count = 0
        accum_count = 0

        for batch in train_loader:
            current_lambda_conf, current_conf_factor = compute_confidence_schedule(step, args)
            criterion.set_confidence_schedule(current_lambda_conf, current_conf_factor)

            next_accum = accum_count + 1
            is_accumulating = (next_accum % args.gradient_accumulation_steps) != 0

            losses = forward_backward_step(
                model=model,
                batch=batch,
                criterion=criterion,
                scaler=scaler,
                args=args,
                device=device,
                is_accumulating=is_accumulating,
            )
            if not losses:
                continue

            update_metric_sums(train_metric_sums, losses)
            local_batch_count += 1
            accum_count = next_accum

            if accum_count % args.gradient_accumulation_steps == 0:
                optimizer_step(model, optimizer, scheduler, scaler, args)
                step += 1
                accum_count = 0
                if step >= scheduler_total_steps:
                    stop_training = True
                    break

        if accum_count > 0 and not stop_training:
            sync_gradients(model)
            optimizer_step(model, optimizer, scheduler, scaler, args)
            step += 1
            if step >= scheduler_total_steps:
                stop_training = True

        train_metrics = reduce_metric_sums(train_metric_sums, local_batch_count, device, world_size)

        if world_size > 1:
            dist.barrier()

        current_lambda_conf, current_conf_factor = compute_confidence_schedule(step, args)
        criterion.set_confidence_schedule(current_lambda_conf, current_conf_factor)

        val_metrics = {}
        should_validate = (
            val_loader is not None
            and (completed_epoch % max(1, args.val_every_epochs) == 0 or completed_epoch == max_epochs or stop_training)
        )
        if is_main and should_validate:
            validation_model = model.module if hasattr(model, "module") else model
            val_metrics = run_validation(validation_model, val_loader, criterion, args, device)

        if world_size > 1:
            dist.barrier()

        elapsed = time.time() - train_start_time
        if args.steps > 0:
            completed_steps = max(1, step - start_step)
            eta_seconds = max(0.0, scheduler_total_steps - step) * (elapsed / completed_steps)
        else:
            epochs_done = max(1, epoch_idx - start_epoch + 1)
            eta_seconds = (args.epochs - completed_epoch) * (elapsed / epochs_done)
        lr = scheduler.get_last_lr()[0]

        if is_main and completed_epoch % max(1, args.log_epochs) == 0:
            print(
                f"Epoch {completed_epoch}/{max_epochs} | "
                f"Step {step}/{scheduler_total_steps} | "
                f"LR: {lr:.2e} | "
                f"ConfLambda: {current_lambda_conf:.3e} | "
                f"ConfBlend: {current_conf_factor:.2f} | "
                f"Train: {format_metrics(train_metrics)} | "
                f"Time: {format_time(time.time() - epoch_start_time)} | "
                f"ETA: {format_time(eta_seconds)}"
            )
            if should_validate:
                print(f"Validation {completed_epoch}/{max_epochs} | {format_metrics(val_metrics)}")
            else:
                print(f"Validation {completed_epoch}/{max_epochs} | skipped")

        if is_main:
            append_metrics_log(
                output_dir,
                completed_epoch,
                step,
                train_metrics,
                val_metrics,
                extra={
                    "lr": lr,
                    "lambda_conf_effective": current_lambda_conf,
                    "confidence_weighting_factor": current_conf_factor,
                },
            )

            if writer is not None:
                writer.add_scalar("train/lr", lr, completed_epoch)
                for key, value in train_metrics.items():
                    writer.add_scalar(f"train_epoch/{key}", value, completed_epoch)
                for key, value in val_metrics.items():
                    writer.add_scalar(f"val_epoch/{key}", value, completed_epoch)
                writer.flush()

            should_save = (
                args.save_epochs > 0 and completed_epoch % args.save_epochs == 0
            ) or completed_epoch == max_epochs or stop_training
            if should_save:
                checkpoint_path = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    step=step,
                    epoch=completed_epoch,
                    args=args,
                    output_dir=output_dir,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                )
                print(f"Saved checkpoint to {checkpoint_path}")

        if stop_training:
            break

    if is_main:
        print("\nTraining complete")
        print(f"Total time: {format_time(time.time() - train_start_time)}")
        if writer is not None:
            writer.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
