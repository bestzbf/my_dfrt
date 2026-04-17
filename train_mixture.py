#!/usr/bin/env python3
"""Mixed dataset training for D4RT."""

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from datasets.factory import create_training_dataset
from datasets.collate import d4rt_collate_fn
from models import create_d4rt
from losses import D4RTLoss
import json
import time
import os
import contextlib
import math


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    batch = {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    batch["targets"] = {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch["targets"].items()
    }
    transform_metadata = batch.get("transform_metadata")
    if isinstance(transform_metadata, dict):
        batch["transform_metadata"] = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in transform_metadata.items()
        }
    if batch["video"].dtype == torch.uint8:
        batch["video"] = batch["video"].float() / 255.0
    return batch


def maybe_fallback_patch_provider(
    model: nn.Module,
    batch: dict,
    configured_provider: str,
    local_rank: int,
    warned: bool,
) -> bool:
    if configured_provider != "sampled_highres":
        return warned
    if batch.get("transform_metadata") is not None:
        return warned

    decoder = unwrap_model(model).decoder
    if getattr(decoder, "patch_provider", None) == "sampled_highres":
        decoder.patch_provider = "auto"
    if local_rank == 0 and not warned:
        print(
            "[Patch Provider] WARNING: batch is missing transform_metadata/highres_video; "
            "falling back from sampled_highres to auto.",
            flush=True,
        )
    return True


def set_dataset_epoch(dataset, epoch: int) -> None:
    current = dataset
    while current is not None:
        if hasattr(current, "set_epoch"):
            current.set_epoch(epoch)
            return
        current = getattr(current, "dataset", None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Dataset config YAML")
    parser.add_argument("--val-config", type=str, default=None, help="Separate val config YAML (optional, defaults to --config)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--lr-warmup-steps", type=int, default=2500)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument("--num-queries", type=int, default=2048)
    parser.add_argument("--loss-w-3d", type=float, default=1.0)
    parser.add_argument("--loss-w-raw-3d", type=float, default=0.0)
    parser.add_argument("--loss-w-2d", type=float, default=0.1)
    parser.add_argument("--loss-w-vis", type=float, default=0.1)
    parser.add_argument("--loss-w-disp", type=float, default=0.1)
    parser.add_argument("--loss-w-conf", type=float, default=0.2)
    parser.add_argument("--loss-w-normal", type=float, default=0.5)
    parser.add_argument("--loss-w-static-reprojection", type=float, default=1.0,
                        help="Weight for static-reprojection (has_tracks=False) queries in 3D loss. "
                             "Default 1.0 = no change. Set <1.0 to down-weight static queries.")
    parser.add_argument("--shared-depth-norm", action="store_true", default=True,
                        help="Use target mean-depth to normalize both pred and target (scale-aware). "
                             "Default True. Use --no-shared-depth-norm for paper's independent normalization.")
    parser.add_argument("--no-shared-depth-norm", dest="shared_depth_norm", action="store_false",
                        help="Use independent normalization (paper default): pred and target each "
                             "divided by their own mean depth. Scale-invariant but blind to depth-scale drift.")
    parser.add_argument("--loss-3d-mode", type=str, default="scale_invariant",
                        choices=["scale_invariant", "raw_l1", "log_space"],
                        help="3D loss mode. 'scale_invariant': paper default (median-norm + log1p). "
                             "'log_space': depth-invariant log/angular loss (no normalization needed). "
                             "'raw_l1': raw L1 for debugging.")
    parser.add_argument("--output-dir", type=str, default="outputs/mixture")
    parser.add_argument("--pretrain", type=str, default=None, help="Path to pretrained checkpoint")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--quick-test", action="store_true", help="Quick test mode with only 10 samples")
    parser.add_argument("--save-interval", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--val-interval", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--val-samples", type=int, default=200, help="Number of val samples per validation run")
    parser.add_argument("--keep-checkpoints", type=int, default=10, help="Keep last N checkpoints (except milestone)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument(
        "--patch-provider", type=str, default="auto",
        help=(
            "Patch provider: auto | precomputed_resized | sampled_resized | sampled_highres.\n"
            "  auto              → resolves to precomputed_resized when local_patches is precomputed.\n"
            "  sampled_highres   → samples from highres_video (original crop res); requires\n"
            "                      dataset config with precompute_patches: false.\n"
            "  NOTE: 'auto' does NOT enable high-res patches. For the paper's strongest setting\n"
            "  use --patch-provider sampled_highres with a config that sets precompute_patches: false."
        )
    )
    args = parser.parse_args()

    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load val config (if separate)
    val_config = config
    if args.val_config is not None:
        with open(args.val_config) as f:
            val_config = yaml.safe_load(f)

    # Create dataset
    train_dataset = create_training_dataset(config, split='train')
    val_loader = None
    if args.val_interval > 0 and args.val_samples > 0:
        val_dataset = create_training_dataset(val_config, split='val')
        from torch.utils.data import Subset
        val_dataset = Subset(val_dataset, range(min(args.val_samples, len(val_dataset))))
    else:
        val_dataset = None
    if local_rank == 0:
        val_len = len(val_dataset) if val_dataset is not None else 0
        print(f"Dataset: {train_dataset.get_dataset_names()}")
        print(f"Train Length: {len(train_dataset)}, Val Length: {val_len}")
        print(f"[Patch Provider] configured={args.patch_provider!r} "
              f"{'⚠ WARNING: high-res patches NOT active (use --patch-provider sampled_highres for paper setting)' if args.patch_provider == 'auto' else ''}")

    # Quick validation mode: use only first 10 samples
    if args.quick_test:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(min(10, len(train_dataset))))
        if local_rank == 0:
            print(f"Quick test mode: using {len(train_dataset)} samples")

    # DataLoader sampler setup
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = RandomSampler(train_dataset)

    train_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=d4rt_collate_fn,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    if args.num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    if val_dataset is not None:
        val_num_workers = max(2, args.num_workers // 2)
        if distributed:
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        else:
            val_sampler = SequentialSampler(val_dataset)
        val_loader_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=val_num_workers,
            collate_fn=d4rt_collate_fn,
            sampler=val_sampler,
            pin_memory=True,
            persistent_workers=val_num_workers > 0,
            drop_last=False,
        )
        if val_num_workers > 0:
            val_loader_kwargs["prefetch_factor"] = 2
        val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    # Setup device and model
    device = torch.device(f"cuda:{local_rank}")
    amp_enabled = (device.type == "cuda")
    model = create_d4rt(variant="base", decoder_depth=6, img_size=args.resolution,
                        temporal_size=args.num_frames, patch_size=(2, 16, 16),
                        query_patch_size=9, videomae_model="/data1/zbf/pretrained/videomae-base",
                        patch_provider=args.patch_provider,).to(device)
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
        )

    # Load pretrained weights
    if args.pretrain:
        if local_rank == 0:
            print(f"Loading pretrained weights from {args.pretrain}")
        checkpoint = torch.load(args.pretrain, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model'))
        if state_dict is None:
            raise KeyError(
                f"Checkpoint {args.pretrain} missing both 'model_state_dict' and 'model' keys"
            )
        unwrap_model(model).load_state_dict(state_dict, strict=True)

    # Optimizer and loss - separate weight decay for bias/norm
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay for bias, norm, and embedding parameters
        if 'bias' in name or 'norm' in name or 'embed' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=args.lr)
    loss_fn = D4RTLoss(lambda_3d=args.loss_w_3d, lambda_raw_3d=args.loss_w_raw_3d,
                       lambda_2d=args.loss_w_2d,
                       lambda_vis=args.loss_w_vis, lambda_disp=args.loss_w_disp,
                       lambda_conf=args.loss_w_conf, lambda_normal=args.loss_w_normal,
                       static_reprojection_weight=args.loss_w_static_reprojection,
                       shared_depth_normalization=args.shared_depth_norm,
                       debug_3d_loss_mode=args.loss_3d_mode)

    # LR scheduler: warmup + cosine annealing
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.epochs * optimizer_steps_per_epoch
    def lr_lambda(step):
        if step < args.lr_warmup_steps:
            return step / args.lr_warmup_steps
        progress = (step - args.lr_warmup_steps) / (total_steps - args.lr_warmup_steps)
        return args.lr_min / args.lr + (1 - args.lr_min / args.lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_list = []  # Track regular checkpoints for rotation

    start_epoch = 0
    global_step = 0
    warned_patch_provider_fallback = False

    # Resume from checkpoint
    if args.resume:
        if local_rank == 0:
            print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', start_epoch * optimizer_steps_per_epoch)
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            # Advance scheduler to match current step for older checkpoints.
            for _ in range(global_step):
                scheduler.step()
        if local_rank == 0:
            print(f"Resumed at epoch {start_epoch}, global_step {global_step}")

    if local_rank == 0:
        print(f"Starting training for {args.epochs} epochs")
        effective_batch = args.batch_size * args.grad_accum * (world_size if distributed else 1)
        print(f"Distributed: {distributed}, world_size: {world_size}")
        print(f"Grad accum steps: {args.grad_accum}, effective batch size: {effective_batch}")
        print(f"Steps per epoch: {len(train_loader)}, samples per epoch: {len(train_dataset)}")
        print(f"Optimizer steps per epoch: {optimizer_steps_per_epoch}")
    for epoch in range(start_epoch, args.epochs):
        if distributed and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        set_dataset_epoch(train_dataset, epoch)
        model.train()
        epoch_loss = 0
        optimizer.zero_grad(set_to_none=True)
        t_data_start = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            t_data_end = time.perf_counter()
            t_data = t_data_end - t_data_start

            batch = move_batch_to_device(batch, device)
            warned_patch_provider_fallback = maybe_fallback_patch_provider(
                model, batch, args.patch_provider, local_rank, warned_patch_provider_fallback
            )

            query_frames_arg = batch.get('highres_video') if args.patch_provider == 'sampled_highres' else None
            if local_rank == 0 and epoch == start_epoch and batch_idx == 0:
                info = unwrap_model(model).decoder.get_patch_provider_info(
                    batch.get('local_patches'),
                    batch.get('transform_metadata'),
                )
                print("\n[Patch Provider Info]", flush=True)
                print(f"  Configured: {info['configured']}", flush=True)
                print(f"  Resolved: {info['resolved']}", flush=True)
                print(f"  Has local_patches: {info['has_local_patches']}", flush=True)
                print(f"  Has transform_metadata: {info['has_transform_metadata']}", flush=True)
                print(f"  Has highres_video: {batch.get('highres_video') is not None}", flush=True)

            is_last_accum = (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader)

            t_fwd_start = time.perf_counter()
            # 梯度累积：只在最后一步才同步梯度，减少NCCL通信频次
            use_no_sync = distributed and hasattr(model, "no_sync") and not is_last_accum
            ctx = model.no_sync() if use_no_sync else contextlib.nullcontext()
            with ctx:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                    outputs = model(batch['video'], batch['coords'], batch['t_src'], batch['t_tgt'], batch['t_cam'],
                                    aspect_ratio=batch.get('aspect_ratio'),
                                    local_patches=batch.get('local_patches'),
                                    transform_metadata=batch.get('transform_metadata'),
                                    query_frames=query_frames_arg,)
                    # Per-(dataset, frame) depth normalization:
                    # dataset_id * num_frames + t_cam gives each (dataset, frame) pair a unique
                    # group id, so mean-depth is computed independently per frame within each
                    # dataset. This is correct for both single- and multi-dataset batches.
                    normalize_groups = batch['dataset_id'] * args.num_frames + batch['t_cam']
                    loss_dict = loss_fn(outputs, batch['targets'], normalize_groups=normalize_groups)
                    loss = loss_dict['loss'] / args.grad_accum

                loss.backward()

            if is_last_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            t_fwd = time.perf_counter() - t_fwd_start

            real_loss = loss.item() * args.grad_accum  # 还原除法，得到真实loss值
            epoch_loss += real_loss
            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                lr_info = f"LR: {current_lr:.2e} (warmup {global_step}/{args.lr_warmup_steps} → {args.lr:.2e})" \
                    if global_step < args.lr_warmup_steps else f"LR: {current_lr:.2e}"
                # Print from ALL ranks so we can compare data/compute time per rank
                print(f"[{time.strftime('%H:%M:%S')}][rank{local_rank}] Epoch {epoch}, Batch {batch_idx}, "
                      f"data={t_data*1000:.0f}ms fwd+bwd={t_fwd*1000:.0f}ms Loss: {real_loss:.4f}, {lr_info}", flush=True)

                if local_rank == 0:
                    # Save loss log
                    log_entry = {
                        'epoch': epoch,
                        'step': global_step,
                        'batch': batch_idx,
                        'loss': f"{real_loss:.4f}",
                        'loss_3d': f"{loss_dict.get('loss_3d', 0):.4f}",
                        'loss_3d_nocon': f"{loss_dict.get('loss_3d_unweighted', 0):.4f}",
                        'loss_raw_3d': f"{loss_dict.get('loss_raw_3d', 0):.4f}",
                        'loss_2d': f"{loss_dict.get('loss_2d', 0):.4f}",
                        'loss_vis': f"{loss_dict.get('loss_vis', 0):.4f}",
                        'loss_disp': f"{loss_dict.get('loss_disp', 0):.4f}",
                        'loss_conf': f"{loss_dict.get('loss_conf', 0):.4f}",
                        'loss_normal': f"{loss_dict.get('loss_normal', 0):.4f}",
                        'raw_3d_l1': f"{loss_dict.get('metric_raw_3d_l1', 0):.4f}",
                        'raw_3d_euc': f"{loss_dict.get('metric_raw_3d_euclidean', 0):.4f}",
                        'raw_3d_l1_static': f"{loss_dict.get('metric_raw_3d_l1_static', 0):.4f}",
                        'raw_3d_l1_temporal': f"{loss_dict.get('metric_raw_3d_l1_temporal', 0):.4f}",
                        'raw_3d_euc_static': f"{loss_dict.get('metric_raw_3d_euclidean_static', 0):.4f}",
                        'raw_3d_euc_temporal': f"{loss_dict.get('metric_raw_3d_euclidean_temporal', 0):.4f}",
                        'loss_3d_static_nocon': f"{loss_dict.get('metric_loss_3d_static_unweighted', 0):.4f}",
                        'loss_3d_temporal_nocon': f"{loss_dict.get('metric_loss_3d_temporal_unweighted', 0):.4f}",
                        'valid_3d_ratio': f"{loss_dict.get('metric_valid_3d_query_ratio', 0):.4f}",
                        'static_query_ratio': f"{loss_dict.get('metric_static_query_ratio', 0):.4f}",
                        'temporal_query_ratio': f"{loss_dict.get('metric_temporal_query_ratio', 0):.4f}",
                        'static_valid3d_ratio': f"{loss_dict.get('metric_static_valid3d_ratio', 0):.4f}",
                        'temporal_valid3d_ratio': f"{loss_dict.get('metric_temporal_valid3d_ratio', 0):.4f}",
                        'normal_query_ratio': f"{loss_dict.get('metric_normal_query_ratio', 0):.4f}",
                        'normal_valid3d_ratio': f"{loss_dict.get('metric_normal_valid3d_ratio', 0):.4f}",
                        'conf_mean': f"{loss_dict.get('metric_conf_mean', 0):.4f}",
                        'lr': f"{current_lr:.6f}"
                    }
                    with open(output_dir / 'loss_log.jsonl', 'a') as f:
                        f.write(json.dumps(log_entry) + '\n')

            t_data_start = time.perf_counter()

        avg_loss = epoch_loss / len(train_loader)
        if local_rank == 0:
            print(f"Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}")

        # Validation
        if val_loader is not None and (epoch + 1) % args.val_interval == 0:
            model.eval()
            val_metrics = {'loss': 0, 'loss_3d': 0, 'loss_3d_nocon': 0, 'loss_raw_3d': 0, 'loss_2d': 0, 'loss_vis': 0,
                          'loss_disp': 0, 'loss_conf': 0, 'loss_normal': 0,
                          'raw_3d_l1': 0, 'raw_3d_euc': 0,
                          'raw_3d_l1_static': 0, 'raw_3d_l1_temporal': 0,
                          'raw_3d_euc_static': 0, 'raw_3d_euc_temporal': 0,
                          'loss_3d_static_nocon': 0, 'loss_3d_temporal_nocon': 0,
                          'valid_3d_ratio': 0,
                          'static_query_ratio': 0, 'temporal_query_ratio': 0,
                          'static_valid3d_ratio': 0, 'temporal_valid3d_ratio': 0,
                          'normal_query_ratio': 0, 'normal_valid3d_ratio': 0,
                          'conf_mean': 0}
            with torch.no_grad():
                for batch in val_loader:
                    batch = move_batch_to_device(batch, device)
                    warned_patch_provider_fallback = maybe_fallback_patch_provider(
                        model, batch, args.patch_provider, local_rank, warned_patch_provider_fallback
                    )
                    query_frames_arg = batch.get('highres_video') if args.patch_provider == 'sampled_highres' else None
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                        outputs = model(batch['video'], batch['coords'], batch['t_src'], batch['t_tgt'], batch['t_cam'],
                                       aspect_ratio=batch.get('aspect_ratio'),
                                       local_patches=batch.get('local_patches'),
                                       transform_metadata=batch.get('transform_metadata'),
                                       query_frames=query_frames_arg,)
                        loss_dict = loss_fn(outputs, batch['targets'],
                                           normalize_groups=batch['dataset_id'] * args.num_frames + batch['t_cam'])
                    val_metrics['loss'] += loss_dict['loss'].item()
                    val_metrics['loss_3d'] += loss_dict.get('loss_3d', 0).item()
                    val_metrics['loss_3d_nocon'] += loss_dict.get('loss_3d_unweighted', 0).item()
                    val_metrics['loss_raw_3d'] += loss_dict.get('loss_raw_3d', 0).item()
                    val_metrics['loss_2d'] += loss_dict.get('loss_2d', 0).item()
                    val_metrics['loss_vis'] += loss_dict.get('loss_vis', 0).item()
                    val_metrics['loss_disp'] += loss_dict.get('loss_disp', 0).item()
                    val_metrics['loss_conf'] += loss_dict.get('loss_conf', 0).item()
                    val_metrics['loss_normal'] += loss_dict.get('loss_normal', 0).item()
                    val_metrics['raw_3d_l1'] += loss_dict.get('metric_raw_3d_l1', 0).item()
                    val_metrics['raw_3d_euc'] += loss_dict.get('metric_raw_3d_euclidean', 0).item()
                    val_metrics['raw_3d_l1_static'] += loss_dict.get('metric_raw_3d_l1_static', 0).item()
                    val_metrics['raw_3d_l1_temporal'] += loss_dict.get('metric_raw_3d_l1_temporal', 0).item()
                    val_metrics['raw_3d_euc_static'] += loss_dict.get('metric_raw_3d_euclidean_static', 0).item()
                    val_metrics['raw_3d_euc_temporal'] += loss_dict.get('metric_raw_3d_euclidean_temporal', 0).item()
                    val_metrics['loss_3d_static_nocon'] += loss_dict.get('metric_loss_3d_static_unweighted', 0).item()
                    val_metrics['loss_3d_temporal_nocon'] += loss_dict.get('metric_loss_3d_temporal_unweighted', 0).item()
                    val_metrics['valid_3d_ratio'] += loss_dict.get('metric_valid_3d_query_ratio', 0).item()
                    val_metrics['static_query_ratio'] += loss_dict.get('metric_static_query_ratio', 0).item()
                    val_metrics['temporal_query_ratio'] += loss_dict.get('metric_temporal_query_ratio', 0).item()
                    val_metrics['static_valid3d_ratio'] += loss_dict.get('metric_static_valid3d_ratio', 0).item()
                    val_metrics['temporal_valid3d_ratio'] += loss_dict.get('metric_temporal_valid3d_ratio', 0).item()
                    val_metrics['normal_query_ratio'] += loss_dict.get('metric_normal_query_ratio', 0).item()
                    val_metrics['normal_valid3d_ratio'] += loss_dict.get('metric_normal_valid3d_ratio', 0).item()
                    val_metrics['conf_mean'] += loss_dict.get('metric_conf_mean', 0).item()
            for k in val_metrics:
                val_metrics[k] /= len(val_loader)
                if distributed:
                    val_metrics[k] = torch.tensor(val_metrics[k], device=device)
                    dist.all_reduce(val_metrics[k], op=dist.ReduceOp.AVG)
                    val_metrics[k] = val_metrics[k].item()
            if distributed:
                dist.barrier()
            if local_rank == 0:
                print(f"Validation Loss: {val_metrics['loss']:.4f}")
                val_log = {'epoch': epoch + 1}
                val_log.update({k: f"{v:.4f}" for k, v in val_metrics.items()})
                with open(output_dir / 'val_log.jsonl', 'a') as f:
                    f.write(json.dumps(val_log) + '\n')

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 and local_rank == 0:
            is_milestone = (epoch + 1) % 1000 == 0
            if is_milestone:
                checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            else:
                checkpoint_path = output_dir / f"checkpoint_latest_{epoch+1}.pth"
                checkpoint_list.append(checkpoint_path)
                # Remove old checkpoints if exceeds limit
                if len(checkpoint_list) > args.keep_checkpoints:
                    old_ckpt = checkpoint_list.pop(0)
                    if old_ckpt.exists():
                        old_ckpt.unlink()

            torch.save({
                'model': unwrap_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
