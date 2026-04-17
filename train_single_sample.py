#!/usr/bin/env python3
"""Single sample training for D4RT - overfit on one sample."""

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.factory import create_training_dataset
from datasets.collate import d4rt_collate_fn
from models import create_d4rt
from losses import D4RTLoss
import json
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Dataset config YAML")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--lr-warmup-steps", type=int, default=100)
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
    parser.add_argument("--output-dir", type=str, default="outputs/single_sample")
    parser.add_argument("--pretrain", type=str, default=None, help="Path to pretrained checkpoint")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--save-interval", type=int, default=100, help="Save checkpoint every N epochs")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N epochs")
    parser.add_argument("--patch-provider", type=str, default="auto", help="Patch provider: auto, precomputed_resized, sampled_resized, sampled_highres")
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    amp_enabled = (device.type == "cuda")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Create dataset - 同一个样本用于训练和验证
    print("Loading dataset...")
    train_dataset = create_training_dataset(config, split='train')

    # 验证集只需要1个clip
    val_config = config.copy()
    val_config['epoch_size'] = 1
    val_dataset = create_training_dataset(val_config, split='train')

    print(f"Dataset: {train_dataset.get_dataset_names()}")
    print(f"Train Length: {len(train_dataset)}, Val Length: {len(val_dataset)} (只需1个clip)")
    print(f"Data augmentation: {config.get('use_augs', True)}")

    # DataLoader - 优化配置
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=d4rt_collate_fn,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if args.num_workers > 0 else False,  # 保持workers存活
        prefetch_factor=4 if args.num_workers > 0 else None,  # 预取4个batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=d4rt_collate_fn,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # Setup model
    print("Creating model...")
    model = create_d4rt(
        variant="base",
        decoder_depth=6,
        img_size=args.resolution,
        temporal_size=args.num_frames,
        patch_size=(2, 16, 16),
        query_patch_size=9,
        videomae_model="/data1/zbf/pretrained/videomae-base",
        patch_provider=args.patch_provider,
    ).to(device)

    # Load pretrained weights
    if args.pretrain:
        print(f"Loading pretrained weights from {args.pretrain}")
        checkpoint = torch.load(args.pretrain, map_location=device)
        state_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'model'
        model.load_state_dict(checkpoint[state_key], strict=True)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = D4RTLoss(
        lambda_3d=args.loss_w_3d,
        lambda_raw_3d=args.loss_w_raw_3d,
        lambda_2d=args.loss_w_2d,
        lambda_vis=args.loss_w_vis,
        lambda_disp=args.loss_w_disp,
        lambda_conf=args.loss_w_conf,
        lambda_normal=args.loss_w_normal,
        static_reprojection_weight=args.loss_w_static_reprojection,
    )

    # LR scheduler: warmup + cosine annealing
    total_steps = args.epochs * len(train_loader)
    def lr_lambda(step):
        if step < args.lr_warmup_steps:
            return step / args.lr_warmup_steps
        progress = (step - args.lr_warmup_steps) / (total_steps - args.lr_warmup_steps)
        return args.lr_min / args.lr + (1 - args.lr_min / args.lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0

    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=True)
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch'] + 1
                print("Loaded optimizer state")
            except (KeyError, ValueError) as e:
                print(f"Could not load optimizer state: {e}. Starting with fresh optimizer.")
                start_epoch = 0
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            try:
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Loaded optimizer state")
            except (KeyError, ValueError) as e:
                print(f"Could not load optimizer state: {e}. Starting with fresh optimizer.")
            start_epoch = 0  # Start from epoch 0 with fresh optimizer
        else:
            raise KeyError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")

        global_step = start_epoch * len(train_loader)
        # advance scheduler to match current step
        for _ in range(global_step):
            scheduler.step()
        print(f"Resumed at epoch {start_epoch}, global_step {global_step}")

    print(f"Starting training for {args.epochs} epochs")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Patch provider: {args.patch_provider}")
    print(f"Output directory: {output_dir}")

    # Log patch provider configuration on first batch
    first_batch_logged = False

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        epoch_metrics = {}

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch['targets'] = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch['targets'].items()}
            if batch.get('transform_metadata'):
                batch['transform_metadata'] = {k: v.to(device) for k, v in batch['transform_metadata'].items()}
            if batch['video'].dtype == torch.uint8:
                batch['video'] = batch['video'].float() / 255.0

            # Log patch provider info on first batch
            if not first_batch_logged:
                info = model.decoder.get_patch_provider_info(
                    batch.get('local_patches'),
                    batch.get('transform_metadata')
                )
                print(f"\n[Patch Provider Info]")
                print(f"  Configured: {info['configured']}")
                print(f"  Resolved: {info['resolved']}")
                print(f"  Has local_patches: {info['has_local_patches']}")
                print(f"  Has transform_metadata: {info['has_transform_metadata']}")
                print(f"  Has highres_video: {batch.get('highres_video') is not None}")
                if batch.get('highres_video') is not None:
                    if isinstance(batch['highres_video'], list):
                        print(f"  highres_video: list of {len(batch['highres_video'])} tensors")
                    else:
                        print(f"  highres_video shape: {batch['highres_video'].shape}")
                if info['resolved'] == 'precomputed_resized' and info['has_local_patches']:
                    print(f"  WARNING: using precomputed patches from resized video (low-res)")
                elif info['resolved'] == 'precomputed_highres' and info['has_local_patches']:
                    print(f"  OK: using precomputed patches from highres video")
                print()
                first_batch_logged = True

            optimizer.zero_grad()

            # For sampled_highres: pass original-resolution frames as query_frames.
            # highres_video is a list (variable crop sizes), so we pass the list directly.
            # The model uses query_frames only when patch_provider='sampled_highres'.
            query_frames_arg = batch.get('highres_video') if args.patch_provider == 'sampled_highres' else None

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                outputs = model(batch['video'], batch['coords'], batch['t_src'], batch['t_tgt'], batch['t_cam'],
                               aspect_ratio=batch.get('aspect_ratio'),
                               query_frames=query_frames_arg,
                               local_patches=batch.get('local_patches'),
                               transform_metadata=batch.get('transform_metadata'))
                normalize_groups = batch['dataset_id'] * args.num_frames + batch['t_cam']
                loss_dict = loss_fn(outputs, batch['targets'], normalize_groups=normalize_groups)
                loss = loss_dict['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += loss.item()

            # Accumulate metrics
            for k, v in loss_dict.items():
                if k != 'loss':
                    if k not in epoch_metrics:
                        epoch_metrics[k] = 0
                    epoch_metrics[k] += v.item() if isinstance(v, torch.Tensor) else v

        avg_loss = epoch_loss / len(train_loader)
        avg_metrics = {k: v / len(train_loader) for k, v in epoch_metrics.items()}

        # Logging
        if (epoch + 1) % args.log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1}/{args.epochs}, "
                  f"Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

            # Save loss log
            log_entry = {
                'epoch': epoch + 1,
                'step': global_step,
                'loss': f"{avg_loss:.4f}",
                'loss_3d': f"{avg_metrics.get('loss_3d', 0):.4f}",
                'loss_3d_nocon': f"{avg_metrics.get('loss_3d_unweighted', 0):.4f}",
                'loss_raw_3d': f"{avg_metrics.get('loss_raw_3d', 0):.4f}",
                'loss_2d': f"{avg_metrics.get('loss_2d', 0):.4f}",
                'loss_vis': f"{avg_metrics.get('loss_vis', 0):.4f}",
                'loss_disp': f"{avg_metrics.get('loss_disp', 0):.4f}",
                'loss_conf': f"{avg_metrics.get('loss_conf', 0):.4f}",
                'loss_normal': f"{avg_metrics.get('loss_normal', 0):.4f}",
                'raw_3d_l1': f"{avg_metrics.get('metric_raw_3d_l1', 0):.4f}",
                'raw_3d_euc': f"{avg_metrics.get('metric_raw_3d_euclidean', 0):.4f}",
                'raw_3d_l1_static': f"{avg_metrics.get('metric_raw_3d_l1_static', 0):.4f}",
                'raw_3d_l1_temporal': f"{avg_metrics.get('metric_raw_3d_l1_temporal', 0):.4f}",
                'raw_3d_euc_static': f"{avg_metrics.get('metric_raw_3d_euclidean_static', 0):.4f}",
                'raw_3d_euc_temporal': f"{avg_metrics.get('metric_raw_3d_euclidean_temporal', 0):.4f}",
                'loss_3d_static_nocon': f"{avg_metrics.get('metric_loss_3d_static_unweighted', 0):.4f}",
                'loss_3d_temporal_nocon': f"{avg_metrics.get('metric_loss_3d_temporal_unweighted', 0):.4f}",
                'valid_3d_ratio': f"{avg_metrics.get('metric_valid_3d_query_ratio', 0):.4f}",
                'static_query_ratio': f"{avg_metrics.get('metric_static_query_ratio', 0):.4f}",
                'temporal_query_ratio': f"{avg_metrics.get('metric_temporal_query_ratio', 0):.4f}",
                'static_valid3d_ratio': f"{avg_metrics.get('metric_static_valid3d_ratio', 0):.4f}",
                'temporal_valid3d_ratio': f"{avg_metrics.get('metric_temporal_valid3d_ratio', 0):.4f}",
                'normal_query_ratio': f"{avg_metrics.get('metric_normal_query_ratio', 0):.4f}",
                'normal_valid3d_ratio': f"{avg_metrics.get('metric_normal_valid3d_ratio', 0):.4f}",
                'conf_mean': f"{avg_metrics.get('metric_conf_mean', 0):.4f}",
                'lr': f"{current_lr:.6f}"
            }
            with open(output_dir / 'loss_log.jsonl', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

        # Validation
        if (epoch + 1) % args.log_interval == 0:
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
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    batch['targets'] = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch['targets'].items()}
                    if batch.get('transform_metadata'):
                        batch['transform_metadata'] = {k: v.to(device) for k, v in batch['transform_metadata'].items()}
                    if batch['video'].dtype == torch.uint8:
                        batch['video'] = batch['video'].float() / 255.0
                    # For sampled_highres: pass original-resolution frames as query_frames.
                    # highres_video is a list (variable crop sizes), so we pass the list directly.
                    # The model uses query_frames only when patch_provider='sampled_highres'.
                    query_frames_arg = batch.get('highres_video') if args.patch_provider == 'sampled_highres' else None

                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                        outputs = model(batch['video'], batch['coords'], batch['t_src'], batch['t_tgt'], batch['t_cam'],
                                       aspect_ratio=batch.get('aspect_ratio'),
                                       query_frames=query_frames_arg,
                                       local_patches=batch.get('local_patches'),
                                       transform_metadata=batch.get('transform_metadata'))
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

            print(f"  Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val 3D: {val_metrics['loss_3d']:.4f}, "
                  f"Val 2D: {val_metrics['loss_2d']:.4f}")

            val_log = {'epoch': epoch + 1}
            val_log.update({k: f"{v:.4f}" for k, v in val_metrics.items()})
            with open(output_dir / 'val_log.jsonl', 'a') as f:
                f.write(json.dumps(val_log) + '\n')

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()
