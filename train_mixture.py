#!/usr/bin/env python3
"""Mixed dataset training for D4RT."""

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
    parser.add_argument("--loss-w-2d", type=float, default=0.1)
    parser.add_argument("--loss-w-vis", type=float, default=0.1)
    parser.add_argument("--loss-w-disp", type=float, default=0.1)
    parser.add_argument("--loss-w-conf", type=float, default=0.2)
    parser.add_argument("--loss-w-normal", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="outputs/mixture")
    parser.add_argument("--pretrain", type=str, default=None, help="Path to pretrained checkpoint")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Create dataset
    train_dataset = create_training_dataset(config, split='train')
    print(f"Dataset: {train_dataset.get_dataset_names()}")
    print(f"Length: {len(train_dataset)}")

    # DataLoader (shuffle=True for mixture training per paper)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=d4rt_collate_fn,
        shuffle=True,
        pin_memory=True,
    )

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_d4rt(encoder="base", decoder_depth=12, img_size=args.resolution,
                        num_frames=args.num_frames, patch_size=(2, 16, 16),
                        query_patch_size=9, videomae_model="/data1/zbf/pretrained/videomae-base").to(device)

    # Load pretrained weights
    if args.pretrain:
        print(f"Loading pretrained weights from {args.pretrain}")
        checkpoint = torch.load(args.pretrain, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=True)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = D4RTLoss(lambda_3d=args.loss_w_3d, lambda_2d=args.loss_w_2d,
                       lambda_vis=args.loss_w_vis, lambda_disp=args.loss_w_disp,
                       lambda_conf=args.loss_w_conf, lambda_normal=args.loss_w_normal)

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

    global_step = 0
    print(f"Starting training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(batch)
            loss_dict = loss_fn(outputs, batch)
            loss = loss_dict['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}, Step {global_step}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                       'epoch': epoch}, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()
