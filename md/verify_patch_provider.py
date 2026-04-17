#!/usr/bin/env python3
"""
快速验证 patch provider 配置是否正确。

Usage:
    python verify_patch_provider.py --config configs/pointodyssey.yaml --patch-provider sampled_highres
"""

import argparse
import yaml
from pathlib import Path
import torch
from datasets.factory import create_training_dataset
from datasets.collate import d4rt_collate_fn
from torch.utils.data import DataLoader
from models import create_d4rt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--patch-provider", type=str, default="auto")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=48)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Config: {args.config}")
    print(f"  precompute_patches: {config.get('precompute_patches', True)}")
    print(f"  patch_provider (model): {args.patch_provider}\n")

    # Create dataset
    print("Loading dataset...")
    dataset = create_training_dataset(config, split='train')
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=d4rt_collate_fn,
        shuffle=False,
    )

    # Create model
    print("Creating model...")
    model = create_d4rt(
        variant="base",
        decoder_depth=6,
        img_size=args.resolution,
        temporal_size=args.num_frames,
        patch_size=(2, 16, 16),
        query_patch_size=9,
        patch_provider=args.patch_provider,
    ).to(device)

    # Get first batch
    print("Loading first batch...\n")
    batch = next(iter(loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    if batch.get('transform_metadata'):
        batch['transform_metadata'] = {k: v.to(device) for k, v in batch['transform_metadata'].items()}

    # Check patch provider info
    info = model.decoder.get_patch_provider_info(
        batch.get('local_patches'),
        batch.get('transform_metadata')
    )

    print("=" * 60)
    print("PATCH PROVIDER VERIFICATION")
    print("=" * 60)
    print(f"Configured:             {info['configured']}")
    print(f"Resolved:               {info['resolved']}")
    print(f"Has local_patches:      {info['has_local_patches']}")
    print(f"Has transform_metadata: {info['has_transform_metadata']}")
    print(f"Has highres_video:      {batch.get('highres_video') is not None}")

    if batch.get('local_patches') is not None:
        print(f"local_patches shape:    {batch['local_patches'].shape}")

    if batch.get('highres_video') is not None:
        if isinstance(batch['highres_video'], list):
            print(f"highres_video:          list of {len(batch['highres_video'])} tensors")
            for i, hv in enumerate(batch['highres_video']):
                if hv is not None:
                    print(f"  [{i}] shape: {hv.shape}")
        else:
            print(f"highres_video shape:    {batch['highres_video'].shape}")

    print(f"video shape:            {batch['video'].shape}")
    print("=" * 60)

    # Validate configuration
    print("\nVALIDATION:")

    expected_resolved = args.patch_provider if args.patch_provider != "auto" else (
        "precomputed_resized" if info['has_local_patches'] else "sampled_resized"
    )

    if info['resolved'] == expected_resolved:
        print(f"✅ Patch provider resolved correctly: {info['resolved']}")
    else:
        print(f"❌ Unexpected patch provider: {info['resolved']} (expected {expected_resolved})")

    # Check for common misconfigurations
    if args.patch_provider == "sampled_highres":
        if not info['has_transform_metadata']:
            print("❌ ERROR: sampled_highres requires transform_metadata")
        elif not batch.get('highres_video'):
            print("⚠️  WARNING: sampled_highres configured but no highres_video in batch")
            print("    Make sure training script passes query_frames=batch.get('highres_video')")
        elif info['has_local_patches']:
            print("⚠️  WARNING: local_patches present but will be ignored")
            print("    Consider setting precompute_patches=false in config to save time")
        else:
            print("✅ sampled_highres configuration looks correct")

    elif info['resolved'] == "precomputed_resized":
        if batch.get('highres_video'):
            print("⚠️  WARNING: highres_video present but will be ignored")
            print("    Use --patch-provider sampled_highres to enable high-res patches")
        print("ℹ️  Using precomputed resized patches (256×256)")
        print("    This may limit fine-grained detail capability")

    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
