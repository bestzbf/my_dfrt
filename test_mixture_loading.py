#!/usr/bin/env python3
"""Test mixture dataset loading."""

import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from datasets.factory import create_training_dataset
from datasets.collate import d4rt_collate_fn
from torch.utils.data import DataLoader

def test_mixture_config(config_path):
    print(f"Testing config: {config_path}")
    print("=" * 60)

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Mode: {config['mode']}")
    print(f"Datasets: {len(config['datasets'])}")
    for i, ds in enumerate(config['datasets']):
        print(f"  [{i}] {ds['name']}: weight={ds['weight']}, root={ds['root']}")
    print()

    # Create dataset
    print("Creating dataset...")
    try:
        dataset = create_training_dataset(config, split='train')
        print(f"✓ Dataset created successfully")
        print(f"  Dataset names: {dataset.get_dataset_names()}")
        print(f"  Dataset length: {len(dataset)}")
        print()
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        return False

    # Test single sample
    print("Loading sample 0...")
    try:
        sample = dataset[0]
        print(f"✓ Sample loaded successfully")
        print(f"  video shape: {sample.video.shape}")
        print(f"  coords shape: {sample.coords.shape}")
        print(f"  t_src shape: {sample.t_src.shape}")
        print(f"  dataset_name: {sample.dataset_name}")
        print(f"  sequence_name: {sample.sequence_name}")
        print()
    except Exception as e:
        print(f"✗ Failed to load sample: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test dataloader
    print("Testing DataLoader...")
    try:
        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=0,
            collate_fn=d4rt_collate_fn,
        )
        batch = next(iter(loader))
        print(f"✓ DataLoader works")
        print(f"  batch video shape: {batch['video'].shape}")
        print(f"  batch coords shape: {batch['coords'].shape}")
        print(f"  batch dataset_names: {batch['dataset_names']}")
        print()
    except Exception as e:
        print(f"✗ DataLoader failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("=" * 60)
    print("✓ All tests passed!")
    return True

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/mixture_full_11datasets.yaml"
    test_mixture_config(config_path)
