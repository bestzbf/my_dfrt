"""
Dataset factory for D4RT training.

Supports three training modes:
  - 'single'  : train on one entire dataset
  - 'scene'   : train on specific sequences from one dataset
  - 'mixture' : train on multiple datasets mixed by weight

Usage:
    # Single dataset
    dataset = create_training_dataset(config)

    # Single scene
    dataset = create_training_dataset(scene_config)

    # Mixture dataset
    dataset = create_training_dataset(mixture_config)
"""

from __future__ import annotations
from typing import Union

from datasets.registry import create_adapter
from datasets.mixture import MixtureDataset


def create_training_dataset(
    config: dict,
    split: str = 'train',
) -> MixtureDataset:
    """
    Create dataset from config.

    Args:
        config: Dataset configuration dict
        split: 'train', 'val', or 'test'

    Returns:
        MixtureDataset

    ---
    Config format for single dataset (all sequences):
        mode: single
        name: pointodyssey
        root: /path/to/data
        clip_len: 48
        img_size: 256
        num_queries: 2048

    ---
    Config format for single scene (one or more specific sequences):
        mode: scene
        name: pointodyssey
        root: /path/to/data
        sequences:
          - ani
        # or multiple:
        # sequences:
        #   - ani
        #   - ani11_new_
        clip_len: 48
        img_size: 256
        num_queries: 2048

    ---
    Config format for mixture:
        mode: mixture
        datasets:
          - name: pointodyssey
            root: /path
            weight: 0.5
          - name: scannet
            root: /path
            weight: 0.3
        clip_len: 48
        img_size: 256
        num_queries: 2048
    """
    mode = config.get('mode', 'single')

    if mode == 'single':
        return _create_single_dataset(config, split)
    elif mode == 'scene':
        return _create_scene_dataset(config, split)
    elif mode == 'mixture':
        return _create_mixture_dataset(config, split)
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'single', 'scene', or 'mixture'")


def _create_single_dataset(config: dict, split: str) -> MixtureDataset:
    """Train on all sequences of one dataset."""
    index_cache_dir = config.get('index_cache_dir', None)
    extra = {'cache_dir': index_cache_dir} if index_cache_dir else {}
    adapter = create_adapter(
        name=config['name'],
        root=config['root'],
        split=split,
        **config.get('adapter_kwargs', {}),
        **extra,
    )

    return MixtureDataset(
        adapters=[adapter],
        dataset_weights=[1.0],
        clip_len=config.get('clip_len', 48),
        img_size=config.get('img_size', 256),
        use_augs=config.get('use_augs', True),
        num_queries=config.get('num_queries', 2048),
        boundary_ratio=config.get('boundary_ratio', 0.3),
        t_tgt_eq_t_cam_ratio=config.get('t_tgt_eq_t_cam_ratio', 0.4),
        seed=config.get('seed', 42),
        sampling_mode=config.get('sampling_mode', 'stride'),
        epoch_size=config.get('epoch_size', 10000),
        precompute_patches=config.get('precompute_patches', True),
        precompute_from_highres=config.get('precompute_from_highres', False),
        allow_track_fallback=config.get('allow_track_fallback', False),
        reshuffle_each_epoch=config.get('reshuffle_each_epoch', split == 'train'),
    )


def _create_scene_dataset(config: dict, split: str) -> MixtureDataset:
    """Train on specific sequences (scenes) from one dataset."""
    sequences = config.get('sequences')
    if not sequences:
        raise ValueError(
            "mode='scene' requires a non-empty 'sequences' list. "
            "Example:\n  sequences:\n    - ani"
        )

    index_cache_dir = config.get('index_cache_dir', None)
    extra = {'cache_dir': index_cache_dir} if index_cache_dir else {}
    adapter = create_adapter(
        name=config['name'],
        root=config['root'],
        split=split,
        **config.get('adapter_kwargs', {}),
        **extra,
    )

    custom_stride_range = None
    if 'stride_range' in config:
        stride_range = config['stride_range']
        if isinstance(stride_range, list) and len(stride_range) == 2:
            custom_stride_range = tuple(stride_range)

    return MixtureDataset(
        adapters=[adapter],
        dataset_weights=[1.0],
        clip_len=config.get('clip_len', 48),
        img_size=config.get('img_size', 256),
        use_augs=config.get('use_augs', True),
        num_queries=config.get('num_queries', 2048),
        boundary_ratio=config.get('boundary_ratio', 0.3),
        t_tgt_eq_t_cam_ratio=config.get('t_tgt_eq_t_cam_ratio', 0.4),
        seed=config.get('seed', 42),
        allowed_sequences_per_adapter=[list(sequences)],
        sampling_mode=config.get('sampling_mode', 'stride'),
        epoch_size=config.get('epoch_size', 10000),
        custom_stride_range=custom_stride_range,
        precompute_patches=config.get('precompute_patches', True),
        precompute_from_highres=config.get('precompute_from_highres', False),
        allow_track_fallback=config.get('allow_track_fallback', False),
        reshuffle_each_epoch=config.get('reshuffle_each_epoch', split == 'train'),
    )


def _create_mixture_dataset(config: dict, split: str) -> MixtureDataset:
    """Train on multiple datasets mixed by weight."""
    adapters = []
    weights = []
    allowed_per_adapter = []
    index_cache_dir = config.get('index_cache_dir', None)
    custom_stride_range = None
    if 'stride_range' in config:
        stride_range = config['stride_range']
        if isinstance(stride_range, list) and len(stride_range) == 2:
            custom_stride_range = tuple(stride_range)

    for ds_config in config['datasets']:
        actual_split = ds_config.get('val_split', split) if split == 'val' else ds_config.get('split', split)
        extra = {'cache_dir': index_cache_dir} if index_cache_dir else {}
        adapter = create_adapter(
            name=ds_config['name'],
            root=ds_config['root'],
            split=actual_split,
            **ds_config.get('adapter_kwargs', {}),
            **extra,
        )
        adapters.append(adapter)
        weights.append(ds_config.get('weight', 1.0))
        # Each dataset entry may optionally restrict to specific sequences
        if split == 'val' and 'val_sequences' in ds_config:
            allowed_per_adapter.append(ds_config['val_sequences'])
        elif split == 'train' and 'train_sequences' in ds_config:
            allowed_per_adapter.append(ds_config['train_sequences'])
        else:
            allowed_per_adapter.append(ds_config.get('sequences', None))

    return MixtureDataset(
        adapters=adapters,
        dataset_weights=weights,
        clip_len=config.get('clip_len', 48),
        img_size=config.get('img_size', 256),
        use_augs=config.get('use_augs', True) if split == 'train' else False,
        num_queries=config.get('num_queries', 2048),
        boundary_ratio=config.get('boundary_ratio', 0.3),
        t_tgt_eq_t_cam_ratio=config.get('t_tgt_eq_t_cam_ratio', 0.4),
        seed=config.get('seed', 42),
        allowed_sequences_per_adapter=allowed_per_adapter,
        sampling_mode=config.get('sampling_mode', 'stride'),
        epoch_size=config.get('epoch_size', 10000),
        custom_stride_range=custom_stride_range,
        precompute_patches=config.get('precompute_patches', True),
        precompute_from_highres=config.get('precompute_from_highres', False),
        allow_track_fallback=config.get('allow_track_fallback', False),
        reshuffle_each_epoch=config.get('reshuffle_each_epoch', split == 'train'),
    )
