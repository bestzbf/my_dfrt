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
import os
from typing import Union

from datasets.registry import create_adapter
from datasets.mixture import MixtureDataset
from datasets.planned_dataset import PlannedMixtureDataset


def create_training_dataset(
    config: dict,
    split: str = 'train',
    rank: int = 0,
    world_size: int = 1,
) -> Union[MixtureDataset, PlannedMixtureDataset]:
    """
    Create dataset from config.

    Args:
        config: Dataset configuration dict
        split: 'train', 'val', or 'test'
        rank: DDP rank (for planned mode)
        world_size: DDP world size (for planned mode)

    Returns:
        MixtureDataset or PlannedMixtureDataset

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
        return _create_single_dataset(config, split, rank, world_size)
    elif mode == 'scene':
        return _create_scene_dataset(config, split, rank, world_size)
    elif mode == 'mixture':
        return _create_mixture_dataset(config, split, rank, world_size)
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'single', 'scene', or 'mixture'")


def _create_single_dataset(config: dict, split: str, rank: int = 0, world_size: int = 1) -> Union[MixtureDataset, PlannedMixtureDataset]:
    """Train on all sequences of one dataset."""
    index_cache_dir = config.get('index_cache_dir', None)
    index_workers = config.get('index_workers', None)
    extra = {}
    if index_cache_dir:
        extra['cache_dir'] = index_cache_dir
    if index_workers is not None:
        extra['index_workers'] = index_workers
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


def _create_scene_dataset(config: dict, split: str, rank: int = 0, world_size: int = 1) -> Union[MixtureDataset, PlannedMixtureDataset]:
    """Train on specific sequences (scenes) from one dataset."""
    sequences = config.get('sequences')
    if not sequences:
        raise ValueError(
            "mode='scene' requires a non-empty 'sequences' list. "
            "Example:\n  sequences:\n    - ani"
        )

    index_cache_dir = config.get('index_cache_dir', None)
    index_workers = config.get('index_workers', None)
    extra = {}
    if index_cache_dir:
        extra['cache_dir'] = index_cache_dir
    if index_workers is not None:
        extra['index_workers'] = index_workers
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


def _create_mixture_dataset(config: dict, split: str, rank: int = 0, world_size: int = 1) -> Union[MixtureDataset, PlannedMixtureDataset]:
    """Train on multiple datasets mixed by weight."""
    index_cache_dir = config.get('index_cache_dir', None)
    index_workers = config.get('index_workers', None)
    custom_stride_range = None
    if 'stride_range' in config:
        stride_range = config['stride_range']
        if isinstance(stride_range, list) and len(stride_range) == 2:
            custom_stride_range = tuple(stride_range)

    ds_configs = config['datasets']

    def _build_one(ds_config):
        actual_split = ds_config.get('val_split', split) if split == 'val' else ds_config.get('split', split)
        extra = {}
        if index_cache_dir:
            extra['cache_dir'] = index_cache_dir
        if index_workers is not None:
            extra['index_workers'] = index_workers
        adapter = create_adapter(
            name=ds_config['name'],
            root=ds_config['root'],
            split=actual_split,
            **ds_config.get('adapter_kwargs', {}),
            **extra,
        )
        if split == 'val' and 'val_sequences' in ds_config:
            allowed = ds_config['val_sequences']
        elif split == 'train' and 'train_sequences' in ds_config:
            allowed = ds_config['train_sequences']
        else:
            allowed = ds_config.get('sequences', None)
        return adapter, ds_config.get('weight', 1.0), allowed

    serialize_build = os.getenv("D4RT_SERIALIZE_ADAPTER_INIT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    results = [None] * len(ds_configs)
    if serialize_build or len(ds_configs) <= 1:
        for idx, ds_config in enumerate(ds_configs):
            results[idx] = _build_one(ds_config)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

        with ThreadPoolExecutor(max_workers=len(ds_configs)) as executor:
            future_to_idx = {executor.submit(_build_one, dc): i for i, dc in enumerate(ds_configs)}
            for fut in _as_completed(future_to_idx):
                results[future_to_idx[fut]] = fut.result()

    adapters = [r[0] for r in results]
    weights = [r[1] for r in results]
    allowed_per_adapter = [r[2] for r in results]

    # Check if planned mode is enabled
    planned_mode = config.get('planned_mode', False)

    if planned_mode:
        # Create PlannedMixtureDataset
        from datasets.mixture import MixtureSampler
        from datasets.sampling import DatasetSampler
        from datasets.query_builder import D4RTQueryBuilder
        from datasets.transforms import GeometryTransformPipeline

        # Build dataset samplers
        dataset_samplers = []
        for adapter, allowed_seqs in zip(adapters, allowed_per_adapter):
            sampler = DatasetSampler(
                adapter=adapter,
                clip_len=config.get('clip_len', 48),
                sampling_mode=config.get('sampling_mode', 'stride'),
                min_frames=config.get('clip_len', 48),
                allowed_sequences=allowed_seqs,
                custom_stride_range=custom_stride_range,
                sequence_locality_size=config.get('sequence_locality_size', 3),
                frame_locality_radius=config.get('frame_locality_radius', config.get('clip_len', 48)),
            )
            dataset_samplers.append(sampler)

        # Build mixture sampler
        mixture_sampler = MixtureSampler(
            samplers=dataset_samplers,
            dataset_weights=weights,
            dataset_locality_size=config.get('dataset_locality_size', 2),
        )

        # Build transform pipeline
        transform = GeometryTransformPipeline(
            img_size=config.get('img_size', 256),
            use_augs=config.get('use_augs', True) if split == 'train' else False,
        )

        # Build query builder
        query_builder = D4RTQueryBuilder(
            num_queries=config.get('num_queries', 2048),
            boundary_ratio=config.get('boundary_ratio', 0.3),
            t_tgt_eq_t_cam_ratio=config.get('t_tgt_eq_t_cam_ratio', 0.4),
            precompute_patches=config.get('precompute_patches', True),
            precompute_from_highres=config.get('precompute_from_highres', False),
            allow_track_fallback=config.get('allow_track_fallback', False),
        )

        return PlannedMixtureDataset(
            adapters=adapters,
            dataset_weights=weights,
            transform=transform,
            query_builder=query_builder,
            mixture_sampler=mixture_sampler,
            clip_len=config.get('clip_len', 48),
            seed=config.get('seed', 42),
            epoch_size=config.get('epoch_size', 10000),
            reshuffle_each_epoch=config.get('reshuffle_each_epoch', split == 'train'),
            builder_workers=config.get('builder_workers', 2),
            prefetch_depth=config.get('prefetch_depth', 32),
            spool_dir=config.get('spool_dir', None),
            rank=rank,
            world_size=world_size,
        )
    else:
        # Create standard MixtureDataset
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
