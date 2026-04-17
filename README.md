# D4RT Local Notes

This directory contains the current D4RT training/inference code and a few project-local training entrypoints.

## Training: important current defaults

### 1) Default training is **not** the paper's strongest high-res setting

`train_mixture.py` defaults to:

- `--patch-provider auto`

That does **not** mean high-res patches are active.

If you want the paper-style strongest local high-res patch path, use:

```bash
cd /workspace/openclaw/d4rt
python train_mixture.py \
  --config configs/mixture_train_highres.yaml \
  --patch-provider sampled_highres \
  --output-dir outputs/mixture_highres
```

Multi-GPU example:

```bash
cd /workspace/openclaw/d4rt
torchrun --nproc_per_node=8 train_mixture.py \
  --config configs/mixture_train_highres.yaml \
  --patch-provider sampled_highres \
  --output-dir outputs/mixture_highres
```

Related config:

- `configs/mixture_train_highres.yaml`

### Co3Dv2 quality filtering

`Co3Dv2Adapter` now supports sequence-level filtering in `adapter_kwargs`, for example:

```yaml
- name: co3dv2
  root: /path/to/Co3Dv2
  weight: 0.3
  adapter_kwargs:
    min_viewpoint_quality: 0.5
    min_pointcloud_quality: 0.4
    min_valid_depth_ratio: 0.1
    min_foreground_ratio: 0.05
    require_pointcloud: true
    require_precomputed: true
```

The standard training configs already enable a conservative Co3Dv2 default:

- `min_viewpoint_quality: 0.5`
- `min_valid_depth_ratio: 0.1`
- `min_foreground_ratio: 0.05`
- `quality_probe_frames: 3`
- `require_pointcloud: true`

See:

- `configs/mixture_train.yaml`
- `configs/mixture_train_highres.yaml`

You can also export an allowlist offline:

```bash
python datasets/computer/filter_co3dv2_sequences.py \
  --root /path/to/Co3Dv2 \
  --min-viewpoint-quality 0.5 \
  --min-pointcloud-quality 0.4 \
  --min-valid-depth-ratio 0.1 \
  --require-precomputed \
  --output-json tmp/co3d_filter.json \
  --output-txt tmp/co3d_allowlist.txt
```

---

### 2) Static reprojection queries now have an explicit 3D-loss weight

`has_tracks=False` samples are static reprojection queries. Their 3D contribution can now be controlled explicitly.

New training arg:

- `--loss-w-static-reprojection <float>`

Default:

- `1.0` → preserves previous behavior

Examples:

```bash
# Keep current behavior
python train_mixture.py --config configs/mixture_train_highres.yaml \
  --patch-provider sampled_highres \
  --loss-w-static-reprojection 1.0

# Down-weight static-reprojection 3D supervision
python train_mixture.py --config configs/mixture_train_highres.yaml \
  --patch-provider sampled_highres \
  --loss-w-static-reprojection 0.5

# Ignore static-reprojection contribution in 3D loss
python train_mixture.py --config configs/mixture_train_highres.yaml \
  --patch-provider sampled_highres \
  --loss-w-static-reprojection 0.0
```

The same flag also exists in:

- `train_single_sample.py`

What it changes:

- only the **3D loss aggregation** for queries marked `is_static_reprojection`
- it does **not** change 2D / visibility / displacement logic
- it does **not** change the unweighted diagnostic metric `loss_3d_unweighted`

---

### 3) New observability metrics in training logs

Recent training logs now expose static-vs-temporal split metrics so you can inspect how much of the batch is coming from static reprojection and how their errors differ.

New log fields include:

- `static_query_ratio`
- `temporal_query_ratio`
- `valid_3d_ratio`
- `static_valid3d_ratio`
- `temporal_valid3d_ratio`
- `loss_3d_static_nocon`
- `loss_3d_temporal_nocon`
- `raw_3d_l1_static`
- `raw_3d_l1_temporal`
- `raw_3d_euc_static`
- `raw_3d_euc_temporal`
- `normal_query_ratio`
- `normal_valid3d_ratio`

These appear in:

- `outputs/.../loss_log.jsonl`
- `outputs/.../val_log.jsonl`

---

## Regression / contract tests added

### Collate target preservation

Ensures `d4rt_collate_fn` does not silently drop extra target fields such as:

- `is_static_reprojection`
- `source_is_boundary`
- `source_is_depth_boundary`
- `source_is_motion_boundary`
- `point_indices`

Run:

```bash
python /workspace/openclaw/d4rt/datasets/tests/test_collate_targets.py
```

### Static reprojection weight behavior

Ensures:

- default weight `1.0` keeps old behavior
- `0.0` down-weights static queries fully in 3D loss
- intermediate values interpolate as expected
- `loss_3d_unweighted` remains unchanged

Run:

```bash
python /workspace/openclaw/d4rt/losses/test_static_reprojection_weight.py
```

### transform_metadata / highres contract tests

Locks current supported behavior for decoder-side high-res patch extraction:

- `sampled_highres` requires `transform_metadata`
- non-`crop-normalized` `canonical_space` must fail
- happy path with `crop_size_hw` should run
- `sampled_resized` should still work without `transform_metadata`

Run:

```bash
python /workspace/openclaw/d4rt/models/tests/test_transform_metadata_contracts.py
```

---

## Practical guidance

If your goal is **paper-faithful strongest training**, start from:

1. `configs/mixture_train_highres.yaml`
2. `--patch-provider sampled_highres`
3. inspect `static_query_ratio / temporal_query_ratio`
4. inspect `normal_valid3d_ratio`
5. only then decide whether to tune `--loss-w-static-reprojection`

If your goal is **minimal behavior change with better observability**, keep:

- `--loss-w-static-reprojection 1.0`

and just use the new metrics to understand the dataset mix before changing training semantics.
