# Latest Best Evaluation And Visualization Code

Curated runnable copy of the newest local visualization and metric code for D4RT.

This folder merges the strongest pieces from:

- Worktree visualization code: `/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration`
- Main repository evaluation code: `/data/zbf/openclaw/d4rt`

The entry scripts default to:

- GPU: `CUDA_VISIBLE_DEVICES=0`
- Model: `MODEL_VARIANT=large`
- Checkpoint: newest `checkpoint_latest_*.pth` under `CHECKPOINT_DIR`
- Default `CHECKPOINT_DIR`: `/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration/outputs/mixture_6datasets_cos_planned_from200`
- Current newest checkpoint at README update time: `checkpoint_latest_492.pth`
- Strict metric scripts default to no confidence filtering and rigid Umeyama pose solving.
- Camera pose visualization still uses the visualization defaults because it is for qualitative inspection, not paper metrics.

Set `CHECKPOINT=/abs/path/to/checkpoint.pth` to reproduce an older result.

## Selected Code

| Area | Selected source | Why this version |
|---|---|---|
| Multi-dataset dense visualization | worktree `md/visualize_dynamic_replica_checkpoint.py` | Best current dense visualization path; includes confidence filtering, dense predicted reference/world clouds, GT dynamic world clouds, canonical output, camera trajectory, and intrinsics plots. |
| Sintel visualization | worktree `md/visualize_sintel_checkpoint.py` | Camera-aware Sintel path; emits dynamic point cloud, canonical reconstruction, GT/pred camera trajectory, and intrinsics visualization. |
| Video/image-folder visualization | worktree `md/visualize_video_checkpoint.py` | Newer confidence-aware video visualization path. |
| Depth visualization | worktree `md/visualize_co3dv2_depth.py` | Newer depth visualization helper. |
| Sintel and ScanNet metrics | main repo `md/eval_sintel_checkpoint.py`, `md/eval_scannet_checkpoint.py`, then patched here | Strict paper-style metrics by default; confidence filtering and RANSAC are explicit diagnostic options only. |
| Metrics and camera utilities | main repo `utils/metrics.py`, `utils/camera.py` | Strict depth S/SS, mean-shift point-cloud L1, Sim(3) pose alignment, ATE/RPE/AUC helpers. |

The quick code map is in `CODE_INDEX.md`. Organized result locations are in `RESULTS_INDEX.md`.
Source hashes are recorded in `source_notes/`; `source_notes/current_curated_sha256.txt` is the hash list for the current curated code after local patches. Selection details are in `VERSION_SELECTION.md`.

## Paper-Aligned Evaluation Protocol

The strict entry points are:

```bash
bash visualization_latest_best_20260518/run_eval_sintel.sh
bash visualization_latest_best_20260518/run_eval_scannet.sh
```

These scripts are aligned to `/data/zbf/openclaw/论文/5_experiments.tex`:

- 3D point cloud: decode all observed pixels into one shared reference camera, align prediction and GT by mean-shift only, then report paired coordinate Mean L1. `pointcloud_l1` is the paper metric; `pointcloud_l1_sim3` is diagnostic only.
- Video depth: query `t_src=t_tgt=t_cam`, use the z coordinate as depth, apply one global sequence alignment, and report AbsRel. `depth_S_abs_rel` uses scale-only alignment; `depth_SS_abs_rel` uses scale-and-shift alignment.
- Camera pose: Sintel defaults to the 14 final-pass sequence subset. Relative poses are solved from paired D4RT query point sets with rigid Umeyama, then the predicted trajectory is Sim(3)-aligned to GT before ATE/RPE-T/RPE-R. `pose_auc_30` follows the VGGT-style AUC@30 convention and is reported as a percentage; `pose_auc_30_frac` keeps the raw 0-1 value.

Strict defaults:

- `DEPTH_CONFIDENCE_QUANTILE=0.0`
- `POINTCLOUD_CONFIDENCE_QUANTILE=0.0`
- `POSE_CONFIDENCE_THRESHOLD=0.0`
- `POSE_SOLVER=umeyama`
- `DEPTH_STRIDE=1`
- `POINTCLOUD_STRIDE=1`
- `MAX_POINTCLOUD_POINTS=0`

Optional tuning knobs such as confidence filtering and `POSE_SOLVER=ransac` are retained for debugging but should not be reported as strict paper numbers unless explicitly labeled.
For fast smoke tests, override density explicitly, for example `DEPTH_STRIDE=2 POINTCLOUD_STRIDE=4 MAX_POINTCLOUD_POINTS=50000`.

## Confidence Filtering And Tuned Results

Confidence filtering is useful for visual quality and diagnostic "best effort" numbers, but it is not the strict paper protocol. The reason is simple: filtering removes low-confidence pixels or 3D correspondences, which usually removes hard regions such as motion boundaries, occlusions, sky/infinity depths, thin structures, and correspondence failures. The resulting metric measures "accuracy on retained predictions", not "accuracy on all observed pixels". That is why strict paper runs keep confidence quantiles at `0.0`.

To measure the best tuned result, run a sweep and report the winning config together with its retained coverage settings:

```bash
bash visualization_latest_best_20260518/run_eval_sintel_tuned_sweep.sh
```

The default sweep is intentionally fast: 1 Sintel scene, `DEPTH_STRIDE=2`, `POINTCLOUD_STRIDE=4`, `MAX_POINTCLOUD_POINTS=50000`. It writes:

- `sweep_summary.md`
- `sweep_summary.json`
- one `summary.json` per config

For final tuned numbers on the paper Sintel 14-scene subset, run:

```bash
NUM_SCENES=0 DEPTH_STRIDE=1 POINTCLOUD_STRIDE=1 MAX_POINTCLOUD_POINTS=0 \
SWEEP_CONFIGS="strict:0.0:0.0:umeyama:0.0 conf20:0.2:0.2:umeyama:0.0 conf40:0.4:0.4:umeyama:0.0 ransac:0.0:0.0:ransac:0.0 ransac_conf20:0.2:0.2:ransac:0.3" \
bash visualization_latest_best_20260518/run_eval_sintel_tuned_sweep.sh
```

Interpretation rule:

- Report `run_eval_sintel.sh` default output as paper-aligned.
- Report `run_eval_sintel_tuned_sweep.sh` winners as tuned/diagnostic.
- Do not compare tuned confidence-filtered numbers directly against paper table numbers unless the table uses the same coverage rule.

External protocol references checked while aligning this folder:

- MegaSaM Sintel depth evaluation uses sequence-level median-centered scale+shift before AbsRel: `https://github.com/mega-sam/mega-sam/blob/main/evaluations_depth/evaluate_depth_ours_sintel.py`
- CUT3R video-depth evaluation provides the scale-only robust single-scale alignment: `https://github.com/CUT3R/CUT3R/blob/main/eval/video_depth/tools.py`
- CUT3R/evo pose evaluation uses Sim(3)-style trajectory alignment for ATE/RPE: `https://github.com/CUT3R/CUT3R/blob/main/eval/relpose/evo_utils.py`

## Visualization Commands

Run from repo root `/data/zbf/openclaw/d4rt`, or use absolute script paths.

```bash
bash visualization_latest_best_20260518/run_dataset.sh scannet_test
bash visualization_latest_best_20260518/run_dataset.sh scannetpp_val
bash visualization_latest_best_20260518/run_dataset.sh dynamic_replica_val
bash visualization_latest_best_20260518/run_dataset.sh kubric_val
bash visualization_latest_best_20260518/run_dataset.sh pointodyssey_val
bash visualization_latest_best_20260518/run_dataset.sh blendedmvs_val
bash visualization_latest_best_20260518/run_dataset.sh co3dv2_val
```

Sintel geometry and camera-view visualization:

```bash
bash visualization_latest_best_20260518/run_sintel_visualization.sh
```

Single video or image directory:

```bash
bash visualization_latest_best_20260518/run_video.sh /path/to/video_or_images
```

Depth-only visualization:

```bash
bash visualization_latest_best_20260518/run_depth.sh co3dv2_val
bash visualization_latest_best_20260518/run_depth.sh scannet_test
```

Useful overrides:

```bash
NUM_SAMPLES=1 OUTPUT_DIR=/tmp/d4rt_smoke bash visualization_latest_best_20260518/run_dataset.sh scannet_test
DENSE_PRED_VIS_THRESHOLD=0.5 DENSE_PRED_CONFIDENCE_PERCENTILE=40 bash visualization_latest_best_20260518/run_dataset.sh scannet_test
CHECKPOINT=/path/to/checkpoint_latest_480.pth bash visualization_latest_best_20260518/run_sintel_visualization.sh
```

## Evaluation Commands

Sintel paper-style metrics, including camera pose metrics:

```bash
bash visualization_latest_best_20260518/run_eval_sintel.sh
```

Camera-only Sintel evaluation, optimized for Camera Extrinsics / Intrinsics:

```bash
bash visualization_latest_best_20260518/run_eval_sintel_camera.sh --all-scenes
```

This uses the current best camera configuration by default:
`pose_mode=adjacent`, `pose_solver=umeyama`, `pose_grid=16x16`, `pose_confidence_quantile=0.4`.

Camera pose tuning sweep:

```bash
bash visualization_latest_best_20260518/run_eval_sintel_camera_pose_sweep.sh
```

This is the right entry when the question is "which pose configuration gives the best metric?" It sweeps reference/adjacent composition, grid size, confidence quantile, solver, and confidence weighting. It is separate from `run_eval_sintel.sh`, which is the stricter paper-style multi-task evaluator.

Fast Sintel pose smoke test on one sequence:

```bash
NUM_SCENES=1 OUTPUT_DIR=/tmp/d4rt_sintel_pose_smoke bash visualization_latest_best_20260518/run_eval_sintel.sh
```

ScanNet metrics remain available, but camera pose validation requested for this folder should use Sintel unless another dataset is explicitly required:

```bash
bash visualization_latest_best_20260518/run_eval_scannet.sh
```

Sintel confidence/pose tuning sweep:

```bash
bash visualization_latest_best_20260518/run_eval_sintel_tuned_sweep.sh
```

Latest camera-pose sweep on Sintel paper 14-sequence subset with `checkpoint_latest_492.pth`:

`/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration/outputs/mixture_6datasets_cos_planned_from200/eval_checkpoint_latest_492_sintel_camera_pose_sweep_smoke_20260518/camera_pose_sweep_summary.md`

Best-by-metric from the 8 tested configs:

- Best ATE: `0.081287`, `reference`, `16x16`, `pose_confidence_quantile=0.0`
- Best RPE-T: `0.020354`, `reference`, `16x16`, `pose_confidence_quantile=0.0`
- Best RPE-R: `0.614662`, `adjacent`, `16x16`, `pose_confidence_quantile=0.0`
- Best Pose AUC@30: `8.1302`, `reference`, `8x8`, `pose_confidence_quantile=0.0`

On this checkpoint, hard confidence filtering (`q=0.2` or `q=0.4`) did not improve the main camera-pose metrics; the best runs used all grid correspondences with confidence weighting.

## Output Contents

Dataset/Sintel visualization folders contain:

- `gt_dense_dynamic_world.gif`: dense GT dynamic world point cloud when GT geometry exists
- `pred_dense_reference.gif`: dense prediction in reference camera coordinates
- `pred_dense_world.gif`: dense prediction transformed to camera/world trajectory view
- `canonical.gif`: canonical reconstruction, useful when reference 3D looks static
- `camera_trajectory_gt_pred.png`: GT vs predicted camera pose visualization
- `camera_intrinsics.png`: intrinsics visualization
- `summary.json`: sample metadata and aggregate visualization stats
- static PNG/PLY side products for inspection

Evaluation folders contain `summary.json` with:

- Depth: `depth_S_abs_rel`, `depth_SS_abs_rel`
- Camera pose: `pose_ate`, `pose_rpe_trans`, `pose_rpe_rot`, `pose_auc_30` as percent
- Point cloud: `pointcloud_l1` as the paper metric, plus diagnostic `pointcloud_l1_sim3`

## Known Organized Results

Full dense `s2` results already organized for checkpoint 480:

`/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration/outputs/mixture_6datasets_cos_planned_from200/organized_checkpoint_latest_480_full_results_20260517`

That folder contains 10 samples each for ScanNet test, ScanNet++ val, Dynamic Replica val, Kubric val, PointOdyssey val, BlendedMVS val, CO3Dv2 val, and Sintel geometry, with dense dynamic clouds and camera/intrinsics outputs.

Use the scripts here for new runs with the newest checkpoint. By default, they currently pick `checkpoint_latest_492.pth`.

## Latest Smoke / Metric Verification

Sintel paper-metric smoke on 2026-05-18 with `checkpoint_latest_492.pth` on GPU 0.
This was compute-bounded (`DEPTH_STRIDE=2`, `POINTCLOUD_STRIDE=4`, `MAX_POINTCLOUD_POINTS=50000`), so it validates the metric path but is not a full-density paper number:

`/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration/outputs/mixture_6datasets_cos_planned_from200/latest_best_smoke_20260518/sintel_eval_checkpoint_latest_492_paper_metric_smoke_scene1`

- Protocol flags in `summary.json`: `paper_metric_settings=true`, `paper_density_settings=false`, `full_paper_scene_set=false`
- Scene: `alley_1`, final pass, 48 real frames
- Depth: `AbsRel(S)=0.4841`, `AbsRel(SS)=0.3012`
- Point cloud paper metric: `L1(mean-shift)=1.6785`
- Camera pose: `ATE=0.01021`, `RPE-T=0.00830`, `RPE-R=0.04379`, `Pose AUC@30=0.0`

Historical verification on 2026-05-18 with `checkpoint_latest_490.pth` on GPU 0:

`/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration/outputs/mixture_6datasets_cos_planned_from200/latest_best_smoke_20260518`

Sintel visualization, 1 sequence (`alley_1`):

- Output: `sintel_visualization_checkpoint_latest_490_scene1`
- Dense predicted visible points: `14396.67`
- Dense predicted world visible points: `14396.67`
- Dense GT selected points: `7396.0`
- Canonical visible points: `14754.54`
- Camera pose from visualization summary: `ATE=0.0332`, `RPE-T=0.00855`, `RPE-R=0.0521`

Updated camera-pose visualization fix:

- Output: `sintel_visualization_checkpoint_latest_490_scene1_posefix_grid8`
- Fix: apply Sim(3) global rotation to camera-to-world orientation as `R_global @ R_c2w`, matching `utils.metrics.compute_pose_metrics`
- Pose grid: `8x8`, matching the evaluator defaults
- Camera pose after fix: `ATE=0.0167`, `RPE-T=0.00834`, `RPE-R=0.0496`
- A 16x16 grid was also tested; it improved rotation slightly but worsened center ATE to `0.0188`, so 8x8 remains the default.

Sintel paper-style metrics, 14-sequence subset:

- Output: `sintel_eval_checkpoint_latest_490_paper14`
- Depth: `AbsRel(S)=0.4478`, `AbsRel(SS)=1.8133`
- Point cloud: `L1(mean-shift)=7.6576`, `L1(Sim3)=7.3895`
- Camera pose: `ATE=0.1244`, `RPE-T=0.0731`, `RPE-R=5.0424`, legacy raw `Pose AUC@30=0.06735`

Camera-only all-Sintel verification with `checkpoint_latest_491.pth`:

- Output: `/data/zbf/openclaw/d4rt/.claude/worktrees/cos-acceleration/outputs/mixture_6datasets_cos_planned_from200/camera_eval_debug_20260518/sintel_all23_checkpoint_latest_491_best_camera_adj_umeyama_g16q40`
- All 23 final-pass sequences: `ATE=0.36746`, `RPE-T=0.17761`, `RPE-R=0.50204`, legacy raw `Pose AUC@30=0.06641`
- Paper 14-sequence subset from the same run: `ATE=0.07956`, `RPE-T=0.02133`, `RPE-R=0.66295`, legacy raw `Pose AUC@30=0.05218`
- Intrinsics all 23: focal `AbsRel=0.53000`, `fx AbsRel=0.61407`, `fy AbsRel=0.44593`
