import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import os
from tqdm import tqdm
import matplotlib.cm

from data.dataset import PointOdysseyDataset
from models import create_d4rt
from utils.camera import umeyama_alignment

# --- Helper Functions ---

def get_inference_autocast_dtype(device):
    if device.type != "cuda":
        return None
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def inference_autocast_context(device):
    dtype = get_inference_autocast_dtype(device)
    if dtype is None:
        from contextlib import nullcontext
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)

def load_model(args, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    
    videomae_model = args.videomae_model
    if videomae_model is None:
        videomae_model = ckpt_args.get("videomae_model", None)

    print(f"Loading model with VideoMAE: {videomae_model}")
    model = create_d4rt(
        variant=ckpt_args.get("encoder", "base"),
        img_size=ckpt_args.get("img_size", 256),
        temporal_size=ckpt_args.get("num_frames", 48),
        decoder_depth=ckpt_args.get("decoder_depth", 8),
        query_patch_size=ckpt_args.get("patch_size", 9),
        patch_provider=ckpt_args.get("patch_provider", "auto"),
        videomae_model=videomae_model,
        disable_query_patch_embedding=ckpt_args.get("disable_query_patch_embedding", False),
        disable_query_timestep_embedding=ckpt_args.get("disable_query_timestep_embedding", False),
        disable_decoder_cross_attention=ckpt_args.get("disable_decoder_cross_attention", False),
        debug_3d_head_mode=ckpt_args.get("debug_3d_head_mode", "linear"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt_args

def build_source_contact_sheet(video_np, coords_input, t_src, num_queries, max_source_frames=6):
    """Build a static summary panel showing the source frames used by dataset queries."""
    T, H, W, _ = video_np.shape
    unique_frames, counts = np.unique(t_src.astype(np.int64), return_counts=True)
    sort_order = np.argsort(-counts)
    selected_frames = unique_frames[sort_order][:max_source_frames]
    omitted = max(0, len(unique_frames) - len(selected_frames))

    if len(selected_frames) == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    cols = 2 if len(selected_frames) <= 4 else 3
    rows = int(np.ceil(len(selected_frames) / cols))
    tile_w = max(1, W // cols)
    tile_h = max(1, H // rows)

    panel_bgr = np.zeros((H, W, 3), dtype=np.uint8)
    cmap = matplotlib.colormaps.get_cmap('tab10')
    colors = (cmap(np.linspace(0, 1, num_queries))[:, :3] * 255).astype(np.uint8)

    for tile_idx, frame_idx in enumerate(selected_frames):
        row = tile_idx // cols
        col = tile_idx % cols
        y0 = row * tile_h
        x0 = col * tile_w
        y1 = H if row == rows - 1 else min(H, y0 + tile_h)
        x1 = W if col == cols - 1 else min(W, x0 + tile_w)
        curr_tile_h = max(1, y1 - y0)
        curr_tile_w = max(1, x1 - x0)

        frame_rgb = video_np[int(np.clip(frame_idx, 0, T - 1))]
        tile_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        tile_bgr = cv2.resize(tile_bgr, (curr_tile_w, curr_tile_h), interpolation=cv2.INTER_LINEAR)

        query_indices = np.where(t_src.astype(np.int64) == frame_idx)[0]
        for query_idx in query_indices:
            px = int(coords_input[query_idx, 0] * (curr_tile_w - 1))
            py = int(coords_input[query_idx, 1] * (curr_tile_h - 1))
            if 0 <= px < curr_tile_w and 0 <= py < curr_tile_h:
                color = tuple(int(c) for c in colors[query_idx])
                cv2.circle(tile_bgr, (px, py), 4, color, -1)
                cv2.circle(tile_bgr, (px, py), 2, (255, 255, 255), -1)

        label = f"src t={int(frame_idx)}  n={len(query_indices)}"
        cv2.putText(tile_bgr, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        panel_bgr[y0:y1, x0:x1] = tile_bgr

    if omitted > 0:
        footer = f"+ {omitted} more source frames"
        cv2.putText(panel_bgr, footer, (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)

def select_fixed_source_point_indices(
    trajs_2d_all,
    valids_all,
    visibs_all,
    frame_indices,
    source_frame,
    crop_offset_xy,
    crop_size_hw,
    max_queries,
):
    """Select points that are truly visible on one fixed source frame inside the sampled clip."""
    clip_source_frame = int(source_frame)
    abs_source_frame = int(frame_indices[clip_source_frame])
    source_xy = trajs_2d_all[abs_source_frame]

    x0, y0 = crop_offset_xy
    crop_h, crop_w = crop_size_hw
    crop_x = source_xy[:, 0] - x0
    crop_y = source_xy[:, 1] - y0
    in_bounds = (
        (crop_x >= 0.0)
        & (crop_x < crop_w)
        & (crop_y >= 0.0)
        & (crop_y < crop_h)
    )
    visible = (
        (valids_all[abs_source_frame] > 0.5)
        & (visibs_all[abs_source_frame] > 0.5)
        & in_bounds
    )
    candidates = np.flatnonzero(visible)
    if len(candidates) == 0:
        raise ValueError(
            f"No valid query points are visible on clip source frame {clip_source_frame} "
            f"(absolute frame {abs_source_frame})."
        )

    if len(candidates) <= max_queries:
        return candidates

    # Evenly subsample to keep the visualization deterministic.
    take_idx = np.floor(
        np.linspace(0, len(candidates), num=max_queries, endpoint=False)
    ).astype(np.int64)
    return candidates[take_idx]

def get_panel_frame(panel_np, frame_idx, fallback_frame_rgb):
    """Return one RGB frame for a panel specification."""
    if panel_np is None:
        return fallback_frame_rgb
    if panel_np.ndim == 3:
        return panel_np
    return panel_np[frame_idx]

def draw_tracks_2d_compare(
    video_np,
    coords_2d_pred, # [N, T, 2] in pixel coords
    coords_2d_gt, # [N, T, 2] in pixel coords
    input_panel_np=None,
    coords_input=None,
    input_visibility=None,
    input_title="Input",
    output_path=None,
):
    """
    video_np: [T, H, W, 3], uint8
    coords_2d_pred: [N, T, 2], float PIXEL COORDS (0..W-1)
    coords_2d_gt: [N, T, 2], float PIXEL COORDS (0..W-1)
    """
    T, H, W, C = video_np.shape
    N = coords_2d_pred.shape[0]
    
    # Calculate output width: 3 * W (GT | Input | Output)
    out_W = 3 * W
    out_H = H
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, 10.0, (out_W, out_H))
    
    cmap = matplotlib.colormaps.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, N))[:, :3] * 255 # [N, 3]
    
    for t in tqdm(range(T), desc="Rendering video"):
        frame = video_np[t].copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_input_base = get_panel_frame(input_panel_np, t, video_np[t])
        frame_input_base = cv2.cvtColor(frame_input_base.copy(), cv2.COLOR_RGB2BGR)
        
        # Create 3 panels
        frame_gt = frame.copy()
        frame_input = frame_input_base
        frame_pred = frame.copy()
        
        cv2.putText(frame_gt, "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_input, input_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame_pred, "Pred", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        for i in range(N):
            color = tuple(int(c) for c in colors[i])
            
            # --- Draw GT (Left) ---
            # coords are ALREADY in pixels! DO NOT MULTIPLY BY W/H AGAIN!
            curr_x_gt = int(coords_2d_gt[i, t, 0])
            curr_y_gt = int(coords_2d_gt[i, t, 1])
            
            # Draw if in bounds (and maybe even if slightly out of bounds for debugging?)
            # Usually we only draw if visible. 
            # But let's draw everything in bounds.
            if -1000 < curr_x_gt < 10000: # Loose check to avoid cv2 errors
                 # Only draw if roughly inside image for visibility
                 if 0 <= curr_x_gt < W and 0 <= curr_y_gt < H:
                     cv2.circle(frame_gt, (curr_x_gt, curr_y_gt), 4, color, -1)
                     cv2.circle(frame_gt, (curr_x_gt, curr_y_gt), 2, (255, 255, 255), -1)
            
            # --- Draw Pred (Right) ---
            curr_x_pred = int(coords_2d_pred[i, t, 0])
            curr_y_pred = int(coords_2d_pred[i, t, 1])
            
            if 0 <= curr_x_pred < W and 0 <= curr_y_pred < H:
                 cv2.circle(frame_pred, (curr_x_pred, curr_y_pred), 4, color, -1)
                 cv2.circle(frame_pred, (curr_x_pred, curr_y_pred), 2, (255, 255, 255), -1)
                 
        # --- Draw Input (Middle) ---
        # Input panel drawing logic remains (assuming coords_input is handled correctly upstream)
        # But wait, we didn't touch input panel drawing here.
        # Assuming input_panel_np is already a rendered image or coords_input logic is separate.
        # Actually `draw_tracks_2d_compare` usually doesn't re-draw input points if input_panel_np is provided as an image?
        # Let's check the original code logic for input panel.
        
        # Concatenate
        combined = np.hstack([frame_gt, frame_input, frame_pred])
        if output_path:
            out.write(combined)
            
    if output_path:
        out.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--videomae-model", type=str, default=None)
    parser.add_argument("--num-queries", type=int, default=16, help="Number of points to track")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:1 or cpu")
    parser.add_argument(
        "--source-mode",
        type=str,
        default="fixed",
        choices=["fixed", "clip0", "dataset", "per_frame"],
        help=(
            "How to visualize source queries: fixed/clip0 fixes all queries to one clip frame "
            "(controlled by --source-frame); "
            "dataset uses the dataset-sampled (coords, t_src); per_frame uses identity queries "
            "with t_src=t_tgt=t_cam=t and the GT point location on each frame."
        ),
    )
    parser.add_argument(
        "--source-frame",
        type=int,
        default=0,
        help=(
            "Clip-relative source frame for fixed-source visualization. "
            "Used when --source-mode=fixed or clip0."
        ),
    )
    parser.add_argument(
        "--camera-frame",
        type=int,
        default=None,
        help=(
            "Clip-relative camera frame for visualization. "
            "If not set, t_cam follows t_tgt (the current frame being rendered)."
        ),
    )
    parser.add_argument(
        "--camera-mode",
        type=str,
        default="follow_tgt",
        choices=["follow_tgt", "fixed", "follow_src", "dynamic"],
        help=(
            "How the camera moves during visualization: "
            "'follow_tgt' means t_cam=t_tgt (moves with the current frame); "
            "'fixed' means t_cam is fixed to --camera-frame; "
            "'follow_src' means t_cam=t_src (fixed to where the point was queried)."
        )
    )
    parser.add_argument(
        "--patch-provider",
        type=str,
        default=None,
        help="Override patch provider mode (e.g., sampled_highres)",
    )
    args = parser.parse_args()
    
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model, ckpt_args = load_model(args, Path(args.checkpoint), device)
    
    # [DEBUG] Force model to train mode to see if it fixes the discrepancy
    # Some layers like LayerNorm/BatchNorm/Dropout behave differently in eval mode.
    # VideoMAE uses LayerNorm which should be fine, but let's check.
    # model.train() 
    # print("WARNING: Model is in TRAIN mode for debugging!")

    resolved_data_root = args.data_root or ckpt_args.get("data_root")
    resolved_split = args.split or ckpt_args.get("val_split") or ckpt_args.get("train_split")
    resolved_sequence = args.sequence or ckpt_args.get("val_sequence") or ckpt_args.get("train_sequence")
    if resolved_data_root is None:
        raise ValueError("Could not resolve --data-root from CLI or checkpoint args")
    if resolved_split is None:
        raise ValueError("Could not resolve --split from CLI or checkpoint args")
    if resolved_sequence is None:
        split_root = Path(resolved_data_root) / resolved_split
        if split_root.is_dir():
            candidate_sequences = sorted(
                path.name for path in split_root.iterdir()
                if path.is_dir()
            )
            if len(candidate_sequences) == 1:
                resolved_sequence = candidate_sequences[0]
            elif candidate_sequences:
                preview = ", ".join(candidate_sequences[:8])
                raise ValueError(
                    "Could not resolve --sequence from CLI or checkpoint args. "
                    f"This checkpoint was trained on split={resolved_split} without a fixed sequence. "
                    f"Pass --sequence explicitly, for example one of: {preview}"
                )
    if resolved_sequence is None:
        raise ValueError("Could not resolve --sequence from CLI or checkpoint args")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.checkpoint).resolve().parent / "visualize_tracks"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the args used for visualization
    with open(output_dir / "viz_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(
        "Resolved visualization config: "
        f"data_root={resolved_data_root}, split={resolved_split}, sequence={resolved_sequence}"
    )
    
    # [DEBUG] Check VideoMAE Normalization
    # dataset.py normalizes to [0, 1]. VideoMAE usually expects ImageNet Mean/Std normalization.
    # Check if the model config or dataset config implies normalization.
    # If not, and the model was pretrained on ImageNet, this is a huge issue.
    # But we are finetuning, so maybe it learned to adapt?
    # Unless we froze the backbone?
    # Let's check model parameters.
    print(f"Model backbone frozen: {ckpt_args.get('freeze_backbone', False)}")
    
    # [CRITICAL DEBUG] Check if we are running in a different precision
    # Training uses AMP (float16/bfloat16).
    # Inference here uses autocast if available.
    
    # 2. Load Dataset with FORCED STRIDE = 1 and SMALL num_queries
    print(f"Loading sequence {resolved_sequence} with stride=1 and num_queries={args.num_queries}...")
    
    # 优先使用命令行参数，否则使用 ckpt 参数，最后默认 auto
    patch_provider = args.patch_provider or ckpt_args.get("patch_provider", "auto")
    print(f"Using patch_provider: {patch_provider}")
    
    precompute_local_patches = (
        not ckpt_args.get("disable_precompute_local_patches", False)
        and patch_provider not in {"sampled_resized", "sampled_highres"}
    )
    dataset = PointOdysseyDataset(
        dataset_location=resolved_data_root,
        dset=resolved_split,
        patch_size=ckpt_args.get("patch_size", 9),
        S=ckpt_args.get("num_frames", 48),
        img_size=ckpt_args.get("img_size", 256),
        num_queries=args.num_queries, # Sparse queries
        use_augs=False,
        verbose=True,
        sequence_name=resolved_sequence,
        strides=[1], # Force stride 1 for testing (so frames are continuous)
        query_mode=ckpt_args.get("query_mode", "full"),
        precompute_local_patches=precompute_local_patches,
        return_query_video=patch_provider == "sampled_highres",
        static_scene_frame_idx=ckpt_args.get("static_scene_frame_idx", None),
        t_tgt_eq_t_cam_ratio=ckpt_args.get("t_tgt_eq_t_cam_ratio", 0.4), # Match training distribution of queries
        # use_motion_boundaries=not ckpt_args.get("disable_motion_boundary_oversampling", True), # MUST MATCH TRAIN
    )
    
    # [HACK FOR OVERFIT TEST]
    # We must reset the seed to the exact state it was during training index=0
    # to get the exact same points
    # dataset.seed = 42
    
    if len(dataset) == 0:
        print("No data found!")
        return

    # [CRITICAL DEBUG] Apply ImageNet normalization manually to see if it fixes things
    # VideoMAE expects normalized input if pretrained.
    # Dataset.py returns [0, 1].
    # Check if we should normalize.
    # The training code doesn't seem to have explicit normalization in dataset.py either!
    # Let's check `models/encoder.py` if it normalizes internally.
    
    # 3. Process one sample
    sample, success = dataset[0]
    if not success:
        print("Failed to load sample")
        return
        
    print(f"Sample loaded. Frame indices: {sample['frame_indices'].tolist()}")
    
    input_dtype = next(model.parameters()).dtype
    
    # Let's check exactly what the dataloader is providing as target_2d!
    if "target_2d" in sample:
        dataset_target_2d = sample["target_2d"]
        print(f"Dataset target_2d shape: {dataset_target_2d.shape}")
        print(f"Dataset target_2d min/max: {dataset_target_2d.min().item():.4f}, {dataset_target_2d.max().item():.4f}")
        print(f"Dataset target_2d first 5: \n{dataset_target_2d[0, :5]}")
    
    video = sample["video"].unsqueeze(0).to(device=device, dtype=input_dtype)
    coords = sample["coords"].unsqueeze(0).to(device=device, dtype=input_dtype)
    
    # Let's print the actual video shape and dtype
    print(f"Video shape: {video.shape}, dtype: {video.dtype}, min/max: {video.min():.2f}/{video.max():.2f}")

    # Verify dataset targets
    if "target_2d" in sample and "t_tgt" in sample:
        # Check validation target exactly
        val_t_tgt = sample["t_tgt"].cpu().numpy()
        val_t_src = sample["t_src"].cpu().numpy()
        print(f"Validation sample t_src: {val_t_src[:5]}")
        print(f"Validation sample t_tgt: {val_t_tgt[:5]}")

    t_src = sample["t_src"].unsqueeze(0).to(device=device)
    t_tgt = sample["t_tgt"].unsqueeze(0).to(device=device)
    t_cam = sample["t_cam"].unsqueeze(0).to(device=device)
    
    aspect_ratio = sample.get("aspect_ratio")
    if aspect_ratio is not None:
        aspect_ratio = aspect_ratio.unsqueeze(0).to(device=device, dtype=input_dtype)
    else:
        aspect_ratio = torch.tensor([1.0], device=device, dtype=input_dtype)
        
    local_patches = sample.get("local_patches")
    if local_patches is not None:
        local_patches = local_patches.unsqueeze(0).to(device=device, dtype=input_dtype)

    video_query = sample.get("video_query")
    if video_query is not None:
        video_query = video_query.unsqueeze(0).to(device=device, dtype=input_dtype)

    transform_metadata_input = {
        key: value.unsqueeze(0).to(device=device)
        for key, value in sample["transform_metadata"].items()
    }

    # [CRITICAL DEBUG] Manual VideoMAE normalization
    # VideoMAE models trained on Kinetics usually expect ImageNet mean/std normalized inputs.
    # dataset.py loads RGB in [0, 1].
    # Let's check if the model backbone expects normalization.
    # If we are using `VideoMAEModel.from_pretrained`, it usually expects normalized inputs!
    # The `train.py` might rely on `VideoMAEFeatureExtractor` or similar? No, it uses `dataset.py` directly.
    # If `dataset.py` does not normalize, and `train.py` does not normalize, then the model is trained on [0, 1].
    # UNLESS `VideoMAEModel` has a built-in normalization layer? No, usually not.
    # BUT, if we initialized from pretrained VideoMAE, the weights expect normalized input!
    # If we finetuned on [0, 1], the model has to "unlearn" the normalization or adapt.
    # With 0.003 loss, it must have adapted!
    
    # However, let's try to print the mean/std of the video batch.
    print(f"Video batch mean: {video.mean():.4f}, std: {video.std():.4f}")
    
    # 4. Generate Full Trajectories
    print("Generating full trajectories for sparse points...")
    
    # 强制让 model.decoder 使用正确的 patch_provider
    if hasattr(model, "module"):
        model.module.decoder.patch_provider = patch_provider
    else:
        model.decoder.patch_provider = patch_provider
    
    S = video.shape[1] # Temporal dim
    full_tracks_2d = []
    query_mode = ckpt_args.get("query_mode", "full")

    if query_mode == "same_frame":
        effective_source_mode = "per_frame"
        print(
            "Checkpoint was trained with query_mode=same_frame; "
            "overriding source mode to per_frame so the input panel matches the actual queries."
        )
    else:
        effective_source_mode = "fixed" if args.source_mode == "clip0" else args.source_mode

    fixed_source_frame = int(args.source_frame)
    if not (0 <= fixed_source_frame < S):
        raise ValueError(
            f"--source-frame must be in [0, {S - 1}] for this clip, got {fixed_source_frame}"
        )
    
    # Pre-encode
    with torch.no_grad():
        with inference_autocast_context(device):
             encoder_features = model.encode(video, aspect_ratio=aspect_ratio)
    
    # Load GT annotations so visualization can build source queries explicitly.
    print("Loading annotations early to fix query coordinates...")
    trajs_2d_all, visibs_all, valids_all, frame_indices = None, None, None, None
    if "trajs_2d" in sample and "visibs" in sample:
        # these are provided by the dataset if return_annotations=True
        trajs_2d_all = sample["trajs_2d"].numpy()
        visibs_all = sample["visibs"].numpy()
        valids_all = sample.get("valids", torch.ones_like(sample["visibs"])).numpy()
        frame_indices = sample["frame_indices"].numpy()
        print(f"Loaded trajs_2d_all from sample: min/max = {trajs_2d_all.min():.2f}, {trajs_2d_all.max():.2f}")
    else:
        seq_name = resolved_sequence
        seq_path = os.path.join(dataset.root, seq_name)
        
        anno_path = os.path.join(seq_path, "anno.npz")
        if not os.path.exists(anno_path):
            import glob
            npzs = glob.glob(os.path.join(seq_path, "*.npz"))
            if len(npzs) > 0: anno_path = npzs[0]
            else:
                 print("Error: No annotation file found!")
                 return
            
        anno = np.load(anno_path, allow_pickle=True)
        trajs_2d_all = anno["trajs_2d"] # [TotalFrames, TotalPoints, 2]
        valids_all = anno["valids"]
        visibs_all = anno["visibs"] if "visibs" in anno else valids_all
        frame_indices = sample['frame_indices'].numpy() # [S]
        print(f"Loaded trajs_2d_all from sample: min/max = {trajs_2d_all.min():.2f}, {trajs_2d_all.max():.2f}")
    
    # Check what dataset target_2d looks like compared to raw trajs
    if "target_2d" in sample:
        print(f"Sample target_2d shape: {sample['target_2d'].shape}")
        print(f"Sample target_2d min/max: {sample['target_2d'].min():.4f}, {sample['target_2d'].max():.4f}")
        
        # [DEBUG OVERFIT] For overfitting on a single sequence, if we are in dataset mode,
        # we MUST use exactly the same T_SRC as training!
        # In train.py, the model only ever saw t_src=0 for the fixed queries, or random if not overfit.
        # But wait, t_tgt is what we want to evaluate. In training, t_tgt is random.
        # In evaluation, we predict all frames!
        
        # In D4RT, pos_2d = model(video, coords, t_src, t_tgt, t_cam)
        # It predicts the 2D position at time t_tgt!
        # Our evaluation loop goes over t in range(S), and sets curr_t_tgt = t.
        # So we ask the model: "Where is the point at time t?"
        
        # But look at how dataset target_2d is formed in dataset.py:
        # target_2d is the GT position at `t_tgt`!
        # And the L1 loss during training is computed ONLY at `t_tgt`.
        
        # Our `mean_l1_01` is computed across ALL FRAMES simultaneously.
        # Maybe the model is only good at the specific `t_tgt` it was trained on recently?
        # NO, it should learn all frames if we train long enough.
    
    meta = sample["transform_metadata"]
    x0 = meta["crop_offset_xy"][0].item()
    y0 = meta["crop_offset_xy"][1].item()
    crop_h = meta["crop_size_hw"][0].item()
    crop_w = meta["crop_size_hw"][1].item()

    dataset_point_indices = sample["targets"]["point_indices"].cpu().numpy()
    print(f"Point indices from sample: {dataset_point_indices.tolist()}")

    if effective_source_mode == "fixed":
        point_indices = select_fixed_source_point_indices(
            trajs_2d_all=trajs_2d_all,
            valids_all=valids_all,
            visibs_all=visibs_all,
            frame_indices=frame_indices,
            source_frame=fixed_source_frame,
            crop_offset_xy=(x0, y0),
            crop_size_hw=(crop_h, crop_w),
            max_queries=args.num_queries,
        )
        print(
            f"Using fixed source frame t={fixed_source_frame} with "
            f"{len(point_indices)} visible query points."
        )
    else:
        point_indices = dataset_point_indices

    num_queries = int(len(point_indices))
    full_tracks_gt = np.zeros((num_queries, S, 2), dtype=np.float32)
    full_visibility = np.zeros((num_queries, S), dtype=bool)

    coords_fixed = None
    t_src_fixed = None
    if effective_source_mode == "fixed":
        coords_fixed = torch.zeros((1, num_queries, 2), device=device, dtype=torch.float32)
        t_src_fixed = torch.full((1, num_queries), fixed_source_frame, device=device, dtype=torch.long)

    for i, pidx in enumerate(point_indices.tolist() if isinstance(point_indices, np.ndarray) else point_indices):
        
        # --- GT Extraction Logic (Same as debug_alignment.py) ---
        # 1. Get raw trajectory
        # trajs_2d_all is [TotalFrames, TotalPoints, 2]
        # We need [S, 2] for the full clip
        gt_traj_raw = trajs_2d_all[frame_indices, pidx, :] 
        
        # 2. Apply Crop
        gt_traj_crop_x = gt_traj_raw[:, 0] - x0
        gt_traj_crop_y = gt_traj_raw[:, 1] - y0
        gt_in_bounds = (
            (gt_traj_crop_x >= 0.0)
            & (gt_traj_crop_x < crop_w)
            & (gt_traj_crop_y >= 0.0)
            & (gt_traj_crop_y < crop_h)
        )
        gt_visible = (
            (valids_all[frame_indices, pidx] > 0.5)
            & (visibs_all[frame_indices, pidx] > 0.5)
            & gt_in_bounds
        )
        
        # 3. Normalize to [-1, 1] for fixed coords to feed to the model properly!
        # wait, the visualization code below relies on full_tracks_gt being in [0, 1]!
        # DO NOT NORMALIZE IT TO [-1, 1] here!
        gt_traj_norm_x = gt_traj_crop_x / max(crop_w - 1, 1)
        gt_traj_norm_y = gt_traj_crop_y / max(crop_h - 1, 1)
        
        # [CRITICAL BUG FIX] dataset.py normalizes based on crop_h and crop_w.
        # But wait, does it use W-1 or W?
        # In PointOdyssey / dataset.py it's usually `coords / [W, H] * 2 - 1` or similar.
        # Let's check the printed dataset target_2d!
        # Query 0 GT extraction vs Dataset target_2d:
        #   Manual GT [0,1] -> [-1,1]: 0.1002, 0.7941
        #   Dataset target_2d [-1,1]: 0.1002, 0.7941
        # It matches perfectly! So our extraction is correct.
        
        # Store for Visualization (keep [0, 1] for visualization overlay)
        full_tracks_gt[i, :, 0] = gt_traj_norm_x
        full_tracks_gt[i, :, 1] = gt_traj_norm_y
        full_visibility[i, :] = gt_visible
        
        if effective_source_mode == "fixed":
            # The model expects query coordinates in [0, 1] range!
            coords_fixed[0, i, 0] = float(gt_traj_norm_x[fixed_source_frame])
            coords_fixed[0, i, 1] = float(gt_traj_norm_y[fixed_source_frame])

    # For dataset mode, coords are in [0, 1] (as we confirmed in dataset.py), no need to convert!
    dataset_coords_cpu = coords.squeeze(0).cpu().numpy()
    dataset_t_src_cpu = t_src.squeeze(0).cpu().numpy()
    
    # For visualization, we need [0, 1] coords
    fixed_coords_cpu = None
    if coords_fixed is not None:
        # fixed_coords is ALREADY IN [0, 1] now!
        fixed_coords_cpu = coords_fixed.squeeze(0).cpu().numpy()
        
    query_frames_for_decode = video_query if video_query is not None else video

    input_title = "Input"
    input_panel_np = None
    input_coords_vis = None
    input_visibility = None
    use_sample_local_patches = False
    if effective_source_mode == "fixed":
        print(f"Using clip frame {fixed_source_frame} as the fixed source for all queries.")
        input_title = f"Input (Clip t={fixed_source_frame})"
        input_panel_np = video.squeeze(0).permute(0, 2, 3, 1).float().cpu().numpy()
        input_coords_vis = fixed_coords_cpu
        use_sample_local_patches = False
    elif effective_source_mode == "dataset":
        print("Using dataset-sampled source coordinates and t_src values for each query.")
        input_title = "Input (Dataset Sources)"
        # [DEBUG] Disable local patches to see if they are causing the bias
        use_sample_local_patches = False 
        # use_sample_local_patches = True
    else:
        print(
            "Using per-frame identity queries (t_src=t_tgt=t_cam=t). "
            "This is useful to inspect per-frame inputs, not a fixed-source track."
        )
        input_title = "Input (Per-frame)"
        input_panel_np = video.squeeze(0).permute(0, 2, 3, 1).float().cpu().numpy()
        input_coords_vis = full_tracks_gt.copy()
        input_visibility = full_visibility.copy()
        use_sample_local_patches = False

    local_patches_for_decode = local_patches if use_sample_local_patches else None
    identity_reported = False
    
    for t in tqdm(range(S), desc="Decoding frames"):
        if effective_source_mode == "per_frame":
            # For per-frame, we need to convert GT [0, 1] coords to model's expected [-1, 1] coords
            # WAIT! The model expects [0, 1] coords! dataset.py produces [0, 1].
            curr_coords_01 = full_tracks_gt[:, t, :]
            curr_coords = torch.from_numpy(curr_coords_01).unsqueeze(0).to(device=device, dtype=torch.float32)
            curr_t_src = torch.full((1, num_queries), t, device=device, dtype=torch.long)
            curr_t_tgt = curr_t_src
            
            if args.camera_mode == "fixed" and args.camera_frame is not None:
                curr_t_cam = torch.full((1, num_queries), args.camera_frame, device=device, dtype=torch.long)
            elif args.camera_mode == "follow_src":
                curr_t_cam = curr_t_src
            else: # follow_tgt or dynamic
                curr_t_cam = curr_t_tgt
                
        elif effective_source_mode == "dataset":
            # Dataset coords are ALREADY IN [0, 1] (done by dataset.py)
            curr_coords = coords
            curr_t_src = t_src
            curr_t_tgt = torch.full_like(t_tgt, t)
            
            if args.camera_mode == "fixed" and args.camera_frame is not None:
                curr_t_cam = torch.full_like(t_cam, args.camera_frame)
            elif args.camera_mode == "follow_src":
                curr_t_cam = curr_t_src
            else: # follow_tgt
                curr_t_cam = torch.full_like(t_cam, t)
                
        else:
            # fixed_coords_cpu is already built in [0, 1] range!
            # The model expects [0, 1]!
            curr_coords = coords_fixed
            curr_t_src = t_src_fixed
            curr_t_tgt = torch.full((1, num_queries), t, device=device, dtype=torch.long)
            
            if args.camera_mode == "fixed" and args.camera_frame is not None:
                curr_t_cam = torch.full((1, num_queries), args.camera_frame, device=device, dtype=torch.long)
            elif args.camera_mode == "follow_src":
                curr_t_cam = curr_t_src
            else: # follow_tgt
                curr_t_cam = torch.full((1, num_queries), t, device=device, dtype=torch.long)
        
        with torch.no_grad():
            with inference_autocast_context(device):
                curr_preds = model.decode(
                    encoder_features,
                    query_frames_for_decode,
                    curr_coords,
                    curr_t_src,
                    curr_t_tgt,
                    curr_t_cam,
                    local_patches=local_patches_for_decode,
                    transform_metadata=transform_metadata_input,
                )
        
        # In D4RT, pos_2d is predicted directly.
        # But if the model is not trained well or something is wrong, let's also check if
        # we can reconstruct it from pos_3d if intrinsics were available (but they are not easily here).
        
        # When t_tgt == t_src, the model is solving an identity query.
        if not identity_reported:
            # Model outputs predictions in [0, 1] (we assume now)
            pred_t_01 = curr_preds["pos_2d"].squeeze(0).float().cpu().numpy() # [N, 2]
            
            # curr_coords was fed in [0, 1]
            input_t_01 = curr_coords.squeeze(0).float().cpu().numpy() # [N, 2]
            
            identity_mask = (curr_t_tgt == curr_t_src).squeeze(0).cpu().numpy().astype(bool)
            if identity_mask.any():
                diff_01 = np.abs(pred_t_01[identity_mask] - input_t_01[identity_mask]).mean()
                print(
                    f"Frame {t} Pred vs Input Diff (identity queries: {int(identity_mask.sum())}): "
                    f"0-1 range={diff_01:.6f}"
                )
                print(f"  First 5 Inputs ([0,1]): \n{input_t_01[identity_mask][:5]}")
                print(f"  First 5 Preds ([0,1]): \n{pred_t_01[identity_mask][:5]}")
                identity_reported = True
        
        full_tracks_2d.append(curr_preds["pos_2d"].squeeze(0).float().cpu()) # [N, 2]

    # Stack to [N, S, 2]
    full_tracks_2d = torch.stack(full_tracks_2d, dim=1).numpy()
    if not identity_reported:
        print("No identity queries were encountered for the selected source mode.")

    # [DEBUG] Compare raw predictions directly against target_2d from dataset!
    if "target_2d" in sample and "t_tgt" in sample:
        ds_target = sample["target_2d"].cpu().numpy() # [1, N, 2]
        ds_t_tgt = sample["t_tgt"].cpu().numpy()[0] # [N]
        ds_vis = sample.get("target_vis", torch.ones_like(sample["target_2d"][..., 0])).cpu().numpy()[0] # [N]
        
        preds_for_ds_tgt = np.zeros_like(ds_target[0]) # [N, 2]
        for i in range(num_queries):
            preds_for_ds_tgt[i] = full_tracks_2d[i, ds_t_tgt[i]]
            
        # compute error on valid points
        valid_mask = ds_vis > 0.5
        if valid_mask.any():
            raw_diff = np.abs(preds_for_ds_tgt[valid_mask] - ds_target[valid_mask]).mean()
            print(f"!!! REAL METRIC !!! Mean L1 Diff vs Dataset target_2d ([0, 1] scale): {raw_diff:.6f}")
        else:
            print("No valid points in dataset target_2d!")
    
    # 5. Extract GT Trajectories (Already loaded above, just process for vis)
    # full_tracks_gt = np.zeros((args.num_queries, S, 2), dtype=np.float32)
    
    # for i in range(args.num_queries):
    #     point_idx = point_indices[i].item()
    #     
    #     # Extract full trajectory for this exact point for the sampled frames
    #     gt_traj_raw = trajs_2d_all[frame_indices, point_idx, :] 
    #     
    #     # Apply crop
    #     gt_traj_crop_x = gt_traj_raw[:, 0] - x0
    #     gt_traj_crop_y = gt_traj_raw[:, 1] - y0
    #     
    #     # Normalize
    #     gt_traj_norm_x = gt_traj_crop_x / max(crop_w - 1, 1)
    #     gt_traj_norm_y = gt_traj_crop_y / max(crop_h - 1, 1)
    #     
    #     full_tracks_gt[i, :, 0] = gt_traj_norm_x
    #     full_tracks_gt[i, :, 1] = gt_traj_norm_y

    # Visualize
    # video tensor is [1, S, 3, H, W] -> [S, H, W, 3]
    video_np = (video.squeeze(0).permute(0, 2, 3, 1).float().cpu().numpy() * 255).astype(np.uint8)
    H, W = video_np.shape[1:3]

    if effective_source_mode == "fixed":
        input_panel_np = np.repeat(video_np[fixed_source_frame:fixed_source_frame + 1], S, axis=0)
    elif effective_source_mode == "dataset":
        # dataset_coords_cpu is now [0, 1]
        input_panel_np = build_source_contact_sheet(
            video_np=video_np,
            coords_input=dataset_coords_cpu,
            t_src=dataset_t_src_cpu,
            num_queries=num_queries,
        )
        input_coords_vis = None
        input_visibility = None
    elif effective_source_mode == "per_frame":
        input_panel_np = video_np.copy()
    
    # Check shape to debug OpenCV error
    print(f"Video shape for visualization: {video_np.shape}")
    pred_tracks_px = np.empty_like(full_tracks_2d)
    gt_tracks_px = np.empty_like(full_tracks_gt)
    
    # [CRITICAL FIX] The model predicts coordinates in [-1, 1] range!
    # We must unnormalize from [-1, 1] to [0, W-1] / [0, H-1]
    # WAIT! In D4RT dataset.py, normalize_coords is defined as:
    # coords_x = coords_x / max(crop_w - 1, 1) * 2 - 1
    # BUT crop_w and crop_h are from the random crop during training.
    # When visualizing the WHOLE image (which is 256x256), the model thinks it's a 256x256 image!
    # So we should unnormalize based on W and H (the actual video dimensions fed to the model),
    # not the original crop_w/crop_h (which is 1920x1080) if we didn't crop the validation video!
    
    # [CRITICAL FIX] The model predicts coordinates in [0, 1] range!
    # We must unnormalize from [0, 1] to [0, W-1] / [0, H-1]
    
    # NOTE: The model was trained on 256x256 images (or 512x512). 
    # But here we are feeding it the FULL RESOLUTION image (e.g. 1920x1080) which is resized to 256x256 by the model or dataset?
    # Wait, in `dataset` mode, the `video` tensor comes from the dataloader.
    # The dataloader resizes the full image to `img_size` (256 or 512).
    # So the model sees a 256x256 image that contains the SQUEEZED 1920x1080 content.
    # The model outputs [0, 1] coordinates relative to this 256x256 image.
    # Since the 256x256 image is a squeezed version of the 1920x1080 image, 
    # the relative [0, 1] coordinates are preserved!
    # e.g. a point at the center (0.5, 0.5) of the 1920x1080 image is also at (0.5, 0.5) of the 256x256 image.
    
    # So, to visualize on the original resolution (or whatever resolution we want to plot on),
    # we just multiply [0, 1] coords by the target resolution.
    
    # BUT, `pred_tracks_px` and `gt_tracks_px` are used for calculating metrics below.
    # `mean_l1_px` is calculated as `l1_px.mean()`.
    # `l1_px` is `norm(pred - gt, ord=1)`.
    # If we scale both to (W, H) (1920, 1080), then the error is in original pixels.
    # If we scale both to (256, 256), then the error is in training pixels.
    
    # The user is complaining that metrics are poor. 
    # mean_l1_px: 84.5. 
    # If W=1920, 84px is about 4.4% of the width.
    # If W=256, 84px is about 33% of the width!
    
    # Let's check what `W` and `H` are.
    # `B, S, C, H, W = video.shape`
    # In `dataset` mode, `video` comes from the dataloader.
    # The dataloader resizes images to `img_size` (e.g. 256).
    # So W=256, H=256!
    # So `pred_tracks_px` are scaled to 256x256.
    
    # BUT `full_tracks_gt` comes from `trajs_2d_all`.
    # In `dataset` mode, `trajs_2d_all` comes from `sample["trajs_2d"]`.
    # Let's check `dataset.py` to see if `trajs_2d` are normalized or raw pixels.
    # In `dataset.py`, `trajs_2d` are usually absolute pixels in the original video resolution!
    # Wait, `PointOdysseyDataset` usually returns `trajs_2d` in original pixels?
    # Or does it return normalized [-1, 1]?
    
    # If `full_tracks_gt` is in [0, 1] (as we assumed in the extraction logic for `per_frame` mode),
    # then `gt_tracks_px` will be scaled to 256x256.
    
    # Let's look at the extraction logic again.
    # `gt_traj_norm_x = gt_traj_crop_x / max(crop_w - 1, 1)`
    # This logic was used to CREATE `full_tracks_gt` manually in `per_frame` mode.
    # But in `dataset` mode, we skipped this!
    # In `dataset` mode, we didn't populate `full_tracks_gt`!
    # We only populated `full_tracks_2d` (predictions).
    
    # WHERE IS `full_tracks_gt` POPULATED IN DATASET MODE?
    # It seems we missed that part in previous edits or it's hidden.
    # Ah, I see `gt_tracks_px = np.empty_like(full_tracks_gt)`.
    # But `full_tracks_gt` is initialized as `np.zeros((num_queries, S, 2))`.
    # If we are in `dataset` mode, we need to fill `full_tracks_gt` from `sample["target_2d"]` or `sample["trajs_2d"]`.
    
    # In the code block above (lines 750+), we have:
    # `if "target_2d" in sample and "t_tgt" in sample:` -> check logic.
    
    # We need to ensure `full_tracks_gt` is correctly filled for `dataset` mode!
    
    if effective_source_mode == "dataset" and "trajs_2d" in sample:
        # trajs_2d_all is [S, N, 2] in ORIGINAL RAW PIXELS (e.g. 1920x1080)
        # We need to normalize it to [0, 1] range to fill `full_tracks_gt`.
        # BUT we don't know the original resolution unless we check `video_orig_shape` if available.
        # However, `full_tracks_gt` is currently zeros.
        
        # Let's try to infer original resolution from the max value of trajs_2d
        max_x = trajs_2d_all[..., 0].max()
        max_y = trajs_2d_all[..., 1].max()
        
        # Heuristic: if max_x > 256, it's original resolution.
        # If we assume 1080p video (common in PointOdyssey):
        W_orig, H_orig = 1920.0, 1080.0 # This is a guess, but likely correct for this dataset
        # Or better, we can assume the max coordinate is within the image bounds.
        
        # Actually, `sample` might have "original_size" or similar?
        # Let's just use the max values as a proxy if we can't find metadata.
        # But wait, `dataset.py` might store original size.
        
        # For now, let's normalize by max_x/max_y or just assume 1920x1080 if not provided.
        # Or even better, let's just populate `gt_tracks_px` directly with `trajs_2d_all`
        # and then reverse-engineer `full_tracks_gt` (0-1) from it.
        
        # Assuming `trajs_2d_all` is the ground truth in RAW PIXELS.
        gt_tracks_px = trajs_2d_all.transpose(1, 0, 2) # [N, S, 2]
        
        # Now we need to make `pred_tracks_px` comparable to `gt_tracks_px`.
        # `pred_tracks_px` was scaled by (W-1, H-1) where W=256.
        # This means `pred_tracks_px` is in 256x256 pixel space.
        
        # IF `gt_tracks_px` is in 1920x1080 space, we CANNOT compare them directly!
        # We must scale `pred_tracks_px` to 1920x1080 space too!
        
        # We need the scale factor!
        # Scale X = W_orig / 256
        # Scale Y = H_orig / 256
        
        # Let's assume W_orig=1920, H_orig=1080 for PointOdyssey ani1_new_f.
        scale_x = 1920.0 / float(W) # 1920 / 256
        scale_y = 1080.0 / float(H) # 1080 / 256
        
        # BUT wait, the aspect ratio changed!
        # 1920/1080 = 1.77
        # 256/256 = 1.0
        # The image was SQUEEZED.
        
        # The model predicts in [0, 1] space relative to the SQUEEZED image.
        # This [0, 1] space corresponds to [0, 1] space in the ORIGINAL image too!
        # (Assuming linear interpolation resizing).
        
        # So:
        # pred_px_orig_x = pred_01_x * 1920
        # pred_px_orig_y = pred_01_y * 1080
        
        # So we should recalculate `pred_tracks_px` using the ORIGINAL resolution if we want to compare with GT!
        # OR we normalize GT to [0, 1] using original resolution.
        
        # Let's verify max GT values to confirm resolution
        print(f"GT Max X: {max_x}, GT Max Y: {max_y}")
        
        if max_x > 256:
             # GT is definitely high-res. Let's normalize GT to [0, 1]
             # We'll approximate resolution from max values if needed, or use standard 1920x1080
             # For PointOdyssey, it's usually 1920x1080 or 1280x720.
             # ani1_new_f is likely 1920x1080.
             w_guess = 1920 if max_x > 1280 else 1280
             h_guess = 1080 if max_y > 720 else 720
             # Fallback to max + margin if uncertain? No, let's use standard.
             print(f"Guessing original resolution: {w_guess}x{h_guess}")
             
             full_tracks_gt = gt_tracks_px / np.array([w_guess - 1, h_guess - 1])
             
             # Now `full_tracks_gt` is [0, 1].
             # We can re-calculate `gt_tracks_px` to be in 256x256 space for visualization?
             # OR we keep `gt_tracks_px` in original space, and scale `pred_tracks_px` to original space.
             
             # Let's scale PREDICTIONS to ORIGINAL space for metrics to match the user's expectation of pixel error.
             pred_tracks_px[..., 0] = full_tracks_2d[..., 0] * (w_guess - 1)
             pred_tracks_px[..., 1] = full_tracks_2d[..., 1] * (h_guess - 1)
             
             # But we also want to visualize on the video...
             # The video tensor `video` is 256x256.
             # So for plotting on `video`, we need 256x256 coordinates.
             # Let's create a separate `pred_tracks_vis` for plotting.
             pass
    
    # [CRITICAL FIX] The model predicts coordinates in [0, 1] range!
    # We must unnormalize from [0, 1] to [0, W-1] / [0, H-1]
    
    # We need to correctly handle the resolution.
    # W=256, H=256 (because of dataloader resizing)
    
    # [CRITICAL FIX] 
    # Do NOT attempt to reconstruct full_tracks_gt from trajs_2d_all if we are in dataset mode!
    # Because dataset.py might have applied random cropping (even in val mode if configured so)
    # or specific resizing that we can't easily replicate here without knowing x0, y0.
    
    # If we are in dataset mode, we should ONLY trust the `target_2d` provided by the dataset
    # for the specific t_tgt.
    # However, for visualization, we want to see the track over ALL frames.
    
    # The only way to get ground truth for ALL frames that matches the cropped video
    # is to ask the dataset to provide it, OR to rely on the fact that we are in OVERFIT mode
    # where we know use_augs=False means NO RANDOM CROP (x0=0, y0=0) IF img_size matches crop_size?
    
    # Let's check dataset.py again.
    # If use_augs=False:
    # x0, y0, crop_w, crop_h = self._sample_crop(original_h, original_w, py_rng)
    # In _sample_crop: if not use_augs: return 0, 0, min(h, w), min(h, w) (Center Crop) or similar?
    # Actually, standard D4RT dataset often does a Center Crop or Random Crop even in val.
    
    # BUT, if we can't get the crop parameters, we can't visualize GT correctly on the cropped video.
    # However, we DO have the `coords` (t_src position) which is correct.
    # And we have `target_2d` (t_tgt position) which is correct.
    
    # Let's trust `full_tracks_gt` ONLY if we are sure it aligns.
    # For now, let's assume `full_tracks_gt` might be misaligned and NOT use it for the final video 
    # if we detect a massive discrepancy at t_src.
    
    # Let's check t_src alignment!
    # full_tracks_gt is derived from trajs_2d_all (raw).
    # full_tracks_2d is derived from model (on cropped video).
    
    # If we want to visualize correctly, we should project the PREDICTIONS back to the RAW image space?
    # Or project the RAW GT to the CROP space?
    # Since we only have the CROP video (from dataloader), we must project RAW GT -> CROP.
    # But we don't know the Crop!
    
    # WAIT! We can infer the crop/resize from `dataset_coords_cpu` (which is [0, 1] on crop) 
    # vs `trajs_2d_all` (which is raw pixels) at `t_src`.
    
    # Let's try to estimate the transformation!
    # We have `dataset_coords_cpu` [N, 2] (0-1 in Crop)
    # We have `trajs_2d_all` at `t_src`.
    # `dataset_t_src_cpu` tells us which frame.
    
    # Let's pick the first query.
    q_idx = 0
    t = int(dataset_t_src_cpu[q_idx])
    raw_xy = trajs_2d_all[t, point_indices[q_idx]] # Raw pixel
    crop_uv = dataset_coords_cpu[q_idx] # 0-1 in crop
    
    # crop_x = x0 + crop_uv_x * crop_w
    # crop_y = y0 + crop_uv_y * crop_h
    # raw_x = crop_x? No. 
    # The relationship is: 
    # crop_pixel_x = (raw_x - x0) * (img_size / crop_w)
    # crop_uv_x = crop_pixel_x / (img_size - 1)
    
    # It's hard to solve for x0, y0, crop_w without more points.
    # But we have 256 points! We can estimate the affine transform!
    
    from skimage import transform
    
    # Collect paired points
    src_pts_raw = []
    src_pts_crop_01 = []
    
    for i in range(num_queries):
        t = int(dataset_t_src_cpu[i])
        raw = trajs_2d_all[t, point_indices[i]]
        crop = dataset_coords_cpu[i]
        src_pts_raw.append(raw)
        src_pts_crop_01.append(crop)
        
    src_pts_raw = np.array(src_pts_raw)
    src_pts_crop_01 = np.array(src_pts_crop_01)
    
    # Target domain: Crop Pixels (0..W-1)
    src_pts_crop_px = src_pts_crop_01 * np.array([W-1, H-1])
    
    # Estimate transform: Raw -> Crop Px
    # SimilarityTransform (scale, rotation, translation) should be enough if it's just crop+resize
    tform = transform.SimilarityTransform()
    tform.estimate(src_pts_raw, src_pts_crop_px)
    
    print(f"Estimated Raw->Crop Transform: scale={tform.scale}, translation={tform.translation}")
    
    # --- Apply Transform to GT ---
    
    # 1. Select the relevant frames and points from the full GT annotations
    # trajs_2d_all is [TotalFrames, TotalPoints, 2]
    # frame_indices is [S]
    # point_indices is [N]
    
    # Use np.ix_ to slice correctly: [S, N, 2]
    gt_tracks_selected = trajs_2d_all[np.ix_(frame_indices, point_indices)]
    
    # 2. Reshape to [N, S, 2] to match our convention
    gt_tracks_selected = gt_tracks_selected.transpose(1, 0, 2) # [N, S, 2]
    
    # 3. Flatten for transform
    gt_tracks_flat = gt_tracks_selected.reshape(-1, 2) # [N*S, 2]
    
    # 4. Apply transform to get GT in Crop Pixel Space (256x256)
    gt_tracks_crop_px_flat = tform(gt_tracks_flat)
    gt_tracks_vis = gt_tracks_crop_px_flat.reshape(num_queries, S, 2)
    
    # --- Prepare Predictions in Crop Pixel Space ---
    # full_tracks_2d is [N, S, 2] in [0, 1] range
    pred_tracks_vis = full_tracks_2d * np.array([W-1, H-1])
    
    # --- Update Visibility ---
    # Extract visibility for selected points/frames
    if visibs_all is not None:
        gt_vis_selected = visibs_all[np.ix_(frame_indices, point_indices)] # [S, N]
        gt_vis_selected = gt_vis_selected.transpose(1, 0) # [N, S]
        visible_mask = gt_vis_selected > 0.5
        full_visibility = gt_vis_selected # Update for later use
    else:
        visible_mask = np.ones((num_queries, S), dtype=bool)
        full_visibility = visible_mask.astype(float)
    
    # --- Calculate Metrics in Crop Pixel Space ---
    # Now `gt_tracks_vis` is aligned with `pred_tracks_vis`
    pred_tracks_px = pred_tracks_vis
    gt_tracks_px = gt_tracks_vis

    
    l1_px = np.abs(pred_tracks_px - gt_tracks_px).mean(axis=-1)
    l2_px = np.linalg.norm(pred_tracks_px - gt_tracks_px, axis=-1)
    
    print(f"!!! ALIGNED METRIC !!! Mean L1 Px: {l1_px[visible_mask].mean():.4f}")
    
    # Update full_tracks_gt for 0-1 metric if needed, but pixel metric is more intuitive now.
    
    # The image rendering uses W and H (e.g. 256x256), not crop_w/crop_h (e.g. 1920x1080)
    # The points must be scaled to the visualization video size!
    pred_tracks_01 = full_tracks_2d.copy()
    
    visible_mask = full_visibility.astype(bool)
    if visible_mask.any():
        # Calculate L1 norm in [0, 1] space
        l1_norm = np.abs(full_tracks_2d - full_tracks_gt).mean(axis=-1)
        
        # Wait! Is `full_tracks_gt` strictly in [0, 1] for dataset mode?
        # In dataset mode, we populated `full_tracks_gt` by dividing `gt_tracks_px` by `(w_guess-1, h_guess-1)`.
        # `w_guess` was 1920, `h_guess` was 1080.
        # This means `full_tracks_gt` is the coordinate relative to the FULL 1920x1080 image.
        
        # But wait! What if the training `loss_2d` is calculated on the CROPPED image?
        # No, for full image overfit, the crop size is 512x512 (or 256x256) which is the WHOLE image resized!
        # So `target_2d` in the dataset is normalized by the RESIZED image dimensions (e.g. 512, 512) or the ORIGINAL dimensions?
        # Let's check `dataset.py` normalize_coords logic:
        # coords_x = coords_x / max(crop_w - 1, 1)
        # Here `crop_w` is the width of the crop. If no crop, it's the width of the resized image (512).
        # WAIT! The GT points are loaded from the raw dataset, which are in 1920x1080!
        # When `dataset.py` creates `target_2d`, it transforms the 1920x1080 coords into the crop space, and then normalizes them by the crop size!
        
        # If we resize the whole 1920x1080 image to 256x256 without maintaining aspect ratio:
        # A point at (x=1920, y=1080) becomes (x=256, y=256).
        # Normalized, it is (x=1.0, y=1.0).
        # A point at (x=960, y=540) becomes (x=128, y=128).
        # Normalized, it is (x=0.5, y=0.5).
        
        # So the normalized `[0, 1]` coordinates should be EXACTLY THE SAME whether you normalize 
        # the raw pixel by (1920, 1080) or the resized pixel by (256, 256)!
        # EXCEPT for one thing: aspect ratio preserving resize!
        # Does `dataset.py` preserve aspect ratio when resizing the validation image?
        
        # Let's look at `loss_2d` in training: 0.003 ~ 0.008.
        # But our `mean_l1_norm` is 0.13!
        # Why is there a 10x to 30x difference?
        
        # 1. VISIBILITY MASK:
        # In training, `loss_2d` is computed ONLY on `target_vis > 0.5` AND `mask_2d`.
        # Our `mean_l1_norm` here is computed on `visible_mask` which is `full_visibility > 0.5`.
        # In dataset mode, we DID NOT properly populate `full_visibility` from the dataset!
        # Wait, in `dataset` mode, `full_visibility` is left as `np.zeros((N, S))`.
        # Oh, wait! If `full_visibility` is zeros, then `visible_mask.any()` is FALSE!
        # But we do get output metrics, meaning `full_visibility` is NOT zeros.
        # How is it populated?
        # Ah! `full_visibility[i, :] = gt_visible` was executed inside the `for i in range(num_queries):` loop!
        # But `gt_visible` was calculated from `trajs_2d_all`.
        
        # 2. THE T_TGT DISTRIBUTION:
        # The training loss is the average error OVER THE SAMPLED `t_tgt` FRAMES.
        # In `dataset` mode, the model predicts the trajectory for ALL frames (`t=0..47`).
        # `mean_l1_norm` is the average over ALL frames.
        # If the model is good at `t_tgt` but bad at other frames, this would explain the discrepancy.
        # But wait, our `mean_l1_norm` is 0.13. Training `loss_2d` is 0.003.
        # In training, the `loss_2d` is calculated on `target_pos_2d` which is:
        # `target_pos_2d[query_idx, 0] = float(tgt_xy[0] / max(crop_w - 1, 1))`
        
        # In `d4rt.py`, how is `loss_2d` computed?
        # It's an L1 loss between `pos_2d` (which is in `[0, 1]`) and `target_pos_2d` (which is in `[0, 1]`).
        # YES, `target_pos_2d` is in `[0, 1]`!
        # Wait, if `target_pos_2d` is in `[0, 1]`, then `loss_2d = 0.003` means an error of 0.3%.
        # On a 256x256 image, 0.3% is 0.76 pixels.
        # But wait, our `mean_l1_px` is 34px, which is ~13%.
        
        # WHY is there such a massive discrepancy?
        # The training loss is the average OVER THE SAMPLED `t_tgt` FRAMES.
        # In `dataset.py`, `t_tgt` is randomly sampled.
        # However, our testing script `mean_l1_px` is evaluated on ALL frames (t=0..47).
        # In a single-video overfit, the model might not generalize perfectly to ALL frames
        # if the training hasn't converged completely, or if the motion is too large.
        
        # Let's verify the EXACT loss on the EXACT SAME metric.
        # Training `loss_2d` is `L1Loss(pred, target)`.
        # `target` is `target_pos_2d`.
        
        # Wait, I noticed something in the log for `可视化_DEBUG_GT13`:
        # `mean_l1_norm: 0.136`
        # `metric_raw_3d_l1` in training log: 0.014
        # `loss_2d` in training log: 0.003
        
        # Are we computing the error correctly?
        # `l1_norm = np.abs(full_tracks_2d - full_tracks_gt).mean(axis=-1)`
        # Then we average it: `l1_norm[visible_mask].mean()`.
        # `full_tracks_2d` is [N, S, 2]. `full_tracks_gt` is [N, S, 2].
        
        # If the model is only evaluated on valid/visible GT points, it should match the loss!
        # The ONLY difference is that `loss_2d` is computed over `t_tgt` (one frame per query),
        # while our test computes it over ALL frames.
        
        # Let's compute the error specifically on `t_tgt` to see if it matches!
        if "t_tgt" in sample and "target_2d" in sample:
            ds_t_tgt = sample["t_tgt"][0].cpu().numpy() # [N]
            ds_target = sample["target_2d"][0].cpu().numpy() # [N, 2]
            ds_vis = sample["target_vis"][0].cpu().numpy() # [N]
            
            preds_at_t_tgt = np.zeros_like(ds_target)
            for i in range(num_queries):
                preds_at_t_tgt[i] = full_tracks_2d[i, ds_t_tgt[i]]
                
            # ds_target is ALREADY in [0, 1]! DO NOT scale it!
            mask = ds_vis > 0.5
            if mask.any():
                l1_at_t_tgt = np.abs(preds_at_t_tgt[mask] - ds_target[mask]).mean()
                print(f"!!! REAL METRIC at T_TGT !!! Mean L1 Diff vs Dataset target_2d ([0, 1] scale): {l1_at_t_tgt:.6f}")
                
                # Let's also print the L1 diff on the first frame (t_src)
                # Since ds_t_tgt might be anything, let's just look at the prediction vs input for the first frame
                first_frame_pred = full_tracks_2d[:, 0, :] # [N, 2]
                first_frame_gt = dataset_coords_cpu # [N, 2]
                first_frame_diff = np.abs(first_frame_pred - first_frame_gt).mean()
                print(f"!!! REAL METRIC at T_SRC !!! Mean L1 Diff vs Dataset coords ([0, 1] scale): {first_frame_diff:.6f}")
        
        # Calculate pixel errors
        # If we inferred W_orig/H_orig, we use the scaled pred_tracks_px and gt_tracks_px
        # Otherwise we use the default 256x256 scaling
        
        # Check if we successfully scaled to high-res
        is_high_res_metric = (gt_tracks_px[..., 0].max() > W)
        
        if not is_high_res_metric:
             # Fallback to 256x256 metrics if we didn't detect high-res GT
             pred_tracks_px = pred_tracks_vis
             gt_tracks_px = gt_tracks_vis
        
        l1_px = np.abs(pred_tracks_px - gt_tracks_px).mean(axis=-1)
        l2_px = np.linalg.norm(pred_tracks_px - gt_tracks_px, axis=-1)
        
        # Let's also compute the diff in 0-1 range to see if it matches train loss
        l1_01 = np.linalg.norm(pred_tracks_01 - full_tracks_gt, ord=1, axis=-1)
        mean_l1_01 = float(l1_01[visible_mask].mean())
        print(f"Mean L1 Error (0-1 range): {mean_l1_01:.6f}")
        
        summary = {
            "num_queries": int(num_queries),
            "num_frames": int(S),
            "num_visible_points": int(visible_mask.sum()),
            "query_mode": query_mode,
            "source_mode_requested": args.source_mode,
            "source_mode_effective": effective_source_mode,
            "source_frame_requested": int(args.source_frame),
            "source_frame_effective": int(fixed_source_frame) if effective_source_mode == "fixed" else None,
            "mean_l1_norm": float(l1_norm[visible_mask].mean()),
            "mean_l1_px": float(l1_px[visible_mask].mean()),
            "median_l1_px": float(np.median(l1_px[visible_mask])),
            "mean_l2_px": float(l2_px[visible_mask].mean()),
            "median_l2_px": float(np.median(l2_px[visible_mask])),
            "pck_1px": float((l2_px[visible_mask] <= 1.0).mean()),
            "pck_2px": float((l2_px[visible_mask] <= 2.0).mean()),
            "pck_4px": float((l2_px[visible_mask] <= 4.0).mean()),
            "pck_8px": float((l2_px[visible_mask] <= 8.0).mean()),
        }
    else:
        summary = {
            "num_queries": int(num_queries),
            "num_frames": int(S),
            "num_visible_points": 0,
            "query_mode": query_mode,
            "source_mode_requested": args.source_mode,
            "source_mode_effective": effective_source_mode,
            "source_frame_requested": int(args.source_frame),
            "source_frame_effective": int(fixed_source_frame) if effective_source_mode == "fixed" else None,
        }
    print("Tracking summary:", json.dumps(summary, ensure_ascii=False))
    summary_path = output_dir / "tracking_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved tracking summary to {summary_path}")
    
    preview_frame_idx = fixed_source_frame if effective_source_mode == "fixed" else 0

    # --- Verify Reference Frame Alignment ---
    preview_frame = video_np[preview_frame_idx].copy()
    preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_RGB2BGR)
    input_frame0 = get_panel_frame(input_panel_np, preview_frame_idx, video_np[preview_frame_idx])
    input_frame0 = cv2.cvtColor(input_frame0.copy(), cv2.COLOR_RGB2BGR)
    
    frame_gt = preview_frame.copy()
    frame_input = input_frame0
    frame_pred = preview_frame.copy()
    
    cv2.putText(frame_gt, f"GT (Frame {preview_frame_idx})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame_input, input_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame_pred, f"Pred (Frame {preview_frame_idx})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cmap = matplotlib.colormaps.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, num_queries))[:, :3] * 255 

    # Also convert prediction tracking results to 0-1 for rendering
    # pred_tracks_01 already calculated above!
    # pred_tracks_01 = np.empty_like(full_tracks_2d)
    # pred_tracks_01[..., 0] = (full_tracks_2d[..., 0] + 1.0) / 2.0
    # pred_tracks_01[..., 1] = (full_tracks_2d[..., 1] + 1.0) / 2.0

    for i in range(num_queries):
        color = tuple(int(c) for c in colors[i])
        
        gt_x = int(full_tracks_gt[i, preview_frame_idx, 0] * (W - 1))
        gt_y = int(full_tracks_gt[i, preview_frame_idx, 1] * (H - 1))
        cv2.circle(frame_gt, (gt_x, gt_y), 4, color, -1)
        
        if input_coords_vis is not None:
            if input_coords_vis.ndim == 2:
                input_xy = input_coords_vis[i]
                input_is_visible = True
            else:
                input_xy = input_coords_vis[i, preview_frame_idx]
                input_is_visible = input_visibility is None or bool(input_visibility[i, preview_frame_idx])
            in_x = int(input_xy[0] * (W - 1))
            in_y = int(input_xy[1] * (H - 1))
            if input_is_visible and 0 <= in_x < W and 0 <= in_y < H:
                cv2.circle(frame_input, (in_x, in_y), 4, color, -1)
                cv2.circle(frame_input, (in_x, in_y), 2, (255, 255, 255), -1)

        pred_x = int(pred_tracks_01[i, preview_frame_idx, 0] * (W - 1))
        pred_y = int(pred_tracks_01[i, preview_frame_idx, 1] * (H - 1))
        cv2.circle(frame_pred, (pred_x, pred_y), 4, color, -1)
        
    combined_first = np.hstack((frame_gt, frame_input, frame_pred))
    first_frame_path = str(output_dir / f"{resolved_sequence}_frame{preview_frame_idx}_check.png")
    cv2.imwrite(first_frame_path, combined_first)
    print(f"Saved first frame alignment check to {first_frame_path}")
    
    # If we are in `dataset` mode, and we calculated `gt_tracks_px` from raw pixels,
    # we need to be careful about what we pass to `plot_tracks_v2`.
    # `plot_tracks_v2` takes `pred_tracks` and `gt_tracks` and plots them on `video`.
    # `video` is 256x256.
    # So we MUST pass 256x256 coordinates to `plot_tracks_v2`.
    
    # We prepared `pred_tracks_vis` and `gt_tracks_vis` for exactly this purpose.
    
    vis_save_path = os.path.join(output_dir, f"{resolved_sequence}_compare_tracks.mp4")
    
    # [CRITICAL FIX] `input_visibility` is passed to `draw_tracks_2d_compare`
    # But `visibs_all` shape is [S, N] (loaded from dataset)
    # `full_visibility` shape is [N, S] (if created manually)
    # The function likely expects [N, S] or [S, N].
    # `visibs_all.transpose(1, 0)` -> [N, S].
    
    # Wait, `draw_tracks_2d_compare` signature:
    # (video, tracks_pred, tracks_gt, input_panel_np=None, coords_input=None, input_visibility=None, ...)
    # It does NOT take `gt_visibility` separately?
    # It usually assumes GT tracks are visible if `tracks_gt` is not None.
    # Or it might use `input_visibility` for the input panel only?
    
    # Let's check `draw_tracks_2d_compare` implementation if possible, or assume standard API.
    # Usually visualization functions in these repos take tracks in pixel coords.
    # `pred_tracks_vis` and `gt_tracks_vis` are in [0, 255] range (for 256x256 video).
    
    # ISSUE: If `gt_tracks_vis` has zeros or invalid values where visibility is 0, they might be plotted at (0,0).
    # We should ensure we pass the visibility mask if the function supports it.
    
    # If `draw_tracks_2d_compare` doesn't support explicit visibility mask for tracks,
    # we might need to filter points or they will be drawn.
    
    # But the user said "points disappeared". This means they are NOT drawn.
    # This usually happens if:
    # 1. Coordinates are NaN or Inf.
    # 2. Coordinates are out of bounds (negative or > W/H).
    # 3. The plotting function filters them out based on some condition.
    
    # We know `mean_l1_px` is ~34px, so coords are not NaN.
    # Are they out of bounds?
    # `pred_tracks_vis` = `full_tracks_2d * (W-1)`. `full_tracks_2d` is [0, 1].
    # So `pred_tracks_vis` should be [0, 255].
    
    # Wait, did we pass the correct `video_np`?
    # `video_np` is 256x256.
    
    # Let's explicitly check bounds before plotting.
    print(f"Pred tracks vis min/max: {pred_tracks_vis.min():.2f}/{pred_tracks_vis.max():.2f}")
    print(f"GT tracks vis min/max: {gt_tracks_vis.min():.2f}/{gt_tracks_vis.max():.2f}")
    
    draw_tracks_2d_compare(
        video_np, # 256x256
        pred_tracks_vis, # 256x256
        gt_tracks_vis, # 256x256
        input_panel_np=input_panel_np,
        coords_input=input_coords_vis,
        input_visibility=visibs_all.transpose(1, 0) if visibs_all is not None else full_visibility,
        input_title=input_title,
        output_path=vis_save_path,
    )
    print(f"Saved comparison video to {vis_save_path}")
    
    # Also save the actual first input panel for quick inspection.
    cv2.imwrite(str(output_dir / "source_points_sparse.png"), frame_input)

if __name__ == "__main__":
    main()
