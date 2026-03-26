import argparse
import json
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import os
from tqdm import tqdm
import matplotlib.cm

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

try:
    from PIL import Image
except ImportError:
    Image = None

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
    coords_2d_pred,
    coords_2d_gt,
    input_panel_np=None,
    coords_input=None,
    input_visibility=None,
    input_title="Input",
    output_path=None,
    fps=10.0,
):
    """
    video_np: [T, H, W, 3], uint8
    coords_2d_pred: [N, T, 2], float normalized [0,1]
    coords_2d_gt: [N, T, 2], float normalized [0,1]
    coords_input: [N, 2] fixed input points or [N, T, 2] per-frame input points
    input_visibility: optional [N, T] bool mask for coords_input when it is per-frame
    output_path: path to save video
    """
    T, H, W, C = video_np.shape
    N = coords_2d_pred.shape[0]
    
    # Calculate output width: 3 * W (GT | Input | Output)
    out_W = 3 * W
    out_H = H
    
    writer = None
    gif_frames = None
    output_suffix = None
    if output_path:
        output_path = Path(output_path)
        output_suffix = output_path.suffix.lower()
        if output_suffix == ".mp4":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_W, out_H))
        elif output_suffix == ".gif":
            gif_frames = []
        else:
            raise ValueError(f"Unsupported output suffix {output_suffix!r}; use .mp4 or .gif")
    
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
            curr_x_gt = int(coords_2d_gt[i, t, 0] * (W - 1))
            curr_y_gt = int(coords_2d_gt[i, t, 1] * (H - 1))
            if 0 <= curr_x_gt < W and 0 <= curr_y_gt < H:
                 cv2.circle(frame_gt, (curr_x_gt, curr_y_gt), 3, color, -1)
            
            if t > 0:
                for k in range(1, t + 1):
                    prev_x = int(coords_2d_gt[i, t-k, 0] * (W - 1))
                    prev_y = int(coords_2d_gt[i, t-k, 1] * (H - 1))
                    curr_trail_x = int(coords_2d_gt[i, t-k+1, 0] * (W - 1))
                    curr_trail_y = int(coords_2d_gt[i, t-k+1, 1] * (H - 1))
                    cv2.line(frame_gt, (prev_x, prev_y), (curr_trail_x, curr_trail_y), color, 2)

            # --- Draw Input (Middle) ---
            if coords_input is not None:
                if coords_input.ndim == 2:
                    input_xy = coords_input[i]
                    input_is_visible = True
                else:
                    input_xy = coords_input[i, t]
                    input_is_visible = input_visibility is None or bool(input_visibility[i, t])
                px = int(input_xy[0] * (W - 1))
                py = int(input_xy[1] * (H - 1))
                if input_is_visible and 0 <= px < W and 0 <= py < H:
                    cv2.circle(frame_input, (px, py), 4, color, -1)
                    cv2.circle(frame_input, (px, py), 2, (255, 255, 255), -1)

            # --- Draw Pred (Right) ---
            curr_x_pred = int(coords_2d_pred[i, t, 0] * (W - 1))
            curr_y_pred = int(coords_2d_pred[i, t, 1] * (H - 1))
            if 0 <= curr_x_pred < W and 0 <= curr_y_pred < H:
                cv2.circle(frame_pred, (curr_x_pred, curr_y_pred), 3, color, -1)
            
            if t > 0:
                for k in range(1, t + 1):
                    prev_x = int(coords_2d_pred[i, t-k, 0] * (W - 1))
                    prev_y = int(coords_2d_pred[i, t-k, 1] * (H - 1))
                    curr_trail_x = int(coords_2d_pred[i, t-k+1, 0] * (W - 1))
                    curr_trail_y = int(coords_2d_pred[i, t-k+1, 1] * (H - 1))
                    cv2.line(frame_pred, (prev_x, prev_y), (curr_trail_x, curr_trail_y), color, 2)
        
        # Concatenate 3 panels
        combined = np.hstack((frame_gt, frame_input, frame_pred))
        
        if writer is not None:
            writer.write(combined)
        elif gif_frames is not None:
            gif_frames.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

    if writer is not None:
        writer.release()
        print(f"Saved comparison video to {output_path}")
    elif gif_frames is not None:
        if imageio is not None:
            imageio.mimsave(output_path, gif_frames, duration=1.0 / max(fps, 1e-6), loop=0)
        elif Image is not None:
            pil_frames = [Image.fromarray(frame) for frame in gif_frames]
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(round(1000.0 / max(fps, 1e-6))),
                loop=0,
            )
        else:
            raise RuntimeError("Saving GIF requires either imageio or Pillow to be installed")
        print(f"Saved comparison animation to {output_path}")

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
        "--sampling-mode",
        type=str,
        default="deterministic",
        choices=["deterministic", "random"],
        help=(
            "How to sample the clip/crop/queries from PointOdyssey. "
            "'deterministic' replays the same sample for a given dataset index, "
            "while 'random' resamples on each run."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional global RNG seed. Useful with --sampling-mode=random when you want "
            "a different sample configuration to be reproducible."
        ),
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="gif",
        choices=["gif", "mp4"],
        help="Animation format for the track comparison output",
    )
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
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        print(f"Using global seed {args.seed}")
    
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model, ckpt_args = load_model(args, Path(args.checkpoint), device)

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

    print(
        "Resolved visualization config: "
        f"data_root={resolved_data_root}, split={resolved_split}, sequence={resolved_sequence}"
    )
    
    # 2. Load Dataset with FORCED STRIDE = 1 and SMALL num_queries
    print(
        f"Loading sequence {resolved_sequence} with stride=1, num_queries={args.num_queries}, "
        f"sampling_mode={args.sampling_mode}..."
    )
    patch_provider = ckpt_args.get("patch_provider", "auto")
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
        strides=[1], # Force stride 1
        query_mode=ckpt_args.get("query_mode", "full"),
        precompute_local_patches=precompute_local_patches,
        return_query_video=patch_provider == "sampled_highres",
        static_scene_frame_idx=ckpt_args.get("static_scene_frame_idx", None),
    )
    
    if len(dataset) == 0:
        print("No data found!")
        return

    if args.sampling_mode == "random" and args.seed is None:
        print("Sampling mode is random with no seed; clip/crop/query selection may change on every run.")

    # 3. Process one sample
    sample, success = dataset[0]
    if not success:
        print("Failed to load sample")
        return
        
    print(f"Sample loaded. Frame indices: {sample['frame_indices'].tolist()}")
    
    # Prepare inputs
    input_dtype = torch.float32
    # video in dataset is [S, 3, H, W]
    # We need to unsqueeze batch dim -> [1, S, 3, H, W]
    video = sample["video"].unsqueeze(0).to(device=device, dtype=input_dtype)
    coords = sample["coords"].unsqueeze(0).to(device=device, dtype=input_dtype)
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

    # 4. Generate Full Trajectories
    print("Generating full trajectories for sparse points...")
    
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
        
        # 3. Normalize
        gt_traj_norm_x = gt_traj_crop_x / max(crop_w - 1, 1)
        gt_traj_norm_y = gt_traj_crop_y / max(crop_h - 1, 1)
        
        # Store for Visualization
        full_tracks_gt[i, :, 0] = gt_traj_norm_x
        full_tracks_gt[i, :, 1] = gt_traj_norm_y
        full_visibility[i, :] = gt_visible
        
        if effective_source_mode == "fixed":
            coords_fixed[0, i, 0] = float(gt_traj_norm_x[fixed_source_frame])
            coords_fixed[0, i, 1] = float(gt_traj_norm_y[fixed_source_frame])

    dataset_coords_cpu = coords.squeeze(0).cpu().numpy()
    dataset_t_src_cpu = t_src.squeeze(0).cpu().numpy()
    fixed_coords_cpu = None if coords_fixed is None else coords_fixed.squeeze(0).cpu().numpy()
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
        use_sample_local_patches = True
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
            curr_coords = torch.from_numpy(full_tracks_gt[:, t, :]).unsqueeze(0).to(device=device, dtype=torch.float32)
            curr_t_src = torch.full((1, num_queries), t, device=device, dtype=torch.long)
            curr_t_tgt = curr_t_src
            curr_t_cam = curr_t_src
        elif effective_source_mode == "dataset":
            curr_coords = coords
            curr_t_src = t_src
            curr_t_tgt = torch.full_like(t_tgt, t)
            curr_t_cam = torch.full_like(t_cam, t)
        else:
            curr_coords = coords_fixed
            curr_t_src = t_src_fixed
            curr_t_tgt = torch.full((1, num_queries), t, device=device, dtype=torch.long)
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
            pred_t = curr_preds["pos_2d"].squeeze(0).float().cpu().numpy() # [N, 2]
            input_t = curr_coords.squeeze(0).float().cpu().numpy() # [N, 2]
            identity_mask = (curr_t_tgt == curr_t_src).squeeze(0).cpu().numpy().astype(bool)
            if identity_mask.any():
                diff = np.abs(pred_t[identity_mask] - input_t[identity_mask]).mean()
                print(
                    f"Frame {t} Pred vs Input Diff (identity queries: {int(identity_mask.sum())}): "
                    f"{diff:.6f}"
                )
                identity_reported = True
        
        full_tracks_2d.append(curr_preds["pos_2d"].squeeze(0).float().cpu()) # [N, 2]

    # Stack to [N, S, 2]
    full_tracks_2d = torch.stack(full_tracks_2d, dim=1).numpy()
    if not identity_reported:
        print("No identity queries were encountered for the selected source mode.")
    
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
    pred_tracks_px[..., 0] = full_tracks_2d[..., 0] * (W - 1)
    pred_tracks_px[..., 1] = full_tracks_2d[..., 1] * (H - 1)
    gt_tracks_px[..., 0] = full_tracks_gt[..., 0] * (W - 1)
    gt_tracks_px[..., 1] = full_tracks_gt[..., 1] * (H - 1)

    visible_mask = full_visibility.astype(bool)
    if visible_mask.any():
        l1_norm = np.abs(full_tracks_2d - full_tracks_gt).mean(axis=-1)
        l1_px = np.abs(pred_tracks_px - gt_tracks_px).mean(axis=-1)
        l2_px = np.linalg.norm(pred_tracks_px - gt_tracks_px, axis=-1)
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

        pred_x = int(full_tracks_2d[i, preview_frame_idx, 0] * (W - 1))
        pred_y = int(full_tracks_2d[i, preview_frame_idx, 1] * (H - 1))
        cv2.circle(frame_pred, (pred_x, pred_y), 4, color, -1)
        
    combined_first = np.hstack((frame_gt, frame_input, frame_pred))
    first_frame_path = str(output_dir / f"{resolved_sequence}_frame{preview_frame_idx}_check.png")
    cv2.imwrite(first_frame_path, combined_first)
    print(f"Saved first frame alignment check to {first_frame_path}")
    
    output_video_path = output_dir / f"{resolved_sequence}_compare_tracks.{args.output_format}"
    draw_tracks_2d_compare(
        video_np,
        full_tracks_2d,
        full_tracks_gt,
        input_panel_np=input_panel_np,
        coords_input=input_coords_vis,
        input_visibility=input_visibility,
        input_title=input_title,
        output_path=output_video_path,
        fps=10.0,
    )
    print(f"Saved comparison animation to {output_video_path}")
    
    # Also save the actual first input panel for quick inspection.
    cv2.imwrite(str(output_dir / "source_points_sparse.png"), frame_input)

if __name__ == "__main__":
    main()
