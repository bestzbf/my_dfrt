import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    import matplotlib
except ImportError as exc:
    matplotlib = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None

try:
    from PIL import Image
except ImportError as exc:
    Image = None
    _PIL_IMPORT_ERROR = exc
else:
    _PIL_IMPORT_ERROR = None
import numpy as np
import torch
from tqdm import tqdm

from data import PointOdysseyDataset
from models import create_d4rt
from utils.misc import farthest_point_sample_py
from utils.visualization import save_point_cloud_ply


def require_visualization_dependencies():
    if cv2 is None:
        raise ImportError(
            "visualize_tracks.py requires OpenCV. Install a package like `opencv-python` "
            "in the environment used for visualization."
        ) from _CV2_IMPORT_ERROR
    if matplotlib is None:
        raise ImportError(
            "visualize_tracks.py requires matplotlib in the visualization environment."
        ) from _MATPLOTLIB_IMPORT_ERROR


def require_gif_dependency():
    if Image is None:
        raise ImportError(
            "Generating point-cloud GIFs requires Pillow. Install a package like `pillow` "
            "in the visualization environment."
        ) from _PIL_IMPORT_ERROR


def get_inference_autocast_dtype(device):
    if device.type != "cuda":
        return None
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def inference_autocast_context(device):
    dtype = get_inference_autocast_dtype(device)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def load_model(args, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {})

    videomae_model = args.videomae_model
    if videomae_model is None:
        videomae_model = ckpt_args.get("videomae_model")

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


def build_identity_transform_metadata(original_h, original_w, resized_h, resized_w):
    return {
        "canonical_space": torch.tensor(0, dtype=torch.long),
        "original_hw": torch.tensor([float(original_h), float(original_w)], dtype=torch.float32),
        "crop_offset_xy": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "crop_size_hw": torch.tensor([float(original_h), float(original_w)], dtype=torch.float32),
        "resized_hw": torch.tensor([float(resized_h), float(resized_w)], dtype=torch.float32),
    }


def load_video_clip_for_visualization(
    video_path,
    *,
    start_frame,
    num_frames,
    frame_stride,
    img_size,
):
    require_visualization_dependencies()

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if int(num_frames) <= 0:
        raise ValueError(f"--video-num-frames must be > 0, got {num_frames}")
    if int(frame_stride) <= 0:
        raise ValueError(f"--video-frame-stride must be > 0, got {frame_stride}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0:
        fps = 10.0

    start_frame = int(start_frame)
    if start_frame < 0:
        raise ValueError(f"--video-start-frame must be >= 0, got {start_frame}")
    if total_frames > 0 and start_frame >= total_frames:
        raise ValueError(
            f"--video-start-frame={start_frame} is outside the video with {total_frames} frames"
        )

    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    raw_frames_rgb = []
    frame_indices = []
    current_frame_idx = start_frame
    last_frame_rgb = None

    try:
        while len(raw_frames_rgb) < int(num_frames):
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if (current_frame_idx - start_frame) % int(frame_stride) == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                raw_frames_rgb.append(frame_rgb)
                frame_indices.append(current_frame_idx)
                last_frame_rgb = frame_rgb
            current_frame_idx += 1
    finally:
        capture.release()

    if not raw_frames_rgb:
        raise RuntimeError(f"No frames could be read from {video_path}")

    while len(raw_frames_rgb) < int(num_frames):
        raw_frames_rgb.append(last_frame_rgb.copy())
        frame_indices.append(frame_indices[-1])

    original_h, original_w = raw_frames_rgb[0].shape[:2]
    resized_frames_rgb = [
        cv2.resize(frame_rgb, (int(img_size), int(img_size)), interpolation=cv2.INTER_LINEAR)
        for frame_rgb in raw_frames_rgb
    ]

    resized_video = torch.stack(
        [torch.from_numpy(frame).permute(2, 0, 1) for frame in resized_frames_rgb],
        dim=0,
    ).float() / 255.0
    query_video = torch.stack(
        [torch.from_numpy(frame).permute(2, 0, 1) for frame in raw_frames_rgb],
        dim=0,
    ).float() / 255.0

    return {
        "video": resized_video,
        "video_query": query_video,
        "video_rgb": np.stack(resized_frames_rgb, axis=0).astype(np.uint8),
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "fps": fps,
        "original_height": int(original_h),
        "original_width": int(original_w),
        "resized_height": int(img_size),
        "resized_width": int(img_size),
        "source_path": str(video_path),
    }


def build_source_contact_sheet(video_np, coords_input, t_src, num_queries, max_source_frames=6):
    require_visualization_dependencies()
    t_src = np.asarray(t_src, dtype=np.int64)
    coords_input = np.asarray(coords_input, dtype=np.float32)
    num_queries = min(int(num_queries), int(coords_input.shape[0]))
    video_np = np.asarray(video_np)
    total_frames, height, width, _ = video_np.shape

    unique_frames, counts = np.unique(t_src[:num_queries], return_counts=True)
    sort_order = np.argsort(-counts)
    selected_frames = unique_frames[sort_order][:max_source_frames]
    omitted = max(0, len(unique_frames) - len(selected_frames))

    if len(selected_frames) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    cols = 2 if len(selected_frames) <= 4 else 3
    rows = int(np.ceil(len(selected_frames) / cols))
    tile_width = max(1, width // cols)
    tile_height = max(1, height // rows)

    panel_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    cmap = matplotlib.colormaps.get_cmap("tab10")
    colors = (cmap(np.linspace(0, 1, num_queries))[:, :3] * 255).astype(np.uint8)

    for tile_idx, frame_idx in enumerate(selected_frames):
        row = tile_idx // cols
        col = tile_idx % cols
        y0 = row * tile_height
        x0 = col * tile_width
        y1 = height if row == rows - 1 else min(height, y0 + tile_height)
        x1 = width if col == cols - 1 else min(width, x0 + tile_width)
        curr_tile_h = max(1, y1 - y0)
        curr_tile_w = max(1, x1 - x0)

        frame_rgb = video_np[int(np.clip(frame_idx, 0, total_frames - 1))]
        tile_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        tile_bgr = cv2.resize(tile_bgr, (curr_tile_w, curr_tile_h), interpolation=cv2.INTER_LINEAR)

        query_indices = np.where(t_src[:num_queries] == frame_idx)[0]
        for query_idx in query_indices:
            px = int(coords_input[query_idx, 0] * (curr_tile_w - 1))
            py = int(coords_input[query_idx, 1] * (curr_tile_h - 1))
            if 0 <= px < curr_tile_w and 0 <= py < curr_tile_h:
                color = tuple(int(channel) for channel in colors[query_idx])
                cv2.circle(tile_bgr, (px, py), 4, color, -1)
                cv2.circle(tile_bgr, (px, py), 2, (255, 255, 255), -1)

        label = f"src t={int(frame_idx)}  n={len(query_indices)}"
        cv2.putText(tile_bgr, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        panel_bgr[y0:y1, x0:x1] = tile_bgr

    if omitted > 0:
        footer = f"+ {omitted} more source frames"
        cv2.putText(panel_bgr, footer, (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

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
    clip_source_frame = int(source_frame)
    abs_source_frame = int(frame_indices[clip_source_frame])
    source_xy = trajs_2d_all[abs_source_frame]

    x0, y0 = crop_offset_xy
    crop_h, crop_w = crop_size_hw
    crop_x = source_xy[:, 0] - x0
    crop_y = source_xy[:, 1] - y0
    in_bounds = (
        np.isfinite(crop_x)
        & np.isfinite(crop_y)
        & (crop_x >= 0.0)
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
            f"No visible query points on clip source frame {clip_source_frame} "
            f"(absolute frame {abs_source_frame})."
        )

    if len(candidates) <= max_queries:
        return candidates

    take_idx = np.floor(
        np.linspace(0, len(candidates), num=max_queries, endpoint=False)
    ).astype(np.int64)
    return candidates[take_idx]


def get_panel_frame(panel_np, frame_idx, fallback_frame_rgb):
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
    gt_visibility=None,
    input_visibility=None,
    input_title="Input",
    output_path=None,
):
    require_visualization_dependencies()
    total_frames, height, width, _ = video_np.shape
    num_queries = coords_2d_pred.shape[0]

    out_width = 3 * width
    out_height = height
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, 10.0, (out_width, out_height))

    cmap = matplotlib.colormaps.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, num_queries))[:, :3] * 255

    for frame_idx in tqdm(range(total_frames), desc="Rendering video"):
        frame_bgr = cv2.cvtColor(video_np[frame_idx].copy(), cv2.COLOR_RGB2BGR)
        input_base = get_panel_frame(input_panel_np, frame_idx, video_np[frame_idx])
        input_bgr = cv2.cvtColor(input_base.copy(), cv2.COLOR_RGB2BGR)

        frame_gt = frame_bgr.copy()
        frame_input = input_bgr
        frame_pred = frame_bgr.copy()

        cv2.putText(frame_gt, "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_input, input_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame_pred, "Pred", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        for query_idx in range(num_queries):
            color = tuple(int(channel) for channel in colors[query_idx])

            gt_is_visible = True if gt_visibility is None else bool(gt_visibility[query_idx, frame_idx])
            if gt_is_visible:
                gt_x = int(coords_2d_gt[query_idx, frame_idx, 0])
                gt_y = int(coords_2d_gt[query_idx, frame_idx, 1])
                if 0 <= gt_x < width and 0 <= gt_y < height:
                    cv2.circle(frame_gt, (gt_x, gt_y), 4, color, -1)
                    cv2.circle(frame_gt, (gt_x, gt_y), 2, (255, 255, 255), -1)

            pred_x = int(coords_2d_pred[query_idx, frame_idx, 0])
            pred_y = int(coords_2d_pred[query_idx, frame_idx, 1])
            if 0 <= pred_x < width and 0 <= pred_y < height:
                cv2.circle(frame_pred, (pred_x, pred_y), 4, color, -1)
                cv2.circle(frame_pred, (pred_x, pred_y), 2, (255, 255, 255), -1)

            if coords_input is not None:
                if coords_input.ndim == 2:
                    input_xy = coords_input[query_idx]
                    input_is_visible = True
                else:
                    input_xy = coords_input[query_idx, frame_idx]
                    input_is_visible = input_visibility is None or bool(input_visibility[query_idx, frame_idx])
                in_x = int(input_xy[0])
                in_y = int(input_xy[1])
                if input_is_visible and 0 <= in_x < width and 0 <= in_y < height:
                    cv2.circle(frame_input, (in_x, in_y), 4, color, -1)
                    cv2.circle(frame_input, (in_x, in_y), 2, (255, 255, 255), -1)

        combined = np.hstack([frame_gt, frame_input, frame_pred])
        if writer is not None:
            writer.write(combined)

    if writer is not None:
        writer.release()


def resolve_sampled_patch_provider(patch_provider: str) -> str:
    if patch_provider in {"auto", "precomputed_resized"}:
        return "sampled_resized"
    if patch_provider == "precomputed_highres":
        return "sampled_highres"
    return patch_provider


def colorize_scalar_map(values, *, valid_mask=None, vmin=None, vmax=None, cmap_name="viridis"):
    require_visualization_dependencies()
    values = np.asarray(values, dtype=np.float32)
    if valid_mask is None:
        valid_mask = np.isfinite(values)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)

    if not np.any(valid_mask):
        return np.zeros((*values.shape, 3), dtype=np.uint8), 0.0, 1.0

    valid_values = values[valid_mask]
    if vmin is None:
        vmin = float(np.percentile(valid_values, 2.0))
    if vmax is None:
        vmax = float(np.percentile(valid_values, 98.0))
    if not np.isfinite(vmin):
        vmin = float(valid_values.min())
    if not np.isfinite(vmax):
        vmax = float(valid_values.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm = np.clip((values - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    colored = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    colored[~valid_mask] = 0
    return colored, float(vmin), float(vmax)


def add_panel_text(panel_bgr, title, subtitle=None, title_color=(255, 255, 255)):
    cv2.putText(panel_bgr, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, title_color, 2)
    if subtitle:
        cv2.putText(panel_bgr, subtitle, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    return panel_bgr


def draw_query_points(panel_bgr, coords_2d_px, colors_rgb, visible_mask=None, radius=4):
    height, width = panel_bgr.shape[:2]
    num_queries = coords_2d_px.shape[0]
    for query_idx in range(num_queries):
        if visible_mask is not None and not bool(visible_mask[query_idx]):
            continue
        x = int(coords_2d_px[query_idx, 0])
        y = int(coords_2d_px[query_idx, 1])
        if 0 <= x < width and 0 <= y < height:
            color = tuple(int(channel) for channel in colors_rgb[query_idx])
            cv2.circle(panel_bgr, (x, y), radius, color, -1)
            cv2.circle(panel_bgr, (x, y), max(1, radius // 2), (255, 255, 255), -1)
    return panel_bgr


def make_track_panel(frame_rgb, coords_2d_px, colors_rgb, title, visible_mask=None, subtitle=None):
    panel_bgr = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    draw_query_points(panel_bgr, coords_2d_px, colors_rgb, visible_mask=visible_mask)
    return add_panel_text(panel_bgr, title, subtitle=subtitle)


def make_visibility_panel(
    frame_rgb,
    coords_2d_px,
    pred_visibility,
    *,
    gt_visibility=None,
    threshold=0.5,
):
    panel_bgr = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    vis_scores = np.clip(np.asarray(pred_visibility, dtype=np.float32), 0.0, 1.0)
    colors_rgb = (matplotlib.colormaps.get_cmap("RdYlGn")(vis_scores)[..., :3] * 255).astype(np.uint8)
    height, width = panel_bgr.shape[:2]

    mismatch_count = 0
    for query_idx in range(coords_2d_px.shape[0]):
        x = int(coords_2d_px[query_idx, 0])
        y = int(coords_2d_px[query_idx, 1])
        if not (0 <= x < width and 0 <= y < height):
            continue

        color = tuple(int(channel) for channel in colors_rgb[query_idx])
        is_pred_visible = bool(vis_scores[query_idx] >= threshold)
        cv2.circle(panel_bgr, (x, y), 5 if is_pred_visible else 3, color, -1)
        cv2.circle(panel_bgr, (x, y), 2, (255, 255, 255), -1)

        if gt_visibility is not None:
            gt_visible = bool(gt_visibility[query_idx])
            if is_pred_visible != gt_visible:
                mismatch_count += 1
                cv2.circle(panel_bgr, (x, y), 7, (255, 0, 255), 1)

    subtitle = f"mean={vis_scores.mean():.3f}"
    if gt_visibility is not None and len(gt_visibility) > 0:
        gt_visibility = np.asarray(gt_visibility, dtype=bool)
        acc = float(((vis_scores >= threshold) == gt_visibility).mean())
        subtitle += f"  acc@{threshold:.2f}={acc:.3f}  mismatch={mismatch_count}"

    return add_panel_text(panel_bgr, "Pred Visibility", subtitle=subtitle)


def make_error_panel(
    frame_rgb,
    pred_tracks_px,
    gt_tracks_px,
    gt_visibility,
    *,
    error_cap_px=16.0,
    title="2D Error",
):
    panel_bgr = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    gt_visibility = np.asarray(gt_visibility, dtype=bool)
    errors_px = np.linalg.norm(pred_tracks_px - gt_tracks_px, axis=-1)
    colors_rgb = (
        matplotlib.colormaps.get_cmap("turbo")(np.clip(errors_px / max(error_cap_px, 1e-6), 0.0, 1.0))[..., :3] * 255
    ).astype(np.uint8)
    height, width = panel_bgr.shape[:2]

    if np.any(gt_visibility):
        visible_errors = errors_px[gt_visibility]
        subtitle = (
            f"mean={visible_errors.mean():.2f}px  "
            f"median={np.median(visible_errors):.2f}px  cap={error_cap_px:.1f}"
        )
    else:
        subtitle = "no visible GT points"

    for query_idx in range(pred_tracks_px.shape[0]):
        if not gt_visibility[query_idx]:
            continue
        pred_x = int(pred_tracks_px[query_idx, 0])
        pred_y = int(pred_tracks_px[query_idx, 1])
        gt_x = int(gt_tracks_px[query_idx, 0])
        gt_y = int(gt_tracks_px[query_idx, 1])
        if not (0 <= pred_x < width and 0 <= pred_y < height and 0 <= gt_x < width and 0 <= gt_y < height):
            continue
        color = tuple(int(channel) for channel in colors_rgb[query_idx])
        cv2.line(panel_bgr, (gt_x, gt_y), (pred_x, pred_y), color, 1, cv2.LINE_AA)
        cv2.circle(panel_bgr, (gt_x, gt_y), 2, (255, 255, 255), 1)
        cv2.circle(panel_bgr, (pred_x, pred_y), 4, color, -1)

    return add_panel_text(panel_bgr, title, subtitle=subtitle)


def save_diagnostic_video(
    *,
    video_np,
    gt_tracks_px,
    pred_tracks_px,
    gt_visibility,
    pred_visibility,
    output_path,
    error_cap_px=16.0,
    visibility_threshold=0.5,
):
    require_visualization_dependencies()
    total_frames, height, width, _ = video_np.shape
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (2 * width, 2 * height),
    )

    colors_rgb = (matplotlib.colormaps.get_cmap("tab10")(np.linspace(0, 1, pred_tracks_px.shape[0]))[:, :3] * 255)

    for frame_idx in tqdm(range(total_frames), desc="Rendering diagnostic video"):
        gt_panel = make_track_panel(
            video_np[frame_idx],
            gt_tracks_px[:, frame_idx, :],
            colors_rgb,
            "GT Tracks",
            visible_mask=gt_visibility[:, frame_idx],
            subtitle=f"frame={frame_idx}",
        )
        pred_panel = make_track_panel(
            video_np[frame_idx],
            pred_tracks_px[:, frame_idx, :],
            colors_rgb,
            "Pred Tracks",
            subtitle=f"frame={frame_idx}",
        )
        visibility_panel = make_visibility_panel(
            video_np[frame_idx],
            pred_tracks_px[:, frame_idx, :],
            pred_visibility[:, frame_idx],
            gt_visibility=gt_visibility[:, frame_idx],
            threshold=visibility_threshold,
        )
        error_panel = make_error_panel(
            video_np[frame_idx],
            pred_tracks_px[:, frame_idx, :],
            gt_tracks_px[:, frame_idx, :],
            gt_visibility[:, frame_idx],
            error_cap_px=error_cap_px,
        )

        combined = np.vstack([np.hstack([gt_panel, pred_panel]), np.hstack([visibility_panel, error_panel])])
        writer.write(combined)

    writer.release()


def project_predicted_3d_to_pixels(pred_points_3d, intrinsics_by_frame, t_cam_indices, depth_eps=1e-6):
    pred_points_3d = np.asarray(pred_points_3d, dtype=np.float32)
    intrinsics_by_frame = np.asarray(intrinsics_by_frame, dtype=np.float32)
    t_cam_indices = np.asarray(t_cam_indices, dtype=np.int64)

    num_queries, num_frames, _ = pred_points_3d.shape
    flat_points = pred_points_3d.reshape(-1, 3)
    flat_cam_indices = np.clip(t_cam_indices.reshape(-1), 0, intrinsics_by_frame.shape[0] - 1)
    flat_intrinsics = intrinsics_by_frame[flat_cam_indices]

    x = flat_points[:, 0]
    y = flat_points[:, 1]
    z = flat_points[:, 2]
    fx = flat_intrinsics[:, 0, 0]
    fy = flat_intrinsics[:, 1, 1]
    cx = flat_intrinsics[:, 0, 2]
    cy = flat_intrinsics[:, 1, 2]

    valid = (
        np.isfinite(flat_points).all(axis=-1)
        & np.isfinite(flat_intrinsics).all(axis=(1, 2))
        & (z > depth_eps)
        & (fx > 0.0)
        & (fy > 0.0)
    )

    z_safe = np.where(valid, z, 1.0)
    u = fx * (x / z_safe) + cx
    v = fy * (y / z_safe) + cy
    proj = np.stack([u, v], axis=-1).astype(np.float32)
    proj[~valid] = 0.0

    return proj.reshape(num_queries, num_frames, 2), valid.reshape(num_queries, num_frames)


def save_reprojection_comparison_video(
    *,
    video_np,
    gt_tracks_px,
    head_tracks_px,
    reproj_tracks_px,
    gt_visibility,
    reproj_comparable_mask,
    output_path,
    error_cap_px=16.0,
):
    require_visualization_dependencies()
    total_frames, height, width, _ = video_np.shape
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (2 * width, 2 * height),
    )

    colors_rgb = (matplotlib.colormaps.get_cmap("tab10")(np.linspace(0, 1, head_tracks_px.shape[0]))[:, :3] * 255)

    for frame_idx in tqdm(range(total_frames), desc="Rendering reprojection video"):
        comparable_mask = reproj_comparable_mask[:, frame_idx]
        gt_panel = make_track_panel(
            video_np[frame_idx],
            gt_tracks_px[:, frame_idx, :],
            colors_rgb,
            "GT Tracks",
            visible_mask=gt_visibility[:, frame_idx],
            subtitle=f"frame={frame_idx}",
        )
        head_panel = make_track_panel(
            video_np[frame_idx],
            head_tracks_px[:, frame_idx, :],
            colors_rgb,
            "Pred 2D Head",
            subtitle=f"frame={frame_idx}",
        )
        reproj_panel = make_track_panel(
            video_np[frame_idx],
            reproj_tracks_px[:, frame_idx, :],
            colors_rgb,
            "Reproj 3D Head",
            visible_mask=comparable_mask,
            subtitle=f"gt-comparable={int(comparable_mask.sum())}/{head_tracks_px.shape[0]}",
        )
        reproj_error_panel = make_error_panel(
            video_np[frame_idx],
            reproj_tracks_px[:, frame_idx, :],
            gt_tracks_px[:, frame_idx, :],
            gt_visibility[:, frame_idx] & comparable_mask,
            error_cap_px=error_cap_px,
            title="Reproj3D vs GT",
        )

        combined = np.vstack([np.hstack([gt_panel, head_panel]), np.hstack([reproj_panel, reproj_error_panel])])
        writer.write(combined)

    writer.release()


def predict_dense_depth_maps(
    *,
    model,
    encoder_features,
    frames,
    transform_metadata,
    num_frames,
    height,
    width,
    device,
    input_dtype,
    chunk_size,
):
    grid_u = torch.linspace(0.0, 1.0, width, device=device, dtype=input_dtype)
    grid_v = torch.linspace(0.0, 1.0, height, device=device, dtype=input_dtype)
    mesh_u, mesh_v = torch.meshgrid(grid_u, grid_v, indexing="xy")
    coords = torch.stack([mesh_u, mesh_v], dim=-1).reshape(1, -1, 2)
    num_queries = coords.shape[1]
    depth_frames = []

    if hasattr(model, "module"):
        decoder = model.module.decoder
    else:
        decoder = model.decoder

    original_patch_provider = decoder.patch_provider
    decoder.patch_provider = resolve_sampled_patch_provider(original_patch_provider)

    try:
        for frame_idx in tqdm(range(num_frames), desc="Predicting dense depth"):
            depth_chunks = []
            for start in range(0, num_queries, chunk_size):
                end = min(start + chunk_size, num_queries)
                curr_coords = coords[:, start:end]
                curr_t = torch.full((1, end - start), frame_idx, device=device, dtype=torch.long)
                with torch.no_grad():
                    with inference_autocast_context(device):
                        curr_outputs = model.decode(
                            encoder_features=encoder_features,
                            frames=frames,
                            coords=curr_coords,
                            t_src=curr_t,
                            t_tgt=curr_t,
                            t_cam=curr_t,
                            local_patches=None,
                            transform_metadata=transform_metadata,
                        )
                depth_chunks.append(curr_outputs["pos_3d"][0, :, 2].float().cpu())
            depth_frame = torch.cat(depth_chunks, dim=0).reshape(height, width)
            depth_frames.append(depth_frame)
    finally:
        decoder.patch_provider = original_patch_provider

    return torch.stack(depth_frames, dim=0).numpy()


def save_depth_comparison_video(
    *,
    video_np,
    pred_depths,
    gt_depths=None,
    output_path,
):
    require_visualization_dependencies()
    total_frames, height, width, _ = video_np.shape
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (2 * width, 2 * height),
    )

    pred_valid = np.isfinite(pred_depths)
    gt_valid = None if gt_depths is None else (np.isfinite(gt_depths) & (gt_depths > 0.0))

    scale_values = pred_depths[pred_valid]
    if gt_valid is not None and np.any(gt_valid):
        scale_values = np.concatenate([scale_values, gt_depths[gt_valid]])
    if scale_values.size == 0:
        depth_vmin, depth_vmax = 0.0, 1.0
    else:
        depth_vmin = float(np.percentile(scale_values, 2.0))
        depth_vmax = float(np.percentile(scale_values, 98.0))
        if depth_vmax <= depth_vmin:
            depth_vmax = depth_vmin + 1e-6

    if gt_depths is not None:
        depth_error = np.abs(pred_depths - gt_depths)
        error_valid = pred_valid & gt_valid
        if np.any(error_valid):
            error_vmax = float(np.percentile(depth_error[error_valid], 98.0))
            if error_vmax <= 0.0:
                error_vmax = 1.0
        else:
            error_vmax = 1.0
    else:
        depth_error = None
        error_valid = None
        error_vmax = 1.0

    for frame_idx in tqdm(range(total_frames), desc="Rendering depth video"):
        rgb_panel = add_panel_text(
            cv2.cvtColor(video_np[frame_idx].copy(), cv2.COLOR_RGB2BGR),
            "RGB",
            subtitle=f"frame={frame_idx}",
        )

        pred_depth_rgb, _, _ = colorize_scalar_map(
            pred_depths[frame_idx],
            valid_mask=pred_valid[frame_idx],
            vmin=depth_vmin,
            vmax=depth_vmax,
            cmap_name="magma",
        )
        pred_depth_panel = add_panel_text(
            cv2.cvtColor(pred_depth_rgb, cv2.COLOR_RGB2BGR),
            "Pred Depth",
            subtitle=f"range=[{depth_vmin:.3f}, {depth_vmax:.3f}]",
        )

        if gt_depths is not None:
            gt_depth_rgb, _, _ = colorize_scalar_map(
                gt_depths[frame_idx],
                valid_mask=gt_valid[frame_idx],
                vmin=depth_vmin,
                vmax=depth_vmax,
                cmap_name="magma",
            )
            gt_depth_panel = add_panel_text(
                cv2.cvtColor(gt_depth_rgb, cv2.COLOR_RGB2BGR),
                "GT Depth",
            )
            depth_error_rgb, _, _ = colorize_scalar_map(
                depth_error[frame_idx],
                valid_mask=error_valid[frame_idx],
                vmin=0.0,
                vmax=error_vmax,
                cmap_name="inferno",
            )
            error_panel = add_panel_text(
                cv2.cvtColor(depth_error_rgb, cv2.COLOR_RGB2BGR),
                "Depth Abs Error",
                subtitle=f"p98={error_vmax:.3f}",
            )
        else:
            gt_depth_panel = add_panel_text(
                np.zeros((height, width, 3), dtype=np.uint8),
                "GT Depth",
                subtitle="not available",
            )
            error_panel = add_panel_text(
                np.zeros((height, width, 3), dtype=np.uint8),
                "Depth Abs Error",
                subtitle="not available",
            )

        combined = np.vstack([np.hstack([rgb_panel, gt_depth_panel]), np.hstack([pred_depth_panel, error_panel])])
        writer.write(combined)

    writer.release()


def predict_dense_world_points(
    *,
    model,
    encoder_features,
    frames,
    transform_metadata,
    extrinsics_w2c,
    video_rgb,
    num_frames,
    height,
    width,
    device,
    input_dtype,
    stride,
    chunk_size,
):
    stride = max(1, int(stride))
    x_pixels = torch.arange(0, width, stride, device=device, dtype=input_dtype)
    y_pixels = torch.arange(0, height, stride, device=device, dtype=input_dtype)
    if x_pixels.numel() == 0 or y_pixels.numel() == 0:
        raise ValueError(f"Invalid point-cloud stride {stride} for image size {(height, width)}")

    grid_u, grid_v = torch.meshgrid(
        x_pixels / max(width - 1, 1),
        y_pixels / max(height - 1, 1),
        indexing="xy",
    )
    coords = torch.stack([grid_u, grid_v], dim=-1).reshape(1, -1, 2)
    pixel_xy = torch.stack([grid_u, grid_v], dim=-1).reshape(-1, 2)
    pixel_xy[:, 0] *= (width - 1)
    pixel_xy[:, 1] *= (height - 1)
    num_queries = coords.shape[1]

    colors = []
    x_idx_np = np.clip(np.round(pixel_xy[:, 0].cpu().numpy()).astype(np.int32), 0, width - 1)
    y_idx_np = np.clip(np.round(pixel_xy[:, 1].cpu().numpy()).astype(np.int32), 0, height - 1)
    for frame_idx in range(num_frames):
        colors.append(video_rgb[frame_idx, y_idx_np, x_idx_np, :].copy())
    colors = np.stack(colors, axis=0)

    if hasattr(model, "module"):
        decoder = model.module.decoder
    else:
        decoder = model.decoder

    original_patch_provider = decoder.patch_provider
    decoder.patch_provider = resolve_sampled_patch_provider(original_patch_provider)

    world_points = []
    visibility_probs = []
    c2w_all = np.linalg.inv(extrinsics_w2c.astype(np.float64)).astype(np.float32)

    try:
        for frame_idx in tqdm(range(num_frames), desc="Predicting world point clouds"):
            point_chunks = []
            visibility_chunks = []
            for start in range(0, num_queries, chunk_size):
                end = min(start + chunk_size, num_queries)
                curr_coords = coords[:, start:end]
                curr_t = torch.full((1, end - start), frame_idx, device=device, dtype=torch.long)
                with torch.no_grad():
                    with inference_autocast_context(device):
                        curr_outputs = model.decode(
                            encoder_features=encoder_features,
                            frames=frames,
                            coords=curr_coords,
                            t_src=curr_t,
                            t_tgt=curr_t,
                            t_cam=curr_t,
                            local_patches=None,
                            transform_metadata=transform_metadata,
                        )
                point_chunks.append(curr_outputs["pos_3d"].squeeze(0).float().cpu())
                visibility_chunks.append(
                    torch.sigmoid(curr_outputs["visibility"].squeeze(0).squeeze(-1)).float().cpu()
                )

            cam_points = torch.cat(point_chunks, dim=0).numpy()
            vis_probs = torch.cat(visibility_chunks, dim=0).numpy()
            pts_cam_h = np.concatenate([cam_points, np.ones((cam_points.shape[0], 1), dtype=np.float32)], axis=-1)
            pts_world = (c2w_all[frame_idx] @ pts_cam_h.T).T[:, :3]

            valid = np.isfinite(cam_points).all(axis=-1) & np.isfinite(pts_world).all(axis=-1) & (cam_points[:, 2] > 1e-6)
            pts_world[~valid] = 0.0
            vis_probs = np.where(valid, vis_probs, 0.0)

            world_points.append(pts_world.astype(np.float32))
            visibility_probs.append(vis_probs.astype(np.float32))
    finally:
        decoder.patch_provider = original_patch_provider

    return {
        "world_points": np.stack(world_points, axis=0),
        "visibility": np.stack(visibility_probs, axis=0),
        "colors": colors.astype(np.uint8),
        "pixel_coords": pixel_xy.cpu().numpy().astype(np.float32),
        "num_points_per_frame": int(num_queries),
        "stride": stride,
    }


def predict_dense_reference_points(
    *,
    model,
    encoder_features,
    frames,
    transform_metadata,
    video_rgb,
    num_frames,
    height,
    width,
    reference_frame,
    device,
    input_dtype,
    stride,
    chunk_size,
):
    stride = max(1, int(stride))
    reference_frame = int(reference_frame)
    if not (0 <= reference_frame < int(num_frames)):
        raise ValueError(
            f"reference_frame must be in [0, {int(num_frames) - 1}], got {reference_frame}"
        )

    x_pixels = torch.arange(0, width, stride, device=device, dtype=input_dtype)
    y_pixels = torch.arange(0, height, stride, device=device, dtype=input_dtype)
    if x_pixels.numel() == 0 or y_pixels.numel() == 0:
        raise ValueError(f"Invalid point-cloud stride {stride} for image size {(height, width)}")

    grid_u, grid_v = torch.meshgrid(
        x_pixels / max(width - 1, 1),
        y_pixels / max(height - 1, 1),
        indexing="xy",
    )
    coords = torch.stack([grid_u, grid_v], dim=-1).reshape(1, -1, 2)
    pixel_xy = torch.stack([grid_u, grid_v], dim=-1).reshape(-1, 2)
    pixel_xy[:, 0] *= (width - 1)
    pixel_xy[:, 1] *= (height - 1)
    num_queries = coords.shape[1]

    colors = []
    x_idx_np = np.clip(np.round(pixel_xy[:, 0].cpu().numpy()).astype(np.int32), 0, width - 1)
    y_idx_np = np.clip(np.round(pixel_xy[:, 1].cpu().numpy()).astype(np.int32), 0, height - 1)
    for frame_idx in range(num_frames):
        colors.append(video_rgb[frame_idx, y_idx_np, x_idx_np, :].copy())
    colors = np.stack(colors, axis=0)

    if hasattr(model, "module"):
        decoder = model.module.decoder
    else:
        decoder = model.decoder

    original_patch_provider = decoder.patch_provider
    decoder.patch_provider = resolve_sampled_patch_provider(original_patch_provider)

    reference_points = []
    visibility_probs = []

    try:
        for frame_idx in tqdm(range(num_frames), desc="Predicting reference-frame point clouds"):
            point_chunks = []
            visibility_chunks = []
            for start in range(0, num_queries, chunk_size):
                end = min(start + chunk_size, num_queries)
                curr_coords = coords[:, start:end]
                curr_t = torch.full((1, end - start), frame_idx, device=device, dtype=torch.long)
                curr_t_cam = torch.full((1, end - start), reference_frame, device=device, dtype=torch.long)
                with torch.no_grad():
                    with inference_autocast_context(device):
                        curr_outputs = model.decode(
                            encoder_features=encoder_features,
                            frames=frames,
                            coords=curr_coords,
                            t_src=curr_t,
                            t_tgt=curr_t,
                            t_cam=curr_t_cam,
                            local_patches=None,
                            transform_metadata=transform_metadata,
                        )
                point_chunks.append(curr_outputs["pos_3d"].squeeze(0).float().cpu())
                visibility_chunks.append(
                    torch.sigmoid(curr_outputs["visibility"].squeeze(0).squeeze(-1)).float().cpu()
                )

            ref_points = torch.cat(point_chunks, dim=0).numpy()
            vis_probs = torch.cat(visibility_chunks, dim=0).numpy()
            valid = np.isfinite(ref_points).all(axis=-1) & (ref_points[:, 2] > 1e-6)
            ref_points[~valid] = 0.0
            vis_probs = np.where(valid, vis_probs, 0.0)

            reference_points.append(ref_points.astype(np.float32))
            visibility_probs.append(vis_probs.astype(np.float32))
    finally:
        decoder.patch_provider = original_patch_provider

    return {
        "reference_points": np.stack(reference_points, axis=0),
        "visibility": np.stack(visibility_probs, axis=0),
        "colors": colors.astype(np.uint8),
        "pixel_coords": pixel_xy.cpu().numpy().astype(np.float32),
        "num_points_per_frame": int(num_queries),
        "stride": stride,
        "reference_frame": reference_frame,
    }


def compute_world_bounds(world_points, visibility_probs, vis_threshold):
    valid_mask = (
        np.isfinite(world_points).all(axis=-1)
        & (visibility_probs >= vis_threshold)
    )
    if not np.any(valid_mask):
        return np.zeros(3, dtype=np.float32), 1.0

    pts = world_points[valid_mask]
    lower = np.percentile(pts, 2.0, axis=0)
    upper = np.percentile(pts, 98.0, axis=0)
    center = ((lower + upper) * 0.5).astype(np.float32)
    radius = float(max(np.max(upper - lower) * 0.55, 1e-3))
    return center, radius


def voxel_downsample_point_cloud(points, colors, visibility, voxel_size):
    if len(points) == 0 or voxel_size is None or voxel_size <= 0.0:
        return points, colors, visibility

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    visibility = np.asarray(visibility, dtype=np.float32)

    min_corner = points.min(axis=0, keepdims=True)
    voxel_coords = np.floor((points - min_corner) / float(voxel_size)).astype(np.int64)
    _, inverse, counts = np.unique(voxel_coords, axis=0, return_inverse=True, return_counts=True)
    num_voxels = int(counts.shape[0])

    point_sums = np.zeros((num_voxels, 3), dtype=np.float64)
    color_sums = np.zeros((num_voxels, 3), dtype=np.float64)
    vis_sums = np.zeros((num_voxels,), dtype=np.float64)
    np.add.at(point_sums, inverse, points)
    np.add.at(color_sums, inverse, colors)
    np.add.at(vis_sums, inverse, visibility)

    voxel_points = (point_sums / counts[:, None]).astype(np.float32)
    voxel_colors = np.clip(color_sums / counts[:, None], 0.0, 255.0).astype(np.uint8)
    voxel_visibility = (vis_sums / counts).astype(np.float32)
    return voxel_points, voxel_colors, voxel_visibility


def collect_pointcloud_for_export(
    *,
    points,
    colors,
    visibility_probs,
    vis_threshold,
    uniformization_mode,
    voxel_size,
):
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    visibility_probs = np.asarray(visibility_probs, dtype=np.float32)

    per_frame = []
    merged_points = []
    merged_colors = []

    for frame_idx in range(points.shape[0]):
        valid_mask = (
            np.isfinite(points[frame_idx]).all(axis=-1)
            & (visibility_probs[frame_idx] >= float(vis_threshold))
        )
        pts = points[frame_idx][valid_mask]
        cols = colors[frame_idx][valid_mask]
        vis = visibility_probs[frame_idx][valid_mask]

        if uniformization_mode == "voxel":
            pts, cols, vis = voxel_downsample_point_cloud(pts, cols, vis, voxel_size)

        frame_entry = {
            "points": pts.astype(np.float32),
            "colors": cols.astype(np.uint8),
            "visibility": vis.astype(np.float32),
        }
        per_frame.append(frame_entry)

        if len(pts) > 0:
            merged_points.append(frame_entry["points"])
            merged_colors.append(frame_entry["colors"])

    if merged_points:
        merged_points = np.concatenate(merged_points, axis=0)
        merged_colors = np.concatenate(merged_colors, axis=0)
        if uniformization_mode == "voxel":
            merged_points, merged_colors, _ = voxel_downsample_point_cloud(
                merged_points,
                merged_colors,
                np.ones((merged_points.shape[0],), dtype=np.float32),
                voxel_size,
            )
    else:
        merged_points = np.zeros((0, 3), dtype=np.float32)
        merged_colors = np.zeros((0, 3), dtype=np.uint8)

    return {
        "per_frame": per_frame,
        "merged_points": merged_points.astype(np.float32),
        "merged_colors": merged_colors.astype(np.uint8),
    }


def prepare_world_pointcloud_render_data(
    *,
    world_points,
    colors,
    visibility_probs,
    max_render_points,
    vis_threshold,
    uniformization_mode,
    voxel_size,
):
    valid_mask = (
        np.isfinite(world_points).all(axis=-1)
        & np.isfinite(visibility_probs)
        & (visibility_probs >= vis_threshold)
    )

    pts = np.asarray(world_points[valid_mask], dtype=np.float32)
    cols = np.asarray(colors[valid_mask], dtype=np.uint8)
    vis = np.asarray(visibility_probs[valid_mask], dtype=np.float32)

    if uniformization_mode == "voxel":
        pts, cols, vis = voxel_downsample_point_cloud(pts, cols, vis, voxel_size=voxel_size)

    if max_render_points is not None and max_render_points > 0 and len(pts) > max_render_points:
        try:
            keep_idx = farthest_point_sample_py(pts, int(max_render_points))
        except Exception:
            keep_idx = np.floor(
                np.linspace(0, len(pts), num=max_render_points, endpoint=False)
            ).astype(np.int64)
        pts = pts[keep_idx]
        cols = cols[keep_idx]
        vis = vis[keep_idx]

    return pts, cols, vis


def render_world_pointcloud_panel(
    *,
    points,
    colors,
    visibility_probs,
    center,
    radius,
    frame_idx,
    elev,
    azim,
    point_size,
    panel_label=None,
):
    import matplotlib.pyplot as plt

    pts = np.asarray(points, dtype=np.float32)
    cols = np.asarray(colors, dtype=np.uint8)
    vis = np.asarray(visibility_probs, dtype=np.float32)

    fig = plt.figure(figsize=(6, 6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    if len(pts) > 0:
        alpha = np.clip(0.15 + 0.85 * vis, 0.15, 1.0)
        rgba = np.concatenate([cols.astype(np.float32) / 255.0, alpha[:, None]], axis=-1)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=rgba,
            s=point_size,
            depthshade=False,
            linewidths=0.0,
        )

    ax.view_init(elev=float(elev), azim=float(azim))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_axis_off()
    title = f"World Point Cloud | frame={frame_idx} | points={len(pts)}"
    if panel_label:
        title += f" | {panel_label}"
    ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.0)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb, int(len(pts))


def render_world_pointcloud_frame(
    *,
    world_points,
    colors,
    visibility_probs,
    center,
    radius,
    frame_idx,
    elev,
    azim,
    point_size,
    max_render_points,
    vis_threshold,
    uniformization_mode,
    voxel_size,
    panel_label=None,
):
    pts, cols, vis = prepare_world_pointcloud_render_data(
        world_points=world_points,
        colors=colors,
        visibility_probs=visibility_probs,
        max_render_points=max_render_points,
        vis_threshold=vis_threshold,
        uniformization_mode=uniformization_mode,
        voxel_size=voxel_size,
    )
    return render_world_pointcloud_panel(
        points=pts,
        colors=cols,
        visibility_probs=vis,
        center=center,
        radius=radius,
        frame_idx=frame_idx,
        elev=elev,
        azim=azim,
        point_size=point_size,
        panel_label=panel_label,
    )


def save_world_pointcloud_gif(
    *,
    world_points,
    colors,
    visibility_probs,
    output_path,
    fps,
    vis_threshold,
    elev,
    azim,
    rotate_azim_per_frame,
    point_size,
    max_render_points,
    uniformization_mode,
    voxel_size,
):
    require_visualization_dependencies()
    require_gif_dependency()

    center, radius = compute_world_bounds(world_points, visibility_probs, vis_threshold)
    gif_frames = []
    preview_rgb = None
    rendered_counts = []

    for frame_idx in tqdm(range(world_points.shape[0]), desc="Rendering world point-cloud GIF"):
        pts, cols, vis = prepare_world_pointcloud_render_data(
            world_points=world_points[frame_idx],
            colors=colors[frame_idx],
            visibility_probs=visibility_probs[frame_idx],
            max_render_points=max_render_points,
            vis_threshold=vis_threshold,
            uniformization_mode=uniformization_mode,
            voxel_size=voxel_size,
        )
        rgb, rendered_count = render_world_pointcloud_panel(
            points=pts,
            colors=cols,
            visibility_probs=vis,
            center=center,
            radius=radius,
            frame_idx=frame_idx,
            elev=elev,
            azim=azim + rotate_azim_per_frame * frame_idx,
            point_size=point_size,
            panel_label=None,
        )
        if preview_rgb is None:
            preview_rgb = rgb
        gif_frames.append(Image.fromarray(rgb))
        rendered_counts.append(rendered_count)

    if not gif_frames:
        raise RuntimeError("No frames available for point-cloud GIF")

    duration_ms = max(1, int(round(1000.0 / max(float(fps), 1e-6))))
    gif_frames[0].save(
        str(output_path),
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return preview_rgb, center, radius, rendered_counts


def combine_rgb_panels(panels_rgb, *, fill_value=255):
    if not panels_rgb:
        raise ValueError("Need at least one panel to combine")

    panel_h, panel_w = panels_rgb[0].shape[:2]
    num_panels = len(panels_rgb)
    cols = 2 if num_panels <= 4 else 3
    rows = int(np.ceil(num_panels / cols))
    canvas = np.full((rows * panel_h, cols * panel_w, 3), fill_value, dtype=np.uint8)

    for idx, panel in enumerate(panels_rgb):
        row = idx // cols
        col = idx % cols
        y0 = row * panel_h
        x0 = col * panel_w
        canvas[y0:y0 + panel_h, x0:x0 + panel_w] = panel

    return canvas


def save_world_pointcloud_multiview_gif(
    *,
    world_points,
    colors,
    visibility_probs,
    output_path,
    fps,
    vis_threshold,
    elev,
    azims,
    rotate_azim_per_frame,
    point_size,
    max_render_points,
    uniformization_mode,
    voxel_size,
):
    require_visualization_dependencies()
    require_gif_dependency()

    center, radius = compute_world_bounds(world_points, visibility_probs, vis_threshold)
    gif_frames = []
    preview_rgb = None
    rendered_counts = []

    for frame_idx in tqdm(range(world_points.shape[0]), desc="Rendering world point-cloud multiview GIF"):
        pts, cols, vis = prepare_world_pointcloud_render_data(
            world_points=world_points[frame_idx],
            colors=colors[frame_idx],
            visibility_probs=visibility_probs[frame_idx],
            max_render_points=max_render_points,
            vis_threshold=vis_threshold,
            uniformization_mode=uniformization_mode,
            voxel_size=voxel_size,
        )
        frame_panels = []
        frame_rendered_counts = []
        for base_azim in azims:
            panel_rgb, rendered_count = render_world_pointcloud_panel(
                points=pts,
                colors=cols,
                visibility_probs=vis,
                center=center,
                radius=radius,
                frame_idx=frame_idx,
                elev=elev,
                azim=base_azim + rotate_azim_per_frame * frame_idx,
                point_size=point_size,
                panel_label=f"azim={base_azim:.0f}",
            )
            frame_panels.append(panel_rgb)
            frame_rendered_counts.append(rendered_count)

        combined_rgb = combine_rgb_panels(frame_panels)
        if preview_rgb is None:
            preview_rgb = combined_rgb
        gif_frames.append(Image.fromarray(combined_rgb))
        rendered_counts.append(int(max(frame_rendered_counts) if frame_rendered_counts else 0))

    if not gif_frames:
        raise RuntimeError("No frames available for point-cloud multiview GIF")

    duration_ms = max(1, int(round(1000.0 / max(float(fps), 1e-6))))
    gif_frames[0].save(
        str(output_path),
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return preview_rgb, center, radius, rendered_counts


def run_video_only_pointcloud_visualization(args, model, ckpt_args, device):
    video_path = Path(args.video_path)
    video_stem = video_path.stem

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.checkpoint).resolve().parent / "visualize_tracks" / video_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_provider = args.patch_provider or ckpt_args.get("patch_provider", "auto")
    print(f"Running video-only point-cloud visualization for {video_path}")
    print(f"Using patch_provider: {patch_provider}")

    requested_frames = args.video_num_frames
    if requested_frames is None:
        requested_frames = int(ckpt_args.get("num_frames", 48))
    requested_frames = int(requested_frames)

    img_size = int(ckpt_args.get("img_size", 256))
    video_bundle = load_video_clip_for_visualization(
        video_path=video_path,
        start_frame=args.video_start_frame,
        num_frames=requested_frames,
        frame_stride=args.video_frame_stride,
        img_size=img_size,
    )

    viz_args = dict(vars(args))
    viz_args.update(
        {
            "resolved_dataset": "video",
            "resolved_sequence": video_stem,
            "resolved_video_path": str(video_path),
            "resolved_video_frame_indices": video_bundle["frame_indices"].tolist(),
        }
    )
    with open(output_dir / "viz_args.json", "w", encoding="utf-8") as f:
        json.dump(viz_args, f, indent=2, ensure_ascii=False)

    input_dtype = next(model.parameters()).dtype
    video = video_bundle["video"].unsqueeze(0).to(device=device, dtype=input_dtype)
    aspect_ratio = torch.tensor(
        [[float(video_bundle["original_width"]) / max(float(video_bundle["original_height"]), 1.0)]],
        device=device,
        dtype=input_dtype,
    )

    if hasattr(model, "module"):
        model.module.decoder.patch_provider = patch_provider
    else:
        model.decoder.patch_provider = patch_provider

    sampled_provider = resolve_sampled_patch_provider(patch_provider)
    if sampled_provider == "sampled_highres":
        query_frames_for_decode = video_bundle["video_query"].unsqueeze(0).to(device=device, dtype=input_dtype)
    else:
        query_frames_for_decode = video

    transform_metadata = build_identity_transform_metadata(
        original_h=video_bundle["original_height"],
        original_w=video_bundle["original_width"],
        resized_h=video_bundle["resized_height"],
        resized_w=video_bundle["resized_width"],
    )
    transform_metadata_input = {
        key: value.unsqueeze(0).to(device=device)
        for key, value in transform_metadata.items()
    }

    with torch.no_grad():
        with inference_autocast_context(device):
            encoder_features = model.encode(video, aspect_ratio=aspect_ratio)

    num_frames = int(video.shape[1])
    reference_frame = int(args.reference_camera_frame)
    if not (0 <= reference_frame < num_frames):
        raise ValueError(f"--reference-camera-frame must be in [0, {num_frames - 1}], got {reference_frame}")

    video_np = video_bundle["video_rgb"]
    height, width = video_np.shape[1:3]
    summary = {
        "dataset": "video",
        "sequence": video_stem,
        "video_path": str(video_path),
        "num_frames": num_frames,
        "frame_indices": video_bundle["frame_indices"].tolist(),
        "frame_stride": int(args.video_frame_stride),
        "source_video_fps": float(video_bundle["fps"]),
        "source_video_resolution": [
            int(video_bundle["original_height"]),
            int(video_bundle["original_width"]),
        ],
        "model_input_resolution": [int(height), int(width)],
        "reference_camera_frame": reference_frame,
        "coordinate_frame": f"reference_camera_{reference_frame}",
        "patch_provider": patch_provider,
    }

    if args.with_depth_video:
        pred_depths = predict_dense_depth_maps(
            model=model,
            encoder_features=encoder_features,
            frames=query_frames_for_decode,
            transform_metadata=transform_metadata_input,
            num_frames=num_frames,
            height=height,
            width=width,
            device=device,
            input_dtype=input_dtype,
            chunk_size=max(1, int(args.depth_chunk_size)),
        )
        depth_video_path = output_dir / f"{video_stem}_depth.mp4"
        save_depth_comparison_video(
            video_np=video_np,
            pred_depths=pred_depths,
            gt_depths=None,
            output_path=depth_video_path,
        )
        print(f"Saved video-only depth comparison to {depth_video_path}")

    pointcloud_result = predict_dense_reference_points(
        model=model,
        encoder_features=encoder_features,
        frames=query_frames_for_decode,
        transform_metadata=transform_metadata_input,
        video_rgb=video_np,
        num_frames=num_frames,
        height=height,
        width=width,
        reference_frame=reference_frame,
        device=device,
        input_dtype=input_dtype,
        stride=max(1, int(args.pointcloud_stride)),
        chunk_size=max(1, int(args.pointcloud_chunk_size)),
    )

    single_view_requested = args.with_world_pointcloud_gif or not args.with_world_pointcloud_multiview_gif
    if single_view_requested:
        pointcloud_gif_path = output_dir / f"{video_stem}_reference_pointcloud.gif"
        pointcloud_preview_rgb, pointcloud_center, pointcloud_radius, rendered_counts = save_world_pointcloud_gif(
            world_points=pointcloud_result["reference_points"],
            colors=pointcloud_result["colors"],
            visibility_probs=pointcloud_result["visibility"],
            output_path=pointcloud_gif_path,
            fps=args.pointcloud_fps,
            vis_threshold=args.pointcloud_vis_threshold,
            elev=args.pointcloud_elev,
            azim=args.pointcloud_azim,
            rotate_azim_per_frame=args.pointcloud_rotate_azim_per_frame,
            point_size=args.pointcloud_point_size,
            max_render_points=args.pointcloud_max_render_points,
            uniformization_mode=args.pointcloud_uniformization,
            voxel_size=args.pointcloud_voxel_size,
        )
        pointcloud_preview_path = output_dir / f"{video_stem}_reference_pointcloud_preview.png"
        cv2.imwrite(str(pointcloud_preview_path), cv2.cvtColor(pointcloud_preview_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved reference-frame point-cloud preview to {pointcloud_preview_path}")
        print(f"Saved reference-frame point-cloud GIF to {pointcloud_gif_path}")

        summary["reference_pointcloud_rendered_points_mean"] = float(np.mean(rendered_counts))
        summary["reference_pointcloud_rendered_points_min"] = int(np.min(rendered_counts))
        summary["reference_pointcloud_rendered_points_max"] = int(np.max(rendered_counts))
        summary["reference_pointcloud_bounds_center"] = [float(x) for x in pointcloud_center.tolist()]
        summary["reference_pointcloud_bounds_radius"] = float(pointcloud_radius)

    if args.with_world_pointcloud_multiview_gif:
        multiview_azims = [float(x.strip()) for x in args.pointcloud_view_azims.split(",") if x.strip()]
        if not multiview_azims:
            raise ValueError("--pointcloud-view-azims must contain at least one angle")

        multiview_gif_path = output_dir / f"{video_stem}_reference_pointcloud_multiview.gif"
        multiview_preview_rgb, multiview_center, multiview_radius, multiview_rendered_counts = (
            save_world_pointcloud_multiview_gif(
                world_points=pointcloud_result["reference_points"],
                colors=pointcloud_result["colors"],
                visibility_probs=pointcloud_result["visibility"],
                output_path=multiview_gif_path,
                fps=args.pointcloud_fps,
                vis_threshold=args.pointcloud_vis_threshold,
                elev=args.pointcloud_elev,
                azims=multiview_azims,
                rotate_azim_per_frame=args.pointcloud_rotate_azim_per_frame,
                point_size=args.pointcloud_point_size,
                max_render_points=args.pointcloud_max_render_points,
                uniformization_mode=args.pointcloud_uniformization,
                voxel_size=args.pointcloud_voxel_size,
            )
        )
        multiview_preview_path = output_dir / f"{video_stem}_reference_pointcloud_multiview_preview.png"
        cv2.imwrite(str(multiview_preview_path), cv2.cvtColor(multiview_preview_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved reference-frame multiview preview to {multiview_preview_path}")
        print(f"Saved reference-frame multiview GIF to {multiview_gif_path}")

        summary["reference_pointcloud_multiview_azims"] = multiview_azims
        summary["reference_pointcloud_multiview_rendered_points_mean"] = float(np.mean(multiview_rendered_counts))
        summary["reference_pointcloud_multiview_rendered_points_min"] = int(np.min(multiview_rendered_counts))
        summary["reference_pointcloud_multiview_rendered_points_max"] = int(np.max(multiview_rendered_counts))
        if "reference_pointcloud_bounds_center" not in summary:
            summary["reference_pointcloud_bounds_center"] = [float(x) for x in multiview_center.tolist()]
            summary["reference_pointcloud_bounds_radius"] = float(multiview_radius)

    export_data = collect_pointcloud_for_export(
        points=pointcloud_result["reference_points"],
        colors=pointcloud_result["colors"],
        visibility_probs=pointcloud_result["visibility"],
        vis_threshold=args.pointcloud_vis_threshold,
        uniformization_mode=args.pointcloud_uniformization,
        voxel_size=args.pointcloud_voxel_size,
    )
    merged_points = export_data["merged_points"]
    merged_colors = export_data["merged_colors"]
    merged_ply_path = output_dir / f"{video_stem}_reference_pointcloud_merged.ply"
    save_point_cloud_ply(str(merged_ply_path), merged_points, merged_colors)
    print(f"Saved merged reference-frame point cloud to {merged_ply_path}")

    if args.save_pointcloud_per_frame_ply:
        ply_dir = output_dir / f"{video_stem}_reference_pointcloud_frames"
        ply_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx, frame_data in enumerate(export_data["per_frame"]):
            frame_ply_path = ply_dir / f"frame_{frame_idx:04d}.ply"
            save_point_cloud_ply(str(frame_ply_path), frame_data["points"], frame_data["colors"])
        print(f"Saved per-frame reference point clouds to {ply_dir}")

    visible_mask = (
        np.isfinite(pointcloud_result["reference_points"]).all(axis=-1)
        & (pointcloud_result["visibility"] >= args.pointcloud_vis_threshold)
    )
    points_per_frame = visible_mask.sum(axis=1)
    summary["reference_pointcloud_stride"] = int(pointcloud_result["stride"])
    summary["reference_pointcloud_points_per_frame"] = int(pointcloud_result["num_points_per_frame"])
    summary["reference_pointcloud_visible_points_mean"] = float(points_per_frame.mean())
    summary["reference_pointcloud_visible_points_min"] = int(points_per_frame.min())
    summary["reference_pointcloud_visible_points_max"] = int(points_per_frame.max())
    summary["reference_pointcloud_uniformization"] = args.pointcloud_uniformization
    summary["reference_pointcloud_voxel_size"] = float(args.pointcloud_voxel_size)
    summary["reference_pointcloud_merged_points"] = int(merged_points.shape[0])

    print("Tracking summary:", json.dumps(summary, ensure_ascii=False))
    summary_path = output_dir / "tracking_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved tracking summary to {summary_path}")


def resolve_data_root(dataset_type, data_root, split):
    if dataset_type != "pointodyssey":
        return data_root
    data_root = os.path.abspath(data_root)
    if os.path.isdir(os.path.join(data_root, split)):
        return data_root
    if os.path.basename(data_root) == split:
        return os.path.dirname(data_root)
    return data_root


def is_kubric_scene_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "frames").is_dir()
        and (path / "depths").is_dir()
        and any(path.glob("*_trajs_2d.npy"))
        and any(path.glob("*_visibility.npy"))
        and any(path.glob("*_with_rank.npz"))
    )


def resolve_sequence_name(dataset_type: str, data_root: str, split: str | None, sequence: str | None) -> str:
    root_path = Path(data_root)

    if dataset_type == "kubric":
        if is_kubric_scene_dir(root_path):
            return root_path.name

        candidate_names = sorted(
            path.name for path in root_path.iterdir()
            if path.is_dir() and not path.name.startswith(".") and is_kubric_scene_dir(path)
        )
        if not candidate_names:
            raise FileNotFoundError(f"No valid Kubric scenes found in {data_root}")
        if sequence is None:
            if len(candidate_names) == 1:
                return candidate_names[0]
            preview = ", ".join(candidate_names[:8])
            raise ValueError(f"Pass --sequence explicitly. Example Kubric scenes: {preview}")

        if sequence in candidate_names:
            return sequence
        normalized = sequence.lstrip("0") or "0"
        matches = [name for name in candidate_names if (name.lstrip("0") or "0") == normalized]
        if len(matches) == 1:
            return matches[0]
        if sequence.isdigit():
            padded = sequence.zfill(4)
            if padded in candidate_names:
                return padded
        preview = ", ".join(candidate_names[:8])
        raise ValueError(f"Could not resolve Kubric sequence {sequence!r}. Example scenes: {preview}")

    if split is None:
        raise ValueError("PointOdyssey visualization requires a split")

    split_root = Path(data_root) / split
    if not split_root.is_dir():
        split_root = Path(data_root)

    candidate_names = sorted(path.name for path in split_root.iterdir() if path.is_dir())
    if not candidate_names:
        raise FileNotFoundError(f"No sequences found under {split_root}")
    if sequence is None:
        if len(candidate_names) == 1:
            return candidate_names[0]
        preview = ", ".join(candidate_names[:8])
        raise ValueError(f"Pass --sequence explicitly. Example sequences: {preview}")
    if sequence in candidate_names:
        return sequence
    preview = ", ".join(candidate_names[:8])
    raise ValueError(f"Could not resolve sequence {sequence!r}. Example sequences: {preview}")


def build_dataset_for_visualization(
    *,
    dataset_type: str,
    data_root: str,
    split: str,
    sequence: str,
    ckpt_args: dict,
    patch_provider: str,
    num_queries: int,
):
    precompute_local_patches = (
        not ckpt_args.get("disable_precompute_local_patches", False)
        and patch_provider not in {"sampled_resized", "sampled_highres"}
    )

    common_kwargs = dict(
        patch_size=ckpt_args.get("patch_size", 9),
        S=ckpt_args.get("num_frames", 48),
        img_size=ckpt_args.get("img_size", 256),
        num_queries=num_queries,
        use_augs=False,
        verbose=True,
        query_mode=ckpt_args.get("query_mode", "full"),
        precompute_local_patches=precompute_local_patches,
        return_query_video=patch_provider == "sampled_highres",
        static_scene_frame_idx=ckpt_args.get("static_scene_frame_idx"),
        t_tgt_eq_t_cam_ratio=ckpt_args.get("t_tgt_eq_t_cam_ratio", 0.4),
        use_motion_boundaries=not ckpt_args.get("disable_motion_boundary_oversampling", False),
        strides=[1],
    )

    # if dataset_type == "kubric":
    #     root_path = Path(data_root)
    #     if is_kubric_scene_dir(root_path):
    #         return KubricForD4RT(scene_dir=str(root_path), dset=split, **common_kwargs)
    #     return KubricForD4RT(scene_root=data_root, dset=split, scene_names=[sequence], **common_kwargs)

    return PointOdysseyDataset(
        dataset_location=data_root,
        dset=split,
        sequence_name=sequence,
        **common_kwargs,
    )


def load_raw_annotations(dataset_type: str, data_root: str, split: str, sequence: str):
    if dataset_type == "kubric":
        root_path = Path(data_root)
        scene_dir = root_path if is_kubric_scene_dir(root_path) else root_path / sequence
        trajs_2d_path = next(iter(sorted(scene_dir.glob("*_trajs_2d.npy"))), None)
        visibility_path = next(iter(sorted(scene_dir.glob("*_visibility.npy"))), None)
        if trajs_2d_path is None or visibility_path is None:
            raise FileNotFoundError(f"Missing Kubric annotations in {scene_dir}")
        trajs_2d_all = np.load(trajs_2d_path).astype(np.float32).transpose(1, 0, 2)
        visibs_all = np.load(visibility_path).astype(np.float32).T
        valids_all = visibs_all.copy()
        return trajs_2d_all, valids_all, visibs_all

    seq_root = Path(data_root) / split / sequence

    # Fast annotation cache (anno_fast/*.npy) takes priority over anno.npz
    fast_dir = seq_root / "anno_fast"
    fast_trajs = fast_dir / "trajs_2d.npy"
    if fast_trajs.exists():
        trajs_2d_all = np.load(fast_trajs).astype(np.float32)
        valids_path = fast_dir / "valids.npy"
        visibs_path = fast_dir / "visibs.npy"
        valids_all = np.load(valids_path).astype(np.float32) if valids_path.exists() else np.ones(trajs_2d_all.shape[:2], dtype=np.float32)
        visibs_all = np.load(visibs_path).astype(np.float32) if visibs_path.exists() else valids_all
        return trajs_2d_all, valids_all, visibs_all

    anno_path = seq_root / "anno.npz"
    if not anno_path.exists():
        npzs = sorted(seq_root.glob("*.npz"))
        if not npzs:
            raise FileNotFoundError(f"No annotation file found in {seq_root}")
        anno_path = npzs[0]

    anno = np.load(anno_path, allow_pickle=True)
    trajs_2d_all = anno["trajs_2d"].astype(np.float32)
    valids_all = anno["valids"].astype(np.float32)
    visibs_all = anno["visibs"].astype(np.float32) if "visibs" in anno else valids_all
    return trajs_2d_all, valids_all, visibs_all


def build_gt_tracks_01(
    *,
    trajs_2d_all,
    valids_all,
    visibs_all,
    frame_indices,
    point_indices,
    transform_metadata,
):
    crop_offset_xy = transform_metadata["crop_offset_xy"].cpu().numpy()
    crop_size_hw = transform_metadata["crop_size_hw"].cpu().numpy()
    x0, y0 = float(crop_offset_xy[0]), float(crop_offset_xy[1])
    crop_h, crop_w = float(crop_size_hw[0]), float(crop_size_hw[1])

    num_queries = len(point_indices)
    num_frames = len(frame_indices)
    tracks_01 = np.zeros((num_queries, num_frames, 2), dtype=np.float32)
    visibility = np.zeros((num_queries, num_frames), dtype=bool)

    for query_idx, point_idx in enumerate(point_indices):
        gt_traj_raw = trajs_2d_all[frame_indices, point_idx, :]
        gt_traj_crop_x = gt_traj_raw[:, 0] - x0
        gt_traj_crop_y = gt_traj_raw[:, 1] - y0
        gt_in_bounds = (
            np.isfinite(gt_traj_crop_x)
            & np.isfinite(gt_traj_crop_y)
            & (gt_traj_crop_x >= 0.0)
            & (gt_traj_crop_x < crop_w)
            & (gt_traj_crop_y >= 0.0)
            & (gt_traj_crop_y < crop_h)
        )
        gt_visible = (
            (valids_all[frame_indices, point_idx] > 0.5)
            & (visibs_all[frame_indices, point_idx] > 0.5)
            & gt_in_bounds
        )

        tracks_01[query_idx, :, 0] = gt_traj_crop_x / max(crop_w - 1.0, 1.0)
        tracks_01[query_idx, :, 1] = gt_traj_crop_y / max(crop_h - 1.0, 1.0)
        visibility[query_idx, :] = gt_visible

    return tracks_01, visibility


def build_camera_indices(camera_mode, camera_frame, curr_t_src, curr_t_tgt):
    if camera_mode == "fixed" and camera_frame is not None:
        return torch.full_like(curr_t_tgt, int(camera_frame))
    if camera_mode == "follow_src":
        return curr_t_src
    return curr_t_tgt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None, choices=["pointodyssey", "kubric"])
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Optional pure-video input path. When set, the script skips dataset loading and predicts a reference-frame point cloud directly from the video.",
    )
    parser.add_argument(
        "--video-start-frame",
        type=int,
        default=0,
        help="Start frame index for pure-video mode.",
    )
    parser.add_argument(
        "--video-num-frames",
        type=int,
        default=None,
        help="Number of frames to read in pure-video mode. Defaults to the checkpoint temporal size.",
    )
    parser.add_argument(
        "--video-frame-stride",
        type=int,
        default=1,
        help="Temporal stride when reading frames in pure-video mode.",
    )
    parser.add_argument("--videomae-model", type=str, default=None)
    parser.add_argument("--num-queries", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--source-mode",
        type=str,
        default="fixed",
        choices=["fixed", "clip0", "dataset", "per_frame"],
        help="How to choose query source points for visualization.",
    )
    parser.add_argument("--source-frame", type=int, default=0, help="Clip-relative source frame for fixed mode.")
    parser.add_argument("--camera-frame", type=int, default=None, help="Clip-relative camera frame for fixed camera mode.")
    parser.add_argument(
        "--camera-mode",
        type=str,
        default="follow_tgt",
        choices=["follow_tgt", "fixed", "follow_src", "dynamic"],
        help="How camera frame is chosen while decoding.",
    )
    parser.add_argument("--patch-provider", type=str, default=None, help="Optional patch provider override.")
    parser.add_argument(
        "--skip-diagnostic-video",
        action="store_true",
        help="Disable the extra 2x2 diagnostic video with GT/pred/visibility/error panels.",
    )
    parser.add_argument(
        "--with-depth-video",
        action="store_true",
        help="Run dense depth prediction and save a depth comparison video. This is slower than track-only visualization.",
    )
    parser.add_argument(
        "--depth-chunk-size",
        type=int,
        default=4096,
        help="Number of dense depth queries to decode at once.",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.5,
        help="Threshold used when comparing predicted visibility against GT visibility.",
    )
    parser.add_argument(
        "--error-cap-px",
        type=float,
        default=16.0,
        help="Clip per-point 2D error colors to this pixel value in the diagnostic panel.",
    )
    parser.add_argument(
        "--with-world-pointcloud-gif",
        action="store_true",
        help="Save a dynamic point-cloud GIF. In dataset mode points are shown in world coordinates; in pure-video mode they are shown in the chosen reference-camera coordinates.",
    )
    parser.add_argument(
        "--with-world-pointcloud-multiview-gif",
        action="store_true",
        help="Save an extra multi-view point-cloud GIF with several azimuths arranged in a grid.",
    )
    parser.add_argument(
        "--pointcloud-stride",
        type=int,
        default=8,
        help="Uniform pixel stride for dynamic point-cloud sampling. Smaller values are denser and slower.",
    )
    parser.add_argument(
        "--pointcloud-chunk-size",
        type=int,
        default=4096,
        help="Number of dense point-cloud queries to decode at once.",
    )
    parser.add_argument(
        "--pointcloud-vis-threshold",
        type=float,
        default=0.5,
        help="Visibility threshold for keeping points in the dynamic point-cloud render.",
    )
    parser.add_argument(
        "--pointcloud-fps",
        type=float,
        default=8.0,
        help="Frames per second for the dynamic point-cloud GIF.",
    )
    parser.add_argument(
        "--pointcloud-point-size",
        type=float,
        default=5.0,
        help="Rendered point size for the dynamic point-cloud GIF.",
    )
    parser.add_argument(
        "--pointcloud-elev",
        type=float,
        default=22.0,
        help="Camera elevation for the world point-cloud render.",
    )
    parser.add_argument(
        "--pointcloud-azim",
        type=float,
        default=40.0,
        help="Camera azimuth for the world point-cloud render.",
    )
    parser.add_argument(
        "--reference-camera-frame",
        type=int,
        default=0,
        help="Reference camera frame used to unify point clouds in pure-video mode.",
    )
    parser.add_argument(
        "--pointcloud-view-azims",
        type=str,
        default="40,130,220,310",
        help="Comma-separated azimuths used for the multi-view world point-cloud GIF.",
    )
    parser.add_argument(
        "--pointcloud-rotate-azim-per-frame",
        type=float,
        default=0.0,
        help="Optional azimuth rotation added each frame when rendering the dynamic point-cloud GIF.",
    )
    parser.add_argument(
        "--pointcloud-max-render-points",
        type=int,
        default=12000,
        help="Optional cap on rendered points per frame in the dynamic point-cloud GIF.",
    )
    parser.add_argument(
        "--pointcloud-uniformization",
        type=str,
        default="none",
        choices=["none", "voxel"],
        help="Optional world-space point-cloud uniformization mode before rendering.",
    )
    parser.add_argument(
        "--pointcloud-voxel-size",
        type=float,
        default=0.05,
        help="Voxel size for `--pointcloud-uniformization voxel`, in world-coordinate units.",
    )
    parser.add_argument(
        "--save-pointcloud-per-frame-ply",
        action="store_true",
        help="Also save one PLY per frame after visibility filtering and optional voxel uniformization.",
    )
    args = parser.parse_args()

    require_visualization_dependencies()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt_args = load_model(args, Path(args.checkpoint), device)
    dataset_type = args.dataset or ckpt_args.get("dataset", "pointodyssey")

    if args.video_path is not None:
        run_video_only_pointcloud_visualization(args, model, ckpt_args, device)
        return

    resolved_data_root = args.data_root or ckpt_args.get("data_root")
    if resolved_data_root is None:
        raise ValueError("Could not resolve --data-root from CLI or checkpoint args")

    resolved_split = args.split or ckpt_args.get("val_split") or ckpt_args.get("train_split") or "val"
    resolved_data_root = resolve_data_root(dataset_type, resolved_data_root, resolved_split)
    resolved_sequence = resolve_sequence_name(
        dataset_type=dataset_type,
        data_root=resolved_data_root,
        split=resolved_split,
        sequence=args.sequence or ckpt_args.get("val_sequence") or ckpt_args.get("train_sequence"),
    )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.checkpoint).resolve().parent / "visualize_tracks"
    output_dir.mkdir(parents=True, exist_ok=True)

    viz_args = dict(vars(args))
    viz_args.update(
        {
            "resolved_dataset": dataset_type,
            "resolved_data_root": resolved_data_root,
            "resolved_split": resolved_split,
            "resolved_sequence": resolved_sequence,
        }
    )
    with open(output_dir / "viz_args.json", "w", encoding="utf-8") as f:
        json.dump(viz_args, f, indent=2, ensure_ascii=False)

    print(
        "Resolved visualization config: "
        f"dataset={dataset_type}, data_root={resolved_data_root}, "
        f"split={resolved_split}, sequence={resolved_sequence}"
    )

    patch_provider = args.patch_provider or ckpt_args.get("patch_provider", "auto")
    print(f"Using patch_provider: {patch_provider}")

    dataset = build_dataset_for_visualization(
        dataset_type=dataset_type,
        data_root=resolved_data_root,
        split=resolved_split,
        sequence=resolved_sequence,
        ckpt_args=ckpt_args,
        patch_provider=patch_provider,
        num_queries=args.num_queries,
    )
    if len(dataset) == 0:
        raise RuntimeError("Visualization dataset is empty")

    sample, success = dataset[0]
    if not success:
        raise RuntimeError("Failed to load visualization sample")

    frame_indices = sample["frame_indices"].cpu().numpy()
    print(f"Sample loaded. Frame indices: {frame_indices.tolist()}")

    input_dtype = next(model.parameters()).dtype
    video = sample["video"].unsqueeze(0).to(device=device, dtype=input_dtype)
    coords = sample["coords"].unsqueeze(0).to(device=device, dtype=input_dtype)
    t_src = sample["t_src"].unsqueeze(0).to(device=device)
    t_tgt = sample["t_tgt"].unsqueeze(0).to(device=device)
    t_cam = sample["t_cam"].unsqueeze(0).to(device=device)
    aspect_ratio = sample.get("aspect_ratio")
    if aspect_ratio is None:
        aspect_ratio = torch.tensor([1.0], device=device, dtype=input_dtype)
    else:
        aspect_ratio = aspect_ratio.unsqueeze(0).to(device=device, dtype=input_dtype)

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

    if hasattr(model, "module"):
        model.module.decoder.patch_provider = patch_provider
    else:
        model.decoder.patch_provider = patch_provider

    with torch.no_grad():
        with inference_autocast_context(device):
            encoder_features = model.encode(video, aspect_ratio=aspect_ratio)

    trajs_2d_all, valids_all, visibs_all = load_raw_annotations(
        dataset_type=dataset_type,
        data_root=resolved_data_root,
        split=resolved_split,
        sequence=resolved_sequence,
    )

    dataset_point_indices = sample["targets"]["point_indices"].cpu().numpy().astype(np.int64)
    dataset_coords_01 = coords.squeeze(0).float().cpu().numpy()
    dataset_t_src_cpu = t_src.squeeze(0).cpu().numpy()
    intrinsics_resized = sample["intrinsics"].float().cpu().numpy()
    extrinsics_w2c = sample["extrinsics"].float().cpu().numpy()
    num_frames = int(video.shape[1])
    query_mode = ckpt_args.get("query_mode", "full")

    if query_mode == "same_frame":
        effective_source_mode = "per_frame"
        print("query_mode=same_frame, override source-mode to per_frame for identity visualization.")
    else:
        effective_source_mode = "fixed" if args.source_mode == "clip0" else args.source_mode

    fixed_source_frame = int(args.source_frame)
    if not (0 <= fixed_source_frame < num_frames):
        raise ValueError(f"--source-frame must be in [0, {num_frames - 1}], got {fixed_source_frame}")

    gt_tracks_dataset_01, gt_visibility_dataset = build_gt_tracks_01(
        trajs_2d_all=trajs_2d_all,
        valids_all=valids_all,
        visibs_all=visibs_all,
        frame_indices=frame_indices,
        point_indices=dataset_point_indices,
        transform_metadata=sample["transform_metadata"],
    )

    if effective_source_mode == "fixed":
        meta = sample["transform_metadata"]
        crop_offset_xy = meta["crop_offset_xy"].cpu().numpy()
        crop_size_hw = meta["crop_size_hw"].cpu().numpy()
        point_indices = select_fixed_source_point_indices(
            trajs_2d_all=trajs_2d_all,
            valids_all=valids_all,
            visibs_all=visibs_all,
            frame_indices=frame_indices,
            source_frame=fixed_source_frame,
            crop_offset_xy=(float(crop_offset_xy[0]), float(crop_offset_xy[1])),
            crop_size_hw=(float(crop_size_hw[0]), float(crop_size_hw[1])),
            max_queries=args.num_queries,
        )
        gt_tracks_01, gt_visibility = build_gt_tracks_01(
            trajs_2d_all=trajs_2d_all,
            valids_all=valids_all,
            visibs_all=visibs_all,
            frame_indices=frame_indices,
            point_indices=point_indices,
            transform_metadata=sample["transform_metadata"],
        )
        num_queries = len(point_indices)
        coords_fixed = torch.from_numpy(gt_tracks_01[:, fixed_source_frame, :]).unsqueeze(0).to(
            device=device,
            dtype=input_dtype,
        )
        t_src_fixed = torch.full((1, num_queries), fixed_source_frame, device=device, dtype=torch.long)
        print(f"Using fixed source frame t={fixed_source_frame} with {num_queries} visible query points.")
    else:
        point_indices = dataset_point_indices
        gt_tracks_01 = gt_tracks_dataset_01
        gt_visibility = gt_visibility_dataset
        num_queries = len(point_indices)
        coords_fixed = None
        t_src_fixed = None

    query_frames_for_decode = video_query if video_query is not None else video
    local_patches_for_decode = local_patches if effective_source_mode == "dataset" else None

    # When not in dataset mode, local_patches are unavailable.
    # Fall back to a sampled provider; prefer sampled_highres only when video_query is available.
    active_decoder = model.module.decoder if hasattr(model, "module") else model.decoder
    if effective_source_mode != "dataset":
        fallback = resolve_sampled_patch_provider(patch_provider)
        if fallback == "sampled_highres" and video_query is None:
            fallback = "sampled_resized"
        active_decoder.patch_provider = fallback
    else:
        active_decoder.patch_provider = patch_provider

    pred_tracks_01 = []
    pred_points_3d = []
    pred_visibility_probs = []
    pred_t_cam_indices = []
    for frame_idx in tqdm(range(num_frames), desc="Decoding frames"):
        if effective_source_mode == "per_frame":
            curr_coords_np = np.clip(gt_tracks_01[:, frame_idx, :], 0.0, 1.0)
            curr_coords = torch.from_numpy(curr_coords_np).unsqueeze(0).to(device=device, dtype=input_dtype)
            curr_t_src = torch.full((1, num_queries), frame_idx, device=device, dtype=torch.long)
            curr_t_tgt = curr_t_src
        elif effective_source_mode == "dataset":
            curr_coords = coords
            curr_t_src = t_src
            curr_t_tgt = torch.full_like(t_tgt, frame_idx)
        else:
            curr_coords = coords_fixed
            curr_t_src = t_src_fixed
            curr_t_tgt = torch.full((1, num_queries), frame_idx, device=device, dtype=torch.long)

        curr_t_cam = build_camera_indices(args.camera_mode, args.camera_frame, curr_t_src, curr_t_tgt)

        with torch.no_grad():
            with inference_autocast_context(device):
                curr_preds = model.decode(
                    encoder_features=encoder_features,
                    frames=query_frames_for_decode,
                    coords=curr_coords,
                    t_src=curr_t_src,
                    t_tgt=curr_t_tgt,
                    t_cam=curr_t_cam,
                    local_patches=local_patches_for_decode,
                    transform_metadata=transform_metadata_input,
                )

        pred_tracks_01.append(curr_preds["pos_2d"].squeeze(0).float().cpu())
        pred_points_3d.append(curr_preds["pos_3d"].squeeze(0).float().cpu())
        pred_visibility_probs.append(
            torch.sigmoid(curr_preds["visibility"].squeeze(0).squeeze(-1)).float().cpu()
        )
        pred_t_cam_indices.append(curr_t_cam.squeeze(0).cpu())

    pred_tracks_01 = torch.stack(pred_tracks_01, dim=1).numpy()
    pred_points_3d = torch.stack(pred_points_3d, dim=1).numpy()
    pred_visibility_probs = torch.stack(pred_visibility_probs, dim=1).numpy()
    pred_t_cam_indices = torch.stack(pred_t_cam_indices, dim=1).numpy().astype(np.int64)

    video_np = (video.squeeze(0).permute(0, 2, 3, 1).float().cpu().numpy() * 255.0).astype(np.uint8)
    height, width = video_np.shape[1:3]
    pred_tracks_px = pred_tracks_01.copy()
    pred_tracks_px[..., 0] *= (width - 1)
    pred_tracks_px[..., 1] *= (height - 1)
    gt_tracks_px = gt_tracks_01.copy()
    gt_tracks_px[..., 0] *= (width - 1)
    gt_tracks_px[..., 1] *= (height - 1)
    l2_px = np.linalg.norm(pred_tracks_px - gt_tracks_px, axis=-1)
    reproj_tracks_px, reproj_valid_mask = project_predicted_3d_to_pixels(
        pred_points_3d,
        intrinsics_by_frame=intrinsics_resized,
        t_cam_indices=pred_t_cam_indices,
    )
    target_frame_indices = np.broadcast_to(
        np.arange(num_frames, dtype=np.int64)[None, :],
        pred_t_cam_indices.shape,
    )
    reproj_same_camera_mask = pred_t_cam_indices == target_frame_indices
    reproj_comparable_mask = reproj_valid_mask & reproj_same_camera_mask
    head_vs_reproj_l2_px = np.linalg.norm(pred_tracks_px - reproj_tracks_px, axis=-1)
    reproj_vs_gt_l2_px = np.linalg.norm(reproj_tracks_px - gt_tracks_px, axis=-1)

    visible_mask = gt_visibility
    if np.any(visible_mask):
        l1_norm = np.abs(pred_tracks_01 - gt_tracks_01).mean(axis=-1)
        l2_norm = np.linalg.norm(pred_tracks_01 - gt_tracks_01, axis=-1)
        l1_px = np.abs(pred_tracks_px - gt_tracks_px).mean(axis=-1)
        reproj_gt_mask = reproj_comparable_mask & gt_visibility
        head_reproj_mask = reproj_comparable_mask
        summary = {
            "dataset": dataset_type,
            "sequence": resolved_sequence,
            "num_queries": int(num_queries),
            "num_frames": int(num_frames),
            "num_visible_points": int(visible_mask.sum()),
            "query_mode": query_mode,
            "source_mode_requested": args.source_mode,
            "source_mode_effective": effective_source_mode,
            "camera_mode": args.camera_mode,
            "mean_l1_norm": float(l1_norm[visible_mask].mean()),
            "mean_l2_norm": float(l2_norm[visible_mask].mean()),
            "mean_l1_px": float(l1_px[visible_mask].mean()),
            "median_l1_px": float(np.median(l1_px[visible_mask])),
            "mean_l2_px": float(l2_px[visible_mask].mean()),
            "median_l2_px": float(np.median(l2_px[visible_mask])),
            "pck_1px": float((l2_px[visible_mask] <= 1.0).mean()),
            "pck_2px": float((l2_px[visible_mask] <= 2.0).mean()),
            "pck_4px": float((l2_px[visible_mask] <= 4.0).mean()),
            "pck_8px": float((l2_px[visible_mask] <= 8.0).mean()),
            "pred_visibility_mean": float(pred_visibility_probs.mean()),
            "pred_visibility_mean_gt_visible": float(pred_visibility_probs[visible_mask].mean()),
            "pred_visibility_mean_gt_invisible": float(pred_visibility_probs[~visible_mask].mean())
            if np.any(~visible_mask) else None,
            "visibility_acc@threshold": float(
                ((pred_visibility_probs >= args.visibility_threshold) == gt_visibility).mean()
            ),
            "reproj_num_comparable": int(reproj_comparable_mask.sum()),
            "reproj_num_gt_comparable": int(reproj_gt_mask.sum()),
            "head_vs_reproj_mean_l2_px": float(head_vs_reproj_l2_px[head_reproj_mask].mean())
            if np.any(head_reproj_mask) else None,
            "reproj3d_mean_l2_px": float(reproj_vs_gt_l2_px[reproj_gt_mask].mean())
            if np.any(reproj_gt_mask) else None,
            "reproj3d_median_l2_px": float(np.median(reproj_vs_gt_l2_px[reproj_gt_mask]))
            if np.any(reproj_gt_mask) else None,
            "reproj3d_pck_4px": float((reproj_vs_gt_l2_px[reproj_gt_mask] <= 4.0).mean())
            if np.any(reproj_gt_mask) else None,
        }
    else:
        summary = {
            "dataset": dataset_type,
            "sequence": resolved_sequence,
            "num_queries": int(num_queries),
            "num_frames": int(num_frames),
            "num_visible_points": 0,
            "query_mode": query_mode,
            "source_mode_requested": args.source_mode,
            "source_mode_effective": effective_source_mode,
            "camera_mode": args.camera_mode,
            "pred_visibility_mean": float(pred_visibility_probs.mean()),
            "reproj_num_comparable": int(reproj_comparable_mask.sum()),
        }

    input_title = "Input"
    input_panel_np = None
    input_coords_vis = None
    input_visibility = None
    if effective_source_mode == "fixed":
        input_title = f"Input (Clip t={fixed_source_frame})"
        input_panel_np = np.repeat(video_np[fixed_source_frame:fixed_source_frame + 1], num_frames, axis=0)
        input_coords_vis = coords_fixed.squeeze(0).float().cpu().numpy().copy()
        input_coords_vis[:, 0] *= (width - 1)
        input_coords_vis[:, 1] *= (height - 1)
    elif effective_source_mode == "dataset":
        input_title = "Input (Dataset Sources)"
        input_panel_np = build_source_contact_sheet(
            video_np=video_np,
            coords_input=dataset_coords_01,
            t_src=dataset_t_src_cpu,
            num_queries=num_queries,
        )
    else:
        input_title = "Input (Per-frame)"
        input_panel_np = video_np.copy()
        input_coords_vis = gt_tracks_px.copy()
        input_visibility = gt_visibility.copy()

    preview_frame_idx = fixed_source_frame if effective_source_mode == "fixed" else 0
    preview_frame = cv2.cvtColor(video_np[preview_frame_idx].copy(), cv2.COLOR_RGB2BGR)
    input_preview = cv2.cvtColor(get_panel_frame(input_panel_np, preview_frame_idx, video_np[preview_frame_idx]).copy(), cv2.COLOR_RGB2BGR)
    frame_gt = preview_frame.copy()
    frame_input = input_preview.copy()
    frame_pred = preview_frame.copy()

    cv2.putText(frame_gt, f"GT (Frame {preview_frame_idx})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame_input, input_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame_pred, f"Pred (Frame {preview_frame_idx})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    colors = matplotlib.colormaps.get_cmap("tab10")(np.linspace(0, 1, num_queries))[:, :3] * 255
    for query_idx in range(num_queries):
        color = tuple(int(channel) for channel in colors[query_idx])
        if gt_visibility[query_idx, preview_frame_idx]:
            gt_x = int(gt_tracks_px[query_idx, preview_frame_idx, 0])
            gt_y = int(gt_tracks_px[query_idx, preview_frame_idx, 1])
            if 0 <= gt_x < width and 0 <= gt_y < height:
                cv2.circle(frame_gt, (gt_x, gt_y), 4, color, -1)
                cv2.circle(frame_gt, (gt_x, gt_y), 2, (255, 255, 255), -1)

        pred_x = int(pred_tracks_px[query_idx, preview_frame_idx, 0])
        pred_y = int(pred_tracks_px[query_idx, preview_frame_idx, 1])
        if 0 <= pred_x < width and 0 <= pred_y < height:
            cv2.circle(frame_pred, (pred_x, pred_y), 4, color, -1)
            cv2.circle(frame_pred, (pred_x, pred_y), 2, (255, 255, 255), -1)

        if input_coords_vis is not None:
            if input_coords_vis.ndim == 2:
                input_xy = input_coords_vis[query_idx]
                input_is_visible = True
            else:
                input_xy = input_coords_vis[query_idx, preview_frame_idx]
                input_is_visible = input_visibility is None or bool(input_visibility[query_idx, preview_frame_idx])
            in_x = int(input_xy[0])
            in_y = int(input_xy[1])
            if input_is_visible and 0 <= in_x < width and 0 <= in_y < height:
                cv2.circle(frame_input, (in_x, in_y), 4, color, -1)
                cv2.circle(frame_input, (in_x, in_y), 2, (255, 255, 255), -1)

    combined_first = np.hstack((frame_gt, frame_input, frame_pred))
    first_frame_path = output_dir / f"{resolved_sequence}_frame{preview_frame_idx}_check.png"
    cv2.imwrite(str(first_frame_path), combined_first)
    print(f"Saved first frame alignment check to {first_frame_path}")

    vis_save_path = output_dir / f"{resolved_sequence}_compare_tracks.mp4"
    draw_tracks_2d_compare(
        video_np=video_np,
        coords_2d_pred=pred_tracks_px,
        coords_2d_gt=gt_tracks_px,
        input_panel_np=input_panel_np,
        coords_input=input_coords_vis,
        gt_visibility=gt_visibility,
        input_visibility=input_visibility,
        input_title=input_title,
        output_path=str(vis_save_path),
    )
    print(f"Saved comparison video to {vis_save_path}")

    source_points_path = output_dir / "source_points_sparse.png"
    cv2.imwrite(str(source_points_path), frame_input)
    print(f"Saved source point preview to {source_points_path}")

    reproj_preview_path = output_dir / f"{resolved_sequence}_reprojection_frame{preview_frame_idx}.png"
    gt_reproj_panel = make_track_panel(
        video_np[preview_frame_idx],
        gt_tracks_px[:, preview_frame_idx, :],
        colors,
        "GT Tracks",
        visible_mask=gt_visibility[:, preview_frame_idx],
        subtitle=f"frame={preview_frame_idx}",
    )
    head_reproj_panel = make_track_panel(
        video_np[preview_frame_idx],
        pred_tracks_px[:, preview_frame_idx, :],
        colors,
        "Pred 2D Head",
        subtitle=f"frame={preview_frame_idx}",
    )
    reproj_panel = make_track_panel(
        video_np[preview_frame_idx],
        reproj_tracks_px[:, preview_frame_idx, :],
        colors,
        "Reproj 3D Head",
        visible_mask=reproj_comparable_mask[:, preview_frame_idx],
        subtitle=f"gt-comparable={int(reproj_comparable_mask[:, preview_frame_idx].sum())}/{num_queries}",
    )
    reproj_error_panel = make_error_panel(
        video_np[preview_frame_idx],
        reproj_tracks_px[:, preview_frame_idx, :],
        gt_tracks_px[:, preview_frame_idx, :],
        gt_visibility[:, preview_frame_idx] & reproj_comparable_mask[:, preview_frame_idx],
        error_cap_px=args.error_cap_px,
        title="Reproj3D vs GT",
    )
    combined_reproj = np.vstack([
        np.hstack([gt_reproj_panel, head_reproj_panel]),
        np.hstack([reproj_panel, reproj_error_panel]),
    ])
    cv2.imwrite(str(reproj_preview_path), combined_reproj)
    print(f"Saved reprojection preview to {reproj_preview_path}")

    reproj_video_path = output_dir / f"{resolved_sequence}_reprojection_compare.mp4"
    save_reprojection_comparison_video(
        video_np=video_np,
        gt_tracks_px=gt_tracks_px,
        head_tracks_px=pred_tracks_px,
        reproj_tracks_px=reproj_tracks_px,
        gt_visibility=gt_visibility,
        reproj_comparable_mask=reproj_comparable_mask,
        output_path=reproj_video_path,
        error_cap_px=args.error_cap_px,
    )
    print(f"Saved reprojection comparison video to {reproj_video_path}")

    if not args.skip_diagnostic_video:
        diagnostic_preview_path = output_dir / f"{resolved_sequence}_diagnostic_frame{preview_frame_idx}.png"
        gt_diag = make_track_panel(
            video_np[preview_frame_idx],
            gt_tracks_px[:, preview_frame_idx, :],
            colors,
            "GT Tracks",
            visible_mask=gt_visibility[:, preview_frame_idx],
            subtitle=f"frame={preview_frame_idx}",
        )
        pred_diag = make_track_panel(
            video_np[preview_frame_idx],
            pred_tracks_px[:, preview_frame_idx, :],
            colors,
            "Pred Tracks",
            subtitle=f"frame={preview_frame_idx}",
        )
        vis_diag = make_visibility_panel(
            video_np[preview_frame_idx],
            pred_tracks_px[:, preview_frame_idx, :],
            pred_visibility_probs[:, preview_frame_idx],
            gt_visibility=gt_visibility[:, preview_frame_idx],
            threshold=args.visibility_threshold,
        )
        err_diag = make_error_panel(
            video_np[preview_frame_idx],
            pred_tracks_px[:, preview_frame_idx, :],
            gt_tracks_px[:, preview_frame_idx, :],
            gt_visibility[:, preview_frame_idx],
            error_cap_px=args.error_cap_px,
        )
        combined_diag = np.vstack([np.hstack([gt_diag, pred_diag]), np.hstack([vis_diag, err_diag])])
        cv2.imwrite(str(diagnostic_preview_path), combined_diag)
        print(f"Saved diagnostic preview to {diagnostic_preview_path}")

        diagnostic_video_path = output_dir / f"{resolved_sequence}_diagnostic_panels.mp4"
        save_diagnostic_video(
            video_np=video_np,
            gt_tracks_px=gt_tracks_px,
            pred_tracks_px=pred_tracks_px,
            gt_visibility=gt_visibility,
            pred_visibility=pred_visibility_probs,
            output_path=diagnostic_video_path,
            error_cap_px=args.error_cap_px,
            visibility_threshold=args.visibility_threshold,
        )
        print(f"Saved diagnostic video to {diagnostic_video_path}")

    if args.with_depth_video:
        gt_depths = sample["depths"].squeeze(1).float().cpu().numpy()
        pred_depths = predict_dense_depth_maps(
            model=model,
            encoder_features=encoder_features,
            frames=query_frames_for_decode,
            transform_metadata=transform_metadata_input,
            num_frames=num_frames,
            height=height,
            width=width,
            device=device,
            input_dtype=input_dtype,
            chunk_size=max(1, int(args.depth_chunk_size)),
        )
        depth_valid = np.isfinite(gt_depths) & np.isfinite(pred_depths) & (gt_depths > 0.0)
        if np.any(depth_valid):
            depth_abs_error = np.abs(pred_depths - gt_depths)
            summary["depth_mae"] = float(depth_abs_error[depth_valid].mean())
            summary["depth_rmse"] = float(np.sqrt(np.square(pred_depths - gt_depths)[depth_valid].mean()))
            summary["depth_rel_l1"] = float((depth_abs_error[depth_valid] / np.maximum(gt_depths[depth_valid], 1e-6)).mean())
        else:
            summary["depth_mae"] = None
            summary["depth_rmse"] = None
            summary["depth_rel_l1"] = None

        depth_video_path = output_dir / f"{resolved_sequence}_depth_comparison.mp4"
        save_depth_comparison_video(
            video_np=video_np,
            pred_depths=pred_depths,
            gt_depths=gt_depths,
            output_path=depth_video_path,
        )
        print(f"Saved depth comparison video to {depth_video_path}")

    if args.with_world_pointcloud_gif or args.with_world_pointcloud_multiview_gif:
        pointcloud_result = predict_dense_world_points(
            model=model,
            encoder_features=encoder_features,
            frames=query_frames_for_decode,
            transform_metadata=transform_metadata_input,
            extrinsics_w2c=extrinsics_w2c,
            video_rgb=video_np,
            num_frames=num_frames,
            height=height,
            width=width,
            device=device,
            input_dtype=input_dtype,
            stride=max(1, int(args.pointcloud_stride)),
            chunk_size=max(1, int(args.pointcloud_chunk_size)),
        )

        world_valid_mask = (
            np.isfinite(pointcloud_result["world_points"]).all(axis=-1)
            & (pointcloud_result["visibility"] >= args.pointcloud_vis_threshold)
        )
        points_per_frame = world_valid_mask.sum(axis=1)
        summary["world_pointcloud_stride"] = int(pointcloud_result["stride"])
        summary["world_pointcloud_points_per_frame"] = int(pointcloud_result["num_points_per_frame"])
        summary["world_pointcloud_visible_points_mean"] = float(points_per_frame.mean())
        summary["world_pointcloud_visible_points_min"] = int(points_per_frame.min())
        summary["world_pointcloud_visible_points_max"] = int(points_per_frame.max())
        summary["world_pointcloud_uniformization"] = args.pointcloud_uniformization
        summary["world_pointcloud_voxel_size"] = float(args.pointcloud_voxel_size)

        if args.with_world_pointcloud_gif:
            world_gif_path = output_dir / f"{resolved_sequence}_world_pointcloud.gif"
            pointcloud_preview_rgb, pointcloud_center, pointcloud_radius, rendered_counts = save_world_pointcloud_gif(
                world_points=pointcloud_result["world_points"],
                colors=pointcloud_result["colors"],
                visibility_probs=pointcloud_result["visibility"],
                output_path=world_gif_path,
                fps=args.pointcloud_fps,
                vis_threshold=args.pointcloud_vis_threshold,
                elev=args.pointcloud_elev,
                azim=args.pointcloud_azim,
                rotate_azim_per_frame=args.pointcloud_rotate_azim_per_frame,
                point_size=args.pointcloud_point_size,
                max_render_points=args.pointcloud_max_render_points,
                uniformization_mode=args.pointcloud_uniformization,
                voxel_size=args.pointcloud_voxel_size,
            )
            pointcloud_preview_path = output_dir / f"{resolved_sequence}_world_pointcloud_preview.png"
            cv2.imwrite(str(pointcloud_preview_path), cv2.cvtColor(pointcloud_preview_rgb, cv2.COLOR_RGB2BGR))
            print(f"Saved world point-cloud preview to {pointcloud_preview_path}")
            print(f"Saved world point-cloud GIF to {world_gif_path}")

            summary["world_pointcloud_rendered_points_mean"] = float(np.mean(rendered_counts))
            summary["world_pointcloud_rendered_points_min"] = int(np.min(rendered_counts))
            summary["world_pointcloud_rendered_points_max"] = int(np.max(rendered_counts))
            summary["world_pointcloud_bounds_center"] = [float(x) for x in pointcloud_center.tolist()]
            summary["world_pointcloud_bounds_radius"] = float(pointcloud_radius)

        if args.with_world_pointcloud_multiview_gif:
            multiview_azims = [float(x.strip()) for x in args.pointcloud_view_azims.split(",") if x.strip()]
            if not multiview_azims:
                raise ValueError("--pointcloud-view-azims must contain at least one angle")

            multiview_gif_path = output_dir / f"{resolved_sequence}_world_pointcloud_multiview.gif"
            multiview_preview_rgb, _, _, multiview_rendered_counts = save_world_pointcloud_multiview_gif(
                world_points=pointcloud_result["world_points"],
                colors=pointcloud_result["colors"],
                visibility_probs=pointcloud_result["visibility"],
                output_path=multiview_gif_path,
                fps=args.pointcloud_fps,
                vis_threshold=args.pointcloud_vis_threshold,
                elev=args.pointcloud_elev,
                azims=multiview_azims,
                rotate_azim_per_frame=args.pointcloud_rotate_azim_per_frame,
                point_size=args.pointcloud_point_size,
                max_render_points=args.pointcloud_max_render_points,
                uniformization_mode=args.pointcloud_uniformization,
                voxel_size=args.pointcloud_voxel_size,
            )
            multiview_preview_path = output_dir / f"{resolved_sequence}_world_pointcloud_multiview_preview.png"
            cv2.imwrite(str(multiview_preview_path), cv2.cvtColor(multiview_preview_rgb, cv2.COLOR_RGB2BGR))
            print(f"Saved world point-cloud multiview preview to {multiview_preview_path}")
            print(f"Saved world point-cloud multiview GIF to {multiview_gif_path}")

            summary["world_pointcloud_multiview_azims"] = multiview_azims
            summary["world_pointcloud_multiview_rendered_points_mean"] = float(np.mean(multiview_rendered_counts))
            summary["world_pointcloud_multiview_rendered_points_min"] = int(np.min(multiview_rendered_counts))
            summary["world_pointcloud_multiview_rendered_points_max"] = int(np.max(multiview_rendered_counts))
            if "world_pointcloud_bounds_center" not in summary:
                multiview_center, multiview_radius = compute_world_bounds(
                    pointcloud_result["world_points"],
                    pointcloud_result["visibility"],
                    args.pointcloud_vis_threshold,
                )
                summary["world_pointcloud_bounds_center"] = [float(x) for x in multiview_center.tolist()]
                summary["world_pointcloud_bounds_radius"] = float(multiview_radius)

    print("Tracking summary:", json.dumps(summary, ensure_ascii=False))
    summary_path = output_dir / "tracking_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved tracking summary to {summary_path}")


if __name__ == "__main__":
    main()
