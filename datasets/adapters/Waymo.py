from __future__ import annotations

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import glob
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils
frame_utils.bytearray = bytes

import torch
import torchvision.transforms.functional as F_vision
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

from .base import BaseAdapter, UnifiedClip


class RAFTFlowEstimator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"⏳ Loading RAFT flow model to {device}...")
        weights = Raft_Large_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.model = raft_large(weights=weights, progress=False).to(self.device)
        self.model.eval()
        print("✅ RAFT model loaded")

    @torch.no_grad()
    def compute_flow(self, img1_np: np.ndarray, img2_np: np.ndarray) -> np.ndarray:
        t1 = F_vision.to_tensor(img1_np).unsqueeze(0).to(self.device)
        t2 = F_vision.to_tensor(img2_np).unsqueeze(0).to(self.device)
        t1, t2 = self.transforms(t1, t2)
        list_of_flows = self.model(t1, t2)
        flow = list_of_flows[-1][0]
        return flow.cpu().numpy().transpose(1, 2, 0)


class WaymoAdapter(BaseAdapter):
    dataset_name: str = "waymo"

    def __init__(
        self,
        root: str,
        split: str = "training",
        camera_name: int = dataset_pb2.CameraName.FRONT,
        extract_flow: bool = True,
        precompute_root: Optional[str] = None,
        verbose: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.camera_name = camera_name
        self.extract_flow = extract_flow
        self.precompute_root = Path(precompute_root) if precompute_root else self.root
        self.verbose = verbose

        if self.extract_flow:
            self.flow_estimator = RAFTFlowEstimator()

        self.sequences: list[str] = []
        self.seq_to_path: dict[str, Path] = {}
        self._build_index()

        if len(self.sequences) == 0:
            raise RuntimeError(f"No valid Waymo sequences found under {self.root}")

        if self.verbose:
            print(f"[WaymoAdapter] split={self.split}, num_sequences={len(self.sequences)}, "
                  f"extract_flow={self.extract_flow}, precompute={'yes' if self.precompute_root else 'no'}")

    def __len__(self) -> int:
        return len(self.sequences)

    def list_sequences(self) -> list[str]:
        return self.sequences.copy()

    def get_sequence_name(self, index: int) -> str:
        return self.sequences[index]

    def _build_index(self):
        # Search both root and split subdirectory (train/val/test)
        record_files = glob.glob(str(self.root / f"*{self.split}*.tfrecord"))
        record_files += glob.glob(str(self.root / self.split / "*.tfrecord"))
        for file_path in record_files:
            seq_name = Path(file_path).stem
            self.sequences.append(seq_name)
            self.seq_to_path[seq_name] = Path(file_path)

    def _load_precomputed(self, sequence_name: str) -> Optional[dict]:
        if self.precompute_root is None:
            return None
        cache_path = self.precompute_root / sequence_name / "precomputed.npz"
        if not cache_path.exists():
            return None
        try:
            data = np.load(cache_path)
            return {k: data[k] for k in data.files}
        except Exception as e:
            if self.verbose:
                print(f"[WaymoAdapter] Warning: failed to load cache for {sequence_name}: {e}")
            return None

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        has_precomputed = (
            self.precompute_root is not None
            and (self.precompute_root / sequence_name / "precomputed.npz").exists()
        )
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "sequence_name": sequence_name,
            "num_frames": 198,
            "has_depth": True,
            "has_flow": self.extract_flow,
            "has_normals": has_precomputed,
            "has_tracks": has_precomputed,
            "has_visibility": has_precomputed,
        }

    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        tfrecord_path = self.seq_to_path[sequence_name]
        dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type='')

        images, extrinsics, intrinsics, depths = [], [], [], []
        target_indices = sorted(list(set(frame_indices)))
        frames_extracted = 0

        for frame_idx, data in enumerate(dataset):
            if frame_idx not in target_indices:
                continue

            frame = dataset_pb2.Frame()
            frame.ParseFromString(data.numpy())

            img_np = None
            for camera_image in frame.images:
                if camera_image.name == self.camera_name:
                    img_np = tf.image.decode_jpeg(camera_image.image).numpy()
                    images.append(img_np)  # Keep as uint8 [0, 255]
                    # camera_image.pose.transform is vehicle-to-world
                    vehicle_to_world = np.array(camera_image.pose.transform).reshape(4, 4)
                    break

            for calibration in frame.context.camera_calibrations:
                if calibration.name == self.camera_name:
                    calib = calibration.intrinsic
                    intrinsic_mat = np.array([
                        [calib[0], 0,        calib[2]],
                        [0,        calib[1], calib[3]],
                        [0,        0,        1       ]
                    ], dtype=np.float32)
                    intrinsics.append(intrinsic_mat)
                    # camera_to_vehicle extrinsic from calibration
                    cam_to_vehicle = np.array(calibration.extrinsic.transform).reshape(4, 4)
                    # Waymo camera axes: X=forward, Y=left, Z=up
                    # Standard camera axes: X=right, Y=down, Z=forward
                    # std = R_ws @ way  =>  std_X=-way_Y, std_Y=-way_Z, std_Z=way_X
                    R_ws = np.array([[ 0, -1,  0,  0],
                                     [ 0,  0, -1,  0],
                                     [ 1,  0,  0,  0],
                                     [ 0,  0,  0,  1]], dtype=np.float64)
                    # std_cam -> vehicle = cam_to_vehicle @ inv(R_ws) @ std_pts
                    # c2w (std_cam -> world) = vehicle_to_world @ cam_to_vehicle @ inv(R_ws)
                    cam_to_vehicle_std = cam_to_vehicle @ np.linalg.inv(R_ws)
                    c2w = vehicle_to_world @ cam_to_vehicle_std
                    R = c2w[:3, :3]
                    t = c2w[:3, 3]
                    w2c = np.eye(4, dtype=np.float32)
                    w2c[:3, :3] = R.T
                    w2c[:3, 3]  = (-R.T @ t).astype(np.float32)
                    extrinsics.append(w2c)
                    break

            if img_np is not None:
                (range_images, camera_projections, _, range_image_top_pose) = \
                    frame_utils.parse_range_image_and_camera_projection(frame)
                points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                    frame, range_images, camera_projections, range_image_top_pose)
                points_all = np.concatenate(points, axis=0)       # [N,3] world xyz
                cp_points_all = np.concatenate(cp_points, axis=0) # [N,6] cam_id,px,py,...

                # Filter points projected onto this camera (primary projection)
                mask = (cp_points_all[:, 0] == int(self.camera_name))
                pts_veh = points_all[mask].astype(np.float64)  # [M,3] vehicle frame
                px = cp_points_all[mask, 1]           # pixel x (from Waymo projection)
                py = cp_points_all[mask, 2]           # pixel y

                # Transform vehicle points to Waymo camera frame (X=forward=depth)
                vehicle_to_cam = np.linalg.inv(cam_to_vehicle)
                pts_h = np.concatenate([pts_veh, np.ones((len(pts_veh), 1))], axis=1)
                pts_cam = (vehicle_to_cam @ pts_h.T).T
                depth_x = pts_cam[:, 0]               # Waymo camera X = depth

                h, w_img, _ = img_np.shape
                depth_map = np.zeros((h, w_img), dtype=np.float32)
                xi = np.round(px).astype(np.int32)
                yi = np.round(py).astype(np.int32)
                valid = (xi >= 0) & (xi < w_img) & (yi >= 0) & (yi < h) & (depth_x > 0)
                depth_map[yi[valid], xi[valid]] = depth_x[valid].astype(np.float32)
                depths.append(depth_map)

            frames_extracted += 1
            if frames_extracted == len(target_indices):
                break

        flows = []
        if self.extract_flow and len(images) > 0:
            if self.verbose:
                print(f"🌊 Computing RAFT flow...")
            for i in range(len(images)):
                if i < len(images) - 1:
                    fwd_flow = self.flow_estimator.compute_flow(images[i], images[i+1])
                    flows.append(fwd_flow)
                else:
                    flows.append(None)

        # Load precomputed normals/tracks if available
        normals_out, trajs_2d_out, trajs_3d_out, valids_out, visibs_out = None, None, None, None, None
        has_precomputed = False
        cache = self._load_precomputed(sequence_name)
        if cache is not None:
            try:
                normals_out  = [cache["normals"][i].astype(np.float32) for i in target_indices]
                trajs_2d_out = cache["trajs_2d"][np.array(target_indices)]
                trajs_3d_out = cache["trajs_3d_world"][np.array(target_indices)]
                valids_out   = cache["valids"][np.array(target_indices)]
                visibs_out   = cache["visibs"][np.array(target_indices)]
                has_precomputed = True
            except Exception as e:
                if self.verbose:
                    print(f"[WaymoAdapter] Warning: precomputed indexing failed for {sequence_name}: {e}")

        # Generate pseudo-tracks from depth for visualization if no precomputed tracks
        if not has_precomputed and len(depths) > 0 and len(extrinsics) > 0:
            T = len(images)
            n_sample = 2000
            trajs_2d_out = np.zeros((T, n_sample, 2), dtype=np.float32)
            trajs_3d_out = np.zeros((T, n_sample, 3), dtype=np.float32)
            valids_out   = np.zeros((T, n_sample), dtype=bool)
            visibs_out   = np.zeros((T, n_sample), dtype=bool)

            rng = np.random.default_rng(42)
            for t in range(T):
                K  = intrinsics[t]
                E  = extrinsics[t]
                E_inv = np.linalg.inv(E.astype(np.float64))
                fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
                depth_t = depths[t]

                ys, xs = np.where(depth_t > 0)
                if len(ys) == 0:
                    continue
                sel = rng.choice(len(ys), min(n_sample, len(ys)), replace=False)
                xs_t = xs[sel]; ys_t = ys[sel]
                n = len(xs_t)

                z = depth_t[ys_t, xs_t].astype(np.float64)
                x_c = (xs_t - cx) / fx * z
                y_c = (ys_t - cy) / fy * z
                pts_cam = np.stack([x_c, y_c, z, np.ones(n)], axis=1)  # [n,4]
                pts_world = (E_inv @ pts_cam.T).T[:, :3]                # [n,3]

                trajs_2d_out[t, :n, 0] = xs_t
                trajs_2d_out[t, :n, 1] = ys_t
                trajs_3d_out[t, :n]    = pts_world.astype(np.float32)
                valids_out[t, :n]      = True
                visibs_out[t, :n]      = True

            has_precomputed = True

        return UnifiedClip(
            dataset_name=self.dataset_name,
            sequence_name=sequence_name,
            frame_paths=None,
            images=images,
            depths=depths if depths else None,
            normals=normals_out,
            trajs_2d=trajs_2d_out,
            trajs_3d_world=trajs_3d_out,
            valids=valids_out,
            visibs=visibs_out,
            intrinsics=np.stack(intrinsics, axis=0) if intrinsics else np.zeros((len(images), 3, 3), dtype=np.float32),
            extrinsics=np.stack(extrinsics, axis=0) if extrinsics else np.eye(4, dtype=np.float32)[None].repeat(len(images), axis=0),
            flows=flows if any(f is not None for f in flows) else None,
            metadata={
                "dataset_name": self.dataset_name,
                "split": self.split,
                "has_depth": len(depths) > 0,
                "has_flow": any(f is not None for f in flows),
                "has_normals": has_precomputed,
                "has_tracks": has_precomputed,
                "has_visibility": has_precomputed,
                "pose_convention": "world_to_camera",
                "intrinsics_convention": "pinhole",
                "depth_unit": "meters",
            }
        )

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        info = self.get_sequence_info(sequence_name)
        info["ok"] = True
        return info
