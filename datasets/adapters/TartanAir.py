from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .base import BaseAdapter, UnifiedClip


def quat_to_rotation_matrix(quat):
    """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
    qx, qy, qz, qw = quat
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ], dtype=np.float32)


class TartanAirAdapter(BaseAdapter):
    dataset_name: str = "tartanair"

    def __init__(
        self,
        root: str,
        split: str = "train",
        camera: str = "left",
        precompute_root: Optional[str] = None,
        verbose: bool = True,
        strict: bool = True,
    ):
        """
        Args:
            root:             TartanAir root directory (contains P001/, P003/, …).
            split:            Dataset split tag (informational only).
            camera:           Which camera to use: 'left' or 'right'.
            precompute_root:  If set, load precomputed normals/tracks from
                              <precompute_root>/<seq_name>/precomputed.npz.
                              Run computer/run_tartanair.py to generate these files.
            verbose:          Print loading info.
        """
        self.root = Path(root)
        self.split = split
        self.camera = camera
        self.precompute_root = Path(precompute_root) if precompute_root else self.root
        self.verbose = verbose

        # TartanAir standard intrinsics (640x480)
        self.K = np.array([
            [320.0, 0.0,   320.0],
            [0.0,   320.0, 240.0],
            [0.0,   0.0,   1.0]
        ], dtype=np.float32)

        self.sequences: list[str] = []
        self.seq_to_dir: dict[str, Path] = {}
        # Maps seq_name -> sorted list of frame indices that have valid+visible precomputed points.
        # If no precomputed cache exists for a sequence, maps to None (use all image frames).
        self.seq_valid_frames: dict[str, Optional[list[int]]] = {}
        self._build_index()

        if len(self.sequences) == 0:
            raise RuntimeError(f"No valid TartanAir sequences found under {self.root}")

        if self.verbose:
            print(f"[TartanAirAdapter] split={self.split}, camera={self.camera}, "
                  f"num_sequences={len(self.sequences)}, "
                  f"precompute={'yes' if self.precompute_root else 'no'}")

    def __len__(self) -> int:
        return len(self.sequences)

    def list_sequences(self) -> list[str]:
        return self.sequences.copy()

    def get_sequence_name(self, index: int) -> str:
        return self.sequences[index]

    def _build_index(self):
        subdirs = [f for f in self.root.iterdir() if f.is_dir()]
        for seq_dir in sorted(subdirs):
            seq_name = seq_dir.name
            self.sequences.append(seq_name)
            self.seq_to_dir[seq_name] = seq_dir
            self.seq_valid_frames[seq_name] = self._get_valid_frame_list(seq_dir, seq_name)

    def _get_valid_frame_list(self, seq_dir: Path, seq_name: str) -> Optional[list[int]]:
        """Return sorted list of usable frame indices for this sequence.

        When a precomputed cache exists, returns only the contiguous run of frames
        starting from ref_frame that have at least MIN_VALID_PTS valid+visible points.
        This ensures the sampler always picks frames with dense track coverage.
        Falls back to all image frames when no cache is found.
        """
        MIN_VALID_PTS = 5
        h5_path = self.precompute_root / seq_name / "precomputed.h5"
        npz_path = self.precompute_root / seq_name / "precomputed.npz"
        try:
            if h5_path.exists():
                import h5py
                with h5py.File(h5_path, 'r') as f:
                    valids = f['valids'][()]
                    visibs = f['visibs'][()]
                    ref_frame = int(f['ref_frame'][()])
            elif npz_path.exists():
                raw = np.load(npz_path, allow_pickle=True)
                valids = raw['valids']
                visibs = raw['visibs']
                ref_frame = int(raw['ref_frame'])
            else:
                return None

            per_frame = (valids & visibs).sum(axis=1)
            # Find the contiguous run from ref_frame where valid pts >= threshold
            n = len(per_frame)
            end = ref_frame
            while end < n and per_frame[end] >= MIN_VALID_PTS:
                end += 1
            good = list(range(ref_frame, end))
            if len(good) == 0:
                # fallback: any frame with valid pts
                good = np.where(per_frame >= MIN_VALID_PTS)[0].tolist()
            return good if good else None
        except Exception as e:
            if self.verbose:
                print(f"[TartanAirAdapter] Warning: could not read valid frames for {seq_name}: {e}")
        return None

    def _load_precomputed(self, sequence_name: str, frame_indices: list[int]) -> Optional[dict]:
        """Load precomputed normals/tracks for frame_indices. Prefers .h5 over .npz."""
        if self.precompute_root is None:
            return None
        from datasets.adapters.base import load_precomputed_fast
        cache_path = self.precompute_root / sequence_name / "precomputed.npz"
        h5_path = cache_path.with_suffix('.h5')
        if not cache_path.exists() and not h5_path.exists():
            return None
        try:
            return load_precomputed_fast(cache_path, frame_indices)
        except Exception as e:
            if self.verbose:
                print(f"[TartanAirAdapter] Warning: failed to load precomputed cache "
                      f"for {sequence_name}: {e}")
            return None

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        seq_dir = self.seq_to_dir[sequence_name]
        img_dir = seq_dir / f"image_{self.camera}"

        total_img_frames = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0

        valid_frames = self.seq_valid_frames.get(sequence_name)
        num_frames = len(valid_frames) if valid_frames is not None else total_img_frames

        has_precomputed = (
            self.precompute_root is not None
            and (
                (self.precompute_root / sequence_name / "precomputed.npz").exists()
                or (self.precompute_root / sequence_name / "precomputed.h5").exists()
            )
        )

        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "sequence_name": sequence_name,
            "num_frames": num_frames,
            "has_depth": (seq_dir / f"depth_{self.camera}").exists(),
            "has_flow": (seq_dir / "flow").exists(),
            "has_normals": has_precomputed,
            "has_tracks": has_precomputed,
            "has_visibility": has_precomputed,
        }

    def get_num_frames(self, sequence_name: str) -> int:
        valid_frames = self.seq_valid_frames.get(sequence_name)
        if valid_frames is not None:
            return len(valid_frames)
        seq_dir = self.seq_to_dir[sequence_name]
        img_dir = seq_dir / f"image_{self.camera}"
        return len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0

    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        seq_dir = self.seq_to_dir[sequence_name]

        img_dir   = seq_dir / f"image_{self.camera}"
        depth_dir = seq_dir / f"depth_{self.camera}"
        pose_file = seq_dir / f"pose_{self.camera}.txt"
        flow_dir  = seq_dir / "flow"

        # Map logical frame_indices to actual frame numbers via valid_frames list
        valid_frames = self.seq_valid_frames.get(sequence_name)
        if valid_frames is not None:
            actual_indices = [valid_frames[i] for i in frame_indices]
        else:
            actual_indices = frame_indices

        all_img_paths   = sorted(img_dir.glob("*.*"))
        all_depth_paths = sorted(depth_dir.glob("*.*")) if depth_dir.exists() else []
        all_flow_paths  = sorted(flow_dir.glob("*_flow.npy")) if flow_dir.exists() else []

        with open(pose_file, 'r') as f:
            all_pose_lines = f.readlines()

        images, depths, extrinsics, intrinsics, flows = [], [], [], [], []
        frame_paths = []

        for idx in actual_indices:
            # Load image
            img_path = all_img_paths[idx]
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)  # uint8 [0,255], consistent with other adapters
            frame_paths.append(str(img_path))

            # Load depth (.npy float32, metres; sentinel values (sky/infinity) are clipped)
            if idx < len(all_depth_paths):
                depth_path = all_depth_paths[idx]
                if depth_path.suffix == '.npy':
                    depth = np.load(depth_path).astype(np.float32)
                else:
                    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                    if depth is not None:
                        depth = depth.astype(np.float32) / 1000.0
                # Clip sky/infinity sentinel values (>200m treated as invalid)
                if depth is not None:
                    depth = np.where(depth > 200.0, np.nan, depth)
                depths.append(depth)

            # Parse pose
            # TartanAir pose file stores camera-to-world (c2w) transforms:
            #   tx ty tz qx qy qz qw
            # Invert analytically to get world-to-camera (w2c) for consistency.
            if idx < len(all_pose_lines):
                line = all_pose_lines[idx].strip().split()
                if len(line) == 7:
                    tx, ty, tz, qx, qy, qz, qw = map(float, line)
                    rot_mat = quat_to_rotation_matrix([qx, qy, qz, qw])
                    # Build c2w first
                    c2w = np.eye(4, dtype=np.float32)
                    c2w[:3, :3] = rot_mat
                    c2w[0, 3] = tx
                    c2w[1, 3] = ty
                    c2w[2, 3] = tz
                    # Invert analytically to get w2c
                    R = c2w[:3, :3]
                    t = c2w[:3, 3]
                    w2c = np.eye(4, dtype=np.float32)
                    w2c[:3, :3] = R.T
                    w2c[:3, 3]  = -R.T @ t
                    extrinsics.append(w2c)

            intrinsics.append(self.K.copy())

            # Load flow
            if idx < len(all_flow_paths):
                flow_path = all_flow_paths[idx]
                if flow_path.suffix == '.npy':
                    flows.append(np.load(flow_path))
                else:
                    flows.append(None)
            else:
                flows.append(None)

        # ---- precomputed normals / tracks ----
        normals_out    = None
        trajs_2d_out   = None
        trajs_3d_out   = None
        valids_out     = None
        visibs_out     = None
        has_precomputed = False

        cache = self._load_precomputed(sequence_name, actual_indices)
        if cache is not None:
            try:
                normals_out  = [n.astype(np.float32) for n in cache["normals"]]
                trajs_2d_out = cache["trajs_2d"]
                trajs_3d_out = cache["trajs_3d_world"]
                valids_out   = cache["valids"]
                visibs_out   = cache["visibs"]
                has_precomputed = True
            except Exception as e:
                if self.verbose:
                    print(f"[TartanAirAdapter] Warning: precomputed indexing failed "
                          f"for {sequence_name}: {e}")

        extrinsics_arr = np.stack(extrinsics, axis=0)  # [T,4,4] w2c in NED world

        # ── NED → ENU coordinate conversion ──────────────────────────────
        # TartanAir world frame: NED (x=North, y=East,  z=Down)
        # Target  world frame:   ENU (x=East,  y=North, z=Up)   — right-hand Z-up,
        # friendlier for 3-D visualisers (rerun default grid is Z=0 horizontal).
        #
        #   x_enu =  y_ned   (East  → East)
        #   y_enu =  x_ned   (North → North)
        #   z_enu = -z_ned   (Down  → Up, negated)
        #
        # For a world point p:  p_enu = R @ p_ned
        # For w2c:  w2c_enu takes an ENU world point and gives camera coords.
        #   camera = w2c_ned @ p_ned = w2c_ned @ (R^T @ p_enu)
        #   => w2c_enu = w2c_ned @ R^T  =  w2c_ned @ T_ned2enu^T
        R_ned2enu = np.array([
            [0,  1,  0],
            [1,  0,  0],
            [0,  0, -1],
        ], dtype=np.float32)
        T_ned2enu = np.eye(4, dtype=np.float32)
        T_ned2enu[:3, :3] = R_ned2enu

        # Transform extrinsics: w2c_enu = w2c_ned @ R_ned2enu^T
        extrinsics_arr = extrinsics_arr @ T_ned2enu.T  # broadcast over T frames

        # Transform 3-D tracks: p_enu = R @ p_ned
        if trajs_3d_out is not None:
            sh = trajs_3d_out.shape          # [T, N, 3]
            pts = trajs_3d_out.reshape(-1, 3).astype(np.float32)
            trajs_3d_out = (R_ned2enu @ pts.T).T.reshape(sh).astype(np.float32)

        return UnifiedClip(
            dataset_name=self.dataset_name,
            sequence_name=sequence_name,
            frame_paths=frame_paths,
            images=images,
            depths=depths if depths else None,
            normals=normals_out,
            trajs_2d=trajs_2d_out,
            trajs_3d_world=trajs_3d_out,
            valids=valids_out,
            visibs=visibs_out,
            intrinsics=np.stack(intrinsics, axis=0),
            extrinsics=extrinsics_arr,
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
        info["ok"] = info["num_frames"] > 0
        return info
