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
        load_precomputed: bool = True,
        verbose: bool = True,
        strict: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            root:             TartanAir root directory (contains scene/Easy|Hard/P00X/).
            split:            Dataset split tag (informational only).
            camera:           Which camera to use: 'left' or 'right'.
            precompute_root:  If set, load precomputed normals/tracks from
                              <precompute_root>/<seq_name>/precomputed.npz.
            verbose:          Print loading info.
            cache_dir:        If set, cache sequence index to avoid slow COS iterdir().
        """
        self.root = Path(root)
        self.split = split
        self.camera = camera
        self.load_precomputed = bool(load_precomputed)
        self.precompute_root = (
            Path(precompute_root) if precompute_root and self.load_precomputed
            else (self.root if self.load_precomputed else None)
        )
        self.verbose = verbose

        # TartanAir standard intrinsics (640x480)
        self.K = np.array([
            [320.0, 0.0,   320.0],
            [0.0,   320.0, 240.0],
            [0.0,   0.0,   1.0]
        ], dtype=np.float32)

        if cache_dir is not None:
            from datasets.index_cache import load_or_build
            import hashlib, json
            cache_key = {"dataset": self.dataset_name, "split": self.split, "camera": self.camera, "cache_schema": 1}
            suffix = hashlib.sha1(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()[:12]
            cache_path = Path(cache_dir) / f"{self.dataset_name}_{self.split}_{suffix}.pkl"
            index: list[tuple[str, int]] = load_or_build(self._build_index, cache_path)
        else:
            index = self._build_index()

        self.sequences: list[str] = [name for name, _ in index]
        self.seq_to_dir: dict[str, Path] = {name: self.root / name for name in self.sequences}
        self._num_frames_map: dict[str, int] = {name: nf for name, nf in index}

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

    def _build_index(self) -> list[tuple[str, int]]:
        results: list[tuple[str, int]] = []
        for scene_dir in sorted(self.root.iterdir()):
            if not scene_dir.is_dir():
                continue
            # Two-level layout: scene/difficulty/P00X  (TartanAirV1)
            difficulty_dirs = sorted(
                d for d in scene_dir.iterdir()
                if d.is_dir() and d.name not in (f"image_{self.camera}", f"depth_{self.camera}", "flow")
            )
            candidates = []
            for d in difficulty_dirs:
                for p in sorted(d.iterdir()):
                    if p.is_dir() and (p / f"image_{self.camera}").exists():
                        candidates.append((f"{scene_dir.name}/{d.name}/{p.name}", p))
            if not candidates and (scene_dir / f"image_{self.camera}").exists():
                candidates = [(scene_dir.name, scene_dir)]
            for seq_name, seq_dir in candidates:
                pose_file = seq_dir / f"pose_{self.camera}.txt"
                try:
                    with open(pose_file, 'r') as f:
                        nf = sum(1 for line in f if line.strip())
                except Exception:
                    nf = 0
                if nf > 0:
                    results.append((seq_name, nf))
        return results

    def _get_valid_frame_list(self, seq_dir: Path, seq_name: str) -> Optional[list[int]]:
        """Return sorted list of usable frame indices for this sequence.

        When a precomputed cache exists, returns only the contiguous run of frames
        starting from ref_frame that have at least MIN_VALID_PTS valid+visible points.
        This ensures the sampler always picks frames with dense track coverage.
        Falls back to all image frames when no cache is found.
        """
        MIN_VALID_PTS = 5
        h5_path = seq_dir / "precomputed.h5"
        npz_path = seq_dir / "precomputed.npz"
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
        if not self.load_precomputed or self.precompute_root is None:
            return None
        from datasets.adapters.base import load_precomputed_fast
        cache_path = self.precompute_root / sequence_name / "precomputed.npz"
        h5_path = cache_path.with_suffix(".h5")
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
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "sequence_name": sequence_name,
            "num_frames": self.get_num_frames(sequence_name),
            "has_depth": True,
            "has_flow": False,
            "has_normals": False,
            "has_tracks": False,
            "has_visibility": False,
        }

    def get_num_frames(self, sequence_name: str) -> int:
        return self._num_frames_map.get(sequence_name, 0)

    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        seq_dir = self.seq_to_dir[sequence_name]

        img_dir   = seq_dir / f"image_{self.camera}"
        depth_dir = seq_dir / f"depth_{self.camera}"
        pose_file = seq_dir / f"pose_{self.camera}.txt"
        flow_dir  = seq_dir / "flow"

        actual_indices = frame_indices

        with open(pose_file, 'r') as f:
            all_pose_lines = f.readlines()

        images, depths, extrinsics, intrinsics, flows = [], [], [], [], []
        frame_paths = []

        for idx in actual_indices:
            # Direct path construction — avoids slow glob on COS
            img_path = img_dir / f"{idx:06d}_{self.camera}.png"
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            frame_paths.append(str(img_path))

            # Load depth
            depth_path = depth_dir / f"{idx:06d}_{self.camera}_depth.npy"
            try:
                depth = np.load(str(depth_path)).astype(np.float32)
                depth = np.where(depth > 200.0, np.nan, depth)
            except Exception:
                depth = None
            depths.append(depth)

            # Parse pose
            # TartanAir pose file stores camera-to-world (c2w) transforms:
            # TartanAir pose: c2w in NED world frame (tx ty tz qx qy qz qw)
            if idx < len(all_pose_lines):
                line = all_pose_lines[idx].strip().split()
                if len(line) == 7:
                    tx, ty, tz, qx, qy, qz, qw = map(float, line)
                    R = quat_to_rotation_matrix([qx, qy, qz, qw])
                    t = np.array([tx, ty, tz], dtype=np.float32)
                    # Invert c2w -> w2c (still in NED world; converted later)
                    w2c = np.eye(4, dtype=np.float32)
                    w2c[:3, :3] = R.T
                    w2c[:3, 3]  = -R.T @ t
                    extrinsics.append(w2c)

            intrinsics.append(self.K.copy())

            flows.append(None)

        R_ned2enu = np.array([[0,1,0],[1,0,0],[0,0,-1]], dtype=np.float32)
        R_ned2cv  = np.array([[0,1,0],[0,0,1],[1,0,0]], dtype=np.float32)
        R_edn2enu = np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=np.float32)

        # ---- precomputed normals / tracks ----
        normals_out    = None
        trajs_2d_out   = None
        trajs_3d_out   = None
        valids_out     = None
        visibs_out     = None
        has_precomputed = False
        trajs_3d_source = None

        cache = self._load_precomputed(sequence_name, actual_indices)
        if cache is not None:
            try:
                normals_out  = [n.astype(np.float32) for n in cache["normals"]]
                trajs_2d_out = cache["trajs_2d"]
                if "trajs_3d_world" in cache:
                    trajs_3d_cache = cache["trajs_3d_world"].astype(np.float32)
                    sh = trajs_3d_cache.shape
                    pts = trajs_3d_cache.reshape(-1, 3)
                    trajs_3d_out = (R_edn2enu @ pts.T).T.reshape(sh).astype(np.float32)
                    trajs_3d_source = "precomputed_trajs_3d_world_edn_to_enu"
                valids_out   = cache["valids"]
                visibs_out   = cache["visibs"]
                has_precomputed = True
            except Exception as e:
                if self.verbose:
                    print(f"[TartanAirAdapter] Warning: precomputed indexing failed "
                          f"for {sequence_name}: {e}")

        extrinsics_arr = np.stack(extrinsics, axis=0)  # [T,4,4] w2c in NED world

        # Fallback for older caches that do not contain trajs_3d_world:
        # recompute from trajs_2d + depth (OpenCV cam coords -> NED world).
        if trajs_3d_out is None and trajs_2d_out is not None and len(depths) > 0:
            T_frames = len(actual_indices)
            N = trajs_2d_out.shape[1]
            trajs_3d_ned = np.zeros((T_frames, N, 3), dtype=np.float32)
            for ti in range(T_frames):
                uv = trajs_2d_out[ti]  # [N,2] pixel coords in original resolution
                K_t = intrinsics[ti]
                E_t = extrinsics_arr[ti]  # w2c NED
                depth_t = depths[ti] if ti < len(depths) else None
                if depth_t is None:
                    continue
                H, W = depth_t.shape
                xi = np.clip(np.round(uv[:,0]).astype(np.int32), 0, W-1)
                yi = np.clip(np.round(uv[:,1]).astype(np.int32), 0, H-1)
                d = depth_t[yi, xi]
                # OpenCV unproject: Z=depth
                fx, fy = K_t[0,0], K_t[1,1]
                cx, cy = K_t[0,2], K_t[1,2]
                pts_cam = np.stack([(uv[:,0]-cx)/fx*d, (uv[:,1]-cy)/fy*d, d], axis=-1)
                # OpenCV cam -> NED cam -> NED world
                R_ned2cv = np.array([[0,1,0],[0,0,1],[1,0,0]], dtype=np.float32)
                pts_cam_ned = (R_ned2cv.T @ pts_cam.T).T  # OpenCV -> NED camera
                R = E_t[:3,:3]; t_w = E_t[:3,3]
                R_c2w = R.T; t_c2w = -R.T @ t_w
                pts_world = (R_c2w @ pts_cam_ned.T).T + t_c2w
                trajs_3d_ned[ti] = pts_world
            sh = trajs_3d_ned.shape
            pts = trajs_3d_ned.reshape(-1, 3).astype(np.float32)
            trajs_3d_out = (R_ned2enu @ pts.T).T.reshape(sh).astype(np.float32)
            trajs_3d_source = "depth_recomputed_ned_to_enu"

        # TartanAir uses NED world frame and NED camera frame.
        # Convert to: ENU world frame + OpenCV camera frame (x=right, y=down, z=fwd).
        #
        # R_ned2enu: NED world → ENU world  [[0,1,0],[1,0,0],[0,0,-1]]
        # R_ned2cv:  NED cam  → OpenCV cam  [[0,1,0],[0,0,1],[1,0,0]]
        #
        # Full extrinsic: p_cam_cv = R_ned2cv @ w2c_ned @ R_ned2enu^T @ p_enu
        T_left  = np.eye(4, dtype=np.float32); T_left[:3,:3]  = R_ned2cv
        T_right = np.eye(4, dtype=np.float32); T_right[:3,:3] = R_ned2enu.T  # = R_enu2ned

        extrinsics_arr = T_left @ extrinsics_arr @ T_right  # [T,4,4]

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
                "trajs_3d_source": trajs_3d_source,
                "world_convention": "enu",
                "pose_convention": "world_to_camera",
                "intrinsics_convention": "pinhole",
                "depth_unit": "meters",
            }
        )

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        info = self.get_sequence_info(sequence_name)
        info["ok"] = info["num_frames"] > 0
        return info
