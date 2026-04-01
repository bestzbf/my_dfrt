from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


def load_precomputed_fast(
    npz_path: Path,
    frame_indices: list[int],
) -> Optional[dict]:
    """
    Load precomputed tracks/normals for specific frame indices.

    Prefers .h5 (chunked HDF5, O(frames) random access) over .npz
    (requires full zlib decompression of the entire array).
    Falls back to .npz if .h5 is not found.

    Returns a dict with arrays already indexed to frame_indices order,
    or None if neither .h5 nor .npz exists.

    Run computer/convert_precomputed_to_h5.py once to generate .h5 files.
    """
    npz_path = Path(npz_path)
    h5_path = npz_path.with_suffix('.h5')

    if h5_path.exists():
        import h5py
        # h5py fancy indexing requires sorted unique indices
        sorted_idx = sorted(set(frame_indices))
        idx_map = {v: i for i, v in enumerate(sorted_idx)}
        reorder = [idx_map[i] for i in frame_indices]
        needs_reorder = reorder != list(range(len(frame_indices)))

        result: dict = {}
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                ds = f[key]
                if ds.ndim >= 1 and ds.shape[0] > 1:
                    data = ds[sorted_idx]       # reads only the needed chunks
                    if needs_reorder:
                        data = data[reorder]
                    result[key] = data
                else:
                    result[key] = ds[()]        # scalar / metadata
        return result

    elif npz_path.exists():
        raw = np.load(npz_path, allow_pickle=True)
        result = {}
        for k in raw.files:
            arr = raw[k]
            if arr.ndim >= 1 and arr.shape[0] > 1:
                result[k] = arr[np.array(frame_indices)]
            else:
                result[k] = arr[()]
        return result

    return None


@dataclass
class UnifiedClip:
    dataset_name: str
    sequence_name: str
    frame_paths: Optional[list[str]]
    images: list[np.ndarray]                  # [T][H,W,3]
    depths: Optional[list[np.ndarray]]        # [T][H,W]
    normals: Optional[list[np.ndarray]]       # [T][H,W,3]
    trajs_2d: Optional[np.ndarray]            # [T,N,2]
    trajs_3d_world: Optional[np.ndarray]      # [T,N,3]
    valids: Optional[np.ndarray]              # [T,N]
    visibs: Optional[np.ndarray]              # [T,N]
    intrinsics: np.ndarray                    # [T,3,3]
    extrinsics: np.ndarray                    # [T,4,4]
    flows: Optional[list[np.ndarray]] = None  # [T][H,W,2], optical flow
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return len(self.images)

    @property
    def image_size(self) -> tuple[int, int]:
        h, w = self.images[0].shape[:2]
        return h, w


class BaseAdapter(ABC):
    dataset_name: str = "base"

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def list_sequences(self) -> list[str]:
        pass

    @abstractmethod
    def get_sequence_name(self, index: int) -> str:
        pass

    @abstractmethod
    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        pass

    @abstractmethod
    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        pass

    @abstractmethod
    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        pass