#!/usr/bin/env python3
"""Regression tests for Co3Dv2 precomputed-track temporal validity guards."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets.adapters.co3dv2 import (
    _precomputed_tracks_look_temporal,
    _summarize_precomputed_track_validity,
)


def test_single_frame_tracks_are_rejected():
    valids = np.zeros((48, 8), dtype=bool)
    valids[7, :] = True

    stats = _summarize_precomputed_track_validity(valids)
    assert stats["mean_visible_frames"] == 1.0
    assert stats["tracks_ge_2_ratio"] == 0.0
    assert not _precomputed_tracks_look_temporal(valids)


def test_multi_frame_tracks_are_accepted():
    valids = np.zeros((48, 8), dtype=bool)
    valids[3:9, :] = True

    stats = _summarize_precomputed_track_validity(valids)
    assert stats["mean_visible_frames"] == 6.0
    assert stats["tracks_ge_2_ratio"] == 1.0
    assert _precomputed_tracks_look_temporal(valids)


if __name__ == "__main__":
    test_single_frame_tracks_are_rejected()
    test_multi_frame_tracks_are_accepted()
    print("ok")
