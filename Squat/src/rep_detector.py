"""Detect squat reps from a sequence of FramePose."""
import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from src.utils import OneEuroFilter, get_side_keypoints

PROMINENCE = 30       # min pixel drop for a valid squat bottom
MIN_DISTANCE_RATIO = 0.5  # min gap between bottoms = fps * this


@dataclass
class Rep:
    rep_n: int
    frame_start: int
    frame_bottom: int
    frame_end: int


def detect_reps(poses: list, fps: float, side: str = "right") -> list[Rep]:
    """Detect reps from FramePose list. Returns list of Rep."""
    if not poses:
        return []

    f = OneEuroFilter(min_cutoff=1.0, beta=0.1)
    hip_y: list[float] = []
    frame_idxs: list[int] = []

    for i, pose in enumerate(poses):
        kp = get_side_keypoints(pose, side)
        if kp is None:
            continue
        y_smooth = f(float(kp["hip"][1]), i / fps)
        hip_y.append(y_smooth)
        frame_idxs.append(pose.frame_idx)

    if len(hip_y) < 3:
        logging.warning("Not enough valid frames to detect reps")
        return []

    arr = np.array(hip_y)
    min_dist = max(1, int(fps * MIN_DISTANCE_RATIO))

    # In image coords y grows downward: squat bottom = max hip_y
    peaks, _ = find_peaks(arr, prominence=PROMINENCE, distance=min_dist)

    if len(peaks) == 0:
        logging.warning("No squat bottoms detected")
        return []

    reps = []
    for i, pk in enumerate(peaks):
        # frame_start: last standing position before descent (min hip_y before peak)
        left_start = int(np.argmin(arr[:pk])) if pk > 0 else 0
        # frame_end: next standing position after ascent (min hip_y after peak)
        right = arr[pk + 1:]
        right_end = pk + 1 + int(np.argmin(right)) if len(right) > 0 else len(arr) - 1

        reps.append(Rep(
            rep_n=i + 1,
            frame_start=frame_idxs[left_start],
            frame_bottom=frame_idxs[pk],
            frame_end=frame_idxs[right_end],
        ))

    return reps
