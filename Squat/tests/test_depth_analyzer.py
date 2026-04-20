"""Tests for depth_analyzer.py — AAA pattern."""
import numpy as np
import pytest

from src.depth_analyzer import UNCERTAIN_ZONE_PX, judge_rep
from src.pose_estimator import FramePose
from src.rep_detector import Rep


def _make_pose(frame_idx: int, hip_xy: tuple, knee_xy: tuple, score: float = 0.9) -> FramePose:
    """FramePose with controlled right-side hip/knee. Ankle set below knee."""
    kp = np.zeros((17, 2), dtype=float)
    sc = np.zeros(17, dtype=float)
    kp[12], sc[12] = hip_xy, score   # right_hip
    kp[14], sc[14] = knee_xy, score  # right_knee
    kp[16] = (knee_xy[0], knee_xy[1] + 100)
    sc[16] = score                   # right_ankle
    return FramePose(frame_idx=frame_idx, keypoints=kp, scores=sc)


def _make_rep(frame_bottom: int) -> Rep:
    return Rep(rep_n=1, frame_start=0, frame_bottom=frame_bottom, frame_end=frame_bottom + 10)


# When limb is vertical (same x), hip_crease_y == hip_y.
# So delta = hip_y - knee_y.
# knee at y=200 throughout.

def test_good_lift():
    # Arrange: hip at y=215, knee at y=200 → delta = +15 > 5
    pose = _make_pose(10, hip_xy=(100, 215), knee_xy=(100, 200))
    rep = _make_rep(10)
    # Act
    v = judge_rep(rep, {10: pose})
    # Assert
    assert v is not None
    assert v.result == "GOOD_LIFT"
    assert v.delta_px > UNCERTAIN_ZONE_PX


def test_no_lift():
    # Arrange: hip at y=190, knee at y=200 → delta = -10 < -5
    pose = _make_pose(10, hip_xy=(100, 190), knee_xy=(100, 200))
    rep = _make_rep(10)
    # Act
    v = judge_rep(rep, {10: pose})
    # Assert
    assert v is not None
    assert v.result == "NO_LIFT"
    assert v.delta_px < -UNCERTAIN_ZONE_PX


def test_uncertain():
    # Arrange: hip at y=202, knee at y=200 → delta = +2, within zone
    pose = _make_pose(10, hip_xy=(100, 202), knee_xy=(100, 200))
    rep = _make_rep(10)
    # Act
    v = judge_rep(rep, {10: pose})
    # Assert
    assert v is not None
    assert v.result == "UNCERTAIN"
    assert -UNCERTAIN_ZONE_PX <= v.delta_px <= UNCERTAIN_ZONE_PX


def test_low_confidence_keypoints():
    # Arrange: score below MIN_CONFIDENCE → skip
    pose = _make_pose(10, hip_xy=(100, 215), knee_xy=(100, 200), score=0.1)
    rep = _make_rep(10)
    # Act
    v = judge_rep(rep, {10: pose})
    # Assert: no crash, returns None
    assert v is None


def test_missing_frame():
    # Arrange: frame_bottom not in poses dict
    rep = _make_rep(99)
    # Act
    v = judge_rep(rep, {})
    # Assert: no crash, returns None
    assert v is None
