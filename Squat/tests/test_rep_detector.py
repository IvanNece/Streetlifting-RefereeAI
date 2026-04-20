"""Tests for rep_detector.py — AAA pattern."""
import numpy as np
import pytest

from src.pose_estimator import FramePose
from src.rep_detector import detect_reps


def _make_poses(hip_y_values: list[float], score: float = 0.9) -> list[FramePose]:
    """FramePose list from hip_y trajectory (right side, vertical limb)."""
    poses = []
    for i, y in enumerate(hip_y_values):
        kp = np.zeros((17, 2), dtype=float)
        sc = np.full(17, score)
        kp[12] = (100, y)    # right_hip
        kp[14] = (100, 300)  # right_knee (fixed)
        kp[16] = (100, 400)  # right_ankle (fixed)
        poses.append(FramePose(frame_idx=i, keypoints=kp, scores=sc))
    return poses


def _squat_curve(n_reps: int, fps: float = 30.0) -> list[float]:
    """Synthetic hip_y: standing=100, bottom=300, n_reps squats."""
    frames_per_half = int(fps)  # 1s down, 1s up
    curve: list[float] = []
    for _ in range(n_reps):
        curve.extend(np.linspace(100, 300, frames_per_half).tolist())
        curve.extend(np.linspace(300, 100, frames_per_half).tolist())
    curve.extend([100.0] * int(fps))  # trailing standing frames
    return curve


def test_three_reps():
    # Arrange
    fps = 30.0
    poses = _make_poses(_squat_curve(3, fps))
    # Act
    reps = detect_reps(poses, fps)
    # Assert
    assert len(reps) == 3
    for i, rep in enumerate(reps):
        assert rep.rep_n == i + 1
        assert rep.frame_start < rep.frame_bottom
        assert rep.frame_bottom < rep.frame_end


def test_one_rep():
    # Arrange
    fps = 30.0
    poses = _make_poses(_squat_curve(1, fps))
    # Act
    reps = detect_reps(poses, fps)
    # Assert
    assert len(reps) == 1
    assert reps[0].rep_n == 1


def test_no_squat():
    # Arrange: flat curve — no depth change
    poses = _make_poses([100.0] * 90)
    # Act
    reps = detect_reps(poses, fps=30.0)
    # Assert
    assert reps == []


def test_low_confidence_all_frames():
    # Arrange: score below MIN_CONFIDENCE → all frames skipped
    poses = _make_poses(_squat_curve(2), score=0.1)
    # Act
    reps = detect_reps(poses, fps=30.0)
    # Assert
    assert reps == []
