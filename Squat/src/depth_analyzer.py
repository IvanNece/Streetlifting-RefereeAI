"""IPF depth judgment — hip crease vs knee."""
import logging
from dataclasses import dataclass

import numpy as np

from src.utils import get_side_keypoints

HIP_CREASE_ANTERIOR_RATIO = 0.20   # DA CALIBRARE su dataset reale
UNCERTAIN_ZONE_PX = 5


@dataclass
class Verdict:
    rep_n: int
    result: str    # "GOOD_LIFT" | "NO_LIFT" | "UNCERTAIN"
    delta_px: float


def _detect_facing(poses: list, side: str) -> int:
    """Detect which direction athlete faces: +1=right, -1=left.

    Uses average horizontal offset knee_x - hip_x across all valid frames.
    When facing right the knee is forward (larger x) relative to the hip, and vice versa.
    """
    diffs = [
        float(kp["knee"][0] - kp["hip"][0])
        for pose in poses
        if (kp := get_side_keypoints(pose, side)) is not None
    ]
    if not diffs:
        return 1
    facing = 1 if np.mean(diffs) >= 0 else -1
    logging.info("Detected athlete facing: %s", "right" if facing == 1 else "left")
    return facing


def judge_rep(rep, poses_by_frame: dict, side: str = "right", facing: int = 1) -> Verdict | None:
    """Judge depth at frame_bottom. Returns None if keypoints missing or low confidence."""
    pose = poses_by_frame.get(rep.frame_bottom)
    if pose is None:
        logging.warning("Rep %d: frame_bottom %d not found in poses", rep.rep_n, rep.frame_bottom)
        return None

    kp = get_side_keypoints(pose, side)
    if kp is None:
        logging.warning("Rep %d: low-confidence keypoints at frame_bottom", rep.rep_n)
        return None

    hip = kp["hip"].astype(float)
    knee = kp["knee"].astype(float)

    limb_vec = knee - hip
    limb_len = np.linalg.norm(limb_vec)
    if limb_len < 1e-6:
        logging.warning("Rep %d: degenerate limb vector (hip == knee)", rep.rep_n)
        return None

    perp = np.array([limb_vec[1], -limb_vec[0]])
    perp_norm = perp / np.linalg.norm(perp)

    # Guarantee perp_norm points ANTERIOR (toward the direction the athlete faces).
    # At the bottom of a deep squat limb_vec.y can be negative (hip below knee in image),
    # which flips the clockwise rotation to the posterior side — we must correct for this.
    if perp_norm[0] * facing < 0:
        perp_norm = -perp_norm

    hip_crease = hip + perp_norm * limb_len * HIP_CREASE_ANTERIOR_RATIO

    # positive delta = hip_crease lower than knee in image = good depth
    delta = float(hip_crease[1] - knee[1])

    if delta > UNCERTAIN_ZONE_PX:
        result = "GOOD_LIFT"
    elif delta < -UNCERTAIN_ZONE_PX:
        result = "NO_LIFT"
    else:
        result = "UNCERTAIN"

    return Verdict(rep_n=rep.rep_n, result=result, delta_px=delta)


def analyze_depth(reps: list, poses: list, side: str = "right") -> list[Verdict]:
    """Judge depth for all reps. Returns only successful verdicts."""
    poses_by_frame = {p.frame_idx: p for p in poses}
    facing = _detect_facing(poses, side)
    return [v for rep in reps if (v := judge_rep(rep, poses_by_frame, side, facing)) is not None]
