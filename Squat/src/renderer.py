"""Write annotated output video with keypoint overlay and verdict."""
import cv2
import numpy as np

from src.depth_analyzer import HIP_CREASE_ANTERIOR_RATIO
from src.utils import get_side_keypoints

_COLORS = {
    "GOOD_LIFT": (0, 200, 0),
    "NO_LIFT":   (0, 0, 220),
    "UNCERTAIN": (0, 200, 220),
}


def render(
    input_path: str,
    output_path: str,
    poses: list,
    reps: list,
    verdicts: list,
    side: str = "right",
) -> None:
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    poses_by_frame = {p.frame_idx: p for p in poses}
    verdict_by_rep = {v.rep_n: v for v in verdicts}
    bottom_to_rep = {r.frame_bottom: r for r in reps}
    display_frames = int(fps)  # show semaphore for ~1 second

    active: list[tuple[int, str]] = []   # (expire_frame, result)
    completed: list[tuple[int, str]] = []  # (rep_n, result)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Trigger verdict at rep bottom
        if frame_idx in bottom_to_rep:
            rep = bottom_to_rep[frame_idx]
            v = verdict_by_rep.get(rep.rep_n)
            if v:
                active.append((frame_idx + display_frames, v.result))
                completed.append((rep.rep_n, v.result))

        active = [(exp, res) for exp, res in active if frame_idx <= exp]

        # Draw keypoints and reference lines
        pose = poses_by_frame.get(frame_idx)
        if pose is not None:
            kp = get_side_keypoints(pose, side)
            if kp is not None:
                for pt in kp.values():
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, (255, 255, 255), -1)

                # Hip crease reference line
                hip = kp["hip"].astype(float)
                knee = kp["knee"].astype(float)
                limb_vec = knee - hip
                limb_len = np.linalg.norm(limb_vec)
                if limb_len > 1e-6:
                    perp = np.array([limb_vec[1], -limb_vec[0]])
                    perp_norm = perp / np.linalg.norm(perp)
                    hip_crease = hip + perp_norm * limb_len * HIP_CREASE_ANTERIOR_RATIO
                    _dashed_hline(frame, int(hip_crease[1]), (200, 200, 200))
                _dashed_hline(frame, int(knee[1]), (150, 150, 255))

        # Semaphore circle
        for _, result in active:
            color = _COLORS[result]
            cv2.circle(frame, (50, 50), 30, color, -1)
            cv2.putText(frame, result.replace("_", " "), (90, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Permanent rep summary (bottom of frame)
        for i, (rep_n, result) in enumerate(completed):
            color = _COLORS[result]
            text = f"REP {rep_n}: {result.replace('_', ' ')}"
            cv2.putText(frame, text, (10, h - 15 - i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


def _dashed_hline(frame: np.ndarray, y: int, color: tuple, dash: int = 10) -> None:
    w = frame.shape[1]
    for x in range(0, w, dash * 2):
        cv2.line(frame, (x, y), (min(x + dash, w - 1), y), color, 1)
