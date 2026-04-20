"""AI Squat Referee — entry point.

Usage:
    python main.py video.mp4 [--side left|right]
"""
import argparse
import logging
import os
import time

import cv2

from src.depth_analyzer import analyze_depth
from src.pose_estimator import PoseEstimator
from src.rep_detector import detect_reps
from src.renderer import render

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Squat Referee (IPF standard)")
    parser.add_argument("video", help="Input video path (.mp4 / .mov)")
    parser.add_argument("--side", choices=["left", "right"], default="right",
                        help="Side facing the camera (default: right)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        logging.error("Video not found: %s", args.video)
        return

    base = os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs("output", exist_ok=True)
    output_path = f"output/{base}_analyzed.mp4"

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Phase 1 — pose estimation
    logging.info("Estimating pose (%d frames at %.1f fps)...", total_frames, fps)
    t0 = time.perf_counter()
    estimator = PoseEstimator()
    poses = []
    cap = cv2.VideoCapture(args.video)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pose = estimator.estimate(frame, frame_idx)
        if pose is not None:
            poses.append(pose)
        frame_idx += 1
    cap.release()
    logging.info("Pose done in %.1fs (%d/%d frames valid)",
                 time.perf_counter() - t0, len(poses), total_frames)

    if not poses:
        logging.error("No poses detected — check video orientation and lighting")
        return

    # Phase 2 — rep detection
    reps = detect_reps(poses, fps, side=args.side)
    if not reps:
        logging.error("No reps detected — try adjusting prominence/distance params")
        return
    logging.info("%d rep(s) detected", len(reps))

    # Phase 3 — depth analysis
    verdicts = analyze_depth(reps, poses, side=args.side)

    # Phase 4 — render
    logging.info("Rendering output video...")
    render(args.video, output_path, poses, reps, verdicts, side=args.side)
    logging.info("Saved: %s", output_path)

    # Report
    print("\n=== SQUAT REFEREE REPORT ===")
    for v in verdicts:
        sign = "+" if v.delta_px >= 0 else ""
        print(f"REP {v.rep_n}: {v.result:<12}  (delta: {sign}{v.delta_px:.1f} px)")
    print("============================\n")


if __name__ == "__main__":
    main()
