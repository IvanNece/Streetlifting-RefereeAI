import argparse
import sys
import os
import cv2
import yaml
import numpy as np
from typing import List, Optional, Dict, Any
from dip_validator.video_io import load_video, save_video
from dip_validator.pose import PoseEstimator, PoseResult
from dip_validator.phases import compute_depth_signal, smooth_signal, detect_bottom_frame, segment_phases
from dip_validator.refinement import refine_landmarks, smooth_landmarks_temporal, RefinedLandmarks
from dip_validator.rules import evaluate_dip, DipDecision
from dip_validator.reporting import generate_report

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_landmarks_trace(landmarks: List[Optional[RefinedLandmarks]]) -> List[Dict[str, Any]]:
    """Creates a serializable trace of landmark data."""
    trace = []
    for i, lm in enumerate(landmarks):
        if lm:
            trace.append({
                "frame": i,
                "deltoid": [round(lm.deltoid_apex[0], 2), round(lm.deltoid_apex[1], 2)],
                "elbow": [round(lm.elbow_tip[0], 2), round(lm.elbow_tip[1], 2)],
                "margin_px": round(lm.deltoid_apex[1] - lm.elbow_tip[1], 2),
                "deltoid_conf": round(lm.deltoid_confidence, 2),
                "elbow_conf": round(lm.elbow_confidence, 2)
            })
    return trace

def generate_overlay_video(
    frames: List[np.ndarray],
    landmarks: List[Optional[RefinedLandmarks]],
    phases: Dict[int, str],
    decision: DipDecision,
    bottom_idx: int,
    config: Dict[str, Any]
) -> List[np.ndarray]:
    """Generates frames with analysis overlay."""
    overlay_frames = []
    num_frames = len(frames)
    width = frames[0].shape[1]
    bottom_win = config['phases']['bottom_window']
    show_margin = config['output']['overlay_show_margin']

    for i, frame in enumerate(frames):
        canvas = frame.copy()
        lm = landmarks[i]
        
        # 1. Draw Landmarks
        if lm:
            d_pt = (int(lm.deltoid_apex[0]), int(lm.deltoid_apex[1]))
            e_pt = (int(lm.elbow_tip[0]), int(lm.elbow_tip[1]))
            
            cv2.line(canvas, (0, e_pt[1]), (width, e_pt[1]), (255, 0, 0), 1)
            cv2.circle(canvas, d_pt, 6, (0, 255, 0), -1)
            cv2.circle(canvas, e_pt, 6, (255, 0, 0), -1)
            
            margin_val = lm.deltoid_apex[1] - lm.elbow_tip[1]
            if abs(i - bottom_idx) <= bottom_win:
                color = (0, 255, 0) if margin_val >= 0 else (0, 0, 255)
                cv2.line(canvas, d_pt, (d_pt[0], e_pt[1]), color, 2)
                if show_margin:
                    cv2.putText(canvas, f"{margin_val:+.1f}px", (d_pt[0] + 10, (d_pt[1] + e_pt[1]) // 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 2. Draw Phase
        phase = phases.get(i, "unknown")
        p_colors = {"bottom": (0, 255, 255), "descending": (0, 165, 255), "ascending": (0, 255, 0)}
        color = p_colors.get(phase, (255, 255, 255))
        cv2.putText(canvas, f"PHASE: {phase.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # 3. Draw Decision
        if i >= bottom_idx:
            res_txt = "VALID" if decision.valid else "INVALID"
            res_col = (0, 255, 0) if decision.valid else (0, 0, 255)
            cv2.putText(canvas, res_txt, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, res_col, 3)
            
        overlay_frames.append(canvas)
        if (i + 1) % 50 == 0 or (i + 1) == num_frames:
            print(f"\rOverlay generation: {(i + 1) / num_frames * 100:.1f}%", end="", flush=True)
            
    return overlay_frames

def save_debug_images(
    output_dir: str,
    frames: List[np.ndarray],
    overlay_frames: List[np.ndarray],
    results: List[Optional[PoseResult]],
    bottom_idx: int,
    conf_thresh: float
):
    """Saves visual debug information."""
    if 0 <= bottom_idx < len(overlay_frames):
        cv2.imwrite(os.path.join(output_dir, "debug_landmarks.jpg"), overlay_frames[bottom_idx])
        
        for i, res in enumerate(results):
            if res:
                debug_pose = frames[i].copy()
                for pt, conf in zip(res.keypoints, res.confidences):
                    if conf > conf_thresh:
                        cv2.circle(debug_pose, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
                cv2.imwrite(os.path.join(output_dir, "debug_pose.jpg"), debug_pose)
                break

def main():
    parser = argparse.ArgumentParser(description="Dip Validator CLI")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
        video_output_dir = os.path.join(args.output_dir, video_basename)
        os.makedirs(video_output_dir, exist_ok=True)
        
        print(f"Processing: {os.path.basename(args.video_path)}")
        frames, meta = load_video(args.video_path)
        num_frames = len(frames)
        
        # 1. Pose Estimation
        print("Starting pose estimation...")
        mode_map = {"rtmpose-s": "lightweight", "rtmpose-m": "balanced", "rtmpose-l": "performance"}
        mode = mode_map.get(config['pose']['model'], "balanced")
        estimator = PoseEstimator(device=config['pose']['device'], mode=mode)
        
        results = []
        conf_thresh = config['pose']['confidence_threshold']
        for i, frame in enumerate(frames):
            results.append(estimator.estimate_poses([frame], conf_threshold=conf_thresh)[0])
            if (i + 1) % 20 == 0 or (i + 1) == num_frames:
                print(f"\rPose estimation: {(i + 1) / num_frames * 100:.1f}%", end="", flush=True)
        print("\nPose estimation complete.")
        
        # 2. Phase Detection
        print("Starting phase detection...")
        depth_signal = compute_depth_signal(results, conf_threshold=conf_thresh)
        smoothed = smooth_signal(depth_signal, 
                                 window=config['phases']['smoothing_window'], 
                                 polyorder=config['phases']['smoothing_polyorder'])
        bottom_idx = detect_bottom_frame(smoothed)
        phases = segment_phases(smoothed, bottom_idx, bottom_window=config['phases']['bottom_window'])
        
        # 3. Refinement
        print("Starting landmark refinement...")
        raw_l, raw_r = [], []
        ref_params = {
            "elbow_offset_ratio": config['landmarks']['elbow_offset_ratio'],
            "deltoid_offset_ratio": config['landmarks']['deltoid_offset_ratio']
        }
        for i, (f, r) in enumerate(zip(frames, results)):
            raw_l.append(refine_landmarks(f, r, "left", **ref_params))
            raw_r.append(refine_landmarks(f, r, "right", **ref_params))
        
        left_refined = smooth_landmarks_temporal(raw_l, alpha=config['landmarks']['ema_alpha'])
        right_refined = smooth_landmarks_temporal(raw_r, alpha=config['landmarks']['ema_alpha'])
        
        # 4. Decision
        print("Evaluating dip decision...")
        decision = evaluate_dip(
            left_refined, right_refined, bottom_idx, 
            window_half_size=config['phases']['bottom_window'],
            min_confidence=config['decision']['min_confidence']
        )
        
        # 5. Reporting & Trace
        trace = create_landmarks_trace(left_refined if decision.selected_side == "left" else right_refined) \
                if config['output']['save_landmarks_trace'] else None
        
        report_path = generate_report(args.video_path, decision, num_frames, meta['fps'], video_output_dir, trace)
        
        print(f"\nResult: {'VALID' if decision.valid else 'INVALID'} (Margin: {decision.best_margin_px:.1f}px)")
        print(f"Report saved: {report_path}")
        
        # 6. Overlay & Debug
        print("Generating overlay video...")
        selected_lms = left_refined if decision.selected_side == "left" else right_refined
        overlay_frames = generate_overlay_video(frames, selected_lms, phases, decision, bottom_idx, config)
        save_video(overlay_frames, os.path.join(video_output_dir, "overlay.mp4"), meta['fps'])
        print(f"\nOverlay saved to {video_output_dir}")
        
        save_debug_images(video_output_dir, frames, overlay_frames, results, bottom_idx, conf_thresh)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
