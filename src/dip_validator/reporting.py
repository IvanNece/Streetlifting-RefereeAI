import json
import os
from typing import Dict, Any
from .rules import DipDecision

def generate_report(
    video_path: str,
    decision: DipDecision,
    num_frames: int,
    fps: float,
    output_dir: str = "output",
    landmarks_trace: list = None
) -> str:
    """
    Generates a JSON report for the dip analysis.
    
    Args:
        video_path: Path to the input video.
        decision: DipDecision object.
        num_frames: Total number of frames analyzed.
        fps: Frames per second of the video.
        output_dir: Directory where the report.json will be saved.
        landmarks_trace: Optional list of per-frame landmark data.
        
    Returns:
        str: Path to the generated JSON report.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    report_path = os.path.join(output_dir, "report.json")
    
    report_data = {
        "video": os.path.basename(video_path),
        "result": "VALID" if decision.valid else "INVALID",
        "margin_px": round(decision.margin_px, 2),
        "best_margin_px": round(decision.best_margin_px, 2),
        "selected_side": decision.selected_side,
        "bottom_frame_index": decision.bottom_frame_index,
        "confidence": round(decision.confidence, 2),
        "warnings": decision.warnings,
        "frames_analyzed": num_frames,
        "fps": round(fps, 2)
    }

    if landmarks_trace is not None:
        report_data["landmarks_trace"] = landmarks_trace
    
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
        
    return report_path
