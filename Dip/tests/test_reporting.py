import json
import os
import pytest
from dip_validator.reporting import generate_report
from dip_validator.rules import DipDecision

def test_generate_report(tmp_path):
    output_dir = tmp_path / "output"
    video_path = "test_video.mp4"
    
    decision = DipDecision(
        valid=True,
        margin_px=15.5,
        best_margin_px=15.5,
        selected_side="right",
        bottom_frame_index=50,
        confidence=0.85,
        warnings=["angle_warning"]
    )
    
    report_path = generate_report(
        video_path=video_path,
        decision=decision,
        num_frames=100,
        fps=30.0,
        output_dir=str(output_dir)
    )
    
    assert os.path.exists(report_path)
    assert os.path.basename(report_path) == "report.json"    
    
    with open(report_path, "r") as f:
        data = json.load(f)
        
    assert data["video"] == "test_video.mp4"
    assert data["result"] == "VALID"
    assert data["margin_px"] == 15.5
    assert data["best_margin_px"] == 15.5
    assert data["selected_side"] == "right"
    assert data["bottom_frame_index"] == 50
    assert data["confidence"] == 0.85
    assert "angle_warning" in data["warnings"]
    assert data["frames_analyzed"] == 100
    assert data["fps"] == 30.0
