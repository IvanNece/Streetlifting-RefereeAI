import pytest
from dip_validator.rules import evaluate_dip, DipDecision
from dip_validator.refinement import RefinedLandmarks

def create_mock_landmark(y_d, y_e, conf=0.8, angle_warn=False):
    return RefinedLandmarks(
        elbow_tip=(100.0, float(y_e)),
        deltoid_apex=(100.0, float(y_d)),
        elbow_confidence=conf,
        deltoid_confidence=conf,
        side="left",
        angle_warning=angle_warn
    )

def test_evaluate_dip_valid():
    # Shoulder (D) below Elbow (E) -> y_D > y_E
    # Frame 10 is detected bottom. Margin = 120 - 110 = 10
    landmarks = [None] * 20
    for i in range(5, 16):
        landmarks[i] = create_mock_landmark(y_d=120, y_e=110)
    
    # Right side is empty/low conf
    right_landmarks = [None] * 20
    
    decision = evaluate_dip(landmarks, right_landmarks, detected_bottom_idx=10)
    
    assert decision.valid is True
    assert decision.margin_px == 10.0
    assert decision.best_margin_px == 10.0
    assert decision.selected_side == "left"
    assert "angle_warning" not in decision.warnings

def test_evaluate_dip_invalid():
    # Shoulder (D) above Elbow (E) -> y_D < y_E
    # Even best margin is negative
    landmarks = [None] * 20
    for i in range(5, 16):
        landmarks[i] = create_mock_landmark(y_d=100, y_e=110)
        
    decision = evaluate_dip(landmarks, [None]*20, detected_bottom_idx=10)
    
    assert decision.valid is False
    assert decision.margin_px == -10.0
    assert decision.best_margin_px == -10.0

def test_evaluate_dip_best_margin():
    # Invalid at detected bottom (frame 10), but VALID at frame 12
    # Logic should pick frame 12 as the true bottom
    landmarks = [None] * 20
    for i in range(5, 16):
        if i == 12:
            landmarks[i] = create_mock_landmark(y_d=115, y_e=110) # Margin +5 (VALID)
        else:
            landmarks[i] = create_mock_landmark(y_d=100, y_e=110) # Margin -10 (INVALID)
            
    decision = evaluate_dip(landmarks, [None]*20, detected_bottom_idx=10)
    
    assert decision.valid is True
    assert decision.margin_px == 5.0
    assert decision.best_margin_px == 5.0
    assert decision.bottom_frame_index == 12 # Found the deeper frame

def test_evaluate_dip_side_selection():
    left_landmarks = [None] * 20
    right_landmarks = [None] * 20
    
    for i in range(5, 16):
        left_landmarks[i] = create_mock_landmark(120, 110, conf=0.5)
        right_landmarks[i] = create_mock_landmark(120, 110, conf=0.9)
        
    decision = evaluate_dip(left_landmarks, right_landmarks, detected_bottom_idx=10)
    
    assert decision.selected_side == "right"
    assert decision.confidence == pytest.approx(0.9)

def test_evaluate_dip_angle_warning():
    landmarks = [None] * 20
    for i in range(5, 16):
        landmarks[i] = create_mock_landmark(120, 110)
        
    # Add warning near the best frame (which will be frame 10 since all equal)
    landmarks[10] = create_mock_landmark(120, 110, angle_warn=True)
        
    decision = evaluate_dip(landmarks, [None]*20, detected_bottom_idx=10)
    assert "angle_warning" in decision.warnings