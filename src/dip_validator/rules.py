from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np
from .refinement import RefinedLandmarks

@dataclass
class DipDecision:
    valid: bool
    margin_px: float  # Margin at the best frame (deepest point)
    best_margin_px: float  # Maximum margin found (should be same as margin_px)
    selected_side: str  # "left" or "right"
    bottom_frame_index: int # Frame index where best margin occurred
    confidence: float
    warnings: List[str]

def evaluate_dip(
    left_landmarks: List[Optional[RefinedLandmarks]],
    right_landmarks: List[Optional[RefinedLandmarks]],
    detected_bottom_idx: int,
    window_half_size: int = 5,
    min_confidence: float = 0.3
) -> DipDecision:
    """
    Evaluates the dip depth based on refined landmarks.
    
    Args:
        left_landmarks: List of refined landmarks for the left side.
        right_landmarks: List of refined landmarks for the right side.
        detected_bottom_idx: Index of the detected bottom frame (from phase detection).
        window_half_size: Half-size of the window to analyze for side selection.
        min_confidence: Minimum confidence threshold for warnings.
        
    Returns:
        DipDecision: The final decision object.
    """
    num_frames = len(left_landmarks)
    
    # 1. Side Selection (still use the detected bottom window for stability check)
    start_idx = max(0, detected_bottom_idx - window_half_size)
    end_idx = min(num_frames - 1, detected_bottom_idx + window_half_size)
    
    left_conf = []
    right_conf = []
    
    for i in range(start_idx, end_idx + 1):
        if left_landmarks[i]:
            left_conf.append((left_landmarks[i].elbow_confidence + left_landmarks[i].deltoid_confidence) / 2)
        if right_landmarks[i]:
            right_conf.append((right_landmarks[i].elbow_confidence + right_landmarks[i].deltoid_confidence) / 2)
            
    avg_left_conf = np.mean(left_conf) if left_conf else 0.0
    avg_right_conf = np.mean(right_conf) if right_conf else 0.0
    
    warnings = []
    if abs(avg_left_conf - avg_right_conf) < 0.1:
        warnings.append("low_side_confidence_diff")
        
    if avg_left_conf >= avg_right_conf:
        selected_side = "left"
        selected_landmarks = left_landmarks
        confidence = float(avg_left_conf)
    else:
        selected_side = "right"
        selected_landmarks = right_landmarks
        confidence = float(avg_right_conf)
        
    if confidence < min_confidence:
        warnings.append("low_overall_confidence")
        
    # 2. Global Margin Search (Deepest Point Detection)
    # Iterate over ALL frames to find the maximum margin (deepest point)
    
    max_margin = -float('inf')
    best_frame_idx = detected_bottom_idx # Default to detected bottom if no landmarks
    
    # We only care about frames where landmarks are present
    frames_with_landmarks = []
    
    for i, lm in enumerate(selected_landmarks):
        if lm:
            # y_D - y_E. Positive = D below E (VALID).
            margin = lm.deltoid_apex[1] - lm.elbow_tip[1]
            frames_with_landmarks.append((i, margin))
            
            if margin > max_margin:
                max_margin = margin
                best_frame_idx = i
                
    if not frames_with_landmarks:
        return DipDecision(
            valid=False,
            margin_px=0.0,
            best_margin_px=0.0,
            selected_side=selected_side,
            bottom_frame_index=detected_bottom_idx,
            confidence=confidence,
            warnings=warnings + ["no_landmarks_for_decision"]
        )
    
    # Check for angle warning in the best frame (or nearby)
    # Let's check a small window around the best frame for warnings
    warn_start = max(0, best_frame_idx - 5)
    warn_end = min(num_frames - 1, best_frame_idx + 5)
    if any(selected_landmarks[i] and selected_landmarks[i].angle_warning for i in range(warn_start, warn_end + 1)):
        warnings.append("angle_warning")
        
    # Decision: Valid if at ANY point margin >= 0 (strictly speaking > 0 or >= 0. Let's use >= -0.0 for float tolerance)
    valid = max_margin >= 0.0
    
    return DipDecision(
        valid=valid,
        margin_px=float(max_margin),
        best_margin_px=float(max_margin),
        selected_side=selected_side,
        bottom_frame_index=best_frame_idx,
        confidence=confidence,
        warnings=list(set(warnings)) # unique warnings
    )