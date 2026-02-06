import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from .pose import PoseResult

@dataclass
class RefinedLandmarks:
    elbow_tip: Tuple[float, float]
    deltoid_apex: Tuple[float, float]
    elbow_confidence: float
    deltoid_confidence: float
    side: str
    angle_warning: bool

def estimate_elbow_tip(pose: PoseResult, side: str, offset_ratio: float = 0.18) -> Tuple[Tuple[float, float], float]:
    """
    Estimates the elbow tip (olecranon).
    The elbow tip points BACKWARD (opposite to the forearm direction).
    """
    elbow_idx = 7 if side == "left" else 8
    wrist_idx = 9 if side == "left" else 10
    shoulder_idx = 5 if side == "left" else 6
    
    elbow_kp = pose.keypoints[elbow_idx]
    wrist_kp = pose.keypoints[wrist_idx]
    shoulder_kp = pose.keypoints[shoulder_idx]
    
    conf = float(pose.confidences[elbow_idx])
    
    # Forearm vector: elbow to wrist
    forearm_vec = wrist_kp - elbow_kp
    forearm_len = np.linalg.norm(forearm_vec)
    
    if forearm_len < 1e-6:
        return (float(elbow_kp[0]), float(elbow_kp[1])), conf
    
    # Upper arm vector: shoulder to elbow
    upper_arm_vec = elbow_kp - shoulder_kp
    upper_arm_len = np.linalg.norm(upper_arm_vec)
    
    if upper_arm_len < 1e-6:
        return (float(elbow_kp[0]), float(elbow_kp[1])), conf
    
    # Elbow tip points opposite to forearm
    elbow_tip_dir = -forearm_vec / forearm_len
    offset_magnitude = upper_arm_len * offset_ratio
    
    elbow_tip = elbow_kp + elbow_tip_dir * offset_magnitude
    
    return (float(elbow_tip[0]), float(elbow_tip[1])), conf

def estimate_deltoid_apex(pose: PoseResult, side: str, offset_ratio: float = 0.22) -> Tuple[Tuple[float, float], float]:
    """
    Estimates the posterior deltoid apex.
    It is offset BACKWARD from the shoulder joint.
    """
    shoulder_idx = 5 if side == "left" else 6
    elbow_idx = 7 if side == "left" else 8
    wrist_idx = 9 if side == "left" else 10
    
    shoulder_kp = pose.keypoints[shoulder_idx]
    elbow_kp = pose.keypoints[elbow_idx]
    wrist_kp = pose.keypoints[wrist_idx]
    
    conf = float(pose.confidences[shoulder_idx])
    
    # Upper arm vector: shoulder to elbow
    upper_arm_vec = elbow_kp - shoulder_kp
    upper_arm_len = np.linalg.norm(upper_arm_vec)
    
    if upper_arm_len < 1e-6:
        return (float(shoulder_kp[0]), float(shoulder_kp[1])), conf
    
    # Forearm vector: elbow to wrist
    forearm_vec = wrist_kp - elbow_kp
    forearm_len = np.linalg.norm(forearm_vec)
    
    if forearm_len < 1e-6:
        posterior_dir = upper_arm_vec / upper_arm_len
    else:
        posterior_dir = -forearm_vec / forearm_len
    
    backward_offset = upper_arm_len * offset_ratio
    deltoid_apex = shoulder_kp + posterior_dir * backward_offset
    
    return (float(deltoid_apex[0]), float(deltoid_apex[1])), conf

def detect_angle_warning(pose: PoseResult) -> bool:
    """Detects if camera is likely at ~45 degrees."""
    l_sh = pose.keypoints[5]
    r_sh = pose.keypoints[6]
    l_hip = pose.keypoints[11]
    
    shoulder_width = np.abs(l_sh[0] - r_sh[0])
    torso_height = np.abs(l_sh[1] - l_hip[1])
    
    if torso_height == 0: return False
    return (shoulder_width / torso_height) < 0.5

def refine_landmarks(
    pose: Optional[PoseResult], 
    side: str, 
    elbow_offset_ratio: float = 0.18, 
    deltoid_offset_ratio: float = 0.22
) -> Optional[RefinedLandmarks]:
    """Refines landmarks for a single frame."""
    if pose is None:
        return None
        
    elbow, e_conf = estimate_elbow_tip(pose, side, offset_ratio=elbow_offset_ratio)
    deltoid, d_conf = estimate_deltoid_apex(pose, side, offset_ratio=deltoid_offset_ratio)
    angle_warn = detect_angle_warning(pose)
    
    return RefinedLandmarks(
        elbow_tip=elbow,
        deltoid_apex=deltoid,
        elbow_confidence=e_conf,
        deltoid_confidence=d_conf,
        side=side,
        angle_warning=angle_warn
    )

def smooth_landmarks_temporal(landmarks_list: List[Optional[RefinedLandmarks]], alpha: float = 0.4) -> List[Optional[RefinedLandmarks]]:
    """Apply EMA smoothing to E and D positions across frames."""
    smoothed_list = []
    prev_e = None
    prev_d = None
    
    for lm in landmarks_list:
        if lm is None:
            smoothed_list.append(None)
            prev_e = None
            prev_d = None
            continue
            
        curr_e = np.array(lm.elbow_tip)
        curr_d = np.array(lm.deltoid_apex)
        
        if prev_e is None:
            smooth_e = curr_e
            smooth_d = curr_d
        else:
            smooth_e = alpha * curr_e + (1 - alpha) * prev_e
            smooth_d = alpha * curr_d + (1 - alpha) * prev_d
            
        smoothed_list.append(RefinedLandmarks(
            elbow_tip=(float(smooth_e[0]), float(smooth_e[1])),
            deltoid_apex=(float(smooth_d[0]), float(smooth_d[1])),
            elbow_confidence=lm.elbow_confidence,
            deltoid_confidence=lm.deltoid_confidence,
            side=lm.side,
            angle_warning=lm.angle_warning
        ))
        
        prev_e = smooth_e
        prev_d = smooth_d
        
    return smoothed_list
