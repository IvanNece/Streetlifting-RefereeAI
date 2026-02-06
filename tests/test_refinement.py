import numpy as np
import pytest
from dip_validator.refinement import (
    refine_landmarks, 
    RefinedLandmarks, 
    detect_angle_warning,
    estimate_elbow_tip,
    estimate_deltoid_apex,
    smooth_landmarks_temporal
)
from dip_validator.pose import PoseResult

def test_estimate_functions_fallback():
    # Test that functions return offset points based on geometry
    
    # Mock pose: vertical upper arm, slanted forearm
    kp = np.zeros((17, 2))
    kp[5] = [20, 20] # L_SHOULDER
    kp[7] = [20, 50] # L_ELBOW
    kp[9] = [0, 50]  # L_WRIST (forearm pointing left)
    pose = PoseResult(keypoints=kp, confidences=np.ones(17), bbox=(0,0,100,100))
    
    # Test elbow tip estimation
    # Forearm is [0, 50] - [20, 50] = [-20, 0]
    # Posterior dir is [1, 0]
    # E = [20, 50] + [1, 0] * (30 * 0.18) = [20 + 5.4, 50] = [25.4, 50.0]
    e_pt, e_conf = estimate_elbow_tip(pose, "left")
    assert e_pt[0] == pytest.approx(25.4)
    assert e_pt[1] == pytest.approx(50.0)
    
    # Test deltoid apex estimation
    # Posterior dir is [1, 0]
    # D = [20, 20] + [1, 0] * (30 * 0.22) = [20 + 6.6, 20] = [26.6, 20.0]
    d_pt, d_conf = estimate_deltoid_apex(pose, "left")
    assert d_pt[0] == pytest.approx(26.6)
    assert d_pt[1] == pytest.approx(20.0)

def test_smooth_landmarks_temporal():
    # Create a list of landmarks with a jump
    lm1 = RefinedLandmarks((20.0, 50.0), (20.0, 20.0), 1.0, 1.0, "left", False)
    lm2 = RefinedLandmarks((30.0, 50.0), (20.0, 20.0), 1.0, 1.0, "left", False)
    
    smoothed = smooth_landmarks_temporal([lm1, lm2], alpha=0.5)
    
    assert smoothed[0].elbow_tip == (20.0, 50.0)
    # Frame 2: 0.5 * 30 + 0.5 * 20 = 25
    assert smoothed[1].elbow_tip == (25.0, 50.0)

def test_detect_angle_warning():
    # Case 1: Wide shoulders (front view) -> False
    kp_front = np.zeros((17, 2))
    kp_front[5] = [100, 100] # L_SH
    kp_front[6] = [200, 100] # R_SH
    kp_front[11] = [100, 300] # L_HIP
    pose_front = PoseResult(keypoints=kp_front, confidences=np.ones(17), bbox=(0,0,500,500))
    assert detect_angle_warning(pose_front) == False 
    
    # Case 2: Narrow shoulders (side/45 view) -> True
    kp_side = np.zeros((17, 2))
    kp_side[5] = [100, 100]
    kp_side[6] = [130, 100] # Narrow
    kp_side[11] = [100, 300]
    pose_side = PoseResult(keypoints=kp_side, confidences=np.ones(17), bbox=(0,0,500,500))
    assert detect_angle_warning(pose_side) == True