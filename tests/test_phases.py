import numpy as np
import pytest
from dip_validator.phases import smooth_signal, detect_bottom_frame, segment_phases, compute_depth_signal
from dip_validator.pose import PoseResult

def test_smooth_signal():
    # Synthetic noisy signal: a parabola with noise
    t = np.linspace(0, 1, 100)
    signal = -4 * (t - 0.5)**2 + 1  # Peak at 0.5
    noise = np.random.normal(0, 0.05, 100)
    noisy_signal = signal + noise
    
    smoothed = smooth_signal(noisy_signal, window=15, polyorder=2)
    
    assert len(smoothed) == len(noisy_signal)
    # Smoothing should reduce variance
    assert np.var(smoothed - signal) < np.var(noisy_signal - signal)

def test_detect_bottom_frame():
    # Signal with a clear peak
    signal = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0], dtype=float)
    bottom_idx = detect_bottom_frame(signal)
    assert bottom_idx == 5

def test_segment_phases():
    # 21 frames: 0-10 ascending y (descending dip), 10 is bottom, 11-20 descending y (ascending dip)
    signal = np.concatenate([
        np.linspace(0, 10, 11), # 0 to 10
        np.linspace(9, 0, 10)   # 9 to 0
    ])
    bottom_idx = 10
    phases = segment_phases(signal, bottom_idx)
    
    # Bottom window is +/- 5 frames: 5 to 15
    assert phases[10] == "bottom"
    assert phases[5] == "bottom"
    assert phases[15] == "bottom"
    
    # Before bottom window: 0 to 4
    # Since signal is strictly increasing, they should be "descending"
    assert phases[0] in ["top", "descending"] 
    assert phases[4] == "descending"
    
    # After bottom window: 16 to 20
    # Since signal is strictly decreasing, they should be "ascending"
    assert phases[20] in ["top", "ascending"]
    assert phases[16] == "ascending"

def test_compute_depth_signal():
    # Mock PoseResults
    def make_pose(hip_y, shoulder_y=None):
        kp = np.zeros((17, 2))
        conf = np.zeros(17)
        
        if hip_y is not None:
            kp[11, 1] = hip_y # L_HIP
            kp[12, 1] = hip_y # R_HIP
            conf[11] = 0.9
            conf[12] = 0.9
            
        if shoulder_y is not None:
            kp[5, 1] = shoulder_y # L_SHOULDER
            kp[6, 1] = shoulder_y # R_SHOULDER
            conf[5] = 0.9
            conf[6] = 0.9
            
        return PoseResult(keypoints=kp, confidences=conf, bbox=(0,0,10,10))

    poses = [
        make_pose(100),
        make_pose(110),
        make_pose(120),
        None, # Gap
        make_pose(None, 130), # Fallback to shoulder
        make_pose(140)
    ]
    
    signal = compute_depth_signal(poses)
    assert len(signal) == 6
    assert signal[0] == 100.0
    assert signal[1] == 110.0
    assert signal[2] == 120.0
    assert signal[3] == 120.0 # Fill from previous
    assert signal[4] == 130.0 # Shoulder fallback
    assert signal[5] == 140.0
