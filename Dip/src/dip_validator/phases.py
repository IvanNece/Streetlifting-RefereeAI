import numpy as np
from scipy.signal import savgol_filter
from typing import List, Optional, Dict
from .pose import PoseResult

def compute_depth_signal(poses: List[Optional[PoseResult]], conf_threshold: float = 0.3) -> np.ndarray:
    """
    Computes a 1D depth signal from pose results using hip or shoulder y-coordinates.
    
    Args:
        poses: List of PoseResult objects (or None if no pose was detected).
        conf_threshold: Minimum confidence for keypoints.
        
    Returns:
        np.ndarray: 1D array of depth values per frame.
    """
    signal = []
    
    # Indices for COCO keypoints
    L_SHOULDER, R_SHOULDER = 5, 6
    L_HIP, R_HIP = 11, 12
    
    for pose in poses:
        if pose is None:
            # If no pose, use the last known value or 0 if it's the first frame
            signal.append(signal[-1] if signal else 0.0)
            continue
            
        kp = pose.keypoints
        conf = pose.confidences
        
        # Try to use hips first (usually more stable depth proxy)
        hip_y = []
        if conf[L_HIP] > conf_threshold: hip_y.append(kp[L_HIP, 1])
        if conf[R_HIP] > conf_threshold: hip_y.append(kp[R_HIP, 1])
        
        if hip_y:
            signal.append(np.mean(hip_y))
        else:
            # Fallback to shoulders
            shoulder_y = []
            if conf[L_SHOULDER] > conf_threshold: shoulder_y.append(kp[L_SHOULDER, 1])
            if conf[R_SHOULDER] > conf_threshold: shoulder_y.append(kp[R_SHOULDER, 1])
            
            if shoulder_y:
                signal.append(np.mean(shoulder_y))
            else:
                # If neither hips nor shoulders, use last known value
                signal.append(signal[-1] if signal else 0.0)
                
    return np.array(signal)

def smooth_signal(signal: np.ndarray, window: int = 15, polyorder: int = 2) -> np.ndarray:
    """
    Applies Savitzky-Golay filter to smooth the depth signal.
    
    Args:
        signal: 1D array of depth values.
        window: The length of the filter window.
        polyorder: The order of the polynomial used to fit the samples.
        
    Returns:
        np.ndarray: Smoothed signal.
    """
    if len(signal) < window:
        # If signal is too short for the window, return it as is or use a smaller window
        if len(signal) > polyorder + 1:
            w = len(signal) if len(signal) % 2 != 0 else len(signal) - 1
            return savgol_filter(signal, w, polyorder)
        return signal
        
    return savgol_filter(signal, window, polyorder)

def detect_bottom_frame(smoothed_signal: np.ndarray) -> int:
    """
    Detects the bottom frame index (maximum depth / highest y).
    
    Args:
        smoothed_signal: 1D array of smoothed depth values.
        
    Returns:
        int: Index of the bottom frame.
    """
    if len(smoothed_signal) == 0:
        return 0
    return int(np.argmax(smoothed_signal))

def segment_phases(smoothed_signal: np.ndarray, bottom_idx: int, bottom_window: int = 5) -> Dict[int, str]:
    """
    Segments the dip into phases: top, descending, bottom, ascending.
    
    Args:
        smoothed_signal: 1D array of smoothed depth values.
        bottom_idx: Index of the detected bottom frame.
        bottom_window: +/- frames around detected bottom to consider as "bottom" phase.
        
    Returns:
        Dict[int, str]: Mapping from frame index to phase name.
    """
    phases = {}
    num_frames = len(smoothed_signal)
    
    if num_frames == 0:
        return phases
        
    # Define bottom window
    bottom_start = max(0, bottom_idx - bottom_window)
    bottom_end = min(num_frames - 1, bottom_idx + bottom_window)
    
    # Compute derivative to detect motion
    # We use a simple difference for now, but smoothed_signal is already SG-filtered
    # SG filter can also provide derivatives, but for v1 this is enough.
    diff = np.diff(smoothed_signal, prepend=smoothed_signal[0])
    
    # Threshold for "motionless" to detect top
    # This might need tuning or a more robust approach in v2
    motion_threshold = np.std(diff) * 0.2 if len(diff) > 1 else 0.0
    
    for i in range(num_frames):
        if bottom_start <= i <= bottom_end:
            phases[i] = "bottom"
        elif i < bottom_start:
            if diff[i] > motion_threshold:
                phases[i] = "descending"
            else:
                phases[i] = "top"
        else: # i > bottom_end
            if diff[i] < -motion_threshold:
                phases[i] = "ascending"
            else:
                # After bottom and not ascending much, could be "top" again or just end
                phases[i] = "top"
                
    return phases
