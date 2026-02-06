import cv2
import numpy as np
import os

def load_video(path: str) -> tuple[list[np.ndarray], dict]:
    """
    Load video frames using OpenCV.
    Returns:
        frames: list of numpy arrays (BGR)
        metadata: dict with fps, width, height, frame_count, duration
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    # Note: CAP_PROP_FRAME_WIDTH/HEIGHT might be raw dimensions before rotation
    # We will trust the dimensions of the read frames.
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    if not frames:
        raise ValueError(f"No frames read from video: {path}")

    # Update metadata based on actual read frames
    height, width = frames[0].shape[:2]
    frame_count = len(frames)
    duration = frame_count / fps if fps > 0 else 0

    metadata = {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration": duration
    }
    
    return frames, metadata

def save_video(frames: list[np.ndarray], path: str, fps: float):
    """
    Save list of frames to a video file.
    """
    if not frames:
        return
        
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
        
    out.release()
