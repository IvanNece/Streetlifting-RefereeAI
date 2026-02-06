import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from rtmlib import Body

@dataclass
class PoseResult:
    """Dataclass to store pose estimation results for a single frame."""
    keypoints: np.ndarray  # (17, 2) - x, y coordinates
    confidences: np.ndarray  # (17,) - per-keypoint confidence
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)

class PoseEstimator:
    """Wrapper around rtmlib for pose estimation."""
    def __init__(self, device: str = 'cpu', mode: str = 'balanced'):
        # mode can be 'lightweight', 'balanced', or 'performance'
        # 'balanced' uses rtmpose-m and yolox-m
        # 'performance' uses rtmpose-l and yolox-l
        # 'lightweight' uses rtmpose-s and yolox-s
        self.model = Body(
            mode=mode,
            device=device
        )

    def estimate_poses(self, frames: List[np.ndarray], conf_threshold: float = 0.3) -> List[Optional[PoseResult]]:
        """
        Estimate poses for a list of frames.
        
        Args:
            frames: List of RGB frames as numpy arrays.
            conf_threshold: Minimum confidence for keypoints.
            
        Returns:
            List of PoseResult objects, one per frame. 
            Returns None for frames where no person is detected.
        """
        results = []
        for i, frame in enumerate(frames):
            # rtmlib Body returns (keypoints, scores)
            # keypoints shape: (N, 17, 2), scores shape: (N, 17)
            keypoints, scores = self.model(frame)
            
            if keypoints is not None and len(keypoints) > 0:
                # We assume the first detected person is the subject
                kp = keypoints[0]
                conf = scores[0]
                
                # Check if we have enough confident keypoints
                if np.mean(conf) < conf_threshold:
                    results.append(None)
                    continue

                # Calculate bbox from keypoints as fallback
                x1, y1 = np.min(kp, axis=0)
                x2, y2 = np.max(kp, axis=0)
                bbox = (float(x1), float(y1), float(x2), float(y2))
                
                results.append(PoseResult(
                    keypoints=kp,
                    confidences=conf,
                    bbox=bbox
                ))
            else:
                results.append(None)
                
        return results

def estimate_poses(frames: List[np.ndarray], device: str = 'cpu', mode: str = 'balanced', conf_threshold: float = 0.3) -> List[Optional[PoseResult]]:
    """Convenience function for pose estimation."""
    estimator = PoseEstimator(device=device, mode=mode)
    return estimator.estimate_poses(frames, conf_threshold=conf_threshold)
