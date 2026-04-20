"""Thin wrapper on rtmlib.Body — frame → FramePose."""
import logging
from dataclasses import dataclass

import numpy as np
from rtmlib import Body

_DETECTOR_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "yolox_s_8xb8-300e_humanart-3ef259a7.zip"
)
_POSE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)


@dataclass
class FramePose:
    frame_idx: int
    keypoints: np.ndarray  # shape (17, 2) — [x, y]
    scores: np.ndarray     # shape (17,)


class PoseEstimator:
    def __init__(self) -> None:
        logging.info("Initializing PoseEstimator — first run downloads ~50 MB to ~/.rtmlib/")
        self._body = Body(
            det=_DETECTOR_URL,
            det_input_size=(640, 640),
            pose=_POSE_URL,
            pose_input_size=(192, 256),
            backend="onnxruntime",
            device="cpu",
        )

    def estimate(self, frame: np.ndarray, frame_idx: int) -> FramePose | None:
        keypoints, scores = self._body(frame)
        if len(keypoints) == 0:
            logging.warning("Frame %d: no person detected, skipping", frame_idx)
            return None
        return FramePose(frame_idx=frame_idx, keypoints=keypoints[0], scores=scores[0])
