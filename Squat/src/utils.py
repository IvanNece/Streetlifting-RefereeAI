"""OneEuroFilter and keypoint helpers."""
import numpy as np

MIN_CONFIDENCE = 0.5

# COCO keypoint indices
_SIDE_INDICES = {
    "left":  {"hip": 11, "knee": 13, "ankle": 15},
    "right": {"hip": 12, "knee": 14, "ankle": 16},
}


class OneEuroFilter:
    """Adaptive smoothing: smooth on slow motion, reactive on fast."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.1, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def __call__(self, x: float, t: float) -> float:
        if self._t_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x
        dt = t - self._t_prev
        if dt <= 0:
            return self._x_prev  # type: ignore[return-value]
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self._x_prev) / dt  # type: ignore[operator]
        dx_hat = alpha_d * dx + (1 - alpha_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1 - alpha) * self._x_prev  # type: ignore[operator]
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


def get_side_keypoints(pose, side: str) -> dict | None:
    """Return {hip, knee, ankle} coords for `side`, or None if any score < MIN_CONFIDENCE."""
    indices = _SIDE_INDICES[side]
    result = {}
    for name, idx in indices.items():
        if pose.scores[idx] < MIN_CONFIDENCE:
            return None
        result[name] = pose.keypoints[idx]
    return result
