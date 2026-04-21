# Architecture

## Pipeline Overview

```
Video File (lateral view)
        │
        ▼
┌─────────────────────┐
│   Frame Extractor   │  OpenCV VideoCapture, frame-by-frame iteration
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Pose Estimator    │  MediaPipe Pose (model_complexity=2)
│                     │  → 33 landmarks, normalized + pixel coords
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Depth Analyzer    │  Extract: LEFT_HIP, LEFT_KNEE (or RIGHT_*)
│                     │  Rule: hip_y > knee_y  (pixel coords, y↓)
│                     │  depth_ratio = (hip_y - knee_y) / frame_height
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  State Machine      │  STANDING → DESCENDING → BOTTOM → ASCENDING
│                     │  Verdict locked at BOTTOM frame (min hip_y)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Renderer          │  Draw skeleton, hip/knee points, depth line
│                     │  Overlay verdict badge on each frame
│                     │  Write annotated video via VideoWriter
└─────────────────────┘
        │
        ▼
Annotated Video + stdout verdict
```

---

## Key Components

### `analyzer.py` — `SquatAnalyzer`

Responsible for:
1. Running MediaPipe inference on each frame.
2. Extracting the **hip crease** (landmark 23/24) and **knee** (landmark 25/26) pixel coords.
3. Computing `depth_ratio` per frame.
4. Running the state machine to detect the rep bottom.
5. Returning a `SquatResult` dataclass: `{verdict, bottom_frame_idx, depth_ratio_at_bottom, landmarks_per_frame}`.

**Depth rule implementation:**
```python
# In pixel space: y increases downward.
# Hip below knee ↔ hip_y > knee_y
depth_ratio = (hip_y - knee_y) / frame_height
# depth_ratio > 0  →  hip is below knee  →  VALID
# depth_ratio <= 0 →  hip is at or above knee → INVALID
DEPTH_THRESHOLD = 0.0  # strict: any positive value = valid
```

> **Why pixel coords and not normalized?** Normalized landmarks (0-1) are relative to the bounding box, not the frame, making cross-frame comparison unreliable during lateral motion. Use `landmark.x/y * frame_w/h`.

### `renderer.py` — `SquatRenderer`

- Draws MediaPipe skeleton (lower body only: hips, knees, ankles).
- Highlights hip landmark (circle, red/green based on depth).
- Draws a horizontal reference line at knee height.
- Overlays depth ratio and verdict badge on each frame.
- Handles `cv2.VideoWriter` with original video FPS and resolution.

### State Machine

```
IDLE → STANDING (hip visibility confirmed)
STANDING → DESCENDING (hip_y increasing across N frames)
DESCENDING → BOTTOM (hip_y stops decreasing, local minimum)
BOTTOM → ASCENDING (hip_y decreasing again)
ASCENDING → DONE (hip_y back to standing baseline ± threshold)
```

Verdict is evaluated **at BOTTOM state only**, not as a running average.

---

## Coordinate System

MediaPipe uses a normalized coordinate system (0-1). This pipeline converts to pixel space:

```
(0,0) ───────────────────▶ x
  │
  │    [image content]
  │
  ▼
  y
```

Hip **below** knee means `hip_y_pixel > knee_y_pixel`.

---

## Landmark IDs Used

| ID | Name | Role |
|----|------|------|
| 23 | LEFT_HIP | Hip crease reference |
| 24 | RIGHT_HIP | Hip crease reference (fallback) |
| 25 | LEFT_KNEE | Knee reference |
| 26 | RIGHT_KNEE | Knee reference (fallback) |
| 27 | LEFT_ANKLE | Skeleton drawing |
| 28 | RIGHT_ANKLE | Skeleton drawing |

Side selection: use the side **facing the camera** (higher landmark visibility score). For a strict lateral view, one side will have systematically higher visibility.

---

## Performance Notes

- MediaPipe `model_complexity=2` is the accuracy-optimized model. For real-time use switch to `1`.
- Expected throughput: ~30-40 FPS on M1/M2 Mac CPU for 1080p input at complexity=1.
- This is an **offline batch processor**, not real-time. Complexity=2 is the correct default.