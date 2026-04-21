# Technical Decisions

## ADR-001: MediaPipe Pose over YOLO11 Pose

**Decision:** Use MediaPipe Pose (`model_complexity=2`) as the sole pose estimator.

**Context:**
This is an offline, single-person, lateral-view video analyzer. We need accurate lower-body landmark localization, particularly hip and knee keypoints in a sagittal plane view.

**Rationale:**

| Factor | MediaPipe | YOLO11 Pose |
|---|---|---|
| Setup complexity | `pip install mediapipe` | `pip install ultralytics` + weights download |
| Lower-body accuracy | Good, 33 landmarks incl. hip crease | Good, 17 COCO landmarks |
| Single-person | Native design | Designed for multi-person (overkill) |
| CPU inference | Excellent (~30+ FPS) | Heavier on CPU |
| Hip crease landmark | Landmark 23/24 maps well to hip joint | Landmark 11/12 = hip, acceptable |
| Offline batch use | ✓ | ✓ |

**Verdict:** MediaPipe wins for this use case. YOLO11 adds unnecessary weight without a meaningful accuracy delta for a single-person, controlled lateral-view scenario. IEEE 2025 paper on deep squat detection with MediaPipe+YOLOv5 reports >96% accuracy — the precision ceiling is already high with MediaPipe alone.

**When to revisit:** If we extend to multi-person or non-lateral camera angles, YOLO11 becomes the right call due to its superior occlusion handling.

---

## ADR-002: Offline Batch Processing (No Real-Time)

**Decision:** Process pre-recorded video files only. No webcam/live stream.

**Rationale:**
- Competition context: referees review footage post-lift.
- Removes latency constraints → can use `model_complexity=2` (highest accuracy).
- Simpler error handling: video has a known length and FPS.
- No streaming infrastructure needed.

**Trade-off:** Cannot be used as live feedback during training without a refactor to swap complexity=1 and add a frame buffer.

---

## ADR-003: Verdict at Minimum Depth Frame, Not Average

**Decision:** Lock the VALID/INVALID verdict to the single frame where hip depth is maximum (bottom of the squat), not an average across all frames below parallel.

**Rationale:**
- Mirrors how a human referee judges: the **deepest point** of the lift is what counts.
- Averaging would make borderline squats ambiguous.
- A lifter can hit depth briefly; that is sufficient per powerlifting/streetlifting rules.

**Risk:** If pose estimation is noisy and produces a single erroneous outlier frame at the bottom, it could flip the verdict. Mitigation: apply a 3-frame rolling median on `depth_ratio` before evaluating the bottom.

---

## ADR-004: Lateral View Constraint (Hard Requirement)

**Decision:** The system only accepts lateral (sagittal plane) video. No frontal/diagonal camera support.

**Rationale:**
- Hip crease depth is **not reliably measurable** from a frontal view with 2D pose estimation.
- In a lateral view, hip_y and knee_y directly encode depth in the sagittal plane — the geometry is unambiguous.
- Adding multi-view support would require stereo or 3D pose estimation (e.g. MotionBERT, VideoPose3D) — out of scope.

**Implementation note:** Add a camera angle validator at startup: check that shoulder width (landmark 11-12 x-distance) is below a threshold relative to frame width. If the person is facing the camera, this distance is large; if lateral, it collapses. Warn the user if the view looks non-lateral but do not hard-block.

---

## ADR-005: Pure Python + OpenCV Stack (No API Calls)

**Decision:** The entire pipeline runs locally. No external API (Vision APIs, cloud inference) required.

**Rationale:**
- Competition video may contain sensitive athlete data — local processing is the only viable option.
- MediaPipe runs offline with no network dependency after install.
- Eliminates cost and latency of cloud inference.
- Consistent with the Dip module's approach in the parent repo.

---

## Known Risks & Open Issues

| Risk | Severity | Mitigation |
|---|---|---|
| Loose/baggy shorts obscure hip crease position | Medium | Annotate hip landmark visually so referee can override |
| Athlete positioned off-center in frame | Low | MediaPipe's ROI tracker handles this well |
| Fast descent causes motion blur | Medium | Process at original FPS; don't downsample |
| Single-frame noise flipping verdict at bottom | Medium | 3-frame rolling median on depth_ratio (ADR-003) |
| Non-lateral camera angle | High | Camera validator warning (ADR-004) |
| Occlusion of knee by arm/equipment | Low | Use visibility score to select best side |