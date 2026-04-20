# DECISIONS.md — Dip Validator v1

## Architecture Decisions

### D1: Base Pose System — RTMPose via rtmlib
**Decision:** Use `rtmlib` (lightweight RTMPose wrapper) instead of full MMPose installation.

**Rationale:**
- `rtmlib` is a minimal inference-only library (~50MB vs >500MB for full MMPose)
- Same RTMPose models, just easier installation
- Better suited for CLI tool deployment
- CPU inference is fast enough (~10-20 fps on modern CPU)

**Alternatives considered:**
- Full MMPose: More features but heavy dependency, complex installation
- MediaPipe: Explicitly forbidden in GEMINI.md

**Trade-offs:**
- ✅ Easy installation, small footprint
- ✅ Same model accuracy as MMPose
- ❌ Fewer model options (but RTMPose-m is sufficient)

---

### D2: Landmark Refinement Strategy — Keypoint Geometry for v1
**Decision:** Use pose keypoints + geometric offsets to estimate D (deltoid apex) and E (elbow tip).

**Rationale:**
- **No training data required** — can implement immediately
- More robust than contour-based approach (doesn't fail on clothing/lighting)
- Uses arm geometry (shoulder→elbow→wrist vectors) to determine directions

**How it works:**

**E (Elbow Tip):**
1. Take elbow and wrist keypoints from RTMPose
2. Calculate forearm vector (elbow → wrist)
3. Elbow tip is in the OPPOSITE direction of forearm (olecranon sticks backward)
4. Offset: 18% of upper arm length in -forearm direction

**D (Posterior Deltoid Apex):**
1. Take shoulder keypoint from RTMPose
2. Calculate forearm vector to determine "forward" direction
3. Posterior = OPPOSITE of forearm direction (back is opposite to hands)
4. Offset: 22% of upper arm length in posterior direction from shoulder

**Advantages over contour-based:**
- Works regardless of clothing color/texture
- Not affected by lighting conditions
- No dependency on edge detection quality

**Trade-offs:**
- ✅ Robust to visual conditions
- ✅ No segmentation needed
- ✅ Simple and fast
- ❌ Fixed offset ratios may need tuning for different body types
- ❌ Assumes arm geometry is visible (fails if arm is very occluded)

---

### D3: Side Selection — Automatic Based on Confidence
**Decision:** Automatically select left or right side based on landmark confidence scores.

**Rationale:**
- In side-view videos, one side is typically more visible
- In 45° videos, the camera-facing side has better visibility
- Selecting the more confident side reduces errors

**Algorithm:**
```python
left_conf = mean(left_elbow_conf, left_deltoid_conf)
right_conf = mean(right_elbow_conf, right_deltoid_conf)
selected_side = "left" if left_conf > right_conf else "right"
```

**Trade-offs:**
- ✅ Adaptive to video angle
- ✅ No manual configuration needed
- ❌ Could be confused if subject rotates mid-rep

---

### D4: Bottom Detection — Depth Signal with Smoothing
**Decision:** Use hip/shoulder y-coordinate as depth signal, apply Savitzky-Golay filter, find maximum.

**Rationale:**
- Hip and shoulder keypoints are the most stable in RTMPose
- Savitzky-Golay preserves peak shape while removing noise
- Maximum of depth signal = deepest point = bottom

**Parameters:**
- Signal: average of hip y-coordinates (or shoulder if hip occluded)
- Smoothing: Savitzky-Golay, window=15, polyorder=2
- Bottom window: ±5 frames around detected bottom

**Trade-offs:**
- ✅ Robust to single-frame jitter
- ✅ Works for different body types
- ❌ May fail if subject moves laterally during dip

---

### D5: Validity Rule — Best Margin Over Entire Video
**Decision:** Find the BEST (maximum) margin across ALL frames. VALID if best_margin >= 0.

**Rationale:**
- A dip is valid if the athlete reaches proper depth at ANY point during the movement
- The bottom_frame_index becomes the frame where the deepest point occurs
- This is more intuitive: "Did D ever reach or go below E?"

**Formula:**
```python
margins = [y_D[i] - y_E[i] for all frames i]
best_margin = max(margins)
best_frame_idx = argmax(margins)
valid = best_margin >= 0  # D touched or went below E at least once
```

**Trade-offs:**
- ✅ Matches athletic judging intent (reached depth at any point)
- ✅ Not fooled by pauses at partial depth
- ✅ bottom_frame_index now represents the deepest point of the dip
- ❌ Single-frame spike could trigger false positive (mitigated by temporal smoothing)

---

### D6: Video Codec — H.264 via OpenCV
**Decision:** Encode overlay video as H.264 (mp4v/avc1 fourcc) for maximum compatibility.

**Rationale:**
- H.264 is universally supported on all devices
- OpenCV can write H.264 if system has ffmpeg
- Alternative: fallback to XVID if H.264 unavailable

**Implementation:**
```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1'
writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
```

**Trade-offs:**
- ✅ Universal playback support
- ✅ Good compression (reasonable file size)
- ❌ Requires ffmpeg on system (usually pre-installed)

---

### D7: CPU-Only Inference
**Decision:** Target CPU inference only, no CUDA requirement.

**Rationale:**
- User explicitly requested CPU-only
- RTMPose via rtmlib is optimized with ONNX runtime
- Inference speed of ~10-20 fps is acceptable for offline analysis

**Trade-offs:**
- ✅ Works on any machine (no GPU needed)
- ✅ Simpler installation (no CUDA/cuDNN)
- ❌ Slower than GPU (~3-5x)

---

### D8: Python Version — 3.10+
**Decision:** Require Python 3.10 or higher.

**Rationale:**
- Required by rtmlib and modern type hints
- Match expressions, improved error messages
- User confirmed 3.10+ is acceptable

---

### D9: Temporal Smoothing — EMA for Landmark Positions
**Decision:** Apply Exponential Moving Average to D and E positions across frames.

**Rationale:**
- Reduces frame-to-frame jitter in landmark positions
- EMA is computationally cheap
- Alpha of 0.3-0.5 provides good balance

**Formula:**
```python
smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t-1]
```

**Trade-offs:**
- ✅ Reduces visual jitter in overlay
- ✅ More stable margin calculation
- ❌ Slight lag (acceptable for offline analysis)

---

### D10: 45-Degree Angle Detection
**Decision:** Detect oblique camera angles via shoulder width ratio and emit warning.

**Rationale:**
- At 45°, posterior deltoid is partially occluded
- Users should know when confidence is lower
- System still provides judgment (not a hard failure)

**Detection heuristic:**
```python
# If shoulders appear compressed (side view: wide, front view: narrow)
shoulder_width = abs(left_shoulder_x - right_shoulder_x)
if shoulder_width < threshold * torso_height:
    angle_warning = True
```

**Trade-offs:**
- ✅ Informs user of reduced precision
- ✅ Still attempts judgment (useful info)
- ❌ Heuristic may have false positives/negatives

---

## Library Choices

| Need | Library | Why |
|------|---------|-----|
| Pose estimation | rtmlib | Lightweight RTMPose wrapper, CPU-optimized |
| Video I/O | opencv-python | Standard, well-tested, handles mp4/mov |
| Signal processing | scipy | Savitzky-Golay filter for smoothing |
| CLI | argparse | Built-in, no extra dependency |
| Config | PyYAML | Simple, human-readable configs |
| Testing | pytest | Standard Python testing |

---

## Open Questions (for future versions)

1. **Custom keypoint training:** When should we switch from geometry-based to trained model?
   - *Answer:* After collecting 100+ annotated frames with clear ground truth

2. **Multi-rep support:** How to handle videos with multiple dips?
   - *Deferred to v2:* v1 assumes single dip or uses first complete rep

3. **Real-time mode:** Should we support live camera feed?
   - *Deferred to v2:* v1 is offline-only

4. **Calibration:** Should users be able to calibrate with a known object?
   - *Deferred to v2:* Current pixel-based margin is sufficient for valid/invalid
