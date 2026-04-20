# TASKS.md — Dip Validator v1

## Overview
Checklist tracking implementation progress. Each tranche is a reviewable unit.

---

## Tranche 1: Project Skeleton + Video I/O ✅ COMPLETE
- [x] Create local virtual environment (`python -m venv .venv`) and activate it
- [x] Create `pyproject.toml` with project metadata and dependencies
- [x] Create `requirements.txt` for pip users
- [x] Create `src/dip_validator/__init__.py` with version
- [x] Implement `src/dip_validator/video_io.py`:
  - [x] Load video with OpenCV
  - [x] Extract rotation metadata and fix orientation
  - [x] Return frames as numpy array + metadata dict
- [x] Implement minimal `src/dip_validator/cli.py`:
  - [x] Parse arguments (input video, output dir)
  - [x] Load video and print metadata
- [x] Verify: `pip install -e . && python -m dip_validator input_videos/dip_45deg_1.mp4`

---

## Tranche 2: Pose Estimation with RTMPose ✅ COMPLETE
- [x] Implement `src/dip_validator/pose.py`:
  - [x] Initialize rtmlib RTMPose model
  - [x] Run inference on all frames (CPU mode)
  - [x] Return list of PoseResult with keypoints + confidences
- [x] Update `cli.py` to run pose estimation
- [x] Verify: Print keypoint counts, visualize one frame

---

## Tranche 3: Phase Detection + Bottom Frame ✅ COMPLETE
- [x] Implement `src/dip_validator/phases.py`:
  - [x] Compute depth signal from stable keypoints (hip/shoulder y)
  - [x] Apply Savitzky-Golay smoothing
  - [x] Detect bottom_frame_index as maximum
  - [x] Segment phases: top → descending → bottom → ascending
- [x] Create `tests/test_phases.py`:
  - [x] Test smoothing function
  - [x] Test bottom detection with synthetic signal
  - [x] Test phase segmentation logic
- [x] Update `cli.py` to print bottom_frame_index
- [x] Verify: `pytest tests/test_phases.py -v`

---

## Tranche 4: Landmark Refinement (D and E) ✅ COMPLETE
- [x] Implement `src/dip_validator/segmentation.py`:
  - [x] Extract arm/shoulder ROI from pose keypoints
  - [x] Generate binary masks for arm regions
- [x] Implement `src/dip_validator/refinement.py`:
  - [x] Estimate E (elbow tip) using geometry-based approach
  - [x] Estimate D (deltoid apex) using geometry-based approach
  - [x] Compute confidence scores
  - [x] Detect ~45° angle and set warning flag
  - [x] Apply temporal smoothing (EMA)
- [x] Create `tests/test_refinement.py`:
  - [x] Test E estimation
  - [x] Test D estimation
  - [x] Test confidence thresholding
- [x] Verify: Print D and E coordinates on test video

**Implementation Note:** 
- E (elbow tip) = elbow keypoint + offset in OPPOSITE direction of forearm (18% of upper arm length)
- D (deltoid apex) = shoulder keypoint + offset in POSTERIOR direction (22% of upper arm length)

---

## Tranche 5: Rules + Decision Logic ✅ COMPLETE
- [x] Implement `src/dip_validator/rules.py`:
  - [x] Calculate margin_px = y_D - y_E
  - [x] Select best side based on confidence
  - [x] Find BEST margin over ENTIRE video (not worst in window)
  - [x] Return DipDecision with valid/invalid, margin, warnings
- [x] Create `tests/test_rules.py`:
  - [x] Test margin calculation (positive/negative cases)
  - [x] Test side selection logic
  - [x] Test best margin search logic
- [x] Update `cli.py` to print VALID/INVALID + margin
- [x] Verify: Manual E2E on test videos

**Rule:** `VALID = best_margin_px >= 0` (D touched or went below E at least once)

---

## Tranche 6: Overlay Video Generation ✅ COMPLETE
- [x] Implement overlay generation in `cli.py`:
  - [x] Draw phase label on each frame
  - [x] Draw D point (green)
  - [x] Draw E point (blue)
  - [x] Draw horizontal elbow line
  - [x] Show margin indicator
  - [x] Add VALID/INVALID banner
  - [x] Encode to MP4 with H.264
- [x] Update `cli.py` to generate overlay
- [x] Verify: Play `output/dip_45deg_1_overlay.mp4`

---

## Tranche 7: JSON Report ✅ COMPLETE
- [x] Implement `src/dip_validator/reporting.py`:
  - [x] Generate JSON with all required fields
  - [x] Include confidence stats
  - [x] Include video metadata
- [x] Update `cli.py` to generate report
- [x] Verify: Check JSON schema matches SPEC.md

---

## Tranche 8: E2E Tests + Final Verification ✅ COMPLETE
- [x] Run on `dip_45deg_1.mp4` — **VALID** (margin: +6.77px) ✓
- [x] Run on `dip_slide_1.mp4` — **VALID** (margin: +7.94px) ✓
- [x] Run on `dip_slide_2.mp4` — **INVALID** (margin: -3.66px) ✓
- [x] All unit tests passing (`pytest tests/ -v`)

---

## Remaining Tasks (Optional Polish)

### Documentation
- [ ] Update README.md with usage instructions
- [ ] Add example output screenshots

### Optional Enhancements
- [x] Create `configs/default.yaml` for configurable parameters
- [x] Add per-frame landmark trace to JSON report
- [x] Improve overlay with margin value text display

---

## Quality Gates ✅ ALL PASSED
- [x] `pip install -e .` succeeds
- [x] `pytest tests/ -v` all 13 tests pass
- [x] E2E on all 3 videos produces overlay + JSON
- [x] Overlay visually shows correct D and E positions
- [x] JSON report matches expected schema
- [x] Correct results: 45deg=VALID, slide1=VALID, slide2=INVALID

---

## Final Summary

**v1 Implementation Complete!**

The Dip Validator successfully:
1. Loads smartphone videos with rotation correction
2. Runs RTMPose for body keypoint detection
3. Detects movement phases and bottom frame
4. Estimates D (posterior deltoid) and E (elbow tip) using geometry
5. Calculates depth margin and determines validity
6. Generates overlay video and JSON report

Validation rule: **VALID if D reaches or goes below E line at ANY point during the movement**
