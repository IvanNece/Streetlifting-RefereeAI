# SPEC — Dip validity from side-view video (v1)

## 1) Goal
Build a software that takes a raw side-view smartphone video of a Dip and decides if the rep is valid.

## 2) Inputs
- A raw phone video file (mp4/mov), single rep or a short set.
- Side view assumption: camera placed laterally, roughly perpendicular to athlete sagittal plane.

Input folder convention:
- input_videos/ contains raw videos (no preprocessing required by the user).

## 3) Outputs
1) On-screen result after processing:
- VALID or INVALID
- bottom_frame_index
- depth_margin_px (positive means valid depth, negative means not deep enough)
- confidence/warnings if landmarks are unreliable

2) Files generated in output/:
- overlay video (same fps/resolution as input) with:
  - phases/segments timeline (top, descending, bottom, ascending)
  - landmarks and rule lines for the chosen side
  - a text banner with current phase and final decision
- JSON decision report with the same fields displayed on screen

## 4) Dip depth rule (v1)
Rule:
- At bottom position, posterior deltoid must be below elbow.
In image coordinates (y grows downward):
- valid if y_posterior_deltoid > y_elbow at the bottom.

Important details:
- Evaluate only one side (the side with higher landmark confidence).
- Use a small bottom window (e.g., 5–15 frames around the bottom) and take the worst margin in that window to be conservative.

Depth margin definition (pixels):
- margin = y_shoulder - y_elbow
- margin > 0 => valid depth
- margin < 0 => invalid depth

## 5) Bottom detection (v1 heuristic)
Define bottom as the time where the tracked shoulder reaches its lowest point (max y) after smoothing.

Segment phases:
- top: shoulder y near local minimum
- descending: shoulder y increasing trend
- bottom: frames around the global/local max y
- ascending: shoulder y decreasing trend

This is heuristic in v1. Later we can improve with velocity thresholds and repetition parsing.

## 6) Acceptance criteria (v1)
- Runs end-to-end on a raw smartphone dip video without manual preprocessing.
- Produces decision JSON and overlay video.
- Correctly identifies a bottom frame and outputs a depth margin.
- Has unit tests for:
  - margin computation
  - decision logic on synthetic landmarks
  - side selection logic (pick higher confidence side)

## 7) Non-goals (v1)
- No multi-lift classification (MU/pull/squat not required)
- No counting multiple reps reliably in a set (can be added later)
- No production UI (web/mobile) yet
