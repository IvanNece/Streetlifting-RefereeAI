# GEMINI.md — Agent Operating Rules

You are working in a new repository. You must follow this document.

## 0) Mission
Implement a Dip validity checker from raw side-view smartphone video.

Default model/tool: Gemini (fast/cheap).
Use Claude Opus ONLY if:
- the system is stuck after two concrete attempts to fix a bug, or
- a design decision is ambiguous and requires deep reasoning.

## 1) Non-negotiables
- Do not claim “done” unless you ran the Quality Gates.
- Keep changes small and reviewable (one task per branch).
- Do not add heavy dependencies without proposing at least one lighter alternative.
- Always update docs when behavior changes (SPEC/ARCHITECTURE).

## 2) Technical constraints
- Prefer Python for v1 (fast prototyping, CV ecosystem).
- Modular design:
  - pose backend as an interchangeable module
  - rule evaluation as pure functions
  - pipeline separated from UI/output
- The video is raw (phone). Handle rotation metadata if needed.

## 3) Required deliverables in v1
- CLI command that:
  - reads input video from input_videos/
  - produces output report JSON
  - produces overlay mp4 in output/
  - prints VALID/INVALID + bottom_frame_index + margin

- Overlay must show:
  - selected side (left/right)
  - shoulder and elbow markers
  - a horizontal line at elbow y
  - margin numeric value
  - phase label (top/descending/bottom/ascending) per frame

## 4) Quality Gates (must run before “done”)
- Install dependencies
- Run unit tests
- Run the pipeline on at least one example video and generate outputs

If any gate fails, fix it.

## 5) Output expectations for each task
For each completed task, provide:
- What changed (short)
- Commands to run (exact)
- Files touched
- Known limitations + next steps
