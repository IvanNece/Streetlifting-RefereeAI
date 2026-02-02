# Streetlifting Video Judge

Computer vision pipeline to judge Streetlifting reps from a raw smartphone side-view video.
Current lift: Dip on parallel bars.

## What it does (v1)
Input: a raw phone video (e.g., iPhone mp4) placed in an input folder.
Output:
- VALID or INVALID
- Bottom frame index
- Depth margin (how deep)
- An overlay video that shows:
  - the detected phases/segments (top, descending, bottom, ascending)
  - the landmarks/lines used for the rule
  - the final decision

## Current rule (Dip depth)
At the bottom position, the posterior deltoid landmark must be below the elbow landmark (side view).
We evaluate ONE side only (the visible side).

