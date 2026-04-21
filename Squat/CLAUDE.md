# CLAUDE.md — Instructions for Claude Code

This file is the authoritative guide for any AI agent (Claude Code or otherwise) working on this codebase.

---

## Project in One Line

Offline squat depth validator: lateral video in → VALID/INVALID verdict + annotated video out.

---

## Struttura del Progetto (STATO ATTUALE)

```
squat-referee/
├── Docs/                ← sei qui (CLAUDE.md, ARCHITECTURE.md, DECISIONS.md)
├── Input_Videos/        ← 4 video di squat laterali già presenti, usali per sviluppo e test
├── Output_Videos/       ← crea questa cartella se non esiste, ci vanno i video annotati
├── main.py              ← DA CREARE
├── analyzer.py          ← DA CREARE
├── renderer.py          ← DA CREARE
├── tests/               ← DA CREARE
│   ├── test_analyzer.py
│   └── fixtures/
└── requirements.txt     ← già presente
```

**Quando Claude Code legge questo file**, il primo passo è:
1. Leggere `Docs/ARCHITECTURE.md` e `Docs/DECISIONS.md`.
2. Listare i file in `Input_Videos/` per conoscere i nomi reali dei video disponibili.
3. Produrre un piano di implementazione prima di scrivere qualsiasi codice.

---

## Stack

| Layer | Scelta | Note |
|---|---|---|
| Language | Python 3.11+ | No versioni precedenti |
| Pose estimation | `mediapipe>=0.10` | `model_complexity=2` per uso offline |
| Video I/O | `opencv-python` | VideoCapture + VideoWriter |
| Numerics | `numpy` | Solo math sulle coordinate |
| CLI | `argparse` | Semplice, no Click/Typer |
| Tests | `pytest` | Solo unit test |

**Non introdurre** FastAPI, Flask, SDK cloud, o ML framework oltre mediapipe/opencv.

---

## File Responsibilities (strict)

| File | Cosa fa | Cosa NON deve fare |
|---|---|---|
| `main.py` | CLI entry, wiring dei componenti | Business logic |
| `analyzer.py` | Inferenza pose + regola profondità + state machine | Rendering/drawing |
| `renderer.py` | Tutto il drawing OpenCV + VideoWriter | Logica della regola |
| `Input_Videos/` | Video di input già presenti — non modificare | — |
| `Output_Videos/` | Output annotati generati a runtime — crea se non esiste | — |

---

## Core Rule — Do Not Change Without Explicit Instruction

```python
# Hip below knee = VALID
# In pixel coords (y increases downward):
depth_ratio = (hip_y_pixel - knee_y_pixel) / frame_height
VERDICT = "VALID LIFT" if depth_ratio > DEPTH_THRESHOLD else "INVALID LIFT"
DEPTH_THRESHOLD = 0.0  # strict: any positive margin = valid
```

The 3-frame rolling median on `depth_ratio` must be applied before evaluating. Do not remove it.

---

## Landmark Selection

- Use MediaPipe landmark indices: HIP=23 (left) / 24 (right), KNEE=25 (left) / 26 (right).
- Always select the side with the **higher visibility score** (`landmark.visibility`).
- If visibility < 0.5 on both sides, skip the frame and log a warning. Do not hallucinate a verdict from invisible landmarks.

---

## State Machine States

```
IDLE → STANDING → DESCENDING → BOTTOM → ASCENDING → DONE
```

- Verdict is evaluated **only at BOTTOM**.
- BOTTOM is defined as the frame where `depth_ratio` reaches its per-rep maximum (deepest hip position).
- A rep is complete when state reaches DONE.

---

## Output Contract

`main.py` must print to stdout exactly one of:
```
VALID LIFT
INVALID LIFT
```
No other text on that line. Additional debug info must go to stderr or be gated behind `--debug`.

The annotated video must preserve original resolution and FPS.

---

## What to Annotate in the Output Video

1. Lower-body skeleton (hips → knees → ankles), line thickness=2.
2. Hip landmark: filled circle, **green** if `depth_ratio > 0`, **red** otherwise.
3. Knee landmark: filled circle, white.
4. Horizontal dashed line at knee height.
5. `depth_ratio` value (2 decimal places) top-left corner.
6. Verdict badge (bottom-center): green "VALID LIFT" or red "INVALID LIFT", shown from BOTTOM frame onward.

---

## Error Handling

- If MediaPipe fails to detect a pose in a frame: skip frame, log to stderr, continue.
- If more than 30% of frames have no detection: exit with code 1 and message `"ERROR: Poor pose detection. Check camera angle (must be lateral view)."`.
- If input file does not exist: exit with code 1 immediately.

---

## Testing

```bash
pytest tests/ -v
```

Tests must cover:
- `depth_ratio` calculation for a known hip/knee pixel pair.
- State machine transitions (unit test with mock frames).
- Verdict at bottom frame (parametrized: valid case, invalid case, borderline case).

Do not write tests that require actual video files unless they use fixtures in `tests/fixtures/`.

---

## Do Not

- Do not use `mediapipe.solutions.pose` legacy API — use the Tasks API (`mediapipe.tasks`) if on mediapipe >= 0.10.3, otherwise use `mp.solutions.pose` if Tasks API is unavailable.
- Do not hardcode file paths.
- Do not print to stdout anything other than the final verdict line (unless `--debug`).
- Do not add GUI windows (`cv2.imshow`) — this is a headless batch tool.