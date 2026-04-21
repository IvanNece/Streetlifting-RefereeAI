# Squat Referee AI

Automated squat depth validator using computer vision. Processes a lateral-view video and outputs **VALID LIFT / INVALID LIFT** verdict + annotated video.

Part of the [Streetlifting-RefereeAI](https://github.com/IvanNece/Streetlifting-RefereeAI) ecosystem.

---

## Rule Implemented

**Below-parallel depth**: the crease of the hip (proximal end of the femur at the hip joint) must descend **visibly below** the top of the knee joint.

> Camera constraint: **lateral view only** (sagittal plane). The system is not designed for frontal or angled shots.

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run su uno dei video di test già presenti
python main.py --input Input_Videos/squat1.mp4 --output Output_Videos/squat1_result.mp4

# Con overlay di debug
python main.py --input Input_Videos/squat1.mp4 --output Output_Videos/squat1_result.mp4 --debug
```

---

## Output

| Artefact | Description |
|---|---|
| `result.mp4` | Input video annotated with skeleton, depth line, hip/knee markers |
| `stdout` | `VALID LIFT` or `INVALID LIFT` + frame-level depth ratio |

---

## Project Structure

```
squat-referee/
├── main.py              # Entry point (da creare)
├── analyzer.py          # Core logic: pose extraction + depth rule (da creare)
├── renderer.py          # OpenCV annotation & video writer (da creare)
├── Docs/                # Documentazione progetto
│   ├── ARCHITECTURE.md
│   ├── DECISIONS.md
│   └── CLAUDE.md
├── Input_Videos/        # Video di test già presenti (4 squat laterali)
├── Output_Videos/       # Output annotati (da creare a runtime)
├── tests/               # Da creare
│   ├── test_analyzer.py
│   └── fixtures/
├── requirements.txt
└── README.md
```

---

## Docs

- [`Docs/ARCHITECTURE.md`](./Docs/ARCHITECTURE.md) — pipeline design and component breakdown
- [`Docs/DECISIONS.md`](./Docs/DECISIONS.md) — tech choices and rejected alternatives
- [`Docs/CLAUDE.md`](./Docs/CLAUDE.md) — istruzioni per Claude Code