# TECH_STACK — AI Squat Referee

## Stack

| Componente | Scelta | Perché |
|---|---|---|
| Video I/O | `opencv-python-headless` | Standard per manipolazione video, nessuna alternativa reale |
| Pose Engine | **RTMPose-m** (ONNX) | 90+ FPS su CPU i7, 75.8% AP — significativamente più accurato di YOLOv8n (50% AP) |
| Person Detector | **RTMDet-nano** (ONNX) | Necessario per la pipeline top-down di RTMPose, leggerissimo |
| Inference Runtime | `onnxruntime` | Platform-agnostic (CPU/GPU/Apple Silicon), stesso modello ovunque |
| Calcoli numerici | `numpy` | Ovvio |
| Keypoint smoothing | `OneEuroFilter` | Adattivo: smooth sui movimenti lenti, reattivo su quelli veloci — ideale per squat |
| Package manager | `uv` + `pyproject.toml` | Setup rapido, lock file deterministico |

## Perché NON le scelte di Gemini

**MediaPipe Holistic:** deprecated da Google. `Holistic` include face/hands — overhead inutile. La nuova API non esporta in ONNX.

**YOLOv8n-pose:** il suffisso `n` (nano) sacrifica ~25 punti di accuracy rispetto a `m`. Per un sistema arbitrale è la scelta sbagliata. RTMPose-m fa più FPS *e* più accuracy su CPU.

**OpenVINO:** Intel-only. Su laptop AMD o Mac non funziona. ONNX Runtime gestisce già tutto.

## Dipendenze (pyproject.toml)

```toml
[project]
name = "squat-referee"
requires-python = ">=3.10"
dependencies = [
    "opencv-python-headless>=4.9",
    "onnxruntime>=1.18",
    "numpy>=1.26",
    "scipy>=1.13",
]
```

## Come si scaricano i modelli

```bash
python scripts/download_models.py
# Scarica rtmdet_nano.onnx e rtmpose_m.onnx da HuggingFace nella cartella models/
```

I file `.onnx` non si committano su git (troppo grandi).