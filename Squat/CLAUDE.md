# CLAUDE.md — AI Squat Referee

Istruzioni di progetto per Claude Code. Leggi prima di toccare qualsiasi file.

---

## Panoramica

Script Python che prende un video di squat (vista laterale) e produce:
- Video annotato con overlay grafici (keypoint, linee, semaforo GOOD/NO LIFT)
- Report testuale con giudizio per ogni ripetizione (standard IPF)

Entry point: `python main.py input_video.mp4`

---

## Struttura del Progetto

```
squat-referee/
├── pyproject.toml
├── main.py                  # Entry point
├── src/
│   ├── pose_estimator.py    # Thin wrapper su rtmlib.Body → FramePose per frame
│   ├── rep_detector.py      # State machine: riconosce inizio/bottom/fine rep
│   ├── depth_analyzer.py    # Logica IPF: anca sotto il ginocchio?
│   ├── renderer.py          # Overlay video di output
│   └── utils.py             # OneEuroFilter, helpers
└── tests/
    ├── test_depth_analyzer.py
    └── test_rep_detector.py
```

Niente cartella `models/` — rtmlib scarica i pesi automaticamente al primo avvio e li cachea in `~/.rtmlib/`.

---

## Stack Tecnico

| Componente | Libreria |
|---|---|
| Video I/O | `opencv-python-headless` |
| Pose Engine | `rtmlib` — wrapper ufficiale OpenMMLab su RTMPose |
| Modelli | YOLOX-s (detector) + RTMPose-m (pose) — URL stabili da OpenMMLab CDN |
| Runtime sottostante | `onnxruntime` (gestito da rtmlib) |
| Smoothing | `OneEuroFilter` (adattivo, implementato in `utils.py`) |
| Peak detection | `scipy.signal.find_peaks` |
| Package manager | `uv` + `pyproject.toml` |

Non usare MediaPipe (deprecated), YOLOv8n-pose (troppo bassa accuracy), OpenVINO (Intel-only).
Non gestire manualmente i file `.onnx` — rtmlib li scarica e cachea in automatico.

---

## Regole di Sviluppo

### Codice
- Python 3.10+, type hints sulle funzioni pubbliche
- `snake_case` tutto, `UPPER_SNAKE` costanti, `PascalCase` classi
- **Max 150 righe per file** — se ci si avvicina, spezza
- Funzioni semplici > classi se non c'è stato da mantenere
- Solo dipendenze in `pyproject.toml`

### Parametri critici — costanti in cima al file, non hardcoded
```python
# depth_analyzer.py
HIP_CREASE_ANTERIOR_RATIO = 0.20   # DA CALIBRARE su dataset reale
UNCERTAIN_ZONE_PX = 5
MIN_CONFIDENCE = 0.5
```

### Gestione errori
- Primo avvio: rtmlib scarica i modelli (~50 MB) — nessuna azione richiesta, ma logga un messaggio informativo
- Frame senza keypoint sufficienti (nessuna persona rilevata o `scores < MIN_CONFIDENCE`) → skippa, `logging.warning(...)`, non crashare
- Mai `except Exception: pass`

### Performance
- Frame in streaming, non tutti in memoria
- Non copiare array NumPy inutilmente
- Misura con `time.perf_counter` prima di ottimizzare

---

## Logica Core — Hip Crease IPF

RTMPose rileva il **centro articolare dell'anca**, non la piega inguinale. Approssimazione geometrica:

```python
limb_vec = knee - hip
perp = np.array([limb_vec[1], -limb_vec[0]])
perp_norm = perp / np.linalg.norm(perp)
hip_crease = hip + perp_norm * np.linalg.norm(limb_vec) * HIP_CREASE_ANTERIOR_RATIO
```

`HIP_CREASE_ANTERIOR_RATIO = 0.20` è il parametro più critico — va calibrato su video reali etichettati da un arbitro IPF.

## Logica Core — Rep Detector

State machine sull'asse Y dell'anca nel tempo:
```
STANDING → DESCENDING → BOTTOM → ASCENDING → STANDING
```
Profondità giudicata al `BOTTOM`. Implementazione: `scipy.signal.find_peaks` sulla curva y invertita.

---

## Testing

Test **obbligatori** solo per `depth_analyzer.py` e `rep_detector.py`. Pattern AAA. Casi da coprire:
- Squat borderline (delta vicino a zero → `UNCERTAIN`)
- Video con una sola rep
- Frame con keypoint mancanti (occlusion)

---

## Cosa NON fare

- No Clean Architecture, layer, DI — è un progettino
- No chiamate di rete durante il processing
- Non sovrascrivere mai il video originale in input

---

## Video di Test

```
Input_Videos/1_Milo.mp4
Input_Videos/2_Mase.mp4
Input_Videos/3_Cerve.mp4
Input_Videos/4_Ivan.mp4
```

## Setup Ambiente

- Venv gestito da `uv`: `uv sync` crea `.venv/` automaticamente
- Python interpreter: `.venv/Scripts/python` (Windows)
- `pyproject.toml` usa `[tool.uv] package = false` — progetto script, non libreria
- Non usare `hatchling` build-backend senza configurare `packages`
