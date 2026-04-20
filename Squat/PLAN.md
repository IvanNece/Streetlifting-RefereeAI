# PLAN.md — AI Squat Referee (IPF Standard)

Piano implementativo per l'MVP. Ogni task ha un criterio di successo verificabile.

---

## Obiettivo MVP

```bash
python main.py Input_Videos/1_Milo.mp4
# → output/1_Milo_analyzed.mp4  (video con overlay)
# → report a schermo: REP 1: GOOD LIFT | REP 2: NO LIFT | ...
```

Criteri di successo globali:
- Accuratezza giudizio ≥ 90% rispetto a un giudice umano
- Elaborazione ≤ 2× la durata del video su laptop normale
- Setup con un solo comando (`uv sync`)

---

## Fase 0 — Setup Progetto ✅

**Task 0.1** — `pyproject.toml` con dipendenze

```toml
[project]
name = "squat-referee"
requires-python = ">=3.10"
dependencies = [
    "rtmlib>=0.0.13",
    "opencv-python-headless>=4.9",
    "scipy>=1.13",
]
```

`rtmlib` porta con sé `onnxruntime` e `numpy` come dipendenze transitive — non dichiararli esplicitamente.

**Task 0.2** — `.gitignore`
- `output/`
- `__pycache__/`, `.venv/`

Niente `models/` da gestire: rtmlib scarica i pesi automaticamente al primo avvio e li cachea in `~/.rtmlib/` (fuori dal repo).

**Verifica Fase 0**: `uv sync` → nessun errore; `python -c "from rtmlib import Body"` → nessun ImportError.

> ✅ **DONE** — `pyproject.toml` creato con `[tool.uv] package = false`, `.gitignore` ok, `uv sync` → 14 pacchetti installati, import verificato.

---

## Fase 1 — Pose Estimator (`src/pose_estimator.py`)

**Responsabilità**: thin wrapper su `rtmlib.Body`. Dato un frame BGR, ritorna i 17 keypoint COCO dell'atleta con confidence.

**Inizializzazione** (una volta sola, passata come argomento o singleton):
```python
from rtmlib import Body

_DETECTOR_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "yolox_s_8xb8-300e_humanart-3ef259a7.zip"
)
_POSE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)

body = Body(
    det=_DETECTOR_URL,
    det_input_size=(640, 640),
    pose=_POSE_URL,
    pose_input_size=(192, 256),   # (width, height) — modello 256x192
    backend="onnxruntime",
    device="cpu",
)
```

Al primo avvio rtmlib scarica i .zip da OpenMMLab CDN (~50 MB totali) e li cachea in `~/.rtmlib/`.

**Chiamata per frame**:
```python
keypoints, scores = body(frame)
# keypoints: np.ndarray (N_persons, 17, 2)
# scores:    np.ndarray (N_persons, 17)
```

**Output per frame** (si prende sempre la prima persona rilevata, indice 0):
```python
@dataclass
class FramePose:
    frame_idx: int
    keypoints: np.ndarray  # shape (17, 2) — coordinate [x, y]
    scores: np.ndarray     # shape (17,)
```

**Keypoint COCO rilevanti**:
- 11: left_hip, 12: right_hip
- 13: left_knee, 14: right_knee
- 15: left_ankle, 16: right_ankle

**Gestione frame vuoti**: se `keypoints` è vuoto (nessuna persona rilevata) → skippa il frame con `logging.warning`, non crashare.

**Verifica**: su un singolo frame di `1_Milo.mp4`, i keypoint anca/ginocchio sono visualizzabili con `cv2.circle`.

---

## Fase 2 — Utils (`src/utils.py`)

**Task 2.1** — `OneEuroFilter`
- Implementazione standard (parametri: `min_cutoff=1.0`, `beta=0.1`, `d_cutoff=1.0`)
- Usato per smoothing coordinate keypoint nel tempo

**Task 2.2** — `get_side_keypoints(pose: FramePose, side: str) -> dict`
- Estrae hip/knee/ankle per il lato `"left"` o `"right"`
- Filtra per `MIN_CONFIDENCE = 0.5`

**Verifica**: unit test banale — input noto → output atteso.

---

## Fase 3 — Rep Detector (`src/rep_detector.py`)

**Responsabilità**: data la lista di `FramePose`, identifica le ripetizioni.

**Algoritmo**:
1. Estrae curva `hip_y(t)` — coordinata Y dell'anca nel tempo (smoothed via OneEuroFilter)
2. `scipy.signal.find_peaks(-hip_y, prominence=30, distance=fps//2)` → frame dei minimi (= bottom delle rep)
3. Per ogni minimo: cerca il frame di inizio (hip_y sale prima del minimo) e fine (hip_y torna in alto)

**Output**:
```python
@dataclass
class Rep:
    rep_n: int
    frame_start: int
    frame_bottom: int
    frame_end: int
```

**Verifica** (`tests/test_rep_detector.py`):
- Input sintetico con 3 rep → output 3 `Rep` con frame corretti
- Input con 1 sola rep → output 1 `Rep`
- Input con curve piatte (no squat) → output lista vuota

---

## Fase 4 — Depth Analyzer (`src/depth_analyzer.py`)

**Responsabilità**: per ogni rep, al `frame_bottom`, giudica la profondità IPF.

**Costanti**:
```python
HIP_CREASE_ANTERIOR_RATIO = 0.20   # DA CALIBRARE
UNCERTAIN_ZONE_PX = 5
MIN_CONFIDENCE = 0.5
```

**Algoritmo**:
1. Al `frame_bottom`, estrae hip e knee del lato scelto
2. Calcola `hip_crease`:
   ```python
   limb_vec = knee - hip
   perp = np.array([limb_vec[1], -limb_vec[0]])
   perp_norm = perp / np.linalg.norm(perp)
   hip_crease = hip + perp_norm * np.linalg.norm(limb_vec) * HIP_CREASE_ANTERIOR_RATIO
   ```
3. `delta = hip_crease[1] - knee[1]`  (positivo = anca più bassa, in coordinate immagine y cresce verso il basso)
4. Giudizio:
   - `delta > UNCERTAIN_ZONE_PX` → `GOOD_LIFT`
   - `delta < -UNCERTAIN_ZONE_PX` → `NO_LIFT`
   - altrimenti → `UNCERTAIN`

**Output**:
```python
@dataclass
class Verdict:
    rep_n: int
    result: str   # "GOOD_LIFT" | "NO_LIFT" | "UNCERTAIN"
    delta_px: float
```

**Verifica** (`tests/test_depth_analyzer.py`):
- Hip nettamente sotto knee → `GOOD_LIFT`
- Hip nettamente sopra knee → `NO_LIFT`
- Delta = 3px → `UNCERTAIN`
- Keypoint mancante (confidence < 0.5) → skip, no crash

---

## Fase 5 — Renderer (`src/renderer.py`)

**Responsabilità**: riscrive il video con overlay grafico.

**Overlay per frame**:
- Cerchi bianchi sui keypoint anca/ginocchio/caviglia
- Linea orizzontale tratteggiata su `hip_crease` e `knee`
- Dal `frame_bottom` per 1 secondo: semaforo verde (`GOOD_LIFT`) o rosso (`NO_LIFT`) o giallo (`UNCERTAIN`)
- Testo `"REP N: GOOD LIFT"` in overlay permanente dopo la rep

**Verifica**: output `1_Milo_analyzed.mp4` riproducibile con un player standard, overlay visibili.

---

## Fase 6 — Entry Point (`main.py`)

```python
# python main.py video.mp4 [--side left|right]
```

**Flusso**:
1. Legge video con OpenCV in streaming
2. `pose_estimator` → lista `FramePose`
3. `rep_detector` → lista `Rep`
4. `depth_analyzer` → lista `Verdict`
5. `renderer` → scrive video di output
6. Stampa report a schermo

**Report**:
```
=== SQUAT REFEREE REPORT ===
REP 1: GOOD LIFT  (delta: +12.3 px)
REP 2: NO LIFT    (delta: -8.1 px)
REP 3: UNCERTAIN  (delta: +2.0 px)
============================
```

**Verifica**: `python main.py Input_Videos/1_Milo.mp4` → file di output creato, report stampato.

---

## Fase 7 — Calibrazione

**Dopo aver girato l'MVP sui 4 video di test:**

1. Guardare ogni bottom frame e confrontare il giudizio automatico con quello visivo
2. Aggiustare `HIP_CREASE_ANTERIOR_RATIO` finché concordanza ≥ 90%
3. Documentare il valore finale calibrato in `depth_analyzer.py`

**Video di test**: `1_Milo.mp4`, `2_Mase.mp4`, `3_Cerve.mp4`, `4_Ivan.mp4`

---

## Ordine di Implementazione

```
Fase 0 (setup)
    ↓
Fase 1 (pose_estimator) — blocca tutto il resto
    ↓
Fase 2 (utils) + Fase 3 (rep_detector)  ← parallelizzabili
    ↓
Fase 4 (depth_analyzer)
    ↓
Fase 5 (renderer) + Fase 6 (main.py)   ← parallelizzabili
    ↓
Fase 7 (calibrazione)
```

---

## Note Critiche

- `HIP_CREASE_ANTERIOR_RATIO = 0.20` è un'assunzione iniziale non calibrata
- Il `side` (left/right) dipende dall'angolazione della camera — defaultare a `right` e permettere override CLI
- I modelli vengono scaricati automaticamente da rtmlib al primo avvio (~50 MB) e cachati in ~/.rtmlib/
- Non sovrascrivere mai il video originale
