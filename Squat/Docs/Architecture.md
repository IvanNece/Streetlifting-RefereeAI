# ARCHITECTURE — AI Squat Referee

## Struttura del Progetto

```
squat-referee/
├── pyproject.toml
├── README.md
├── main.py                  # Entry point: python main.py video.mp4
│
├── src/
│   ├── pose_estimator.py    # Wrapper RTMPose ONNX: frame → keypoints
│   ├── rep_detector.py      # Rileva inizio/fine di ogni ripetizione
│   ├── depth_analyzer.py    # Logica IPF: anca sotto il ginocchio?
│   ├── renderer.py          # Disegna overlay sul video di output
│   └── utils.py             # OneEuroFilter, helpers vari
│
├── models/
│   ├── rtmdet_nano.onnx     # (non committato, scaricato da download_models.py)
│   └── rtmpose_m.onnx
│
├── scripts/
│   └── download_models.py
│
└── tests/
    ├── test_depth_analyzer.py
    └── test_rep_detector.py
```

## Flusso dei Dati

```
video.mp4
    │
    ▼
[pose_estimator.py]
    Legge frame per frame con OpenCV
    RTMDet → bounding box atleta
    RTMPose → 17 keypoints (COCO) per frame
    Output: lista di dict {frame_idx, left_hip, right_hip, left_knee, right_knee, ...}
    │
    ▼
[rep_detector.py]
    Analizza la traiettoria verticale dell'anca nel tempo
    State machine semplice: SCENDENDO → FONDO → SALENDO
    Output: lista di rep con [frame_start, frame_bottom, frame_end]
    │
    ▼
[depth_analyzer.py]
    Per ogni rep, al frame_bottom:
    Controlla se hip_crease_y > knee_top_y (anca più bassa del ginocchio)
    Output: lista di verdetti {rep_n, GOOD_LIFT/NO_LIFT, delta_px}
    │
    ▼
[renderer.py]
    Riscrive il video con overlay:
    - Cerchi sui keypoint
    - Linea orizzontale su anca e ginocchio
    - Semaforo verde/rosso al momento della profondità
    - Testo "REP 1: GOOD LIFT" o "REP 1: NO LIFT"
    │
    ▼
output_video.mp4 + report stampato a schermo
```

## Nota Critica: Hip Crease

I modelli COCO rilevano il **centro articolare dell'anca**, non la **piega inguinale** (hip crease IPF). Sono punti diversi.

Per vista laterale, si approssima geometricamente:

```python
# La hip crease è ~20% della lunghezza femore, 
# in direzione anteriore rispetto al centro articolare
limb_vec = knee - hip
perp = np.array([limb_vec[1], -limb_vec[0]])  # perpendicolare = direzione anteriore
perp_norm = perp / np.linalg.norm(perp)
hip_crease = hip + perp_norm * np.linalg.norm(limb_vec) * 0.20
```

**Il valore `0.20` deve essere calibrato su video reali etichettati da un arbitro IPF.** È il parametro più critico del sistema.

## Rep Detector — State Machine

```
STANDING → (anca scende per N frame) → DESCENDING
DESCENDING → (velocità ≈ 0) → BOTTOM        ← qui si giudica la profondità
BOTTOM → (anca sale per N frame) → ASCENDING
ASCENDING → (anca stabile in alto) → STANDING
```

Niente di elaborato: si traccia la posizione y dell'anca nel tempo e si cercano il minimo locale.
`scipy.signal.find_peaks` sulla curva invertita fa il lavoro.