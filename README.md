# Streetlifting Dip Validator (High-Precision)

Analizza video smartphone di Dip alle parallele e determina se l'alzata è valida secondo le regole tecniche dello Streetlifting.

## ✅ Status: v1 Complete

| Video | Risultato | Margin |
|-------|-----------|--------|
| dip_45deg_1.mp4 | VALID ✅ | +6.77px |
| dip_slide_1.mp4 | VALID ✅ | +7.94px |
| dip_slide_2.mp4 | INVALID ✓ | -3.66px |

---

## Quick Start

```bash
# 1. Setup
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -e .

# 2. Run on a video
python -m dip_validator input_videos/dip_45deg_1.mp4

# 3. Check outputs
dir output/
# -> dip_45deg_1_overlay.mp4 (video con overlay)
# -> dip_45deg_1_report.json  (report JSON)
```

---

## Configurazione

Il tool utilizza un file di configurazione YAML per gestire tutti i parametri. Il file di default è `configs/default.yaml`.

È possibile specificare un file di configurazione personalizzato:
```bash
python -m dip_validator input_videos/video.mp4 --config configs/my_params.yaml
```

I parametri includono:
- **Pose**: Modello (rtmpose-s/m/l), device (cpu/cuda), confidence threshold.
- **Phases**: Smoothing window e polyorder, bottom window size.
- **Landmarks**: Rapporti di offset per D ed E, temporal smoothing factor (EMA).
- **Decision**: Minima confidenza richiesta.
- **Output**: Abilitazione trace per-frame nel JSON, visualizzazione margin nell'overlay.

---

## Regola di Validità

Un dip è **VALIDO** se, in qualsiasi momento durante l'esecuzione, il punto del **deltoide posteriore (D)** raggiunge o supera la linea orizzontale del **gomito (E)**.

```
margin_px = y_D - y_E
VALID = margin_px >= 0 (almeno in un frame)
```

In coordinate immagine Y cresce verso il basso, quindi:
- margin positivo = D è più in basso di E = VALIDO
- margin negativo = D è sopra E = NON VALIDO

---

## Output

### Console
```
=== DIP ANALYSIS RESULT ===
Video: dip_45deg_1.mp4
Result: VALID ✓
Best Margin: 6.8 px (at frame 120)
Side: left (conf: 0.78)
Detected Bottom: 120 (phase-based)
Report saved: output\dip_45deg_1_report.json
```

### Overlay Video
- Mostra fase corrente (DESCENDING, BOTTOM, ASCENDING)
- Punto D (verde) e punto E (blu)
- Linea orizzontale all'altezza del gomito
- Banner VALID/INVALID
- (Opzionale) Valore del margin in tempo reale vicino al deltoide

### Report JSON
Include i risultati sintetici e (se abilitato) il trace per-frame:
```json
{
  "video": "dip_45deg_1.mp4",
  "result": "VALID",
  "margin_px": 6.77,
  "best_margin_px": 6.77,
  "selected_side": "left",
  "bottom_frame_index": 120,
  "confidence": 0.78,
  "warnings": [],
  "frames_analyzed": 196,
  "fps": 30.0,
  "landmarks_trace": [
    {
      "frame": 0,
      "deltoid": [647.45, 93.75],
      "elbow": [671.16, 195.35],
      "margin_px": -101.6,
      "deltoid_conf": 0.86,
      "elbow_conf": 0.82
    },
    ...
  ]
}
```

---

## Architettura

1. **Video I/O** — Carica video con correzione rotazione
2. **Pose Estimation** — RTMPose via rtmlib (17 keypoints)
3. **Phase Detection** — Rileva bottom frame con Savitzky-Golay smoothing
4. **Landmark Refinement** — Stima D ed E con geometria basata su keypoints
5. **Decision Logic** — Calcola best margin su tutto il video
6. **Reporting** — Genera overlay MP4 e report JSON

---

## Limitazioni Note

- **Angolo obliquo (~45°)**: Funziona ma con warning `angle_warning`
- **Video ruotati**: Gestiti automaticamente via metadati
- **CPU only**: Nessun requisito GPU

---

## Development

```bash
# Run tests
pytest tests/ -v

# All 13 tests should pass
```

---

## Files

```
Streetlifting-RefereeAI/
├── input_videos/         # Video di input
├── output/               # Overlay e report generati
├── src/dip_validator/    # Codice principale
│   ├── cli.py           # Entry point
│   ├── video_io.py      # Caricamento video
│   ├── pose.py          # RTMPose wrapper
│   ├── phases.py        # Rilevamento fasi
│   ├── refinement.py    # Stima D ed E
│   ├── rules.py         # Logica decisione
│   └── reporting.py     # Report JSON
├── tests/                # Unit tests
└── docs/                 # Documentazione
```
