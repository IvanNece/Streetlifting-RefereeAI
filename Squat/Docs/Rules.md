# RULES — Istruzioni per Claude Code

Regole di sviluppo per il progetto `squat-referee`. Semplici e pragmatiche.

---

## Stile del Codice

- **Python 3.10+**, type hints sulle funzioni pubbliche, nient'altro di obbligatorio
- Naming: `snake_case` per tutto, `UPPER_SNAKE` per costanti, `PascalCase` per classi
- Nessun file deve superare **150 righe**. Se ci si avvicina, spezza in un modulo separato
- Preferisci funzioni semplici a classi se non c'è stato da mantenere
- Niente dipendenze esterne oltre a quelle in `pyproject.toml`

## Docstring

Brevi e utili. Solo dove serve davvero.

```python
def analyze_depth(hip_y: float, knee_y: float, delta_threshold: float = 5.0) -> str:
    """Ritorna 'GOOD_LIFT', 'NO_LIFT' o 'UNCERTAIN' secondo le regole IPF.
    
    UNCERTAIN se il delta è < delta_threshold pixel (zona grigia).
    """
```

## Commenti

Commenta il **perché**, non il cosa:
- ✅ `# IPF: se incerto, beneficio del dubbio → UNCERTAIN non NO_LIFT`  
- ❌ `# incrementa il contatore`

## Gestione Errori

- Se il modello ONNX non è trovato → stampa un messaggio chiaro con il path e termina
- Se un frame non ha keypoint a sufficiente confidence → skippa quel frame, logga un warning, non crashare
- Niente `except Exception: pass`. Mai.

## Testing

- Test solo per `depth_analyzer.py` e `rep_detector.py` — sono la logica core
- Struttura AAA (Arrange / Act / Assert), niente framework elaborati
- Casi da coprire obbligatoriamente:
  - Squat borderline (delta vicino a zero)
  - Video con una sola rep
  - Video con keypoint mancanti (occlusion)

## Performance

- I frame si leggono in streaming, non tutti in memoria
- Non copiare array NumPy se non strettamente necessario
- Se qualcosa è lento, misura prima con `time.perf_counter`, poi ottimizza

## Cosa NON fare

- Niente architetture elaborate (Clean Architecture, layer, dependency injection) — è un progettino
- Niente ORM, database, API REST — tutto su file locali
- Niente chiamate di rete durante il processing del video
- Non sovrascrivere mai il video originale in input

## Parametri Configurabili

Tienili come costanti in cima al file che li usa, non hardcoded nel mezzo del codice:

```python
# depth_analyzer.py
HIP_CREASE_ANTERIOR_RATIO = 0.20   # DA CALIBRARE su dataset reale
UNCERTAIN_ZONE_PX = 5              # Delta sotto cui il giudizio è incerto
MIN_CONFIDENCE = 0.5               # Threshold keypoint confidence
```

## Entry Point

```bash
python main.py input_video.mp4
# Output: input_video_analyzed.mp4 + report a schermo
```

Opzione utile da aggiungere subito: `--side [left|right]` per indicare da che lato è la camera.