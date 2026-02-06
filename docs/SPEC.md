# SPEC — Dip Validator v1 (High-Precision Landmarks)

## 1) Goal
Costruire una pipeline che, dato un video smartphone di un Dip, determini:
- VALID o INVALID
- bottom_frame_index
- depth_margin_px (quanto e profondo)
- overlay video per debug e revisione

## 2) Input
- File video mp4/mov (iPhone/Android), “raw”
- Video tipico: 5–15 secondi, uno o pochi dip
- Camera: side view preferita, ma deve funzionare anche a ~45 gradi da davanti

Convenzione:
- input_videos/ contiene i video grezzi
- output/ contiene overlay e report

## 3) Output
Console:
- VALID/INVALID
- bottom_frame_index
- depth_margin_px
- selected_side (left/right) e warnings

File:
- output/<name>_overlay.mp4: video con overlay
- output/<name>_report.json: decision + dettagli
- (opzionale) output/<name>_landmarks.jsonl: trace per-frame

## 4) Rule (Dip depth) — definizione precisa
Target anatomici richiesti:
- E = elbow tip (punta del gomito, olecrano)
- D = posterior deltoid apex (apice del deltoide posteriore)

Regola:
- Un dip è VALIDO se, in QUALSIASI momento durante l'intera esecuzione, 
  il punto D raggiunge o supera (va sotto) la linea orizzontale che passa per E.
- Coordinate immagine (y cresce verso il basso):
  - margin_px = y_D - y_E
  - **VALID se esiste almeno un frame dove margin_px >= 0**

Valutazione:
- Si valuta un solo lato (auto-selezione in base ad affidabilità)
- Si cerca il **BEST margin** (massimo margin_px) su TUTTO il video
- **VALID = best_margin_px >= 0** (D ha toccato o superato la linea di E almeno una volta)
- Il bottom_frame_index è il frame dove si raggiunge il best_margin (punto più profondo del dip)

## 5) Bottom detection + segmentazione fasi (v1)
Obiettivo: mostrare segmenti utili nel video e scegliere bottom_frame_index.

v1:
- definire un segnale di “profondita” basato su keypoints stabili (es. shoulder/hip o torso proxy)
- smoothing del segnale (EMA o Savitzky-Golay)
- bottom = indice del massimo del segnale (posizione piu bassa)
- fasi:
  - top: vicino a un minimo locale
  - descending: derivata positiva
  - bottom: finestra attorno al massimo
  - ascending: derivata negativa

## 6) Pose & Landmark Strategy (NO MediaPipe, SI MMPose/RTMPose)
Requisito: alta precisione sui due punti D ed E.

Osservazione critica:
- I dataset standard (COCO 17 keypoints, COCO-WholeBody 133 keypoints) non includono “posterior deltoid apex”.
- I 133 keypoints estendono body+feet+face+hands, non aggiungono deltoide specifico.

Quindi la pipeline deve avere due livelli:
A) Coarse pose (base)
- RTMPose / MMPose per stimare joints e stabilizzare tracking (spalla, gomito, anche, ecc.)

B) Landmark refinement (target anatomici)
- Modulo dedicato che produce D ed E con confidenza per frame.
- Deve supportare almeno una di queste opzioni:
  1) Custom keypoint detector (consigliato): allenare un piccolo modello su frames annotati con D ed E.
  2) Mesh / dense surface (opzionale): stimare mesh umana e derivare un punto coerente sul deltoide posteriore.
  3) Segmentation + geometria (fallback): segmentare braccio/spalla e stimare D ed E da contorni (meno robusto ma senza training).

Il planning deve scegliere la strategia v1 (e motivarla) e definire un percorso per arrivare a precisione alta.

## 7) Affidabilita e warnings
Dato che il video puo essere a 45 gradi e/o con occlusioni:
- Se confidenza di D o E e sotto soglia per troppi frame nella bottom window:
  - output = INVALID
  - warnings: "low_confidence_landmarks" e dettagli

## 8) Acceptance Criteria (v1)
- E2E su un video raw senza preprocessing manuale
- produce overlay mp4 e report JSON
- mostra segmentazione fasi nel video (anche se euristica)
- calcola bottom_frame_index e depth_margin_px
- test unitari per:
  - calcolo margin e decision
  - selezione lato
  - bottom window + worst margin logic
