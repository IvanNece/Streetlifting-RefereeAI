# GEMINI.md — Agent Operating Rules (High-Precision Dip Validator)

## 0) Mission
Implementare una pipeline CV/ML con alta precisione per giudicare la profondita del Dip da video smartphone.

Modello di default per implementazione: Gemini (costi/velocita).
Claude Opus 4.5:
- usalo per planning, architettura, scelte difficili, oppure debugging duro dopo 2 tentativi falliti.

## 1) Non-negotiables
- Non dichiarare “done” senza aver eseguito i Quality Gates.
- Cambi piccoli e reviewabili: una tranche per volta.
- Nessun refactor creativo fuori scope.
- Nessuna dipendenza pesante senza motivazione e alternativa.

## 2) Core constraints
- NO MediaPipe.
- SI MMPose/RTMPose (OpenMMLab) per pose di base.
- Devi gestire video “raw” da telefono (rotazione possibile).
- Deve funzionare anche a ~45 gradi con warnings quando l’affidabilita scende.

## 3) Landmark reality check (fondamentale)
Non assumere che 133 keypoints includano il deltoide posteriore:
- COCO-WholeBody 133 keypoints = body + feet + face + hands.
Per ottenere posterior deltoid apex e elbow tip serve un modulo dedicato (landmark refinement).

## 4) Required deliverables (v1)
- CLI che legge input_videos/<video> e produce in output/:
  - overlay mp4 con fasi + punti/linee + decisione
  - report JSON con valid/invalid, bottom_frame_index, margin_px, selected_side, warnings
- Algoritmo:
  - coarse pose (RTMPose) + refinement (D ed E)
  - bottom detection con smoothing
  - decision su bottom window (worst margin)

## 5) Quality Gates
- install deps
- unit tests
- run E2E su almeno un video e generare overlay + JSON

## 6) Output expectations per tranche
- cosa e cambiato
- comandi esatti per verificare
- files toccati
- limiti noti + next step
