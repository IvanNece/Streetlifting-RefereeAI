# PRD — AI Squat Referee (IPF Standard)

## Cos'è

Script Python che prende in input un video di uno squat (vista laterale) e produce:
- Un video annotato con overlay grafici
- Un report testuale con il giudizio per ogni ripetizione

## Utenti Target

- Atleti di powerlifting che vogliono verificare la profondità in allenamento
- Coach che analizzano i video dopo la sessione
- Arbitri che vogliono un secondo parere rapido in gara

## Cosa fa (MVP)

1. Legge un file video `.mp4` / `.mov`
2. Rileva i keypoint dell'atleta frame per frame
3. Identifica automaticamente inizio e fine di ogni ripetizione
4. Per ogni rep, determina se la profondità IPF è stata raggiunta (anca sotto il ginocchio)
5. Genera un video di output con overlay (keypoint, linee, semaforo GOOD/NO LIFT)
6. Stampa un report a schermo con il riepilogo

## Cosa NON fa (fuori scope per ora)

- Vista frontale o diagonale
- Live webcam
- Interfaccia grafica
- Bench press / stacco
- Multi-atleta nello stesso frame

## Criteri di Successo

| Cosa | Target |
|---|---|
| Accuratezza giudizio | ≥ 90% concordanza con giudice umano |
| Velocità | Elabora un video in ≤ 2× la sua durata su laptop normale |
| Usabilità | Un solo comando da terminale per far girare tutto |

## User Story Principale

> Come atleta, voglio trascinare un video nella cartella e lanciare uno script per sapere se le mie ripetizioni erano valide IPF, senza installare roba complicata.