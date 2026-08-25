# Phase 1 conclusions and Phase 2 gate

## Conclusions

- The three small models are eliminated for a CPU-first real-time target:
  YOLOv8s, YOLO11s, and YOLOv10s deliver only 6.59–6.76 FPS at 320 px and
  1.89–1.94 FPS at 640 px. They also use more RAM and have roughly double
  the tail latency of the nano models.
- YOLOv8n, YOLOv10n, YOLO11n, and YOLO26n remain Phase 2 candidates. At
  320 px they deliver 12.63–13.91 FPS. At 416 px, none reaches 10 FPS;
  YOLOv8n leads at 9.42 FPS.
- YOLOv8n is fastest at every size and has the lowest p95 latency among the
  nano models. YOLOv10n has the lowest RAM footprint (924 MB at 320 px),
  while YOLO26n has the lowest post-processing time and the best 640 px
  nano FPS after YOLOv8n.
- These are relative measurements, not deployment numbers: the run used
  plain PyTorch CPU rather than the NCNN format required by the selection
  plan. No candidate can be selected on speed alone yet.
- The short thermal check showed no observed drop for YOLO11n at 320 px,
  but it lasted about one minute rather than the plan's ten-minute test.
  Thermal deployability remains to be confirmed for the selected Phase 3
  configuration.

## Phase 2 decision

Run the four surviving nano models on the same `uav.yaml` with `imgsz=640`,
60 epochs, `batch=32`, `seed=0`, `deterministic=True`, `patience=0`, and the
specified aerial augmentations. Evaluate the untouched `test` split and rank
first by `mAP@0.5:0.95`, then by `military_equipment` recall. Inspect each
confusion matrix, especially vehicle/equipment confusion, before selecting
the Phase 3 finalists.

Command:

```bash
python scripts/training/run_bakeoff.py --project runs/phase2
python scripts/training/eval_results.py
```
