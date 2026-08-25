# Phase 2 decision and Phase 3 handoff

Phase 2 used the four surviving nano candidates from Phase 1. All runs used
the same dataset, `imgsz=640`, 60 epochs, `batch=32`, seed 0,
`deterministic=True`, `patience=0`, and the prescribed aerial augmentation
overrides. The held-out test split contains 703 images and 9,740 instances.

| Model | mAP@0.5 | mAP@0.5:0.95 | military recall | Decision |
|---|---:|---:|---:|---|
| YOLO26n | 0.527 | **0.286** | **0.696** | **Primary finalist** |
| YOLO11n | 0.523 | 0.284 | 0.739 | Secondary finalist: strongest equipment recall |
| YOLOv8n | 0.508 | 0.270 | 0.701 | Eliminate |
| YOLOv10n | 0.478 | 0.247 | 0.654 | Eliminate |

YOLO26n is the accuracy finalist because it has the best mAP@0.5:0.95 and
the best mAP@0.5. YOLO11n remains in Phase 3 because it has higher
`military_equipment` recall, and the documented decision rule prioritizes that
class after the FPS constraint is applied. The confusion matrices show that
vehicle/equipment confusion remains a material error for both finalists.

## Phase 3 action

Run `scripts/training/eval_imgsz_sweep.py` for both finalists at 320, 416,
512, and 640, then join its accuracy table with the same-size NCNN benchmark
from the Raspberry Pi. Do not make a final deployment choice until the target
FPS and thermal-soak results are available.

The generated artifacts are `docs/phase3_imgsz_results.md` and
`docs/phase3_imgsz_results.json`.
