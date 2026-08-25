# Phase 2 accuracy results

Test split evaluation after the fixed 60-epoch bake-off. See each
Ultralytics run directory for confusion matrices and PR curves.

| Model | imgsz | mAP@0.5 | mAP@0.5:0.95 | military recall | person recall | vehicle recall |
|---|---:|---:|---:|---:|---:|---:|
| yolov8n | 640 | 0.5083 | 0.2700 | 0.7012 | 0.2633 | 0.5593 |
| yolov10n | 640 | 0.4780 | 0.2470 | 0.6541 | 0.2479 | 0.5610 |
| yolo11n | 640 | 0.5231 | 0.2839 | 0.7388 | 0.2563 | 0.5799 |
| yolo26n | 640 | 0.5266 | 0.2862 | 0.6965 | 0.3216 | 0.5619 |
