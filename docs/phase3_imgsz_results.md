# Phase 3 input-size accuracy sweep

Existing Phase 2 checkpoints evaluated on the untouched test split.
Runtime columns are intentionally left for the Raspberry Pi NCNN benchmark.

| Model | imgsz | mAP@0.5 | mAP@0.5:0.95 | military recall | person recall | vehicle recall | FPS (NCNN) | p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo26n | 320 | 0.3551 | 0.1773 | 0.5995 | 0.1602 | 0.4079 | — | — |
| yolo26n | 416 | 0.4415 | 0.2312 | 0.6480 | 0.2205 | 0.4791 | — | — |
| yolo26n | 512 | 0.4903 | 0.2618 | 0.6541 | 0.2703 | 0.5241 | — | — |
| yolo26n | 640 | 0.5266 | 0.2862 | 0.6965 | 0.3216 | 0.5619 | — | — |
| yolo11n | 320 | 0.3437 | 0.1674 | 0.6000 | 0.1253 | 0.3895 | — | — |
| yolo11n | 416 | 0.4220 | 0.2159 | 0.6588 | 0.1759 | 0.4755 | — | — |
| yolo11n | 512 | 0.4860 | 0.2569 | 0.7129 | 0.2173 | 0.5325 | — | — |
| yolo11n | 640 | 0.5231 | 0.2839 | 0.7388 | 0.2563 | 0.5799 | — | — |
