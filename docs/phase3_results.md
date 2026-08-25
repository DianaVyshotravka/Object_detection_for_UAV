# Phase 3 results and conclusions

Phase 3 evaluated the two finalists, YOLO26n and YOLO11n, after export to
NCNN on a Raspberry Pi 5 Model B. Accuracy was measured on the complete
703-image UAV test split; runtime was measured on 250 frames per
configuration after 20 warm-up inferences.

## NCNN runtime results

| Model | imgsz | FPS | p50 latency, ms | p95 latency, ms | RAM, MB |
|---|---:|---:|---:|---:|---:|
| YOLO26n | 320 | 59.19 | 15.7 | 16.2 | 934 |
| YOLO26n | 416 | 36.66 | 25.9 | 27.3 | 944 |
| YOLO26n | 512 | **23.43** | 40.6 | 47.0 | 954 |
| YOLO26n | 640 | 14.96 | 63.8 | 71.4 | 973 |
| YOLO11n | 320 | 55.63 | 16.8 | 17.7 | 938 |
| YOLO11n | 416 | 34.78 | 27.6 | 28.6 | 946 |
| YOLO11n | 512 | 22.05 | 43.5 | 50.1 | 957 |
| YOLO11n | 640 | 13.95 | 68.8 | 83.0 | 970 |

All short-run configurations completed without throttle flags.

## NCNN accuracy results

| Model | imgsz | mAP@0.5 | mAP@0.5:0.95 | military recall | person recall | vehicle recall |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26n | 320 | 0.3519 | 0.1678 | 0.5671 | 0.1564 | 0.3975 |
| YOLO26n | 416 | 0.4444 | 0.2170 | 0.6306 | 0.2350 | 0.4874 |
| YOLO26n | 512 | 0.5007 | **0.2511** | **0.6639** | 0.2905 | 0.5454 |
| YOLO26n | 640 | **0.5519** | **0.2830** | 0.6918 | **0.3449** | **0.5981** |
| YOLO11n | 320 | 0.3113 | 0.1441 | 0.5200 | 0.1261 | 0.3681 |
| YOLO11n | 416 | 0.3992 | 0.1904 | 0.6047 | 0.1754 | 0.4646 |
| YOLO11n | 512 | 0.4660 | 0.2264 | 0.6525 | 0.2214 | 0.5274 |
| YOLO11n | 640 | 0.5137 | 0.2544 | **0.7106** | 0.2563 | 0.5833 |

## NCNN conversion delta

Compared with the same checkpoints evaluated in PyTorch:

| Model | imgsz | mAP@0.5:0.95 delta | Military recall delta |
|---|---:|---:|---:|
| YOLO26n | 320 | -0.0094 | -0.0324 |
| YOLO26n | 416 | -0.0142 | -0.0174 |
| YOLO26n | 512 | -0.0107 | +0.0098 |
| YOLO26n | 640 | -0.0032 | -0.0047 |
| YOLO11n | 320 | -0.0232 | -0.0800 |
| YOLO11n | 416 | -0.0255 | -0.0541 |
| YOLO11n | 512 | -0.0305 | -0.0605 |
| YOLO11n | 640 | -0.0296 | -0.0282 |

YOLO26n preserves accuracy considerably better during NCNN conversion.

## Thermal soak

Both 512 px configurations ran for approximately ten minutes with four CPU
threads:

| Model | Duration | FPS first 60s | FPS last 60s | Change | Peak temp | Throttling |
|---|---:|---:|---:|---:|---:|---|
| YOLO11n 512 | 9.99 min | 21.83 | 21.85 | +0.07% | 68.3°C | None |
| YOLO26n 512 | 9.98 min | 23.00 | 23.31 | +1.32% | 67.75°C | None |

Both configurations pass the thermal stability gate: no FPS drop greater
than 10% and no throttle flags.

## Conclusion

The recommended deployment configuration is:

```text
YOLO26n, 512 px, NCNN, Raspberry Pi 5
```

It provides approximately 23.4 FPS, 47 ms p95 latency, 0.2511 NCNN
mAP@0.5:0.95, and 0.6639 military-equipment recall while remaining thermally
stable.

YOLO26n at 640 has the highest accuracy, but its speed falls to approximately
15 FPS. YOLO11n at 640 has the highest military-equipment recall (`0.7106`),
so it remains an alternative if that metric is more important than overall
mAP and speed. Under the combined accuracy/speed criteria, YOLO26n at 512 is
the strongest balanced choice.

Source artifacts:

- `phase3/ncnn_speed_retry.csv`
- `phase3/ncnn_accuracy.md`
- `phase3/soak_yolo11n_512.csv`
- `phase3/soak_yolo26n_512.csv`
- `docs/phase3_imgsz_results.md`
