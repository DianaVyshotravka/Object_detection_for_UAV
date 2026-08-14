# Phase 1 results — speed screening

Format is **plain PyTorch `.pt` on CPU** (no NCNN, no quantization), so these
are *not* deployable FPS numbers — NCNN will lift all of them. What this
table is good for is the **relative ranking** of the candidate
architectures, which is what Phase 1 needs.

Latency is end-to-end `predict()` on a single frame (letterbox + forward +
NMS), batch=1. Frame decode is excluded. FPS = 1000 / mean latency,
single-stream.

Device(s): Raspberry Pi 5 Model B Rev 1.1  
Frames per config: 250  
Threads: 4  
Stack: torch 2.13.0+cpu, ultralytics 8.4.118

### FPS by model and input size

| model | 320 px | 416 px | 512 px | 640 px |
|---|---|---|---|---|
| yolo11n.pt | 12.97 | 8.78 | 6.13 | 4.18 |
| yolo11s.pt | 6.59 | 4.24 | 2.80 | 1.89 |
| yolo26n.pt | 12.67 | 8.68 | 6.16 | 4.24 |
| yolov8n.pt | 13.91 | 9.41 | 6.64 | 4.57 |

### Full metrics

| Model | imgsz | FPS | p50, ms | p95, ms | RAM, MB | Weights, MB | temp end, C | throttled |
|---|---|---|---|---|---|---|---|---|
| yolo11n.pt | 320 | 12.97 | 80.11 | 88.65 | 1065.02 | 5.35 | 69.95 |  |
| yolo11n.pt | 416 | 8.78 | 121.20 | 143.15 | 1081.78 | 5.35 | 71.05 |  |
| yolo11n.pt | 512 | 6.13 | 179.14 | 198.55 | 1066.98 | 5.35 | 72.15 |  |
| yolo11n.pt | 640 | 4.18 | 259.41 | 291.23 | 1111.64 | 5.35 | 72.15 |  |
| yolo11s.pt | 320 | 6.59 | 161.11 | 182.56 | 1118.06 | 18.42 | 74.35 |  |
| yolo11s.pt | 416 | 4.24 | 253.92 | 295.65 | 1129.22 | 18.42 | 76.55 |  |
| yolo11s.pt | 512 | 2.80 | 397.81 | 441.08 | 1185.41 | 18.42 | 75.45 |  |
| yolo11s.pt | 640 | 1.89 | 581.66 | 630.10 | 1275.34 | 18.42 | 76.55 |  |
| yolo26n.pt | 320 | 12.67 | 82.69 | 96.59 | 1074.33 | 5.29 | 69.40 |  |
| yolo26n.pt | 416 | 8.68 | 122.07 | 143.22 | 1068.58 | 5.29 | 71.05 |  |
| yolo26n.pt | 512 | 6.16 | 176.46 | 200.69 | 1090.72 | 5.29 | 70.50 |  |
| yolo26n.pt | 640 | 4.24 | 255.68 | 284.11 | 1095.05 | 5.29 | 73.80 |  |
| yolov8n.pt | 320 | 13.91 | 74.38 | 89.36 | 995.97 | 6.25 | 66.10 |  |
| yolov8n.pt | 416 | 9.41 | 111.92 | 126.13 | 1037.41 | 6.25 | 71.05 |  |
| yolov8n.pt | 512 | 6.64 | 165.00 | 186.85 | 1072.56 | 6.25 | 74.35 |  |
| yolov8n.pt | 640 | 4.57 | 238.72 | 271.03 | 1075.75 | 6.25 | 76.00 |  |

### Latency breakdown

| Model | imgsz | pre, ms | inference, ms | post, ms | end-to-end, ms |
|---|---|---|---|---|---|
| yolo11n.pt | 320 | 1.36 | 74.58 | 0.82 | 77.08 |
| yolo11n.pt | 416 | 1.65 | 110.93 | 0.96 | 113.87 |
| yolo11n.pt | 512 | 2.23 | 159.35 | 1.15 | 163.07 |
| yolo11n.pt | 640 | 3.11 | 234.20 | 1.44 | 239.23 |
| yolo11s.pt | 320 | 1.31 | 149.34 | 0.84 | 151.82 |
| yolo11s.pt | 416 | 1.65 | 232.99 | 1.00 | 235.99 |
| yolo11s.pt | 512 | 2.23 | 352.67 | 1.14 | 357.11 |
| yolo11s.pt | 640 | 3.11 | 523.36 | 1.42 | 528.78 |
| yolo26n.pt | 320 | 1.28 | 77.03 | 0.29 | 78.93 |
| yolo26n.pt | 416 | 1.56 | 113.05 | 0.30 | 115.26 |
| yolo26n.pt | 512 | 2.13 | 159.49 | 0.30 | 162.25 |
| yolo26n.pt | 640 | 3.01 | 232.46 | 0.31 | 236.12 |
| yolov8n.pt | 320 | 1.31 | 69.46 | 0.79 | 71.87 |
| yolov8n.pt | 416 | 1.66 | 103.28 | 0.94 | 106.22 |
| yolov8n.pt | 512 | 2.21 | 146.74 | 1.16 | 150.56 |
| yolov8n.pt | 640 | 3.05 | 214.04 | 1.41 | 218.85 |

### Sustained load (throttling test)

| Model | imgsz | minutes | FPS first 60s | FPS last 60s | delta % | peak temp, C | throttled |
|---|---|---|---|---|---|---|---|
| yolo11n.pt | 320 | 0.98 | 12.88 | 12.88 | 0.00 | 70.50 | none |
