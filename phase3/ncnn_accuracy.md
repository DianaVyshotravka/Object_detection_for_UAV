# Raspberry Pi NCNN accuracy

Full held-out test-split validation of exported NCNN models.
Compare with `docs/phase3_imgsz_results.md` from the PyTorch checkpoints.

| Model | imgsz | mAP@0.5 | mAP@0.5:0.95 | military recall | person recall | vehicle recall |
|---|---:|---:|---:|---:|---:|---:|
| models/ncnn/yolo26n_320_ncnn_model | 320 | 0.3519 | 0.1678 | 0.5671 | 0.1564 | 0.3975 |
| models/ncnn/yolo26n_416_ncnn_model | 416 | 0.4444 | 0.2170 | 0.6306 | 0.2350 | 0.4874 |
| models/ncnn/yolo26n_512_ncnn_model | 512 | 0.5007 | 0.2511 | 0.6639 | 0.2905 | 0.5454 |
| models/ncnn/yolo26n_640_ncnn_model | 640 | 0.5519 | 0.2830 | 0.6918 | 0.3449 | 0.5981 |
| models/ncnn/yolo11n_320_ncnn_model | 320 | 0.3113 | 0.1441 | 0.5200 | 0.1261 | 0.3681 |
| models/ncnn/yolo11n_416_ncnn_model | 416 | 0.3992 | 0.1904 | 0.6047 | 0.1754 | 0.4646 |
| models/ncnn/yolo11n_512_ncnn_model | 512 | 0.4660 | 0.2264 | 0.6525 | 0.2214 | 0.5274 |
| models/ncnn/yolo11n_640_ncnn_model | 640 | 0.5137 | 0.2544 | 0.7106 | 0.2563 | 0.5833 |
