# Raspberry Pi onboard bring-up: camera, flight controller, final model

Date: 2026-10-25. Target: `beacon@raspberrypi.local`.

The Pi now runs the final model, `yolo26n_512_visdrone_crops_v1` (NCNN,
512 px), as a systemd service that starts on boot. The camera and the
flight controller (FC) both work. Results:

| Check | Result |
|---|---|
| Camera | OV64A40 (Arducam 64MP OwlSight) detected on CAM0. Captures work. |
| FC link | ArduPlane 4.6.2, MAVLink2, `/dev/ttyAMA0` @ 115200, heartbeat and position streams OK |
| Model speed (benchmark) | **22.07 FPS**, p50 45.3 ms, p95 45.7 ms, 957 MB RSS, no throttling |
| Model speed (live, camera + FC) | **~18.7 FPS** processed (camera delivers 30 FPS at 1920x1080) |
| Accuracy (703-image test split, NCNN on the Pi) | mAP@0.5 **0.552**, mAP@0.5:0.95 **0.305**, military recall **0.699** |
| Onboard mode | `uav-detector.service` enabled. Autostart confirmed after a reboot. |

---

## 1. Hardware and OS

| Item | Value |
|---|---|
| Board | Raspberry Pi 5 Model B Rev 1.1, 16 GB RAM, active cooling |
| OS | Debian 13 (trixie), kernel 6.18.50+rpt-rpi-2712, aarch64 |
| Python | 3.13.5, venv at `~/uav/.venv` (`--system-site-packages` for picamera2) |
| Key packages | ultralytics 8.4.189, torch 2.14.1+cpu, ncnn, pymavlink, opencv-python-headless, python3-picamera2 0.3.37 (apt) |
| Idle state | 39.5 °C, `throttled=0x0` |

## 2. Camera

The connected module is the **Arducam 64MP OwlSight (OV64A40)**. 

Final `/boot/firmware/config.txt` changes:

```
camera_auto_detect=0                                  # was 1
[all]
dtoverlay=ov64a40,clock-frequency=37125000,cam0       # unchanged from the original
dtparam=uart0=on                                      # new, see §3
```

`rpicam-hello --list-cameras`:

```
0 : ov64a40 [9248x6944 10-bit]
    1920x1080 [70.56 fps - (784, 1312)/7712x4352 crop]
    2312x1736 [32.24 fps - (0, 0)/9280x6976 crop]
    3840x2160 [19.40 fps - (784, 1312)/7712x4352 crop]
    4624x3472 [9.86 fps]  8000x6000 [3.18 fps]  9248x6944 [2.60 fps]
```


## 3. Flight controller (MAVLink)

Wiring: Pi GPIO14/15 (UART0). On a Pi 5 that is `/dev/ttyAMA0`, which
appears only with `dtparam=uart0=on`. `/dev/serial0` points at the separate
debug UART (`ttyAMA10`), which carries the kernel console. That console
does not conflict with GPIO14/15.

A raw sniff found MAVLink2 framing (`0xFD`) at 115200. pymavlink confirmed:

```
115200 HEARTBEAT sys 1 comp 0 type MAV_TYPE_FIXED_WING autopilot MAV_AUTOPILOT_ARDUPILOTMEGA v2
AUTOPILOT_VERSION flight_sw_version 4.6.2 (official), git 1ebd4d99
```

Messages received within 5 s include `GLOBAL_POSITION_INT`, `GPS_RAW_INT`,
`ATTITUDE`, `SYSTEM_TIME`, `SYS_STATUS`, `VFR_HUD`, `BATTERY_STATUS`,
`EKF_STATUS_REPORT` and `AUTOPILOT_VERSION` (answered to
`MAV_CMD_REQUEST_MESSAGE`). Battery 30.8 V.

The GPS had **no fix** (`fix_type 0`, 0 satellites) because the test was
indoors. Until there is a fix, `lat/lon` are 0 and `SYSTEM_TIME.time_unix_usec`
is 0, so saved images get no usable geotag. Verify outdoors before a flight.

## 4. Model speed on the Pi

Model: `models/yolo26n_512_visdrone_crops_v1/yolo26n_512_visdrone_crops_v1_512_ncnn_model`
(fine-tuned from `final_yolo26n_512-3` on VisDrone crops).

Benchmark: `scripts/benchmark/bench_ncnn.py`, 4 threads, 20 warm-up runs and
250 timed runs on `data/benchmark/frames`, end-to-end `predict()`:

| FPS | mean | p50 | p95 | p99 | pre / inf / post | RSS peak | temp start → end | throttling |
|---:|---:|---:|---:|---:|---|---:|---|---|
| **22.07** | 45.3 ms | 45.3 ms | 45.7 ms | 45.9 ms | 2.9 / 40.8 / 1.1 ms | 957 MB | 49.6 → 55.7 °C | none |

Phase 3 measured 23.43 FPS for the earlier YOLO26n 512 checkpoint (same
architecture) on the same board. The ~6% gap is within the difference
between software stacks (Debian trixie, ultralytics 8.4.189 / torch 2.14).


## 5. Accuracy on the Pi

The NCNN model was run on the Pi over the full held-out test split
(`uav_mar20_cap100_v25/test`: 703 images, 9740 objects), conf/iou at the
Ultralytics val defaults (`scripts/benchmark/validate_ncnn.py`):

| Class | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| all | 703 | 9740 | 0.652 | 0.546 | **0.552** | **0.305** |
| person | 238 | 3433 | 0.577 | 0.341 | 0.361 | 0.136 |
| vehicle | 431 | 5882 | 0.711 | 0.599 | 0.647 | 0.385 |
| military_equipment | 262 | 425 | 0.667 | **0.699** | 0.648 | 0.392 |

Against Phase 3 YOLO26n 512 NCNN (0.5007 / 0.2511, military recall 0.664),
the final fine-tuned model scores +0.051 mAP@0.5, +0.053 mAP@0.5:0.95 and
+0.035 military recall at the same speed. Its accuracy is close to the
Phase 3 640 px model (0.552 / 0.283) at about 1.5x the speed.

Raw data: `benchmarks/onboard/ncnn_accuracy.json`, `.md`.

Fixed along the way: `validate_ncnn.py` and
`scripts/training/eval_results.py` wrote per-class `class_result()` values
under the wrong keys (mAP@0.5 as `ap50_95`, mAP@0.5:0.95 as `f1`). The keys
are now `ap50` and `ap50_95`. Only `recall` was used in earlier reports, so
the published tables are unaffected. Older JSON files still carry the old
labels.

## 6. Onboard mode

Layout on the Pi (deploy root `~/uav`):

```
~/uav/
  .venv/                     python env (system site packages for picamera2)
  onboard/                   service code + config.yaml
  models/yolo26n_512_visdrone_crops_v1/..._512_ncnn_model/
  scripts/benchmark/         benchmark/validation scripts
  data/benchmark/frames/     250 timing frames
  data/uav-test/ + data/uav-test.yaml   703-image test split
  benchmarks/onboard/        results of this run
  flights/                   service output (one folder per start) + onboard.log
```

`onboard/config.yaml` changes for this hardware:

| Setting | Was | Now |
|---|---|---|
| `camera.source` | `"0"` (USB) | `"csi"` |
| `mavlink.url` | `/dev/serial0` | `/dev/ttyAMA0` |
| `mavlink.baud` | `57600` | `115200` |
