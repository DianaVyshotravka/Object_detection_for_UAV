# Onboard detection service

Runs the UAV object detector on a Raspberry Pi 4/5 against a live camera,
tags every result with geo reference from the flight controller (MAVLink), and writes
geotagged JPEGs plus a JSONL log of detections. Optionally serves a local web
page with the annotated video and a live map.

```
camera ──► video thread ──► frame queue (newest 2) ──► pipeline thread ──► flights/<start time>/
                                                          ▲      │  │         detections.jsonl
flight controller ──► mavlink thread ──► telemetry buffer ┘      │  └► saver thread ──► *.jpg
        ▲                                                         │     (queue of 8)    (boxes, GPS EXIF)
        │                                                         └──► live view (optional)
        └── clock_sync thread (TIMESYNC: Pi clock ↔ FC clock)          http://<pi>:8080/
```

| File | Role |
|---|---|
| `main.py` | entry point: init, thread wiring, signal handling, shutdown |
| `config.yaml` / `config.py` | all runtime settings |
| `core/video_stream.py` | camera capture (picamera2 for CSI, OpenCV for USB/video files) |
| `core/mavlink_client.py` | flight-controller link, the only MAVLink reader, auto-reconnect |
| `core/clock_sync.py` | Pi ↔ FC clock offset (TIMESYNC) and GPS UTC (SYSTEM_TIME) |
| `core/detector_model.py` | NCNN inference via Ultralytics |
| `core/pipeline.py` | inference, telemetry matching, JSONL log; `ImageSaver` thread writes the JPEGs |
| `core/geotag.py` | GPS EXIF write/read |
| `core/live_view.py` | optional web page: MJPEG stream + detection map |
| `build_map.py` | after the flight: map of one run's detections |
| `systemd/uav-detector.service` | start on boot, restart on failure |

---

## 1. Hardware

- Raspberry Pi 4 or 5, **64-bit** Raspberry Pi OS (Bookworm), active cooling.
- Camera: CSI ribbon camera (Camera Module 2/3, HQ) **or** a USB/UVC camera.
- Flight controller running ArduPilot or PX4 (optional, without it frames
  are saved without GPS).
- SD card with room for images (`save_mode: detections` writes only frames
  with objects; `all` is ~25 GB/hour).

### Wiring the flight controller (UART)

Use a free TELEM/SERIAL port on the FC. Logic levels are 3.3 V on both sides.

| Pi pin | | FC TELEM port |
|---|---|---|
| GPIO14 / pin 8 (TX) | → | RX |
| GPIO15 / pin 10 (RX) | ← | TX |
| GND / pin 6 | — | GND |


### Flight controller parameters

ArduPilot, for the port you wired (example: `SERIAL2`):

```
SERIAL2_PROTOCOL = 2      # MAVLink2
SERIAL2_BAUD     = 57     # 57600 (must match mavlink.baud); 921 for 921600
```

PX4: set `MAV_1_CONFIG` to that TELEM port, `MAV_1_MODE = Onboard`,
`SER_TELx_BAUD` to match `mavlink.baud`.

The service asks the FC for `GLOBAL_POSITION_INT` at `mavlink.telemetry_hz`
(20 Hz). If ArduPilot sends fewer, raise `SRx_POSITION` for that port.

---

## 2. Prepare the Pi

### Enable the UART, disable the serial console

```bash
sudo raspi-config
#   Interface Options → Serial Port
#     login shell over serial?  → No
#     serial port hardware?     → Yes
sudo reboot
```

Which device to put in `mavlink.url`:

| Board | GPIO14/15 device | Notes |
|---|---|---|
| Pi 5 | `/dev/ttyAMA0` | add `dtparam=uart0=on` to `/boot/firmware/config.txt`. On a Pi 5, `/dev/serial0` may point at the separate debug-UART connector. Check with `ls -l /dev/serial0`. |
| Pi 4 | `/dev/serial0` | add `dtoverlay=disable-bt` to `/boot/firmware/config.txt` so GPIO14/15 gets the full PL011 UART instead of the mini UART, then reboot. |

### User groups

The service user needs serial and camera access:

```bash
sudo usermod -aG dialout,video $USER
# log out and back in (or reboot)
```

### Check the camera

```bash
rpicam-hello --list-cameras        # CSI camera
ls /dev/video*                     # USB camera
```

---

## 3. Install

### Copy the code and the model from the dev machine

The deploy root on the Pi is `~/uav` (the parent of `onboard/`). Relative paths
in `config.yaml` resolve against it.

```bash
PI=<user>@<pi-hostname>.local
MODEL=models/yolo26n_512_visdrone_crops_v1/yolo26n_512_visdrone_crops_v1_512_ncnn_model

rsync -av --exclude __pycache__ onboard/ "$PI:~/uav/onboard/"
rsync -av "$MODEL/" "$PI:~/uav/$MODEL/"
```

### Python environment (on the Pi)

`picamera2` is an apt package and cannot be pip-installed, so the venv must
see system packages:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-venv

cd ~/uav
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics ncnn pymavlink opencv-python-headless PyYAML Pillow
```


Check:

```bash
python -c "import picamera2, pymavlink, cv2, ultralytics; print('ok')"
uname -m                    # aarch64
vcgencmd get_throttled      # throttled=0x0
```

---

## 4. Configure

Edit `~/uav/onboard/config.yaml`. The settings that usually change:

| Setting | Typical value | Meaning |
|---|---|---|
| `camera.source` | `"csi"` / `"0"` / `/path/video.mp4` | CSI camera, USB camera index, or a recorded video |
| `camera.width`, `camera.height` | `1920`, `1080` | capture resolution |
| `camera.latency_ms` | `0` | USB cameras only: measured delay from exposure to delivery. CSI frames carry the sensor timestamp. |
| `mavlink.enabled` | `true` | `false` = run without a flight controller |
| `mavlink.url` | `/dev/ttyAMA0` | serial device (see table above) or `udp:127.0.0.1:14550` for SITL |
| `mavlink.baud` | `57600` | must match the FC's `SERIALx_BAUD` |
| `detector.model_path` | NCNN model directory | relative to `~/uav` |
| `detector.imgsz` | `512` | must match the NCNN export |
| `detector.conf` | `0.25` | detection confidence threshold |
| `detector.threads` | `3` | NCNN CPU threads. 4 is fastest alone but stalls while the camera and saver threads run; 3 keeps one core free |
| `storage.save_mode` | `detections` | `detections` = only frames with objects, `all` = every frame |
| `storage.draw_boxes` | `true` | draw boxes and labels on saved images; `false` = raw frames (boxes stay in `detections.jsonl`) |
| `storage.save_min_interval_s` | `0.0` | minimum seconds between saved images |
| `storage.min_free_mb` | `512` | below this free space, stop saving images (detections are still logged) |
| `live.enabled` | `false` | web page with video and map, see §7 |

Unknown keys are rejected at startup, so a typo fails loudly instead of being
silently ignored.

---

## 5. First run (manually)

```bash
cd ~/uav
source .venv/bin/activate
python -m onboard.main --config onboard/config.yaml
```

Healthy startup log:

```
INFO mavlink: heartbeat from system 1 component 1
INFO storage: writing to /home/<user>/uav/flights/2026-10-24_14-03-12
INFO clock: bootstrapped offset ... s from telemetry
INFO detector: loaded .../yolo26n_..._ncnn_model at imgsz=512, 3 threads
INFO onboard service running (5 threads)
INFO camera: 690 frames read, 12 dropped          # every 30 s
INFO pipeline: 650 frames, 41 saved, 97 detections, 0 saves dropped
```

Stop with **Ctrl+C**. It shuts down cleanly in about 1 s.

### Dry runs without hardware

- **No flight controller:** `mavlink.enabled: false`. Frames are saved
  without GPS.
- **No camera:** `camera.source: /path/to/flight.mp4` with `camera.backend: cv2`.
- **Simulator:** run ArduPilot SITL and set `mavlink.url: udp:127.0.0.1:14550`.

---

## 6. Run on boot (systemd)

The unit file assumes user `pi` and deploy root `/home/pi/uav`. Bookworm has
no default `pi` user, so edit `User=`, `WorkingDirectory=` and the two paths
in `ExecStart=` to match yours first.

```bash
sudo cp ~/uav/onboard/systemd/uav-detector.service /etc/systemd/system/
sudo nano /etc/systemd/system/uav-detector.service     # fix user and paths
sudo systemctl daemon-reload
sudo systemctl enable --now uav-detector
```

Day to day:

```bash
systemctl status uav-detector
journalctl -u uav-detector -f          # live log
sudo systemctl restart uav-detector    # after editing config.yaml
sudo systemctl stop uav-detector       # before pulling the SD card
```

Behaviour:
- A camera failure, a crashed thread or a failed model load exits with code 1,
  and systemd restarts the service after 5 s. After 5 failures in 60 s it
  gives up. Check `journalctl` in that case.
- A missing flight controller does **not** stop the service. It runs without
  GPS and reconnects every 5 s; `mavlink: link up` appears when the FC is
  found.
- The log also goes to `flights/onboard.log` (rotated, max 4 × 5 MB).

---

## 7. Live view (optional)

```yaml
live:
  enabled: true
  host: 127.0.0.1    # browser on the Pi itself
  port: 8080
```

Open `http://127.0.0.1:8080/` on the Pi. To watch from a laptop on the same
network set `host: 0.0.0.0` and open `http://<pi-hostname>.local:8080/`.

- Left: the video at 960 px / 5 FPS, with boxes, class and confidence.
- Right: the map. Red points show where the drone was when objects were seen
  (click for counts), blue is the current drone position. Refreshes every 2 s.
- No viewer connected = no extra CPU. Each open viewer costs one JPEG encode
  per stream frame.
- The map loads Leaflet and OpenStreetMap tiles from the internet. Without an
  uplink, the video works and the map stays blank.

> **Security:** the page has no password. With `host: 0.0.0.0` anyone on
> the network can see the video and the drone's position. Use it only on a
> trusted link.

---

## 8. Output

Each start of the service creates a new folder named after the Pi's clock:

```
~/uav/flights/
  onboard.log
  2026-10-24_14-03-12/
    000017.jpg            # geotagged frame (GPS position, altitude, heading, UTC time in EXIF)
    000018.jpg
    detections.jsonl      # one line per processed frame
  2026-10-24_15-20-41/
    ...
```

The Pi has no battery clock, so without network time the folder name may be
off. A clash gets a `_2` suffix, never an overwrite. The UTC time **inside**
the records and EXIF comes from GPS once the FC has a fix.

One `detections.jsonl` record:

```json
{"frame_id": 17, "timestamp": "2026-10-24T11:03:14.512Z", "image_size": [1920, 1080],
 "detections": [{"class": "vehicle", "conf": 0.8756, "bbox_xyxy": [812.0, 403.5, 870.2, 441.0]}],
 "image": "000017.jpg",
 "telemetry": {"lat": 50.4501234, "lon": 30.5234567, "alt_msl_m": 182.4, "rel_alt_m": 60.1, "heading_deg": 93.5}}
```

`image` is `null` when the frame was not saved; `telemetry` is `null` before
the first GPS sample, or if none was within `max_telemetry_dt_s` of the frame.

### Map after the flight

Copy a run folder off the Pi and build an HTML map plus GeoJSON (opens in
QGIS):

```bash
rsync -av "$PI:~/uav/flights/2026-10-24_14-03-12/" flights/2026-10-24_14-03-12/
python -m onboard.build_map --flight-dir flights/2026-10-24_14-03-12
# -> flights/2026-10-24_14-03-12/map.html, track.geojson
```

---

## 9. Troubleshooting

| Log message | Cause / fix |
|---|---|
| `mavlink: no MAVLink heartbeat from ... -- retrying every 5 s` | Wrong `mavlink.url` or baud, TX/RX swapped, serial console still enabled, or FC port not set to MAVLink |
| `pymavlink is not installed` | `pip install pymavlink` inside the venv |
| `Permission denied: '/dev/ttyAMA0'` | user not in `dialout` group |
| `mavlink: no position for N s, re-requesting streams` | FC rebooted or ignores the stream request. Check `SRx_POSITION` / `MAV_x_MODE`. |
| `camera: picamera2 unavailable (...), falling back to cv2` | venv created without `--system-site-packages`, or `python3-picamera2` not installed |
| `cannot open camera source '0'` | no USB camera at that index, check `ls /dev/video*` |
| `camera: N consecutive read failures, stopping capture` | camera unplugged or ribbon loose; systemd restarts the service |
| `camera: SensorTimestamp ... off monotonic, using delivery time` | CSI timestamp not usable on this kernel; frames fall back to delivery time minus `latency_ms` |
| `Model not found: ...` | model directory not copied, or `model_path` wrong |
| `detector: load failed: ...` | `ncnn` / `ultralytics` missing in the venv, or `imgsz` does not match the export |
| `storage: ... MB free < 512 MB floor, no longer saving images` | SD card full; detections still go to the JSONL |
| `live view: cannot start ([Errno 98] Address already in use)` | another process on `live.port`; the service runs on without the page |
| `thread X died unexpectedly, stopping` | bug or hardware fault in that thread; the traceback is just above in the log |
| `vcgencmd get_throttled` ≠ `0x0` | undervoltage or overheating. Fix power and cooling before trusting FPS. |

---

## 10. Tests

Run on the dev machine. No camera, FC, pymavlink or model needed:

```bash
pytest onboard/tests -q
```
