"""Ten-minute sustained-load test for one exported NCNN model."""

import argparse
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from bench_common import (
    append_csv, decode_throttled, device_info, latency_stats, load_frames,
    read_cpu_freq_mhz, read_cpu_temp_c, read_throttled, round_floats, set_threads,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIELDS = [
    "timestamp", "device_label", "model", "imgsz", "format", "device",
    "threads", "elapsed_s", "window_idx", "frames", "fps_window",
    "lat_p50_ms", "lat_p95_ms", "temp_c", "freq_mhz", "throttled_flags",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--frames-dir", type=Path, default=REPO_ROOT / "data/benchmark/frames")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--device-label", default=None)
    args = parser.parse_args()
    if not args.model.is_dir() or not args.frames_dir.is_dir():
        raise SystemExit("Model directory or frames directory does not exist")

    from ultralytics import YOLO

    info = device_info()
    args.device_label = args.device_label or info["board"]
    set_threads(args.threads)
    frames = load_frames(args.frames_dir)
    model = YOLO(str(args.model))
    predict_args = dict(imgsz=args.imgsz, device="cpu", conf=0.25, iou=0.7, verbose=False)
    for i in range(args.warmup):
        model.predict(frames[i % len(frames)], **predict_args)

    start = time.monotonic()
    window_latencies = []
    window_idx = 0
    rows = []
    frame_idx = 0
    all_flags = set()
    max_temp = read_cpu_temp_c()
    while time.monotonic() - start < args.minutes * 60:
        t0 = time.perf_counter()
        model.predict(frames[frame_idx % len(frames)], **predict_args)
        frame_idx += 1
        window_latencies.append((time.perf_counter() - t0) * 1000.0)
        if len(window_latencies) < args.window:
            continue
        window_idx += 1
        elapsed = time.monotonic() - start
        stats = latency_stats(window_latencies)
        temp = read_cpu_temp_c()
        flags = decode_throttled(read_throttled())
        all_flags.update(flags)
        if temp is not None and (max_temp is None or temp > max_temp):
            max_temp = temp
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device_label": args.device_label, "model": str(args.model),
            "imgsz": args.imgsz, "format": "ncnn", "device": "cpu",
            "threads": args.threads, "elapsed_s": elapsed,
            "window_idx": window_idx, "frames": len(window_latencies),
            "fps_window": 1000.0 * len(window_latencies) / sum(window_latencies),
            "lat_p50_ms": stats["lat_p50_ms"], "lat_p95_ms": stats["lat_p95_ms"],
            "temp_c": temp, "freq_mhz": read_cpu_freq_mhz(),
            "throttled_flags": ";".join(flags),
        }
        rows.append(row)
        append_csv(args.out, {field: round_floats(row).get(field, "") for field in FIELDS})
        print(f"t={elapsed:6.0f}s | {row['fps_window']:.2f} FPS | "
              f"p95 {row['lat_p95_ms']:.1f} ms | temp {temp or 'n/a'} C")
        window_latencies = []

    if not rows:
        raise SystemExit("No complete measurement window recorded")
    first = [r["fps_window"] for r in rows if r["elapsed_s"] <= 60]
    last = [r["fps_window"] for r in rows if r["elapsed_s"] >= rows[-1]["elapsed_s"] - 60]
    first_fps = statistics.fmean(first or [rows[0]["fps_window"]])
    last_fps = statistics.fmean(last or [rows[-1]["fps_window"]])
    delta = (last_fps - first_fps) / first_fps * 100 if first_fps else 0.0
    print(f"\nFirst 60s: {first_fps:.2f} FPS")
    print(f"Last 60s:  {last_fps:.2f} FPS ({delta:+.1f}%)")
    print(f"Peak temp: {max_temp if max_temp is not None else 'n/a'} C")
    print(f"Throttle:  {';'.join(sorted(all_flags)) or 'none'}")
    if delta < -10 or all_flags:
        print("VERDICT: FAIL thermal stability gate")
    else:
        print("VERDICT: PASS thermal stability gate")


if __name__ == "__main__":
    main()
