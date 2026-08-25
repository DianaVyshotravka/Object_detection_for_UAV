"""Benchmark exported NCNN models on a Raspberry Pi.

Measures end-to-end single-frame prediction on the same fixed frame set for
each model and input size. The first warm-up predictions are discarded.
Results include mean FPS, p50/p95 latency, peak process RSS, temperature, and
Raspberry Pi throttle flags.
"""

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from bench_common import (
    PeakRss, append_csv, cooldown, decode_throttled, device_info,
    latency_stats, load_frames, read_cpu_freq_mhz, read_cpu_temp_c,
    read_throttled, round_floats, set_threads,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAMES = REPO_ROOT / "data/benchmark/frames"
DEFAULT_OUT = REPO_ROOT / "benchmarks/phase3/ncnn_speed.csv"
FIELDS = [
    "timestamp", "device_label", "model", "imgsz", "format", "device",
    "threads", "n_frames", "warmup", "status", "error", "fps_mean",
    "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms", "lat_min_ms",
    "lat_max_ms", "pre_ms", "inf_ms", "post_ms", "rss_peak_mb",
    "temp_start_c", "temp_end_c", "freq_start_mhz", "freq_end_mhz",
    "throttled_flags", "board", "machine", "torch_version",
    "ultralytics_version",
]


def benchmark_one(model_path: Path, imgsz: int, frames: list, args, info: dict) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    predict_args = dict(imgsz=imgsz, device="cpu", conf=args.conf,
                        iou=args.iou, verbose=False)
    for i in range(args.warmup):
        model.predict(frames[i % len(frames)], **predict_args)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device_label": args.device_label, "model": str(model_path),
        "imgsz": imgsz, "format": "ncnn", "device": "cpu",
        "threads": args.threads, "n_frames": 0, "warmup": args.warmup,
        "status": "ok", "error": "", "board": info["board"],
        "machine": info["machine"], "torch_version": info["torch_version"],
        "ultralytics_version": info["ultralytics_version"],
        "temp_start_c": read_cpu_temp_c(),
        "freq_start_mhz": read_cpu_freq_mhz(),
    }
    latencies = []
    totals = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    with PeakRss() as rss:
        for i in range(args.n_frames):
            start = time.perf_counter()
            result = model.predict(frames[i % len(frames)], **predict_args)[0]
            latencies.append((time.perf_counter() - start) * 1000.0)
            for key in totals:
                totals[key] += result.speed.get(key, 0.0) or 0.0

    row.update(latency_stats(latencies))
    row["n_frames"] = len(latencies)
    row["pre_ms"] = totals["preprocess"] / len(latencies)
    row["inf_ms"] = totals["inference"] / len(latencies)
    row["post_ms"] = totals["postprocess"] / len(latencies)
    row["rss_peak_mb"] = rss.peak_mb
    row["temp_end_c"] = read_cpu_temp_c()
    row["freq_end_mhz"] = read_cpu_freq_mhz()
    row["throttled_flags"] = ";".join(decode_throttled(read_throttled()))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="+", required=True,
                        help="Pairs in MODEL_DIR:IMGSZ form")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-frames", type=int, default=250)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--cooldown-c", type=float, default=60.0)
    parser.add_argument("--cooldown-timeout", type=float, default=300.0)
    parser.add_argument("--device-label", default=None)
    args = parser.parse_args()
    if args.n_frames < 200:
        raise SystemExit("--n-frames must be at least 200 for the real benchmark")
    configs = []
    for value in args.configs:
        try:
            model_text, size_text = value.rsplit(":", 1)
            model_path, size = Path(model_text), int(size_text)
        except ValueError as exc:
            raise SystemExit(f"Invalid config {value!r}; use MODEL_DIR:IMGSZ") from exc
        configs.append((model_path, size))
    missing = [str(p) for p in [*(p for p, _ in configs), args.frames_dir] if not p.exists()]
    if missing:
        raise SystemExit("Missing input(s):\n" + "\n".join(missing))

    info = device_info()
    args.device_label = args.device_label or info["board"]
    set_threads(args.threads)
    frames = load_frames(args.frames_dir)
    print(f"Device: {args.device_label} | frames: {len(frames)} | NCNN")
    for model_path, size in configs:
        print(f"=== {model_path} | imgsz={size} ===")
        cooldown(args.cooldown_c, args.cooldown_timeout)
        try:
            row = benchmark_one(model_path, size, frames, args, info)
            print(f"{row['fps_mean']:.2f} FPS | p50 {row['lat_p50_ms']:.1f} ms | "
                  f"p95 {row['lat_p95_ms']:.1f} ms | RAM {row['rss_peak_mb']:.0f} MB")
        except Exception as exc:  # keep the grid running
            row = {"timestamp": datetime.now(timezone.utc).isoformat(),
                   "model": str(model_path), "imgsz": size, "format": "ncnn",
                   "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            print(f"FAILED: {exc}")
        append_csv(args.out, {field: round_floats(row).get(field, "")
                              for field in FIELDS})
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
