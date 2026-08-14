"""Sustained-load throttling test (model_selection.md Phase 1,
project_plan.md §3).

`bench_speed.py` measures a cold, short burst. That is the optimistic
number. This runs one config flat out for ten minutes and logs FPS +
temperature per window, because "holds 12 FPS for 30 seconds and sags at
minute 5" is not deployable -- and on a drone the CPU never gets a break.

Verdict printed at the end: first-minute FPS vs last-minute FPS, the
delta, peak temperature, and every throttle bit that fired at any point.

Usage:
    python thermal_soak.py --model yolo11n.pt --imgsz 416
    python thermal_soak.py --model yolo26n.pt --imgsz 320 --minutes 15
    python thermal_soak.py --model yolo11n.pt --imgsz 320 --minutes 1   # smoke
"""

import argparse
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from bench_common import (
    append_csv,
    decode_throttled,
    device_info,
    latency_stats,
    load_frames,
    read_cpu_freq_mhz,
    read_cpu_temp_c,
    read_throttled,
    round_floats,
    set_threads,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAMES_DIR = REPO_ROOT / "data/benchmark/frames"
DEFAULT_OUT_DIR = REPO_ROOT / "benchmarks/phase1"

CSV_FIELDS = [
    "timestamp", "device_label", "model", "imgsz", "format", "device", "threads",
    "elapsed_s", "window_idx", "frames", "fps_window",
    "lat_p50_ms", "lat_p95_ms", "temp_c", "freq_mhz", "throttled_flags", "rss_mb",
]


def resolve_weights(name: str) -> str:
    local = REPO_ROOT / name
    return str(local) if local.is_file() else name


def rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / 1024**2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Weights to soak, e.g. yolo11n.pt")
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--minutes", type=float, default=10.0,
                        help="Soak duration (default: 10)")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--window", type=int, default=30,
                        help="Frames per logged window (default: 30)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Discarded warm-up inferences (default: 10)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--device-label", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--out", type=Path, default=None,
                        help="CSV path (default: benchmarks/phase1/soak_<model>_<imgsz>.csv)")
    args = parser.parse_args()

    info = device_info()
    if args.device_label is None:
        args.device_label = info["board"]
    if args.out is None:
        args.out = DEFAULT_OUT_DIR / f"soak_{Path(args.model).stem}_{args.imgsz}.csv"

    from ultralytics import YOLO

    set_threads(args.threads)
    frames = load_frames(args.frames_dir)
    model = YOLO(resolve_weights(args.model))
    predict_kwargs = dict(
        imgsz=args.imgsz, device=args.device,
        conf=args.conf, iou=args.iou, verbose=False,
    )

    print(f"Soak: {args.model} imgsz={args.imgsz} for {args.minutes:.1f} min "
          f"on {args.device_label}")
    print(f"Frames: {len(frames)} | window: {args.window} frames | out: {args.out}")
    for i in range(args.warmup):
        model.predict(frames[i % len(frames)], **predict_kwargs)

    duration_s = args.minutes * 60
    start = time.monotonic()
    frame_idx = 0
    window_idx = 0
    window_latencies: list[float] = []
    rows: list[dict] = []
    max_temp = read_cpu_temp_c()
    all_flags: set[str] = set()

    while time.monotonic() - start < duration_s:
        frame = frames[frame_idx % len(frames)]
        frame_idx += 1
        t0 = time.perf_counter()
        model.predict(frame, **predict_kwargs)
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
            "device_label": args.device_label,
            "model": args.model,
            "imgsz": args.imgsz,
            "format": "pytorch",
            "device": args.device,
            "threads": args.threads,
            "elapsed_s": elapsed,
            "window_idx": window_idx,
            "frames": len(window_latencies),
            # Window FPS from the summed latency of this window only, so a
            # slowdown shows up immediately instead of being averaged away.
            "fps_window": 1000.0 * len(window_latencies) / sum(window_latencies),
            "lat_p50_ms": stats["lat_p50_ms"],
            "lat_p95_ms": stats["lat_p95_ms"],
            "temp_c": temp,
            "freq_mhz": read_cpu_freq_mhz(),
            "throttled_flags": ";".join(flags),
            "rss_mb": rss_mb(),
        }
        rows.append(row)
        append_csv(args.out, round_floats({f: row.get(f, "") for f in CSV_FIELDS}))
        print(f"  t={elapsed:6.0f}s  {row['fps_window']:6.2f} FPS  "
              f"p95 {row['lat_p95_ms']:7.1f} ms  "
              f"temp {'n/a' if temp is None else f'{temp:.1f}C'}"
              + (f"  [{';'.join(flags)}]" if flags else ""))
        window_latencies = []

    if not rows:
        raise SystemExit("Soak ended before a single window completed; "
                         "lower --window or raise --minutes")

    # Compare the first and last 60 s of the run -- that is the whole point.
    first = [r["fps_window"] for r in rows if r["elapsed_s"] <= 60]
    last = [r["fps_window"] for r in rows if r["elapsed_s"] >= rows[-1]["elapsed_s"] - 60]
    fps_first = statistics.fmean(first or [rows[0]["fps_window"]])
    fps_last = statistics.fmean(last or [rows[-1]["fps_window"]])
    delta_pct = (fps_last - fps_first) / fps_first * 100 if fps_first else 0.0

    print(f"\n--- {args.model} imgsz={args.imgsz}, {args.minutes:.1f} min, "
          f"{frame_idx} frames ---")
    print(f"FPS first 60s : {fps_first:.2f}")
    print(f"FPS last 60s  : {fps_last:.2f}  ({delta_pct:+.1f}%)")
    print(f"Peak temp     : {'n/a' if max_temp is None else f'{max_temp:.1f} C'}")
    print(f"Throttle bits : {';'.join(sorted(all_flags)) or 'none'}")
    if delta_pct < -10:
        print("VERDICT: FPS sagged >10% under sustained load -- fix cooling "
              "before trusting any Phase 1 number.")
    print(f"CSV: {args.out}")


if __name__ == "__main__":
    main()
