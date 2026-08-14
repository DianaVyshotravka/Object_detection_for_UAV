"""Phase 1 speed screening: time every model x imgsz config on the target
device and write one CSV row per config (model_selection.md Phase 1).

Deviation from the doc, on purpose: this measures **plain PyTorch .pt on
CPU**, not NCNN. No export, no quantization -- those come later. So the
absolute FPS here is NOT deployable FPS (NCNN lifts all of them); what
this produces is the relative ranking of the architectures, which is what
Phase 1 needs in order to eliminate candidates before spending GPU hours.

What is measured: end-to-end `model.predict()` on a single frame,
batch=1, i.e. letterbox + forward + NMS, the shape the drone actually
runs. project_plan.md §3 insists pre/post is included -- on ARM CPU it is
a real share of the cost. Frame decode is excluded (it is disk cost) by
pre-loading the frames into RAM.

Fair-comparison guards: fixed frame set in a fixed order, locked thread
count, warm-up discarded, >=200 timed frames, and a cool-down to a stable
thermal state between configs.

Usage:
    python bench_speed.py --dry-run
    python bench_speed.py                          # full 4x4 grid
    python bench_speed.py --models yolo11n.pt --imgsz 320 640 --n-frames 20
"""

import argparse
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from bench_common import (
    PeakRss,
    append_csv,
    cooldown,
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
DEFAULT_MODELS = ["yolov8n.pt", "yolo11n.pt", "yolo26n.pt", "yolo11s.pt"]
DEFAULT_IMGSZ = [320, 416, 512, 640]
DEFAULT_FRAMES_DIR = REPO_ROOT / "data/benchmark/frames"
DEFAULT_OUT = REPO_ROOT / "benchmarks/phase1/speed.csv"

# Fixed column order. Error rows carry fewer keys than ok rows, so every
# row is padded to this list before it is appended -- otherwise a later
# append would not line up with the header written by the first one.
CSV_FIELDS = [
    "timestamp", "device_label", "board", "machine",
    "model", "weights_mb", "imgsz", "format", "device", "threads",
    "n_frames", "status", "error",
    "fps_mean", "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms",
    "lat_min_ms", "lat_max_ms",
    "pre_ms", "inf_ms", "post_ms", "rss_peak_mb",
    "temp_start_c", "temp_end_c", "freq_start_mhz", "freq_end_mhz",
    "throttled_flags", "torch_version", "ultralytics_version",
]


def to_csv_row(row: dict) -> dict:
    """Pad/order a results row to CSV_FIELDS so appends stay aligned."""
    unknown = set(row) - set(CSV_FIELDS)
    if unknown:
        raise KeyError(f"Row has fields missing from CSV_FIELDS: {sorted(unknown)}")
    return {field: row.get(field, "") for field in CSV_FIELDS}


def resolve_weights(name: str) -> str:
    """Prefer a checkpoint sitting at the repo root over an auto-download.

    yolo11n.pt / yolo26n.pt are already there; yolov8n.pt / yolo11s.pt are
    fetched by Ultralytics on first use (needs network on the Pi once).
    """
    local = REPO_ROOT / name
    return str(local) if local.is_file() else name


def weights_mb(name: str) -> float | None:
    path = Path(resolve_weights(name))
    return path.stat().st_size / 1024**2 if path.is_file() else None


def bench_one(model_name: str, imgsz: int, frames: list, args: argparse.Namespace,
              info: dict) -> dict:
    """Time one config and return its results row."""
    from ultralytics import YOLO

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device_label": args.device_label,
        "board": info["board"],
        "machine": info["machine"],
        "model": model_name,
        "weights_mb": weights_mb(model_name),
        "imgsz": imgsz,
        "format": "pytorch",
        "device": args.device,
        "threads": args.threads,
        "n_frames": 0,
        "status": "ok",
        "error": "",
    }

    model = YOLO(resolve_weights(model_name))
    predict_kwargs = dict(
        imgsz=imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )

    # Warm-up: first inferences pay lazy kernel/allocator init and would
    # otherwise land in the p95.
    for i in range(args.warmup):
        model.predict(frames[i % len(frames)], **predict_kwargs)

    row["temp_start_c"] = read_cpu_temp_c()
    row["freq_start_mhz"] = read_cpu_freq_mhz()

    latencies: list[float] = []
    speed_totals = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    with PeakRss() as rss:
        for i in range(args.n_frames):
            frame = frames[i % len(frames)]
            start = time.perf_counter()
            results = model.predict(frame, **predict_kwargs)
            latencies.append((time.perf_counter() - start) * 1000.0)
            # Ultralytics' own breakdown: shows whether YOLO26n's NMS-free
            # head actually cuts postprocess on ARM CPU.
            for key in speed_totals:
                speed_totals[key] += results[0].speed.get(key, 0.0) or 0.0

    row["temp_end_c"] = read_cpu_temp_c()
    row["freq_end_mhz"] = read_cpu_freq_mhz()
    row["n_frames"] = len(latencies)
    row.update(latency_stats(latencies))
    row["pre_ms"] = speed_totals["preprocess"] / len(latencies)
    row["inf_ms"] = speed_totals["inference"] / len(latencies)
    row["post_ms"] = speed_totals["postprocess"] / len(latencies)
    row["rss_peak_mb"] = rss.peak_mb
    row["throttled_flags"] = ";".join(decode_throttled(read_throttled()))
    row["torch_version"] = info["torch_version"]
    row["ultralytics_version"] = info["ultralytics_version"]
    return row


def error_row(model_name: str, imgsz: int, args: argparse.Namespace, info: dict,
              exc: Exception) -> dict:
    """Placeholder row so one broken config does not hide the rest of the grid."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device_label": args.device_label,
        "board": info["board"],
        "machine": info["machine"],
        "model": model_name,
        "weights_mb": weights_mb(model_name),
        "imgsz": imgsz,
        "format": "pytorch",
        "device": args.device,
        "threads": args.threads,
        "n_frames": 0,
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}"[:300],
        "torch_version": info["torch_version"],
        "ultralytics_version": info["ultralytics_version"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Candidates to screen (default: {DEFAULT_MODELS})")
    parser.add_argument("--imgsz", nargs="+", type=int, default=DEFAULT_IMGSZ,
                        help=f"Input sizes to sweep (default: {DEFAULT_IMGSZ})")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR,
                        help="Frame set from make_bench_frames.py")
    parser.add_argument("--n-frames", type=int, default=250,
                        help="Timed frames per config, >=200 (default: 250)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Discarded warm-up inferences per config (default: 20)")
    parser.add_argument("--device", default="cpu",
                        help="Torch device; keep 'cpu' -- the Pi has no usable NN GPU")
    parser.add_argument("--threads", type=int, default=os.cpu_count(),
                        help="Torch/OpenCV threads, locked for comparability")
    parser.add_argument("--cooldown-c", type=float, default=60.0,
                        help="Wait until CPU is below this before each config (default: 60)")
    parser.add_argument("--cooldown-timeout", type=float, default=300.0,
                        help="Give up waiting after this many seconds (default: 300)")
    parser.add_argument("--device-label", default=None,
                        help="Name for this device in the results table (default: board model)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold; affects NMS/postprocess cost")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Results CSV, appended to (default: benchmarks/phase1/speed.csv)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the grid and check weights resolve, time nothing")
    args = parser.parse_args()

    if args.n_frames < 200 and not args.dry_run:
        print(f"warning: --n-frames {args.n_frames} is below the >=200 "
              f"model_selection.md asks for; smoke test only")

    info = device_info()
    if args.device_label is None:
        args.device_label = info["board"]

    configs = [(m, sz) for m in args.models for sz in args.imgsz]
    print(f"Device : {args.device_label} ({info['machine']}, {info['cpu_count']} cores)")
    print(f"Stack  : torch {info['torch_version']}, "
          f"ultralytics {info['ultralytics_version']}")
    print(f"Grid   : {len(configs)} configs = {len(args.models)} models "
          f"x {len(args.imgsz)} sizes, format=pytorch")
    print(f"Out    : {args.out}")

    if args.dry_run:
        for model_name, imgsz in configs:
            resolved = resolve_weights(model_name)
            where = "repo root" if Path(resolved).is_file() else "auto-download"
            print(f"  {model_name:14s} imgsz={imgsz:<4d} weights: {where}")
        frames = sorted(p.name for p in args.frames_dir.glob("*")
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        print(f"Frames : {len(frames)} in {args.frames_dir}")
        return

    set_threads(args.threads)
    frames = load_frames(args.frames_dir)
    print(f"Frames : {len(frames)} loaded from {args.frames_dir}")

    for index, (model_name, imgsz) in enumerate(configs, start=1):
        print(f"\n=== [{index}/{len(configs)}] {model_name} imgsz={imgsz} ===")
        cooldown(args.cooldown_c, args.cooldown_timeout)
        try:
            row = bench_one(model_name, imgsz, frames, args, info)
            rss = row["rss_peak_mb"]
            print(f"  {row['fps_mean']:.2f} FPS  |  p50 {row['lat_p50_ms']:.1f} ms  "
                  f"|  p95 {row['lat_p95_ms']:.1f} ms  |  "
                  f"RSS {f'{rss:.0f} MB' if rss else 'n/a'}")
            if row["throttled_flags"]:
                print(f"  THROTTLED: {row['throttled_flags']} -- numbers are suspect")
        except Exception as exc:  # noqa: BLE001 - one bad arch must not kill the grid
            traceback.print_exc()
            print(f"  FAILED: {exc}")
            row = error_row(model_name, imgsz, args, info, exc)
        append_csv(args.out, to_csv_row(round_floats(row)))

    print(f"\nDone. {args.out}\n"
          f"Summarize: python scripts/benchmark/summarize_results.py "
          f"--speed-csv {args.out}")


if __name__ == "__main__":
    main()
