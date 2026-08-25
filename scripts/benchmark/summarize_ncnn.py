"""Summarize the Raspberry Pi NCNN benchmark CSV."""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("benchmarks/phase3/ncnn_speed.csv"))
    parser.add_argument("--out", type=Path, default=Path("benchmarks/phase3/ncnn_results.md"))
    parser.add_argument("--target-fps", type=float, default=None)
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    df = df[df["status"].fillna("") == "ok"].copy()
    if df.empty:
        raise SystemExit("No successful NCNN rows found")
    df = df.sort_values("timestamp").drop_duplicates(["device_label", "model", "imgsz"], keep="last")
    columns = [
        "device_label", "model", "imgsz", "fps_mean", "lat_p50_ms",
        "lat_p95_ms", "rss_peak_mb", "temp_start_c", "temp_end_c",
        "throttled_flags",
    ]
    shown = df[[c for c in columns if c in df]].rename(columns={
        "device_label": "Device", "model": "Model", "imgsz": "imgsz",
        "fps_mean": "FPS", "lat_p50_ms": "p50 ms", "lat_p95_ms": "p95 ms",
        "rss_peak_mb": "RAM MB", "temp_start_c": "start C", "temp_end_c": "end C",
        "throttled_flags": "throttling",
    })
    if args.target_fps is not None:
        shown[f"FPS >= {args.target_fps:g}"] = df["fps_mean"].ge(args.target_fps).map({True: "yes", False: "no"})
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    shown = shown.fillna("").astype(str)
    lines = [
        "# Raspberry Pi NCNN benchmark",
        "",
        "Runtime measurements for the Phase 3 finalists. Join this table with `docs/phase3_imgsz_results.md`.",
        "",
        "| " + " | ".join(shown.columns) + " |",
        "|" + "|".join("---" for _ in shown.columns) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in shown.itertuples(index=False, name=None))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
