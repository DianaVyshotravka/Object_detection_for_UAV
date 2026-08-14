"""Turn the Phase 1 CSVs into the markdown artifact
(model_selection.md Phase 1 output, project_plan.md §3 "results table").

Tables emitted:
  1. model x imgsz -> FPS pivot (the elimination view)
  2. full metric table, column order matching model_selection.md's
     decision table
  3. latency breakdown (preprocess / inference / postprocess)
  4. soak summary, if any soak CSVs are given

`--target-fps` is optional. Without it no pass/fail column is emitted --
the target gets chosen after seeing the numbers, not before.

Usage:
    python summarize_results.py
    python summarize_results.py --target-fps 10 --plot
    python summarize_results.py --speed-csv /tmp/speed_smoke.csv --out /tmp/out.md
"""

import argparse
import glob
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEED_CSV = REPO_ROOT / "benchmarks/phase1/speed.csv"
DEFAULT_SOAK_GLOB = str(REPO_ROOT / "benchmarks/phase1/soak_*.csv")
DEFAULT_OUT = REPO_ROOT / "benchmarks/phase1/phase1_results.md"

PREAMBLE = """\
# Phase 1 results — speed screening

Format is **plain PyTorch `.pt` on CPU** (no NCNN, no quantization), so these
are *not* deployable FPS numbers — NCNN will lift all of them. What this
table is good for is the **relative ranking** of the candidate
architectures, which is what Phase 1 needs.

Latency is end-to-end `predict()` on a single frame (letterbox + forward +
NMS), batch=1. Frame decode is excluded. FPS = 1000 / mean latency,
single-stream.
"""


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    """Markdown table without depending on `tabulate`."""
    shown = df.copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(
                lambda v: "" if pd.isna(v) else floatfmt.format(v)
            )
    shown = shown.fillna("").astype(str)
    header = "| " + " | ".join(shown.columns) + " |"
    sep = "|" + "|".join("---" for _ in shown.columns) + "|"
    rows = ["| " + " | ".join(r) + " |" for r in shown.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def load_speed(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Speed CSV not found: {path}\n"
                         f"Run scripts/benchmark/bench_speed.py first.")
    df = pd.read_csv(path)
    if "status" in df.columns:
        failed = df[df["status"] == "error"]
        if not failed.empty:
            print(f"note: {len(failed)} failed config(s) excluded from the tables")
            for _, r in failed.iterrows():
                print(f"  {r['model']} imgsz={r['imgsz']}: {r['error']}")
        df = df[df["status"] != "error"].copy()
    if df.empty:
        raise SystemExit("No successful configs in the speed CSV")
    # A config can be measured more than once (consistency re-runs); keep
    # the most recent row per device/model/imgsz.
    keys = ["device_label", "model", "imgsz"]
    df = df.sort_values("timestamp").drop_duplicates(keys, keep="last")
    return df.sort_values(["device_label", "model", "imgsz"])


def fps_pivot(df: pd.DataFrame) -> str:
    pivot = df.pivot_table(index="model", columns="imgsz", values="fps_mean")
    pivot.columns = [f"{c} px" for c in pivot.columns]
    return md_table(pivot.reset_index())


def metric_table(df: pd.DataFrame, target_fps: float | None) -> str:
    cols = {
        "model": "Model", "imgsz": "imgsz", "fps_mean": "FPS",
        "lat_p50_ms": "p50, ms", "lat_p95_ms": "p95, ms",
        "rss_peak_mb": "RAM, MB", "weights_mb": "Weights, MB",
        "temp_end_c": "temp end, C", "throttled_flags": "throttled",
    }
    out = df[[c for c in cols if c in df.columns]].rename(columns=cols)
    if target_fps is not None:
        out[f"FPS >= {target_fps:g}?"] = [
            "yes" if v >= target_fps else "no" for v in df["fps_mean"]
        ]
    return md_table(out)


def breakdown_table(df: pd.DataFrame) -> str:
    cols = ["model", "imgsz", "pre_ms", "inf_ms", "post_ms", "lat_mean_ms"]
    if not all(c in df.columns for c in cols):
        return ""
    out = df[cols].rename(columns={
        "model": "Model", "imgsz": "imgsz", "pre_ms": "pre, ms",
        "inf_ms": "inference, ms", "post_ms": "post, ms",
        "lat_mean_ms": "end-to-end, ms",
    })
    return md_table(out)


def soak_section(pattern: str) -> str:
    paths = sorted(glob.glob(pattern))
    if not paths:
        return ""
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        last_t = df["elapsed_s"].max()
        first = df[df["elapsed_s"] <= 60]["fps_window"]
        last = df[df["elapsed_s"] >= last_t - 60]["fps_window"]
        fps_first = first.mean() if not first.empty else df["fps_window"].iloc[0]
        fps_last = last.mean() if not last.empty else df["fps_window"].iloc[-1]
        flags = sorted({
            f for cell in df["throttled_flags"].dropna().astype(str)
            for f in cell.split(";") if f
        })
        rows.append({
            "Model": df["model"].iloc[0],
            "imgsz": df["imgsz"].iloc[0],
            "minutes": last_t / 60,
            "FPS first 60s": fps_first,
            "FPS last 60s": fps_last,
            "delta %": (fps_last - fps_first) / fps_first * 100 if fps_first else 0.0,
            "peak temp, C": df["temp_c"].max(),
            "throttled": ";".join(flags) or "none",
        })
    if not rows:
        return ""
    return md_table(pd.DataFrame(rows))


def make_plot(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_fps, ax_lat) = plt.subplots(1, 2, figsize=(11, 4.2))
    for model, group in df.groupby("model"):
        group = group.sort_values("imgsz")
        ax_fps.plot(group["imgsz"], group["fps_mean"], marker="o", label=model)
        ax_lat.plot(group["imgsz"], group["lat_p95_ms"], marker="o", label=model)
    ax_fps.set(xlabel="imgsz (px)", ylabel="FPS (single-stream)", title="FPS vs input size")
    ax_lat.set(xlabel="imgsz (px)", ylabel="p95 latency (ms)", title="p95 latency vs input size")
    for ax in (ax_fps, ax_lat):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--speed-csv", type=Path, default=DEFAULT_SPEED_CSV)
    parser.add_argument("--soak-glob", default=DEFAULT_SOAK_GLOB,
                        help="Glob for soak CSVs (default: benchmarks/phase1/soak_*.csv)")
    parser.add_argument("--target-fps", type=float, default=None,
                        help="Adds a pass/fail column; omit to record only")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--plot", action="store_true",
                        help="Also write fps_vs_imgsz.png next to --out")
    args = parser.parse_args()

    df = load_speed(args.speed_csv)
    devices = sorted(df["device_label"].astype(str).unique())

    parts = [PREAMBLE]
    parts.append(f"Device(s): {', '.join(devices)}  \n"
                 f"Frames per config: {int(df['n_frames'].max())}  \n"
                 f"Threads: {', '.join(str(t) for t in sorted(df['threads'].unique()))}  \n"
                 f"Stack: torch {df['torch_version'].iloc[0]}, "
                 f"ultralytics {df['ultralytics_version'].iloc[0]}\n")

    for device in devices:
        sub = df[df["device_label"].astype(str) == device]
        if len(devices) > 1:
            parts.append(f"## {device}\n")
        parts.append("### FPS by model and input size\n")
        parts.append(fps_pivot(sub) + "\n")
        parts.append("### Full metrics\n")
        parts.append(metric_table(sub, args.target_fps) + "\n")
        table = breakdown_table(sub)
        if table:
            parts.append("### Latency breakdown\n")
            parts.append(table + "\n")

    soak = soak_section(args.soak_glob)
    if soak:
        parts.append("### Sustained load (throttling test)\n")
        parts.append(soak + "\n")
    else:
        parts.append("_No soak CSVs found — run thermal_soak.py to complete Phase 1._\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts))
    print(f"Wrote {args.out}")

    if args.plot:
        make_plot(df, args.out.with_name("fps_vs_imgsz.png"))


if __name__ == "__main__":
    main()
