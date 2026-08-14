#!/usr/bin/env python3
"""Create visual summaries for the Phase 1 Raspberry Pi speed screen."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "visualizations"
OUT.mkdir(exist_ok=True)


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(row, key):
    return float(row[key])


def model_label(name):
    return name.removesuffix(".pt")


plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 180,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

agg = [r for r in read_csv(ROOT / "speed_all.csv") if r["status"] == "ok"]
models = sorted({r["model"] for r in agg}, key=lambda x: ("s" in x, x))
sizes = sorted({int(r["imgsz"]) for r in agg})
colors = {m: colormaps["tab10"](i % 10) for i, m in enumerate(models)}
legend_cols = min(4, max(2, len(models)))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


# 1. Main speed/latency comparison.
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for m in models:
    rows = sorted([r for r in agg if r["model"] == m], key=lambda r: int(r["imgsz"]))
    x = [int(r["imgsz"]) for r in rows]
    axes[0].plot(x, [f(r, "fps_mean") for r in rows], marker="o", lw=2, label=model_label(m), color=colors[m])
    axes[1].plot(x, [f(r, "lat_p95_ms") for r in rows], marker="o", lw=2, label=model_label(m), color=colors[m])
axes[0].set(title="Speed by input size", xlabel="Input size (px)", ylabel="Mean FPS")
axes[1].set(title="Tail latency by input size", xlabel="Input size (px)", ylabel="P95 latency (ms)")
axes[0].set_xticks(sizes); axes[1].set_xticks(sizes)
axes[0].legend(frameon=False, ncol=legend_cols)
save(fig, "01_speed_and_latency.png")


# 2. Pareto-style resource view at each resolution.
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for m in models:
    rows = [r for r in agg if r["model"] == m]
    axes[0].scatter([f(r, "weights_mb") for r in rows], [f(r, "fps_mean") for r in rows],
                    s=55, color=colors[m], label=model_label(m), alpha=.9)
    axes[1].plot([int(r["imgsz"]) for r in sorted(rows, key=lambda r: int(r["imgsz"]))],
                 [f(r, "rss_peak_mb") for r in sorted(rows, key=lambda r: int(r["imgsz"]))],
                 marker="o", lw=2, color=colors[m], label=model_label(m))
axes[0].set(title="Model size vs speed", xlabel="Weights (MB)", ylabel="Mean FPS")
axes[1].set(title="Peak RSS by input size", xlabel="Input size (px)", ylabel="Peak RSS (MB)")
axes[1].set_xticks(sizes)
axes[0].legend(frameon=False, ncol=legend_cols)
save(fig, "02_size_memory_tradeoffs.png")


# 3. Latency breakdown, normalized to the measured end-to-end path.
ordered = sorted(agg, key=lambda r: (int(r["imgsz"]), models.index(r["model"])))
fig, ax = plt.subplots(figsize=(max(12, len(ordered) * 0.42), 5.5))
labels = [f"{model_label(r['model'])}\n{r['imgsz']}" for r in ordered]
bottom = [0.0] * len(ordered)
for key, label, color in [("pre_ms", "Pre", "#93c5fd"), ("inf_ms", "Inference", "#2563eb"), ("post_ms", "Post", "#1e3a8a")]:
    vals = [f(r, key) for r in ordered]
    ax.bar(labels, vals, bottom=bottom, label=label, color=color)
    bottom = [a + b for a, b in zip(bottom, vals)]
for idx, total in enumerate(bottom):
    ax.text(idx, total + max(bottom) * 0.012, f"{total:.1f}", ha="center", va="bottom", fontsize=8)
ax.set(title="Where the end-to-end latency goes", ylabel="Milliseconds")
ax.tick_params(axis="x", labelrotation=55, labelsize=8)
ax.legend(frameon=False, ncol=3)
save(fig, "03_latency_breakdown.png")


# 4. Temperature and throttling signal.
fig, ax = plt.subplots(figsize=(13, 4.8))
for m in models:
    rows = sorted([r for r in agg if r["model"] == m], key=lambda r: int(r["imgsz"]))
    ax.plot([int(r["imgsz"]) for r in rows], [f(r, "temp_end_c") for r in rows], marker="o", lw=2,
            color=colors[m], label=model_label(m))
ax.axhline(80, color="#b91c1c", ls="--", lw=1, label="80 °C reference")
ax.set(title="End temperature remained below the reference line", xlabel="Input size (px)", ylabel="End temperature (°C)", xticks=sizes)
ax.legend(frameon=False, ncol=legend_cols)
save(fig, "04_temperature.png")


# 5. Per-window 320 px stability traces.
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
trace_files = sorted(ROOT.glob("*_320.csv"))
for path in trace_files:
    rows = read_csv(path)
    m = rows[0]["model"]
    x = [int(r["window_idx"]) for r in rows]
    axes[0].plot(x, [f(r, "fps_window") for r in rows], lw=1.4, label=model_label(m), color=colors.get(m))
    axes[1].plot(x, [f(r, "temp_c") for r in rows], lw=1.4, label=model_label(m), color=colors.get(m))
axes[0].set(title="320 px window-by-window throughput", ylabel="FPS")
axes[1].set(title="320 px temperature trace", xlabel="30-frame window", ylabel="Temperature (°C)")
axes[0].legend(frameon=False, ncol=legend_cols)
axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
save(fig, "05_320px_stability.png")


# 6. One compact ranking table rendered as a figure for easy sharing.
rows320 = sorted([r for r in agg if int(r["imgsz"]) == 320], key=lambda r: f(r, "fps_mean"), reverse=True)
fig, ax = plt.subplots(figsize=(max(8, len(rows320) * 1.35), 2.7))
ax.axis("off")
columns = ["Model", "FPS", "P50 (ms)", "P95 (ms)", "Weights (MB)", "End temp (°C)"]
data = [[model_label(r["model"]), f'{f(r, "fps_mean"):.2f}', f'{f(r, "lat_p50_ms"):.1f}',
         f'{f(r, "lat_p95_ms"):.1f}', f'{f(r, "weights_mb"):.2f}', f'{f(r, "temp_end_c"):.1f}'] for r in rows320]
table = ax.table(cellText=data, colLabels=columns, cellLoc="center", loc="center")
table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.55)
for cell in table.get_celld().values():
    cell.set_edgecolor("#d1d5db")
for col in range(len(columns)):
    cell = table[(0, col)]
    cell.set_facecolor("#1f2937"); cell.set_text_props(color="white", weight="bold")
ax.set_title("320 px screening ranking", pad=12, weight="bold")
save(fig, "06_320px_ranking.png")

print(f"Wrote {len(list(OUT.glob('*.png')))} plots to {OUT}")
