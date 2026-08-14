"""Shared helpers for the Phase 1 speed screening on the Raspberry Pi
(model_selection.md Phase 1).

Device sensors, thread pinning, frame loading, latency stats and CSV
append -- everything both `bench_speed.py` and `thermal_soak.py` need.

Every sensor reader returns None when the underlying file / tool is
absent, so the same scripts run unchanged on the x86 dev machine (for a
smoke test) and on the Pi (for the real numbers).
"""

import csv
import os
import platform
import socket
import statistics
import subprocess
import threading
import time
from pathlib import Path

# Bits of the `vcgencmd get_throttled` word. Low nibble = happening now,
# bits 16-19 = has happened at some point since boot.
THROTTLED_BITS = {
    0: "under-voltage",
    1: "arm-freq-capped",
    2: "throttled",
    3: "soft-temp-limit",
    16: "under-voltage-occurred",
    17: "arm-freq-capped-occurred",
    18: "throttled-occurred",
    19: "soft-temp-limit-occurred",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


# --------------------------------------------------------------------------
# device sensors
# --------------------------------------------------------------------------

def read_cpu_temp_c() -> float | None:
    """CPU temperature in degrees C, or None if the sensor is unavailable."""
    path = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return int(path.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def read_cpu_freq_mhz() -> float | None:
    """Current cpu0 frequency in MHz, or None if cpufreq is unavailable."""
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    try:
        return int(path.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def read_throttled() -> str | None:
    """Raw `vcgencmd get_throttled` word (e.g. '0x0'), or None off-Pi."""
    try:
        out = subprocess.run(
            ["vcgencmd", "get_throttled"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    # Output looks like: throttled=0x0
    _, _, word = out.partition("=")
    return word.strip() or None


def decode_throttled(word: str | None) -> list[str]:
    """Names of the throttle bits set in a `get_throttled` word."""
    if not word:
        return []
    try:
        value = int(word, 16)
    except ValueError:
        return []
    return [name for bit, name in THROTTLED_BITS.items() if value & (1 << bit)]


def board_model() -> str:
    """Board string from the device tree ('Raspberry Pi 5 Model B ...')."""
    path = Path("/proc/device-tree/model")
    try:
        # Device-tree strings are NUL-terminated.
        return path.read_bytes().decode("utf-8", "replace").strip("\x00").strip()
    except OSError:
        return platform.platform()


def device_info() -> dict:
    """Everything about the machine that belongs in a results row."""
    import torch
    import ultralytics

    return {
        "hostname": socket.gethostname(),
        "board": board_model(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
    }


def set_threads(n: int) -> None:
    """Pin torch + OpenCV to `n` threads so runs are comparable.

    Left implicit, torch and OpenCV each pick their own default and a
    rerun on a differently loaded machine is not the same measurement.
    """
    import cv2
    import torch

    torch.set_num_threads(n)
    cv2.setNumThreads(n)
    print(f"threads: torch={torch.get_num_threads()} cv2={n}")


def cooldown(target_c: float, timeout_s: float, poll_s: float = 5.0) -> float | None:
    """Block until CPU temp drops below `target_c`. No-op without a sensor.

    Returns the temperature it settled at (None off-Pi). A run started hot
    is throttled from frame 1 and is not comparable to one started cold --
    model_selection.md's "stable thermal state" rule.
    """
    temp = read_cpu_temp_c()
    if temp is None:
        return None
    deadline = time.monotonic() + timeout_s
    while temp is not None and temp > target_c:
        if time.monotonic() > deadline:
            print(f"cooldown: still {temp:.1f}C after {timeout_s:.0f}s, continuing")
            break
        print(f"cooldown: {temp:.1f}C > {target_c:.1f}C, waiting {poll_s:.0f}s")
        time.sleep(poll_s)
        temp = read_cpu_temp_c()
    return temp


# --------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------

class PeakRss:
    """Context manager sampling this process' peak RSS in the background.

    psutil gives a live sample every `interval`; without it we fall back
    to getrusage's high-water mark, which is coarser (it never goes down)
    but still answers "did this config blow up RAM".
    """

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = None
        try:
            import psutil

            self._proc = psutil.Process()
        except ImportError:
            print("PeakRss: psutil not installed, falling back to getrusage")

    def _sample(self) -> None:
        while not self._stop.is_set():
            rss_mb = self._proc.memory_info().rss / 1024**2
            if self.peak_mb is None or rss_mb > self.peak_mb:
                self.peak_mb = rss_mb
            self._stop.wait(self.interval)

    def __enter__(self) -> "PeakRss":
        if self._proc is not None:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
            return
        import resource

        # ru_maxrss is KiB on Linux.
        self.peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------

def list_frames(frames_dir: Path) -> list[Path]:
    """Image paths in `frames_dir`, sorted so every model sees the same order."""
    return sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    )


def load_frames(frames_dir: Path, limit: int | None = None) -> list:
    """Decode frames into RAM as BGR numpy arrays.

    Decode is deliberately outside the timed loop: it is disk/JPEG cost,
    not model-pipeline cost. Letterboxing and NMS stay inside the timing
    because project_plan.md §3 wants end-to-end pre/post included.
    """
    import cv2

    paths = list_frames(frames_dir)
    if not paths:
        raise SystemExit(f"No images found in {frames_dir}")
    if limit is not None:
        paths = paths[:limit]
    frames = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"load_frames: cannot decode {path.name}, skipping")
            continue
        frames.append(img)
    if not frames:
        raise SystemExit(f"No decodable images in {frames_dir}")
    return frames


# --------------------------------------------------------------------------
# stats + output
# --------------------------------------------------------------------------

def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100) of a non-empty list."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
    return ordered[idx]


def latency_stats(latencies_ms: list[float]) -> dict:
    """mean / p50 / p95 / p99 / min / max plus the derived single-stream FPS."""
    mean_ms = statistics.fmean(latencies_ms)
    return {
        "fps_mean": 1000.0 / mean_ms if mean_ms else 0.0,
        "lat_mean_ms": mean_ms,
        "lat_p50_ms": percentile(latencies_ms, 50),
        "lat_p95_ms": percentile(latencies_ms, 95),
        "lat_p99_ms": percentile(latencies_ms, 99),
        "lat_min_ms": min(latencies_ms),
        "lat_max_ms": max(latencies_ms),
    }


def round_floats(row: dict, ndigits: int = 3) -> dict:
    return {
        k: round(v, ndigits) if isinstance(v, float) else v
        for k, v in row.items()
    }


def append_csv(path: Path, row: dict) -> None:
    """Append one row, writing the header first if the file is new.

    Same pattern as scripts/training/log_gpu.py: flush per row so a run
    killed halfway still leaves usable results on disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
