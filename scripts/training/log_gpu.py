"""Poll GPU stats during a training run and append them to a CSV
(finetuning_high_level.md step 2).

TensorBoard already covers loss / mAP / P / R / LR. This covers the GPU
side the user asked for: utilization, memory, temperature, power. Every
`--interval` seconds it runs

    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,\
temperature.gpu,power.draw --format=csv,noheader,nounits

and appends one timestamped row to `<run_dir>/gpu_log.csv`.

Two ways to use it:

CLI (started/stopped by the bake-off wrapper as a subprocess) --
    python log_gpu.py --run-dir runs/bakeoff_yolo11n --interval 5
The wrapper terminates it (SIGTERM/SIGINT) when the run ends; the final
partial interval is not lost because each row is flushed as written.

Import (context manager) --
    from log_gpu import GpuLogger
    with GpuLogger(run_dir):
        model.train(...)
"""

import argparse
import csv
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

QUERY_FIELDS = [
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
]
CSV_HEADER = ["timestamp", "gpu_index", *QUERY_FIELDS]


def query_gpus() -> list[list[str]]:
    """Return one list of field values per visible GPU (empty on failure)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu=index,{','.join(QUERY_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"log_gpu: nvidia-smi failed ({exc}); stopping poll")
        return []
    return [
        [c.strip() for c in line.split(",")]
        for line in out.splitlines()
        if line.strip()
    ]


class GpuLogger:
    """Background thread that appends GPU stats to `<run_dir>/gpu_log.csv`.

    Usable as a context manager or via start()/stop(). Safe if the run
    dir does not exist yet -- it is created on start.
    """

    def __init__(self, run_dir: Path | str, interval: float = 5.0):
        self.run_dir = Path(run_dir)
        self.interval = interval
        self.csv_path = self.run_dir / "gpu_log.csv"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll_loop(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(CSV_HEADER)
                f.flush()
            while not self._stop.is_set():
                rows = query_gpus()
                if not rows:
                    break
                ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
                for row in rows:
                    writer.writerow([ts, *row])
                f.flush()
                self._stop.wait(self.interval)

    def start(self) -> "GpuLogger":
        if self._thread is not None:
            raise RuntimeError("GpuLogger already started")
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def __enter__(self) -> "GpuLogger":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run folder; log written to <run-dir>/gpu_log.csv",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds between polls (default: 5)",
    )
    args = parser.parse_args()

    logger = GpuLogger(args.run_dir, args.interval)

    def handle_signal(signum, frame):
        logger.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"log_gpu: polling every {args.interval}s -> {logger.csv_path}")
    logger.start()
    # Block the main thread until a signal stops the daemon poll thread.
    while logger._thread is not None and logger._thread.is_alive():
        logger._thread.join(timeout=1.0)
    print(f"log_gpu: stopped, wrote {logger.csv_path}")


if __name__ == "__main__":
    main()
