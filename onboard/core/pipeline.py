"""DetectionPipeline -- the sync thread that joins frames, telemetry and the
clock offset, runs inference and writes results.
"""

import json
import shutil
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

from . import geotag
from .types import Frame, Telemetry


def nearest_telemetry(buffer: deque, lock: threading.Lock, frame_ts: float,
                      offset_s: Optional[float], max_dt_s: float) -> Optional[Telemetry]:
    """Telemetry sample closest in time to the frame, or None.

    The clock offset is applied here, at match time, rather than when samples
    are stored: every buffered sample then benefits from the latest TIMESYNC
    correction with no rewrite pass.
    """
    if offset_s is None:
        return None  # clock not yet synced -- a guessed position is worse than none
    with lock:
        # deque.append is atomic but ITERATION is not: iterating while the
        # reader thread appends raises "deque mutated during iteration".
        samples = tuple(buffer)

    best, best_dt = None, max_dt_s
    for sample in samples:
        dt = abs((sample.t_boot_s + offset_s) - frame_ts)
        if dt <= best_dt:
            best, best_dt = sample, dt
    return best
    # ponytail: O(n) scan over 50 items per frame, ~2 us. bisect on a sorted
    # list only if the buffer ever grows past a few hundred samples.


class DetectionPipeline:
    def __init__(self, frame_queue: Queue, telemetry_buffer: deque,
                 telemetry_lock: threading.Lock, clock_sync, detector, config,
                 logger, stop_event: threading.Event):
        self.frame_queue = frame_queue
        self.telemetry_buffer = telemetry_buffer
        self.telemetry_lock = telemetry_lock
        self.clock_sync = clock_sync
        self.detector = detector
        self.config = config
        self.storage = config.storage
        self.logger = logger
        self.stop_event = stop_event

        self.out_dir = Path(self.storage.out_dir)
        # Created up front, not lazily on first write: the disk-space check
        # statvfs's this path before anything has been written to it.
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.out_dir / "detections.jsonl"
        self._records = None
        self._last_save = 0.0
        self._saves = 0
        self.frames_processed = 0
        self.frames_saved = 0
        self.detections_total = 0
        self._disk_full_logged = False

    # --- storage ----------------------------------------------------------

    def _has_space(self) -> bool:
        """Checked every 100 saves -- a statvfs per frame is wasted syscalls."""
        if self._saves % 100 != 0:
            return not self._disk_full_logged
        free_mb = shutil.disk_usage(self.out_dir).free / 1e6
        if free_mb < self.storage.min_free_mb:
            if not self._disk_full_logged:
                self.logger.log_error(
                    "storage: %.0f MB free < %d MB floor, no longer saving images"
                    % (free_mb, self.storage.min_free_mb))
                self._disk_full_logged = True
            return False
        self._disk_full_logged = False
        return True

    def _should_save(self, detections: list, now: float) -> bool:
        if self.storage.save_mode == "detections" and not detections:
            return False
        if now - self._last_save < self.storage.save_min_interval_s:
            return False
        return self._has_space()

    # --- one frame --------------------------------------------------------

    def process(self, frame: Frame) -> dict:
        detections = self.detector.infer(frame.image)
        frame.telemetry = nearest_telemetry(
            self.telemetry_buffer, self.telemetry_lock, frame.timestamp,
            self.clock_sync.get_offset() if self.clock_sync else None,
            self.config.max_telemetry_dt_s)

        # Two different clocks: the TIMESYNC offset above converts autopilot
        # boot-time to our clock for matching, and cannot date a photo.
        # to_utc() uses the GPS UTC from SYSTEM_TIME for that.
        utc = self.clock_sync.to_utc(frame.timestamp) if self.clock_sync else frame.timestamp
        height, width = frame.image.shape[:2]

        image_name = None
        if self._should_save(detections, frame.timestamp):
            image_name = "%06d.jpg" % frame.frame_id
            # BGR -> RGB without importing cv2 just for a channel swap.
            geotag.save_jpeg(self.out_dir / image_name, frame.image[:, :, ::-1],
                             frame.telemetry, utc, self.storage.jpeg_quality)
            self._last_save = frame.timestamp
            self._saves += 1
            self.frames_saved += 1

        record = {
            "frame_id": frame.frame_id,
            "timestamp": datetime.fromtimestamp(utc, tz=timezone.utc)
                                 .isoformat(timespec="milliseconds")
                                 .replace("+00:00", "Z"),
            "image_size": [width, height],
            "detections": [d.as_dict() for d in detections],
            "image": image_name,
            "telemetry": frame.telemetry.as_dict() if frame.telemetry else None,
        }
        self.frames_processed += 1
        self.detections_total += len(detections)
        return record

    def _write(self, record: dict) -> None:
        if self._records is None:
            self._records = self.records_path.open("a")
        self._records.write(json.dumps(record) + "\n")
        # Flush per record: a power cut mid-flight still leaves every result
        # written so far readable (same rationale as bench_common.append_csv).
        self._records.flush()

    # --- thread -----------------------------------------------------------

    def run(self) -> None:
        last_report = time.monotonic()
        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.5)
                except Empty:
                    continue  # timeout, not a tight loop -- keeps SIGTERM responsive
                try:
                    self._write(self.process(frame))
                except Exception as exc:  # one bad frame must not end the flight
                    self.logger.log_error(f"pipeline: frame {frame.frame_id} failed: {exc}")

                now = time.monotonic()
                if now - last_report >= 30.0:
                    self.logger.log_info(
                        "pipeline: %d frames, %d saved, %d detections"
                        % (self.frames_processed, self.frames_saved, self.detections_total))
                    last_report = now
        finally:
            if self._records is not None:
                self._records.close()
                self._records = None
            self.logger.log_info("pipeline: stopped (%d frames, %d saved, %d detections)"
                                 % (self.frames_processed, self.frames_saved,
                                    self.detections_total))
