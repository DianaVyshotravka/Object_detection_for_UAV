"""Onboard detection service entry point.

Run:  python -m onboard.main --config onboard/config.yaml
Stop: SIGTERM (systemd) or Ctrl+C. Both exit cleanly within ~2 s.
"""

import argparse
import logging
import os
import queue
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

from onboard.config import load_config
from onboard.core.clock_sync import ClockSync
from onboard.core.logger import Logger
from onboard.core.mavlink_client import MavlinkClient
from onboard.core.pipeline import DetectionPipeline
from onboard.core.video_stream import VideoStream, open_capture

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def shutdown_exit(code: int) -> None:
    """Exit now, skipping interpreter teardown.

    The worker threads stop in ~0.3 s, but torch/ultralytics atexit teardown
    adds an unpredictable 1-2 s on top, which intermittently pushed a SIGTERM
    stop past the 2 s budget in section 3.

    note: os._exit skips atexit handlers and GC. Safe only because there is
    nothing left to flush -- the JSONL is flushed per record and closed in
    DetectionPipeline.run's finally, the camera is released and the MAVLink
    link closed, and logging is shut down here. Anything added later that
    buffers data must be flushed BEFORE this call, not in an atexit hook.
    """
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


# --- initialization ---------------------------

def init_camera(config, logger):
    return open_capture(config.camera, logger)


def init_mavlink(config, logger):
    if not config.mavlink.enabled:
        logger.log_info("mavlink: disabled by config, running without telemetry")
        return None
    client = MavlinkClient(config.mavlink.url, config.mavlink.baud,
                           config.mavlink.telemetry_hz, logger)
    client.connect(timeout=config.init_timeout_s)
    return client


def parallel_init(config, logger):
    """Open camera and autopilot concurrently, with a deadline on both.

    Plain daemon threads rather than ThreadPoolExecutor: the executor's workers
    are non-daemon and it registers an atexit hook that joins them, so a worker
    stuck in VideoCapture() on a dead device hangs interpreter shutdown -- the
    "init error -> End" branch would never actually reach exit.
    """
    results, errors = {}, {}

    def worker(name, fn):
        try:
            results[name] = fn(config, logger)
        except Exception as exc:
            errors[name] = exc

    threads = [threading.Thread(target=worker, args=(n, f), daemon=True)
               for n, f in (("camera", init_camera), ("mavlink", init_mavlink))]
    for t in threads:
        t.start()
    deadline = time.monotonic() + config.init_timeout_s
    for t in threads:  # the AND-join, with a deadline
        t.join(timeout=max(0.0, deadline - time.monotonic()))

    if "camera" in errors:
        raise RuntimeError(f"camera: {errors['camera']}")
    if "camera" not in results:
        raise RuntimeError(f"camera: open timed out after {config.init_timeout_s}s")
    if "mavlink" in errors or "mavlink" not in results:
        # Degraded mode on purpose: a dead telemetry link is no reason to
        # abandon a mission the camera can still fly. Frames lose their GPS.
        reason = errors.get("mavlink", "connect timed out")
        logger.log_error(f"mavlink: {reason} -- continuing without telemetry")
        results["mavlink"] = None
    return results["camera"], results["mavlink"]


# --- the running service --------------------

def run(config, camera, mavlink_client, logger, detector=None) -> int:
    """Wire the threads, block until stopped, shut down. Returns an exit code.

    ``detector`` is injectable so the integration test can exercise the whole
    thread graph without loading a model.
    """
    stop_event = threading.Event()

    def handle_signal(signum, _frame):
        logger.log_info(f"{signal.Signals(signum).name} received - stop signal")
        stop_event.set()

    # SIGINT as well as SIGTERM: without it Ctrl+C during a manual Pi run
    # raises out of the wait below and skips the whole shutdown block.
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handle_signal)

    frame_queue = queue.Queue(maxsize=config.frame_queue_size)
    telemetry_buffer = deque(maxlen=config.telemetry_buffer_size)
    telemetry_lock = threading.Lock()

    if detector is None:
        from onboard.core.detector_model import DetectorModel
        detector = DetectorModel(config.detector.model_path, config.detector.imgsz,
                                 config.detector.conf, config.detector.iou, logger)

    clock_sync = None
    workers = []
    if mavlink_client is not None:
        clock_sync = ClockSync(mavlink_client.send_timesync, stop_event, logger,
                               config.mavlink.timesync_interval_s, config.mavlink.max_rtt_s)
        mavlink_client.attach(telemetry_buffer, telemetry_lock, clock_sync, stop_event)
        workers += [("mavlink", mavlink_client), ("clock_sync", clock_sync)]

    video_stream = VideoStream(camera, frame_queue, stop_event, logger,
                               config.camera.latency_ms / 1000.0,
                               config.camera.max_read_failures)
    pipeline = DetectionPipeline(frame_queue, telemetry_buffer, telemetry_lock,
                                 clock_sync, detector, config, logger, stop_event)
    workers += [("video", video_stream), ("pipeline", pipeline)]

    threads = [threading.Thread(target=w.run, name=name, daemon=True) for name, w in workers]
    for t in threads:
        t.start()
    logger.log_info("onboard service running (%d threads)" % len(threads))

    while not stop_event.wait(0.2):
        pass  # poll rather than block forever, so the stop latency is bounded

    logger.log_info("stopping threads, releasing resources")
    for t in threads:
        # Bounded join so an unresponsive thread cannot block shutdown past
        # the unit's TimeoutStopSec; daemon=True means a straggler is killed.
        t.join(timeout=2.0)
        if t.is_alive():
            logger.log_error(f"thread {t.name} did not stop within 2 s")
    try:
        camera.release()
    except Exception as exc:
        logger.log_error(f"camera: release failed: {exc}")
    if mavlink_client is not None:
        mavlink_client.close()

    return 1 if video_stream.failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    logger = Logger(config.log_file)
    try:
        camera, mavlink_client = parallel_init(config, logger)
    except Exception as exc:
        logger.log_error(f"initialization failed: {exc}")
        return 1  # -> systemd Restart=on-failure
    return run(config, camera, mavlink_client, logger)


if __name__ == "__main__":
    shutdown_exit(main())
