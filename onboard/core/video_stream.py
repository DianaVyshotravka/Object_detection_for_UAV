"""VideoStream -- frame capture thread.

VideoStream itself never opens a device: it takes an already-opened capture
object exposing ``read() -> (ok, image)`` and ``release()``. That keeps cv2 and
picamera2 out of this class and makes the thread testable against a fake.
``open_capture()`` below is the factory that does touch the hardware.
"""

import queue
import threading
import time

from .types import Frame


class Picamera2Capture:
    """cv2-shaped adapter over Picamera2 (CSI ribbon cameras on Pi OS Bookworm,
    which libcamera owns and cv2.VideoCapture cannot open)."""

    def __init__(self, width: int, height: int):
        from picamera2 import Picamera2  # imported here so this module loads on a dev box

        self._cam = Picamera2()
        # picamera2's "RGB888" is BGR in memory, which is what cv2/Ultralytics expect.
        self._cam.configure(self._cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}, buffer_count=2))
        self._cam.start()

    def read(self):
        return True, self._cam.capture_array()

    def release(self):
        self._cam.stop()
        self._cam.close()


def open_capture(camera_cfg, logger):
    """Open the camera per config: auto | picamera2 | cv2.

    A file path or a plain integer source always goes to cv2 -- that is what
    lets the whole service run against a recorded video on a dev box.
    """
    import cv2

    source = camera_cfg.source
    want_pi = camera_cfg.backend in ("auto", "picamera2") and str(source) in ("", "csi", "picamera2")

    if want_pi:
        try:
            return Picamera2Capture(camera_cfg.width, camera_cfg.height)
        except Exception as exc:
            if camera_cfg.backend == "picamera2":
                raise
            logger.log_warning(f"camera: picamera2 unavailable ({exc}), falling back to cv2")

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera source {source!r}")
    # Depth-1 buffer: a deeper V4L2 queue hands us frames that are already old,
    # which geotags them at the wrong position. Ignored by file sources.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.height)
    return cap


class VideoStream:
    def __init__(self, capture, frame_queue: queue.Queue, stop_event: threading.Event,
                 logger, latency_s: float = 0.0, max_read_failures: int = 30,
                 report_every_s: float = 30.0):
        self.capture = capture
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.logger = logger
        self.latency_s = latency_s
        self.max_read_failures = max_read_failures
        self.report_every_s = report_every_s

        self.frames_read = 0
        self.frames_dropped = 0
        self.failed = False

    def run(self) -> None:
        fails = 0
        frame_id = 0
        last_report = time.monotonic()

        while not self.stop_event.is_set():
            ok, image = self.capture.read()
            if not ok or image is None:
                fails += 1
                if fails >= self.max_read_failures:
                    self.logger.log_error(
                        "camera: %d consecutive read failures, stopping capture" % fails)
                    self.failed = True
                    self.stop_event.set()  # -> exit(1) -> systemd Restart=on-failure
                    return
                # Backoff instead of the plan's bare `continue`, which spins a
                # core at 100% and floods the log the moment the camera unplugs.
                self.stop_event.wait(min(0.05 * fails, 1.0))
                continue

            fails = 0
            frame_id += 1
            self.frames_read += 1
            # Subtract the measured capture latency so the timestamp is the
            # moment of exposure, not the moment of delivery.
            frame = Frame(frame_id=frame_id, timestamp=time.time() - self.latency_s,
                          image=image)
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                # Drop deliberately: blocking here would desync capture from
                # real time and geotag later frames at stale positions.
                self.frames_dropped += 1

            now = time.monotonic()
            if now - last_report >= self.report_every_s:
                self.logger.log_info("camera: %d frames read, %d dropped"
                                     % (self.frames_read, self.frames_dropped))
                last_report = now

        self.logger.log_info("camera: capture stopped (%d read, %d dropped)"
                             % (self.frames_read, self.frames_dropped))
