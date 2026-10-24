"""VideoStream -- frame capture thread.

VideoStream itself never opens a device: it takes an already-opened capture
object exposing ``read() -> (ok, image)`` and ``release()``, plus an optional
``last_timestamp`` (monotonic seconds of the last frame's exposure, or None
when the device cannot say). That keeps cv2 and
picamera2 out of this class and makes the thread testable against a fake.
``open_capture()`` below is the factory that does touch the hardware.
"""

import queue
import threading
import time

from .types import Frame

MAX_SENSOR_SKEW_S = 1.0


class Picamera2Capture:
    """cv2-shaped adapter over Picamera2 (CSI ribbon cameras on Pi OS Bookworm,
    which libcamera owns and cv2.VideoCapture cannot open).

    Also reports the true exposure time: libcamera's SensorTimestamp (ns,
    CLOCK_MONOTONIC on the Pi's CSI drivers) is stamped by the kernel at
    frame start, so buffer depth and delivery lag drop out of the geotag
    without calibrating camera.latency_ms.
    """

    def __init__(self, width: int, height: int, logger=None):
        from picamera2 import Picamera2  # imported here so this module loads on a dev box

        self.logger = logger
        self.last_timestamp = None
        self._skew_logged = False
        self._cam = Picamera2()
        # picamera2's "RGB888" is BGR in memory, which is what cv2/Ultralytics expect.
        self._cam.configure(self._cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}, buffer_count=2))
        self._cam.start()

    def read(self):
        request = self._cam.capture_request()
        try:
            image = request.make_array("main")
            sensor_ns = request.get_metadata().get("SensorTimestamp")
        finally:
            request.release()  # unreleased requests starve the 2-buffer pool
        self.last_timestamp = self._check(sensor_ns)
        return True, image

    def _check(self, sensor_ns):
        if sensor_ns is None:
            return None
        stamp = sensor_ns / 1e9
        skew = time.monotonic() - stamp
        if not 0.0 <= skew <= MAX_SENSOR_SKEW_S:
            if not self._skew_logged and self.logger is not None:
                self.logger.log_warning("camera: SensorTimestamp %.3f s off monotonic, "
                                        "using delivery time" % skew)
                self._skew_logged = True
            return None
        return stamp

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
            return Picamera2Capture(camera_cfg.width, camera_cfg.height, logger)
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

    def _enqueue_newest(self, frame) -> None:
        """Queue the frame, evicting the oldest rather than dropping this one.

        Never blocks: blocking here would desync capture from real time and
        geotag later frames at stale positions.
        """
        try:
            self.frame_queue.put_nowait(frame)
            return
        except queue.Full:
            pass
        try:
            self.frame_queue.get_nowait()  # discard oldest, keep the freshest
        except queue.Empty:
            pass 
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass 
        self.frames_dropped += 1

    def run(self) -> None:
        fails = 0
        frame_id = 0
        last_report = time.monotonic()

        while not self.stop_event.is_set():
            try:
                ok, image = self.capture.read()
            except Exception as exc: 
                self.logger.log_error(f"camera: read raised: {exc}")
                ok, image = False, None
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
            # Sensor exposure time when the device reports one (Picamera2),
            # else delivery time.
            stamp = getattr(self.capture, "last_timestamp", None)
            frame = Frame(frame_id=frame_id,
                          timestamp=(stamp if stamp is not None else time.monotonic())
                          - self.latency_s,
                          image=image)
            self._enqueue_newest(frame)

            now = time.monotonic()
            if now - last_report >= self.report_every_s:
                self.logger.log_info("camera: %d frames read, %d dropped"
                                     % (self.frames_read, self.frames_dropped))
                last_report = now

        self.logger.log_info("camera: capture stopped (%d read, %d dropped)"
                             % (self.frames_read, self.frames_dropped))
