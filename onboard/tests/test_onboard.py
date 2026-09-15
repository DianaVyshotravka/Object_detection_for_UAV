"""Tests for the onboard service.

Runs on a dev box with no camera, no autopilot, no pymavlink and no model:
the capture device, the MAVLink link and the detector are all injected, which
is why VideoStream takes a capture object and run() takes a detector.

    pytest onboard/tests -q
"""

import json
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from onboard.config import Config, StorageConfig  # noqa: E402
from onboard.core import geotag  # noqa: E402
from onboard.core.clock_sync import ClockSync  # noqa: E402
from onboard.core.logger import Logger  # noqa: E402
from onboard.core.pipeline import DetectionPipeline, nearest_telemetry  # noqa: E402
from onboard.core.types import Detection, Frame, Telemetry  # noqa: E402
from onboard.core.video_stream import VideoStream  # noqa: E402

MS = 1_000_000


# --- fakes ---------------------------------------------------------------

class FakeCapture:
    """cv2-shaped capture. ``ok=False`` simulates an unplugged camera."""

    def __init__(self, ok=True, size=(48, 64)):
        self.ok = ok
        self.size = size
        self.reads = 0
        self.released = False

    def read(self):
        self.reads += 1
        if not self.ok:
            return False, None
        return True, np.zeros((*self.size, 3), np.uint8)

    def release(self):
        self.released = True


class FakeDetector:
    def __init__(self, detections=None):
        self.detections = detections if detections is not None else []

    def infer(self, image):
        return list(self.detections)


def make_config(out_dir, **storage):
    return Config(storage=StorageConfig(out_dir=str(out_dir), **storage))


def make_clock_sync(sent=None, **kwargs):
    sent = sent if sent is not None else []
    return ClockSync(lambda tc1, ts1: sent.append((tc1, ts1)),
                     threading.Event(), Logger(), **kwargs)


# --- ClockSync: the TIMESYNC math ---------------------------------------

def test_timesync_offset_math():
    """offset = (send + recv)/2 - autopilot_time, i.e. local = remote + offset."""
    sync = make_clock_sync()
    sync._ts1 = ts1 = 1_000 * MS
    recv_ns = 1_040 * MS                       # 40 ms round trip
    tc1 = 500 * MS                             # autopilot is 520 ms behind us
    sync.handle_timesync(SimpleNamespace(tc1=tc1, ts1=ts1), recv_ns)
    assert sync.get_offset() == pytest.approx(0.520, abs=1e-9)
    # Sign check: applying the offset to the autopilot's clock lands mid-flight
    # of the round trip, not 520 ms the wrong way.
    assert (tc1 / 1e9) + sync.get_offset() == pytest.approx((ts1 + recv_ns) / 2e9)


def test_timesync_ignores_requests():
    """tc1 == 0 is a request, not a response -- including our own echo."""
    sync = make_clock_sync()
    sync._ts1 = 1_000 * MS
    sync.handle_timesync(SimpleNamespace(tc1=0, ts1=1_000 * MS), 1_040 * MS)
    assert sync.get_offset() is None


def test_timesync_ignores_foreign_reply():
    sync = make_clock_sync()
    sync._ts1 = 1_000 * MS
    sync.handle_timesync(SimpleNamespace(tc1=500 * MS, ts1=999 * MS), 1_040 * MS)
    assert sync.get_offset() is None


def test_timesync_rejects_bad_rtt():
    """A late reply must not poison a good offset."""
    sync = make_clock_sync(max_rtt_s=0.5)
    sync._ts1 = 1_000 * MS
    sync.handle_timesync(SimpleNamespace(tc1=500 * MS, ts1=1_000 * MS), 1_040 * MS)
    good = sync.get_offset()

    sync._ts1 = 2_000 * MS
    sync.handle_timesync(SimpleNamespace(tc1=100 * MS, ts1=2_000 * MS), 4_000 * MS)  # 2 s rtt
    assert sync.get_offset() == good
    sync._ts1 = 3_000 * MS
    sync.handle_timesync(SimpleNamespace(tc1=100 * MS, ts1=3_000 * MS), 2_900 * MS)  # negative
    assert sync.get_offset() == good


def test_timesync_send_failure_survives():
    """A dropped link must not kill the sync thread."""
    def boom(tc1, ts1):
        raise OSError("link down")
    sync = ClockSync(boom, threading.Event(), Logger())
    sync._sync_once()  # must not raise
    assert sync.get_offset() is None


def test_clock_bootstrap_then_timesync_wins():
    sync = make_clock_sync()
    sync.bootstrap(t_boot_s=10.0, recv_wall_s=1010.0)
    assert sync.get_offset() == pytest.approx(1000.0)
    sync.bootstrap(t_boot_s=11.0, recv_wall_s=9999.0)   # only the first counts
    assert sync.get_offset() == pytest.approx(1000.0)

    sync._ts1 = 1_000 * MS
    sync.handle_timesync(SimpleNamespace(tc1=500 * MS, ts1=1_000 * MS), 1_040 * MS)
    assert sync.get_offset() == pytest.approx(0.520)


def test_utc_offset_is_separate_from_timesync():
    """TIMESYNC carries time-since-boot and cannot date a photo; SYSTEM_TIME can."""
    sync = make_clock_sync()
    sync._ts1 = 1_000 * MS
    sync.handle_timesync(SimpleNamespace(tc1=500 * MS, ts1=1_000 * MS), 1_040 * MS)
    assert sync.to_utc(5000.0) == 5000.0        # no SYSTEM_TIME yet -> Pi clock unchanged

    sync.set_utc_offset(unix_s=1_700_000_000.0, recv_wall_s=5000.0)
    assert sync.to_utc(5000.0) == pytest.approx(1_700_000_000.0)
    sync.set_utc_offset(unix_s=0.0, recv_wall_s=5000.0)   # no GPS fix -> ignored
    assert sync.to_utc(5000.0) == pytest.approx(1_700_000_000.0)


def test_clock_sync_stops_immediately():
    """A 30 s interval must still shut down fast: stop_event.wait, not sleep."""
    stop = threading.Event()
    sync = ClockSync(lambda tc1, ts1: None, stop, Logger(), interval_s=30.0)
    thread = threading.Thread(target=sync.run)
    thread.start()
    time.sleep(0.05)
    stop.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()


# --- telemetry matching ---------------------------------------------------

def _sample(t_boot, lat=1.0):
    return Telemetry(t_boot_s=t_boot, lat=lat, lon=2.0, alt_msl_m=100.0)


def test_nearest_telemetry_picks_closest():
    buffer = deque(_sample(t, lat=t) for t in (10.0, 10.2, 10.4, 10.6))
    lock = threading.Lock()
    # offset 1000 -> samples sit at local 1010.0 .. 1010.6
    best = nearest_telemetry(buffer, lock, 1010.38, offset_s=1000.0, max_dt_s=0.5)
    assert best.t_boot_s == pytest.approx(10.4)


def test_nearest_telemetry_rejects_stale_and_unsynced():
    buffer = deque([_sample(10.0)])
    lock = threading.Lock()
    assert nearest_telemetry(buffer, lock, 1015.0, 1000.0, 0.5) is None   # 5 s stale
    assert nearest_telemetry(buffer, lock, 1010.0, None, 0.5) is None     # clock unsynced
    assert nearest_telemetry(deque(), lock, 1010.0, 1000.0, 0.5) is None  # no telemetry


def test_nearest_telemetry_snapshots_under_lock():
    """deque.append is atomic but ITERATION is not: iterating while the MAVLink
    reader appends raises "deque mutated during iteration". Asserted directly
    rather than by racing threads, which fails only intermittently."""

    class Tracker:
        held = False

        def __enter__(self):
            self.held = True
            return self

        def __exit__(self, *exc):
            self.held = False

    class WatchedDeque(deque):
        iterated_while_held = None

        def __iter__(self):
            WatchedDeque.iterated_while_held = tracker.held
            return super().__iter__()

    tracker = Tracker()
    buffer = WatchedDeque([_sample(10.0)])
    nearest_telemetry(buffer, tracker, 1010.0, offset_s=1000.0, max_dt_s=0.5)
    assert WatchedDeque.iterated_while_held is True, "snapshot taken without the lock"


# --- VideoStream ----------------------------------------------------------

def test_video_stream_drops_when_queue_full():
    """Full queue must drop, not block (the plan's put(timeout=1) stalls capture)."""
    frame_queue = queue.Queue(maxsize=2)
    stop = threading.Event()
    stream = VideoStream(FakeCapture(), frame_queue, stop, Logger())

    thread = threading.Thread(target=stream.run, daemon=True)
    started = time.monotonic()
    thread.start()
    while stream.frames_dropped < 20 and time.monotonic() - started < 2.0:
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=1.0)

    assert stream.frames_dropped >= 20, "capture blocked instead of dropping"
    assert frame_queue.qsize() == 2
    assert not thread.is_alive()


def test_video_stream_bails_on_dead_camera():
    """Failed reads must back off and stop, not spin a core at 100%."""
    capture = FakeCapture(ok=False)
    stop = threading.Event()
    stream = VideoStream(capture, queue.Queue(maxsize=2), stop, Logger(),
                         max_read_failures=5)
    started = time.monotonic()
    stream.run()

    assert stream.failed and stop.is_set()
    assert capture.reads == 5, "gave up at the wrong failure count"
    # 0.05*1 + 0.05*2 + 0.05*3 + 0.05*4 = 0.5 s of backoff before bailing.
    assert time.monotonic() - started > 0.4, "no backoff -- this is a busy-spin"


def test_video_stream_timestamps_compensate_latency():
    frame_queue = queue.Queue(maxsize=4)
    stop = threading.Event()
    stream = VideoStream(FakeCapture(), frame_queue, stop, Logger(), latency_s=0.25)
    thread = threading.Thread(target=stream.run, daemon=True)
    thread.start()
    frame = frame_queue.get(timeout=2.0)
    stop.set()
    thread.join(timeout=1.0)
    assert time.time() - frame.timestamp >= 0.25


# --- pipeline -------------------------------------------------------------

def make_pipeline(tmp_path, detections=None, clock_sync=None, **storage):
    config = make_config(tmp_path, **storage)
    return DetectionPipeline(queue.Queue(), deque(maxlen=50), threading.Lock(),
                             clock_sync, FakeDetector(detections), config,
                             Logger(), threading.Event())


def a_frame(frame_id=1):
    return Frame(frame_id=frame_id, timestamp=time.time(),
                 image=np.zeros((48, 64, 3), np.uint8))


def test_pipeline_saves_only_on_detection(tmp_path):
    pipeline = make_pipeline(tmp_path, detections=[])
    assert pipeline.process(a_frame(1))["image"] is None
    assert list(tmp_path.glob("*.jpg")) == []

    pipeline.detector.detections = [Detection("person", 0.88, [10, 20, 30, 40])]
    assert pipeline.process(a_frame(2))["image"] == "000002.jpg"
    assert (tmp_path / "000002.jpg").exists()


def test_pipeline_creates_a_fresh_out_dir(tmp_path):
    """First flight: out_dir does not exist yet, and the disk-space check
    statvfs's it before anything is written."""
    fresh = tmp_path / "flights" / "2026-09-14"
    pipeline = make_pipeline(fresh, detections=[Detection("person", 0.9, [1, 2, 3, 4])])
    record = pipeline.process(a_frame(1))
    pipeline._write(record)
    assert (fresh / "000001.jpg").exists()
    assert (fresh / "detections.jsonl").exists()


def test_pipeline_save_mode_all(tmp_path):
    pipeline = make_pipeline(tmp_path, detections=[], save_mode="all")
    pipeline.process(a_frame(7))
    assert (tmp_path / "000007.jpg").exists()


def test_record_matches_project_plan_schema(tmp_path):
    """Pinned against docs/project_plan.md:21-32 and the dicts replay_app.py emits."""
    pipeline = make_pipeline(tmp_path, detections=[Detection("person", 0.8756, [1, 2, 3, 4])])
    record = pipeline.process(a_frame(1287))
    pipeline._write(record)

    assert {"frame_id", "timestamp", "image_size", "detections"} <= set(record)
    assert record["frame_id"] == 1287
    assert record["image_size"] == [64, 48]          # [width, height]
    assert record["timestamp"].endswith("Z")
    datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))

    detection = record["detections"][0]
    assert set(detection) == {"class", "conf", "bbox_xyxy"}
    assert detection["class"] == "person" and len(detection["bbox_xyxy"]) == 4

    written = json.loads((tmp_path / "detections.jsonl").read_text().splitlines()[0])
    assert written == record


def test_pipeline_attaches_nearest_telemetry(tmp_path):
    sync = make_clock_sync()
    sync.bootstrap(t_boot_s=0.0, recv_wall_s=0.0)     # offset 0: boot time == local time
    pipeline = make_pipeline(tmp_path, detections=[Detection("vehicle", 0.9, [1, 2, 3, 4])],
                             clock_sync=sync)
    frame = a_frame(1)
    with pipeline.telemetry_lock:
        pipeline.telemetry_buffer.append(_sample(frame.timestamp - 0.05, lat=50.45))
        pipeline.telemetry_buffer.append(_sample(frame.timestamp - 9.0, lat=11.11))

    record = pipeline.process(frame)
    assert record["telemetry"]["lat"] == pytest.approx(50.45)
    assert geotag.read_gps(tmp_path / "000001.jpg")[0] == pytest.approx(50.45, abs=1e-6)


def test_storage_guard_blocks_writes(tmp_path, monkeypatch):
    """A full card stops images but must not stop the flight."""
    import shutil as shutil_module
    pipeline = make_pipeline(tmp_path, detections=[Detection("person", 0.9, [1, 2, 3, 4])],
                             min_free_mb=10_000_000)
    monkeypatch.setattr(shutil_module, "disk_usage",
                        lambda _p: SimpleNamespace(total=0, used=0, free=1_000_000))

    record = pipeline.process(a_frame(1))
    assert record["image"] is None
    assert list(tmp_path.glob("*.jpg")) == []
    assert record["detections"], "detections must still be recorded"


def test_save_min_interval_throttles(tmp_path):
    pipeline = make_pipeline(tmp_path, detections=[Detection("person", 0.9, [1, 2, 3, 4])],
                             save_min_interval_s=10.0)
    now = time.time()
    assert pipeline.process(Frame(1, now, np.zeros((48, 64, 3), np.uint8)))["image"]
    assert pipeline.process(Frame(2, now + 0.1, np.zeros((48, 64, 3), np.uint8)))["image"] is None
    assert pipeline.process(Frame(3, now + 11.0, np.zeros((48, 64, 3), np.uint8)))["image"]


def test_pipeline_survives_a_bad_frame(tmp_path):
    """One exploding frame must not end the flight."""
    pipeline = make_pipeline(tmp_path)
    pipeline.detector.infer = lambda image: (_ for _ in ()).throw(RuntimeError("ncnn blew up"))
    pipeline.frame_queue.put(a_frame(1))
    thread = threading.Thread(target=pipeline.run, daemon=True)
    thread.start()
    time.sleep(0.2)
    assert thread.is_alive()
    pipeline.stop_event.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


# --- geotag ---------------------------------------------------------------

@pytest.mark.parametrize("lat,lon,alt", [
    (50.4501, 30.5234, 180.0),      # Kyiv, N/E, above sea level
    (-33.8688, -70.6693, -25.5),    # S/W, below sea level
])
def test_geotag_roundtrip(tmp_path, lat, lon, alt):
    path = tmp_path / "f.jpg"
    telemetry = Telemetry(t_boot_s=1.0, lat=lat, lon=lon, alt_msl_m=alt, heading_deg=93.5)
    geotag.save_jpeg(path, np.zeros((16, 16, 3), np.uint8), telemetry, time.time())

    got_lat, got_lon, got_alt = geotag.read_gps(path)
    assert got_lat == pytest.approx(lat, abs=1e-6)
    assert got_lon == pytest.approx(lon, abs=1e-6)
    assert got_alt == pytest.approx(alt, abs=0.01)


def test_geotag_without_telemetry(tmp_path):
    """No GPS is never a reason to lose the frame."""
    path = tmp_path / "f.jpg"
    geotag.save_jpeg(path, np.zeros((16, 16, 3), np.uint8), None, time.time())
    assert path.exists() and geotag.read_gps(path) is None


# --- map builder ----------------------------------------------------------

def test_build_map(tmp_path):
    from onboard.build_map import collect, main as build_main

    telemetry = Telemetry(t_boot_s=1.0, lat=50.4501, lon=30.5234, alt_msl_m=180.0)
    geotag.save_jpeg(tmp_path / "000001.jpg", np.zeros((16, 16, 3), np.uint8),
                     telemetry, time.time())
    geotag.save_jpeg(tmp_path / "000002.jpg", np.zeros((16, 16, 3), np.uint8),
                     None, time.time())  # no fix -> no marker
    (tmp_path / "detections.jsonl").write_text(json.dumps({
        "frame_id": 1, "image": "000001.jpg", "timestamp": "2026-09-13T10:00:00.000Z",
        "detections": [{"class": "person", "conf": 0.9, "bbox_xyxy": [1, 2, 3, 4]}]}) + "\n")

    features = collect(tmp_path)["features"]
    assert len(features) == 1
    assert features[0]["geometry"]["coordinates"] == [30.5234, 50.4501]
    assert features[0]["properties"]["counts"] == {"person": 1}

    assert build_main(["--flight-dir", str(tmp_path)]) == 0
    assert (tmp_path / "map.html").exists() and (tmp_path / "track.geojson").exists()


# --- config ---------------------------------------------------------------

def test_load_config_rejects_unknown_and_missing(tmp_path):
    from onboard.config import load_config

    with pytest.raises(SystemExit):
        load_config(tmp_path / "nope.yaml")

    path = tmp_path / "c.yaml"
    path.write_text("detector:\n  model_path: /nope\n  typo_key: 1\n")
    with pytest.raises(SystemExit):
        load_config(path)

    path.write_text("detector:\n  model_path: /definitely/not/here\n")
    with pytest.raises(SystemExit, match="Model not found"):
        load_config(path)


def test_shipped_config_yaml_is_valid(tmp_path):
    """The committed config.yaml must parse -- a typo there is a failed boot."""
    import yaml
    data = yaml.safe_load((REPO_ROOT / "onboard" / "config.yaml").read_text())
    assert data["detector"]["imgsz"] == 512          # must match the NCNN export
    assert data["storage"]["save_mode"] in ("detections", "all")


def test_mavlink_module_imports_without_pymavlink():
    """The dev box has no pymavlink; only connect() may complain."""
    from onboard.core import mavlink_client

    if mavlink_client.mavutil is None:
        with pytest.raises(RuntimeError, match="pymavlink"):
            mavlink_client.MavlinkClient("udp:127.0.0.1:14550").connect(timeout=0.1)


# --- integration: the two terminal branches of Fig. 2.3 -------------------

def child_main():
    """Runs the full thread graph in a subprocess with fakes, so SIGTERM can be
    sent for real. Invoked by the test below, not by pytest."""
    out_dir = Path(sys.argv[-1])
    config = make_config(out_dir, save_mode="all")
    config.mavlink.enabled = False
    detector = FakeDetector([Detection("person", 0.9, [1, 2, 3, 4])])
    from onboard.main import run, shutdown_exit
    shutdown_exit(run(config, FakeCapture(), None, Logger(), detector=detector))


def test_sigterm_stops_within_2s(tmp_path):
    """Fig. 2.3 clean-stop branch: SIGTERM -> every thread down, results flushed."""
    process = subprocess.Popen(
        [sys.executable, "-c",
         "import onboard.tests.test_onboard as t; t.child_main()", str(tmp_path)],
        cwd=str(REPO_ROOT))
    try:
        deadline = time.monotonic() + 10.0
        records = tmp_path / "detections.jsonl"
        while not records.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert records.exists(), "service never started producing records"

        started = time.monotonic()
        process.terminate()                       # SIGTERM
        returncode = process.wait(timeout=5.0)
        elapsed = time.monotonic() - started
    finally:
        if process.poll() is None:
            process.kill()

    assert elapsed < 2.0, f"shutdown took {elapsed:.1f}s"
    assert returncode == 0, "clean stop must exit 0 so systemd does not restart"
    # Flushed, not truncated -- proves the join-before-exit path.
    lines = records.read_text().splitlines()
    assert lines and json.loads(lines[-1])["detections"]


def test_main_exits_1_on_init_failure(tmp_path):
    """Fig. 2.3 init-error branch -> exit 1 -> systemd Restart=on-failure.

    Also guards the ThreadPoolExecutor deadlock: this must not hang.
    """
    config = tmp_path / "c.yaml"
    config.write_text(
        "camera:\n  source: /nonexistent/video.mp4\n  backend: cv2\n"
        "mavlink:\n  enabled: false\n"
        f"detector:\n  model_path: {REPO_ROOT}\n"
        f"storage:\n  out_dir: {tmp_path}\n")

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; from onboard.main import main; sys.exit(main())",
         "--config", str(config)],
        cwd=str(REPO_ROOT), timeout=30, capture_output=True, text=True)
    assert result.returncode == 1, result.stderr[-2000:]
