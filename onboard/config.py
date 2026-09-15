"""Runtime configuration (StorageConfig and friends, table 2.3).

Loaded from config.yaml so the systemd unit stays a single ExecStart line and
the operator can retune the service on the Pi without touching code.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import yaml

DEPLOY_ROOT = Path(__file__).resolve().parents[1]  # the parent of onboard/


@dataclass
class CameraConfig:
    source: str = "0"          # "0"/"1" = cv2 device index, a path = video file, "csi" = Picamera2
    backend: str = "auto"      # auto | picamera2 | cv2
    width: int = 1920
    height: int = 1080
    latency_ms: float = 0.0    # calibration knob: V4L2 buffer depth + exposure, see below
    max_read_failures: int = 30


@dataclass
class MavlinkConfig:
    enabled: bool = True
    url: str = "/dev/serial0"
    baud: int = 57600
    telemetry_hz: int = 5
    timesync_interval_s: float = 10.0
    max_rtt_s: float = 0.5


@dataclass
class DetectorConfig:
    model_path: str = ""
    imgsz: int = 512
    conf: float = 0.25
    iou: float = 0.7


@dataclass
class StorageConfig:
    out_dir: str = "flights"
    save_mode: str = "detections"   # detections | all
    jpeg_quality: int = 85
    save_min_interval_s: float = 0.0
    min_free_mb: int = 512


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    mavlink: MavlinkConfig = field(default_factory=MavlinkConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    log_file: Optional[str] = None
    init_timeout_s: float = 10.0
    max_telemetry_dt_s: float = 0.5
    frame_queue_size: int = 2       # drop-newest: a deep queue geotags stale frames
    telemetry_buffer_size: int = 50


_SECTIONS = {"camera": CameraConfig, "mavlink": MavlinkConfig,
             "detector": DetectorConfig, "storage": StorageConfig}


def _build(cls, data: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise SystemExit(f"{cls.__name__}: unknown setting(s) {sorted(unknown)}")
    return cls(**data)


def load_config(path: Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    sections = {name: _build(cls, data.pop(name, None) or {})
                for name, cls in _SECTIONS.items()}
    config = _build(Config, {**data, **sections})

    if not config.detector.model_path:
        raise SystemExit("detector.model_path is required")
    # Relative paths resolve against the deploy root, so the same config works
    # from a systemd WorkingDirectory and from a manual run in a subdirectory.
    for obj, attr in ((config.detector, "model_path"), (config.storage, "out_dir")):
        value = Path(getattr(obj, attr))
        if not value.is_absolute():
            setattr(obj, attr, str((DEPLOY_ROOT / value).resolve()))
    if not Path(config.detector.model_path).exists():
        raise SystemExit(f"Model not found: {config.detector.model_path}")
    if config.storage.save_mode not in ("detections", "all"):
        raise SystemExit("storage.save_mode must be 'detections' or 'all'")
    return config
