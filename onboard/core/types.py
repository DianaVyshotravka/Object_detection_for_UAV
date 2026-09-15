"""Core data classes for the onboard pipeline.

Frame, Telemetry and Detection carry data
between threads. They live in one module rather than three 8-line files;
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Telemetry:
    """One GLOBAL_POSITION_INT sample.

    ``t_boot_s`` is the autopilot's own clock (time since its boot), NOT our
    wall clock -- ClockSync.get_offset() converts it. Storing it raw means a
    later TIMESYNC correction improves every sample already in the buffer.
    """

    t_boot_s: float
    lat: float
    lon: float
    alt_msl_m: float
    rel_alt_m: float = 0.0
    heading_deg: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "lat": round(self.lat, 7),
            "lon": round(self.lon, 7),
            "alt_msl_m": round(self.alt_msl_m, 2),
            "rel_alt_m": round(self.rel_alt_m, 2),
            "heading_deg": self.heading_deg,
        }


@dataclass
class Detection:
    """One detected object."""

    cls_name: str
    conf: float
    bbox_xyxy: list

    def as_dict(self) -> dict:
        return {
            "class": self.cls_name,
            "conf": round(float(self.conf), 4),
            "bbox_xyxy": [round(float(v), 1) for v in self.bbox_xyxy],
        }


@dataclass
class Frame:
    """A captured frame on its way through the pipeline.

    ``timestamp`` is local wall-clock seconds at the moment of exposure, i.e.
    already corrected by the camera latency knob (see VideoStream).
    """

    frame_id: int
    timestamp: float
    image: Any = field(repr=False, default=None)
    telemetry: Optional[Telemetry] = None
