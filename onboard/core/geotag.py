"""Geotagging -- write GPS EXIF into the saved JPEG.

Pillow (already a dependency) builds and reads the GPS IFD correctly, so no
piexif. OpenCV encodes the pixels (12 ms vs Pillow's 66 ms for 1080p on a
Pi 5) and the EXIF block is spliced in as an APP1 segment. Two non-obvious details make this 50 lines rather than the "single
call" the plan assumed:

  * GPS rationals must be ``IFDRational``; plain ``(num, den)`` tuples raise
    ``TypeError: bad operand type for abs()`` inside Pillow's _limit_rational.
  * ``GPSAltitudeRef`` reads back as ``bytes`` (b"\\x00"), not ``int``.
"""

import struct
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from PIL.TiffImagePlugin import IFDRational

GPS_IFD = 0x8825
EXIF_IFD = 0x8769


def _dms(value: float) -> tuple:
    """Decimal degrees -> (deg, min, sec) as EXIF rationals. Sign dropped --
    the N/S/E/W ref tag carries it."""
    # Round once, in integer 1/10000 s, then split
    total = round(abs(value) * 3600 * 10000)
    deg, rest = divmod(total, 3600 * 10000)
    minutes, seconds = divmod(rest, 60 * 10000)
    return (IFDRational(deg, 1), IFDRational(minutes, 1), IFDRational(seconds, 10000))


def build_exif(telemetry, when: float) -> Image.Exif:
    """EXIF block with GPS position and the UTC timestamp of the exposure."""
    stamp = datetime.fromtimestamp(when, tz=timezone.utc)
    exif = Image.Exif()
    text = stamp.strftime("%Y:%m:%d %H:%M:%S")
    exif[0x0132] = text 
    sub = exif.get_ifd(EXIF_IFD)
    sub[0x9003] = text        # DateTimeOriginal
    sub[0x9011] = "+00:00"    # OffsetTimeOriginal

    alt = telemetry.alt_msl_m
    gps = {
        1: "N" if telemetry.lat >= 0 else "S",
        2: _dms(telemetry.lat),
        3: "E" if telemetry.lon >= 0 else "W",
        4: _dms(telemetry.lon),
        5: b"\x00" if alt >= 0 else b"\x01",  # 0 = above sea level, 1 = below
        6: IFDRational(round(abs(alt) * 100), 100),
        7: (IFDRational(stamp.hour, 1), IFDRational(stamp.minute, 1),
            IFDRational(stamp.second, 1)),
        29: stamp.strftime("%Y:%m:%d"),
    }
    if telemetry.heading_deg is not None:
        gps[16] = "T"  # true north
        gps[17] = IFDRational(round(telemetry.heading_deg * 100), 100)
    exif[GPS_IFD] = gps
    return exif


def save_jpeg(path: Path, bgr_image, telemetry, when: float, quality: int = 85) -> None:
    """Write one BGR frame as JPEG, geotagged when telemetry is available."""
    import cv2

    jpeg = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
    if telemetry is not None:  # no GPS is not a reason to lose the frame
        exif = build_exif(telemetry, when).tobytes()  # starts with b"Exif\0\0"
        # APP1 right after SOI (FFD8), where EXIF readers look for it.
        jpeg = jpeg[:2] + b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif + jpeg[2:]
    # Write then rename: the live view lists *.jpg and must never serve half a file.
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_bytes(jpeg)
    tmp.replace(path)


def read_gps(path: Path):
    """Read (lat, lon, alt_m) back out of a geotagged JPEG, or None."""
    gps = Image.open(path).getexif().get_ifd(GPS_IFD)
    if not gps or 2 not in gps or 4 not in gps:
        return None

    def to_deg(dms, ref) -> float:
        deg = float(dms[0]) + float(dms[1]) / 60 + float(dms[2]) / 3600
        return -deg if ref in ("S", "W") else deg

    alt = float(gps.get(6, 0))
    ref = gps.get(5, 0)
    if (ref if isinstance(ref, int) else ref[0]) == 1:  # bytes on read-back
        alt = -alt
    return to_deg(gps[2], gps[1]), to_deg(gps[4], gps[3]), alt
