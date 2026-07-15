"""Shared helpers for source-dataset -> YOLO txt converters."""

from pathlib import Path

from PIL import Image


def get_image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) without decoding full pixel data."""
    with Image.open(path) as img:
        return img.size


def hbb_from_quad(xs: list[float], ys: list[float]) -> tuple[float, float, float, float]:
    """Axis-aligned min/max HBB (xmin, ymin, xmax, ymax) from 4 corner points."""
    return min(xs), min(ys), max(xs), max(ys)


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def to_yolo_line(
    class_id: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: int,
    img_h: int,
) -> str | None:
    """Convert a pixel-space HBB to a normalized YOLO txt line.

    Returns None if the box is degenerate after clipping to image bounds
    (fully outside the image, or zero area).
    """
    xmin = clip(xmin, 0, img_w)
    xmax = clip(xmax, 0, img_w)
    ymin = clip(ymin, 0, img_h)
    ymax = clip(ymax, 0, img_h)
    w = xmax - xmin
    h = ymax - ymin
    if w <= 0 or h <= 0:
        return None
    cx = xmin + w / 2
    cy = ymin + h / 2
    return f"{class_id} {cx / img_w:.6f} {cy / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"
