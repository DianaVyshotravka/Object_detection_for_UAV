"""Tile large source images + their YOLO labels (project_plan.md §2 step 3).

DOTA/VisDrone-scale images (up to 20000x20000) lose small objects when
downscaled to the 320-640px training resolution, so large images are
sliced into overlapping windows before training. Images already small
enough (e.g. VEDAI's 512/1024px frames) pass through unchanged.

This mirrors what SAHI's slicing does conceptually, but operates
directly on already-converted YOLO txt (produced by convert_dota.py /
convert_vedai.py) instead of round-tripping through COCO -- SAHI itself
remains the right tool for *sliced inference* later (project_plan.md §2
and §4), which is a separate, inference-time step from this one.

Input:  <interim-dir>/images/*, <interim-dir>/labels/*.txt (normalized
        to each image's own size)
Output: <out-dir>/images/*, <out-dir>/labels/*.txt (normalized to each
        tile's size), <out-dir>/manifest.csv (tile -> parent image, so
        build_dataset.py can split by parent and never leak a source
        image's tiles across train/val/test)

Usage:
    python tile_images.py --interim-dir data/interim/dota/train \
        --out-dir data/interim/dota_tiled/train --source-name dota
"""

import argparse
import csv
from pathlib import Path

from PIL import Image

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, w, h = line.split()
        boxes.append((int(cls), float(cx), float(cy), float(w), float(h)))
    return boxes


def tile_origins(size: int, tile: int, overlap: float) -> list[int]:
    """1D tile start coordinates covering [0, size) with the given overlap fraction."""
    if size <= tile:
        return [0]
    stride = int(tile * (1 - overlap))
    origins = list(range(0, size - tile + 1, stride))
    if origins[-1] != size - tile:
        origins.append(size - tile)  # flush last tile to the far edge, no ragged remainder
    return origins


def slice_image(
    image_path: Path,
    labels_path: Path,
    out_images: Path,
    out_labels: Path,
    tile_size: int,
    overlap: float,
    min_visibility: float,
) -> list[str]:
    """Returns list of tile image filenames written for this source image."""
    with Image.open(image_path) as img:
        img_w, img_h = img.size

        if img_w <= tile_size and img_h <= tile_size:
            out_path = out_images / image_path.name
            img.save(out_path)
            labels_out = out_labels / f"{image_path.stem}.txt"
            labels_out.write_text(labels_path.read_text() if labels_path.exists() else "")
            return [image_path.name]

        boxes = read_yolo_labels(labels_path)
        # denormalize once to pixel-space absolute boxes (xmin, ymin, xmax, ymax)
        abs_boxes = []
        for cls, cx, cy, w, h in boxes:
            bw, bh = w * img_w, h * img_h
            bx, by = cx * img_w, cy * img_h
            abs_boxes.append((cls, bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2))

        written = []
        for row, y0 in enumerate(tile_origins(img_h, tile_size, overlap)):
            for col, x0 in enumerate(tile_origins(img_w, tile_size, overlap)):
                # clamp to the real image bounds -- tile_origins emits a single
                # origin=0 for any axis shorter than tile_size, and a fixed
                # x0+tile_size/y0+tile_size window would then reach past the
                # image edge, which Image.crop pads with solid black rather
                # than raising. Clamping keeps the tile's actual pixel content
                # (and its normalization denominator, tile_w/tile_h) truthful.
                x1, y1 = min(x0 + tile_size, img_w), min(y0 + tile_size, img_h)
                tile_w, tile_h = x1 - x0, y1 - y0
                tile_lines = []
                for cls, bxmin, bymin, bxmax, bymax in abs_boxes:
                    ix0, iy0 = max(bxmin, x0), max(bymin, y0)
                    ix1, iy1 = min(bxmax, x1), min(bymax, y1)
                    if ix1 <= ix0 or iy1 <= iy0:
                        continue
                    orig_area = (bxmax - bxmin) * (bymax - bymin)
                    inter_area = (ix1 - ix0) * (iy1 - iy0)
                    if orig_area <= 0 or inter_area / orig_area < min_visibility:
                        continue
                    tw, th = ix1 - ix0, iy1 - iy0
                    tcx, tcy = (ix0 + ix1) / 2 - x0, (iy0 + iy1) / 2 - y0
                    tile_lines.append(
                        f"{cls} {tcx / tile_w:.6f} {tcy / tile_h:.6f} "
                        f"{tw / tile_w:.6f} {th / tile_h:.6f}"
                    )

                tile_name = f"{image_path.stem}_tile_{row}_{col}{image_path.suffix}"
                img.crop((x0, y0, x1, y1)).save(out_images / tile_name)
                (out_labels / f"{Path(tile_name).stem}.txt").write_text(
                    "\n".join(tile_lines) + ("\n" if tile_lines else "")
                )
                written.append(tile_name)
        return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interim-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-name", required=True, help="Tag stored in manifest.csv, e.g. 'dota'")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.2,
        help="Drop a box from a tile if less than this fraction of its original area survives clipping",
    )
    args = parser.parse_args()

    images_dir = args.interim_dir / "images"
    labels_dir = args.interim_dir / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"{images_dir} not found -- run the relevant convert_*.py first")

    out_images = args.out_dir / "images"
    out_labels = args.out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for image_path in image_paths:
        labels_path = labels_dir / f"{image_path.stem}.txt"
        tiles = slice_image(
            image_path, labels_path, out_images, out_labels,
            args.tile_size, args.overlap, args.min_visibility,
        )
        for tile_name in tiles:
            manifest_rows.append({"tile": tile_name, "parent_image": image_path.name, "source_dataset": args.source_name})

    with (args.out_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tile", "parent_image", "source_dataset"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"done: {len(image_paths)} source images -> {len(manifest_rows)} tiles/images in {args.out_dir}")


if __name__ == "__main__":
    main()
