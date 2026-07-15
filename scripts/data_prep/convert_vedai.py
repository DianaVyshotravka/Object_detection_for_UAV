"""Convert VEDAI labels to YOLO HBB txt (project_plan.md §2 step 1).

VEDAI ships 512px and 1024px co-registered RGB+IR image variants plus a
10-fold cross-validation split, but this project re-splits everything
itself (70/15/15 by source image, see build_dataset.py), so folds are
ignored here -- every annotated image is converted.

Expected raw layout (adjust with --images-dir/--annotations-dir if your
download differs):
    data/raw/vedai/images/<8-digit-id>_co.png   (RGB; *_ir.png variants are skipped)
    data/raw/vedai/metadata/<8-digit-id>.txt
The metadata/ folder also ships fold01..fold10(+test).txt,  fold.mat and
a combined annotation1024.txt -- only files whose name is exactly an
8-digit id are treated as per-image annotations, everything else in
that folder is ignored.

Each annotation line has 14 columns (confirmed against both a reference
devkit reimplementation and the actual per-image class-id counts in
this download):
    cx cy angle class_id is_contained is_occluded x1 x2 x3 x4 y1 y2 y3 y4
Only ids present in class_map.VEDAI_CLASS_MAP are converted; anything
else raises rather than being silently mismapped.

A handful of files (observed: 4/1250) instead use a reduced 5-column
format with no angle or corner geometry at all (cx cy class_id
is_contained is_occluded) -- there's no way to build a box from a
center point alone, so those files are skipped entirely (image
excluded too, not just the unlabelable objects).

Writes YOLO txt + an image symlink per kept image to:
    data/interim/vedai/images/<id>.<ext>   (symlink to raw image)
    data/interim/vedai/labels/<id>.txt

Usage:
    python convert_vedai.py --raw-dir data/raw/vedai --out-dir data/interim/vedai
"""

import argparse
import re
import sys
from pathlib import Path

from class_map import NAME_TO_ID, VEDAI_CLASS_MAP
from common import get_image_size, hbb_from_quad, to_yolo_line

RGB_SUFFIXES = ("_co.png", "_co.jpg", ".png", ".jpg")
PER_IMAGE_ANNOTATION_NAME = re.compile(r"\d{8}")


def find_rgb_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in RGB_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def parse_annotation_file(path: Path) -> list[tuple[int, list[float]]] | None:
    """Return list of (class_id, [x1,x2,x3,x4,y1,y2,y3,y4]) per object line.

    Returns None if the file uses the reduced 5-column format (no corner
    geometry) instead of the normal 14 -- the whole file is unusable
    then, not just individual lines.
    """
    objects = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 14:
            return None
        class_id = int(float(parts[3]))
        corners = [float(v) for v in parts[6:14]]
        objects.append((class_id, corners))
    return objects


def convert(raw_dir: Path, out_dir: Path, images_dir: Path | None, annotations_dir: Path | None) -> dict:
    images_dir = images_dir or raw_dir / "images"
    annotations_dir = annotations_dir or raw_dir / "metadata"
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise FileNotFoundError(
            f"expected {images_dir} and {annotations_dir} -- "
            "pass --images-dir/--annotations-dir if your VEDAI download uses different folder names"
        )

    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "instances_kept": 0, "instances_degenerate": 0, "files_skipped_no_geometry": 0}
    unknown_ids = set()

    for ann_path in sorted(annotations_dir.glob("*.txt")):
        stem = ann_path.stem
        if not PER_IMAGE_ANNOTATION_NAME.fullmatch(stem):
            continue  # fold01.txt, fold01test.txt, annotation1024.txt, etc.
        image_path = find_rgb_image(images_dir, stem)
        if image_path is None:
            print(f"  warning: no RGB image found for {ann_path.name}, skipping", file=sys.stderr)
            continue

        img_w, img_h = get_image_size(image_path)
        objects = parse_annotation_file(ann_path)
        if objects is None:
            stats["files_skipped_no_geometry"] += 1
            continue

        yolo_lines = []
        for class_id, corners in objects:
            if class_id not in VEDAI_CLASS_MAP:
                unknown_ids.add(class_id)
                continue
            xs, ys = corners[0:4], corners[4:8]
            xmin, ymin, xmax, ymax = hbb_from_quad(xs, ys)
            unified_id = NAME_TO_ID[VEDAI_CLASS_MAP[class_id]]
            line = to_yolo_line(unified_id, xmin, ymin, xmax, ymax, img_w, img_h)
            if line is None:
                stats["instances_degenerate"] += 1
                continue
            yolo_lines.append(line)
            stats["instances_kept"] += 1

        out_image_link = out_images / image_path.name
        if not out_image_link.exists():
            out_image_link.symlink_to(image_path.resolve())
        # label filename must match the IMAGE's stem (e.g. "00000001_co"), not the
        # annotation file's stem (e.g. "00000001") -- every downstream consumer
        # (tile_images.py, YOLO training itself) looks up a label by image stem.
        (out_labels / f"{image_path.stem}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
        stats["images"] += 1

    if unknown_ids:
        raise ValueError(
            f"Unknown/unverified VEDAI class ids encountered (not in class_map.VEDAI_CLASS_MAP): "
            f"{sorted(unknown_ids)}. Confirm each id against your VEDAI devkit README, "
            f"then add it to class_map.py before re-running."
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/vedai"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/interim/vedai"))
    parser.add_argument("--images-dir", type=Path, default=None, help="Override if your download's image folder isn't <raw-dir>/images")
    parser.add_argument("--annotations-dir", type=Path, default=None, help="Override if your download's annotation folder isn't <raw-dir>/metadata")
    args = parser.parse_args()

    if not args.raw_dir.is_dir():
        print(f"skip {args.raw_dir}: not found (place your VEDAI download there first)", file=sys.stderr)
        return

    stats = convert(args.raw_dir, args.out_dir, args.images_dir, args.annotations_dir)
    print(
        f"done: {stats['images']} images, {stats['instances_kept']} instances kept, "
        f"{stats['instances_degenerate']} dropped (degenerate box), "
        f"{stats['files_skipped_no_geometry']} files skipped (no corner geometry)"
    )


if __name__ == "__main__":
    main()
