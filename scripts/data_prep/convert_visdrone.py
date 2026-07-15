"""Convert VisDrone2019-DET labels to unified YOLO txt (project_plan.md §2).

Boxes are already axis-aligned (unlike DOTA/VEDAI's oriented quads), so
this is a straight remap + normalize, no HBB-from-corners math needed.

Expected raw layout:
    data/raw/VisDrone2019/images/<name>.jpg
    data/raw/VisDrone2019/annotations/<name>.txt
Each annotation line (official VisDrone-DET format):
    bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion
A line is dropped if score==0 (VisDrone's own "ignore in evaluation"
flag) or category==0 (ignored region) -- either on its own is enough,
they're usually but not always paired. Category 11 ("others") and
bicycle (category 3) are dropped per class_map.VISDRONE_CLASS_MAP
(unspecified/no clear bucket).

This dataset isn't pre-split (all images in one flat pool), which is
fine -- like KIIT/VEDAI, build_dataset.py re-splits everything itself.

Writes YOLO txt + an image symlink per kept image to:
    data/interim/visdrone/images/<name>.jpg   (symlink to raw image)
    data/interim/visdrone/labels/<name>.txt

Usage:
    python convert_visdrone.py --raw-dir data/raw/VisDrone2019 --out-dir data/interim/visdrone
"""

import argparse
from pathlib import Path

from class_map import NAME_TO_ID, VISDRONE_CLASS_MAP
from common import get_image_size, to_yolo_line


def convert(raw_dir: Path, out_dir: Path) -> dict:
    images_dir = raw_dir / "images"
    annotations_dir = raw_dir / "annotations"
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise FileNotFoundError(f"expected {images_dir} and {annotations_dir}")

    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "instances_kept": 0, "instances_dropped_class": 0, "instances_ignored": 0, "instances_degenerate": 0}
    unknown_categories = set()

    for ann_path in sorted(annotations_dir.glob("*.txt")):
        stem = ann_path.stem
        image_path = images_dir / f"{stem}.jpg"
        if not image_path.exists():
            continue

        img_w, img_h = get_image_size(image_path)
        yolo_lines = []
        for line in ann_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split(",")
            left, top, w, h, score, category = (float(parts[0]), float(parts[1]), float(parts[2]),
                                                 float(parts[3]), int(parts[4]), int(parts[5]))
            if score == 0:
                stats["instances_ignored"] += 1
                continue
            if category not in VISDRONE_CLASS_MAP:
                unknown_categories.add(category)
                continue
            unified_name = VISDRONE_CLASS_MAP[category]
            if unified_name is None:
                stats["instances_dropped_class"] += 1
                continue
            yolo_line = to_yolo_line(NAME_TO_ID[unified_name], left, top, left + w, top + h, img_w, img_h)
            if yolo_line is None:
                stats["instances_degenerate"] += 1
                continue
            yolo_lines.append(yolo_line)
            stats["instances_kept"] += 1

        out_image_link = out_images / image_path.name
        if not out_image_link.exists():
            out_image_link.symlink_to(image_path.resolve())
        (out_labels / f"{stem}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
        stats["images"] += 1

    if unknown_categories:
        raise ValueError(
            f"Unknown VisDrone categories encountered (not in class_map.VISDRONE_CLASS_MAP): "
            f"{sorted(unknown_categories)}. Add them to class_map.py before re-running."
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/VisDrone2019"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/interim/visdrone"))
    args = parser.parse_args()

    if not args.raw_dir.is_dir():
        print(f"skip {args.raw_dir}: not found")
        return

    stats = convert(args.raw_dir, args.out_dir)
    print(
        f"done: {stats['images']} images, {stats['instances_kept']} instances kept, "
        f"{stats['instances_dropped_class']} dropped (out-of-scope class), "
        f"{stats['instances_ignored']} dropped (score=0/ignore), "
        f"{stats['instances_degenerate']} dropped (degenerate box)"
    )


if __name__ == "__main__":
    main()
