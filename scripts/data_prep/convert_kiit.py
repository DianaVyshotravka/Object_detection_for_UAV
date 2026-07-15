"""Remap KIIT-MiTA's own YOLO labels to the unified 3-class taxonomy
(project_plan.md §2).

KIIT-MiTA ships already in YOLO txt format (class cx cy w h) with its
own 7-class KIIT-MiTA.yml (Artilary, Missile, Radar, M. Rocket
Launcher, Soldier, Tank, Vehicle), so no coordinate conversion is
needed -- just a class-id remap via class_map.KIIT_CLASS_MAP.

KIIT-MiTA.yml's own train/val/test paths are unreliable (val points at
/test/images and test at /valid/images -- looks like a typo in their
yaml), and this project re-splits everything itself anyway (see
build_dataset.py), so all three of the dataset's own folders are
flattened into one interim pool here rather than trusted as a split.

Expected raw layout:
    data/raw/kiit/{train,valid,test}/images/*.jpeg
    data/raw/kiit/{train,valid,test}/labels/*.txt

Writes remapped labels + an image symlink per image to:
    data/interim/kiit/images/<name>.jpeg   (symlink to raw image)
    data/interim/kiit/labels/<name>.txt

Usage:
    python convert_kiit.py --raw-dir data/raw/kiit --out-dir data/interim/kiit
"""

import argparse
import sys
from pathlib import Path

from class_map import KIIT_CLASS_MAP, NAME_TO_ID, UNIFIED_CLASSES

UNIFIED_ID_BY_KIIT_NAME = {name: NAME_TO_ID[name] for name in UNIFIED_CLASSES.values()}


def remap_label_file(path: Path) -> list[str]:
    lines = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        kiit_id = int(parts[0])
        if kiit_id not in KIIT_CLASS_MAP:
            raise ValueError(
                f"{path}: unknown KIIT class id {kiit_id} (not in class_map.KIIT_CLASS_MAP)"
            )
        unified_name = KIIT_CLASS_MAP[kiit_id]
        if unified_name is None:
            continue  # dropped class
        parts[0] = str(UNIFIED_ID_BY_KIIT_NAME[unified_name])
        lines.append(" ".join(parts))
    return lines


def convert_split(split_dir: Path, out_dir: Path) -> dict:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(f"  skip {split_dir}: missing images/ or labels/", file=sys.stderr)
        return {"images": 0, "instances_kept": 0, "instances_dropped_class": 0}

    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "instances_kept": 0}
    for image_path in sorted(images_dir.iterdir()):
        label_path = labels_dir / f"{image_path.stem}.txt"
        remapped = remap_label_file(label_path) if label_path.exists() else []

        out_image_link = out_images / image_path.name
        if not out_image_link.exists():
            out_image_link.symlink_to(image_path.resolve())
        (out_labels / f"{image_path.stem}.txt").write_text(
            "\n".join(remapped) + ("\n" if remapped else "")
        )
        stats["images"] += 1
        stats["instances_kept"] += len(remapped)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/kiit"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/interim/kiit"))
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = parser.parse_args()

    total = {"images": 0, "instances_kept": 0}
    for split in args.splits:
        split_dir = args.raw_dir / split
        if not split_dir.is_dir():
            print(f"skip {split_dir}: not found", file=sys.stderr)
            continue
        print(f"converting {split_dir} ...")
        stats = convert_split(split_dir, args.out_dir)
        for k in total:
            total[k] += stats.get(k, 0)

    print(f"done: {total['images']} images, {total['instances_kept']} instances kept")


if __name__ == "__main__":
    main()
