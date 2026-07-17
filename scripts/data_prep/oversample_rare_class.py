"""Oversample train images containing a rare class (project_plan.md §5,
"Handling class imbalance", step 1).

EDA showed military_equipment at ~4% of instances. Duplicating every
train image that contains the rare class raises its per-epoch frequency
without touching val/test or the loss function. Duplicates are physical
copies named `<stem>_dup<N>.<ext>` so the standard Ultralytics loader
picks them up with zero config.

Usage:
    python oversample_rare_class.py --uav-dir data/processed/uav --factor 2

`--factor N` means each rare-class image appears N times total in train
(original + N-1 copies). Re-running is safe: existing `_dup` files are
ignored when scanning and never duplicated again, but the script refuses
to overwrite a duplicate that already exists (use a fresh dataset build
or remove old dups to change the factor).

If split_manifest.csv exists, duplicate rows are appended with the same
source_dataset/parent_image so the notebook's leakage check stays valid.
"""

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

from class_map import UNIFIED_CLASSES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DUP_MARKER = "_dup"


def class_counts(labels_dir: Path) -> Counter:
    counts: Counter = Counter()
    for label_path in labels_dir.glob("*.txt"):
        for line in label_path.read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    return counts


def print_counts(title: str, counts: Counter) -> None:
    total = sum(counts.values()) or 1
    print(title)
    for cls_id, name in UNIFIED_CLASSES.items():
        n = counts.get(cls_id, 0)
        print(f"  {name:<20} {n:>7,}  ({n / total:6.2%})")


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--uav-dir", type=Path, default=Path("data/processed/uav"))
    parser.add_argument("--split", default="train", help="split to oversample (train only, unless you know better)")
    parser.add_argument("--class-id", type=int, default=2, help="rare class id (default: 2 = military_equipment)")
    parser.add_argument("--factor", type=int, default=2, help="total appearances per rare-class image (original + factor-1 copies)")
    parser.add_argument("--dry-run", action="store_true", help="report what would be copied without writing anything")
    args = parser.parse_args()

    if args.factor < 2:
        parser.error("--factor must be >= 2 (1 would be a no-op)")

    labels_dir = args.uav_dir / "labels" / args.split
    images_dir = args.uav_dir / "images" / args.split
    if not labels_dir.is_dir() or not images_dir.is_dir():
        raise SystemExit(f"Missing images/labels dirs under {args.uav_dir} for split '{args.split}'")

    before = class_counts(labels_dir)
    print_counts(f"Before ({args.split}):", before)

    rare_stems = [
        p.stem
        for p in sorted(labels_dir.glob("*.txt"))
        if DUP_MARKER not in p.stem
        and any(line.split()[0] == str(args.class_id) for line in p.read_text().splitlines() if line.strip())
    ]
    print(f"\nImages containing class {args.class_id} ({UNIFIED_CLASSES.get(args.class_id, '?')}): {len(rare_stems):,}")

    copied = 0
    new_manifest_rows: list[dict[str, str]] = []
    manifest_path = args.uav_dir / "split_manifest.csv"
    manifest_by_name: dict[str, dict[str, str]] = {}
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest_by_name = {row["final_name"]: row for row in csv.DictReader(f)}

    for stem in rare_stems:
        image_path = find_image(images_dir, stem)
        if image_path is None:
            print(f"  WARNING: no image found for label {stem}.txt; skipped")
            continue
        for i in range(1, args.factor):
            dup_stem = f"{stem}{DUP_MARKER}{i}"
            dup_image = images_dir / f"{dup_stem}{image_path.suffix}"
            dup_label = labels_dir / f"{dup_stem}.txt"
            if dup_image.exists() or dup_label.exists():
                raise SystemExit(
                    f"Duplicate already exists: {dup_image.name}. "
                    "Remove old _dup files (or rebuild the dataset) before re-running."
                )
            if not args.dry_run:
                shutil.copy2(image_path, dup_image)
                shutil.copy2(labels_dir / f"{stem}.txt", dup_label)
            copied += 1
            source_row = manifest_by_name.get(image_path.name)
            if source_row:
                new_manifest_rows.append({**source_row, "final_name": dup_image.name})

    if manifest_by_name and new_manifest_rows and not args.dry_run:
        with manifest_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["final_name", "source_dataset", "parent_image", "split"])
            writer.writerows(new_manifest_rows)
        print(f"Appended {len(new_manifest_rows):,} rows to {manifest_path.name}")
    elif manifest_by_name and copied and len(new_manifest_rows) < copied:
        print(f"  WARNING: {copied - len(new_manifest_rows):,} duplicates had no manifest row to copy")

    verb = "Would copy" if args.dry_run else "Copied"
    print(f"{verb} {copied:,} image+label pairs (factor={args.factor})\n")
    if not args.dry_run:
        print_counts(f"After ({args.split}):", class_counts(labels_dir))


if __name__ == "__main__":
    main()
