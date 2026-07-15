"""Combine tiled per-source datasets into the final train/val/test layout
and emit uav.yaml (project_plan.md §2 "Train/val/test split" + §5).

Splits by *source image* (the manifest's parent_image), never by tile,
so tiles from the same image can't straddle splits and leak. Groups
that contain at least one military_equipment instance are split
separately from the rest so the rare class is stratified roughly
70/15/15 across train/val/test too, rather than landing lopsided in
one split by chance.

Input: one or more --sources directories, each shaped like
tile_images.py's output (images/, labels/, manifest.csv with columns
tile, parent_image, source_dataset).

Output:
    data/processed/uav/images/{train,val,test}/<source>_<tile>
    data/processed/uav/labels/{train,val,test}/<source>_<tile>.txt
    data/processed/uav/split_manifest.csv  (final_name, source_dataset, parent_image, split)
    data/processed/uav/uav.yaml

Usage:
    python build_dataset.py --sources data/interim/dota_tiled/train data/interim/dota_tiled/val data/interim/vedai_tiled \
        --out-dir data/processed/uav
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

from class_map import NAME_TO_ID, UNIFIED_CLASSES

RARE_CLASS_ID = NAME_TO_ID["military_equipment"]


def load_manifest(source_dir: Path) -> list[dict]:
    manifest_path = source_dir / "manifest.csv"
    if manifest_path.exists():
        with manifest_path.open() as f:
            return list(csv.DictReader(f))
    # fallback for a source that wasn't run through tile_images.py
    images_dir = source_dir / "images"
    return [
        {"tile": p.name, "parent_image": p.name, "source_dataset": source_dir.name}
        for p in sorted(images_dir.iterdir())
    ]


def has_rare_class(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    for line in label_path.read_text().splitlines():
        if line.split() and int(line.split()[0]) == RARE_CLASS_ID:
            return True
    return False


def split_groups(group_keys: list, ratios: tuple[float, float, float], rng: random.Random) -> dict:
    """Shuffle group_keys and assign each to train/val/test by the given ratios."""
    keys = list(group_keys)
    rng.shuffle(keys)
    n = len(keys)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    assignment = {}
    for key in keys[:n_train]:
        assignment[key] = "train"
    for key in keys[n_train:n_train + n_val]:
        assignment[key] = "val"
    for key in keys[n_train + n_val:]:
        assignment[key] = "test"
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/uav"))
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise ValueError(f"--ratios must sum to 1.0, got {args.ratios}")

    rows = []
    for source_dir in args.sources:
        for row in load_manifest(source_dir):
            row["_source_dir"] = source_dir
            rows.append(row)

    # group by (source_dataset, parent_image); flag rare-class groups for separate stratified split
    groups: dict[tuple, list[dict]] = {}
    group_is_rare: dict[tuple, bool] = {}
    for row in rows:
        key = (row["source_dataset"], row["parent_image"])
        groups.setdefault(key, []).append(row)
        label_path = row["_source_dir"] / "labels" / f"{Path(row['tile']).stem}.txt"
        if has_rare_class(label_path):
            group_is_rare[key] = True

    rare_keys = [k for k in groups if group_is_rare.get(k)]
    common_keys = [k for k in groups if not group_is_rare.get(k)]

    rng = random.Random(args.seed)
    assignment = {}
    assignment.update(split_groups(rare_keys, tuple(args.ratios), rng))
    assignment.update(split_groups(common_keys, tuple(args.ratios), rng))

    for split in ("train", "val", "test"):
        (args.out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_out = []
    counts = {"train": 0, "val": 0, "test": 0}
    for key, group_rows in groups.items():
        split = assignment[key]
        for row in group_rows:
            source_dir = row["_source_dir"]
            tile_stem = Path(row["tile"]).stem
            tile_ext = Path(row["tile"]).suffix
            final_stem = f"{row['source_dataset']}_{tile_stem}"

            src_image = source_dir / "images" / row["tile"]
            src_label = source_dir / "labels" / f"{tile_stem}.txt"
            dst_image = args.out_dir / "images" / split / f"{final_stem}{tile_ext}"
            dst_label = args.out_dir / "labels" / split / f"{final_stem}.txt"

            shutil.copy2(src_image, dst_image)
            shutil.copy2(src_label, dst_label) if src_label.exists() else dst_label.write_text("")

            manifest_out.append({
                "final_name": dst_image.name,
                "source_dataset": row["source_dataset"],
                "parent_image": row["parent_image"],
                "split": split,
            })
            counts[split] += 1

    with (args.out_dir / "split_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["final_name", "source_dataset", "parent_image", "split"])
        writer.writeheader()
        writer.writerows(manifest_out)

    yaml_lines = [
        f"path: {args.out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ] + [f"  {idx}: {name}" for idx, name in UNIFIED_CLASSES.items()]
    (args.out_dir / "uav.yaml").write_text("\n".join(yaml_lines) + "\n")

    print(f"done: {counts} -> {args.out_dir}/uav.yaml")
    print(f"  {len(rare_keys)} source images contain military_equipment, {len(common_keys)} do not")


if __name__ == "__main__":
    main()
