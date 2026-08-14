"""Build the fixed frame set used by the Phase 1 speed screening
(model_selection.md Phase 1).

Every model x imgsz config must be timed on byte-identical input, and
that input has to survive an rsync to the Pi -- so this copies real files
out of `data/processed/uav` instead of symlinking (data/interim images are
symlinks into data/raw, which would not follow).

Sampling is stratified across source datasets by default so the frame set
is not accidentally all VEDAI (small, uniform images) or all VisDrone
tiles; image dimensions differ per source and pre-processing cost with
them.

Usage:
    python make_bench_frames.py
    python make_bench_frames.py --split val --n 400
    python make_bench_frames.py --n 20 --out-dir /tmp/frames_smoke
"""

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# scripts/data_prep is a flat script dir, not an installed package -- same
# import style as scripts/training/run_bakeoff.py importing log_gpu.
sys.path.insert(0, str(REPO_ROOT / "scripts/data_prep"))
from common import get_image_size  # noqa: E402

DEFAULT_UAV_DIR = REPO_ROOT / "data/processed/uav"
DEFAULT_OUT_DIR = REPO_ROOT / "data/benchmark/frames"
MANIFEST_NAME = "frames_manifest.csv"


def read_split_manifest(uav_dir: Path, split: str) -> list[dict]:
    """Rows of split_manifest.csv belonging to `split`."""
    path = uav_dir / "split_manifest.csv"
    if not path.is_file():
        raise SystemExit(f"Split manifest not found: {path}")
    with path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    if not rows:
        raise SystemExit(f"No rows for split '{split}' in {path}")
    return rows


def sample_stratified(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Pick `n` rows, allocated across source_dataset in proportion to size.

    Largest-remainder allocation, then a deterministic shuffle of the
    result so the sources are interleaved rather than grouped.
    """
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source_dataset"]].append(row)

    total = len(rows)
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for source, source_rows in sorted(by_source.items()):
        exact = n * len(source_rows) / total
        quotas[source] = min(len(source_rows), int(exact))
        remainders.append((exact - int(exact), source))

    # Hand out the leftover slots to the largest fractional parts.
    assigned = sum(quotas.values())
    for _, source in sorted(remainders, reverse=True):
        if assigned >= n:
            break
        if quotas[source] < len(by_source[source]):
            quotas[source] += 1
            assigned += 1

    picked: list[dict] = []
    for source, source_rows in sorted(by_source.items()):
        picked.extend(rng.sample(source_rows, quotas[source]))
    rng.shuffle(picked)
    return picked


def sample_flat(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--uav-dir", type=Path, default=DEFAULT_UAV_DIR,
                        help="Processed dataset root (default: data/processed/uav)")
    parser.add_argument("--split", default="test",
                        help="Split to sample from (default: test)")
    parser.add_argument("--n", type=int, default=250,
                        help="Frames to copy; >=200 per model_selection.md (default: 250)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Destination for the copied frames (default: data/benchmark/frames)")
    parser.add_argument("--no-stratify", action="store_true",
                        help="Sample uniformly instead of proportionally per source dataset")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the per-source allocation without copying")
    args = parser.parse_args()

    if args.n < 200:
        print(f"warning: --n {args.n} is below the >=200 frames "
              f"model_selection.md asks for; fine for a smoke test only")

    rows = read_split_manifest(args.uav_dir, args.split)
    picked = (sample_flat if args.no_stratify else sample_stratified)(
        rows, args.n, args.seed
    )

    counts: dict[str, int] = defaultdict(int)
    for row in picked:
        counts[row["source_dataset"]] += 1
    print(f"{args.split}: {len(rows)} available -> sampling {len(picked)}")
    for source, count in sorted(counts.items()):
        print(f"  {source:10s} {count:4d}")
    if args.dry_run:
        return

    images_dir = args.uav_dir / "images" / args.split
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.out_dir.iterdir():
        if stale.is_file():
            stale.unlink()

    manifest_rows = []
    for row in picked:
        src = images_dir / row["final_name"]
        if not src.is_file():
            raise SystemExit(f"Image listed in manifest is missing: {src}")
        shutil.copy2(src, args.out_dir / src.name)
        width, height = get_image_size(src)
        manifest_rows.append({
            "filename": src.name,
            "source_dataset": row["source_dataset"],
            "parent_image": row["parent_image"],
            "width": width,
            "height": height,
        })

    manifest_path = args.out_dir / MANIFEST_NAME
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Copied {len(manifest_rows)} frames -> {args.out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
