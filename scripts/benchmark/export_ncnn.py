"""Export trained finalist checkpoints to NCNN for Raspberry Pi testing.

Run this on the training/development machine. The resulting directories can
be copied to the Pi; no training dataset is needed for runtime benchmarking.
"""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", type=Path, required=True,
                        help="Trained .pt checkpoints to export")
    parser.add_argument("--imgsz", nargs="+", type=int,
                        default=[320, 416, 512, 640])
    parser.add_argument("--out-dir", type=Path, default=Path("models/ncnn"))
    args = parser.parse_args()

    from ultralytics import YOLO

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint in args.models:
        if not checkpoint.is_file():
            raise SystemExit(f"Checkpoint not found: {checkpoint}")
        for size in args.imgsz:
            model_label = (checkpoint.parent.parent.name or checkpoint.stem).removeprefix("bakeoff_")
            # Ultralytics detects NCNN by the `_ncnn_model` suffix.
            export_name = f"{model_label}_{size}_ncnn_model"
            print(f"Exporting {checkpoint} at imgsz={size}")
            # Ultralytics writes `<checkpoint-dir>/<stem>_ncnn_model` and
            # ignores project/name for this exporter. Copy each result to a
            # unique destination before exporting the next input size.
            exported = Path(YOLO(str(checkpoint)).export(
                format="ncnn", imgsz=size, device="cpu"
            ))
            destination = args.out_dir / export_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(exported, destination, dirs_exist_ok=True)
            print(f"  exported: {destination}")


if __name__ == "__main__":
    main()
