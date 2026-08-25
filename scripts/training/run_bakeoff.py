"""Phase 2 accuracy bake-off on identical data and training recipe.

The model is the only variable. This follows docs/model_selection.md:
60 epochs, early stopping disabled, fixed seed/determinism, and identical
aerial augmentations for every candidate.

Runs are sequential: a single 6 GB GPU cannot hold two trainings at
once. Each run gets its own folder under `runs/` plus a GPU-usage CSV
written by GpuLogger (step 2) alongside the standard TensorBoard events.

Usage:
    python run_bakeoff.py
    python run_bakeoff.py --models yolov8n.pt yolo10n.pt yolo11n.pt yolo26n.pt
    python run_bakeoff.py --epochs 3 --gpu-interval 2   # quick smoke test
"""

import argparse
from pathlib import Path

from log_gpu import GpuLogger

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "data/processed/uav/uav.yaml"
DEFAULT_MODELS = ["yolov8n.pt", "yolov10n.pt", "yolo11n.pt", "yolo26n.pt"]


def train_one(model_name: str, args: argparse.Namespace) -> None:
    from ultralytics import YOLO  # imported lazily so --help stays fast

    run_name = f"bakeoff_{Path(model_name).stem}"
    run_dir = args.project / run_name
    print(f"\n=== {model_name} -> {run_dir} ===")

    model = YOLO(model_name)
    # GpuLogger creates run_dir; Ultralytics reuses it (exist_ok=True) so
    # gpu_log.csv and the training outputs land in the same folder.
    with GpuLogger(run_dir, interval=args.gpu_interval):
        model.train(
            data=str(args.data),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            seed=0,
            deterministic=True,
            patience=0,
            copy_paste=0.3,
            degrees=180.0,
            flipud=0.5,
            cache=args.cache,
            project=str(args.project),
            name=run_name,
            exist_ok=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Model weights to bake off (default: {DEFAULT_MODELS})")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Dataset YAML (default: data/processed/uav/uav.yaml)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--cache", default="disk",
                        help="Ultralytics cache mode: ram / disk / False")
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs",
                        help="Parent folder for run outputs (default: runs/)")
    parser.add_argument("--gpu-interval", type=float, default=5.0,
                        help="Seconds between GPU stat polls (default: 5)")
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Dataset YAML not found: {args.data}")
    if str(args.cache).lower() in {"false", "none", "0"}:
        args.cache = False

    # Ultralytics resolves relative project paths below its configured
    # runs_dir (usually runs/detect). Resolve here so the GPU logger and
    # Ultralytics always write to the same directory.
    args.data = args.data.resolve()
    args.project = args.project.resolve()

    print(f"Bake-off: {args.models}  |  {args.epochs} epochs  |  imgsz={args.imgsz}")
    for model_name in args.models:
        train_one(model_name, args)
    print(f"\nDone. Runs under {args.project}/bakeoff_*  "
          f"(view: tensorboard --logdir {args.project})")


if __name__ == "__main__":
    main()
