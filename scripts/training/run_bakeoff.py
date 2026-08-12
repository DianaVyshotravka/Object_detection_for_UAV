"""Baseline bake-off: finetune YOLO11n and YOLO26n on identical recipe
and data (finetuning_high_level.md step 3, project_plan.md §3).

The model is the ONLY variable. Both runs use Ultralytics default
hyperparameters and augmentation on purpose -- this is the §3 baseline.
The §5 aerial recipe (heavy augmentation, 200 epochs, oversampling) is
applied later to the winner only, not here.

Runs are sequential: a single 6 GB GPU cannot hold two trainings at
once. Each run gets its own folder under `runs/` plus a GPU-usage CSV
written by GpuLogger (step 2) alongside the standard TensorBoard events.

Usage:
    python run_bakeoff.py
    python run_bakeoff.py --models yolo11n.pt yolo26n.pt --epochs 50
    python run_bakeoff.py --epochs 3 --gpu-interval 2   # quick smoke test
"""

import argparse
from pathlib import Path

from log_gpu import GpuLogger

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "data/processed/uav/uav.yaml"
DEFAULT_MODELS = ["yolo11n.pt", "yolo26n.pt"]


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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1,
                        help="-1 auto-sizes batch for available VRAM")
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

    print(f"Bake-off: {args.models}  |  {args.epochs} epochs  |  imgsz={args.imgsz}")
    for model_name in args.models:
        train_one(model_name, args)
    print(f"\nDone. Runs under {args.project}/bakeoff_*  "
          f"(view: tensorboard --logdir {args.project})")


if __name__ == "__main__":
    main()
