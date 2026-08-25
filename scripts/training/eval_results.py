"""Evaluate Phase 2 test checkpoints and write comparable metrics.

Usage:
    python scripts/training/eval_results.py
    python scripts/training/eval_results.py --weights runs/phase2/bakeoff_*/weights/best.pt
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "data/processed/uav/uav.yaml"
DEFAULT_WEIGHTS = [
    REPO_ROOT / "runs/runs/bakeoff_yolov8n/weights/best.pt",
    REPO_ROOT / "runs/runs/bakeoff_yolov10n/weights/best.pt",
    REPO_ROOT / "runs/runs/bakeoff_yolo11n/weights/best.pt",
    REPO_ROOT / "runs/runs/bakeoff_yolo26n/weights/best.pt",
]


def evaluate(weights: Path, data: Path, imgsz: int) -> dict:
    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(
        data=str(data), split="test", imgsz=imgsz, plots=True
    )
    names = metrics.names
    per_class = {}
    for index, class_id in enumerate(metrics.box.ap_class_index):
        class_id = int(class_id)
        result = metrics.box.class_result(index)
        per_class[names[class_id]] = {
            "precision": float(result[0]),
            "recall": float(result[1]),
            "ap50_95": float(result[2]),
            "f1": float(result[3]),
        }
    run_name = weights.parent.parent.name
    model_name = run_name.removeprefix("bakeoff_")
    return {
        "model": model_name or weights.stem,
        "weights": str(weights),
        "imgsz": imgsz,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class": per_class,
    }


def write_markdown(results: list[dict], output: Path) -> None:
    lines = [
        "# Phase 2 accuracy results",
        "",
        "Test split evaluation after the fixed 60-epoch bake-off. See each",
        "Ultralytics run directory for confusion matrices and PR curves.",
        "",
        "| Model | imgsz | mAP@0.5 | mAP@0.5:0.95 | military recall | person recall | vehicle recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        per_class = item["per_class"]
        lines.append(
            f"| {item['model']} | {item['imgsz']} | {item['map50']:.4f} | "
            f"{item['map50_95']:.4f} | "
            f"{per_class.get('military_equipment', {}).get('recall', float('nan')):.4f} | "
            f"{per_class.get('person', {}).get('recall', float('nan')):.4f} | "
            f"{per_class.get('vehicle', {}).get('recall', float('nan')):.4f} |"
        )
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--json", type=Path, default=REPO_ROOT / "docs/phase2_results.json")
    parser.add_argument("--markdown", type=Path, default=REPO_ROOT / "docs/phase2_results.md")
    args = parser.parse_args()
    missing = [str(path) for path in args.weights if not path.is_file()]
    if missing:
        raise SystemExit("Missing checkpoint(s):\n" + "\n".join(missing))
    if not args.data.is_file():
        raise SystemExit(f"Dataset YAML not found: {args.data}")
    results = [evaluate(path, args.data, args.imgsz) for path in args.weights]
    args.json.write_text(json.dumps(results, indent=2) + "\n")
    write_markdown(results, args.markdown)
    print(f"Wrote {args.markdown} and {args.json}")


if __name__ == "__main__":
    main()
