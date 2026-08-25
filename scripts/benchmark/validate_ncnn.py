"""Validate exported NCNN models on the held-out test split.

Unlike the speed benchmark's 250-frame sample, this uses every image in the
test split so mAP and per-class recall are comparable with Phase 2/3 PyTorch
results.
"""

import argparse
import json
from pathlib import Path


def evaluate(model_path: Path, data: Path, imgsz: int) -> dict:
    from ultralytics import YOLO

    metrics = YOLO(str(model_path)).val(
        data=str(data), split="test", imgsz=imgsz, plots=True, device="cpu"
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
    return {
        "model": str(model_path), "imgsz": imgsz,
        "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="+", required=True,
                        help="Pairs in MODEL_DIR:IMGSZ form")
    parser.add_argument("--data", type=Path, required=True,
                        help="Pi-local YAML containing train/val/test paths")
    parser.add_argument("--json", type=Path, default=Path("benchmarks/phase3/ncnn_accuracy.json"))
    parser.add_argument("--markdown", type=Path, default=Path("benchmarks/phase3/ncnn_accuracy.md"))
    args = parser.parse_args()

    configs = []
    for value in args.configs:
        model_text, size_text = value.rsplit(":", 1)
        configs.append((Path(model_text), int(size_text)))
    missing = [str(path) for path in [args.data, *(p for p, _ in configs)] if not path.exists()]
    if missing:
        raise SystemExit("Missing input(s):\n" + "\n".join(missing))

    results = [evaluate(model, args.data, size) for model, size in configs]
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=2) + "\n")

    lines = [
        "# Raspberry Pi NCNN accuracy",
        "",
        "Full held-out test-split validation of exported NCNN models.",
        "Compare with `docs/phase3_imgsz_results.md` from the PyTorch checkpoints.",
        "",
        "| Model | imgsz | mAP@0.5 | mAP@0.5:0.95 | military recall | person recall | vehicle recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        per_class = item["per_class"]
        recall = lambda name: per_class.get(name, {}).get("recall", float("nan"))
        lines.append(
            f"| {item['model']} | {item['imgsz']} | {item['map50']:.4f} | "
            f"{item['map50_95']:.4f} | {recall('military_equipment'):.4f} | "
            f"{recall('person'):.4f} | {recall('vehicle'):.4f} |"
        )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.markdown} and {args.json}")


if __name__ == "__main__":
    main()
