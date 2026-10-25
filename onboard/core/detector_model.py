"""DetectorModel -- NCNN inference wrapper.

Same call the Phase 3 benchmark measured at 23.4 FPS on a Pi 5
(scripts/benchmark/bench_ncnn.py:38-40), so onboard throughput matches the
number already reported in docs/phase3_results.md.
"""

from pathlib import Path

import numpy as np

from .types import Detection


class DetectorModel:
    def __init__(self, model_path: Path, imgsz: int = 512, conf: float = 0.25,
                 iou: float = 0.7, logger=None, threads: int = None):
        from ultralytics import YOLO  # heavy import, kept out of module load

        self.model = YOLO(str(model_path))  # a *_ncnn_model directory, or a .pt
        self.predict_args = dict(imgsz=imgsz, device="cpu", conf=conf, iou=iou,
                                 verbose=False)
        # First NCNN predict is seconds slower than the rest; burn it here
        # rather than on the first real frame of the flight.
        warmup = np.zeros((imgsz, imgsz, 3), np.uint8)
        self.model.predict(warmup, **self.predict_args)
        net = getattr(getattr(self.model.predictor.model, "backend", None), "net", None)
        if threads and net is not None:
            # NCNN on all 4 cores stalls whenever the camera or saver thread
            # takes one: 21 -> 10 FPS measured on a Pi 5. Convolutions read the
            # thread count at load time, so reload after setting it.
            param = next(Path(model_path).glob("*.param"))
            net.clear()
            net.opt.num_threads = threads
            net.load_param(str(param))
            net.load_model(str(param.with_suffix(".bin")))
            self.model.predict(warmup, **self.predict_args)
        if logger is not None:
            logger.log_info(f"detector: loaded {model_path} at imgsz={imgsz}"
                            + (f", {threads} threads" if threads and net is not None else ""))

    def infer(self, image) -> list:
        result = self.model.predict(image, **self.predict_args)[0]
        return [
            Detection(
                cls_name=self.model.names[int(box.cls[0])],
                conf=float(box.conf[0]),
                bbox_xyxy=[float(v) for v in box.xyxy[0].tolist()],
            )
            for box in result.boxes
        ]
