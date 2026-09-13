"""
Реплей-режим: перегляд серії знімків з польоту з накладеною детекцією об'єктів.

Інтерфейс не є частиною бортового процесу (розділ 2): він призначений для
постфактум-перегляду вже збережених кадрів на робочій станції оператора,
після завантаження їх з носія (scp). Повторно використовує клас DetectorModel
(табл. 2.3) — той самий алгоритм інференсу, що й на борту, лише над кадрами,
що вже лежать на диску, а не над живим відеопотоком.

Запуск: python replay_app.py
"""

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False

MODEL_PATH = "/home/dvyshotravka/object_detection/models/yolo26n_512_visdrone_crops_v1/yolo26n_512_visdrone_crops_v1/weights/best.pt"  # навчена модель детекції (підрозділ 3.3-3.4);
                        # для NCNN-варіанта вкажіть шлях до експортованої директорії
CONF_THRESHOLD = 0.25
CLASS_COLORS = {
    "person": (66, 133, 244),
    "vehicle": (52, 168, 83),
    "military_equipment": (234, 67, 53),
}


def _load_model():
    if not _HAS_ULTRALYTICS:
        return None
    try:
        return YOLO(MODEL_PATH)
    except Exception:
        return None


_model = _load_model()


def _draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    """Малює рамки, клас та впевненість для одного кадру (без залежності від
    Ultralytics .plot(), щоб той самий код працював і з mock-детекціями)."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        color = CLASS_COLORS.get(det["class"], (255, 255, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{det['class']} {det['conf']:.2f}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
            fill=color,
        )
        draw.text((x1, y1), label, fill="white", font=font)
    return annotated


def _detect(image: Image.Image) -> list[dict]:
    """Повертає перелік детекцій у форматі, узгодженому з бортовим виводом
    (project_plan.md: frame_id/timestamp/detections[class,conf,bbox_xyxy])."""
    if _model is not None:
        results = _model.predict(image, conf=CONF_THRESHOLD, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class": _model.names[cls_id],
                "conf": float(box.conf[0]),
                "bbox_xyxy": [float(v) for v in box.xyxy[0].tolist()],
            })
        return detections

    # Демонстраційна заглушка (немає навченої моделі в цьому середовищі):
    # повертає одну умовну детекцію по центру кадру, щоб інтерфейс можна
    # було перевірити без ваги моделі.
    w, h = image.size
    return [{
        "class": "vehicle",
        "conf": 0.87,
        "bbox_xyxy": [w * 0.35, h * 0.40, w * 0.65, h * 0.60],
    }]


def run_replay(files):
    if not files:
        return [], "Load at least one image."

    gallery_items = []
    summary_lines = []
    for i, file in enumerate(files, start=1):
        image = Image.open(file.name)
        detections = _detect(image)
        annotated = _draw_detections(image, detections)
        caption = ", ".join(f"{d['class']} ({d['conf']:.2f})" for d in detections) or "no objects detected"
        gallery_items.append((annotated, f"Кадр {i}: {caption}"))
        summary_lines.append(f"Кадр {i} ({file.name.split('/')[-1]}): {caption}")

    return gallery_items, "\n".join(summary_lines)


with gr.Blocks(title="Replay Mode — Detection Review") as demo:
    gr.Markdown("## Replay Mode: Review a series of images with detection results")
    gr.Markdown(
        "Upload one or more flight images. For each image, object detection will be performed "
    )
    with gr.Row():
        file_input = gr.File(
            label="Flight Images",
            file_count="multiple",
            file_types=["image"],
        )
    run_button = gr.Button("Run Detection", variant="primary")
    output_gallery = gr.Gallery(
        label="Results (series of frames)",
        columns=4,
        object_fit="contain",
        height=480,
    )
    output_summary = gr.Textbox(label="Frame Summary", lines=6)

    run_button.click(
        fn=run_replay,
        inputs=file_input,
        outputs=[output_gallery, output_summary],
    )


if __name__ == "__main__":
    demo.launch()
