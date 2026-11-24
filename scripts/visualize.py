"""Side-by-side detections for the demo, [ground truth |] RT-DETR | YOLOv8, on the same images and score threshold."""

import argparse
import colorsys
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root, so `import src` works
from src.data.transforms import build_transforms  # noqa: E402
from src.model import RTDETR  # noqa: E402
from src.postprocess import post_process  # noqa: E402

Detections = tuple[list[list[float]], list[int], list[float]]      # boxes xyxy (original pixels), labels, scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rtdetr", default="rtdetr_r50", help="rtdetr_r18 | rtdetr_r50")
    p.add_argument("--yolo", default="yolov8l", help="yolov8s | yolov8m | yolov8l | yolov8x")
    p.add_argument("--coco", default="data/coco")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--num", type=int, default=8, help="first N val2017 images")
    src.add_argument("--ids", type=int, nargs="+", help="COCO val2017 image ids")
    src.add_argument("--images", nargs="+", help="arbitrary image files (no ground truth)")
    p.add_argument("--gt", action="store_true", help="add a ground-truth panel (COCO images only)")
    p.add_argument("--threshold", type=float, default=0.5, help="score threshold, the same for both models")
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--out", default="results/vis")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ------------------------------------------------------------------------------------------- models ----


def rtdetr_detector(name: str, size: int, threshold: float, device: str):
    """→ fn(PIL image) -> Detections, with our whole model (RTDETR.from_hf)."""
    model = RTDETR.from_hf(int(name.split("_r")[1])).to(device)
    transforms = build_transforms(size)

    @torch.inference_mode()
    def detect(img: Image.Image) -> Detections:
        x, _ = transforms(img, {})
        out = model(x[None].to(device))
        sizes = torch.tensor([[img.height, img.width]], device=device)
        r = post_process(out["logits"], out["pred_boxes"], sizes, threshold=threshold)[0]
        return r["boxes"].tolist(), r["labels"].tolist(), r["scores"].tolist()

    return detect


def yolo_detector(name: str, weights_dir: str, size: int, threshold: float, device: str):
    """→ (fn(image path) -> Detections, class names), with ultralytics predict (its letterbox + NMS)."""
    from ultralytics import YOLO

    weights = Path(weights_dir, f"{name}.pt")
    weights.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))

    def detect(path: str) -> Detections:
        r = model.predict(path, imgsz=size, conf=threshold, iou=0.7, max_det=300, device=device, verbose=False)[0]
        return r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()

    return detect, [model.names[i] for i in range(len(model.names))]   # same order as COCO's sorted category ids


# ------------------------------------------------------------------------------------------ drawing ----


def class_color(label: int) -> tuple[int, int, int]:
    """Stable, well-separated colour per class (golden-ratio hue steps)."""
    r, g, b = colorsys.hsv_to_rgb((label * 0.618034) % 1.0, 0.75, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_panel(img: Image.Image, det: Detections, names: list[str], title: str) -> Image.Image:
    """Copy of `img` with boxes + "name score" labels and a title bar on top."""
    w, h = img.size
    scale = max(w, h) / 640
    font = ImageFont.load_default(size=max(12, round(14 * scale)))
    title_font = ImageFont.load_default(size=max(16, round(20 * scale)))
    line = max(2, round(2 * scale))

    canvas = img.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for (x1, y1, x2, y2), label, score in sorted(zip(*det, strict=True), key=lambda d: d[2] or 0):   # best drawn last = on top
        color = class_color(label)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line)
        text = names[label] if score is None else f"{names[label]} {score:.2f}"
        tx1, ty1, tx2, ty2 = draw.textbbox((x1, y1), text, font=font)
        ty = y1 - (ty2 - ty1) - 2 * line if y1 - (ty2 - ty1) - 2 * line >= 0 else y1   # above the box if it fits
        draw.rectangle((x1, ty, x1 + (tx2 - tx1) + 2 * line, ty + (ty2 - ty1) + 2 * line), fill=color)
        draw.text((x1 + line, ty + line - (ty1 - y1)), text, fill=(0, 0, 0), font=font)

    bar = round(36 * scale)
    panel = Image.new("RGB", (w, h + bar), (255, 255, 255))
    panel.paste(canvas, (0, bar))
    ImageDraw.Draw(panel).text((round(8 * scale), bar // 2), title, fill=(0, 0, 0), font=title_font, anchor="lm")
    return panel


def side_by_side(panels: list[Image.Image], gap: int = 8) -> Image.Image:
    w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    out = Image.new("RGB", (w, max(p.height for p in panels)), (255, 255, 255))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + gap
    return out


# --------------------------------------------------------------------------------------------- main ----


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # Images: (path, COCO image id or None)
    coco = None
    if args.images:
        items = [(p, None) for p in args.images]
    else:
        from pycocotools.coco import COCO

        coco = COCO(str(Path(args.coco, "annotations/instances_val2017.json")))
        ids = args.ids or sorted(coco.getImgIds())[: args.num]
        items = [(str(Path(args.coco, "val2017", coco.imgs[i]["file_name"])), i) for i in ids]
    if args.gt and coco is None:
        raise SystemExit("--gt needs COCO images (--num or --ids), not --images")

    rtdetr = rtdetr_detector(args.rtdetr, args.size, args.threshold, device)
    yolo, names = yolo_detector(args.yolo, args.weights_dir, args.size, args.threshold, device)
    rt_title = args.rtdetr.replace("rtdetr_r", "RT-DETR-R")              # rtdetr_r50 → RT-DETR-R50
    yolo_title = f"YOLOv8-{args.yolo.removeprefix('yolov8').upper()}"    # yolov8l   → YOLOv8-L
    cat_ids = sorted(coco.getCatIds()) if coco else []

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, image_id in items:
        img = Image.open(path).convert("RGB")
        panels = []
        if args.gt:
            anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id, iscrowd=False))
            gt = ([[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in anns)],
                  [cat_ids.index(a["category_id"]) for a in anns], [None] * len(anns))
            panels.append(draw_panel(img, gt, names, f"Ground truth ({len(anns)})"))
        for title, det in ((rt_title, rtdetr(img)), (yolo_title, yolo(path))):
            panels.append(draw_panel(img, det, names, f"{title}  ({len(det[0])} boxes, score > {args.threshold})"))
        stem = f"{image_id:012d}" if image_id is not None else Path(path).stem
        out_path = out_dir / f"{stem}_{args.rtdetr}_vs_{args.yolo}.jpg"
        side_by_side(panels).save(out_path, quality=92)
        print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
