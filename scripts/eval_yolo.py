"""COCO val2017 AP for YOLOv8 (ultralytics), with the same metric code as scripts/eval.py."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root, so `import src` works
from src.data.coco import CocoDetection  # noqa: E402
from src.eval.coco_eval import evaluate_coco, to_coco_detections  # noqa: E402

STAT_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolov8s", help="yolov8s | yolov8m | yolov8l | yolov8x")
    p.add_argument("--weights-dir", default="weights", help="where the .pt files are downloaded (gitignored)")
    p.add_argument("--coco", default="data/coco", help="folder with annotations/ and val2017/")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001, help="NMS confidence threshold")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--half", action=argparse.BooleanOptionalAction, default=True, help="FP16 on GPU")
    p.add_argument("--limit", type=int, default=None, help="only the first N images (quick check)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="save the metrics as JSON here (e.g. results/yolov8l.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    half = args.half and device == "cuda"

    # Only for image paths / ids / category map / ground truth — YOLO does its own preprocessing.
    dataset = CocoDetection(Path(args.coco, "val2017"), Path(args.coco, "annotations/instances_val2017.json"))
    ids = dataset.ids[: args.limit] if args.limit is not None else dataset.ids
    paths = [str(Path(args.coco, "val2017", dataset.coco.imgs[i]["file_name"])) for i in ids]

    weights = Path(args.weights_dir, f"{args.model}.pt")
    weights.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))                       # downloads on first use

    # YOLO's 80 classes must be in the same order as our contiguous labels (sorted COCO category ids).
    coco_names = [dataset.coco.cats[c]["name"] for c in dataset.cat_ids]
    yolo_names = [model.names[i] for i in range(len(model.names))]
    assert yolo_names == coco_names, "YOLO class order differs from COCO's sorted category ids"

    detections = []
    t0 = time.perf_counter()
    for start in tqdm(range(0, len(paths), args.batch_size), desc=f"{args.model} conf={args.conf}"):
        batch_paths = paths[start : start + args.batch_size]
        batch_ids = ids[start : start + args.batch_size]
        preds = model.predict(batch_paths, imgsz=args.imgsz, conf=args.conf, iou=args.iou, max_det=args.max_det,
                              quantize=16 if half else 32, device=device, verbose=False)
        results = [{"boxes": r.boxes.xyxy.cpu(),               # already in ORIGINAL image pixels
                    "labels": r.boxes.cls.long().cpu(),
                    "scores": r.boxes.conf.cpu()} for r in preds]
        detections += to_coco_detections(results, batch_ids, dataset.cat_ids)
    elapsed = time.perf_counter() - t0

    stats = evaluate_coco(dataset.coco, detections, image_ids=list(ids))
    metrics = dict(zip(STAT_NAMES, (round(float(s), 4) for s in stats), strict=True))
    print(f"\n{args.model} (conf {args.conf}, iou {args.iou}) on {len(ids)} val2017 images "
          f"({len(ids) / elapsed:.1f} img/s incl. loading + NMS, {device}):")
    print(f"AP: {stats[0]:.3f}, AP50: {stats[1]:.3f}, AP75: {stats[2]:.3f}")
    print(f"APs: {stats[3]:.3f}, APm: {stats[4]:.3f}, APl: {stats[5]:.3f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "model": args.model, "conf": args.conf, "iou": args.iou, "max_det": args.max_det,
            "images": len(ids),
            "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
            "throughput_img_s": round(len(ids) / elapsed, 2),
            **metrics,
        }
        out_path.write_text(json.dumps(record, indent=2))
        print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
