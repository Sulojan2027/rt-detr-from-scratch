"""Convert detections to COCO result format and compute bbox mAP with pycocotools.

COCO results use the original COCO category id (1..90, via `dataset.cat_ids[label]`) and
[x, y, w, h] boxes in original-image pixels.
"""

from pycocotools.cocoeval import COCOeval
from torch import Tensor


def to_coco_detections(
    results: list[dict[str, Tensor]],
    image_ids: list[int],
    label_to_cat_id: list[int],
) -> list[dict]:
    """Per-image post-processed results (xyxy pixel boxes, 0..79 labels) → flat list of COCO result dicts."""
    detections = []
    for result, image_id in zip(results, image_ids, strict=True):
        for box, label, score in zip(result["boxes"].tolist(), result["labels"].tolist(), result["scores"].tolist()):
            x1, y1, x2, y2 = box
            detections.append(
                {
                    "image_id": int(image_id),
                    "category_id": label_to_cat_id[label],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )
    return detections


def evaluate_coco(coco_gt, detections: list[dict], image_ids: list[int] | None = None) -> list[float]:
    """Run COCOeval (bbox). Returns the 12 stats; stats[0] = AP@[.5:.95], stats[1] = AP50."""
    if not detections:
        raise ValueError("No detections to evaluate...")
    coco_dt = coco_gt.loadRes(detections)
    E = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if image_ids is not None:                 
        E.params.imgIds = sorted(image_ids)
    E.evaluate()
    E.accumulate()
    E.summarize()
    return list(E.stats)
