"""Detections → COCO json results → mAP with pycocotools.  [SHARED — skeleton]

Flow:  post_process(threshold=0.0) per image  →  to_coco_detections  →  evaluate_coco  →  12 COCO stats.

COCO result format (one dict per box, pixels, ORIGINAL image size):
    {"image_id": 139, "category_id": 1, "bbox": [x, y, w, h], "score": 0.93}
    - category_id is the COCO id (1..90 with gaps), NOT our 0..79 label → map with dataset.cat_ids[label]
    - bbox is [x, y, w, h], NOT xyxy                                        → w = x2 - x1, h = y2 - y1

Self-checks once written:
    - Feed the GROUND TRUTH back in as detections (score 1.0) → AP = 1.0. Best way to test the mapping + bbox format.
    - HF R18 on full val2017 → AP ≈ 0.465 (paper Table 1: 46.5). R50 ≈ 0.531.
    - Our backbone + encoder inside HF's decoder (`encoder_outputs=`) → the same AP (to ~0.001).
"""

from torch import Tensor


def to_coco_detections(
    results: list[dict[str, Tensor]],
    image_ids: list[int],
    label_to_cat_id: list[int],
) -> list[dict]:
    """Per-image post-processed results (xyxy pixel boxes, 0..79 labels) → flat list of COCO result dicts."""
    # Hints:
    # 1. zip(results, image_ids); for each box: x1, y1, x2, y2 = box.tolist()
    # 2. "bbox": [x1, y1, x2 - x1, y2 - y1], "category_id": label_to_cat_id[int(label)], "score": float(score)
    # 3. Plain Python floats / ints only — json + pycocotools choke on tensors.
    raise NotImplementedError


def evaluate_coco(coco_gt, detections: list[dict], image_ids: list[int] | None = None) -> list[float]:
    """Run COCOeval (bbox). Returns the 12 stats; stats[0] = AP@[.5:.95], stats[1] = AP50."""
    # Hints:
    # 1. from pycocotools.cocoeval import COCOeval
    # 2. coco_dt = coco_gt.loadRes(detections)       (empty list crashes — guard it)
    # 3. E = COCOeval(coco_gt, coco_dt, iouType="bbox")
    # 4. If you only ran a subset: E.params.imgIds = image_ids   ← else missing images count as misses, AP drops.
    # 5. E.evaluate(); E.accumulate(); E.summarize(); return list(E.stats)
    raise NotImplementedError
