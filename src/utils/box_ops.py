"""Box helpers shared by query selection, denoising, matcher, losses and post-processing.  [SHARED]

Conventions (same as HF):
- Boxes are **normalised** to [0, 1] by image width / height.
- The model and the targets use **(cx, cy, w, h)** (centre format).
- IoU / GIoU and COCO use **corners (x1, y1, x2, y2)**; COCO json itself uses (x, y, w, h) in pixels.

HF reference: `transformers/image_transforms.py` (`center_to_corners_format`, `corners_to_center_format`),
`transformers/loss/loss_for_object_detection.py` (`box_iou`, `generalized_box_iou`),
`modeling_rt_detr.py::inverse_sigmoid`.

Self-checks once written:
    b = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
    box_cxcywh_to_xyxy(b)                         → [[0.4, 0.3, 0.6, 0.7]]
    box_xyxy_to_cxcywh(box_cxcywh_to_xyxy(b))     → b   (round trip)
    box_iou(xyxy, xyxy)[0]                        → [[1.0]]
    generalized_box_iou(far_apart_a, far_apart_b) → negative (→ -1 as they move apart)
    inverse_sigmoid(torch.sigmoid(x))             ≈ x   (for |x| < ~10)
"""

import torch
from torch import Tensor


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """[..., 4] (cx, cy, w, h) → (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """[..., 4] (x1, y1, x2, y2) → (cx, cy, w, h)."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = 0.5 * (x2 + x1)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)

def box_area(boxes: Tensor) -> Tensor:
    """[N, 4] xyxy → [N] areas."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return (x2 - x1) * (y2 - y1)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    """Pairwise IoU. [N, 4], [M, 4] xyxy → (iou [N, M], union [N, M]).  [optional — training only: matcher / loss]"""
    # Hints:
    # 1. Broadcast: boxes1[:, None, :2] is [N, 1, 2], boxes2[None, :, :2] is [1, M, 2].
    # 2. Intersection top-left = max of the two top-lefts; bottom-right = min of the two bottom-rights.
    # 3. wh = (rb - lt).clamp(min=0)   ← no overlap must give 0, not a negative area.
    # 4. union = area1[:, None] + area2[None, :] - inter. Return union too: GIoU needs it.
    raise NotImplementedError


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise GIoU in [-1, 1]. [N, 4], [M, 4] xyxy → [N, M].  [optional — training only: matcher / loss]"""
    # Hints:
    # 1. iou, union = box_iou(boxes1, boxes2)
    # 2. Enclosing box C: top-left = MIN of top-lefts, bottom-right = MAX of bottom-rights (opposite of step 2 above).
    # 3. giou = iou - (area_C - union) / area_C
    # 4. Optional guard (HF does it): raise if any box has x2 < x1 or y2 < y1 — catches cxcywh passed by mistake.
    raise NotImplementedError


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """logit(x) = log(x / (1 - x)), clamped so 0 and 1 don't give ±inf. Used for box refinement in logit space."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


__all__ = [
    "box_area",
    "box_cxcywh_to_xyxy",
    "box_iou",
    "box_xyxy_to_cxcywh",
    "generalized_box_iou",
    "inverse_sigmoid",
]
