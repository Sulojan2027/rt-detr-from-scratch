"""Box format conversions, IoU / GIoU and inverse sigmoid (same conventions as HF).

Model outputs and targets are normalised (cx, cy, w, h); IoU / GIoU take corners (x1, y1, x2, y2).
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
    """Pairwise IoU. [N, 4], [M, 4] xyxy → (iou [N, M], union [N, M])."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # Overlap rectangle of every pair: the larger top-left corner and the smaller bottom-right corner.
    top_left = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    bottom_right = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    # Boxes that do not overlap would give a negative size, so clamp it to 0.
    overlap_wh = (bottom_right - top_left).clamp(min=0)
    intersection = overlap_wh[..., 0] * overlap_wh[..., 1]  # [N, M]

    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise GIoU in [-1, 1]. [N, 4], [M, 4] xyxy → [N, M]."""
    # A box with x2 < x1 usually means (cx, cy, w, h) was passed by mistake; fail loudly instead of giving nan.
    for name, boxes in (("boxes1", boxes1), ("boxes2", boxes2)):
        if not (boxes[:, 2:] >= boxes[:, :2]).all():
            raise ValueError(f"{name} must be in (x1, y1, x2, y2) corner format with x2 >= x1 and y2 >= y1.")

    iou, union = box_iou(boxes1, boxes2)

    # Smallest box that contains both boxes: the opposite corners from the IoU overlap above.
    top_left = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing_wh = (bottom_right - top_left).clamp(min=0)
    enclosing_area = enclosing_wh[..., 0] * enclosing_wh[..., 1]

    # GIoU = IoU minus the share of the enclosing box that is empty (neither box covers it).
    return iou - (enclosing_area - union) / enclosing_area


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
