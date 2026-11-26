"""Turn model outputs into final detections (scores, labels, xyxy pixel boxes) per image.

No NMS is needed: one-to-one Hungarian matching trains each object to a single query. Matches HF
`RTDetrImageProcessor.post_process_object_detection` (focal-loss branch: sigmoid, not softmax).
"""

import torch
from torch import Tensor

from src.utils.box_ops import box_cxcywh_to_xyxy


def topk_detections(
    logits: Tensor,
    pred_boxes: Tensor,
    target_sizes: Tensor,
    num_top: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Top-k over all (query, class) pairs; fixed-size outputs, so it exports to ONNX / TensorRT.

    logits [B, Q, C], pred_boxes [B, Q, 4] normalised cxcywh, target_sizes [B, 2] original (h, w)
    → scores [B, K], labels [B, K], boxes [B, K, 4] xyxy in original-image pixels.
    """
    boxes = box_cxcywh_to_xyxy(pred_boxes)
    img_h, img_w = target_sizes.unbind(1)
    scale = torch.stack([img_w, img_h, img_w, img_h], 1).to(boxes)      # sizes are (h, w), boxes are (x, y, x, y)
    boxes = boxes * scale[:, None, :]
    scores = logits.sigmoid()                                          # no softmax: no "no-object" class
    _, Q, C = scores.shape
    if num_top is None:
        num_top = Q
    scores, index = scores.flatten(1).topk(num_top, dim=-1)
    labels = index % C
    query = index // C
    boxes = boxes.gather(1, query.unsqueeze(-1).repeat(1, 1, 4))
    return scores, labels, boxes


def post_process(
    logits: Tensor,
    pred_boxes: Tensor,
    target_sizes: Tensor,
    threshold: float = 0.5,
    num_top: int | None = None,
) -> list[dict[str, Tensor]]:
    """[B, Q, C] logits + [B, Q, 4] cxcywh boxes → per-image dicts of scores / labels / xyxy pixel boxes."""
    scores, labels, boxes = topk_detections(logits, pred_boxes, target_sizes, num_top)
    results = []
    for score, label, box in zip(scores, labels, boxes, strict=True):
        keep = score > threshold
        results.append({"scores": score[keep], "labels": label[keep], "boxes": box[keep]})
    return results
