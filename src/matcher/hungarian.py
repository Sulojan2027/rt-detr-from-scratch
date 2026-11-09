"""Hungarian (bipartite) matcher: assign each ground-truth box to exactly one query.  [PERSON B — skeleton]

Paper: RT-DETR §4 (inherits DETR's set prediction, arXiv:2005.12872 §3.1). HF: `RTDetrHungarianMatcher`
in `transformers/loss/loss_rt_detr.py`. Training only — no weights, nothing to load.

Why: the model outputs 300 boxes but an image has e.g. 5 objects. The matcher picks the cheapest one-to-one
pairing (5 queries ↔ 5 GTs); those 5 get box + class losses, the other 295 learn "no object". This is what
makes NMS unnecessary at inference.

Cost per (query, GT) pair = 2 · class_cost + 5 · L1(boxes) + 2 · (−GIoU)   (hf.config.matcher_* values)
Class cost is focal-style (matcher_alpha=0.25, matcher_gamma=2.0) on sigmoid probabilities, not softmax.

Self-check: give the same random outputs/targets to this and to HF's matcher
    from transformers.loss.loss_rt_detr import RTDetrHungarianMatcher; ref = RTDetrHungarianMatcher(hf.config)
→ identical index lists. Also: a GT box that equals query 7's box exactly must be matched to query 7.
"""

import torch.nn as nn
from torch import Tensor


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        class_cost: float = 2.0,
        bbox_cost: float = 5.0,
        giou_cost: float = 2.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        # Hint: store the five numbers; raise if all three costs are 0.
        raise NotImplementedError

    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> list[tuple[Tensor, Tensor]]:
        """outputs: {"logits" [B, Q, C], "pred_boxes" [B, Q, 4] cxcywh}; targets: [{"class_labels" [N], "boxes" [N, 4]}]
        → per image (query_idx [K], gt_idx [K]) LongTensors, K = min(Q, N)."""
        # Hints (decorate with @torch.no_grad() — matching is not differentiated):
        # 1. Flatten the batch: prob = logits.flatten(0, 1).sigmoid() [B*Q, C]; boxes [B*Q, 4]
        #    tgt_ids = cat(all class_labels) [sum N]; tgt_boxes = cat(all boxes) [sum N, 4]
        # 2. p = prob[:, tgt_ids]                                            [B*Q, sum N]
        #    neg = (1 - alpha) * p**gamma * -log(1 - p + 1e-8)
        #    pos = alpha * (1 - p)**gamma * -log(p + 1e-8)
        #    cost_class = pos - neg
        # 3. cost_bbox = torch.cdist(boxes, tgt_boxes, p=1)
        #    cost_giou = -generalized_box_iou(cxcywh_to_xyxy(boxes), cxcywh_to_xyxy(tgt_boxes))
        # 4. C = (bbox_cost*cost_bbox + class_cost*cost_class + giou_cost*cost_giou).view(B, Q, -1).cpu()
        # 5. Split the last dim by per-image GT counts; image i uses block i only:
        #       scipy.optimize.linear_sum_assignment(c[i])  for i, c in enumerate(C.split(sizes, -1))
        # 6. Images with 0 GTs → (empty, empty) int64 tensors.
        raise NotImplementedError
