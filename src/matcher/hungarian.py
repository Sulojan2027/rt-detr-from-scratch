"""Hungarian (bipartite) matcher: assign each ground-truth box to exactly one query.

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

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from src.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


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
        if class_cost == bbox_cost == giou_cost == 0:
            raise ValueError("All costs of the matcher can't be 0.")
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()  # matching only picks indices, nothing here needs gradients
    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> list[tuple[Tensor, Tensor]]:
        """outputs: {"logits" [B, Q, C], "pred_boxes" [B, Q, 4] cxcywh}; targets: [{"class_labels" [N], "boxes" [N, 4]}]
        → per image (query_idx [K], gt_idx [K]) LongTensors, K = min(Q, N)."""
        batch_size, num_queries = outputs["logits"].shape[:2]

        # Put the whole batch into one cost matrix: rows = all B*Q queries, columns = all GT boxes of all images.
        probs = outputs["logits"].flatten(0, 1).sigmoid()  # [B*Q, C]
        boxes = outputs["pred_boxes"].flatten(0, 1)  # [B*Q, 4]
        tgt_labels = torch.cat([t["class_labels"] for t in targets])  # [sum N]
        tgt_boxes = torch.cat([t["boxes"] for t in targets])  # [sum N, 4]

        # Class cost (focal-loss style): how bad is it to give this query that GT's class?
        p = probs[:, tgt_labels]  # [B*Q, sum N]
        neg_cost = (1 - self.alpha) * p**self.gamma * -(1 - p + 1e-8).log()
        pos_cost = self.alpha * (1 - p) ** self.gamma * -(p + 1e-8).log()
        cost_class = pos_cost - neg_cost

        # Box costs: L1 distance between (cx, cy, w, h), and minus the GIoU of the corner boxes.
        cost_bbox = torch.cdist(boxes, tgt_boxes, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(tgt_boxes))

        cost = self.bbox_cost * cost_bbox + self.class_cost * cost_class + self.giou_cost * cost_giou

        # A nan / inf would crash the solver. Give it the largest value instead, so such a pair is only
        # chosen when nothing better exists (a high training loss will show the problem).
        max_value = torch.finfo(cost.dtype).max
        cost = torch.nan_to_num(cost, nan=max_value, posinf=max_value, neginf=max_value)
        cost = cost.view(batch_size, num_queries, -1).cpu()

        # Image i may only be matched with its own GT boxes: split the columns by GT count and solve per image.
        sizes = [len(t["boxes"]) for t in targets]
        matches = []
        for image_cost in (c[i] for i, c in enumerate(cost.split(sizes, dim=-1))):
            query_idx, gt_idx = linear_sum_assignment(image_cost)  # empty arrays when the image has no GT
            matches.append((torch.as_tensor(query_idx, dtype=torch.int64), torch.as_tensor(gt_idx, dtype=torch.int64)))
        return matches
