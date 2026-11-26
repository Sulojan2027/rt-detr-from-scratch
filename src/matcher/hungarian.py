"""Hungarian matcher: one-to-one assignment of ground-truth boxes to queries (DETR, arXiv:2005.12872 §3.1).

Mirrors HF `RTDetrHungarianMatcher` (`transformers/loss/loss_rt_detr.py`). Training only, no weights.
Cost per (query, GT) pair = 2 · class_cost + 5 · L1(cxcywh boxes) + 2 · (−GIoU).
The class cost is focal-style on sigmoid probabilities (alpha=0.25, gamma=2.0), not softmax.
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

    @torch.no_grad()
    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> list[tuple[Tensor, Tensor]]:
        """outputs: {"logits" [B, Q, C], "pred_boxes" [B, Q, 4] cxcywh}; targets: [{"class_labels" [N], "boxes" [N, 4]}]
        → per image (query_idx [K], gt_idx [K]) LongTensors, K = min(Q, N)."""
        batch_size, num_queries = outputs["logits"].shape[:2]

        # One cost matrix for the whole batch: rows = all B*Q queries, columns = all GT boxes of all images.
        probs = outputs["logits"].flatten(0, 1).sigmoid()  # [B*Q, C]
        boxes = outputs["pred_boxes"].flatten(0, 1)  # [B*Q, 4]
        tgt_labels = torch.cat([t["class_labels"] for t in targets])  # [sum N]
        tgt_boxes = torch.cat([t["boxes"] for t in targets])  # [sum N, 4]

        # Focal-style class cost.
        p = probs[:, tgt_labels]  # [B*Q, sum N]
        neg_cost = (1 - self.alpha) * p**self.gamma * -(1 - p + 1e-8).log()
        pos_cost = self.alpha * (1 - p) ** self.gamma * -(p + 1e-8).log()
        cost_class = pos_cost - neg_cost

        # L1 on cxcywh, and −GIoU on xyxy.
        cost_bbox = torch.cdist(boxes, tgt_boxes, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(tgt_boxes))

        cost = self.bbox_cost * cost_bbox + self.class_cost * cost_class + self.giou_cost * cost_giou

        # nan / inf would crash the solver; replace with the largest finite value.
        max_value = torch.finfo(cost.dtype).max
        cost = torch.nan_to_num(cost, nan=max_value, posinf=max_value, neginf=max_value)
        cost = cost.view(batch_size, num_queries, -1).cpu()

        # Solve per image: image i is matched only against its own GT columns.
        sizes = [len(t["boxes"]) for t in targets]
        matches = []
        for image_cost in (c[i] for i, c in enumerate(cost.split(sizes, dim=-1))):
            query_idx, gt_idx = linear_sum_assignment(image_cost)  # empty arrays when the image has no GT
            matches.append((torch.as_tensor(query_idx, dtype=torch.int64), torch.as_tensor(gt_idx, dtype=torch.int64)))
        return matches
