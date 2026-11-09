"""Model outputs → final detections per image.  [SHARED — skeleton]

RT-DETR needs **no NMS**: the Hungarian matcher trains each object to one query, so post-processing is only
"score, pick top-k, convert boxes".

HF reference: `RTDetrImageProcessor.post_process_object_detection` (use_focal_loss=True branch),
`transformers/models/rt_detr/image_processing_rt_detr.py`.

Inputs (from the decoder's last layer, HF `outputs.logits` / `outputs.pred_boxes`):
    logits      [B, Q, C]   raw class logits, Q = 300 queries, C = 80 classes (sigmoid, NOT softmax — no "no-object" class)
    pred_boxes  [B, Q, 4]   normalised (cx, cy, w, h)
    target_sizes [B, 2]     ORIGINAL image (height, width) — boxes must be scaled back to the original image, not 640×640

Output: list of B dicts {"scores": [K], "labels": [K] (0..79), "boxes": [K, 4] xyxy in pixels}.

Self-check once written (should be EXACT, diff 0.0):
    hf = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r18vd").eval()
    proc = AutoImageProcessor.from_pretrained("PekingU/rtdetr_r18vd")
    out = hf(**proc(images=img, return_tensors="pt"))
    ref = proc.post_process_object_detection(out, threshold=0.5, target_sizes=[img.size[::-1]])
    ours = post_process(out.logits, out.pred_boxes, torch.tensor([img.size[::-1]]), threshold=0.5)
    → same scores / labels / boxes (try the COCO cats image, val2017/000000039769.jpg).
"""

from torch import Tensor


def post_process(
    logits: Tensor,
    pred_boxes: Tensor,
    target_sizes: Tensor,
    threshold: float = 0.5,
    num_top: int | None = None,
) -> list[dict[str, Tensor]]:
    """[B, Q, C] logits + [B, Q, 4] cxcywh boxes → per-image dicts of scores / labels / xyxy pixel boxes."""
    # Hints:
    # 1. boxes = box_cxcywh_to_xyxy(pred_boxes)                       (src/utils/box_ops.py)
    # 2. Scale: img_h, img_w = target_sizes.unbind(1); scale = stack([img_w, img_h, img_w, img_h], 1) → [B, 4];
    #    boxes = boxes * scale[:, None, :]. Watch the order: sizes are (h, w) but boxes are (x, y, x, y).
    # 3. scores = logits.sigmoid()  → [B, Q, C]. One query can score for several classes.
    # 4. Top-k over the FLATTENED [B, Q*C] scores (k = num_top or Q):
    #       scores, index = scores.flatten(1).topk(k, dim=-1)
    #       labels = index % C          ← which class
    #       query  = index // C         ← which query (box)
    # 5. boxes = boxes.gather(1, query.unsqueeze(-1).repeat(1, 1, 4))
    # 6. Per image keep score > threshold. For COCO mAP use threshold=0.0 (mAP needs low-score detections too).
    raise NotImplementedError
