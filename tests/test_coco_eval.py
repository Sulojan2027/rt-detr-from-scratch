import pytest
import torch

pytest.importorskip("pycocotools")
from src.eval.coco_eval import to_coco_detections  # noqa: E402


def test_to_coco_detections_format():
    results = [
        {"boxes": torch.tensor([[10.0, 20.0, 50.0, 80.0], [0.0, 0.0, 5.0, 5.0]]),
         "labels": torch.tensor([0, 11]), "scores": torch.tensor([0.9, 0.1])},
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long), "scores": torch.zeros(0)},
    ]
    cat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]          # COCO ids skip 12
    dets = to_coco_detections(results, [139, 285], cat_ids)
    assert len(dets) == 2                                     # empty image adds nothing
    assert dets[0] == {"image_id": 139, "category_id": 1, "bbox": [10.0, 20.0, 40.0, 60.0], "score": pytest.approx(0.9)}
    assert dets[1]["category_id"] == 13                       # label 11 → id 13 (gap at 12)
    assert all(type(v) is float for v in dets[0]["bbox"])
