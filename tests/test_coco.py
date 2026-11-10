import copy
from pathlib import Path

import pytest
import torch

COCO_ROOT = Path(__file__).resolve().parent.parent / "data/coco"
ANN = COCO_ROOT / "annotations/instances_val2017.json"
pytestmark = pytest.mark.skipif(not ANN.exists(), reason="COCO val2017 not downloaded (see README → Data)")


@pytest.fixture(scope="module")
def ds():
    pytest.importorskip("pycocotools")
    from src.data.coco import CocoDetection
    from src.data.transforms import build_transforms

    return CocoDetection(COCO_ROOT / "val2017", ANN, build_transforms())


def test_size_and_categories(ds):
    assert len(ds) == 5000
    assert len(ds.cat_ids) == 80 and ds.cat_ids[0] == 1 and ds.cat_ids[-1] == 90


def test_first_item(ds):
    img, t = ds[0]
    assert img.shape == (3, 640, 640) and img.dtype == torch.float32
    assert t["image_id"] == 139 and t["orig_size"].tolist() == [426, 640]
    assert ((t["boxes"] >= 0) & (t["boxes"] <= 1)).all() and len(t["boxes"]) == len(t["class_labels"])


def test_empty_image_and_gt_untouched(ds):
    before = copy.deepcopy(ds.coco.dataset["annotations"][:500])
    empty = next(i for i, iid in enumerate(ds.ids) if not ds.coco.getAnnIds(imgIds=iid))
    _, t = ds[empty]
    assert t["boxes"].shape == (0, 4) and t["class_labels"].shape == (0,)
    for i in range(5):
        ds[i]
    assert ds.coco.dataset["annotations"][:500] == before


def test_collate(ds):
    from src.data.coco import collate_fn

    images, targets = collate_fn([ds[0], ds[1]])
    assert images.shape == (2, 3, 640, 640) and isinstance(targets, list) and len(targets) == 2
