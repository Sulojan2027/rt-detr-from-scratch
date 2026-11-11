import numpy as np
import pytest
import torch
from PIL import Image

from src.data.transforms import Resize, build_transforms


def _img(h, w, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8))


@pytest.mark.parametrize("hw", [(427, 640), (640, 480)])
def test_output_and_target(hw):
    boxes = torch.rand(4, 4)
    target = {"boxes": boxes.clone(), "orig_size": torch.tensor(hw), "size": torch.tensor(hw)}
    x, t = build_transforms()(_img(*hw), target)
    assert x.shape == (3, 640, 640) and x.dtype == torch.float32
    assert 0 <= x.min() and x.max() <= 1
    assert torch.equal(t["boxes"], boxes)
    assert t["size"].tolist() == [640, 640] and t["orig_size"].tolist() == list(hw)
    assert target["size"].tolist() == list(hw)          # caller's dict not mutated


def test_resize_int_and_tuple():
    img = torch.zeros(3, 100, 150, dtype=torch.uint8)
    assert Resize(64)(img, {})[0].shape == (3, 64, 64)
    assert Resize((32, 48))(img, {})[0].shape == (3, 32, 48)


@pytest.mark.slow
def test_matches_hf_processor():
    transformers = pytest.importorskip("transformers")
    proc = transformers.AutoImageProcessor.from_pretrained("PekingU/rtdetr_r18vd")
    for hw in [(427, 640), (640, 480), (333, 517)]:
        img = _img(*hw)
        x, _ = build_transforms()(img, {})
        assert (x - proc(images=img, return_tensors="pt").pixel_values[0]).abs().max().item() == 0.0
