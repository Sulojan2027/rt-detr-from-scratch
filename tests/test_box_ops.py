import pytest
import torch

from src.utils.box_ops import (
    box_area,
    box_cxcywh_to_xyxy,
    box_iou,
    box_xyxy_to_cxcywh,
    generalized_box_iou,
    inverse_sigmoid,
)


def test_cxcywh_to_xyxy_known_value():
    b = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
    assert torch.allclose(box_cxcywh_to_xyxy(b), torch.tensor([[0.4, 0.3, 0.6, 0.7]]))
    assert torch.allclose(box_area(box_cxcywh_to_xyxy(b)), torch.tensor([0.08]))


@pytest.mark.parametrize("shape", [(4,), (300, 4), (2, 300, 4)])
def test_round_trip_any_leading_dims(shape):
    x = torch.rand(shape)
    xyxy = box_cxcywh_to_xyxy(x)
    assert xyxy.shape == x.shape
    assert torch.allclose(box_xyxy_to_cxcywh(xyxy), x, atol=1e-6)


def test_inverse_sigmoid():
    x = torch.linspace(-8, 8, 101)
    assert torch.allclose(inverse_sigmoid(torch.sigmoid(x)), x, atol=1e-3)
    edge = inverse_sigmoid(torch.tensor([0.0, 1.0, -0.5, 1.5]))
    assert torch.isfinite(edge).all()
    assert edge[2] == edge[0] and edge[3] == edge[1]          # out-of-range is clamped to [0, 1] first


def _random_xyxy(n):
    xy = torch.rand(n, 2) * 0.7
    return torch.cat([xy, xy + torch.rand(n, 2) * 0.3 + 0.01], dim=-1)


def test_iou_known_values():
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    b = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0], [5.0, 5.0, 6.0, 6.0]])
    iou, union = box_iou(a, b)
    assert iou.shape == union.shape == (1, 3)
    assert torch.allclose(iou, torch.tensor([[1.0, 1 / 7, 0.0]]))  # overlap 1, union 4 + 4 - 1 = 7
    assert torch.allclose(union, torch.tensor([[4.0, 7.0, 5.0]]))


def test_giou_known_values():
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    far = torch.tensor([[2.0, 0.0, 3.0, 1.0]])  # same size, one box-width gap: enclosing 3×1, union 2
    assert torch.allclose(generalized_box_iou(a, a), torch.tensor([[1.0]]))
    assert torch.allclose(generalized_box_iou(a, far), torch.tensor([[0.0 - (3 - 2) / 3]]))
    assert generalized_box_iou(a, torch.tensor([[1e6, 0.0, 1e6 + 1, 1.0]])).item() < -0.99


def test_giou_rejects_centre_format():
    cxcywh = torch.tensor([[0.5, 0.5, 0.2, 0.4]])  # read as xyxy: x2 (0.2) < x1 (0.5)
    with pytest.raises(ValueError):
        generalized_box_iou(cxcywh, cxcywh)


def test_iou_and_giou_match_torchvision():
    from torchvision.ops import box_iou as tv_iou
    from torchvision.ops import generalized_box_iou as tv_giou

    a, b = _random_xyxy(40), _random_xyxy(25)
    assert torch.allclose(box_iou(a, b)[0], tv_iou(a, b), atol=1e-6)
    assert torch.allclose(generalized_box_iou(a, b), tv_giou(a, b), atol=1e-6)


def test_matches_hf():
    from transformers.loss.loss_for_object_detection import box_iou as hf_iou
    from transformers.loss.loss_for_object_detection import (
        generalized_box_iou as hf_giou,
    )

    a, b = _random_xyxy(40), _random_xyxy(25)
    assert torch.equal(box_iou(a, b)[0], hf_iou(a, b)[0])
    assert torch.equal(generalized_box_iou(a, b), hf_giou(a, b))
