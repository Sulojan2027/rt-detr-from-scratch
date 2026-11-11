import pytest
import torch

from src.utils.box_ops import box_area, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, inverse_sigmoid


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
