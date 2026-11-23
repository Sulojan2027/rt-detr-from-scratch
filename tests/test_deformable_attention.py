import pytest
import torch

from src.decoder.deformable_attention import (
    MultiscaleDeformableAttention,
    multi_scale_deformable_attn,
)

SHAPES_640 = [(80, 80), (40, 40), (20, 20)]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _inputs(shapes, batch=2, queries=30, d_model=256):
    tokens = sum(h * w for h, w in shapes)
    hidden = torch.randn(batch, queries, d_model)
    pos = torch.randn(batch, queries, d_model)
    memory = torch.randn(batch, tokens, d_model)
    # sigmoid boxes (cx, cy, w, h), with a size-1 level axis that broadcasts over levels
    ref = torch.rand(batch, queries, 1, 4) * 0.8 + 0.1
    return hidden, pos, memory, ref


def test_param_count():
    attn = MultiscaleDeformableAttention()
    assert sum(p.numel() for p in attn.parameters()) == 205_600
    assert set(dict(attn.named_children())) == {"sampling_offsets", "attention_weights", "value_proj", "output_proj"}


def test_output_shape():
    attn = MultiscaleDeformableAttention().eval()
    hidden, pos, memory, ref = _inputs(SHAPES_640)
    out = attn(hidden, memory, ref, SHAPES_640, pos)
    assert out.shape == hidden.shape


def test_works_without_position_embeddings_and_on_non_square_maps():
    shapes = [(40, 52), (20, 26), (10, 13)]
    attn = MultiscaleDeformableAttention().eval()
    hidden, _, memory, ref = _inputs(shapes)
    assert attn(hidden, memory, ref, shapes).shape == hidden.shape


def test_rejects_bad_config_and_bad_shapes():
    with pytest.raises(ValueError):
        MultiscaleDeformableAttention(d_model=250, n_heads=8)
    attn = MultiscaleDeformableAttention()
    hidden, _, memory, ref = _inputs(SHAPES_640)
    with pytest.raises(ValueError):
        attn(hidden, memory, ref, [(80, 80), (40, 40)])


def test_gradients_reach_all_weights():
    attn = MultiscaleDeformableAttention()
    hidden, pos, memory, ref = _inputs([(8, 8), (4, 4), (2, 2)])
    attn(hidden, memory, ref, [(8, 8), (4, 4), (2, 2)], pos).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in attn.parameters())


def test_sampling_exactly_on_a_pixel_centre_returns_that_pixel():
    # One level 4×4, one point, weight 1: sampling the centre of pixel (row 1, col 2) must give its value.
    H, W = 4, 4
    value = torch.arange(H * W, dtype=torch.float32).view(1, H * W, 1, 1)  # [B, S, h=1, d=1]
    loc = torch.tensor([(2 + 0.5) / W, (1 + 0.5) / H]).view(1, 1, 1, 1, 1, 2)  # (x, y)
    weights = torch.ones(1, 1, 1, 1, 1)
    out = multi_scale_deformable_attn(value, [(H, W)], loc, weights)
    assert out.item() == pytest.approx(1 * W + 2)


def _hf_module():
    from transformers import RTDetrConfig
    from transformers.models.rt_detr.modeling_rt_detr import (
        RTDetrMultiscaleDeformableAttention,
    )

    return RTDetrMultiscaleDeformableAttention(RTDetrConfig(), num_heads=8, n_points=4).eval()


def test_core_matches_hf():
    from transformers.models.rt_detr.modeling_rt_detr import (
        MultiScaleDeformableAttention as HFCore,
    )

    B, Q, h, d, L, P = 2, 17, 8, 32, 3, 4
    shapes = [(16, 20), (8, 10), (4, 5)]
    value = torch.randn(B, sum(a * b for a, b in shapes), h, d)
    loc = torch.rand(B, Q, h, L, P, 2) * 1.2 - 0.1  # includes points outside the image
    weights = torch.rand(B, Q, h, L, P).flatten(-2).softmax(-1).view(B, Q, h, L, P)

    ours = multi_scale_deformable_attn(value, shapes, loc, weights)
    theirs = HFCore()(value, torch.tensor(shapes), shapes, None, loc, weights, 64)
    assert (ours - theirs).abs().max() < 1e-6


@pytest.mark.parametrize("shapes", [SHAPES_640, [(40, 52), (20, 26), (10, 13)]])
def test_module_matches_hf(shapes):
    hf = _hf_module()
    ours = MultiscaleDeformableAttention().eval()
    ours.load_state_dict(hf.state_dict(), strict=True)  # same key names, no remapping

    hidden, pos, memory, ref = _inputs(shapes)
    with torch.no_grad():
        expected, _ = hf(
            hidden_states=hidden,
            encoder_hidden_states=memory,
            position_embeddings=pos,
            reference_points=ref,
            spatial_shapes=torch.tensor(shapes),
            spatial_shapes_list=shapes,
        )
        actual = ours(hidden, memory, ref, shapes, pos)
    assert (actual - expected).abs().max() < 1e-5
