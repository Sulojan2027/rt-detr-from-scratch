import ast
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.encoder import HybridEncoder
from src.encoder.aifi import AIFILayer, SelfAttention
from src.encoder.ccfm import CCFM, CSPRepLayer
from src.encoder.position_encoding import build_2d_sinusoidal_position_embedding
from src.utils.hf_weights import HF_CONFIGS, encoder_state_dict

DOCS = Path(__file__).resolve().parent.parent / "docs"
STRIDES = (8, 16, 32)
# depth → total params (input proj + AIFI + CCFM)
EXPECTED_PARAMS = {18: 4_965_120, 50: 11_950_848}


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _feats(depth, h, w, batch=1, requires_grad=False):
    chans = HF_CONFIGS[depth]["encoder"]["in_channels"]
    return [torch.randn(batch, c, h // s, w // s, requires_grad=requires_grad) for c, s in zip(chans, STRIDES)]


def _hf_dump(depth):
    """Encoder keys + shapes from docs/hf_keys_r{depth}.txt, in our naming (no network)."""
    out, section = {}, None
    for line in (DOCS / f"hf_keys_r{depth}.txt").read_text().splitlines():
        if line.startswith("## "):
            section = line.split()[1]
        elif line and not line.startswith("#") and section in ("encoder_input_proj", "encoder"):
            key, shape = line.split(" ", 1)
            out[("encoder_input_proj." if section == "encoder_input_proj" else "") + key] = tuple(ast.literal_eval(shape))
    return out


# ---------------------------------------------------------------- position embedding
@pytest.mark.parametrize("hw", [(20, 20), (10, 13)])
def test_pos_embed_shape_and_values(hw):
    h, w = hw
    pe = build_2d_sinusoidal_position_embedding(h, w, 256)
    assert pe.shape == (h * w, 256) and pe.dtype == torch.float32
    q = 256 // 4
    # token 0 is (row 0, col 0): sin(0) = 0, cos(0) = 1 in all four blocks [sin_h, cos_h, sin_w, cos_w]
    assert torch.equal(pe[0], torch.cat([torch.zeros(q), torch.ones(q), torch.zeros(q), torch.ones(q)]))
    # token 1 is (row 0, col 1): h part unchanged, first w frequency is 1 → sin(1)
    assert torch.equal(pe[1, : 2 * q], pe[0, : 2 * q])
    assert pe[1, 2 * q].item() == pytest.approx(math.sin(1.0), abs=1e-6)


def test_pos_embed_bad_dim():
    with pytest.raises(ValueError):
        build_2d_sinusoidal_position_embedding(4, 4, 250)


# ---------------------------------------------------------------- AIFI
def test_self_attention_bad_heads():
    with pytest.raises(ValueError):
        SelfAttention(256, 7)


@pytest.mark.parametrize("hw", [(20, 20), (10, 13)])
def test_aifi_shape(hw):
    x = torch.randn(2, 256, *hw)
    assert AIFILayer().eval()(x).shape == x.shape


# ---------------------------------------------------------------- CCFM
@pytest.mark.parametrize("expansion, has_conv3", [(0.5, True), (1.0, False)])
def test_csp_conv3(expansion, has_conv3):
    assert isinstance(CSPRepLayer(512, 256, expansion=expansion).conv3, nn.Identity) is not has_conv3


def test_ccfm_act_passed_down():
    acts = {type(m) for m in CCFM(32, act="relu").modules() if isinstance(m, (nn.SiLU, nn.ReLU))}
    assert acts == {nn.ReLU}


# ---------------------------------------------------------------- HybridEncoder
@pytest.mark.parametrize("depth", [18, 50])
def test_hybrid_keys_match_hf_dump(depth):
    want = _hf_dump(depth)
    got = {k: tuple(v.shape) for k, v in HybridEncoder(**HF_CONFIGS[depth]["encoder"]).state_dict().items()}
    assert set(got) == set(want), (sorted(set(want) - set(got))[:3], sorted(set(got) - set(want))[:3])
    assert got == want


@pytest.mark.parametrize("depth", [18, 50])
def test_hybrid_param_count(depth):
    model = HybridEncoder(**HF_CONFIGS[depth]["encoder"])
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMS[depth]


def test_hybrid_attrs():
    model = HybridEncoder(**HF_CONFIGS[18]["encoder"])
    assert model.out_channels == [256, 256, 256]
    assert model.out_strides == [8, 16, 32]


@pytest.mark.parametrize("depth", [18, 50])
@pytest.mark.parametrize("hw", [(640, 640), (320, 416)])
def test_hybrid_shapes(depth, hw):
    h, w = hw
    model = HybridEncoder(**HF_CONFIGS[depth]["encoder"]).eval()
    with torch.no_grad():
        outs = model(_feats(depth, h, w, batch=2))
    assert [tuple(o.shape) for o in outs] == [(2, 256, h // s, w // s) for s in STRIDES]


def test_hybrid_bad_num_feats():
    model = HybridEncoder(**HF_CONFIGS[18]["encoder"])
    with pytest.raises(ValueError):
        model(_feats(18, 64, 64)[:2])


def test_hybrid_gradients():
    model = HybridEncoder(**HF_CONFIGS[18]["encoder"])
    feats = _feats(18, 64, 96, requires_grad=True)
    sum(o.sum() for o in model(feats)).backward()
    assert all(f.grad is not None for f in feats)
    assert all(p.grad is not None for p in model.parameters()), "a sub-module is not in the graph"


# ---------------------------------------------------------------- HF parity (slow)
@pytest.mark.slow
@pytest.mark.parametrize("depth", [18, 50])
@pytest.mark.parametrize("hw", [(640, 640), (320, 416)])
def test_hf_parity(depth, hw):
    transformers = pytest.importorskip("transformers")
    hf = transformers.RTDetrForObjectDetection.from_pretrained(f"PekingU/rtdetr_r{depth}vd").eval()
    m = hf.model

    model = HybridEncoder(**HF_CONFIGS[depth]["encoder"]).eval()
    model.load_state_dict(encoder_state_dict(hf.state_dict()), strict=True)

    with torch.no_grad():
        feats = m.backbone.model(torch.randn(1, 3, *hw)).feature_maps
        ref = m.encoder(inputs_embeds=[p(f) for p, f in zip(m.encoder_input_proj, feats, strict=True)]).last_hidden_state
        outs = model(list(feats))
    for out, r in zip(outs, ref, strict=True):
        assert out.shape == r.shape
        assert (out - r).abs().max().item() < 1e-4


@pytest.mark.slow
@pytest.mark.parametrize("depth", [18, 50])
def test_end_to_end_parity(depth):
    """image → our PResNet → our HybridEncoder == HF full forward's encoder_last_hidden_state."""
    transformers = pytest.importorskip("transformers")
    from src.backbone import PResNet
    from src.utils.hf_weights import backbone_state_dict

    hf = transformers.RTDetrForObjectDetection.from_pretrained(f"PekingU/rtdetr_r{depth}vd").eval()
    sd = hf.state_dict()
    backbone = PResNet(**HF_CONFIGS[depth]["backbone"]).eval()
    backbone.load_state_dict(backbone_state_dict(sd), strict=True)
    encoder = HybridEncoder(**HF_CONFIGS[depth]["encoder"]).eval()
    encoder.load_state_dict(encoder_state_dict(sd), strict=True)

    x = torch.randn(1, 3, 320, 416)
    with torch.no_grad():
        ref = hf(pixel_values=x).encoder_last_hidden_state
        outs = encoder(backbone(x))
    for out, r in zip(outs, ref, strict=True):
        assert out.shape == r.shape
        assert (out - r).abs().max().item() < 1e-4
