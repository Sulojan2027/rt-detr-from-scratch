import pytest
import torch
import torch.nn as nn

from src.decoder.decoder import Decoder, DecoderLayer, MLPPredictionHead
from src.encoder.aifi import SelfAttention

SHAPES = [(16, 20), (8, 10), (4, 5)]  # small, non-square maps
TOKENS = sum(h * w for h, w in SHAPES)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _n_params(module):
    return sum(p.numel() for p in module.parameters())


def test_param_counts():
    assert _n_params(MLPPredictionHead(256, 256, 4, 3)) == 132_612
    assert _n_params(MLPPredictionHead(4, 512, 256, 2)) == 133_888
    assert _n_params(DecoderLayer()) == 995_872
    assert _n_params(Decoder(num_layers=3)) == 3_581_020
    assert _n_params(Decoder(num_layers=6)) == 7_028_152


def test_output_shapes():
    dec = Decoder(num_layers=3).eval()
    target, ref, memory = torch.randn(2, 30, 256), torch.randn(2, 30, 4), torch.randn(2, TOKENS, 256)
    logits, boxes = dec(target, ref, memory, SHAPES)
    assert logits.shape == (2, 3, 30, 80)
    assert boxes.shape == (2, 3, 30, 4)
    assert boxes.min() >= 0 and boxes.max() <= 1


def test_gradient_flow_and_detached_reference():
    dec = Decoder(num_layers=3)
    target, ref, memory = torch.randn(1, 10, 256), torch.randn(1, 10, 4, requires_grad=True), torch.randn(1, TOKENS, 256)
    logits, boxes = dec(target, ref, memory, SHAPES)
    (logits.sum() + boxes.sum()).backward()
    assert all(p.grad is not None for p in dec.parameters())
    # the first reference box is detached from layer 2 onward, but layer 1's own output still reaches it
    assert ref.grad is not None


def test_self_attention_mask_blocks_attention():
    attn = SelfAttention().eval()
    x = torch.randn(1, 6, 256)
    mask = torch.zeros(6, 6)
    mask[:, 5] = float("-inf")  # nobody may look at token 5
    changed = x.clone()
    changed[:, 5] += 10.0  # so changing token 5 must not change the others' outputs
    out_a, out_b = attn(x, attn_mask=mask), attn(changed, attn_mask=mask)
    assert torch.allclose(out_a[:, :5], out_b[:, :5], atol=1e-5)
    assert not torch.allclose(attn(x)[:, :5], attn(changed)[:, :5], atol=1e-5)


def _hf_decoder(num_layers):
    """HF decoder with random weights, with the per-layer heads attached the way RTDetrForObjectDetection does."""
    from transformers import RTDetrConfig
    from transformers.models.rt_detr.modeling_rt_detr import (
        RTDetrDecoder,
        RTDetrMLPPredictionHead,
    )

    cfg = RTDetrConfig(decoder_layers=num_layers, num_labels=80)
    hf = RTDetrDecoder(cfg)
    hf.class_embed = nn.ModuleList(nn.Linear(cfg.d_model, cfg.num_labels) for _ in range(num_layers))
    hf.bbox_embed = nn.ModuleList(
        RTDetrMLPPredictionHead(cfg.d_model, cfg.d_model, 4, 3) for _ in range(num_layers)
    )
    return hf.eval()


@pytest.mark.parametrize("num_layers", [3, 6])
def test_matches_hf(num_layers):
    hf = _hf_decoder(num_layers)
    ours = Decoder(num_layers=num_layers).eval()
    ours.load_state_dict(hf.state_dict(), strict=True)  # same key names, no remapping

    target, ref, memory = torch.randn(2, 30, 256), torch.randn(2, 30, 4), torch.randn(2, TOKENS, 256)
    with torch.no_grad():
        expected = hf(
            inputs_embeds=target,
            encoder_hidden_states=memory,
            reference_points=ref,
            spatial_shapes=torch.tensor(SHAPES),
            spatial_shapes_list=SHAPES,
            level_start_index=torch.tensor([0, 320, 400]),
        )
        logits, boxes = ours(target, ref, memory, SHAPES)

    assert (logits - expected.intermediate_logits).abs().max() < 1e-4
    assert (boxes - expected.intermediate_reference_points).abs().max() < 1e-4
