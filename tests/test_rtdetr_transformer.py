import ast
from pathlib import Path

import pytest
import torch

from src.decoder.rtdetr_transformer import RTDetrTransformer, generate_anchors
from src.losses.rtdetr_loss import RTDetrLoss
from src.matcher.hungarian import HungarianMatcher
from src.utils.hf_weights import decoder_state_dict

DOCS = Path(__file__).resolve().parent.parent / "docs"
# depth → (decoder layers, total params of query selection + decoder, number of checkpoint keys)
CONFIGS = {18: (3, 4_019_376, 137), 50: (6, 7_466_508, 239)}
STRIDES = (8, 16, 32)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _feats(h, w, batch=1):
    return [torch.randn(batch, 256, h // s, w // s) for s in STRIDES]


def _hf_dump(depth):
    """Decoder-section keys + shapes from docs/hf_keys_r{depth}.txt (no network)."""
    out, in_section = {}, False
    for line in (DOCS / f"hf_keys_r{depth}.txt").read_text().splitlines():
        if line.startswith("## "):
            in_section = line.split()[1] == "decoder"
        elif in_section and line and not line.startswith("#"):
            key, shape = line.split(" ", 1)
            out[key] = tuple(ast.literal_eval(shape))
    return out


@pytest.mark.parametrize("depth", [18, 50])
def test_params_and_keys_match_checkpoint(depth):
    layers, n_params, n_keys = CONFIGS[depth]
    model = RTDetrTransformer(num_decoder_layers=layers)
    assert sum(p.numel() for p in model.parameters()) == n_params
    ours = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    assert len(ours) == n_keys
    assert ours == _hf_dump(depth)


def test_anchors_basic_properties():
    shapes = [(80, 80), (40, 40), (20, 20)]
    anchors, valid = generate_anchors(shapes)
    assert anchors.shape == (1, 8400, 4)
    assert valid.shape == (1, 8400, 1) and valid.dtype == torch.bool
    # first token = top-left cell of P3: centre (0.5/80), size 0.05 → touches the border → invalid
    assert not valid[0, 0, 0]
    assert anchors[0, 0, 0] == torch.finfo(torch.float32).max
    # a middle cell of P3 is valid and sits at its cell centre
    middle = 40 * 80 + 40
    assert valid[0, middle, 0]
    centre = torch.sigmoid(anchors[0, middle])
    assert torch.allclose(centre, torch.tensor([40.5 / 80, 40.5 / 80, 0.05, 0.05]), atol=1e-6)
    # box size doubles on every coarser level
    assert torch.sigmoid(anchors[0, 6400 + 20 * 40 + 20, 2]) == pytest.approx(0.1, abs=1e-6)
    assert torch.sigmoid(anchors[0, 8000 + 10 * 20 + 10, 2]) == pytest.approx(0.2, abs=1e-6)


def test_forward_output_structure():
    model = RTDetrTransformer(num_decoder_layers=3).eval()
    out = model(_feats(320, 416, batch=2))
    assert out["logits"].shape == (2, 300, 80)
    assert out["pred_boxes"].shape == (2, 300, 4)
    assert out["enc_outputs_class"].shape == (2, (40 * 52) + (20 * 26) + (10 * 13), 80)
    # 2 earlier decoder layers + the encoder top-k predictions
    assert len(out["aux_outputs"]) == 3
    assert all(a["logits"].shape == (2, 300, 80) and a["pred_boxes"].shape == (2, 300, 4) for a in out["aux_outputs"])


def _gt_targets():
    return [
        {"class_labels": torch.tensor([1, 5, 7]), "boxes": torch.rand(3, 4) * 0.5 + 0.2},
        {"class_labels": torch.tensor([3]), "boxes": torch.rand(1, 4) * 0.5 + 0.2},
    ]


def test_training_backward_without_targets():
    model = RTDetrTransformer(num_decoder_layers=3).train()
    out = model(_feats(320, 416))
    assert "dn_outputs" not in out  # no ground truth → no denoising queries
    (out["logits"].sum() + out["pred_boxes"].sum()).backward()
    assert model.decoder.layers[0].self_attn.q_proj.weight.grad is not None


def test_denoising_is_only_used_in_training_with_targets():
    targets = _gt_targets()[:1]
    model = RTDetrTransformer(num_decoder_layers=3)
    assert "dn_outputs" not in model.eval()(_feats(320, 416), targets)
    assert "dn_outputs" in model.train()(_feats(320, 416), targets)


def test_training_with_denoising_output_structure_and_backward():
    targets = _gt_targets()
    model = RTDetrTransformer(num_decoder_layers=3).train()
    out = model(_feats(320, 416, batch=2), targets)

    # normal outputs look exactly like at inference: 300 queries, denoising queries already split off
    assert out["logits"].shape == (2, 300, 80) and out["pred_boxes"].shape == (2, 300, 4)
    assert len(out["aux_outputs"]) == 3
    # one denoising output per decoder layer; 3 GT max → 33 groups → 198 queries
    assert len(out["dn_outputs"]) == 3
    assert out["dn_outputs"][0]["logits"].shape == (2, 198, 80)
    assert out["dn_meta"]["dn_num_group"] == 33

    # the whole training path works: transformer → matcher → loss → backward
    loss = RTDetrLoss(HungarianMatcher())(out, targets)
    assert "loss_vfl_dn_2" in loss
    sum(loss.values()).backward()
    assert model.denoising_class_embed.weight.grad is not None


def test_state_dict_loader_strips_only_the_model_prefix():
    fake = {"model.decoder_input_proj.0.0.weight": 1, "model.decoder.layers.0.x": 2, "model.encoder.y": 3, "z": 4}
    assert decoder_state_dict(fake) == {"decoder_input_proj.0.0.weight": 1, "decoder.layers.0.x": 2}


def _hf_model(num_layers):
    from transformers import RTDetrConfig, RTDetrForObjectDetection

    return RTDetrForObjectDetection(RTDetrConfig(decoder_layers=num_layers, num_labels=80)).eval()


@pytest.mark.parametrize(("height", "width"), [(640, 640), (320, 416)])
def test_matches_hf_with_random_weights(height, width):
    hf = _hf_model(3)
    ours = RTDetrTransformer(num_decoder_layers=3).eval()
    ours.load_state_dict(decoder_state_dict(hf.state_dict()), strict=True)

    feats = _feats(height, width)
    with torch.no_grad():
        expected = hf.model(pixel_values=None, inputs_embeds=feats[0], encoder_outputs=(feats,))
        actual = ours(feats)

    assert (actual["enc_outputs_class"] - expected.enc_outputs_class).abs().max() < 1e-4
    assert (actual["logits"] - expected.intermediate_logits[:, -1]).abs().max() < 1e-4
    assert (actual["pred_boxes"] - expected.intermediate_reference_points[:, -1]).abs().max() < 1e-4


@pytest.mark.slow
@pytest.mark.parametrize("depth", [18, 50])
def test_matches_pretrained_hf_end_to_end(depth):
    """Pretrained PekingU weights (downloads): our decoder fed HF's own encoder output vs HF's full forward."""
    from transformers import RTDetrForObjectDetection

    hf = RTDetrForObjectDetection.from_pretrained(f"PekingU/rtdetr_r{depth}vd").eval()
    ours = RTDetrTransformer(num_decoder_layers=CONFIGS[depth][0]).eval()
    ours.load_state_dict(decoder_state_dict(hf.state_dict()), strict=True)

    x = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        expected = hf(pixel_values=x)
        actual = ours(list(expected.encoder_last_hidden_state))

    assert (actual["enc_outputs_class"] - expected.enc_outputs_class).abs().max() < 1e-4
    assert (actual["logits"].sigmoid() - expected.logits.sigmoid()).abs().max() < 1e-4
    assert (actual["pred_boxes"] - expected.pred_boxes).abs().max() < 1e-4


def test_training_matches_hf_with_denoising():
    """Same seed → same noise → our training forward equals HF's, including the denoising queries."""
    hf = _hf_model(3).train()
    ours = RTDetrTransformer(num_decoder_layers=3).train()
    ours.load_state_dict(decoder_state_dict(hf.state_dict()), strict=True)

    feats = _feats(320, 416, batch=2)
    targets = _gt_targets()
    with torch.no_grad():
        torch.manual_seed(1)
        expected = hf.model(pixel_values=None, inputs_embeds=feats[0], encoder_outputs=(feats,), labels=targets)
        torch.manual_seed(1)
        actual = ours(feats, targets)

    num_dn = expected.denoising_meta_values["dn_num_split"][0]
    assert actual["dn_meta"]["dn_num_split"] == expected.denoising_meta_values["dn_num_split"]
    hf_logits, hf_boxes = expected.intermediate_logits, expected.intermediate_reference_points
    assert (actual["logits"] - hf_logits[:, -1, num_dn:]).abs().max() < 1e-4
    assert (actual["pred_boxes"] - hf_boxes[:, -1, num_dn:]).abs().max() < 1e-4
    for layer, dn in enumerate(actual["dn_outputs"]):
        assert (dn["logits"] - hf_logits[:, layer, :num_dn]).abs().max() < 1e-4
        assert (dn["pred_boxes"] - hf_boxes[:, layer, :num_dn]).abs().max() < 1e-4
