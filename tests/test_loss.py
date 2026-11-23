import pytest
import torch

from src.losses.rtdetr_loss import RTDetrLoss
from src.matcher.hungarian import HungarianMatcher


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _boxes(*shape):
    centre = torch.rand(*shape, 2) * 0.6 + 0.2
    size = torch.rand(*shape, 2) * 0.3 + 0.05
    return torch.cat([centre, size], dim=-1)


def _predictions(batch, queries, classes=80):
    return {"logits": torch.randn(batch, queries, classes, requires_grad=True), "pred_boxes": _boxes(batch, queries).requires_grad_()}


def _targets(gt_counts, classes=80):
    return [{"class_labels": torch.randint(0, classes, (n,)), "boxes": _boxes(n)} for n in gt_counts]


def _make_loss():
    return RTDetrLoss(HungarianMatcher())


def test_default_weights_and_keys():
    loss = _make_loss()
    assert loss.weight_dict == {"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    out = _predictions(2, 30)
    out["aux_outputs"] = [_predictions(2, 30), _predictions(2, 30)]
    losses = loss(out, _targets((3, 5)))
    assert set(losses) == {
        "loss_vfl", "loss_bbox", "loss_giou",
        "loss_vfl_aux_0", "loss_bbox_aux_0", "loss_giou_aux_0",
        "loss_vfl_aux_1", "loss_bbox_aux_1", "loss_giou_aux_1",
    }  # fmt: skip
    assert all(torch.isfinite(v) and v >= 0 for v in losses.values())


def test_backward_reaches_logits_and_boxes():
    out = _predictions(2, 30)
    total = sum(_make_loss()(out, _targets((3, 5))).values())
    total.backward()
    assert out["logits"].grad.abs().sum() > 0
    assert out["pred_boxes"].grad.abs().sum() > 0


def test_perfect_boxes_give_zero_box_loss():
    targets = _targets((4,))
    out = _predictions(1, 20)
    with torch.no_grad():
        out["pred_boxes"][0, :4] = targets[0]["boxes"]  # 4 queries that equal the GT boxes exactly
    losses = _make_loss()(out, targets)
    assert losses["loss_bbox"].item() == pytest.approx(0.0, abs=1e-6)
    assert losses["loss_giou"].item() == pytest.approx(0.0, abs=1e-6)


def test_images_without_objects_still_work():
    out = _predictions(2, 30)
    losses = _make_loss()(out, _targets((0, 0)))
    assert all(torch.isfinite(v) for v in losses.values())
    assert losses["loss_bbox"].item() == 0.0 and losses["loss_giou"].item() == 0.0  # nothing to regress
    assert losses["loss_vfl"].item() > 0.0  # every query is still taught "no object"


def test_confident_correct_class_lowers_vfl():
    targets = _targets((1,))
    out = _predictions(1, 10)
    with torch.no_grad():
        out["logits"][:] = -8.0
        out["pred_boxes"][0, 3] = targets[0]["boxes"][0]
    loss = _make_loss()
    before = loss(out, targets)["loss_vfl"].item()
    with torch.no_grad():
        out["logits"][0, 3, targets[0]["class_labels"][0]] = 8.0
    assert loss(out, targets)["loss_vfl"].item() < before


def _hf_format(ours, dn_meta=None):
    """Our output dict → the dict HF's RTDetrLoss expects."""
    hf = {"logits": ours["logits"], "pred_boxes": ours["pred_boxes"], "auxiliary_outputs": ours["aux_outputs"]}
    if dn_meta is not None:
        hf["dn_auxiliary_outputs"] = ours["dn_outputs"]
        hf["denoising_meta_values"] = dn_meta
    return hf


@pytest.mark.parametrize("gt_counts", [(5, 0, 12), (1, 3, 2)])
def test_matches_hf(gt_counts):
    from transformers import RTDetrConfig
    from transformers.loss.loss_rt_detr import RTDetrLoss as HFLoss

    config = RTDetrConfig(num_labels=80)
    hf_loss = HFLoss(config)
    ours_loss = RTDetrLoss(HungarianMatcher(), alpha=config.focal_loss_alpha, gamma=config.focal_loss_gamma)

    outputs = _predictions(3, 300)
    outputs["aux_outputs"] = [_predictions(3, 300) for _ in range(3)]
    targets = _targets(gt_counts)

    expected = hf_loss(_hf_format(outputs), targets)
    actual = ours_loss(outputs, targets)
    assert set(actual) == set(expected)
    for key in expected:
        assert torch.allclose(actual[key], expected[key], atol=1e-5), key


def test_denoising_matches_hf():
    from transformers import RTDetrConfig
    from transformers.loss.loss_rt_detr import RTDetrLoss as HFLoss

    config = RTDetrConfig(num_labels=80)
    gt_counts, groups, dn_queries = (3, 0, 4), 2, 40
    targets = _targets(gt_counts)

    outputs = _predictions(3, 300)
    outputs["aux_outputs"] = [_predictions(3, 300)]
    outputs["dn_outputs"] = [_predictions(3, dn_queries), _predictions(3, dn_queries)]
    # every group holds one noisy copy of each GT, so image i has n_gt * groups positive queries
    positive = [torch.randperm(dn_queries)[: n * groups] for n in gt_counts]
    dn_meta = {"dn_positive_idx": positive, "dn_num_group": groups}
    outputs["dn_meta"] = dn_meta

    expected = HFLoss(config)(_hf_format(outputs, dn_meta), targets)
    actual = RTDetrLoss(HungarianMatcher(), alpha=config.focal_loss_alpha, gamma=config.focal_loss_gamma)(outputs, targets)
    assert set(actual) == set(expected)
    assert "loss_vfl_dn_1" in actual
    for key in expected:
        assert torch.allclose(actual[key], expected[key], atol=1e-5), key
