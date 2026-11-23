import pytest
import torch

from src.matcher.hungarian import HungarianMatcher


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _random_boxes(*shape):
    """Valid normalised (cx, cy, w, h) boxes."""
    centre = torch.rand(*shape, 2) * 0.6 + 0.2
    size = torch.rand(*shape, 2) * 0.3 + 0.05
    return torch.cat([centre, size], dim=-1)


def _batch(batch=3, queries=40, classes=80, gt_counts=(5, 0, 12)):
    outputs = {"logits": torch.randn(batch, queries, classes), "pred_boxes": _random_boxes(batch, queries)}
    targets = [
        {"class_labels": torch.randint(0, classes, (n,)), "boxes": _random_boxes(n)} for n in gt_counts
    ]
    return outputs, targets


def test_rejects_all_zero_costs():
    with pytest.raises(ValueError):
        HungarianMatcher(0, 0, 0)


def test_output_structure_and_one_to_one():
    outputs, targets = _batch()
    matches = HungarianMatcher()(outputs, targets)
    assert len(matches) == 3
    for (query_idx, gt_idx), target in zip(matches, targets):
        n = len(target["boxes"])
        assert query_idx.dtype == gt_idx.dtype == torch.int64
        assert len(query_idx) == len(gt_idx) == n
        assert len(set(query_idx.tolist())) == n  # no query is used twice
        assert sorted(gt_idx.tolist()) == list(range(n))  # every GT is used exactly once


def test_image_without_gt_gives_empty_indices():
    outputs, targets = _batch(gt_counts=(0, 0, 0))
    for query_idx, gt_idx in HungarianMatcher()(outputs, targets):
        assert query_idx.numel() == 0 and gt_idx.numel() == 0


def test_more_gt_than_queries_matches_min():
    outputs, targets = _batch(batch=1, queries=4, gt_counts=(9,))
    query_idx, gt_idx = HungarianMatcher()(outputs, targets)[0]
    assert len(query_idx) == len(gt_idx) == 4


def test_perfect_prediction_is_matched_to_its_query():
    outputs, targets = _batch(batch=1, queries=20, gt_counts=(1,))
    # make query 7 an exact copy of the GT: right box, confident right class
    outputs["pred_boxes"][0, 7] = targets[0]["boxes"][0]
    outputs["logits"][0, 7] = -10.0
    outputs["logits"][0, 7, targets[0]["class_labels"][0]] = 10.0
    query_idx, gt_idx = HungarianMatcher()(outputs, targets)[0]
    assert query_idx.tolist() == [7] and gt_idx.tolist() == [0]


def test_each_image_only_uses_its_own_gt():
    outputs, targets = _batch(batch=2, queries=10, gt_counts=(3, 4))
    for (_, gt_idx), target in zip(HungarianMatcher()(outputs, targets), targets):
        assert gt_idx.max() < len(target["boxes"])  # indices are local to the image


def test_nan_scores_do_not_crash():
    outputs, targets = _batch(batch=1, queries=10, gt_counts=(3,))
    outputs["logits"][0, 2] = float("nan")  # (a nan *box* is rejected earlier, by the GIoU corner-format check)
    query_idx, _ = HungarianMatcher()(outputs, targets)[0]
    assert 2 not in query_idx.tolist()  # the broken query is only used as a last resort


def test_matches_hf():
    from transformers import RTDetrConfig
    from transformers.loss.loss_rt_detr import RTDetrHungarianMatcher

    config = RTDetrConfig()
    ours = HungarianMatcher(
        config.matcher_class_cost, config.matcher_bbox_cost, config.matcher_giou_cost,
        config.matcher_alpha, config.matcher_gamma,
    )
    hf = RTDetrHungarianMatcher(config)
    for gt_counts in [(5, 0, 12), (1, 1, 1), (30, 2, 7)]:
        outputs, targets = _batch(batch=3, queries=300, gt_counts=gt_counts)
        for (qi, gi), (qh, gh) in zip(ours(outputs, targets), hf(outputs, targets)):
            assert torch.equal(qi, qh) and torch.equal(gi, gh)
