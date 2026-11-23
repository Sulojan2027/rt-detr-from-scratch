import pytest
import torch
from torch import nn

from src.decoder.denoising import get_contrastive_denoising_training_group

NUM_CLASSES, NUM_QUERIES = 80, 300


def _targets(gt_counts):
    return [
        {"class_labels": torch.randint(0, NUM_CLASSES, (n,)), "boxes": torch.rand(n, 4) * 0.5 + 0.2}
        for n in gt_counts
    ]


def _embed():
    return nn.Embedding(NUM_CLASSES + 1, 256, padding_idx=NUM_CLASSES)


def _group(targets, **kwargs):
    return get_contrastive_denoising_training_group(targets, NUM_CLASSES, NUM_QUERIES, _embed(), **kwargs)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_shapes_and_meta_match_the_documented_example():
    dn_class, dn_bbox, attn_mask, meta = _group(_targets((3, 1)))
    # max_gt = 3, groups = 100 // 3 = 33, queries = 3 * 2 * 33 = 198
    assert dn_class.shape == (2, 198, 256)
    assert dn_bbox.shape == (2, 198, 4)
    assert attn_mask.shape == (498, 498)
    assert meta["dn_num_group"] == 33
    assert meta["dn_num_split"] == [198, 300]
    assert [len(idx) for idx in meta["dn_positive_idx"]] == [99, 33]  # n_gt * groups


def test_nothing_to_denoise_returns_none():
    assert _group(_targets((0, 0))) == (None, None, None, None)
    assert _group(_targets((2,)), num_denoising=0) == (None, None, None, None)


def test_image_without_gt_has_no_positive_queries():
    _, _, _, meta = _group(_targets((2, 0)))
    assert len(meta["dn_positive_idx"][1]) == 0


def test_more_gt_than_budget_still_makes_one_group():
    _, _, _, meta = _group(_targets((150,)), num_denoising=100)
    assert meta["dn_num_group"] == 1


def test_attention_mask_rules():
    _, _, mask, meta = _group(_targets((3, 1)))
    num_dn, groups, block = meta["dn_num_split"][0], meta["dn_num_group"], 3 * 2
    blocked = torch.isinf(mask)
    assert not blocked.diagonal().any()  # everyone sees itself
    assert blocked[num_dn:, :num_dn].all()  # normal queries never see denoising queries
    assert not blocked[:, num_dn:].any()  # but everything may see the normal queries
    for g in range(groups):  # a group sees itself, and nobody else's group
        own = slice(g * block, (g + 1) * block)
        assert not blocked[own, own].any()
        assert blocked[own, : g * block].all()
        assert blocked[own, (g + 1) * block : num_dn].all()


def test_positive_queries_are_the_real_first_half_of_each_group():
    _, _, _, meta = _group(_targets((3, 1)))
    max_gt = 3
    for n_gt, idx in zip((3, 1), meta["dn_positive_idx"], strict=True):
        # inside a group's positive half, and not on a padded slot
        assert all((i % (2 * max_gt)) < n_gt for i in idx.tolist())


def test_box_noise_stays_in_range_and_negatives_are_noisier():
    targets = [{"class_labels": torch.tensor([4]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]
    _, dn_bbox, _, _ = _group(targets, num_denoising=2000)
    assert torch.isfinite(dn_bbox).all()
    boxes = dn_bbox.sigmoid()[0]  # back from logit space; one GT per half-group → even rows positive, odd negative
    positives, negatives = boxes[0::2], boxes[1::2]

    def shift(b):
        return (b[:, :2] - 0.5).abs().mean() + (b[:, 2:] - 0.2).abs().mean()

    assert shift(negatives) > shift(positives)


def test_matches_hf_with_same_seed():
    from transformers.models.rt_detr.modeling_rt_detr import (
        get_contrastive_denoising_training_group as hf_group,
    )

    targets = _targets((3, 1, 0, 7))
    embed = _embed()
    torch.manual_seed(5)
    expected = hf_group(targets, NUM_CLASSES, NUM_QUERIES, embed, 100, 0.5, 1.0)
    torch.manual_seed(5)
    actual = get_contrastive_denoising_training_group(targets, NUM_CLASSES, NUM_QUERIES, embed, 100, 0.5, 1.0)

    for ours, theirs in zip(actual[:3], expected[:3], strict=True):
        assert torch.equal(ours, theirs)
    assert actual[3]["dn_num_group"] == expected[3]["dn_num_group"]
    assert actual[3]["dn_num_split"] == expected[3]["dn_num_split"]
    for ours, theirs in zip(actual[3]["dn_positive_idx"], expected[3]["dn_positive_idx"], strict=True):
        assert torch.equal(ours, theirs)
