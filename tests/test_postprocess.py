import torch

from src.postprocess import post_process


def test_known_values_and_scaling():
    logits = torch.full((1, 2, 3), -10.0)
    logits[0, 1, 2] = 10.0                                    # query 1, class 2 → only confident detection
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.5, 0.25, 0.5, 0.5]]])
    sizes = torch.tensor([[200, 400]])                        # (h, w)
    r = post_process(logits, boxes, sizes, threshold=0.5)[0]
    assert r["labels"].tolist() == [2]
    assert torch.allclose(r["boxes"], torch.tensor([[100.0, 0.0, 300.0, 100.0]]))   # x scaled by w, y by h


def test_topk_over_queries_times_classes():
    logits = torch.randn(2, 5, 4)
    out = post_process(logits, torch.rand(2, 5, 4), torch.tensor([[10, 20], [30, 40]]), threshold=0.0)
    assert [len(o["scores"]) for o in out] == [5, 5]
    assert (out[0]["scores"][:-1] >= out[0]["scores"][1:]).all()          # sorted by score
    assert out[0]["labels"].max() < 4
