import pytest
import torch
import torch.nn as nn

from src.backbone import PResNet
from src.backbone.common import FrozenBatchNorm2d
from src.backbone.presnet import BasicBlock, Blocks, BottleNeck

HF_PREFIX = "model.backbone.model."

# depth → (out_channels, params in millions with freeze_norm=False)
EXPECTED = {
    18: ([128, 256, 512], 11.20),
    34: ([128, 256, 512], 21.31),
    50: ([512, 1024, 2048], 23.53),
    101: ([512, 1024, 2048], 42.52),
}


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.mark.parametrize("depth", [18, 50])
@pytest.mark.parametrize("hw", [(64, 64), (65, 97), (96, 128)])
def test_presnet_shapes(depth, hw):
    model = PResNet(depth).eval()
    h, w = hw
    with torch.no_grad():
        outs = model(torch.randn(2, 3, h, w))
    channels, _ = EXPECTED[depth]
    assert model.out_channels == channels
    assert model.out_strides == [8, 16, 32]
    for out, c, s in zip(outs, channels, model.out_strides, strict=True):
        # every downsample uses padding / ceil_mode, so size = ceil(size / stride)
        assert out.shape == (2, c, -(-h // s), -(-w // s))


@pytest.mark.parametrize("depth", sorted(EXPECTED))
def test_presnet_param_count(depth):
    model = PResNet(depth, freeze_norm=False)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    assert n == pytest.approx(EXPECTED[depth][1], abs=0.01)


def test_hf_key_names():
    keys = PResNet(18).state_dict().keys()
    for k in [
        "embedder.embedder.0.convolution.weight",
        "encoder.stages.0.layers.0.shortcut.convolution.weight",    # stride-1 projection
        "encoder.stages.1.layers.0.shortcut.1.convolution.weight",  # vd: AvgPool at index 0
        "encoder.stages.3.layers.1.layer.1.normalization.running_var",
    ]:
        assert k in keys, k
    assert not any(k.startswith("encoder.stages.1.layers.1.shortcut") for k in keys)
    assert not any("num_batches_tracked" in k for k in keys)


def test_blocks_structure():
    stage = Blocks(BottleNeck, 256, 128, count=4, stride=2)
    layers = list(stage.layers)
    assert not isinstance(layers[0].shortcut, nn.Identity)
    assert all(isinstance(b.shortcut, nn.Identity) for b in layers[1:])
    assert layers[0].layer[1].convolution.stride == (2, 2)  # stride on the 3x3
    assert layers[1].layer[0].convolution.in_channels == 512
    with pytest.raises(ValueError):
        Blocks(BasicBlock, 64, 64, count=0, stride=1)


def test_freezing():
    model = PResNet(18, freeze_at=1, freeze_norm=True)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert any(isinstance(m, FrozenBatchNorm2d) for m in model.modules())
    assert not any(p.requires_grad for p in model.embedder.parameters())
    assert not any(p.requires_grad for p in model.encoder["stages"][0].parameters())
    assert all(p.requires_grad for p in model.encoder["stages"][1].parameters())


@pytest.mark.parametrize("bad", [(), (3, 1), (4,), (1, 1)])
def test_bad_return_idx(bad):
    with pytest.raises(ValueError):
        PResNet(18, return_idx=bad)


def test_gradients_flow():
    model = PResNet(18, freeze_norm=False)
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    sum(o.sum() for o in model(x)).backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.slow
@pytest.mark.parametrize("depth", [18, 50])
def test_hf_parity(depth):
    transformers = pytest.importorskip("transformers")
    hf = transformers.RTDetrForObjectDetection.from_pretrained(f"PekingU/rtdetr_r{depth}vd").eval()
    sd = {k[len(HF_PREFIX):]: v for k, v in hf.state_dict().items() if k.startswith(HF_PREFIX)}

    model = PResNet(depth, freeze_norm=True).eval()
    model.load_state_dict(sd, strict=True)

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        ref = hf.model.backbone.model(x).feature_maps
        outs = model(x)
    for out, r in zip(outs, ref, strict=True):
        assert out.shape == r.shape
        assert (out - r).abs().max().item() < 1e-4
