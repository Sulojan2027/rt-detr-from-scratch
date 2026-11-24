"""src/utils/trt.py

- Fake-TensorRT tests run everywhere: they check the plumbing (logger, error messages, IO bookkeeping, addresses).
- The real self-check (tiny conv net → ONNX → engine → parity with PyTorch) needs `tensorrt` + `onnx` + CUDA,
  so it is skipped on the laptop and runs on the 3090 server.
"""

import sys
import types

import pytest
import torch
from torch import nn

import src.utils.trt as trt_utils

# ---------------------------------------------------------------- fake TensorRT (laptop) ----


def _fake_trt(parse_ok=True, build_ok=True, load_ok=True):
    """Just enough of the TensorRT 10 API for trt.py: one input `x` [1, 3] fp32, one output `y` [1, 2] fp32."""
    m = types.ModuleType("tensorrt")
    m.float32, m.float16, m.int32, m.int64, m.bool, m.int8 = "f32", "f16", "i32", "i64", "bool", "i8"
    m.calls = {"plugins": 0}

    class Logger:
        WARNING = 2

        def __init__(self, level):
            self.level = level

    def init_libnvinfer_plugins(logger, ns):
        m.calls["plugins"] += 1

    class Parser:
        num_errors = 1

        def __init__(self, network, logger):
            pass

        def parse_from_file(self, path):
            return parse_ok

        def get_error(self, i):
            return "Unsupported op: FancyOp"

    class Config:
        def set_memory_pool_limit(self, pool, size):
            m.calls["workspace"] = size

        def set_flag(self, flag):
            m.calls["flag"] = flag

    class Builder:
        def __init__(self, logger):
            pass

        def create_network(self, flags):
            return object()

        def create_builder_config(self):
            return Config()

        def build_serialized_network(self, network, config):
            return b"ENGINE" if build_ok else None

    class Context:
        def __init__(self):
            self.addresses = {}

        def set_tensor_address(self, name, ptr):
            self.addresses[name] = ptr

        def execute_async_v3(self, stream):
            return True

    class Engine:
        num_io_tensors = 2
        _io = {"x": ("in", (1, 3)), "y": ("out", (1, 2))}

        def get_tensor_name(self, i):
            return list(self._io)[i]

        def get_tensor_mode(self, name):
            return m.TensorIOMode.INPUT if self._io[name][0] == "in" else m.TensorIOMode.OUTPUT

        def get_tensor_shape(self, name):
            return self._io[name][1]

        def get_tensor_dtype(self, name):
            return m.float32

        def create_execution_context(self):
            return Context()

    class Runtime:
        def __init__(self, logger):
            pass

        def deserialize_cuda_engine(self, data):
            return Engine() if load_ok else None

    m.Logger, m.init_libnvinfer_plugins, m.OnnxParser, m.Builder, m.Runtime = (
        Logger, init_libnvinfer_plugins, Parser, Builder, Runtime)
    m.MemoryPoolType = types.SimpleNamespace(WORKSPACE="ws")
    m.BuilderFlag = types.SimpleNamespace(FP16="fp16")
    m.TensorIOMode = types.SimpleNamespace(INPUT="INPUT", OUTPUT="OUTPUT")
    return m


@pytest.fixture
def fake(monkeypatch):
    def install(**kw):
        m = _fake_trt(**kw)
        monkeypatch.setitem(sys.modules, "tensorrt", m)
        monkeypatch.setattr(trt_utils, "_LOGGER", None)
        return m

    return install


def test_logger_created_once_and_plugins_registered(fake):
    m = fake()
    a, b = trt_utils._logger(), trt_utils._logger()
    assert a is b and m.calls["plugins"] == 1


def test_build_engine_writes_file_in_new_folder(fake, tmp_path):
    m = fake()
    out = trt_utils.build_engine("model.onnx", str(tmp_path / "engines" / "m.engine"), fp16=True, workspace_gb=0.5)
    assert out == tmp_path / "engines" / "m.engine" and out.read_bytes() == b"ENGINE"
    assert m.calls["flag"] == "fp16" and m.calls["workspace"] == 1 << 29


def test_build_engine_reports_parser_errors(fake, tmp_path):
    fake(parse_ok=False)
    with pytest.raises(RuntimeError, match="FancyOp"):
        trt_utils.build_engine("model.onnx", tmp_path / "m.engine")


def test_build_engine_raises_on_failed_build(fake, tmp_path):
    fake(build_ok=False)
    with pytest.raises(RuntimeError, match="build failed"):
        trt_utils.build_engine("model.onnx", tmp_path / "m.engine")
    assert not (tmp_path / "m.engine").exists()


def test_trtmodule_raises_on_unloadable_engine(fake, tmp_path):
    fake(load_ok=False)
    (tmp_path / "m.engine").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="could not load"):
        trt_utils.TRTModule(tmp_path / "m.engine", device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="execute uses torch.cuda.current_stream()")
def test_trtmodule_io_bookkeeping(fake, tmp_path):
    fake()
    (tmp_path / "m.engine").write_bytes(b"x")
    mod = trt_utils.TRTModule(tmp_path / "m.engine")
    assert mod.input_names == ["x"] and mod.input_dtypes == [torch.float32]
    assert mod.outputs["y"].shape == (1, 2) and mod.outputs["y"].is_cuda
    assert mod.context.addresses["y"] == mod.outputs["y"].data_ptr()

    x = torch.ones(1, 3, dtype=torch.float64)              # CPU + wrong dtype → converted, moved, kept alive
    out = mod(x)
    assert out is mod.outputs
    assert mod._inputs[0].is_cuda and mod._inputs[0].dtype == torch.float32
    assert mod.context.addresses["x"] == mod._inputs[0].data_ptr()
    with pytest.raises(ValueError):                         # wrong number of inputs (zip strict)
        mod(x, x)


# ---------------------------------------------------------------- real TensorRT (3090 server) ----


@pytest.mark.slow
@pytest.mark.parametrize("fp16, tol", [(False, 1e-4), (True, 1e-2)])
def test_real_engine_matches_pytorch(fp16, tol, tmp_path, monkeypatch):
    pytest.importorskip("tensorrt")
    pytest.importorskip("onnx")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    monkeypatch.setattr(trt_utils, "_LOGGER", None)

    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 4, 1)).eval()
    x = torch.randn(1, 3, 32, 32)
    onnx_path = tmp_path / "tiny.onnx"
    torch.onnx.export(net, (x,), str(onnx_path), input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)

    engine = trt_utils.TRTModule(trt_utils.build_engine(onnx_path, tmp_path / "tiny.engine", fp16=fp16))
    assert engine.input_names == ["x"] and list(engine.outputs) == ["y"]

    y = engine(x.cuda())["y"]
    torch.cuda.synchronize()
    with torch.no_grad():
        ref = net(x)
    assert (y.float().cpu() - ref).abs().max() < tol
