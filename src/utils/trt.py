"""Minimal TensorRT helpers: ONNX → engine, and run an engine on torch CUDA tensors.  [SHARED — Step 1]

Big picture (see the explanation in the chat / docs):
    model.onnx  ──(1) build_engine──▶  model.engine  ──(2) TRTModule──▶  outputs
    portable recipe                    compiled for THIS GPU + TensorRT version

TensorRT 10 Python API — the only objects you need:
    trt.Logger(trt.Logger.WARNING)           prints TensorRT warnings/errors; ONE logger for everything
    trt.init_libnvinfer_plugins(logger, "")  registers built-in plugins (EfficientNMS_TRT for YOLO) — call once
    trt.Builder(logger)                      makes networks + engines
      .create_network(0)                     empty network (TRT 10: explicit batch is the only mode → flag 0)
      .create_builder_config()               build options (FP16, workspace memory)
      .build_serialized_network(net, cfg)    → engine bytes (or None on failure) — SLOW: autotunes kernels
    trt.OnnxParser(network, logger)          fills the network from an ONNX file
      .parse_from_file(path) → bool          .num_errors / .get_error(i) explain failures
    trt.Runtime(logger)
      .deserialize_cuda_engine(bytes)        bytes → ICudaEngine (the loaded engine)
    engine
      .num_io_tensors, .get_tensor_name(i)   inputs + outputs, by name
      .get_tensor_mode(name)                 trt.TensorIOMode.INPUT / OUTPUT
      .get_tensor_shape(name), .get_tensor_dtype(name)
      .create_execution_context()            one "runner" of the engine
    context
      .set_tensor_address(name, ptr)         where in GPU memory a tensor lives (torch: tensor.data_ptr())
      .execute_async_v3(stream_handle)       run on a CUDA stream (torch: torch.cuda.current_stream().cuda_stream)

Design choice: PyTorch owns all GPU memory. Inputs/outputs are torch CUDA tensors, the engine runs on torch's
current stream, so torch.cuda.synchronize() in scripts/benchmark.py `time_fn` times an engine exactly like a model.

`tensorrt` must be imported LAZILY (inside functions): the rest of the project must still import on machines
without TensorRT (tests, the laptop without the package).

Self-check (tests/test_trt.py, skipped without TensorRT): tiny conv net → torch.onnx.export → build_engine → TRTModule → compare with PyTorch:
    FP32 engine: max |diff| < 1e-4      FP16 engine: max |diff| < 1e-2
"""

from pathlib import Path

import torch


def _trt():
    """Lazy import."""
    import tensorrt as trt

    return trt


_LOGGER = None


def _logger():
    """One shared trt.Logger; the first call also registers the built-in plugins."""
    global _LOGGER                                         # without this, assigning below makes _LOGGER local
    if _LOGGER is None:
        trt = _trt()
        _LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(_LOGGER, "")           # EfficientNMS_TRT etc. (YOLO engine)
    return _LOGGER


def build_engine(onnx_path: str | Path, engine_path: str | Path, fp16: bool = True, workspace_gb: float = 4.0) -> Path:
    """Parse an ONNX file and save a serialized TensorRT engine (static shapes, optional FP16). Returns engine_path."""
    # Core
    trt = _trt()
    logger = _logger()
    builder = trt.Builder(logger)
    network = builder.create_network(0)

    # ONNX parser
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errors}")

    # Config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Engine bytes
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path} (see the TensorRT log above)")

    # Write
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    return engine_path


class TRTModule:
    """Callable TensorRT engine: `outputs = module(*inputs)` — torch CUDA tensors in, dict {name: tensor} out.

    Output buffers are allocated ONCE and REUSED on every call → clone() them if you keep results across calls.
    """

    def __init__(self, engine_path: str | Path, device: str = "cuda") -> None:
        trt = _trt()
        self.device = torch.device(device)
        dtypes = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
            trt.bool: torch.bool,
            trt.int8: torch.int8,
        }

        # Load
        self.engine = trt.Runtime(_logger()).deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not load engine {engine_path} (built with another TensorRT version / GPU?)")
        # One runner of the engine
        self.context = self.engine.create_execution_context()

        # Inputs
        self.input_names: list[str] = []
        self.input_dtypes: list[torch.dtype] = []
        self.outputs: dict[str, torch.Tensor] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = dtypes[self.engine.get_tensor_dtype(name)]
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self.input_dtypes.append(dtype)
            else:
                buf = torch.empty(tuple(self.engine.get_tensor_shape(name)), dtype=dtype, device=self.device)
                self.outputs[name] = buf
                self.context.set_tensor_address(name, buf.data_ptr())
        self._inputs: list[torch.Tensor] = []

    def __call__(self, *inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        self._inputs = [
            x.to(self.device, dtype).contiguous() for x, dtype in zip(inputs, self.input_dtypes, strict=True)
        ]
        # Tell TensorRT
        for name, x in zip(self.input_names, self._inputs, strict=True):
            self.context.set_tensor_address(name, x.data_ptr())
        # Run on torch's current stream
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return self.outputs
