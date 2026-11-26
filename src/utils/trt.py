"""Minimal TensorRT 10 helpers: build an engine from ONNX, and run it on torch CUDA tensors.

PyTorch owns all GPU memory and the engine runs on torch's current CUDA stream, so engines can be timed
exactly like PyTorch models. `tensorrt` is imported lazily so the rest of the project works without it.
"""

from pathlib import Path

import torch


def _trt():
    """Import tensorrt lazily."""
    import tensorrt as trt

    return trt


_LOGGER = None


def _logger():
    """One shared trt.Logger; the first call also registers the built-in plugins."""
    global _LOGGER
    if _LOGGER is None:
        trt = _trt()
        _LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(_LOGGER, "")           # EfficientNMS_TRT etc. (YOLO engine)
    return _LOGGER


def build_engine(onnx_path: str | Path, engine_path: str | Path, fp16: bool = True, workspace_gb: float = 4.0) -> Path:
    """Parse an ONNX file and save a serialized TensorRT engine (static shapes, optional FP16). Returns engine_path."""
    trt = _trt()
    logger = _logger()
    builder = trt.Builder(logger)
    network = builder.create_network(0)

    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path} (see the TensorRT log above)")

    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    return engine_path


class TRTModule:
    """Callable TensorRT engine: torch CUDA tensors in, dict {output name: tensor} out.

    Output buffers are allocated once and reused on every call; clone() results that must outlive the next call.
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

        self.engine = trt.Runtime(_logger()).deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not load engine {engine_path} (built with another TensorRT version / GPU?)")
        self.context = self.engine.create_execution_context()

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
        for name, x in zip(self.input_names, self._inputs, strict=True):
            self.context.set_tensor_address(name, x.data_ptr())
        # Run on torch's current stream
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return self.outputs
