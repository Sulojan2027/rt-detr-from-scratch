import torch
import torch.nn as nn

_ACTIVATIONS: dict[str, nn.Module] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU
}

def get_activation(activation: str| None | nn.Module) -> nn.Module:
    if activation is None:
        return None
    if isinstance(activation, nn.Module):
        return activation
    if isinstance(activation, str):
        activation = activation.lower()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Valid options: {sorted(_ACTIVATIONS)}"
            )
        return _ACTIVATIONS[activation]()
    raise TypeError(f"activation must be str, nn.Module or None, got {type(act).__name__}")
    
class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        # TODO: 4 buffers + store num_features, eps
        self.register_buffer("weight", torch.ones(num_features, num_features))
        self.register_buffer("bias", torch.zeros(num_features, 1))
        self.register_buffer("running_mean", torch.zeros(num_features, 1))
        self.register_buffer("running_variance", torch.ones(num_features, 1))
        self.num_features = num_features
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # TODO: drop prefix + "num_batches_tracked" if present, then call super
        

    def forward(self, x: Tensor) -> Tensor:
        # TODO: compute scale & shift, reshape to [1, C, 1, 1], return x * scale + shift
        

    def extra_repr(self) -> str:
        ...