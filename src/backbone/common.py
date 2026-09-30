import torch
import torch.nn as nn
from torch import Tensor

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU
}

def get_activation(activation: str| None | nn.Module) -> nn.Module:
    if activation is None:
        return nn.Identity()
    if isinstance(activation, nn.Module):
        return activation
    if isinstance(activation, str):
        act = activation.lower()
        if act not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Valid options: {sorted(_ACTIVATIONS)}"
            )
        return _ACTIVATIONS[act]()
    raise TypeError(f"activation must be str, nn.Module or None, got {type(activation).__name__}")
    
class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.num_features = num_features
        self.eps = eps

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str, 
        *args, 
        **kwargs
    ) -> None:
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
            
        super()._load_from_state_dict(
            state_dict, prefix, *args, **kwargs
        )
        

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        
        scale = weight * (rv + self.eps).rsqrt()
        shift = bias - rm * scale
        
        return x * scale + shift

    def extra_repr(self) -> str:
        return f"{self.num_features}, eps={self.eps}"

class ConvNormLayer(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        act: str | nn.Module | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)

def freeze_batch_norm2d(module: nn.Module) -> nn.Module:
    """Recursively replace every nn.BatchNorm2d with FrozenBatchNorm2d.

    Copies weight/bias/running stats/eps so it works before or after
    loading pretrained weights. Returns the (possibly new) module:
        model = freeze_batch_norm2d(model)
    """
    
    # Check if module is standard BatchNorm
    if isinstance(module, nn.BatchNorm2d):
        frozen_bn = FrozenBatchNorm2d(module.num_features, eps=module.eps)
        
        ref = module.running_mean if module.running_mean is not None else module.weight
        if ref is not None:
            frozen_bn = frozen_bn.to(device=ref.device, dtype=ref.dtype)
        
        with torch.no_grad():
            if module.weight is not None:
                frozen_bn.weight.copy_(module.weight)
                frozen_bn.bias.copy_(module.bias)
            if module.running_mean is not None:
                frozen_bn.running_mean.copy_(module.running_mean)
                frozen_bn.running_var.copy_(module.running_var)
            
        return frozen_bn
        
    # Check with children and update recursively
    for name, child in module.named_children():
        new_child = freeze_batch_norm2d(child)
        if new_child is not child:
            setattr(module, name, new_child)
            
    return module
