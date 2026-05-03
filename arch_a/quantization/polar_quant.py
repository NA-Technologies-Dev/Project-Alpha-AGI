
from __future__ import annotations
import math
import torch

def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

def hadamard_rotate(x: torch.Tensor):
    """
    Fast Walsh-Hadamard transform on the last dimension.
    Pads to next power of two if needed and trims back.
    """
    orig_shape = x.shape
    d = x.shape[-1]
    n = _next_pow2(d)
    if n != d:
        pad = n - d
        x = torch.nn.functional.pad(x, (0, pad))
    y = x.clone()
    h = 1
    while h < n:
        y = y.view(*y.shape[:-1], -1, 2, h)
        a = y[..., 0, :].clone()
        b = y[..., 1, :].clone()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        y = y.view(*y.shape[:-3], -1)
        h *= 2
    y = y / math.sqrt(n)
    return y[..., :d]

class PolarQuantizer:
    def __init__(self, bits: int = 4, block_size: int = 32):
        self.bits = bits
        self.block_size = block_size

    def quantize(self, x: torch.Tensor):
        x_rot = hadamard_rotate(x.float())
        flat = x_rot.flatten()
        n = flat.numel()
        pad = (-n) % self.block_size
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        blocks = flat.view(-1, self.block_size)
        scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
        levels = 2 ** (self.bits - 1) - 1
        q = torch.clamp((blocks / scale) * levels, -levels - 1, levels).round().to(torch.int8)
        return q, scale.to(torch.float16), x.shape

    def dequantize(self, q: torch.Tensor, scale: torch.Tensor, shape):
        levels = 2 ** (self.bits - 1) - 1
        flat = (q.to(torch.float32) / levels) * scale.to(torch.float32)
        flat = flat.flatten()[: int(torch.tensor(shape).prod())]
        return flat.view(*shape)
