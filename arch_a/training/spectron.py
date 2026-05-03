
from __future__ import annotations
import torch

def _spectral_norm_estimate(w: torch.Tensor, power_iters: int = 1):
    if w.ndim != 2:
        return None
    device = w.device
    dtype = w.dtype
    u = torch.randn(w.shape[0], 1, device=device, dtype=dtype)
    for _ in range(power_iters):
        v = torch.nn.functional.normalize(w.t() @ u, dim=0)
        u = torch.nn.functional.normalize(w @ v, dim=0)
    sigma = (u.t() @ w @ v).abs().squeeze()
    return sigma

@torch.no_grad()
def spectral_renormalize_model(model: torch.nn.Module, target_norm: float = 1.0, power_iters: int = 1):
    """
    Spectron: keep matrix weights in a controlled spectral envelope.
    """
    for p in model.parameters():
        if p.ndim != 2:
            continue
        sigma = _spectral_norm_estimate(p.data.float(), power_iters=power_iters)
        if sigma is None:
            continue
        sigma_val = float(sigma.detach().cpu())
        if sigma_val > target_norm and sigma_val > 0:
            scale = target_norm / sigma_val
            p.data.mul_(scale)
