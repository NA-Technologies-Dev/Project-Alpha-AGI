from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from .normalization import RMSNorm

def semantic_global_anchor(logits: torch.Tensor, anchor: Optional[torch.Tensor] = None, strength: Optional[torch.Tensor] = None):
    if anchor is None:
        return logits
    if anchor.dim() == logits.dim() - 1:
        anchor = anchor.unsqueeze(1)
    if strength is None:
        return logits + anchor
    while strength.dim() < logits.dim():
        strength = strength.unsqueeze(-1)
    return logits + strength * anchor

class NADDRefiner(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 2, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        inner = d_model * hidden_mult
        self.norm = RMSNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, inner, bias=use_bias),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inner, d_model, bias=use_bias),
        )
        self.gate = nn.Linear(d_model, d_model, bias=use_bias)

    def forward(self, x: torch.Tensor):
        x_norm = self.norm(x)
        delta = self.net(x_norm)
        g = torch.sigmoid(self.gate(x_norm))
        return x + g * delta

class NADDDecoder(nn.Module):
    """
    Non-autoregressive diffusion-style iterative refinement decoder.
    Works as a denoising latent refinement stack, not a discrete diffusion solver.
    """
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        steps: int = 6,
        hidden_mult: int = 2,
        dropout: float = 0.0,
        use_bias: bool = False,
        noise_scale: float = 0.15,
    ):
        super().__init__()
        self.steps = steps
        self.noise_scale = noise_scale
        self.refiners = nn.ModuleList([NADDRefiner(d_model, hidden_mult, dropout, use_bias) for _ in range(steps)])
        self.out_norm = RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.anchor_proj = nn.Linear(d_model, vocab_size, bias=use_bias)

    def corrupt(self, x: torch.Tensor):
        if not self.training or self.noise_scale <= 0:
            return x
        noise = torch.randn_like(x) * self.noise_scale
        return x + noise

    def forward(self, x: torch.Tensor, anchor_state: Optional[torch.Tensor] = None):
        x = x.to(self.out_proj.weight.dtype)
        x = self.corrupt(x)
        for refiner in self.refiners:
            x = refiner(x)
        logits = self.out_proj(self.out_norm(x))
        if anchor_state is not None:
            anchor_state = anchor_state.to(x.dtype)
            anchor = self.anchor_proj(anchor_state).unsqueeze(1)  # [B,1,V]
            strength = torch.sigmoid(anchor_state.float().mean(dim=-1, keepdim=True)).unsqueeze(-1)  # [B,1,1]
            logits = semantic_global_anchor(logits, anchor, strength)
        return x, logits
