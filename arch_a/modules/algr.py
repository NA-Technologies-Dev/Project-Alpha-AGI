from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import torch
from torch import nn
from .alpha_window import AlphaWindow, AlphaWindowState
from .normalization import RMSNorm

@dataclass
class ALGRMeta:
    loops: List[int]
    halt_prob: List[float]
    entropy: List[float]
    confidence: List[float]

class ALGRBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_head: int,
        ssm_state_dim: int,
        alpha_window: int,
        d_ff: int,
        rope_theta: float,
        norm_eps: float,
        dropout: float,
        use_bias: bool,
        residual_fp32: bool,
        summary_dim: int,
    ):
        super().__init__()
        self.alpha_window = AlphaWindow(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            ssm_state_dim=ssm_state_dim,
            alpha_window=alpha_window,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            attn_dropout=dropout,
            use_bias=use_bias,
            residual_fp32=residual_fp32,
            summary_dim=summary_dim,
        )
        self.norm2 = RMSNorm(d_model, norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=use_bias),
            nn.SiLU(),
            nn.Linear(d_ff, d_model, bias=use_bias),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, ssm_state: Optional[AlphaWindowState] = None, loop_idx: int = 0):
        residual = x
        x, new_state = self.alpha_window(x, ssm_state)
        x = residual + self.dropout(x)
        residual2 = x
        x = self.mlp(self.norm2(x))
        x = residual2 + self.dropout(x)
        return x, new_state

class ALGRController(nn.Module):
    """
    Adaptive Logic-Gated Recurrence controller.
    The controller loops each block until a halting gate crosses a threshold.
    """
    def __init__(
        self,
        layers: nn.ModuleList,
        d_model: int,
        max_loops: int = 3,
        confidence_threshold: float = 0.82,
        temperature: float = 1.0,
        device_map: Optional[Sequence[torch.device]] = None,
    ):
        super().__init__()
        self.layers = layers
        self.max_loops = max_loops
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.device_map = list(device_map) if device_map is not None else None
        self.halt_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

    def _move_optional_state(self, state, device):
        if state is None:
            return None
        if isinstance(state, tuple):
            return tuple(s.to(device, non_blocking=True) if s is not None else None for s in state)
        return state.to(device, non_blocking=True)

    def forward(self, layers, x, ssm_states, training: bool = True):
        if ssm_states is None:
            ssm_states = [None] * len(layers)
        meta_loops, meta_halt, meta_entropy, meta_conf = [], [], [], []

        for i, layer in enumerate(layers):
            dev = self.device_map[i] if self.device_map is not None else x.device
            if x.device != dev:
                x = x.to(dev, non_blocking=True)
            ssm_states[i] = self._move_optional_state(ssm_states[i], dev)

            # keep halting head on the same device as the activations
            halt_dev = x.device
            if next(self.halt_head.parameters()).device != halt_dev:
                self.halt_head.to(halt_dev)

            loops = 0
            halted = False
            conf = 0.0
            entropy = 0.0
            layer_input = x
            while True:
                x, ssm_states[i] = layer(layer_input, ssm_states[i], loop_idx=loops)
                pooled = x.float().mean(dim=1)
                halt_logit = self.halt_head(pooled).squeeze(-1) / max(self.temperature, 1e-6)
                prob = torch.sigmoid(halt_logit)
                conf = float(prob.mean().detach().cpu())
                p = prob.clamp(1e-6, 1 - 1e-6)
                ent = (-p * torch.log(p) - (1 - p) * torch.log(1 - p)).mean()
                entropy = float(ent.detach().cpu())
                loops += 1
                if (prob >= self.confidence_threshold).all() or loops >= self.max_loops:
                    halted = True
                    break
                # Only the SSM state carries forward recurrently;
                # the residual input to the layer remains fixed.
            meta_loops.append(loops)
            meta_halt.append(float(halted))
            meta_entropy.append(entropy)
            meta_conf.append(conf)

        return x, ssm_states, ALGRMeta(meta_loops, meta_halt, meta_entropy, meta_conf)
