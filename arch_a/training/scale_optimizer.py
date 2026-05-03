
from __future__ import annotations
from typing import Iterable, Optional
import torch

class ScaleOptimizer(torch.optim.Optimizer):
    """
    SCALE optimizer:
    - critical params: momentum + decoupled weight decay
    - noncritical params: plain SGD step
    This keeps optimizer state minimal.
    """
    def __init__(self, params, lr_critical=1e-4, lr_noncritical=3e-4, momentum=0.9, weight_decay=0.0):
        defaults = dict(lr_critical=lr_critical, lr_noncritical=lr_noncritical,
                        momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr_critical"] if group.get("critical", False) else group["lr_noncritical"]
            wd = group.get("weight_decay", 0.0)
            mom = group.get("momentum", 0.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                if grad.is_sparse:
                    raise RuntimeError("ScaleOptimizer does not support sparse gradients.")
                if group.get("critical", False):
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                    buf = state["momentum_buffer"]
                    buf.mul_(mom).add_(grad)
                    update = buf
                else:
                    update = grad
                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(update.to(p.dtype), alpha=-lr)
        return loss
