from __future__ import annotations
import torch
from arch_a import ArchAConfig, ArchAForCausalLM
from arch_a.quantization import ShadowResidualQuantizer, PolarQuantizer
from arch_a.training import (ScaleOptimizer, project_gradients_galore2,
                              spectral_renormalize_model,
                              blockwise_mxfp8_quantize, blockwise_mxfp8_dequantize)

def run():
    cfg   = ArchAConfig.for_debug()
    model = ArchAForCausalLM(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    y = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model(input_ids=x, labels=y, training_mode="hybrid", output_hidden_states=True)
    assert out.logits.shape == (1, 4, cfg.vocab_size)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.hidden_states is not None
    assert len(out.ssm_states) == cfg.n_layers
    assert all(isinstance(v, int) for v in out.algr_meta.loops)
    out_ar = model(input_ids=x, labels=y, training_mode="ar")
    assert torch.isfinite(out_ar.loss)
    out_nd = model(input_ids=x, labels=y, training_mode="nadd")
    assert torch.isfinite(out_nd.loss)
    qz = ShadowResidualQuantizer(bits=4)
    q, r, s, shape = qz.quantize(torch.randn(3, 17))
    assert qz.dequantize(q, r, s, shape).shape == (3, 17)
    pq = PolarQuantizer(bits=4)
    q2, s2, sh2 = pq.quantize(torch.randn(4, 16))
    assert pq.dequantize(q2, s2, sh2).shape == (4, 16)
    tensor = torch.randn(4, 64)
    q3, s3, sh3 = blockwise_mxfp8_quantize(tensor, block_size=32)
    assert blockwise_mxfp8_dequantize(q3, s3, sh3, block_size=32).shape == tensor.shape
    params = list(model.parameters())
    opt = ScaleOptimizer([
        {"params": params[:4],  "critical": True,  "lr_critical": 1e-4, "lr_noncritical": 3e-4},
        {"params": params[4:],  "critical": False, "lr_critical": 1e-4, "lr_noncritical": 3e-4},
    ])
    out.loss.backward()
    project_gradients_galore2(model, rank=2)
    spectral_renormalize_model(model, target_norm=2.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    model.eval()
    gen = model.generate_ar(x, max_new_tokens=4)
    assert gen.shape == (1, 8)
    print("✓ All tests passed")

if __name__ == "__main__":
    run()
