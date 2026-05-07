from dataclasses import dataclass

@dataclass
class TrainConfig:
    batch_size: int = 8
    grad_accum_steps: int = 4
    lr: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    max_steps: int = 100000
    eval_every: int = 1000
    save_every: int = 5000
    seed: int = 42
    mixed_precision: str = "bf16"
    compile: bool = False
