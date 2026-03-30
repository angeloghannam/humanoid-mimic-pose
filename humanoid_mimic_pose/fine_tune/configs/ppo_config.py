from dataclasses import dataclass, field
from typing import List


@dataclass
class PPOConfig:
    gamma_range: List[float] = field(default_factory=lambda: [0.97, 0.999])
    n_steps_pow_range: List[float] = field(default_factory=lambda: [5, 10])
    lr_range: List[float] = field(default_factory=lambda: [3e-5, 3e-3])
    activation_func: List[str] = field(
        default_factory=lambda: ["tanh", "relu"])
