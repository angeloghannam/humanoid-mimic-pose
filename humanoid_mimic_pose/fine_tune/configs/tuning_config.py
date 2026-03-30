from __future__ import annotations

import yaml

from pprint import pprint
from dacite import from_dict
from dataclasses import dataclass, field

from humanoid_mimic_pose.fine_tune.configs.ppo_config import PPOConfig


@dataclass(frozen=True)
class TuningConfig:
    env: str = "UnitreeG1-v0"
    n_trials: int = 25
    n_startup_trials: int = 5
    n_evaluations_per_trial: int = 4
    n_timesteps_per_trial: int = 100_000
    n_eval_episodes: int = 10
    n_parallel_envs: int = 5

    ppo_config: PPOConfig = field(default_factory=PPOConfig)

    save_study: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> TuningConfig:
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        config = from_dict(data_class=cls, data=data)

        print(f"[INFO] Loaded config from: {path}")
        pprint(config)

        return config
