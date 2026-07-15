from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class fixed_posture:
    """Penalize deviation from default pose with a fixed per-joint tolerance.

    Reward is exp(-mean(error^2 / std^2)) across the given joints. Map joint
    name patterns to std values via cfg.params["std"], e.g. {".*knee.*": 0.35}.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]

        default_joint_pos = asset.data.default_joint_pos
        if default_joint_pos is None:
            raise ValueError("Default joint position is None. Cannot compute fixed posture penalty.")

        self.default_joint_pos = default_joint_pos

        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
        _, _, std = resolve_matching_names_values(data=cfg.params["std"], list_of_strings=joint_names)

        std_t = torch.tensor(std, device=env.device, dtype=torch.float32)

        if not torch.all(std_t > 0):
            raise ValueError("Posture std must be strictly positive.")

        self.std = std_t

    def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, **_) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error_sq = torch.square(current_joint_pos - desired_joint_pos)
        return torch.exp(-torch.mean(error_sq / (self.std ** 2), dim=1))


def feet_air_time_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold_max: float = 0.1,
) -> torch.Tensor:
    """Penalize feet air time."""

    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data

    current_air_time = sensor_data.current_air_time

    if current_air_time is None:
        raise ValueError("Feet air time contact sensor returned None for current_air_time.")

    in_range = current_air_time > threshold_max
    penalty = torch.sum(in_range.float(), dim=1)
    in_air = current_air_time > 0
    num_in_air = torch.sum(in_air.float())
    mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
        num_in_air, min=1
    )
    env.extras["log"]["Metrics/air_time_mean"] = mean_air_time

    return penalty
