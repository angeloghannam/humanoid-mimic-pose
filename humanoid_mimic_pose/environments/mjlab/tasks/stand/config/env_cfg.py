"""Unitree G1 stand environment configuration."""

from mjlab.asset_zoo.robots import (
    G1_ACTION_SCALE,
    get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
)
from mjlab.tasks.velocity import mdp as velocity_mdp

from ..stand_env_cfg import make_stand_env_cfg


def unitree_g1_stand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 flat-terrain stand configuration."""
    cfg = make_stand_env_cfg()

    # Flat-terrain sim limits (standing is always on a plane).
    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    cfg.scene.entities = {"robot": get_g1_robot_cfg()}

    geom_names = tuple(
        f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
    )

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
    )

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE

    cfg.viewer.body_name = "torso_link"

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

    cfg.rewards["pose"].params["std"] = {
        # Lower body.
        r".*hip_pitch.*": 0.05,
        r".*hip_roll.*": 0.05,
        r".*hip_yaw.*": 0.05,
        r".*knee.*": 0.05,
        r".*ankle_pitch.*": 0.05,
        r".*ankle_roll.*": 0.05,
        # Waist.
        r".*waist_yaw.*": 0.05,
        r".*waist_roll.*": 0.05,
        r".*waist_pitch.*": 0.05,
        # Arms.
        r".*shoulder_pitch.*": 0.05,
        r".*shoulder_roll.*": 0.05,
        r".*shoulder_yaw.*": 0.05,
        r".*elbow.*": 0.05,
        r".*wrist.*": 0.05,
    }
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

    cfg.rewards["body_ang_vel"].weight = -0.1
    cfg.rewards["angular_momentum"].weight = -0.05
    cfg.rewards["air_time"].weight = -0.1

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=velocity_mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
    )

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

    return cfg