import math
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict
from importlib.resources import files

from humanoid_mimic_pose.utils.utils import angle_between

from humanoid_mimic_pose.environments.mujoco.mujoco_base_env import MujocoRobotEnv

DEFAULT_MODEL_PATH = str(files("humanoid_mimic_pose.assets") / "scene_mjx.xml")
DEFAULT_N_ACTIONS = 29
DEFAULT_N_SUBSTEPS = 5

SUPPORTED_TRAINING_MODES = ['stand', 'walk']
DEFAULT_TRAINING_MODE = 'stand'

### --- REWARD CONSTANTS --- ###
IDEAL_TORSO_HEIGHT = 0.8    # m
MIN_TERMINATION_HEIGHT = 0.5    # m
MAX_TORSO_VEL = 2.5     # m/s
MAX_TORSO_ANG_VEL = 10.0     # rad/s


class UnitreeG1(MujocoRobotEnv):

    def __init__(self,
                 model_path: str = DEFAULT_MODEL_PATH,
                 n_actions: int = DEFAULT_N_ACTIONS,
                 n_substeps: int = DEFAULT_N_SUBSTEPS,
                 training_mode: str = DEFAULT_TRAINING_MODE,
                 **kwargs):
        """Unitree G1 Mujoco/Gymnasium environment

        Args:
            model_path (str, optional): Path to mjcf mujoco model. Defaults to DEFAULT_MODEL_PATH.
            n_actions (int, optional): Size of the action space. Defaults to DEFAULT_N_ACTIONS.
            n_substeps (int, optional): Number of mujoco simulation timesteps per Gymnasium step. Defaults to DEFAULT_N_SUBSTEPS.
            initial_qpos (NDArray, optional): Initial joint positions. Defaults to None.
        """

        super().__init__(
            model_path=model_path,
            n_actions=n_actions,
            n_substeps=n_substeps,
            **kwargs)

        if training_mode not in SUPPORTED_TRAINING_MODES:
            raise ValueError(
                f"Training mode {training_mode} not supported. \n Supported modes are {SUPPORTED_TRAINING_MODES}")

        self.training_mode = training_mode

    def get_null_action(self) -> NDArray:
        return np.zeros(self.model.nu)

    def compute_robot_reward(self) -> float:
        match self.training_mode:
            case 'stand':
                goal_reward, components = self.stand_reward()
            case _:
                raise ValueError(
                    f"Training mode {self.training_mode} not supported. \n Supported modes are {SUPPORTED_TRAINING_MODES}")

        healthy_reward = self._healthy_reward()
        action_penalty = self._action_magnitude_penalty()

        total_reward = 0.7 * goal_reward + 0.2 * healthy_reward + 0.1 * action_penalty

        self._last_reward_components = {
            **components,
            "healthy": healthy_reward,
            "action_penalty": action_penalty,
            "total": total_reward,
        }

        return total_reward

    def stand_reward(self) -> Tuple[float, Dict[str, float]]:
        # 6D velocity: [angular (wx, wy, wz), linear (vx, vy, vz)]
        cvel = self.data.body("torso_link").cvel

        torso_angular_velocity = cvel[:3]
        torso_linear_velocity = cvel[3:]

        vx, vy, _ = torso_linear_velocity
        vel_penalty = -np.clip((abs(vx) + abs(vy)) / MAX_TORSO_VEL, 0.0, 1.0)

        torso_angular_velocity_norm = np.linalg.norm(torso_angular_velocity)
        ang_penalty = -np.clip(torso_angular_velocity_norm / MAX_TORSO_ANG_VEL, 0.0, 1.0)

        upright_reward = self._upright_reward()

        contact_reward = self._foot_contact_reward()

        stand_reward = (
            0.3 * upright_reward +
            0.3 * contact_reward +
            0.2 * vel_penalty +
            0.2 * ang_penalty
        )

        components = {
            "upright": upright_reward,
            "contact": contact_reward,
            "vel_penalty": vel_penalty,
            "ang_penalty": ang_penalty,
        }

        return stand_reward, components

    def _action_magnitude_penalty(self) -> float:
        action = self.data.ctrl
        action_magnitude = np.linalg.norm(action)
        n_actuators = self.model.nu
        max_action_magnitude = math.sqrt(n_actuators)
        normalized_penalty = - action_magnitude / max_action_magnitude
        return normalized_penalty

    def _healthy_reward(self) -> float:
        height = self.data.body("torso_link").xpos[2]
        height_margin = height - MIN_TERMINATION_HEIGHT
        ideal_height_margin = IDEAL_TORSO_HEIGHT - MIN_TERMINATION_HEIGHT
        height_reward = height_margin / ideal_height_margin
        return np.clip(height_reward, 0, 1)

    def _upright_reward(self) -> float:
        torso_z = self.__get_torso_orientation()
        world_z = np.array([0, 0, 1])
        upright_reward = math.cos(angle_between(torso_z, world_z))
        return np.clip(upright_reward, 0, 1)

    def _foot_contact_reward(self) -> float:
        # IDs for floor and foot geoms
        floor_id = self.model.geom("floor").id

        left_geom_ids = [
            self.model.geom("left_foot1_collision").id,
            self.model.geom("left_foot2_collision").id,
            self.model.geom("left_foot3_collision").id,
        ]

        right_geom_ids = [
            self.model.geom("right_foot1_collision").id,
            self.model.geom("right_foot2_collision").id,
            self.model.geom("right_foot3_collision").id,
        ]

        # Track which geoms are in contact
        left_contacts = set()
        right_contacts = set()

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2

            # Check left foot contacts
            if (g1 in left_geom_ids and g2 == floor_id) or (g2 in left_geom_ids and g1 == floor_id):
                left_contacts.add(g1 if g1 in left_geom_ids else g2)

            # Check right foot contacts
            if (g1 in right_geom_ids and g2 == floor_id) or (g2 in right_geom_ids and g1 == floor_id):
                right_contacts.add(g1 if g1 in right_geom_ids else g2)

        left_ratio = len(left_contacts) / len(left_geom_ids)
        right_ratio = len(right_contacts) / len(right_geom_ids)

        contact_reward = 0.5 * left_ratio + 0.5 * right_ratio

        return contact_reward

    def is_terminated(self) -> bool:
        height = self.data.body("torso_link").xpos[2]
        return bool(height < MIN_TERMINATION_HEIGHT)

    def _get_obs(self) -> NDArray:

        # --- Pelvis and Joints ---
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        qpos = qpos[2:]  # keep height + joints

        # --- Torso ---
        cvel = self.data.body("torso_link").cvel
        torso_angular_velocity = cvel[:3]
        torso_linear_velocity = cvel[3:]

        torso_z = self.__get_torso_orientation()

        obs = np.concatenate([
            qpos,
            qvel,
            torso_z,
            torso_angular_velocity,
            torso_linear_velocity
        ])

        return obs.astype(np.float32)

    def __get_torso_orientation(self) -> NDArray:
        torso_trsf = self.data.body("torso_link").xmat.reshape(3, 3)
        torso_z = torso_trsf[:, 2]
        return torso_z
