import math
import numpy as np
from numpy.typing import NDArray
from importlib.resources import files

from humanoid_mimic_pose.utils.utils import angle_between

from humanoid_mimic_pose.environments.mujoco.mujoco_base_env import MujocoRobotEnv

DEFAULT_MODEL_PATH = str(files("humanoid_mimic_pose.assets") / "scene_mjx.xml")
DEFAULT_N_ACTIONS = 29
DEFAULT_N_SUBSTEPS = 5
MIN_TERMINATION_HEIGHT = 0.5    # meters

SUPPORTED_TRAINING_MODES = ['stand', 'walk']
DEFAULT_TRAINING_MODE = 'stand'

TARGET_VELOCITY = 1.0  # m/s


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
                goal_reward = self.stand_reward()
            case 'walk':
                goal_reward = self.walk_reward()
            case _:
                raise ValueError(
                    f"Training mode {self.training_mode} not supported. \n Supported modes are {SUPPORTED_TRAINING_MODES}")

        optimizing_reward = self.optimizing_reward()

        return goal_reward + optimizing_reward

    def stand_reward(self) -> float:
        # 6D velocity: [angular (wx, wy, wz), linear (vx, vy, vz)]
        cvel = self.data.body("torso_link").cvel

        torso_angular_velocity = np.linalg.norm(cvel[:3])
        torso_linear_velocity = np.linalg.norm(cvel[3:])

        linear_reward = 1 + math.tanh(-torso_linear_velocity)
        angular_reward = 1 + math.tanh(-torso_angular_velocity)

        orientation_reward = self._upright_reward()

        return (linear_reward + angular_reward + orientation_reward) / 3

    def walk_reward(self) -> float:
        torso_x_vel = self.data.body("torso_link").cvel[3]
        velocity_reward = math.tanh(torso_x_vel)
        orientation_reward = self._upright_reward()

        return (velocity_reward + orientation_reward) / 2

    def optimizing_reward(self) -> float:
        return self._action_magnitude_reward() + self._healthy_reward()

    def _action_magnitude_reward(self) -> float:
        return math.tanh(-np.linalg.norm(self.data.ctrl))

    def _healthy_reward(self) -> float:
        height = self.data.body("torso_link").xpos[2]
        height_margin = height - MIN_TERMINATION_HEIGHT
        height_reward = math.tanh(5.0 * height_margin)

        if self.is_terminated():
            return -1.0

        return height_reward

    def _upright_reward(self) -> float:
        torso_trsf = self.data.body("torso_link").xmat.reshape(3, 3)
        torso_z = torso_trsf[:, 2]
        world_z = np.array([0, 0, 1])
        return 1 + math.tanh(-angle_between(torso_z, world_z))

    def is_terminated(self) -> bool:
        height = self.data.body("torso_link").xpos[2]
        return bool(height < MIN_TERMINATION_HEIGHT)
