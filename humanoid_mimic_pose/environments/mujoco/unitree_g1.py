from numpy import zeros
from numpy.typing import NDArray
from importlib.resources import files

from humanoid_mimic_pose.environments.mujoco.mujoco_base_env import MujocoRobotEnv

DEFAULT_MODEL_PATH = str(files("humanoid_mimic_pose.assets") / "scene_mjx.xml")
DEFAULT_N_ACTIONS = 29
DEFAULT_N_SUBSTEPS = 5
MIN_TERMINATION_HEIGHT = 0.5    # meters


class UnitreeG1(MujocoRobotEnv):

    def __init__(self,
                 model_path: str = DEFAULT_MODEL_PATH,
                 n_actions: int = DEFAULT_N_ACTIONS,
                 n_substeps: int = DEFAULT_N_SUBSTEPS,
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

    def get_null_action(self) -> NDArray:
        return zeros(self.model.nu)

    def compute_reward(self) -> float:
        height = self.data.body("torso_link").xpos[2]
        return height

    def is_terminated(self) -> bool:
        height = self.data.body("torso_link").xpos[2]
        print(f"height: {height}")
        return height < MIN_TERMINATION_HEIGHT
