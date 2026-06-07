import numpy as np

from typing import Dict, Optional
from numpy.typing import NDArray

import mujoco
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from humanoid_mimic_pose.environments.robot_base_env import BaseRobotEnv

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.0,
    "lookat": [0, 0, 0.5],
    "elevation": -30,
    "azimuth": 45,
}


class MujocoRobotEnv(BaseRobotEnv):

    """
     Mujoco robot base class.
     """

    def __init__(self, model_path: str, n_substeps: int, default_camera_config: Dict = None, **kwargs):
        """
        Args:
            model_path (str): Path to mjcf mujoco model.
            n_substeps (int): Number of mujoco simulation timesteps per Gymnasium step.
            default_camera_config (Dict, optional): dictionary of default mujoco camera parameters for human rendering. Defaults to DEFAULT_CAMERA_CONFIG.. Defaults to DEFAULT_CAMERA_CONFIG.
        """

        self.model_path = model_path
        self.n_substeps = n_substeps

        super().__init__(**kwargs)

        if default_camera_config is None:
            default_camera_config = DEFAULT_CAMERA_CONFIG

        if self.render_mode is not None:
            self.mujoco_renderer = MujocoRenderer(
                self.model,
                self.data,
                default_camera_config,
                width=self.width,
                height=self.height,
            )
        else:
            self.mujoco_renderer = None

    def _initialize_simulation(self):
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        self.ctrl_range = self.model.actuator_ctrlrange

        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        self.initial_time = self.data.time
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)

    def _reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)

        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        mujoco.mj_forward(self.model, self.data)

    def render(self) -> Optional[NDArray]:
        """Render a frame of the MuJoCo simulation.

        Returns:
            rgb image (NDArray): if render_mode is "rgb_array", return a 3D image array.
        """
        if self.mujoco_renderer is not None:
            return self.mujoco_renderer.render(self.render_mode)
        else:
            return None

    def close(self):
        """Close contains the code necessary to "clean up" the environment.

        Terminates any existing WindowViewer instances in the Gymnaisum MujocoRenderer.
        """
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    @property
    def dt(self) -> float:
        """Return the timestep of each Gymanisum step."""
        return self.model.opt.timestep * self.n_substeps

    def _mujoco_step(self, action: NDArray):
        """Apply action command and step the simulation.

        Args:
            action (NDArray): Should be of shape (nac)
        """
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    def _default_qpos(self):
        return np.zeros(self.n_actions)

    def get_body_pos(self, name: str) -> NDArray:
        return self.data.body(name).xpos

    def get_body_vel(self, name: str) -> NDArray:
        return self.data.body(name).cvel
