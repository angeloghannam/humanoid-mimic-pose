import numpy as np

from typing import Optional, Dict, Tuple
from numpy.typing import NDArray

from gymnasium import spaces, Env

DEFAULT_SIZE = 480
MAX_EPISODE_STEPS = 1000


class BaseRobotEnv(Env):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 25,
    }

    def __init__(
        self,
        n_actions: int,
        render_mode: Optional[str] = "rgb_array",
        max_episode_steps: int = MAX_EPISODE_STEPS,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
    ):
        """Initialize robot gym environment.

        Args:
            model_path (str): Path to mjcf mujoco model.
            initial_qpos (NDArray): Initial joint positions.
            n_actions (int): Size of the action space.
            render_mode (Optional[str], optional): type of rendering mode, "human" for window rendeirng and "rgb_array" for offscreen. Defaults to "rgb_array".
            width (int, optional): width of each rendered frame. Defaults to DEFAULT_SIZE.
            height (int, optional): height of each rendered frame. Defaults to DEFAULT_SIZE.
        """
        super(BaseRobotEnv, self).__init__()

        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        self._last_rewards: Dict = {}

        self._initialize_simulation()

        self._last_action: NDArray = np.zeros(self.model.nu, dtype=np.float32)

        obs = self._get_obs()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)

    # Env methods
    # ----------------------------

    def step(self, action: NDArray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (NDArray): Control action to be applied to the agent and update the simulation.

        Returns:
            observation (Dict): Next observation due to the agent actions.
            reward (float): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state. 
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
            info (Dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
        """
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._mujoco_step(action)
        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        reward = self.compute_robot_reward()

        self._step_count += 1

        terminated = self.is_terminated()
        truncated = self.is_truncated()

        info = {}

        if truncated:
            info["TimeLimit.truncated"] = True

        info["reward_components"] = self._last_reward_components

        self._last_action = np.copy(action)

        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

        self._reset_sim()
        obs = self._get_obs()

        self._last_action = np.zeros(self.model.nu, dtype=np.float32)

        if self.render_mode == "human":
            self.render()

        self._step_count = 0

        return obs, {}

    # --- methods to implement
    def _initialize_simulation(self):
        raise NotImplementedError

    def _reset_sim(self):
        raise NotImplementedError

    def _mujoco_step(self, action: NDArray):
        raise NotImplementedError

    def _get_obs(self) -> NDArray:
        raise NotImplementedError

    def compute_robot_reward(self) -> float:
        raise NotImplementedError

    def is_terminated(self):
        raise NotImplementedError

    def is_truncated(self):
        return self._step_count >= self.max_episode_steps
