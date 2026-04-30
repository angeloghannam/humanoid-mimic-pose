from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class RewardLoggingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        # Temporary storage for the current episode's steps
        self.episode_storage = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if "reward_components" in info:
                comps = info["reward_components"]
                for k, v in comps.items():
                    if k not in self.episode_storage:
                        self.episode_storage[k] = []
                    self.episode_storage[k].append(v)

            # Only log to TensorBoard when the episode actually finishes
            if dones[i] and self.episode_storage:
                for k, v_list in self.episode_storage.items():
                    # Record the MEAN of the components over the whole episode
                    self.logger.record(f"reward_mean/{k}", np.mean(v_list))

                # Clear storage for the next episode
                self.episode_storage = {}

                # Optional: Force a write so you see it immediately
                # self.logger.dump(step=self.num_timesteps)

        return True
