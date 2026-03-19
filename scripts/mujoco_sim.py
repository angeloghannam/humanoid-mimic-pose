import gymnasium as gym
from humanoid_mimic_pose.environments.mujoco.unitree_g1 import UnitreeG1

if __name__ == "__main__":

    # env = UnitreeG1(render_mode="human")
    env = gym.make("UnitreeG1-v0", render_mode="human")

    obs, _ = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):

        action = env.action_space.sample()  # or env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(obs)

    env.close()
