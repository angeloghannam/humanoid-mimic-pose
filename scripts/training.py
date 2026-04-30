import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from humanoid_mimic_pose.environments.mujoco.unitree_g1 import UnitreeG1
from humanoid_mimic_pose.training.callbacks import RewardLoggingCallback

if __name__ == "__main__":
    env = gym.make("UnitreeG1-v0")

    check_env(env)

    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        tensorboard_log="./ppo_unitree_tensorboard/"
    )

    callback = RewardLoggingCallback()

    # Train
    model.learn(total_timesteps=2_000_000, callback=callback)

    # Save model
    model.save("ppo_unitree_g1")

    env.close()
