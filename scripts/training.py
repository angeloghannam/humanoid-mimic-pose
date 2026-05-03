import os
import git
import argparse
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from humanoid_mimic_pose.environments.mujoco.unitree_g1 import UnitreeG1, DEFAULT_TRAINING_MODE
from humanoid_mimic_pose.training.callbacks import RewardLoggingCallback

N_PARALLEL_ENVS = 8

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--training_mode",
        type=str,
        required=False
    )
    args, _ = parser.parse_known_args()

    if args.training_mode is not None:
        env_kwargs = {"training_mode": args.training_mode}
    else:
        env_kwargs = {"training_mode": DEFAULT_TRAINING_MODE}

    env = make_vec_env("UnitreeG1-v0",
                       n_envs=N_PARALLEL_ENVS,
                       env_kwargs=env_kwargs,
                       vec_env_cls=SubprocVecEnv)

    work_tree_dir = git.Repo('.', search_parent_directories=True).working_tree_dir

    logging_path = os.path.join(work_tree_dir, 'logging', 'ppo_unitree_tensorboard')

    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        gamma=0.99,
        tensorboard_log=logging_path
    )

    callback = RewardLoggingCallback()

    # Train
    model.learn(total_timesteps=10_000_000, callback=callback)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = os.path.join(work_tree_dir, 'models', 'ppo_unitree_g1_' + env_kwargs["training_mode"] + timestamp)

    # Save model
    model.save(model_path)

    env.close()
