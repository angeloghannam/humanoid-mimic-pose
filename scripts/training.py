from humanoid_mimic_pose.training.callbacks import RewardLoggingCallback
from humanoid_mimic_pose.environments.mujoco.unitree_g1 import UnitreeG1, DEFAULT_TRAINING_MODE
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from datetime import datetime
import argparse
import git
import os


N_PARALLEL_ENVS = 8

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--training_mode", type=str, required=False)
    parser.add_argument("--load_path", type=str, required=False, help="Path to a saved .zip model to continue training")
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

    if args.load_path:
        print(f"Loading existing model from: {args.load_path}")
        # Load the model and connect it to the new environment
        model = PPO.load(args.load_path, env=env, tensorboard_log=logging_path)
    else:
        print("Creating new model from scratch.")
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
    model.learn(total_timesteps=10_000_000, callback=callback, reset_num_timesteps=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(work_tree_dir, 'models', 'ppo_unitree_g1_' +
                              env_kwargs["training_mode"] + "_" + timestamp)

    # Save model
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    env.close()
