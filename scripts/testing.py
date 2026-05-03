import argparse
from stable_baselines3 import PPO
from humanoid_mimic_pose.environments.mujoco.unitree_g1 import UnitreeG1
import gymnasium as gym

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True
    )
    args, _ = parser.parse_known_args()

    env = gym.make("UnitreeG1-v0", render_mode="human", n_substeps=1)

    model = PPO.load(args.model_path)

    obs, _ = env.reset()

    try:
        while True:
            action, _ = model.predict(obs, deterministic=False)

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("Stopped.")

    env.close()
