from gymnasium.envs.registration import register

register(
    id="UnitreeG1-v0",
    entry_point="humanoid_mimic_pose.environments.mujoco.unitree_g1:UnitreeG1",
    max_episode_steps=1000,
)
