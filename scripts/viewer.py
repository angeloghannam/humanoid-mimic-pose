import mujoco
from mujoco import viewer

from humanoid_mimic_pose.environments.robot.robot_model import Robot

if __name__ == '__main__':
    robot = Robot()

    with viewer.launch_passive(robot.model, robot.data) as mj_viewer:
        mj_viewer.sync()

        while mj_viewer.is_running():
            # Step the physics.
            mujoco.mj_step(robot.model, robot.data)
            mj_viewer.sync()
