import mujoco
from mujoco import viewer

from humanoid_mimic_pose.environments.robot.robot_model import Robot

if __name__ == '__main__':
    robot = Robot()

    # --- Find keyframe id ---
    key_id = None
    for i in range(robot.model.nkey):
        if robot.model.key(i).name == "knees_bent":
            key_id = i
            break

    assert key_id is not None, "Keyframe 'home' not found"

    # --- Apply keyframe ---
    mujoco.mj_resetDataKeyframe(robot.model, robot.data, key_id)

    home_qpos = robot.model.key(key_id).qpos.copy()

    kp = 10.0
    kd = 0.1

    with viewer.launch_passive(robot.model, robot.data) as mj_viewer:
        while mj_viewer.is_running():

            qpos = robot.data.qpos
            qvel = robot.data.qvel

            # --- correct slicing ---
            qpos_err = home_qpos[7:] - qpos[7:]
            qvel_err = -qvel[6:]

            ctrl = kp * qpos_err + kd * qvel_err

            robot.data.ctrl[:] = ctrl

            mujoco.mj_step(robot.model, robot.data)
            mj_viewer.sync()
