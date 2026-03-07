import mujoco
from importlib.resources import files


class Robot:
    def __init__(self):

        scene_xml = str(files("humanoid_mimic_pose.assets") / "scene_mjx.xml")

        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
