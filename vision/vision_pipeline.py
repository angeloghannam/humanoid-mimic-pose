from camera import Camera
from pose2d import Pose2D

import numpy as np

class VisionSystem:

    def __init__(self):

        self.camera = Camera()
        self.pose = Pose2D()

    def step(self):

        frame = self.camera.read()

        keypoints = self.pose.infer(frame)

        if keypoints is None:
            return None

        return keypoints