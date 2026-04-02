import mediapipe as mp
import numpy as np

class Pose2D:

    def __init__(self):

        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False
        )

    def infer(self, frame):

        results = self.pose.process(frame)

        if not results.pose_world_landmarks:
            return None

        keypoints3d = []

        for lm in results.pose_world_landmarks.landmark:
            keypoints3d.append([lm.x, lm.y, lm.z])

        return np.array(keypoints3d)