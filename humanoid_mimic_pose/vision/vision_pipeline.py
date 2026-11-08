import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python.vision import drawing_utils

from humanoid_mimic_pose.vision.camera import Camera
from humanoid_mimic_pose.vision.pose_detector import PoseDetector


class VisionSystem:

    def __init__(self):

        self.camera = Camera()
        self.pose_detector = PoseDetector()
        self.frame = None

    def step(self):

        self.frame = self.camera.read()

        if self.frame is None:
            return None

        return self.pose_detector.infer(self.frame)

    def draw_landmarks_on_image(self, skeleton):

        annotated_image = np.copy(self.frame)

        pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
        pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=20, circle_radius=10)

        for pose_landmarks in skeleton.raw.pose_landmarks:
            drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks,
                vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=pose_landmark_style,
                connection_drawing_spec=pose_connection_style
            )

        return annotated_image

    def release(self):
        self.camera.release()
