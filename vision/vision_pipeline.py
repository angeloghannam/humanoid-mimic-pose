from vision.camera import Camera
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from vision.pose_detector import PoseDetector
import matplotlib.pyplot as plt

import numpy as np

class VisionSystem:

    def __init__(self):

        self.camera = Camera()
        self.pose_detector = PoseDetector()

    def step(self):

        self.frame = self.camera.read()

        return self.pose_detector.infer(self.frame)
    
    def draw_landmarks_on_image(self, skeleton):

        mp_drawing = drawing_utils
        mp_drawing_styles = drawing_styles

        annotated_image = np.copy(self.frame)

        pose_landmark_style = mp_drawing_styles.get_default_pose_landmarks_style()
        pose_connection_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=20, circle_radius=10)

        for pose_landmarks in skeleton.raw.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                pose_landmarks,
                vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=pose_landmark_style,
                connection_drawing_spec=pose_connection_style
            )

        return annotated_image