from camera import Camera
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pose2d import Pose2D
from pose3d import Pose3D
import matplotlib.pyplot as plt

import numpy as np

class VisionSystem:

    def __init__(self, poseDimension=2):

        self.camera = Camera()

        if poseDimension == 2:
            self.pose = Pose2D()
        elif poseDimension == 3:
            self.pose = Pose3D()
        else:
            raise ValueError("poseDimension must be either 2 or 3")
        
        
    def step(self):

        self.frame = self.camera.read()

        result = self.pose.infer_result(self.frame)

        if result is None:
            return None

        return result
    
    def draw_landmarks_on_image(self, result):

        mp_drawing = drawing_utils
        mp_drawing_styles = drawing_styles

        annotated_image = np.copy(self.frame)

        pose_landmark_style = mp_drawing_styles.get_default_pose_landmarks_style()
        pose_connection_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=20, circle_radius=10)

        for pose_landmarks in result.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                pose_landmarks,
                vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=pose_landmark_style,
                connection_drawing_spec=pose_connection_style
            )

        return annotated_image