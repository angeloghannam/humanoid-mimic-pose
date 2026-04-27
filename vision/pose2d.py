import mediapipe as mp
import numpy as np

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class Pose2D:

    def __init__(self, model_path=None):

        if model_path is None:
            print(Path(__file__).parent)
            model_path = Path(__file__).parent / "models" / "pose_landmarker_full.task"

        model_path = str(model_path)

        BaseOptions = python.BaseOptions
        PoseLandmarker = vision.PoseLandmarker
        PoseLandmarkerOptions = vision.PoseLandmarkerOptions
        VisionRunningMode = vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE
        )

        self.detector = PoseLandmarker.create_from_options(options)

    def infer(self, frame):

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        result = self.detector.detect(mp_image)

        if len(result.pose_landmarks) == 0:
            return None

        landmarks = result.pose_landmarks[0]

        keypoints = np.array([[lm.x, lm.y] for lm in landmarks])

        return keypoints
    
    def infer_result(self, frame):

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        result = self.detector.detect(mp_image)

        if len(result.pose_landmarks) == 0:
            return None

        return result