from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from humanoid_mimic_pose.vision.skeleton import Skeleton


class PoseDetector:
    """Runs mediapipe's pose landmarker once per frame and returns 2D and 3D keypoints."""

    def __init__(self, model_path=None):

        if model_path is None:
            model_path = Path(__file__).parent / "models" / "pose_landmarker_full.task"

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)

    def infer(self, frame) -> Skeleton | None:
        """Takes an RGB frame, returns the Skeleton of the first detected person or None."""

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        result = self.detector.detect(mp_image)

        if len(result.pose_landmarks) == 0:
            return None

        landmarks = result.pose_landmarks[0]
        world_landmarks = result.pose_world_landmarks[0]

        return Skeleton(
            keypoints2d=np.array([[lm.x, lm.y] for lm in landmarks]),
            keypoints3d=np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks]),
            visibility=np.array([lm.visibility for lm in landmarks]),
            raw=result
        )
