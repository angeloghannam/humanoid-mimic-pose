from dataclasses import dataclass

from numpy.typing import NDArray

# Landmark names/indices (NOSE, LEFT_WRIST, ...), so keypoints can be accessed by name
from mediapipe.tasks.python.vision import PoseLandmark, PoseLandmarkerResult


@dataclass
class Skeleton:
    """Output of a single pose detection, holding both 2D and 3D keypoints."""

    keypoints2d: NDArray  # (33, 2) image coordinates, normalized to [0, 1]
    keypoints3d: NDArray  # (33, 3) world coordinates in meters, origin at the hips
    visibility: NDArray  # (33,) likelihood that each landmark is visible
    raw: PoseLandmarkerResult  # raw mediapipe result, used for drawing

    def get2d(self, landmark: PoseLandmark) -> NDArray:
        return self.keypoints2d[landmark]

    def get3d(self, landmark: PoseLandmark) -> NDArray:
        return self.keypoints3d[landmark]
