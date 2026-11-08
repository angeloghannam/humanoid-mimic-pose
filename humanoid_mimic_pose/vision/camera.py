import cv2


class Camera:

    def __init__(self, device=0):
        self.cap = cv2.VideoCapture(device)

    def read(self):
        """Returns the next frame as RGB, or None if no frame could be read."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        self.cap.release()
