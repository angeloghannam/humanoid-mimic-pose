import cv2

class Camera:

    def __init__(self, device=0):
        self.cap = cv2.VideoCapture(device)

    def read(self):
        ret, frame = self.cap.read()
        return frame

    def release(self):
        self.cap.release()