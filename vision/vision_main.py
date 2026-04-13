import numpy as np
from vision_pipeline import VisionSystem
import matplotlib.pyplot as plt
import time
import keyboard
import cv2

terminated = False

t = time.time()
while not terminated or t < 30:  # Example condition, replace with actual termination criteria

    vision_system = VisionSystem(poseDimension=2)
    result = vision_system.step()

    annotated_image = vision_system.draw_landmarks_on_image(result)
    cv2.imshow('Annotated Image', annotated_image)

    t2 = time.time()
    t = t2 - t

    if keyboard.is_pressed('q'):
        terminated = True

    if cv2.waitKey(10) == 100:  # ESC pour quitter
        break
    vision_system.camera.release() 

cv2.destroyAllWindows()


