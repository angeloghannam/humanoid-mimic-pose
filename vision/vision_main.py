import numpy as np
from vision_pipeline import VisionSystem
import matplotlib.pyplot as plt
import time
import keyboard
import cv2

terminated = False
no_pose = False
vision_system = VisionSystem(poseDimension=2)
t = time.time()
while not terminated or t < 30:  # Example condition, replace with actual termination criteria
    # ADD LIVE VIDEO PARAMETER TO VISION SYSTEMs

    
    result = vision_system.step()

    try :
        print(result.pose_landmarks[0])
    except:
        print("No pose detected")
        no_pose = True
    if no_pose == False:
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


