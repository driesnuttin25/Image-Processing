import cv2
import numpy as np

cap = cv2.VideoCapture("c:/Users/dries/Desktop/Image_Processing/Image_processing_task/test_video.mp4")

if not cap.isOpened():
    raise IOError("Could not open video source")

while True:
    ret, frame = cap.read()
    if not ret:
        break  

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    lower_bound = np.array([0, 156, 0])   # LowerH=0, LowerS=156, LowerV=0
    upper_bound = np.array([179, 255, 255])  # UpperH=179, UpperS=255, UpperV=255

    # Create mask to isolate "red" range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Show only the red regions from the original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display
    cv2.imshow('Original', frame)
    cv2.imshow('Red Tubes Only', result)

    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
