import cv2
import numpy as np

cap = cv2.VideoCapture("c:/Users/dries/Desktop/Image_Processing/Image_processing_task/test_video.mp4")
if not cap.isOpened():
    raise IOError("Could not open video source")

smoothed_x = None
alpha = 0.8  # smoothing factor

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) Convert to Lab color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # 2) Threshold to isolate red tubes
    lower_bound = np.array([0, 150, 0])
    upper_bound = np.array([255, 255, 255])
    mask = cv2.inRange(lab, lower_bound, upper_bound)

    # 3) Morphological ops to help fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 4) Compute the “real” centroid (average x-position) of all red pixels
    height, width = mask.shape
    column_sums = np.sum(mask, axis=0)  # sum of pixel intensities per column
    column_counts = column_sums // 255  # count of white pixels per column
    total_red_pixels = np.sum(column_counts)

    if total_red_pixels > 0:
        # Weighted average of column indices
        indices = np.arange(width)
        weighted_sum = np.sum(indices * column_counts)
        centroid_x = weighted_sum / total_red_pixels

        # 5) Invert the centroid horizontally
        inverted_x = (width - 1) - centroid_x

        # 6) Smooth the inverted centroid
        if smoothed_x is None:
            smoothed_x = inverted_x
        else:
            smoothed_x = alpha * smoothed_x + (1 - alpha) * inverted_x

        # 7) Steering logic based on the inverted centroid
        center_x = width / 2
        offset = smoothed_x - center_x
        threshold = 20

        if offset < -threshold:
            print("Steer LEFT")
        elif offset > threshold:
            print("Steer RIGHT")
        else:
            print("Go STRAIGHT")

        # Visualization: draw a line at the smoothed centroid
        output = frame.copy()
        cv2.line(output, (int(smoothed_x), 0), (int(smoothed_x), height), (0, 255, 0), 2)
        # Draw a blue line at the image center for reference
        cv2.line(output, (int(center_x), 0), (int(center_x), height), (255, 0, 0), 2)
    else:
        print("No red tubes detected – fallback or go straight")
        output = frame.copy()

    # Display
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
