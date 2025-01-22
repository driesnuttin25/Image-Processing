import cv2
import numpy as np

cap = cv2.VideoCapture("c:/Users/dries/Desktop/Image_Processing/Image_processing_task/test_video.mp4")
if not cap.isOpened():
    raise IOError("Could not open video source")

smoothed_x = None
alpha = 0.9  
BIN_COUNT = 20
distance_factor = 0.3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) Convert to Lab color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # 2) Threshold to isolate red tubes
    lower_bound = np.array([0, 145, 0])
    upper_bound = np.array([255, 255, 255])
    mask = cv2.inRange(lab, lower_bound, upper_bound)

    # 3) Morphological ops to help fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    height, width = mask.shape
    center_x = width / 2

    # 4) Divide the frame into BIN_COUNT vertical strips,
    #    compute the number of red pixels in each, and a cost function.
    bin_width = width // BIN_COUNT
    
    best_bin_index = None
    best_cost = float('inf')  # Initialize with a large number
    total_red_pixels = cv2.countNonZero(mask)
    
    if total_red_pixels > 0:
        for i in range(BIN_COUNT):
            start_x = i * bin_width
            
            end_x = width if (i == BIN_COUNT - 1) else (start_x + bin_width)

            # Extract the portion of the mask in this bin
            bin_mask = mask[:, start_x:end_x]
            red_count = cv2.countNonZero(bin_mask)

            # Bin's center
            bin_center = (start_x + end_x) / 2.0
            dist_from_center = abs(bin_center - center_x)

            # Cost function:
            #   - We want to AVOID bins with lots of red pixels
            #   - We also want to avoid bins far from the image center
            cost = red_count + distance_factor * dist_from_center

            if cost < best_cost:
                best_cost = cost
                best_bin_index = i

        # 5) Compute the target x = center of the best bin
        start_x = best_bin_index * bin_width
        end_x = width if (best_bin_index == BIN_COUNT - 1) else (start_x + bin_width)
        chosen_bin_center = (start_x + end_x) / 2.0

        # 6) Exponential smoothing on that target x
        if smoothed_x is None:
            smoothed_x = chosen_bin_center
        else:
            smoothed_x = alpha * smoothed_x + (1 - alpha) * chosen_bin_center

        # 7) Steering logic
        offset = smoothed_x - center_x
        threshold = 20

        if offset < -threshold:
            print("Steer LEFT")
        elif offset > threshold:
            print("Steer RIGHT")
        else:
            print("Go STRAIGHT")

        # 8) Visualization: draw all bin lines + highlight chosen bin
        output = frame.copy()

        # Draw bin boundaries
        for i in range(1, BIN_COUNT):
            line_x = int(i * bin_width)
            cv2.line(output, (line_x, 0), (line_x, height), (200, 200, 200), 1)

        # Draw green line at chosen bin's center
        cv2.line(output, (int(smoothed_x), 0), (int(smoothed_x), height), (0, 255, 0), 2)

        # Draw a blue line at the image center
        cv2.line(output, (int(center_x), 0), (int(center_x), height), (255, 0, 0), 2)
    else:
        print("No red tubes detected – fallback or go straight")
        output = frame.copy()

    # Show windows
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
