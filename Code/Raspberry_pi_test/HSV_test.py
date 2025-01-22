from picamera2 import Picamera2
import cv2
import numpy as np

def convert_to_bgr_if_needed(frame):
    # If colors look inverted or off, convert from RGB to BGR
    # If not needed, comment this line out.
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def preprocess_frame(frame):
    """
    Detect red areas in the frame using the HSV color space.
    We'll use two hue ranges to capture red hue wrap-around.
    """

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Lower range for red:
    lower_red1 = np.array([0, 145, 0], dtype=np.uint8)
    upper_red1 = np.array([255, 255, 255], dtype=np.uint8)

    # Upper range for red:
    lower_red2 = np.array([0, 145, 0], dtype=np.uint8)
    upper_red2 = np.array([255, 255, 255], dtype=np.uint8)

    # Create two masks and combine them
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    # Morphological operations to clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask

def find_tubes(mask, min_area=200):
    """
    Find contours of red areas and filter them to find tubes representing red lines.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tubes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            tubes.append({'contour': cnt, 'x': x, 'y': y, 'w': w, 'h': h})

    # Sort by x-coordinate (left to right)
    tubes.sort(key=lambda t: t['x'])
    return tubes

def calculate_driving_point(tubes, frame_width, prev_point=None, alpha=0.2):
    """
    Determine the driving point based on detected tubes.
    If two or more tubes, midpoint of top centers.
    If one, steer slightly away.
    If none, center.
    """
    height = 240  # Based on camera config
    default_point = (frame_width // 2, height - 1)

    if len(tubes) == 0:
        new_point = default_point
    elif len(tubes) == 1:
        # One tube
        tube = tubes[0]
        left_top_center = (tube['x'] + tube['w'] // 2, tube['y'])
        if left_top_center[0] < frame_width // 2:
            # Tube on left, steer right
            new_point = (int(frame_width * 0.75), height // 2)
        else:
            # Tube on right, steer left
            new_point = (int(frame_width * 0.25), height // 2)
    else:
        # Two or more tubes
        left_tube = tubes[0]
        right_tube = tubes[-1]
        left_top_center = (left_tube['x'] + left_tube['w'] // 2, left_tube['y'])
        right_top_center = (right_tube['x'] + right_tube['w'] // 2, right_tube['y'])
        mid_x = (left_top_center[0] + right_top_center[0]) // 2
        mid_y = (left_top_center[1] + right_top_center[1]) // 2
        new_point = (mid_x, mid_y)

    # Smooth the driving point
    if prev_point is None:
        return new_point
    else:
        smoothed_x = int(alpha * new_point[0] + (1 - alpha) * prev_point[0])
        smoothed_y = int(alpha * new_point[1] + (1 - alpha) * prev_point[1])
        return (smoothed_x, smoothed_y)

def main():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))
    picam2.start()

    print("Press 'q' to exit.")

    smoothed_driving_point = None
    frame_width = 320
    frame_height = 240

    while True:
        frame = picam2.capture_array()
        # Convert if needed from RGB to BGR
        frame = convert_to_bgr_if_needed(frame)

        mask = preprocess_frame(frame)
        tubes = find_tubes(mask, min_area=200)
        smoothed_driving_point = calculate_driving_point(tubes, frame_width, prev_point=smoothed_driving_point, alpha=0.2)

        # Highlight red areas
        red_highlighted = cv2.bitwise_and(frame, frame, mask=mask)

        # Draw driving point and line
        cv2.circle(red_highlighted, smoothed_driving_point, 5, (0, 255, 255), -1)
        cv2.putText(red_highlighted, "Driving Point", (smoothed_driving_point[0] - 20, smoothed_driving_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        bottom_center = (frame_width // 2, frame_height)
        cv2.line(red_highlighted, bottom_center, smoothed_driving_point, (0, 255, 0), 2)

        # Show original and processed side by side
        combined = np.hstack((frame, red_highlighted))
        cv2.imshow("Original (Left) vs Processed (Right)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == "__main__":
    main()
