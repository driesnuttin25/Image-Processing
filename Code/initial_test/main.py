import cv2
import numpy as np

def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return None, None, None, None

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    return cap, frame_width, frame_height, fps

def preprocess_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    # Define red color range in HSV
    lower_red1 = np.array([0, 145, 0])
    upper_red1 = np.array([255, 255, 255])
    lower_red2 = np.array([0, 145, 0])
    upper_red2 = np.array([255, 255, 255])

    # Create masks for red color ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine masks to capture all red hues
    mask = mask1 | mask2

    # Remove noise from mask using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Apply the mask to the original frame to get the red tubes only
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return mask, masked_frame

def find_tube_contours(mask, min_area=100):
    # Find contours on the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    tube_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    return tube_contours

def calculate_driving_point(tubes, frame_width, smoothed_driving_point, alpha=0.1):
    # Sort tubes by centroid x-coordinate
    tubes.sort(key=lambda tube: tube['cx'])
    if len(tubes) == 1:
        # Only one tube detected
        tube = tubes[0]
        # Determine if tube is on the left or right side of the frame
        if tube['cx'] < frame_width // 2:
            # Tube is on the left, set driving point to the right
            driving_point = (int(frame_width * 0.9), tube['cy'])
        else:
            # Tube is on the right, set driving point to the left
            driving_point = (int(frame_width * 0.1), tube['cy'])
    else:
        # Extract the leftmost and rightmost tubes
        left_tube = tubes[0]
        right_tube = tubes[-1]
        # Calculate the driving point as the midpoint between the centroids
        left_x = left_tube['cx']
        right_x = right_tube['cx']
        avg_y = (left_tube['cy'] + right_tube['cy']) // 2
        driving_point = ((left_x + right_x) // 2, avg_y)

    # Apply Exponential Moving Average smoothing
    if smoothed_driving_point is None:
        smoothed_driving_point = driving_point
    else:
        smoothed_driving_point = (
            int(alpha * driving_point[0] + (1 - alpha) * smoothed_driving_point[0]),
            int(alpha * driving_point[1] + (1 - alpha) * smoothed_driving_point[1])
        )
    return smoothed_driving_point

def annotate_frames(frame_display, masked_frame_display, tubes, smoothed_driving_point, frame_width, max_width_threshold=200):
    # Draw contours and annotations on both frames
    for img in [frame_display, masked_frame_display]:
        for tube in tubes:
            # Draw contour
            cv2.drawContours(img, [tube['contour']], -1, (0, 255, 0), 2)
            # Draw centroid
            cv2.circle(img, (tube['cx'], tube['cy']), 5, (255, 0, 0), -1)
            # Put width text
            cv2.putText(img, f"Width: {tube['width']}", (tube['cx'] - 40, tube['cy'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Draw smoothed driving point and path
    for img in [frame_display, masked_frame_display]:
        cv2.circle(img, smoothed_driving_point, 5, (0, 0, 255), -1)
        img_height, img_width = img.shape[:2]
        bottom_center = (img_width // 2, img_height)
        cv2.line(img, bottom_center, smoothed_driving_point, (0, 255, 255), 2)

    # Steering suggestion based on tube widths
    if len(tubes) == 1:
        tube = tubes[0]
        if tube['width'] > max_width_threshold:
            # Determine if tube is on the left or right side of the frame
            img_width = frame_display.shape[1]
            if tube['cx'] < img_width // 2:
                direction = 'Steer Right'
            else:
                direction = 'Steer Left'
            for img in [frame_display, masked_frame_display]:
                cv2.putText(img, direction, (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif len(tubes) >= 2:
        # Check if the tubes are too close (i.e., tubes are too wide)
        left_tube = tubes[0]
        right_tube = tubes[-1]
        if left_tube['width'] > max_width_threshold or right_tube['width'] > max_width_threshold:
            # Determine which side is closer and suggest steering direction
            if left_tube['width'] > right_tube['width']:
                direction = 'Steer Right'
            else:
                direction = 'Steer Left'
            for img in [frame_display, masked_frame_display]:
                cv2.putText(img, direction, (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def display_frames(frame_display, masked_frame_display):
    combined_frame = np.hstack((frame_display, masked_frame_display))
    cv2.imshow('Processed Frames', combined_frame)

def main():
    video_path = 'c:/Users/dries/Desktop/Image_Processing/test-car/test_drive.mp4'

    # Initialize video capture
    cap, frame_width, frame_height, fps = initialize_video_capture(video_path)
    if cap is None:
        return

    # Initialize smoothed driving point
    smoothed_driving_point = None
    alpha = 0.01  # Smoothing factor for Exponential Moving Average

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Preprocess the frame to get the mask and masked frame
        mask, masked_frame = preprocess_frame(frame)

        # Find tube contours
        contours = find_tube_contours(mask, min_area=200)

        # Prepare frames for display
        frame_display = frame.copy()
        masked_frame_display = masked_frame.copy()

        if len(contours) < 1:
            # If no tubes are detected, display a message
            cv2.putText(frame_display, 'No tubes detected', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(masked_frame_display, 'No tubes detected', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            tubes = []
            for cnt in contours:
                # Calculate moments to find centroid
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    cx, cy = 0, 0  # Avoid division by zero

                # Calculate contour width as the difference between max and min x-coordinates
                x_coords = cnt[:, 0, 0]
                width = x_coords.max() - x_coords.min()

                tubes.append({'contour': cnt, 'cx': cx, 'cy': cy, 'width': width})

            # Calculate the driving point with smoothing
            smoothed_driving_point = calculate_driving_point(tubes, frame_width, smoothed_driving_point, alpha=alpha)

            # Annotate frames with contours, driving path, and suggestions
            annotate_frames(frame_display, masked_frame_display, tubes, smoothed_driving_point, frame_width, max_width_threshold=200)

        # Display the frames
        display_frames(frame_display, masked_frame_display)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
