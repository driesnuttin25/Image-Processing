import cv2
import numpy as np

def nothing(x):
    pass

# 1. Load image
img = cv2.imread('c:/Users/dries/Desktop/Image_Processing/Image_processing_task/screenshot.jpg')
if img is None:
    raise IOError("Could not read the image file. Check the path!")

# 2. Create windows (one for trackbars, optional ones for visualization)
cv2.namedWindow('Trackbars')
cv2.namedWindow('Original')
cv2.namedWindow('Mask')
cv2.namedWindow('Result')

# 3. Create trackbars for lower/upper L, A, B
# In OpenCV, L, A, and B each go from 0 to 255
cv2.createTrackbar('LowerL', 'Trackbars', 0,   255, nothing)
cv2.createTrackbar('UpperL', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('LowerA', 'Trackbars', 0,   255, nothing)
cv2.createTrackbar('UpperA', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('LowerB', 'Trackbars', 0,   255, nothing)
cv2.createTrackbar('UpperB', 'Trackbars', 255, 255, nothing)

while True:
    # 4. Convert to Lab color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # 5. Get current positions of the six trackbars
    l_l = cv2.getTrackbarPos('LowerL','Trackbars')
    u_l = cv2.getTrackbarPos('UpperL','Trackbars')
    l_a = cv2.getTrackbarPos('LowerA','Trackbars')
    u_a = cv2.getTrackbarPos('UpperA','Trackbars')
    l_b = cv2.getTrackbarPos('LowerB','Trackbars')
    u_b = cv2.getTrackbarPos('UpperB','Trackbars')

    # 6. Define lower and upper arrays for the mask
    lower_bound = np.array([l_l, l_a, l_b])
    upper_bound = np.array([u_l, u_a, u_b])

    # 7. Create the mask based on those Lab thresholds
    mask = cv2.inRange(lab, lower_bound, upper_bound)

    # 8. Use the mask to keep only the selected region
    result = cv2.bitwise_and(img, img, mask=mask)

    # 9. Show everything
    cv2.imshow('Original', img)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    # 10. ESC key breaks the loop
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()
