import cv2
import numpy as np

image_path = "c:/Users/dries/Desktop/Image_Processing/Image_processing_task/screenshot_2.jpg"

# 1. Load your image
img = cv2.imread(image_path)
if img is None:
    raise IOError("Could not read the image. Check the file path!")

# 2. Convert to Lab color space
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# 3. Threshold to isolate the “red tube” region
#    Adjust lower/upper bounds as needed for your red tubes
lower_bound = np.array([0, 145, 0])
upper_bound = np.array([255, 255, 255])
mask = cv2.inRange(lab, lower_bound, upper_bound)

# 4. (Optional) Morphological operations to clean up the mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 5. Extract only the red tube region from the original image
red_region = cv2.bitwise_and(img, img, mask=mask)

# 6. Convert that region to grayscale for edge detection
gray_region = cv2.cvtColor(red_region, cv2.COLOR_BGR2GRAY)

# 7. Canny edge detection
#    Tweak the thresholds (100, 200) for your scenario
edges = cv2.Canny(gray_region, 100, 200)

# 8. (Optional) To visualize the edges in a more telling way:
#    - We can create a 3-channel edge image to overlay on original
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
overlay = cv2.addWeighted(img, 0.7, edges_colored, 0.9, 0)

# 9. Display
cv2.imshow("Original", img)
cv2.imshow("Mask (Red Areas)", mask)
cv2.imshow("Red Region", red_region)
cv2.imshow("Edges within Red Region", edges)
cv2.imshow("Edges Overlaid on Original", overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()
