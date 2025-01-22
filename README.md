# Image-Processing

## Objectives

- Detect the red tubes using computer vision techniques.
- Process the camera feed to identify the path between the tubes.
- Use the detected path to control the car’s steering, keeping it centered on the track.

---

## Hardware Setup

- **Raspberry Pi**: The central controller that processes the camera feed.
- **Camera**: Attached to the Raspberry Pi, captures live video of the track.
- **Self-driving car chassis**: Includes motors and wheels controlled by the Raspberry Pi.

---

## Software Approach

### 1. Color Space Selection
To isolate the red tubes from the track, we experimented with different color spaces to find the most effective representation. The following color spaces were tested:

- **RGB**: The default color space captured by the camera. While simple, it is highly sensitive to lighting changes.
- **HSV (Hue, Saturation, Value)**: Separates color (Hue) from brightness (Value), making it robust to moderate lighting changes.
- **Lab** (Lightness, A*, B*): Offers better color consistency under varying lighting conditions, as the A* and B* channels represent color independent of brightness.


### 2. Thresholding and Masking
Using the chosen color space (Lab for its robustness to lighting variations), we applied a threshold to isolate the red regions representing the tubes. The threshold values were determined experimentally to ensure reliable detection under different conditions.


- **Thresholding Result**: 
![image](https://github.com/user-attachments/assets/61dc7114-ff63-4f20-8351-83eaeccb852d)

---

### 3. Path Detection and Steering Logic
To steer the car through the track, we implemented the following logic:

1. **Frame Division**:
   - The frame is divided into multiple vertical bins (e.g., 20 strips).
   - Each bin is analyzed for the number of red pixels it contains.

2. **Cost Function**:
   - A cost function is used to prioritize bins with fewer red pixels (indicating a clear path).
   - Additional weight is given to bins closer to the center of the frame to keep the car centered on the track.

3. **Steering Decision**:
   - The car steers toward the bin with the lowest cost.
   - Exponential smoothing is applied to the steering direction to reduce jitter.

### Placeholder for Images:

- **Visualization of Bins**: 
![image](https://github.com/user-attachments/assets/e839c202-cc43-48d4-8c43-4287b9dd0fbb)

---

### 4. Morphological Operations
To improve the mask quality, morphological operations (e.g., closing) are applied. This helps fill small gaps in the red tube detection and reduces noise.

---

## Challenges

1. **Lighting Variations**:
   - Shadows and highlights affect color detection.
   - Lab color space proved effective at mitigating these issues.

2. **Gaps in the Tubes**:
   - Connectors and breaks in the red tubes caused discontinuities.
   - Morphological operations and logic that doesn't rely on continuous contours were implemented to handle this.


