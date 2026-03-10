---
marp: true
math: true
mathjax: true
---

# Automatic Basketball Video Zooming System: Real-Time Camera Tracking Based on PID Control, YOLOv5, and Classification Models
## Project Background and Objectives

### Project Background:
- **Problem Description**: Current mid- to low-level basketball games lack sufficient photographers for recording and tracking.
- **Implementation Goal**: Automatically zoom the camera based on the area where players are concentrated, simulating the effect of following the players.
- **Application Scenario**: Campus basketball games.

---

### Core Objectives:
- Real-time detection and precise classification of player positions.
- Dynamically adjust the camera focus area.
- Smooth camera movement to avoid unnecessary shaking.

---

## Mathematical Model Analysis

### Target Position Calculation:
Assume the player's position is represented by a bounding box, and calculate the center point of each player:
$$
x_c = \frac{x_1 + x_2}{2}, \quad y_c = \frac{y_1 + y_2}{2}
$$

The center point of the concentrated area:
$$
x = \frac{\sum_{i=1}^n x_{c_i}}{n}, \quad y = \frac{\sum_{i=1}^n y_{c_i}}{n}
$$
Where $n$ represents the number of players, and $x_{c_i}, y_{c_i}$ represent the center coordinates of the $i$-th player.

---

## PID Controller
Basic formula:
$$
e_t = \text{setpoint} - \text{current\_value}
$$
PID controller output:
$$
\text{Output} = K_p \cdot e_t + K_i \cdot \sum_{t=1}^T e_t + K_d \cdot (e_t - e_{t-1})
$$
Where:
- $e_t$ is the current error.
- $K_p$, $K_i$, $K_d$ are the proportional, integral, and derivative constants of the PID controller.
- $\sum_{t=1}^T e_t$ is the accumulated error.
- $e_{t-1}$ is the previous error.

---

Adjust the camera position:
$$
x_{\text{new}} = x_{\text{current}} + \text{Output}_x, \quad y_{\text{new}} = y_{\text{current}} + \text{Output}_y
$$

Limit the camera position to avoid exceeding the frame boundary:
$$
x_{\text{new}} = \max(500, \min(\text{frame\_width} - 500, x_{\text{new}})), \quad y_{\text{new}} = \max(300, \min(\text{frame\_height} - 300, y_{\text{new}}))
$$

---

## Smoothing Processing

### Smoothing Processing:
To reduce fluctuations in the target position, use a sliding window to average the target position:
$$
\bar{x} = \frac{\sum_{i=1}^{W} x_i}{W}, \quad \bar{y} = \frac{\sum_{i=1}^{W} y_i}{W}
$$
Where $W$ is the window size, and $x_i$ and $y_i$ are the target coordinates in the $i$-th frame.

---

## System Architecture and Process Flow

### System Architecture:
1. **Video Input**: Read the basketball game video.
2. **Target Detection**: Use the YOLO model to detect player positions.
3. **Target Tracking**: Calculate the center of player positions and select the camera focus.
4. **PID Control**: Calculate and adjust the camera position, using the PID controller to smooth the camera movement.

---

### Processing Flow:
1. Capture video frames.
2. Detect the bounding boxes of players and the basketball.
3. Calculate the concentrated target area of the players. 
4. Adjust the camera position using the PID controller.
5. Output and save the video.

---

## Results Display

### Effect Display:
- **Comparison between the original video and the automatic zoom effect**: Show how the area with concentrated players automatically aligns and stabilizes.

### Performance Evaluation:
- **Accuracy**: Detection accuracy ≥ 90%.
- **Response Speed**: On my computer, processing 20 frames per second, meeting real-time requirements.
- **Smoothing Effect**: Significant reduction in jitter.

---

## Continuous Optimization and Future Prospects

### Optimization Directions:
- **Improve Detection Accuracy**: Reduce false positives, optimize the target detection model.
- **Introduce Deep Learning for Enhanced Target Recognition**: For example, using more powerful models to improve player recognition capabilities.
- **Introduce Basketball Position as a New Variable for Better Tracking Effects**.
- **More Efficient Smoothing Algorithm**: Reduce smoothing processing delays and improve response speed.

---

### Future Applications:
- **Multi-Scenario Applications**: Not limited to basketball games, it can be extended to live broadcasting and analysis of other sports events such as soccer, marathon running, etc.
- **Enhanced Interactivity**: Real-time weight adjustment to provide a more personalized viewing experience.

---

## Q&A
**Thank you, everyone!**  
- **Q&A** time.

