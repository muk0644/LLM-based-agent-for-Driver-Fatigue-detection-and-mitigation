# Feature Extraction Module

## Overview

This module implements comprehensive multimodal feature extraction pipelines for the drowsiness detection system. It extracts facial action units, eye state, gaze direction, head pose, audio features, and driving metrics from raw sensor data (camera, microphone, vehicle telemetry).

## Key Components

### camera_pipeline.py
Real-time facial feature extraction using computer vision techniques.

**Core Features:**

1. **Facial Landmark Detection (68-point)**
   - Uses dlib for robust facial landmark detection
   - Computes eye closure ratio (ECR) and blink frequency
   - Detects head nod patterns and face position
   - Measures mouth opening (yawning detection)

2. **Eye State Analysis**
   - Eye Aspect Ratio (EAR) calculation
   - Blink frequency and duration tracking
   - Blink pattern anomalies (repetitive, prolonged)
   - Pupil tracking for gaze direction

3. **Head Pose Estimation**
   - 3D head rotation angles (pitch, roll, yaw)
   - Forward/backward lean detection
   - Head stability metrics
   - Nodding frequency and amplitude

4. **Facial Action Units**
   - Mouth corner positions (smile/grimace)
   - Cheek muscle activation
   - Eyebrow height and position
   - Forehead wrinkles

**Facial Landmark Indices (68-point format):**
```
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
MOUTH = [48, 54, 51, 62, 66, 57]
NOSE = [27-35]
JAW = [0-16]
EYEBROWS = [17-26]
```

**Usage Example:**
```python
from camera_pipeline import CameraPipeline
import cv2

# Initialize ROS 2 node
pipeline = CameraPipeline()

# Subscribe to image topic and process frames
# Publishes: facial_features, eye_state, head_pose topics

# Manual processing
import dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

frame = cv2.imread("driver_face.jpg")
faces = detector(frame)

for face in faces:
    landmarks = predictor(frame, face)
    
    # Extract features
    left_eye_points = landmarks[36:42]
    right_eye_points = landmarks[42:48]
    mouth_points = landmarks[48:68]
    
    # Compute metrics
    left_ear = compute_eye_aspect_ratio(left_eye_points)
    right_ear = compute_eye_aspect_ratio(right_eye_points)
    eye_closed = (left_ear + right_ear) / 2 < 0.2
```

**Output Features:**
- eye_closure_ratio: 0-1 (0 = open, 1 = closed)
- blink_frequency: blinks per minute
- blink_duration: average blink duration (ms)
- head_pitch: head tilt angle (-90 to 90 degrees)
- head_roll: head rotation angle (-90 to 90 degrees)
- head_yaw: head left-right angle (-90 to 90 degrees)
- mouth_opening: 0-1 (0 = closed, 1 = wide open)
- yawn_detected: boolean
- gaze_direction: horizontal and vertical angle

### carla_data_feature_extraction.py
Feature extraction from CARLA simulator data for training and validation.

**CARLA Integration:**
```python
from carla_data_feature_extraction import CarlaFeatureExtractor

extractor = CarlaFeatureExtractor(
    carla_ip="localhost",
    carla_port=2000
)

# Connect to CARLA world
extractor.connect()

# Extract features from simulation
features = extractor.extract_frame(
    rgb_image=carla_rgb_frame,
    thermal_image=carla_thermal_frame,
    actor_list=carla_actors
)

# Includes
print(f"Extracted {len(features)} feature types")
print(f"Fatigue level: {features['fatigue_level']}")
```

**CARLA Data Advantages:**
- Controlled environment for reproducible testing
- Ground truth labels available
- Multiple sensor modalities
- Weather and lighting variation
- Diverse driving scenarios

### fatigue_estimation.py
High-level fatigue level estimation combining multiple feature sources.

**Fatigue Score Computation:**

```
Fatigue Score = w1 * eye_closure_ratio
              + w2 * (1 - blink_frequency/normal_rate)
              + w3 * head_nod_intensity
              + w4 * (1 - voice_quality)
              + w5 * steering_instability
              + w6 * (time_awake / max_time)

Weights (sum = 1):
w1 = 0.25  (eye closure)
w2 = 0.20  (blink rate)
w3 = 0.15  (head movement)
w4 = 0.15  (voice quality)
w5 = 0.15  (steering)
w6 = 0.10  (time awake)
```

**Fatigue Levels:**
- Alert (0.0-0.33): Normal driving condition
- Drowsy (0.33-0.66): Moderate fatigue, intervention recommended
- Sleepy (0.66-1.0): Severe fatigue, immediate action needed

**Usage:**
```python
from fatigue_estimation import FatigueEstimator

estimator = FatigueEstimator(
    model_path="trained_fatigue_model.pkl",
    confidence_threshold=0.75
)

features = {
    'eye_closure_ratio': 0.45,
    'blink_frequency': 8.5,
    'head_nod_magnitude': 0.3,
    'voice_quality': 0.72,
    'steering_stability': 0.85,
    'time_awake': 120
}

fatigue_level, confidence = estimator.estimate(features)
print(f"Fatigue: {fatigue_level} (confidence: {confidence:.2f})")
```

## Feature Types and Ranges

### Visual Features
| Feature | Range | Unit | Extraction Source |
|---------|-------|------|------------------|
| eye_closure_ratio | 0-1 | ratio | Eye aspect ratio |
| blink_frequency | 0-20 | blinks/min | Temporal tracking |
| blink_duration | 50-500 | ms | Frame counting |
| head_pitch | -90 to 90 | degrees | 3D head pose |
| head_roll | -90 to 90 | degrees | 3D head pose |
| head_yaw | -90 to 90 | degrees | 3D head pose |
| mouth_opening | 0-1 | ratio | Landmark distance |
| face_confidence | 0-1 | score | Detection quality |

### Audio Features
| Feature | Range | Unit | Extraction Source |
|---------|-------|------|------------------|
| mfcc_coeff[0-12] | varied | coefficient | MFCC transform |
| pitch | 80-250 | Hz | Fundamental frequency |
| energy | 0-1 | normalized | RMS energy |
| spectral_centroid | 0-8000 | Hz | Frequency center |
| zero_crossing_rate | 0-0.5 | ratio | Silence detection |

### Driving Features
| Feature | Range | Unit | Extraction Source |
|---------|-------|------|------------------|
| steering_angle | -100 to 100 | degrees | CAN bus |
| steering_rate | 0-90 | deg/sec | Steering derivative |
| lane_deviation | 0-1 | normalized | Lane detection |
| speed_variance | 0-1 | normalized | Speed changes |
| acceleration | -10 to 10 | m/s^2 | CAN bus |
| brake_pressure | 0-100 | percent | CAN bus |

## ROS 2 Integration

### Published Topics
```
/camera_pipeline/facial_features (std_msgs/Float32MultiArray)
  - Facial feature vector [eye_closure, blink_freq, head_pitch, ...]

/camera_pipeline/eye_state (std_msgs/String)
  - "open", "closing", "closed", "blinking"

/camera_pipeline/head_pose (geometry_msgs/Vector3)
  - Pitch, roll, yaw angles

/camera_pipeline/mouth_state (std_msgs/Float32)
  - Mouth opening ratio

/feature_extraction/combined_features (std_msgs/Float32MultiArray)
  - All features for downstream processing
```

### Subscribed Topics
```
/camera/image_raw (sensor_msgs/Image)
  - RGB video from camera

/microphone/audio (sensor_msgs/Audio)
  - Audio stream from microphone

/vehicle/telemetry (custom_msgs/VehicleTelemetry)
  - CAN bus data and vehicle metrics
```

## Data Flow

```
Raw Sensor Data
├── RGB Camera Stream
├── Audio Stream
└── Vehicle CAN Bus
        |
        v
    Feature Extraction
├── camera_pipeline.py (facial/head features)
├── Audio extraction (MFCC, pitch, energy)
└── Driving metrics (steering, speed, acceleration)
        |
        v
    Feature Normalization
        |
        v
    Fatigue Estimation
        |
        v
    LLM Feature Formatting
        |
        v
    Vector Embedding
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Specific packages
pip install opencv-python dlib librosa numpy scipy pandas

# For ROS 2 integration
pip install rclpy sensor_msgs cv-bridge

# For CARLA
pip install carla

# Download dlib face detector
# https://github.com/davisking/dlib-models
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

## Performance Specifications

### Camera Pipeline
- Processing time: 33 ms per frame (30 FPS)
- Accuracy: 95%+ detection rate
- Robustness: Handles various lighting, angles

### Audio Extraction
- Processing: Real-time, 16 kHz sample rate
- Features computed: Every 512 samples (32 ms)
- Latency: <50 ms

### Overall System
- End-to-end latency: 100-150 ms
- Feature update rate: 10 Hz
- Memory footprint: 500 MB

## Troubleshooting

**Issue:** Facial landmarks not detected
- Solution: Check lighting conditions
- Ensure face is clearly visible
- Verify dlib model is loaded correctly

**Issue:** High false positive blink detection
- Solution: Adjust EAR threshold (default 0.2)
- Increase consecutive frame requirement
- Filter with temporal smoothing

**Issue:** Poor head pose estimation
- Solution: Ensure camera calibration
- Use front-facing camera angles
- Verify landmark detection accuracy

## Related Modules

- **Input Signal Processing:** Raw data preprocessing
- **Vector Database:** Embedding generation
- **LLM and Fatigue Handling:** Feature consumption

## Advanced Topics

### Custom Feature Addition
```python
def extract_custom_feature(frame, landmarks):
    """Add custom feature extraction"""
    # Process frame and landmarks
    custom_value = compute_custom_metric(frame, landmarks)
    return custom_value

# Register with pipeline
CameraPipeline.register_feature('custom_feature', extract_custom_feature)
```

### Calibration
```python
# Camera calibration for improved head pose
camera_matrix = calibrate_camera(calibration_images)
pipeline.set_camera_calibration(camera_matrix)
```

### Real-time Visualization
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(eye_closure_history)
axes[0, 1].plot(blink_frequency_history)
axes[1, 0].plot(head_pose_history)
axes[1, 1].plot(fatigue_score_history)
plt.show()
```

## Future Enhancements

- Deep learning-based eye detection (YOLOv8)
- Gaze tracking with eye-in-head coordinate system
- Facial expression recognition (AU-based)
- Multi-person tracking for validation
- GPU acceleration with CUDA
- Real-time visualization dashboard
