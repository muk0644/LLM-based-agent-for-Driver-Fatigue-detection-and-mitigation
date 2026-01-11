# Input Signal Processing Module

## Overview

This module handles raw data preprocessing and sensor synchronization for the drowsiness detection system. It implements data cleaning, normalization, and temporal alignment across multiple sensor streams (infrared images, RGB camera, audio, and vehicle telemetry).

## Key Components

### data_process.py
Main preprocessing pipeline with the following stages:

**STEP 1: Cleaning**
- Validates infrared image data for black/white frames and corrupt files
- Removes NaN and infinite values
- Filters out empty or unreadable image files
- Tracks quality metrics (total vs. dropped frames)

**STEP 2: Normalization**
- Applies histogram equalization for consistent image brightness
- Standardizes pixel ranges across different infrared camera models
- Handles dynamic range compression for thermal imagery

**STEP 3: Alignment**
- Synchronizes timestamps across asynchronous sensors
- Resamples data to common frequency (e.g., 30 FPS for vision, 16 kHz for audio)
- Handles clock drift between distributed sensor nodes

**STEP 4: Feature Preparation**
- Crops regions of interest (face, hands for steering wheel)
- Applies data augmentation for training robustness
- Creates synchronized data bundles for downstream processing

### data_process_only_sync.py
Lightweight synchronization module focusing only on timestamp alignment without extensive cleaning. Useful for real-time processing with pre-validated data streams.

### ROS 2 Integration

#### sync_node.py (ROS 2 Node)
Real-time synchronization node that:
- Subscribes to multiple ROS 2 topics from different sensors
- Maintains synchronized message buffers
- Publishes synchronized data bundles via custom message types
- Handles dropped frames and latency compensation

#### preprocess_node.py (ROS 2 Node)
Applies preprocessing transformations in real-time:
- Receives raw sensor data from sync_node
- Applies cleaning and normalization algorithms
- Publishes preprocessed data for feature extraction

## Configuration

Key parameters in source files:

```python
SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp'}  # Accepted image formats
SYNC_TOLERANCE_MS = 100  # Maximum timestamp difference for alignment
FRAME_RATE = 30  # Target frames per second
NORMALIZATION_METHOD = 'histogram'  # Contrast enhancement method
```

## Usage

### Offline Processing
```python
from data_process import clean_ir_images, normalize_images, align_sequences

# Clean raw thermal images
clean_ir_images(input_dir='./raw_data/thermal', 
                output_dir='./processed/cleaned')

# Normalize and prepare
normalize_images(input_dir='./processed/cleaned',
                 output_dir='./processed/normalized')

# Align with other modalities
align_sequences(thermal_dir='./processed/normalized',
                video_dir='./processed/video',
                audio_dir='./processed/audio',
                output_dir='./processed/aligned')
```

### Real-time Processing (ROS 2)
```bash
# Terminal 1: Start sensor nodes (assumed to be running)
ros2 launch sensor_drivers all_sensors.launch.py

# Terminal 2: Start synchronization
ros2 run input_signal_processing sync_node

# Terminal 3: Start preprocessing
ros2 run input_signal_processing preprocess_node

# Monitor output
ros2 topic echo /preprocessed_data
```

## Input Data Format

### Infrared Images
- Format: PNG or TIFF (16-bit depth for thermal data)
- Resolution: Varies by camera (320x256 or 640x512 typical)
- Naming convention: `frame_YYYYMMDD_HHMMSS_milliseconds.png`

### RGB Video
- Format: H.264/H.265 encoded video streams
- Resolution: 1280x720 or 1920x1080
- Frame rate: 30 FPS
- ROS 2 Topic: `/camera/image_raw`

### Audio
- Format: 16-bit PCM WAV or ROS 2 audio messages
- Sample rate: 16 kHz
- Mono or stereo channels
- ROS 2 Topic: `/microphone/audio`

### Vehicle Telemetry
- CAN bus messages or ROS 2 topics
- Signals: speed, steering angle, brake/throttle, lane detection
- Update rate: 100 Hz

## Output Data Format

### Synchronized Data Bundle
Contains temporally aligned data from all modalities:
- Frame timestamp (nanosecond precision)
- Thermal image array
- RGB image array
- Audio chunk (512 samples)
- Vehicle telemetry snapshot
- Data quality flags

### Custom ROS 2 Message (SyncedOutput.msg)
```
std_msgs/Header header
sensor_msgs/Image thermal_image
sensor_msgs/Image rgb_image
sensor_msgs/CompressedImage rgb_compressed
std_msgs/Float32MultiArray audio_chunk
std_msgs/Float32MultiArray vehicle_telemetry
uint8[] quality_flags
```

## Error Handling

The module includes robust error handling for:
- Missing or corrupted sensor data
- Timestamp synchronization failures
- Buffer overflows in real-time processing
- ROS 2 connection drops

Errors are logged with context for debugging and recovery.

## Performance Considerations

- Processing overhead: ~5-10 ms per frame on standard hardware
- Memory usage: ~2-3 GB for offline processing of 10-minute sequences
- Real-time capability: Supports up to 4 concurrent sensor streams at 30 FPS

## Dependencies

- OpenCV (cv2) for image processing
- NumPy for numerical operations
- ROS 2 (Humble or later) for real-time components
- Pandas for data manipulation
- Scipy for signal processing

## Testing

Run the test suite:
```bash
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_synchronization.py -v
```

## Related Modules

- **Feature Extraction:** Consumes preprocessed data from this module
- **Vector Database:** Stores extracted features from aligned data
- **LLM and Fatigue Handling:** Receives final processed features

## Troubleshooting

**Issue:** Timestamp synchronization errors across sensors
- Solution: Verify all sensors have synchronized clocks (NTP if networked)
- Check ROS 2 time synchronization settings

**Issue:** Dropped frames during real-time processing
- Solution: Increase buffer size or reduce processing load
- Monitor CPU/GPU usage and adjust quality settings

**Issue:** Memory overflow with long sequences
- Solution: Process data in chunks or use streaming mode
- Implement circular buffer for real-time applications

## Future Enhancements

- GPU-accelerated image preprocessing with CUDA
- Adaptive frame rate adjustment based on load
- Anomaly detection for faulty sensor streams
- Integration with cloud-based data storage for fleet management
