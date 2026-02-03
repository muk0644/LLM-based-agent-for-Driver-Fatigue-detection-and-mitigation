# LLM-based Agent for Driver Sleepiness Detection and Mitigation in Automotive Systems

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![LLM](https://img.shields.io/badge/LLM-Llama--2--7B-34A853?logo=meta&logoColor=white)](https://huggingface.co/meta-llama)
[![FAISS](https://img.shields.io/badge/FAISS-Vector--DB-FF6F00?logo=facebook&logoColor=white)](https://faiss.ai/)
[![Computer Vision](https://img.shields.io/badge/CV-dlib%20%2B%20OpenCV-FF69B4)](https://opencv.org/)
[![ROS](https://img.shields.io/badge/ROS%202-Humble-22314E?logo=ros&logoColor=white)](https://docs.ros.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## Project Overview

Driver fatigue is one of the leading causes of road accidents globally. This project presents an intelligent system that detects driver sleepiness using multimodal data and provides real-time interventions through Large Language Models (LLMs).

The system collects data from cameras (facial monitoring), vehicle telemetry (steering angle, lane deviation), and audio signals. It then extracts fatigue-related features, estimates the driver's alertness level, and uses an LLM to generate appropriate interventions like playing music, initiating calls, or adjusting cabin temperature.

<div align="center">
  <img src="carla_simulator_setup.png" alt="CARLA Simulator Setup for Driver Fatigue Detection" width="800"/>
  <p><i>Research setup: Driver fatigue detection testing environment using CARLA simulator with steering wheel hardware interface</i></p>
</div>

---

## ⚙️ Project Workflow & System Architecture

This section explains the complete pipeline from data collection to real-time deployment. The workflow consists of six main stages:

### 1. 📹 Data Collection (CARLA Simulation)

Obtaining real-world drowsiness data is difficult and inconsistent. To address this, the **CARLA Simulator** was used to generate a synthetic dataset under controlled conditions.

**Setup:**
- Multiple participants "drove" in the simulation for approximately 20 minutes each
- Different CARLA maps were used to ensure variability in driving conditions

**Data Sources Collected:**
| Data Type | Description |
|-----------|-------------|
| **Camera** | Dashboard view capturing facial expressions for monitoring |
| **Telemetry** | Steering angle, lane deviation, and vehicle speed |
| **Audio** | Driver audio for potential speech analysis |

**Storage:** All sensor data was recorded in **ROS 2 bag files (.mcap)** using `bag_writer_for_offline_data.py` to preserve raw sensor timestamps with high precision.

---

### 2. 🔄 Data Processing & Synchronization

A major challenge was aligning data from sensors running at different frequencies (e.g., high-frame-rate camera vs. slower steering angle updates). A Python-based processing pipeline was developed to read ROS bags and convert them into CSV format.

**Two synchronization techniques were implemented:**

| Technique | Description |
|-----------|-------------|
| **Fast-Sensor Alignment** | Upsampling slower data to match the high-speed sensor |
| **Slow-Sensor Alignment** | Downsampling the fast sensor data (deleting extra frames) to match the slower sensor |

**Synchronization Threshold:** If the time difference between frames was less than **9 nanoseconds**, they were considered effectively synchronized.

**Implementation:** `data_process.py` handles the Fast/Slow alignment logic.

---

### 3. 🎯 Feature Extraction

Feature extraction was handled by two specialized teams focusing on different modalities:

#### Vision-Based Features (Camera)
Using **dlib** (68-point facial landmarks) and **OpenCV**, the following features were extracted from the video stream:

- **Blink Rate** – Frequency of eye blinks
- **Yawn Rate** – Frequency and duration of yawns
- **Head Nodding** – Patterns indicating drowsiness
- **Eye Closure Duration** – Time eyes remain closed

**Implementation:** `camera_pipeline.py`

#### Vehicle-Based Features (Telemetry)
A rule-based approach was used to extract numerical features from driving data:

- **Steering Angle Variance** – Irregular steering movements
- **Lane Deviation** – How much the vehicle drifts from the lane center

**Implementation:** `carla_data_feature_extraction.py`

---

### 4. 📊 Fatigue Estimation (Ground Truth Generation)

Training machine learning models requires "Ground Truth" labels (knowing exactly when a driver was tired). However, publicly available datasets were inconsistent—some had camera data but no steering data, and vice versa.

**Solution:** A **Rule-Based Fatigue Score** was calculated from the extracted features:

```
IF (Blink Rate > threshold_X) AND (Lane Deviation > threshold_Y) 
THEN Fatigue Score = HIGH
```

This score served as the label for training data.

**Implementation:** `fatigue_estimation.py`

---

### 5. 🤖 LLM Agents (Decision Making)

**LLaMA 2** was implemented as the reasoning engine to determine the best intervention. Three different versions were explored to optimize performance:

#### Version A: Simple LLaMA 2 (Prompt Engineering)

| Aspect | Details |
|--------|---------|
| **Method** | Raw feature data converted to textual prompt and fed to the model |
| **Result** | Successfully suggested interventions (e.g., "Play music," "Call a contact," "Turn on the dashboard fan") |
| **Limitation** | Slow inference because textual prompts consumed many tokens |

**Implementation:** `llama2_7B/simple_inference.py`

---

#### Version B: LLaMA 2 with Prefix Adapter Vector Injection (Novel Approach)

To solve the speed and context issues, a **Prefix Adapter** technique was developed:

**How it works:**
1. Instead of converting features to text, an **MLP (Multi-Layer Perceptron)** projects numerical features directly into an embedding space (dimension 4096, matching LLaMA's embedding size)
2. These feature embeddings are **injected as prefix tokens** along with a shortened textual prompt
3. The model treats these vectors as context without needing long text descriptions

**Training Process:**
- The model receives both standard Token Embeddings and custom Feature Embeddings
- The MLP adapter learns to map fatigue features into the LLM's embedding space

**Benefits:**
- Drastically shorter textual prompts
- Faster inference time
- Better integration of numerical features

**RAG & FAISS Integration:**
- Feature embeddings are stored in a **FAISS Vector Database**
- **Retrieval-Augmented Generation (RAG)** finds similar past fatigue scenarios
- Similarity search uses **Euclidean Distance** or **Cosine Similarity**
- Retrieved examples help the model make better, context-aware decisions

**Implementation:** `llama2_7B_with_prefix_adapter_vector/model_wrapper_with_mlp_adapter.py`

---

#### Version C: TinyLLaMA (Lightweight Variant)

**TinyLLaMA** was tested to evaluate feasibility for resource-constrained edge devices with limited compute power.

**Implementation:** `tiny_llama/tinyllama_inference.ipynb`

---

### 6. 🚀 Deployment: ROS 2 & Docker

The complete system is integrated into a real-time application using **ROS 2 (Robot Operating System)** and containerized with **Docker**.

#### Node Architecture

In ROS 2, every algorithm runs as a **"Node"** (an independent Python executable). Nodes communicate via **Topics** using a Publisher/Subscriber model.

#### End-to-End Data Flow

```
┌─────────────────┐    ┌──────────────────────┐    ┌────────────────────────┐
│   Sensor Node   │───►│  Preprocessing Node  │───►│ Feature Extraction Node│
│ (Publishes raw  │    │ (Subscribes, syncs,  │    │  (Calculates fatigue   │
│     data)       │    │  publishes clean)    │    │      metrics)          │
└─────────────────┘    └──────────────────────┘    └───────────┬────────────┘
                                                               │
                                                               ▼
                       ┌──────────────────────┐    ┌────────────────────────┐
                       │   Actuator Node      │◄───│      LLM Node          │
                       │ (Triggers physical   │    │ (Runs inference,       │
                       │   car systems)       │    │  publishes commands)   │
                       └──────────────────────┘    └────────────────────────┘
```

**Workflow Steps:**
1. **Sensor Node** → Publishes raw camera, telemetry, and audio data
2. **Preprocessing Node** → Subscribes to raw data, synchronizes timestamps, publishes clean data
3. **Feature Extraction Node** → Subscribes to clean data, calculates fatigue metrics, publishes features
4. **LLM Node** → Subscribes to features, runs model inference, publishes intervention commands (e.g., "Start Fan")
5. **Actuator Node** → Receives commands and triggers the actual car system

**Implementation:** `llm_node.py` serves as the main ROS 2 wrapper that runs the LLM in real-time.

---

## 🔗 Architecture to Code Mapping

This table shows how each workflow step maps to specific files in the repository:

| Workflow Step | Corresponding File/Folder | Description |
|---------------|--------------------------|-------------|
| Data Collection | `input_signal_processing/src/bag_writer_for_offline_data.py` | Writing ROS 2 bags from CARLA |
| Synchronization | `input_signal_processing/data_process.py` | Fast/Slow alignment logic |
| Vision Features | `feature_extraction/camera_pipeline.py` | dlib-based landmark detection |
| Vehicle Features | `feature_extraction/carla_data_feature_extraction.py` | Telemetry feature extraction |
| Ground Truth | `feature_extraction/fatigue_estimation.py` | Rule-based fatigue score calculation |
| LLM (Simple) | `llm_and_fatigue_handling/llama2_7B/` | Basic prompt engineering approach |
| LLM (Prefix Adapter) | `llm_and_fatigue_handling/llama2_7B_with_prefix_adapter_vector/` | MLP adapter and vector injection |
| LLM (Lightweight) | `llm_and_fatigue_handling/tiny_llama/` | TinyLLaMA for edge devices |
| Vector Database | `vector_database/faiss_vd.py` | FAISS integration for RAG |
| ROS 2 Deployment | `llm_and_fatigue_handling/llm_node.py` | Real-time LLM ROS 2 node |
| Docker | `docker/Dockerfile` | Container configuration |

---

## 📁 Project Structure

```
LLM-based-Agent-for-Driver-Sleepiness-Detection/
│
├── 🖼️  carla_simulator_setup.png            CARLA simulator testing setup image
├── 📝 README.md                             Project documentation
├── 📄 addingthingstoreadme.md               Additional documentation notes
│
├── 📊 input_signal_processing/              Data preprocessing and synchronization
│   ├── 🐍 data_process.py                  Multi-stage data cleaning and normalization
│   ├── 🔄 data_process_only_sync.py        Lightweight synchronization only
│   ├── 📁 src/
│   │   ├── 💾 bag_writer_for_offline_data.py  ROS 2 bag file writer for CARLA data
│   │   ├── ⚙️  preprocess_node.py          ROS 2 preprocessing node
│   │   └── 🔗 sync_node.py                 ROS 2 synchronization node
│   ├── 📄 readme.txt
│   └── 📖 README.md
│
├── 🎥 feature_extraction/                   Multimodal feature extraction pipelines
│   ├── 📹 camera_pipeline.py               Facial feature extraction using dlib landmarks
│   ├── 🏎️  carla_data_feature_extraction.py CARLA telemetry data processing
│   ├── 😴 fatigue_estimation.py            Rule-based fatigue score computation
│   └── 📖 README.md
│
├── 🔍 vector_database/                      FAISS-based vector storage for RAG
│   ├── 🗂️  faiss_vd.py                     Vector database implementation
│   ├── 📄 readme.txt
│   └── 📖 README.md
│
├── 🤖 llm_and_fatigue_handling/             LLM inference and decision-making
│   ├── 🧠 llm_node.py                      Main LLM ROS 2 node for real-time inference
│   ├── 🏷️  generate_fatigue_labels.py      Label generation utility
│   │
│   ├── 🦙 llama2_7B/                       Base LLaMA 2 7B (Prompt Engineering)
│   │   ├── 🔮 simple_inference.py          Inference engine
│   │   ├── 📥 simple_input_process.py      Feature preprocessing
│   │   ├── 🎓 simple_fine_tuning_pipeline.py Fine-tuning pipeline
│   │   ├── 🗂️  simple_faiss_vd.py          Vector database integration
│   │   ├── 📊 captured_data.csv            Training dataset
│   │   ├── 📄 readme.txt
│   │   └── 📖 README.md
│   │
│   ├── ⚡ llama2_7B_with_prefix_adapter_vector/  Prefix Adapter (Novel Approach)
│   │   ├── 🔧 model_wrapper_with_mlp_adapter.py  MLP adapter for feature-to-embedding
│   │   ├── 🎓 fine_tuning_pipeline.py        Prefix-tuning pipeline
│   │   ├── 📓 inference_v2.ipynb             Inference notebook
│   │   ├── 📥 input_process.py               Feature normalization
│   │   ├── 🗂️  faiss_vd.py                   Vector database integration
│   │   ├── 📊 dummy_data.csv                 Training dataset
│   │   ├── 📄 readme.txt
│   │   └── 📖 README.md
│   │
│   └── 🏃 tiny_llama/                      Lightweight TinyLLaMA for edge devices
│       ├── 📓 tinyllama_inference.ipynb    Inference notebook
│       ├── 🎓 tinyllama_fine_tuning_pipeline.py Fine-tuning pipeline
│       ├── 📥 tinyllama_input_process.py   Feature preprocessing
│       ├── 🗂️  tinyllama_faiss_vd.py       Vector database integration
│       ├── 📊 dummy_data.csv               Training dataset
│       ├── 📄 readme.txt
│       └── 📖 README.md
│
└── 🐳 docker/                               Containerized deployment setup
    ├── 📦 Dockerfile                       Container build configuration
    └── 📁 src/
        └── 🔗 data_sync/                   ROS 2 data synchronization package
            ├── 🐍 __init__.py
            ├── 📋 package.xml              ROS 2 package manifest
            ├── 🔨 CMakeLists.txt
            ├── 📁 scripts/
            │   └── 🔗 sync_node.py
            └── 📁 msg/
                └── 💬 SyncedOutput.msg
```

---

## 🛠️ Technical Stack

| Category | Technologies |
|----------|--------------|
| **Framework** | ROS 2 (Humble/Foxy), Docker |
| **AI/ML** | PyTorch, PEFT (LoRA, Prefix Tuning), BitsAndBytes quantization |
| **Language Models** | Meta LLaMA 2 (7B), TinyLLaMA |
| **Vision Processing** | dlib (68-point landmarks), OpenCV |
| **Vector Database** | FAISS (Facebook AI Similarity Search) |
| **Hardware Acceleration** | NVIDIA CUDA |

---

## ⚠️ Important Implementation Note

While the initial research proposal references "Vision Transformers" and "Informer Transformers," the actual implementation primarily uses **dlib** and **OpenCV** for vision processing, and **rule-based heuristics** for telemetry analysis.

**Reasoning:** In testing, standard computer vision pipelines (dlib) provided significantly **lower latency** and **higher reliability** for the real-time constraints of a moving vehicle compared to heavy Transformer-based vision models.

---

## 🚀 Installation & Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- ROS 2 (Humble or Foxy)
- 16GB+ RAM (32GB recommended for LLM inference)

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/muk0644/LLM-based-agent-for-Driver-Fatigue-detection-and-mitigation
cd LLM-based-Agent-for-Driver-Sleepiness-Detection

# Build and run with Docker
cd docker/
docker build -t drowsiness-detection:latest .
docker run -it --gpus all drowsiness-detection:latest
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/muk0644/LLM-based-agent-for-Driver-Fatigue-detection-and-mitigation
cd LLM-based-Agent-for-Driver-Sleepiness-Detection

# Install Python dependencies
pip install -r requirements.txt

# Download LLaMA 2 model (requires Hugging Face access token)
huggingface-cli login

# Build ROS 2 packages
colcon build

# Run the system
ros2 launch drowsiness_detection system.launch.py
```

---

## 📊 Dataset

The system was trained and evaluated on:

- **CARLA Simulation Data** – Synthetic driving data with synchronized sensors
- **Multimodal Recordings** – Camera, telemetry, and audio data
- **Ground Truth Labels** – Rule-based fatigue scores (Alert/Drowsy/Sleepy)
- **Intervention Outcomes** – Effectiveness of suggested actions

---

## 📈 Evaluation Metrics

The system is evaluated on:

| Metric | Description |
|--------|-------------|
| **Detection Accuracy** | Correctness of fatigue state classification |
| **Response Latency** | Time from feature input to intervention output |
| **Intervention Relevance** | Appropriateness of suggested actions |
| **False Positive Rate** | Incorrect fatigue alerts (critical for user trust) |

---

## ⚙️ Configuration

Key parameters can be adjusted in configuration files:

- Model IDs and paths for different LLM variants
- Feature extraction thresholds (blink rate, yawn detection sensitivity)
- Fatigue score thresholds
- Intervention strategies and priorities
- ROS 2 topic names and publishing rates

---

## 👥 Supervisor and Institution

This project was completed as a team project at **Technische Hochschule Ingolstadt**, Masters Program, under the supervision of:

**Prof. Dr. Ignacio Alvarez**  
*Professor für Human-Centered Intelligent Systems*

---

## 📜 License

This project is provided for academic and research purposes.

---

## ⚠️ Disclaimer

The details provided here are subject to change as the project evolves. Performance in real-world scenarios may vary based on environmental conditions, camera calibration, and individual driver variations. This system is designed as a **research prototype** and should be integrated with additional safety systems for production use.
