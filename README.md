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

**The Problem:** Driver fatigue (sleepiness) is one of the leading causes of road accidents. This project creates a smart system that can detect when a driver is getting tired and automatically suggests helpful actions to keep them alert.

**How It Works:** The system collects data from multiple sources:
- **Camera** – Records facial expressions to detect yawning, blinking, and eye closure
- **Vehicle Sensors** – Monitors steering angle and lane position (irregular driving indicates drowsiness)
- **Audio** – Captures driver sounds for voice analysis

**What It Does:** Using an Artificial Intelligence model called LLaMA 2, the system:
1. Analyzes all this data in real-time
2. Calculates a "fatigue score" (is the driver alert, drowsy, or sleepy?)
3. Recommends appropriate actions like:
   - Playing music to wake them up
   - Calling a contact for conversation
   - Adjusting cabin temperature (fan, AC)
   - Suggesting rest stops

This is a complete automated solution that runs continuously during driving and reacts instantly when drowsiness is detected.

<div align="center">
  <img src="carla_simulator_setup.png" alt="CARLA Simulator Setup for Driver Fatigue Detection" width="800"/>
  <p><i>Research setup: Driver fatigue detection testing environment using CARLA simulator with steering wheel hardware interface</i></p>
</div>

---

## ⚙️ Project Workflow & System Architecture

This section explains the complete pipeline from data collection to real-time deployment. The workflow consists of six main stages:

### 1. 📹 Data Collection (CARLA Simulation)

**Why Simulation?** Real-world drowsy driving data is hard to collect safely and ethically. Instead, the project used the **CARLA Simulator** – a realistic virtual driving environment that allows testing without real safety risks.

**How Data Was Collected:**
- Multiple people "drove" in the simulator for about 20 minutes each session
- Different routes and maps were used to get varied driving scenarios
- All sensor data (camera, steering, audio) was recorded simultaneously with precise timestamps

**What Data Was Captured:**

| Data Type | What It Captures | Why It Matters |
|-----------|-----------------|---------------|
| **Camera** | Facial video from dashboard view | Detects eye blinking, yawning, head position |
| **Telemetry** | Steering angle, lane position, speed | Detects swerving and loss of control |
| **Audio** | Driver sounds and speech | Detects slurred speech, yawning sounds |

**Storage Method:** All data was saved in **ROS 2 bag files (.mcap)** – a special format that preserves exact timing information so sensors can be synchronized later. File: `bag_writer_for_offline_data.py`

---

### 2. 🔄 Data Processing & Synchronization

**The Challenge:** Different sensors run at different speeds:
- **Fast sensor:** Camera records 30 frames per second
- **Slow sensor:** Steering wheel sensor records data only 10 times per second

If you try to combine these directly, the timing gets misaligned (like trying to sync audio and video that are out of sync).

**The Solution:** A Python-based processing pipeline reads all the data and aligns the timestamps. It converts ROS bag files into easy-to-use CSV (spreadsheet) format.

**Two Alignment Methods:**

| Method | How It Works | When to Use |
|--------|------------|------------|
| **Fast-Sensor Alignment** | Slow sensors get "filled in" to match fast sensor's rate (make 10 readings into 30) | Want all timestamps synchronized |
| **Slow-Sensor Alignment** | Fast sensors get reduced to match slow sensor's rate (keep only some of the 30 frames) | Need less data, smaller files |

**Precision Threshold:** If two sensor readings arrived within **9 nanoseconds** of each other, they're considered synchronized (that's 0.000000009 seconds – extremely precise!).

**Result:** Clean, time-aligned data ready for feature extraction. File: `data_process.py`

---

### 3. 🎯 Feature Extraction

**What are Features?** Features are specific measurements or observations that indicate drowsiness. The system extracts them from multiple data sources.

#### Vision-Based Features (What the Camera Detects)

Using **dlib** (an open-source tool for facial analysis) and **OpenCV** (image processing), the system detects:

| Feature | What It Measures | Drowsy Indicator |
|---------|-----------------|-----------------|
| **Blink Rate** | How often the driver blinks | Slower blinks = drowsier |
| **Yawn Rate** | Frequency of yawning | More yawns = more tired |
| **Head Nodding** | Repeated head movements | Nodding = falling asleep |
| **Eye Closure Duration** | How long eyes stay closed | Longer closures = drowsy |

**Technical Details:** The system tracks 68 specific landmarks on the face (eyes, nose, mouth, chin) to calculate these measurements.

**File:** `camera_pipeline.py`

#### Vehicle-Based Features (What the Sensors Detect)

Using steering and lane data, the system calculates:

| Feature | What It Measures | Drowsy Indicator |
|---------|-----------------|-----------------|
| **Steering Variance** | How much steering changes (jittery vs. smooth) | Erratic steering = loss of control |
| **Lane Deviation** | How far the car drifts from the center | Drifting = inattentive |

**Logic:** Tired drivers lose focus and their driving becomes irregular and unpredictable.

**File:** `carla_data_feature_extraction.py`

---

### 4. 📊 Fatigue Estimation (Ground Truth Generation)

**What is Ground Truth?** To train AI models, you need correct answers. For example, you can't train a drowsiness detector without knowing exactly when the driver WAS actually drowsy. This is called "ground truth."

**The Problem:** Most public datasets are incomplete – some have camera data but no steering data, others have the opposite. There's no reliable dataset with all sensors synchronized.

**The Solution:** Instead of using unreliable public data, the system creates its own "ground truth" using simple rules based on the extracted features.

**How the Rule Works:**

```
IF (Blink Rate is HIGH) AND (Lane Deviation is HIGH)
THEN Fatigue Level = HIGH (Driver is drowsy)

IF (Blink Rate is MEDIUM) AND (Lane Deviation is MEDIUM)
THEN Fatigue Level = MEDIUM (Driver is drowsy but not critical)

IF (Blink Rate is LOW) AND (Lane Deviation is LOW)
THEN Fatigue Level = LOW (Driver is alert)
```

**Simple Logic:** When multiple drowsiness indicators are high at the same time, the driver is drowsy. When all indicators are normal, the driver is alert.

**Result:** Automatically generated labels for training data based on actual observed behavior in the simulator.

**File:** `fatigue_estimation.py`

---

### 5. 🤖 LLM Agents (Decision Making)

**What's an LLM?** An LLM (Large Language Model) like ChatGPT is an AI that can understand information and make smart decisions. This project uses **LLaMA 2** – Meta's powerful AI language model.

**Why Use an LLM?** Simple rules aren't enough. When the system detects drowsiness, it needs to decide: What's the best action for THIS specific driver? Different people respond to different interventions. An LLM can reason and personalize responses.

**Three Versions Tested:**

#### Version A: Simple LLaMA 2 (Prompt Engineering)

**The Idea:** Convert numbers into words that the LLM can understand.

**How It Works:**
1. Take the extracted features (blink rate = 25/min, yawn rate = 3/min, lane deviation = 15cm)
2. Convert to natural English: "The driver is blinking 25 times per minute, yawning 3 times, and drifting 15cm from the lane center"
3. Send this text to LLaMA 2
4. Ask the model: "Given this information, what should we do to keep the driver awake?"
5. The LLM generates smart responses like:
   - "Play upbeat music to increase alertness"
   - "Call a friend to have a conversation"
   - "Turn on the air conditioner to provide sensory stimulation"

| Aspect | Details |
|--------|---------|
| **Strength** | Works well, generates intelligent suggestions |
| **Weakness** | Slow – longer text = more processing time = slower response |
| **Speed Issue** | The textual description takes many "tokens" (word pieces) for the AI to process. More tokens = slower inference |
| **Best For** | When response time isn't critical; good for learning |

**LLaMA 2 Variants Explored:**
1. **LoRA-fine-tuned LLaMA 2** – Efficiently adapts the base model to your specific task
2. **Prompt-based LLaMA 2** – Uses prompt engineering without additional training
3. **TinyLLaMA** – Smaller, faster version for resource-limited devices

**Output:** The system generates intelligent intervention suggestions, stored in CSV files for later review and improvement.

**File:** `llama2_7B/simple_inference.py`

---

#### Version B: LLaMA 2 with Prefix Adapter Vector Injection (Novel Approach - Our Innovation)

**The Problem with Version A:** Converting numbers to text takes time. The LLM has to read and process all that text, which slows down the response.

**The Innovation:** Instead of converting numbers to words, directly feed the numbers into the AI's brain as "embeddings" (a special mathematical representation).

**How It Works:**

1. **Feature Conversion:** Use a small neural network (MLP - Multi-Layer Perceptron) to convert raw numbers into a special format (4096 dimensions) that LLaMA 2 understands
   - Input: [blink_rate=25, yawn_rate=3, lane_deviation=15, ...]
   - Output: A 4096-dimensional mathematical representation

2. **Injection:** Instead of writing long text, inject these numbers directly as "context" (prefix tokens)
   - The LLM understands these vectors as drowsiness information
   - Much shorter than writing out a description
   - Faster for the AI to process

3. **Training:** The model learns to interpret these embedded features correctly
   - The MLP learns what each feature pattern means
   - The LLM learns to respond appropriately

| Aspect | Details |
|--------|---------|
| **Speed Improvement** | 10-100x faster than Version A (much fewer tokens to process) |
| **Text Used** | Can use much shorter prompts (fewer words) |
| **Accuracy** | Better understanding of numerical patterns |
| **Memory** | More efficient use of AI's attention capacity |
| **Best For** | Real-time vehicle systems where speed matters |

**Memory & Learning Enhancement – RAG (Retrieval-Augmented Generation):**

The system can learn from its past decisions:

1. **Storage:** Every feature vector is stored in a **FAISS Vector Database** (super-fast search engine for AI vectors)

2. **Similarity Search:** When facing a new drowsy situation, the system searches for similar past situations
   - Uses **Euclidean Distance** or **Cosine Similarity** (two ways to measure "closeness" in AI space)
   - Finds the 3-5 most similar past scenarios

3. **Context Learning:** The LLM says: "I've seen this pattern before, and here's what worked..."
   - Uses past successes to make better current decisions
   - Continuously improves through experience

**Example:**
- Past: "When blink_rate=25 and yawn_rate=3, playing upbeat music worked best"
- New Situation: "Blink rate is 24 and yawn rate is 3 – very similar!"
- Decision: "Let's play upbeat music again"

**Training Process:**
- MLP learns: How to convert features into meaningful embeddings
- LLaMA learns: How to interpret these embeddings and make decisions
- FAISS learns: Store embeddings for fast similarity matching

**File:** `llama2_7B_with_prefix_adapter_vector/model_wrapper_with_mlp_adapter.py`

---

#### Version C: TinyLLaMA (Lightweight Option)

**When Size Matters:** Some vehicles have limited computing power (phones, old car computers). Full-size LLaMA 2 is too heavy.

**The Solution:** TinyLLaMA is a smaller, faster version of LLaMA 2
- Uses much less memory (fits on smaller devices)
- Runs faster (good for older hardware)
- Still makes intelligent decisions (though slightly less advanced)

**Trade-off:** Speed and efficiency for slightly less sophistication

**Best For:** Older vehicles, edge devices, mobile phones, resource-constrained environments

**File:** `tiny_llama/tinyllama_inference.ipynb`

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

**ROS 2 System Details:**

Each component is implemented as an independent Python executable (Node). Nodes communicate through ROS topics using a Publisher/Subscriber (Pub/Sub) model:

- **Nodes:** Independent workers that perform specific tasks
- **Topics:** Channels where data is broadcast (like message queues)
- **Publishers:** Nodes that send data to a topic
- **Subscribers:** Nodes that receive data from a topic

**Design Benefits:**
- Modular and scalable system design
- Each node can act as both a publisher and subscriber
- Easy to update or replace individual components
- Real-time message passing between processing modules
- Nodes can run in parallel on different CPU cores

**Data Integration:**
- The trained LLM models are integrated directly into ROS nodes
- A node subscribes to feature topics, performs inference, and publishes the resulting fatigue or intervention data
- Downstream nodes (including those controlling autonomous driving) receive these predictions
- The system supports end-to-end real-time operation

**Synchronization in ROS 2:**
- If the timestamp difference between sensor messages was less than **9 nanoseconds**, the data was considered synchronized
- This ensures precise temporal alignment across all sensor streams

---

## � Architecture to Code Mapping

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

## 📤 Output & Predictions Storage

After the LLM makes fatigue predictions and recommends interventions, the results are handled as follows:

**Output Format:** Fatigue predictions and intervention recommendations are stored in **CSV format** for:
- Further training and model improvement
- Evaluation and performance analysis
- Historical logging of driver sessions
- Feedback loops for continuous learning

**Data Stored:** Each record contains:
- Timestamp of prediction
- Extracted features (blink rate, yawn rate, steering variance, etc.)
- Predicted fatigue level (Alert/Drowsy/Sleepy)
- Recommended intervention
- Confidence scores

**Performance Factors:**
- Inference speed and response time depend on token length in prompt-based models
- Token length affects generation size and overall latency
- Prefix Adapter models have faster inference due to reduced token usage

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
