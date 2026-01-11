# LLaMA 2 7B Base Model Implementation

## Overview

This directory contains the base LLaMA 2 7B implementation for the drowsiness detection system. It provides core functionality for inference, fine-tuning, and feature processing using Meta's LLaMA 2 7B language model.

## Components

### simple_inference.py
Inference engine for real-time fatigue assessment and intervention generation.

**Key Functions:**
- Load pre-trained LLaMA 2 7B model with BitsAndBytes quantization
- Process driver fatigue features and generate natural language interventions
- Handle token generation with temperature and top-k sampling
- Stream output for low-latency response

**Usage:**
```python
from simple_inference import LlamaInferenceEngine

engine = LlamaInferenceEngine(
    model_id="meta-llama/Llama-2-7b-hf",
    hf_token="your_token_here",
    device="cuda"
)

fatigue_report = {
    'eye_closure_ratio': 0.55,
    'blink_frequency': 5.2,
    'head_nod': True,
    'confidence': 0.89,
    'time_awake': 600  # seconds
}

intervention = engine.generate_intervention(fatigue_report, max_length=256)
print(intervention)
```

### simple_input_process.py
Feature processing and input formatting for the LLaMA model.

**Responsibilities:**
- Convert raw sensor features into text prompts
- Normalize numerical features for consistent input
- Create contextual prompts for fatigue assessment
- Handle edge cases and missing data

**Feature Processing:**
```python
from simple_input_process import FeatureProcessor

processor = FeatureProcessor()

# Raw sensor features
sensor_data = {
    'eye_closure_ratio': 0.45,
    'blink_frequency': 8.5,
    'head_movement': 'moderate',
    'speech_quality': 'degraded',
    'steering_stability': 0.72,
    'time_on_road': 180
}

# Generate prompt for LLM
prompt = processor.create_prompt(sensor_data)
# Output: "The driver shows signs of fatigue: eye closure ratio is 45%, blink frequency is 8.5 per minute..."
```

### simple_fine_tuning_pipeline.py
Training pipeline for task-specific adaptation of the base model.

**Fine-Tuning Process:**
1. Load pre-trained LLaMA 2 7B
2. Apply LoRA (Low-Rank Adaptation) for efficient fine-tuning
3. Train on fatigue-intervention pairs
4. Validate on held-out test set
5. Save adapter weights separately

**Training Example:**
```python
from simple_fine_tuning_pipeline import FineTuningPipeline

pipeline = FineTuningPipeline(
    model_id="meta-llama/Llama-2-7b-hf",
    output_dir="./trained_models/llama2_7b_ft"
)

# Load training data
training_data = pipeline.load_data("captured_data.csv")

# Fine-tune model
pipeline.train(
    train_data=training_data,
    num_epochs=3,
    batch_size=8,
    learning_rate=1e-4
)

# Save model
pipeline.save_model()
```

### captured_data.csv
Training dataset containing:
- Driver fatigue features (numerical)
- Corresponding interventions (text)
- Context information (time, weather, road conditions)
- Outcomes (driver response, effectiveness rating)

**Data Format:**
```
eye_closure_ratio,blink_freq,head_nod,confidence,intervention,effectiveness
0.45,8.5,true,0.89,"Increase cabin temperature to 72F",4
0.67,5.2,true,0.95,"Recommend rest stop in 10 minutes",5
0.23,12.1,false,0.78,"Conversation starter: Tell me about your destination",3
```

## Model Configuration

### BitsAndBytes Quantization
```python
# 8-bit quantization for memory efficiency
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf8",
)
```

### LoRA Configuration (Fine-tuning)
```python
# Low-rank adaptation for efficient training
lora_config = LoraConfig(
    r=8,                           # Rank
    lora_alpha=16,                 # Scaling factor
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Specific packages needed
pip install transformers peft bitsandbytes torch accelerate pandas
```

## Usage Examples

### Real-time Inference
```python
from simple_inference import LlamaInferenceEngine

# Initialize engine
engine = LlamaInferenceEngine(
    model_id="meta-llama/Llama-2-7b-hf",
    hf_token="your_huggingface_token",
    device="cuda",
    quantize_8bit=True
)

# Process features
features = {
    'fatigue_level': 'drowsy',
    'eye_closure': 0.55,
    'blink_rate': 5.0,
    'head_position': 'forward_drooping'
}

# Generate intervention
output = engine.generate_intervention(
    features=features,
    max_length=256,
    temperature=0.7,
    top_k=50
)

print("LLM Output:", output)
```

### Batch Processing
```python
from simple_inference import LlamaInferenceEngine

engine = LlamaInferenceEngine()

# Process multiple feature sets
batch_features = [
    {'eye_closure': 0.45, 'blink_rate': 8.5, ...},
    {'eye_closure': 0.67, 'blink_rate': 5.2, ...},
    {'eye_closure': 0.23, 'blink_rate': 12.1, ...},
]

interventions = []
for features in batch_features:
    intervention = engine.generate_intervention(features)
    interventions.append(intervention)
```

### Fine-tuning with Custom Data
```python
from simple_fine_tuning_pipeline import FineTuningPipeline

# Create pipeline
pipeline = FineTuningPipeline(
    model_id="meta-llama/Llama-2-7b-hf",
    output_dir="./my_trained_model"
)

# Load your data
train_df = pd.read_csv("my_training_data.csv")

# Train
pipeline.train(
    train_data=train_df,
    val_split=0.1,
    num_epochs=5,
    batch_size=16,
    learning_rate=2e-4,
    warmup_steps=100
)

# Evaluate
metrics = pipeline.evaluate(test_data=test_df)
print(f"Validation Loss: {metrics['val_loss']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## Performance Specifications

### Inference Performance
- Latency: 100-300 ms per intervention generation (GPU)
- Throughput: 3-10 inference requests per second
- Memory: 4-6 GB VRAM with 8-bit quantization
- Model size: 7B parameters

### Fine-tuning Performance
- Training time: 1-2 hours per epoch (batch size 8)
- Memory required: 16 GB VRAM
- Convergence: 3-5 epochs typical
- Model size after LoRA: Base model + 5% additional parameters

## Key Features

**Efficient Inference**
- BitsAndBytes 8-bit quantization reduces memory footprint
- Cached attention for faster token generation
- Batch processing support for multiple requests

**Task-Specific Adaptation**
- LoRA fine-tuning preserves base model knowledge
- Requires minimal additional parameters
- Fast adaptation to new fatigue patterns

**Contextual Understanding**
- Generates human-readable interventions
- Incorporates historical context from vector database
- Personalized responses based on driver history

## Input Feature Description

The model expects driver fatigue features:
- Eye closure ratio (0-1): Percentage of time eyes are closed
- Blink frequency (0-20): Blinks per minute
- Head nod detection (boolean): Whether head nodding is detected
- Speech quality (0-1): Normalized quality metric
- Steering stability (0-1): Regularity of steering inputs
- Time awake (seconds): Duration since last significant rest
- Confidence (0-1): Model confidence in fatigue assessment

## Output Intervention Types

LLM generates interventions in these categories:
- Temperature control: "Increase cabin temperature to 74F for alertness"
- Engagement: "Ask the driver about their destination and preferred route"
- Timing: "Recommend a rest stop in 15 minutes"
- Navigation: "Suggest a safer route with better visibility"
- Alerts: "ALERT: Critical fatigue detected - immediate action required"

## Troubleshooting

**Issue:** Out of memory errors during inference
- Solution: Enable 8-bit quantization
- Reduce batch size
- Use fp16 precision instead of fp32

**Issue:** Slow inference speed
- Solution: Enable GPU acceleration
- Implement token caching
- Use smaller context window

**Issue:** Poor quality interventions
- Solution: Fine-tune on domain-specific data
- Adjust temperature and sampling parameters
- Verify feature preprocessing

## Related Modules

- **Input Signal Processing:** Provides raw features
- **Feature Extraction:** Generates feature vectors
- **Vector Database:** Supplies historical context
- **LLM Node:** ROS 2 wrapper for real-time deployment

## Future Enhancements

- Integration with larger models (13B, 70B parameters)
- Multi-language support for international drivers
- Personalization learning from individual driver feedback
- Real-time adaptation with online learning
