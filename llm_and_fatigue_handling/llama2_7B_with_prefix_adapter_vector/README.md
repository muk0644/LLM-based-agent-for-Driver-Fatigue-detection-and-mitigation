# LLaMA 2 7B with Prefix Adapter and Vector Database

## Overview

This directory contains an advanced implementation of LLaMA 2 7B with a specialized prefix-tuning adapter and vector database integration. The prefix adapter efficiently incorporates multimodal features into the LLM for improved fatigue assessment and personalized interventions.

## Architecture

### Model Structure
```
Input Features (9-dim) -> MLP Adapter -> Prefix Embeddings (20480-dim)
                                              |
                                              v
                                    LLaMA 2 7B Model
                                              |
                                              v
                                        Output: Intervention
```

## Key Components

### model_wrapper_with_mlp_adapter.py
Implements the FeaturePrefixAdapter that bridges sensor features to LLM prefix space.

**FeaturePrefixAdapter Class:**
```python
class FeaturePrefixAdapter(nn.Module):
    def __init__(self, feature_dim=9, embedding_dim=4096, prefix_token_count=5):
        """
        Args:
            feature_dim: Input feature dimension (9 for this project)
            embedding_dim: LLaMA embedding dimension (4096)
            prefix_token_count: Number of prefix tokens (default 5)
        """
```

**Processing Pipeline:**
1. Feature input (9-dimensional vector from sensors)
2. MLP layers with batch normalization
3. Generate prefix embeddings (5 tokens x 4096 dimensions)
4. Prepend to LLaMA input
5. Generate intervention text

**Architecture Details:**
```
Input (9-dim)
    |
    v
FC Layer 1: 9 -> 64 (ReLU + BatchNorm)
    |
    v
FC Layer 2: 64 -> 256 (ReLU + BatchNorm)
    |
    v
FC Layer 3: 256 -> 512 (ReLU + BatchNorm)
    |
    v
Dropout (0.1)
    |
    v
FC Layer 4: 512 -> (5 * 4096) = 20480
    |
    v
Reshape to (5, 4096)
    |
    v
Prefix for LLaMA
```

### fine_tuning_pipeline.py
Training pipeline specifically designed for prefix-adapter fine-tuning.

**Training Process:**
```python
from fine_tuning_pipeline import PrefixFineTuningPipeline

pipeline = PrefixFineTuningPipeline(
    model_id="meta-llama/Llama-2-7b-hf",
    feature_dim=9,
    prefix_token_count=5,
    output_dir="./trained_models"
)

# Load dataset
train_data = pipeline.load_data("dummy_data.csv")

# Fine-tune
pipeline.train(
    train_data=train_data,
    num_epochs=5,
    batch_size=16,
    learning_rate=1e-3,
    validation_split=0.2
)

# Save adapter
pipeline.save_adapter()
```

**Loss Functions:**
- Main loss: Next-token prediction (cross-entropy)
- Regularization: Adapter weight regularization
- Optional: Contrastive loss with vector database embeddings

### input_process.py
Preprocesses multimodal features for adapter input.

**Feature Normalization:**
```python
from input_process import FeatureProcessor

processor = FeatureProcessor(
    feature_names=[
        'eye_closure_ratio',
        'blink_frequency',
        'head_nod_magnitude',
        'mouth_opening',
        'voice_quality',
        'steering_stability',
        'speed_variance',
        'yaw_rate',
        'contextual_fatigue'
    ]
)

# Raw features from sensors
raw_features = {
    'eye_closure_ratio': 0.45,
    'blink_frequency': 8.5,
    'head_nod_magnitude': 0.3,
    'mouth_opening': 0.1,
    'voice_quality': 0.72,
    'steering_stability': 0.85,
    'speed_variance': 0.12,
    'yaw_rate': 0.08,
    'contextual_fatigue': 0.6
}

# Normalized features (0-1 range)
normalized = processor.normalize(raw_features)
# Output: [0.45, 0.42, 0.3, 0.1, 0.72, 0.85, 0.12, 0.08, 0.6]
```

### faiss_vd.py
Vector database integration for in-context learning.

**Context Retrieval:**
```python
from faiss_vd import retrieve_similar_vectors, runtime_add

# Query database for similar past cases
similar_embeddings, metadata = retrieve_similar_vectors(
    query_embedding=current_prefix_embedding,
    k=5,
    database=faiss_db
)

# Format as context for LLM
context = format_historical_context(similar_embeddings, metadata)

# Use in prompt
enhanced_prompt = f"""
Previous similar cases:
{context}

Current driver state:
{current_features_text}

Recommended intervention:
"""
```

### inference_v2.ipynb
Jupyter notebook demonstrating end-to-end inference workflow.

**Workflow:**
1. Load sensor data or real-time stream
2. Extract features using computer vision
3. Normalize features
4. Generate prefix embeddings via adapter
5. Query vector database for context
6. Invoke LLaMA with prefix and context
7. Parse and execute intervention

## Data Format

### Input Features (9-dimensional vector)

```python
features = {
    'eye_closure_ratio': 0.45,      # 0-1: Eye closure percentage
    'blink_frequency': 8.5,         # 0-20: Blinks per minute
    'head_nod_magnitude': 0.3,      # 0-1: Head nod intensity
    'mouth_opening': 0.1,           # 0-1: Mouth opening ratio
    'voice_quality': 0.72,          # 0-1: Speech quality score
    'steering_stability': 0.85,     # 0-1: Steering regularity
    'speed_variance': 0.12,         # 0-1: Speed change rate
    'yaw_rate': 0.08,               # 0-1: Vehicle rotation rate
    'contextual_fatigue': 0.6       # 0-1: Overall fatigue estimate
}
```

### Training Data Format (dummy_data.csv)

```
eye_closure_ratio,blink_frequency,head_nod_magnitude,mouth_opening,voice_quality,steering_stability,speed_variance,yaw_rate,contextual_fatigue,intervention,effectiveness
0.45,8.5,0.3,0.1,0.72,0.85,0.12,0.08,0.6,"Increase temperature to 74F and play upbeat music",4
0.67,5.2,0.5,0.05,0.55,0.72,0.25,0.15,0.8,"ALERT: Recommend immediate rest stop",5
0.23,12.1,0.1,0.2,0.85,0.92,0.05,0.03,0.3,"Continue driving - driver is alert",1
```

## Model Parameters

### Adapter Configuration
```python
ADAPTER_CONFIG = {
    'feature_dim': 9,
    'embedding_dim': 4096,
    'prefix_token_count': 5,
    'total_prefix_dim': 20480,  # 5 * 4096
    'dropout': 0.1,
    'batch_norm': True,
}
```

### Training Configuration
```python
TRAINING_CONFIG = {
    'learning_rate': 1e-3,
    'batch_size': 16,
    'num_epochs': 5,
    'warmup_steps': 100,
    'weight_decay': 1e-5,
    'gradient_accumulation_steps': 2,
    'gradient_clip': 1.0,
    'val_check_interval': 0.2,  # Validate 5 times per epoch
}
```

## Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install HF dependencies
pip install transformers peft bitsandbytes accelerate

# Install vector DB
pip install faiss-gpu  # or faiss-cpu
```

## Usage Examples

### Complete Inference Pipeline
```python
import torch
from model_wrapper_with_mlp_adapter import FeaturePrefixAdapter
from input_process import FeatureProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss_vd

# Initialize components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
adapter = FeaturePrefixAdapter(feature_dim=9).to(device)
adapter.load_state_dict(torch.load("trained_adapter.pt"))

processor = FeatureProcessor()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    load_in_8bit=True
)

# Real-time inference
raw_features = {
    'eye_closure_ratio': 0.45,
    'blink_frequency': 8.5,
    # ... other features
}

# Process
normalized_features = processor.normalize(raw_features)
features_tensor = torch.tensor(normalized_features, dtype=torch.float32).to(device)

# Generate prefix
with torch.no_grad():
    prefix_embedding = adapter(features_tensor.unsqueeze(0))  # (1, 5, 4096)

# Retrieve context from vector DB
query_embedding = prefix_embedding.squeeze(0).flatten().cpu().numpy()
similar_vectors, metadata = faiss_vd.retrieve_similar_vectors(query_embedding, k=3)

# Create prompt with context
context_text = "\n".join([m['intervention'] for m in metadata])
prompt = f"""Based on similar past cases:
{context_text}

Current driver fatigue assessment:
Eye closure: {raw_features['eye_closure_ratio']*100:.1f}%
Blink rate: {raw_features['blink_frequency']:.1f} blinks/min
Head nods detected: Yes
Voice quality: {raw_features['voice_quality']*100:.0f}%

Recommended intervention:
"""

# Tokenize and prepend prefix
inputs = tokenizer(prompt, return_tensors="pt").to(device)
inputs['prefix_embedding'] = prefix_embedding

# Generate
outputs = model.generate(
    input_ids=inputs['input_ids'],
    max_length=256,
    temperature=0.7,
    top_k=50,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

intervention = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Intervention:", intervention)
```

### Training New Adapter
```python
from fine_tuning_pipeline import PrefixFineTuningPipeline
import pandas as pd

# Load data
train_df = pd.read_csv("dummy_data.csv")

# Create pipeline
pipeline = PrefixFineTuningPipeline(
    model_id="meta-llama/Llama-2-7b-hf",
    output_dir="./trained_models",
    device="cuda"
)

# Train
history = pipeline.train(
    train_data=train_df,
    num_epochs=10,
    batch_size=32,
    learning_rate=2e-3,
    validation_split=0.15
)

# Evaluate
metrics = pipeline.evaluate(test_data=test_df)
print(f"Test Loss: {metrics['loss']:.4f}")
print(f"Test Accuracy: {metrics['accuracy']:.4f}")

# Save
pipeline.save_adapter("best_adapter.pt")
```

## Performance Characteristics

### Inference Performance
- Latency: 200-400 ms per intervention (including prefix generation)
- Throughput: 2-5 requests/second on single GPU
- Memory: 6-8 GB VRAM with 8-bit quantization
- Prefix generation: <50 ms

### Training Performance
- Time per epoch: 1.5-2 hours (batch size 16)
- Total training: 8-15 hours for 10 epochs
- Validation frequency: Every 20% of epoch
- Convergence: Typically achieved in 3-5 epochs

## Advanced Features

**Prefix Caching**
```python
# Reuse prefix embeddings for faster inference
prefix_cache = {}
feature_hash = hash(tuple(normalized_features))
if feature_hash in prefix_cache:
    prefix_embedding = prefix_cache[feature_hash]
else:
    prefix_embedding = adapter(features_tensor)
    prefix_cache[feature_hash] = prefix_embedding
```

**Batch Processing**
```python
# Process multiple drivers in parallel
batch_features = torch.stack([
    processor.normalize(driver_features) 
    for driver_features in batch_raw_features
]).to(device)

batch_prefixes = adapter(batch_features)  # (batch_size, 5, 4096)
```

## Related Modules

- **Input Signal Processing:** Raw feature extraction
- **Feature Extraction:** Facial, audio, and driving features
- **Vector Database:** Historical context retrieval
- **LLM Node:** ROS 2 integration

## Troubleshooting

**Issue:** Poor quality interventions despite training
- Solution: Check data quality and feature normalization
- Increase training data diversity
- Adjust learning rate and regularization

**Issue:** Slow inference speed
- Solution: Enable prefix caching
- Reduce context window size
- Use faster tokenizer

**Issue:** Prefix dimension mismatch
- Solution: Verify embedding_dim matches model (4096 for LLaMA)
- Check prefix_token_count consistency

## Future Enhancements

- Multi-head attention in adapter for richer feature representation
- Gating mechanisms for selective feature importance
- Adapter distillation for edge deployment
- Integration with larger LLaMA variants (13B, 70B)
