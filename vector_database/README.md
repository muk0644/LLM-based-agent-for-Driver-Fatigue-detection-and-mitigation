# Vector Database Module

## Overview

This module implements a FAISS-based (Facebook AI Similarity Search) vector database system for efficient storage, retrieval, and management of feature embeddings. It enables fast nearest-neighbor searches for context-aware LLM interventions and in-context learning with relevant historical examples.

## Key Components

### faiss_vd.py - PrefixVectorDB Class

High-performance vector database optimized for prefix embedding storage and retrieval.

#### Core Features

**Index Management**
- Uses FAISS IndexFlatL2 for exact L2 distance computation
- Supports immediate persistence to disk
- Maintains metadata mapping (embedding ID to fatigue event information)

**Vector Operations**
- Stores 20480-dimensional prefix embeddings (5 tokens x 4096 dimensions)
- Handles vector normalization for cosine similarity
- Batch insertion and retrieval operations

**Metadata Storage**
- Pickle-based serialization for Python object storage
- Stores fatigue labels, timestamps, driver context, and outcomes
- Enables comprehensive logging of interventions

#### API Methods

```python
# Initialization
db = PrefixVectorDB(dim=20480, 
                   index_path='index.faiss', 
                   metadata_path='metadata.pkl')

# Add embeddings
db.add_vector(embedding_vector, label='drowsy', context_dict)

# Search for similar vectors
distances, indices = db.search(query_vector, k=5)

# Batch operations
db.bulk_insert(embedding_array, label_array, metadata_list)

# Persistence
db.save()  # Saves index and metadata to disk
```

## Data Structure

### Embedding Format

Each stored embedding represents a driver fatigue state snapshot:

```
Embedding (20480 dim):
├── Prefix tokens [0:4096]     - Vision features (eye, mouth, head)
├── Prefix tokens [4096:8192]  - Audio features (voice, tone)
├── Prefix tokens [8192:12288] - Driving behavior features
├── Prefix tokens [12288:16384] - Temporal context
└── Prefix tokens [16384:20480] - Contextual metadata
```

### Metadata Storage

For each vector ID:
```python
{
    'timestamp': '2024-01-15T14:30:45Z',
    'fatigue_label': 'drowsy',  # alert, drowsy, sleepy
    'confidence': 0.87,
    'features': {
        'eye_closure_ratio': 0.45,
        'blink_frequency': 8.2,
        'head_nod': True,
        'steering_irregularity': 0.31,
        'voice_energy': 0.56
    },
    'intervention': 'increase_ac_temperature',
    'outcome': 'driver_acknowledged',
    'driver_id': 'D001',
    'vehicle_id': 'V0042',
    'location': {'latitude': 37.7749, 'longitude': -122.4194},
    'weather': 'clear',
    'time_of_day': 'evening'
}
```

## Usage Examples

### Basic Usage

```python
from faiss_vd import PrefixVectorDB, runtime_add, retrieve_similar_vectors

# Initialize database
db = PrefixVectorDB(dim=20480, 
                   index_path='fatigue_database.faiss',
                   metadata_path='fatigue_metadata.pkl')

# Add new fatigue event embedding
embedding = model.get_prefix_embedding(features)  # 20480 dim
db.add_vector(embedding, 
             label='drowsy',
             metadata={
                 'driver_id': 'D001',
                 'confidence': 0.92,
                 'features': features_dict
             })

# Search for similar past events
query_embedding = model.get_prefix_embedding(current_features)
distances, indices = db.search(query_embedding, k=10)

# Retrieve metadata for top matches
for idx in indices:
    context = db.metadata[idx]
    print(f"Similar event: {context['fatigue_label']}")
    print(f"Suggested intervention: {context['intervention']}")
```

### Runtime Integration with LLM

```python
from faiss_vd import runtime_add, retrieve_similar_vectors

# During inference - retrieve context for LLM
similar_vectors, metadata = retrieve_similar_vectors(
    query_embedding=current_embedding,
    k=5,
    faiss_index=db.index,
    metadata_store=db.metadata
)

# Format for LLM in-context learning
context_prompt = format_context(similar_vectors, metadata)

# LLM generates intervention using historical context
intervention = llm.generate(
    query=current_fatigue_features,
    context=context_prompt,
    max_length=256
)

# Log result for future learning
runtime_add(
    embedding=current_embedding,
    label=predicted_label,
    metadata=event_metadata,
    db=db
)
```

## Configuration

### Index Parameters

```python
# Database configuration
DB_CONFIG = {
    'embedding_dimension': 20480,
    'index_type': 'IndexFlatL2',  # L2 distance (Euclidean)
    'index_path': '/data/models/faiss_database.faiss',
    'metadata_path': '/data/models/metadata.pkl',
    'max_vectors': 1_000_000,
    'search_k': 5,  # Default number of results per query
}

# Normalization
NORMALIZE_EMBEDDINGS = True  # For cosine similarity
NORMALIZE_METHOD = 'l2'  # L2 normalization before storage
```

## Storage Requirements

- Index file size: ~80 MB per 100,000 vectors (4KB per vector)
- Metadata size: ~1-2 KB per vector
- Total for 100K vectors: ~100-200 MB

## Performance Characteristics

- **Insertion:** 10,000+ vectors/second
- **Search latency:** 1-5 ms for k=5 nearest neighbors
- **Memory footprint:** 4-5 KB per stored embedding
- **Scalability:** Efficient up to 10+ million vectors

## Database Maintenance

### Periodic Operations

```python
# Backup database
import shutil
shutil.copy('fatigue_database.faiss', 'backup/database_20240115.faiss')
shutil.copy('metadata.pkl', 'backup/metadata_20240115.pkl')

# Rebuild index for optimization (if fragmented)
db.rebuild_index()

# Statistics
print(f"Total vectors: {db.index.ntotal}")
print(f"Metadata entries: {len(db.metadata)}")
```

### Cleanup Old Entries

```python
# Remove entries older than 1 year
threshold_date = datetime.now() - timedelta(days=365)
ids_to_remove = [
    vid for vid, meta in db.metadata.items()
    if datetime.fromisoformat(meta['timestamp']) < threshold_date
]
db.remove_vectors(ids_to_remove)
```

## Integration with Other Modules

### Feature Extraction -> Vector Database

```
Camera Pipeline / Audio Pipeline / Driving Data
        |
        v
  Feature Extraction
        |
        v
  Embedding Generation (20480 dim)
        |
        v
  Vector Database Storage
```

### Vector Database -> LLM Node

```
Current Sensor Data
        |
        v
  Feature Extraction
        |
        v
  Query Embedding Generation
        |
        v
  Vector Database Similarity Search
        |
        v
  Retrieve Top-K Similar Historical Events
        |
        v
  Format Context for LLM
        |
        v
  LLM Inference
        |
        v
  Intervention Output
```

## Monitoring and Debugging

### Database Health Checks

```python
def check_database_health(db):
    """Verify database integrity"""
    print(f"Total vectors: {db.index.ntotal}")
    print(f"Metadata consistency: {len(db.metadata) == db.index.ntotal}")
    print(f"Index size: {os.path.getsize(db.index_path) / 1e6:.2f} MB")
    print(f"Metadata size: {os.path.getsize(db.metadata_path) / 1e6:.2f} MB")
    
    # Test retrieval
    test_embedding = db.normalize_vector(np.random.randn(20480))
    distances, indices = db.search(test_embedding, k=1)
    print(f"Search test passed: retrieved {len(indices)} results")

check_database_health(db)
```

### Query Analysis

```python
# Analyze query distribution
from collections import Counter

labels = [db.metadata[vid]['fatigue_label'] for vid in db.metadata]
label_counts = Counter(labels)
print(f"Database composition: {label_counts}")
# Output: Counter({'alert': 5000, 'drowsy': 3500, 'sleepy': 1500})
```

## Related Modules

- **LLM and Fatigue Handling:** Uses embeddings for in-context learning
- **Feature Extraction:** Generates embeddings for storage
- **Input Signal Processing:** Provides raw data for features

## Troubleshooting

**Issue:** Search returns poor quality results
- Solution: Verify embeddings are properly normalized
- Check that metadata matches stored vectors
- Consider rebuilding index

**Issue:** Out of memory errors
- Solution: Implement vector pruning for old entries
- Use disk-based index instead of in-memory
- Implement sharding across multiple databases

**Issue:** Metadata pickle corruption
- Solution: Maintain backup copies
- Use JSON alternative for critical metadata
- Implement transaction logging

## Future Enhancements

- GPU-accelerated FAISS operations with CUDA
- Distributed database for fleet-scale systems
- Automatic rebalancing and optimization
- Integration with cloud storage (S3, GCS)
- Real-time online learning with vector updates
