# Smart Grid AI - FastAPI Inference Server Documentation

**Version**: 1.0.0  
**Date**: January 30, 2026  
**Status**: Production Ready  

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [API Endpoints](#api-endpoints)
3. [Request/Response Examples](#request-response-examples)
4. [Error Handling](#error-handling)
5. [Performance](#performance)
6. [Deployment](#deployment)

---

## Quick Start

### Installation

```bash
# Install API dependencies
pip install -r requirements_api.txt

# Or individual packages
pip install fastapi uvicorn pydantic torch scikit-learn
```

### Running the Server

```bash
# Development mode with auto-reload
python inference_api.py

# Production mode with uvicorn
uvicorn inference_api:app --host 0.0.0.0 --port 8000 --workers 4

# With custom configuration
uvicorn inference_api:app --host localhost --port 5000 --log-level warning
```

### Health Check

```bash
# Verify server is running
curl http://localhost:8000/health

# Expected response
{
  "status": "healthy",
  "timestamp": "2026-01-30T12:45:00.123456",
  "uptime_seconds": 45.67,
  "models_loaded": {
    "moe": true,
    "baseline": true,
    "anomaly": true
  },
  "version": "1.0.0"
}
```

---

## API Endpoints

### 1. Health Check

**Endpoint**: `GET /health`

**Purpose**: Monitor server status and model availability

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-30T12:45:00.123456",
  "uptime_seconds": 3600.5,
  "models_loaded": {
    "moe": true,
    "baseline": true,
    "anomaly": true
  },
  "version": "1.0.0"
}
```

**Status Codes**:
- `200`: Server healthy, models loaded
- `503`: Server unavailable or models not loaded

**Use Case**: 
- Kubernetes liveness/readiness probes
- Load balancer health monitoring
- CI/CD deployment verification

---

### 2. Single Prediction

**Endpoint**: `POST /predict/single`

**Purpose**: Predict energy consumption for a single sample

**Request Body**:
```json
{
  "features": [
    1000.0, 2000.0, 1500.0, 3000.0, 500.0,
    100.0, 50.0, 75.0, 25.0, 10.0,
    20.0, 65.0, 5.0, 0.5, 50.0,
    1200.0, 110.0, 10500.0, 100.0, 50.0,
    30.0, 70.0, 2.0, 0.3, 100.0,
    200.0, 0.5, 0.3, 1000.0, 50.0,
    5000.0
  ]
}
```

**Response**:
```json
{
  "prediction": 5234.56,
  "model": "moe",
  "confidence": 0.95,
  "timestamp": "2026-01-30T12:45:00.123456",
  "input_features_count": 31
}
```

**Request Parameters**:
- `features`: Array of 31 floats (required)
  - Must be between -∞ and +∞
  - No NaN or Inf values allowed
  - Order matters: follows training data feature order

**Response Fields**:
- `prediction`: Energy consumption in kW
- `model`: Model used ("moe" by default)
- `confidence`: Confidence score (0-1)
- `timestamp`: Response time
- `input_features_count`: Number of features received

**Status Codes**:
- `200`: Success
- `422`: Validation error (wrong number of features, invalid types)
- `500`: Server error

**Example Usage**:

```python
import requests

url = "http://localhost:8000/predict/single"
features = [1000.0] * 31  # 31 features

response = requests.post(url, json={"features": features})
result = response.json()

print(f"Predicted consumption: {result['prediction']:.2f} kW")
print(f"Confidence: {result['confidence']:.1%}")
```

```bash
# Using curl
curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -d '{"features": [1000.0, 2000.0, ...]}'
```

---

### 3. Batch Predictions

**Endpoint**: `POST /predict/batch`

**Purpose**: Predict energy consumption for multiple samples (up to 1000)

**Request Body**:
```json
{
  "features": [
    [1000.0, 2000.0, ..., 5000.0],  // Sample 1 (31 features)
    [1100.0, 2100.0, ..., 5100.0],  // Sample 2 (31 features)
    [1200.0, 2200.0, ..., 5200.0]   // Sample 3 (31 features)
  ],
  "model": "moe"
}
```

**Response**:
```json
{
  "predictions": [5234.56, 5334.67, 5434.78],
  "model": "moe",
  "batch_size": 3,
  "mean_prediction": 5334.67,
  "std_prediction": 100.11,
  "timestamp": "2026-01-30T12:45:00.123456",
  "processing_time_ms": 125.34
}
```

**Request Parameters**:
- `features`: Array of feature arrays (1-1000 samples, each with 31 features)
- `model`: Model to use ("moe", "baseline", or "anomaly") - default: "moe"

**Response Fields**:
- `predictions`: Array of predicted consumptions (kW)
- `model`: Model used
- `batch_size`: Number of samples processed
- `mean_prediction`: Average prediction
- `std_prediction`: Standard deviation of predictions
- `timestamp`: Response time
- `processing_time_ms`: Total processing time in milliseconds

**Batch Size Recommendations**:
- Small batches (1-100): ~10-50ms per sample
- Medium batches (100-500): ~5-20ms per sample
- Large batches (500-1000): ~2-10ms per sample

**Example Usage**:

```python
import requests
import numpy as np

url = "http://localhost:8000/predict/batch"

# Generate 100 samples with 31 features each
features = np.random.randn(100, 31).tolist()

response = requests.post(url, json={
    "features": features,
    "model": "moe"
})

result = response.json()
print(f"Processed {result['batch_size']} samples")
print(f"Mean prediction: {result['mean_prediction']:.2f} kW")
print(f"Processing time: {result['processing_time_ms']:.2f}ms")
```

```bash
# Using curl with file
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d @batch_request.json
```

---

### 4. Anomaly Detection

**Endpoint**: `POST /anomaly-detect`

**Purpose**: Detect anomalies in energy consumption patterns

**Request Body**:
```json
{
  "features": [
    [1000.0, 2000.0, ..., 5000.0],
    [1100.0, 2100.0, ..., 5100.0],
    [85000.0, 90000.0, ..., 92000.0]  // Anomaly (very high values)
  ],
  "voting_threshold": 2
}
```

**Response**:
```json
{
  "total_samples": 3,
  "anomalies_detected": 1,
  "anomaly_indices": [2],
  "anomaly_scores": [0.87],
  "percentage_anomaly": 33.33,
  "timestamp": "2026-01-30T12:45:00.123456"
}
```

**Request Parameters**:
- `features`: Array of feature arrays (1-1000 samples)
- `voting_threshold`: Minimum number of models voting anomaly (1-3)
  - 1: High sensitivity, more false positives
  - 2: Balanced (recommended)
  - 3: High precision, fewer false positives

**Response Fields**:
- `total_samples`: Number of samples analyzed
- `anomalies_detected`: Number of anomalies found
- `anomaly_indices`: Zero-indexed positions of anomalies
- `anomaly_scores`: Confidence scores (0-1) for each anomaly
- `percentage_anomaly`: Percentage of samples flagged as anomalies
- `timestamp`: Response time

**Ensemble Details**:
- Uses 3 models: IsolationForest, OneClassSVM, Autoencoder
- Voting mechanism for high confidence
- Each model contributes 1 vote for anomalies
- Result: Majority vote (≥threshold) = anomaly

**Example Usage**:

```python
import requests
import numpy as np

url = "http://localhost:8000/anomaly-detect"

# Generate normal and anomalous samples
normal_features = np.random.randn(50, 31).tolist()
anomaly_features = (np.random.randn(5, 31) + 10).tolist()  # Shifted distribution
features = normal_features + anomaly_features

response = requests.post(url, json={
    "features": features,
    "voting_threshold": 2
})

result = response.json()
print(f"Anomalies: {result['anomalies_detected']}/{result['total_samples']}")
print(f"Indices: {result['anomaly_indices']}")
print(f"Scores: {result['anomaly_scores']}")
```

---

### 5. List Models

**Endpoint**: `GET /models`

**Purpose**: List available models and their status

**Response**:
```json
{
  "available_models": ["moe", "baseline", "anomaly"],
  "models_loaded": {
    "moe": true,
    "baseline": true,
    "anomaly": true
  },
  "total_models": 3,
  "timestamp": "2026-01-30T12:45:00.123456"
}
```

---

### 6. Model Information

**Endpoint**: `GET /models/{model_id}`

**Path Parameters**:
- `model_id`: Model identifier ("moe", "baseline", or "anomaly")

**Response**:
```json
{
  "model_id": "moe",
  "name": "Mixture of Experts Ensemble",
  "type": "Neural Ensemble",
  "mape": 0.0031,
  "r2": 0.9818,
  "params": 450000,
  "date": "2026-01-30",
  "is_loaded": true,
  "timestamp": "2026-01-30T12:45:00.123456"
}
```

**Available Models**:

1. **MoE (Mixture of Experts)**
   - Type: Neural Ensemble
   - Performance: 0.31% MAPE, 0.9818 R²
   - Experts: GRU, CNN-LSTM, Transformer, Attention
   - Best for: Maximum accuracy

2. **Baseline (SimpleEnsemble)**
   - Type: ML Ensemble
   - Performance: 17.05% MAPE, 0.9662 R²
   - Models: RandomForest, ExtraTrees, Ridge
   - Best for: Interpretability, legacy compatibility

3. **Anomaly**
   - Type: Hybrid Ensemble
   - Detection Rate: 99.95%
   - Models: IsolationForest, OneClassSVM, Autoencoder
   - Best for: Anomaly detection only

---

## Request/Response Examples

### Example 1: Single Prediction with Python

```python
import requests
import json

# API Configuration
API_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

# Prepare features (31 values)
features = [
    1000.0,  # consumption_industrial
    2000.0,  # consumption_commercial
    1500.0,  # consumption_residential
    500.0,   # generation_solar
    100.0,   # generation_wind
    # ... (26 more features)
    5000.0   # grid_load
]

# Make prediction
response = requests.post(
    f"{API_URL}/predict/single",
    json={"features": features},
    headers=HEADERS
)

if response.status_code == 200:
    result = response.json()
    print(f"Prediction: {result['prediction']:.2f} kW")
    print(f"Model: {result['model']}")
    print(f"Confidence: {result['confidence']:.1%}")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

### Example 2: Batch Processing with Pandas

```python
import requests
import pandas as pd

# Load data
df = pd.read_csv('energy_data.csv')
features = df.iloc[:100, :-1].values.tolist()  # First 100 samples

# Send batch request
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"features": features, "model": "moe"}
)

results = response.json()
predictions = results['predictions']

# Add predictions to dataframe
df['predicted_consumption'] = predictions[:len(df)]
print(df.head())
```

### Example 3: Anomaly Detection Pipeline

```python
import requests
import numpy as np

def detect_grid_anomalies(data, threshold=2):
    """Detect anomalies in grid consumption data"""
    
    response = requests.post(
        "http://localhost:8000/anomaly-detect",
        json={
            "features": data.tolist(),
            "voting_threshold": threshold
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Found {result['anomalies_detected']} anomalies")
        print(f"Indices: {result['anomaly_indices']}")
        return result
    else:
        raise Exception(f"API Error: {response.text}")

# Usage
data = np.random.randn(1000, 31)
anomalies = detect_grid_anomalies(data)
```

### Example 4: Health Monitoring

```python
import requests
import time

def monitor_api_health(interval=30):
    """Monitor API health at regular intervals"""
    
    while True:
        try:
            response = requests.get("http://localhost:8000/health")
            health = response.json()
            
            print(f"Status: {health['status']}")
            print(f"Uptime: {health['uptime_seconds']:.1f}s")
            print(f"Models loaded: {health['models_loaded']}")
            
            # Alert if models not loaded
            if not all(health['models_loaded'].values()):
                print("⚠️  WARNING: Some models not loaded!")
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
        
        time.sleep(interval)

monitor_api_health()
```

---

## Error Handling

### Common Error Responses

#### 1. Validation Error (422)

**Cause**: Invalid input format or constraints violated

```json
{
  "detail": [
    {
      "loc": ["body", "features"],
      "msg": "ensure this value has at least 31 items",
      "type": "value_error.list.min_items"
    }
  ]
}
```

**Solutions**:
- Verify feature count is exactly 31
- Check all values are numeric (no strings)
- Validate no NaN or Inf values

#### 2. Model Not Found (404)

**Cause**: Invalid model ID requested

```json
{
  "detail": "Model 'gpt' not found"
}
```

**Solutions**:
- Use valid model IDs: "moe", "baseline", "anomaly"
- Check available models: `GET /models`

#### 3. Server Error (500)

**Cause**: Internal server error during processing

```json
{
  "error": "Internal server error",
  "detail": "Model prediction failed",
  "status_code": 500,
  "timestamp": "2026-01-30T12:45:00.123456"
}
```

**Solutions**:
- Check server logs
- Verify models are loaded: `GET /health`
- Retry the request
- Contact support if issue persists

### Error Codes Summary

| Code | Status | Meaning |
|------|--------|---------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid request format |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation error |
| 500 | Server Error | Internal server error |
| 503 | Service Unavailable | Server unavailable |

---

## Performance

### Latency Benchmarks

**Single Prediction**:
- Model: MoE
- Latency: ~50-100ms (99th percentile: <150ms)
- Throughput: ~100-200 predictions/second

**Batch Prediction (100 samples)**:
- Model: MoE
- Latency: ~100-200ms
- Throughput: ~500-1000 predictions/second

**Batch Prediction (1000 samples)**:
- Model: MoE
- Latency: ~500-1000ms
- Throughput: ~1000-2000 predictions/second

**Anomaly Detection (100 samples)**:
- Ensemble: IsoForest + SVM + Autoencoder
- Latency: ~100-300ms
- Throughput: ~300-1000 samples/second

### Optimization Tips

1. **Batch Processing**: Use larger batches when possible
   - 100 samples vs 1 sample = 50-100x throughput improvement

2. **Connection Pooling**: Reuse HTTP connections
   ```python
   session = requests.Session()
   for _ in range(100):
       session.post(url, json=data)
   ```

3. **Async Processing**: Use async/await for concurrent requests
   ```python
   import asyncio
   tasks = [async_predict(features) for features in all_features]
   results = await asyncio.gather(*tasks)
   ```

4. **Caching**: Cache predictions for identical inputs
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def cached_predict(features_tuple):
       return predict(list(features_tuple))
   ```

---

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements_api.txt .
RUN pip install -r requirements_api.txt

# Copy application
COPY inference_api.py .
COPY models/ ./models/
COPY data/ ./data/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smartgrid-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: smartgrid-api
  template:
    metadata:
      labels:
        app: smartgrid-api
    spec:
      containers:
      - name: api
        image: smartgrid-ai:1.0.0
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: smartgrid-api-service
spec:
  selector:
    app: smartgrid-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Production Checklist

- [ ] Enable SSL/TLS encryption
- [ ] Configure rate limiting
- [ ] Set up monitoring and alerts
- [ ] Configure logging aggregation
- [ ] Set up auto-scaling policies
- [ ] Enable CORS for web clients
- [ ] Configure request timeouts
- [ ] Set up backup/disaster recovery

---

## Support & Contact

**Documentation**: `/docs` (Swagger UI)  
**Alternative Docs**: `/redoc` (ReDoc)  
**Issues**: Open GitHub issue with:
  - API version
  - Request/response examples
  - Error message and stack trace
  - System information

---

**Last Updated**: January 30, 2026  
**Maintained By**: Smart Grid AI Team
