
# DEPLOYMENT GUIDE - Smart Grid AI Inference Service

## Option A: Local Deployment (Development)

### 1. Install dependencies
```bash
pip install fastapi uvicorn pandas numpy scikit-learn joblib
```

### 2. Run API server
```bash
cd outputs/8_code_models
python -m uvicorn fastapi_service:app --reload --port 8000
```

### 3. Test API
```bash
# Health check
curl http://localhost:8000/health

# Get forecast (24 hours)
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "horizon": 24,
    "include_intervals": true,
    "scenario": "base"
  }'

# Get anomalies
curl http://localhost:8000/anomalies \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Option B: Docker Deployment (Production)

### 1. Build Docker image
```bash
cd outputs/8_code_models
docker build -t smartgrid-ai:latest .
```

### 2. Run container
```bash
docker run -d \
  --name smartgrid-api \
  -p 8000:8000 \
  -e API_KEY="your_secret_key" \
  smartgrid-ai:latest
```

### 3. Test container
```bash
curl http://localhost:8000/health
```

## Option C: Cloud Deployment (AWS/GCP/Azure)

### AWS Lambda + API Gateway
1. Create Lambda function from Docker image
2. Attach API Gateway trigger
3. Set environment variables (API keys)
4. Deploy

### Google Cloud Run
```bash
gcloud run deploy smartgrid-ai \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure Container Instances
```bash
az container create \
  --resource-group myResourceGroup \
  --name smartgrid-api \
  --image smartgrid-ai:latest \
  --port 8000
```

## API Authentication

### Setup API Keys
```python
# Generate secure key
import secrets
api_key = secrets.token_urlsafe(32)
print(f"API Key: {api_key}")
```

### Use in requests
```bash
curl -H "Authorization: Bearer <your_api_key>" \
  http://api.example.com/forecast
```

## Monitoring & Logging

### Setup CloudWatch (AWS)
```python
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
```

### Setup Application Insights (Azure)
```python
from azure.monitor.opentelemetry import configure_azure_monitor
configure_azure_monitor()
```

## Performance Optimization

- Response time target: <200ms ✓
- Throughput: 168,000 req/day = 1.9 req/sec ✓
- Concurrent users: 10-50 (adjust based on load)
- Caching: 1 hour for forecasts
- Auto-scaling: Enable in cloud platform

## Health Checks

Monitor these metrics:
- API availability: >99.9%
- Response time: <200ms (p99)
- Error rate: <0.1%
- Model accuracy: MAPE <5%

---
For questions: support@smartgrid-ai.com
