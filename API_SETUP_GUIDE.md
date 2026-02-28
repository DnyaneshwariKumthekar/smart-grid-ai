# Smart Grid AI - FastAPI Inference Server Setup Guide

**Version**: 1.0.0  
**Date**: January 30, 2026  
**Status**: Production Ready  

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Starting the Server](#starting-the-server)
4. [Verifying the Setup](#verifying-the-setup)
5. [Quick Start Examples](#quick-start-examples)
6. [Troubleshooting](#troubleshooting)
7. [Production Deployment](#production-deployment)

---

## System Requirements

### Minimum Hardware

- **CPU**: 2 cores
- **RAM**: 4 GB
- **Disk**: 2 GB (for models)
- **Network**: Ethernet or WiFi

### Recommended Hardware

- **CPU**: 4+ cores
- **RAM**: 8 GB
- **Disk**: 4 GB SSD
- **Network**: 1 Gbps connection

### Software Requirements

- **Python**: 3.11 or 3.13+
- **OS**: Windows 10+, Ubuntu 20.04+, macOS 10.15+
- **Port**: 8000 (configurable)

### Network Requirements

- Outbound HTTP/HTTPS (for dependencies)
- Inbound on port 8000 (for API)
- Optional: TLS/SSL certificates (for HTTPS)

---

## Installation

### Step 1: Install Python Dependencies

```bash
# Navigate to project directory
cd "C:\Users\Dnyaneshwari\Desktop\new projects\smart-grid-ai"

# Install API requirements
pip install -r requirements_api.txt

# Verify installation
python -c "import fastapi; import torch; print('✓ Dependencies installed')"
```

### Step 2: Verify Model Files

```bash
# Check if all models are present
ls -la models/

# Expected files:
# - baseline_day8_9.pkl (7.22 MB)
# - moe_day10_11.pkl (3.28 MB)
# - anomaly_day12_13.pkl (1.44 MB)
```

### Step 3: Create Logs Directory

```bash
# Create logs directory for API server
mkdir logs

# Verify
ls -la logs/
```

### Step 4: Configure Environment (Optional)

```bash
# Create .env file for configuration
cat > .env << EOF
API_PORT=8000
API_HOST=0.0.0.0
LOG_LEVEL=INFO
WORKERS=4
EOF
```

---

## Starting the Server

### Option 1: Development Mode (with auto-reload)

**Best for**: Development and testing

```bash
python inference_api.py
```

**Expected Output**:
```
Starting Smart Grid AI API Server...
✓ All models loaded successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### Option 2: Production Mode (recommended)

**Best for**: Production deployments

```bash
# Single worker (simpler)
uvicorn inference_api:app --host 0.0.0.0 --port 8000

# Multiple workers (better throughput)
uvicorn inference_api:app --host 0.0.0.0 --port 8000 --workers 4

# With custom logging
uvicorn inference_api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info \
  --access-log
```

### Option 3: Using Windows Batch Script

**Create `start_api.bat`**:

```batch
@echo off
echo Starting Smart Grid AI API Server...
cd /d "%~dp0"

REM Activate virtual environment if using venv
REM call venv\Scripts\activate.bat

REM Start the API server
python inference_api.py

pause
```

**Run the script**:
```batch
start_api.bat
```

### Option 4: Background Process (Windows PowerShell)

```powershell
# Start in background
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "inference_api.py"

# Verify it's running
Invoke-WebRequest -Uri "http://localhost:8000/health"
```

### Option 5: Docker Container

```bash
# Build image
docker build -t smartgrid-api:1.0.0 .

# Run container
docker run -p 8000:8000 \
  -v ${PWD}/models:/app/models \
  -v ${PWD}/logs:/app/logs \
  smartgrid-api:1.0.0

# Run with custom port
docker run -p 5000:8000 smartgrid-api:1.0.0
```

---

## Verifying the Setup

### Quick Health Check

```bash
# Using curl
curl http://localhost:8000/health

# Using PowerShell
Invoke-WebRequest -Uri "http://localhost:8000/health" | ConvertTo-Json
```

**Expected Response**:
```json
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

### Run Test Suite

```bash
# Run all tests
python test_api.py

# Expected output
=====================================
Smart Grid AI - API Test Suite
=====================================
Health Check                       PASS
Single Prediction                  PASS
Batch Prediction                   PASS
Anomaly Detection                  PASS
List Models                         PASS
Model Info                          PASS
Error Handling                      PASS
Performance                         PASS
=====================================
Results: 8/8 tests passed
✓ All tests passed!
```

### Access Interactive Documentation

Open a web browser and navigate to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Check API Logs

```bash
# View recent logs
tail -f logs/api_server.log

# Count requests
grep "POST" logs/api_server.log | wc -l
```

---

## Quick Start Examples

### Python - Single Prediction

**Save as `example_single.py`**:

```python
import requests

# Single prediction
features = [1000.0] * 31  # 31 features

response = requests.post(
    "http://localhost:8000/predict/single",
    json={"features": features}
)

result = response.json()
print(f"Predicted consumption: {result['prediction']:.2f} kW")
print(f"Confidence: {result['confidence']:.1%}")
```

**Run**:
```bash
python example_single.py
```

### Python - Batch Prediction

**Save as `example_batch.py`**:

```python
import requests
import numpy as np

# Generate 100 samples
features = np.random.randn(100, 31).tolist()

response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"features": features, "model": "moe"}
)

result = response.json()
print(f"Processed: {result['batch_size']} samples")
print(f"Mean prediction: {result['mean_prediction']:.2f} kW")
print(f"Processing time: {result['processing_time_ms']:.2f}ms")
```

**Run**:
```bash
python example_batch.py
```

### cURL - Health Check

```bash
curl -X GET http://localhost:8000/health
```

### cURL - Single Prediction

```bash
curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -d '{
    "features": [1000.0, 2000.0, 1500.0, 3000.0, 500.0, 100.0, 50.0, 75.0, 25.0, 10.0, 20.0, 65.0, 5.0, 0.5, 50.0, 1200.0, 110.0, 10500.0, 100.0, 50.0, 30.0, 70.0, 2.0, 0.3, 100.0, 200.0, 0.5, 0.3, 1000.0, 50.0, 5000.0]
  }'
```

### PowerShell - Health Check

```powershell
$response = Invoke-WebRequest -Uri "http://localhost:8000/health"
$response.Content | ConvertFrom-Json | Format-Table
```

### Batch File for Windows Task Scheduler

**Create `schedule_api.xml`**:

```xml
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Date>2026-01-30T12:45:00</Date>
    <Author>SmartGridAI</Author>
    <Description>Start Smart Grid AI API Server</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>S-1-5-21-3623811015-3361044348-30300820-1013</UserId>
      <LogonType>ServiceAccount</LogonType>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>true</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>true</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>false</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>true</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <DisallowStartOnRemoteAppSession>false</DisallowStartOnRemoteAppSession>
    <UseUnifiedSchedulingEngine>true</UseUnifiedSchedulingEngine>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <DeleteExpiredTaskAfter>PT0S</DeleteExpiredTaskAfter>
    <VisibleOnlyWhenLoggedOn>false</VisibleOnlyWhenLoggedOn>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>python.exe</Command>
      <Arguments>inference_api.py</Arguments>
      <WorkingDirectory>C:\Users\Dnyaneshwari\Desktop\new projects\smart-grid-ai</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
```

---

## Troubleshooting

### Issue: Port 8000 Already in Use

**Solution 1**: Use a different port

```bash
uvicorn inference_api:app --port 5000
```

**Solution 2**: Kill process using port 8000

```bash
# Windows PowerShell
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process

# Linux/Mac
sudo lsof -ti:8000 | xargs kill -9
```

### Issue: Models Not Loading

**Symptoms**: `models_loaded: {moe: false, baseline: false, anomaly: false}`

**Check**:
1. Verify model files exist:
   ```bash
   ls -la models/
   ```

2. Check file permissions:
   ```bash
   chmod 644 models/*.pkl
   ```

3. Verify pickle files are not corrupted:
   ```python
   import pickle
   with open('models/moe_day10_11.pkl', 'rb') as f:
       data = pickle.load(f)
       print(f"✓ Model loaded successfully")
   ```

### Issue: Out of Memory

**Symptoms**: Process killed or memory error

**Solutions**:
1. Reduce batch size in requests
2. Enable memory monitoring:
   ```bash
   python -m memory_profiler inference_api.py
   ```
3. Use multiple workers instead of increasing batch size

### Issue: Slow Predictions

**Symptoms**: P99 latency > 500ms

**Check**:
1. CPU usage:
   ```bash
   # Windows
   Get-Process python | Select ProcessName, CPU, Memory
   
   # Linux
   top -p $(pgrep -f inference_api)
   ```

2. Network latency:
   ```bash
   ping -t localhost
   ```

3. Model inference time:
   - Enable profiling in inference_api.py
   - Check if GPU is available

### Issue: Connection Refused

**Solution**: Verify server is running

```bash
# Check if server is listening
netstat -ano | findstr :8000

# Restart server
python inference_api.py
```

### Issue: CORS Errors in Browser

**Solution**: CORS is already configured in the API, but verify:

```python
# In inference_api.py - should be enabled
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Production Deployment

### Pre-Deployment Checklist

- [ ] All tests pass (`python test_api.py`)
- [ ] Models are loaded (`GET /health`)
- [ ] SSL/TLS certificate obtained
- [ ] Environment variables configured
- [ ] Logging configured
- [ ] Backups in place
- [ ] Monitoring setup complete
- [ ] Load testing successful

### AWS EC2 Deployment

```bash
# Connect to instance
ssh -i key.pem ubuntu@ec2-instance.amazonaws.com

# Install dependencies
sudo apt-get update
sudo apt-get install python3-pip
pip install -r requirements_api.txt

# Start server with systemd
sudo nano /etc/systemd/system/smartgrid-api.service

# Add:
[Unit]
Description=Smart Grid AI API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/smart-grid-ai
ExecStart=/usr/bin/python3 /home/ubuntu/smart-grid-ai/inference_api.py
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable smartgrid-api
sudo systemctl start smartgrid-api
sudo systemctl status smartgrid-api
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name api.smartgrid.ai;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### SSL/TLS Configuration

```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

# Run with SSL
uvicorn inference_api:app \
  --host 0.0.0.0 \
  --port 443 \
  --ssl-keyfile=key.pem \
  --ssl-certfile=cert.pem \
  --workers 4
```

---

## Support

**Documentation**: See [API_DOCUMENTATION.md](API_DOCUMENTATION.md)  
**Test Suite**: Run `python test_api.py`  
**Interactive Docs**: Visit http://localhost:8000/docs after starting server  

---

**Last Updated**: January 30, 2026  
**Maintained By**: Smart Grid AI Team
