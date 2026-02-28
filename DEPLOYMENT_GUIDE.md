# Smart Grid AI - Deployment Guide

**Version**: 1.0.0  
**Date**: January 30, 2026  
**Target Audience**: DevOps, System Administrators, Operations Team  

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Installation & Configuration](#installation--configuration)
4. [Deployment Scenarios](#deployment-scenarios)
5. [Monitoring & Alerting](#monitoring--alerting)
6. [Scaling & Performance](#scaling--performance)
7. [Disaster Recovery](#disaster-recovery)
8. [Maintenance](#maintenance)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      End Users / Clients                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                    REST API Calls
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Load Balancer (Nginx/HAProxy)                  │
│              Port 80/443 (HTTP/HTTPS)                       │
└────────┬─────────────────────────────────────────┬──────────┘
         │                                         │
    Instance 1                                 Instance N
         │                                         │
┌────────▼────────────────────────────────────────▼──────────┐
│            FastAPI Server (Uvicorn)                        │
│              Workers: 4 (configurable)                     │
│              Port: 8000                                    │
└────────┬─────────────────────────────────────────┬──────────┘
         │                                         │
    Model Inference                          Logging/Metrics
         │                                         │
┌────────▼─────────────────────────────────────────▼──────────┐
│                 Shared Storage                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Models:                                              │  │
│  │ - baseline_day8_9.pkl (7.22 MB)                      │  │
│  │ - moe_day10_11.pkl (3.28 MB)                         │  │
│  │ - anomaly_day12_13.pkl (1.44 MB)                     │  │
│  │ Total: 12 MB                                         │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Logs:                                                │  │
│  │ - api_server.log (rotating, 100MB max)               │  │
│  │ - access.log (nginx)                                 │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
         │
    ┌────┴────┬──────────┬─────────────┐
    │          │          │             │
   DB    Metrics   Backups    Monitoring
```

### Component Specifications

| Component | Type | Count | Specs |
|-----------|------|-------|-------|
| API Server | FastAPI | 1+ | 1 CPU, 2GB RAM |
| Load Balancer | Nginx/HAProxy | 1 | 1 CPU, 1GB RAM |
| Database | PostgreSQL | 1 | 2 CPU, 4GB RAM |
| Cache | Redis | 1 | 1 CPU, 2GB RAM |
| Total | - | - | 5-10 CPU, 9-18GB RAM |

---

## Pre-Deployment Checklist

### 1. Requirements Validation

- [ ] Python 3.11+ installed
- [ ] 12 MB disk space for models
- [ ] 4GB RAM minimum available
- [ ] Port 8000 available (or custom configured)
- [ ] Outbound internet access for logs/metrics

### 2. Security Review

- [ ] SSL/TLS certificate obtained
- [ ] API authentication configured
- [ ] Rate limiting enabled
- [ ] CORS properly configured
- [ ] Firewall rules in place
- [ ] No sensitive data in logs

### 3. Performance Testing

- [ ] Single prediction latency < 100ms (P99)
- [ ] Batch processing 1000 samples < 2s
- [ ] Memory usage stable < 2GB
- [ ] CPU usage < 80% under peak load
- [ ] Zero error rate during 1000s requests

### 4. Documentation Review

- [ ] API documentation complete
- [ ] Deployment guide reviewed
- [ ] Runbooks created for common issues
- [ ] Alert thresholds defined
- [ ] Escalation procedures established

### 5. Team Readiness

- [ ] Operations team trained
- [ ] On-call rotation established
- [ ] Incident response procedures documented
- [ ] Communication channels set up
- [ ] Change management approved

---

## Installation & Configuration

### Production Installation Script

**`deploy.sh`**:

```bash
#!/bin/bash
set -e

echo "=== Smart Grid AI - Production Deployment ==="

# Configuration
DEPLOY_USER="smartgrid"
DEPLOY_DIR="/opt/smartgrid-ai"
PYTHON_VERSION="3.11"
PORT="8000"
WORKERS="4"

# 1. Create deployment user
echo "[1/7] Creating deployment user..."
sudo useradd -m -s /bin/bash $DEPLOY_USER || true

# 2. Create directory structure
echo "[2/7] Creating directory structure..."
sudo mkdir -p $DEPLOY_DIR/{models,logs,data,backups}
sudo chown -R $DEPLOY_USER:$DEPLOY_USER $DEPLOY_DIR

# 3. Install Python dependencies
echo "[3/7] Installing dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-pip \
    python${PYTHON_VERSION}-venv

# 4. Create virtual environment
echo "[4/7] Creating virtual environment..."
cd $DEPLOY_DIR
python${PYTHON_VERSION} -m venv venv
source venv/bin/activate

# 5. Install requirements
echo "[5/7] Installing Python packages..."
pip install --upgrade pip
pip install -r requirements_api.txt

# 6. Deploy code and models
echo "[6/7] Deploying application..."
cp inference_api.py $DEPLOY_DIR/
cp models/*.pkl $DEPLOY_DIR/models/
cp -r data $DEPLOY_DIR/

# 7. Setup systemd service
echo "[7/7] Setting up systemd service..."
sudo tee /etc/systemd/system/smartgrid-api.service > /dev/null << EOF
[Unit]
Description=Smart Grid AI API
After=network.target

[Service]
Type=notify
User=$DEPLOY_USER
WorkingDirectory=$DEPLOY_DIR
Environment="PATH=$DEPLOY_DIR/venv/bin"
ExecStart=$DEPLOY_DIR/venv/bin/uvicorn \\
    inference_api:app \\
    --host 0.0.0.0 \\
    --port $PORT \\
    --workers $WORKERS \\
    --log-level info
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable smartgrid-api
sudo systemctl start smartgrid-api

echo "✓ Deployment complete!"
echo "  API running at http://localhost:$PORT"
echo "  Check status: sudo systemctl status smartgrid-api"
echo "  View logs: journalctl -u smartgrid-api -f"
```

**Run deployment**:

```bash
chmod +x deploy.sh
./deploy.sh
```

### Configuration Files

**`/etc/nginx/sites-available/smartgrid-api`**:

```nginx
upstream smartgrid_backend {
    server localhost:8000 max_fails=3 fail_timeout=30s;
    server localhost:8001 max_fails=3 fail_timeout=30s;
    server localhost:8002 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.smartgrid.ai;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.smartgrid.ai;
    
    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/api.smartgrid.ai/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.smartgrid.ai/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Logging
    access_log /var/log/nginx/smartgrid-access.log combined;
    error_log /var/log/nginx/smartgrid-error.log;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=general:10m rate=100r/s;
    limit_req_zone $binary_remote_addr zone=batch:10m rate=10r/s;
    
    # API proxy
    location / {
        limit_req zone=general burst=200 nodelay;
        
        proxy_pass http://smartgrid_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 16 4k;
    }
    
    # Batch endpoint with different rate limit
    location /predict/batch {
        limit_req zone=batch burst=20 nodelay;
        
        proxy_pass http://smartgrid_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Higher timeout for batch
        proxy_read_timeout 300s;
    }
    
    # Health check endpoint (no rate limit)
    location /health {
        access_log off;
        proxy_pass http://smartgrid_backend;
    }
}
```

---

## Deployment Scenarios

### Scenario 1: Single Server (Development)

**Setup Time**: 15 minutes  
**Cost**: Lowest  
**Availability**: SLA not applicable  

```bash
# Simple deployment
python inference_api.py

# Or with uvicorn
uvicorn inference_api:app --host 0.0.0.0 --port 8000
```

### Scenario 2: High Availability (Production)

**Setup Time**: 1 hour  
**Cost**: Medium  
**Availability**: 99.9% SLA  

```yaml
# Docker Compose for HA
version: '3.8'
services:
  api-1:
    image: smartgrid-api:1.0.0
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
    restart: always
    
  api-2:
    image: smartgrid-api:1.0.0
    ports:
      - "8001:8000"
    environment:
      - LOG_LEVEL=INFO
    restart: always
    
  api-3:
    image: smartgrid-api:1.0.0
    ports:
      - "8002:8000"
    environment:
      - LOG_LEVEL=INFO
    restart: always
    
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api-1
      - api-2
      - api-3
    restart: always
```

### Scenario 3: Kubernetes (Enterprise)

**Setup Time**: 2-4 hours  
**Cost**: Highest  
**Availability**: 99.99% SLA  

```yaml
# Kubernetes Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smartgrid-api
  namespace: production
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
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
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        
        # Resource limits
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        
        # Health checks
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          failureThreshold: 3
        
        # Environment
        env:
        - name: LOG_LEVEL
          value: "INFO"
        
        # Logging
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      
      volumes:
      - name: logs
        persistentVolumeClaim:
          claimName: smartgrid-logs

---
apiVersion: v1
kind: Service
metadata:
  name: smartgrid-api-service
  namespace: production
spec:
  type: LoadBalancer
  selector:
    app: smartgrid-api
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
```

---

## Monitoring & Alerting

### Prometheus Metrics

**Add to `inference_api.py`**:

```python
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
request_count = Counter('api_requests_total', 'Total requests', ['endpoint', 'method'])
request_duration = Histogram('api_request_duration_seconds', 'Request duration')
prediction_errors = Histogram('prediction_error_absolute', 'Prediction error')

# In endpoints, record metrics
@app.post("/predict/single")
async def predict_single(request: PredictionRequest):
    with request_duration.time():
        # prediction logic
        pass
    request_count.labels(endpoint='/predict/single', method='POST').inc()
```

### AlertManager Configuration

**`alerting_rules.yml`**:

```yaml
groups:
- name: smartgrid_alerts
  interval: 30s
  rules:
  
  # High error rate
  - alert: HighErrorRate
    expr: rate(api_requests_total{status="500"}[5m]) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate > 5% for 5 minutes"
  
  # Slow requests
  - alert: SlowRequests
    expr: histogram_quantile(0.99, api_request_duration_seconds) > 1
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Slow requests detected"
      description: "P99 latency > 1 second"
  
  # High memory usage
  - alert: HighMemoryUsage
    expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage > 90%"
  
  # API down
  - alert: APIDown
    expr: up{job="smartgrid-api"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "API is down"
      description: "API has been down for > 1 minute"
```

### Dashboard Queries

**Grafana Dashboard**:

```sql
-- Requests per second
rate(api_requests_total[1m])

-- P99 latency
histogram_quantile(0.99, api_request_duration_seconds)

-- Error rate
rate(api_requests_total{status="500"}[5m])

-- Active connections
increase(api_requests_total[5m])

-- Prediction accuracy (if tracking)
avg(prediction_accuracy)
```

---

## Scaling & Performance

### Horizontal Scaling

**Recommended Setup**:
- Start: 2 instances (HA)
- Target Load: 1000 requests/second
- Scale Up Trigger: CPU > 70% or Latency P99 > 200ms
- Instance Size: 2 CPU, 4GB RAM

**Scaling Script**:

```python
# Auto-scaling logic
def calculate_required_instances(current_rps, target_rps_per_instance=500):
    return max(2, math.ceil(current_rps / target_rps_per_instance))

# Monitor and scale
while True:
    current_rps = get_requests_per_second()
    required_instances = calculate_required_instances(current_rps)
    current_instances = get_running_instances()
    
    if required_instances > current_instances:
        scale_up(required_instances - current_instances)
    elif required_instances < current_instances - 1:  # Hysteresis
        scale_down(current_instances - required_instances)
    
    time.sleep(60)  # Check every minute
```

### Performance Tuning

| Parameter | Default | Recommended | Impact |
|-----------|---------|-------------|--------|
| Workers | 4 | CPU cores * 2 | Throughput |
| Max Connections | 100 | 1000 | Concurrency |
| Batch Size | 1 | 100-1000 | Latency |
| Cache TTL | None | 3600s | Hit Rate |

---

## Disaster Recovery

### Backup Strategy

```bash
# Daily backup script
#!/bin/bash

BACKUP_DIR="/backups/smartgrid-ai"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup models
tar -czf $BACKUP_DIR/models_$TIMESTAMP.tar.gz models/

# Backup database
pg_dump smartgrid > $BACKUP_DIR/db_$TIMESTAMP.sql

# Backup logs (last 30 days)
find logs/ -mtime -30 -tar -czf $BACKUP_DIR/logs_$TIMESTAMP.tar.gz

# Upload to S3
aws s3 cp $BACKUP_DIR s3://smartgrid-backups/ --recursive

# Keep only last 90 days
find $BACKUP_DIR -mtime +90 -delete

echo "Backup completed: $TIMESTAMP"
```

### Recovery Procedures

**Recovery Time Objective (RTO)**: 1 hour  
**Recovery Point Objective (RPO)**: 1 day  

**Restore Steps**:

```bash
# 1. Stop API
sudo systemctl stop smartgrid-api

# 2. Restore models
tar -xzf $BACKUP_DIR/models_$TIMESTAMP.tar.gz

# 3. Restore database
psql smartgrid < $BACKUP_DIR/db_$TIMESTAMP.sql

# 4. Verify integrity
python -c "
import pickle
with open('models/moe_day10_11.pkl', 'rb') as f:
    pickle.load(f)
print('✓ Models restored successfully')
"

# 5. Start API
sudo systemctl start smartgrid-api

# 6. Verify
curl http://localhost:8000/health
```

---

## Maintenance

### Weekly Tasks

- [ ] Review error logs
- [ ] Check disk space usage
- [ ] Verify backup completion
- [ ] Monitor performance metrics
- [ ] Test health check endpoint

### Monthly Tasks

- [ ] Review access logs for anomalies
- [ ] Update dependencies
- [ ] Performance optimization review
- [ ] Security audit
- [ ] Disaster recovery drill

### Quarterly Tasks

- [ ] Retrain models with latest data
- [ ] Update documentation
- [ ] Capacity planning review
- [ ] Security assessment
- [ ] Cost optimization review

---

## Support & Escalation

**Tier 1 (Response: 1 hour)**:
- Non-critical issues
- Feature requests
- Documentation

**Tier 2 (Response: 15 minutes)**:
- Performance degradation
- Partial service outage
- Data inconsistencies

**Tier 3 (Response: 5 minutes)**:
- Complete service outage
- Data loss
- Security incidents

**Escalation Contact**: ops-team@smartgrid.ai

---

**Last Updated**: January 30, 2026  
**Maintained By**: Smart Grid AI Operations Team
