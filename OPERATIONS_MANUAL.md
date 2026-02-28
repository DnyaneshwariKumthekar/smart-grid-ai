# Smart Grid AI - Operations Manual

**Version**: 1.0.0  
**Date**: January 30, 2026  
**Audience**: Operations Team, System Administrators, DevOps  

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [System Architecture](#system-architecture)
3. [Daily Operations](#daily-operations)
4. [Monitoring & Health Checks](#monitoring--health-checks)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Maintenance Procedures](#maintenance-procedures)
7. [Performance Tuning](#performance-tuning)
8. [Disaster Recovery](#disaster-recovery)
9. [Contact & Escalation](#contact--escalation)

---

## Quick Reference

### Critical Commands

```bash
# Status Check
sudo systemctl status smartgrid-api

# View Recent Logs
sudo journalctl -u smartgrid-api -n 50 -f

# Restart Service
sudo systemctl restart smartgrid-api

# View Active Connections
netstat -tlnp | grep 8000

# CPU/Memory Usage
top -p $(pgrep -f uvicorn | head -1)

# Disk Space
df -h | grep /opt

# Check API Health
curl -s http://localhost:8000/health | jq .
```

### Service Health Quick Check

```bash
#!/bin/bash
# Run every 5 minutes

API_ENDPOINT="http://localhost:8000"

# Check API availability
if ! curl -s "${API_ENDPOINT}/health" | jq -e '.status=="ok"' > /dev/null; then
    echo "❌ API DOWN - CRITICAL"
    systemctl restart smartgrid-api
    sleep 10
fi

# Check error rate (from logs)
ERROR_RATE=$(journalctl -u smartgrid-api --since "5 min ago" | grep ERROR | wc -l)
if [ $ERROR_RATE -gt 10 ]; then
    echo "⚠️  HIGH ERROR RATE: $ERROR_RATE in last 5 minutes"
fi

# Check memory
MEM_PERCENT=$(top -bn1 -p $(pgrep -f uvicorn | head -1) | tail -1 | awk '{print $10}')
if (( $(echo "$MEM_PERCENT > 80" | bc -l) )); then
    echo "⚠️  HIGH MEMORY: ${MEM_PERCENT}%"
fi

# Check CPU
CPU_PERCENT=$(top -bn1 -p $(pgrep -f uvicorn | head -1) | tail -1 | awk '{print $9}')
if (( $(echo "$CPU_PERCENT > 80" | bc -l) )); then
    echo "⚠️  HIGH CPU: ${CPU_PERCENT}%"
fi

echo "✅ All systems nominal"
```

### Dashboard URLs

| Service | URL | Login |
|---------|-----|-------|
| Grafana | https://monitoring.smartgrid.ai | ops@smartgrid.ai |
| Prometheus | https://metrics.smartgrid.ai:9090 | N/A (internal) |
| Kibana | https://logs.smartgrid.ai | ops@smartgrid.ai |
| API Docs | https://api.smartgrid.ai/docs | API Key required |

---

## System Architecture

### Production Deployment

```
Load Balancer (Nginx)
    Port 80/443
         │
    ┌────┴────┬─────────┬──────────┐
    │         │         │          │
    ▼         ▼         ▼          ▼
API-1      API-2      API-3    API-N
Port 8000  Port 8001  Port 8002
    │         │         │          │
    └────┬────┴─────────┴──────────┘
         │
    Shared Storage
    ├─ Models (3.28 MB)
    ├─ Logs (rotating)
    └─ Config Files
         │
    ┌────┴────┬──────────┬─────────┐
    │         │          │         │
    ▼         ▼          ▼         ▼
  PostgreSQL  Redis   Prometheus  Grafana
  (Database) (Cache)  (Metrics)  (Dashboard)
```

### Service Ports

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| Nginx | 80 | HTTP | Public API (redirect to HTTPS) |
| Nginx | 443 | HTTPS | Public API (encrypted) |
| API | 8000-8099 | HTTP | Internal (load balanced) |
| Prometheus | 9090 | HTTP | Metrics (internal only) |
| Grafana | 3000 | HTTP | Dashboard (internal only) |
| PostgreSQL | 5432 | TCP | Database (internal only) |
| Redis | 6379 | TCP | Cache (internal only) |

### Firewall Rules

**Ingress (Allowed)**:
- Port 80 from anywhere (HTTP redirect)
- Port 443 from anywhere (HTTPS)
- Port 22 from admin VPN only (SSH)

**Egress (Allowed)**:
- Port 53 (DNS)
- Port 123 (NTP)
- Port 443 (HTTPS outbound)

---

## Daily Operations

### Morning Startup Checklist (09:00 AM)

```bash
#!/bin/bash
# Daily startup verification

echo "=== Smart Grid AI - Morning Startup Check ==="
echo ""

# 1. System Status
echo "1. System Status"
if systemctl is-system-running > /dev/null 2>&1; then
    echo "   ✓ System running"
else
    echo "   ✗ System not ready"
    exit 1
fi
echo ""

# 2. Database Status
echo "2. Database Status"
if pg_isready -h localhost -p 5432; then
    echo "   ✓ PostgreSQL ready"
else
    echo "   ✗ PostgreSQL not responding"
    sudo systemctl start postgresql
fi
echo ""

# 3. Redis Status
echo "3. Cache Status"
if redis-cli ping > /dev/null 2>&1; then
    echo "   ✓ Redis responding"
else
    echo "   ✗ Redis not responding"
    sudo systemctl start redis-server
fi
echo ""

# 4. API Service Status
echo "4. API Service Status"
if sudo systemctl is-active --quiet smartgrid-api; then
    echo "   ✓ API running"
else
    echo "   ✗ API not running, starting..."
    sudo systemctl start smartgrid-api
    sleep 5
fi
echo ""

# 5. API Health Check
echo "5. API Health Check"
HEALTH=$(curl -s http://localhost:8000/health)
if echo $HEALTH | jq -e '.status=="ok"' > /dev/null; then
    UPTIME=$(echo $HEALTH | jq -r '.uptime')
    MODELS=$(echo $HEALTH | jq -r '.models_loaded')
    echo "   ✓ API healthy (uptime: ${UPTIME}s, models: ${MODELS})"
else
    echo "   ✗ API unhealthy"
    sudo systemctl restart smartgrid-api
fi
echo ""

# 6. Disk Space
echo "6. Disk Space"
DISK_USAGE=$(df /opt | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -lt 80 ]; then
    echo "   ✓ Disk OK (${DISK_USAGE}% used)"
else
    echo "   ✗ Disk high (${DISK_USAGE}% used)"
fi
echo ""

# 7. Model Status
echo "7. Model Files"
if [ -f "/opt/smartgrid-ai/models/moe_day10_11.pkl" ]; then
    MODEL_SIZE=$(ls -lh /opt/smartgrid-ai/models/moe_day10_11.pkl | awk '{print $5}')
    echo "   ✓ Primary model loaded (${MODEL_SIZE})"
else
    echo "   ✗ Primary model missing"
fi
echo ""

echo "=== All checks complete ==="
```

### Monitoring During Business Hours

**Every hour**:

```bash
# Check error rate
journalctl -u smartgrid-api --since "1 hour ago" | grep ERROR | wc -l

# Check average response time
journalctl -u smartgrid-api --since "1 hour ago" | grep "duration" | \
  awk '{print $NF}' | sed 's/ms//' | awk '{sum+=$1; count++} END {print sum/count" ms"}'

# Check predictions processed
journalctl -u smartgrid-api --since "1 hour ago" | grep "prediction" | wc -l
```

**Every 4 hours**:

```bash
# Full system status
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')"
echo "MEM: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100}')"
echo "DISK: $(df /opt | tail -1 | awk '{print $5}')"
echo "DB: $(sudo -u postgres psql -c 'SELECT now()' | grep -v '^$' | tail -1)"
echo "CACHE: $(redis-cli info | grep connected_clients)"
```

### Evening Shutdown Checklist (17:00)

```bash
#!/bin/bash
# Evening shutdown verification

echo "=== Smart Grid AI - Evening Shutdown Check ==="

# 1. Current Load
CURRENT_CONNECTIONS=$(netstat -tln | grep ESTABLISHED | wc -l)
echo "Current connections: $CURRENT_CONNECTIONS"

# 2. Pending Requests
PENDING=$(ps aux | grep uvicorn | grep -v grep | wc -l)
echo "API workers running: $PENDING"

# 3. Recent Errors
ERRORS=$(journalctl -u smartgrid-api --since "4 hours ago" | grep ERROR | wc -l)
echo "Errors in last 4 hours: $ERRORS"

# 4. Backup Status
LAST_BACKUP=$(ls -t /backups/smartgrid-ai/*.tar.gz | head -1 | xargs stat -c %y)
echo "Last backup: $LAST_BACKUP"

# 5. Disk Usage
DISK_FREE=$(df /opt | tail -1 | awk '{print $4}')
echo "Free disk space: $DISK_FREE KB"

if [ $ERRORS -gt 50 ]; then
    echo "⚠️  ALERT: High error count in last 4 hours"
fi

echo "✅ Evening check complete"
```

---

## Monitoring & Health Checks

### Prometheus Metrics to Track

```
# Request Metrics
api_requests_total
api_request_duration_seconds
api_requests_errors_total

# Model Metrics
model_prediction_error_absolute
model_inference_duration_seconds
model_anomalies_detected

# System Metrics
process_resident_memory_bytes
process_cpu_seconds_total
process_open_fds
```

### Alert Thresholds

| Alert | Condition | Action |
|-------|-----------|--------|
| **CRITICAL** | API down for 1+ minute | Page on-call immediately |
| **CRITICAL** | Error rate > 5% for 5 min | Page on-call, assess |
| **CRITICAL** | Disk usage > 95% | Alert ops, investigate |
| **WARNING** | Latency P99 > 1 second | Monitor, investigate |
| **WARNING** | CPU > 80% for 10 min | Consider scaling |
| **WARNING** | Memory > 80% for 10 min | Consider scaling |
| **WARNING** | Error rate > 1% for 15 min | Investigate root cause |
| **INFO** | Latency P99 > 500ms | Log and monitor |
| **INFO** | 100k+ predictions processed | Monthly report |

### Health Check Indicators

```json
GET /health
{
  "status": "ok",
  "uptime": 86400,
  "models_loaded": 3,
  "database": "connected",
  "cache": "connected",
  "timestamp": "2026-01-30T09:00:00Z"
}
```

**Status Values**:
- `ok`: All systems operational
- `degraded`: One component non-critical down
- `critical`: Critical component down

---

## Troubleshooting Guide

### API Not Responding

**Symptom**: curl returns connection refused

```bash
# 1. Check if service is running
sudo systemctl status smartgrid-api

# 2. Check listening ports
sudo netstat -tlnp | grep :8000

# 3. Check process
ps aux | grep uvicorn | grep -v grep

# 4. View recent logs
sudo journalctl -u smartgrid-api -n 100 -p err

# 5. Restart service
sudo systemctl restart smartgrid-api

# 6. If still not working
sudo systemctl start smartgrid-api
sleep 5
curl -v http://localhost:8000/health
```

### High Error Rate

**Symptom**: Error rate > 1%

```bash
# 1. Check error types
sudo journalctl -u smartgrid-api | grep ERROR | tail -50

# 2. Check request volume
sudo journalctl -u smartgrid-api | wc -l

# 3. Check database connection
psql -h localhost -U smartgrid -c "SELECT 1"

# 4. Check model loading
curl http://localhost:8000/models

# 5. Check recent changes
git log --oneline -5

# 6. Restart to clear transient issues
sudo systemctl restart smartgrid-api
```

### Memory Usage High

**Symptom**: Memory > 2 GB

```bash
# 1. Check current usage
top -b -n 1 -p $(pgrep -f uvicorn | head -1) | tail -1

# 2. Check what's using memory
ps aux --sort=-%mem | head -10

# 3. Check for memory leaks
# Monitor over 1 hour
watch -n 60 'ps aux | grep uvicorn | grep -v grep | awk "{print \$6}"'

# 4. Options:
# - Reduce batch size limit
# - Restart service (clears accumulated memory)
# - Scale horizontally

sudo systemctl restart smartgrid-api
```

### Slow Predictions

**Symptom**: P99 latency > 1 second

```bash
# 1. Check load
ps aux | grep uvicorn | wc -l

# 2. Check CPU usage
top -b -n 1 | grep -i cpu

# 3. Check disk I/O
iostat -x 1 5

# 4. Check database performance
psql -c "SELECT query, calls, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10"

# 5. Options:
# - Add more workers
# - Enable caching
# - Scale to more instances
# - Optimize models (quantization)
```

### Database Connection Issues

**Symptom**: Cannot connect to PostgreSQL

```bash
# 1. Check if PostgreSQL is running
sudo systemctl status postgresql

# 2. Test connection
psql -h localhost -U smartgrid -d smartgrid_ai -c "SELECT 1"

# 3. Check connection limits
psql -c "SELECT sum(numbackends) FROM pg_stat_database"
psql -c "SHOW max_connections"

# 4. Check for stuck connections
psql -c "SELECT pid, usename, state FROM pg_stat_activity WHERE state != 'idle'"

# 5. Kill stuck connections if needed
psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid <> pg_backend_pid()"

# 6. Restart database
sudo systemctl restart postgresql
```

### Cache (Redis) Issues

**Symptom**: Redis connection refused

```bash
# 1. Check Redis status
sudo systemctl status redis-server

# 2. Test connection
redis-cli ping

# 3. Check memory usage
redis-cli info memory

# 4. Check for many keys
redis-cli dbsize

# 5. Clear cache if needed (WARNING: data loss)
redis-cli FLUSHALL

# 6. Restart Redis
sudo systemctl restart redis-server
```

---

## Maintenance Procedures

### Daily Backup

```bash
#!/bin/bash
# Daily backup script (runs at 02:00 AM)

BACKUP_DIR="/backups/smartgrid-ai"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting backup: $TIMESTAMP"

# 1. Backup models
tar -czf $BACKUP_DIR/models_$TIMESTAMP.tar.gz \
    /opt/smartgrid-ai/models/ || exit 1

# 2. Backup database
PGPASSWORD=$DB_PASSWORD pg_dump \
    -h localhost -U smartgrid smartgrid_ai | \
    gzip > $BACKUP_DIR/db_$TIMESTAMP.sql.gz || exit 1

# 3. Backup configuration
tar -czf $BACKUP_DIR/config_$TIMESTAMP.tar.gz \
    /opt/smartgrid-ai/config/ || exit 1

# 4. Upload to S3
aws s3 cp $BACKUP_DIR/ s3://smartgrid-backups/ --recursive || exit 1

# 5. Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

# 6. Verify backup
if [ -f "$BACKUP_DIR/models_$TIMESTAMP.tar.gz" ]; then
    echo "✓ Backup successful: $TIMESTAMP"
    echo "Backup size: $(du -sh $BACKUP_DIR/models_$TIMESTAMP.tar.gz | awk '{print $1}')"
else
    echo "✗ Backup failed!"
    # Alert ops team
    mail -s "ALERT: Backup failed" ops@smartgrid.ai
    exit 1
fi
```

### Weekly Log Rotation

```bash
# Rotate logs weekly (Sundays at 02:00 AM)
/opt/smartgrid-ai/logs/
  ├─ api_server.log (current)
  ├─ api_server.log.1 (last week)
  ├─ api_server.log.2 (2 weeks ago)
  └─ ... (keep 8 weeks = 60 days)

# Automated by logrotate
/etc/logrotate.d/smartgrid-api:
--
/opt/smartgrid-ai/logs/api_server.log {
    daily
    rotate 60
    compress
    delaycompress
    notifempty
    create 0640 smartgrid smartgrid
    sharedscripts
    postrotate
        systemctl reload smartgrid-api > /dev/null 2>&1 || true
    endscript
}
```

### Monthly Model Retraining

```bash
#!/bin/bash
# Monthly retraining (1st of month at 00:00 AM)

TIMESTAMP=$(date +%Y%m%d)

echo "Starting monthly model retraining..."

# 1. Backup current models
cp -r /opt/smartgrid-ai/models /opt/smartgrid-ai/models.backup.$TIMESTAMP

# 2. Run training job
python /opt/smartgrid-ai/training/train_all_models.py \
    --data /opt/smartgrid-ai/data/latest.csv \
    --output /opt/smartgrid-ai/models.new \
    --epochs 50 \
    --batch-size 32

# 3. Evaluate performance
python /opt/smartgrid-ai/training/evaluate_models.py \
    --models /opt/smartgrid-ai/models.new \
    --output /tmp/eval_$TIMESTAMP.json

# 4. Compare with current models
CURRENT_MAPE=$(cat /opt/smartgrid-ai/model_metrics.json | jq '.mape')
NEW_MAPE=$(cat /tmp/eval_$TIMESTAMP.json | jq '.mape')

if (( $(echo "$NEW_MAPE < $CURRENT_MAPE" | bc -l) )); then
    echo "✓ New models better ($NEW_MAPE < $CURRENT_MAPE)"
    
    # 5. Deploy new models
    mv /opt/smartgrid-ai/models.new/* /opt/smartgrid-ai/models/
    
    # 6. Restart API to load new models
    sudo systemctl restart smartgrid-api
    
    echo "✓ New models deployed and API restarted"
else
    echo "✗ New models worse ($NEW_MAPE >= $CURRENT_MAPE)"
    echo "Keeping current models"
    rm -rf /opt/smartgrid-ai/models.new
fi
```

### Quarterly Security Update

```bash
# Check for updates
apt list --upgradable

# Update system (non-production hours)
sudo apt update
sudo apt upgrade -y

# Update Python packages
pip list --outdated
pip install --upgrade smartgrid-api-requirements

# Verify after update
curl http://localhost:8000/health
```

---

## Performance Tuning

### Worker Configuration

```python
# /opt/smartgrid-ai/config/uvicorn.conf

# Based on: (CPU cores × 2) + 1
# For 4-core machine: 9 workers
workers = 9

# Connection pool
backlog = 2048
limit_concurrency = 1000
limit_max_requests = 10000
```

### Caching Strategy

```python
# Cache responses for 1 hour
@app.get("/models", cache_control="max-age=3600")
def get_models():
    return models

# Cache predictions for identical inputs
@app.post("/predict/single")
@cache(expire=300)  # 5 minutes
async def predict_single(request: PredictionRequest):
    return predictions
```

### Batch Optimization

```python
# Tune batch size based on memory
# 32: Fast, uses 200MB
# 64: Faster, uses 400MB
# 128: Fastest, uses 800MB

# Limit to safe values
MAX_BATCH_SIZE = 1000  # Don't exceed
RECOMMENDED_BATCH_SIZE = 256
```

### Database Optimization

```sql
-- Create indexes on frequently queried columns
CREATE INDEX idx_predictions_timestamp ON predictions(created_at);
CREATE INDEX idx_predictions_user_id ON predictions(user_id);

-- Analyze query performance
ANALYZE;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan FROM pg_stat_user_indexes;
```

---

## Disaster Recovery

### Recovery Time Objectives (RTO)

- **Critical Components**: 15 minutes
- **API Service**: 30 minutes
- **Full System**: 1 hour

### Recovery Point Objective (RPO)

- **Database**: 1 day (daily backups)
- **Models**: 1 day (daily backups)
- **Configurations**: 1 day (daily backups)

### Recovery Procedures

**Scenario 1: API Service Crash**

```bash
# 1. Check if it's just a process crash
sudo systemctl status smartgrid-api

# 2. Restart
sudo systemctl restart smartgrid-api
sleep 5

# 3. Verify
curl http://localhost:8000/health

# RTO: 1-2 minutes
```

**Scenario 2: Database Corruption**

```bash
# 1. Stop API
sudo systemctl stop smartgrid-api

# 2. Find latest backup
ls -t /backups/smartgrid-ai/db_*.sql.gz | head -1

# 3. Restore
BACKUP_FILE="/backups/smartgrid-ai/db_20260130_020000.sql.gz"
gunzip < $BACKUP_FILE | psql smartgrid_ai

# 4. Verify
psql smartgrid_ai -c "SELECT COUNT(*) FROM predictions"

# 5. Restart API
sudo systemctl start smartgrid-api

# RTO: 10-15 minutes
```

**Scenario 3: Model Files Lost**

```bash
# 1. Find backup
ls -t /backups/smartgrid-ai/models_*.tar.gz | head -1

# 2. Stop API
sudo systemctl stop smartgrid-api

# 3. Restore models
BACKUP_FILE="/backups/smartgrid-ai/models_20260130_020000.tar.gz"
tar -xzf $BACKUP_FILE -C /opt/smartgrid-ai/

# 4. Restart API
sudo systemctl start smartgrid-api

# RTO: 5 minutes
```

**Scenario 4: Complete Server Loss**

```bash
# 1. Provision new server (same specs)
# 2. Install OS and dependencies
# 3. Deploy using deployment guide
# 4. Restore from backups:

# Database
gunzip < backup_db.sql.gz | psql smartgrid_ai

# Models
tar -xzf backup_models.tar.gz -C /opt/smartgrid-ai/

# Configuration
tar -xzf backup_config.tar.gz -C /opt/smartgrid-ai/

# 5. Verify all services
curl http://localhost:8000/health

# RTO: 45-60 minutes
```

---

## Contact & Escalation

### Support Contacts

| Role | Name | Phone | Email | Hours |
|------|------|-------|-------|-------|
| **L1 Support** | Operations Team | +1-555-0100 | ops@smartgrid.ai | 24/7 |
| **L2 Support** | Senior DevOps | +1-555-0101 | devops-lead@smartgrid.ai | 08:00-18:00 |
| **L3 Support** | Engineering Lead | +1-555-0102 | engineering-lead@smartgrid.ai | Escalation |
| **On-Call** | Rotating | +1-555-0199 | on-call@smartgrid.ai | 24/7 |

### Escalation Matrix

**Level 1: Warning (Notify Operations)**
- Error rate 1-5%
- Latency P99 500ms-1s
- Memory 70-80%
- Contact: ops@smartgrid.ai
- Response Time: 1 hour

**Level 2: Alert (Page Senior DevOps)**
- Error rate > 5%
- Latency P99 > 1s
- Memory > 80%
- Contact: devops-lead@smartgrid.ai
- Response Time: 15 minutes

**Level 3: Critical (Page Engineering Lead)**
- API down
- Database down
- Data corruption
- Security breach
- Contact: engineering-lead@smartgrid.ai + on-call
- Response Time: 5 minutes

### Incident Communication

```
INCIDENT NOTIFICATION TEMPLATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INCIDENT_ID: INC-2026-01-30-001
SEVERITY: Critical
SERVICE: SmartGrid API
TIME_DETECTED: 2026-01-30 14:23:45 UTC
DURATION: 5 minutes
USERS_AFFECTED: All

DESCRIPTION:
API Service unavailable, returning 503 errors

ROOT_CAUSE: (to be determined)

ACTIONS_TAKEN:
1. Service restarted at 14:28 UTC
2. Recovery verified at 14:28:30 UTC

STATUS: RESOLVED
RESOLUTION_TIME: 5 minutes

FOLLOW_UP:
- Root cause analysis to be completed
- Preventive measures to be documented
- Post-incident review scheduled
```

---

**Document Created**: January 30, 2026  
**Last Updated**: January 30, 2026  
**Status**: Active  
**Maintained By**: Operations Team

---

## Appendix: Useful Links

- **API Documentation**: https://api.smartgrid.ai/docs
- **Grafana Dashboards**: https://monitoring.smartgrid.ai
- **Prometheus Metrics**: https://metrics.smartgrid.ai:9090
- **GitHub Repository**: https://github.com/smartgrid-ai/api
- **Internal Wiki**: https://wiki.smartgrid.ai
