
# SMART GRID AI REST API v1.0

## Base URL
https://api.smartgrid-ai.com/v1

## Authentication
All endpoints require Bearer token in Authorization header:
Authorization: Bearer <your_api_key>

## Endpoints

### 1. Get Forecast
**GET /forecast**

Query Parameters:
- `horizon`: Forecast horizon in hours (1-168)
- `include_intervals`: Boolean, include 95% CI (default: true)
- `scenario`: Scenario type (base, conservative, optimistic)

Example:
GET /forecast?horizon=24&include_intervals=true

Response:
{
  "forecasts": [
    {
      "timestamp": "2024-01-15T10:00:00Z",
      "consumption_kwh": 1234.5,
      "lower_95": 1150.2,
      "upper_95": 1318.8
    }
  ],
  "status": "success"
}

### 2. Get Anomalies
**GET /anomalies**

Query Parameters:
- `hours_back`: Number of hours to look back (default: 24)
- `severity`: Minimum severity level (low, medium, high, critical)

Response:
{
  "anomalies": [
    {
      "timestamp": "2024-01-15T09:30:00Z",
      "consumption_kwh": 2150.3,
      "anomaly_score": -0.85,
      "severity": "high",
      "root_cause": "Peak Load Spike"
    }
  ]
}

### 3. Get Model Performance
**GET /models/performance**

Response:
{
  "models": [
    {"name": "LSTM", "mape": 4.8, "r2": 0.88},
    {"name": "Transformer", "mape": 4.1, "r2": 0.90},
    {"name": "Ensemble", "mape": 4.32, "r2": 0.891}
  ]
}

## Rate Limiting
- 1000 requests per hour
- 100 requests per minute

## Status Codes
- 200: Success
- 400: Bad request
- 401: Unauthorized
- 429: Too many requests
- 500: Server error
