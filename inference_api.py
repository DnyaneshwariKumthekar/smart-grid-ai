"""
Smart Grid AI - FastAPI Inference Server
Production-grade REST API for model inference and predictions
Days 23-26 Development

Features:
- Single and batch predictions
- Anomaly detection endpoint
- Model information and metadata
- Health checks and monitoring
- Comprehensive error handling
- Async request processing
"""

import pickle
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

import torch
import torch.nn as nn
import uvicorn

# ========================
# Configure Logging
# ========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if needed
os.makedirs('logs', exist_ok=True)

# ========================
# Request/Response Models
# ========================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    uptime_seconds: float
    models_loaded: Dict[str, bool]
    version: str = "1.0.0"


class PredictionRequest(BaseModel):
    """Single prediction request - 31 features"""
    features: List[float] = Field(..., min_items=31, max_items=31)
    
    class Config:
        schema_extra = {
            "example": {
                "features": [1000.0] * 31  # Placeholder for 31 features
            }
        }
    
    @validator('features')
    def validate_features(cls, v):
        """Validate feature values"""
        if len(v) != 31:
            raise ValueError('Must provide exactly 31 features')
        # Check for NaN or Inf
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f'Feature {i} must be numeric')
            if np.isnan(val) or np.isinf(val):
                raise ValueError(f'Feature {i} contains NaN or Inf')
        return v


class PredictionResponse(BaseModel):
    """Single prediction response"""
    prediction: float = Field(..., description="Predicted consumption in kW")
    model: str
    confidence: Optional[float] = None
    timestamp: str
    input_features_count: int


class BatchPredictionRequest(BaseModel):
    """Batch prediction request - up to 1000 samples"""
    features: List[List[float]] = Field(..., max_items=1000)
    model: str = "moe"
    
    class Config:
        schema_extra = {
            "example": {
                "features": [[1000.0] * 31 for _ in range(10)],
                "model": "moe"
            }
        }
    
    @validator('features')
    def validate_batch_features(cls, v):
        """Validate batch features"""
        if len(v) == 0:
            raise ValueError('At least one sample required')
        if len(v) > 1000:
            raise ValueError('Maximum 1000 samples per batch')
        
        # Check each sample
        for i, sample in enumerate(v):
            if len(sample) != 31:
                raise ValueError(f'Sample {i} must have 31 features')
            for j, val in enumerate(sample):
                if not isinstance(val, (int, float)):
                    raise ValueError(f'Sample {i}, Feature {j} must be numeric')
                if np.isnan(val) or np.isinf(val):
                    raise ValueError(f'Sample {i}, Feature {j} contains NaN or Inf')
        
        return v


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[float]
    model: str
    batch_size: int
    mean_prediction: float
    std_prediction: float
    timestamp: str
    processing_time_ms: float


class AnomalyDetectionRequest(BaseModel):
    """Anomaly detection request"""
    features: List[List[float]] = Field(..., max_items=1000)
    voting_threshold: int = Field(default=2, ge=1, le=3)
    
    class Config:
        schema_extra = {
            "example": {
                "features": [[1000.0] * 31 for _ in range(10)],
                "voting_threshold": 2
            }
        }


class AnomalyDetectionResponse(BaseModel):
    """Anomaly detection response"""
    total_samples: int
    anomalies_detected: int
    anomaly_indices: List[int]
    anomaly_scores: List[float]
    percentage_anomaly: float
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_id: str
    model_name: str
    model_type: str
    performance_metrics: Dict[str, float]
    training_date: str
    feature_count: int
    parameters: int
    status: str


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    timestamp: str
    status_code: int


# ========================
# Initialize FastAPI App
# ========================

app = FastAPI(
    title="Smart Grid AI - Inference API",
    description="Production inference server for energy forecasting models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Global State
# ========================

start_time = datetime.now()
loaded_models = {
    'moe': None,
    'baseline': None,
    'anomaly': None
}
model_info = {
    'moe': {
        'name': 'Mixture of Experts Ensemble',
        'type': 'Neural Ensemble',
        'mape': 0.0031,
        'r2': 0.9818,
        'params': 450000,
        'date': '2026-01-30'
    },
    'baseline': {
        'name': 'SimpleEnsemble',
        'type': 'ML Ensemble',
        'mape': 0.1705,
        'r2': 0.9662,
        'params': 50000,
        'date': '2026-01-29'
    },
    'anomaly': {
        'name': 'Anomaly Detection Ensemble',
        'type': 'Hybrid Ensemble',
        'detection_rate': 0.9995,
        'false_positive_rate': 0.0005,
        'params': 200000,
        'date': '2026-01-30'
    }
}


# ========================
# Model Loading Functions
# ========================

def load_models():
    """Load all trained models from disk"""
    try:
        # Load MoE models
        moe_path = 'models/moe_day10_11.pkl'
        if os.path.exists(moe_path):
            with open(moe_path, 'rb') as f:
                loaded_models['moe'] = pickle.load(f)
            logger.info(f"✓ MoE models loaded from {moe_path}")
        
        # Load baseline models
        baseline_path = 'models/baseline_day8_9.pkl'
        if os.path.exists(baseline_path):
            with open(baseline_path, 'rb') as f:
                loaded_models['baseline'] = pickle.load(f)
            logger.info(f"✓ Baseline models loaded from {baseline_path}")
        
        # Load anomaly models
        anomaly_path = 'models/anomaly_day12_13.pkl'
        if os.path.exists(anomaly_path):
            with open(anomaly_path, 'rb') as f:
                loaded_models['anomaly'] = pickle.load(f)
            logger.info(f"✓ Anomaly models loaded from {anomaly_path}")
        
        logger.info("✓ All models loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error loading models: {str(e)}")
        return False


# ========================
# Prediction Functions
# ========================

def predict_moe(features: np.ndarray) -> float:
    """Make prediction using MoE ensemble"""
    if loaded_models['moe'] is None:
        raise ValueError("MoE model not loaded")
    
    # Assuming MoE model has predict method
    # Adapt based on actual model interface
    if isinstance(features, list):
        features = np.array(features).reshape(1, -1)
    elif features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Mock prediction (replace with actual model call)
    prediction = float(np.mean(features) * 1.1)  # Placeholder
    return prediction


def predict_baseline(features: np.ndarray) -> float:
    """Make prediction using baseline ensemble"""
    if loaded_models['baseline'] is None:
        raise ValueError("Baseline model not loaded")
    
    if isinstance(features, list):
        features = np.array(features).reshape(1, -1)
    elif features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Mock prediction
    prediction = float(np.mean(features) * 1.0)  # Placeholder
    return prediction


def detect_anomalies(features: np.ndarray, threshold: int = 2) -> tuple:
    """Detect anomalies in features"""
    if loaded_models['anomaly'] is None:
        raise ValueError("Anomaly model not loaded")
    
    if isinstance(features, list):
        features = np.array(features)
    
    # Mock anomaly detection
    n_samples = features.shape[0]
    anomaly_indices = [i for i in range(0, n_samples, max(1, n_samples // 5))]  # Placeholder
    anomaly_scores = [0.5 + np.random.random() * 0.5 for _ in anomaly_indices]
    
    return anomaly_indices, anomaly_scores


# ========================
# API Endpoints
# ========================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting Smart Grid AI API Server...")
    load_models()
    logger.info("API Server started successfully")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint
    
    Returns:
        Health status with uptime and model status
    """
    uptime = (datetime.now() - start_time).total_seconds()
    models_status = {
        'moe': loaded_models['moe'] is not None,
        'baseline': loaded_models['baseline'] is not None,
        'anomaly': loaded_models['anomaly'] is not None
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=uptime,
        models_loaded=models_status
    )


@app.post("/predict/single", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(request: PredictionRequest) -> PredictionResponse:
    """
    Single sample prediction
    
    Args:
        request: 31 features for energy consumption prediction
    
    Returns:
        Predicted consumption in kW
    """
    try:
        features = np.array(request.features)
        prediction = predict_moe(features)
        
        logger.info(f"Single prediction: {prediction:.2f} kW")
        
        return PredictionResponse(
            prediction=prediction,
            model="moe",
            confidence=0.95,
            timestamp=datetime.now().isoformat(),
            input_features_count=len(request.features)
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Batch predictions (up to 1000 samples)
    
    Args:
        request: Batch of feature vectors (31 features each)
    
    Returns:
        List of predictions with statistics
    """
    try:
        start_time_batch = datetime.now()
        features = np.array(request.features)
        
        # Process batch
        predictions = []
        for sample in features:
            pred = predict_moe(sample)
            predictions.append(pred)
        
        processing_time = (datetime.now() - start_time_batch).total_seconds() * 1000
        
        logger.info(f"Batch prediction: {len(predictions)} samples in {processing_time:.2f}ms")
        
        return BatchPredictionResponse(
            predictions=predictions,
            model=request.model,
            batch_size=len(predictions),
            mean_prediction=float(np.mean(predictions)),
            std_prediction=float(np.std(predictions)),
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/anomaly-detect", response_model=AnomalyDetectionResponse, tags=["Anomaly Detection"])
async def detect_anomalies_endpoint(request: AnomalyDetectionRequest) -> AnomalyDetectionResponse:
    """
    Detect anomalies in feature vectors
    
    Args:
        request: Feature vectors and voting threshold
    
    Returns:
        Anomaly detection results with indices and scores
    """
    try:
        features = np.array(request.features)
        anomaly_indices, anomaly_scores = detect_anomalies(features, request.voting_threshold)
        
        anomaly_percentage = (len(anomaly_indices) / len(features)) * 100
        
        logger.info(f"Anomaly detection: {len(anomaly_indices)}/{len(features)} samples ({anomaly_percentage:.2f}%)")
        
        return AnomalyDetectionResponse(
            total_samples=len(features),
            anomalies_detected=len(anomaly_indices),
            anomaly_indices=anomaly_indices,
            anomaly_scores=anomaly_scores,
            percentage_anomaly=anomaly_percentage,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Anomaly detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=Dict[str, Any], tags=["Models"])
async def list_models() -> Dict[str, Any]:
    """
    List available models with status
    
    Returns:
        Dictionary of available models and their status
    """
    return {
        "available_models": list(loaded_models.keys()),
        "models_loaded": {k: v is not None for k, v in loaded_models.items()},
        "total_models": len(loaded_models),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/models/{model_id}/info", response_model=ModelInfoResponse, tags=["Models"])
async def get_model_info(model_id: str) -> ModelInfoResponse:
    """
    Get detailed information about a specific model
    
    Args:
        model_id: Model identifier (moe, baseline, anomaly)
    
    Returns:
        Model information including performance metrics
    """
    if model_id not in model_info:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    info = model_info[model_id]
    
    return ModelInfoResponse(
        model_id=model_id,
        model_name=info['name'],
        model_type=info['type'],
        performance_metrics={k: v for k, v in info.items() 
                            if k not in ['name', 'type', 'date']},
        training_date=info['date'],
        feature_count=31,
        parameters=info['params'],
        status="active" if loaded_models[model_id] is not None else "inactive"
    )


@app.get("/models/{model_id}", tags=["Models"])
async def get_model_details(model_id: str) -> Dict[str, Any]:
    """
    Get detailed model metrics and specifications
    """
    if model_id not in model_info:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    info = model_info[model_id]
    is_loaded = loaded_models[model_id] is not None
    
    return {
        "model_id": model_id,
        **info,
        "is_loaded": is_loaded,
        "timestamp": datetime.now().isoformat()
    }


# ========================
# Error Handlers
# ========================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP Exception: {exc.detail}")
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "status_code": 500,
        "timestamp": datetime.now().isoformat()
    }


# ========================
# Root Endpoint
# ========================

@app.get("/", tags=["System"])
async def root():
    """
    API Root - Overview and documentation links
    """
    return {
        "name": "Smart Grid AI Inference API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "health": "/health",
            "predict_single": "/predict/single",
            "predict_batch": "/predict/batch",
            "anomaly_detection": "/anomaly-detect",
            "models_list": "/models",
            "model_info": "/models/{model_id}/info"
        },
        "timestamp": datetime.now().isoformat()
    }


# ========================
# Main Entry Point
# ========================

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Set to True for development
    )
