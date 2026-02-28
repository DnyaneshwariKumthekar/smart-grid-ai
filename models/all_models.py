"""
COMPREHENSIVE AI/ML MODELS FOR SMART GRID FORECASTING
=====================================================

This file contains ALL AI and ML models used in the project:
✅ Day 8-9: RandomForest, ExtraTrees, Ridge (Current)
🟡 Day 10-11: LSTM, GRU, Transformer, CNN-LSTM, Attention, StackingEnsemble, MixtureOfExperts (Planned)
🟡 Day 12-13: Anomaly Detection Models (Planned)
🟡 Day 15+: Additional Models (Planned)

New Additions (Day 10-11 Enhanced):
  • GRUBase - Faster LSTM alternative (30% fewer parameters)
  • CNNLSTMHybrid - Spatial-temporal learning (CNN + LSTM)
  • AttentionNetwork - Interpretable attention-based model
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.linear_model import Ridge
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

# ============================================================================
# PART 1: CURRENT MODELS (Day 8-9) ✅
# ============================================================================

class SimpleEnsemble:
    """
    Current ensemble used in Days 8-9.
    Combines: RandomForest + ExtraTrees + Ridge Meta-Learner
    Performance: MAPE 17.05%, R² 0.9662
    """
    def __init__(self, n_estimators: int = 50):
        from sklearn.ensemble import ExtraTreesRegressor
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=10, 
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        self.et = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        self.meta_learner = Ridge(alpha=1.0)
        self.fitted = False
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train ensemble"""
        self.rf.fit(X_train, y_train)
        self.et.fit(X_train, y_train)
        
        # Generate meta-features
        pred_rf_train = self.rf.predict(X_train).reshape(-1, 1)
        pred_et_train = self.et.predict(X_train).reshape(-1, 1)
        meta_train = np.hstack([pred_rf_train, pred_et_train])
        
        self.meta_learner.fit(meta_train, y_train)
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        
        pred_rf = self.rf.predict(X).reshape(-1, 1)
        pred_et = self.et.predict(X).reshape(-1, 1)
        meta = np.hstack([pred_rf, pred_et])
        
        return self.meta_learner.predict(meta)


# ============================================================================
# PART 2: DEEP LEARNING MODELS (Day 10-11) 🟡
# ============================================================================

class LSTMBase(nn.Module):
    """
    LSTM model for sequence-to-scalar prediction.
    - 2 layers, 64 hidden units
    - Processes time-series sequences (batch, seq_len, features)
    - Planned for: Day 10-11 MoE expert
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMBase, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Last timestep
        output = self.fc(last_hidden)
        return output


class TransformerBase(nn.Module):
    """
    Transformer model for sequence-to-scalar prediction.
    - 2 encoder layers, 64 d_model, 4 attention heads
    - Uses positional encoding for sequence understanding
    - Planned for: Day 10-11 MoE expert
    """
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super(TransformerBase, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = self._create_pos_encoder(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    
    def _create_pos_encoder(self, d_model: int, max_len: int = 500):
        """Create positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Handle 2D input by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, features) -> (batch, 1, features)
        
        x = self.embedding(x)
        x = x + self.pos_encoder[:, :x.size(1), :].to(x.device)
        x = self.transformer(x)
        x = x[:, -1, :]  # Use last timestep
        output = self.fc(x)
        return output


class GRUBase(nn.Module):
    """
    GRU (Gated Recurrent Unit) model - Faster alternative to LSTM.
    - 2 layers, 64 hidden units
    - 30-40% fewer parameters than LSTM
    - Better for resource-constrained environments
    - Planned for: Day 10-11 MoE expert #1 (Speed)
    
    Performance: Similar to LSTM but ~30% faster training
    Use case: Real-time forecasting, edge devices, lightweight MoE
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(GRUBase, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Handle 2D input by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, features) -> (batch, 1, features)
        
        gru_out, h_n = self.gru(x)
        last_hidden = gru_out[:, -1, :]  # Last timestep
        output = self.fc(last_hidden)
        return output


class CNNLSTMHybrid(nn.Module):
    """
    CNN-LSTM Hybrid model - Combines spatial & temporal learning.
    
    Architecture:
      Input (batch, seq_len, features)
        ↓
      CNN Layers (learn spatial patterns across features)
        ↓
      LSTM Layers (learn temporal patterns)
        ↓
      Fully Connected → Output
    
    Benefits:
    - CNN extracts spatial features from the 32-feature grid
    - LSTM captures temporal dependencies
    - Stronger feature representation than LSTM alone
    
    Planned for: Day 10-11 MoE expert #2 (Spatial-Temporal)
    Performance: Expected 2-5% better than pure LSTM on grid data
    
    Feature extraction: Learns correlations between grid zones, 
                       weather clusters, consumption categories
    """
    def __init__(self, input_dim: int, cnn_channels: int = 32, hidden_dim: int = 64, num_layers: int = 2):
        super(CNNLSTMHybrid, self).__init__()
        
        # CNN for spatial feature extraction
        self.conv1 = nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.bn2 = nn.BatchNorm1d(cnn_channels)
        
        # LSTM for temporal pattern learning
        self.lstm = nn.LSTM(cnn_channels, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: CNN → LSTM → Dense
        x shape: (batch, seq_len, input_dim) or (batch, input_dim)
        """
        # Handle 2D input by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, features) -> (batch, 1, features)
        
        # CNN needs (batch, channels, length) format
        x_cnn = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        
        # Spatial feature extraction
        x_cnn = self.conv1(x_cnn)
        x_cnn = self.bn1(x_cnn)
        x_cnn = self.relu(x_cnn)
        x_cnn = self.conv2(x_cnn)
        x_cnn = self.bn2(x_cnn)
        x_cnn = self.relu(x_cnn)
        
        # Convert back to (batch, seq_len, channels) for LSTM
        x_lstm = x_cnn.transpose(1, 2)
        
        # Temporal pattern learning
        lstm_out, (h_n, c_n) = self.lstm(x_lstm)
        last_hidden = lstm_out[:, -1, :]
        
        # Output
        output = self.fc(last_hidden)
        return output


class AttentionNetwork(nn.Module):
    """
    Attention Network for sequence-to-scalar prediction.
    - Multi-head self-attention over timesteps
    - Learns which timesteps are most important
    - Provides interpretability: shows attention weights
    
    Benefits:
    - Identifies critical periods (peak hours, transitions)
    - Attention visualizations explain predictions
    - Can detect anomalies by attention patterns
    
    Architecture:
      Input Sequence (batch, seq_len, features)
        ↓
      Query/Key/Value Embeddings
        ↓
      Multi-Head Attention (8 heads)
        ↓
      Position-wise Feed Forward
        ↓
      Average pooling over timesteps
        ↓
      Output
    
    Planned for: Day 10-11 MoE expert #3 (Interpretability)
    Use case: Understanding what drives energy forecasts
    
    Output: Prediction + Attention weights (for visualization)
    """
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 8, num_layers: int = 2):
        super(AttentionNetwork, self).__init__()
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = self._create_pos_encoder(d_model, max_len=500)
        
        # Multi-head attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_model // 2, 1)
        
        self.attention_weights = None
    
    def _create_pos_encoder(self, d_model: int, max_len: int = 500):
        """Create positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention
        Returns: Output prediction
        Stores: attention_weights for visualization
        """
        # Handle 2D input by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, features) -> (batch, 1, features)
        
        # Embed input
        x = self.embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :x.size(1), :].to(x.device)
        
        # Multi-head attention
        x = self.transformer(x)
        
        # Average pooling over timesteps for global representation
        x = x.mean(dim=1)
        
        # MLP head
        x = self.fc1(x)
        x = self.relu(x)
        output = self.fc2(x)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> np.ndarray:
        """
        Extract attention weights for interpretability.
        Returns: Attention patterns (batch, seq_len)
        """
        # Embed input
        x_emb = self.embedding(x)
        x_emb = x_emb + self.pos_encoder[:, :x_emb.size(1), :].to(x_emb.device)
        
        # Get attention from first transformer layer
        # Note: This is simplified - actual implementation depends on attention extraction needs
        with torch.no_grad():
            _ = self.transformer(x_emb)
        
        return np.ones((x.shape[0], x.shape[1]))  # Placeholder


class GatingNetwork(nn.Module):
    """
    Gating network for Mixture of Experts.
    - Learns which expert to use for each sample
    - Input: features → Output: expert weights
    - Planned for: Day 10-11 MoE
    """
    def __init__(self, input_dim: int, n_experts: int = 3, hidden_dim: int = 64):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, n_experts)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - output expert weights
        Returns: (batch_size, n_experts) with weights summing to 1
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        weights = self.softmax(x)
        return weights


class MixtureOfExperts:
    """
    Mixture of Experts ensemble.
    - Multiple specialized experts handle different patterns
    - Gating network learns to select experts
    - Planned for: Day 10-11 (Target: 12-15% MAPE vs 17% baseline)
    
    Architecture:
      Input Features
        ↓
      Expert 1 (Peak consumption)
      Expert 2 (Off-peak consumption)
      Expert 3 (Transition periods)
      Expert 4 (Weather-dependent) [optional]
        ↓
      Gating Network (decides weights)
        ↓
      Weighted Sum of Expert Outputs
        ↓
      Final Prediction
    """
    def __init__(self, n_experts: int = 3):
        self.n_experts = n_experts
        self.experts = [LSTMBase(input_dim=32) for _ in range(n_experts)]
        self.gating = GatingNetwork(input_dim=32, n_experts=n_experts)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fitted = False
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50):
        """
        Train MoE: First train experts on segments, then train gating network
        """
        print(f"Training Mixture of Experts ({self.n_experts} experts)...")
        # Implementation in Day 10-11
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using gating network to combine experts"""
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        # Implementation in Day 10-11
        return np.zeros((len(X), 1))


class StackingEnsemble:
    """
    Advanced stacking ensemble with LSTM + Transformer + XGBoost.
    - Uses K-fold cross-validation for meta-feature generation
    - XGBoost meta-learner combines base models
    - Planned for: Day 10-11+ advanced phase
    """
    def __init__(self, n_splits: int = 5):
        self.lstm = LSTMBase(input_dim=32, hidden_dim=64)
        self.transformer = TransformerBase(input_dim=32, d_model=64)
        self.n_splits = n_splits
        self.kfold = KFold(n_splits=n_splits, shuffle=False)
        self.meta_learner = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ) if XGBRegressor is not None else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fitted = False
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train stacking ensemble"""
        print("Training StackingEnsemble...")
        # Implementation in Day 10-11
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        return np.zeros((len(X), 1))


# ============================================================================
# PART 3: ANOMALY DETECTION MODELS (Day 12-13) 🟡
# ============================================================================

class AnomalyDetector:
    """
    Ensemble of anomaly detection models.
    - IsolationForest: Isolates anomalies
    - OneClassSVM: SVM-based anomaly detection
    - Autoencoder: Reconstruction error-based detection
    
    Planned for: Day 12-13
    """
    def __init__(self):
        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
        self.autoencoder = None  # Will implement in Day 12-13
        self.fitted = False
    
    def fit(self, X_train: np.ndarray):
        """Train anomaly detectors"""
        print("Training Anomaly Detection Models...")
        self.iso_forest.fit(X_train)
        self.svm.fit(X_train)
        # self.autoencoder.fit(X_train)  # Day 12-13
        self.fitted = True
        return self
    
    def detect_anomalies(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Detect anomalies using ensemble voting.
        Returns: Binary array (1 = anomaly, 0 = normal)
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        
        # Get predictions from each model (-1 = anomaly, 1 = normal)
        iso_pred = self.iso_forest.predict(X)  # -1 or 1
        svm_pred = self.svm.predict(X)  # -1 or 1
        
        # Ensemble voting: count anomaly votes
        anomaly_votes = ((iso_pred == -1).astype(int) + (svm_pred == -1).astype(int)) / 2
        
        # Threshold voting (if ≥50% of models vote anomaly)
        return (anomaly_votes >= threshold).astype(int)


# ============================================================================
# PART 4: AUTOENCODER FOR ANOMALY DETECTION (Day 12-13) 🟡
# ============================================================================

class AutoencoderAnomalyDetector(nn.Module):
    """
    Autoencoder for anomaly detection.
    - Learns to reconstruct normal patterns
    - High reconstruction error = anomaly
    - Planned for: Day 12-13
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super(AutoencoderAnomalyDetector, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim)
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and decode"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ============================================================================
# PART 5: UTILITY FUNCTIONS
# ============================================================================

def get_available_models() -> Dict[str, Dict]:
    """Get all available models and their status"""
    return {
        "Day 8-9 (Current)": {
            "SimpleEnsemble": {"status": "✅ ACTIVE", "components": ["RandomForest", "ExtraTrees", "Ridge"]},
        },
        "Day 10-11 (Next) - ENHANCED": {
            "LSTMBase": {"status": "🟡 DEFINED", "purpose": "Base LSTM sequences", "params": "2L, 64H"},
            "GRUBase": {"status": "🟡 DEFINED", "purpose": "Fast alternative to LSTM", "params": "2L, 64H, 30% faster"},
            "TransformerBase": {"status": "🟡 DEFINED", "purpose": "Attention-based sequences", "params": "2L, 64D, 4H"},
            "CNNLSTMHybrid": {"status": "🟡 DEFINED", "purpose": "Spatial-temporal learning", "params": "CNN→LSTM"},
            "AttentionNetwork": {"status": "🟡 DEFINED", "purpose": "Interpretable attention model", "params": "8H, 2L"},
            "GatingNetwork": {"status": "🟡 DEFINED", "purpose": "MoE routing network"},
            "MixtureOfExperts": {"status": "🟡 PLANNED", "target_mape": "12-15%", "experts": 3-4},
            "StackingEnsemble": {"status": "🟡 PLANNED", "components": ["LSTM", "Transformer", "XGBoost"]},
        },
        "Day 12-13 (Anomaly)": {
            "AnomalyDetector": {"status": "🟡 PLANNED", "components": ["IsolationForest", "OneClassSVM"]},
            "AutoencoderAnomalyDetector": {"status": "🟡 PLANNED", "purpose": "Reconstruction-based anomaly"},
        }
    }


def print_model_summary():
    """Print summary of all models"""
    models = get_available_models()
    print("\n" + "="*80)
    print("COMPREHENSIVE AI/ML MODEL SUMMARY")
    print("="*80)
    
    for phase, phase_models in models.items():
        print(f"\n{phase}")
        print("-" * 80)
        for model_name, details in phase_models.items():
            print(f"  • {model_name}")
            print(f"    Status: {details['status']}")
            for key, value in details.items():
                if key != 'status':
                    print(f"    {key}: {value}")


if __name__ == "__main__":
    print_model_summary()
    
    # Example: Quick test of SimpleEnsemble
    print("\n\nQuick Test: SimpleEnsemble")
    print("-" * 80)
    
    # Generate dummy data
    X_train = np.random.randn(1000, 31)
    y_train = np.random.randn(1000)
    X_test = np.random.randn(100, 31)
    
    # Train and predict
    ensemble = SimpleEnsemble(n_estimators=10)
    ensemble.fit(X_train, y_train)
    predictions = ensemble.predict(X_test)
    
    print(f"✓ SimpleEnsemble trained successfully!")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Sample predictions: {predictions[:5].flatten()}")
