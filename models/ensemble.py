"""
StackingEnsemble: Meta-learner approach combining LSTM and Transformer.
This ensemble uses base models' predictions as meta-features for XGBoost.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class LSTMBase(nn.Module):
    """LSTM base model for ensemble."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMBase, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x shape (batch, seq_len, input_dim)"""
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Last timestep
        output = self.fc(last_hidden)
        return output


class TransformerBase(nn.Module):
    """Transformer base model for ensemble."""
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super(TransformerBase, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = self._create_pos_encoder(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def _create_pos_encoder(self, d_model: int, max_len: int = 500):
        """Create positional encoding."""
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
        """Forward pass: x shape (batch, seq_len, input_dim)"""
        x = self.embedding(x)
        x = x + self.pos_encoder[:, :x.size(1), :].to(x.device)
        x = self.transformer(x)
        x = x[:, -1, :]  # Use last timestep
        output = self.fc(x)
        return output


class StackingEnsemble:
    """
    Stacking ensemble combining LSTM + Transformer base models with XGBoost meta-learner.
    Uses K-fold cross-validation to generate meta-features.
    """
    
    def __init__(self, 
                 lstm_hidden: int = 64, 
                 transformer_d_model: int = 64,
                 xgb_params: Dict = None,
                 n_splits: int = 5):
        """
        Initialize StackingEnsemble.
        
        Args:
            lstm_hidden: Hidden dimension for LSTM
            transformer_d_model: Model dimension for Transformer
            xgb_params: Parameters for XGBoost meta-learner
            n_splits: Number of folds for K-fold CV
        """
        self.lstm = LSTMBase(input_dim=32, hidden_dim=lstm_hidden)
        self.transformer = TransformerBase(input_dim=32, d_model=transformer_d_model)
        self.n_splits = n_splits
        self.kfold = KFold(n_splits=n_splits, shuffle=False, random_state=42)
        
        if xgb_params is None:
            xgb_params = {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42,
                'verbosity': 0
            }
        self.meta_learner = XGBRegressor(**xgb_params)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_models_fitted = False
    
    def _train_base_model(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, 
                         epochs: int = 50, batch_size: int = 32, lr: float = 0.001) -> float:
        """
        Train a base model (LSTM or Transformer).
        
        Args:
            model: PyTorch model to train
            X_train: Training features (n_samples, seq_len, n_features)
            y_train: Training targets (n_samples, 1)
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            
        Returns:
            Final training loss
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        model.train()
        
        n_batches = len(X_train) // batch_size
        for epoch in range(epochs):
            total_loss = 0
            indices = np.random.permutation(len(X_train))
            
            for i in range(n_batches):
                batch_idx = indices[i*batch_size:(i+1)*batch_size]
                X_batch = torch.FloatTensor(X_train[batch_idx]).to(self.device)
                y_batch = torch.FloatTensor(y_train[batch_idx]).to(self.device)
                
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / n_batches
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate meta-features using K-fold cross-validation.
        Each fold: train base models on fold_train, predict on fold_test.
        
        Args:
            X: Features (n_samples, seq_len, n_features)
            y: Targets (n_samples, 1)
            
        Returns:
            Meta-features (n_samples, 2) - predictions from LSTM and Transformer
        """
        print("  Generating meta-features using K-fold CV...")
        meta_features = np.zeros((X.shape[0], 2))  # 2 base models
        
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(X)):
            print(f"    Fold {fold+1}/{self.n_splits}")
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Train base models on this fold
            lstm_fold = LSTMBase(input_dim=32, hidden_dim=64)
            transformer_fold = TransformerBase(input_dim=32, d_model=64)
            
            self._train_base_model(lstm_fold, X_fold_train, y_fold_train, epochs=30)
            self._train_base_model(transformer_fold, X_fold_train, y_fold_train, epochs=30)
            
            # Generate predictions on validation fold
            lstm_fold.eval()
            transformer_fold.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_fold_val).to(self.device)
                lstm_preds = lstm_fold(X_val_tensor).cpu().numpy()
                transformer_preds = transformer_fold(X_val_tensor).cpu().numpy()
            
            meta_features[val_idx, 0] = lstm_preds.flatten()
            meta_features[val_idx, 1] = transformer_preds.flatten()
        
        print("  ✓ Meta-features generated")
        return meta_features
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            epochs: int = 50, verbose: bool = True) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble.
        
        Args:
            X_train: Training features (n_samples, seq_len, n_features)
            y_train: Training targets (n_samples, 1)
            epochs: Number of epochs for base models
            verbose: Whether to print progress
            
        Returns:
            self
        """
        if verbose:
            print("Training StackingEnsemble...")
        
        # Step 1: Train base models on full training data
        print("  Training base models...")
        self._train_base_model(self.lstm, X_train, y_train, epochs=epochs)
        self._train_base_model(self.transformer, X_train, y_train, epochs=epochs)
        self.base_models_fitted = True
        print("  ✓ Base models trained")
        
        # Step 2: Generate meta-features using K-fold CV
        meta_features_train = self._generate_meta_features(X_train, y_train)
        
        # Step 3: Train meta-learner on meta-features
        print("  Training meta-learner...")
        self.meta_learner.fit(meta_features_train, y_train.flatten())
        print("  ✓ Meta-learner trained")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the stacking ensemble.
        
        Args:
            X: Features (n_samples, seq_len, n_features)
            
        Returns:
            Predictions (n_samples, 1)
        """
        if not self.base_models_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.lstm.eval()
        self.transformer.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            lstm_preds = self.lstm(X_tensor).cpu().numpy()
            transformer_preds = self.transformer(X_tensor).cpu().numpy()
        
        meta_features = np.hstack([lstm_preds, transformer_preds])
        final_preds = self.meta_learner.predict(meta_features)
        
        return final_preds.reshape(-1, 1)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the ensemble on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics (MAPE, RMSE, MAE, R²)
        """
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mape = self._calculate_mape(y_test, y_pred)
        rmse = self._calculate_rmse(y_test, y_pred)
        mae = self._calculate_mae(y_test, y_pred)
        r2 = self._calculate_r2(y_test, y_pred)
        
        metrics = {
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        return metrics
    
    @staticmethod
    def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    
    @staticmethod
    def _calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def _calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def _calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)


# Example usage
if __name__ == "__main__":
    from data_loader import generate_synthetic_data, preprocess_data
    
    # Generate and preprocess data
    print("Loading data...")
    df = generate_synthetic_data(n_samples=100000)  # Smaller for testing
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df, test_size=0.2)
    
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    
    # Train stacking ensemble
    print(f"\nTraining StackingEnsemble...")
    ensemble = StackingEnsemble(n_splits=3)
    ensemble.fit(X_train, y_train, epochs=10)
    
    # Evaluate
    print(f"\nEvaluating on test set...")
    metrics = ensemble.evaluate(X_test, y_test)
    print(f"\nTest Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    print("\n✓ StackingEnsemble training complete!")
