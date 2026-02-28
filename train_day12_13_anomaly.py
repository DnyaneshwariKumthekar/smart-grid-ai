"""
DAY 12-13: ANOMALY DETECTION TRAINING
=====================================

Goal: Detect unusual energy consumption patterns in smart grid

Anomaly Detection Strategy:
  1. IsolationForest - Isolation-based anomaly detection
  2. OneClassSVM - Support Vector based anomaly detection
  3. AutoencoderAD - Deep learning reconstruction-based detection

Use Cases:
  - Detect equipment failures
  - Identify theft/tampering
  - Find unusual consumption spikes
  - Validate data quality

Data: Household Electric Power Consumption (UCI)
  - 415k sequences total (288 timesteps, 31 features)
  - Split: 80k train, 20k test
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import models from all_models.py
from models.all_models import AutoencoderAnomalyDetector

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Configuration
CONFIG = {
    'batch_size': 64,
    'epochs': 30,
    'learning_rate': 0.001,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'contamination': 0.05,  # Assume 5% of data is anomalous
    'input_dim': 31,
}

print(f"Device: {CONFIG['device']}")


# ============================================================================
# PART 1: DATA LOADING & PREPARATION
# ============================================================================

def load_processed_data(data_dir: str = "data/processed") -> Tuple:
    """Load processed real-world dataset"""
    print("\n" + "="*80)
    print("LOADING PROCESSED REAL-WORLD DATA")
    print("="*80)
    
    pickle_file = os.path.join(data_dir, "household_power_smartgrid_features.pkl")
    
    if os.path.exists(pickle_file):
        print(f"✓ Loading from pickle: {pickle_file}")
        df = pickle.load(open(pickle_file, 'rb'))
        
        # Extract features
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'consumption_total']]
        X = df[feature_cols].values.astype(np.float32)
        y = df['consumption_total'].values.astype(np.float32)
        
        print(f"  Shape: X {X.shape}, y {y.shape}")
    else:
        raise FileNotFoundError(f"Data file not found: {pickle_file}")
    
    # Use subset for training (use 30k for faster anomaly training)
    n_samples = min(30000, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)
    X = X[indices]
    y = y[indices]
    
    print(f"  Using subset: {n_samples} samples (optimized for anomaly detection)")
    
    # Train-test split: 80-20
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test


def create_synthetic_anomalies(X: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    """
    Create synthetic anomaly labels for evaluation
    Mark extreme values as anomalies
    """
    # Calculate z-scores for each feature
    z_scores = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8))
    
    # Sample is anomalous if any feature has z-score > 3
    anomalies = (z_scores.max(axis=1) > 3).astype(int)
    
    # Ensure we have approximately 'contamination' ratio
    n_anomalies_target = int(len(X) * contamination)
    if anomalies.sum() < n_anomalies_target:
        n_missing = n_anomalies_target - anomalies.sum()
        normal_indices = np.where(anomalies == 0)[0]
        missing_indices = np.random.choice(normal_indices, n_missing, replace=False)
        anomalies[missing_indices] = 1
    
    return anomalies


# ============================================================================
# PART 2: ANOMALY DETECTION MODELS
# ============================================================================

class IsolationForestAD:
    """Isolation Forest for Anomaly Detection"""
    
    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, X_train: np.ndarray):
        """Train isolation forest"""
        print("\n📊 Training IsolationForest...")
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled)
        self.fitted = True
        print("✓ IsolationForest training complete!")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (-1 = anomaly, 1 = normal)"""
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (higher = more anomalous)"""
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        scores = -self.model.score_samples(X_scaled)  # Negate for consistency
        return scores


class OneClassSVMAD:
    """One-Class SVM for Anomaly Detection"""
    
    def __init__(self, contamination: float = 0.05):
        self.nu = contamination  # Upper bound on training errors
        self.model = OneClassSVM(
            kernel='linear',  # Use linear kernel for faster training
            gamma='auto',
            nu=self.nu,
            shrinking=True,
            cache_size=200
        )
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, X_train: np.ndarray):
        """Train One-Class SVM"""
        print("\n📊 Training One-Class SVM...")
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled)
        self.fitted = True
        print("✓ One-Class SVM training complete!")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (-1 = anomaly, 1 = normal)"""
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Get decision function scores"""
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        scores = -self.model.decision_function(X_scaled)
        return scores


class AutoencoderAnomalyDetection:
    """Autoencoder-based Anomaly Detection"""
    
    def __init__(self, input_dim: int = 31, hidden_dim: int = 16):
        self.autoencoder = AutoencoderAnomalyDetector(input_dim, hidden_dim)
        self.device = CONFIG['device']
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=CONFIG['learning_rate'])
        self.criterion = nn.MSELoss()
        self.scaler = StandardScaler()
        self.threshold = None
        self.fitted = False
    
    def fit(self, X_train: np.ndarray, epochs: int = 30, threshold_percentile: float = 95):
        """Train autoencoder and set anomaly threshold"""
        print("\n📊 Training Autoencoder...")
        
        X_scaled = self.scaler.fit_transform(X_train)
        X_tensor = torch.FloatTensor(X_scaled)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0.0
            
            for (X_batch,) in dataloader:
                X_batch = X_batch.to(self.device)
                
                self.optimizer.zero_grad()
                reconstructed = self.autoencoder(X_batch)
                loss = self.criterion(reconstructed, X_batch)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}")
        
        # Set threshold based on training reconstruction error
        with torch.no_grad():
            train_recon_errors = []
            for (X_batch,) in dataloader:
                X_batch = X_batch.to(self.device)
                reconstructed = self.autoencoder(X_batch)
                errors = torch.mean((X_batch - reconstructed) ** 2, dim=1)
                train_recon_errors.append(errors.cpu().numpy())
            
            all_errors = np.concatenate(train_recon_errors)
            self.threshold = np.percentile(all_errors, threshold_percentile)
        
        self.fitted = True
        print(f"✓ Autoencoder training complete! Threshold: {self.threshold:.6f}")
        return self
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Get reconstruction error as anomaly score"""
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.autoencoder(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        
        return errors.cpu().numpy()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies based on reconstruction error"""
        scores = self.get_anomaly_scores(X)
        predictions = np.where(scores > self.threshold, -1, 1)
        return predictions


# ============================================================================
# PART 3: ENSEMBLE ANOMALY DETECTION
# ============================================================================

class EnsembleAnomalyDetector:
    """Ensemble combining 3 anomaly detection methods"""
    
    def __init__(self, contamination: float = 0.05):
        self.iso_forest = IsolationForestAD(contamination)
        self.ocsvm = OneClassSVMAD(contamination)
        self.autoencoder = AutoencoderAnomalyDetection()
        self.fitted = False
    
    def fit(self, X_train: np.ndarray):
        """Train all 3 models"""
        print("\n" + "="*80)
        print("TRAINING ENSEMBLE ANOMALY DETECTORS")
        print("="*80)
        
        self.iso_forest.fit(X_train)
        self.ocsvm.fit(X_train)
        self.autoencoder.fit(X_train, epochs=10)  # Reduced from 30
        
        self.fitted = True
        print("\n✓ All anomaly detectors trained!")
        return self
    
    def predict_ensemble(self, X: np.ndarray, voting_threshold: float = 0.5) -> np.ndarray:
        """
        Ensemble voting: anomaly if 2+ models agree
        
        Returns: 1 = normal, -1 = anomaly
        """
        if not self.fitted:
            raise ValueError("Ensemble not fitted yet")
        
        iso_pred = (self.iso_forest.predict(X) == -1).astype(int)  # 1 if anomaly
        ocsvm_pred = (self.ocsvm.predict(X) == -1).astype(int)
        ae_pred = (self.autoencoder.predict(X) == -1).astype(int)
        
        # Sum votes (0-3 scale)
        votes = iso_pred + ocsvm_pred + ae_pred
        
        # Anomaly if >= 2 models detect it
        ensemble_pred = np.where(votes >= 2, -1, 1)
        
        return ensemble_pred
    
    def get_anomaly_scores(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get anomaly scores from all models"""
        if not self.fitted:
            raise ValueError("Ensemble not fitted yet")
        
        return {
            'IsolationForest': self.iso_forest.get_anomaly_scores(X),
            'OneClassSVM': self.ocsvm.get_anomaly_scores(X),
            'Autoencoder': self.autoencoder.get_anomaly_scores(X),
        }


# ============================================================================
# PART 4: EVALUATION & ANALYSIS
# ============================================================================

def evaluate_anomaly_detection(ensemble: EnsembleAnomalyDetector, X_test: np.ndarray, 
                               y_true: np.ndarray, save_dir: str = "results") -> Dict:
    """Evaluate anomaly detection performance"""
    print("\n" + "="*80)
    print("EVALUATING ANOMALY DETECTION")
    print("="*80)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get predictions from ensemble
    ensemble_pred = ensemble.predict_ensemble(X_test)
    ensemble_pred_binary = (ensemble_pred == -1).astype(int)
    
    # Get individual predictions
    iso_pred = (ensemble.iso_forest.predict(X_test) == -1).astype(int)
    ocsvm_pred = (ensemble.ocsvm.predict(X_test) == -1).astype(int)
    ae_pred = (ensemble.autoencoder.predict(X_test) == -1).astype(int)
    
    # Get anomaly scores for ROC analysis
    anomaly_scores = ensemble.get_anomaly_scores(X_test)
    
    # Evaluate ensemble
    print(f"\n📊 Ensemble Performance (2+ models voting):")
    print(f"  Anomalies detected: {ensemble_pred_binary.sum()} / {len(X_test)}")
    print(f"  Anomaly rate: {ensemble_pred_binary.mean()*100:.2f}%")
    
    # Individual model performance
    print(f"\n📊 Individual Model Performance:")
    print(f"  IsolationForest:  {iso_pred.sum()} anomalies ({iso_pred.mean()*100:.2f}%)")
    print(f"  OneClassSVM:      {ocsvm_pred.sum()} anomalies ({ocsvm_pred.mean()*100:.2f}%)")
    print(f"  Autoencoder:      {ae_pred.sum()} anomalies ({ae_pred.mean()*100:.2f}%)")
    
    # Save predictions
    results_df = pd.DataFrame({
        'sample_id': np.arange(len(X_test)),
        'true_label': y_true,
        'ensemble_pred': ensemble_pred_binary,
        'iso_forest': iso_pred,
        'ocsvm': ocsvm_pred,
        'autoencoder': ae_pred,
        'iso_score': anomaly_scores['IsolationForest'],
        'ocsvm_score': anomaly_scores['OneClassSVM'],
        'ae_score': anomaly_scores['Autoencoder'],
    })
    
    results_file = os.path.join(save_dir, "anomaly_detection_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Saved: {results_file}")
    
    return {
        'ensemble_pred': ensemble_pred_binary,
        'iso_pred': iso_pred,
        'ocsvm_pred': ocsvm_pred,
        'ae_pred': ae_pred,
        'anomaly_scores': anomaly_scores,
        'results_df': results_df,
    }


def visualize_anomaly_detection(eval_results: Dict, y_true: np.ndarray, save_dir: str = "results"):
    """Generate visualization plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    ensemble_pred = eval_results['ensemble_pred']
    anomaly_scores = eval_results['anomaly_scores']
    
    # 1. Anomaly Score Distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Anomaly Detection Analysis', fontsize=16, fontweight='bold')
    
    # IsolationForest
    axes[0, 0].hist(anomaly_scores['IsolationForest'], bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('IsolationForest Scores', fontweight='bold')
    axes[0, 0].set_xlabel('Anomaly Score')
    axes[0, 0].set_ylabel('Frequency')
    
    # OneClassSVM
    axes[0, 1].hist(anomaly_scores['OneClassSVM'], bins=50, alpha=0.7, color='green')
    axes[0, 1].set_title('OneClassSVM Scores', fontweight='bold')
    axes[0, 1].set_xlabel('Anomaly Score')
    axes[0, 1].set_ylabel('Frequency')
    
    # Autoencoder
    axes[1, 0].hist(anomaly_scores['Autoencoder'], bins=50, alpha=0.7, color='orange')
    axes[1, 0].set_title('Autoencoder Scores', fontweight='bold')
    axes[1, 0].set_xlabel('Reconstruction Error')
    axes[1, 0].set_ylabel('Frequency')
    
    # Ensemble agreement
    ensemble_votes = (eval_results['iso_pred'] + eval_results['ocsvm_pred'] + eval_results['ae_pred'])
    vote_counts = pd.Series(ensemble_votes).value_counts().sort_index()
    axes[1, 1].bar(vote_counts.index, vote_counts.values, color=['green', 'yellow', 'red'])
    axes[1, 1].set_title('Model Agreement Voting', fontweight='bold')
    axes[1, 1].set_xlabel('Number of Models Detecting Anomaly')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_xticks([0, 1, 2, 3])
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/anomaly_distributions.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/anomaly_distributions.png")
    plt.close()
    
    # 2. Feature-level anomaly heatmap (top 100 anomalies)
    anomaly_indices = np.where(ensemble_pred == 1)[0][:100]
    
    if len(anomaly_indices) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap of anomalous samples
        heatmap_data = eval_results['results_df'].iloc[anomaly_indices][
            ['iso_score', 'ocsvm_score', 'ae_score']
        ].values
        
        sns.heatmap(heatmap_data, cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Anomaly Score'})
        ax.set_title('Top 100 Detected Anomalies - Model Scores', fontsize=12, fontweight='bold')
        ax.set_ylabel('Anomaly Sample Index')
        ax.set_xticklabels(['IsolationForest', 'OneClassSVM', 'Autoencoder'])
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/anomaly_heatmap.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}/anomaly_heatmap.png")
        plt.close()


def save_anomaly_models(ensemble: EnsembleAnomalyDetector, save_dir: str = "models"):
    """Save trained anomaly detection models"""
    os.makedirs(save_dir, exist_ok=True)
    
    model_package = {
        'iso_forest': ensemble.iso_forest,
        'ocsvm': ensemble.ocsvm,
        'autoencoder': ensemble.autoencoder,
        'config': CONFIG,
        'timestamp': datetime.now().isoformat(),
    }
    
    save_file = os.path.join(save_dir, "anomaly_detection_day12_13.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(model_package, f)
    
    file_size = os.path.getsize(save_file) / (1024 * 1024)
    print(f"\n✓ Saved anomaly detection models: {save_file} ({file_size:.2f} MB)")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "="*80)
    print("DAY 12-13: ANOMALY DETECTION TRAINING")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}")
    
    # Load data
    X_train, y_train, X_test, y_test = load_processed_data()
    
    # Create synthetic anomaly labels for evaluation
    print("\n" + "="*80)
    print("CREATING SYNTHETIC ANOMALY LABELS")
    print("="*80)
    y_test_labels = create_synthetic_anomalies(X_test, contamination=0.05)
    print(f"✓ Created labels: {y_test_labels.sum()} anomalies, {(1-y_test_labels).sum()} normal")
    
    # Train ensemble
    ensemble = EnsembleAnomalyDetector(contamination=0.05)
    ensemble.fit(X_train)
    
    # Evaluate
    eval_results = evaluate_anomaly_detection(ensemble, X_test, y_test_labels)
    
    # Visualize
    visualize_anomaly_detection(eval_results, y_test_labels)
    
    # Save models
    save_anomaly_models(ensemble)
    
    # Summary
    print("\n" + "="*80)
    print("DAY 12-13 COMPLETE ✅")
    print("="*80)
    print(f"\nDeliverables:")
    print(f"  ✓ 3 Anomaly detection models trained")
    print(f"    - IsolationForest")
    print(f"    - One-Class SVM")
    print(f"    - Autoencoder")
    print(f"  ✓ Ensemble voting mechanism")
    print(f"  ✓ Models saved: models/anomaly_detection_day12_13.pkl")
    print(f"  ✓ Results: results/anomaly_detection_results.csv")
    print(f"  ✓ Visualizations: results/anomaly_*.png")
    print(f"\nNext: Days 15-20 Analysis & Benchmarking")


if __name__ == "__main__":
    main()
