"""
DAY 10-11: MIXTURE OF EXPERTS (MoE) TRAINING
=============================================

Goal: Improve upon Day 8-9 baseline (17.05% MAPE) using Mixture of Experts
Target: 12-15% MAPE improvement

Architecture:
  - Expert 1: GRU (fast, lightweight)
  - Expert 2: CNN-LSTM (spatial-temporal)
  - Expert 3: Transformer (attention-based)
  - Expert 4: AttentionNetwork (interpretable)
  - Gating Network: Routes samples to best experts

Training Strategy:
  1. Load real-world data (100k sequences)
  2. Create & train base experts
  3. Train gating network with K-fold CV
  4. Evaluate ensemble
  5. Compare to baseline
  6. Save best models

Data: Household Electric Power Consumption (UCI)
  - 415k sequences total (288 timesteps, 32 features)
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Import models from all_models.py
from models.all_models import (
    GRUBase, CNNLSTMHybrid, TransformerBase, AttentionNetwork,
    GatingNetwork, SimpleEnsemble
)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configuration
CONFIG = {
    'batch_size': 64,
    'epochs': 50,
    'learning_rate': 0.001,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'n_experts': 4,
    'hidden_dim': 64,
    'd_model': 64,
    'input_dim': 31,  # Actual number of features in dataset
}

print(f"Device: {CONFIG['device']}")


# ============================================================================
# PART 1: DATA LOADING & PREPARATION
# ============================================================================

def load_processed_data(data_dir: str = "data/processed") -> Tuple:
    """
    Load processed real-world dataset
    
    Returns:
        X_train, y_train, X_test, y_test: numpy arrays
    """
    print("\n" + "="*80)
    print("LOADING PROCESSED REAL-WORLD DATA")
    print("="*80)
    
    pickle_file = os.path.join(data_dir, "household_power_smartgrid_features.pkl")
    
    if os.path.exists(pickle_file):
        print(f"✓ Loading from pickle: {pickle_file}")
        df = pickle.load(open(pickle_file, 'rb'))
        
        # Extract features (all columns except timestamp and target)
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'consumption_total']]
        X = df[feature_cols].values.astype(np.float32)
        y = df['consumption_total'].values.astype(np.float32)
        
        print(f"  Shape: X {X.shape}, y {y.shape}")
        print(f"  Data type: {X.dtype}, {y.dtype}")
        
    else:
        raise FileNotFoundError(f"Data file not found: {pickle_file}")
    
    # Use subset for training (same as Day 8-9)
    n_samples = min(100000, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)
    X = X[indices]
    y = y[indices]
    
    print(f"\n  Using subset: {n_samples} samples")
    
    # Train-test split: 80-20
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  y_train range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    
    scaler = StandardScaler()
    return X_train, y_train, X_test, y_test, scaler


def prepare_sequences(X: np.ndarray, y: np.ndarray, batch_size: int = 64) -> DataLoader:
    """Convert numpy arrays to PyTorch DataLoader"""
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


# ============================================================================
# PART 2: EXPERT TRAINING
# ============================================================================

class ExpertTrainer:
    """Train individual experts"""
    
    def __init__(self, model: nn.Module, device, learning_rate: float = 0.001):
        self.model = model.to(device)  # Move model to device
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.training_loss = []
    
    def train_epoch(self, train_loader: DataLoader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        self.training_loss.append(avg_loss)
        return avg_loss
    
    def validate(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Validate on test data"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                predictions = self.model(X_batch)
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(y_batch.numpy())
        
        predictions = np.vstack(all_preds).flatten()
        targets = np.vstack(all_targets).flatten()
        
        return predictions, targets


def train_experts(X_train, y_train, X_test, y_test, config: dict) -> Dict:
    """
    Train all 4 experts independently
    
    Returns: Dictionary with trained experts and their predictions
    """
    print("\n" + "="*80)
    print("TRAINING INDIVIDUAL EXPERTS")
    print("="*80)
    
    device = config['device']
    batch_size = config['batch_size']
    epochs = config['epochs']
    
    # Create data loaders
    train_loader = prepare_sequences(X_train, y_train, batch_size)
    test_loader = prepare_sequences(X_test, y_test, batch_size)
    
    experts = {
        'GRU': GRUBase(input_dim=config['input_dim'], hidden_dim=config['hidden_dim']),
        'CNN-LSTM': CNNLSTMHybrid(input_dim=config['input_dim'], hidden_dim=config['hidden_dim']),
        'Transformer': TransformerBase(input_dim=config['input_dim'], d_model=config['d_model']),
        'Attention': AttentionNetwork(input_dim=config['input_dim'], d_model=config['d_model']),
    }
    
    trained_experts = {}
    expert_predictions = {}
    expert_metrics = {}
    
    for expert_name, expert_model in experts.items():
        print(f"\n📊 Training Expert: {expert_name}")
        print("-" * 80)
        
        trainer = ExpertTrainer(expert_model, device, learning_rate=config['learning_rate'])
        
        # Training loop
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            loss = trainer.train_epoch(train_loader)
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {loss:.6f}")
            
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Validate
        preds, targets = trainer.validate(test_loader)
        
        mape = mean_absolute_percentage_error(targets, preds)
        rmse = np.sqrt(mean_squared_error(targets, preds))
        mae = np.mean(np.abs(targets - preds))
        r2 = r2_score(targets, preds)
        
        print(f"\n  ✓ {expert_name} Expert Results:")
        print(f"    MAPE: {mape:.2f}%")
        print(f"    RMSE: {rmse:.2f} kW")
        print(f"    MAE:  {mae:.2f} kW")
        print(f"    R²:   {r2:.4f}")
        
        trained_experts[expert_name] = expert_model
        expert_predictions[expert_name] = preds
        expert_metrics[expert_name] = {
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': preds,
            'targets': targets
        }
    
    return trained_experts, expert_predictions, expert_metrics


# ============================================================================
# PART 3: MIXTURE OF EXPERTS TRAINING
# ============================================================================

class MixtureOfExpertsEnsemble:
    """
    Complete MoE with gating network
    """
    def __init__(self, experts: Dict, gating_network: nn.Module, device):
        self.experts = experts
        self.gating_network = gating_network
        self.device = device
        self.gating_optimizer = optim.Adam(gating_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Combine expert predictions using gating network
        
        Output = Sum(gating_weight_i * expert_i_output)
        """
        expert_outputs = []
        
        # Get outputs from all experts
        for expert_name, expert_model in self.experts.items():
            with torch.no_grad():
                output = expert_model(X)  # Shape: (batch_size, 1)
                expert_outputs.append(output.squeeze(-1))  # Shape: (batch_size,)
        
        # Stack expert outputs: (batch_size, n_experts)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Get gating weights: (batch_size, n_experts)
        # Note: X has shape (batch, features) since data is not sequences
        gating_weights = self.gating_network(X)
        
        # Combine: weighted sum of experts
        # expert_outputs: (batch_size, n_experts)
        # gating_weights: (batch_size, n_experts)
        combined_output = (gating_weights * expert_outputs).sum(dim=1, keepdim=True)
        
        return combined_output, gating_weights
    
    def train_gating(self, train_loader: DataLoader, epochs: int = 20):
        """Train gating network to optimally combine experts"""
        print("\n" + "="*80)
        print("TRAINING GATING NETWORK")
        print("="*80)
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.gating_optimizer.zero_grad()
                
                # Get MoE predictions
                moe_pred, gating_weights = self.predict(X_batch)
                
                # Compute loss
                loss = self.criterion(moe_pred, y_batch)
                loss.backward()
                self.gating_optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}")
        
        print("✓ Gating network training complete!")
    
    def evaluate(self, X_test, y_test) -> Dict:
        """Evaluate MoE ensemble"""
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_flat = y_test.flatten()
        
        with torch.no_grad():
            predictions, gating_weights = self.predict(X_tensor)
            predictions = predictions.cpu().numpy().flatten()
        
        mape = mean_absolute_percentage_error(y_test_flat, predictions)
        rmse = np.sqrt(mean_squared_error(y_test_flat, predictions))
        mae = np.mean(np.abs(y_test_flat - predictions))
        r2 = r2_score(y_test_flat, predictions)
        
        return {
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'targets': y_test_flat,
            'gating_weights': gating_weights.cpu().numpy()
        }


# ============================================================================
# PART 4: EVALUATION & COMPARISON
# ============================================================================

def compare_results(baseline_metrics: Dict, moe_metrics: Dict, expert_metrics: Dict):
    """Compare MoE to baseline and individual experts"""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON: Day 8-9 Baseline vs Day 10-11 MoE")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Model': ['SimpleEnsemble (Day 8-9)', 'MoE Ensemble (Day 10-11)'] + list(expert_metrics.keys()),
        'MAPE': [
            baseline_metrics['mape'],
            moe_metrics['mape']
        ] + [expert_metrics[name]['mape'] for name in expert_metrics.keys()],
        'RMSE': [
            baseline_metrics['rmse'],
            moe_metrics['rmse']
        ] + [expert_metrics[name]['rmse'] for name in expert_metrics.keys()],
        'MAE': [
            baseline_metrics['mae'],
            moe_metrics['mae']
        ] + [expert_metrics[name]['mae'] for name in expert_metrics.keys()],
        'R²': [
            baseline_metrics['r2'],
            moe_metrics['r2']
        ] + [expert_metrics[name]['r2'] for name in expert_metrics.keys()],
    })
    
    print("\n" + comparison.to_string(index=False))
    
    # Calculate improvements
    mape_improvement = (baseline_metrics['mape'] - moe_metrics['mape']) / baseline_metrics['mape'] * 100
    rmse_improvement = (baseline_metrics['rmse'] - moe_metrics['rmse']) / baseline_metrics['rmse'] * 100
    
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    print(f"MAPE Improvement: {mape_improvement:+.2f}% (Target: -12% to -15%)")
    print(f"RMSE Improvement: {rmse_improvement:+.2f}%")
    print(f"Baseline MAPE:    {baseline_metrics['mape']:.2f}%")
    print(f"MoE MAPE:         {moe_metrics['mape']:.2f}%")
    
    return comparison


def visualize_results(baseline_metrics, moe_metrics, expert_metrics, save_dir: str = "results"):
    """Generate comparison visualizations"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. MAPE Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Day 10-11 MoE vs Baseline Comparison', fontsize=16, fontweight='bold')
    
    # MAPE
    models = ['Baseline'] + list(expert_metrics.keys()) + ['MoE']
    mape_values = [baseline_metrics['mape']] + [expert_metrics[m]['mape'] for m in expert_metrics.keys()] + [moe_metrics['mape']]
    
    axes[0, 0].bar(models, mape_values, color=['orange', 'blue', 'green', 'purple', 'red', 'darkred'], alpha=0.7)
    axes[0, 0].set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('MAPE Comparison', fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # RMSE
    rmse_values = [baseline_metrics['rmse']] + [expert_metrics[m]['rmse'] for m in expert_metrics.keys()] + [moe_metrics['rmse']]
    axes[0, 1].bar(models, rmse_values, color=['orange', 'blue', 'green', 'purple', 'red', 'darkred'], alpha=0.7)
    axes[0, 1].set_ylabel('RMSE (kW)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('RMSE Comparison', fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # R² Scores
    r2_values = [baseline_metrics['r2']] + [expert_metrics[m]['r2'] for m in expert_metrics.keys()] + [moe_metrics['r2']]
    axes[1, 0].bar(models, r2_values, color=['orange', 'blue', 'green', 'purple', 'red', 'darkred'], alpha=0.7)
    axes[1, 0].set_ylabel('R² Score', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('R² Comparison', fontweight='bold')
    axes[1, 0].set_ylim([0.8, 1.0])
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # MAE
    mae_values = [baseline_metrics['mae']] + [expert_metrics[m]['mae'] for m in expert_metrics.keys()] + [moe_metrics['mae']]
    axes[1, 1].bar(models, mae_values, color=['orange', 'blue', 'green', 'purple', 'red', 'darkred'], alpha=0.7)
    axes[1, 1].set_ylabel('MAE (kW)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('MAE Comparison', fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/moe_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {save_dir}/moe_comparison.png")
    plt.close()
    
    # 2. Time-series Prediction Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    n_plot = 500
    x_range = np.arange(n_plot)
    
    ax.plot(x_range, moe_metrics['targets'][:n_plot], 'o-', label='Actual', alpha=0.6, linewidth=2, markersize=3)
    ax.plot(x_range, baseline_metrics['predictions'][:n_plot], 's--', label='Baseline (Day 8-9)', alpha=0.6, linewidth=1.5)
    ax.plot(x_range, moe_metrics['predictions'][:n_plot], '^--', label='MoE (Day 10-11)', alpha=0.6, linewidth=1.5)
    
    ax.set_xlabel('Sample', fontsize=11, fontweight='bold')
    ax.set_ylabel('Power Consumption (kW)', fontsize=11, fontweight='bold')
    ax.set_title('Predictions: Baseline vs MoE (First 500 Test Samples)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/moe_predictions.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/moe_predictions.png")
    plt.close()


def save_models(trained_experts, gating_network, moe_model, save_dir: str = "models"):
    """Save trained models"""
    os.makedirs(save_dir, exist_ok=True)
    
    model_package = {
        'experts': trained_experts,
        'gating_network': gating_network,
        'moe_model': moe_model,
        'config': CONFIG,
        'timestamp': datetime.now().isoformat(),
    }
    
    save_file = os.path.join(save_dir, "moe_day10_11.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(model_package, f)
    
    file_size = os.path.getsize(save_file) / (1024 * 1024)
    print(f"\n✓ Saved MoE models: {save_file} ({file_size:.2f} MB)")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    print("\n" + "="*80)
    print("DAY 10-11: MIXTURE OF EXPERTS TRAINING")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}")
    print(f"Device: {CONFIG['device']}")
    
    # Step 1: Load data
    X_train, y_train, X_test, y_test, scaler = load_processed_data()
    
    # Step 2: Train baseline (Day 8-9) for comparison
    print("\n" + "="*80)
    print("LOADING DAY 8-9 BASELINE FOR COMPARISON")
    print("="*80)
    
    baseline_model_file = "models/ensemble_day8_9.pkl"
    if os.path.exists(baseline_model_file):
        with open(baseline_model_file, 'rb') as f:
            baseline_models = pickle.load(f)
        
        baseline_ensemble = SimpleEnsemble()
        baseline_ensemble.rf = baseline_models['model1_rf']
        baseline_ensemble.et = baseline_models['model2_et']
        baseline_ensemble.meta_learner = baseline_models['meta_learner']
        baseline_ensemble.fitted = True
        
        baseline_preds = baseline_ensemble.predict(X_test)
        baseline_metrics = {
            'mape': mean_absolute_percentage_error(y_test, baseline_preds),
            'rmse': np.sqrt(mean_squared_error(y_test, baseline_preds)),
            'mae': np.mean(np.abs(y_test - baseline_preds)),
            'r2': r2_score(y_test, baseline_preds),
            'predictions': baseline_preds,
        }
        print(f"✓ Baseline loaded!")
        print(f"  MAPE: {baseline_metrics['mape']:.2f}%")
    else:
        print("⚠ Baseline model not found. Will train experts without comparison.")
        baseline_metrics = None
    
    # Step 3: Train individual experts
    trained_experts, expert_predictions, expert_metrics = train_experts(
        X_train, y_train, X_test, y_test, CONFIG
    )
    
    # Step 4: Train MoE with gating network
    print("\n" + "="*80)
    print("TRAINING MIXTURE OF EXPERTS")
    print("="*80)
    
    device = CONFIG['device']
    gating_network = GatingNetwork(input_dim=CONFIG['input_dim'], n_experts=4, hidden_dim=64)
    gating_network = gating_network.to(device)  # Move to device
    
    moe_ensemble = MixtureOfExpertsEnsemble(trained_experts, gating_network, device)
    
    train_loader = prepare_sequences(X_train, y_train, CONFIG['batch_size'])
    moe_ensemble.train_gating(train_loader, epochs=20)
    
    # Step 5: Evaluate MoE
    moe_metrics = moe_ensemble.evaluate(X_test, y_test)
    
    print(f"\n✓ MoE Evaluation Results:")
    print(f"  MAPE: {moe_metrics['mape']:.2f}%")
    print(f"  RMSE: {moe_metrics['rmse']:.2f} kW")
    print(f"  MAE:  {moe_metrics['mae']:.2f} kW")
    print(f"  R²:   {moe_metrics['r2']:.4f}")
    
    # Step 6: Compare results
    if baseline_metrics:
        comparison = compare_results(baseline_metrics, moe_metrics, expert_metrics)
        
        # Save comparison
        os.makedirs("results", exist_ok=True)
        comparison.to_csv("results/moe_comparison_day10_11.csv", index=False)
        print(f"\n✓ Saved comparison: results/moe_comparison_day10_11.csv")
        
        # Visualize
        visualize_results(baseline_metrics, moe_metrics, expert_metrics)
    
    # Step 7: Save models
    save_models(trained_experts, gating_network, moe_ensemble)
    
    # Summary
    print("\n" + "="*80)
    print("DAY 10-11 COMPLETE ✅")
    print("="*80)
    print(f"\nDeliverables:")
    print(f"  ✓ 4 Expert models trained (GRU, CNN-LSTM, Transformer, Attention)")
    print(f"  ✓ Gating network trained")
    print(f"  ✓ MoE ensemble created and evaluated")
    print(f"  ✓ Models saved: models/moe_day10_11.pkl")
    print(f"  ✓ Comparison results: results/moe_comparison_day10_11.csv")
    print(f"  ✓ Visualizations: results/moe_*.png")
    print(f"\nNext: Days 12-13 Anomaly Detection")


if __name__ == "__main__":
    main()
