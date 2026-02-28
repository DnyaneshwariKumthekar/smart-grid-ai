"""
Day 8-9: Simplified StackingEnsemble Implementation and Testing
Minimal dependencies version for reliable execution.
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_loader import generate_synthetic_data, preprocess_data, get_data_stats

# Simple XGBoost replacement using sklearn's GradientBoosting
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold


def main():
    """Main training pipeline for Day 8-9 - Simplified Version."""
    
    print("\n" + "="*70)
    print("SMART GRID ENSEMBLE - DAY 8-9 IMPLEMENTATION (SIMPLIFIED)")
    print("="*70)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    print(f"🔧 PyTorch version: {torch.__version__}")
    
    # ============================================================
    # STEP 1: Data Generation
    # ============================================================
    print("\n" + "-"*70)
    print("STEP 1: Data Generation")
    print("-"*70)
    
    print("Generating synthetic data (50,000 samples for testing)...")
    df = generate_synthetic_data(n_samples=50000, n_features=32)
    
    stats = get_data_stats(df)
    print(f"\n✓ Dataset Statistics:")
    print(f"  Shape: {stats['shape']}")
    print(f"  Missing: {stats['missing_percentage']:.4f}%")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # ============================================================
    # STEP 2: Data Preprocessing
    # ============================================================
    print("\n" + "-"*70)
    print("STEP 2: Data Preprocessing")
    print("-"*70)
    
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df, test_size=0.2)
    
    print(f"\n✓ Preprocessing Complete:")
    print(f"  Training set: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"  Test set:     X_test={X_test.shape}, y_test={y_test.shape}")
    
    # ============================================================
    # STEP 3: Generate Meta-Features Using Ensemble
    # ============================================================
    print("\n" + "-"*70)
    print("STEP 3: Training Base Models & Generating Meta-Features")
    print("-"*70)
    
    print("\nUsing simplified approach with GradientBoosting...")
    
    # Flatten sequences for simpler model
    n_samples, seq_len, n_features = X_train.shape
    X_train_flat = X_train.reshape(n_samples, seq_len * n_features)
    X_test_flat = X_test.reshape(X_test.shape[0], seq_len * n_features)
    
    print(f"  Flattened shape: {X_train_flat.shape}")
    
    # Train base models with K-fold for meta-feature generation
    n_splits = 3
    kfold = KFold(n_splits=n_splits, shuffle=False)
    meta_features_train = np.zeros((X_train_flat.shape[0], 2))  # 2 base models
    
    print(f"  Training with {n_splits}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_flat)):
        print(f"    Fold {fold+1}/{n_splits}...")
        
        X_fold_train, X_fold_val = X_train_flat[train_idx], X_train_flat[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Model 1: GradientBoosting
        model1 = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42)
        model1.fit(X_fold_train, y_fold_train.flatten())
        meta_features_train[val_idx, 0] = model1.predict(X_fold_val)
        
        # Model 2: GradientBoosting with different params
        model2 = GradientBoostingRegressor(n_estimators=50, learning_rate=0.05, max_depth=7, random_state=42)
        model2.fit(X_fold_train, y_fold_train.flatten())
        meta_features_train[val_idx, 1] = model2.predict(X_fold_val)
    
    print("  ✓ Meta-features generated")
    
    # ============================================================
    # STEP 4: Train Meta-Learner
    # ============================================================
    print("\n" + "-"*70)
    print("STEP 4: Training Meta-Learner")
    print("-"*70)
    
    print("  Training XGBoost replacement (GradientBoosting meta-learner)...")
    meta_learner = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    meta_learner.fit(meta_features_train, y_train.flatten())
    print("  ✓ Meta-learner trained")
    
    # ============================================================
    # STEP 5: Evaluation on Test Set
    # ============================================================
    print("\n" + "-"*70)
    print("STEP 5: Evaluation on Test Set")
    print("-"*70)
    
    # Generate test meta-features
    model1_test = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42)
    model1_test.fit(X_train_flat, y_train.flatten())
    model1_test_pred = model1_test.predict(X_test_flat)
    
    model2_test = GradientBoostingRegressor(n_estimators=50, learning_rate=0.05, max_depth=7, random_state=42)
    model2_test.fit(X_train_flat, y_train.flatten())
    model2_test_pred = model2_test.predict(X_test_flat)
    
    meta_features_test = np.column_stack([model1_test_pred, model2_test_pred])
    
    # Final predictions
    y_pred = meta_learner.predict(meta_features_test)
    y_pred = y_pred.reshape(-1, 1)
    
    # Calculate metrics
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n✓ Test Set Metrics:")
    print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    print(f"  RMSE (Root Mean Squared Error):        {rmse:.4f}")
    print(f"  MAE (Mean Absolute Error):             {mae:.4f}")
    print(f"  R² Score:                              {r2:.4f}")
    
    # Check success criteria
    print(f"\n📊 Success Criteria (Day 8-9):")
    if mape < 8.0:
        print(f"  ✓ MAPE < 8.0%: {mape:.2f}% (TARGET ACHIEVED)")
    else:
        print(f"  ✗ MAPE < 8.0%: {mape:.2f}% (needs improvement)")
    
    # ============================================================
    # STEP 6: Save Results
    # ============================================================
    print("\n" + "-"*70)
    print("STEP 6: Save Results")
    print("-"*70)
    
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'MAPE': [mape],
        'RMSE': [rmse],
        'MAE': [mae],
        'R2': [r2]
    }, index=['GradientBoostingEnsemble'])
    
    metrics_path = results_dir / 'day8_9_metrics.csv'
    metrics_df.to_csv(metrics_path)
    print(f"  ✓ Metrics saved: {metrics_path}")
    
    # Save sample predictions
    preds_df = pd.DataFrame({
        'actual': y_test[:100].flatten(),
        'predicted': y_pred[:100].flatten(),
        'error': (y_test[:100].flatten() - y_pred[:100].flatten()),
        'mape_sample': (np.abs(y_test[:100].flatten() - y_pred[:100].flatten()) / 
                       (np.abs(y_test[:100].flatten()) + 1e-6) * 100)
    })
    
    preds_path = results_dir / 'day8_9_predictions_sample.csv'
    preds_df.to_csv(preds_path, index=False)
    print(f"  ✓ Predictions saved: {preds_path}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("SUMMARY - DAY 8-9 COMPLETE")
    print("="*70)
    print(f"""
✓ Ensemble Implementation Status: COMPLETE (Simplified Version)

Completed Tasks:
  ✓ Data generation (50k samples, 32 features)
  ✓ Data preprocessing and normalization
  ✓ Sequence creation (288-timestep windows)
  ✓ Base model training (2 GradientBoosting models)
  ✓ K-fold meta-feature generation
  ✓ Meta-learner training
  ✓ Test set evaluation

Results:
  • MAPE: {mape:.2f}% (Target < 8.0%)
  • RMSE: {rmse:.4f}
  • MAE:  {mae:.4f}
  • R²:   {r2:.4f}

Output Files:
  • results/day8_9_metrics.csv
  • results/day8_9_predictions_sample.csv

Ready for Next Phase:
  → If MAPE < 8%, proceed to Days 10-11 (MixtureOfExperts)
  → If MAPE >= 8%, adjust hyperparameters and retry

Notes:
  • This simplified version uses GradientBoosting instead of XGBoost
  • Ensures all dependencies are readily available
  • Demonstrates complete ensemble workflow
  • Production version (full_training.py) will use PyTorch + XGBoost
""")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
