"""
Day 8-9: StackingEnsemble Implementation and Testing
Main training script to verify the ensemble works correctly.
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_loader import generate_synthetic_data, preprocess_data, get_data_stats
from models.ensemble import StackingEnsemble


def main():
    """Main training pipeline for Day 8-9."""
    
    print("\n" + "="*70)
    print("SMART GRID ENSEMBLE - DAY 8-9 IMPLEMENTATION")
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
    
    df = generate_synthetic_data(n_samples=100000, n_features=32)
    
    stats = get_data_stats(df)
    print(f"\n✓ Dataset Statistics:")
    print(f"  Shape: {stats['shape']}")
    print(f"  Missing: {stats['missing_percentage']:.4f}%")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Features: 32 (consumption, generation, weather, time-based, system, derived)")
    
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
    print(f"  Sequence length: 288 (24-hour lookback)")
    print(f"  Normalization: StandardScaler (fitted on training data)")
    
    # ============================================================
    # STEP 3: Train StackingEnsemble
    # ============================================================
    print("\n" + "-"*70)
    print("STEP 3: Training StackingEnsemble")
    print("-"*70)
    print("\nArchitecture:")
    print("  Base Model 1: LSTM (hidden=64, layers=2)")
    print("  Base Model 2: Transformer (d_model=64, heads=4, layers=2)")
    print("  Meta-Learner: XGBoost (n_estimators=200, max_depth=6)")
    print("  K-Fold CV: 5 folds for meta-feature generation")
    
    ensemble = StackingEnsemble(
        lstm_hidden=64,
        transformer_d_model=64,
        n_splits=5
    )
    
    ensemble.fit(X_train, y_train, epochs=20, verbose=True)
    
    # ============================================================
    # STEP 4: Evaluation on Test Set
    # ============================================================
    print("\n" + "-"*70)
    print("STEP 4: Evaluation on Test Set")
    print("-"*70)
    
    metrics = ensemble.evaluate(X_test, y_test)
    
    print(f"\n✓ Test Set Metrics:")
    print(f"  MAPE (Mean Absolute Percentage Error): {metrics['MAPE']:.2f}%")
    print(f"  RMSE (Root Mean Squared Error):        {metrics['RMSE']:.4f}")
    print(f"  MAE (Mean Absolute Error):             {metrics['MAE']:.4f}")
    print(f"  R² Score:                              {metrics['R2']:.4f}")
    
    # Check success criteria
    print(f"\n📊 Success Criteria (Day 8-9):")
    if metrics['MAPE'] < 8.0:
        print(f"  ✓ MAPE < 8.0%: {metrics['MAPE']:.2f}% (TARGET ACHIEVED)")
    else:
        print(f"  ✗ MAPE < 8.0%: {metrics['MAPE']:.2f}% (needs improvement)")
    
    # Target is < 6% for final ensemble
    if metrics['MAPE'] < 6.0:
        print(f"  ✓ BONUS: MAPE < 6.0%: {metrics['MAPE']:.2f}% (EXCEEDS EXPECTATIONS)")
    else:
        print(f"  → Target for final ensemble: MAPE < 6.0% (currently {metrics['MAPE']:.2f}%)")
    
    # ============================================================
    # STEP 5: Save Results
    # ============================================================
    print("\n" + "-"*70)
    print("STEP 5: Save Results")
    print("-"*70)
    
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics], index=['StackingEnsemble'])
    metrics_path = results_dir / 'day8_9_metrics.csv'
    metrics_df.to_csv(metrics_path)
    print(f"  ✓ Metrics saved: {metrics_path}")
    
    # Save sample predictions
    y_pred = ensemble.predict(X_test[:100])
    preds_df = pd.DataFrame({
        'actual': y_test[:100].flatten(),
        'predicted': y_pred.flatten(),
        'error': (y_test[:100].flatten() - y_pred.flatten()),
        'mape_sample': (np.abs(y_test[:100].flatten() - y_pred.flatten()) / 
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
✓ StackingEnsemble Implementation Status: COMPLETE

Completed Tasks:
  ✓ Data generation (100k samples, 32 features)
  ✓ Data preprocessing and normalization
  ✓ Sequence creation (288-timestep windows)
  ✓ LSTM base model training
  ✓ Transformer base model training
  ✓ K-fold meta-feature generation
  ✓ XGBoost meta-learner training
  ✓ Test set evaluation

Results:
  • MAPE: {metrics['MAPE']:.2f}% (Target < 8.0%)
  • RMSE: {metrics['RMSE']:.4f}
  • MAE:  {metrics['MAE']:.4f}
  • R²:   {metrics['R2']:.4f}

Next Steps (Day 10-11):
  ▶ Implement MixtureOfExperts with 3 specialist networks
  ▶ Add gating mechanism for dynamic expert selection
  ▶ Implement load balancing loss to prevent expert collapse
  ▶ Target: MAPE < 5% with MoE

Files Created:
  • models/ensemble.py (StackingEnsemble class)
  • data_loader.py (Data preprocessing pipeline)
  • train_day8_9.py (This training script)
  • results/day8_9_metrics.csv
  • results/day8_9_predictions_sample.csv
""")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
