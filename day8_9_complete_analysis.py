"""
Day 8-9: Complete Ensemble - Save Models, Feature Importance, Visualizations
All-in-one script: Train → Save → Analyze → Visualize
"""

import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Setup
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10


def load_real_dataset():
    """Load preprocessed real-world dataset."""
    pickle_file = PROJECT_ROOT / 'data' / 'processed' / 'household_power_smartgrid_features.pkl'
    csv_file = PROJECT_ROOT / 'data' / 'processed' / 'household_power_smartgrid_features.csv'
    
    if pickle_file.exists():
        print(f"Loading from {pickle_file.name}...")
        df = pd.read_pickle(pickle_file)
    elif csv_file.exists():
        print(f"Loading from {csv_file.name}...")
        df = pd.read_csv(csv_file)
    else:
        raise FileNotFoundError(f"Run: python prepare_real_dataset.py")
    
    print(f"✓ Loaded {len(df):,} records × {len(df.columns)-1} features")
    return df


def main():
    """Main pipeline: Train → Save → Analyze → Visualize"""
    
    print("\n" + "="*80)
    print("DAY 8-9: COMPLETE ENSEMBLE ANALYSIS")
    print("✓ Train Models  |  ✓ Save Models  |  ✓ Feature Importance  |  ✓ Visualizations")
    print("="*80 + "\n")
    
    # ============================================================
    # STEP 1: Load & Prepare Data
    # ============================================================
    print("-"*80)
    print("STEP 1: Load & Prepare Data")
    print("-"*80)
    
    df = load_real_dataset()
    
    # Sample for faster processing
    df_sample = df.sample(n=100000, random_state=42)
    print(f"✓ Sampled to {len(df_sample):,} records for analysis")
    
    X = df_sample.drop(['timestamp', 'consumption_total'], axis=1).values
    y = df_sample['consumption_total'].values
    feature_names = df_sample.drop(['timestamp', 'consumption_total'], axis=1).columns.tolist()
    
    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Train: {X_train_scaled.shape} | Test: {X_test_scaled.shape}")
    
    # ============================================================
    # STEP 2: Train Models
    # ============================================================
    print("\n" + "-"*80)
    print("STEP 2: Train Ensemble")
    print("-"*80)
    
    print("Training RandomForest...")
    model1 = RandomForestRegressor(
        n_estimators=50, max_depth=10, max_features='sqrt',
        random_state=42, n_jobs=-1, verbose=0
    )
    model1.fit(X_train_scaled, y_train)
    pred1_train = model1.predict(X_train_scaled).reshape(-1, 1)
    pred1_test = model1.predict(X_test_scaled).reshape(-1, 1)
    print(f"  ✓ RF trained")
    
    print("Training ExtraTrees...")
    model2 = ExtraTreesRegressor(
        n_estimators=50, max_depth=10, max_features='sqrt',
        random_state=42, n_jobs=-1, verbose=0
    )
    model2.fit(X_train_scaled, y_train)
    pred2_train = model2.predict(X_train_scaled).reshape(-1, 1)
    pred2_test = model2.predict(X_test_scaled).reshape(-1, 1)
    print(f"  ✓ ExtraTrees trained")
    
    # Meta-learner
    meta_train = np.hstack([pred1_train, pred2_train])
    meta_test = np.hstack([pred1_test, pred2_test])
    
    print("Training Meta-Learner (Ridge)...")
    meta_learner = Ridge(alpha=1.0)
    meta_learner.fit(meta_train, y_train)
    print(f"  ✓ Meta-learner trained")
    
    y_pred = meta_learner.predict(meta_test)
    
    # ============================================================
    # STEP 3: SAVE MODELS
    # ============================================================
    print("\n" + "-"*80)
    print("STEP 3: Save Models")
    print("-"*80)
    
    models = {
        'model1_rf': model1,
        'model2_et': model2,
        'meta_learner': meta_learner,
        'scaler': scaler,
        'feature_names': feature_names
    }
    
    model_path = MODELS_DIR / 'ensemble_day8_9.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(models, f)
    print(f"✓ Models saved: {model_path}")
    print(f"  - RandomForest (50 trees)")
    print(f"  - ExtraTrees (50 trees)")
    print(f"  - Ridge Meta-Learner")
    print(f"  - StandardScaler")
    print(f"  - Feature names ({len(feature_names)} features)")
    
    # ============================================================
    # STEP 4: FEATURE IMPORTANCE ANALYSIS
    # ============================================================
    print("\n" + "-"*80)
    print("STEP 4: Feature Importance Analysis")
    print("-"*80)
    
    # Get importance from both models
    importance_rf = model1.feature_importances_
    importance_et = model2.feature_importances_
    
    # Average importance
    importance_avg = (importance_rf + importance_et) / 2
    
    # Sort
    indices = np.argsort(importance_avg)[::-1]
    
    # Print top 15 features
    print("\n📊 Top 15 Most Important Features:")
    print("─" * 80)
    print(f"{'Rank':<6} {'Feature':<30} {'Importance':<15} {'%':<10}")
    print("─" * 80)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_rf': importance_rf,
        'importance_et': importance_et,
        'importance_avg': importance_avg
    }).sort_values('importance_avg', ascending=False)
    
    for i, (idx, row) in enumerate(importance_df.head(15).iterrows()):
        pct = row['importance_avg'] * 100
        print(f"{i+1:<6} {row['feature']:<30} {row['importance_avg']:<15.6f} {pct:<10.2f}%")
    
    # Save feature importance
    importance_df.to_csv(RESULTS_DIR / 'feature_importance.csv', index=False)
    print(f"\n✓ Feature importance saved: {RESULTS_DIR}/feature_importance.csv")
    
    # ============================================================
    # STEP 5: METRICS
    # ============================================================
    print("\n" + "-"*80)
    print("STEP 5: Model Metrics")
    print("-"*80)
    
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n✓ Ensemble Performance:")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")
    
    # ============================================================
    # STEP 6: VISUALIZATIONS
    # ============================================================
    print("\n" + "-"*80)
    print("STEP 6: Generate Visualizations")
    print("-"*80)
    
    # 1. Actual vs Predicted (Time Series)
    print("Creating plot 1/4: Actual vs Predicted...")
    fig, ax = plt.subplots(figsize=(16, 6))
    time_idx = np.arange(len(y_test[:2000]))
    ax.plot(time_idx, y_test[:2000], 'b-', label='Actual', linewidth=2, alpha=0.7)
    ax.plot(time_idx, y_pred[:2000], 'r--', label='Predicted', linewidth=2, alpha=0.7)
    ax.fill_between(time_idx, y_test[:2000], y_pred[:2000], alpha=0.2, color='gray')
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Consumption (kW)')
    ax.set_title('Smart Grid Energy Consumption: Actual vs Predicted (First 2000 samples)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot1_path = RESULTS_DIR / 'plot_1_actual_vs_predicted.png'
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot1_path}")
    
    # 2. Error Distribution
    print("Creating plot 2/4: Error Distribution...")
    errors = y_test - y_pred
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.hist(errors, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label=f'Mean Error: {np.mean(errors):.2f}')
    ax.axvline(x=np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median Error: {np.median(errors):.2f}')
    ax.set_xlabel('Prediction Error (kW)')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plot2_path = RESULTS_DIR / 'plot_2_error_distribution.png'
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot2_path}")
    
    # 3. Actual vs Predicted (Scatter)
    print("Creating plot 3/4: Residuals & Scatter...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot
    ax1.scatter(y_test, y_pred, alpha=0.5, s=10, color='blue')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Consumption (kW)')
    ax1.set_ylabel('Predicted Consumption (kW)')
    ax1.set_title(f'Actual vs Predicted (R² = {r2:.4f})', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals plot
    ax2.scatter(y_pred, errors, alpha=0.5, s=10, color='green')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Consumption (kW)')
    ax2.set_ylabel('Residuals (Actual - Predicted)')
    ax2.set_title('Residuals Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot3_path = RESULTS_DIR / 'plot_3_scatter_residuals.png'
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot3_path}")
    
    # 4. Feature Importance (Top 15)
    print("Creating plot 4/4: Feature Importance...")
    fig, ax = plt.subplots(figsize=(12, 8))
    top_n = 15
    top_features = importance_df.head(top_n)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    bars = ax.barh(range(len(top_features)), top_features['importance_avg'].values, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features (Averaged RF + ExtraTrees)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (idx, bar) in enumerate(zip(top_features.index, bars)):
        val = top_features.loc[idx, 'importance_avg']
        ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.4f} ({val*100:.2f}%)', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    plot4_path = RESULTS_DIR / 'plot_4_feature_importance.png'
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot4_path}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*80)
    print("✅ COMPLETE ANALYSIS FINISHED!")
    print("="*80)
    print(f"""
📁 SAVED ARTIFACTS:

1️⃣  TRAINED MODELS (Ready for production):
    • {model_path}
    • Contains: RF, ExtraTrees, Ridge, Scaler, Feature names
    • Load with: pickle.load(open('{model_path}', 'rb'))

2️⃣  FEATURE IMPORTANCE:
    • {RESULTS_DIR}/feature_importance.csv
    • Shows which features matter most
    • Top 3: {importance_df.iloc[0:3]['feature'].tolist()}

3️⃣  VISUALIZATIONS (Publication-quality PNG):
    • {plot1_path}
      → Time-series: Actual vs Predicted
    • {plot2_path}
      → Error distribution histogram
    • {plot3_path}
      → Scatter plot + residuals
    • {plot4_path}
      → Feature importance bars

📊 MODEL PERFORMANCE:
   • MAPE: {mape:.2f}%  {'✅ TARGET!' if mape < 8 else ''}
   • RMSE: {rmse:.2f}
   • MAE: {mae:.2f}
   • R²: {r2:.4f}

🔑 KEY INSIGHTS:
   • Model explains {r2*100:.2f}% of variance
   • Average error: ±{mae:.2f} kW
   • Worst error: {np.max(np.abs(errors)):.2f} kW
   
🚀 NEXT STEPS:
   Day 10-11: Implement Mixture of Experts
   - Use these saved models as building blocks
   - Create specialized experts for different patterns
   - Implement gating network for dynamic selection

""")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
