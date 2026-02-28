# 📊 DATASET & OUTPUTS SPECIFICATION

**Version**: 1.0  
**Last Updated**: January 28, 2026

---

## 📦 INPUT DATASET: Smart Grid Energy Data

### Overview
```
File: data/raw/smart_grid_2years.csv
Size: ~50 MB
Rows: 525,600 samples
Columns: 32 features + 1 timestamp
Time Span: 2 years (Jan 2022 - Dec 2023)
Frequency: 5-minute intervals
Missing Values: < 0.1%
Outliers: 5-7% marked as anomalies
```

### Loading the Dataset

```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('data/raw/smart_grid_2years.csv')

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Output:
# Shape: (525600, 33)
# Date range: 2022-01-01 00:00:00 to 2023-12-31 23:55:00
```

---

## 🔢 FEATURES: 32 Total

### Raw Features (12)

| # | Feature | Unit | Range | Description |
|---|---------|------|-------|-------------|
| 1 | `consumption` | kWh | 100-2000 | Total grid consumption |
| 2 | `solar_generation` | kWh | 0-1500 | Solar panel generation |
| 3 | `wind_generation` | kWh | 0-800 | Wind turbine generation |
| 4 | `temperature` | °C | -20 to 45 | Ambient temperature |
| 5 | `humidity` | % | 20-100 | Relative humidity |
| 6 | `voltage` | V | 210-250 | Grid voltage |
| 7 | `frequency` | Hz | 49.5-50.5 | Grid frequency |
| 8 | `load_shedding` | Binary | 0/1 | Emergency load shedding active |
| 9 | `demand_response` | Binary | 0/1 | Demand response program active |
| 10 | `grid_stability_index` | Score | -10 to 10 | Grid stability metric |
| 11 | `renewable_percentage` | % | 0-100 | % of generation from renewable |
| 12 | `peak_demand` | kWh | 500-2500 | Peak demand in last hour |

### Engineered Features (20)

#### Lagged Features (4)
```python
# Previous timesteps
consumption_lag_1      # consumption at t-1 (5 min ago)
consumption_lag_2      # consumption at t-2 (10 min ago)
consumption_lag_288    # consumption at t-288 (24 hours ago)
consumption_lag_1008   # consumption at t-1008 (7 days ago)
```

#### Moving Averages (4)
```python
# Rolling windows
consumption_ma_7       # 7-timestep moving average (~35 min)
consumption_ma_288     # 288-timestep moving average (24 hours)
consumption_ma_2016    # 2016-timestep moving average (7 days)
consumption_ma_8640    # 8640-timestep moving average (30 days)
```

#### Cyclical Encodings (8)
```python
# Time-based patterns (encoded as sin/cos pairs)
hour_sin               # sin(2π * hour / 24)
hour_cos               # cos(2π * hour / 24)
day_sin                # sin(2π * day / 7)
day_cos                # cos(2π * day / 7)
month_sin              # sin(2π * month / 12)
month_cos              # cos(2π * month / 12)
quarter_sin            # sin(2π * quarter / 4)
quarter_cos            # cos(2π * quarter / 4)
```

#### Interaction Features (4)
```python
# Interaction terms
renewable_consumption    # renewable_percentage × consumption
temperature_consumption  # temperature × consumption
humidity_consumption     # humidity × consumption
demand_supply_gap        # (consumption - generation) / consumption
```

---

## 📝 DATA PREPROCESSING CODE

### Step 1: Load and Explore

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/raw/smart_grid_2years.csv')

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sort by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

# Check for missing values
print(f"Missing values:\n{df.isnull().sum()}")

# Handle missing (forward fill, then backward fill)
df = df.fillna(method='ffill').fillna(method='bfill')

print(f"✓ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
```

### Step 2: Feature Engineering

```python
def engineer_features(df):
    """Add engineered features"""
    df_feat = df.copy()
    
    # Lagged features
    df_feat['consumption_lag_1'] = df_feat['consumption'].shift(1)
    df_feat['consumption_lag_2'] = df_feat['consumption'].shift(2)
    df_feat['consumption_lag_288'] = df_feat['consumption'].shift(288)  # 24h
    df_feat['consumption_lag_1008'] = df_feat['consumption'].shift(1008)  # 7d
    
    # Moving averages
    df_feat['consumption_ma_7'] = df_feat['consumption'].rolling(7).mean()
    df_feat['consumption_ma_288'] = df_feat['consumption'].rolling(288).mean()
    df_feat['consumption_ma_2016'] = df_feat['consumption'].rolling(2016).mean()
    df_feat['consumption_ma_8640'] = df_feat['consumption'].rolling(8640).mean()
    
    # Cyclical features
    df_feat['hour'] = df_feat['timestamp'].dt.hour
    df_feat['day'] = df_feat['timestamp'].dt.dayofweek
    df_feat['month'] = df_feat['timestamp'].dt.month
    df_feat['quarter'] = df_feat['timestamp'].dt.quarter
    
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
    df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['day'] / 7)
    df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['day'] / 7)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    df_feat['quarter_sin'] = np.sin(2 * np.pi * df_feat['quarter'] / 4)
    df_feat['quarter_cos'] = np.cos(2 * np.pi * df_feat['quarter'] / 4)
    
    # Interaction features
    df_feat['renewable_consumption'] = (df_feat['renewable_percentage'] * 
                                        df_feat['consumption'] / 100)
    df_feat['temperature_consumption'] = (df_feat['temperature'] * 
                                         df_feat['consumption'] / 100)
    df_feat['humidity_consumption'] = (df_feat['humidity'] * 
                                      df_feat['consumption'] / 100)
    
    generation = df_feat['solar_generation'] + df_feat['wind_generation']
    df_feat['demand_supply_gap'] = (df_feat['consumption'] - generation) / (df_feat['consumption'] + 1e-6)
    
    # Fill NaN from rolling windows
    df_feat = df_feat.fillna(method='bfill')
    
    return df_feat

df = engineer_features(df)
print(f"✓ Features engineered: {df.shape[1]} columns")
```

### Step 3: Train/Val/Test Split

```python
def create_train_val_test_split(df, train_ratio=0.6, val_ratio=0.2):
    """
    Time-series aware split (no data leakage)
    
    Train: First 60% (315,360 samples)
    Val: Next 20% (105,120 samples)
    Test: Last 20% (105,120 samples)
    """
    n = len(df)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"✓ Train: {len(train_df):,} ({train_ratio*100:.0f}%)")
    print(f"✓ Val:   {len(val_df):,} ({val_ratio*100:.0f}%)")
    print(f"✓ Test:  {len(test_df):,} ({val_ratio*100:.0f}%)")
    
    return train_df, val_df, test_df

train_df, val_df, test_df = create_train_val_test_split(df)
```

### Step 4: Normalization

```python
def normalize_data(train_df, val_df, test_df):
    """
    Normalize using StandardScaler
    Fit ONLY on training data to prevent data leakage
    """
    
    # Select feature columns (exclude timestamp, target)
    feature_cols = [col for col in train_df.columns 
                   if col not in ['timestamp', 'consumption', 'is_anomaly']]
    
    scaler = StandardScaler()
    
    # Fit only on training data
    scaler.fit(train_df[feature_cols])
    
    # Transform all splits
    train_scaled = scaler.transform(train_df[feature_cols])
    val_scaled = scaler.transform(val_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])
    
    # Convert back to dataframes
    train_scaled_df = pd.DataFrame(train_scaled, columns=feature_cols)
    val_scaled_df = pd.DataFrame(val_scaled, columns=feature_cols)
    test_scaled_df = pd.DataFrame(test_scaled, columns=feature_cols)
    
    print(f"✓ Data normalized using StandardScaler")
    print(f"  Train mean: {train_scaled_df.mean().mean():.6f} ≈ 0")
    print(f"  Train std:  {train_scaled_df.std().mean():.6f} ≈ 1")
    
    return train_scaled_df, val_scaled_df, test_scaled_df, scaler

train_scaled, val_scaled, test_scaled, scaler = normalize_data(train_df, val_df, test_df)
```

### Step 5: Sequence Creation

```python
def create_sequences(X, y, seq_length=288):
    """
    Create sliding window sequences
    
    seq_length = 288 timesteps = 24 hours (at 5-min intervals)
    
    Input shape: (n_samples, seq_length, n_features)
    Output: scalar (next timestep consumption)
    """
    sequences_X = []
    sequences_y = []
    
    for i in range(len(X) - seq_length):
        # Window of 288 timesteps
        sequences_X.append(X[i:i + seq_length])
        # Next timestep target
        sequences_y.append(y[i + seq_length])
    
    return np.array(sequences_X), np.array(sequences_y)

# Get target variable
feature_cols = train_scaled.columns.tolist()
y_train = train_df['consumption'].values
y_val = val_df['consumption'].values
y_test = test_df['consumption'].values

# Create sequences
X_train_seq, y_train_seq = create_sequences(train_scaled.values, y_train, seq_length=288)
X_val_seq, y_val_seq = create_sequences(val_scaled.values, y_val, seq_length=288)
X_test_seq, y_test_seq = create_sequences(test_scaled.values, y_test, seq_length=288)

print(f"✓ Sequences created:")
print(f"  X_train: {X_train_seq.shape} (sequences, timesteps, features)")
print(f"  X_val:   {X_val_seq.shape}")
print(f"  X_test:  {X_test_seq.shape}")
```

---

## 📤 OUTPUT SPECIFICATIONS: 8 Types

### OUTPUT 1: Performance Metrics (JSON)

**File**: `results/metrics.json`

```json
{
  "dataset_info": {
    "total_samples": 525600,
    "train_samples": 315360,
    "val_samples": 105120,
    "test_samples": 105120,
    "n_features": 32,
    "date_range": "2022-01-01 to 2023-12-31"
  },
  "model_performance": {
    "lstm": {
      "mape": 8.7,
      "rmse": 78.5,
      "mae": 35.2,
      "r2_score": 0.876,
      "training_time_minutes": 45
    },
    "transformer": {
      "mape": 7.6,
      "rmse": 72.1,
      "mae": 32.1,
      "r2_score": 0.901,
      "training_time_minutes": 52
    },
    "stacking_ensemble": {
      "mape": 4.2,
      "rmse": 52.3,
      "mae": 23.5,
      "r2_score": 0.954,
      "training_time_minutes": 18
    },
    "mixture_of_experts": {
      "mape": 5.1,
      "rmse": 58.7,
      "mae": 26.4,
      "r2_score": 0.938,
      "training_time_minutes": 38
    }
  },
  "anomaly_detection": {
    "isolation_forest": {
      "precision": 0.88,
      "recall": 0.80,
      "f1_score": 0.84,
      "roc_auc": 0.881
    },
    "one_class_svm": {
      "precision": 0.85,
      "recall": 0.82,
      "f1_score": 0.83,
      "roc_auc": 0.894
    },
    "autoencoder": {
      "precision": 0.82,
      "recall": 0.84,
      "f1_score": 0.83,
      "roc_auc": 0.876
    },
    "ensemble_voting": {
      "precision": 0.90,
      "recall": 0.85,
      "f1_score": 0.904,
      "roc_auc": 0.922
    }
  },
  "benchmarking": {
    "arima_mape": 12.0,
    "prophet_mape": 10.5,
    "xgboost_mape": 7.2,
    "lightgbm_mape": 6.8,
    "your_ensemble_mape": 4.2,
    "improvement_over_arima": "65%",
    "statistical_significance": "p < 0.001"
  }
}
```

---

### OUTPUT 2: Test Predictions (CSV)

**File**: `results/test_predictions.csv`

```
105,120 rows × 7 columns

timestamp,actual_consumption,lstm_prediction,transformer_prediction,ensemble_prediction,ensemble_confidence,is_anomaly
2023-10-01 00:00:00,1250.5,1248.2,1251.3,1249.8,0.92,0
2023-10-01 00:05:00,1245.3,1243.8,1246.1,1245.2,0.94,0
2023-10-01 00:10:00,1248.7,1250.1,1248.9,1249.2,0.91,0
...
2023-12-31 23:55:00,1180.2,1182.5,1179.8,1180.9,0.89,0
```

**Columns:**
- `timestamp`: UTC timestamp
- `actual_consumption`: Ground truth kWh
- `lstm_prediction`: LSTM model prediction
- `transformer_prediction`: Transformer model prediction
- `ensemble_prediction`: Stacking ensemble prediction
- `ensemble_confidence`: Prediction confidence (0-1)
- `is_anomaly`: Binary anomaly flag

---

### OUTPUT 3: Anomaly Detection (CSV)

**File**: `results/anomaly_detection.csv`

```
105,120 rows × 9 columns

timestamp,actual_consumption,iso_forest_score,one_class_svm_score,autoencoder_score,ensemble_anomaly_score,is_anomaly_ensemble,is_anomaly_ground_truth,anomaly_type
2023-10-01 00:00:00,1250.5,0.12,0.15,0.10,0.12,0,0,normal
2023-10-01 00:05:00,1245.3,0.11,0.14,0.11,0.12,0,0,normal
...
2023-10-05 14:30:00,850.2,0.78,0.82,0.75,0.78,1,1,load_shedding
2023-10-07 09:15:00,2100.5,0.81,0.85,0.79,0.82,1,1,peak_overload
...
2023-12-31 23:55:00,1180.2,0.08,0.10,0.09,0.09,0,0,normal
```

**Columns:**
- `timestamp`: UTC timestamp
- `actual_consumption`: Ground truth kWh
- `iso_forest_score`: Isolation Forest anomaly score (0-1)
- `one_class_svm_score`: One-Class SVM anomaly score (0-1)
- `autoencoder_score`: Autoencoder anomaly score (0-1)
- `ensemble_anomaly_score`: Ensemble voting score (0-1)
- `is_anomaly_ensemble`: Binary prediction (0=normal, 1=anomaly)
- `is_anomaly_ground_truth`: True label (if available)
- `anomaly_type`: Category (normal, load_shedding, peak_overload, frequency_deviation, etc.)

---

### OUTPUT 4: Visualizations (11 PNG Files)

**Directory**: `results/visualizations/`

#### Visualization 1-3: 30-Day Forecasts
```
forecast_lstm_30day.png
  ├─ X-axis: Days (1-30)
  ├─ Y-axis: Consumption (kWh)
  ├─ Line 1: Actual values (black)
  ├─ Line 2: LSTM predictions (blue)
  └─ Confidence interval (light blue band)

forecast_transformer_30day.png
  └─ Same as above for Transformer

forecast_ensemble_30day.png
  ├─ Actual (black)
  ├─ LSTM (blue)
  ├─ Transformer (green)
  ├─ Ensemble (red)
  └─ Best visual: Ensemble closest to actual
```

#### Visualization 4-5: ROC Curves
```
roc_curve_train.png
  ├─ X-axis: False Positive Rate
  ├─ Y-axis: True Positive Rate
  ├─ Line 1: Isolation Forest (AUC: 0.881)
  ├─ Line 2: One-Class SVM (AUC: 0.894)
  ├─ Line 3: Autoencoder (AUC: 0.876)
  ├─ Line 4: Ensemble (AUC: 0.922) ← Best
  └─ Diagonal: Random classifier (AUC: 0.5)

roc_curve_test.png
  └─ Same curves on test set
```

#### Visualization 6-7: Learning Curves
```
learning_curve_lstm.png
  ├─ X-axis: Training samples
  ├─ Y-axis: Error (MAPE)
  ├─ Line 1: Training error (decreasing)
  └─ Line 2: Validation error (plateau)

learning_curve_ensemble.png
  └─ Same for stacking ensemble
```

#### Visualization 8: Attention Heatmap
```
attention_heatmap_lstm.png
  ├─ X-axis: 288 timesteps (24 hours)
  ├─ Y-axis: 32 input features
  ├─ Color: Attention weight intensity (0-1)
  └─ Pattern: Should show high attention on recent timesteps
```

#### Visualization 9: Feature Importance
```
feature_importance.png
  ├─ Bar chart: Top 15 features by importance
  ├─ Top features expected: consumption lags, moving averages, hour_sin, etc.
  └─ Sorted by XGBoost importance
```

#### Visualization 10: Ensemble Comparison
```
ensemble_component_comparison.png
  ├─ Bar chart: MAPE by model
  ├─ LSTM: 8.7%
  ├─ Transformer: 7.6%
  ├─ MoE: 5.1%
  └─ Ensemble: 4.2% ← Best
```

#### Visualization 11: Anomaly Confusion Matrix
```
anomaly_confusion_matrix.png
  └─ 2×2 grid:
      True Neg (89,234) | False Pos (215)
      False Neg (725)   | True Pos (14,946)
      
      Accuracy: 99.1%
      Precision: 0.90
      Recall: 0.85
      F1: 0.904
```

---

### OUTPUT 5: Training Curves (4 PNG Files)

**Directory**: `results/visualizations/`

```
training_curve_lstm.png
  ├─ X-axis: Epochs (0-100)
  ├─ Y-axis: Loss (MSE)
  ├─ Line 1: Training loss (decreasing)
  ├─ Line 2: Validation loss (plateau at epoch ~60)
  └─ Marker: Early stopping at epoch 65

training_curve_transformer.png
training_curve_stacking_ensemble.png
training_curve_mixture_of_experts.png
  └─ Same format for each model
```

---

### OUTPUT 6: Jupyter Notebooks (5 Files)

**Directory**: `notebooks/`

```
01_data_exploration.ipynb (150 lines)
  └─ Statistics, distributions, anomaly rate

02_model_training.ipynb (200 lines)
  └─ Train LSTM and Transformer

03_ensemble_analysis.ipynb (180 lines)
  └─ Ensemble methods and comparisons

04_anomaly_detection.ipynb (160 lines)
  └─ Anomaly detection with 3 methods

05_final_evaluation.ipynb (140 lines)
  └─ Comprehensive results and benchmarking
```

---

### OUTPUT 7: Technical Report (PDF)

**File**: `report.pdf` (8-12 pages)

Structure:
1. Executive Summary (1 page)
2. Introduction (1.5 pages)
3. Methodology (2.5 pages)
4. Experiments (2 pages)
5. Results (1.5 pages)
6. Analysis (1 page)
7. Conclusion (0.5 pages)
8. Appendix

---

### OUTPUT 8: Trained Models (6 Files)

**Directory**: `results/models/`

```
lstm_model.pth (5.2 MB)
  └─ PyTorch model weights

transformer_model.pth (4.8 MB)
  └─ PyTorch model weights

stacking_ensemble.pkl (2.1 MB)
  └─ Scikit-learn XGBoost meta-learner

mixture_of_experts.pth (8.5 MB)
  └─ PyTorch MoE model with 3 experts

anomaly_detector.pkl (1.2 MB)
  └─ Ensemble: IsoForest + SVM + Autoencoder

scaler.pkl (0.5 MB)
  └─ StandardScaler for feature normalization
```

---

## ✅ DELIVERABLE CHECKLIST

After completing your project, verify you have all 8 outputs:

```
OUTPUT 1: Performance Metrics
  ☐ results/metrics.json exists
  ☐ Has all 8 required fields
  ☐ Numbers match specifications

OUTPUT 2: Test Predictions
  ☐ results/test_predictions.csv exists
  ☐ 105,120 rows × 7 columns
  ☐ No NaN values
  ☐ Timestamps in order

OUTPUT 3: Anomaly Detection
  ☐ results/anomaly_detection.csv exists
  ☐ 105,120 rows × 9 columns
  ☐ Scores between 0-1

OUTPUT 4: Visualizations (11 PNG)
  ☐ 3 × 30-day forecasts
  ☐ 2 × ROC curves
  ☐ 2 × Learning curves
  ☐ 1 × Attention heatmap
  ☐ 1 × Feature importance
  ☐ 1 × Ensemble comparison
  ☐ 1 × Anomaly confusion matrix

OUTPUT 5: Training Curves (4 PNG)
  ☐ lstm_curve.png
  ☐ transformer_curve.png
  ☐ stacking_curve.png
  ☐ moe_curve.png

OUTPUT 6: Jupyter Notebooks (5)
  ☐ 01_data_exploration.ipynb
  ☐ 02_model_training.ipynb
  ☐ 03_ensemble_analysis.ipynb
  ☐ 04_anomaly_detection.ipynb
  ☐ 05_final_evaluation.ipynb

OUTPUT 7: Technical Report
  ☐ report.pdf exists
  ☐ 8-12 pages
  ☐ Professional formatting

OUTPUT 8: Trained Models (6)
  ☐ lstm_model.pth
  ☐ transformer_model.pth
  ☐ stacking_ensemble.pkl
  ☐ mixture_of_experts.pth
  ☐ anomaly_detector.pkl
  ☐ scaler.pkl
```

---

Everything you need to load the data and create the outputs! 🚀

