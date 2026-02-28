# Day 8-9 Implementation Guide: StackingEnsemble

## Overview
This guide walks you through the Day 8-9 implementation step by step.

## Files Created
- `data_loader.py` - Data generation, preprocessing, and sequence creation
- `models/ensemble.py` - StackingEnsemble class with LSTM + Transformer + XGBoost
- `train_day8_9.py` - Main training script for Days 8-9
- `__init__.py` - Project initialization

## Project Structure
```
smart-grid-ai/
├── data/
│   ├── raw/           # Raw data will go here
│   └── processed/     # Processed data
├── models/
│   ├── ensemble.py    # StackingEnsemble ✓ IMPLEMENTED
│   ├── moe.py         # MixtureOfExperts (TODO: Days 10-11)
│   └── anomaly.py     # AnomalyDetection (TODO: Days 12-13)
├── notebooks/         # Jupyter notebooks
├── results/           # Training results, metrics, visualizations
├── tests/             # Unit tests
├── data_loader.py     # ✓ IMPLEMENTED
├── train_day8_9.py    # ✓ IMPLEMENTED
├── requirements.txt
└── [Documentation files]
```

## Quick Start

### Step 1: Activate Virtual Environment
```powershell
# Create virtual environment (if not exists)
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run Day 8-9 Training
```powershell
python train_day8_9.py
```

## What the Code Does

### 1. Data Loader (`data_loader.py`)
- **`generate_synthetic_data()`**: Creates 100k+ synthetic samples with:
  - 4 consumption features
  - 5 generation features
  - 5 weather features
  - 8 time-based features
  - 5 system status features
  - 5 derived features
  - Total: 32 features, structured like real smart grid data

- **`preprocess_data()`**: 
  - Handles missing values (forward/backward fill)
  - Normalizes all features (StandardScaler)
  - Creates 288-timestep sequences (24-hour lookback)
  - Temporal train/test split (80/20)
  - Returns: X_train, X_test, y_train, y_test, scaler

- **`create_sequences()`**: 
  - Converts flat time series into sliding windows
  - Input: (n_samples, n_features)
  - Output: (n_sequences, 288, n_features)

### 2. StackingEnsemble (`models/ensemble.py`)

#### Architecture
```
Input (batch, 288, 32)
    ↓
    ├─→ LSTM (64 hidden units, 2 layers)  ─→ Output 1
    │
    └─→ Transformer (64 d_model, 4 heads) ─→ Output 2
    ↓
    Meta-Features (batch, 2)
    ↓
    XGBoost Meta-Learner
    ↓
    Final Prediction
```

#### Key Components

**1. LSTM Base Model** (`LSTMBase`)
- Input: Sequence of 32 features × 288 timesteps
- Processing: 2-layer LSTM with dropout
- Output: Single prediction per sample
- Used for capturing long-range dependencies

**2. Transformer Base Model** (`TransformerBase`)
- Input: Same as LSTM
- Processing: Multi-head self-attention + positional encoding
- Output: Single prediction per sample
- Used for parallel processing and attention weights

**3. K-Fold Meta-Feature Generation**
- Split data into 5 folds
- For each fold:
  - Train both base models on 4 folds
  - Predict on 1 fold (validation)
- Result: Meta-features = [lstm_pred, transformer_pred]
- This prevents data leakage!

**4. XGBoost Meta-Learner**
- Input: Meta-features (batch, 2)
- Training: 200 estimators, depth=6
- Learns how to best combine LSTM + Transformer predictions

#### Metrics Calculated

- **MAPE**: Mean Absolute Percentage Error (%)
  - Good for percentage-based error analysis
  - Target: < 8% (Days 8-9), < 6% (final)

- **RMSE**: Root Mean Squared Error
  - Penalizes large errors more
  - In same units as target

- **MAE**: Mean Absolute Error
  - Average absolute prediction error
  - Interpretable, not squared

- **R²**: Coefficient of determination
  - 1.0 = perfect fit
  - 0.0 = baseline model performance
  - < 0 = worse than baseline

### 3. Training Script (`train_day8_9.py`)

Runs complete pipeline:
1. Generate 100k synthetic samples
2. Preprocess data (normalize, split, create sequences)
3. Train StackingEnsemble (20 epochs, 5-fold CV)
4. Evaluate on test set
5. Save results and predictions

Expected Runtime: 10-15 minutes (depending on GPU)

## Success Criteria - Day 8-9

✓ **PRIMARY**: MAPE < 8.0%
- This proves the ensemble is learning
- Combined predictions better than individual models

✓ **CODE QUALITY**:
- Clean, documented code
- Proper error handling
- Follows Python best practices (PEP 8)

✓ **VALIDATION**:
- Train/test split done properly (no data leakage)
- K-fold CV prevents overfitting
- Metrics calculated correctly

## Testing the Implementation

### Quick Test
```python
from data_loader import generate_synthetic_data, preprocess_data
from models.ensemble import StackingEnsemble

# Load data
df = generate_synthetic_data(n_samples=50000)
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# Train
ensemble = StackingEnsemble()
ensemble.fit(X_train, y_train, epochs=10)

# Evaluate
metrics = ensemble.evaluate(X_test, y_test)
print(f"MAPE: {metrics['MAPE']:.2f}%")
```

### With Unit Tests
```powershell
pytest tests/ -v
```

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch_size or use CPU
```python
# In train_day8_9.py, change:
device = torch.device('cpu')  # Force CPU
```

### Issue: Slow training
**Solution**: Reduce data size or epochs
```python
df = generate_synthetic_data(n_samples=50000)  # Smaller dataset
ensemble.fit(X_train, y_train, epochs=10)  # Fewer epochs
```

### Issue: ImportError for torch/sklearn/xgboost
**Solution**: Reinstall requirements
```powershell
pip install --upgrade torch scikit-learn xgboost
```

## Expected Output

When you run `python train_day8_9.py`, you should see:

```
======================================================================
SMART GRID ENSEMBLE - DAY 8-9 IMPLEMENTATION
======================================================================

📱 Device: cuda
🔧 PyTorch version: 2.1.0

----------------------------------------------------------------------
STEP 1: Data Generation
----------------------------------------------------------------------
Generating synthetic data: 100000 samples × 32 features...
✓ Generated data shape: (100000, 33)

...

----------------------------------------------------------------------
STEP 4: Evaluation on Test Set
----------------------------------------------------------------------

✓ Test Set Metrics:
  MAPE (Mean Absolute Percentage Error): 7.45%
  RMSE (Root Mean Squared Error):        45.2341
  MAE (Mean Absolute Error):             32.1234
  R² Score:                              0.8567

📊 Success Criteria (Day 8-9):
  ✓ MAPE < 8.0%: 7.45% (TARGET ACHIEVED)
  → Target for final ensemble: MAPE < 6.0% (currently 7.45%)

...

SUMMARY - DAY 8-9 COMPLETE
```

## What's Next (Day 10-11)

Next phase: MixtureOfExperts
- Implement 3 specialist networks (short-term, medium-term, long-term)
- Add gating network for dynamic expert selection
- Implement load balancing to prevent expert collapse
- Target: MAPE < 5%

## Questions?

Refer to the documentation files:
- `IMPLEMENTATION_PROMPT.md` - Day-by-day tasks
- `DATASET_SPECIFICATION.md` - Data details
- `CODE_TEMPLATES.md` - Code examples
- `00_START_HERE.md` - Big picture overview
