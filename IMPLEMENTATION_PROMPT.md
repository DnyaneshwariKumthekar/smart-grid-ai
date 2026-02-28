# 🔧 IMPLEMENTATION PROMPT - Week 2-4 Detailed Tasks

**Project**: Smart Grid Energy Forecasting  
**Duration**: Days 8-28 (21 days, 50-60 hours)  
**Focus**: Build ensemble methods, analyze results, document findings

---

## 📋 WEEK 2 TASKS (Days 8-14): ENSEMBLE METHODS

Your goal this week: Build 3 ensemble methods that significantly improve forecasting accuracy and anomaly detection.

### DAY 8: StackingEnsemble Setup (3 hours)

**Goal**: Create the foundation for your stacking ensemble

#### Tasks:
```
☐ Create models/ensemble.py (new file)
☐ Copy StackingEnsemble class from CODE_TEMPLATES.md
☐ Implement __init__() method
☐ Implement _generate_meta_features() method
☐ Write unit test for initialization
☐ Run test: python -m pytest tests/test_ensemble.py::test_stacking_init -v
```

#### Expected Output:
```python
# File: models/ensemble.py
class StackingEnsemble:
    def __init__(self, base_models, meta_learner, cv_folds=5):
        """Initialize stacking ensemble"""
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.cv_folds = cv_folds
        
    def _generate_meta_features(self, X, y=None):
        """Generate meta-features from base models"""
        # Implementation here
```

#### Testing:
```python
# tests/test_ensemble.py
def test_stacking_init():
    ensemble = StackingEnsemble(base_models=[lstm, transformer], 
                                 meta_learner=xgb_model)
    assert ensemble.base_models is not None
    assert ensemble.meta_learner is not None
    print("✓ StackingEnsemble initialized correctly")
```

---

### DAY 9: StackingEnsemble Training (3 hours)

**Goal**: Implement the full training pipeline for stacking ensemble

#### Tasks:
```
☐ Implement fit() method
☐ Implement predict() method
☐ Implement evaluate() method
☐ Test on sample data (10k samples)
☐ Verify MAPE < 8%
☐ Create simple visualization (30-day forecast plot)
☐ Run test: python -m pytest tests/test_ensemble.py::test_stacking_train -v
```

#### Expected Output:
```
Test Results:
  ✓ fit() - Trains on 8k samples in <30 sec
  ✓ predict() - Predicts 2k samples in <5 sec
  ✓ evaluate() - MAPE: 7.8% ✓ (target: <8%)
  ✓ Visualization saved: results/visualizations/stacking_30day.png
```

#### Code Structure:
```python
def fit(self, X_train, y_train, X_val, y_val):
    """Train stacking ensemble with cross-validation"""
    # Generate meta-features
    meta_X_train = self._generate_meta_features(X_train, y_train)
    
    # Train meta-learner
    self.meta_learner.fit(meta_X_train, y_train)
    
    return self

def predict(self, X):
    """Generate predictions"""
    meta_X = self._generate_meta_features(X)
    return self.meta_learner.predict(meta_X)

def evaluate(self, X_test, y_test):
    """Calculate metrics"""
    y_pred = self.predict(X_test)
    mape = calculate_mape(y_test, y_pred)
    rmse = calculate_rmse(y_test, y_pred)
    return {"mape": mape, "rmse": rmse}
```

---

### DAYS 10-11: MixtureOfExperts Implementation (6 hours)

**Goal**: Build a mixture of experts with 3 specialist models

#### Tasks:
```
☐ Create models/mixture_of_experts.py
☐ Copy MixtureOfExperts class from CODE_TEMPLATES.md
☐ Implement __init__() - 3 specialists (short, medium, long-term)
☐ Implement forward() - Gating mechanism
☐ Implement load_balancing_loss()
☐ Implement training loop
☐ Test expert specialization
☐ Verify each expert focuses on different time horizons
☐ Run test: python -m pytest tests/test_moe.py -v
```

#### Expert Specialization (verify this works):
```
Short-term expert (horizon: 1-6 steps):
  └─ Focus: Immediate fluctuations
  └─ Test: MAPE on next hour: <5%

Medium-term expert (horizon: 7-48 steps):
  └─ Focus: Daily patterns
  └─ Test: MAPE on next day: 6-8%

Long-term expert (horizon: 49-288 steps):
  └─ Focus: Weekly trends
  └─ Test: MAPE on next week: 8-10%
```

#### Testing:
```
Test Results:
  ✓ Short-term expert MAPE: 4.8% (1-hour horizon)
  ✓ Medium-term expert MAPE: 7.2% (24-hour horizon)
  ✓ Long-term expert MAPE: 9.1% (168-hour horizon)
  ✓ Gating mechanism working (soft selection)
  ✓ Load balancing loss < 0.1
```

---

### DAYS 12-13: AnomalyDetectionEnsemble (6 hours)

**Goal**: Build anomaly detection using 3 methods with voting

#### Tasks:
```
☐ Create models/anomaly_detection.py
☐ Copy AnomalyDetectionEnsemble class from CODE_TEMPLATES.md
☐ Implement Isolation Forest detector
☐ Implement One-Class SVM detector
☐ Implement Autoencoder detector
☐ Implement ensemble voting mechanism
☐ Test on synthetic anomalies
☐ Verify F1 > 0.85
☐ Calculate ROC AUC
☐ Run test: python -m pytest tests/test_anomaly.py -v
```

#### Anomaly Detection Methods:

**1. Isolation Forest**
```python
# Unsupervised - detects statistical outliers
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.05, random_state=42)

# Test: F1 on synthetic anomalies
expected_f1 = 0.80+
```

**2. One-Class SVM**
```python
# Detects points far from normal data boundary
from sklearn.svm import OneClassSVM
svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')

# Test: F1 on synthetic anomalies
expected_f1 = 0.82+
```

**3. Autoencoder**
```python
# Neural network - detects reconstruction error
class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_size=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_size)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Test: F1 on synthetic anomalies
expected_f1 = 0.84+
```

**Ensemble Voting**:
```python
def detect_anomalies(self, X):
    """Vote across 3 detectors"""
    scores = []
    
    # Get scores from each detector (0-1)
    scores.append(self.iso_forest.predict(X))
    scores.append(self.svm.predict(X))
    scores.append(self.autoencoder_anomaly_score(X))
    
    # Average scores
    ensemble_score = np.mean(scores, axis=0)
    
    # Threshold at 0.5
    predictions = (ensemble_score > 0.5).astype(int)
    
    return predictions, ensemble_score
```

#### Testing:
```
Test Results on 10k synthetic test set:
  ✓ Isolation Forest: Precision=0.88, Recall=0.80, F1=0.84
  ✓ One-Class SVM: Precision=0.85, Recall=0.82, F1=0.83
  ✓ Autoencoder: Precision=0.82, Recall=0.84, F1=0.83
  ✓ Ensemble Voting: Precision=0.90, Recall=0.85, F1=0.87 ✓
  ✓ ROC AUC: 0.922
```

---

### DAY 14: Integration & Testing (4 hours)

**Goal**: Ensure all ensemble methods work together

#### Tasks:
```
☐ Create integration test (test_full_pipeline.py)
☐ Test: LSTM → Transformer → Ensemble pipeline
☐ Test: Anomaly detection on same data
☐ Create combined results CSV
☐ Generate 4 visualizations:
   ├─ 30-day forecast (all methods)
   ├─ ROC curves (anomaly detection)
   ├─ Ensemble comparison bar chart
   └─ Feature importance plot
☐ Verify all tests passing
☐ Document results
☐ Commit to git
```

#### Integration Test:
```python
def test_full_pipeline():
    """Test: Load data → All ensembles → Save results"""
    
    # Load data
    X_test, y_test = load_test_data()
    
    # Make predictions
    lstm_pred = lstm_model.predict(X_test)
    transformer_pred = transformer_model.predict(X_test)
    ensemble_pred = stacking_ensemble.predict(X_test)
    
    # Detect anomalies
    anomaly_pred, anomaly_scores = anomaly_detector.detect(X_test)
    
    # Calculate metrics
    results = {
        'lstm_mape': calculate_mape(y_test, lstm_pred),
        'transformer_mape': calculate_mape(y_test, transformer_pred),
        'ensemble_mape': calculate_mape(y_test, ensemble_pred),
        'anomaly_f1': calculate_f1(true_anomalies, anomaly_pred)
    }
    
    # Assertions
    assert results['lstm_mape'] < 9.0
    assert results['transformer_mape'] < 8.0
    assert results['ensemble_mape'] < 6.0  # TARGET
    assert results['anomaly_f1'] > 0.87     # TARGET
    
    print("✅ ALL INTEGRATION TESTS PASSED")
    return results
```

#### Expected Output:
```
WEEK 2 RESULTS:
├─ LSTM MAPE: 8.7%
├─ Transformer MAPE: 7.6%
├─ StackingEnsemble MAPE: 4.2% ✓ (improved from 7.6%)
├─ MixtureOfExperts MAPE: 5.1%
├─ AnomalyDetection F1: 0.904 ✓
└─ Visualizations saved: 4 PNG files

Code:
├─ models/ensemble.py (350 lines)
├─ models/mixture_of_experts.py (280 lines)
├─ models/anomaly_detection.py (420 lines)
└─ tests/test_ensemble.py (400 lines)
```

---

## 📊 WEEK 3 TASKS (Days 15-20): ANALYSIS & BENCHMARKING

### DAYS 15-16: Attention Visualization (4 hours)

**Goal**: Visualize what the models pay attention to

#### Tasks:
```
☐ Create visualizations/attention_viz.py
☐ Extract attention weights from LSTM model
☐ Extract attention weights from Transformer model
☐ Create heatmaps showing attention patterns
☐ Visualize for 7-day period
☐ Save 2 high-quality PNG files
```

#### Visualization:
```
Heatmap: Model Attention Over Time
  Rows: 32 input features
  Columns: 288 timesteps (24 hours)
  Color intensity: Attention weight (0-1)
  
Expected patterns:
  ✓ LSTM: High attention on recent timesteps
  ✓ Transformer: Distributed attention pattern
  ✓ Both: Should focus on consumption features
```

---

### DAYS 17-18: Uncertainty Quantification (4 hours)

**Goal**: Calculate prediction confidence intervals

#### Tasks:
```
☐ Implement Monte Carlo Dropout
☐ Generate 100 stochastic predictions per sample
☐ Calculate mean and std deviation
☐ Create 95% confidence intervals
☐ Visualize uncertainty bands
☐ Analyze where uncertainty is highest
☐ Save visualization
```

#### Uncertainty Output:
```
For each prediction:
  ├─ Point estimate: 450 kWh
  ├─ Uncertainty: ±25 kWh (95% CI)
  └─ Confidence: High for peak hours, Low for transitions
```

---

### DAYS 19-20: Comprehensive Benchmarking (6 hours)

**Goal**: Compare your ensemble vs 8 baseline methods

#### Baseline Methods to Compare:
```
1. ARIMA (statistical baseline)
2. Prophet (Facebook's forecaster)
3. SARIMA (seasonal ARIMA)
4. ETS (exponential smoothing)
5. XGBoost (traditional ML)
6. LightGBM (gradient boosting)
7. Random Forest (ensemble baseline)
8. Support Vector Regression (SVM)
```

#### Benchmarking Test:
```python
def benchmark_all_methods():
    """Compare your ensemble vs 8 baselines"""
    
    results = {}
    
    for method_name, method in methods.items():
        predictions = method.predict(X_test)
        mape = calculate_mape(y_test, predictions)
        rmse = calculate_rmse(y_test, predictions)
        mae = calculate_mae(y_test, predictions)
        
        results[method_name] = {
            'mape': mape,
            'rmse': rmse,
            'mae': mae
        }
    
    # Statistical significance testing
    for method_name in results:
        p_value = statistical_test(results['Your Ensemble'], results[method_name])
        results[method_name]['p_value'] = p_value
        results[method_name]['significant'] = p_value < 0.05
    
    return results
```

#### Expected Results:
```
Method                  MAPE    RMSE    Improvement
─────────────────────────────────────────────────
ARIMA                   12.0%   89.2    baseline
Prophet                 10.5%   85.1    -12.5%
SARIMA                  9.8%    82.3    -18.3%
ETS                     11.2%   87.5    -6.7%
XGBoost                 7.2%    71.4    -40%
LightGBM                6.8%    69.1    -43.3%
Random Forest           8.1%    75.2    -32.5%
SVM                     7.9%    73.6    -34.2%
───────────────────────────────────────────────
YOUR ENSEMBLE           4.2%    52.1    -65% ✓✓✓
───────────────────────────────────────────────
```

---

## 📓 WEEK 4 TASKS (Days 21-28): DOCUMENTATION & FINALIZATION

### DAYS 21-23: Jupyter Notebooks (8 hours)

**Goal**: Create 5 comprehensive Jupyter notebooks

#### Notebook 1: Data Exploration (150 lines)
```
1. Load data
2. Statistical summary (mean, std, quantiles)
3. Missing values analysis
4. Feature distributions (histograms)
5. Correlation matrix
6. Time series plot
7. Seasonal decomposition
8. Anomaly rate analysis
```

#### Notebook 2: Model Training (200 lines)
```
1. Data preparation
2. LSTM training (with loss curve)
3. Transformer training (with loss curve)
4. Training time comparison
5. Hyperparameter impact
6. Validation performance
7. Test set performance
8. Model comparison
```

#### Notebook 3: Ensemble Analysis (180 lines)
```
1. Load base models
2. Generate meta-features
3. Train stacking ensemble
4. Compare ensemble vs single models
5. Mixture of Experts specialization
6. Performance breakdown
7. Visualization: ensemble components
8. Statistical significance tests
```

#### Notebook 4: Anomaly Detection (160 lines)
```
1. Data preparation for anomaly detection
2. Train 3 detectors
3. Test on synthetic anomalies
4. ROC curves
5. Confusion matrices
6. Ensemble voting analysis
7. Real anomaly examples
8. Detection performance vs threshold
```

#### Notebook 5: Final Evaluation (140 lines)
```
1. Load all trained models
2. Generate final predictions
3. Calculate all metrics
4. Benchmarking vs baselines
5. Statistical significance
6. Uncertainty quantification
7. Key findings summary
8. Recommendations
```

---

### DAYS 24-26: Technical Report (8 hours)

**Goal**: Write 8-12 page PDF report

#### Report Structure:

**1. Executive Summary (1 page)**
```
- Problem statement
- Approach overview
- Key results
- Improvement over baseline
```

**2. Introduction (1.5 pages)**
```
- Background on smart grids
- Motivation for forecasting
- Related work
- Contributions
```

**3. Methodology (2.5 pages)**
```
- Data description
- Feature engineering
- LSTM architecture
- Transformer architecture
- Stacking ensemble approach
- Mixture of experts approach
- Anomaly detection ensemble
- Validation strategy (walk-forward)
```

**4. Experiments (2 pages)**
```
- Experimental setup
- Hardware and training time
- Hyperparameter selection
- Cross-validation results
- Test set results
```

**5. Results (1.5 pages)**
```
- Performance metrics table
- Visualizations (30-day forecast, ROC curves)
- Benchmarking vs baselines
- Ensemble component contribution
- Anomaly detection performance
```

**6. Analysis (1 page)**
```
- Why ensemble works
- Expert specialization insights
- Attention patterns
- Uncertainty quantification
- Failure cases
```

**7. Conclusion (0.5 pages)**
```
- Summary of contributions
- Practical implications
- Future work
- Recommendations
```

**8. Appendix**
```
- Code snippets
- Additional results
- Hyperparameter tables
- Computational requirements
```

---

### DAYS 27-28: Final Polish & Submission (4 hours)

#### Tasks:
```
☐ Code cleanup (PEP 8, docstrings)
☐ Run all tests (pytest)
☐ Update README.md
☐ Create requirements.txt
☐ Test on fresh environment
☐ Generate all outputs
☐ Verify all 8 output types present
☐ PDF report final review
☐ Git commit and push
☐ Create final checklist
☐ Submit!
```

#### Final Checklist:
```
Code:
  ☐ All .py files PEP 8 compliant
  ☐ 90%+ test coverage
  ☐ No runtime warnings
  ☐ Docstrings on all classes/functions
  
Results:
  ☐ metrics.json exists with all 8 fields
  ☐ test_predictions.csv has 105k rows
  ☐ anomaly_detection.csv has 105k rows
  ☐ 11 visualizations in results/
  ☐ 6 trained models in results/models/
  
Documentation:
  ☐ README.md complete and clear
  ☐ 5 notebooks (all runnable)
  ☐ report.pdf (8-12 pages)
  ☐ requirements.txt accurate
  
Performance:
  ☐ LSTM MAPE: 7-9% ✓
  ☐ Transformer MAPE: 6-8% ✓
  ☐ Ensemble MAPE: <6% ✓
  ☐ Anomaly F1: >0.87 ✓
  ☐ Improvement: >60% ✓
```

---

## 📊 EXPECTED TIMELINE

| Week | Days | Tasks | Hours | Expected Output |
|------|------|-------|-------|-----------------|
| 2 | 8-9 | Stacking Ensemble | 6 | MAPE 7.8% |
| 2 | 10-11 | Mixture of Experts | 6 | 3 specialists |
| 2 | 12-13 | Anomaly Detection | 6 | F1 0.87+ |
| 2 | 14 | Integration & Testing | 4 | Full pipeline |
| 3 | 15-16 | Attention Visualization | 4 | 2 heatmaps |
| 3 | 17-18 | Uncertainty Quantification | 4 | Confidence intervals |
| 3 | 19-20 | Benchmarking | 6 | vs 8 baselines |
| 4 | 21-23 | Jupyter Notebooks | 8 | 5 notebooks |
| 4 | 24-26 | Technical Report | 8 | 10-page PDF |
| 4 | 27-28 | Polish & Submit | 4 | Final deliverable |
| | **TOTAL** | | **56 hours** | **A+ grade** |

---

## ✅ SUCCESS CRITERIA BY WEEK

### Week 2 (End of Day 14):
- ✓ StackingEnsemble working (MAPE <8%)
- ✓ MixtureOfExperts implemented (3 specialists)
- ✓ AnomalyDetection working (F1 >0.85)
- ✓ All tests passing
- ✓ Code committed to git

### Week 3 (End of Day 20):
- ✓ Attention visualizations complete
- ✓ Uncertainty quantification working
- ✓ Benchmarking vs 8 methods done
- ✓ Statistical significance tested
- ✓ Results documented

### Week 4 (End of Day 28):
- ✓ 5 Jupyter notebooks complete
- ✓ 10-page PDF report done
- ✓ All code cleaned up
- ✓ All tests passing
- ✓ Ready for submission
- ✓ Expected grade: A+ (90-100%)

---

## 🚀 DAILY QUICK CHECKLIST

**Every day:**
```
☐ Write code (main task)
☐ Write tests (unit tests)
☐ Run tests (pytest)
☐ Commit to git (save work)
☐ Document progress (notes)
☐ Update TODO.md (track tasks)
```

**Twice per week:**
```
☐ Generate visualizations
☐ Check metrics
☐ Verify improvement
☐ Update documentation
```

**End of each week:**
```
☐ Review progress
☐ Update README
☐ Generate outputs
☐ Take final metrics
☐ Plan next week
```

---

This document has everything you need to succeed. Follow it day by day, and you'll complete a production-grade forecasting system! 🎓

