@echo off
REM Quick start script for Day 8-9 training
REM This script sets up the environment and runs the training

echo.
echo ===================================================================
echo SMART GRID AI - Day 8-9 Implementation
echo ===================================================================
echo.

REM Step 1: Create virtual environment if not exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created!
)

REM Step 2: Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Step 3: Install requirements
echo.
echo Installing requirements...
pip install -r requirements.txt -q

REM Step 4: Run training
echo.
echo ===================================================================
echo Starting Day 8-9 Training...
echo ===================================================================
echo.

python train_day8_9.py

echo.
echo ===================================================================
echo Training Complete!
echo ===================================================================
echo.
echo Check results in:
echo   - results/day8_9_metrics.csv (Performance metrics)
echo   - results/day8_9_predictions_sample.csv (Sample predictions)
echo.
echo Next Steps:
echo   1. Review the metrics
echo   2. If MAPE < 8%%, proceed to Days 10-11 (MixtureOfExperts)
echo   3. If MAPE > 8%%, adjust hyperparameters and retry
echo.

pause
