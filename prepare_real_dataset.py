"""
Real-World Dataset Preprocessing and Adapter
Converts Household Electric Power Consumption data to smart grid format
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def load_household_data(downsample_to_5min=True, nrows=None):
    """
    Load and preprocess household power consumption data.
    
    Args:
        downsample_to_5min: Aggregate 1-min data to 5-min (matches synthetic data)
        nrows: Limit rows for testing (None = all 2.1M rows)
    
    Returns:
        DataFrame with cleaned data
    """
    print(f"Loading household power consumption data...")
    
    file_path = DATA_RAW / 'household_power_consumption.txt'
    
    if not file_path.exists():
        raise FileNotFoundError(f"Download dataset first: python download_real_dataset.py household")
    
    # Load data
    df = pd.read_csv(file_path, sep=';', nrows=nrows)
    print(f"  ✓ Loaded {len(df):,} records")
    
    # Parse datetime
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df = df.drop(['Date', 'Time'], axis=1)
    
    # Handle missing values (marked as '?')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    print(f"  ✓ Missing values: {missing_pct:.2f}%")
    
    # Fill missing values with forward fill then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Downsample from 1-minute to 5-minute (to match synthetic data freq)
    if downsample_to_5min:
        print("  Downsampling to 5-minute intervals...")
        df = df.set_index('timestamp').resample('5min').mean().reset_index()
        print(f"  ✓ Downsampled to {len(df):,} records")
    
    return df


def create_features_from_household(df):
    """
    Transform household data into smart-grid-like features.
    Maps from single-household to grid-scale features.
    """
    print("\nCreating smart-grid features from household data...")
    
    features = pd.DataFrame()
    features['timestamp'] = df['timestamp']
    
    # Main consumption (scale up to grid level - assume this house is 1/10000 of grid)
    SCALE = 10000  # Simulate scaling to grid level
    
    features['consumption_total'] = df['Global_active_power'] * SCALE
    features['consumption_industrial'] = df['Sub_metering_1'] * SCALE * 0.4  # ~40% industrial
    features['consumption_commercial'] = df['Sub_metering_2'] * SCALE * 0.3  # ~30% commercial
    features['consumption_residential'] = df['Sub_metering_3'] * SCALE * 0.3  # ~30% residential
    
    # Voltage proxy -> simulate generation with inverse relationship
    # (low voltage = high generation needed, vice versa)
    voltage_norm = (df['Voltage'] - df['Voltage'].min()) / (df['Voltage'].max() - df['Voltage'].min())
    
    features['generation_solar'] = (1 - voltage_norm) * 1000 + np.random.normal(0, 50, len(df))
    features['generation_wind'] = np.random.normal(500, 100, len(df))
    features['generation_hydro'] = 800 + np.random.normal(0, 50, len(df))
    features['generation_thermal'] = 1200 + np.random.normal(0, 100, len(df))
    features['generation_nuclear'] = 1500 + np.random.normal(0, 50, len(df))
    
    # Reactive power -> simulate weather
    reactive_norm = (df['Global_reactive_power'] - df['Global_reactive_power'].min()) / \
                   (df['Global_reactive_power'].max() - df['Global_reactive_power'].min())
    
    features['temperature'] = 15 + 10 * np.sin(np.arange(len(df)) * 2*np.pi/(288*365)) + \
                             reactive_norm * 20 + np.random.normal(0, 2, len(df))
    features['humidity'] = 60 + 20 * np.sin(np.arange(len(df)) * 2*np.pi/(288*365)) + \
                          reactive_norm * 15 + np.random.normal(0, 5, len(df))
    features['wind_speed'] = 5 + 8 * np.abs(np.sin(np.arange(len(df)) * 2*np.pi/1440)) + \
                            np.random.normal(0, 1, len(df))
    features['cloud_cover'] = 50 + 30 * np.sin(np.arange(len(df)) * 2*np.pi/1440) + \
                             np.random.normal(0, 10, len(df))
    
    # Grid status features
    features['frequency'] = 50 + (voltage_norm - 0.5) * 0.5  # ±0.25 Hz variation
    features['reactive_power'] = df['Global_reactive_power'] * SCALE
    features['voltage'] = df['Voltage']
    features['grid_load'] = df['Global_intensity'] * SCALE
    
    # Add random other features to match 32 features
    np.random.seed(42)
    for i in range(32 - len(features.columns) + 1):  # +1 for timestamp
        features[f'feature_{i}'] = np.random.normal(0, 1, len(df))
    
    # Ensure exactly 33 columns (32 features + timestamp)
    features = features.iloc[:, :33]
    
    print(f"  ✓ Created {len(features.columns)-1} features")
    return features


def prepare_real_dataset(n_samples=50000, downsample_to_5min=True):
    """
    Full pipeline: Load real data -> Transform -> Save
    
    Args:
        n_samples: How many records to use (None = all ~2.1M)
        downsample_to_5min: Downsample from 1-min to 5-min
    
    Returns:
        DataFrame ready for training
    """
    print("\n" + "="*70)
    print("REAL-WORLD DATASET PREPARATION")
    print("="*70)
    
    # Load and preprocess
    raw_df = load_household_data(downsample_to_5min=downsample_to_5min, nrows=n_samples)
    
    # Create smart-grid features
    features_df = create_features_from_household(raw_df)
    
    # Save processed data
    print(f"\nSaving processed data...")
    
    # Save as pickle (fastest, binary format)
    output_path = DATA_PROCESSED / 'household_power_smartgrid_features.pkl'
    features_df.to_pickle(output_path)
    print(f"  ✓ Saved: {output_path}")
    print(f"    Shape: {features_df.shape}")
    print(f"    Size: {features_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Display sample
    print(f"\n📊 Dataset Preview:")
    print(f"  Columns: {list(features_df.columns)}")
    print(f"  Time range: {features_df['timestamp'].min()} to {features_df['timestamp'].max()}")
    print(f"\n  First 5 rows:")
    print(features_df.head().to_string())
    
    return features_df


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) < 2:
        n_samples = 50000  # Default for quick testing
        print(f"Using default: {n_samples:,} samples")
    else:
        try:
            n_samples = int(sys.argv[1])
        except:
            n_samples = None  # Use all data
    
    prepare_real_dataset(n_samples=n_samples, downsample_to_5min=True)


if __name__ == "__main__":
    main()
