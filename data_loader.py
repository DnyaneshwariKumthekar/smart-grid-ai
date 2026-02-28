"""
Data loading and preprocessing module for smart grid energy forecasting.
Handles data loading, feature engineering, and sequence creation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


def generate_synthetic_data(n_samples: int = 525600, n_features: int = 32) -> pd.DataFrame:
    """
    Generate synthetic smart grid data for testing.
    
    Args:
        n_samples: Number of 5-minute interval samples (2 years = 525,600)
        n_features: Number of features per timestep
        
    Returns:
        DataFrame with synthetic data
    """
    print(f"Generating synthetic data: {n_samples} samples × {n_features} features...")
    
    # Create timestamps (5-minute intervals for 2 years)
    timestamps = pd.date_range('2022-01-01', periods=n_samples, freq='5min')
    
    data = {'timestamp': timestamps}
    
    # Consumption features (4 features)
    data['consumption_total'] = 500 + 200 * np.sin(np.arange(n_samples) * 2*np.pi/288) + np.random.normal(0, 50, n_samples)
    data['consumption_industrial'] = 300 + 100 * np.sin(np.arange(n_samples) * 2*np.pi/288) + np.random.normal(0, 30, n_samples)
    data['consumption_commercial'] = 150 + 50 * np.sin(np.arange(n_samples) * 2*np.pi/288) + np.random.normal(0, 20, n_samples)
    data['consumption_residential'] = 50 + 50 * np.sin(np.arange(n_samples) * 2*np.pi/288) + np.random.normal(0, 15, n_samples)
    
    # Generation features (5 features)
    data['generation_solar'] = 100 + 150 * np.abs(np.sin(np.arange(n_samples) * 2*np.pi/288)) + np.random.normal(0, 20, n_samples)
    data['generation_wind'] = 80 + 120 * np.random.random(n_samples) + np.random.normal(0, 15, n_samples)
    data['generation_hydro'] = 200 + np.random.normal(0, 10, n_samples)
    data['generation_thermal'] = 250 + np.random.normal(0, 20, n_samples)
    data['generation_nuclear'] = 300 + np.random.normal(0, 5, n_samples)
    
    # Weather features (5 features)
    data['temperature'] = 15 + 10 * np.sin(np.arange(n_samples) * 2*np.pi/(288*365)) + np.random.normal(0, 2, n_samples)
    data['humidity'] = 60 + 20 * np.sin(np.arange(n_samples) * 2*np.pi/288) + np.random.normal(0, 5, n_samples)
    data['wind_speed'] = 5 + 8 * np.abs(np.sin(np.arange(n_samples) * 2*np.pi/1440)) + np.random.normal(0, 1, n_samples)
    data['cloud_cover'] = 50 + 30 * np.sin(np.arange(n_samples) * 2*np.pi/1440) + np.random.normal(0, 10, n_samples)
    data['precipitation'] = np.maximum(0, 5 * np.random.exponential(0.1, n_samples))
    
    # Time-based features (8 features)
    hour_of_day = (np.arange(n_samples) % 288) / 12  # 288 5-min intervals in 24h
    data['hour_of_day'] = hour_of_day
    data['hour_sin'] = np.sin(2*np.pi*hour_of_day/24)
    data['hour_cos'] = np.cos(2*np.pi*hour_of_day/24)
    day_of_week = ((np.arange(n_samples) // 288) % 7)
    data['day_of_week'] = day_of_week
    data['day_sin'] = np.sin(2*np.pi*day_of_week/7)
    data['day_cos'] = np.cos(2*np.pi*day_of_week/7)
    day_of_year = ((np.arange(n_samples) // 288) % 365)
    data['month_sin'] = np.sin(2*np.pi*day_of_year/365)
    data['month_cos'] = np.cos(2*np.pi*day_of_year/365)
    
    # System status features (5 features)
    data['frequency'] = 50 + np.random.normal(0, 0.1, n_samples)
    data['voltage'] = 240 + np.random.normal(0, 5, n_samples)
    data['active_power'] = 600 + 200 * np.sin(np.arange(n_samples) * 2*np.pi/288) + np.random.normal(0, 40, n_samples)
    data['reactive_power'] = 100 + 50 * np.sin(np.arange(n_samples) * 2*np.pi/288) + np.random.normal(0, 10, n_samples)
    data['power_factor'] = 0.95 + np.random.normal(0, 0.02, n_samples)
    
    # Derived features (5 features)
    total_consumption = np.array(data['consumption_total'])
    total_generation = np.array(data['generation_solar'] + data['generation_wind'] + 
                                 data['generation_hydro'] + data['generation_thermal'] + 
                                 data['generation_nuclear'])
    data['demand_supply_gap'] = total_consumption - total_generation
    data['renewable_percentage'] = ((data['generation_solar'] + data['generation_wind']) / 
                                   (total_generation + 1e-6) * 100)
    data['peak_indicator'] = (total_consumption > np.percentile(total_consumption, 90)).astype(int)
    data['is_weekend'] = (day_of_week >= 5).astype(int)
    data['grid_load'] = (total_consumption / (total_generation + 1e-6))
    
    df = pd.DataFrame(data)
    
    # Ensure all values are positive/reasonable
    for col in df.columns:
        if col != 'timestamp':
            df[col] = df[col].clip(lower=0)
    
    print(f"✓ Generated data shape: {df.shape}")
    return df


def preprocess_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Preprocess data: handle missing values, normalize, split into train/test.
    
    Args:
        df: Input DataFrame with raw data
        test_size: Proportion of data for testing (default 0.2 for 80/20 split)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler)
    """
    print("Preprocessing data...")
    
    # Step 1: Handle missing values
    df_filled = df.fillna(method='ffill').fillna(method='bfill')
    print(f"✓ Missing values handled")
    
    # Step 2: Separate features and target
    feature_cols = [col for col in df_filled.columns if col != 'timestamp']
    target_col = 'consumption_total'
    
    X = df_filled[feature_cols].values
    y = df_filled[target_col].values.reshape(-1, 1)
    
    # Step 3: Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)
    
    print(f"✓ Features normalized")
    
    # Step 4: Create sequences
    sequence_length = 288  # 24-hour lookback (288 × 5-min = 24 hours)
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)
    print(f"✓ Sequences created: {X_seq.shape}")
    
    # Step 5: Temporal train/test split (NOT random shuffle)
    split_idx = int(len(X_seq) * (1 - test_size))
    X_train = X_seq[:split_idx]
    X_test = X_seq[split_idx:]
    y_train = y_seq[:split_idx]
    y_test = y_seq[split_idx:]
    
    print(f"✓ Train/test split: {X_train.shape} / {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler


def create_sequences(X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.
    
    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples, 1)
        sequence_length: Length of each sequence (default 288 for 24-hour windows)
        
    Returns:
        Tuple of (X_sequences, y_sequences)
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    
    return np.array(X_seq), np.array(y_seq)


def get_data_stats(df: pd.DataFrame) -> Dict:
    """Get basic statistics about the dataset."""
    stats = {
        'shape': df.shape,
        'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
        'feature_means': df.drop('timestamp', axis=1).mean().to_dict(),
        'feature_stds': df.drop('timestamp', axis=1).std().to_dict(),
    }
    return stats


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    df = generate_synthetic_data(n_samples=525600, n_features=32)
    
    # Show statistics
    stats = get_data_stats(df)
    print(f"\nDataset Statistics:")
    print(f"  Shape: {stats['shape']}")
    print(f"  Missing: {stats['missing_percentage']:.4f}%")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df, test_size=0.2)
    print(f"\nPreprocessing Complete:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Sample X_train[0] shape: {X_train[0].shape}")
