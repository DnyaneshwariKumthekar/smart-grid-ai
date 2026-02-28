"""
Download Real-World Energy Datasets

Three options provided. Choose the one that fits your needs:
1. Household Electric Power Consumption (2.1M records) - LARGE, REALISTIC
2. UCI Energy Efficiency (768 buildings) - SMALL, BUILDING-LEVEL
3. UCI Appliances Energy (19k records) - MEDIUM, APARTMENT-LEVEL
"""

import os
import urllib.request
import zipfile
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_RAW.mkdir(parents=True, exist_ok=True)

def download_household_power(force=False):
    """
    Download Household Electric Power Consumption dataset
    - 2.1M records (Dec 2006 - Nov 2010)
    - 7 features: date, time, global active/reactive power, voltage, intensity, sub-metering
    - 1-minute granularity
    - Size: ~20 MB
    
    Source: UCI Machine Learning Repository
    """
    print("\n" + "="*70)
    print("OPTION 1: Household Electric Power Consumption")
    print("="*70)
    print("""
    📊 Dataset Info:
    - Records: 2,075,259 (Dec 2006 - Nov 2010, 4 years)
    - Granularity: 1-minute intervals
    - Features: Global active/reactive power, voltage, intensity, sub-metering (3 circuits)
    - Size: ~20 MB
    - Real-world: ✓ YES (French household)
    - Suitability: ⭐⭐⭐⭐⭐ (Perfect for energy forecasting)
    """)
    
    file_path = DATA_RAW / 'household_power_consumption.zip'
    csv_path = DATA_RAW / 'household_power_consumption.txt'
    
    if csv_path.exists() and not force:
        print(f"✓ Already downloaded: {csv_path}")
        return csv_path
    
    if not file_path.exists() or force:
        print(f"Downloading (~20 MB)...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
        try:
            urllib.request.urlretrieve(url, file_path)
            print(f"✓ Downloaded to {file_path}")
            
            # Extract
            print("Extracting...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_RAW)
            print("✓ Extracted")
            
            # Verify
            if csv_path.exists():
                print(f"✓ File ready: {csv_path}")
                return csv_path
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return None
    
    return csv_path if csv_path.exists() else None


def download_energy_efficiency(force=False):
    """
    Download UCI Energy Efficiency Data
    - 768 building samples
    - Building-level energy simulation data
    - 10 features: heating load, cooling load, etc.
    - Size: ~45 KB
    
    Source: UCI Machine Learning Repository
    """
    print("\n" + "="*70)
    print("OPTION 2: UCI Energy Efficiency Data")
    print("="*70)
    print("""
    📊 Dataset Info:
    - Records: 768 buildings
    - Granularity: Building-level (not time-series)
    - Features: Building parameters (surface area, wall area, roof area, height, orientation, etc.)
    - Size: ~45 KB
    - Real-world: ✓ YES (Building simulation)
    - Suitability: ⭐⭐⭐ (Good for energy analysis, not time-series forecasting)
    """)
    
    file_path = DATA_RAW / 'energy_efficiency.xlsx'
    
    if file_path.exists() and not force:
        print(f"✓ Already downloaded: {file_path}")
        return file_path
    
    print(f"Downloading (~50 KB)...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    try:
        urllib.request.urlretrieve(url, file_path)
        print(f"✓ Downloaded to {file_path}")
        return file_path
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return None


def download_appliances_energy(force=False):
    """
    Download UCI Appliances Energy Prediction Data
    - 19,735 records (Jan 2016 - May 2017)
    - 28 features: appliance energy consumption + weather
    - 10-minute granularity from single apartment
    - Size: ~2 MB
    
    Source: UCI Machine Learning Repository
    """
    print("\n" + "="*70)
    print("OPTION 3: UCI Appliances Energy Prediction")
    print("="*70)
    print("""
    📊 Dataset Info:
    - Records: 19,735 (Jan 2016 - May 2017, 1.5 years)
    - Granularity: 10-minute intervals
    - Features: Appliance energy consumption + 27 environmental/weather features
    - Size: ~2 MB
    - Real-world: ✓ YES (Single apartment in Belgium)
    - Suitability: ⭐⭐⭐⭐ (Good for appliance load forecasting)
    """)
    
    file_path = DATA_RAW / 'appliances_energy.zip'
    csv_path = DATA_RAW / 'energydata_complete.csv'
    
    if csv_path.exists() and not force:
        print(f"✓ Already downloaded: {csv_path}")
        return csv_path
    
    if not file_path.exists() or force:
        print(f"Downloading (~2 MB)...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.zip"
        try:
            urllib.request.urlretrieve(url, file_path)
            print(f"✓ Downloaded to {file_path}")
            
            # Extract
            print("Extracting...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_RAW)
            print("✓ Extracted")
            
            if csv_path.exists():
                print(f"✓ File ready: {csv_path}")
                return csv_path
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return None
    
    return csv_path if csv_path.exists() else None


def show_dataset_preview(file_path):
    """Show preview of downloaded dataset."""
    if file_path.suffix == '.xlsx':
        df = pd.read_excel(file_path)
    elif file_path.suffix == '.txt':
        df = pd.read_csv(file_path, sep=';', nrows=5000)  # Load sample
    else:
        df = pd.read_csv(file_path)
    
    print(f"\n📋 Preview ({len(df)} rows × {len(df.columns)} columns):")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\n   First few rows:")
    print(df.head(3).to_string())
    return df


def main():
    """Download and display real-world energy datasets."""
    
    print("\n" + "="*70)
    print("REAL-WORLD ENERGY DATASETS")
    print("="*70)
    
    print("""
Choose one dataset (run this script with argument):

    python download_real_dataset.py household    # 2.1M records, 1-min granularity ⭐⭐⭐⭐⭐
    python download_real_dataset.py efficiency   # 768 buildings, cross-sectional
    python download_real_dataset.py appliances   # 19k records, 10-min granularity ⭐⭐⭐⭐
    python download_real_dataset.py all          # Download all three
    python download_real_dataset.py show         # Show what you already have

RECOMMENDATION for Smart Grid Forecasting:
    ✓ Use "household" for realistic time-series forecasting
    ✓ Matches your project goal (energy consumption prediction)
    ✓ 4 years of data = good for training deep learning models
    """)
    
    import sys
    
    if len(sys.argv) < 2:
        print("\nNo dataset selected. Use: python download_real_dataset.py <household|efficiency|appliances|all|show>")
        return
    
    choice = sys.argv[1].lower()
    
    # Show what's already downloaded
    if choice == 'show' or choice == 'all':
        print("\n" + "="*70)
        print("ALREADY DOWNLOADED:")
        print("="*70)
        files = list(DATA_RAW.glob('*'))
        if files:
            for f in files:
                print(f"  ✓ {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
        else:
            print("  (None yet)")
        
        if choice == 'show':
            return
    
    # Download selected dataset(s)
    datasets = []
    
    if choice in ['household', 'all']:
        path = download_household_power()
        if path:
            datasets.append(('Household Power Consumption', path))
    
    if choice in ['efficiency', 'all']:
        path = download_energy_efficiency()
        if path:
            datasets.append(('Energy Efficiency', path))
    
    if choice in ['appliances', 'all']:
        path = download_appliances_energy()
        if path:
            datasets.append(('Appliances Energy', path))
    
    # Show previews
    if datasets:
        print("\n" + "="*70)
        print("DATASET PREVIEWS")
        print("="*70)
        for name, path in datasets:
            print(f"\n▶ {name}: {path}")
            try:
                show_dataset_preview(path)
            except Exception as e:
                print(f"  ✗ Error reading: {e}")


if __name__ == "__main__":
    main()
