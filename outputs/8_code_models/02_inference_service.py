
"""
Smart Grid AI - Inference Service
Deployment-ready inference code for energy consumption forecasting
"""

import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

class EnergyForecaster:
    """Production-grade energy forecasting service"""

    def __init__(self, model_path='models/ensemble_model.pkl', scaler_path='models/scaler.pkl'):
        """Initialize forecaster with pre-trained model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = ['hour', 'dow', 'month', 'temp', 'humidity']

    def prepare_features(self, timestamp, weather_data):
        """Prepare features for prediction"""
        features = {
            'hour': timestamp.hour,
            'dow': timestamp.dayofweek,
            'month': timestamp.month,
            'temp': weather_data['temperature'],
            'humidity': weather_data['humidity']
        }
        return pd.DataFrame([features])

    def predict(self, timestamp, weather_data, return_intervals=True):
        """Generate forecast with optional prediction intervals"""
        X = self.prepare_features(timestamp, weather_data)
        X_scaled = self.scaler.transform(X)

        prediction = self.model.predict(X_scaled)[0]

        if return_intervals:
            # Bootstrap-based intervals
            prediction_std = np.std([self.model.predict(X_scaled) 
                                     for _ in range(100)])
            lower_95 = prediction - 1.96 * prediction_std
            upper_95 = prediction + 1.96 * prediction_std
            return {
                'forecast': prediction,
                'lower_95': lower_95,
                'upper_95': upper_95,
                'timestamp': timestamp
            }
        return prediction

    def batch_forecast(self, timestamps, weather_df):
        """Generate forecasts for multiple timestamps"""
        results = []
        for ts in timestamps:
            weather = weather_df[weather_df['timestamp'] == ts].iloc[0].to_dict()
            result = self.predict(ts, weather)
            results.append(result)
        return pd.DataFrame(results)

# Usage example
if __name__ == '__main__':
    forecaster = EnergyForecaster()

    # Single forecast
    now = datetime.now()
    weather = {'temperature': 22.5, 'humidity': 65}
    forecast = forecaster.predict(now, weather)
    print(f"Forecast: {forecast['forecast']:.2f} kWh")
    print(f"95% CI: [{forecast['lower_95']:.2f}, {forecast['upper_95']:.2f}]")
