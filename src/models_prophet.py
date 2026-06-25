import os
import pandas as pd
import numpy as np
from prophet import Prophet
import joblib

def evaluate_prophet(df: pd.DataFrame, val_steps: int, freq: str) -> tuple[float, float]:
    """Fits Prophet on training data and computes MAE and RMSE on a validation window."""
    train_seq = df.iloc[:-val_steps]
    val_seq = df.iloc[-val_steps:]
    
    # Format for Prophet
    train_prophet = train_seq[['Datetime', 'Global_active_power']].rename(columns={'Datetime': 'ds', 'Global_active_power': 'y'})
    val_prophet = val_seq[['Datetime', 'Global_active_power']].rename(columns={'Datetime': 'ds', 'Global_active_power': 'y'})
    
    model = Prophet()
    model.fit(train_prophet)
    
    future = model.make_future_dataframe(periods=val_steps, freq=freq)
    forecast = model.predict(future)
    
    # Align validation targets with forecast predictions
    val_forecast = forecast[forecast['ds'] >= val_prophet['ds'].min()][['ds', 'yhat']].merge(val_prophet, on='ds')
    
    mae = np.mean(np.abs(val_forecast['yhat'] - val_forecast['y']))
    rmse = np.sqrt(np.mean((val_forecast['yhat'] - val_forecast['y']) ** 2))
    return float(mae), float(rmse)

def generate_future_prophet(df: pd.DataFrame, periods: int, freq: str) -> pd.DataFrame:
    """Trains Prophet on full dataset and predicts future steps."""
    train_prophet = df[['Datetime', 'Global_active_power']].rename(columns={'Datetime': 'ds', 'Global_active_power': 'y'})
    
    model = Prophet()
    model.fit(train_prophet)
    
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    
    # Return future forecast only
    future_forecast = forecast[forecast['ds'] > train_prophet['ds'].max()][['ds', 'yhat']]
    future_forecast = future_forecast.rename(columns={'ds': 'Datetime', 'yhat': 'Global_active_power'})
    return future_forecast, model

if __name__ == '__main__':
    # Load processed data
    h_df = pd.read_csv('data/cleaned_hourly.csv', parse_dates=['Datetime'])
    d_df = pd.read_csv('data/cleaned_daily.csv', parse_dates=['Datetime'])
    w_df = pd.read_csv('data/cleaned_weekly.csv', parse_dates=['Datetime'])
    
    # Train on recent windows to optimize speed
    h_train_data = h_df.tail(1000)
    d_train_data = d_df.tail(1000)
    w_train_data = w_df
    
    metrics = []
    
    # 1. Hourly (1 Day)
    h_mae, h_rmse = evaluate_prophet(h_train_data, 24, 'h')
    h_fc, h_model = generate_future_prophet(h_train_data, 24, 'h')
    h_fc.to_csv('data/forecast_prophet_1day.csv', index=False)
    joblib.dump(h_model, 'models/prophet_hourly.joblib')
    metrics.append({'Horizon': '1day', 'MAE': h_mae, 'RMSE': h_rmse})
    
    # 2. Daily (10 Days)
    d_mae, d_rmse = evaluate_prophet(d_train_data, 10, 'D')
    d_fc, d_model = generate_future_prophet(d_train_data, 10, 'D')
    d_fc.to_csv('data/forecast_prophet_10day.csv', index=False)
    joblib.dump(d_model, 'models/prophet_daily.joblib')
    metrics.append({'Horizon': '10day', 'MAE': d_mae, 'RMSE': d_rmse})
    
    # 3. Weekly (1 Year)
    w_mae, w_rmse = evaluate_prophet(w_train_data, 52, 'W')
    w_fc, w_model = generate_future_prophet(w_train_data, 52, 'W')
    w_fc.to_csv('data/forecast_prophet_1year.csv', index=False)
    joblib.dump(w_model, 'models/prophet_weekly.joblib')
    metrics.append({'Horizon': '1year', 'MAE': w_mae, 'RMSE': w_rmse})
    
    # Save performance metrics
    pd.DataFrame(metrics).to_csv('data/metrics_prophet.csv', index=False)
