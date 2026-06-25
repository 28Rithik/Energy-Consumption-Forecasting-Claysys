import os
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

def evaluate_sarimax(series: pd.Series, order: tuple, val_steps: int) -> tuple[float, float]:
    """Evaluates SARIMAX performance on a validation split, returning MAE and RMSE."""
    train_seq = series.iloc[:-val_steps]
    val_seq = series.iloc[-val_steps:]
    
    model = SARIMAX(train_seq, order=order, seasonal_order=(0, 0, 0, 0))
    fit_model = model.fit(disp=False)
    preds = fit_model.forecast(steps=val_steps)
    
    mae = np.mean(np.abs(preds - val_seq))
    rmse = np.sqrt(np.mean((preds - val_seq) ** 2))
    return float(mae), float(rmse)

def generate_future_forecast(series: pd.Series, order: tuple, steps: int) -> pd.Series:
    """Trains SARIMAX on full dataset and projects forecasts into the future."""
    model = SARIMAX(series, order=order, seasonal_order=(0, 0, 0, 0))
    fit_model = model.fit(disp=False)
    return fit_model.forecast(steps=steps), fit_model

if __name__ == '__main__':
    # Load processed data
    h_df = pd.read_csv('data/cleaned_hourly.csv', parse_dates=['Datetime'], index_col='Datetime').asfreq('h')
    d_df = pd.read_csv('data/cleaned_daily.csv', parse_dates=['Datetime'], index_col='Datetime').asfreq('D')
    w_df = pd.read_csv('data/cleaned_weekly.csv', parse_dates=['Datetime'], index_col='Datetime').asfreq('W')
    
    # Train on recent windows to optimize speed
    h_series = h_df['Global_active_power'].tail(1000)
    d_series = d_df['Global_active_power'].tail(1000)
    w_series = w_df['Global_active_power']
    
    metrics = []
    
    # 1. Hourly (1 Day)
    h_mae, h_rmse = evaluate_sarimax(h_series, (1, 1, 1), 24)
    h_fc, h_model = generate_future_forecast(h_series, (1, 1, 1), 24)
    pd.DataFrame({'Datetime': h_fc.index, 'Global_active_power': h_fc.values}).to_csv('data/forecast_stats_1day.csv', index=False)
    joblib.dump(h_model, 'models/stats_hourly.joblib')
    metrics.append({'Horizon': '1day', 'MAE': h_mae, 'RMSE': h_rmse})
    
    # 2. Daily (10 Days)
    d_mae, d_rmse = evaluate_sarimax(d_series, (1, 1, 1), 10)
    d_fc, d_model = generate_future_forecast(d_series, (1, 1, 1), 10)
    pd.DataFrame({'Datetime': d_fc.index, 'Global_active_power': d_fc.values}).to_csv('data/forecast_stats_10day.csv', index=False)
    joblib.dump(d_model, 'models/stats_daily.joblib')
    metrics.append({'Horizon': '10day', 'MAE': d_mae, 'RMSE': d_rmse})
    
    # 3. Weekly (1 Year)
    w_mae, w_rmse = evaluate_sarimax(w_series, (1, 1, 1), 52)
    w_fc, w_model = generate_future_forecast(w_series, (1, 1, 1), 52)
    pd.DataFrame({'Datetime': w_fc.index, 'Global_active_power': w_fc.values}).to_csv('data/forecast_stats_1year.csv', index=False)
    joblib.dump(w_model, 'models/stats_weekly.joblib')
    metrics.append({'Horizon': '1year', 'MAE': w_mae, 'RMSE': w_rmse})
    
    # Save performance metrics
    pd.DataFrame(metrics).to_csv('data/metrics_stats.csv', index=False)
