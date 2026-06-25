import os
import pandas as pd
import numpy as np

def load_raw_data(file_path: str) -> pd.DataFrame:
    """Loads the raw semicolon-separated household power consumption file."""
    return pd.read_csv(file_path, sep=';', low_memory=False)

def clean_and_interpolate(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans columns, merges datetime, and interpolates missing values."""
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df = df.drop(columns=['Date', 'Time'])
    
    cols = ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.set_index('Datetime')
    df = df[cols]
    
    # Linear interpolation + forward/backward fill to guarantee zero NaNs
    df = df.interpolate(method='linear').ffill().bfill()
    return df

def detect_anomalies(df: pd.DataFrame, window: int = 30, threshold: float = 3.0) -> pd.DataFrame:
    """Detects active power anomalies using rolling Z-score method."""
    rolling_mean = df['Global_active_power'].rolling(window=window, min_periods=1).mean()
    rolling_std = df['Global_active_power'].rolling(window=window, min_periods=1).std()
    
    z_scores = (df['Global_active_power'] - rolling_mean) / (rolling_std + 1e-8)
    df['Anomaly'] = z_scores.abs() > threshold
    return df

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    
    # Run the processing pipeline
    raw_df = load_raw_data('data/household_power_consumption.txt')
    cleaned_df = clean_and_interpolate(raw_df)
    
    # Resample and detect anomalies at the respective scale
    hourly_df = cleaned_df.resample('h').mean()
    hourly_df = detect_anomalies(hourly_df, window=24, threshold=3.0)
    hourly_df.to_csv('data/cleaned_hourly.csv')

    daily_df = cleaned_df.resample('D').mean()
    daily_df = detect_anomalies(daily_df, window=30, threshold=3.0)
    daily_df.to_csv('data/cleaned_daily.csv')

    weekly_df = cleaned_df.resample('W').mean()
    weekly_df = detect_anomalies(weekly_df, window=10, threshold=3.0)
    weekly_df.to_csv('data/cleaned_weekly.csv')
