import os
import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds calendar and cyclical time features to a datetime-indexed DataFrame."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['dayofyear'] = df.index.dayofyear
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    # Cyclical encoding for hour and month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Season flag: 0=Winter, 1=Spring, 2=Summer, 3=Autumn
    df['season'] = df['month'].map(
        lambda m: 0 if m in [12, 1, 2] else (1 if m in [3, 4, 5] else (2 if m in [6, 7, 8] else 3))
    )
    return df


def add_lag_features(df: pd.DataFrame, target_col: str, lags: list) -> pd.DataFrame:
    """Adds lag features for a target column."""
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str, windows: list) -> pd.DataFrame:
    """Adds rolling mean and std features for specified windows."""
    df = df.copy()
    for w in windows:
        df[f'rolling_mean_{w}'] = df[target_col].rolling(window=w, min_periods=1).mean()
        df[f'rolling_std_{w}'] = df[target_col].rolling(window=w, min_periods=1).std().fillna(0)
    return df


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    # ── Hourly features ──────────────────────────────────────────────────────
    h_df = pd.read_csv('data/cleaned_hourly.csv', parse_dates=['Datetime'], index_col='Datetime')
    h_df = add_time_features(h_df)
    h_df = add_lag_features(h_df, 'Global_active_power', [1, 2, 3, 6, 12, 24])
    h_df = add_rolling_features(h_df, 'Global_active_power', [6, 12, 24])
    h_df = h_df.dropna()
    h_df.to_csv('data/features_hourly.csv')
    print(f"Hourly features: {h_df.shape}")

    # ── Daily features ────────────────────────────────────────────────────────
    d_df = pd.read_csv('data/cleaned_daily.csv', parse_dates=['Datetime'], index_col='Datetime')
    d_df = add_time_features(d_df)
    d_df = add_lag_features(d_df, 'Global_active_power', [1, 2, 3, 7, 14, 30])
    d_df = add_rolling_features(d_df, 'Global_active_power', [7, 14, 30])
    d_df = d_df.dropna()
    d_df.to_csv('data/features_daily.csv')
    print(f"Daily features: {d_df.shape}")

    # ── Weekly features ───────────────────────────────────────────────────────
    w_df = pd.read_csv('data/cleaned_weekly.csv', parse_dates=['Datetime'], index_col='Datetime')
    w_df = add_time_features(w_df)
    w_df = add_lag_features(w_df, 'Global_active_power', [1, 2, 3, 4, 8, 12])
    w_df = add_rolling_features(w_df, 'Global_active_power', [4, 8, 12])
    w_df = w_df.dropna()
    w_df.to_csv('data/features_weekly.csv')
    print(f"Weekly features: {w_df.shape}")
