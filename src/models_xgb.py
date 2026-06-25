import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib


# ── Feature columns used for training ────────────────────────────────────────
LAG_COLS_HOURLY = ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'lag_24',
                   'rolling_mean_6', 'rolling_mean_12', 'rolling_mean_24',
                   'rolling_std_6', 'rolling_std_12', 'rolling_std_24',
                   'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                   'dayofweek', 'is_weekend', 'season']

LAG_COLS_DAILY = ['lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_30',
                  'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30',
                  'rolling_std_7', 'rolling_std_14', 'rolling_std_30',
                  'month_sin', 'month_cos', 'dayofweek', 'is_weekend', 'season']

LAG_COLS_WEEKLY = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_8', 'lag_12',
                   'rolling_mean_4', 'rolling_mean_8', 'rolling_mean_12',
                   'rolling_std_4', 'rolling_std_8', 'rolling_std_12',
                   'month_sin', 'month_cos', 'season']


def train_ensemble(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """Trains XGBoost and RandomForest models and returns the fitted pair."""
    xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    return xgb, rf


def ensemble_predict(xgb, rf, X: np.ndarray) -> np.ndarray:
    """Returns the averaged prediction from XGBoost and RandomForest."""
    return (xgb.predict(X) + rf.predict(X)) / 2.0


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Computes MAE and RMSE."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return mae, rmse


def build_forecast_features(last_row: pd.Series, step: int,
                             future_dt: pd.Timestamp,
                             feat_cols: list,
                             recent_preds: list) -> np.ndarray:
    """Constructs a single-row feature vector for the next forecast step."""
    row = last_row.copy()

    # Update lag features with previously predicted values
    n_preds = len(recent_preds)
    for i, col in enumerate([c for c in feat_cols if c.startswith('lag_')]):
        lag_idx = n_preds - int(col.split('_')[1])
        row[col] = recent_preds[lag_idx] if lag_idx >= 0 else last_row[col]

    # Update calendar features
    row['hour'] = future_dt.hour if hasattr(future_dt, 'hour') else 0
    row['dayofweek'] = future_dt.dayofweek
    row['month'] = future_dt.month
    row['is_weekend'] = int(future_dt.dayofweek >= 5)
    row['season'] = (0 if future_dt.month in [12, 1, 2]
                     else (1 if future_dt.month in [3, 4, 5]
                           else (2 if future_dt.month in [6, 7, 8] else 3)))
    row['hour_sin'] = np.sin(2 * np.pi * row['hour'] / 24)
    row['hour_cos'] = np.cos(2 * np.pi * row['hour'] / 24)
    row['month_sin'] = np.sin(2 * np.pi * row['month'] / 12)
    row['month_cos'] = np.cos(2 * np.pi * row['month'] / 12)

    return row[feat_cols].values.reshape(1, -1)


def run_horizon(feat_file: str, feat_cols: list, steps: int,
                freq: str, name: str, val_frac: float = 0.1):
    """Full pipeline: load → split → train → validate → forecast for one horizon."""
    df = pd.read_csv(feat_file, parse_dates=['Datetime'], index_col='Datetime')
    target = 'Global_active_power'

    val_size = max(steps, int(len(df) * val_frac))
    train_df = df.iloc[:-val_size]
    val_df = df.iloc[-val_size:]

    X_train = train_df[feat_cols].values
    y_train = train_df[target].values
    X_val = val_df[feat_cols].values
    y_val = val_df[target].values

    xgb, rf = train_ensemble(X_train, y_train)
    val_preds = ensemble_predict(xgb, rf, X_val)
    mae, rmse = evaluate_metrics(y_val, val_preds)

    # Retrain on full dataset for future forecast
    xgb_full, rf_full = train_ensemble(df[feat_cols].values, df[target].values)

    last_row = df[feat_cols].iloc[-1]
    last_dt = df.index[-1]
    if freq == 'h':
        future_dates = pd.date_range(start=last_dt + pd.Timedelta(hours=1), periods=steps, freq='h')
    elif freq == 'D':
        future_dates = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=steps, freq='D')
    else:
        future_dates = pd.date_range(start=last_dt + pd.Timedelta(weeks=1), periods=steps, freq='W')

    recent_preds = list(df[target].values[-max([int(c.split('_')[1]) for c in feat_cols if c.startswith('lag_')], default=1):])
    fc_preds = []
    cur_row = last_row.copy()

    for i, dt in enumerate(future_dates):
        X_next = build_forecast_features(cur_row, i, dt, feat_cols, recent_preds)
        pred = float(ensemble_predict(xgb_full, rf_full, X_next)[0])
        fc_preds.append(pred)
        recent_preds.append(pred)

    fc_df = pd.DataFrame({'Datetime': future_dates, 'Global_active_power': fc_preds})
    return fc_df, mae, rmse, (xgb_full, rf_full)


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    metrics = []

    h_fc, h_mae, h_rmse, h_models = run_horizon('data/features_hourly.csv', LAG_COLS_HOURLY, 24, 'h', '1day')
    h_fc.to_csv('data/forecast_xgb_1day.csv', index=False)
    joblib.dump(h_models, 'models/xgb_hourly.joblib')
    metrics.append({'Horizon': '1day', 'MAE': h_mae, 'RMSE': h_rmse})
    print(f"[XGB] 1day — MAE: {h_mae:.4f}  RMSE: {h_rmse:.4f}")

    d_fc, d_mae, d_rmse, d_models = run_horizon('data/features_daily.csv', LAG_COLS_DAILY, 10, 'D', '10day')
    d_fc.to_csv('data/forecast_xgb_10day.csv', index=False)
    joblib.dump(d_models, 'models/xgb_daily.joblib')
    metrics.append({'Horizon': '10day', 'MAE': d_mae, 'RMSE': d_rmse})
    print(f"[XGB] 10day — MAE: {d_mae:.4f}  RMSE: {d_rmse:.4f}")

    w_fc, w_mae, w_rmse, w_models = run_horizon('data/features_weekly.csv', LAG_COLS_WEEKLY, 52, 'W', '1year')
    w_fc.to_csv('data/forecast_xgb_1year.csv', index=False)
    joblib.dump(w_models, 'models/xgb_weekly.joblib')
    metrics.append({'Horizon': '1year', 'MAE': w_mae, 'RMSE': w_rmse})
    print(f"[XGB] 1year — MAE: {w_mae:.4f}  RMSE: {w_rmse:.4f}")

    pd.DataFrame(metrics).to_csv('data/metrics_xgb.csv', index=False)
    print("XGBoost Ensemble forecasts saved.")
