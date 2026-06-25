import pandas as pd
import numpy as np
import joblib
import torch
import os

# Import model architectures and helper functions
from src.models_pytorch import LSTMRegressor, forecast_autoregressive
from src.models_tft import TemporalFusionTransformer, forecast_tft
from src.models_xgb import ensemble_predict, build_forecast_features

def predict_custom_horizon(model_type: str, freq_name: str, steps: int, hist_df: pd.DataFrame, feat_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generates a dynamic forecast using saved models.
    model_type: 'stats', 'prophet', 'xgb', 'pytorch', 'tft'
    freq_name: 'hourly', 'daily', 'weekly'
    steps: int
    hist_df: DataFrame of historical data (cleaned_*.csv)
    feat_df: DataFrame of features (features_*.csv) used by XGBoost
    """
    if freq_name == 'hourly':
        freq = 'h'
        last_dt = hist_df.index[-1]
        future_dates = pd.date_range(start=last_dt + pd.Timedelta(hours=1), periods=steps, freq=freq)
    elif freq_name == 'daily':
        freq = 'D'
        last_dt = hist_df.index[-1]
        future_dates = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=steps, freq=freq)
    else:
        freq = 'W'
        last_dt = hist_df.index[-1]
        future_dates = pd.date_range(start=last_dt + pd.Timedelta(weeks=1), periods=steps, freq=freq)
        
    fc_preds = []

    if model_type == 'stats':
        model = joblib.load(f'models/stats_{freq_name}.joblib')
        preds = model.forecast(steps=steps)
        fc_preds = preds.values

    elif model_type == 'prophet':
        model = joblib.load(f'models/prophet_{freq_name}.joblib')
        future = model.make_future_dataframe(periods=steps, freq=freq, include_history=False)
        forecast = model.predict(future)
        fc_preds = forecast['yhat'].values
        future_dates = forecast['ds'].values

    elif model_type == 'xgb':
        if feat_df is None:
            raise ValueError("Feature DataFrame is required for XGBoost.")
        models = joblib.load(f'models/xgb_{freq_name}.joblib')
        xgb_model, rf_model = models
        
        target = 'Global_active_power'
        feat_cols = [c for c in feat_df.columns if c != target]
        last_row = feat_df[feat_cols].iloc[-1]
        recent_preds = list(feat_df[target].values[-max([int(c.split('_')[1]) for c in feat_cols if c.startswith('lag_')], default=1):])
        
        cur_row = last_row.copy()
        for i, dt in enumerate(future_dates):
            X_next = build_forecast_features(cur_row, i, dt, feat_cols, recent_preds)
            pred = float(ensemble_predict(xgb_model, rf_model, X_next)[0])
            fc_preds.append(pred)
            recent_preds.append(pred)

    elif model_type == 'pytorch':
        checkpoint = torch.load(f'models/pytorch_{freq_name}.pt', weights_only=True)
        seq_len = checkpoint['seq_len']
        min_val = checkpoint['min_val']
        max_val = checkpoint['max_val']
        
        model = LSTMRegressor()
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        # Prepare seed
        vals = hist_df['Global_active_power'].values[-seq_len:]
        scaled_seed = (vals - min_val) / (max_val - min_val + 1e-8)
        
        fc_preds_scaled = forecast_autoregressive(model, list(scaled_seed), seq_len, steps)
        fc_preds = [p * (max_val - min_val) + min_val for p in fc_preds_scaled]

    elif model_type == 'tft':
        checkpoint = torch.load(f'models/tft_{freq_name}.pt', weights_only=True)
        seq_len = checkpoint['seq_len']
        min_val = checkpoint['min_val']
        max_val = checkpoint['max_val']
        
        model = TemporalFusionTransformer(seq_len=seq_len)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        vals = hist_df['Global_active_power'].values[-seq_len:]
        scaled_seed = (vals - min_val) / (max_val - min_val + 1e-8)
        
        fc_preds = forecast_tft(model, list(scaled_seed), seq_len, steps, min_val, max_val)

    fc_df = pd.DataFrame({'Datetime': future_dates, 'Global_active_power': fc_preds})
    fc_df = fc_df.set_index('Datetime')
    return fc_df
