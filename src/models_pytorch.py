import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    """Custom PyTorch Dataset for building sliding sequence window slices."""
    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return x, y

class LSTMRegressor(nn.Module):
    """LSTM regressor model with a single hidden LSTM layer followed by a linear mapping."""
    def __init__(self, input_size: int = 1, hidden_size: int = 16, num_layers: int = 1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = self.linear(lstm_out[:, -1, :])
        return out

def train_lstm(model: nn.Module, data: np.ndarray, seq_len: int, epochs: int = 5, lr: float = 0.01) -> nn.Module:
    """Trains the LSTM Regressor using MSE Loss and the Adam Optimizer."""
    dataset = TimeSeriesDataset(data, seq_len)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
    return model

def forecast_autoregressive(model: nn.Module, seed_seq: list, seq_len: int, steps: int) -> list:
    """Performs autoregressive forecast loop projecting predictions step-by-step."""
    model.eval()
    curr_seq = list(seed_seq)
    predictions = []
    
    for _ in range(steps):
        input_tensor = torch.tensor(curr_seq, dtype=torch.float32).view(1, seq_len, 1)
        with torch.no_grad():
            pred = model(input_tensor).item()
        predictions.append(pred)
        curr_seq.append(pred)
        curr_seq.pop(0)
    return predictions

if __name__ == '__main__':
    # Load processed data
    h_df = pd.read_csv('data/cleaned_hourly.csv', parse_dates=['Datetime'])
    d_df = pd.read_csv('data/cleaned_daily.csv', parse_dates=['Datetime'])
    w_df = pd.read_csv('data/cleaned_weekly.csv', parse_dates=['Datetime'])
    
    horizons_configs = [
        {'name': '1day', 'df': h_df, 'seq_len': 24, 'steps': 24, 'freq': 'h', 'train_tail': 2000, 'save_name': 'hourly'},
        {'name': '10day', 'df': d_df, 'seq_len': 30, 'steps': 10, 'freq': 'D', 'train_tail': 1000, 'save_name': 'daily'},
        {'name': '1year', 'df': w_df, 'seq_len': 10, 'steps': 52, 'freq': 'W', 'train_tail': None, 'save_name': 'weekly'}
    ]
    
    metrics = []
    
    for config in horizons_configs:
        df = config['df']
        seq_len = config['seq_len']
        steps = config['steps']
        freq = config['freq']
        
        # Filter training slice for hourly/daily to speed up
        if config['train_tail']:
            df_slice = df.tail(config['train_tail'])
        else:
            df_slice = df
            
        vals = df_slice['Global_active_power'].values
        min_val = vals.min()
        max_val = vals.max()
        scaled = (vals - min_val) / (max_val - min_val + 1e-8)
        
        # 1. Validation Split Evaluation
        # We hold out the last 'steps' data points for validation
        train_scaled = scaled[:-steps]
        val_scaled = scaled[-steps:]
        
        val_model = LSTMRegressor()
        val_model = train_lstm(val_model, train_scaled, seq_len)
        
        # Seed sequence is the last segment of the training scaled sequence
        seed_seq_val = train_scaled[-seq_len:]
        val_preds_scaled = forecast_autoregressive(val_model, seed_seq_val, seq_len, steps)
        val_preds = [p * (max_val - min_val) + min_val for p in val_preds_scaled]
        val_targets = df_slice['Global_active_power'].values[-steps:]
        
        mae = np.mean(np.abs(np.array(val_preds) - val_targets))
        rmse = np.sqrt(np.mean((np.array(val_preds) - val_targets) ** 2))
        metrics.append({'Horizon': config['name'], 'MAE': float(mae), 'RMSE': float(rmse)})
        
        # 2. Train on full series and forecast future
        full_model = LSTMRegressor()
        full_model = train_lstm(full_model, scaled, seq_len)
        
        seed_seq_full = scaled[-seq_len:]
        fc_preds_scaled = forecast_autoregressive(full_model, seed_seq_full, seq_len, steps)
        fc_preds = [p * (max_val - min_val) + min_val for p in fc_preds_scaled]
        
        # Datetime construction
        last_dt = pd.to_datetime(df['Datetime'].iloc[-1])
        if freq == 'h':
            future_dts = pd.date_range(start=last_dt + pd.Timedelta(hours=1), periods=steps, freq='h')
        elif freq == 'D':
            future_dts = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=steps, freq='D')
        else:
            future_dts = pd.date_range(start=last_dt + pd.Timedelta(weeks=1), periods=steps, freq='W')
            
        fc_out = pd.DataFrame({'Datetime': future_dts, 'Global_active_power': fc_preds})
        fc_out.to_csv(f'data/forecast_pytorch_{config["name"]}.csv', index=False)
        
        torch.save({
            'state_dict': full_model.state_dict(),
            'seq_len': seq_len,
            'min_val': min_val,
            'max_val': max_val
        }, f'models/pytorch_{config["save_name"]}.pt')
        
    pd.DataFrame(metrics).to_csv('data/metrics_pytorch.csv', index=False)
