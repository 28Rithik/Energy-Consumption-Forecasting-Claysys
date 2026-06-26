import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for LSTM training."""

    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len].unsqueeze(-1)  # (T, 1)
        y = self.data[idx + self.seq_len]                       # scalar
        return x, y


class AttentionLayer(nn.Module):
    """
    Scaled-dot-product temporal attention.
    Weights each timestep of the LSTM output and produces a context vector.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """lstm_out: (B, T, H*2)  ->  context: (B, H*2)"""
        scores  = self.attn(lstm_out)               # (B, T, 1)
        weights = torch.softmax(scores, dim=1)       # (B, T, 1)
        context = (weights * lstm_out).sum(dim=1)   # (B, H*2)
        return context


class PowerfulLSTMRegressor(nn.Module):
    """
    Advanced LSTM Regressor with:
      - 2-layer Bidirectional LSTM  (captures both past and future context)
      - Temporal self-attention     (focuses on the most informative timesteps)
      - LayerNorm                   (stabilises training)
      - 2-layer MLP decoder with GELU activation + Dropout
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AttentionLayer(hidden_size)
        self.norm      = nn.LayerNorm(hidden_size * 2)
        self.dropout   = nn.Dropout(dropout)
        self.fc1       = nn.Linear(hidden_size * 2, hidden_size)
        self.act       = nn.GELU()
        self.fc2       = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)            # (B, T, H*2)
        context     = self.attention(lstm_out) # (B, H*2)
        context     = self.norm(context)
        context     = self.dropout(context)
        out = self.act(self.fc1(context))      # (B, H)
        out = self.dropout(out)
        return self.fc2(out)                   # (B, 1)


def train_model(model: nn.Module, data: np.ndarray, seq_len: int,
                epochs: int = 20, lr: float = 5e-3,
                batch_size: int = 256) -> nn.Module:
    """
    Trains the model using:
      - MSELoss
      - Adam with weight decay
      - CosineAnnealingLR scheduling
      - Gradient norm clipping
    """
    dataset   = TimeSeriesDataset(data, seq_len)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x).squeeze(-1)
            loss  = criterion(preds, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            avg = epoch_loss / len(loader)
            print(f"  Epoch {epoch+1:02d}/{epochs}  loss={avg:.6f}")
    return model


def forecast_autoregressive(model: nn.Module, seed: list,
                             seq_len: int, steps: int) -> list:
    """Generates future predictions step-by-step (closed-loop autoregressive)."""
    model.eval()
    cur: list = list(seed)
    preds: list = []
    with torch.no_grad():
        for _ in range(steps):
            inp = torch.tensor(cur[-seq_len:], dtype=torch.float32).view(1, seq_len, 1)
            p   = model(inp).item()
            preds.append(p)
            cur.append(p)
    return preds


def run_horizon(clean_file: str, seq_len: int, steps: int,
                freq: str, name: str,
                train_tail: int = None) -> tuple:
    """
    Full pipeline for one horizon:
      load -> scale -> validate -> retrain-full -> forecast -> inverse-scale.
    Returns (forecast_df, mae, rmse).
    """
    df   = pd.read_csv(clean_file, parse_dates=['Datetime'], index_col='Datetime')
    vals = df['Global_active_power'].values.astype(np.float32)
    if train_tail:
        vals = vals[-train_tail:]

    v_min, v_max = vals.min(), vals.max()
    scaled = (vals - v_min) / (v_max - v_min + 1e-8)

    # Validation split
    train_sc = scaled[:-steps]
    val_sc   = scaled[-steps:]

    val_model = PowerfulLSTMRegressor()
    val_model = train_model(val_model, train_sc, seq_len)

    val_preds_sc  = np.array(forecast_autoregressive(val_model, list(train_sc[-seq_len:]), seq_len, steps))
    val_preds_kw  = val_preds_sc * (v_max - v_min) + v_min
    val_true_kw   = val_sc       * (v_max - v_min) + v_min
    mae  = float(np.mean(np.abs(val_preds_kw  - val_true_kw)))
    rmse = float(np.sqrt(np.mean((val_preds_kw - val_true_kw) ** 2)))

    # Retrain on full series for final forecast
    full_model = PowerfulLSTMRegressor()
    full_model = train_model(full_model, scaled, seq_len)

    fc_sc  = np.array(forecast_autoregressive(full_model, list(scaled[-seq_len:]), seq_len, steps))
    fc_kw  = fc_sc * (v_max - v_min) + v_min

    last_dt = df.index[-1]
    if freq == 'h':
        future_dts = pd.date_range(start=last_dt + pd.Timedelta(hours=1), periods=steps, freq='h')
    elif freq == 'D':
        future_dts = pd.date_range(start=last_dt + pd.Timedelta(days=1),  periods=steps, freq='D')
    else:
        future_dts = pd.date_range(start=last_dt + pd.Timedelta(weeks=1), periods=steps, freq='W')

    fc_df = pd.DataFrame({'Datetime': future_dts, 'Global_active_power': fc_kw})
    return fc_df, mae, rmse


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    metrics = []

    print("=== 1-Day  (hourly, seq=48) ===")
    h_fc, h_mae, h_rmse = run_horizon('data/cleaned_hourly.csv', seq_len=48, steps=24, freq='h', name='1day', train_tail=5000)
    h_fc.to_csv('data/forecast_pytorch_1day.csv', index=False)
    metrics.append({'Horizon': '1day',  'MAE': h_mae, 'RMSE': h_rmse})
    print(f"  -> MAE={h_mae:.4f}  RMSE={h_rmse:.4f}\n")

    print("=== 10-Day (daily,  seq=60) ===")
    d_fc, d_mae, d_rmse = run_horizon('data/cleaned_daily.csv',  seq_len=60, steps=10, freq='D', name='10day')
    d_fc.to_csv('data/forecast_pytorch_10day.csv', index=False)
    metrics.append({'Horizon': '10day', 'MAE': d_mae, 'RMSE': d_rmse})
    print(f"  -> MAE={d_mae:.4f}  RMSE={d_rmse:.4f}\n")

    print("=== 1-Year (weekly, seq=26) ===")
    w_fc, w_mae, w_rmse = run_horizon('data/cleaned_weekly.csv', seq_len=26, steps=52, freq='W', name='1year')
    w_fc.to_csv('data/forecast_pytorch_1year.csv', index=False)
    metrics.append({'Horizon': '1year', 'MAE': w_mae, 'RMSE': w_rmse})
    print(f"  -> MAE={w_mae:.4f}  RMSE={w_rmse:.4f}\n")

    pd.DataFrame(metrics).to_csv('data/metrics_pytorch.csv', index=False)
    print("All improved LSTM forecasts saved.")
