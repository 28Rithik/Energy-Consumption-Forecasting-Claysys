import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TFTDataset(Dataset):
    """Sliding-window dataset returning (encoder_seq, decoder_len, target)."""

    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx: idx + self.seq_len].unsqueeze(-1)   # (T, 1)
        y = self.data[idx + self.seq_len].unsqueeze(-1)        # (1,)
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Gated Residual Network block
# ─────────────────────────────────────────────────────────────────────────────

class GRN(nn.Module):
    """Gated Residual Network for non-linear variable selection."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        g = self.sigmoid(self.gate(x))
        return self.norm(x + g * h)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Fusion Transformer
# ─────────────────────────────────────────────────────────────────────────────

class TemporalFusionTransformer(nn.Module):
    """
    Simplified TFT:  input projection → LSTM encoder
                     → multi-head self-attention → GRN → linear output.
    """

    def __init__(self, input_size: int = 1, d_model: int = 32,
                 n_heads: int = 4, num_layers: int = 1,
                 seq_len: int = 24, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.grn = GRN(d_model, dropout)
        self.fc_out = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1)
        h = self.input_proj(x)                          # (B, T, d_model)
        h, _ = self.lstm(h)                             # (B, T, d_model)
        attn_out, _ = self.attn(h, h, h)                # (B, T, d_model)
        h = self.norm(h + attn_out)                     # residual
        h = self.grn(h[:, -1, :])                       # use last timestep
        return self.fc_out(h)                           # (B, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_tft(model: nn.Module, data: np.ndarray, seq_len: int,
              epochs: int = 10, lr: float = 5e-3, batch_size: int = 256) -> nn.Module:
    """Trains the TFT model with CosineAnnealing LR scheduling."""
    dataset = TFTDataset(data, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
    return model


def forecast_tft(model: nn.Module, seed: np.ndarray,
                 seq_len: int, steps: int,
                 data_min: float, data_max: float) -> list[float]:
    """Autoregressive forecasting loop, returning inverse-scaled predictions."""
    model.eval()
    cur = list(seed)
    preds = []
    with torch.no_grad():
        for _ in range(steps):
            inp = torch.tensor(cur[-seq_len:], dtype=torch.float32).view(1, seq_len, 1)
            p = model(inp).item()
            preds.append(p)
            cur.append(p)
    return [p * (data_max - data_min) + data_min for p in preds]


def evaluate_tft(model: nn.Module, data_scaled: np.ndarray, seq_len: int,
                 val_steps: int, data_min: float, data_max: float) -> tuple[float, float]:
    """Evaluates TFT on held-out validation steps."""
    seed = data_scaled[-(seq_len + val_steps): -val_steps]
    preds_scaled = forecast_tft(model, seed, seq_len, val_steps, data_min, data_max)
    targets = data_scaled[-val_steps:] * (data_max - data_min) + data_min
    preds_arr = np.array(preds_scaled)
    tgt_arr = np.array(targets)
    mae = float(np.mean(np.abs(preds_arr - tgt_arr)))
    rmse = float(np.sqrt(np.mean((preds_arr - tgt_arr) ** 2)))
    return mae, rmse


def run_tft_horizon(clean_file: str, seq_len: int, steps: int,
                    freq: str, name: str):
    """Full TFT pipeline for one horizon."""
    df = pd.read_csv(clean_file, parse_dates=['Datetime'], index_col='Datetime')
    vals = df['Global_active_power'].values.astype(np.float32)
    vmin, vmax = vals.min(), vals.max()
    scaled = (vals - vmin) / (vmax - vmin + 1e-8)

    # Train on full sequence (TFT is sample-efficient)
    model = TemporalFusionTransformer(seq_len=seq_len)
    model = train_tft(model, scaled, seq_len)

    # Validation on last `steps` slice
    mae, rmse = evaluate_tft(model, scaled, seq_len, steps, vmin, vmax)

    # Future forecast
    fc_vals = forecast_tft(model, scaled, seq_len, steps, vmin, vmax)
    last_dt = df.index[-1]
    if freq == 'h':
        future_dts = pd.date_range(start=last_dt + pd.Timedelta(hours=1), periods=steps, freq='h')
    elif freq == 'D':
        future_dts = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=steps, freq='D')
    else:
        future_dts = pd.date_range(start=last_dt + pd.Timedelta(weeks=1), periods=steps, freq='W')

    fc_df = pd.DataFrame({'Datetime': future_dts, 'Global_active_power': fc_vals})
    return fc_df, mae, rmse, model, vmin, vmax


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    metrics = []

    h_fc, h_mae, h_rmse, h_model, h_vmin, h_vmax = run_tft_horizon('data/cleaned_hourly.csv', 48, 24, 'h', '1day')
    h_fc.to_csv('data/forecast_tft_1day.csv', index=False)
    torch.save({'state_dict': h_model.state_dict(), 'seq_len': 48, 'min_val': h_vmin, 'max_val': h_vmax}, 'models/tft_hourly.pt')
    metrics.append({'Horizon': '1day', 'MAE': h_mae, 'RMSE': h_rmse})
    print(f"[TFT] 1day  — MAE: {h_mae:.4f}  RMSE: {h_rmse:.4f}")

    d_fc, d_mae, d_rmse, d_model, d_vmin, d_vmax = run_tft_horizon('data/cleaned_daily.csv', 60, 10, 'D', '10day')
    d_fc.to_csv('data/forecast_tft_10day.csv', index=False)
    torch.save({'state_dict': d_model.state_dict(), 'seq_len': 60, 'min_val': d_vmin, 'max_val': d_vmax}, 'models/tft_daily.pt')
    metrics.append({'Horizon': '10day', 'MAE': d_mae, 'RMSE': d_rmse})
    print(f"[TFT] 10day — MAE: {d_mae:.4f}  RMSE: {d_rmse:.4f}")

    w_fc, w_mae, w_rmse, w_model, w_vmin, w_vmax = run_tft_horizon('data/cleaned_weekly.csv', 20, 52, 'W', '1year')
    w_fc.to_csv('data/forecast_tft_1year.csv', index=False)
    torch.save({'state_dict': w_model.state_dict(), 'seq_len': 20, 'min_val': w_vmin, 'max_val': w_vmax}, 'models/tft_weekly.pt')
    metrics.append({'Horizon': '1year', 'MAE': w_mae, 'RMSE': w_rmse})
    print(f"[TFT] 1year — MAE: {w_mae:.4f}  RMSE: {w_rmse:.4f}")

    pd.DataFrame(metrics).to_csv('data/metrics_tft.csv', index=False)
    print("TFT Transformer forecasts saved.")
