import json
import math
import random
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ----------------------------
# Configuration
# ----------------------------
lookback = 60   # timesteps per sequence
horizon = 5     # steps to predict ahead (autoregressive chunk size)
batch_size = 64
train_frac = 0.8
test_frac = 0.2
max_epochs = 100
patience = 10
lr = 1e-3
hidden = 64
layers = 2
dropout = 0.2
save_csv_alongside_json = True   # also save CSVs next to JSONs

TICKER_FILES = {
    "NVDA": "nvda_30min.json",
    "MSFT": "msft_30min.json",
    "TXN" : "txn_30min.json",
    "AAPL": "aapl_30min.json",
}

# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Model
# ----------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, hidden=64, layers=2, dropout=0.2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ----------------------------
# Helpers
# ----------------------------
def make_sequences(arr, lookback, horizon):
    X, y = [], []
    for i in range(lookback, len(arr) - horizon + 1):
        X.append(arr[i - lookback : i, 0])
        y.append(arr[i : i + horizon, 0])
    X = np.array(X)[:, :, None]          # (N, lookback, 1)
    y = np.array(y)                       # (N, horizon)
    return X, y

def save_prediction_db(ticker, test_times, x_pred, y_true_test, pred_recursive,
                       mae, rmse, mape, granularity="30min"):
    """Write per-ticker prediction database to JSON (and optional CSV)."""
    records = []
    for i, (ts, xi, pred, act) in enumerate(zip(test_times, x_pred,
                                                pred_recursive.reshape(-1),
                                                y_true_test.reshape(-1))):
        records.append({
            "ticker": ticker,
            "index_in_test": i,
            "x_index": int(xi),
            "iso_time": ts.isoformat(),          # NY-local ISO8601 with offset
            "pred_close": float(pred),
            "actual_close": float(act),
        })

    db = {
        "meta": {
            "ticker": ticker,
            "granularity": granularity,
            "lookback": lookback,
            "horizon": horizon,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "metrics": {"MAE": float(mae), "RMSE": float(rmse), "MAPE_pct": float(mape)}
        },
        "predictions": records
    }

    json_path = f"pred_{ticker}_{granularity}.json"
    with open(json_path, "w") as f:
        json.dump(db, f, indent=2)

    csv_path = None
    if save_csv_alongside_json:
        csv_path = f"pred_{ticker}_{granularity}.csv"
        pd.DataFrame(records).to_csv(csv_path, index=False)

    print(f"[{ticker}] Saved prediction DB -> {json_path} ({len(records)} rows)")
    if csv_path:
        print(f"[{ticker}] Saved CSV -> {csv_path}")

    return json_path, csv_path

def train_one_ticker(times_ny, close_prices, ticker):
    # Sort by time
    series = pd.Series(close_prices, index=times_ny).sort_index()
    values = series.values.reshape(-1, 1)
    sorted_times = series.index
    print(f"\n[{ticker}] Loaded {len(values)} 30-min points.")

    # Build x-axis positions and gap lines (visual only)
    gap_threshold = timedelta(minutes=35)
    x_positions = [0]
    gap_locations = []
    for i in range(1, len(sorted_times)):
        time_diff = sorted_times[i] - sorted_times[i - 1]
        x_positions.append(x_positions[-1] + 1)  # advance by 1 always
        if time_diff > gap_threshold:
            gap_locations.append(x_positions[-1])

    # Train/test split indices
    n = len(values)
    train_end_index = int(math.ceil(train_frac * n))
    test_end_index  = int(math.ceil((train_frac + test_frac) * n))
    train_vals = values[:train_end_index]
    test_vals  = values[train_end_index - lookback : test_end_index]

    # Scale fit on train only
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_vals)
    test_scaled  = scaler.transform(test_vals)

    # Supervised sets
    X_train, y_train = make_sequences(train_scaled, lookback, horizon)
    X_test,  y_test  = make_sequences(test_scaled,  lookback, horizon)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False
    )

    # Build & train model
    model = LSTMForecaster(hidden=hidden, layers=layers, dropout=dropout, output_size=horizon).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    no_improve = 0
    best_path = f"best_lstm_{ticker}.pt"

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(test_loader.dataset)

        print(f"[{ticker}] Epoch {epoch:03d} | train MSE={train_loss:.6f} | val MSE={val_loss:.6f}")

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{ticker}] Early stopping.")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    # ----------------------------
    # Autoregressive (recursive) forecast over test window
    # ----------------------------
    y_true_test = values[train_end_index:test_end_index]  # (T_test, 1)
    num_test_pts = len(y_true_test)

    recursive_scaled = []
    with torch.no_grad():
        current_input = torch.tensor(train_scaled[-lookback:], dtype=torch.float32)\
                              .reshape(1, lookback, 1).to(device)
        for _ in range(int(math.ceil(num_test_pts / horizon))):
            step_pred = model(current_input)              # (1, horizon)
            recursive_scaled.extend(step_pred.cpu().squeeze().tolist())
            old_seq = current_input.squeeze(0)[horizon:]  # (lookback - horizon, 1)
            new_seq = step_pred.detach().T                # (horizon, 1)
            current_input = torch.cat((old_seq, new_seq), dim=0).unsqueeze(0)

    recursive_scaled = np.array(recursive_scaled[:num_test_pts]).reshape(-1, 1)
    pred_recursive = scaler.inverse_transform(recursive_scaled)

    # Metrics
    mae  = mean_absolute_error(y_true_test, pred_recursive)
    rmse = np.sqrt(mean_squared_error(y_true_test, pred_recursive))
    mape = np.mean(np.abs((y_true_test - pred_recursive) / y_true_test)) * 100.0
    print(f"[{ticker}] Recursive {horizon}-Step Test: MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape:.2f}%")

    # Build x-axis for predictions on test region only
    x_pos = x_positions
    x_test_start = train_end_index
    x_test_end   = train_end_index + num_test_pts
    x_pred = x_pos[x_test_start:x_test_end]

    # SAVE prediction database (JSON + optional CSV)
    test_times = list(sorted_times[x_test_start:x_test_end])
    json_path, csv_path = save_prediction_db(
        ticker=ticker,
        test_times=test_times,
        x_pred=x_pred,
        y_true_test=y_true_test,
        pred_recursive=pred_recursive,
        mae=mae, rmse=rmse, mape=mape,
        granularity="30min"
    )

    # Pack results for plotting and index
    return {
        "ticker": ticker,
        "times": sorted_times,
        "x_positions": x_pos,
        "gap_locations": gap_locations,
        "actual_full": values.reshape(-1),
        "x_pred": x_pred,
        "pred_recursive": pred_recursive.reshape(-1),
        "metrics": (mae, rmse, mape),
        "pred_json_path": json_path,
        "pred_csv_path": csv_path,
    }

# ----------------------------
# Load all files, train per ticker, collect results
# ----------------------------
all_results = []
for ticker, fname in TICKER_FILES.items():
    with open(fname, "r") as f:
        raw = json.load(f)
    # Use NY-local timestamps already in your files; these are offset-aware
    times_ny = [datetime.fromisoformat(entry["iso_ny"]) for entry in raw]
    close_prices = [entry["c"] for entry in raw]
    res = train_one_ticker(times_ny, close_prices, ticker)
    all_results.append(res)

# ----------------------------
# Write a small index of saved databases (handy for downstream jobs)
# ----------------------------
pred_index = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "granularity": "30min",
    "lookback": lookback,
    "horizon": horizon,
    "tickers": []
}
for r in all_results:
    m_mae, m_rmse, m_mape = r["metrics"]
    pred_index["tickers"].append({
        "ticker": r["ticker"],
        "pred_json_path": r["pred_json_path"],
        "pred_csv_path": r["pred_csv_path"],
        "metrics": {"MAE": float(m_mae), "RMSE": float(m_rmse), "MAPE_pct": float(m_mape)}
    })

with open("pred_index.json", "w") as f:
    json.dump(pred_index, f, indent=2)
print("Saved prediction index -> pred_index.json")

# ----------------------------
# Plot: one figure, all tickers overlaid
# ----------------------------
fig, ax = plt.subplots(figsize=(16, 8))

# Assign a distinct color per ticker
color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
ticker_colors = {}
for (res, col) in zip(all_results, color_cycle):
    ticker_colors[res["ticker"]] = col

# Choose a reference series (longest) for x-ticks/labels
ref_idx = int(np.argmax([len(r["x_positions"]) for r in all_results]))
ref_res = all_results[ref_idx]

# Plot all tickers: solid = actual, dotted = AR forecast
for res in all_results:
    ticker = res["ticker"]
    col = ticker_colors[ticker]

    # Actual (solid)
    ax.plot(
        res["x_positions"],
        res["actual_full"],
        linewidth=1.8,
        color=col,
        label=f"{ticker} Actual"
    )

    # Autoregressive forecast (dotted)
    ax.plot(
        res["x_pred"],
        res["pred_recursive"],
        linestyle=":",
        linewidth=2.2,
        color=col,
        label=f"{ticker} AR"
    )

# Optional: VERY faint trading gap lines (from reference series only to reduce clutter)
for gx in ref_res["gap_locations"]:
    ax.axvline(x=gx, color='gray', linestyle=':', alpha=0.12, linewidth=0.8)

# X ticks from the reference ticker
times_ref = ref_res["times"]
xpos_ref = ref_res["x_positions"]
tick_idx = np.linspace(0, len(xpos_ref) - 1, num=10, dtype=int)
ax.set_xticks([xpos_ref[i] for i in tick_idx])
ax.set_xticklabels([times_ref[i].strftime('%Y-%m-%d') for i in tick_idx], rotation=30, ha='right')

ax.set_title(f"30-min LSTM Autoregressive ({horizon}-step) Forecasts — All Tickers Overlaid", fontsize=16)
ax.set_xlabel("Trading Period Index (30-min intervals)", fontsize=12)
ax.set_ylabel("Price (USD)", fontsize=12)
ax.margins(x=0.01)

# Make legend compact
ax.legend(ncol=2, loc="upper left", frameon=False)

plt.tight_layout()
plt.show()
