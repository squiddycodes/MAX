import json
import math
import random
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
lookback = 60  # timesteps per sequence
horizon = 5    # timesteps to predict ahead
batch_size = 64
train_frac = 0.8 #ain on first 50%
test_frac = 0.2  # Test on next 20% (data from 50% to 70%)


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
# Load the data
# ----------------------------
with open("txn_30min.json", "r") as f:
    data = json.load(f)

times_ny = [datetime.fromisoformat(entry["iso_ny"]) for entry in data]
close_prices = [entry["c"] for entry in data]
series = pd.Series(close_prices, index=times_ny).sort_index()
values = series.values.reshape(-1, 1)
full_close_prices = series.values
sorted_times = series.index
print(f"Loaded {len(values)} 30-min data points.")


# ----------------------------
# Detect gaps and create continuous x-axis
# ----------------------------
gap_threshold = timedelta(minutes=35)
x_positions = [0]
gap_locations = []
for i in range(1, len(sorted_times)):
    time_diff = sorted_times[i] - sorted_times[i - 1]
    if time_diff > gap_threshold:
        x_positions.append(x_positions[-1] + 1)
        gap_locations.append(x_positions[-1])
    else:
        x_positions.append(x_positions[-1] + 1)
print(f"Detected {len(gap_locations)} trading gaps.")


# ----------------------------
# Split 50/20 (MODIFIED)
# ----------------------------
n = len(values)
train_end_index = int(math.ceil(train_frac * n))
test_end_index = int(math.ceil((train_frac + test_frac) * n))

train_vals = values[:train_end_index]
# Need to grab data starting 'lookback' steps *before* the split
test_vals = values[train_end_index - lookback : test_end_index]


# ----------------------------
# Scale (fit only on train)
# ----------------------------
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_vals)
test_scaled = scaler.transform(test_vals)


# ----------------------------
# Create supervised sequences (MODIFIED FOR HORIZON)
# ----------------------------
def make_sequences(arr, lookback, horizon):
    X, y = [], []
    # Stop loop 'horizon' steps earlier to get a full y-sequence
    for i in range(lookback, len(arr) - horizon + 1):
        X.append(arr[i - lookback : i, 0])
        y.append(arr[i : i + horizon, 0]) # y is now a sequence of 'horizon' length
    X = np.array(X)[:, :, None]
    y = np.array(y) # Shape will be (N, horizon)
    return X, y


X_train, y_train = make_sequences(train_scaled, lookback, horizon)
X_test, y_test = make_sequences(test_scaled, lookback, horizon)

# y_test_full is the actual, unscaled data we want to predict recursively
y_true_test_data = values[train_end_index:test_end_index]

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                  torch.tensor(y_train, dtype=torch.float32)),
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                  torch.tensor(y_test, dtype=torch.float32)),
    batch_size=batch_size,
    shuffle=False
)


# ----------------------------
# LSTM Model (MODIFIED FOR HORIZON)
# ----------------------------
class LSTMForecaster(nn.Module):
    # Add 'output_size'
    def __init__(self, hidden=64, layers=2, dropout=0.2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        # Output layer now predicts 'output_size' steps
        self.fc = nn.Linear(hidden, output_size) 

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# Pass in the new output_size
model = LSTMForecaster(hidden=64, layers=2, dropout=0.2, output_size=horizon).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ----------------------------
# Training loop (No changes)
# ----------------------------
best_val, patience, counter = float("inf"), 10, 0
for epoch in range(1, 101):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred_logits = model(xb)
        loss = criterion(pred_logits, yb)
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

    print(f"Epoch {epoch:03d} | train MSE={train_loss:.6f} | val MSE={val_loss:.6f}")

    if val_loss < best_val - 1e-8:
        best_val = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_lstm.pt")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

model.load_state_dict(torch.load("best_lstm.pt", map_location=device))
model.eval()


# ----------------------------------------------------
# Evaluation 1: "One-Step" (Sliding) 5-Step Forecast
# ----------------------------------------------------
def predict_sliding(loader):
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred_logits = model(xb.to(device)).cpu().numpy()
            preds.append(pred_logits)
            trues.append(yb.numpy())
    preds = np.vstack(preds) # Shape (N, 5)
    trues = np.vstack(trues) # Shape (N, 5)
    return preds, trues

pred_scaled, y_scaled = predict_sliding(test_loader)
pred_sliding = scaler.inverse_transform(pred_scaled)
y_true_sliding = scaler.inverse_transform(y_scaled)

# Compare the 5th predicted step to the 5th actual step
mae_sliding = mean_absolute_error(y_true_sliding[:, -1], pred_sliding[:, -1])
rmse_sliding = np.sqrt(mean_squared_error(y_true_sliding[:, -1], pred_sliding[:, -1]))
mape_sliding = np.mean(np.abs((y_true_sliding[:, -1] - pred_sliding[:, -1]) / y_true_sliding[:, -1])) * 100
print(f"\nSliding 5-Step Test (MAE={mae_sliding:.4f} | RMSE={rmse_sliding:.4f} | MAPE={mape_sliding:.2f}%)")


# ----------------------------------------------------
# Evaluation 2: Recursive 5-Step Forecast
# ----------------------------------------------------
recursive_preds_scaled = []
# Get the last 60 points from the training set as the initial input
current_input = torch.tensor(train_scaled[-lookback:], dtype=torch.float32).reshape(1, lookback, 1).to(device)
num_test_points = len(y_true_test_data)

print("Starting 5-step recursive forecast...")
with torch.no_grad():
    # Loop, predicting 5 steps at a time
    for _ in range(int(math.ceil(num_test_points / horizon))):
        # 1. Get 5-step prediction (shape: [1, 5])
        pred_scaled_batch = model(current_input)
        
        # 2. Store all 5 scalar values
        recursive_preds_scaled.extend(pred_scaled_batch.cpu().squeeze().tolist())
        
        # 3. Prepare next input
        old_seq_part = current_input.squeeze(0)[horizon:] # Shape: [55, 1]
        
        # 4. New part is the 5 predictions (shape [5, 1])
        new_seq_part = pred_scaled_batch.detach().T 
        new_input_seq = torch.cat((old_seq_part, new_seq_part), dim=0)
        
        # 5. Update current_input for the next loop
        current_input = new_input_seq.unsqueeze(0) # Shape: [1, 60, 1]

print("Recursive forecast complete.")

# Trim predictions to the exact length of the test set
recursive_preds_scaled = np.array(recursive_preds_scaled[:num_test_points]).reshape(-1, 1)
pred_recursive = scaler.inverse_transform(recursive_preds_scaled)

# Calculate metrics
mae_rec = mean_absolute_error(y_true_test_data, pred_recursive)
rmse_rec = np.sqrt(mean_squared_error(y_true_test_data, pred_recursive))
mape_rec = np.mean(np.abs((y_true_test_data - pred_recursive) / y_true_test_data)) * 100
print(f"Recursive 5-Step Test (MAE={mae_rec:.4f} | RMSE={rmse_rec:.4f} | MAPE={mape_rec:.2f}%)")


# ----------------------------
# Plot Results (MODIFIED BLOCK)
# ----------------------------
# 1. X-axis for the full test set (for recursive)
x_pos_test_full_len = len(y_true_test_data)
x_pos_test_recursive = x_positions[train_end_index : train_end_index + x_pos_test_full_len]

# 2. X-axis for the sliding forecast
# This plot is shorter by (horizon-1) points
x_pos_test_sliding_len = len(y_true_sliding)
x_pos_test_sliding = x_positions[train_end_index + horizon - 1 : train_end_index + horizon - 1 + x_pos_test_sliding_len]


# --- Create the plot ---
fig, ax = plt.subplots(figsize=(14, 7))

# Plot 1: All actual data
ax.plot(x_positions, full_close_prices, label='Actual Price', linewidth=2, color='royalblue', alpha=0.8)

# Plot 2: Sliding 5-Step-Ahead Forecast (plots the 5th predicted point)
ax.plot(x_pos_test_sliding, pred_sliding[:, -1], label=f'Sliding {horizon}-Step Forecast (GND Truth Input)', linewidth=2, color='darkorange', linestyle='--')

# Plot 3: Recursive 5-Step-Ahead Forecast
ax.plot(x_pos_test_recursive, pred_recursive, label=f'Recursive {horizon}-Step Forecast (Model Input)', linewidth=2, color='green', linestyle=':')

# --- Add vertical lines ---
for gap_x in gap_locations:
    ax.axvline(x=gap_x, color='red', linestyle=':', alpha=0.3, linewidth=1)

# --- Format axes ---
ax.set_title(f"TXN 30-min LSTM {horizon}-Step Forecast Comparison (Trading Gaps Removed)", fontsize=16)
ax.set_xlabel("Trading Period Index (30-min intervals)", fontsize=12)
ax.set_ylabel("Price (USD)", fontsize=12)
#ax.grid(True, which='major', linestyle='--', alpha=0.4)
ax.legend()

# --- Create custom x-axis labels ---
tick_indices = np.linspace(0, len(x_positions) - 1, num=15, dtype=int)
tick_positions = [x_positions[i] for i in tick_indices]
tick_labels = [sorted_times[i].strftime('%Y-%m-%d') for i in tick_indices]

if x_positions[-1] not in tick_positions:
    tick_positions.append(x_positions[-1])
    tick_labels.append(sorted_times[-1].strftime('%Y-%m-%d'))

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, ha='right')
ax.margins(x=0.01)

plt.tight_layout()
plt.show()