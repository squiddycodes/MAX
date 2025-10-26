from django.shortcuts import render, redirect
import json
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server environments
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import base64
from io import BytesIO

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from django.db import transaction

import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
from django.http import JsonResponse
from Ingest.models import Case, Prediction, PredictionMeta, Stock


MODEL_NAME_MONTE = "MonteCarloGBM"
N_SIMS = 1000

def run_monte_carlo_simulation(historical_prices, n_sims=N_SIMS):
    """Run Monte Carlo GBM simulation."""
    split_index = int(len(historical_prices) * 0.80)
    train_data = historical_prices.iloc[:split_index]
    test_data = historical_prices.iloc[split_index:]

    if len(train_data) < 2:
        print("Not enough training data. Skipping.")
        return None, None, None

    log_returns = np.log(train_data / train_data.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()

    S0 = train_data.iloc[-1]
    T = len(test_data)
    dt = 1

    all_walks = np.zeros((T + 1, n_sims))
    all_walks[0, :] = S0

    for i in range(n_sims):
        for t in range(1, T + 1):
            Z = np.random.standard_normal()
            all_walks[t, i] = all_walks[t - 1, i] * np.exp(
                (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
            )

    avg_walk_values = np.mean(all_walks, axis=1)
    combined_index = train_data.index[[-1]].union(test_data.index)
    avg_walk_series = pd.Series(avg_walk_values, index=combined_index)

    return train_data, test_data, avg_walk_series


@transaction.atomic
def generate_monte_predictions():
    """Generate Monte Carlo predictions for all tickers in DB."""
    tickers = Stock.objects.values_list("ticker", flat=True)
    print(f"Found {len(tickers)} tickers to process: {list(tickers)}")

    for ticker in tickers:
        print(f"\n--- Processing {ticker} ---")

        # Fetch historical cases for this ticker
        cases = Case.objects.filter(ticker=ticker).order_by("iso_ny")
        if not cases.exists():
            print(f"No Case data for {ticker}. Skipping.")
            continue

        times = [c.iso_ny for c in cases]
        prices = [c.c for c in cases]

        df = pd.DataFrame({"iso_ny": times, "c": prices}).set_index("iso_ny")

        train, test, avg_walk = run_monte_carlo_simulation(df["c"])
        if train is None:
            continue

        test_values = test.values
        predicted_values = avg_walk.iloc[1:].values  # skip initial S0

        errors = predicted_values - test_values
        me = float(np.mean(errors))
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        safe_test_values = np.where(test_values == 0, 1e-9, test_values)
        mape = float(np.mean(np.abs(errors / safe_test_values)) * 100)

        print(f"  Metrics for {ticker}: ME={me:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")

        # Save Prediction rows
        split_index = len(train)
        test_cases = cases[split_index:]

        predictions_to_create = []
        for i, (actual, pred, case_obj) in enumerate(zip(test_values, predicted_values, test_cases)):
            predictions_to_create.append(
                Prediction(
                    ticker=ticker,
                    x_index=i,
                    iso_time=case_obj.iso_ny,
                    pred_close=float(pred),
                    actual_close=float(actual),
                    model=MODEL_NAME_MONTE
                )
            )
        print("PREDS[0]",MODEL_NAME)
        Prediction.objects.bulk_create(predictions_to_create, batch_size=500)
        print(f"  Saved {len(predictions_to_create)} predictions for {ticker}")

        # Save PredictionMeta row
        PredictionMeta.objects.update_or_create(
            ticker=ticker,
            model=MODEL_NAME_MONTE,
            mape=mape
        )
        print(f"  Saved PredictionMeta for {ticker}")

    print("\n✅ Monte Carlo predictions complete for all tickers.")



def GenerateMontePredictions(request):
    generate_monte_predictions()
    return redirect("/")


def Home(request):
    return render(request, 'home.html')

def Display(request):
    with open('msft_30min.json', 'r') as f:
        data = json.load(f)

    # --- Extract time and closing prices ---
    #times = [datetime.fromisoformat(entry['iso_ny']) for entry in data] #GET ARRAY OF ISO_NY - X
    times = []
    timesNVDA = []
    timesTXN = []
    timesMSFT = []
    close_prices = []
    close_pricesNVDA = []
    close_pricesTXN = []
    close_pricesMSFT = []
    for case in Case.objects.all():
        if case.ticker == "AAPL":
            times.append(case.iso_ny)
            close_prices.append(case.c)
        elif case.ticker == "NVDA":
            timesNVDA.append(case.iso_ny)
            close_pricesNVDA.append(case.c)
        elif case.ticker == "TXN":
            timesTXN.append(case.iso_ny)
            close_pricesTXN.append(case.c)
        elif case.ticker == "MSFT":
            timesMSFT.append(case.iso_ny)
            close_pricesMSFT.append(case.c)

    # --- Detect gaps and create continuous x-axis ---
    gap_threshold = timedelta(minutes=35)  # Slightly more than 30 min to detect gaps
    x_positions = [0]  # Start at position 0
    xPosNVDA = [0]
    xPosTXN = [0]
    xPosMSFT = [0]
    gap_locations = []  # Store where gaps occur (start index of new sessions)
    gapLocationsNVDA = []
    gapLocationsTXN = []
    gapLocationsMSFT = []

    for i in range(1, len(times)):
        time_diff = times[i] - times[i - 1]
        if time_diff > gap_threshold:
            # Gap detected - record position where the new session starts
            x_positions.append(x_positions[-1] + 1)
            gap_locations.append(x_positions[-1])  # mark session boundary
        else:
            # Normal progression
            x_positions.append(x_positions[-1] + 1)

    for i in range(1, len(timesNVDA)):
        time_diff = timesNVDA[i] - timesNVDA[i - 1]
        if time_diff > gap_threshold:
            # Gap detected - record position where the new session starts
            xPosNVDA.append(xPosNVDA[-1] + 1)
            gapLocationsNVDA.append(xPosNVDA[-1])  # mark session boundary
        else:
            # Normal progression
            xPosNVDA.append(xPosNVDA[-1] + 1)

    for i in range(1, len(timesTXN)):
            time_diff = timesTXN[i] - timesTXN[i - 1]
            if time_diff > gap_threshold:
                # Gap detected - record position where the new session starts
                xPosTXN.append(xPosTXN[-1] + 1)
                gapLocationsTXN.append(xPosTXN[-1])  # mark session boundary
            else:
                # Normal progression
                xPosTXN.append(xPosTXN[-1] + 1)

    for i in range(1, len(timesMSFT)):
        time_diff = timesMSFT[i] - timesMSFT[i - 1]
        if time_diff > gap_threshold:
            # Gap detected - record position where the new session starts
            xPosMSFT.append(xPosMSFT[-1] + 1)
            gapLocationsMSFT.append(xPosMSFT[-1])  # mark session boundary
        else:
            # Normal progression
            xPosMSFT.append(xPosMSFT[-1] + 1)


    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_positions, close_prices, label='MSFT Closing Price', linewidth=2)
    ax.plot(xPosNVDA, close_pricesNVDA, label='NVDA Closing Price', linewidth=2)
    ax.plot(xPosTXN, close_pricesTXN, label='TXN Closing Price', linewidth=2)
    ax.plot(xPosMSFT, close_pricesMSFT, label='MSFT Closing Price', linewidth=2)

    # --- NO vertical lines (removed) ---

    # --- Format axes ---
    ax.set_title("MSFT, NVDA, TXN and AAPL Stock Closing Prices Over Time (Gaps Removed)", fontsize=14)
    ax.set_xlabel("Trading Period Index", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Custom x-axis labels: use session start dates, but drop the first date ---
    # Choose ~10 evenly spaced session boundaries for labeling
    if gap_locations:
        stride = max(1, len(gap_locations) // 10)
        # positions to label (session starts), EXCLUDING the first overall start
        tick_positions = gap_locations[::stride]
        # Build matching labels
        tick_labels = []
        for pos in tick_positions:
            # map x-position back to corresponding time index
            idx = x_positions.index(pos)
            if idx < len(times):
                tick_labels.append(times[idx].strftime('%m/%d'))
        # Apply ticks/labels (only if we have any)
        if tick_positions and tick_labels:
            ax.set_xticks(tick_positions[:len(tick_labels)])
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    plt.tight_layout()

    # --- Save figure as embedded base64 image inside an HTML file ---
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png', bbox_inches='tight')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>MSFT Plot</title></head>"
        "<body style='margin:0;padding:16px;font-family:sans-serif;'>"
        "<h3>MSFT Stock Closing Prices Over Time (Gaps Removed)</h3>"
        f"<img style='max-width:100%;height:auto;' src='data:image/png;base64,{encoded}' />"
        "</body></html>"
    )

    with open('stock.html', 'w', encoding='utf-8') as f:
        f.write(html)

    # Optional: close the figure
    plt.close(fig)

    return render(request, 'display.html', {'html': html})

def fetchCases(request):
    return

def Ingest(request):
    if Stock.objects.exists() or Case.objects.exists():
        return redirect('/')

    #CREATE STOCKS
    for stock in ["AAPL", "MSFT", "TXN", "NVDA"]:
        Stock(ticker=stock).save()

    #get the absolute path of the json files
    base_dir = Path(__file__).resolve().parent 
    parent_dir = base_dir.parent
    #find all json files in the data folder
    for file in parent_dir.glob('*.json'):
        print("Starting",file)
        with file.open('r') as f:
            data = json.load(f)#load json
            for obj in data:
                stockTicker = Stock.objects.filter(ticker=obj["ticker"])
                case = Case(
                    ticker=obj["ticker"],
                    t=obj["t"],
                    iso_utc=obj["iso_utc"],
                    iso_ny=obj["iso_ny"],
                    o=obj["o"],
                    h=obj["h"],
                    l=obj["l"],
                    c=obj["c"],
                    v=obj["v"],
                    vw=obj["vw"],
                    n=obj["n"],
                    span=obj["span"],
                    stock=stockTicker[0]
                    #vola=
                )
                case.save()
        print("Done with",file)
    return redirect('/')
        

# ----------------------------
# Configuration
# ----------------------------
lookback = 60
horizon = 5
batch_size = 64
train_frac = 0.8
test_frac = 0.2
max_epochs = 100
patience = 10
lr = 1e-3
hidden = 64
layers = 2
dropout = 0.2
MODEL_NAME = "LSTM_30min"

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
            dropout=dropout if layers > 1 else 0.0,
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
        X.append(arr[i - lookback:i, 0])
        y.append(arr[i:i + horizon, 0])
    X = np.array(X)[:, :, None]
    y = np.array(y)
    return X, y

def save_predictions_to_db(ticker, test_times, x_pred, y_true_test, pred_recursive, mape):
    """Store predictions in Django database."""
    # Clear existing predictions for this ticker/model
    Prediction.objects.filter(ticker=ticker, model=MODEL_NAME).delete()
    PredictionMeta.objects.filter(ticker=ticker, model=MODEL_NAME).delete()

    preds = []
    for i, (ts, xi, pred, act) in enumerate(zip(
        test_times, x_pred,
        pred_recursive.reshape(-1),
        y_true_test.reshape(-1)
    )):
        preds.append(Prediction(
            ticker=ticker,
            x_index=int(xi),
            iso_time=ts,
            pred_close=float(pred),
            actual_close=float(act),
            model=MODEL_NAME
        ))
    Prediction.objects.bulk_create(preds, batch_size=500)
    PredictionMeta.objects.create(ticker=ticker, model=MODEL_NAME, mape=float(mape))
    print(f"[{ticker}] Saved {len(preds)} predictions to DB.")
    return len(preds)

# ----------------------------
# Core training logic
# ----------------------------
def train_one_ticker(ticker):
    """Train model for one ticker from Case table."""
    # Pull data for this ticker
    qs = Case.objects.filter(ticker=ticker).order_by("iso_ny")
    if not qs.exists():
        print(f"[{ticker}] No Case data found.")
        return None

    times_ny = [c.iso_ny for c in qs]
    close_prices = [c.c for c in qs]

    series = pd.Series(close_prices, index=times_ny).sort_index()
    values = series.values.reshape(-1, 1)
    sorted_times = series.index
    print(f"\n[{ticker}] Loaded {len(values)} points from DB.")

    # Build time positions
    gap_threshold = timedelta(minutes=35)
    x_positions = [0]
    for i in range(1, len(sorted_times)):
        x_positions.append(x_positions[-1] + 1)

    # Split train/test
    n = len(values)
    train_end_index = int(math.ceil(train_frac * n))
    test_end_index = int(math.ceil((train_frac + test_frac) * n))
    train_vals = values[:train_end_index]
    test_vals = values[train_end_index - lookback:test_end_index]

    # Scale
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_vals)
    test_scaled = scaler.transform(test_vals)

    # Make supervised sequences
    X_train, y_train = make_sequences(train_scaled, lookback, horizon)
    X_test, y_test = make_sequences(test_scaled, lookback, horizon)

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

    # Model setup
    model = LSTMForecaster(hidden=hidden, layers=layers, dropout=dropout, output_size=horizon).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    no_improve = 0
    best_weights = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
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

        print(f"[{ticker}] Epoch {epoch:03d} | Train MSE={train_loss:.6f} | Val MSE={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            best_weights = model.state_dict()
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{ticker}] Early stopping.")
                break

    if best_weights is None:
        best_weights = model.state_dict()
    model.load_state_dict(best_weights)
    model.eval()

    # Recursive forecasting
    y_true_test = values[train_end_index:test_end_index]
    num_test_pts = len(y_true_test)
    recursive_scaled = []

    with torch.no_grad():
        current_input = torch.tensor(train_scaled[-lookback:], dtype=torch.float32).reshape(1, lookback, 1).to(device)
        for _ in range(int(math.ceil(num_test_pts / horizon))):
            step_pred = model(current_input)
            recursive_scaled.extend(step_pred.cpu().squeeze().tolist())
            old_seq = current_input.squeeze(0)[horizon:]
            new_seq = step_pred.detach().T
            current_input = torch.cat((old_seq, new_seq), dim=0).unsqueeze(0)

    recursive_scaled = np.array(recursive_scaled[:num_test_pts]).reshape(-1, 1)
    pred_recursive = scaler.inverse_transform(recursive_scaled)

    # Metrics
    mae = mean_absolute_error(y_true_test, pred_recursive)
    rmse = np.sqrt(mean_squared_error(y_true_test, pred_recursive))
    mape = np.mean(np.abs((y_true_test - pred_recursive) / y_true_test)) * 100.0
    me = float(np.mean(pred_recursive - y_true_test))

    print(f"[{ticker}] Test: MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape:.2f}% | ME={me:.4f}")

    # Build index mapping for predictions
    x_test_start = train_end_index
    x_test_end = train_end_index + num_test_pts
    x_pred = x_positions[x_test_start:x_test_end]
    test_times = list(sorted_times[x_test_start:x_test_end])

    # Save to database
    count = save_predictions_to_db(ticker, test_times, x_pred, y_true_test, pred_recursive, mape)

    return {"ticker": ticker, "MAE": mae, "RMSE": rmse, "MAPE": mape, "ME": me, "rows": count}


# ----------------------------
# Django View
# ----------------------------
def GenerateLSTMPredictions(request):
    """Train LSTM models for all tickers in Case table."""
    if Prediction.objects.exists() or PredictionMeta.objects.exists():
        return redirect("/")
    tickers = Case.objects.values_list("ticker", flat=True).distinct()
    results = []
    for ticker in tickers:
        try:
            res = train_one_ticker(ticker)
            if res:
                results.append(res)
        except Exception as e:
            print(f"[{ticker}] ERROR: {e}")
            results.append({"ticker": ticker, "error": str(e)})

    return JsonResponse({
        "status": "ok",
        "model": MODEL_NAME,
        "generated_at": datetime.now().isoformat(),
        "results": results,
    })
