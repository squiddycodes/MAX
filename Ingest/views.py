from django.shortcuts import render, redirect
import os
import json
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server environments
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import base64
import io
from io import BytesIO
import math
from typing import Dict
import numpy as np
import pandas as pd
from django.http import JsonResponse, HttpRequest
from django.views.decorators.http import require_GET
import cvxpy as cp
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cvxpy as cp
from django.shortcuts import render
from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_GET
from .models import Case, Prediction, PredictionMeta, Stock
from django.shortcuts import render
from django.http import HttpRequest, JsonResponse
# views.py
import math
from typing import Dict, List
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse, HttpRequest
from django.views.decorators.http import require_GET

from .models import Case, Stock


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
    if Prediction.objects.filter(model='MonteCarloGBM').exists() or PredictionMeta.objects.filter(model='MonteCarloGBM').exists():
        return redirect("/")
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
    # ---- Config
    tickers = ["AAPL", "NVDA", "TXN", "MSFT"]
    gap_threshold = timedelta(minutes=35)

    # Fixed palette so actual + preds share the same color per ticker
    ticker_colors = {
        "AAPL": "tab:blue",
        "MSFT": "tab:orange",
        "TXN":  "tab:green",
        "NVDA": "tab:red",
    }

    # Containers
    series = {}            # ticker -> dict(times, closes, x, gaps)
    time_to_x = {}         # ticker -> {datetime -> x_index}

    # ---- Load actuals and build continuous x-axes per ticker
    for t in tickers:
        rows = list(
            Case.objects.filter(ticker=t).order_by("iso_ny").values("iso_ny", "c")
        )
        if not rows:
            continue

        times = [r["iso_ny"] for r in rows]
        prices = [float(r["c"]) for r in rows]

        x = [0]
        gaps = []
        for i in range(1, len(times)):
            if times[i] - times[i-1] > gap_threshold:
                x.append(x[-1] + 1)
                gaps.append(x[-1])
            else:
                x.append(x[-1] + 1)

        # map timestamps → x for aligning predictions
        t2x = {ts: xi for ts, xi in zip(times, x)}

        series[t] = {
            "times": times,
            "prices": prices,
            "x": x,
            "gaps": gaps
        }
        time_to_x[t] = t2x

    # Choose a reference ticker (longest history) for xticks
    if not series:
        return render(request, "display.html", {"html": "<p>No data found.</p>"})
    ref = max(series.keys(), key=lambda k: len(series[k]["x"]))

    # ---- Pull predictions and align to x for each ticker
    preds = {t: {"MC": ([], []), "LSTM": ([], [])} for t in series.keys()}
    for t in series.keys():
        t2x = time_to_x[t]

        # MonteCarloGBM
        rows_mc = list(
            Prediction.objects.filter(ticker=t, model="MonteCarloGBM")
            .order_by("iso_time")
            .values("iso_time", "pred_close")
        )
        x_mc, y_mc = [], []
        for r in rows_mc:
            ts = r["iso_time"]
            if ts in t2x:
                x_mc.append(t2x[ts])
                y_mc.append(float(r["pred_close"]))
        preds[t]["MC"] = (x_mc, y_mc)

        # LSTM_30min
        rows_lstm = list(
            Prediction.objects.filter(ticker=t, model="LSTM_30min")
            .order_by("iso_time")
            .values("iso_time", "pred_close")
        )
        x_l, y_l = [], []
        for r in rows_lstm:
            ts = r["iso_time"]
            if ts in t2x:
                x_l.append(t2x[ts])
                y_l.append(float(r["pred_close"]))
        preds[t]["LSTM"] = (x_l, y_l)

    # ---- Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for t in series.keys():
        col = ticker_colors.get(t, None)
        # Actual (solid)
        ax.plot(series[t]["x"], series[t]["prices"],
                label=f"{t} Actual", linewidth=2, color=col)

        # Monte Carlo predictions (dotted)
        x_mc, y_mc = preds[t]["MC"]
        if x_mc:
            ax.plot(x_mc, y_mc, linestyle=":", linewidth=2,
                    label=f"{t} MC Pred", color=col)

        # LSTM predictions (dashed)
        x_l, y_l = preds[t]["LSTM"]
        if x_l:
            ax.plot(x_l, y_l, linestyle="--", linewidth=2,
                    label=f"{t} LSTM Pred", color=col)

    # ---- X tick labels from reference ticker’s dates (session starts)
    gaps_ref = series[ref]["gaps"]
    if gaps_ref:
        stride = max(1, len(gaps_ref) // 10)
        tick_positions = gaps_ref[::stride]
        tick_labels = []
        times_ref = series[ref]["times"]
        x_ref = series[ref]["x"]
        for pos in tick_positions:
            try:
                idx = x_ref.index(pos)
                tick_labels.append(times_ref[idx].strftime("%m/%d"))
            except ValueError:
                pass
        if tick_positions and tick_labels:
            ax.set_xticks(tick_positions[:len(tick_labels)])
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_title("AAPL, NVDA, TXN, MSFT — Actual vs. Monte Carlo (dotted) & LSTM (dashed)")
    ax.set_xlabel("Trading Period Index")
    ax.set_ylabel("Price (USD)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),  # below the x-axis
        ncol=3,
        frameon=False
    )
    # Reposition axis labels closer to origin
    ax.xaxis.set_label_coords(0.1, -0.17)   # (x=0.5 means centered horizontally)
    ax.yaxis.set_label_coords(-0.05, 0.2)   # (y=0.5 means centered vertically)


    plt.tight_layout()

    # ---- Embed figure as base64 in HTML
    tmp = BytesIO()
    fig.savefig(tmp, format="png", bbox_inches="tight")
    encoded = base64.b64encode(tmp.getvalue()).decode("utf-8")
    plt.close(fig)

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Stocks</title></head>"
        "<body style='margin:0;padding:16px;font-family:sans-serif;'>"
        "<h3>Actual vs. Predictions (Monte Carlo = dotted, LSTM = dashed)</h3>"
        f"<img style='max-width:100%;height:auto;' src='data:image/png;base64,{encoded}' />"
        "</body></html>"
    )

    return render(request, "display.html", {"html": html})




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
# Configuration (kept)
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
# Reproducibility (kept)
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
# Model (same head as your first script)
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

def _ensure_dir(path):
    import os
    os.makedirs(path, exist_ok=True)

def _ckpt_path_for(ticker):
    import os
    _ensure_dir("checkpoints")
    # unique per ticker/model/hyperparams to avoid stale bleed-through
    return os.path.join("checkpoints", f"{MODEL_NAME}_{ticker}_lb{lookback}_hz{horizon}_h{hidden}_L{layers}.pt")

def save_predictions_to_db(ticker, test_times, x_pred, y_true_test, pred_recursive, mape):
    """Store predictions in Django database."""
    Prediction.objects.filter(ticker=ticker, model=MODEL_NAME).delete()
    PredictionMeta.objects.filter(ticker=ticker, model=MODEL_NAME).delete()

    preds = []
    for ts, xi, pred, act in zip(
        test_times, x_pred,
        pred_recursive.reshape(-1),
        y_true_test.reshape(-1)
    ):
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
# Core training logic (now matches single-file logic + per-ticker checkpoints)
# ----------------------------
def train_one_ticker(ticker):
    """Train one LSTM per ticker from Case table with the same logic as your single-file script."""
    # --- Load series from DB ---
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

    # --- Build monotone x positions (like your first script; gaps don't affect training) ---
    x_positions = list(range(len(sorted_times)))

    # --- Train/test split (identical logic) ---
    n = len(values)
    train_end_index = int(math.ceil(train_frac * n))
    test_end_index = int(math.ceil((train_frac + test_frac) * n))
    train_vals = values[:train_end_index]
    test_vals  = values[train_end_index - lookback:test_end_index]  # include overlap for sequences

    # --- Scale (fit on train only) ---
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_vals)
    test_scaled  = scaler.transform(test_vals)

    # --- Supervised sequences (lookback=60, horizon=5) ---
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

    # --- Model, criterion, optimizer ---
    model = LSTMForecaster(hidden=hidden, layers=layers, dropout=dropout, output_size=horizon).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Best-epoch tracking with epsilon (matches the first script) ---
    best_val = float("inf")
    no_improve = 0
    ckpt_path = _ckpt_path_for(ticker)

    for epoch in range(1, max_epochs + 1):
        # train
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

        # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(test_loader.dataset)

        print(f"[{ticker}] Epoch {epoch:03d} | Train MSE={train_loss:.6f} | Val MSE={val_loss:.6f}")

        # epsilon rule to avoid thrashing on float noise
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{ticker}] Early stopping.")
                break

    # --- Load best checkpoint from disk (matches first script’s pattern) ---
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # --- Recursive 5-step forecasting over the test span (closed-loop) ---
    y_true_test = values[train_end_index:test_end_index]
    num_test_pts = len(y_true_test)
    recursive_scaled = []

    with torch.no_grad():
        current_input = torch.tensor(train_scaled[-lookback:], dtype=torch.float32).reshape(1, lookback, 1).to(device)
        for _ in range(int(math.ceil(num_test_pts / horizon))):
            step_pred = model(current_input)            # [1, horizon]
            recursive_scaled.extend(step_pred.cpu().squeeze().tolist())
            old_seq = current_input.squeeze(0)[horizon:]  # [lookback - horizon, 1]
            new_seq = step_pred.detach().T                # [horizon, 1]
            current_input = torch.cat((old_seq, new_seq), dim=0).unsqueeze(0)

    recursive_scaled = np.array(recursive_scaled[:num_test_pts]).reshape(-1, 1)
    pred_recursive = scaler.inverse_transform(recursive_scaled)

    # --- Metrics (like your first script’s recursive block) ---
    mae = mean_absolute_error(y_true_test, pred_recursive)
    rmse = np.sqrt(mean_squared_error(y_true_test, pred_recursive))
    mape = np.mean(np.abs((y_true_test - pred_recursive) / y_true_test)) * 100.0
    me = float(np.mean(pred_recursive - y_true_test))

    print(f"[{ticker}] Test: MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape:.2f}% | ME={me:.4f}")

    # --- Indices & times for DB writes ---
    x_test_start = train_end_index
    x_test_end   = train_end_index + num_test_pts
    x_pred = x_positions[x_test_start:x_test_end]
    test_times = list(sorted_times[x_test_start:x_test_end])

    # --- Save to DB (kept from your second script) ---
    count = save_predictions_to_db(ticker, test_times, x_pred, y_true_test, pred_recursive, mape)

    return {"ticker": ticker, "MAE": mae, "RMSE": rmse, "MAPE": mape, "ME": me, "rows": count}

# ----------------------------
# Django View (unchanged behavior, now uses aligned logic)
# ----------------------------
def GenerateLSTMPredictions(request):
    """Train LSTM models for all tickers in Case table."""
    if Prediction.objects.filter(model=MODEL_NAME).exists() or PredictionMeta.objects.filter(model=MODEL_NAME).exists():
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


# =========================
# A) Metrics (already added before; unchanged except RF hard-coded)
# =========================
DEFAULT_WINDOW = 60
RF_ANNUAL = 0.0386  # hard-coded 3-mo T-bill coupon-equivalent

def _periods_per_year(span: str) -> float:
    span = (span or "").lower().strip()
    days_per_year = 252.0
    if span in ("1d", "day", "1day", "d"):
        return days_per_year
    if span in ("30min", "30m", "30"):
        return 13.0 * days_per_year         # 6.5h/day / 0.5h
    if span in ("15min", "15m"):
        return 26.0 * days_per_year
    if span in ("60min", "1h", "60m"):
        return 6.5 * days_per_year
    if span in ("5min", "5m"):
        return 78.0 * days_per_year
    try:
        if "min" in span:
            mins = float(span.replace("min", "").replace("m", ""))
            per_day = 390.0 / mins
            return per_day * days_per_year
    except Exception:
        pass
    return days_per_year

def _compute_roll_metrics(df: pd.DataFrame, window: int, ppy: float, rf_annual: float) -> pd.DataFrame:
    df = df.sort_values("iso_ny").reset_index(drop=True)
    px = df["c"].astype(float)
    r = np.log(px / px.shift(1))
    df["ret"] = r
    roll_std  = r.rolling(window=window, min_periods=window).std(ddof=1)
    roll_mean = r.rolling(window=window, min_periods=window).mean()
    vola_ann   = roll_std * math.sqrt(ppy)
    excess_ann = (roll_mean * ppy) - rf_annual
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe_ann = excess_ann / vola_ann
    df["vola"]   = vola_ann.replace([np.inf, -np.inf], np.nan)
    df["sharpe"] = sharpe_ann.replace([np.inf, -np.inf], np.nan)
    return df


# =========================
# B) Allocation + Dollar-neutral Hedge (NEW)
# =========================
# Requires cvxpy in your env

DEFAULT_LOOKBACK_DAYS = 252    # anchor in trading days
DEFAULT_SPAN = "30min"

def _fetch_returns(tickers: List[str], span: str, lookback_days: int) -> pd.DataFrame:
    """Returns aligned per-period log-returns DataFrame (rows=time, cols=tickers)."""
    rets = []
    for t in tickers:
        qs = (Case.objects
              .filter(ticker=t, span=span)
              .order_by("iso_ny")
              .values("iso_ny", "c"))
        rows = list(qs)
        if len(rows) < 3:
            continue
        df = pd.DataFrame(rows)
        df["iso_ny"] = pd.to_datetime(df["iso_ny"])
        df = df.dropna(subset=["c"]).set_index("iso_ny").sort_index()
        ppy = _periods_per_year(span)
        periods = int(max(50, round(lookback_days * (ppy / 252.0))))
        df = df.iloc[-periods:]
        r = np.log(df["c"].astype(float) / df["c"].astype(float).shift(1)).dropna()
        r.name = t
        rets.append(r)
    if not rets:
        raise RuntimeError("Insufficient data to build returns matrix.")
    R = pd.concat(rets, axis=1, join="inner").dropna(how="any")
    return R

def _annualize_mu_sigma(R: pd.DataFrame, span: str):
    """Annualized mu and Sigma from per-period log-returns."""
    ppy = _periods_per_year(span)
    mu_per = R.mean(axis=0).values
    cov_per = np.cov(R.values.T, ddof=1)
    mu_ann = mu_per * ppy
    Sigma_ann = cov_per * ppy
    return mu_ann, Sigma_ann, ppy



def _portfolio_stats(w: np.ndarray, mu_ann: np.ndarray, Sigma_ann: np.ndarray, rf: float):
    mu_p = float(w @ mu_ann)
    var_p = float(w @ Sigma_ann @ w)
    sig_p = math.sqrt(max(0.0, var_p))
    sharpe = (mu_p - rf) / sig_p if sig_p > 0 else float("nan")
    return mu_p, sig_p, sharpe

# =========================
# C) The function-based view (both buttons)
# =========================
# @require_GET
# def PortfolioOptimizer(request: HttpRequest):
#     """
#     - Default: render 'portfolio.html'.
#     - ?run=1           -> compute & store rolling annualized volatility and Sharpe into Case.vola / Case.risk
#     - ?optimize=1      -> solve allocation and dollar-neutral hedge; returns JSON unless you omit &format=json
#       Params (optimize):
#         span=30min|1d        (default 30min)
#         tickers=NVDA,MSFT,... (default: all Stock.ticker present)
#         lookback_days=252     (trading-day anchor for history length)
#         mv_long_only=1|0      (default 1)
#         sims=4000             (GBM scenarios)
#         horizon_periods=252   (simulation horizon in periods of given span)
#         cvar_alpha=0.95
#         l1_hedge=0.5          (||h||_1 cap)
#         format=json           (return JSON)
#     """
#     # ---- Button A: compute volatility & Sharpe into DB ----
#     if request.GET.get("run"):
#         window = int(request.GET.get("window", DEFAULT_WINDOW))
#         result = _run_optimizer_metrics(
#             window=window,
#             rf_annual=RF_ANNUAL,
#             tickers_param=request.GET.get("tickers"),
#             spans_param=request.GET.get("spans"),
#         )
#         if request.GET.get("format") == "json":
#             return JsonResponse({"status": "ok", **result})
#         return render(request, "portfolio.html", {"optimizer_result": result})

#     # ---- Button B: optimize portfolio + dollar-neutral hedge overlay ----
#     if request.GET.get("optimize"):
#         # read params
#         span = request.GET.get("span", DEFAULT_SPAN)
#         lookback_days = int(request.GET.get("lookback_days", DEFAULT_LOOKBACK_DAYS))
#         mv_long_only = request.GET.get("mv_long_only", "1") == "1"
#         sims = int(request.GET.get("sims", 4000))
#         horizon_periods = int(request.GET.get("horizon_periods", 252))
#         cvar_alpha = float(request.GET.get("cvar_alpha", 0.95))
#         l1_hedge = float(request.GET.get("l1_hedge", 0.5))

#         if request.GET.get("tickers"):
#             tickers = [t.strip().upper() for t in request.GET.get("tickers").split(",") if t.strip()]
#         else:
#             tickers = list(Stock.objects.values_list("ticker", flat=True).distinct())

#         # build returns, annualize
#         R = _fetch_returns(tickers, span, lookback_days)
#         mu_ann, Sigma_ann, ppy = _annualize_mu_sigma(R, span)

#         # QP: mean-variance base portfolio (sum=1)
#         w_mv = _solve_mean_variance(mu_ann, Sigma_ann, rf=RF_ANNUAL, long_only=mv_long_only)

#         # GBM scenarios -> CVaR LP hedge (sum=0)
#         S = _simulate_gbm(mu_ann, Sigma_ann, ppy, horizon_periods=horizon_periods, sims=sims)
#         h = _solve_cvar_dollar_neutral(S, alpha=cvar_alpha, l1_hedge=l1_hedge)

#         # combine: total weights still sum to 1
#         w_total = w_mv + h

#         # stats
#         mv_stats = _portfolio_stats(w_mv, mu_ann, Sigma_ann, rf=RF_ANNUAL)
#         total_stats = _portfolio_stats(w_total, mu_ann, Sigma_ann, rf=RF_ANNUAL)

#         payload = {
#             "status": "ok",
#             "rf_annual": RF_ANNUAL,
#             "span": span,
#             "tickers": tickers,
#             "lookback_days": lookback_days,
#             "ppy": _periods_per_year(span),
#             "mv_long_only": mv_long_only,
#             "sims": sims,
#             "horizon_periods": horizon_periods,
#             "cvar_alpha": cvar_alpha,
#             "l1_hedge": l1_hedge,
#             "weights_mv": {t: float(w) for t, w in zip(tickers, w_mv)},
#             "weights_hedge_dollar_neutral": {t: float(x) for t, x in zip(tickers, h)},
#             "weights_total": {t: float(w) for t, w in zip(tickers, w_total)},
#             "mv_stats": {"mu_ann": mv_stats[0], "sigma_ann": mv_stats[1], "sharpe": mv_stats[2]},
#             "total_stats": {"mu_ann": total_stats[0], "sigma_ann": total_stats[1], "sharpe": total_stats[2]},
#             "checks": {
#                 "sum_mv": float(np.sum(w_mv)),
#                 "sum_hedge": float(np.sum(h)),  # should be ~0
#                 "sum_total": float(np.sum(w_total))  # should be ~1
#             }
#         }

#         if request.GET.get("format") == "json":
#             return JsonResponse(payload)
#         return render(request, "portfolio.html", {"optimize_result": payload})

#     # default render
#     return render(request, "portfolio.html")

def _compute_roll_metrics(df: pd.DataFrame, window: int, ppy: float, rf_annual: float) -> pd.DataFrame:
    df = df.sort_values("iso_ny").reset_index(drop=True)
    px = df["c"].astype(float)
    r = np.log(px / px.shift(1))
    df["ret"] = r

    roll_std  = r.rolling(window=window, min_periods=window).std(ddof=1)
    roll_mean = r.rolling(window=window, min_periods=window).mean()

    vola_ann   = roll_std * math.sqrt(ppy)
    excess_ann = (roll_mean * ppy) - rf_annual

    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe_ann = excess_ann / vola_ann

    df["vola"]   = vola_ann.replace([np.inf, -np.inf], np.nan)
    df["sharpe"] = sharpe_ann.replace([np.inf, -np.inf], np.nan)
    return df


def _run_optimizer(window: int, rf_annual: float, tickers_param: str | None, spans_param: str | None):
    # Decide which tickers to run
    if tickers_param:
        tickers = [t.strip().upper() for t in tickers_param.split(",") if t.strip()]
    else:
        tickers = list(Stock.objects.values_list("ticker", flat=True).distinct())

    updated_rows_total = 0
    per_group_counts: Dict[str, int] = {}

    for ticker in tickers:
        span_qs = Case.objects.filter(ticker=ticker).values_list("span", flat=True).distinct()
        spans = [s for s in span_qs if s]
        if spans_param:
            spans_req = [s.strip() for s in spans_param.split(",") if s.strip()]
            spans = [s for s in spans if s in spans_req]
        if not spans:
            continue

        for span in spans:
            qs = (Case.objects
                  .filter(ticker=ticker, span=span)
                  .order_by("iso_ny")
                  .values("id", "iso_ny", "c"))
            rows = list(qs)
            if len(rows) < window + 1:
                per_group_counts[f"{ticker}:{span}"] = 0
                continue

            df = pd.DataFrame(rows)
            if not pd.api.types.is_datetime64_any_dtype(df["iso_ny"]):
                df["iso_ny"] = pd.to_datetime(df["iso_ny"])

            ppy = _periods_per_year(span)
            dfm = _compute_roll_metrics(df[["id", "iso_ny", "c"]].copy(), window, ppy, rf_annual)

            ids = dfm["id"].tolist()
            vola_vals = dfm["vola"].tolist()
            sharpe_vals = dfm["sharpe"].tolist()

            case_map = {c.id: c for c in Case.objects.filter(id__in=ids)}
            updates = []
            for _id, v, s in zip(ids, vola_vals, sharpe_vals):
                obj = case_map.get(_id)
                if obj is None:
                    continue
                obj.vola = float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None
                obj.risk = float(s) if s is not None and not (isinstance(s, float) and math.isnan(s)) else None
                updates.append(obj)

            if updates:
                Case.objects.bulk_update(updates, ["vola", "risk"], batch_size=1000)
                per_group_counts[f"{ticker}:{span}"] = len(updates)
                updated_rows_total += len(updates)
            else:
                per_group_counts[f"{ticker}:{span}"] = 0

    return {
        "window": window,
        "rf_annual": rf_annual,
        "updated_total": updated_rows_total,
        "per_group": per_group_counts
    }
# =========================
# Model-driven portfolio optimizer (QCQP) with HTML render
# =========================

# ----- Tunables -----
RF_ANNUAL = 0.0386                         # risk-free (annualized)
W_MAX     = 0.5                           # per-name cap
LAMBDA    = 2.5                            # return-risk tradeoff in objective
GAMMA_MODEL_BLEND = .99                   # weight of model forecasts in expected return
BETA_COV_INFLATE  = 0.25                   # how much model risk inflates covariance
KAPPA_UNCERT      = 0.18                   # SOC cap for forecast-uncertainty exposure
LOOKBACK_DAYS     = 252                    # history anchor
DEFAULT_SPAN      = "30min"
TAU_REL = .45  # model-uncertainty allowed as a fraction of portfolio volatility

# alpha weights for risk score
ALPHA_RMSE, ALPHA_MAPE, ALPHA_ME = 0.7, 0.0, 0.3

# -------------- utilities --------------

def _periods_per_year(span: str) -> float:
    span = (span or "").lower().strip()
    days_per_year = 252.0
    if span in ("1d", "day", "1day", "d"):
        return days_per_year
    if span in ("30min", "30m", "30"):
        return 13.0 * days_per_year
    if span in ("60min", "1h", "60m"):
        return 6.5 * days_per_year
    if span in ("15min", "15m"):
        return 26.0 * days_per_year
    if span in ("5min", "5m"):
        return 78.0 * days_per_year
    try:
        if "min" in span:
            mins = float(span.replace("min", "").replace("m", ""))
            per_day = 390.0 / mins
            return per_day * days_per_year
    except Exception:
        pass
    return days_per_year

def _fetch_returns(tickers, span, lookback_days=LOOKBACK_DAYS) -> pd.DataFrame:
    """Aligned per-period log returns for tickers/span."""
    series = []
    for t in tickers:
        qs = (Case.objects
              .filter(ticker=t, span=span)
              .order_by("iso_ny")
              .values("iso_ny", "c"))
        rows = list(qs)
        if len(rows) < 3:
            continue
        df = pd.DataFrame(rows)
        df["iso_ny"] = pd.to_datetime(df["iso_ny"])
        df = df.dropna(subset=["c"]).set_index("iso_ny").sort_index()

        ppy = _periods_per_year(span)
        periods = int(max(50, round(lookback_days * (ppy / 252.0))))
        df = df.iloc[-periods:]
        r = np.log(df["c"].astype(float) / df["c"].astype(float).shift(1)).dropna()
        r.name = t
        series.append(r)
    if not series:
        raise RuntimeError("Insufficient price history for returns.")
    R = pd.concat(series, axis=1, join="inner").dropna(how="any")
    return R

def _annualize_mu_sigma(R: pd.DataFrame, span: str):
    ppy = _periods_per_year(span)
    mu_per = R.mean(axis=0).values            # (n,)
    cov_per = np.cov(R.values.T, ddof=1)      # (n,n)
    mu_ann = mu_per * ppy
    Sigma_ann = cov_per * ppy
    return mu_ann, Sigma_ann, ppy

def _metrics_from_predictions(ticker: str) -> dict:
    """
    Compute RMSE, MAPE, ME per model for this ticker using Prediction table.
    Returns:
      {
        model_name: {"rmse":..., "mape":..., "me":..., "mu_model_ann": ...}
      }
    """
    rows = list(Prediction.objects.filter(ticker=ticker).values(
        "model", "pred_close", "actual_close", "iso_time"
    ))
    if not rows:
        return {}

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["pred_close", "actual_close"])
    out = {}
    for model, g in df.groupby("model"):
        yhat = g["pred_close"].astype(float).values
        y    = g["actual_close"].astype(float).values
        if len(y) < 5 or np.all(y == 0):
            continue

        rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
        mape = float(np.mean(np.abs((yhat - y) / np.clip(y, 1e-12, None))) * 100.0)
        me   = float(np.mean(yhat - y))

        # crude model-implied simple return (periodized from consecutive preds vs actuals)
        # If you store horizon>1, this still yields an average per-step drift proxy.
        rets = (yhat[1:] / np.where(yhat[:-1] == 0, yhat[:-1] + 1e-12, yhat[:-1])) - 1.0
        mu_model_per = float(np.nanmean(rets)) if len(rets) > 0 else 0.0

        # annualize assuming 30-min by default; we’ll rescale later if needed
        ppy = _periods_per_year(DEFAULT_SPAN)
        mu_model_ann = mu_model_per * ppy

        out[model] = {"rmse": rmse, "mape": mape, "me": me, "mu_model_ann": mu_model_ann}
    return out

def _risk_scores_for_ticker(ticker: str) -> tuple[dict, float]:
    """
    Compute normalized risk scores r_i for each model and derive a ticker-level uncertainty s (scalar).
    Returns (scores_per_model, s_ticker).
    """
    stats = _metrics_from_predictions(ticker)
    if not stats:
        return {}, 0.0

    # collect vectors
    rmses = np.array([v["rmse"] for v in stats.values()], dtype=float)
    mapes = np.array([v["mape"] for v in stats.values()], dtype=float)
    mes   = np.array([abs(v["me"]) for v in stats.values()], dtype=float)

    # avoid zero denominators
    def nzsum(x): 
        s = np.sum(x)
        return s if s > 0 else 1.0

    s_rmse = nzsum(rmses)
    s_mape = nzsum(mapes)
    s_me   = nzsum(mes)

    models = list(stats.keys())
    r = {}
    for i, m in enumerate(models):
        r_i = (ALPHA_RMSE * (rmses[i] / s_rmse)
             + ALPHA_MAPE * (mapes[i] / s_mape)
             + ALPHA_ME   * (mes[i]   / s_me))
        r[m] = float(r_i)

    # ticker-level uncertainty: weighted average of model risks (weight by each model’s share in the blend below)
    # first compute inverse-risk weights (not yet normalized); if all zeros -> equal
    inv = np.array([1.0 / max(1e-12, r[m]) for m in models], dtype=float)
    if not np.isfinite(inv).any() or inv.sum() == 0:
        w_model = np.ones_like(inv) / len(inv)
    else:
        w_model = inv / inv.sum()

    s_ticker = float(np.dot(w_model, np.array([r[m] for m in models], dtype=float)))
    return r, s_ticker

def _blended_mu_from_models(ticker: str, mu_hist_ann: float) -> float:
    """
    Blend historical drift with model-inferred annual drift using inverse-risk model weights.
    """
    stats = _metrics_from_predictions(ticker)
    if not stats:
        return mu_hist_ann

    models = list(stats.keys())
    risks = []
    mus   = []
    for m in models:
        risks.append(max(1e-12, _risk_scores_for_ticker(ticker)[0].get(m, 1.0)))
        mus.append(stats[m]["mu_model_ann"])

    risks = np.array(risks, dtype=float)
    mus   = np.array(mus, dtype=float)

    inv = 1.0 / risks
    w = inv / inv.sum() if inv.sum() > 0 else np.ones_like(inv) / len(inv)
    mu_model_ann = float(np.dot(w, mus))

    mu_blend = (1.0 - GAMMA_MODEL_BLEND) * mu_hist_ann + GAMMA_MODEL_BLEND * mu_model_ann
    return mu_blend

def _psd_sqrt(M: np.ndarray) -> np.ndarray:
    """Return A s.t. A @ A.T = M for symmetric PSD M (numerically robust)."""
    M = 0.5 * (M + M.T)
    evals, evecs = np.linalg.eigh(M)
    evals = np.clip(evals, 1e-12, None)  # avoid tiny negatives
    A = (evecs * np.sqrt(evals)) @ evecs.T
    return A


def _eval_constraint_values_rel(w_vec: np.ndarray,
                                s_vec: np.ndarray,
                                Sigma: np.ndarray,
                                tau_rel: float,
                                tickers: list[str]) -> dict:
    """Constraint values & slacks for the *relative* SOC."""
    w = np.asarray(w_vec, dtype=float)
    sum_w = float(np.sum(w))
    min_w = float(np.min(w))
    max_w = float(np.max(w))
    soc_val = float(np.linalg.norm(s_vec * w, 2))
    vol_val = float(np.sqrt(max(0.0, w @ Sigma @ w)))
    rhs = float(tau_rel * vol_val)
    return {
        "sum_w": sum_w,
        "sum_w_slack": sum_w - 1.0,                     # ~0 when satisfied
        "min_w": min_w,
        "min_w_ge0_slack": min_w,                       # >= 0 when satisfied
        "max_w": max_w,
        "max_w_le_WMAX_slack": float(W_MAX - max_w),    # >= 0 when satisfied
        "soc_value": soc_val,
        "vol_value": vol_val,
        "rhs_tau_times_vol": rhs,
        "rel_soc_slack": float(rhs - soc_val),          # >= 0 when satisfied
        "by_ticker": {
            t: {"w": float(w[i]), "s": float(s_vec[i]), "s*w_abs": float(abs(s_vec[i] * w[i]))}
            for i, t in enumerate(tickers)
        }
    }

# def _solve_qcqp_with_uncertainty(mu_hist_ann: np.ndarray,
#                                  Sigma_ann: np.ndarray,
#                                  tickers: list[str]):
#     """
#     Instrumented QCQP with per-constraint diagnostics.
#     """
#     n = len(tickers)
#     diag = {"n": n, "notes": []}

#     # Build s and blended mu (per ticker)
#     s_vec = np.zeros(n)
#     mu_vec = np.zeros(n)
#     for i, t in enumerate(tickers):
#         try:
#             _, s_i = _risk_scores_for_ticker(t)
#         except Exception as e:
#             s_i = 0.0
#             diag["notes"].append(f"_risk_scores_for_ticker({t}) failed: {e}")
#         s_vec[i] = max(0.0, s_i)
#         try:
#             mu_vec[i] = _blended_mu_from_models(t, mu_hist_ann[i])
#         except Exception as e:
#             mu_vec[i] = mu_hist_ann[i]
#             diag["notes"].append(f"_blended_mu_from_models({t}) failed, using hist: {e}")

#     # Inflate covariance, then enforce PSD (with ridge if needed)
#     Sigma_adj_raw = Sigma_ann + BETA_COV_INFLATE * np.diag(s_vec ** 2)
#     Sigma_adj, psd_info = _make_psd(Sigma_adj_raw, ridge_min=1e-10)
#     diag["Sigma_psd_info"] = psd_info

#     # Quick feasibility checks implied by caps & SOC
#     cap_feasible = (n * W_MAX >= 1.0 - 1e-12)
#     diag["cap_check"] = {
#         "n": n, "W_MAX": W_MAX, "n_times_W_MAX": n * W_MAX, "sum1_possible": cap_feasible
#     }
#     soc_lb, w_soc_lb, note_lb = _soc_lower_bound(s_vec, W_MAX)
#     diag["soc_bound_check"] = {
#         "soc_lower_bound": soc_lb,
#         "KAPPA_UNCERT": KAPPA_UNCERT,
#         "feasible_given_caps": (soc_lb <= KAPPA_UNCERT + 1e-12),
#         "note": note_lb,
#         "w_at_lb": {t: float(w_soc_lb[i]) for i, t in enumerate(tickers)}
#                  if w_soc_lb.size == n else {}
#     }
#     # Auto-relax KAPPA if infeasible (optional)
#     KAPPA_EFF = KAPPA_UNCERT
#     if soc_lb > KAPPA_UNCERT + 1e-12:
#         KAPPA_EFF = float(soc_lb * 1.01)  # +1% margin
#         diag.setdefault("auto_repair", {})["kappa_relaxed_from"] = KAPPA_UNCERT
#         diag["auto_repair"]["kappa_relaxed_to"] = KAPPA_EFF

#     # Build CVXPY problem
#     w = cp.Variable(n)
#     Sigma_psd = cp.psd_wrap(Sigma_adj)
#     obj = cp.Minimize(0.5 * cp.quad_form(w, Sigma_psd) - LAMBDA * (mu_vec @ w))
#     cons = [cp.sum(w) == 1, w >= 0, w <= W_MAX]

#     # (optional) Warm-start with the greedy feasible point if we relaxed kappa
#     if KAPPA_EFF > KAPPA_UNCERT and w_soc_lb.size == n:
#         w.value = w_soc_lb.copy()

#     # IMPORTANT: use KAPPA_EFF here (not KAPPA_UNCERT)
#     soc_con = None
#     if np.any(s_vec > 0):
#         soc_con = (cp.norm(cp.multiply(s_vec, w), 2) <= KAPPA_EFF)
#         cons.append(soc_con)

#     prob = cp.Problem(obj, cons)

#     # Try solvers with diagnostics
#     solver_used = None
#     try:
#         # OSQP can't handle SOC; we'll try it first only if no SOC is present.
#         if soc_con is None:
#             solver_used = "OSQP"
#             prob.solve(solver=cp.OSQP, verbose=False)
#         else:
#             raise RuntimeError("Skip OSQP for SOC problems")
#         if w.value is None:
#             raise RuntimeError("OSQP returned no solution")
#     except Exception:
#         solver_used = "SCS"
#         prob.solve(solver=cp.SCS, verbose=False)

#     status = prob.status
#     diag["solver"] = {"used": solver_used, "status": status}

#     if w.value is None or status not in ("optimal", "optimal_inaccurate"):
#         diag["failure"] = "QCQP failed to converge or no solution produced."
#         # Provide final constraint values at the best-known *heuristic* feasible point, if we had one
#         if cap_feasible:
#             w0 = np.array([w_soc_lb[i] for i in range(n)], dtype=float)
#             diag["at_heuristic_point"] = _eval_constraint_values(w0, s_vec, tickers)
#         raise RuntimeError("QCQP failed to converge.")

#     # Extract solution and evaluate constraints/slacks
#     w_sol = np.array(w.value).reshape(-1)
#     diag["solution_constraints"] = _eval_constraint_values(w_sol, s_vec, tickers)

#     # Duals (when available)
#     duals = {}
#     try:
#         duals["sum_eq"]  = float(cons[0].dual_value) if cons[0].dual_value is not None else None
#         duals["geq_0"]   = cons[1].dual_value.tolist() if cons[1].dual_value is not None else None
#         duals["leq_wmax"]= cons[2].dual_value.tolist() if cons[2].dual_value is not None else None
#         if soc_con is not None:
#             # Some solvers expose SOC duals differently; this may be None
#             duals["soc_dual"] = getattr(soc_con, "dual_value", None)
#     except Exception:
#         pass
#     diag["duals"] = duals

#     return w_sol, mu_vec, s_vec, Sigma_adj, diag

def _solve_qcqp_with_uncertainty(mu_hist_ann: np.ndarray,
                                 Sigma_ann: np.ndarray,
                                 tickers: list[str]):
    """
    QCQP with relative SOC:
        minimize   0.5 w' Σ_adj w - λ μ' w
        s.t.       1'w = 1, 0 <= w <= W_MAX,
                   ||diag(s) w||_2 <= TAU_REL * sqrt(w' Σ_adj w)

    Implemented via epigraph:
        let t >= sqrt(w' Σ_adj w)  (SOC: ||A w||_2 <= t with A = Σ_adj^{1/2})
        enforce ||diag(s) w||_2 <= TAU_REL * t
    """
    n = len(tickers)
    diag = {"n": n, "notes": []}

    # --- build s and blended mu ---
    s_vec = np.zeros(n)
    mu_vec = np.zeros(n)
    for i, t in enumerate(tickers):
        try:
            _, s_i = _risk_scores_for_ticker(t)
        except Exception as e:
            s_i = 0.0
            diag["notes"].append(f"_risk_scores_for_ticker({t}) failed: {e}")
        s_vec[i] = max(0.0, s_i)
        try:
            mu_vec[i] = _blended_mu_from_models(t, mu_hist_ann[i])
        except Exception as e:
            mu_vec[i] = mu_hist_ann[i]
            diag["notes"].append(f"_blended_mu_from_models({t}) failed, using hist: {e}")

    # --- covariance inflation and PSD fix ---
    Sigma_adj_raw = Sigma_ann + BETA_COV_INFLATE * np.diag(s_vec ** 2)
    Sigma_adj, psd_info = _make_psd(Sigma_adj_raw, ridge_min=1e-10)
    diag["Sigma_psd_info"] = psd_info

    # --- quick cap feasibility (box + sum) ---
    cap_feasible = (n * W_MAX >= 1.0 - 1e-12)
    diag["cap_check"] = {"n": n, "W_MAX": W_MAX, "n_times_W_MAX": n * W_MAX, "sum1_possible": cap_feasible}

    # --- greedy point that minimizes ||diag(s) w|| under box+sum (for reference only) ---
    soc_lb, w_soc_lb, note_lb = _soc_lower_bound(s_vec, W_MAX)
    diag["soc_bound_check_absolute"] = {
        "soc_lower_bound": soc_lb,
        "note": note_lb,
        "w_at_lb": {tickers[i]: float(w_soc_lb[i]) for i in range(n)} if w_soc_lb.size == n else {}
    }

    # --- build problem with relative SOC ---
    # factor Sigma for SOC: ||A w||_2 <= t  <=>  t >= sqrt(w' Σ w)
    A = _psd_sqrt(Sigma_adj)

    w = cp.Variable(n)
    t_var = cp.Variable(nonneg=True)
    Sigma_psd = cp.psd_wrap(Sigma_adj)

    obj = cp.Minimize(0.5 * cp.quad_form(w, Sigma_psd) - LAMBDA * (mu_vec @ w))
    cons = [
        cp.sum(w) == 1,
        w >= 0,
        w <= W_MAX,
        cp.norm(A @ w, 2) <= t_var,                               # volatility epigraph
        cp.norm(cp.multiply(s_vec, w), 2) <= TAU_REL * t_var      # relative uncertainty
    ]

    prob = cp.Problem(obj, cons)

    # Try SCS (handles SOC); OSQP isn't applicable here
    solver_used = "SCS"
    prob.solve(solver=cp.SCS, verbose=False)
    status = prob.status
    diag["solver"] = {"used": solver_used, "status": status, "TAU_REL": TAU_REL}

    if w.value is None or status not in ("optimal", "optimal_inaccurate"):
        # Evaluate constraints at the greedy reference point (may not satisfy relative SOC)
        if cap_feasible and w_soc_lb.size == n:
            diag["at_reference_point"] = _eval_constraint_values_rel(w_soc_lb, s_vec, Sigma_adj, TAU_REL, tickers)
        diag["failure"] = "QCQP failed to converge or no solution produced."
        raise RuntimeError("QCQP failed to converge.")

    w_sol = np.array(w.value).reshape(-1)
    diag["solution_constraints"] = _eval_constraint_values_rel(w_sol, s_vec, Sigma_adj, TAU_REL, tickers)

    # (Optional) collect duals when available
    duals = {}
    try:
        duals["sum_eq"]      = float(cons[0].dual_value) if cons[0].dual_value is not None else None
        duals["geq_0"]       = cons[1].dual_value.tolist() if cons[1].dual_value is not None else None
        duals["leq_wmax"]    = cons[2].dual_value.tolist() if cons[2].dual_value is not None else None
        duals["vol_epigraph"] = float(cons[3].dual_value) if cons[3].dual_value is not None else None
        duals["rel_soc"]     = float(cons[4].dual_value) if cons[4].dual_value is not None else None
    except Exception:
        pass
    diag["duals"] = duals

    return w_sol, mu_vec, s_vec, Sigma_adj, diag



def _eval_constraint_values(w_vec: np.ndarray, s_vec: np.ndarray, tickers: list[str]) -> dict:
    """Report each constraint's value and slack at a given point w."""
    w = np.asarray(w_vec, dtype=float)
    vals = {}
    vals["sum_w"] = float(np.sum(w))
    vals["sum_w_slack"] = float(np.sum(w) - 1.0)  # should be ~0
    vals["min_w"] = float(np.min(w))
    vals["min_w_ge0_slack"] = float(np.min(w))  # >= 0, so negative means violation
    vals["max_w"] = float(np.max(w))
    vals["max_w_le_WMAX_slack"] = float(W_MAX - np.max(w))  # >= 0 if satisfied
    soc_val = float(np.linalg.norm(s_vec * w, 2))
    vals["soc_value"] = soc_val
    vals["soc_le_kappa_slack"] = float(KAPPA_UNCERT - soc_val)  # >= 0 if satisfied
    vals["by_ticker"] = {
        t: {
            "w": float(w[i]),
            "s": float(s_vec[i]),
            "s*w_abs": float(abs(s_vec[i] * w[i]))
        }
        for i, t in enumerate(tickers)
    }
    return vals
def _soc_lower_bound(s_vec: np.ndarray, w_max: float) -> tuple[float, np.ndarray, str]:
    """
    Greedy best-case allocation (minimizes ||diag(s) w||_2) under:
      sum w = 1, 0 <= w <= w_max.
    Fills weight onto the *smallest s* first until reaching sum=1.
    Returns (lb, w_lb, note). If n*w_max < 1, it's infeasible at the box+sum level.
    """
    n = len(s_vec)
    if n == 0:
        return float('inf'), np.array([]), "No assets"
    if n * w_max < 1.0 - 1e-12:
        return float('inf'), np.zeros(n), "Infeasible: n * W_MAX < 1 (cannot sum to 1)."

    idx = np.argsort(s_vec)  # ascending s
    w = np.zeros(n)
    remaining = 1.0
    for i in idx:
        take = min(w_max, remaining)
        w[i] = take
        remaining -= take
        if remaining <= 1e-12:
            break

    # minimal possible SOC norm given the caps (best-case placement)
    lb = float(np.linalg.norm(s_vec * w, 2))
    note = "OK"
    return lb, w, note


def _make_psd(Sigma: np.ndarray, ridge_min: float = 1e-10) -> tuple[np.ndarray, dict]:
    """
    Ensure Sigma is numerically PSD by adding a small ridge if needed.
    Returns (Sigma_psd, info)
    """
    # symmetrize
    Sigma = 0.5 * (Sigma + Sigma.T)
    eigvals = np.linalg.eigvalsh(Sigma)
    lam_min = float(np.min(eigvals))
    lam_max = float(np.max(eigvals))
    info = {
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "cond_est": (lam_max / max(lam_min, 1e-18)) if lam_max > 0 else float('inf'),
        "ridge_added": 0.0
    }
    if lam_min < 0:
        ridge = max(ridge_min, -lam_min + 1e-12)
        Sigma = Sigma + ridge * np.eye(Sigma.shape[0])
        info["ridge_added"] = ridge
        # recompute quick stats
        eigvals2 = np.linalg.eigvalsh(Sigma)
        info["lambda_min_after"] = float(np.min(eigvals2))
        info["lambda_max_after"] = float(np.max(eigvals2))
        info["cond_est_after"] = info["lambda_max_after"] / max(info["lambda_min_after"], 1e-18)
    return Sigma, info

def _portfolio_stats(w: np.ndarray, mu_ann: np.ndarray, Sigma_ann: np.ndarray, rf: float):
    mu_p = float(w @ mu_ann)
    var_p = float(w @ Sigma_ann @ w)
    sig_p = float(np.sqrt(max(0.0, var_p)))
    sharpe = (mu_p - rf) / sig_p if sig_p > 0 else float("nan")
    return mu_p, sig_p, sharpe

def _allocations_plot_base64(labels: list[str], weights: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, weights)
    ax.set_ylim(0, max(0.55, float(weights.max()) + 0.05))
    ax.set_title("Optimized Portfolio Weights")
    ax.set_ylabel("Weight")
    ax.grid(True, axis="y", alpha=0.3)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# -------------- main view --------------

def PortfolioOptimizer(request):
    """
    Build a model-aware optimized portfolio and render HTML inline (as you do in Display()).
    Query (optional):
      ?tickers=NVDA,MSFT,TXN,AAPL
      ?span=30min
      ?lookback_days=252
      ?wmax=0.5&lambda=2.5&gamma=0.6&beta=0.25&kappa=0.18
      ?format=json  -> return JSON instead of HTML
    """
    # params
    span = request.GET.get("span", DEFAULT_SPAN)
    lookback_days = int(request.GET.get("lookback_days", LOOKBACK_DAYS))
    if request.GET.get("tickers"):
        tickers = [t.strip().upper() for t in request.GET.get("tickers").split(",") if t.strip()]
    else:
        tickers = list(Stock.objects.values_list("ticker", flat=True).distinct())

    # knobs (optional overrides)
    global W_MAX, LAMBDA, GAMMA_MODEL_BLEND, BETA_COV_INFLATE, KAPPA_UNCERT
    W_MAX = float(request.GET.get("wmax", W_MAX))
    LAMBDA = float(request.GET.get("lambda", LAMBDA))
    GAMMA_MODEL_BLEND = float(request.GET.get("gamma", GAMMA_MODEL_BLEND))
    BETA_COV_INFLATE = float(request.GET.get("beta", BETA_COV_INFLATE))
    KAPPA_UNCERT = float(request.GET.get("kappa", KAPPA_UNCERT))

    # build historical stats
    R = _fetch_returns(tickers, span, lookback_days)
    mu_hist, Sigma, _ = _annualize_mu_sigma(R, span)

    # solve QCQP with model-aware μ and uncertainty
    w, mu_blend, s_vec, Sigma_adj, diag = _solve_qcqp_with_uncertainty(mu_hist, Sigma, tickers)

    # stats (Sharpe uses the adjusted covariance & blended μ since that’s the portfolio the user buys)
    mu_p, sig_p, sharpe = _portfolio_stats(w, mu_blend, Sigma_adj, rf=RF_ANNUAL)

    payload = {
        "status": "ok",
        "tickers": tickers,
        "weights": {t: float(x) for t, x in zip(tickers, w)},
        "mu_annual": float(mu_p),
        "vol_annual": float(sig_p),
        "sharpe": float(sharpe),
        "rf_annual": RF_ANNUAL,
        "uncertainty_s": {t: float(s) for t, s in zip(tickers, s_vec)},
        "mu_hist_annual": {t: float(x) for t, x in zip(tickers, mu_hist)},
        "mu_blended_annual": {t: float(x) for t, x in zip(tickers, mu_blend)},
        "params": {
            "W_MAX": W_MAX,
            "LAMBDA": LAMBDA,
            "GAMMA_MODEL_BLEND": GAMMA_MODEL_BLEND,
            "BETA_COV_INFLATE": BETA_COV_INFLATE,
            "TAU_REL": TAU_REL,           # <--- add
#            "KAPPA_UNCERT": KAPPA_UNCERT,
            "span": span,
            "lookback_days": lookback_days
        },
        "qcqp_diagnostics": diag  # <---- new
    }

    # JSON mode
    if request.GET.get("format") == "json":
        return JsonResponse(payload)

    # --- HTML like your MSFT example ---
    # chart
    encoded = _allocations_plot_base64(tickers, w)
    # info table
    rows_html = []
    for t in tickers:
        rows_html.append(
            f"<tr><td>{t}</td>"
            f"<td style='text-align:right;'>{payload['weights'][t]:.2%}</td>"
            f"<td style='text-align:right;'>{payload['mu_blended_annual'][t]:.2%}</td>"
            f"<td style='text-align:right;'>{payload['uncertainty_s'][t]:.4f}</td>"
            "</tr>"
        )
    table_html = (
        "<table style='border-collapse:collapse;width:100%;max-width:720px;' class='centerTable'>"
        "<thead><tr>"
        "<th>Ticker</th>"
        "<th>Weight</th>"
        "<th>μ (annual, blended)</th>"
        "<th>Model risk s</th>"
        "</tr></thead>"
        "<tbody>" + "".join(
            r.replace("<td", "<td style='padding:6px;border-bottom:1px solid #eee;'")
            for r in rows_html
        ) + "</tbody></table>"
    )

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Optimize Portfolio</title></head>"
        "<body style='margin:0;padding:16px;font-family:sans-serif;'>"
        "<h3>Optimized Portfolio (Model-aware QCQP)</h3>"
        f"<p style='margin:0 0 10px 0;'>Sharpe (vs {RF_ANNUAL:.2%} rf): <b>{sharpe:.3f}</b> &nbsp; "
        f"&middot; Expected μ (annual): <b>{mu_p:.2%}</b> &nbsp; "
        f"&middot; Vol (annual): <b>{sig_p:.2%}</b></p>"
        f"<img style='max-width:100%;height:auto;margin:10px 0 16px 0;' src='data:image/png;base64,{encoded}' />"
        f"{table_html}"
        "</body></html>"
    )

    with open('optimize_portfolio.html', 'w', encoding='utf-8') as f:
        f.write(html)
    return render(request, 'display.html', {'html': html})
