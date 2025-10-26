import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import json
from datetime import datetime, timezone

def load_price_data(filename):
    """
    Loads, sorts, and extracts data from a given JSON file.
    Returns the full, sorted DataFrame.
    """
    try:
        # Load the JSON data
        df = pd.read_json(filename)
        
        # We must sort by time to ensure the series is chronological
        # Assuming 't' is the timestamp column
        df = df.sort_values(by='t')
        
        if df.empty:
            print(f"Warning: No data found in '{filename}'. Skipping.")
            return None
            
        print(f"Successfully loaded {len(df)} records from '{filename}'")
        # Return the full dataframe
        return df
        
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except KeyError:
        # Check for 'c' and 't' and 'iso_ny' for our new requirements
        print(f"Error: The JSON in '{filename}' must contain 'c', 't', and 'iso_ny' keys.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading '{filename}': {e}")
        return None

def run_monte_carlo_simulation(historical_prices, n_sims=1000):
    """
    Runs a Monte Carlo simulation using Geometric Brownian Motion (GBM)
    on a provided price series.
    
    Returns:
        (train_data, test_data, average_walk) or (None, None, None) on failure
    """
    
    # --- 1. Split Data 80/20 ---
    split_index = int(len(historical_prices) * 0.80)
    train_data = historical_prices.iloc[:split_index]
    test_data = historical_prices.iloc[split_index:]

    print(f"  Training points: {len(train_data)}, Testing points: {len(test_data)}")

    # --- 2. Calculate Drift and Volatility ---
    log_returns = np.log(train_data / train_data.shift(1)).dropna()

    if log_returns.empty:
        print("  Error: Not enough training data to calculate returns. Skipping stock.")
        return None, None, None

    mu = log_returns.mean()
    sigma = log_returns.std()

    print(f"  Calculated Drift (mu): {mu:.6f}")
    print(f"  Calculated Volatility (sigma): {sigma:.6f}")

    # --- 3. Set Up Simulation Parameters ---
    S0 = train_data.iloc[-1] # Starting price is the last *actual* price
    T = len(test_data)
    dt = 1

    all_walks = np.zeros((T + 1, n_sims))
    all_walks[0, :] = S0 # All simulations start from the same actual price

    # --- 4. Run the Simulations ---
    # All calculations are in the *actual price* domain
    for i in range(n_sims):
        for t in range(1, T + 1):
            Z = np.random.standard_normal()
            all_walks[t, i] = all_walks[t-1, i] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            )

    # --- 5. Average the Walks ---
    # FIX: Average all T+1 steps, *including* the starting price S0
    average_walk_values = np.mean(all_walks, axis=1) # This is a (T+1,) numpy array
    
    # FIX: Create the correct index: [last_train_index] + [test_indices]
    # This (T+1) length index will align with the (T+1) values
    combined_index = train_data.index[[-1]].union(test_data.index)

    # FIX: Create the final averaged series
    average_walk_series = pd.Series(average_walk_values, index=combined_index)

    # Return results in *actual price*
    return train_data, test_data, average_walk_series


# --- Main execution ---
if __name__ == "__main__":
    
    # Define the stocks to process, their files, and their colors
    stocks_to_process = {
        "MSFT": {"file": "msft_30min.json", "color": "blue"},
        "AAPL": {"file": "aapl_30min.json", "color": "green"},
        "NVDA": {"file": "nvda_30min.json", "color": "purple"},
        "TXN":  {"file": "txn_30min.json",  "color": "orange"}
    }
    
    # --- Set up the single plot ---
    plt.figure(figsize=(15, 8))
    plt.title(f"Multi-Stock Monte Carlo Simulation ({1000} Walks) - Actual Price", fontsize=16)
    plt.xlabel("Time Step")
    plt.ylabel("Actual Price ($)")
    
    processed_count = 0
    
    # --- Loop through each stock and plot on the single chart ---
    for ticker, info in stocks_to_process.items():
        print("-" * 40)
        print(f"Processing {ticker}...")
        
        color = info['color']
        
        # 1. Load data
        # full_df contains all columns, sorted by time
        full_df = load_price_data(info['file'])
        if full_df is None:
            continue
            
        # Extract just the price series for the simulation
        # We reset the index so train/test splits are clean
        price_data = full_df['c'].reset_index(drop=True)
            
        # 2. Run simulation
        # train, test, and avg_walk are all in *actual price*
        # avg_walk now includes S0, so it has T+1 points
        train, test, avg_walk = run_monte_carlo_simulation(price_data, n_sims=1000)
        
        if train is None:
            continue # Skip if simulation failed (e.g., not enough data)
            
        # 3. Plot the data on the single chart
        
        # Plot the full "Actual" data line (Train + Test)
        # We plot 'train' and 'test' directly
        full_actual_data = pd.concat([train, test])
        plt.plot(full_actual_data.index, full_actual_data, 
                 color=color, 
                 label=f"{ticker} Actual", 
                 linewidth=2)
        
        # Plot the "Average Prediction" as a dotted line
        # We plot 'avg_walk' directly
        plt.plot(avg_walk.index, avg_walk, 
                 color=color, # Use the same color as the stock
                 label=f"{ticker} Avg. Prediction", 
                 linestyle="--", 
                 linewidth=2.5)
                 
        # --- 4. Calculate Metrics and Save JSON ---
        
        # Get actual vs predicted values
        # test_values are the T actual prices
        test_values = test.values 
        # predicted_values are the T predicted prices (we skip S0)
        predicted_values = avg_walk.iloc[1:].values 

        # Calculate errors
        errors = predicted_values - test_values
        
        # Calculate ME (Mean Error / Bias)
        me = np.mean(errors)
        
        # Calculate MAE (Mean Absolute Error)
        mae = np.mean(np.abs(errors))

        # Calculate RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean(errors**2))
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # We must handle potential division by zero if test_values has a 0
        safe_test_values = np.where(test_values == 0, 1e-9, test_values)
        mape = np.mean(np.abs(errors / safe_test_values)) * 100

        print(f"  --- {ticker} Metrics ---")
        if me > 0:
            print(f"  ME (Bias): ${me:+.4f} (Overpredicted)")
        else:
            print(f"  ME (Bias): ${me:+.4f} (Underpredicted)")
        print(f"  MAE     : ${mae:.4f}")
        print(f"  RMSE    : ${rmse:.4f}")
        print(f"  MAPE    : {mape:.4f}%")

        # --- 5. Prepare and Save JSON Output ---
        
        # Get the original data for the test split
        split_index = len(train)
        test_df = full_df.iloc[split_index:].copy()
        
        predictions_list = []
        for i in range(len(test_values)):
            pred_obj = {
                "ticker": ticker,
                "index_in_test": i,
                "x_index": test.index[i],
                "iso_time": test_df['iso_ny'].iloc[i],
                "pred_close": predicted_values[i],
                "actual_close": test_values[i]
            }
            predictions_list.append(pred_obj)
            
        output_data = {
            "meta": {
                "ticker": ticker,
                "model": "MonteCarloGBM",
                "granularity": "30min",
                "lookback": len(train),
                "horizon": len(test),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "metrics": {
                    "ME": me,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE_pct": mape
                }
            },
            "predictions": predictions_list
        }
        
        # Save the JSON file
        json_filename = f"{ticker}_monte_carlo_results.json"
        try:
            with open(json_filename, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"  Successfully saved results to '{json_filename}'")
        except Exception as e:
            print(f"  Error saving JSON file: {e}")
        
        print(f"Finished processing {ticker}.")
        processed_count += 1
        
    print("-" * 40)
    
    # --- Finalize and save the plot ---
    plot_filename = "multi_stock_monte_carlo_actual_price.png"
    
    if processed_count == 0:
        print("No data was successfully processed or plotted. Exiting.")
        sys.exit(1)
        
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # Adjust layout

    plt.savefig(plot_filename)
    print(f"\nSimulation complete. Plot saved as '{plot_filename}'")

