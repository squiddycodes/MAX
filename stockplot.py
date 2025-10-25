import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import base64
from io import BytesIO

# --- Load the data ---
with open('msft_30min.json', 'r') as f:
    data = json.load(f)

# --- Extract time and closing prices ---
times = [datetime.fromisoformat(entry['iso_ny']) for entry in data]
close_prices = [entry['c'] for entry in data]

# --- Detect gaps and create continuous x-axis ---
gap_threshold = timedelta(minutes=35)  # Slightly more than 30 min to detect gaps
x_positions = [0]  # Start at position 0
gap_locations = []  # Store where gaps occur (start index of new sessions)

for i in range(1, len(times)):
    time_diff = times[i] - times[i - 1]
    if time_diff > gap_threshold:
        # Gap detected - record position where the new session starts
        x_positions.append(x_positions[-1] + 1)
        gap_locations.append(x_positions[-1])  # mark session boundary
    else:
        # Normal progression
        x_positions.append(x_positions[-1] + 1)

# --- Create the plot ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_positions, close_prices, label='MSFT Closing Price', linewidth=2)

# --- NO vertical lines (removed) ---

# --- Format axes ---
ax.set_title("MSFT Stock Closing Prices Over Time (Gaps Removed)", fontsize=14)
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
