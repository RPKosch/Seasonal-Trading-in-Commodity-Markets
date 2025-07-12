import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# User-adjustable parameters
# ----------------------------
TICKER     = "CO"                   # Ticker to analyze
start_date = "2000-01-01"           # Analysis start date (inclusive)
end_date   = "2026-12-31"           # Analysis end date (inclusive)

# ----------------------------
# File paths (no change needed)
# ----------------------------
project_root = Path().resolve().parent.parent
data_path    = project_root / "Complete Data" / "All_Monthly_Return_Data" / f"{TICKER}_Monthly_Revenues.csv"

# 1) Load CSV and filter ticker
df = pd.read_csv(data_path)
df = df[df['ticker'] == TICKER]

# 2) Ensure return column numeric, drop bad rows
df['return'] = pd.to_numeric(df['return'], errors='coerce')
df = df.dropna(subset=['return', 'year', 'month'])

# 3) Build a proper date column and filter by date range
df['date'] = pd.to_datetime(
    df['year'].astype(int).astype(str) + '-' +
    df['month'].astype(int).astype(str).str.zfill(2) + '-01'
)
mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
df = df.loc[mask]

# 4) Compute average return by calendar month
seasonal_avg = df.groupby(df['date'].dt.month)['return'].mean()

# --- NEW: print the averages ---
print("Average revenue by calendar month (1=Jan … 12=Dec):")
for month, avg in seasonal_avg.items():
    print(f"  Month {month:2d}: {avg:.4f}")
print()

# 5) Compute leave-one-out z-score for each month
loo_z = {}
for m in range(1, 13):
    others = seasonal_avg.drop(m)
    mu    = others.mean()
    sigma = others.std(ddof=0)
    loo_z[m] = (seasonal_avg.get(m, np.nan) - mu) / sigma
loo_z = pd.Series(loo_z)

# 6) Plot
plt.figure(figsize=(8, 4))
plt.bar(loo_z.index, loo_z.values, color="coral")
plt.axhline(0, linestyle="--", color="gray")
plt.xticks(range(1, 13))
plt.xlabel("Month")
plt.ylabel("Leave-One-Out Seasonal Z-Score")
plt.title(f"{TICKER}: Leave-One-Out Standardized Monthly Seasonal Profile\n"
          f"({start_date} through {end_date})")
plt.tight_layout()
plt.show()
