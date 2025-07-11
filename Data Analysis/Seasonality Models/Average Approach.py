import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# User-adjustable parameters
# ----------------------------
TICKER = "CC"
project_root = Path().resolve().parent.parent
data_path = project_root / "Complete Data" / "All_Monthly_Return_Data" / f"{TICKER}_Monthly_Revenues.csv"

# 1) Load CSV and filter ticker
df = pd.read_csv(data_path)
df = df[df['ticker'] == TICKER]

# 2) Ensure return column numeric
df['return'] = pd.to_numeric(df['return'], errors='coerce')
df = df.dropna(subset=['return', 'year', 'month'])

# 3) Compute average return by calendar month
seasonal_avg = df.groupby('month')['return'].mean()

# 4) Compute leave-one-out z-score for each month
loo_z = {}
for m in range(1, 13):
    others = seasonal_avg.drop(m)
    mu = others.mean()
    sigma = others.std(ddof=0)
    loo_z[m] = (seasonal_avg[m] - mu) / sigma
loo_z = pd.Series(loo_z)

# 5) Plot
plt.figure(figsize=(8, 4))
plt.bar(loo_z.index, loo_z.values, color="coral")
plt.axhline(0, linestyle="--", color="gray")
plt.xticks(range(1, 13))
plt.xlabel("Month")
plt.ylabel("Leave-One-Out Seasonal Z-Score")
plt.title(f"{TICKER}: Leave-One-Out Standardized Monthly Seasonal Profile")
plt.tight_layout()
plt.show()
