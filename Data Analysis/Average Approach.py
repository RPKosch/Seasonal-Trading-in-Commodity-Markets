import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Load your two-column CSV (Date,CL=F)
df = pd.read_csv("crude_monthly_close.csv", parse_dates=["Date"], index_col="Date")
price = df["CL=F"].sort_index()

# 2) Compute log returns
returns = np.log(price / price.shift(1)).dropna()

# 3) Compute average return by calendar month
seasonal_avg = returns.groupby(returns.index.month).mean()

# 4) Compute leave-one-out z-score for each month
loo_z = {}
for m in seasonal_avg.index:
    others = seasonal_avg.drop(m)
    mu = others.mean()
    sigma = others.std(ddof=0)
    loo_z[m] = (seasonal_avg[m] - mu) / sigma
loo_z = pd.Series(loo_z).sort_index()

# 5) Plot
plt.figure(figsize=(8, 4))
plt.bar(loo_z.index, loo_z.values, color="coral")
plt.axhline(0, linestyle="--", color="gray")
plt.xticks(range(1, 13))
plt.xlabel("Month")
plt.ylabel("Leave-One-Out Seasonal Z-Score")
plt.title("Leave-One-Out Standardized Monthly Seasonal Profile")
plt.tight_layout()
plt.show()
