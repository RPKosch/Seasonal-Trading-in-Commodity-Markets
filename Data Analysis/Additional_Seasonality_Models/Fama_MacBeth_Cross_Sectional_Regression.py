import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# ----------------------------
# User‑adjustable parameters
# ----------------------------
lookback_years = 20
start_date     = "1999-01-01"      # inclusive
end_date       = "2026-12-31"      # inclusive

project_root = Path().resolve().parent.parent
data_dir     = project_root / "Complete Data" / "All_Monthly_Return_Data"

# ----------------------------
# Load and pivot returns
# ----------------------------
# read all tickers’ CSVs into one DataFrame
paths = data_dir.glob("*_Monthly_Revenues.csv")
df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)

# build proper datetime and filter by date window
df['date'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1))
mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
df = df.loc[mask].sort_values(['ticker', 'date'])

# pivot to date×ticker matrix of returns
ret_matrix = df.pivot(index='date', columns='ticker', values='return')

# ----------------------------
# Compute raw SMR and OMR via shifts (vectorized)
# ----------------------------
lags = [12 * i for i in range(1, lookback_years + 1)]
smr_dfs = [ret_matrix.shift(lag) for lag in lags]
SMR_raw = pd.concat(smr_dfs, axis=1).T.groupby(level=0).mean().T

other_dfs = [
    ret_matrix.shift(lag)
    for lag in range(1, 12 * lookback_years + 1)
    if lag % 12 != 0
]
OMR_raw = pd.concat(other_dfs, axis=1).T.groupby(level=0).mean().T

# stack back to long form and merge
SMR_long = SMR_raw.stack().rename('SMR_raw').reset_index()
OMR_long = OMR_raw.stack().rename('OMR_raw').reset_index()

df = (
    df
    .merge(SMR_long, on=['date', 'ticker'])
    .merge(OMR_long, on=['date', 'ticker'])
)

# ----------------------------
# Cross‐sectional demeaning
# ----------------------------
means = (
    df
    .groupby('date')[['SMR_raw', 'OMR_raw']]
    .mean()
    .rename(columns=lambda c: c.replace('_raw', '_mean'))
)
df = df.join(means, on='date')
df['SMR'] = df['SMR_raw'] - df['SMR_mean']
df['OMR'] = df['OMR_raw'] - df['OMR_mean']

# ----------------------------
# Fama–MacBeth regressions
# ----------------------------
coef_list = []
dates     = []

for t, sub in df.groupby('date'):
    sub = sub.dropna(subset=['SMR', 'OMR', 'return'])
    if len(sub) < 3:
        continue
    X   = sm.add_constant(sub[['SMR', 'OMR']])
    y   = sub['return']
    res = sm.OLS(y, X).fit()
    coef_list.append(res.params.values)
    dates.append(t)

coefs = pd.DataFrame(coef_list,
                     columns=['alpha', 'beta', 'gamma'],
                     index=pd.to_datetime(dates))

# ----------------------------
# Overall FM averages and t‑stats
# ----------------------------
n = len(coefs)
alpha_avg, beta_avg, gamma_avg = coefs.mean().values
alpha_se = coefs['alpha'].std(ddof=1) / np.sqrt(n)
beta_se  = coefs['beta'].std(ddof=1)  / np.sqrt(n)
gamma_se = coefs['gamma'].std(ddof=1) / np.sqrt(n)

alpha_t = alpha_avg / alpha_se
beta_t  = beta_avg  / beta_se
gamma_t = gamma_avg / gamma_se

# ----------------------------
# Month‑of‑year effects on beta
# ----------------------------
coefs['month']       = coefs.index.month
monthly_beta         = coefs.groupby('month')['beta'].mean().sort_values(ascending=False)
best_months          = monthly_beta.head(3).index.tolist()

# ----------------------------
# Results
# ----------------------------
print(f"Data window: {start_date} through {end_date}\n")
print("Fama–MacBeth Average Coefficients and t‑stats:")
print(f"  Alpha: {alpha_avg:.6f} (t = {alpha_t:.2f})")
print(f"  Beta  : {beta_avg:.6f} (t = {beta_t:.2f})")
print(f"  Gamma : {gamma_avg:.6f} (t = {gamma_t:.2f})\n")

print("Average Beta by Calendar Month (stronger seasonality → higher β):")
print(monthly_beta.to_string())

print(f"\nTop 3 months to focus on (by avg Beta): {best_months}")
