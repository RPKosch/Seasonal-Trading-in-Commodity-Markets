import numpy as np
import pandas as pd
from pathlib import Path

# --- User params ---
TICKER = "CC"; L = 12; J = 2; h = 12

def ssa_forecast(series: pd.Series, L: int, J: int, h: int):
    x = series.values.astype(float)
    N = len(x)
    if L > N:
        raise ValueError(f"L ({L}) must be ≤ series length ({N})")
    K = N - L + 1
    X = np.column_stack([x[i:i+L] for i in range(K)])
    S = X @ X.T
    λ, U = np.linalg.eigh(S)
    idx = np.argsort(λ)[::-1]
    λ, U = λ[idx], U[:, idx]
    Uj = U[:, :J]
    Xj = sum(np.sqrt(λ[m]) * np.outer(U[:, m], (X.T @ U[:, m]) / np.sqrt(λ[m]))
             for m in range(J))
    rec = np.zeros(N); counts = np.zeros(N)
    for i in range(L):
        for j in range(K):
            rec[i+j] += Xj[i,j]; counts[i+j] += 1
    rec /= counts
    nu = Uj[-1, :]; mu = Uj[:-1, :]
    den = np.sum(nu**2)
    a = (mu @ nu) / den
    extended = list(rec.copy())
    for _ in range(h):
        past = extended[-(L-1):]
        extended.append(np.dot(a, past[::-1]))
    forecast = np.array(extended[N:])
    return forecast, forecast.mean()

# --- Load data ---
project_root = Path().resolve().parent
data_path = project_root / "Complete Data" / "All_Monthly_Return_Data" / f"{TICKER}_Monthly_Revenues.csv"
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df[['year','month']].assign(DAY=1))
df = df.sort_values('date').dropna(subset=['return'])
series = df.set_index('date')['return'].astype(float)

# --- Run SSA forecast ---
forecast_vals, forecast_score = ssa_forecast(series, L, J, h)

# --- Prepare future dates ---
last_date = series.index[-1]
future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(), periods=h, freq="MS")

# --- Output ---
print("\nSSA Forecast for next 12 months:\n")
for dt, val in zip(future_dates, forecast_vals):
    print(f"{dt.strftime('%Y-%m')} → {val:.6f}")
print(f"\nSSA seasonal score (mean forecast): {forecast_score:.6f}\n")
