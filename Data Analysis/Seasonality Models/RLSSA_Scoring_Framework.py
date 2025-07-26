import numpy as np
import pandas as pd
from pathlib import Path
import scipy.linalg as la
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- User parameters ---
L                   = 12    # SSA window length
q                   = 2     # Number of L1‐robust components
K_SELECT            = 2     # How many tickers to long/short
# Define analysis window by year & month:
START_YEAR, START_MONTH             = 2001, 1
FINAL_END_YEAR, FINAL_END_MONTH     = 2015, 12
# Lookback in years (None => full history)
LOOKBACK_YEARS      = 10

# --- Date endpoints ---
start_date = datetime(START_YEAR, START_MONTH, 1)
final_end  = (datetime(FINAL_END_YEAR, FINAL_END_MONTH, 1)
              + pd.offsets.MonthEnd(0))

# --- Robust low‐rank helper ---
def robust_low_rank(X, q, max_iter=25, eps=1e-7):
    L_, K = X.shape
    U0, s0, V0t = la.svd(X, full_matrices=False)
    U = U0[:, :q] * np.sqrt(s0[:q])
    V = (V0t[:q, :].T) * np.sqrt(s0[:q])
    for _ in range(max_iter):
        R = X - U @ V.T
        W = 1.0 / (np.abs(R) + eps)
        Xw = np.sqrt(W) * X
        Uw, sw, Vwt = la.svd(Xw, full_matrices=False)
        U = Uw[:, :q] * np.sqrt(sw[:q])
        V = (Vwt[:q, :].T) * np.sqrt(sw[:q])
    return U, V

# --- RLSSA‐score function ---
def rlssa_score(series: pd.Series, L: int, q: int):
    x = series.values.astype(float)
    N = len(x)
    if N < L:
        return np.nan
    K = N - L + 1
    X = np.column_stack([x[i:i+L] for i in range(K)])
    U, V = robust_low_rank(X, q)
    X_rob = U @ V.T
    # diagonal averaging
    rec, counts = np.zeros(N), np.zeros(N)
    for i in range(L):
        for j in range(K):
            rec[i+j]    += X_rob[i, j]
            counts[i+j] += 1
    rec /= counts
    # score = mean of last L reconstructed points
    return rec[-L:].mean()

# --- Gather assets ---
project_root = Path().resolve().parent.parent
monthly_dir  = project_root / "Complete Data" / "All_Monthly_Log_Return_Data"
paths        = list(monthly_dir.glob("*_Monthly_Revenues.csv"))
ASSET_LIST   = [p.stem.replace("_Monthly_Revenues","") for p in paths]

scores = {}
for ticker in ASSET_LIST:
    path = monthly_dir / f"{ticker}_Monthly_Revenues.csv"
    df   = pd.read_csv(path)
    df['date'] = pd.to_datetime(df[['year','month']].assign(day=1))
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # restrict to our analysis window
    df = df.loc[start_date:final_end]
    print(df)

    # apply lookback if requested
    if LOOKBACK_YEARS is not None:
        cutoff = final_end - relativedelta(years=LOOKBACK_YEARS)
        df     = df.loc[cutoff:final_end]

    series = df['return'].astype(float)
    scores[ticker] = rlssa_score(series, L, q)

# --- Rank & select ---
final_scores  = pd.Series(scores, name="RLSSA_Score").to_frame()
sorted_scores = final_scores.sort_values("RLSSA_Score", ascending=False)

long_tickers  = sorted_scores.head(K_SELECT).index.tolist()
short_tickers = sorted_scores.tail(K_SELECT).index.tolist()

# --- Display ---
print("Detected tickers:", ASSET_LIST)
print(f"Analysis window: {start_date.date()} → {final_end.date()}")
if LOOKBACK_YEARS is not None:
    print(f"  (using last {LOOKBACK_YEARS} years only)\n")
print("RLSSA Scores:\n", sorted_scores.to_string(), "\n")
print(f"→ Long candidates:  {long_tickers}")
print(f"→ Short candidates: {short_tickers}")
