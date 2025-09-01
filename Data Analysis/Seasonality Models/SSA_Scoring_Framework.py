import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- User parameters ---
SSA_WINDOW          = 12        # SSA window length
SSA_COMPS          = 2         # Number of components to reconstruct
K_SELECT   = 2         # How many tickers to long/short
# Define analysis window by year & month:
START_YEAR, START_MONTH             = 2001, 1
FINAL_END_YEAR, FINAL_END_MONTH     = 2010, 12
# Lookback in years (None => full history)
LOOKBACK_YEARS   = 10

# --- Compute actual date endpoints ---
start_date = datetime(START_YEAR, START_MONTH, 1)
# The “final end” is the last day of the month:
final_end = (datetime(FINAL_END_YEAR, FINAL_END_MONTH, 1)
             + pd.offsets.MonthEnd(0))

# --- Helper: SSA‐based score for one pandas Series ---
def compute_ssa(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    N = len(x); L = int(SSA_WINDOW); r = int(SSA_COMPS)
    if N < max(L, 3) or L <= 1 or L >= N or r < 1: return np.nan
    if np.isnan(x).any(): return np.nan

    K = N - L + 1
    X = np.column_stack([x[i:i+L] for i in range(K)])  # L x K

    S = X @ X.T
    eigvals, eigvecs = np.linalg.eigh(S)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
    eps = 1e-12
    pos = eigvals > eps
    if not np.any(pos): return np.nan
    eigvals = eigvals[pos]; eigvecs = eigvecs[:, pos]

    r_eff = int(min(r, eigvals.size))
    U = eigvecs[:, :r_eff]
    sigma = np.sqrt(eigvals[:r_eff])
    V = (X.T @ U) / sigma

    Xr = (U * sigma) @ V.T

    rec = np.zeros(N); cnt = np.zeros(N)
    for i in range(L):
        for j in range(K):
            rec[i + j] += Xr[i, j]
            cnt[i + j] += 1.0
    if not np.all(cnt > 0): return np.nan
    rec /= cnt

    P_head = U[:-1, :]
    pi = U[-1, :]
    nu2 = float(np.dot(pi, pi))
    if 1.0 - nu2 <= 1e-10: return np.nan

    R = (P_head @ pi) / (1.0 - nu2)
    a = R[::-1]

    lags = rec[-1: -L: -1]
    if lags.size != a.size: return np.nan
    return float(np.dot(a, lags))

# --- Gather all assets ---
project_root = Path().resolve().parent.parent
monthly_dir  = project_root / "Complete Data" / "All_Monthly_Log_Return_Data"
paths        = list(monthly_dir.glob("*_Monthly_Revenues.csv"))
ASSET_LIST   = [p.stem.replace("_Monthly_Revenues","") for p in paths]

scores = {}
for ticker in ASSET_LIST:
    path = monthly_dir / f"{ticker}_Monthly_Revenues.csv"
    df   = pd.read_csv(path)
    # build a datetime index for month‐start
    df['date'] = pd.to_datetime(df[['year','month']].assign(day=1))
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # restrict to our window
    df = df.loc[start_date:final_end]

    # apply lookback if requested
    if LOOKBACK_YEARS is not None:
        cutoff = final_end - relativedelta(years=LOOKBACK_YEARS)
        df     = df.loc[cutoff:final_end]

    print(f"df {df}")

    # ensure returns are numeric
    series = df['return'].astype(float)

    # compute SSA score
    final_score = compute_ssa(series)
    scores[ticker] = final_score

# --- Rank and select ---
final_scores   = pd.Series(scores, name="SSA_Seasonality_Score")
sorted_scores  = final_scores.sort_values(ascending=False)

long_tickers   = sorted_scores.head(K_SELECT).index.tolist()
short_tickers  = sorted_scores.tail(K_SELECT).index.tolist()

# --- Display ---
print("Assets detected:", ASSET_LIST)
print(f"Analysis window: {start_date.date()} → {final_end.date()}")
if LOOKBACK_YEARS is not None:
    print(f"  (only using last {LOOKBACK_YEARS} years of data)")
print("\nSSA Seasonality Scores:")
print(sorted_scores.to_string(), "\n")
print(f"→ Long candidates:  {long_tickers}")
print(f"→ Short candidates: {short_tickers}")
