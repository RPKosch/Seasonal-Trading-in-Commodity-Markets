import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- User parameters ---
L          = 12        # SSA window length
J          = 2         # Number of components to reconstruct
K_SELECT   = 2         # How many tickers to long/short
# Define analysis window by year & month:
START_YEAR, START_MONTH             = 2001, 1
FINAL_END_YEAR, FINAL_END_MONTH     = 2024, 11
# Lookback in years (None => full history)
LOOKBACK_YEARS   = None

# --- Compute actual date endpoints ---
start_date = datetime(START_YEAR, START_MONTH, 1)
# The “final end” is the last day of the month:
final_end = (datetime(FINAL_END_YEAR, FINAL_END_MONTH, 1)
             + pd.offsets.MonthEnd(0))

# --- Helper: SSA‐based score for one pandas Series ---
def ssa_score(series: pd.Series, L: int, J: int):
    x = series.values.astype(float)
    N = len(x)
    if N < L:
        return np.full(N, np.nan), np.nan

    # 1. Hankel embedding
    K = N - L + 1
    X = np.column_stack([x[i:i+L] for i in range(K)])
    # 2. SVD on lag‐covariance
    S = X @ X.T
    λ, U = np.linalg.eigh(S)
    idx = np.argsort(λ)[::-1]
    λ, U = λ[idx], U[:, idx]
    # 3. Reconstruct top‐J components
    Xj = sum(
        np.sqrt(λ[m]) * np.outer(
            U[:, m],
            (X.T @ U[:, m]) / np.sqrt(λ[m])
        )
        for m in range(J)
    )
    # 4. Diagonal averaging (Hankelization)
    rec    = np.zeros(N)
    counts = np.zeros(N)
    for i in range(L):
        for j in range(K):
            rec[i+j]    += Xj[i, j]
            counts[i+j] += 1
    rec /= counts
    # 5. Final score = mean of last L reconstructed values
    score = rec[-L:].mean()
    return rec, score

# --- Gather all assets ---
project_root = Path().resolve().parent.parent
monthly_dir  = project_root / "Complete Data" / "All_Monthly_Return_Data"
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

    # ensure returns are numeric
    series = df['return'].astype(float)

    # compute SSA score
    _, final_score = ssa_score(series, L, J)
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
