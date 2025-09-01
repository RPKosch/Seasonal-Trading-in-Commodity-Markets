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
FINAL_END_YEAR, FINAL_END_MONTH     = 2011, 3
# Lookback in years (None => full history)
LOOKBACK_YEARS      = 10

# --- Date endpoints ---
start_date = datetime(START_YEAR, START_MONTH, 1)
final_end  = (datetime(FINAL_END_YEAR, FINAL_END_MONTH, 1)
              + pd.offsets.MonthEnd(0))

# -----------------------------------------------------------------------------
# 3) RLSSA (Robust Low-Rank SSA)
# -----------------------------------------------------------------------------
def robust_low_rank(X: np.ndarray, q: int, max_iter: int = 25, eps: float = 1e-7):
    """
    Simple IRLS-style robust low-rank approximation:
      minimize ~ sum w_{ij} (X_ij - (UV^T)_ij)^2 with w_{ij} ≈ 1/(|resid|+eps).
    Returns U (L×q), V (K×q) such that S ≈ U V^T.
    """
    U0, s0, V0t = np.linalg.svd(X, full_matrices=False)
    r0 = min(q, s0.size)
    U = U0[:, :r0] * np.sqrt(s0[:r0])
    V = (V0t[:r0, :].T) * np.sqrt(s0[:r0])

    for _ in range(max_iter):
        R = X - U @ V.T
        W = 1.0 / (np.abs(R) + eps)
        Xw = np.sqrt(W) * X
        Uw, sw, Vwt = np.linalg.svd(Xw, full_matrices=False)
        r0 = min(q, sw.size)
        U = Uw[:, :r0] * np.sqrt(sw[:r0])
        V = (Vwt[:r0, :].T) * np.sqrt(sw[:r0])

    return U, V

def compute_rlssa(series: np.ndarray, L: int, q: int) -> float:
    """
    RLSSA one-step forecast:
      1) Hankel embed X (L×K)
      2) Robust low-rank S ≈ U V^T (rank q)
      3) Hankelize S -> robust fitted rec (length N)
      4) SVD(S) -> Uc; build recurrent coefficients a from Uc
      5) Forecast \hat y_{N+1} = sum a_j * rec[N+1-j], j=1..L-1
    Returns scalar forecast or np.nan on failure.
    """
    x = np.asarray(series, dtype=float).ravel()
    if np.any(~np.isfinite(x)):
        return np.nan
    N = x.size
    if not (1 < L < N) or q < 1:
        return np.nan
    K = N - L + 1

    X = np.column_stack([x[i:i+L] for i in range(K)])  # L×K
    U_r, V_r = robust_low_rank(X, q=q)
    S = U_r @ V_r.T

    # Hankelize S
    rec = np.zeros(N, dtype=float)
    cnt = np.zeros(N, dtype=float)
    for i in range(L):
        for j in range(K):
            rec[i + j] += S[i, j]
            cnt[i + j] += 1.0
    if not np.all(cnt > 0):
        return np.nan
    rec /= cnt

    # Classical SVD on S (for recurrence)
    Uc, sc, Vct = np.linalg.svd(S, full_matrices=False)
    r_eff = int(min(q, sc.size))
    if r_eff < 1:
        return np.nan
    Uc = Uc[:, :r_eff]

    P_head = Uc[:-1, :]        # (L-1)×r
    phi    = Uc[-1, :]         # length r
    nu2    = float(np.dot(phi, phi))
    if 1.0 - nu2 <= 1e-10:
        return np.nan

    R = (P_head @ phi) / (1.0 - nu2)   # (a_{L-1},...,a_1)
    a = R[::-1]                         # (a_1,...,a_{L-1})

    lags = rec[-1: -L: -1]
    if lags.size != a.size:
        return np.nan
    return float(np.dot(a, lags))

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

    # apply lookback if requested
    if LOOKBACK_YEARS is not None:
        cutoff = final_end - relativedelta(years=LOOKBACK_YEARS)
        df     = df.loc[cutoff:final_end]
        print(f"current df -> {df}")

    series = df['return'].astype(float)
    scores[ticker] = compute_rlssa(series, L, q)

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
