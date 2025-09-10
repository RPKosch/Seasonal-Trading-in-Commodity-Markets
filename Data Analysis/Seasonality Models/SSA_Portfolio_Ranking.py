import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from decimal import Decimal
import scipy.linalg as la

# -----------------------------------------------------------------------------
# 1) PARAMETERS & WINDOWS
# -----------------------------------------------------------------------------
START_DATE       = datetime(2001, 1, 1)
LOOKBACK_YEARS   = 10     # for building SSA history
SIM_YEARS        = 5      # years for the test simulation
FINAL_END        = datetime(2024,12,31)
DEBUG_MONTH      = None   # e.g. datetime(2016,1,1)

SSA_WINDOW       = 12
SSA_COMPS        = 2
START_VALUE      = 1000.0

# Ranking mode: raw SSA vs. EWMA-adjusted SSA (SSA / sigma_EWMA over same 10y slice)
USE_EWMA_SCALE   = False
EWMA_LAMBDA      = 0.94     # monthly lambda; alpha = 1 - lambda
MIN_OBS_FOR_VOL  = 12       # min months for EWMA

# -----------------------------------------------------------------------------
# 2) DATE RANGES (no split/calibration period)
# -----------------------------------------------------------------------------
SSA_START      = (START_DATE + relativedelta(years=LOOKBACK_YEARS))
SSA_END        = FINAL_END

# Test simulation: immediately after lookback, for SIM_YEARS
TEST_SIM_START = START_DATE + relativedelta(years=LOOKBACK_YEARS)
TEST_SIM_END   = TEST_SIM_START + relativedelta(years=SIM_YEARS) - pd.offsets.MonthEnd(1)

# Final simulation: month after test sim ends → FINAL_END
FINAL_SIM_START = TEST_SIM_END + pd.offsets.MonthBegin(1)
FINAL_SIM_END   = FINAL_END

print(f"SSA history     : {SSA_START.date()} → {SSA_END.date()}")
print(f"Test Simulation : {TEST_SIM_START.date()} → {TEST_SIM_END.date()}")
print(f"Final Simulation: {FINAL_SIM_START.date()} → {FINAL_SIM_END.date()}")

# -----------------------------------------------------------------------------
# 3) HELPERS
# -----------------------------------------------------------------------------
def load_monthly_returns(root_dir: Path) -> dict[str,pd.Series]:
    out = {}
    for p in sorted(root_dir.glob("*_Monthly_Revenues.csv")):
        tkr = p.stem.replace("_Monthly_Revenues", "")
        df = pd.read_csv(p)
        df['date']   = pd.to_datetime(df[['year','month']].assign(day=1))
        df['return'] = pd.to_numeric(df['return'], errors='coerce')
        out[tkr] = df.set_index('date')['return'].sort_index()
    return out

def compute_ssa(series: np.ndarray, L: int, r: int) -> float:
    """
    Standard SSA one-step forecast using the recurrent-coefficients method.

    Steps:
      1) Trajectory (Hankel) matrix X ∈ R^{L×K}, K=N-L+1
      2) Covariance S = X Xᵀ, eigendecompose S to get leading r eigenpairs
      3) Rank-r reconstruction: X_r = (U Σ) Vᵀ with U (L×r), Σ=diag(σ₁..σ_r), V (K×r)
      4) Diagonal averaging (Hankelization) → reconstructed series \hat{x}_1.. \hat{x}_N
      5) Recurrent coefficients a from last row of U:
         Let U = [u₁..u_r], π = last row of U, P_head = U without last row.
         a_rev = (P_head π) / (1 - ||π||²) gives (a_{L-1},...,a_1); flip to a = (a_1..a_{L-1})
      6) Forecast \hat{x}_{N+1} = ∑_{j=1}^{L-1} a_j · \hat{x}_{N+1-j}
    """
    x = np.asarray(series, dtype=float).ravel()
    N = x.size
    if N < max(L, 3) or L <= 1 or L >= N or r < 1 or np.isnan(x).any():
        return np.nan

    K = N - L + 1
    # 1) Trajectory matrix
    X = np.column_stack([x[i:i+L] for i in range(K)])  # L×K

    # 2) Covariance & eigen-decomposition
    S = X @ X.T
    eigvals, eigvecs = la.eigh(S)               # ascending
    order = np.argsort(eigvals)[::-1]           # descending
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eps = 1e-12
    pos = eigvals > eps
    if not np.any(pos):
        return np.nan
    eigvals = eigvals[pos]
    eigvecs = eigvecs[:, pos]

    r_eff = int(min(r, eigvals.size))
    U = eigvecs[:, :r_eff]
    sigma = np.sqrt(eigvals[:r_eff])
    # 3) V via projection (avoid forming full SVD of X)
    V = (X.T @ U) / sigma

    # 4) Rank-r reconstruction & Hankelization
    Xr = (U * sigma) @ V.T
    rec = np.zeros(N); cnt = np.zeros(N)
    for i in range(L):
        for j in range(K):
            rec[i + j] += Xr[i, j]
            cnt[i + j] += 1.0
    if not np.all(cnt > 0):
        return np.nan
    rec /= cnt

    # 5) Recurrent coefficients from U
    P_head = U[:-1, :]
    pi     = U[-1, :]
    nu2    = float(np.dot(pi, pi))
    if 1.0 - nu2 <= 1e-10:
        return np.nan
    a_rev = (P_head @ pi) / (1.0 - nu2)  # (a_{L-1},...,a_1)
    a = a_rev[::-1]                      # (a_1,...,a_{L-1})

    # 6) Forecast using reconstructed last (L-1) values
    lags = rec[-1: -L: -1]               # last L-1 values in reverse, length L-1
    if lags.size != a.size:
        return np.nan
    return float(np.dot(a, lags))

def build_ssa_history(returns: dict[str,pd.Series], L: int, r: int) -> pd.DataFrame:
    dates = pd.date_range(SSA_START, SSA_END, freq='MS')
    df = pd.DataFrame(index=dates, columns=returns.keys(), dtype=float)
    for dt in dates:
        lb0 = dt - relativedelta(years=LOOKBACK_YEARS)
        lb1 = dt - relativedelta(months=1)
        for tkr, ser in returns.items():
            df.at[dt, tkr] = compute_ssa(ser.loc[lb0:lb1].values, L=L, r=r)
    return df

def ewma_vol(series: pd.Series, lb0: datetime, lb1: datetime, lam: float, min_obs: int) -> float:
    """EWMA volatility on monthly *simple* returns over [lb0, lb1]."""
    win = series.loc[lb0:lb1].dropna()
    if len(win) < min_obs:
        return np.nan
    alpha   = 1.0 - lam
    ewma_v2 = (win**2).ewm(alpha=alpha, adjust=False).mean().iloc[-1]
    sigma   = float(np.sqrt(ewma_v2)) if np.isfinite(ewma_v2) else np.nan
    return sigma if (np.isfinite(sigma) and sigma > 0) else np.nan

def sharpe_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    if arr.size == 0:
        return np.nan
    std = arr.std(ddof=1)
    if std == 0:
        return np.nan
    return arr.mean() / std * np.sqrt(12)

def sortino_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    if arr.size == 0:
        return np.nan
    neg = arr[arr < 0]
    if neg.size == 0:
        return np.nan
    return arr.mean() / neg.std(ddof=1) * np.sqrt(12)

def calmar_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    if arr.size == 0:
        return np.nan
    cum  = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(cum)
    dd   = cum/peak - 1
    mdd  = abs(dd.min())
    years= len(arr)/12
    if mdd == 0 or years == 0:
        return np.nan
    cagr = cum[-1] ** (1/years) - 1
    return cagr / mdd

# === Normalization helpers for GPS ===========================================
def minmax_01(arr_like) -> np.ndarray:
    """
    Min–max to [0,1] per column/vector. NaNs are preserved.
    If all finite values are equal, return 1.0 (do not penalize constant columns).
    """
    x = np.asarray(arr_like, dtype=float)
    mask = np.isfinite(x)
    if not mask.any():
        return np.full_like(x, np.nan, dtype=float)
    xmin = np.nanmin(x[mask]); xmax = np.nanmax(x[mask])
    if xmax == xmin:
        out = np.full_like(x, np.nan, dtype=float)
        out[mask] = 1.0
        return out
    out = (x - xmin) / (xmax - xmin)
    out[~mask] = np.nan
    return np.clip(out, 0.0, 1.0)

def gps_harmonic_01(vals: list[float] | np.ndarray) -> float:
    """
    Harmonic mean over values already scaled to (0,1].
    If any value is 0, GPS = 0. If any NaN -> NaN.
    """
    v = np.asarray(vals, dtype=float)
    if np.isnan(v).any():
        return np.nan
    if np.any(v < 0):
        return np.nan
    if np.any(v == 0.0):
        return 0.0
    return len(v) / np.sum(1.0 / v)

def build_metrics(cum_df, rets_dict):
    """
    Build raw metrics per rank (cum_ret, sharpe, sortino, calmar),
    then min–max each column to [0,1] across ranks and compute GPS harmonic
    on the normalized columns.
    """
    rows=[]
    for prev_rank, cum_row in cum_df.iterrows():
        rets = rets_dict[prev_rank]
        sr   = sharpe_ratio(rets)
        sor  = sortino_ratio(rets)
        cr   = calmar_ratio(rets)
        cum  = float(cum_row['cum_ret'])  # may be negative; normalized later

        rows.append({
            'prev_rank': prev_rank,
            'cum_ret'  : cum,
            'sharpe'   : sr,
            'sortino'  : sor,
            'calmar'   : cr,
        })

    df = pd.DataFrame(rows).set_index('prev_rank').sort_index()

    # Normalize each metric column independently to [0,1]
    base_cols = ['cum_ret', 'sharpe', 'sortino', 'calmar']
    for c in base_cols:
        df[f'{c}_01'] = minmax_01(df[c].values)

    # GPS harmonic on normalized columns
    norm_cols = [f'{c}_01' for c in base_cols]
    df['score'] = [
        gps_harmonic_01(df.loc[idx, norm_cols].values)
        for idx in df.index
    ]

    # Rank: higher score = better
    df['new_rank']    = df['score'].rank(ascending=False, method='first')
    df['rank_change'] = df.index - df['new_rank']
    return df

# -----------------------------------------------------------------------------
# 4) LOAD & PREP
# -----------------------------------------------------------------------------
ROOT_DIR     = Path().resolve().parent.parent / "Complete Data"
# SSA is built on LOG monthly returns (signal), performance on SIMPLE monthly returns
log_rets     = load_monthly_returns(ROOT_DIR / "All_Monthly_Log_Return_Data")
simple_rets  = load_monthly_returns(ROOT_DIR / "All_Monthly_Return_Data")

ssa_score    = build_ssa_history(log_rets, L=SSA_WINDOW, r=SSA_COMPS)
tickers      = list(log_rets)
NUM_T        = len(tickers)

# -----------------------------------------------------------------------------
# 5) RANK LONG & SHORT EACH MONTH (SSA raw or EWMA-adjusted)
# -----------------------------------------------------------------------------
long_rank, short_rank = {}, {}
cur = TEST_SIM_START
while cur <= SSA_END:
    lb0 = cur - relativedelta(years=LOOKBACK_YEARS)
    lb1 = cur - relativedelta(months=1)

    raw_list = []
    for t in tickers:
        sc = ssa_score.at[cur, t] if (cur in ssa_score.index) else np.nan
        if pd.isna(sc):
            continue
        if USE_EWMA_SCALE:
            sig = ewma_vol(simple_rets[t], lb0, lb1, EWMA_LAMBDA, MIN_OBS_FOR_VOL)
            if not np.isfinite(sig):
                continue
            score = sc / sig
        else:
            score = sc
        raw_list.append((t, sc, score))

    # Longs: highest score first; Shorts: lowest score first
    ordL = [t for (t, sc, s) in sorted(raw_list, key=lambda x: x[2], reverse=True)]
    ordS = [t for (t, sc, s) in sorted(raw_list, key=lambda x: x[2])]

    # pad with any missing tickers to keep stable dimension
    missing = [t for t in tickers if t not in [x[0] for x in raw_list]]
    ordL += missing
    ordS += missing

    if DEBUG_MONTH and cur == DEBUG_MONTH:
        print(f"--- DEBUG {cur.date()} ---")
        dfdbg = pd.DataFrame(raw_list, columns=['tkr','ssa','score']).set_index('tkr')
        print(dfdbg.sort_values('score', ascending=False).head(10))
        print("Long order:", ordL[:10], "…")
        print("Short order:", ordS[:10], "…")
        print("----------------------")

    long_rank[cur]  = ordL
    short_rank[cur] = ordS
    cur += relativedelta(months=1)

# -----------------------------------------------------------------------------
# 6) TEST & FINAL SIMULATIONS (no entry costs here)
# -----------------------------------------------------------------------------
def compute_cum_and_rets(rankings, direction='long', start_date=None, end_date=None):
    holdings = {r:{} for r in range(1,NUM_T+1)}
    rets     = {r:[] for r in range(1,NUM_T+1)}
    for dt, order in rankings.items():
        if start_date and dt < start_date: continue
        if end_date   and dt > end_date:   continue
        for r, t in enumerate(order, start=1):
            holdings[r][dt] = t
            x = Decimal(str(simple_rets[t].get(dt, 0.0)))
            if direction=='short':
                x = Decimal(1)/(Decimal(1)+x) - Decimal(1)
            rets[r].append(x)
    rows=[]
    START_D = Decimal(str(START_VALUE))
    for r in range(1,NUM_T+1):
        v = START_D
        for x in rets[r]:
            v *= (Decimal(1)+x)
        rows.append({'rank':float(r),'cum_ret':v/START_D - Decimal(1)})
    return pd.DataFrame(rows).set_index('rank'), holdings, rets

# Test simulation
test_long_cum, _, test_long_rets = compute_cum_and_rets(
    long_rank, 'long', TEST_SIM_START, TEST_SIM_END
)
metrics_long_test = build_metrics(test_long_cum, test_long_rets)

test_short_cum, _, test_short_rets = compute_cum_and_rets(
    short_rank, 'short', TEST_SIM_START, TEST_SIM_END
)
metrics_short_test = build_metrics(test_short_cum, test_short_rets)

# Final simulation using original SSA ordering
final_long_cum, _, _  = compute_cum_and_rets(
    long_rank, 'long', FINAL_SIM_START, FINAL_SIM_END
)
final_short_cum, _, _ = compute_cum_and_rets(
    short_rank,'short', FINAL_SIM_START, FINAL_SIM_END
)

# -----------------------------------------------------------------------------
# 7) OUTPUT & COMPARISON (dynamic rank ranges)
# -----------------------------------------------------------------------------
def _print_ranked_df(df: pd.DataFrame, label: str):
    if df is None or df.empty:
        print(f"\n{label}: (no data)")
        return
    dfx = df.copy()
    # try to display integer ranks nicely
    try:
        dfx.index = dfx.index.astype(int)
    except Exception:
        pass
    rmin = int(dfx.index.min()) if len(dfx.index) else 0
    rmax = int(dfx.index.max()) if len(dfx.index) else 0
    print(f"\n{label} (ranks {rmin}–{rmax}):")
    print(dfx.to_string())

print(f"Testing Portfolios for Period {TEST_SIM_START} until {TEST_SIM_END}")

_print_ranked_df(metrics_long_test,  "Metrics for LONG portfolios")
_print_ranked_df(metrics_short_test, "Metrics for SHORT portfolios")
print("-------------------------------------------------------------------------------------------------")

print(f"Final Portfolios for Period {FINAL_SIM_START} until {FINAL_SIM_END}")
_print_ranked_df(final_long_cum,   "Final LONG cumulative returns by rank")
_print_ranked_df(final_short_cum,  "Final SHORT cumulative returns by rank")

def output_comparison(metrics_test: pd.DataFrame, final_cum: pd.DataFrame) -> pd.DataFrame:
    print(f"\nTest Period : {TEST_SIM_START.date()} → {TEST_SIM_END.date()}")
    print(f"Final Period: {FINAL_SIM_START.date()} → {FINAL_SIM_END.date()}")
    rows=[]
    for prev_rank, row in metrics_test.iterrows():
        pr = int(prev_rank)
        nr = int(row['new_rank'])
        now_ret  = float(final_cum.loc[pr, 'cum_ret'])
        same_ret = float(final_cum.loc[nr, 'cum_ret'])
        diff     = now_ret - same_ret
        rows.append({
            'prev_rank': pr,
            'new_rank' : nr,
            'final_ret_now': now_ret,
            'final_ret_same_rank_prev': same_ret,
            'difference': diff,
            'score': float(row['score'])
        })
    out = pd.DataFrame(rows).set_index('new_rank').sort_index()
    try:
        out.index = out.index.astype(int)
    except Exception:
        pass
    return out

print("\nLong SSA comparison:")
print(output_comparison(metrics_long_test, final_long_cum).to_string())

print("\nShort SSA comparison:")
print(output_comparison(metrics_short_test, final_short_cum).to_string())

