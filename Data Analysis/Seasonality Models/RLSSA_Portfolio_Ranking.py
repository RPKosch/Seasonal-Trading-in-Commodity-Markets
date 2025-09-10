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
LOOKBACK_YEARS   = 10     # for building RLSSA history
SIM_YEARS        = 5      # years for the test simulation
FINAL_END        = datetime(2024,12,31)
DEBUG_MONTH      = None   # e.g. datetime(2023,7,1)

RLSSA_WINDOW     = 12
RLSSA_COMPS      = 2
START_VALUE      = 1000.0

# Ranking mode: raw RLSSA vs. EWMA-adjusted RLSSA (RLSSA / sigma_EWMA over same 10y slice)
USE_EWMA_SCALE   = False
EWMA_LAMBDA      = 0.94     # monthly lambda; alpha = 1 - lambda
MIN_OBS_FOR_VOL  = 12       # min months for EWMA

# -----------------------------------------------------------------------------
# 2) DATE RANGES (no split/calibration period)
# -----------------------------------------------------------------------------
RLSSA_START      = (START_DATE + relativedelta(years=LOOKBACK_YEARS))
RLSSA_END        = FINAL_END

# Test simulation: immediately after lookback, for SIM_YEARS
TEST_SIM_START   = START_DATE + relativedelta(years=LOOKBACK_YEARS)
TEST_SIM_END     = TEST_SIM_START + relativedelta(years=SIM_YEARS) - pd.offsets.MonthEnd(1)

# Final simulation: month after test sim ends → FINAL_END
FINAL_SIM_START  = TEST_SIM_END + pd.offsets.MonthBegin(1)
FINAL_SIM_END    = FINAL_END

print(f"RLSSA history   : {RLSSA_START.date()} → {RLSSA_END.date()}")
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

def robust_low_rank(X, q, max_iter=25, eps=1e-7):
    U0, s0, V0t = la.svd(X, full_matrices=False)
    r0 = min(q, s0.size)
    U = U0[:, :r0] * np.sqrt(s0[:r0])
    V = (V0t[:r0, :].T) * np.sqrt(s0[:r0])
    for _ in range(max_iter):
        R = X - U @ V.T
        W = 1.0 / (np.abs(R) + eps)
        Xw = np.sqrt(W) * X
        Uw, sw, Vwt = la.svd(Xw, full_matrices=False)
        r0 = min(q, sw.size)
        U = Uw[:, :r0] * np.sqrt(sw[:r0])
        V = (Vwt[:r0, :].T) * np.sqrt(sw[:r0])
    return U, V

def compute_rlssa(series: np.ndarray, L: int, q: int) -> float:
    x = np.asarray(series, dtype=float).ravel()
    if np.any(~np.isfinite(x)):
        return np.nan
    N = x.size
    if not (1 < L < N) or q < 1:
        return np.nan
    K = N - L + 1
    X = np.column_stack([x[i:i+L] for i in range(K)])
    U, V = robust_low_rank(X, q)
    S = U @ V.T

    # Hankelize S
    rec = np.zeros(N)
    cnt = np.zeros(N)
    for i in range(L):
        for j in range(K):
            rec[i+j] += S[i, j]
            cnt[i+j] += 1
    if not np.all(cnt > 0):
        return np.nan
    rec /= cnt

    # Recurrent coefficients
    Uc, sc, Vct = la.svd(S, full_matrices=False)
    r_eff = int(min(q, sc.size))
    if r_eff < 1:
        return np.nan
    Uc = Uc[:, :r_eff]
    P_head = Uc[:-1, :]
    phi    = Uc[-1, :]
    nu2    = float(np.dot(phi, phi))
    if 1.0 - nu2 <= 1e-10:
        return np.nan
    R = (P_head @ phi) / (1.0 - nu2)   # (a_{L-1},...,a_1)
    a = R[::-1]

    lags = rec[-1: -L: -1]
    if lags.size != a.size:
        return np.nan
    return float(np.dot(a, lags))

def build_rlssa_history(returns: dict[str,pd.Series]) -> pd.DataFrame:
    dates = pd.date_range(RLSSA_START, RLSSA_END, freq='MS')
    df = pd.DataFrame(index=dates, columns=returns.keys(), dtype=float)
    for dt in dates:
        lb0 = dt - relativedelta(years=LOOKBACK_YEARS)
        lb1 = dt - relativedelta(months=1)
        for tkr, ser in returns.items():
            df.at[dt, tkr] = compute_rlssa(ser.loc[lb0:lb1].values, L=RLSSA_WINDOW, q=RLSSA_COMPS)
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

def build_metrics(cum_df: pd.DataFrame, rets_dict: dict[int, list[Decimal]]) -> pd.DataFrame:
    """
    Build raw metrics per rank (cum_ret, sharpe, sortino, calmar),
    then min–max each column to [0,1] across ranks and compute GPS harmonic on the
    normalized columns.
    """
    rows = []
    for prev_rank, cum_row in cum_df.iterrows():
        rets = rets_dict.get(prev_rank, [])
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
log_rets     = load_monthly_returns(ROOT_DIR / "All_Monthly_Log_Return_Data")
simple_rets  = load_monthly_returns(ROOT_DIR / "All_Monthly_Return_Data")
rlssa_score  = build_rlssa_history(log_rets)
tickers      = list(log_rets)
NUM_T        = len(tickers)

# -----------------------------------------------------------------------------
# 5) RANK LONG & SHORT EACH MONTH (RLSSA raw or EWMA-adjusted)
# -----------------------------------------------------------------------------
long_rank, short_rank = {}, {}
cur = TEST_SIM_START
while cur <= RLSSA_END:
    lb0 = cur - relativedelta(years=LOOKBACK_YEARS)
    lb1 = cur - relativedelta(months=1)

    # Build (ticker, rlssa_raw, score_for_ranking)
    raw_list = []
    for t in tickers:
        sc = rlssa_score.at[cur, t] if (cur in rlssa_score.index) else np.nan
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

    # Sort for long/short
    # Longs: highest score first; Shorts: lowest score first
    ordL = [t for (t, sc, s) in sorted(raw_list, key=lambda x: x[2], reverse=True)]
    ordS = [t for (t, sc, s) in sorted(raw_list, key=lambda x: x[2])]

    # pad with any missing tickers (keeps full ordering)
    missing = [t for t in tickers if t not in [x[0] for x in raw_list]]
    ordL += missing
    ordS += missing

    if DEBUG_MONTH and cur == DEBUG_MONTH:
        print(f"--- DEBUG {cur.date()} ---")
        dfdbg = pd.DataFrame(raw_list, columns=['tkr','rlssa','score']).set_index('tkr')
        print(dfdbg.sort_values('score', ascending=False))
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

# Final simulation using original ordering
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

print("\nLong RLSSA comparison:")
print(output_comparison(metrics_long_test, final_long_cum).to_string())

print("\nShort RLSSA comparison:")
print(output_comparison(metrics_short_test, final_short_cum).to_string())

