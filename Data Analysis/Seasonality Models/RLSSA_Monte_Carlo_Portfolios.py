import io, sys, time
import numpy as np
import pandas as pd
import statsmodels.api as sm  # kept for parity/extensibility
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from decimal import Decimal
from collections import defaultdict
from dataclasses import dataclass
from scipy import stats as st
from contextlib import redirect_stdout

# =============================== #
# 0) GLOBAL SWITCHES / SIGNIFICANCE
# =============================== #
# Two-sided significance control implemented as BOTH one-sided tests at α/2
SIGNIF_TWO_SIDED_ALPHA = 0.05
PER_SIDE_ALPHA         = SIGNIF_TWO_SIDED_ALPHA / 2.0  # 0.025

# =============================== #
# 1) PARAMETERS & WINDOWS
# =============================== #
START_DATE              = datetime(2001, 1, 1)
FINAL_END               = datetime(2024, 12, 31)

LOOKBACK_YEARS          = 10                     # RLSSA lookback window
SIM_YEARS               = 5                      # test window (2011-2015 given START_DATE)
# Apply window is 2016-01 -> 2024-12 given the above

START_VALUE             = 1000.0
ENTRY_COST              = 0.0025                 # apply ONCE per month

# RLSSA params
SSA_WINDOW              = 12                     # embedding dimension L
SSA_COMPS               = 2                      # robust rank q (components)
RLSSA_USE_EWMA_SCALE    = True                   # if True, scale score by EWMA sigma of simple rets
RLSSA_EWMA_LAMBDA       = 0.94                   # alpha = 1 - lambda
MIN_OBS_FOR_VOL         = 12

# GPS switches
GPS_ROLLING_ENABLED     = True                   # True=rolling 5y monthly re-calibration; False=fixed pre-apply
GPS_CALIB_YEARS         = SIM_YEARS

# === GPS score metric selection (case-insensitive; aliases allowed) =========
# Choose any subset of:
#   "Seasonality", "Sharpe"/"Sharp", "Treynor", "Calmar",
#   "Information", "Mer_ann", "One_minus_mdd", "Cum_ret", "Sortino"
# Default replicates your current behavior (Seasonality, Sharpe, Treynor, MDD inverted).
GPS_SCORE_COMPONENTS = ["Seasonality", "Sharpe", "Cum_ret", "Information"]

# Contract/IO (only monthly files used here)
ROOT_DIR                = Path().resolve().parent.parent / "Complete Data"

# Monte Carlo params
MC_RUNS                 = 10
LAMBDA_EWMA             = 0.94                   # for EWMA-projected sigma used in noise injection
BACKCAST_N              = 12
RNG_SEED                = 42
SAVE_SERIES             = False                   # save Top-1 monthly series per run
OUT_DIR_MC              = Path().resolve() / "Outputs_MC" / f"RLSSA_MC_{'EWMA' if RLSSA_USE_EWMA_SCALE else 'Raw'}"

# NEW switches
SAVE_TICKER_CHOICES_CSV = True                   # write Top-5 selection-share tables per direction
ZERO_NOISE              = False                  # if True, set z_t = 0 (no randomness)

# =============================== #
# 2) DATE RANGES
# =============================== #
LOOKBACK_END    = (START_DATE + relativedelta(years=LOOKBACK_YEARS) - pd.offsets.MonthEnd(1))
TEST_SIM_START  = START_DATE + relativedelta(years=LOOKBACK_YEARS)                             # 2011-01
TEST_SIM_END    = TEST_SIM_START + relativedelta(years=SIM_YEARS) - pd.offsets.MonthEnd(1)     # 2015-12
FINAL_SIM_START = START_DATE + relativedelta(years=LOOKBACK_YEARS + SIM_YEARS)                 # 2016-01
FINAL_SIM_END   = FINAL_END                                                                     # 2024-12

print(f"Lookback: {START_DATE.date()} -> {LOOKBACK_END.date()}")
print(f"Testing : {TEST_SIM_START.date()} -> {TEST_SIM_END.date()}")
print(f"Apply   : {FINAL_SIM_START.date()} -> {FINAL_SIM_END.date()}")

# =============================== #
# 3) HELPERS
# =============================== #
def rule(ch="─", n=96) -> str:
    return ch * n

def section(title: str, ch="═", n=96):
    print("\n" + rule(ch, n))
    print(title)
    print(rule(ch, n) + "\n")

def load_returns(root_dir: Path) -> dict[str, pd.Series]:
    """
    Load *_Monthly_Revenues.csv as Series(date->return). Assumes columns year, month, return.
    """
    out: dict[str, pd.Series] = {}
    for f in sorted(root_dir.glob("*_Monthly_Revenues.csv")):
        ticker = f.stem.replace("_Monthly_Revenues", "")
        df = (
            pd.read_csv(f)
              .assign(date=lambda d: pd.to_datetime(d[['year','month']].assign(day=1)),
                      rtn=lambda d: pd.to_numeric(d['return'], errors='coerce'))
              .set_index('date')['rtn']
              .sort_index()
        )
        out[ticker] = df
    return out

def nav_from_returns_on_grid(returns: pd.Series,
                             start_value: float,
                             full_index: pd.DatetimeIndex) -> pd.Series:
    """
    Accumulate NAV across the given full monthly grid.
    If a month has NaN return, treat it as 0% (carry forward).
    NAV reported at each month-end after applying that month's return.
    """
    cur = Decimal(str(start_value))
    out_vals, out_idx = [], []
    for dt in full_index:
        r = returns.get(dt, np.nan)
        step = Decimal(0) if pd.isna(r) else Decimal(str(float(r)))
        cur *= (Decimal(1) + step)
        out_vals.append(float(cur))
        out_idx.append(dt)
    return pd.Series(out_vals, index=out_idx, name="nav")

# ---- Benchmark & relative-performance helpers (ported & aligned with DVR)
def build_equal_weight_benchmark(simple_rets_dict: dict[str, pd.Series],
                                 start_dt: pd.Timestamp,
                                 end_dt: pd.Timestamp) -> pd.Series:
    idx = pd.date_range(start_dt, end_dt, freq='MS')
    df = pd.DataFrame({t: s.reindex(idx) for t, s in simple_rets_dict.items()})
    return df.mean(axis=1, skipna=True).rename("benchmark")

def direction_consistent_benchmark(bench: pd.Series, direction: str, entry_cost: float) -> pd.Series:
    """
    Transform a gross, long benchmark to be direction-consistent and net-of-cost:
      - if direction == 'short': r -> 1/(1+r) - 1
      - then apply netting:      r -> (1-c)*(1+r) - 1
    """
    b = bench.astype(float)
    if direction == 'short':
        b = (1.0 / (1.0 + b)) - 1.0
    b = (1.0 - entry_cost) * (1.0 + b) - 1.0
    return b.rename(f"{bench.name or 'benchmark'}_{direction}_net")

def compute_beta(port: pd.Series, bench: pd.Series) -> float:
    x = bench.dropna()
    y = port.dropna()
    z = pd.concat([x, y], axis=1, join="inner")
    if z.shape[0] < 3:
        return np.nan
    b = z.iloc[:, 1]
    m = z.iloc[:, 0]
    var_m = np.var(m.values, ddof=1)
    if var_m == 0 or not np.isfinite(var_m):
        return np.nan
    cov = np.cov(b.values, m.values, ddof=1)[0, 1]
    return cov / var_m

def treynor_ratio_series(port: pd.Series, bench: pd.Series) -> float:
    """
    Full-period Treynor (annualized) from monthly returns:
        Treynor_ann = 12 * mean_m(port) / beta(port, bench)
    """
    beta = compute_beta(port, bench)
    if not np.isfinite(beta) or beta == 0:
        return np.nan
    mu = float(np.nanmean(port.values)) if len(port) else np.nan
    return 12.0 * (mu / beta)

def information_ratio_series(port: pd.Series, bench: pd.Series) -> float:
    """
    Full-period Information Ratio (annualized) from monthly returns:
        IR_ann = (mean_m(active) / std_m(active)) * sqrt(12)
    """
    a = pd.concat([port, bench], axis=1, join="inner")
    if a.shape[0] < 3:
        return np.nan
    active = a.iloc[:, 0] - a.iloc[:, 1]
    std = float(np.nanstd(active.values, ddof=1))
    if std == 0 or not np.isfinite(std):
        return np.nan
    return float(np.nanmean(active.values)) / std * np.sqrt(12)

def mean_excess_return_series(port: pd.Series, bench: pd.Series) -> float:
    """
    Mean monthly excess return vs. benchmark (NET; not annualized).
    """
    a = pd.concat([port, bench], axis=1, join="inner")
    if a.shape[0] == 0:
        return np.nan
    active = a.iloc[:, 0] - a.iloc[:, 1]
    return float(np.nanmean(active.values))

def sharpe_ratio(returns: list[Decimal] | np.ndarray) -> float:
    arr = np.array([float(r) for r in returns], dtype=float)
    if arr.size == 0: return np.nan
    std = arr.std(ddof=1)
    if std == 0 or not np.isfinite(std): return np.nan
    return arr.mean() / std * np.sqrt(12)

def sortino_ratio(returns: list[Decimal] | np.ndarray) -> float:
    arr = np.array([float(r) for r in returns], dtype=float)
    if arr.size == 0: return np.nan
    neg = arr[arr < 0]
    if neg.size == 0: return np.nan
    std_neg = neg.std(ddof=1)
    if std_neg == 0 or not np.isfinite(std_neg): return np.nan
    return arr.mean() / std_neg * np.sqrt(12)

def calmar_ratio(returns: list[Decimal] | np.ndarray) -> float:
    arr = np.array([float(r) for r in returns], dtype=float)
    if arr.size == 0: return np.nan
    cum = np.cumprod(1 + arr)
    if cum.size == 0: return np.nan
    years = len(arr) / 12
    if years <= 0: return np.nan
    cagr = cum[-1] ** (1 / years) - 1
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1
    mdd = abs(dd.min()) if len(dd) else np.nan
    if not np.isfinite(mdd) or mdd == 0: return np.nan
    return cagr / mdd

def minmax_01(arr_like) -> np.ndarray:
    x = np.asarray(arr_like, dtype=float)
    mask = np.isfinite(x)
    if not mask.any():
        return np.full_like(x, np.nan, dtype=float)
    xmin, xmax = np.nanmin(x[mask]), np.nanmax(x[mask])
    if xmax == xmin:
        out = np.full_like(x, np.nan, dtype=float)
        out[mask] = 1.0
        return out
    out = (x - xmin) / (xmax - xmin)
    out[~mask] = np.nan
    return np.clip(out, 0.0, 1.0)

def gps_harmonic_01(vals: list[float] | np.ndarray) -> float:
    v = np.asarray(vals, dtype=float)
    if np.isnan(v).any(): return np.nan
    if np.any(v < 0):     return np.nan
    if np.any(v == 0.0):  return 0.0
    return len(v) / np.sum(1.0 / v)

# =============================== #
# 4) LOAD DATA
# =============================== #
base = ROOT_DIR
# RLSSA is fit on LOG monthly returns; GPS metrics/compounding use SIMPLE monthly returns
log_rets    = load_returns(base / "All_Monthly_Log_Return_Data")
simple_rets = load_returns(base / "All_Monthly_Return_Data")
tickers     = list(log_rets.keys())

# =============================== #
# 5) EWMA NOISE & SIM HELPERS
# =============================== #
def backcast_sigma0(first_vals: np.ndarray, lam=LAMBDA_EWMA) -> float:
    w = lam ** np.arange(len(first_vals)-1, -1, -1)
    var0 = float(np.sum(w * (first_vals**2)) / np.sum(w))
    return np.sqrt(var0)

def ewma_projected_sigma_series(log_series: pd.Series,
                                lam=LAMBDA_EWMA,
                                n_init=BACKCAST_N) -> pd.Series:
    """
    Deterministic σ_proj(t) from base log returns (no look-ahead).
    σ used at month t is σ_{t-1}; seeded by 12m backcast.
    """
    s = log_series.sort_index().astype(float)
    arr = s.values
    sig_proj = np.empty_like(arr)
    if len(arr) < n_init:
        sigma0 = np.std(arr, ddof=1) if len(arr) > 1 else float(np.mean(np.abs(arr)))
    else:
        sigma0 = backcast_sigma0(arr[:n_init], lam)
    prev_sigma = sigma0
    for k in range(len(arr)):
        sig_proj[k] = prev_sigma
        prev_sigma = np.sqrt(lam * (prev_sigma**2) + (1 - lam) * (arr[k]**2))
    return pd.Series(sig_proj, index=s.index)

def precompute_sigma_proj_for_all(log_rets_dict: dict[str, pd.Series], lam=LAMBDA_EWMA) -> dict[str, np.ndarray]:
    """Precompute σ_proj arrays once per ticker."""
    return {tkr: ewma_projected_sigma_series(s, lam=lam).values for tkr, s in log_rets_dict.items()}

def simulate_log_returns_with_sigma(log_rets_dict: dict[str, pd.Series],
                                    sigma_proj: dict[str, np.ndarray] | None,
                                    lam=LAMBDA_EWMA,
                                    rng: np.random.Generator | None = None,
                                    no_noise: bool = False) -> dict[str, pd.Series]:
    """
    r_sim(t) = r_base(t) + σ_proj(t) * z_t,  z_t ~ N(0,1) iid per ticker.
    If no_noise=True, returns base log series (no perturbation).
    If sigma_proj is None, compute on the fly (fallback).
    """
    if no_noise:
        return {tkr: s.dropna().sort_index().astype(float).copy() for tkr, s in log_rets_dict.items()}
    if rng is None:
        rng = np.random.default_rng()
    sim = {}
    for tkr, s in log_rets_dict.items():
        s = s.dropna().sort_index().astype(float)
        if s.empty:
            continue
        sig = sigma_proj[tkr] if sigma_proj and tkr in sigma_proj else ewma_projected_sigma_series(s, lam=lam).values
        z = rng.standard_normal(len(s))
        r_sim = s.values + sig * z
        sim[tkr] = pd.Series(r_sim, index=s.index)
    return sim

def log_to_simple_dict(log_rets_dict: dict[str, pd.Series]) -> dict[str, pd.Series]:
    return {tkr: (np.exp(s.astype(float)) - 1.0).rename(tkr) for tkr, s in log_rets_dict.items()}

# =============================== #
# 6) RLSSA CORE
# =============================== #
def robust_low_rank(X: np.ndarray, q: int, max_iter: int = 25, eps: float = 1e-7):
    """
    Iteratively reweighted low-rank factorization of X (L×K).
    Returns U (L×q_eff) and V (K×q_eff) such that X ≈ U V^T, robust to outliers.
    """
    U0, s0, V0t = np.linalg.svd(X, full_matrices=False)
    r0 = min(q, s0.size)
    U = U0[:, :r0] * np.sqrt(s0[:r0])
    V = (V0t[:r0, :].T) * np.sqrt(s0[:r0])

    for _ in range(max_iter):
        R = X - U @ V.T
        W = 1.0 / (np.abs(R) + eps)                 # elementwise weights
        Xw = np.sqrt(W) * X                         # elementwise weighting
        Uw, sw, Vwt = np.linalg.svd(Xw, full_matrices=False)
        r0 = min(q, sw.size)
        U = Uw[:, :r0] * np.sqrt(sw[:r0])
        V = (Vwt[:r0, :].T) * np.sqrt(sw[:r0])
    return U, V

def compute_rlssa_forecast(series: np.ndarray, L: int, q: int) -> float:
    """
    RLSSA one-step forecast using robust low-rank trajectory approximation.
      1) Build trajectory matrix X (L×K)
      2) Robust low-rank factorization X ≈ U V^T (rank q)
      3) Hankelize S = U V^T to reconstructed series rec (length N)
      4) Linear recurrence from left singular vectors of S (like SSA)
      5) forecast = sum a_j * rec[N-j], j = 1..L-1
    """
    x = np.asarray(series, dtype=float).ravel()
    if np.any(~np.isfinite(x)): return np.nan
    N = x.size
    if not (1 < L < N) or q < 1: return np.nan
    K = N - L + 1

    # 1) trajectory
    X = np.column_stack([x[i:i+L] for i in range(K)])  # L×K

    # 2) robust low-rank
    U_r, V_r = robust_low_rank(X, q=q)
    S = U_r @ V_r.T                                   # L×K

    # 3) hankelize to reconstructed series
    rec = np.zeros(N, dtype=float)
    cnt = np.zeros(N, dtype=float)
    for i in range(L):
        for j in range(K):
            rec[i + j] += S[i, j]
            cnt[i + j] += 1.0
    if not np.all(cnt > 0): return np.nan
    rec /= cnt

    # 4) recurrence from S's left singular vectors
    Uc, sc, Vct = np.linalg.svd(S, full_matrices=False)
    r_eff = int(min(q, sc.size))
    if r_eff < 1: return np.nan
    Uc = Uc[:, :r_eff]
    P_head = Uc[:-1, :]
    phi    = Uc[-1, :]
    nu2    = float(np.dot(phi, phi))
    if 1.0 - nu2 <= 1e-10: return np.nan
    R = (P_head @ phi) / (1.0 - nu2)      # (a_{L-1},...,a_1)
    a = R[::-1]                            # (a_1,...,a_{L-1})

    # 5) one-step forecast
    lags = rec[-1: -L: -1]
    if lags.size != a.size: return np.nan
    return float(np.dot(a, lags))

# =============================== #
# 7) RLSSA RANKINGS ON SIMULATED SERIES
# =============================== #
def build_rlssa_rankings_from_log_rets(log_rets_sim: dict[str, pd.Series],
                                       simple_rets_sim: dict[str, pd.Series],
                                       lookback_years: int) -> tuple[dict, dict, dict]:
    """
    Returns (long_rankings, short_rankings, season_score_by_month) for all months TEST_SIM_START..FINAL_END.
    Only Top-1 will be used downstream.
    """
    tickers_sim = list(log_rets_sim)
    long_rankings, short_rankings = {}, {}
    season_score_by_month_sim = {}

    cur = TEST_SIM_START
    while cur <= FINAL_END:
        stats_rows = []
        lb0 = cur - relativedelta(years=lookback_years)
        lb1 = cur - relativedelta(months=1)

        for t in tickers_sim:
            s_log = log_rets_sim[t].loc[lb0:lb1].dropna()
            if len(s_log) < SSA_WINDOW:
                continue

            rlssa_val = compute_rlssa_forecast(s_log.values, L=SSA_WINDOW, q=SSA_COMPS)

            score = rlssa_val
            if RLSSA_USE_EWMA_SCALE and np.isfinite(rlssa_val):
                s_simple = simple_rets_sim.get(t, pd.Series()).loc[lb0:lb1].dropna()
                if len(s_simple) >= MIN_OBS_FOR_VOL:
                    alpha = 1.0 - RLSSA_EWMA_LAMBDA
                    ewma_var = (s_simple.astype(float)**2).ewm(alpha=alpha, adjust=False).mean().iloc[-1]
                    sigma = float(np.sqrt(ewma_var)) if np.isfinite(ewma_var) else np.nan
                    score = rlssa_val / sigma if (sigma and np.isfinite(sigma) and sigma > 0) else np.nan
                else:
                    score = np.nan

            if np.isfinite(score):
                stats_rows.append({'ticker': t, 'score': float(score)})

        if not stats_rows:
            long_rankings[cur]  = tickers_sim.copy()
            short_rankings[cur] = tickers_sim.copy()
        else:
            dfm = pd.DataFrame(stats_rows).set_index('ticker')

            # keep per-month RLSSA scores for GPS calibration
            season_score_by_month_sim[cur] = dfm['score'].to_dict()

            # LONG: highest score first; SHORT: lowest score first
            orderL = dfm.sort_values('score', ascending=False).index.tolist()
            orderS = dfm.sort_values('score', ascending=True ).index.tolist()

            # include missing names at the end (preserve universe order)
            orderL += [t for t in tickers_sim if t not in orderL]
            orderS += [t for t in tickers_sim if t not in orderS]
            long_rankings[cur]  = orderL[:len(tickers_sim)]
            short_rankings[cur] = orderS[:len(tickers_sim)]

        cur += relativedelta(months=1)

    return long_rankings, short_rankings, season_score_by_month_sim

# =============================== #
# 8) TOP-1 & WEIGHTED MONTHLY SERIES
# =============================== #
def monthly_top1_returns(rankings: dict[pd.Timestamp, list[str]],
                         simple_rets_dict: dict[str, pd.Series],
                         *,
                         direction: str,
                         start_dt: pd.Timestamp,
                         end_dt: pd.Timestamp,
                         entry_cost: float,
                         return_tickers: bool = False) -> pd.Series | tuple[pd.Series, pd.Series]:
    """
    TOP-1 ONLY: uses order[0] for each month.
    If return_tickers=True, also returns a Series of the chosen ticker strings (always recorded),
    while returns are NaN when the chosen ticker has no return for that month.
    """
    out_ret, out_tkr, idx = [], [], []
    for dt in pd.date_range(start_dt, end_dt, freq='MS'):
        order = rankings.get(dt)
        if not order:
            continue
        top = order[0]
        s = simple_rets_dict.get(top)
        out_tkr.append(top)
        if s is None or dt not in s.index:
            r_wc = np.nan
        else:
            r = float(s.loc[dt])
            if direction == 'short':
                r = (1.0 / (1.0 + r)) - 1.0
            r_wc = (1.0 - entry_cost) * (1.0 + r) - 1.0
        out_ret.append(r_wc); idx.append(dt)
    ret_ser = pd.Series(out_ret, index=idx, name='top1_return')
    if return_tickers:
        tkr_ser = pd.Series(out_tkr, index=idx, name='top1_ticker')
        return ret_ser, tkr_ser
    return ret_ser

def equal_weight_portfolio_series(simple_rets_dict: dict[str, pd.Series],
                                  *,
                                  direction: str,
                                  start_dt: pd.Timestamp,
                                  end_dt: pd.Timestamp,
                                  entry_cost: float) -> pd.Series:
    """
    Equal-weight across all tickers each month (skip-NA mean). Apply short transform if direction=='short'.
    Apply entry cost monthly.
    """
    idx = pd.date_range(start_dt, end_dt, freq='MS')
    df = pd.DataFrame({t: s.reindex(idx) for t, s in simple_rets_dict.items()})
    if direction == 'short':
        df = (1.0 / (1.0 + df)) - 1.0
    ew = df.mean(axis=1, skipna=True)
    ew_wc = (1.0 - entry_cost) * (1.0 + ew) - 1.0
    return ew_wc.rename("equal_weight")

# =============================== #
# 9) GPS MAPPING (RLSSA) — includes Treynor & MaxDD like DVR
# =============================== #
def max_drawdown_from_monthly(returns: pd.Series) -> float:
    r = pd.Series(returns, dtype=float).dropna()
    if r.empty:
        return np.nan
    cum = (1.0 + r.values).cumprod()
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    return float(dd.min())  # signed (≤0)

def compute_gps_mapping_for_month_rlssa(dt: pd.Timestamp,
                                        rankings: dict[pd.Timestamp, list[str]],
                                        simple_rets_dict: dict[str, pd.Series],
                                        season_score_by_month: dict[pd.Timestamp, dict[str, float]],
                                        bench_full: pd.Series,
                                        *,
                                        direction: str,
                                        calib_years: int,
                                        rolling: bool,
                                        entry_cost: float) -> dict[int, int]:
    """
    GPS new->prev rank mapping for month dt using:
      • Seasonality (RLSSA) = the single next-month forecast score at dt (built from prior 10y), direction-aware
      • Sharpe, Treynor, MaxDD computed on NET returns over calibration window (no look-ahead):
          rolling=True  -> [dt - calib_years .. dt - 1m]
          rolling=False -> [FINAL_SIM_START - calib_years .. FINAL_SIM_START - 1m]
    """
    # --- calibration window (no future info)
    if rolling:
        dt_minus  = dt - relativedelta(years=calib_years)
        win_start = pd.Timestamp(datetime(dt_minus.year, dt_minus.month, 1))
        win_end   = dt - pd.offsets.MonthEnd(1)
    else:
        fixed_calib_start = pd.Timestamp(datetime(FINAL_SIM_START.year - calib_years, FINAL_SIM_START.month, 1))
        fixed_calib_end   = FINAL_SIM_START - pd.offsets.MonthEnd(1)
        win_start, win_end = fixed_calib_start, fixed_calib_end

    if win_end < win_start:
        return {1: 1}

    # Benchmark over the window (gross long → direction-consistent NET)
    bench_win_raw = bench_full.loc[(bench_full.index >= win_start) & (bench_full.index <= win_end)]
    if bench_win_raw.empty:
        return {1: 1}
    bench_win = direction_consistent_benchmark(bench_win_raw, direction, entry_cost)

    # --- gather returns per previous rank over the window
    order_today = rankings.get(dt, [])
    num_t = len(order_today)
    if num_t == 0:
        return {1: 1}

    rets_by_pr: dict[int, list[float]] = defaultdict(list)
    dates_by_pr: dict[int, list[pd.Timestamp]] = defaultdict(list)

    for d in pd.date_range(win_start, win_end, freq='MS'):
        order_d = rankings.get(d)
        if not order_d:
            continue
        for pr, tkr in enumerate(order_d, start=1):
            s = simple_rets_dict.get(tkr)
            if s is None or d not in s.index:
                continue
            r = float(s.loc[d])
            if direction == 'short':
                r = (1.0 / (1.0 + r)) - 1.0
            r_net = (1.0 - entry_cost) * (1.0 + r) - 1.0
            rets_by_pr[pr].append(r_net)
            dates_by_pr[pr].append(d)

    # --- compute metrics per prev-rank
    # RLSSA seasonality score AT dt (point-in-time; no averaging)
    s_map_dt = season_score_by_month.get(dt, {})

    rows = []
    for pr in range(1, num_t + 1):
        # ticker at prev-rank pr TODAY (dt)
        tkr_at_dt = order_today[pr - 1] if pr - 1 < len(order_today) else None

        # RLSSA seasonality score at dt (direction-aware)
        sc = np.nan
        if tkr_at_dt is not None:
            sc = s_map_dt.get(tkr_at_dt, np.nan)
            if np.isfinite(sc) and direction == 'short':
                sc = -float(sc)

        # ratios on window
        if pr in rets_by_pr:
            port = pd.Series(rets_by_pr[pr], index=pd.DatetimeIndex(dates_by_pr[pr]))
            port = port.loc[(port.index >= win_start) & (port.index <= win_end)].sort_index()
            sr     = sharpe_ratio(list(port.values))
            trey   = treynor_ratio_series(port, bench_win)
            mddmag = abs(max_drawdown_from_monthly(port))
        else:
            sr, trey, mddmag = (np.nan, np.nan, np.nan)

        rows.append({
            'prev_rank': pr,
            'seasonality_score': sc,
            'sharpe': sr,
            'treynor': trey,
            'mdd_mag': mddmag
        })

    mdf = pd.DataFrame(rows).set_index('prev_rank').sort_index()
    if mdf.empty:
        return {1: 1}

    # --- normalize to [0,1] and combine (higher is better); invert MDD for scoring
    mdf['seasonality_score_01'] = minmax_01(mdf['seasonality_score'].values)
    mdf['sharpe_01']            = minmax_01(mdf['sharpe'].values)
    mdf['treynor_01']           = minmax_01(mdf['treynor'].values)
    mdf['mdd_inv_01']           = 1.0 - minmax_01(mdf['mdd_mag'].values)

    norm_cols = ['seasonality_score_01', 'sharpe_01', 'treynor_01', 'mdd_inv_01']
    mdf['score'] = [gps_harmonic_01(mdf.loc[i, norm_cols].values) for i in mdf.index]

    # new_rank by descending score → invert to mapping new->prev
    mdf['new_rank'] = mdf['score'].rank(ascending=False, method='first')
    inv = {}
    for prev_rank, row in mdf.iterrows():
        nr = int(row['new_rank'])
        if nr not in inv:
            inv[nr] = int(prev_rank)
    return inv

def monthly_top1_returns_gps_rlssa(rankings: dict[pd.Timestamp, list[str]],
                                   simple_rets_dict: dict[str, pd.Series],
                                   season_score_by_month: dict[pd.Timestamp, dict[str, float]],
                                   bench_full: pd.Series,
                                   *,
                                   direction: str,
                                   start_dt: pd.Timestamp,
                                   end_dt: pd.Timestamp,
                                   calib_years: int,
                                   rolling: bool,
                                   entry_cost: float,
                                   return_tickers: bool = False) -> pd.Series | tuple[pd.Series, pd.Series]:
    """
    TOP-1 ONLY: GPS maps new_rank=1 to a prev_rank, takes that ticker's NET simple return.
    If return_tickers=True, also returns a Series of chosen tickers (always recorded),
    while returns are NaN when the chosen ticker has no return for that month.
    """
    out_ret, out_tkr, idx = [], [], []
    for dt in pd.date_range(start_dt, end_dt, freq='MS'):
        order_today = rankings.get(dt)
        if not order_today:
            continue
        mapping = compute_gps_mapping_for_month_rlssa(
            dt, rankings, simple_rets_dict, season_score_by_month, bench_full,
            direction=direction, calib_years=GPS_CALIB_YEARS, rolling=GPS_ROLLING_ENABLED,
            entry_cost=ENTRY_COST
        )
        prev_rank = mapping.get(1, 1)
        if prev_rank < 1 or prev_rank > len(order_today):
            continue
        tkr = order_today[prev_rank - 1]
        out_tkr.append(tkr)
        s = simple_rets_dict.get(tkr)
        if s is None or dt not in s.index:
            r_wc = np.nan
        else:
            r = float(s.loc[dt])
            if direction == 'short':
                r = (1.0 / (1.0 + r)) - 1.0
            r_wc = (1.0 - entry_cost) * (1.0 + r) - 1.0
        out_ret.append(r_wc); idx.append(dt)
    ret_ser = pd.Series(out_ret, index=idx, name='top1_return_gps')
    if return_tickers:
        tkr_ser = pd.Series(out_tkr, index=idx, name='top1_ticker_gps')
        return ret_ser, tkr_ser
    return ret_ser

# ---------- NEW: Top-5 monthly selection tables (per direction) ---------- #
def build_monthly_top5_table(df_choices: pd.DataFrame, direction: str, n_runs: int, top_k: int = 5) -> pd.DataFrame:
    """
    For each monthly date in the apply window and a given direction,
    compute the top-k tickers by selection share for Baseline and GPS.
    Each cell formatted like "[TICKER | 55.7%]" (pct of total simulations).
    Returns a DataFrame with columns:
        Date | B1..B5 | G1..G5
    """
    sub = df_choices[df_choices['direction'] == direction].copy()
    if sub.empty:
        return pd.DataFrame(columns=['Date'] + [f'B{i}' for i in range(1, top_k+1)] + [f'G{i}' for i in range(1, top_k+1)])

    sub['date'] = pd.to_datetime(sub['date'])
    dates = pd.date_range(FINAL_SIM_START, FINAL_SIM_END, freq='MS')

    rows = []
    for dt in dates:
        dsel = sub[sub['date'] == dt]
        # counts
        base_counts = dsel['baseline_ticker'].value_counts()  # NaNs excluded by default
        gps_counts  = dsel['gps_ticker'].value_counts()

        def fmt_top(series):
            series = series.head(top_k)
            labels = []
            for tkr, cnt in series.items():
                pct = 100.0 * float(cnt) / float(n_runs)
                labels.append(f"[{tkr} | {pct:.1f}%]")
            while len(labels) < top_k:
                labels.append("")
            return labels

        B = fmt_top(base_counts)
        G = fmt_top(gps_counts)
        rows.append({
            'Date': dt.date().isoformat(),
            **{f'B{i+1}': B[i] for i in range(top_k)},
            **{f'G{i+1}': G[i] for i in range(top_k)},
        })
    return pd.DataFrame(rows)

# =============================== #
# 10) METRICS & WINDOWS
# =============================== #
@dataclass
class Perf:
    cum_ret: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    mdd: float
    stdev: float   # monthly standard deviation

def cagr_from_monthly(returns: pd.Series) -> float:
    if returns.empty: return np.nan
    factor = float(np.prod(1.0 + returns.values))
    years = len(returns) / 12.0
    return factor ** (1/years) - 1.0 if years > 0 else np.nan

def perf_from_monthly(returns: pd.Series) -> Perf:
    arr = returns.astype(float).values
    if arr.size == 0:
        return Perf(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    sr  = sharpe_ratio(arr)       # annualized, full-period
    sor = sortino_ratio(arr)      # annualized, full-period
    cal = calmar_ratio(arr)       # full-period
    cum = float(np.prod(1.0 + arr) - 1.0)  # full-period cumulative return
    years = len(arr) / 12.0
    cagr = (np.prod(1.0 + arr) ** (1/years) - 1.0) if years > 0 else np.nan
    # MDD negative; store as negative; report abs() where displayed
    cum_curve = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(cum_curve)
    dd = cum_curve / peak - 1.0
    mdd = float(np.min(dd)) if dd.size else np.nan
    sd  = float(np.std(arr, ddof=1)) if arr.size >= 2 else np.nan  # monthly within-period stdev
    return Perf(cum, cagr, sr, sor, cal, mdd, sd)

def three_fixed_windows() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Exactly 3x3y windows starting 2016-01 given globals."""
    wins = []
    cur = pd.Timestamp(FINAL_SIM_START)
    for _ in range(3):
        w_end = cur + relativedelta(months=36) - pd.offsets.MonthEnd(1)
        wins.append((cur, w_end))
        cur = w_end + pd.offsets.MonthBegin(1)
    return wins

# =============================== #
# 11) SIGNIFICANCE HELPERS (paired, one-sided both directions at α/2)
# =============================== #
def paired_one_sided_both(left: np.ndarray, right: np.ndarray):
    """
    Returns (t_right>left, p_right>left, t_right<left, p_right<left, n_valid)
    using paired one-sided t-tests on matched pairs.
    """
    mask = np.isfinite(left) & np.isfinite(right)
    n = int(mask.sum())
    if n < 3:
        return None, None, None, None, n
    t_g, p_g = st.ttest_rel(right[mask], left[mask], alternative='greater')
    t_l, p_l = st.ttest_rel(right[mask], left[mask], alternative='less')
    return float(t_g), float(p_g), float(t_l), float(p_l), n

def print_sig_line_generic(metric_name: str,
                           left_label: str, right_label: str,
                           t_g, p_g, t_l, p_l, n: int,
                           better_note: str):
    """
    Report significance with per-side alpha = PER_SIDE_ALPHA (overall two-sided 5%).
    Tests are RIGHT vs LEFT.
    """
    if n < 3:
        print(f"    {right_label} vs {left_label} — {metric_name:<9}: insufficient valid pairs.")
        return
    better = f"{right_label} better at 5% two-sided" if (p_g is not None and p_g < PER_SIDE_ALPHA) else "no better effect at 5%"
    worse  = f"{right_label} worse at 5% two-sided"  if (p_l is not None and p_l < PER_SIDE_ALPHA) else "no worse effect at 5%"
    print(f"    {right_label} vs {left_label} — {metric_name:<9}: n={n:4d} | "
          f"better ({better_note}; α_one-sided={PER_SIDE_ALPHA:.3f}): t={t_g:.3f}, p={p_g:.4f} -> {better} | "
          f"worse: t={t_l:.3f}, p={p_l:.4f} -> {worse}")

def paired_one_sided_both_from_df(df_dir: pd.DataFrame, strat_left: str, strat_right: str, col: str):
    """
    Utility for Treynor (or any column in df_metrics) comparisons: merges by run for two strategies
    and returns paired one-sided t-test results for right vs left.
    """
    L = df_dir[df_dir['strategy'] == strat_left][['run', col]].rename(columns={col: 'L'})
    R = df_dir[df_dir['strategy'] == strat_right][['run', col]].rename(columns={col: 'R'})
    merged = pd.merge(L, R, on='run', how='inner').dropna()
    if merged.empty:
        return None, None, None, None, 0
    return paired_one_sided_both(merged['L'].values.astype(float), merged['R'].values.astype(float))

# =============================== #
# 12) MONTE CARLO: BOTH DIRECTIONS (efficient)
# =============================== #
def run_monte_carlo_both(n_runs=MC_RUNS, lam=LAMBDA_EWMA, save_series=SAVE_SERIES, zero_noise=ZERO_NOISE):
    OUT_DIR_MC.mkdir(parents=True, exist_ok=True)

    # ---- Tee prints to console AND buffer; write buffer to file at end
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st_ in self.streams: st_.write(s)
            return len(s)
        def flush(self):
            for st_ in self.streams:
                try: st_.flush()
                except Exception: pass

    tee = Tee(sys.__stdout__, buf)

    # Unified metrics table + selection records
    metrics_rows: list[dict] = []
    choice_records = []

    with redirect_stdout(tee):
        rng = np.random.default_rng(RNG_SEED)
        t0 = time.time()

        section(f"Starting Monte Carlo — RLSSA (BOTH directions) | runs={n_runs:,} | λ={lam} | zero_noise={zero_noise}")
        print(f"Ranking: RLSSA(L={SSA_WINDOW}, q={SSA_COMPS}, scale={'EWMA' if RLSSA_USE_EWMA_SCALE else 'Raw'})")
        print(f"Windows: Lookback={LOOKBACK_YEARS}y | Test={SIM_YEARS}y ({TEST_SIM_START.date()}→{TEST_SIM_END.date()}) | "
              f"Apply={FINAL_SIM_START.date()}→{FINAL_SIM_END.date()}")
        print(f"GPS: rolling={GPS_ROLLING_ENABLED}, calib={GPS_CALIB_YEARS}y | Entry cost per month={ENTRY_COST:.4f}\n")

        # Precompute sigma projections once (used only if zero_noise=False)
        sigma_proj = None if zero_noise else precompute_sigma_proj_for_all(log_rets, lam=lam)

        # Static grids
        month_grid = pd.date_range(FINAL_SIM_START, FINAL_SIM_END, freq='MS')
        splits = three_fixed_windows()

        # Aggregators per direction (include 'weighted' = equal-weight across universe)
        dirs = ('long', 'short')
        full_metrics = {d: {'baseline': [], 'gps': [], 'weighted': []} for d in dirs}
        sub_metrics  = {d: {'baseline': defaultdict(list), 'gps': defaultdict(list), 'weighted': defaultdict(list)} for d in dirs}
        sub_treynor  = {d: {'baseline': defaultdict(list), 'gps': defaultdict(list), 'weighted': defaultdict(list)} for d in dirs}
        pos_counts   = {d: {'baseline': 0, 'gps': 0, 'weighted': 0} for d in dirs}
        neg_counts   = {d: {'baseline': 0, 'gps': 0, 'weighted': 0} for d in dirs}

        # MAIN LOOP
        for run in range(1, n_runs + 1):
            # 1) simulate log returns (or reuse base logs) and convert to simple
            log_sim = simulate_log_returns_with_sigma(log_rets, sigma_proj, lam=lam, rng=rng, no_noise=zero_noise)
            simple_sim = log_to_simple_dict(log_sim) if not zero_noise else log_to_simple_dict(log_rets)

            # 2) build RLSSA rankings & per-month seasonality scores
            L_rank, S_rank, season_by_month = build_rlssa_rankings_from_log_rets(log_sim, simple_sim, LOOKBACK_YEARS)

            # 3) equal-weight benchmarks (gross long) for test+apply (full) and apply (window)
            bench_full  = build_equal_weight_benchmark(simple_sim, TEST_SIM_START,  FINAL_END)
            bench_apply = build_equal_weight_benchmark(simple_sim, FINAL_SIM_START, FINAL_SIM_END)

            # 4) iterate BOTH directions
            for direction, rank_use in (('long', L_rank), ('short', S_rank)):
                # Direction-consistent, NET benchmark for the APPLY window
                bench_apply_dc = direction_consistent_benchmark(bench_apply, direction, ENTRY_COST)

                # Baseline + GPS Top-1 (NET) and equal-weight (NET)
                top1_baseline, tkr_baseline = monthly_top1_returns(
                    rank_use, simple_sim, direction=direction,
                    start_dt=FINAL_SIM_START, end_dt=FINAL_SIM_END,
                    entry_cost=ENTRY_COST, return_tickers=True
                )
                top1_gps, tkr_gps = monthly_top1_returns_gps_rlssa(
                    rank_use, simple_sim, season_by_month, bench_full,
                    direction=direction,
                    start_dt=FINAL_SIM_START, end_dt=FINAL_SIM_END,
                    calib_years=GPS_CALIB_YEARS, rolling=GPS_ROLLING_ENABLED,
                    entry_cost=ENTRY_COST, return_tickers=True
                )
                eq_weight_series = equal_weight_portfolio_series(
                    simple_sim, direction=direction,
                    start_dt=FINAL_SIM_START, end_dt=FINAL_SIM_END,
                    entry_cost=ENTRY_COST
                )

                # NAV for context + record Top-1 choices (all months)
                nav_baseline = nav_from_returns_on_grid(top1_baseline, START_VALUE, month_grid)
                nav_gps      = nav_from_returns_on_grid(top1_gps,      START_VALUE, month_grid)
                for dt in month_grid:
                    choice_records.append({
                        'direction': direction,
                        'run': run,
                        'date': dt.date().isoformat(),
                        'baseline_ticker': tkr_baseline.get(dt, np.nan) if not tkr_baseline.empty else np.nan,
                        'gps_ticker': tkr_gps.get(dt, np.nan) if not tkr_gps.empty else np.nan,
                        'baseline_value': float(nav_baseline.get(dt, np.nan)),
                        'gps_value': float(nav_gps.get(dt, np.nan)),
                    })

                # Optional per-run series (direction-tagged folders)
                if save_series:
                    ddir = OUT_DIR_MC / f"{direction}_run_{run:03d}"
                    ddir.mkdir(parents=True, exist_ok=True)
                    top1_baseline.to_csv(ddir / "top1_baseline_monthly.csv")
                    top1_gps.to_csv(ddir / "top1_gps_monthly.csv")
                    eq_weight_series.to_csv(ddir / "weighted_equal_monthly.csv")

                # Full-period metrics (apply window) on NET returns vs direction-consistent NET bench
                pb, pg, pw = perf_from_monthly(top1_baseline), perf_from_monthly(top1_gps), perf_from_monthly(eq_weight_series)
                full_metrics[direction]['baseline'].append(pb)
                full_metrics[direction]['gps'].append(pg)
                full_metrics[direction]['weighted'].append(pw)
                pos_counts[direction]['baseline'] += int(pb.cum_ret > 0)
                neg_counts[direction]['baseline'] += int(pb.cum_ret <= 0)
                pos_counts[direction]['gps']      += int(pg.cum_ret > 0)
                neg_counts[direction]['gps']      += int(pg.cum_ret <= 0)
                pos_counts[direction]['weighted'] += int(pw.cum_ret > 0)
                neg_counts[direction]['weighted'] += int(pw.cum_ret <= 0)

                # Per-run metrics row (use direction-consistent NET bench for Treynor/IR/MER)
                metrics_rows.append({
                    'run': run, 'direction': direction, 'strategy': 'baseline',
                    'sharpe': pb.sharpe, 'sortino': pb.sortino,
                    'treynor': treynor_ratio_series(top1_baseline, bench_apply_dc),
                    'information': information_ratio_series(top1_baseline, bench_apply_dc),
                    'calmar': pb.calmar, 'cum_ret': pb.cum_ret,
                    'mdd': abs(pb.mdd) if np.isfinite(pb.mdd) else np.nan,
                    'mer': mean_excess_return_series(top1_baseline, bench_apply_dc),
                })
                metrics_rows.append({
                    'run': run, 'direction': direction, 'strategy': 'gps',
                    'sharpe': pg.sharpe, 'sortino': pg.sortino,
                    'treynor': treynor_ratio_series(top1_gps, bench_apply_dc),
                    'information': information_ratio_series(top1_gps, bench_apply_dc),
                    'calmar': pg.calmar, 'cum_ret': pg.cum_ret,
                    'mdd': abs(pg.mdd) if np.isfinite(pg.mdd) else np.nan,
                    'mer': mean_excess_return_series(top1_gps, bench_apply_dc),
                })
                metrics_rows.append({
                    'run': run, 'direction': direction, 'strategy': 'weighted',
                    'sharpe': pw.sharpe, 'sortino': pw.sortino,
                    'treynor': treynor_ratio_series(eq_weight_series, bench_apply_dc),
                    'information': information_ratio_series(eq_weight_series, bench_apply_dc),
                    'calmar': pw.calmar, 'cum_ret': pw.cum_ret,
                    'mdd': abs(pw.mdd) if np.isfinite(pw.mdd) else np.nan,
                    'mer': mean_excess_return_series(eq_weight_series, bench_apply_dc),
                })

                # Subperiods — three fixed 3y windows (Treynor vs direction-consistent NET bench)
                for i, (s, e) in enumerate(splits, start=1):
                    sb = top1_baseline[(top1_baseline.index >= s) & (top1_baseline.index <= e)]
                    sg = top1_gps[(top1_gps.index >= s) & (top1_gps.index <= e)]
                    sw = eq_weight_series[(eq_weight_series.index >= s) & (eq_weight_series.index <= e)]

                    sub_metrics[direction]['baseline'][i].append(perf_from_monthly(sb))
                    sub_metrics[direction]['gps'][i].append(perf_from_monthly(sg))
                    sub_metrics[direction]['weighted'][i].append(perf_from_monthly(sw))

                    bench_win_dc = bench_apply_dc[(bench_apply_dc.index >= s) & (bench_apply_dc.index <= e)]
                    sub_treynor[direction]['baseline'][i].append(treynor_ratio_series(sb, bench_win_dc))
                    sub_treynor[direction]['gps'][i].append(treynor_ratio_series(sg, bench_win_dc))
                    sub_treynor[direction]['weighted'][i].append(treynor_ratio_series(sw, bench_win_dc))

            # progress
            if (run % 10) == 0:
                elapsed = time.time() - t0
                print(f"  Progress: completed {run:>4}/{n_runs:<4} runs | elapsed {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

        # === Build & save the Top-5 selection-share tables (per direction) ===
        if SAVE_TICKER_CHOICES_CSV and choice_records:
            df_choices = pd.DataFrame(choice_records)

            section("Top-5 selection share tables (per month)")

            for direction in ('long', 'short'):
                tbl = build_monthly_top5_table(df_choices, direction, n_runs, top_k=5)
                out_path = OUT_DIR_MC / f"top5_selections_table_{direction}.csv"
                tbl.to_csv(out_path, index=False)

                print(f"Direction: {direction.upper()} — Top-5 per method (Baseline first 5 columns, GPS next 5)")
                preview = tbl.head(12).to_string(index=False)  # show first 12 months
                print(preview + "\n")
                print(f"Saved full table to: {out_path}\n")

        # === Safe aggregators to avoid warnings (empty/all-NaN) ===
        def _finite(x) -> np.ndarray:
            arr = np.asarray(x, dtype=float)
            return arr[np.isfinite(arr)]

        def safe_mean(x) -> float:
            v = _finite(x)
            return float(v.mean()) if v.size > 0 else np.nan

        def safe_median(x) -> float:
            v = _finite(x)
            return float(np.median(v)) if v.size > 0 else np.nan

        def safe_std(x, ddof=1) -> float:
            v = _finite(x)
            return float(v.std(ddof=ddof)) if v.size > ddof else np.nan

        def agg(perf_list: list[Perf]) -> dict[str, float]:
            if not perf_list:
                keys = ['cum_ret','cagr','sharpe','sortino','calmar','mdd','stdev']
                return {f"{k}_mean": np.nan for k in keys} | \
                       {f"{k}_median": np.nan for k in keys} | \
                       {f"{k}_std": np.nan for k in keys}
            out = {}
            for k in ['cum_ret','cagr','sharpe','sortino','calmar','mdd','stdev']:
                x = np.array([getattr(p, k) for p in perf_list], dtype=float)
                out[f'{k}_mean']   = safe_mean(x)
                out[f'{k}_median'] = safe_median(x)
                out[f'{k}_std']    = safe_std(x, ddof=1)
            return out

        def treynor_stats_from_df(df: pd.DataFrame, direction: str, strategy: str) -> dict[str, float]:
            sub = df[(df['direction'] == direction) & (df['strategy'] == strategy)]
            vals = sub['treynor'].astype(float).values if not sub.empty else np.array([], dtype=float)
            return {
                'treynor_mean': safe_mean(vals),
                'treynor_median': safe_median(vals),
                'treynor_std': safe_std(vals, ddof=1)
            }

        df_metrics_all = pd.DataFrame(metrics_rows) if metrics_rows else None

        # === Print LONG first, divider, then SHORT ===
        for direction in ('long', 'short'):
            n_runs_local = n_runs
            agg_full_b = agg(full_metrics[direction]['baseline'])
            agg_full_g = agg(full_metrics[direction]['gps'])
            agg_full_w = agg(full_metrics[direction]['weighted'])

            trey_b = treynor_stats_from_df(df_metrics_all, direction, 'baseline') if df_metrics_all is not None else {}
            trey_g = treynor_stats_from_df(df_metrics_all, direction, 'gps')      if df_metrics_all is not None else {}
            trey_w = treynor_stats_from_df(df_metrics_all, direction, 'weighted')  if df_metrics_all is not None else {}

            section(f"MONTE CARLO SUMMARY — Direction: {direction.upper()}")

            print("Config")
            print(f"  Lookback: {LOOKBACK_YEARS}y")
            print(f"  Test    : {TEST_SIM_START.date()} → {TEST_SIM_END.date()}  ({SIM_YEARS}y)")
            print(f"  Apply   : {FINAL_SIM_START.date()} → {FINAL_SIM_END.date()}")
            print(f"  GPS     : rolling={GPS_ROLLING_ENABLED}, calib={GPS_CALIB_YEARS}y   | Entry cost={ENTRY_COST:.4f}\n")

            print("Outcome counts (Top-1 over apply)")
            print(f"  {'Strategy':<10} {'Pos(+)':>7} {'Neg(-)':>7}")
            print(f"  {'-'*10} {'-'*7:>7} {'-'*7:>7}")
            print(f"  {'Baseline':<10} {pos_counts[direction]['baseline']:>7} {neg_counts[direction]['baseline']:>7}")
            print(f"  {'GPS':<5}      {pos_counts[direction]['gps']:>7} {neg_counts[direction]['gps']:>7}")
            print(f"  {'Weighted':<10} {pos_counts[direction]['weighted']:>7} {neg_counts[direction]['weighted']:>7}\n")

            def pretty_full(d, t, name):
                print(f"{name} — Full Apply Period (across {n_runs_local} runs)")
                print(f"  CumRet    mean={d['cum_ret_mean']:.4f}   median={d['cum_ret_median']:.4f}   sd={d['cum_ret_std']:.4f}")
                print(f"  CAGR      mean={d['cagr_mean']:.4f}     median={d['cagr_median']:.4f}     sd={d['cagr_std']:.4f}")
                print(f"  Sharpe    mean={d['sharpe_mean']:.3f}   median={d['sharpe_median']:.3f}   sd={d['sharpe_std']:.3f}")
                print(f"  Sortino   mean={d['sortino_mean']:.3f}  median={d['sortino_median']:.3f}  sd={d['sortino_std']:.3f}")
                print(f"  Calmar    mean={d['calmar_mean']:.3f}   median={d['calmar_median']:.3f}   sd={d['calmar_std']:.3f}")
                print(f"  MaxDD     mean={d['mdd_mean']:.3f}      median={d['mdd_median']:.3f}      sd={d['mdd_std']:.3f}")
                print(f"  Treynor   mean={t.get('treynor_mean', np.nan):.3f}   median={t.get('treynor_median', np.nan):.3f}   sd={t.get('treynor_std', np.nan):.3f}\n")

            pretty_full(agg_full_b, trey_b, "BASELINE")
            pretty_full(agg_full_g, trey_g, "GPS")
            pretty_full(agg_full_w, trey_w, "WEIGHTED")

            print("Subperiods — Three fixed 3-year windows")
            for i, (s, e) in enumerate(splits, start=1):
                ab = agg(sub_metrics[direction]['baseline'][i])
                ag = agg(sub_metrics[direction]['gps'][i])
                aw = agg(sub_metrics[direction]['weighted'][i])

                tb_arr = np.array(sub_treynor[direction]['baseline'][i], dtype=float)
                tg_arr = np.array(sub_treynor[direction]['gps'][i], dtype=float)
                tw_arr = np.array(sub_treynor[direction]['weighted'][i], dtype=float)
                tb_mean, tb_sd = safe_mean(tb_arr), safe_std(tb_arr, ddof=1)
                tg_mean, tg_sd = safe_mean(tg_arr), safe_std(tg_arr, ddof=1)
                tw_mean, tw_sd = safe_mean(tw_arr), safe_std(tw_arr, ddof=1)

                print(f"\n  Window {i}: {s.date()} → {e.date()}")
                print(f"    {'Metric':<14} {'Baseline':>12} {'GPS':>12} {'Weighted':>12}")
                print(f"    {'-'*14:<14} {'-'*12:>12} {'-'*12:>12} {'-'*12:>12}")
                def line3(label, f, g, w, fmt=".4f"):
                    lf = f if np.isfinite(f) else np.nan
                    lg = g if np.isfinite(g) else np.nan
                    lw = w if np.isfinite(w) else np.nan
                    print(f"    {label:<14} {lf:>12{fmt}} {lg:>12{fmt}} {lw:>12{fmt}}")

                line3("CumRet mean",  ab['cum_ret_mean'],  ag['cum_ret_mean'],  aw['cum_ret_mean'])
                line3("CAGR mean",    ab['cagr_mean'],     ag['cagr_mean'],     aw['cagr_mean'])
                line3("Sharpe mean",  ab['sharpe_mean'],   ag['sharpe_mean'],   aw['sharpe_mean'], fmt=".3f")
                line3("Calmar mean",  ab['calmar_mean'],   ag['calmar_mean'],   aw['calmar_mean'], fmt=".3f")
                line3("MaxDD mean",   ab['mdd_mean'],      ag['mdd_mean'],      aw['mdd_mean'],    fmt=".3f")
                line3("Treynor mean", tb_mean,             tg_mean,             tw_mean,           fmt=".3f")

            section("Significance — Paired, one-sided both directions (per-side α = {:.3f}; overall two-sided 5%)".format(PER_SIDE_ALPHA))

            # Build arrays for metrics from Perf lists
            b_list = full_metrics[direction]['baseline']
            g_list = full_metrics[direction]['gps']
            w_list = full_metrics[direction]['weighted']
            arr = lambda attr, L: np.array([getattr(p, attr) for p in L], dtype=float)

            def sig_pair_block(left_label, right_label, L_list, R_list, df_dir_metrics):
                print(f"[ {left_label.upper()}  →  {right_label.upper()} ]")
                for name, attr in [("Return", "cum_ret"),
                                   ("CAGR", "cagr"),
                                   ("Sharpe", "sharpe"),
                                   ("Sortino", "sortino"),
                                   ("Calmar", "calmar")]:
                    t_g, p_g, t_l, p_l, n = paired_one_sided_both(arr(attr, L_list), arr(attr, R_list))
                    print_sig_line_generic(name, left_label, right_label, t_g, p_g, t_l, p_l, n, better_note="higher is better")
                # Treynor from df_metrics_all by pair
                df_dir = df_metrics_all[df_metrics_all['direction'] == direction] if df_metrics_all is not None else pd.DataFrame()
                t_g, p_g, t_l, p_l, n = paired_one_sided_both_from_df(df_dir, left_label.lower(), right_label.lower(), 'treynor')
                print_sig_line_generic("Treynor", left_label, right_label, t_g, p_g, t_l, p_l, n, better_note="higher is better")
                # MaxDD: Perf.mdd is negative, less severe (closer to 0, i.e., higher) is better
                t_g, p_g, t_l, p_l, n = paired_one_sided_both(arr("mdd", L_list), arr("mdd", R_list))
                print_sig_line_generic("MaxDD", left_label, right_label, t_g, p_g, t_l, p_l, n, better_note="less severe (higher)")
                print("")  # spacer

            sig_pair_block("baseline", "gps", b_list, g_list, df_metrics_all if df_metrics_all is not None else pd.DataFrame())
            sig_pair_block("baseline", "weighted", b_list, w_list, df_metrics_all if df_metrics_all is not None else pd.DataFrame())
            sig_pair_block("gps", "weighted", g_list, w_list, df_metrics_all if df_metrics_all is not None else pd.DataFrame())

            if direction == 'long':
                print(rule("─"))
                print("")

        total_elapsed = time.time() - t0
        section("Completed")
        print(f"Total runtime: {time.strftime('%H:%M:%S', time.gmtime(total_elapsed))}\n")

        # === Save metrics table (NO correlation/heatmap here) ===
        if metrics_rows:
            df_metrics = pd.DataFrame(metrics_rows)
            df_metrics.to_csv(OUT_DIR_MC / "performance_metrics_by_portfolio.csv", index=False)

    # === Write terminal printout to file ===
    with open(OUT_DIR_MC / "console_report.txt", "w", encoding="utf-8") as f:
        f.write(buf.getvalue())


# =============================== #
# 13) RUN — single efficient call
# =============================== #
if __name__ == "__main__":
    run_monte_carlo_both(n_runs=MC_RUNS, lam=LAMBDA_EWMA, save_series=SAVE_SERIES, zero_noise=ZERO_NOISE)
