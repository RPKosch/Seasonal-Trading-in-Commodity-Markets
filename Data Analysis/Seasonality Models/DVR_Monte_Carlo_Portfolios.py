import re
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from decimal import Decimal
from collections import defaultdict
from dataclasses import dataclass
from scipy import stats as st
import sys, io
from contextlib import redirect_stdout

# =============================== #
# 0) GLOBAL SWITCHES
# =============================== #
# Keep correlation-matrix helpers, but do not run them unless explicitly enabled.
GENERATE_METRICS_CORR_CSV     = False
GENERATE_METRICS_CORR_HEATMAP = False  # requires matplotlib

# Two-sided significance control: we test BOTH one-sided directions at α/2
SIGNIF_TWO_SIDED_ALPHA = 0.05
PER_SIDE_ALPHA         = SIGNIF_TWO_SIDED_ALPHA / 2.0  # 0.025

# =============================== #
# 1) PARAMETERS & WINDOWS
# =============================== #
START_DATE              = datetime(2001, 1, 1)
FINAL_END               = datetime(2024, 12, 31)

LOOKBACK_YEARS          = 10                     # DVR lookback window
SIM_YEARS               = 5                      # test window (2011-2015 given START_DATE)
# Apply window is 2016-01 -> 2024-12 given the above

START_VALUE             = 1000.0
ENTRY_COST              = 0.0025                 # apply ONCE per month

# DVR params
SIG_LEVEL               = 0.05                   # 1.0 ≡ no p-filter (GREEDY by Z); else significance-first

# GPS switches
GPS_ROLLING_ENABLED     = True                   # True=rolling 5y monthly re-calibration; False=fixed pre-apply
GPS_CALIB_YEARS         = SIM_YEARS

# Contract/IO (only monthly files used here)
ROOT_DIR                = Path().resolve().parent.parent / "Complete Data"

# Monte Carlo params
MC_RUNS                 = 10
LAMBDA_EWMA             = 0.94
BACKCAST_N              = 12
RNG_SEED                = 42
SAVE_SERIES             = False                  # save Top-1 monthly series per run
OUT_DIR_MC              = Path().resolve() / "Outputs_MC" / f"DVR_MC_p≤{SIG_LEVEL}"

# NEW switches
SAVE_TICKER_CHOICES_CSV = True                   # write a single CSV with chosen tickers per run & month
ZERO_NOISE              = False                  # if True, set z_t = 0 (no randomness)

# Limit how many Top-1 ticker choices per run & direction are saved.
# Set to an integer (e.g., 10) to save only the first N months per run; or None to save all months.
TOP1_CHOICES_MAX_PER_RUN = None

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
def build_equal_weight_benchmark(simple_rets_dict: dict[str, pd.Series],
                                 start_dt: pd.Timestamp,
                                 end_dt: pd.Timestamp) -> pd.Series:
    """
    Passive monthly benchmark: equally-weighted average across all tickers' simple returns
    available in a given month. No look-ahead. Index is the month grid [start_dt..end_dt].
    """
    idx = pd.date_range(start_dt, end_dt, freq='MS')
    df = pd.DataFrame({t: s.reindex(idx) for t, s in simple_rets_dict.items()})
    return df.mean(axis=1, skipna=True).rename("benchmark")

def compute_beta(port: pd.Series, bench: pd.Series) -> float:
    """
    Beta of portfolio vs. benchmark on overlapping months (monthly simple returns).
    """
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
    Treynor ratio (monthly): mean(port) / beta(port, bench).
    Risk-free assumed 0 for consistency with your Sharpe.
    """
    beta = compute_beta(port, bench)
    if not np.isfinite(beta) or beta == 0:
        return np.nan
    mu = float(np.nanmean(port.values)) if len(port) else np.nan
    return mu / beta

def information_ratio_series(port: pd.Series, bench: pd.Series) -> float:
    """
    Information ratio (annualized): mean(active)/std(active)*sqrt(12).
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
    Mean monthly excess return vs. benchmark (not annualized).
    """
    a = pd.concat([port, bench], axis=1, join="inner")
    if a.shape[0] == 0:
        return np.nan
    active = a.iloc[:, 0] - a.iloc[:, 1]
    return float(np.nanmean(active.values))

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
    cur = float(start_value)
    out_vals, out_idx = [], []
    for dt in full_index:
        r = returns.get(dt, np.nan)
        step = 0.0 if pd.isna(r) else float(r)
        cur *= (1.0 + step)
        out_vals.append(cur)
        out_idx.append(dt)
    return pd.Series(out_vals, index=out_idx, name="nav")

def sharpe_ratio(returns: list[Decimal] | np.ndarray) -> float:
    arr = np.array([float(r) for r in returns], dtype=float)
    if arr.size == 0: return np.nan
    std = arr.std(ddof=1)
    if std == 0: return np.nan
    return arr.mean() / std * np.sqrt(12)

def sortino_ratio(returns: list[Decimal] | np.ndarray) -> float:
    arr = np.array([float(r) for r in returns], dtype=float)
    if arr.size == 0: return np.nan
    neg = arr[arr < 0]
    if neg.size == 0: return np.nan
    return arr.mean() / neg.std(ddof=1) * np.sqrt(12)

def calmar_ratio(returns: list[Decimal] | np.ndarray) -> float:
    arr = np.array([float(r) for r in returns], dtype=float)
    if arr.size == 0: return np.nan
    cum = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1
    mdd = abs(dd.min()) if len(dd) else np.nan
    years = len(arr) / 12
    if not np.isfinite(mdd) or mdd == 0 or years == 0: return np.nan
    cagr = cum[-1] ** (1 / years) - 1
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

def newey_west_lags(T: int) -> int:
    """Rule of thumb: L = floor(0.75 * T^(1/3)), at least 1."""
    return max(1, int(np.floor(0.75 * (T ** (1/3)))))

def dvr_stats(monthly_series: pd.Series, forecast_month: pd.Timestamp,
              lookback_years: int | None) -> tuple[float, float, float]:
    """
    For the target forecast month, run OLS with a dummy for that calendar month over the lookback window.
    Return (beta, pval, zscore=t-stat, HAC Newey-West).
    """
    nm = forecast_month.month
    df = monthly_series.loc[:(forecast_month - relativedelta(months=1))].to_frame('return')
    if lookback_years is not None:
        cutoff = forecast_month - relativedelta(years=lookback_years)
        df = df[df.index >= cutoff]
    if df.empty or df['return'].count() < 12:
        return (np.nan, np.nan, np.nan)

    df = df.copy()
    df['month'] = df.index.month
    df['D']     = (df['month'] == nm).astype(float)

    X = sm.add_constant(df['D'])
    L = newey_west_lags(len(df))
    try:
        model = sm.OLS(df['return'], X).fit(cov_type='HAC', cov_kwds={'maxlags': L})
        beta   = float(model.params.get('D', np.nan))
        pval   = float(model.pvalues.get('D', np.nan))
        zscore = float(model.tvalues.get('D', np.nan))  # NW t-stat
    except Exception:
        beta, pval, zscore = (np.nan, np.nan, np.nan)
    return (beta, pval, zscore)

def _rank_greedy(dfm: pd.DataFrame) -> tuple[list[str], list[str]]:
    """SIG>=1: plain greedy by Z (longs: desc; shorts: asc)."""
    orderL = dfm.sort_values('z', ascending=False).index.tolist()
    orderS = dfm.sort_values('z', ascending=True ).index.tolist()
    return orderL, orderS

def _rank_sig_first(dfm: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Significance-first ranking that respects SIG_LEVEL.
    """
    df = dfm.copy()
    df = df[np.isfinite(df['beta']) & np.isfinite(df['pval'])]
    if df.empty:
        return [], []

    elig_long  = df[(df['pval'] <= float(SIG_LEVEL)) & (df['beta'] > 0)]
    elig_short = df[(df['pval'] <= float(SIG_LEVEL)) & (df['beta'] < 0)]

    rest_long  = df.loc[~df.index.isin(elig_long.index)]
    rest_short = df.loc[~df.index.isin(elig_short.index)]

    elig_long  = elig_long.sort_values('beta', ascending=False)
    rest_long  = rest_long.sort_values('beta', ascending=False)

    elig_short = elig_short.sort_values('beta', ascending=True)
    rest_short = rest_short.sort_values('beta', ascending=True)

    orderL = elig_long.index.tolist()  + [t for t in rest_long.index.tolist()  if t not in elig_long.index]
    orderS = elig_short.index.tolist() + [t for t in rest_short.index.tolist() if t not in elig_short.index]
    return orderL, orderS

# =============================== #
# 4) LOAD DATA
# =============================== #
base = ROOT_DIR
# DVR runs on LOG monthly returns; GPS metrics/compounding use SIMPLE monthly returns
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

def generate_simulated_simple_returns_for_run(rng, lam, zero_noise, sigma_proj):
    """
    One-stop: simulate log returns for all tickers (or reuse base if no_noise), then convert to simple returns.
    Returns dict[ticker] -> pd.Series (simple returns).
    """
    log_sim = simulate_log_returns_with_sigma(log_rets, sigma_proj, lam=lam, rng=rng, no_noise=zero_noise)
    simple_sim = log_to_simple_dict(log_sim) if not zero_noise else log_to_simple_dict(log_rets)
    return simple_sim

# =============================== #
# 6) DVR RANKINGS ON SIMULATED SERIES
# =============================== #
def build_rankings_from_log_rets(log_rets_sim: dict[str, pd.Series],
                                 lookback_years: int) -> tuple[dict, dict, dict]:
    """
    Returns (long_rankings, short_rankings, z_scores_by_month) for all months TEST_SIM_START..FINAL_END.
    Only Top-1 will be used downstream.
    """
    tickers_sim = list(log_rets_sim)
    long_rankings, short_rankings = {}, {}
    z_scores_by_month_sim = {}
    cur = TEST_SIM_START
    while cur <= FINAL_END:
        stats_rows = []
        for t in tickers_sim:
            beta, pval, z = dvr_stats(log_rets_sim[t], cur, lookback_years)
            if not np.isfinite(z):
                continue
            elig = (np.isfinite(pval) and (pval <= float(SIG_LEVEL)))
            stats_rows.append({'ticker': t, 'beta': beta, 'pval': pval, 'z': z, 'elig': elig})
        if not stats_rows:
            long_rankings[cur]  = tickers_sim.copy()
            short_rankings[cur] = tickers_sim.copy()
        else:
            dfm = pd.DataFrame(stats_rows).set_index('ticker')
            z_scores_by_month_sim[cur] = dfm['z'].to_dict()
            if float(SIG_LEVEL) >= 0.999999:
                orderL, orderS = _rank_greedy(dfm)
            else:
                orderL, orderS = _rank_sig_first(dfm)
            orderL += [t for t in tickers_sim if t not in orderL]
            orderS += [t for t in tickers_sim if t not in orderS]
            long_rankings[cur]  = orderL[:len(tickers_sim)]
            short_rankings[cur] = orderS[:len(tickers_sim)]
        cur += relativedelta(months=1)
    return long_rankings, short_rankings, z_scores_by_month_sim

# =============================== #
# 7) TOP-1 & WEIGHTED MONTHLY SERIES (BASELINE, GPS, WEIGHTED)
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

def max_drawdown_from_monthly(returns: pd.Series) -> float:
    if returns.empty: return np.nan
    cum = np.cumprod(1.0 + returns.values)
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    return float(np.min(dd))  # negative

def compute_gps_mapping_for_month(dt: pd.Timestamp,
                                  rankings: dict[pd.Timestamp, list[str]],
                                  simple_rets_dict: dict[str, pd.Series],
                                  z_scores_by_month: dict[pd.Timestamp, dict[str, float]],
                                  bench_full: pd.Series,
                                  *,
                                  direction: str,
                                  calib_years: int,
                                  rolling: bool) -> dict[int, int]:
    """
    Build GPS new->prev rank mapping for month dt using FOUR metrics:
      1) seasonality_score (higher better)
      2) Sharpe (higher better)
      3) Treynor vs. equal-weight benchmark within the calibration window (higher better)
      4) Max Drawdown magnitude (lower better -> inverted)
    Rolling: use [dt-calib_years, dt-1m]; Fixed: first calib_years before FINAL_SIM_START.
    """
    # --- calibration window
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

    # slice benchmark once for this window
    bench_win = bench_full.loc[(bench_full.index >= win_start) & (bench_full.index <= win_end)]
    if bench_win.empty:
        return {1: 1}

    # --- collect per-prev-rank portfolio series across the window
    order_today = rankings.get(dt, [])
    num_t = len(order_today)
    if num_t == 0:
        return {1: 1}

    rets_by_pr: dict[int, list[float]] = defaultdict(list)
    dates_by_pr: dict[int, list[pd.Timestamp]] = defaultdict(list)
    z_by_prev_rank = defaultdict(list)  # for seasonality_score when SIG>=1

    for d in pd.date_range(win_start, win_end, freq='MS'):
        order_d = rankings.get(d)
        if not order_d:
            continue
        zmap = z_scores_by_month.get(d, {})
        for pr, tkr in enumerate(order_d, start=1):
            s = simple_rets_dict.get(tkr)
            if s is None or d not in s.index:
                continue
            r = float(s.loc[d])
            if direction == 'short':
                r = (1.0 / (1.0 + r)) - 1.0
            rets_by_pr[pr].append(r)
            dates_by_pr[pr].append(d)

            zval = zmap.get(tkr, np.nan)
            if direction == 'short' and np.isfinite(zval):
                zval = -float(zval)
            z_by_prev_rank[pr].append(float(zval) if np.isfinite(zval) else np.nan)

    # --- compute metrics per prev-rank
    rows = []
    for pr in range(1, num_t + 1):
        if pr not in rets_by_pr:
            rows.append({'prev_rank': pr,
                         'seasonality_score': np.nan,
                         'sharpe': np.nan,
                         'treynor': np.nan,
                         'mdd_mag': np.nan})
            continue

        port = pd.Series(rets_by_pr[pr], index=pd.DatetimeIndex(dates_by_pr[pr]))
        port = port.loc[(port.index >= win_start) & (port.index <= win_end)].sort_index()

        # Sharpe
        sr = sharpe_ratio(list(port.values))

        # Treynor vs window benchmark (function already does inner join)
        trey = treynor_ratio_series(port, bench_win)

        # Max drawdown magnitude (positive number, lower is better)
        mdd_mag = abs(max_drawdown_from_monthly(port))

        # Seasonality score: mean Z (SIG>=1) else rank proxy
        if float(SIG_LEVEL) >= 1.0:
            z_list = np.asarray(z_by_prev_rank.get(pr, []), dtype=float)
            seasonality_score = float(np.nanmean(z_list)) if np.isfinite(z_list).any() else np.nan
        else:
            seasonality_score = -float(pr)

        rows.append({'prev_rank': pr,
                     'seasonality_score': seasonality_score,
                     'sharpe': sr,
                     'treynor': trey,
                     'mdd_mag': mdd_mag})

    mdf = pd.DataFrame(rows).set_index('prev_rank').sort_index()
    if mdf.empty:
        return {1: 1}

    # --- normalize (higher is better), invert MDD for scoring only (internal)
    mdf['seasonality_score_01'] = minmax_01(mdf['seasonality_score'].values)
    mdf['sharpe_01']            = minmax_01(mdf['sharpe'].values)
    mdf['treynor_01']           = minmax_01(mdf['treynor'].values)
    mdf['mdd_inv_01']           = 1.0 - minmax_01(mdf['mdd_mag'].values)  # invert: lower mdd is better

    norm_cols = ['seasonality_score_01', 'sharpe_01', 'treynor_01', 'mdd_inv_01']
    # harmonic mean (0 if any 0; NaN if any NaN)
    mdf['score'] = [gps_harmonic_01(mdf.loc[i, norm_cols].values) for i in mdf.index]

    # final new_rank (desc by score)
    mdf['new_rank'] = mdf['score'].rank(ascending=False, method='first')

    # invert to mapping new->prev (take first prev for each new)
    inv = {}
    for prev_rank, row in mdf.iterrows():
        nr = int(row['new_rank'])
        if nr not in inv:
            inv[nr] = int(prev_rank)
    return inv

def monthly_top1_returns_gps(rankings: dict[pd.Timestamp, list[str]],
                             simple_rets_dict: dict[str, pd.Series],
                             z_scores_by_month: dict[pd.Timestamp, dict[str, float]],
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
    TOP-1 ONLY: maps GPS new_rank=1 to a prev_rank using combined metrics; then takes that ticker's return.
    """
    out_ret, out_tkr, idx = [], [], []
    for dt in pd.date_range(start_dt, end_dt, freq='MS'):
        order_today = rankings.get(dt)
        if not order_today:
            continue
        mapping = compute_gps_mapping_for_month(
            dt, rankings, simple_rets_dict, z_scores_by_month, bench_full,
            direction=direction, calib_years=calib_years, rolling=rolling
        )
        prev_rank = mapping.get(1, 1)
        if prev_rank < 1 or prev_rank > len(order_today):
            continue
        tkr = order_today[prev_rank - 1]
        out_tkr.append(tkr)  # record ticker regardless of available returns
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
    # Build DF of simple returns aligned to grid
    df = pd.DataFrame({t: s.reindex(idx) for t, s in simple_rets_dict.items()})
    if direction == 'short':
        df = (1.0 / (1.0 + df)) - 1.0
    ew = df.mean(axis=1, skipna=True)
    ew_wc = (1.0 - entry_cost) * (1.0 + ew) - 1.0
    return ew_wc.rename("equal_weight")

# =============================== #
# 8) METRICS & WINDOWS
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
    sr  = sharpe_ratio(list(arr))
    sor = sortino_ratio(list(arr))
    cal = calmar_ratio(list(arr))
    cum = float(np.prod(1.0 + arr) - 1.0)
    cg  = cagr_from_monthly(returns)
    # MDD negative; store as negative; report abs() later when needed
    mdd = max_drawdown_from_monthly(pd.Series(arr))
    sd  = float(np.std(arr, ddof=1)) if arr.size >= 2 else np.nan
    return Perf(cum, cg, sr, sor, cal, mdd, sd)

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
# 8.1) METRICS CORRELATION HELPERS (kept, but not executed unless flags True)
# =============================== #
def compute_metrics_correlation(df_metrics: pd.DataFrame,
                                metric_cols: list[str]) -> pd.DataFrame:
    """Return Pearson correlation matrix across selected metric columns."""
    return df_metrics[metric_cols].astype(float).corr(method='pearson')

def save_correlation_csv(corr_df: pd.DataFrame, out_path: Path) -> None:
    corr_df.to_csv(out_path)

def save_correlation_heatmap(corr_df: pd.DataFrame, out_path_pdf: Path) -> None:
    """Save a heatmap PDF of the correlation matrix (not called unless flag enabled)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_df.values, vmin=-1, vmax=1, cmap='coolwarm')
    metric_cols = list(corr_df.columns)
    ax.set_xticks(range(len(metric_cols)))
    ax.set_yticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=45, ha='right')
    ax.set_yticklabels(metric_cols)
    for i in range(len(metric_cols)):
        for j in range(len(metric_cols)):
            val = corr_df.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation of Performance Metrics")
    fig.tight_layout()
    fig.savefig(out_path_pdf)
    plt.close(fig)

# =============================== #
# 8.2) SIGNIFICANCE HELPERS (paired, one-sided both directions at α/2)
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
        print(f"  {right_label} vs {left_label} — {metric_name:<9}: insufficient valid pairs.")
        return
    better = f"{right_label} better at 5% two-sided" if (p_g is not None and p_g < PER_SIDE_ALPHA) else "no better effect at 5%"
    worse  = f"{right_label} worse at 5% two-sided"  if (p_l is not None and p_l < PER_SIDE_ALPHA) else "no worse effect at 5%"
    print(f"  {right_label} vs {left_label} — {metric_name:<9}: n={n:4d} | "
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
# 9) MONTE CARLO: BOTH DIRECTIONS (efficient)
# =============================== #
def run_monte_carlo_both(n_runs=MC_RUNS, lam=LAMBDA_EWMA, save_series=SAVE_SERIES, zero_noise=ZERO_NOISE):
    OUT_DIR_MC.mkdir(parents=True, exist_ok=True)

    # ---- Tee all prints to console AND to buffer (we save it at the end)
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams: st.write(s)
            return len(s)
        def flush(self):
            for st in self.streams:
                try: st.flush()
                except Exception: pass

    tee = Tee(sys.__stdout__, buf)

    # We collect metrics here (one row per run × direction × strategy)
    metrics_rows: list[dict] = []
    choice_records = []  # unified CSV with 'direction' column

    with redirect_stdout(tee):
        rng = np.random.default_rng(RNG_SEED)
        t0 = time.time()
        print(f"Starting Monte Carlo (BOTH directions): runs={n_runs}, lambda={lam}, zero_noise={zero_noise}")

        # Precompute sigma projections once (used only if zero_noise=False)
        sigma_proj = None if zero_noise else precompute_sigma_proj_for_all(log_rets, lam=lam)

        # Static grids
        month_grid = pd.date_range(FINAL_SIM_START, FINAL_SIM_END, freq='MS')
        splits = three_fixed_windows()

        # Aggregators per direction (now include 'weighted')
        dirs = ('long', 'short')
        full_metrics = {d: {'baseline': [], 'gps': [], 'weighted': []} for d in dirs}
        sub_metrics  = {d: {'baseline': defaultdict(list), 'gps': defaultdict(list), 'weighted': defaultdict(list)} for d in dirs}
        sub_treynor  = {d: {'baseline': defaultdict(list), 'gps': defaultdict(list), 'weighted': defaultdict(list)} for d in dirs}
        pos_counts   = {d: {'baseline': 0, 'gps': 0, 'weighted': 0} for d in dirs}
        neg_counts   = {d: {'baseline': 0, 'gps': 0, 'weighted': 0} for d in dirs}

        # MAIN LOOP
        for run in range(1, n_runs + 1):
            # 1) simulate simple returns ONCE per run (reused for all portfolios)
            simple_sim = generate_simulated_simple_returns_for_run(rng, lam, zero_noise, sigma_proj)

            # 2) build rankings/z ONCE per run — needs LOG series; rebuild from simulated logs
            #    To keep fast, reconstruct logs from simple via log(1+r) (consistent for small r)
            log_sim = {t: np.log1p(s.astype(float)).rename(t) for t, s in simple_sim.items()}

            L_rank, S_rank, z_by_month = build_rankings_from_log_rets(log_sim, LOOKBACK_YEARS)

            # 3) build equal-weight benchmarks (from simulated simple returns)
            bench_full  = build_equal_weight_benchmark(simple_sim, TEST_SIM_START,  FINAL_SIM_END)
            bench_apply = build_equal_weight_benchmark(simple_sim, FINAL_SIM_START, FINAL_SIM_END)

            # 4) iterate BOTH directions using shared computations
            for direction, rank_use in (('long', L_rank), ('short', S_rank)):
                # Baseline + GPS top-1 series
                top1_baseline, tkr_baseline = monthly_top1_returns(
                    rank_use, simple_sim, direction=direction,
                    start_dt=FINAL_SIM_START, end_dt=FINAL_SIM_END,
                    entry_cost=ENTRY_COST, return_tickers=True
                )
                top1_gps, tkr_gps = monthly_top1_returns_gps(
                    rank_use, simple_sim, z_by_month, bench_full,
                    direction=direction,
                    start_dt=FINAL_SIM_START, end_dt=FINAL_SIM_END,
                    calib_years=GPS_CALIB_YEARS, rolling=GPS_ROLLING_ENABLED,
                    entry_cost=ENTRY_COST, return_tickers=True
                )
                # New: Equal-weight portfolio across all tickers (direction-aware), entry cost monthly
                eq_weight_series = equal_weight_portfolio_series(
                    simple_sim, direction=direction,
                    start_dt=FINAL_SIM_START, end_dt=FINAL_SIM_END,
                    entry_cost=ENTRY_COST
                )

                # NAV series (for CSV context of Top-1 only)
                nav_baseline = nav_from_returns_on_grid(top1_baseline, START_VALUE, month_grid)
                nav_gps      = nav_from_returns_on_grid(top1_gps,      START_VALUE, month_grid)

                # Save choices in unified long-form (optionally limited to first N months)
                if SAVE_TICKER_CHOICES_CSV:
                    save_months = month_grid
                    if isinstance(TOP1_CHOICES_MAX_PER_RUN, int) and TOP1_CHOICES_MAX_PER_RUN > 0:
                        save_months = save_months[:TOP1_CHOICES_MAX_PER_RUN]
                    for dt in save_months:
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

                # Full-period metrics (apply window)
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

                # ---- Collect table rows for correlation and key metrics
                # Baseline row
                metrics_rows.append({
                    'run': run, 'direction': direction, 'strategy': 'baseline',
                    'sharpe': pb.sharpe,
                    'sortino': pb.sortino,
                    'treynor': treynor_ratio_series(top1_baseline, bench_apply),
                    'information': information_ratio_series(top1_baseline, bench_apply),
                    'calmar': pb.calmar,
                    'cum_ret': pb.cum_ret,
                    'mdd': abs(pb.mdd) if np.isfinite(pb.mdd) else np.nan,  # magnitude
                    'mer': mean_excess_return_series(top1_baseline, bench_apply),
                })
                # GPS row
                metrics_rows.append({
                    'run': run, 'direction': direction, 'strategy': 'gps',
                    'sharpe': pg.sharpe,
                    'sortino': pg.sortino,
                    'treynor': treynor_ratio_series(top1_gps, bench_apply),
                    'information': information_ratio_series(top1_gps, bench_apply),
                    'calmar': pg.calmar,
                    'cum_ret': pg.cum_ret,
                    'mdd': abs(pg.mdd) if np.isfinite(pg.mdd) else np.nan,  # magnitude
                    'mer': mean_excess_return_series(top1_gps, bench_apply),
                })
                # WEIGHTED row
                metrics_rows.append({
                    'run': run, 'direction': direction, 'strategy': 'weighted',
                    'sharpe': pw.sharpe,
                    'sortino': pw.sortino,
                    'treynor': treynor_ratio_series(eq_weight_series, bench_apply),
                    'information': information_ratio_series(eq_weight_series, bench_apply),
                    'calmar': pw.calmar,
                    'cum_ret': pw.cum_ret,
                    'mdd': abs(pw.mdd) if np.isfinite(pw.mdd) else np.nan,  # magnitude
                    'mer': mean_excess_return_series(eq_weight_series, bench_apply),
                })

                # Subperiod metrics + Treynor capture
                for i, (s, e) in enumerate(splits, start=1):
                    sb = top1_baseline[(top1_baseline.index >= s) & (top1_baseline.index <= e)]
                    sg = top1_gps[(top1_gps.index >= s) & (top1_gps.index <= e)]
                    sw = eq_weight_series[(eq_weight_series.index >= s) & (eq_weight_series.index <= e)]

                    sub_metrics[direction]['baseline'][i].append(perf_from_monthly(sb))
                    sub_metrics[direction]['gps'][i].append(perf_from_monthly(sg))
                    sub_metrics[direction]['weighted'][i].append(perf_from_monthly(sw))

                    bench_win = bench_apply[(bench_apply.index >= s) & (bench_apply.index <= e)]
                    sub_treynor[direction]['baseline'][i].append(treynor_ratio_series(sb, bench_win))
                    sub_treynor[direction]['gps'][i].append(treynor_ratio_series(sg, bench_win))
                    sub_treynor[direction]['weighted'][i].append(treynor_ratio_series(sw, bench_win))

            # progress
            if (run % 10) == 0:
                elapsed = time.time() - t0
                print(f"  Completed {run}/{n_runs} runs | elapsed {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

        # Write unified choices CSV (with optional limit already applied above)
        if SAVE_TICKER_CHOICES_CSV and choice_records:
            df_choices = pd.DataFrame(choice_records)
            df_choices.to_csv(OUT_DIR_MC / "top1_ticker_choices_all_runs.csv", index=False)

        # === Reporting helpers
        def agg(perf_list: list[Perf]) -> dict[str, float]:
            out = {}
            for k in ['cum_ret','cagr','sharpe','sortino','calmar','mdd','stdev']:
                x = np.array([getattr(p, k) for p in perf_list], float)
                out[f'{k}_mean']   = float(np.nanmean(x))
                out[f'{k}_median'] = float(np.nanmedian(x))
                out[f'{k}_std']    = float(np.nanstd(x, ddof=1))
            return out

        def treynor_stats_from_df(df: pd.DataFrame, direction: str, strategy: str) -> dict[str, float]:
            sub = df[(df['direction'] == direction) & (df['strategy'] == strategy)]
            vals = sub['treynor'].astype(float).values
            return {
                'treynor_mean': float(np.nanmean(vals)),
                'treynor_median': float(np.nanmedian(vals)),
                'treynor_std': float(np.nanstd(vals, ddof=1))
            }

        df_metrics_all = None
        if metrics_rows:
            df_metrics_all = pd.DataFrame(metrics_rows)

        # === Print LONG first, divider, then SHORT
        for direction in ('long', 'short'):
            n_runs_local = n_runs
            agg_full_b = agg(full_metrics[direction]['baseline'])
            agg_full_g = agg(full_metrics[direction]['gps'])
            agg_full_w = agg(full_metrics[direction]['weighted'])

            # Treynor full-period stats
            trey_b = treynor_stats_from_df(df_metrics_all, direction, 'baseline') if df_metrics_all is not None else {}
            trey_g = treynor_stats_from_df(df_metrics_all, direction, 'gps')      if df_metrics_all is not None else {}
            trey_w = treynor_stats_from_df(df_metrics_all, direction, 'weighted')  if df_metrics_all is not None else {}

            print("\n" + "="*96)
            print(f"MONTE CARLO MONTHLY (N={n_runs_local}, lambda={lam}, backcast={BACKCAST_N}, dir={direction}, zero_noise={zero_noise})")
            print("-"*96)
            print(f"Lookback={LOOKBACK_YEARS}y | Test={SIM_YEARS}y ({TEST_SIM_START.date()}→{TEST_SIM_END.date()}) | "
                  f"Apply={FINAL_SIM_START.date()}→{FINAL_SIM_END.date()} | SIG_LEVEL={SIG_LEVEL}")
            print(f"GPS: rolling={GPS_ROLLING_ENABLED}, calib_window={GPS_CALIB_YEARS}y | Entry cost per month={ENTRY_COST}")
            print("-"*96)
            print("Final outcome counts over apply period (Top-1 only):")
            print(f"  Baseline : +{pos_counts[direction]['baseline']:3d}   -{neg_counts[direction]['baseline']:3d}")
            print(f"  GPS      : +{pos_counts[direction]['gps']:3d}   -{neg_counts[direction]['gps']:3d}")
            print(f"  WEIGHTED : +{pos_counts[direction]['weighted']:3d}   -{neg_counts[direction]['weighted']:3d}")
            print("-"*96)

            def pretty_full(d, t, name):
                print(f"{name} : FULL PERIOD across runs:")
                print(f"  CumRet   mean={d['cum_ret_mean']:.4f}  median={d['cum_ret_median']:.4f}  sd={d['cum_ret_std']:.4f}")
                print(f"  CAGR     mean={d['cagr_mean']:.4f}    median={d['cagr_median']:.4f}    sd={d['cagr_std']:.4f}")
                print(f"  Sharpe   mean={d['sharpe_mean']:.3f}  median={d['sharpe_median']:.3f}  sd={d['sharpe_std']:.3f}")
                print(f"  Sortino  mean={d['sortino_mean']:.3f} median={d['sortino_median']:.3f} sd={d['sortino_std']:.3f}")
                print(f"  Calmar   mean={d['calmar_mean']:.3f}  median={d['calmar_median']:.3f}  sd={d['calmar_std']:.3f}")
                print(f"  MDD      mean={d['mdd_mean']:.3f}     median={d['mdd_median']:.3f}     sd={d['mdd_std']:.3f}")
                print(f"  Treynor  mean={t.get('treynor_mean', np.nan):.3f}  median={t.get('treynor_median', np.nan):.3f}  sd={t.get('treynor_std', np.nan):.3f}")
                print(f"  Vol(within) mean={d['stdev_mean']:.4f}   median={d['stdev_median']:.4f}   RunSD={d['stdev_std']:.4f}")

            pretty_full(agg_full_b, trey_b, "BASELINE")
            pretty_full(agg_full_g, trey_g, "GPS")
            pretty_full(agg_full_w, trey_w, "WEIGHTED")

            print("-"*96)
            print("Three fixed 3-year subperiod statistics across runs:")
            for i, (s, e) in enumerate(splits, start=1):
                ab = agg(sub_metrics[direction]['baseline'][i])
                ag = agg(sub_metrics[direction]['gps'][i])
                aw = agg(sub_metrics[direction]['weighted'][i])

                # Treynor subwindow stats
                tb_arr = np.array(sub_treynor[direction]['baseline'][i], dtype=float)
                tg_arr = np.array(sub_treynor[direction]['gps'][i], dtype=float)
                tw_arr = np.array(sub_treynor[direction]['weighted'][i], dtype=float)
                tb_mean, tb_sd = float(np.nanmean(tb_arr)), float(np.nanstd(tb_arr, ddof=1))
                tg_mean, tg_sd = float(np.nanmean(tg_arr)), float(np.nanstd(tg_arr, ddof=1))
                tw_mean, tw_sd = float(np.nanmean(tw_arr)), float(np.nanstd(tw_arr, ddof=1))

                print(f"  Window {i}: {s.date()} → {e.date()}")
                print(f"    Baseline: Ret(mean)={ab['cum_ret_mean']:.4f}  Ret(median)={ab['cum_ret_median']:.4f}  "
                      f"RunSD={ab['cum_ret_std']:.4f}  Vol(within)={ab['stdev_mean']:.4f}  "
                      f"CAGR(mean)={ab['cagr_mean']:.4f} (sd={ab['cagr_std']:.4f})  "
                      f"Sharpe(mean)={ab['sharpe_mean']:.3f} (sd={ab['sharpe_std']:.3f})  "
                      f"Calmar(mean)={ab['calmar_mean']:.3f} (sd={ab['calmar_std']:.3f})  "
                      f"Treynor(mean)={tb_mean:.3f} (sd={tb_sd:.3f})  "
                      f"MaxDD(mean)={ab['mdd_mean']:.3f} (sd={ab['mdd_std']:.3f})")
                print(f"    GPS     : Ret(mean)={ag['cum_ret_mean']:.4f}  Ret(median)={ag['cum_ret_median']:.4f}  "
                      f"RunSD={ag['cum_ret_std']:.4f}  Vol(within)={ag['stdev_mean']:.4f}  "
                      f"CAGR(mean)={ag['cagr_mean']:.4f} (sd={ag['cagr_std']:.4f})  "
                      f"Sharpe(mean)={ag['sharpe_mean']:.3f} (sd={ag['sharpe_std']:.3f})  "
                      f"Calmar(mean)={ag['calmar_mean']:.3f} (sd={ag['calmar_std']:.3f})  "
                      f"Treynor(mean)={tg_mean:.3f} (sd={tg_sd:.3f})  "
                      f"MaxDD(mean)={ag['mdd_mean']:.3f} (sd={ag['mdd_std']:.3f})")
                print(f"    WEIGHTED: Ret(mean)={aw['cum_ret_mean']:.4f}  Ret(median)={aw['cum_ret_median']:.4f}  "
                      f"RunSD={aw['cum_ret_std']:.4f}  Vol(within)={aw['stdev_mean']:.4f}  "
                      f"CAGR(mean)={aw['cagr_mean']:.4f} (sd={aw['cagr_std']:.4f})  "
                      f"Sharpe(mean)={aw['sharpe_mean']:.3f} (sd={aw['sharpe_std']:.3f})  "
                      f"Calmar(mean)={aw['calmar_mean']:.3f} (sd={aw['calmar_std']:.3f})  "
                      f"Treynor(mean)={tw_mean:.3f} (sd={tw_sd:.3f})  "
                      f"MaxDD(mean)={aw['mdd_mean']:.3f} (sd={aw['mdd_std']:.3f})")

            # === Significance section: paired, one-sided BOTH directions with per-side α (0.025)
            print("-"*96)
            print(f"Significance (paired, one-sided in both directions; per-side α={PER_SIDE_ALPHA:.3f} so overall two-sided 5%)")

            # Build arrays for metrics from Perf lists
            b_list = full_metrics[direction]['baseline']
            g_list = full_metrics[direction]['gps']
            w_list = full_metrics[direction]['weighted']
            arr = lambda attr, L: np.array([getattr(p, attr) for p in L], dtype=float)

            def sig_pair_block(left_label, right_label, L_list, R_list, df_dir_metrics):
                # Return (cum_ret), CAGR, Sharpe, Sortino, Calmar -> higher is better
                for name, attr in [("Return", "cum_ret"),
                                   ("CAGR", "cagr"),
                                   ("Sharpe", "sharpe"),
                                   ("Sortino", "sortino"),
                                   ("Calmar", "calmar")]:
                    t_g, p_g, t_l, p_l, n = paired_one_sided_both(arr(attr, L_list), arr(attr, R_list))
                    print_sig_line_generic(name, left_label, right_label, t_g, p_g, t_l, p_l, n, better_note="higher is better")

                # Treynor from df_metrics_all by pair
                t_g, p_g, t_l, p_l, n = paired_one_sided_both_from_df(df_dir_metrics, left_label.lower(), right_label.lower(), 'treynor')
                print_sig_line_generic("Treynor", left_label, right_label, t_g, p_g, t_l, p_l, n, better_note="higher is better")

                # Max Drawdown: Perf.mdd is negative, less severe (closer to 0, i.e., higher) is better
                t_g, p_g, t_l, p_l, n = paired_one_sided_both(arr("mdd", L_list), arr("mdd", R_list))
                print_sig_line_generic("MaxDD", left_label, right_label, t_g, p_g, t_l, p_l, n, better_note="less severe (higher)")

            # Prepare direction-filtered metrics DF
            df_dir = df_metrics_all[df_metrics_all['direction'] == direction] if df_metrics_all is not None else pd.DataFrame()

            # GPS vs Baseline
            sig_pair_block("baseline", "gps", b_list, g_list, df_dir)

            # WEIGHTED vs Baseline
            sig_pair_block("baseline", "weighted", b_list, w_list, df_dir)

            # WEIGHTED vs GPS
            sig_pair_block("gps", "weighted", g_list, w_list, df_dir)

            # Divider after LONG block
            if direction == 'long':
                print("")
                print("#" * 96)

        total_elapsed = time.time() - t0
        print(f"Total runtime (both directions): {time.strftime('%H:%M:%S', time.gmtime(total_elapsed))}")
        print("="*96 + "\n")

        # === Save metrics table (ONLY this) ===
        if metrics_rows:
            df_metrics = pd.DataFrame(metrics_rows)
            df_metrics.to_csv(OUT_DIR_MC / "performance_metrics_by_portfolio.csv", index=False)

            # Optional correlation deliverables
            metric_cols = ["sharpe","sortino","treynor","information","calmar","cum_ret","mdd","mer"]
            if GENERATE_METRICS_CORR_CSV or GENERATE_METRICS_CORR_HEATMAP:
                corr = compute_metrics_correlation(df_metrics, metric_cols)
                if GENERATE_METRICS_CORR_CSV:
                    save_correlation_csv(corr, OUT_DIR_MC / "metrics_correlation_matrix.csv")
                if GENERATE_METRICS_CORR_HEATMAP:
                    save_correlation_heatmap(corr, OUT_DIR_MC / "metrics_correlation_heatmap.pdf")

    # === Write terminal printout to file ===
    with open(OUT_DIR_MC / "console_report.txt", "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

# =============================== #
# 10) RUN — single efficient call
# =============================== #
if __name__ == "__main__":
    run_monte_carlo_both(n_runs=MC_RUNS, lam=LAMBDA_EWMA, save_series=SAVE_SERIES, zero_noise=ZERO_NOISE)
