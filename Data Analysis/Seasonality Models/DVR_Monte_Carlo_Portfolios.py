import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime
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
GENERATE_METRICS_CORR_CSV     = True
GENERATE_METRICS_CORR_HEATMAP = True  # requires matplotlib

SIGNIF_TWO_SIDED_ALPHA = 0.05
PER_SIDE_ALPHA         = SIGNIF_TWO_SIDED_ALPHA / 2.0  # 0.025

# =============================== #
# 1) PARAMETERS & WINDOWS
# =============================== #
START_DATE   = datetime(2001, 1, 1)
FINAL_END    = datetime(2024, 12, 31)

LOOKBACK_YEARS  = 10  # DVR lookback
SIM_YEARS       = 5   # 2011-2015 given START_DATE
# Apply window: 2016-01 -> 2024-12

START_VALUE = 1000.0
ENTRY_COST  = 0.0025  # once per month

# DVR params
SIG_LEVEL = 0.05      # 1.0 => greedy by Z (no p filter)

# GPS switches
GPS_ROLLING_ENABLED = True
GPS_CALIB_YEARS     = SIM_YEARS

# GPS score metric selection (case-insensitive; aliases allowed)
# You can include "Treynor" and it will use the adjusted Treynor (abs(beta)).
GPS_SCORE_COMPONENTS = ["Seasonality", "TREYNOR", "ONE_MINUS_MDD"]  # e.g. ["Seasonality","Sharpe","Treynor"]

# IO
ROOT_DIR   = Path().resolve().parent.parent / "Complete Data"
OUT_DIR_MC = Path().resolve() / "Outputs_MC" / f"DVR_MC_p≤{SIG_LEVEL}_{len(GPS_SCORE_COMPONENTS)}_F"

# Monte Carlo
MC_RUNS     = 50
LAMBDA_EWMA = 0.94
BACKCAST_N  = 12
RNG_SEED    = 42
SAVE_SERIES = False
SAVE_TICKER_CHOICES_CSV = True
ZERO_NOISE  = False

# =============================== #
# 2) DATE RANGES
# =============================== #
LOOKBACK_END    = (START_DATE + relativedelta(years=LOOKBACK_YEARS) - pd.offsets.MonthEnd(1))
TEST_SIM_START  = START_DATE + relativedelta(years=LOOKBACK_YEARS)                         # 2011-01
TEST_SIM_END    = TEST_SIM_START + relativedelta(years=SIM_YEARS) - pd.offsets.MonthEnd(1) # 2015-12
FINAL_SIM_START = START_DATE + relativedelta(years=LOOKBACK_YEARS + SIM_YEARS)             # 2016-01
FINAL_SIM_END   = FINAL_END                                                                 # 2024-12

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

def build_equal_weight_benchmark(simple_rets_dict: dict[str, pd.Series],
                                 start_dt: pd.Timestamp,
                                 end_dt: pd.Timestamp) -> pd.Series:
    """
    Passive monthly benchmark (gross, long): equally-weighted average across all tickers' simple returns
    available in a given month. Index is the month grid [start_dt..end_dt].
    """
    idx = pd.date_range(start_dt, end_dt, freq='MS')
    df = pd.DataFrame({t: s.reindex(idx) for t, s in simple_rets_dict.items()})
    return df.mean(axis=1, skipna=True).rename("benchmark")

def long_benchmark_net(bench_gross_long: pd.Series, entry_cost: float) -> pd.Series:
    """
    Convert a gross, LONG benchmark to NET-of-cost (no direction transform).
    r_net = (1-c)*(1+r) - 1
    """
    b = bench_gross_long.astype(float)
    b = (1.0 - entry_cost) * (1.0 + b) - 1.0
    return b.rename(f"{bench_gross_long.name or 'benchmark'}_long_net")

def compute_beta(port: pd.Series, bench: pd.Series) -> float:
    a = pd.concat([bench, port], axis=1, join="inner").dropna()
    if a.shape[0] < 3: return np.nan
    m = a.iloc[:, 0].values
    b = a.iloc[:, 1].values
    var_m = np.var(m, ddof=1)
    if var_m == 0 or not np.isfinite(var_m): return np.nan
    cov = np.cov(b, m, ddof=1)[0, 1]
    return cov / var_m

def treynor_ratio_series(port: pd.Series, bench: pd.Series) -> float:
    """Raw Treynor (annualized)."""
    beta = compute_beta(port, bench)
    if not np.isfinite(beta) or beta == 0: return np.nan
    mu_m = float(np.nanmean(port.values)) if len(port) else np.nan
    return 12.0 * (mu_m / beta)

def treynor_ratio_series_absbeta(port: pd.Series, bench: pd.Series) -> float:
    """
    Adjusted Treynor (annualized): uses |beta| so 'higher is better' for long and short.
    """
    beta = compute_beta(port, bench)
    if not np.isfinite(beta) or beta == 0: return np.nan
    mu_m = float(np.nanmean(port.values)) if len(port) else np.nan
    return 12.0 * (mu_m / abs(beta))

def information_ratio_series(port: pd.Series, bench: pd.Series) -> float:
    a = pd.concat([port, bench], axis=1, join="inner").dropna()
    if a.shape[0] < 3: return np.nan
    active = a.iloc[:, 0] - a.iloc[:, 1]
    std = float(np.nanstd(active.values, ddof=1))
    if std == 0 or not np.isfinite(std): return np.nan
    return float(np.nanmean(active.values)) / std * np.sqrt(12)

def mean_excess_return_series(port: pd.Series, bench: pd.Series) -> float:
    a = pd.concat([port, bench], axis=1, join="inner")
    if a.shape[0] == 0: return np.nan
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
    years = len(arr) / 12.0
    if years <= 0: return np.nan
    cagr = cum[-1] ** (1 / years) - 1
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    mdd = abs(dd.min()) if dd.size else np.nan
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

def newey_west_lags(T: int) -> int:
    return max(1, int(np.floor(0.75 * (T ** (1/3)))))

def dvr_stats(monthly_series: pd.Series, forecast_month: pd.Timestamp,
              lookback_years: int | None) -> tuple[float, float, float]:
    """
    OLS with month dummy over lookback window (Newey-West HAC).
    Returns (beta, pval, zscore).
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
        model  = sm.OLS(df['return'], X).fit(cov_type='HAC', cov_kwds={'maxlags': L})
        beta   = float(model.params.get('D', np.nan))
        pval   = float(model.pvalues.get('D', np.nan))
        zscore = float(model.tvalues.get('D', np.nan))
    except Exception:
        beta, pval, zscore = (np.nan, np.nan, np.nan)
    return (beta, pval, zscore)

def _rank_greedy(dfm: pd.DataFrame) -> tuple[list[str], list[str]]:
    orderL = dfm.sort_values('z', ascending=False).index.tolist()
    orderS = dfm.sort_values('z', ascending=True ).index.tolist()
    return orderL, orderS

def _rank_sig_first(dfm: pd.DataFrame) -> tuple[list[str], list[str]]:
    df = dfm.copy()
    df = df[np.isfinite(df['beta']) & np.isfinite(df['pval'])]
    if df.empty: return [], []
    elig_long  = df[(df['pval'] <= float(SIG_LEVEL)) & (df['beta'] > 0)]
    elig_short = df[(df['pval'] <= float(SIG_LEVEL)) & (df['beta'] < 0)]
    rest_long  = df.loc[~df.index.isin(elig_long.index)].sort_values('beta', ascending=False)
    rest_short = df.loc[~df.index.isin(elig_short.index)].sort_values('beta', ascending=True)
    elig_long  = elig_long.sort_values('beta', ascending=False)
    elig_short = elig_short.sort_values('beta', ascending=True)
    orderL = elig_long.index.tolist()  + [t for t in rest_long.index.tolist()  if t not in elig_long.index]
    orderS = elig_short.index.tolist() + [t for t in rest_short.index.tolist() if t not in elig_short.index]
    return orderL, orderS

# =============================== #
# 4) LOAD DATA
# =============================== #
base = ROOT_DIR
# DVR on LOG; GPS/compounding on SIMPLE
def load_all():
    log_rets    = load_returns(base / "All_Monthly_Log_Return_Data")
    simple_rets = load_returns(base / "All_Monthly_Return_Data")
    return log_rets, simple_rets, list(log_rets.keys())

log_rets, simple_rets, tickers = load_all()

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
    return {tkr: ewma_projected_sigma_series(s, lam=lam).values for tkr, s in log_rets_dict.items()}

def simulate_log_returns_with_sigma(log_rets_dict: dict[str, pd.Series],
                                    sigma_proj: dict[str, np.ndarray] | None,
                                    lam=LAMBDA_EWMA,
                                    rng: np.random.Generator | None = None,
                                    no_noise: bool = False) -> dict[str, pd.Series]:
    if no_noise:
        return {tkr: s.dropna().sort_index().astype(float).copy() for tkr, s in log_rets_dict.items()}
    if rng is None:
        rng = np.random.default_rng()
    sim = {}
    for tkr, s in log_rets_dict.items():
        s = s.dropna().sort_index().astype(float)
        if s.empty: continue
        sig = sigma_proj[tkr] if sigma_proj and tkr in sigma_proj else ewma_projected_sigma_series(s, lam=lam).values
        z = rng.standard_normal(len(s))
        r_sim = s.values + sig * z
        sim[tkr] = pd.Series(r_sim, index=s.index)
    return sim

def log_to_simple_dict(log_rets_dict: dict[str, pd.Series]) -> dict[str, pd.Series]:
    return {tkr: (np.exp(s.astype(float)) - 1.0).rename(tkr) for tkr, s in log_rets_dict.items()}

def generate_simulated_simple_returns_for_run(rng, lam, zero_noise, sigma_proj):
    log_sim = simulate_log_returns_with_sigma(log_rets, sigma_proj, lam=lam, rng=rng, no_noise=zero_noise)
    simple_sim = log_to_simple_dict(log_sim) if not zero_noise else log_to_simple_dict(log_rets)
    return simple_sim

# =============================== #
# 6) DVR RANKINGS ON SIMULATED SERIES
# =============================== #
def build_rankings_from_log_rets(log_rets_sim: dict[str, pd.Series],
                                 lookback_years: int) -> tuple[dict, dict, dict]:
    tickers_sim = list(log_rets_sim)
    long_rankings, short_rankings = {}, {}
    z_scores_by_month_sim = {}
    cur = TEST_SIM_START
    while cur <= FINAL_END:
        stats_rows = []
        for t in tickers_sim:
            beta, pval, z = dvr_stats(log_rets_sim[t], cur, lookback_years)
            if not np.isfinite(z): continue
            stats_rows.append({'ticker': t, 'beta': beta, 'pval': pval, 'z': z})
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
# 7) TOP-1 MONTHLY SERIES (BASELINE, GPS)
# =============================== #
def monthly_top1_returns(rankings: dict[pd.Timestamp, list[str]],
                         simple_rets_dict: dict[str, pd.Series],
                         *,
                         direction: str,
                         start_dt: pd.Timestamp,
                         end_dt: pd.Timestamp,
                         entry_cost: float,
                         return_tickers: bool = False) -> pd.Series | tuple[pd.Series, pd.Series]:
    out_ret, out_tkr, idx = [], [], []
    for dt in pd.date_range(start_dt, end_dt, freq='MS'):
        order = rankings.get(dt)
        if not order: continue
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
    r = pd.Series(returns, dtype=float).dropna()
    if r.empty: return np.nan
    cum = (1.0 + r.values).cumprod()
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    return float(dd.min())  # negative

def compute_gps_mapping_for_month(dt: pd.Timestamp,
                                  rankings: dict[pd.Timestamp, list[str]],
                                  simple_rets_dict: dict[str, pd.Series],
                                  z_scores_by_month: dict[pd.Timestamp, dict[str, float]],
                                  bench_full_gross_long: pd.Series,
                                  *,
                                  direction: str,
                                  calib_years: int,
                                  rolling: bool,
                                  entry_cost: float) -> dict[int, int]:
    """
    GPS new->prev rank mapping for month dt.

    Seasonality:
      • Use the DVR z-score at dt for the ticker at prev-rank pr (built from lookback).
      • Flip sign for shorts.

    Ratio metrics (Sharpe, Sortino, TreynorAdj, Information, Calmar, CumRet, MER_ann, One_minus_mdd):
      • Computed on NET returns over a calibration window with NO look-ahead:
          rolling=True  -> [dt - calib_years .. dt - 1m]
          rolling=False -> [FINAL_SIM_START - calib_years .. FINAL_SIM_START - 1m]
      • IMPORTANT: All ratios vs the SAME long-only, net benchmark.
    """
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

    # Common long-only NET benchmark window
    bench_win_raw = bench_full_gross_long.loc[(bench_full_gross_long.index >= win_start) & (bench_full_gross_long.index <= win_end)]
    if bench_win_raw.empty:
        return {1: 1}
    bench_win = long_benchmark_net(bench_win_raw, entry_cost)

    # Rankings today + collect per-prev-rank returns (NET, direction-aware)
    order_today = rankings.get(dt, [])
    num_t = len(order_today)
    if num_t == 0:
        return {1: 1}

    rets_by_pr: dict[int, list[float]] = defaultdict(list)
    dates_by_pr: dict[int, list[pd.Timestamp]] = defaultdict(list)

    for d in pd.date_range(win_start, win_end, freq='MS'):
        order_d = rankings.get(d)
        if not order_d: continue
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

    z_map_dt = z_scores_by_month.get(dt, {})

    rows = []
    for pr in range(1, num_t + 1):
        tkr_at_dt = order_today[pr - 1] if pr - 1 < len(order_today) else None

        seasonality_score = np.nan
        if tkr_at_dt is not None:
            zval = z_map_dt.get(tkr_at_dt, np.nan)
            if np.isfinite(zval):
                seasonality_score = float(-zval) if direction == 'short' else float(zval)

        if pr in rets_by_pr:
            port = pd.Series(rets_by_pr[pr], index=pd.DatetimeIndex(dates_by_pr[pr]))
            port = port.loc[(port.index >= win_start) & (port.index <= win_end)].sort_index()

            sr     = sharpe_ratio(list(port.values))
            sor    = sortino_ratio(list(port.values))
            trey_r = treynor_ratio_series(port, bench_win)
            trey_a = treynor_ratio_series_absbeta(port, bench_win)
            info   = information_ratio_series(port, bench_win)
            cal    = calmar_ratio(list(port.values))
            cum    = float(np.prod(1.0 + port.values) - 1.0) if len(port) else np.nan
            mer_a  = 12.0 * mean_excess_return_series(port, bench_win)
            mdd_m  = abs(max_drawdown_from_monthly(port))
            one_m  = 1.0 - mdd_m if np.isfinite(mdd_m) else np.nan
        else:
            sr = sor = trey_r = trey_a = info = cal = cum = mer_a = mdd_m = one_m = np.nan

        rows.append({
            'prev_rank': pr,
            'seasonality_score': seasonality_score,
            'sharpe': sr,
            'sortino': sor,
            'treynor_raw': trey_r,
            'treynor_adj': trey_a,
            'information': info,
            'calmar': cal,
            'cum_ret': cum,
            'mer_ann': mer_a,
            'mdd_mag': mdd_m,
            'one_minus_mdd': one_m,
        })

    mdf = pd.DataFrame(rows).set_index('prev_rank').sort_index()
    if mdf.empty:
        return {1: 1}

    alias = {
        'SEASONALITY': 'seasonality_score',
        'SHARPE': 'sharpe', 'SHARP': 'sharpe',
        'SORTINO': 'sortino',
        'CALMAR': 'calmar',
        'INFORMATION': 'information',
        'MER_ANN': 'mer_ann',
        'ONE_MINUS_MDD': 'one_minus_mdd',
        'CUM_RET': 'cum_ret',

        # Treynor aliases:
        'TREYNOR': 'treynor_adj',            # default to adjusted Treynor
        'TREYNOR_ADJ': 'treynor_adj',
        'ADJ_TREYNOR': 'treynor_adj',
        'TREYNOR_ABS': 'treynor_adj',
        'TREYNORABS': 'treynor_adj',
        'TREYNOR_RAW': 'treynor_raw',        # explicit raw if ever needed
    }

    selected_norm_cols = []
    for name in GPS_SCORE_COMPONENTS:
        key = str(name).strip().upper()
        base = alias.get(key)
        if not base or base not in mdf.columns:
            continue
        norm_col = base + "_01"
        mdf[norm_col] = minmax_01(mdf[base].values)
        if np.isfinite(mdf[norm_col].values).any():
            selected_norm_cols.append(norm_col)

    if not selected_norm_cols:
        return {1: 1}

    mdf['score'] = [gps_harmonic_01(mdf.loc[i, selected_norm_cols].values) for i in mdf.index]
    mdf['new_rank'] = mdf['score'].rank(ascending=False, method='first')
    inv = {}
    for prev_rank, row in mdf.iterrows():
        nr = int(row['new_rank'])
        if nr not in inv:
            inv[nr] = int(prev_rank)
    return inv

def monthly_top1_returns_gps(rankings: dict[pd.Timestamp, list[str]],
                             simple_rets_dict: dict[str, pd.Series],
                             z_scores_by_month: dict[pd.Timestamp, dict[str, float]],
                             bench_full_gross_long: pd.Series,
                             *,
                             direction: str,
                             start_dt: pd.Timestamp,
                             end_dt: pd.Timestamp,
                             calib_years: int,
                             rolling: bool,
                             entry_cost: float,
                             return_tickers: bool = False) -> pd.Series | tuple[pd.Series, pd.Series]:
    out_ret, out_tkr, idx = [], [], []
    for dt in pd.date_range(start_dt, end_dt, freq='MS'):
        order_today = rankings.get(dt)
        if not order_today: continue
        mapping = compute_gps_mapping_for_month(
            dt, rankings, simple_rets_dict, z_scores_by_month, bench_full_gross_long,
            direction=direction, calib_years=calib_years, rolling=rolling, entry_cost=entry_cost
        )
        prev_rank = mapping.get(1, 1)
        if prev_rank < 1 or prev_rank > len(order_today): continue
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

def df_to_string_centered(df: pd.DataFrame, index: bool = False) -> str:
    df_str = df.copy()
    if not index:
        df_str = df_str.reset_index(drop=True)
    df_str = df_str.astype(str)
    cols = list(df_str.columns)
    rows = df_str.values.tolist()
    widths = []
    for i, c in enumerate(cols):
        col_cells = [r[i] for r in rows] if rows else []
        widths.append(max(len(str(c)), *(len(x) for x in col_cells)) if col_cells else len(str(c)))
    header = "  ".join(str(c).center(widths[i]) for i, c in enumerate(cols))
    body_lines = []
    for r in rows:
        body_lines.append("  ".join(str(r[i]).center(widths[i]) for i in range(len(cols))))
    return header + "\n" + "\n".join(body_lines)

def build_monthly_top5_table(df_choices: pd.DataFrame, direction: str, n_runs: int, top_k: int = 5) -> pd.DataFrame:
    sub = df_choices[df_choices['direction'] == direction].copy()
    if sub.empty:
        return pd.DataFrame(columns=['Date'] + [f'B{i}' for i in range(1, top_k+1)] + [f'G{i}' for i in range(1, top_k+1)])
    sub['date'] = pd.to_datetime(sub['date'])
    dates = pd.date_range(FINAL_SIM_START, FINAL_SIM_END, freq='MS')
    rows = []
    for dt in dates:
        dsel = sub[sub['date'] == dt]
        base_counts = dsel['baseline_ticker'].value_counts()
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
# 8) METRICS & WINDOWS
# =============================== #
@dataclass
class Perf:
    cum_ret: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    mdd: float     # negative by construction
    stdev: float   # monthly stdev

def perf_from_monthly(returns: pd.Series) -> Perf:
    arr = returns.astype(float).values
    if arr.size == 0:
        return Perf(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    sr  = sharpe_ratio(arr)
    sor = sortino_ratio(arr)
    cal = calmar_ratio(arr)
    cum = float(np.prod(1.0 + arr) - 1.0)
    years = len(arr) / 12.0
    cagr = (np.prod(1.0 + arr) ** (1/years) - 1.0) if years > 0 else np.nan
    cum_curve = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(cum_curve)
    dd = cum_curve / peak - 1.0
    mdd = float(np.min(dd)) if dd.size else np.nan
    sd  = float(np.std(arr, ddof=1)) if arr.size >= 2 else np.nan
    return Perf(cum, cagr, sr, sor, cal, mdd, sd)

def three_fixed_windows() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    wins = []
    cur = pd.Timestamp(FINAL_SIM_START)
    for _ in range(3):
        w_end = cur + relativedelta(months=36) - pd.offsets.MonthEnd(1)
        wins.append((cur, w_end))
        cur = w_end + pd.offsets.MonthBegin(1)
    return wins

def compute_metrics_correlation(df_metrics: pd.DataFrame,
                                metric_cols: list[str]) -> pd.DataFrame:
    return df_metrics[metric_cols].astype(float).corr(method='pearson')

def save_correlation_csv(corr_df: pd.DataFrame, out_path: Path) -> None:
    corr_df.to_csv(out_path)

def save_correlation_heatmap(corr_df: pd.DataFrame, out_path_pdf: Path) -> None:
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

def paired_one_sided_both(left: np.ndarray, right: np.ndarray):
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
    if n < 3:
        print(f"    {right_label} vs {left_label} — {metric_name:<11}: insufficient valid pairs.")
        return
    better = f"{right_label} better at 5% two-sided" if (p_g is not None and p_g < PER_SIDE_ALPHA) else "no better effect at 5%"
    worse  = f"{right_label} worse at 5% two-sided"  if (p_l is not None and p_l < PER_SIDE_ALPHA) else "no worse effect at 5%"
    print(f"    {right_label} vs {left_label} — {metric_name:<11}: n={n:4d} | "
          f"better ({better_note}; α_one-sided={PER_SIDE_ALPHA:.3f}): t={t_g:.3f}, p={p_g:.4f} -> {better} | "
          f"worse: t={t_l:.3f}, p={p_l:.4f} -> {worse}")

def paired_one_sided_both_from_df(df_dir: pd.DataFrame, strat_left: str, strat_right: str, col: str):
    L = df_dir[df_dir['strategy'] == strat_left][['run', col]].rename(columns={col: 'L'})
    R = df_dir[df_dir['strategy'] == strat_right][['run', col]].rename(columns={col: 'R'})
    merged = pd.merge(L, R, on='run', how='inner').dropna()
    if merged.empty: return None, None, None, None, 0
    return paired_one_sided_both(merged['L'].values.astype(float), merged['R'].values.astype(float))

# =============================== #
# 9) MONTE CARLO: BOTH DIRECTIONS (shared long-only benchmark)
# =============================== #
def run_monte_carlo_both(n_runs=MC_RUNS, lam=LAMBDA_EWMA, save_series=SAVE_SERIES, zero_noise=ZERO_NOISE):
    OUT_DIR_MC.mkdir(parents=True, exist_ok=True)

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

    # metrics rows for CSV/corr
    metrics_rows: list[dict] = []
    choice_records = []

    with redirect_stdout(tee):
        rng = np.random.default_rng(RNG_SEED)
        t0 = time.time()
        section(f"Starting Monte Carlo — BOTH directions (common long-only benchmark) | runs={n_runs:,} | lambda={lam} | zero_noise={zero_noise}")

        sigma_proj = None if zero_noise else precompute_sigma_proj_for_all(log_rets, lam=lam)

        month_grid = pd.date_range(FINAL_SIM_START, FINAL_SIM_END, freq='MS')
        splits = three_fixed_windows()

        # Strategies per direction (no 'weighted' on SHORT)
        STRATS_BY_DIR = {'long': ('baseline','gps','weighted'), 'short': ('baseline','gps')}

        full_metrics = {
            'long':  {'baseline': [], 'gps': [], 'weighted': []},
            'short': {'baseline': [], 'gps': []}
        }
        sub_metrics = {
            'long':  {'baseline': defaultdict(list), 'gps': defaultdict(list), 'weighted': defaultdict(list)},
            'short': {'baseline': defaultdict(list), 'gps': defaultdict(list)}
        }
        sub_treynor_adj = {
            'long':  {'baseline': defaultdict(list), 'gps': defaultdict(list), 'weighted': defaultdict(list)},
            'short': {'baseline': defaultdict(list), 'gps': defaultdict(list)}
        }
        pos_counts = {'long': {'baseline':0,'gps':0,'weighted':0}, 'short': {'baseline':0,'gps':0}}
        neg_counts = {'long': {'baseline':0,'gps':0,'weighted':0}, 'short': {'baseline':0,'gps':0}}

        for run in range(1, n_runs + 1):
            # one simulation shared by both directions
            simple_sim = generate_simulated_simple_returns_for_run(rng, lam, zero_noise, sigma_proj)
            log_sim    = {t: np.log1p(s.astype(float)).rename(t) for t, s in simple_sim.items()}

            L_rank, S_rank, z_by_month = build_rankings_from_log_rets(log_sim, LOOKBACK_YEARS)

            # Build long-only EW benchmarks (gross long)
            bench_full_gross_long  = build_equal_weight_benchmark(simple_sim, TEST_SIM_START,  FINAL_END)
            bench_apply_gross_long = build_equal_weight_benchmark(simple_sim, FINAL_SIM_START, FINAL_SIM_END)

            # Common NET long-only benchmark (used for ALL Treynor/IR/MER)
            bench_apply_long_net = long_benchmark_net(bench_apply_gross_long, ENTRY_COST)

            # Also treat this series as the LONG "weighted" portfolio
            weighted_long_series = bench_apply_long_net.copy().rename("equal_weight_long_net")

            for direction, rank_use in (('long', L_rank), ('short', S_rank)):
                # Top-1 Baseline/GPS (NET) with direction-aware P&L
                top1_baseline, tkr_baseline = monthly_top1_returns(
                    rank_use, simple_sim, direction=direction,
                    start_dt=FINAL_SIM_START, end_dt=FINAL_SIM_END,
                    entry_cost=ENTRY_COST, return_tickers=True
                )
                top1_gps, tkr_gps = monthly_top1_returns_gps(
                    rank_use, simple_sim, z_by_month, bench_full_gross_long,
                    direction=direction,
                    start_dt=FINAL_SIM_START, end_dt=FINAL_SIM_END,
                    calib_years=GPS_CALIB_YEARS, rolling=GPS_ROLLING_ENABLED,
                    entry_cost=ENTRY_COST, return_tickers=True
                )

                # NAVs (Top-1 only) for display context
                nav_baseline = nav_from_returns_on_grid(top1_baseline, START_VALUE, month_grid)
                nav_gps      = nav_from_returns_on_grid(top1_gps,      START_VALUE, month_grid)

                # Record choices for Top-5 tables
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

                # Save series (per direction)
                if save_series:
                    ddir = OUT_DIR_MC / f"{direction}_run_{run:03d}"
                    ddir.mkdir(parents=True, exist_ok=True)
                    top1_baseline.to_csv(ddir / "top1_baseline_monthly.csv")
                    top1_gps.to_csv(ddir / "top1_gps_monthly.csv")
                    if direction == 'long':
                        weighted_long_series.to_csv(ddir / "weighted_equal_monthly.csv")  # only for long

                # Full-period metrics (port-only stats) and rows for CSV/corr (bench-relative pieces below)
                pb = perf_from_monthly(top1_baseline)
                pg = perf_from_monthly(top1_gps)
                full_metrics[direction]['baseline'].append(pb)
                full_metrics[direction]['gps'].append(pg)
                pos_counts[direction]['baseline'] += int(pb.cum_ret > 0)
                neg_counts[direction]['baseline'] += int(pb.cum_ret <= 0)
                pos_counts[direction]['gps']      += int(pg.cum_ret > 0)
                neg_counts[direction]['gps']      += int(pg.cum_ret <= 0)

                # Add 'weighted' only for LONG direction
                if direction == 'long':
                    pw = perf_from_monthly(weighted_long_series)
                    full_metrics['long']['weighted'].append(pw)
                    pos_counts['long']['weighted'] += int(pw.cum_ret > 0)
                    neg_counts['long']['weighted'] += int(pw.cum_ret <= 0)

                # ---- Rows for correlation / metrics CSV (use Adjusted Treynor for correlation; save raw too)
                metrics_rows.append({
                    'run': run, 'direction': direction, 'strategy': 'baseline',
                    'sharpe': pb.sharpe, 'sortino': pb.sortino,
                    'treynor_adj': treynor_ratio_series_absbeta(top1_baseline, bench_apply_long_net),
                    'treynor_raw': treynor_ratio_series(top1_baseline, bench_apply_long_net),
                    'information': information_ratio_series(top1_baseline, bench_apply_long_net),
                    'calmar': pb.calmar, 'cum_ret': pb.cum_ret,
                    'mdd_raw': pb.mdd if np.isfinite(pb.mdd) else np.nan,
                    'mdd_mag': abs(pb.mdd) if np.isfinite(pb.mdd) else np.nan,
                    'mer_m': mean_excess_return_series(top1_baseline, bench_apply_long_net),
                })
                metrics_rows.append({
                    'run': run, 'direction': direction, 'strategy': 'gps',
                    'sharpe': pg.sharpe, 'sortino': pg.sortino,
                    'treynor_adj': treynor_ratio_series_absbeta(top1_gps, bench_apply_long_net),
                    'treynor_raw': treynor_ratio_series(top1_gps, bench_apply_long_net),
                    'information': information_ratio_series(top1_gps, bench_apply_long_net),
                    'calmar': pg.calmar, 'cum_ret': pg.cum_ret,
                    'mdd_raw': pg.mdd if np.isfinite(pg.mdd) else np.nan,
                    'mdd_mag': abs(pg.mdd) if np.isfinite(pg.mdd) else np.nan,
                    'mer_m': mean_excess_return_series(top1_gps, bench_apply_long_net),
                })
                if direction == 'long':
                    metrics_rows.append({
                        'run': run, 'direction': 'long', 'strategy': 'weighted',
                        'sharpe': pw.sharpe, 'sortino': pw.sortino,
                        'treynor_adj': treynor_ratio_series_absbeta(weighted_long_series, bench_apply_long_net),
                        'treynor_raw': treynor_ratio_series(weighted_long_series, bench_apply_long_net),
                        'information': information_ratio_series(weighted_long_series, bench_apply_long_net),
                        'calmar': pw.calmar, 'cum_ret': pw.cum_ret,
                        'mdd_raw': pw.mdd if np.isfinite(pw.mdd) else np.nan,
                        'mdd_mag': abs(pw.mdd) if np.isfinite(pw.mdd) else np.nan,
                        'mer_m': mean_excess_return_series(weighted_long_series, bench_apply_long_net),
                    })

                # ---- Subperiods (three 3y windows) + Adjusted Treynor vs long-only benchmark
                for i, (s, e) in enumerate(splits, start=1):
                    sb = top1_baseline[(top1_baseline.index >= s) & (top1_baseline.index <= e)]
                    sg = top1_gps[(top1_gps.index >= s) & (top1_gps.index <= e)]
                    sub_metrics[direction]['baseline'][i].append(perf_from_monthly(sb))
                    sub_metrics[direction]['gps'][i].append(perf_from_monthly(sg))

                    bench_win = bench_apply_long_net[(bench_apply_long_net.index >= s) & (bench_apply_long_net.index <= e)]
                    sub_treynor_adj[direction]['baseline'][i].append(treynor_ratio_series_absbeta(sb, bench_win))
                    sub_treynor_adj[direction]['gps'][i].append(treynor_ratio_series_absbeta(sg, bench_win))

                    if direction == 'long':
                        sw = weighted_long_series[(weighted_long_series.index >= s) & (weighted_long_series.index <= e)]
                        sub_metrics['long']['weighted'][i].append(perf_from_monthly(sw))
                        sub_treynor_adj['long']['weighted'][i].append(treynor_ratio_series_absbeta(sw, bench_win))

            if (run % 10) == 0:
                elapsed = time.time() - t0
                print(f"  Progress: completed {run:>4}/{n_runs:<4} runs | elapsed {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

        # === Top-5 selection tables ===
        if SAVE_TICKER_CHOICES_CSV and choice_records:
            df_choices = pd.DataFrame(choice_records)
            section("Top-5 selection share tables (per month)")
            for direction in ('long', 'short'):
                tbl = build_monthly_top5_table(df_choices, direction, n_runs, top_k=5)
                out_path = OUT_DIR_MC / f"top5_selections_table_{direction}.csv"
                tbl.to_csv(out_path, index=False)
                print(f"Direction: {direction.upper()} — Top-5 per method (Baseline first 5 columns, GPS next 5)")
                print(df_to_string_centered(tbl.head(12), index=False) + "\n")
                print(f"Saved full table to: {out_path}\n")

        # === Safe stats helpers ===
        def _finite(x) -> np.ndarray:
            arr = np.asarray(x, dtype=float)
            return arr[np.isfinite(arr)]
        def safe_mean(x) -> float:
            v = _finite(x);  return float(v.mean()) if v.size > 0 else np.nan
        def safe_median(x) -> float:
            v = _finite(x);  return float(np.median(v)) if v.size > 0 else np.nan
        def safe_std(x, ddof=1) -> float:
            v = _finite(x);  return float(v.std(ddof=ddof)) if v.size > ddof else np.nan
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
            vals = sub['treynor_adj'].astype(float).values if not sub.empty else np.array([], dtype=float)
            return {'treynor_mean': safe_mean(vals), 'treynor_median': safe_median(vals), 'treynor_std': safe_std(vals, ddof=1)}

        df_metrics_all = pd.DataFrame(metrics_rows) if metrics_rows else None

        # === Print LONG then SHORT
        for direction in ('long', 'short'):
            n_runs_local = n_runs
            strats = STRATS_BY_DIR[direction]

            # Aggregates
            agg_full = {s: agg(full_metrics[direction][s]) for s in strats}
            trey     = {s: (treynor_stats_from_df(df_metrics_all, direction, s) if df_metrics_all is not None else {}) for s in strats}

            section(f"MONTE CARLO SUMMARY — Direction: {direction.upper()}")

            print(f"Config")
            print(f"  Lookback: {LOOKBACK_YEARS}y")
            print(f"  Test    : {TEST_SIM_START.date()} → {TEST_SIM_END.date()}  ({SIM_YEARS}y)")
            print(f"  Apply   : {FINAL_SIM_START.date()} → {FINAL_SIM_END.date()}")
            print(f"  SIG     : {SIG_LEVEL}   | GPS rolling={GPS_ROLLING_ENABLED}, calib={GPS_CALIB_YEARS}y   | Entry cost={ENTRY_COST:.4f}\n")

            # Outcome counts
            print("Outcome counts (Top-1 over apply)")
            hdr = f"  {'Strategy':<10} {'Pos(+)':>7} {'Neg(-)':>7}"
            print(hdr)
            print(f"  {'-'*10} {'-'*7:>7} {'-'*7:>7}")
            print(f"  {'Baseline':<10} {pos_counts[direction]['baseline']:>7} {neg_counts[direction]['baseline']:>7}")
            print(f"  {'GPS':<5}      {pos_counts[direction]['gps']:>7} {neg_counts[direction]['gps']:>7}")
            if 'weighted' in strats:
                print(f"  {'Weighted':<10} {pos_counts[direction]['weighted']:>7} {neg_counts[direction]['weighted']:>7}")
            print("")

            def pretty_full(d, t, name):
                print(f"{name} — Full Apply Period (across {n_runs_local} runs)")
                print(f"  CumRet      mean={d['cum_ret_mean']:.4f}   median={d['cum_ret_median']:.4f}   sd={d['cum_ret_std']:.4f}")
                print(f"  CAGR        mean={d['cagr_mean']:.4f}     median={d['cagr_median']:.4f}     sd={d['cagr_std']:.4f}")
                print(f"  Sharpe      mean={d['sharpe_mean']:.3f}   median={d['sharpe_median']:.3f}   sd={d['sharpe_std']:.3f}")
                print(f"  Sortino     mean={d['sortino_mean']:.3f}  median={d['sortino_median']:.3f}  sd={d['sortino_std']:.3f}")
                print(f"  Calmar      mean={d['calmar_mean']:.3f}   median={d['calmar_median']:.3f}   sd={d['calmar_std']:.3f}")
                print(f"  MaxDD       mean={d['mdd_mean']:.3f}      median={d['mdd_median']:.3f}      sd={d['mdd_std']:.3f}")
                print(f"  AdjTreynor  mean={t.get('treynor_mean', np.nan):.3f}   median={t.get('treynor_median', np.nan):.3f}   sd={t.get('treynor_std', np.nan):.3f}")
                print(f"  Vol(within-month) mean={d['stdev_mean']:.4f}   median={d['stdev_median']:.4f}   run-sd={d['stdev_std']:.4f}\n")

            pretty_full(agg_full['baseline'], trey['baseline'], "BASELINE")
            pretty_full(agg_full['gps'],      trey['gps'],      "GPS")
            if 'weighted' in strats:
                pretty_full(agg_full['weighted'], trey['weighted'], "WEIGHTED")

            print("Subperiods — Three fixed 3-year windows")
            for i, (s, e) in enumerate(splits, start=1):
                ab = agg(sub_metrics[direction]['baseline'][i])
                ag = agg(sub_metrics[direction]['gps'][i])
                tb_arr = np.array(sub_treynor_adj[direction]['baseline'][i], dtype=float)
                tg_arr = np.array(sub_treynor_adj[direction]['gps'][i], dtype=float)
                tb_mean = safe_mean(tb_arr); tg_mean = safe_mean(tg_arr)
                print(f"\n  Window {i}: {s.date()} → {e.date()}")
                print(f"    {'Metric':<14} {'Baseline':>12} {'GPS':>12}" + ("" if direction=='short' else " {'Weighted':>12}"))
                print(f"    {'-'*14:<14} {'-'*12:>12} {'-'*12:>12}" + ("" if direction=='short' else " {'-'*12:>12}"))
                def line2(label, f, g, fmt=".4f"):
                    def fmt1(x): return ("{:"+fmt+"}").format(float(x)) if np.isfinite(x) else "nan"
                    print(f"    {label:<14} {fmt1(f):>12} {fmt1(g):>12}")
                def line3(label, f, g, w, fmt=".4f"):
                    def fmt1(x): return ("{:"+fmt+"}").format(float(x)) if np.isfinite(x) else "nan"
                    print(f"    {label:<14} {fmt1(f):>12} {fmt1(g):>12} {fmt1(w):>12}")

                if direction == 'short':
                    line2("CumRet mean",  ab['cum_ret_mean'],  ag['cum_ret_mean'])
                    line2("CAGR mean",    ab['cagr_mean'],     ag['cagr_mean'])
                    line2("Sharpe mean",  ab['sharpe_mean'],   ag['sharpe_mean'], fmt=".3f")
                    line2("Calmar mean",  ab['calmar_mean'],   ag['calmar_mean'], fmt=".3f")
                    line2("MaxDD mean",   ab['mdd_mean'],      ag['mdd_mean'],    fmt=".3f")
                    print(f"    {'AdjTreynor mean':<14} {tb_mean:>12.3f} {tg_mean:>12.3f}")
                else:
                    aw = agg(sub_metrics['long']['weighted'][i])
                    tw_arr = np.array(sub_treynor_adj['long']['weighted'][i], dtype=float)
                    tw_mean = safe_mean(tw_arr)
                    line3("CumRet mean",  ab['cum_ret_mean'],  ag['cum_ret_mean'],  aw['cum_ret_mean'])
                    line3("CAGR mean",    ab['cagr_mean'],     ag['cagr_mean'],     aw['cagr_mean'])
                    line3("Sharpe mean",  ab['sharpe_mean'],   ag['sharpe_mean'],   aw['sharpe_mean'], fmt=".3f")
                    line3("Calmar mean",  ab['calmar_mean'],   ag['calmar_mean'],   aw['calmar_mean'], fmt=".3f")
                    line3("MaxDD mean",   ab['mdd_mean'],      ag['mdd_mean'],      aw['mdd_mean'],    fmt=".3f")
                    print(f"    {'AdjTreynor mean':<14} {tb_mean:>12.3f} {tg_mean:>12.3f} {tw_mean:>12.3f}")

            section("Significance — Paired, one-sided (per-side α = {:.3f}; overall two-sided 5%)".format(PER_SIDE_ALPHA))

            # Build arrays for paired tests
            b_list = full_metrics[direction]['baseline']
            g_list = full_metrics[direction]['gps']
            arr = lambda attr, L: np.array([getattr(p, attr) for p in L], dtype=float)

            def sig_pair_block(left_label, right_label, L_list, R_list, df_dir_metrics):
                print(f"[ {left_label.upper()}  →  {right_label.upper()} ]")

                # Port-only stats (same Perf structs you already built)
                for name, attr in [
                    ("Return", "cum_ret"),
                    ("CAGR", "cagr"),
                    ("Sharpe", "sharpe"),
                    ("Sortino", "sortino"),
                    ("Calmar", "calmar"),
                ]:
                    t_g, p_g, t_l, p_l, n = paired_one_sided_both(arr(attr, L_list), arr(attr, R_list))
                    print_sig_line_generic(name, left_label, right_label, t_g, p_g, t_l, p_l, n,
                                           better_note="higher is better")

                # ---- Benchmark-relative metrics pulled from df_metrics_all ----
                # Helper: extract per-run values for a given strategy/direction/column
                def _metric_by(strategy: str, direction_label: str, col: str) -> pd.DataFrame:
                    sub = df_metrics_all[
                        (df_metrics_all['strategy'] == strategy) &
                        (df_metrics_all['direction'] == direction_label)
                        ]
                    return sub[['run', col]].dropna().rename(columns={col: 'val'})

                # Which direction to use for each side in Treynor & 1−|MDD|
                dir_L = direction
                dir_R = 'long' if right_label.lower() == 'weighted' else direction

                # Adjusted Treynor (annualized, uses |beta|)
                L = _metric_by(left_label.lower(), dir_L, 'treynor_adj')
                R = _metric_by(right_label.lower(), dir_R, 'treynor_adj')
                merged = pd.merge(L, R, on='run', suffixes=('_L', '_R'))
                if not merged.empty:
                    t_g, p_g, t_l, p_l, n = paired_one_sided_both(merged['val_L'].values, merged['val_R'].values)
                    print_sig_line_generic("AdjTreynor", left_label, right_label, t_g, p_g, t_l, p_l, n,
                                           better_note="higher is better")
                else:
                    print_sig_line_generic("AdjTreynor", left_label, right_label, np.nan, np.nan, np.nan, np.nan, 0,
                                           better_note="higher is better")

                # MaxDD via (1 − |MDD|): higher = less severe drawdowns (consistent with GPS)
                L = _metric_by(left_label.lower(), dir_L, 'one_minus_mdd')
                R = _metric_by(right_label.lower(), dir_R, 'one_minus_mdd')
                merged = pd.merge(L, R, on='run', suffixes=('_L', '_R'))
                if not merged.empty:
                    t_g, p_g, t_l, p_l, n = paired_one_sided_both(merged['val_L'].values, merged['val_R'].values)
                    print_sig_line_generic("MaxDD (1−|MDD|)", left_label, right_label, t_g, p_g, t_l, p_l, n,
                                           better_note="less severe (higher)")
                else:
                    print_sig_line_generic("MaxDD (1−|MDD|)", left_label, right_label, np.nan, np.nan, np.nan, np.nan,
                                           0, better_note="less severe (higher)")

                print("")

            # Long-only extra comparisons vs weighted
            if direction == 'long':
                w_list = full_metrics['long']['weighted']
                sig_pair_block("baseline", "weighted", b_list, w_list, df_metrics_all[df_metrics_all['direction'] == 'long'])
                sig_pair_block("gps", "weighted", g_list, w_list, df_metrics_all[df_metrics_all['direction'] == 'long'])

            if direction == 'short':
                w_list = full_metrics['long']['weighted']  # use the long equal-weight benchmark
                sig_pair_block("baseline", "weighted", b_list, w_list, df_metrics_all)
                sig_pair_block("gps", "weighted", g_list, w_list, df_metrics_all)

            print(rule("─"))
            print("")

        total_elapsed = time.time() - t0
        section("Completed")
        print(f"Total runtime: {time.strftime('%H:%M:%S', time.gmtime(total_elapsed))}\n")

        # === Save metrics table + correlations ===
        if metrics_rows:
            df_metrics = pd.DataFrame(metrics_rows)
            df_metrics['one_minus_mdd'] = 1.0 - df_metrics['mdd_mag']
            df_metrics['mer_ann'] = 12.0 * df_metrics['mer_m']
            df_metrics.to_csv(OUT_DIR_MC / "performance_metrics_by_portfolio.csv", index=False)

            # ---- ADD THIS: Sharpe vs CumRet sanity by direction (and by strategy) ----
            metric_cols = ["sharpe", "sortino", "treynor_adj", "information",
                           "calmar", "cum_ret", "one_minus_mdd", "mer_ann"]

            # Per-direction Sharpe <-> CumRet
            for d in ["long", "short"]:
                sub = df_metrics[df_metrics["direction"] == d]
                if not sub.empty:
                    corr = sub[metric_cols].astype(float).corr()
                    if "sharpe" in corr.index and "cum_ret" in corr.columns:
                        val = corr.loc["sharpe", "cum_ret"]
                        print(f"[Sanity] Sharpe↔CumRet (direction={d}): {val:.3f}")
                        # Optional: save per-direction correlation matrix
                        corr.to_csv(OUT_DIR_MC / f"metrics_correlation_matrix_{d}.csv")

            # Per-direction & per-strategy (baseline/gps/weighted) — helpful to see mixing effects
            for d in ["long", "short"]:
                strategies = ["baseline", "gps", "weighted"] if d == "long" else ["baseline", "gps"]
                for s in strategies:
                    sub = df_metrics[(df_metrics["direction"] == d) & (df_metrics["strategy"] == s)]
                    if not sub.empty:
                        corr = sub[metric_cols].astype(float).corr()
                        if "sharpe" in corr.index and "cum_ret" in corr.columns:
                            val = corr.loc["sharpe", "cum_ret"]
                            print(f"[Sanity] Sharpe↔CumRet (direction={d}, strategy={s}): {val:.3f}")
                            # Optional: save per-direction/strategy correlation matrix
                            corr.to_csv(OUT_DIR_MC / f"metrics_correlation_matrix_{d}_{s}.csv")
            # ---- END ADD ----

            # Use adjusted Treynor in the correlation matrix (higher-is-better across directions)
            metric_cols = ["sharpe","sortino","treynor_adj","information",
                           "calmar","cum_ret","one_minus_mdd","mer_ann"]
            if GENERATE_METRICS_CORR_CSV or GENERATE_METRICS_CORR_HEATMAP:
                corr = compute_metrics_correlation(df_metrics, metric_cols)
                if GENERATE_METRICS_CORR_CSV:
                    save_correlation_csv(corr, OUT_DIR_MC / "metrics_correlation_matrix.csv")
                if GENERATE_METRICS_CORR_HEATMAP:
                    save_correlation_heatmap(corr, OUT_DIR_MC / "metrics_correlation_heatmap.pdf")

    with open(OUT_DIR_MC / "console_report.txt", "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

# =============================== #
# 10) RUN
# =============================== #
if __name__ == "__main__":
    run_monte_carlo_both(n_runs=MC_RUNS, lam=LAMBDA_EWMA, save_series=SAVE_SERIES, zero_noise=ZERO_NOISE)
