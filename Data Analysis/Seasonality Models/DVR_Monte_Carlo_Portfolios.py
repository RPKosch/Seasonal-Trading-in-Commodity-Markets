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
SIG_LEVEL               = 1                      # 1.0 ≡ no p-filter (GREEDY by Z); else significance-first

# GPS switches
GPS_ROLLING_ENABLED     = True                   # True=rolling 5y monthly re-calibration; False=fixed pre-apply
GPS_CALIB_YEARS         = SIM_YEARS

# Contract/IO (only monthly files used here)
ROOT_DIR                = Path().resolve().parent.parent / "Complete Data"

# Monte Carlo params
MC_RUNS                 = 5
LAMBDA_EWMA             = 0.94
BACKCAST_N              = 12
RNG_SEED                = 42
SAVE_SERIES             = False                  # save Top-1 monthly series per run
OUT_DIR_MC              = Path().resolve() / "Outputs" / "MC_Monthly"

# NEW switches
SAVE_TICKER_CHOICES_CSV = True                   # write a single CSV with chosen tickers per run & month
ZERO_NOISE              = True                   # if True, set z_t = 0 (no randomness)

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

def sharpe_ratio(returns: list[Decimal] | np.ndarray) -> float:
    arr = np.array([float(r) for r in returns])
    if arr.size == 0: return np.nan
    std = arr.std(ddof=1)
    if std == 0: return np.nan
    return arr.mean() / std * np.sqrt(12)

def sortino_ratio(returns: list[Decimal] | np.ndarray) -> float:
    arr = np.array([float(r) for r in returns])
    if arr.size == 0: return np.nan
    neg = arr[arr < 0]
    if neg.size == 0: return np.nan
    return arr.mean() / neg.std(ddof=1) * np.sqrt(12)

def calmar_ratio(returns: list[Decimal] | np.ndarray) -> float:
    arr = np.array([float(r) for r in returns])
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
# 7) TOP-1 MONTHLY SERIES (BASELINE & GPS)
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

def compute_gps_mapping_for_month(dt: pd.Timestamp,
                                  rankings: dict[pd.Timestamp, list[str]],
                                  simple_rets_dict: dict[str, pd.Series],
                                  z_scores_by_month: dict[pd.Timestamp, dict[str, float]],
                                  *,
                                  direction: str,
                                  calib_years: int,
                                  rolling: bool) -> dict[int, int]:
    """
    Build GPS new->prev rank mapping for month dt.
    Rolling: use [dt-5y, dt-1m]; Fixed: first 5y before FINAL_SIM_START.
    """
    if rolling:
        dt_minus = dt - relativedelta(years=calib_years)
        win_start = pd.Timestamp(datetime(dt_minus.year, dt_minus.month, 1))
        win_end   = dt - pd.offsets.MonthEnd(1)
    else:
        fixed_calib_start = pd.Timestamp(datetime(FINAL_SIM_START.year - calib_years, FINAL_SIM_START.month, 1))
        fixed_calib_end   = FINAL_SIM_START - pd.offsets.MonthEnd(1)
        win_start, win_end = fixed_calib_start, fixed_calib_end

    if win_end < win_start:
        return {1: 1}

    rets_nc = defaultdict(list)
    z_by_prev_rank = defaultdict(list)
    num_t = len(rankings.get(dt, []))

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
            rets_nc[pr].append(Decimal(str(r)))
            zval = zmap.get(tkr, np.nan)
            if direction == 'short':
                zval = -zval if np.isfinite(zval) else zval
            z_by_prev_rank[pr].append(float(zval) if np.isfinite(zval) else np.nan)

    rows = []
    for pr in range(1, num_t + 1):
        sr  = sharpe_ratio(rets_nc.get(pr, []))
        sor = sortino_ratio(rets_nc.get(pr, []))
        cal = calmar_ratio(rets_nc.get(pr, []))
        if float(SIG_LEVEL) >= 1.0:
            z_list = np.asarray(z_by_prev_rank.get(pr, []), dtype=float)
            seasonality_score = float(np.nanmean(z_list)) if np.isfinite(z_list).any() else np.nan
        else:
            seasonality_score = -float(pr)
        rows.append({'prev_rank': pr, 'seasonality_score': seasonality_score,
                     'sharpe': sr, 'sortino': sor, 'calmar': cal})
    mdf = pd.DataFrame(rows).set_index('prev_rank').sort_index()
    if mdf.empty:
        return {1: 1}
    for c in ['seasonality_score','sharpe','sortino','calmar']:
        mdf[f'{c}_01'] = minmax_01(mdf[c].values)
    norm_cols = [f'{c}_01' for c in ['seasonality_score','sharpe','sortino','calmar']]
    mdf['score'] = [gps_harmonic_01(mdf.loc[i, norm_cols].values) for i in mdf.index]
    mdf['new_rank'] = mdf['score'].rank(ascending=False, method='first')
    inv = {}
    for prev_rank, row in mdf.iterrows():
        nr = int(row['new_rank']); pr = int(prev_rank)
        if nr not in inv: inv[nr] = pr
    return inv

def monthly_top1_returns_gps(rankings: dict[pd.Timestamp, list[str]],
                             simple_rets_dict: dict[str, pd.Series],
                             z_scores_by_month: dict[pd.Timestamp, dict[str, float]],
                             *,
                             direction: str,
                             start_dt: pd.Timestamp,
                             end_dt: pd.Timestamp,
                             calib_years: int,
                             rolling: bool,
                             entry_cost: float,
                             return_tickers: bool = False) -> pd.Series | tuple[pd.Series, pd.Series]:
    """
    TOP-1 ONLY: maps GPS new_rank=1 to a prev_rank, takes that single ticker's simple return.
    If return_tickers=True, also returns a Series of the chosen ticker strings (always recorded),
    while returns are NaN when the chosen ticker has no return for that month.
    """
    out_ret, out_tkr, idx = [], [], []
    for dt in pd.date_range(start_dt, end_dt, freq='MS'):
        order_today = rankings.get(dt)
        if not order_today:
            continue
        mapping = compute_gps_mapping_for_month(
            dt, rankings, simple_rets_dict, z_scores_by_month,
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

def max_drawdown_from_monthly(returns: pd.Series) -> float:
    if returns.empty: return np.nan
    cum = np.cumprod(1.0 + returns.values)
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    return float(np.min(dd))  # negative

def perf_from_monthly(returns: pd.Series) -> Perf:
    arr = returns.astype(float).values
    if arr.size == 0:
        return Perf(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    sr  = sharpe_ratio(list(arr))
    sor = sortino_ratio(list(arr))
    cal = calmar_ratio(list(arr))
    cum = float(np.prod(1.0 + arr) - 1.0)
    cg  = cagr_from_monthly(returns)
    mdd = max_drawdown_from_monthly(returns)
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
# 9) MONTE CARLO: BOTH DIRECTIONS (efficient)
# =============================== #
def run_monte_carlo_both(n_runs=MC_RUNS, lam=LAMBDA_EWMA, save_series=SAVE_SERIES, zero_noise=ZERO_NOISE):
    OUT_DIR_MC.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    t0 = time.time()
    print(f"Starting Monte Carlo (BOTH directions): runs={n_runs}, lambda={lam}, zero_noise={zero_noise}")

    # Precompute sigma projections once (used only if zero_noise=False)
    sigma_proj = None if zero_noise else precompute_sigma_proj_for_all(log_rets, lam=lam)

    # Static grids
    month_grid = pd.date_range(FINAL_SIM_START, FINAL_SIM_END, freq='MS')
    splits = three_fixed_windows()

    # Aggregators per direction
    dirs = ('long', 'short')
    full_metrics = {d: {'baseline': [], 'gps': []} for d in dirs}
    sub_metrics = {d: {'baseline': defaultdict(list), 'gps': defaultdict(list)} for d in dirs}
    pos_counts = {d: {'baseline': 0, 'gps': 0} for d in dirs}
    neg_counts = {d: {'baseline': 0, 'gps': 0} for d in dirs}
    choice_records = []  # unified CSV with 'direction' column

    # If zero_noise, the simulated logs are just base logs; compute simple once
    simple_base_if_zero = log_to_simple_dict(log_rets) if zero_noise else None

    for run in range(1, n_runs + 1):
        # 1) simulate (or reuse base) ONCE per run
        log_sim = simulate_log_returns_with_sigma(log_rets, sigma_proj, lam=lam, rng=rng, no_noise=zero_noise)
        simple_sim = simple_base_if_zero if simple_base_if_zero is not None else log_to_simple_dict(log_sim)

        # 2) build rankings/z ONCE per run
        L_rank, S_rank, z_by_month = build_rankings_from_log_rets(log_sim, LOOKBACK_YEARS)

        # 3) iterate BOTH directions using shared computations
        for direction, rank_use in (('long', L_rank), ('short', S_rank)):
            # Baseline + GPS top-1 series
            top1_baseline, tkr_baseline = monthly_top1_returns(
                rank_use, simple_sim, direction=direction,
                start_dt=FINAL_SIM_START, end_dt=FINAL_SIM_END,
                entry_cost=ENTRY_COST, return_tickers=True
            )
            top1_gps, tkr_gps = monthly_top1_returns_gps(
                rank_use, simple_sim, z_by_month, direction=direction,
                start_dt=FINAL_SIM_START, end_dt=FINAL_SIM_END,
                calib_years=GPS_CALIB_YEARS, rolling=GPS_ROLLING_ENABLED,
                entry_cost=ENTRY_COST, return_tickers=True
            )

            # NAV series (for CSV context)
            nav_baseline = nav_from_returns_on_grid(top1_baseline, START_VALUE, month_grid)
            nav_gps      = nav_from_returns_on_grid(top1_gps,      START_VALUE, month_grid)

            # Save choices in unified long-form
            if SAVE_TICKER_CHOICES_CSV:
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

            # Full-period metrics
            pb, pg = perf_from_monthly(top1_baseline), perf_from_monthly(top1_gps)
            full_metrics[direction]['baseline'].append(pb)
            full_metrics[direction]['gps'].append(pg)
            pos_counts[direction]['baseline'] += int(pb.cum_ret > 0)
            neg_counts[direction]['baseline'] += int(pb.cum_ret <= 0)
            pos_counts[direction]['gps']      += int(pg.cum_ret > 0)
            neg_counts[direction]['gps']      += int(pg.cum_ret <= 0)

            # Subperiod metrics
            for i, (s, e) in enumerate(splits, start=1):
                sb = top1_baseline[(top1_baseline.index >= s) & (top1_baseline.index <= e)]
                sg = top1_gps[(top1_gps.index >= s) & (top1_gps.index <= e)]
                sub_metrics[direction]['baseline'][i].append(perf_from_monthly(sb))
                sub_metrics[direction]['gps'][i].append(perf_from_monthly(sg))

        # progress
        if (run % 10) == 0:
            elapsed = time.time() - t0
            print(f"  Completed {run}/{n_runs} runs | elapsed {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

    # Write unified choices CSV
    if SAVE_TICKER_CHOICES_CSV and choice_records:
        df_choices = pd.DataFrame(choice_records)
        df_choices.to_csv(OUT_DIR_MC / "top1_ticker_choices_all_runs.csv", index=False)

    # === Reporting helpers ===
    def agg(perf_list: list[Perf]) -> dict[str, float]:
        out = {}
        for k in ['cum_ret','cagr','sharpe','sortino','calmar','mdd','stdev']:
            x = np.array([getattr(p, k) for p in perf_list], float)
            out[f'{k}_mean']   = float(np.nanmean(x))
            out[f'{k}_median'] = float(np.nanmedian(x))
            out[f'{k}_std']    = float(np.nanstd(x, ddof=1))
        return out

    def paired_test(b_list: list[Perf], g_list: list[Perf], attr: str):
        b = np.array([getattr(p, attr) for p in b_list], float)
        g = np.array([getattr(p, attr) for p in g_list], float)
        mask = np.isfinite(b) & np.isfinite(g)
        if mask.sum() < 3:
            return None, None, 0
        t, p = st.ttest_rel(g[mask], b[mask], alternative='greater')
        return float(t), float(p), int(mask.sum())

    def pretty_full(d, name):
        print(f"{name} — FULL PERIOD across runs (Top-1 only):")
        print(f"  CumRet   mean={d['cum_ret_mean']:.4f}  median={d['cum_ret_median']:.4f}  sd={d['cum_ret_std']:.4f}")
        print(f"  CAGR     mean={d['cagr_mean']:.4f}    median={d['cagr_median']:.4f}    sd={d['cagr_std']:.4f}")
        print(f"  Sharpe   mean={d['sharpe_mean']:.3f}  median={d['sharpe_median']:.3f}  sd={d['sharpe_std']:.3f}")
        print(f"  Sortino  mean={d['sortino_mean']:.3f} median={d['sortino_median']:.3f} sd={d['sortino_std']:.3f}")
        print(f"  Calmar   mean={d['calmar_mean']:.3f}  median={d['calmar_median']:.3f}  sd={d['calmar_std']:.3f}")
        print(f"  MDD      mean={d['mdd_mean']:.3f}     median={d['mdd_median']:.3f}     sd={d['mdd_std']:.3f}")
        print(f"  Vol(within) mean={d['stdev_mean']:.4f}   median={d['stdev_median']:.4f}   RunSD={d['stdev_std']:.4f}")

    # === Print LONG first, divider, then SHORT ===
    for direction in ('long', 'short'):
        n_runs_local = n_runs
        agg_full_b = agg(full_metrics[direction]['baseline'])
        agg_full_g = agg(full_metrics[direction]['gps'])

        # add Sharpe to the significance battery
        t_ret,    p_ret,    n_ret    = paired_test(full_metrics[direction]['baseline'], full_metrics[direction]['gps'], 'cum_ret')
        t_cagr,   p_cagr,   n_cagr   = paired_test(full_metrics[direction]['baseline'], full_metrics[direction]['gps'], 'cagr')
        t_sharpe, p_sharpe, n_sharpe = paired_test(full_metrics[direction]['baseline'], full_metrics[direction]['gps'], 'sharpe')
        t_sort,   p_sort,   n_sort   = paired_test(full_metrics[direction]['baseline'], full_metrics[direction]['gps'], 'sortino')
        t_calmar, p_calmar, n_calmar = paired_test(full_metrics[direction]['baseline'], full_metrics[direction]['gps'], 'calmar')

        print("\n" + "="*96)
        print(f"MONTE CARLO — MONTHLY (N={n_runs_local}, λ={lam}, backcast={BACKCAST_N}, dir={direction}, zero_noise={zero_noise})")
        print("-"*96)
        print(f"Lookback={LOOKBACK_YEARS}y | Test={SIM_YEARS}y ({TEST_SIM_START.date()}→{TEST_SIM_END.date()}) | "
              f"Apply={FINAL_SIM_START.date()}→{FINAL_SIM_END.date()} | SIG_LEVEL={SIG_LEVEL}")
        print(f"GPS: rolling={GPS_ROLLING_ENABLED}, calib_window={GPS_CALIB_YEARS}y | Entry cost per month={ENTRY_COST}")
        print("-"*96)
        print("Final outcome counts over apply period (Top-1 only):")
        print(f"  Baseline : +{pos_counts[direction]['baseline']:3d}   -{neg_counts[direction]['baseline']:3d}")
        print(f"  GPS      : +{pos_counts[direction]['gps']:3d}   -{neg_counts[direction]['gps']:3d}")
        print("-"*96)

        pretty_full(agg_full_b, "BASELINE")
        pretty_full(agg_full_g, "GPS")

        print("-"*96)
        print("Three fixed 3-year subperiod statistics across runs (Top-1 only):")
        for i, (s, e) in enumerate(splits, start=1):
            ab = agg(sub_metrics[direction]['baseline'][i])
            ag = agg(sub_metrics[direction]['gps'][i])
            print(f"  Window {i}: {s.date()} → {e.date()}")
            # Ret stats & across-run variability; within-window monthly vol; and now SDs for CAGR/Sharpe/Calmar.
            print(f"    Baseline: Ret(mean)={ab['cum_ret_mean']:.4f}  Ret(median)={ab['cum_ret_median']:.4f}  "
                  f"RunSD={ab['cum_ret_std']:.4f}  Vol(within)={ab['stdev_mean']:.4f}  "
                  f"CAGR(mean)={ab['cagr_mean']:.4f} (sd={ab['cagr_std']:.4f})  "
                  f"Sharpe(mean)={ab['sharpe_mean']:.3f} (sd={ab['sharpe_std']:.3f})  "
                  f"Calmar(mean)={ab['calmar_mean']:.3f} (sd={ab['calmar_std']:.3f})")
            print(f"    GPS     : Ret(mean)={ag['cum_ret_mean']:.4f}  Ret(median)={ag['cum_ret_median']:.4f}  "
                  f"RunSD={ag['cum_ret_std']:.4f}  Vol(within)={ag['stdev_mean']:.4f}  "
                  f"CAGR(mean)={ag['cagr_mean']:.4f} (sd={ag['cagr_std']:.4f})  "
                  f"Sharpe(mean)={ag['sharpe_mean']:.3f} (sd={ag['sharpe_std']:.3f})  "
                  f"Calmar(mean)={ag['calmar_mean']:.3f} (sd={ag['calmar_std']:.3f})")

        print("-"*96)
        print("Significance (paired, one-sided H1: GPS > Baseline) — full apply period, Top-1 only:")
        if n_ret >= 3:
            print(f"  Return   : t={t_ret:.3f},   p={p_ret:.4f}   -> {'GPS better at 5%' if p_ret is not None and p_ret < 0.05 else 'no sig diff at 5%'}")
        else:
            print("  Return   : insufficient valid pairs.")
        if n_cagr >= 3:
            print(f"  CAGR     : t={t_cagr:.3f},  p={p_cagr:.4f}  -> {'GPS better at 5%' if p_cagr is not None and p_cagr < 0.05 else 'no sig diff at 5%'}")
        else:
            print("  CAGR     : insufficient valid pairs.")
        if n_sharpe >= 3:
            print(f"  Sharpe   : t={t_sharpe:.3f},  p={p_sharpe:.4f}  -> {'GPS better at 5%' if p_sharpe is not None and p_sharpe < 0.05 else 'no sig diff at 5%'}")
        else:
            print("  Sharpe   : insufficient valid pairs.")
        if n_sort >= 3:
            print(f"  Sortino  : t={t_sort:.3f},  p={p_sort:.4f}  -> {'GPS better at 5%' if p_sort is not None and p_sort < 0.05 else 'no sig diff at 5%'}")
        else:
            print("  Sortino  : insufficient valid pairs.")
        if n_calmar >= 3:
            print(f"  Calmar   : t={t_calmar:.3f}, p={p_calmar:.4f} -> {'GPS better at 5%' if p_calmar is not None and p_calmar < 0.05 else 'no sig diff at 5%'}")
        else:
            print("  Calmar   : insufficient valid pairs.")

        # Divider after LONG block
        if direction == 'long':
            print("")
            print("#" * 96)

    total_elapsed = time.time() - t0
    print(f"Total runtime (both directions): {time.strftime('%H:%M:%S', time.gmtime(total_elapsed))}")
    print("="*96 + "\n")

# =============================== #
# 10) RUN — single efficient call
# =============================== #
if __name__ == "__main__":
    run_monte_carlo_both(n_runs=MC_RUNS, lam=LAMBDA_EWMA, save_series=SAVE_SERIES, zero_noise=ZERO_NOISE)
