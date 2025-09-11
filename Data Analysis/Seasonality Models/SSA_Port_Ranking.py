import numpy as np
import pandas as pd
import statsmodels.api as sm  # not used now (kept only if you reuse DVR elsewhere)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from decimal import Decimal

# =============================== #
# 1) PARAMETERS & WINDOWS
# =============================== #
START_DATE              = datetime(2001, 1, 1)
FINAL_END               = datetime(2024, 12, 31)

LOOKBACK_YEARS          = 10
TEST_YEARS              = 5          # printed diagnostics only
SIM_YEARS               = 5          # printed diagnostics only

START_VALUE             = 1000.0
ENTRY_COST              = 0.0025     # apply EVERY month in apply period

# SSA scoring params
SSA_WINDOW              = 12          # embedding dimension L
SSA_COMPS               = 2           # leading components to keep

# Optional EWMA vol scaling of SSA score (score / sigma)
USE_EWMA_SCALE          = False
EWMA_LAMBDA             = 0.94        # alpha = 1 - lambda
MIN_OBS_FOR_VOL         = 12

# GPS switches
GPS_ROLLING_ENABLED     = True        # True = rolling 5y monthly re-calibration; False = fixed first 5y before FINAL_SIM_START
GPS_CALIB_YEARS         = 5

# Plot window (clip to what you want to see)
PLOT_START              = datetime(2016, 1, 1)
PLOT_END                = datetime(2024, 12, 31)

DEBUG_DATE              = None

# =============================== #
# 2) DATE RANGES
# =============================== #
LOOKBACK_END    = (START_DATE + relativedelta(years=LOOKBACK_YEARS) - pd.offsets.MonthEnd(1))
TEST_SIM_START  = START_DATE + relativedelta(years=LOOKBACK_YEARS)
TEST_SIM_END    = TEST_SIM_START + relativedelta(years=SIM_YEARS) - pd.offsets.MonthEnd(1)
FINAL_SIM_START = START_DATE + relativedelta(years=LOOKBACK_YEARS) + relativedelta(years=SIM_YEARS)
FINAL_SIM_END   = FINAL_END

print(f"Lookback: {START_DATE.date()} → {LOOKBACK_END.date()}")
print(f"Testing:  {TEST_SIM_START.date()} → {TEST_SIM_END.date()}")
print(f"Apply:    {FINAL_SIM_START.date()} → {FINAL_SIM_END.date()}")

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

def sharpe_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    if arr.size == 0: return np.nan
    std = arr.std(ddof=1)
    if std == 0: return np.nan
    return arr.mean() / std * np.sqrt(12)

def sortino_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    if arr.size == 0: return np.nan
    neg = arr[arr < 0]
    if neg.size == 0: return np.nan
    return arr.mean() / neg.std(ddof=1) * np.sqrt(12)

def calmar_ratio(returns: list[Decimal]) -> float:
    arr = np.array([float(r) for r in returns])
    if arr.size == 0: return np.nan
    cum = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1
    mdd = abs(dd.min())
    years = len(arr) / 12
    if mdd == 0 or years == 0: return np.nan
    cagr = cum[-1] ** (1 / years) - 1
    return cagr / mdd

# ---------- GPS helpers ----------
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

def build_metrics(cum_df: pd.DataFrame, rets_dict: dict[int, list[Decimal]]) -> pd.DataFrame:
    """
    GPS score across ranks based on cum_ret, sharpe, sortino, calmar (NO-COST for metrics).
    """
    rows = []
    for prev_rank, cum_row in cum_df.iterrows():
        rows.append({
            'prev_rank':  prev_rank,
            'cum_ret':    float(cum_row['cum_ret']),
            'cum_ret_wc': float(cum_row['cum_ret_wc']),
            'sharpe':     sharpe_ratio(rets_dict.get(prev_rank, [])),
            'sortino':    sortino_ratio(rets_dict.get(prev_rank, [])),
            'calmar':     calmar_ratio(rets_dict.get(prev_rank, [])),
        })
    df = pd.DataFrame(rows).set_index('prev_rank').sort_index()
    base_cols = ['cum_ret', 'sharpe', 'sortino', 'calmar']
    for c in base_cols:
        df[f'{c}_01'] = minmax_01(df[c].values)
    norm_cols = [f'{c}_01' for c in base_cols]
    df['score'] = [gps_harmonic_01(df.loc[idx, norm_cols].values) for idx in df.index]
    df['new_rank'] = df['score'].rank(ascending=False, method='first')
    df['rank_change'] = df.index - df['new_rank']
    return df

# ---------- SSA core ----------
def compute_ssa_forecast(x: np.ndarray, L: int, r: int) -> float:
    """
    Basic SSA one-step-ahead forecast.
    Returns forecast or NaN if not feasible.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    if N < max(L, 3) or L <= 1 or L >= N or r < 1:
        return np.nan
    if np.isnan(x).any():
        return np.nan

    K = N - L + 1
    # Trajectory (Hankel) matrix, shape L x K
    X = np.column_stack([x[i:i+L] for i in range(K)])

    # Covariance S = X X^T (L x L)
    S = X @ X.T
    eigvals, eigvecs = np.linalg.eigh(S)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eps = 1e-12
    pos = eigvals > eps
    if not np.any(pos):
        return np.nan
    eigvals = eigvals[pos]
    eigvecs = eigvecs[:, pos]

    r_eff = int(min(r, eigvals.size))
    U = eigvecs[:, :r_eff]                     # L x r
    sigma = np.sqrt(eigvals[:r_eff])           # r
    V = (X.T @ U) / sigma                      # K x r

    # Rank-r reconstruction
    Xr = (U * sigma) @ V.T                     # L x K

    # Diagonal averaging (Hankelization)
    rec = np.zeros(N)
    cnt = np.zeros(N)
    for i in range(L):
        for j in range(K):
            rec[i + j] += Xr[i, j]
            cnt[i + j] += 1.0
    if not np.all(cnt > 0):
        return np.nan
    rec /= cnt

    # Forecast coefficients (minimum-norm linear recurrence)
    P_head = U[:-1, :]     # (L-1) x r
    pi     = U[-1, :]      # r
    nu2    = float(np.dot(pi, pi))
    if 1.0 - nu2 <= 1e-10:
        return np.nan
    R = (P_head @ pi) / (1.0 - nu2)  # (L-1)
    a = R[::-1]                      # flip for convolution

    lags = rec[-1: -L: -1]           # last L-1 values (from end), size L-1
    if lags.size != a.size:
        return np.nan
    return float(np.dot(a, lags))

# =============================== #
# 4) LOAD DATA
# =============================== #
base = Path().resolve().parent.parent / "Complete Data"
# SSA is typically fit on log returns; P&L on simple returns
log_rets    = load_returns(base / "All_Monthly_Log_Return_Data")
simple_rets = load_returns(base / "All_Monthly_Return_Data")
tickers     = list(log_rets)
NUM_T       = len(tickers)

# =============================== #
# 5) SSA-BASED RANKINGS BY MONTH
# =============================== #
long_rankings:  dict[pd.Timestamp, list[str]] = {}
short_rankings: dict[pd.Timestamp, list[str]] = {}

cur = TEST_SIM_START
while cur <= FINAL_END:
    stats = []
    lb0 = cur - relativedelta(years=LOOKBACK_YEARS)
    lb1 = cur - relativedelta(months=1)

    for t in tickers:
        s_log = log_rets[t].loc[lb0:lb1].dropna()
        if len(s_log) < max(SSA_WINDOW, MIN_OBS_FOR_VOL):
            continue

        # SSA forecast on log returns
        ssa_val = compute_ssa_forecast(s_log.values, L=SSA_WINDOW, r=SSA_COMPS)

        # Optional EWMA vol scaling on simple returns
        score = ssa_val
        if USE_EWMA_SCALE:
            s_simple = simple_rets[t].loc[lb0:lb1].dropna()
            if len(s_simple) >= MIN_OBS_FOR_VOL and np.isfinite(ssa_val):
                alpha = 1.0 - EWMA_LAMBDA
                ewma_var = (s_simple**2).ewm(alpha=alpha, adjust=False).mean().iloc[-1]
                sigma = float(np.sqrt(ewma_var)) if np.isfinite(ewma_var) else np.nan
                if sigma and np.isfinite(sigma) and sigma > 0:
                    score = ssa_val / sigma
                else:
                    score = np.nan

        if np.isfinite(score):
            stats.append({'ticker': t, 'ssa': ssa_val, 'score': score})

    if not stats:
        # fallback: keep original list
        long_rankings[cur]  = tickers.copy()
        short_rankings[cur] = tickers.copy()
    else:
        dfm = pd.DataFrame(stats).set_index('ticker')

        # LONG: score > 0 (descending), then rest by descending score
        pos  = dfm[dfm['score'] > 0].sort_values('score', ascending=False).index.tolist()
        rest = dfm.drop(pos, errors='ignore').sort_values('score', ascending=False).index.tolist()
        orderL = pos + rest
        orderL += [t for t in tickers if t not in orderL]
        long_rankings[cur] = orderL[:len(tickers)]

        # SHORT: score < 0 (ascending, i.e., most negative first), then rest by ascending
        neg  = dfm[dfm['score'] < 0].sort_values('score', ascending=True).index.tolist()
        restS= dfm.drop(neg, errors='ignore').sort_values('score', ascending=True).index.tolist()
        orderS = neg + restS
        orderS += [t for t in tickers if t not in orderS]
        short_rankings[cur] = orderS[:len(tickers)]

    cur += relativedelta(months=1)

# =============================== #
# 6) BASELINE (SSA-ONLY) NAV PATHS
# =============================== #
def simulate_baseline_nav_paths(rankings: dict[pd.Timestamp, list[str]],
                                direction: str,
                                start_dt: pd.Timestamp,
                                end_dt: pd.Timestamp,
                                entry_cost: float) -> pd.DataFrame:
    """
    Returns monthly WITH-COST NAV paths for portfolios 1..NUM_T.
    Columns: portfolio_1 ... portfolio_NUM_T ; index = month end + initial start point.
    """
    nav = {k: Decimal(str(START_VALUE)) for k in range(1, NUM_T + 1)}
    rows = []

    # initial start point (shows 1000 before first compounding)
    start_row = {'date': pd.Timestamp(start_dt)}
    for k in range(1, NUM_T + 1):
        start_row[f'portfolio_{k}'] = float(nav[k])
    rows.append(start_row)

    for dt in pd.date_range(start_dt, end_dt, freq='MS'):
        order = rankings.get(dt)
        if order is None:
            continue
        row = {'date': (dt + pd.offsets.MonthEnd(0))}
        for k in range(1, NUM_T + 1):
            if k > len(order):
                continue
            t = order[k-1]
            r = Decimal(str(simple_rets[t].get(dt, 0.0)))
            if direction == 'short':
                r = Decimal(1) / (Decimal(1) + r) - Decimal(1)
            # monthly entry cost always
            r_wc = (Decimal(1) - Decimal(str(entry_cost))) * (Decimal(1) + r) - Decimal(1)
            nav[k] *= (Decimal(1) + r_wc)
            row[f'portfolio_{k}'] = float(nav[k])
        rows.append(row)

    df = pd.DataFrame(rows).set_index('date').sort_index()
    # ensure all columns exist
    for k in range(1, NUM_T + 1):
        col = f'portfolio_{k}'
        if col not in df.columns:
            df[col] = np.nan
    return df

# =============================== #
# 7) GPS-MAPPED NAV PATHS (ROLLING/FIXED, still using SSA-based prev-ranks)
# =============================== #
def compute_cum(rankings: dict[pd.Timestamp, list[str]],
                direction: str,
                start_date: pd.Timestamp,
                end_date: pd.Timestamp,
                entry_cost: float = 0.0):
    """Used only inside calibration to score previous ranks (NO-COST for metrics)."""
    rets_nc: dict[int, list[Decimal]] = {k: [] for k in range(1, NUM_T + 1)}
    rets_wc: dict[int, list[Decimal]] = {k: [] for k in range(1, NUM_T + 1)}

    for dt in pd.date_range(start_date, end_date, freq='MS'):
        order = rankings.get(dt)
        if order is None: continue
        for r, t in enumerate(order, start=1):
            if r > NUM_T: break
            raw = Decimal(str(simple_rets[t].get(dt, 0.0)))
            if direction == 'short':
                raw = Decimal(1) / (Decimal(1) + raw) - Decimal(1)
            rets_nc[r].append(raw)
            r_wc = (Decimal(1) - Decimal(str(entry_cost))) * (Decimal(1) + raw) - Decimal(1) if entry_cost > 0 else raw
            rets_wc[r].append(r_wc)

    rows = []
    START_D = Decimal(str(START_VALUE))
    for r in range(1, NUM_T + 1):
        vc_nc = START_D; vc_wc = START_D
        for x_nc, x_wc in zip(rets_nc[r], rets_wc[r]):
            vc_nc *= (Decimal(1) + x_nc)
            vc_wc *= (Decimal(1) + x_wc)
        rows.append({'rank': float(r),
                     'cum_ret':    vc_nc / START_D - Decimal(1),
                     'cum_ret_wc': vc_wc / START_D - Decimal(1)})
    cum_df = pd.DataFrame(rows).set_index('rank')
    return cum_df, rets_nc

def invert_prev_to_new(metrics_df: pd.DataFrame) -> dict[int, int]:
    inv = {}
    for prev_rank, row in metrics_df.iterrows():
        nr = int(row['new_rank']); pr = int(prev_rank)
        if nr not in inv: inv[nr] = pr
    return inv

def simulate_gps_nav_paths(rankings: dict[pd.Timestamp, list[str]],
                           direction: str,
                           apply_start: pd.Timestamp,
                           apply_end: pd.Timestamp,
                           calib_years: int,
                           rolling: bool,
                           entry_cost: float) -> pd.DataFrame:
    """
    Returns monthly WITH-COST NAV paths for portfolios 1..NUM_T using GPS monthly mapping.
    Includes initial start point at apply_start = 1000.
    """
    nav = {k: Decimal(str(START_VALUE)) for k in range(1, NUM_T + 1)}
    rows = []

    # initial start point
    start_row = {'date': pd.Timestamp(apply_start)}
    for k in range(1, NUM_T + 1):
        start_row[f'portfolio_{k}'] = float(nav[k])
    rows.append(start_row)

    if not rolling:
        fixed_calib_start = pd.Timestamp(datetime(apply_start.year - calib_years, 1, 1))
        fixed_calib_end   = apply_start - pd.offsets.MonthEnd(1)

    for dt in pd.date_range(apply_start, apply_end, freq='MS'):
        order_today = rankings.get(dt)
        if order_today is None:
            continue

        # Calibration window
        if rolling:
            dt_minus = dt - relativedelta(years=calib_years)
            win_start = pd.Timestamp(datetime(dt_minus.year, dt_minus.month, 1))
            win_end   = dt - pd.offsets.MonthEnd(1)
        else:
            win_start = fixed_calib_start
            win_end   = fixed_calib_end
        if win_end < win_start:
            continue

        # Calibrate on prev-ranks (NO cost for metrics)
        cum_df_calib, rets_nc_calib = compute_cum(
            rankings, direction=direction, start_date=win_start, end_date=win_end, entry_cost=0.0
        )
        metrics_df = build_metrics(cum_df_calib, rets_nc_calib)
        map_new_to_prev = invert_prev_to_new(metrics_df)

        # One month invest per portfolio using mapped prev-rank -> today's ticker
        row = {'date': (dt + pd.offsets.MonthEnd(0))}
        for new_rank in range(1, NUM_T + 1):
            prev_rank = map_new_to_prev.get(new_rank, new_rank)
            if prev_rank < 1 or prev_rank > len(order_today):
                continue
            t = order_today[prev_rank - 1]
            r = Decimal(str(simple_rets[t].get(dt, 0.0)))
            if direction == 'short':
                r = Decimal(1) / (Decimal(1) + r) - Decimal(1)
            # monthly entry cost
            r_wc = (Decimal(1) - Decimal(str(entry_cost))) * (Decimal(1) + r) - Decimal(1)
            nav[new_rank] *= (Decimal(1) + r_wc)
            row[f'portfolio_{new_rank}'] = float(nav[new_rank])
        rows.append(row)

    df = pd.DataFrame(rows).set_index('date').sort_index()
    for k in range(1, NUM_T + 1):
        col = f'portfolio_{k}'
        if col not in df.columns:
            df[col] = np.nan
    return df

# =============================== #
# 8) BUILD NAVs & PLOTS
# =============================== #
print("-" * 100)
print(f"Apply Period: {FINAL_SIM_START.date()} → {FINAL_SIM_END.date()}")
print(f"GPS monthly mapping (SSA prev-ranks): rolling={GPS_ROLLING_ENABLED}, window={GPS_CALIB_YEARS}y")
print(f"Entry cost applied EVERY month: {ENTRY_COST}")
print(f"SSA params: L={SSA_WINDOW}, comps={SSA_COMPS}, EWMA_scale={USE_EWMA_SCALE}")

# Baseline WITH-COST monthly NAV paths (SSA rankings, identity mapping)
long_baseline_nav  = simulate_baseline_nav_paths(long_rankings,  'long',  FINAL_SIM_START, FINAL_SIM_END, ENTRY_COST)
short_baseline_nav = simulate_baseline_nav_paths(short_rankings, 'short', FINAL_SIM_START, FINAL_SIM_END, ENTRY_COST)

# GPS WITH-COST monthly NAV paths (SSA prev-ranks → GPS monthly mapping)
long_gps_nav  = simulate_gps_nav_paths(long_rankings,  'long',  FINAL_SIM_START, FINAL_SIM_END,
                                       GPS_CALIB_YEARS, GPS_ROLLING_ENABLED, ENTRY_COST)
short_gps_nav = simulate_gps_nav_paths(short_rankings, 'short', FINAL_SIM_START, FINAL_SIM_END,
                                       GPS_CALIB_YEARS, GPS_ROLLING_ENABLED, ENTRY_COST)

# Output dirs
out_root = Path().resolve() / "Outputs" / f"SSA_vs_GPS_{NUM_T}"
plot_dir_long  = out_root / "plots" / "LONG"
plot_dir_short = out_root / "plots" / "SHORT"
for p in [plot_dir_long, plot_dir_short]:
    p.mkdir(parents=True, exist_ok=True)

# Save NAV CSVs
long_baseline_nav.to_csv(out_root / "LONG_baseline_nav.csv")
long_gps_nav.to_csv(out_root / "LONG_gps_nav.csv")
short_baseline_nav.to_csv(out_root / "SHORT_baseline_nav.csv")
short_gps_nav.to_csv(out_root / "SHORT_gps_nav.csv")

def _clip_plot_range(df: pd.DataFrame, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    return df[(df.index >= pd.Timestamp(start_dt)) & (df.index <= pd.Timestamp(end_dt))]

def plot_pair_series(dates, y1, y2, title, ylabel, save_path):
    plt.figure(figsize=(10,6))
    plt.plot(dates, y1, label='Baseline (SSA only) — With Costs')
    plt.plot(dates, y2, label='GPS-mapped (SSA) — With Costs')
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.title(title)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend()
    plt.grid(True)
    plt.xlim(PLOT_START, PLOT_END)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Plot each portfolio (LONG / SHORT)
for k in range(1, NUM_T + 1):
    col = f'portfolio_{k}'
    # LONG
    bl = _clip_plot_range(long_baseline_nav[[col]].dropna(how='all'), PLOT_START, PLOT_END)
    gp = _clip_plot_range(long_gps_nav[[col]].dropna(how='all'),       PLOT_START, PLOT_END)
    idx = bl.index.union(gp.index).sort_values()
    s1 = bl.reindex(idx)[col]
    s2 = gp.reindex(idx)[col]
    plot_pair_series(
        dates=idx, y1=s1, y2=s2,
        title=f'LONG Portfolio {k} — With Monthly Costs',
        ylabel='NAV (CHF)',
        save_path=plot_dir_long / f'portfolio_{k}.png'
    )

    # SHORT
    bls = _clip_plot_range(short_baseline_nav[[col]].dropna(how='all'), PLOT_START, PLOT_END)
    gps = _clip_plot_range(short_gps_nav[[col]].dropna(how='all'),      PLOT_START, PLOT_END)
    idxs = bls.index.union(gps.index).sort_values()
    s1s = bls.reindex(idxs)[col]
    s2s = gps.reindex(idxs)[col]
    plot_pair_series(
        dates=idxs, y1=s1s, y2=s2s,
        title=f'SHORT Portfolio {k} — With Monthly Costs',
        ylabel='NAV (CHF)',
        save_path=plot_dir_short / f'portfolio_{k}.png'
    )

print(f"\nSaved NAV CSVs and per-portfolio plots under:\n  {out_root}")
print(f"  - LONG plots:  {plot_dir_long}")
print(f"  - SHORT plots: {plot_dir_short}")
