import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from decimal import Decimal
from functools import lru_cache

# =============================== #
# 1) PARAMETERS & WINDOWS
# =============================== #
START_DATE              = datetime(2001, 1, 1)
FINAL_END               = datetime(2024, 12, 31)

LOOKBACK_YEARS          = 10
SIM_YEARS               = 5          # printed diagnostics only

START_VALUE             = 1000.0
ENTRY_COST              = 0.0025     # apply ONCE at month start in invested months

# DVR params
SIG_LEVEL               = 1          # 1.0 ≡ no filter (GREEDY by Z); else significance-first

# GPS switches
GPS_ROLLING_ENABLED     = True       # True = rolling 5y monthly re-calibration; False = fixed first 5y
GPS_CALIB_YEARS         = SIM_YEARS

# Plot window (clip to what you want to see)
PLOT_START              = datetime(2016, 1, 1)
PLOT_END                = datetime(2024, 12, 31)

DEBUG_DATE              = None

# Contract selection
VOLUME_THRESHOLD        = 1000
ROOT_DIR                = Path().resolve().parent.parent / "Complete Data"

# ==== SPEED/FEATURE TOGGLES ====
TOP1_ONLY               = True       # True = simulate only Top-1 portfolio(s)
RUN_DAILY               = True       # True = run daily-contract simulation (shows daily lines + stats)
DIAG_PRINT              = False      # set True to see DVR stats per month (slow/verbose)

# =============================== #
# 2) DATE RANGES
# =============================== #
LOOKBACK_END    = (START_DATE + relativedelta(years=LOOKBACK_YEARS) - pd.offsets.MonthEnd(1))
TEST_SIM_START  = START_DATE + relativedelta(years=LOOKBACK_YEARS)
TEST_SIM_END    = TEST_SIM_START + relativedelta(years=SIM_YEARS) - pd.offsets.MonthEnd(1)
FINAL_SIM_START = START_DATE + relativedelta(years=LOOKBACK_YEARS) + relativedelta(years=SIM_YEARS)
FINAL_SIM_END   = FINAL_END

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
    mdd = abs(dd.min()) if len(dd) else np.nan
    years = len(arr) / 12
    if not np.isfinite(mdd) or mdd == 0 or years == 0: return np.nan
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

# =============================== #
# 4) LOAD DATA
# =============================== #
base = ROOT_DIR
# DVR showcase runs on LOG monthly returns; GPS metrics use SIMPLE monthly returns
log_rets    = load_returns(base / "All_Monthly_Log_Return_Data")
simple_rets = load_returns(base / "All_Monthly_Return_Data")
tickers     = list(log_rets)
NUM_T       = len(tickers)

# Ranks to simulate (Top-1 or full universe)
RANKS_TO_SIM = [1] if TOP1_ONLY else list(range(1, NUM_T + 1))

# =============================== #
# 5) DVR-BASED RANKINGS BY MONTH (Z-score)
# =============================== #
long_rankings:  dict[pd.Timestamp, list[str]] = {}
short_rankings: dict[pd.Timestamp, list[str]] = {}

# keep z-scores per month to feed GPS when SIG_LEVEL >= 1
z_scores_by_month: dict[pd.Timestamp, dict[str, float]] = {}

def newey_west_lags(T: int) -> int:
    """Rule of thumb: L = floor(0.75 * T^(1/3)), at least 1."""
    return max(1, int(np.floor(0.75 * (T ** (1/3)))))

def dvr_stats(monthly_series: pd.Series, forecast_month: pd.Timestamp,
              lookback_years: int | None) -> tuple[float, float, float]:
    """
    For the target forecast month, run OLS with a dummy for that calendar month
    over the lookback window. Return (beta, pval, zscore=t-stat).
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
        zscore = float(model.tvalues.get('D', np.nan))  # NW t-stat (used as Z)
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
    Significance-first ranking that respects the user-chosen p-value threshold (sig_level).
    """
    df = dfm.copy()
    df = df[np.isfinite(df['beta']) & np.isfinite(df['pval'])]
    if df.empty:
        return [], []

    elig_long  = df[(df['pval'] <= float(SIG_LEVEL)) & (df['beta'] > 0)]
    elig_short = df[(df['pval'] <= float(SIG_LEVEL)) & (df['beta'] < 0)]

    rest_long  = df.loc[~df.index.isin(elig_long.index)]
    rest_short = df.loc[~df.index.isin(elig_short.index)]

    elig_long  = elig_long.sort_values('beta', ascending=False)  # more positive beta better
    rest_long  = rest_long.sort_values('beta', ascending=False)

    elig_short = elig_short.sort_values('beta', ascending=True)  # more negative beta better
    rest_short = rest_short.sort_values('beta', ascending=True)

    orderL = elig_long.index.tolist()  + [t for t in rest_long.index.tolist()  if t not in elig_long.index]
    orderS = elig_short.index.tolist() + [t for t in rest_short.index.tolist() if t not in elig_short.index]

    return orderL, orderS

cur = TEST_SIM_START
while cur <= FINAL_END:
    stats = []
    for t in tickers:
        beta, pval, z = dvr_stats(log_rets[t], cur, LOOKBACK_YEARS)
        if not np.isfinite(z):
            continue
        elig = (np.isfinite(pval) and (pval <= float(SIG_LEVEL)))
        stats.append({'ticker': t, 'beta': beta, 'pval': pval, 'z': z, 'elig': elig})

    if not stats:
        long_rankings[cur]  = tickers.copy()
        short_rankings[cur] = tickers.copy()
    else:
        dfm = pd.DataFrame(stats).set_index('ticker')
        z_scores_by_month[cur] = dfm['z'].to_dict()

        if DIAG_PRINT:
            n_tot  = int(len(dfm))
            n_elig = int(dfm['elig'].sum())
            print(f"[{cur.date()}] DVR stats: N={n_tot}, eligible (p <= {SIG_LEVEL}): {n_elig}")

        if float(SIG_LEVEL) >= 0.999999:          # GREEDY mode
            orderL, orderS = _rank_greedy(dfm)
        else:                                      # SIGNIFICANCE-FIRST mode
            orderL, orderS = _rank_sig_first(dfm)

        # Include any missing names at the end (preserve universe)
        orderL += [t for t in tickers if t not in orderL]
        orderS += [t for t in tickers if t not in orderS]
        long_rankings[cur]  = orderL[:len(tickers)]
        short_rankings[cur] = orderS[:len(tickers)]

    cur += relativedelta(months=1)

# =============================== #
# 6) GPS CALIBRATION UTILITIES (monthly rets for metrics only)
# =============================== #
def compute_cum(rankings: dict[pd.Timestamp, list[str]],
                direction: str,
                start_date: pd.Timestamp,
                end_date: pd.Timestamp,
                entry_cost: float = 0.0):
    """
    Used only inside GPS calibration to score previous ranks.
    Returns:
      cum_df           : per prev_rank cumulative returns (no/w cost) over window
      rets_nc          : per prev_rank list of monthly simple returns (NO-COST) over window
      z_by_prev_rank   : per prev_rank list of per-month DVR z-scores for the ticker at that rank.
                         For shorts, z-scores are sign-flipped so 'more negative' counts as larger-is-better.
    """
    rets_nc: dict[int, list[Decimal]] = {}
    rets_wc: dict[int, list[Decimal]] = {}
    z_by_prev_rank: dict[int, list[float]] = {}

    for dt in pd.date_range(start_date, end_date, freq='MS'):
        order = rankings.get(dt)
        if order is None:
            continue
        n_here = len(order)
        for r in range(1, n_here + 1):
            if r not in rets_nc:
                rets_nc[r] = []
                rets_wc[r] = []
                z_by_prev_rank[r] = []

        zmap = z_scores_by_month.get(dt, {})  # ticker -> z
        for r, t in enumerate(order, start=1):
            raw = Decimal(str(simple_rets[t].get(dt, 0.0)))
            if direction == 'short':
                raw = Decimal(1) / (Decimal(1) + raw) - Decimal(1)
            rets_nc[r].append(raw)

            r_wc = (Decimal(1) - Decimal(str(entry_cost))) * (Decimal(1) + raw) - Decimal(1) if entry_cost > 0 else raw
            rets_wc[r].append(r_wc)

            z_val = zmap.get(t, np.nan)
            if direction == 'short':
                z_val = -z_val if np.isfinite(z_val) else z_val
            z_by_prev_rank[r].append(float(z_val) if np.isfinite(z_val) else np.nan)

    # cumulative return per prev rank
    rows = []
    START_D = Decimal(str(START_VALUE))
    for r in sorted(rets_nc.keys()):
        vc_nc = START_D
        vc_wc = START_D
        for x_nc, x_wc in zip(rets_nc[r], rets_wc[r]):
            vc_nc *= (Decimal(1) + x_nc)
            vc_wc *= (Decimal(1) + x_wc)
        rows.append({'rank': float(r),
                     'cum_ret':    vc_nc / START_D - Decimal(1),
                     'cum_ret_wc': vc_wc / START_D - Decimal(1)})
    cum_df = pd.DataFrame(rows).set_index('rank').sort_index()
    return cum_df, rets_nc, z_by_prev_rank

def build_metrics(cum_df: pd.DataFrame,
                  rets_dict: dict[int, list[Decimal]],
                  *,
                  use_metric: str,  # 'zscore' | 'rank'
                  z_rank_dict: dict[int, list[float]] | None = None) -> pd.DataFrame:
    rows = []
    for prev_rank in cum_df.index.astype(int):
        pr = int(prev_rank)

        if use_metric == 'zscore':
            if z_rank_dict is None:
                seasonality_score = np.nan
            else:
                z_list = np.asarray(z_rank_dict.get(pr, []), dtype=float)
                seasonality_score = float(np.nanmean(z_list)) if np.isfinite(z_list).any() else np.nan
        elif use_metric == 'rank':
            seasonality_score = -float(pr)  # invert so rank 1 is best
        else:
            seasonality_score = np.nan

        sr  = sharpe_ratio(rets_dict.get(pr, []))
        sor = sortino_ratio(rets_dict.get(pr, []))
        cal = calmar_ratio(rets_dict.get(pr, []))

        rows.append({
            'prev_rank':    pr,
            'seasonality_score': seasonality_score,
            'sharpe':       sr,
            'sortino':      sor,
            'calmar':       cal,
        })

    df = pd.DataFrame(rows).set_index('prev_rank').sort_index()

    base_cols = ['seasonality_score', 'sharpe', 'sortino', 'calmar']
    for c in base_cols:
        df[f'{c}_01'] = minmax_01(df[c].values)

    norm_cols = [f'{c}_01' for c in base_cols]
    df['score'] = [gps_harmonic_01(df.loc[idx, norm_cols].values) for idx in df.index]
    df['new_rank'] = df['score'].rank(ascending=False, method='first')
    df['rank_change'] = df.index - df['new_rank']
    return df

def invert_prev_to_new(metrics_df: pd.DataFrame) -> dict[int, int]:
    inv = {}
    for prev_rank, row in metrics_df.iterrows():
        nr = int(row['new_rank']); pr = int(prev_rank)
        if nr not in inv: inv[nr] = pr
    return inv

# =============================== #
# 7) DAILY CONTRACT HELPERS
# =============================== #
@lru_cache(maxsize=4096)
def find_contract_cached(ticker: str, year: int, month: int):
    return find_contract(ticker, year, month)

def find_contract(ticker: str, year: int, month: int):
    root    = ROOT_DIR / f"{ticker}_Historic_Data"
    m0      = datetime(year, month, 1)
    mend    = m0 + relativedelta(months=1) - timedelta(days=1)
    pattern = re.compile(rf"^{ticker}[_-](\d{{4}})-(\d{{2}})\.csv$")

    candidates = []
    earliest_first_date = None
    if not root.exists():
        return None, None

    for p in root.iterdir():
        if not p.is_file(): continue
        mobj = pattern.match(p.name)
        if not mobj: continue
        fy, fm = int(mobj.group(1)), int(mobj.group(2))
        lag = (fy - year) * 12 + (fm - month)
        if lag < 2: continue

        try:
            df = pd.read_csv(p, parse_dates=["Date"])
        except Exception:
            continue

        if not df.empty:
            fmin = df["Date"].min()
            if earliest_first_date is None or fmin < earliest_first_date:
                earliest_first_date = fmin

        if df["Date"].max() < mend + timedelta(days=14): continue

        mdf = df[(df["Date"] >= m0) & (df["Date"] <= mend)]
        if mdf.empty: continue

        if "volume" not in mdf.columns:
            continue

        vol = pd.to_numeric(mdf["volume"], errors="coerce")
        avg_vol = float(vol.mean(skipna=True))
        if pd.isna(avg_vol) or avg_vol < VOLUME_THRESHOLD: continue

        candidates.append((lag, mdf.sort_values("Date"), p.name, avg_vol))

    if not candidates:
        return None, None

    _, best_mdf, _, _ = min(candidates, key=lambda x: x[0])
    return ticker, best_mdf

def realized_month_return_from_daily(mdf: pd.DataFrame, is_short: bool) -> float:
    if mdf is None or mdf.empty: return np.nan
    mdf = mdf.sort_values("Date")
    prev = None
    factor = 1.0
    for row in mdf.itertuples():
        c = row.close
        if prev is None:
            o = row.open
            r_long = (c / o) - 1.0
        else:
            r_long = (c / prev) - 1.0
        step_ret = (1.0 / (1.0 + r_long) - 1.0) if is_short else r_long
        factor *= (1.0 + step_ret)
        prev = c
    return factor - 1.0

# =============================== #
# 8) NAV ENGINES — DAILY (BASELINE & GPS)
# =============================== #
def _simulate_daily_nav_from_order_for_month(order: list[str],
                                             direction: str,
                                             dt: pd.Timestamp,
                                             nav_dict: dict[int, Decimal],
                                             entry_cost: float) -> dict[pd.Timestamp, dict[int, float]]:
    # Entry cost once per rank at month start
    for k in RANKS_TO_SIM:
        if k <= len(order) and order[k-1]:
            nav_dict[k] *= (Decimal(1) - Decimal(str(entry_cost)))

    # Load daily bars for ranks
    per_rank_df = {}
    for k in RANKS_TO_SIM:
        if k > len(order) or not order[k-1]:
            per_rank_df[k] = None
            continue
        tkr = order[k - 1]
        _, mdf = find_contract_cached(tkr, dt.year, dt.month)
        if mdf is None or mdf.empty:
            per_rank_df[k] = None
        else:
            m = mdf[['Date','open','close']].sort_values('Date').copy()
            m['rank'] = k
            m['ticker'] = tkr
            per_rank_df[k] = m

    # Union of all dates with bars
    if any(df is not None for df in per_rank_df.values()):
        all_dates = sorted(pd.unique(pd.concat([df for df in per_rank_df.values() if df is not None], axis=0)['Date']))
    else:
        all_dates = []

    prev_close = {k: None for k in RANKS_TO_SIM}
    out = {}

    for d in all_dates:
        for k in RANKS_TO_SIM:
            dfk = per_rank_df.get(k)
            if dfk is None:
                continue
            rows = dfk[dfk['Date'] == d]
            if rows.empty:
                continue

            r = rows.iloc[0]
            open_, close = float(r['open']), float(r['close'])

            if prev_close[k] is None:
                r_long = (close / open_) - 1.0
            else:
                r_long = (close / prev_close[k]) - 1.0

            is_short = (direction == 'short')
            step_ret = (1.0 / (1.0 + r_long) - 1.0) if is_short else r_long

            nav_dict[k] *= (Decimal(1) + Decimal(step_ret))
            prev_close[k] = close

        out[pd.Timestamp(d)] = {k: float(nav_dict[k]) for k in RANKS_TO_SIM}

    return out

def simulate_baseline_nav_paths_daily(rankings: dict[pd.Timestamp, list[str]],
                                      direction: str,
                                      start_dt: pd.Timestamp,
                                      end_dt: pd.Timestamp,
                                      entry_cost: float) -> pd.DataFrame:
    nav = {k: Decimal(str(START_VALUE)) for k in RANKS_TO_SIM}
    ANCHOR_TS = pd.Timestamp(start_dt) - pd.Timedelta(microseconds=1)
    daily_rows = [(ANCHOR_TS, {k: float(nav[k]) for k in RANKS_TO_SIM})]

    for dt in pd.date_range(start_dt, end_dt, freq='MS'):
        order = rankings.get(dt)
        if order is None:
            continue
        day_map = _simulate_daily_nav_from_order_for_month(order, direction, dt, nav, entry_cost)
        for d, snap in day_map.items():
            daily_rows.append((d, snap))

    df = pd.DataFrame(
        [{**{'date': d}, **{f'portfolio_{k}': snap.get(k, np.nan) for k in RANKS_TO_SIM}} for d, snap in daily_rows]
    ).set_index('date').sort_index()
    return df

def simulate_gps_nav_paths_daily(rankings: dict[pd.Timestamp, list[str]],
                                 direction: str,
                                 apply_start: pd.Timestamp,
                                 apply_end: pd.Timestamp,
                                 calib_years: int,
                                 rolling: bool,
                                 entry_cost: float) -> pd.DataFrame:
    nav = {k: Decimal(str(START_VALUE)) for k in RANKS_TO_SIM}
    ANCHOR_TS = pd.Timestamp(apply_start) - pd.Timedelta(microseconds=1)
    daily_rows = [(ANCHOR_TS, {k: float(nav[k]) for k in RANKS_TO_SIM})]

    if not rolling:
        fixed_calib_start = pd.Timestamp(datetime(apply_start.year - calib_years, 1, 1))
        fixed_calib_end   = apply_start - pd.offsets.MonthEnd(1)

    for dt in pd.date_range(apply_start, apply_end, freq='MS'):
        order_today = rankings.get(dt)
        if order_today is None:
            continue

        # calibration window
        if rolling:
            dt_minus = dt - relativedelta(years=calib_years)
            win_start = pd.Timestamp(datetime(dt_minus.year, dt_minus.month, 1))
            win_end   = dt - pd.offsets.MonthEnd(1)
        else:
            win_start = fixed_calib_start
            win_end   = fixed_calib_end
        if win_end < win_start:
            continue

        cum_df_calib, rets_nc_calib, z_rank = compute_cum(
            rankings, direction=direction, start_date=win_start, end_date=win_end, entry_cost=0.0
        )

        # choose the first metric for GPS
        USE_METRIC = 'zscore' if float(SIG_LEVEL) >= 1.0 else 'rank'

        metrics_df = build_metrics(
            cum_df_calib,
            rets_nc_calib,
            use_metric=USE_METRIC,
            z_rank_dict=z_rank if USE_METRIC == 'zscore' else None
        )
        map_new_to_prev = invert_prev_to_new(metrics_df)

        # Build order by prev-rank for today
        mapped_order = []
        for new_rank in range(1, max(RANKS_TO_SIM) + 1):  # only need ranks we'll simulate
            prev_rank = map_new_to_prev.get(new_rank, new_rank)
            if prev_rank < 1 or prev_rank > len(order_today):
                mapped_order.append('')
            else:
                mapped_order.append(order_today[prev_rank - 1])

        day_map = _simulate_daily_nav_from_order_for_month(mapped_order, direction, dt, nav, entry_cost)
        for d, snap in day_map.items():
            daily_rows.append((d, snap))

    df = pd.DataFrame(
        [{**{'date': d}, **{f'portfolio_{k}': snap.get(k, np.nan) for k in RANKS_TO_SIM}} for d, snap in daily_rows]
    ).set_index('date').sort_index()
    return df

# =============================== #
# 8A) NAV ENGINES — MONTHLY (BASELINE & GPS)
# =============================== #
def simulate_baseline_nav_paths_monthly(rankings: dict[pd.Timestamp, list[str]],
                                        direction: str,
                                        start_dt: pd.Timestamp,
                                        end_dt: pd.Timestamp,
                                        entry_cost: float,
                                        return_top1: bool = False):
    nav = {k: Decimal(str(START_VALUE)) for k in RANKS_TO_SIM}
    ANCHOR_TS = pd.Timestamp(start_dt) - pd.Timedelta(microseconds=1)
    rows = [(ANCHOR_TS, {k: float(nav[k]) for k in RANKS_TO_SIM})]
    top1_list = []

    for dt in pd.date_range(start_dt, end_dt, freq='MS'):
        order = rankings.get(dt)
        if not order:
            continue
        if return_top1:
            top1_list.append((dt, order[0] if len(order) >= 1 else ''))

        # cost at month start
        for k in RANKS_TO_SIM:
            if k <= len(order) and order[k-1]:
                nav[k] *= (Decimal(1) - Decimal(str(entry_cost)))

        # apply monthly return once
        for k in RANKS_TO_SIM:
            if k > len(order) or not order[k-1]:
                continue
            tkr = order[k-1]
            r = float(simple_rets[tkr].get(dt, 0.0))
            if direction == 'short':
                r = (1.0 / (1.0 + r)) - 1.0
            nav[k] *= (Decimal(1) + Decimal(str(r)))

        rows.append((pd.Timestamp(dt), {k: float(nav[k]) for k in RANKS_TO_SIM}))

    df = pd.DataFrame([{**{'date': d}, **{f'portfolio_{k}': snap.get(k, np.nan) for k in RANKS_TO_SIM}} for d, snap in rows]).set_index('date').sort_index()
    if return_top1:
        top1_ser = pd.Series({d: t for d, t in top1_list}, name='baseline_top1')
        return df, top1_ser
    return df

def simulate_gps_nav_paths_monthly(rankings: dict[pd.Timestamp, list[str]],
                                   direction: str,
                                   apply_start: pd.Timestamp,
                                   apply_end: pd.Timestamp,
                                   calib_years: int,
                                   rolling: bool,
                                   entry_cost: float,
                                   return_top1: bool = False):
    nav = {k: Decimal(str(START_VALUE)) for k in RANKS_TO_SIM}
    ANCHOR_TS = pd.Timestamp(apply_start) - pd.Timedelta(microseconds=1)
    rows = [(ANCHOR_TS, {k: float(nav[k]) for k in RANKS_TO_SIM})]
    top1_list = []

    if not rolling:
        fixed_calib_start = pd.Timestamp(datetime(apply_start.year - calib_years, 1, 1))
        fixed_calib_end   = apply_start - pd.offsets.MonthEnd(1)

    for dt in pd.date_range(apply_start, apply_end, freq='MS'):
        order_today = rankings.get(dt)
        if not order_today:
            continue

        # calibration window
        if rolling:
            dt_minus = dt - relativedelta(years=calib_years)
            win_start = pd.Timestamp(datetime(dt_minus.year, dt_minus.month, 1))
            win_end   = dt - pd.offsets.MonthEnd(1)
        else:
            win_start = fixed_calib_start
            win_end   = fixed_calib_end
        if win_end < win_start:
            continue

        cum_df_calib, rets_nc_calib, z_rank = compute_cum(
            rankings, direction=direction, start_date=win_start, end_date=win_end, entry_cost=0.0
        )
        USE_METRIC = 'zscore' if float(SIG_LEVEL) >= 1.0 else 'rank'
        metrics_df = build_metrics(cum_df_calib, rets_nc_calib, use_metric=USE_METRIC, z_rank_dict=z_rank if USE_METRIC == 'zscore' else None)
        map_new_to_prev = invert_prev_to_new(metrics_df)

        # Build order by prev-rank for today (only as many ranks as needed)
        mapped_order = []
        for new_rank in range(1, max(RANKS_TO_SIM) + 1):
            prev_rank = map_new_to_prev.get(new_rank, new_rank)
            if prev_rank < 1 or prev_rank > len(order_today):
                mapped_order.append('')
            else:
                mapped_order.append(order_today[prev_rank - 1])

        if return_top1:
            top1_list.append((dt, mapped_order[0] if len(mapped_order) >= 1 else ''))

        # cost at month start
        for k in RANKS_TO_SIM:
            if k <= len(mapped_order) and mapped_order[k-1]:
                nav[k] *= (Decimal(1) - Decimal(str(entry_cost)))

        # apply monthly return once
        for k in RANKS_TO_SIM:
            if k > len(mapped_order) or not mapped_order[k-1]:
                continue
            tkr = mapped_order[k-1]
            r = float(simple_rets[tkr].get(dt, 0.0))
            if direction == 'short':
                r = (1.0 / (1.0 + r)) - 1.0
            nav[k] *= (Decimal(1) + Decimal(str(r)))

        rows.append((pd.Timestamp(dt), {k: float(nav[k]) for k in RANKS_TO_SIM}))

    df = pd.DataFrame([{**{'date': d}, **{f'portfolio_{k}': snap.get(k, np.nan) for k in RANKS_TO_SIM}} for d, snap in rows]).set_index('date').sort_index()
    if return_top1:
        top1_ser = pd.Series({d: t for d, t in top1_list}, name='gps_top1')
        return df, top1_ser
    return df

# =============================== #
# 9) BUILD NAVs & PLOTS (DAILY + MONTHLY)
# =============================== #
print("-" * 100)
print(f"Apply Period: {FINAL_SIM_START.date()} -> {FINAL_SIM_END.date()}")
print(f"GPS monthly mapping (DVR prev-ranks): rolling={GPS_ROLLING_ENABLED}, window={GPS_CALIB_YEARS}y")
print(f"Entry cost applied once per month: {ENTRY_COST}")
print(f"DVR ranking: Z-score (Newey–West t-stat), p eligibility <= {SIG_LEVEL}")
print(f"TOP1_ONLY={TOP1_ONLY} | RUN_DAILY={RUN_DAILY}")

# Monthly-compounded NAVs (fast path) + capture Top-1 selections
long_baseline_nav_monthly,  long_top1_baseline  = simulate_baseline_nav_paths_monthly(
    long_rankings,  'long',  FINAL_SIM_START, FINAL_SIM_END, ENTRY_COST, return_top1=True)
short_baseline_nav_monthly, short_top1_baseline = simulate_baseline_nav_paths_monthly(
    short_rankings, 'short', FINAL_SIM_START, FINAL_SIM_END, ENTRY_COST, return_top1=True)

long_gps_nav_monthly,  long_top1_gps  = simulate_gps_nav_paths_monthly(
    long_rankings,  'long',  FINAL_SIM_START, FINAL_SIM_END, GPS_CALIB_YEARS, GPS_ROLLING_ENABLED, ENTRY_COST, return_top1=True)
short_gps_nav_monthly, short_top1_gps = simulate_gps_nav_paths_monthly(
    short_rankings, 'short', FINAL_SIM_START, FINAL_SIM_END, GPS_CALIB_YEARS, GPS_ROLLING_ENABLED, ENTRY_COST, return_top1=True)

# Optionally: Daily WITH-COST monthly-invest-but-daily-compound NAVs
if RUN_DAILY:
    long_baseline_nav_daily  = simulate_baseline_nav_paths_daily(long_rankings,  'long',  FINAL_SIM_START, FINAL_SIM_END, ENTRY_COST)
    short_baseline_nav_daily = simulate_baseline_nav_paths_daily(short_rankings, 'short', FINAL_SIM_START, FINAL_SIM_END, ENTRY_COST)

    long_gps_nav_daily  = simulate_gps_nav_paths_daily(long_rankings,  'long',  FINAL_SIM_START, FINAL_SIM_END,
                                                       GPS_CALIB_YEARS, GPS_ROLLING_ENABLED, ENTRY_COST)
    short_gps_nav_daily = simulate_gps_nav_paths_daily(short_rankings, 'short', FINAL_SIM_START, FINAL_SIM_END,
                                                       GPS_CALIB_YEARS, GPS_ROLLING_ENABLED, ENTRY_COST)

# Output dirs
out_root = Path().resolve() / "Outputs" / f"DVR_SIG={SIG_LEVEL}_GPS_ROLLING={GPS_ROLLING_ENABLED}_TOP1={TOP1_ONLY}_DAILY={RUN_DAILY}"
plot_dir_long  = out_root / "plots" / "LONG"
plot_dir_short = out_root / "plots" / "SHORT"
out_root.mkdir(parents=True, exist_ok=True)
for p in [plot_dir_long, plot_dir_short]:
    p.mkdir(parents=True, exist_ok=True)

# Save NAV CSVs (monthly always; daily only if generated)
long_baseline_nav_monthly.to_csv(out_root / "LONG_baseline_nav_monthly.csv")
long_gps_nav_monthly.to_csv(out_root / "LONG_gps_nav_monthly.csv")
short_baseline_nav_monthly.to_csv(out_root / "SHORT_baseline_nav_monthly.csv")
short_gps_nav_monthly.to_csv(out_root / "SHORT_gps_nav_monthly.csv")

if RUN_DAILY:
    long_baseline_nav_daily.to_csv(out_root / "LONG_baseline_nav_daily.csv")
    long_gps_nav_daily.to_csv(out_root / "LONG_gps_nav_daily.csv")
    short_baseline_nav_daily.to_csv(out_root / "SHORT_baseline_nav_daily.csv")
    short_gps_nav_daily.to_csv(out_root / "SHORT_gps_nav_daily.csv")

# === Top-1 selection CSVs (format: 1,date,baseline,gps) ===
def write_top1_csv(top1_base: pd.Series, top1_gps: pd.Series, outfile: Path):
    rows = []
    for dt in pd.date_range(FINAL_SIM_START, FINAL_SIM_END, freq='MS'):
        b = str(top1_base.get(dt, "")) if top1_base is not None else ""
        g = str(top1_gps.get(dt, "")) if top1_gps is not None else ""
        rows.append([1, dt.date().isoformat(), b, g])
    pd.DataFrame(rows, columns=['rank','date','baseline_ticker','gps_ticker']) \
      .to_csv(outfile, index=False, header=False)

write_top1_csv(long_top1_baseline,  long_top1_gps,  out_root / "Top1Selections_LONG.csv")
write_top1_csv(short_top1_baseline, short_top1_gps, out_root / "Top1Selections_SHORT.csv")

def _clip_plot_range(df: pd.DataFrame, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    return df[(df.index >= pd.Timestamp(start_dt)) & (df.index <= pd.Timestamp(end_dt))]

# === Utilities to compute total return for annotation ===
def total_return_from_nav(nav: pd.Series) -> float:
    """TR = last / first - 1 (first & last non-NaN)."""
    if nav is None or nav.empty:
        return np.nan
    s = nav.dropna()
    if s.empty:
        return np.nan
    first = float(s.iloc[0])
    last  = float(s.iloc[-1])
    if first == 0:
        return np.nan
    return (last / first) - 1.0

def fmt_pct(x: float) -> str:
    return ("{:+.2f}%".format(100.0 * x)) if np.isfinite(x) else "n/a"

# Plot helpers — now with annotation box for monthly vs daily total returns
def plot_four_series(dates, y_db, y_dg, y_mb_ff, y_mg_ff,
                     title, ylabel, save_path,
                     tr_mb_base=None, tr_mg_base=None, tr_db_base=None, tr_dg_base=None):
    plt.figure(figsize=(10,6))
    if y_db is not None: plt.plot(dates, y_db, label='Baseline — Daily NAV (with costs)')
    if y_mb_ff is not None: plt.plot(dates, y_mb_ff, label='Baseline — Monthly NAV (with costs)')
    if y_dg is not None: plt.plot(dates, y_dg, label='GPS — Daily NAV (with costs)')
    if y_mg_ff is not None: plt.plot(dates, y_mg_ff, label='GPS — Monthly NAV (with costs)')
    plt.xlabel('Date'); plt.ylabel(ylabel); plt.title(title)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend(); plt.grid(True); plt.xlim(PLOT_START, PLOT_END)

    # Build annotation text
    lines = []
    if tr_mb_base is not None and tr_mg_base is not None:
        lines.append(f"Monthly NAV TR  — Base: {fmt_pct(tr_mb_base)}   GPS: {fmt_pct(tr_mg_base)}")
    if tr_db_base is not None and tr_dg_base is not None:
        lines.append(f"Daily NAV TR    — Base: {fmt_pct(tr_db_base)}   GPS: {fmt_pct(tr_dg_base)}")
    if lines:
        txt = "\n".join(lines)
        ax.text(0.01, 0.99, txt, transform=ax.transAxes, va='top', ha='left',
                fontsize=9, bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

# Plot each portfolio (LONG / SHORT)
for k in RANKS_TO_SIM:
    col = f'portfolio_{k}'

    # LONG — get monthly (original) series for TR calculation, and ffill version for plotting on daily axis
    bl_m  = _clip_plot_range(long_baseline_nav_monthly[[col]].dropna(how='all'), PLOT_START, PLOT_END)
    gp_m  = _clip_plot_range(long_gps_nav_monthly[[col]].dropna(how='all'),     PLOT_START, PLOT_END)
    tr_mb_base = total_return_from_nav(bl_m[col])
    tr_mg_base = total_return_from_nav(gp_m[col])

    # base date index
    idxL = bl_m.index.union(gp_m.index).sort_values()
    y_mb = bl_m.reindex(idxL)[col]
    y_mg = gp_m.reindex(idxL)[col]

    if RUN_DAILY:
        bl_d  = _clip_plot_range(long_baseline_nav_daily[[col]].dropna(how='all'),  PLOT_START, PLOT_END)
        gp_d  = _clip_plot_range(long_gps_nav_daily[[col]].dropna(how='all'),       PLOT_START, PLOT_END)
        tr_db_base = total_return_from_nav(bl_d[col])
        tr_dg_base = total_return_from_nav(gp_d[col])
        idxL = idxL.union(bl_d.index).union(gp_d.index).sort_values()
        y_db = bl_d.reindex(idxL)[col]
        y_dg = gp_d.reindex(idxL)[col]
        # forward-fill monthly NAV to draw across days
        y_mb_ff = y_mb.reindex(idxL).ffill()
        y_mg_ff = y_mg.reindex(idxL).ffill()
    else:
        y_db = y_dg = None
        y_mb_ff = y_mb  # plotted on monthly steps
        y_mg_ff = y_mg
        tr_db_base = tr_dg_base = None

    plot_four_series(
        dates=idxL,
        y_db=y_db, y_dg=y_dg, y_mb_ff=y_mb_ff, y_mg_ff=y_mg_ff,
        title=f'LONG Portfolio {k} — Daily vs Monthly Compounding (with Monthly Entry Costs)',
        ylabel='NAV (CHF)',
        save_path=plot_dir_long / f'portfolio_{k}.png',
        tr_mb_base=tr_mb_base, tr_mg_base=tr_mg_base, tr_db_base=tr_db_base, tr_dg_base=tr_dg_base
    )

    # SHORT — same
    bls_m = _clip_plot_range(short_baseline_nav_monthly[[col]].dropna(how='all'), PLOT_START, PLOT_END)
    gps_m = _clip_plot_range(short_gps_nav_monthly[[col]].dropna(how='all'),     PLOT_START, PLOT_END)
    tr_mb_base_s = total_return_from_nav(bls_m[col])
    tr_mg_base_s = total_return_from_nav(gps_m[col])

    idxS = bls_m.index.union(gps_m.index).sort_values()
    ys_mb = bls_m.reindex(idxS)[col]
    ys_mg = gps_m.reindex(idxS)[col]

    if RUN_DAILY:
        bls_d = _clip_plot_range(short_baseline_nav_daily[[col]].dropna(how='all'),  PLOT_START, PLOT_END)
        gps_d = _clip_plot_range(short_gps_nav_daily[[col]].dropna(how='all'),       PLOT_START, PLOT_END)
        tr_db_base_s = total_return_from_nav(bls_d[col])
        tr_dg_base_s = total_return_from_nav(gps_d[col])
        idxS = idxS.union(bls_d.index).union(gps_d.index).sort_values()
        ys_db = bls_d.reindex(idxS)[col]
        ys_dg = gps_d.reindex(idxS)[col]
        ys_mb_ff = ys_mb.reindex(idxS).ffill()
        ys_mg_ff = ys_mg.reindex(idxS).ffill()
    else:
        ys_db = ys_dg = None
        ys_mb_ff = ys_mb
        ys_mg_ff = ys_mg
        tr_db_base_s = tr_dg_base_s = None

    plot_four_series(
        dates=idxS,
        y_db=ys_db, y_dg=ys_dg, y_mb_ff=ys_mb_ff, y_mg_ff=ys_mg_ff,
        title=f'SHORT Portfolio {k} — Daily vs Monthly Compounding (with Monthly Entry Costs)',
        ylabel='NAV (CHF)',
        save_path=plot_dir_short / f'portfolio_{k}.png',
        tr_mb_base=tr_mb_base_s, tr_mg_base=tr_mg_base_s, tr_db_base=tr_db_base_s, tr_dg_base=tr_dg_base_s
    )

print(f"\nSaved NAV CSVs and per-portfolio plots under:\n  {out_root}")
print(f"  - LONG plots:  {plot_dir_long}")
print(f"  - SHORT plots: {plot_dir_short}")
print(f"  - Top-1 CSVs:  {out_root / 'Top1Selections_LONG.csv'} \n {out_root / 'Top1Selections_SHORT.csv'}")
