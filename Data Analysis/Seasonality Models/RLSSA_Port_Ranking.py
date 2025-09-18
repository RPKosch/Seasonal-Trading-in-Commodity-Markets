import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from decimal import Decimal

# =============================== #
# 1) PARAMETERS & WINDOWS
# =============================== #
START_DATE              = datetime(2001, 1, 1)
FINAL_END               = datetime(2024, 12, 31)

LOOKBACK_YEARS          = 10
SIM_YEARS               = 5          # printed diagnostics only

START_VALUE             = 1000.0
ENTRY_COST              = 0.0025     # apply ONCE at month start in invested months

# RLSSA params
SSA_WINDOW              = 12          # embedding dimension L
SSA_COMPS               = 2           # robust rank q

# Optional EWMA vol scaling of RLSSA score (score / sigma)
USE_EWMA_SCALE          = False
EWMA_LAMBDA             = 0.94        # alpha = 1 - lambda
MIN_OBS_FOR_VOL         = 12

# GPS switches
GPS_ROLLING_ENABLED     = True        # True = rolling 5y monthly re-calibration; False = fixed first 5y before FINAL_SIM_START
GPS_CALIB_YEARS         = SIM_YEARS

# Plot window (clip to what you want to see)
PLOT_START              = datetime(2016, 1, 1)
PLOT_END                = datetime(2024, 12, 31)

DEBUG_DATE              = None

# Contract selection
VOLUME_THRESHOLD        = 1000
ROOT_DIR                = Path().resolve().parent.parent / "Complete Data"

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

def build_metrics(cum_df: pd.DataFrame,
                  rets_dict: dict[int, list[Decimal]],
                  season_by_prev_rank: dict[int, list[float]]) -> pd.DataFrame:
    """
    GPS metrics per previous rank:
      - season: mean RLSSA seasonality score observed at that prev_rank over the window
                (already direction-adjusted so larger is better)
      - sharpe, sortino, calmar: computed on NO-COST monthly returns
    """
    rows = []
    for prev_rank, _ in cum_df.iterrows():
        pr = int(prev_rank)

        z = np.asarray(season_by_prev_rank.get(pr, []), dtype=float)
        season_metric = float(np.nanmean(z)) if np.isfinite(z).any() else np.nan

        rows.append({
            'prev_rank': pr,
            'season':    season_metric,
            'sharpe':    sharpe_ratio(rets_dict.get(pr, [])),
            'sortino':   sortino_ratio(rets_dict.get(pr, [])),
            'calmar':    calmar_ratio(rets_dict.get(pr, [])),
        })

    df = pd.DataFrame(rows).set_index('prev_rank').sort_index()

    # Scale to [0,1], larger is better
    base_cols = ['season', 'sharpe', 'sortino', 'calmar']
    for c in base_cols:
        df[f'{c}_01'] = minmax_01(df[c].values)

    norm_cols = [f'{c}_01' for c in base_cols]
    df['score'] = [gps_harmonic_01(df.loc[idx, norm_cols].values) for idx in df.index]
    df['new_rank'] = df['score'].rank(ascending=False, method='first')
    df['rank_change'] = df.index - df['new_rank']
    return df

# ---------- RLSSA core ----------
def robust_low_rank(X: np.ndarray, q: int, max_iter: int = 25, eps: float = 1e-7):
    """
    Simple IRLS-style robust low-rank approximation:
    minimize ~ sum w_ij (X_ij - (UV^T)_ij)^2 with w_ij ≈ 1/(|resid|+eps).
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

def compute_rlssa_forecast(x: np.ndarray, L: int, q: int) -> float:
    """
    RLSSA one-step-ahead forecast. Returns forecast or NaN if not feasible.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    if N < max(L, 3) or L <= 1 or L >= N or q < 1:
        return np.nan
    if np.isnan(x).any():
        return np.nan

    K = N - L + 1
    # Trajectory matrix
    X = np.column_stack([x[i:i+L] for i in range(K)])  # L x K

    # Robust low-rank reconstruction in trajectory space
    U_r, V_r = robust_low_rank(X, q=q)
    S = U_r @ V_r.T  # L x K

    # Hankelization to fitted series rec
    rec = np.zeros(N)
    cnt = np.zeros(N)
    for i in range(L):
        for j in range(K):
            rec[i + j] += S[i, j]
            cnt[i + j] += 1.0
    if not np.all(cnt > 0):
        return np.nan
    rec /= cnt

    # Recurrence from classical SVD of S
    Uc, sc, Vct = np.linalg.svd(S, full_matrices=False)
    r_eff = int(min(q, sc.size))
    if r_eff < 1:
        return np.nan
    Uc = Uc[:, :r_eff]

    P_head = Uc[:-1, :]   # (L-1) x r
    phi    = Uc[-1, :]    # r
    nu2    = float(np.dot(phi, phi))
    if 1.0 - nu2 <= 1e-10:
        return np.nan
    R = (P_head @ phi) / (1.0 - nu2)  # (a_{L-1},...,a_1)
    a = R[::-1]                        # (a_1,...,a_{L-1})

    lags = rec[-1: -L: -1]
    if lags.size != a.size:
        return np.nan
    return float(np.dot(a, lags))

# =============================== #
# 4) LOAD DATA
# =============================== #
base = ROOT_DIR
log_rets    = load_returns(base / "All_Monthly_Log_Return_Data")   # RLSSA fit on LOG returns
simple_rets = load_returns(base / "All_Monthly_Return_Data")       # only for EWMA scaling option
tickers     = list(log_rets)
NUM_T       = len(tickers)

# =============================== #
# 5) RLSSA-BASED RANKINGS BY MONTH
# =============================== #
long_rankings:  dict[pd.Timestamp, list[str]] = {}
short_rankings: dict[pd.Timestamp, list[str]] = {}
season_score_by_month: dict[pd.Timestamp, dict[str, float]] = {}

cur = TEST_SIM_START
while cur <= FINAL_END:
    stats = []
    lb0 = cur - relativedelta(years=LOOKBACK_YEARS)
    lb1 = cur - relativedelta(months=1)

    for t in tickers:
        s_log = log_rets[t].loc[lb0:lb1].dropna()
        if len(s_log) < max(SSA_WINDOW, MIN_OBS_FOR_VOL):
            continue

        # RLSSA forecast on log returns
        rlssa_val = compute_rlssa_forecast(s_log.values, L=SSA_WINDOW, q=SSA_COMPS)

        # Optional EWMA vol scaling on simple returns
        score = rlssa_val
        if USE_EWMA_SCALE:
            s_simple = simple_rets[t].loc[lb0:lb1].dropna()
            if len(s_simple) >= MIN_OBS_FOR_VOL and np.isfinite(rlssa_val):
                alpha = 1.0 - EWMA_LAMBDA
                ewma_var = (s_simple**2).ewm(alpha=alpha, adjust=False).mean().iloc[-1]
                sigma = float(np.sqrt(ewma_var)) if np.isfinite(ewma_var) else np.nan
                if sigma and np.isfinite(sigma) and sigma > 0:
                    score = rlssa_val / sigma
                else:
                    score = np.nan

        if np.isfinite(score):
            stats.append({'ticker': t, 'rlssa': rlssa_val, 'score': score})

    if not stats:
        long_rankings[cur]  = tickers.copy()
        short_rankings[cur] = tickers.copy()
    else:
        dfm = pd.DataFrame(stats).set_index('ticker')

        season_score_by_month[cur] = dfm['score'].to_dict()

        # LONG: score > 0 first (desc), then rest by desc
        pos  = dfm[dfm['score'] > 0].sort_values('score', ascending=False).index.tolist()
        rest = dfm.drop(pos, errors='ignore').sort_values('score', ascending=False).index.tolist()
        orderL = pos + rest
        orderL += [t for t in tickers if t not in orderL]
        long_rankings[cur] = orderL[:len(tickers)]

        # SHORT: score < 0 first (most negative), then rest by asc
        neg  = dfm[dfm['score'] < 0].sort_values('score', ascending=True).index.tolist()
        restS= dfm.drop(neg, errors='ignore').sort_values('score', ascending=True).index.tolist()
        orderS = neg + restS
        orderS += [t for t in tickers if t not in orderS]
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
    """Used only inside calibration to score previous ranks.
       Returns:
         cum_df                : cumulative return per prev_rank (kept for compatibility)
         rets_nc               : list of NO-COST monthly simple returns per prev_rank
         season_by_prev_rank   : list of RLSSA 'score' values per prev_rank (direction-adjusted)
    """
    rets_nc: dict[int, list[Decimal]] = {k: [] for k in range(1, NUM_T + 1)}
    rets_wc: dict[int, list[Decimal]] = {k: [] for k in range(1, NUM_T + 1)}
    season_by_prev_rank: dict[int, list[float]] = {k: [] for k in range(1, NUM_T + 1)}

    for dt in pd.date_range(start_date, end_date, freq='MS'):
        order = rankings.get(dt)
        if order is None:
            continue

        # seasonality score map for this month (ticker -> score)
        s_map = season_score_by_month.get(dt, {})

        for r, t in enumerate(order, start=1):
            if r > NUM_T:
                break

            # monthly simple return for risk metrics (NO-COST)
            raw = Decimal(str(simple_rets[t].get(dt, 0.0)))
            if direction == 'short':
                raw = Decimal(1) / (Decimal(1) + raw) - Decimal(1)
            rets_nc[r].append(raw)

            # with-cost variant (not used for GPS metrics, but kept for compatibility)
            if entry_cost > 0:
                r_wc = (Decimal(1) - Decimal(str(entry_cost))) * (Decimal(1) + raw) - Decimal(1)
            else:
                r_wc = raw
            rets_wc[r].append(r_wc)

            # direction-adjusted seasonality score (larger = better on both sides)
            s_val = s_map.get(t, np.nan)
            if np.isfinite(s_val):
                if direction == 'short':
                    s_val = -float(s_val)
                season_by_prev_rank[r].append(float(s_val))
            else:
                season_by_prev_rank[r].append(np.nan)

    # keep cum_df for compatibility with the rest of the pipeline
    rows = []
    START_D = Decimal(str(START_VALUE))
    for r in range(1, NUM_T + 1):
        vc_nc = START_D
        vc_wc = START_D
        for x_nc, x_wc in zip(rets_nc[r], rets_wc[r]):
            vc_nc *= (Decimal(1) + x_nc)
            vc_wc *= (Decimal(1) + x_wc)
        rows.append({'rank': float(r),
                     'cum_ret':    vc_nc / START_D - Decimal(1),
                     'cum_ret_wc': vc_wc / START_D - Decimal(1)})
    cum_df = pd.DataFrame(rows).set_index('rank')

    return cum_df, rets_nc, season_by_prev_rank

def invert_prev_to_new(metrics_df: pd.DataFrame) -> dict[int, int]:
    inv = {}
    for prev_rank, row in metrics_df.iterrows():
        nr = int(row['new_rank']); pr = int(prev_rank)
        if nr not in inv: inv[nr] = pr
    return inv

# =============================== #
# 7) DAILY CONTRACT HELPERS
# =============================== #
def find_contract(ticker: str, year: int, month: int):
    root    = ROOT_DIR / f"{ticker}_Historic_Data"
    m0      = datetime(year, month, 1)
    mend    = m0 + relativedelta(months=1) - timedelta(days=1)
    pattern = re.compile(rf"^{ticker}[_-](\d{{4}})-(\d{{2}})\.csv$")

    candidates = []
    earliest_first_date = None
    if not root.exists():
        print(f"  ✖ {year}-{month:02d} {ticker}: directory not found: {root}")
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
        except Exception as e:
            print(f"  • skipped {p.name}: {e}")
            continue

        if not df.empty:
            fmin = df["Date"].min()
            if earliest_first_date is None or fmin < earliest_first_date:
                earliest_first_date = fmin

        if df["Date"].max() < mend + timedelta(days=14): continue

        mdf = df[(df["Date"] >= m0) & (df["Date"] <= mend)]
        if mdf.empty: continue

        if "volume" not in mdf.columns:
            print(f"  • rejected {year}-{month:02d} {ticker} {p.name}: no 'volume'.")
            continue

        vol = pd.to_numeric(mdf["volume"], errors="coerce")
        avg_vol = float(vol.mean(skipna=True))
        if pd.isna(avg_vol) or avg_vol < VOLUME_THRESHOLD: continue

        candidates.append((lag, mdf.sort_values("Date"), p.name, avg_vol))

    if not candidates:
        if earliest_first_date is not None and earliest_first_date > mend:
            print(f"  ✖ {year}-{month:02d} {ticker}: earliest file {earliest_first_date.date()} > month-end.")
        else:
            print(f"  ✖ {year}-{month:02d} {ticker}: no contract met criteria.")
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
# 8) DAILY NAV ENGINES (BASELINE AND GPS)
# =============================== #
def _simulate_daily_nav_from_order_for_month(order: list[str],
                                             direction: str,
                                             dt: pd.Timestamp,
                                             nav_dict: dict[int, Decimal],
                                             entry_cost: float) -> dict[pd.Timestamp, dict[int, float]]:
    """
    For month dt, for every rank k, pick ticker = order[k-1], charge entry cost once,
    then compound daily using daily contract bars. Returns {date -> {k -> nav_k}}.
    """
    # Entry cost once per rank at month start (if we have a mapping for k)
    for k in range(1, NUM_T + 1):
        if k <= len(order) and order[k-1]:
            nav_dict[k] *= (Decimal(1) - Decimal(str(entry_cost)))

    # Load daily bars for ranks
    per_rank_df = {}
    for k in range(1, NUM_T + 1):
        if k > len(order) or not order[k-1]:
            per_rank_df[k] = None
            continue
        tkr = order[k - 1]
        _, mdf = find_contract(tkr, dt.year, dt.month)
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

    prev_close = {k: None for k in range(1, NUM_T + 1)}
    out = {}

    for d in all_dates:
        for k in range(1, NUM_T + 1):
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

        out[pd.Timestamp(d)] = {k: float(nav_dict[k]) for k in range(1, NUM_T + 1)}

    return out

def simulate_baseline_nav_paths_daily(rankings: dict[pd.Timestamp, list[str]],
                                      direction: str,
                                      start_dt: pd.Timestamp,
                                      end_dt: pd.Timestamp,
                                      entry_cost: float) -> pd.DataFrame:
    """
    Daily WITH-COST NAV paths for portfolios 1..NUM_T using identity mapping (baseline RLSSA order).
    Index = daily trading dates we actually observe. Includes an initial anchor at start_dt.
    """
    nav = {k: Decimal(str(START_VALUE)) for k in range(1, NUM_T + 1)}
    daily_rows = [(pd.Timestamp(start_dt), {k: float(nav[k]) for k in range(1, NUM_T + 1)})]

    for dt in pd.date_range(start_dt, end_dt, freq='MS'):
        order = rankings.get(dt)
        if order is None:
            continue
        day_map = _simulate_daily_nav_from_order_for_month(order, direction, dt, nav, entry_cost)
        for d, snap in day_map.items():
            daily_rows.append((d, snap))

    df = pd.DataFrame(
        [{**{'date': d}, **{f'portfolio_{k}': snap.get(k, np.nan) for k in range(1, NUM_T + 1)}} for d, snap in daily_rows]
    ).set_index('date').sort_index()

    for k in range(1, NUM_T + 1):
        col = f'portfolio_{k}'
        if col not in df.columns:
            df[col] = np.nan
    return df

def simulate_gps_nav_paths_daily(rankings: dict[pd.Timestamp, list[str]],
                                 direction: str,
                                 apply_start: pd.Timestamp,
                                 apply_end: pd.Timestamp,
                                 calib_years: int,
                                 rolling: bool,
                                 entry_cost: float) -> pd.DataFrame:
    """
    Daily WITH-COST NAV paths for portfolios 1..NUM_T using monthly GPS mapping of prev->new ranks.
    Entry cost applied once at each month start; buy-and-hold within month for the mapped ticker.
    """
    nav = {k: Decimal(str(START_VALUE)) for k in range(1, NUM_T + 1)}
    daily_rows = [(pd.Timestamp(apply_start), {k: float(nav[k]) for k in range(1, NUM_T + 1)})]

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

        cum_df_calib, rets_nc_calib, season_rank = compute_cum(
            rankings, direction=direction, start_date=win_start, end_date=win_end, entry_cost=0.0
        )
        metrics_df = build_metrics(cum_df_calib, rets_nc_calib, season_rank)
        metrics_df = metrics_df.dropna(subset=['new_rank'])
        map_new_to_prev = invert_prev_to_new(metrics_df)


        # Build order by prev-rank for today
        mapped_order = []
        for new_rank in range(1, NUM_T + 1):
            prev_rank = map_new_to_prev.get(new_rank, new_rank)
            if prev_rank < 1 or prev_rank > len(order_today):
                mapped_order.append('')
            else:
                mapped_order.append(order_today[prev_rank - 1])

        day_map = _simulate_daily_nav_from_order_for_month(mapped_order, direction, dt, nav, entry_cost)
        for d, snap in day_map.items():
            daily_rows.append((d, snap))

    df = pd.DataFrame(
        [{**{'date': d}, **{f'portfolio_{k}': snap.get(k, np.nan) for k in range(1, NUM_T + 1)}} for d, snap in daily_rows]
    ).set_index('date').sort_index()

    for k in range(1, NUM_T + 1):
        col = f'portfolio_{k}'
        if col not in df.columns:
            df[col] = np.nan
    return df

# =============================== #
# 9) BUILD NAVs & PLOTS (DAILY)
# =============================== #
print("-" * 100)
print(f"Apply Period: {FINAL_SIM_START.date()} -> {FINAL_SIM_END.date()}")
print(f"GPS monthly mapping (RLSSA prev-ranks): rolling={GPS_ROLLING_ENABLED}, window={GPS_CALIB_YEARS}y")
print(f"Entry cost applied once per month: {ENTRY_COST}")
print(f"RLSSA params: L={SSA_WINDOW}, rank={SSA_COMPS}, EWMA_scale={USE_EWMA_SCALE}")

# Daily WITH-COST monthly-invest-but-daily-compound NAVs
long_baseline_nav_daily  = simulate_baseline_nav_paths_daily(long_rankings,  'long',  FINAL_SIM_START, FINAL_SIM_END, ENTRY_COST)
short_baseline_nav_daily = simulate_baseline_nav_paths_daily(short_rankings, 'short', FINAL_SIM_START, FINAL_SIM_END, ENTRY_COST)

long_gps_nav_daily  = simulate_gps_nav_paths_daily(long_rankings,  'long',  FINAL_SIM_START, FINAL_SIM_END,
                                                   GPS_CALIB_YEARS, GPS_ROLLING_ENABLED, ENTRY_COST)
short_gps_nav_daily = simulate_gps_nav_paths_daily(short_rankings, 'short', FINAL_SIM_START, FINAL_SIM_END,
                                                   GPS_CALIB_YEARS, GPS_ROLLING_ENABLED, ENTRY_COST)

# Output dirs
out_root = Path().resolve() / "Outputs" / f"RLSSA_EWMA={USE_EWMA_SCALE}_vs_GPS_ROLLING={GPS_ROLLING_ENABLED}"
plot_dir_long  = out_root / "plots" / "LONG"
plot_dir_short = out_root / "plots" / "SHORT"
for p in [plot_dir_long, plot_dir_short]:
    p.mkdir(parents=True, exist_ok=True)

# Save NAV CSVs (daily)
long_baseline_nav_daily.to_csv(out_root / "LONG_baseline_nav_daily.csv")
long_gps_nav_daily.to_csv(out_root / "LONG_gps_nav_daily.csv")
short_baseline_nav_daily.to_csv(out_root / "SHORT_baseline_nav_daily.csv")
short_gps_nav_daily.to_csv(out_root / "SHORT_gps_nav_daily.csv")

def _clip_plot_range(df: pd.DataFrame, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    return df[(df.index >= pd.Timestamp(start_dt)) & (df.index <= pd.Timestamp(end_dt))]

def plot_pair_series(dates, y1, y2, title, ylabel, save_path):
    plt.figure(figsize=(10,6))
    plt.plot(dates, y1, label='Baseline (RLSSA only) - With Costs')
    plt.plot(dates, y2, label='GPS-mapped (RLSSA) - With Costs')
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

# Plot each portfolio (LONG / SHORT) using daily series
for k in range(1, NUM_T + 1):
    col = f'portfolio_{k}'
    # LONG
    bl = _clip_plot_range(long_baseline_nav_daily[[col]].dropna(how='all'), PLOT_START, PLOT_END)
    gp = _clip_plot_range(long_gps_nav_daily[[col]].dropna(how='all'),       PLOT_START, PLOT_END)
    idx = bl.index.union(gp.index).sort_values()
    s1 = bl.reindex(idx)[col]
    s2 = gp.reindex(idx)[col]
    plot_pair_series(
        dates=idx, y1=s1, y2=s2,
        title=f'LONG Portfolio {k} - With Monthly Entry Costs',
        ylabel='NAV (CHF)',
        save_path=plot_dir_long / f'portfolio_{k}.png'
    )

    # SHORT
    bls = _clip_plot_range(short_baseline_nav_daily[[col]].dropna(how='all'), PLOT_START, PLOT_END)
    gps = _clip_plot_range(short_gps_nav_daily[[col]].dropna(how='all'),      PLOT_START, PLOT_END)
    idxs = bls.index.union(gps.index).sort_values()
    s1s = bls.reindex(idxs)[col]
    s2s = gps.reindex(idxs)[col]
    plot_pair_series(
        dates=idxs, y1=s1s, y2=s2s,
        title=f'SHORT Portfolio {k} - With Monthly Entry Costs',
        ylabel='NAV (CHF)',
        save_path=plot_dir_short / f'portfolio_{k}.png'
    )

print(f"\nSaved daily NAV CSVs and per-portfolio plots under:\n  {out_root}")
print(f"  - LONG plots:  {plot_dir_long}")
print(f"  - SHORT plots: {plot_dir_short}")
