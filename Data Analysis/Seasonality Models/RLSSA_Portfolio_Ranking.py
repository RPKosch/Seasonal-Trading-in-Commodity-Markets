# -*- coding: utf-8 -*-
# RLSSA baseline + GPS mapping (lean plotting: daily-only, TR in legend; improved titles & filenames)
# Adds Raw/EWMA label to folder, plot titles, and filenames; optional EWMA scaling of RLSSA score.

import re
import numpy as np
import pandas as pd
import statsmodels.api as sm  # unused but kept if you later add diag checks
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

# RLSSA params
SSA_WINDOW              = 12         # embedding dimension L
SSA_COMPS               = 2          # robust rank q

# Optional EWMA vol scaling of RLSSA score (score / sigma)
USE_EWMA_SCALE          = True
EWMA_LAMBDA             = 0.94       # alpha = 1 - lambda
MIN_OBS_FOR_VOL         = 12

# GPS switches
GPS_ROLLING_ENABLED     = True       # True = rolling 5y monthly re-calibration; False = fixed first 5y
GPS_CALIB_YEARS         = SIM_YEARS

# Plot window
PLOT_START              = datetime(2016, 1, 1)
PLOT_END                = datetime(2024, 12, 31)

DEBUG_DATE              = None

# Contract selection
VOLUME_THRESHOLD        = 1000
ROOT_DIR                = Path().resolve().parent.parent / "Complete Data"

# ==== SPEED/FEATURE TOGGLES ====
TOP1_ONLY               = True       # True = simulate only Top-1 portfolio(s)
RUN_DAILY               = True       # True = run daily-contract simulation (shows daily lines)
DIAG_PRINT              = False      # set True to see RLSSA stats per month

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
# RLSSA runs on LOG monthly returns; GPS metrics use SIMPLE monthly returns
log_rets    = load_returns(base / "All_Monthly_Log_Return_Data")
simple_rets = load_returns(base / "All_Monthly_Return_Data")
tickers     = list(log_rets)
NUM_T       = len(tickers)

# Ranks to simulate (Top-1 or full universe)
RANKS_TO_SIM = [1] if TOP1_ONLY else list(range(1, NUM_T + 1))

# =============================== #
# 5) RLSSA-BASED RANKINGS BY MONTH
# =============================== #
# --- Robust low-rank SSA ---
def robust_low_rank(X: np.ndarray, q: int, max_iter: int = 25, eps: float = 1e-7):
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

def compute_rlssa_forecast(series: np.ndarray, L: int, q: int) -> float:
    x = np.asarray(series, dtype=float).ravel()
    if np.any(~np.isfinite(x)): return np.nan
    N = x.size
    if not (1 < L < N) or q < 1: return np.nan
    K = N - L + 1

    X = np.column_stack([x[i:i+L] for i in range(K)])  # L×K
    U_r, V_r = robust_low_rank(X, q=q)
    S = U_r @ V_r.T

    rec = np.zeros(N, dtype=float)
    cnt = np.zeros(N, dtype=float)
    for i in range(L):
        for j in range(K):
            rec[i + j] += S[i, j]
            cnt[i + j] += 1.0
    if not np.all(cnt > 0): return np.nan
    rec /= cnt

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
    lags = rec[-1: -L: -1]
    if lags.size != a.size: return np.nan
    return float(np.dot(a, lags))

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
        if len(s_log) < SSA_WINDOW:
            continue

        # RLSSA forecast on log returns
        rlssa_val = compute_rlssa_forecast(s_log.values, L=SSA_WINDOW, q=SSA_COMPS)

        # Optional: EWMA vol scaling on *simple* returns
        score = rlssa_val
        if USE_EWMA_SCALE and np.isfinite(rlssa_val):
            s_simple = simple_rets[t].loc[lb0:lb1].dropna()
            if len(s_simple) >= MIN_OBS_FOR_VOL:
                alpha = 1.0 - EWMA_LAMBDA
                ewma_var = (s_simple**2).ewm(alpha=alpha, adjust=False).mean().iloc[-1]
                sigma = float(np.sqrt(ewma_var)) if np.isfinite(ewma_var) else np.nan
                score = rlssa_val / sigma if (sigma and np.isfinite(sigma) and sigma > 0) else np.nan
            else:
                score = np.nan  # not enough obs to apply EWMA scale

        if np.isfinite(score):
            stats.append({'ticker': t, 'score': score})

    if not stats:
        long_rankings[cur]  = tickers.copy()
        short_rankings[cur] = tickers.copy()
    else:
        dfm = pd.DataFrame(stats).set_index('ticker')
        season_score_by_month[cur] = dfm['score'].to_dict()

        if DIAG_PRINT:
            n = len(dfm)
            print(f"[{cur.date()}] RLSSA ({'EWMA' if USE_EWMA_SCALE else 'Raw'}): {n} tickers "
                  f"(>0: {(dfm['score']>0).sum()}, <0: {(dfm['score']<0).sum()})")

        orderL = dfm.sort_values('score', ascending=False).index.tolist()
        orderS = dfm.sort_values('score', ascending=True ).index.tolist()

        orderL += [t for t in tickers if t not in orderL]
        orderS += [t for t in tickers if t not in orderS]
        long_rankings[cur]  = orderL[:len(tickers)]
        short_rankings[cur] = orderS[:len(tickers)]

    cur += relativedelta(months=1)

# =============================== #
# 6) GPS CALIBRATION UTILITIES
# =============================== #
def compute_cum(rankings: dict[pd.Timestamp, list[str]],
                direction: str,
                start_date: pd.Timestamp,
                end_date: pd.Timestamp,
                entry_cost: float = 0.0):
    rets_nc: dict[int, list[Decimal]] = {}
    rets_wc: dict[int, list[Decimal]] = {}
    season_by_prev_rank: dict[int, list[float]] = {}

    for dt in pd.date_range(start_date, end_date, freq='MS'):
        order = rankings.get(dt)
        if order is None:
            continue

        for r in range(1, len(order) + 1):
            rets_nc.setdefault(r, []); rets_wc.setdefault(r, []); season_by_prev_rank.setdefault(r, [])

        s_map = season_score_by_month.get(dt, {})

        for r, t in enumerate(order, start=1):
            raw = Decimal(str(simple_rets[t].get(dt, 0.0)))
            if direction == 'short':
                raw = Decimal(1) / (Decimal(1) + raw) - Decimal(1)
            rets_nc[r].append(raw)

            r_wc = (Decimal(1) - Decimal(str(entry_cost))) * (Decimal(1) + raw) - Decimal(1) if entry_cost > 0 else raw
            rets_wc[r].append(r_wc)

            sc = s_map.get(t, np.nan)
            if np.isfinite(sc):
                sc = -float(sc) if direction == 'short' else float(sc)
            season_by_prev_rank[r].append(sc if np.isfinite(sc) else np.nan)

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
    return cum_df, rets_nc, season_by_prev_rank

def build_metrics(cum_df: pd.DataFrame,
                  rets_dict: dict[int, list[Decimal]],
                  season_by_prev_rank: dict[int, list[float]]) -> pd.DataFrame:
    rows = []
    for pr in cum_df.index.astype(int):
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
    base_cols = ['season', 'sharpe', 'sortino', 'calmar']
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

        cum_df_calib, rets_nc_calib, season_rank = compute_cum(
            rankings, direction=direction, start_date=win_start, end_date=win_end, entry_cost=0.0
        )
        metrics_df = build_metrics(cum_df_calib, rets_nc_calib, season_rank)
        map_new_to_prev = invert_prev_to_new(metrics_df)

        mapped_order = []
        for new_rank in range(1, max(RANKS_TO_SIM) + 1):
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

        # stamp at month end (so daily vs monthly align better)
        rows.append((pd.Timestamp(dt) + pd.offsets.MonthEnd(1),
                     {k: float(nav[k]) for k in RANKS_TO_SIM}))

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
        map_new_to_prev = invert_prev_to_new(metrics_df)

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

        rows.append((pd.Timestamp(dt) + pd.offsets.MonthEnd(1),
                     {k: float(nav[k]) for k in RANKS_TO_SIM}))

    df = pd.DataFrame([{**{'date': d}, **{f'portfolio_{k}': snap.get(k, np.nan) for k in RANKS_TO_SIM}} for d, snap in rows]).set_index('date').sort_index()
    if return_top1:
        top1_ser = pd.Series({d: t for d, t in top1_list}, name='gps_top1')
        return df, top1_ser
    return df

# =============================== #
# 9) BUILD NAVs & PLOTS (DAILY ONLY; TR IN LEGEND)
# =============================== #
scale_label = "EWMA" if USE_EWMA_SCALE else "Raw"

print("-" * 100)
print(f"Apply Period: {FINAL_SIM_START.date()} -> {FINAL_SIM_END.date()}")
print(f"GPS monthly mapping (RLSSA prev-ranks): rolling={GPS_ROLLING_ENABLED}, window={GPS_CALIB_YEARS}y")
print(f"Entry cost applied once per month: {ENTRY_COST}")
print(f"RLSSA ranking: window L={SSA_WINDOW}, rank q={SSA_COMPS} | Score scaling: {scale_label}")
print(f"TOP1_ONLY={TOP1_ONLY} | RUN_DAILY={RUN_DAILY}")

# Monthly-compounded NAVs (+ capture Top-1)
long_baseline_nav_monthly,  long_top1_baseline  = simulate_baseline_nav_paths_monthly(
    long_rankings,  'long',  FINAL_SIM_START, FINAL_SIM_END, ENTRY_COST, return_top1=True)
short_baseline_nav_monthly, short_top1_baseline = simulate_baseline_nav_paths_monthly(
    short_rankings, 'short', FINAL_SIM_START, FINAL_SIM_END, ENTRY_COST, return_top1=True)

long_gps_nav_monthly,  long_top1_gps  = simulate_gps_nav_paths_monthly(
    long_rankings,  'long',  FINAL_SIM_START, FINAL_SIM_END, GPS_CALIB_YEARS, GPS_ROLLING_ENABLED, ENTRY_COST, return_top1=True)
short_gps_nav_monthly, short_top1_gps = simulate_gps_nav_paths_monthly(
    short_rankings, 'short', FINAL_SIM_START, FINAL_SIM_END, GPS_CALIB_YEARS, GPS_ROLLING_ENABLED, ENTRY_COST, return_top1=True)

# Daily NAVs (for plotting)
if RUN_DAILY:
    long_baseline_nav_daily  = simulate_baseline_nav_paths_daily(long_rankings,  'long',  FINAL_SIM_START, FINAL_SIM_END, ENTRY_COST)
    short_baseline_nav_daily = simulate_baseline_nav_paths_daily(short_rankings, 'short', FINAL_SIM_START, FINAL_SIM_END, ENTRY_COST)

    long_gps_nav_daily  = simulate_gps_nav_paths_daily(long_rankings,  'long',  FINAL_SIM_START, FINAL_SIM_END,
                                                       GPS_CALIB_YEARS, GPS_ROLLING_ENABLED, ENTRY_COST)
    short_gps_nav_daily = simulate_gps_nav_paths_daily(short_rankings, 'short', FINAL_SIM_START, FINAL_SIM_END,
                                                       GPS_CALIB_YEARS, GPS_ROLLING_ENABLED, ENTRY_COST)

# Output dirs — include Raw/EWMA label in folder name
out_root = Path().resolve() / "Outputs" / f"RLSSA_Baseline_vs_GPS_{scale_label}"
plot_dir_long  = out_root / "plots" / "LONG"
plot_dir_short = out_root / "plots" / "SHORT"
out_root.mkdir(parents=True, exist_ok=True)
for p in [plot_dir_long, plot_dir_short]:
    p.mkdir(parents=True, exist_ok=True)

# Save NAV CSVs
long_baseline_nav_monthly.to_csv(out_root / "LONG_baseline_nav_monthly.csv")
long_gps_nav_monthly.to_csv(out_root / "LONG_gps_nav_monthly.csv")
short_baseline_nav_monthly.to_csv(out_root / "SHORT_baseline_nav_monthly.csv")
short_gps_nav_monthly.to_csv(out_root / "SHORT_gps_nav_monthly.csv")

if RUN_DAILY:
    long_baseline_nav_daily.to_csv(out_root / "LONG_baseline_nav_daily.csv")
    long_gps_nav_daily.to_csv(out_root / "LONG_gps_nav_daily.csv")
    short_baseline_nav_daily.to_csv(out_root / "SHORT_baseline_nav_daily.csv")
    short_gps_nav_daily.to_csv(out_root / "SHORT_gps_nav_daily.csv")

# === Top-1 selection CSVs ===
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

# === As-of total return across full series (for legend) ===
def total_return_asof(nav: pd.Series, start_dt: datetime, end_dt: datetime) -> float:
    """
    TR(as-of) = last(nav ≤ end_dt) / last(nav ≤ start_dt) - 1
    """
    if nav is None or nav.empty:
        return np.nan
    s = nav.dropna().sort_index()
    s_start = s[s.index <= pd.Timestamp(start_dt)]
    s_end   = s[s.index <= pd.Timestamp(end_dt)]
    if s_start.empty or s_end.empty:
        return np.nan
    first = float(s_start.iloc[-1])
    last  = float(s_end.iloc[-1])
    if first == 0:
        return np.nan
    return (last / first) - 1.0

def fmt_pct(x: float) -> str:
    return ("{:+.2f}%".format(100.0 * x)) if np.isfinite(x) else "n/a"

# === Plot helper: TWO daily series only, TRs in legend ===
def plot_two_series(dates, y_base, y_gps, title, ylabel, save_path,
                    tr_base=None, tr_gps=None):
    plt.figure(figsize=(10,6))
    label_base = 'Baseline — Daily NAV (with costs)'
    label_gps  = 'GPS — Daily NAV (with costs)'
    if tr_base is not None:
        label_base += f" — TR {fmt_pct(tr_base)}"
    if tr_gps is not None:
        label_gps  += f" — TR {fmt_pct(tr_gps)}"

    if y_base is not None: plt.plot(dates, y_base, label=label_base)
    if y_gps  is not None: plt.plot(dates, y_gps,  label=label_gps)

    plt.xlabel('Date'); plt.ylabel(ylabel); plt.title(title)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend(); plt.grid(True); plt.xlim(PLOT_START, PLOT_END)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

def rank_title_parts(k: int):
    """Return (title_rank, filename_rank) like ('Top-1','Top1') or ('Rank-3','Rank3')."""
    if k == 1:
        return "Top-1", "Top1"
    return f"Rank-{k}", f"Rank{k}"

entry_cost_pct = f"{ENTRY_COST*100:.2f}%"

# Plot each portfolio (LONG / SHORT), daily-only lines — include Raw/EWMA in title & filename
for k in RANKS_TO_SIM:
    col = f'portfolio_{k}'
    title_rank, file_rank = rank_title_parts(k)

    # LONG
    if RUN_DAILY:
        bl_d  = _clip_plot_range(long_baseline_nav_daily[[col]].dropna(how='all'),  PLOT_START, PLOT_END)
        gp_d  = _clip_plot_range(long_gps_nav_daily[[col]].dropna(how='all'),       PLOT_START, PLOT_END)
        idxL = bl_d.index.union(gp_d.index).sort_values()
        y_db = bl_d.reindex(idxL)[col] if not bl_d.empty else None
        y_dg = gp_d.reindex(idxL)[col] if not gp_d.empty else None

        tr_db_base = total_return_asof(long_baseline_nav_daily[col], PLOT_START, PLOT_END)
        tr_dg_base = total_return_asof(long_gps_nav_daily[col],      PLOT_START, PLOT_END)

        titleL = f"{title_rank} Long Portfolio — Entry cost {entry_cost_pct} (RLSSA Baseline vs GPS — {scale_label})"
        fnameL = f"{file_rank}_Long_with_entry_cost_{scale_label}.png"
        plot_two_series(idxL, y_db, y_dg, titleL, "NAV (CHF)",
                        plot_dir_long / fnameL, tr_base=tr_db_base, tr_gps=tr_dg_base)
    else:
        bl_m  = _clip_plot_range(long_baseline_nav_monthly[[col]].dropna(how='all'), PLOT_START, PLOT_END)
        gp_m  = _clip_plot_range(long_gps_nav_monthly[[col]].dropna(how='all'),      PLOT_START, PLOT_END)
        idxL = bl_m.index.union(gp_m.index).sort_values()
        y_mb = bl_m.reindex(idxL)[col] if not bl_m.empty else None
        y_mg = gp_m.reindex(idxL)[col] if not gp_m.empty else None
        tr_mb = total_return_asof(long_baseline_nav_monthly[col], PLOT_START, PLOT_END)
        tr_mg = total_return_asof(long_gps_nav_monthly[col],      PLOT_START, PLOT_END)
        titleL = f"{title_rank} Long Portfolio — Entry cost {entry_cost_pct} (RLSSA Baseline vs GPS — {scale_label})"
        fnameL = f"{file_rank}_Long_with_entry_cost_{scale_label}.png"
        plot_two_series(idxL, y_mb, y_mg, titleL, "NAV (CHF)",
                        plot_dir_long / fnameL, tr_base=tr_mb, tr_gps=tr_mg)

    # SHORT
    if RUN_DAILY:
        bls_d = _clip_plot_range(short_baseline_nav_daily[[col]].dropna(how='all'),  PLOT_START, PLOT_END)
        gps_d = _clip_plot_range(short_gps_nav_daily[[col]].dropna(how='all'),       PLOT_START, PLOT_END)
        idxS = bls_d.index.union(gps_d.index).sort_values()
        ys_db = bls_d.reindex(idxS)[col] if not bls_d.empty else None
        ys_dg = gps_d.reindex(idxS)[col] if not gps_d.empty else None

        tr_db_base_s = total_return_asof(short_baseline_nav_daily[col], PLOT_START, PLOT_END)
        tr_dg_base_s = total_return_asof(short_gps_nav_daily[col],      PLOT_START, PLOT_END)

        titleS = f"{title_rank} Short Portfolio — Entry cost {entry_cost_pct} (RLSSA Baseline vs GPS — {scale_label})"
        fnameS = f"{file_rank}_Short_with_entry_cost_{scale_label}.png"
        plot_two_series(idxS, ys_db, ys_dg, titleS, "NAV (CHF)",
                        plot_dir_short / fnameS, tr_base=tr_db_base_s, tr_gps=tr_dg_base_s)
    else:
        bls_m = _clip_plot_range(short_baseline_nav_monthly[[col]].dropna(how='all'), PLOT_START, PLOT_END)
        gps_m = _clip_plot_range(short_gps_nav_monthly[[col]].dropna(how='all'),      PLOT_START, PLOT_END)
        idxS = bls_m.index.union(gps_m.index).sort_values()
        ys_mb = bls_m.reindex(idxS)[col] if not bls_m.empty else None
        ys_mg = gps_m.reindex(idxS)[col] if not gps_m.empty else None
        tr_mb_s = total_return_asof(short_baseline_nav_monthly[col], PLOT_START, PLOT_END)
        tr_mg_s = total_return_asof(short_gps_nav_monthly[col],      PLOT_START, PLOT_END)
        titleS = f"{title_rank} Short Portfolio — Entry cost {entry_cost_pct} (RLSSA Baseline vs GPS — {scale_label})"
        fnameS = f"{file_rank}_Short_with_entry_cost_{scale_label}.png"
        plot_two_series(idxS, ys_mb, ys_mg, titleS, "NAV (CHF)",
                        plot_dir_short / fnameS, tr_base=tr_mb_s, tr_gps=tr_mg_s)

print(f"\nSaved NAV CSVs and per-portfolio plots under:\n  {out_root}")
print(f"  - LONG plots:  {plot_dir_long}")
print(f"  - SHORT plots: {plot_dir_short}")
print(f"  - Top-1 CSVs:  {out_root / 'Top1Selections_LONG.csv'} \n                {out_root / 'Top1Selections_SHORT.csv'}")
