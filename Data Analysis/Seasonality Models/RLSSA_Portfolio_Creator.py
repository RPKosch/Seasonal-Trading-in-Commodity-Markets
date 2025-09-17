import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# -----------------------------------------------------------------------------
# 1) PARAMETERS
# -----------------------------------------------------------------------------
START_DATE      = datetime(2001, 1, 1)
LOOKBACK_YEARS  = 10                 # rolling history for RLSSA & EWMA vol
FINAL_END       = datetime(2024, 12, 31)

NUM_SELECT      = 1
MODE            = "Short"            # "Long", "Short", or "LongShort"

SSA_WINDOW      = 12
SSA_COMPS       = 2                  # robust rank (q)

# EWMA volatility on monthly *simple* returns over the same 10y slice
USE_EWMA_SCALE  = True               # required by your spec; keep toggle for flexibility
EWMA_LAMBDA     = 0.94               # monthly lambda; alpha = 1 - lambda
MIN_OBS_FOR_VOL = 12                 # minimum months for EWMA vol

ENTRY_COST      = 0.0025             # apply at month entry when positions exist
START_VALUE     = 1000.0
PLOT_START, PLOT_END = datetime(2011, 1, 1), datetime(2024, 12, 31)

VOLUME_THRESHOLD = 1000
DEBUG_DATE      = datetime(2016, 1, 1)

# Data paths
ROOT_DIR = Path().resolve().parent.parent / "Complete Data"

# -----------------------------------------------------------------------------
# 2) DATA LOADERS
# -----------------------------------------------------------------------------
def load_monthly_returns(root_dir: Path) -> dict[str, pd.Series]:
    """
    Loads all '*_Monthly_Revenues.csv' from root_dir, returns dict ticker->Series(date->return)
    """
    out: dict[str, pd.Series] = {}
    for path in sorted(root_dir.glob("*_Monthly_Revenues.csv")):
        ticker = path.stem.replace("_Monthly_Revenues", "")
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df[['year','month']].assign(day=1))
        df['return'] = pd.to_numeric(df['return'], errors='coerce')
        series = df.set_index('date')['return'].sort_index()
        out[ticker] = series
    return out

# -----------------------------------------------------------------------------
# 3) RLSSA (Robust Low-Rank SSA)
# -----------------------------------------------------------------------------
def robust_low_rank(X: np.ndarray, q: int, max_iter: int = 25, eps: float = 1e-7):
    """
    Simple IRLS-style robust low-rank approximation:
      minimize ~ sum w_{ij} (X_ij - (UV^T)_ij)^2 with w_{ij} ≈ 1/(|resid|+eps).
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

def compute_rlssa(series: np.ndarray, L: int, q: int) -> float:
    """
    RLSSA one-step forecast:
      1) Hankel embed X (L×K)
      2) Robust low-rank S ≈ U V^T (rank q)
      3) Hankelize S -> robust fitted rec (length N)
      4) SVD(S) -> Uc; build recurrent coefficients a from Uc
      5) Forecast y_{N+1} = sum a_j * rec[N+1-j], j=1..L-1
    Returns scalar forecast or np.nan on failure.
    """
    x = np.asarray(series, dtype=float).ravel()
    if np.any(~np.isfinite(x)):
        return np.nan
    N = x.size
    if not (1 < L < N) or q < 1:
        return np.nan
    K = N - L + 1

    X = np.column_stack([x[i:i+L] for i in range(K)])  # L×K
    U_r, V_r = robust_low_rank(X, q=q)
    S = U_r @ V_r.T

    # Hankelize S
    rec = np.zeros(N, dtype=float)
    cnt = np.zeros(N, dtype=float)
    for i in range(L):
        for j in range(K):
            rec[i + j] += S[i, j]
            cnt[i + j] += 1.0
    if not np.all(cnt > 0):
        return np.nan
    rec /= cnt

    # Classical SVD on S (for recurrence)
    Uc, sc, Vct = np.linalg.svd(S, full_matrices=False)
    r_eff = int(min(q, sc.size))
    if r_eff < 1:
        return np.nan
    Uc = Uc[:, :r_eff]

    P_head = Uc[:-1, :]        # (L-1)×r
    phi    = Uc[-1, :]         # length r
    nu2    = float(np.dot(phi, phi))
    if 1.0 - nu2 <= 1e-10:
        return np.nan

    R = (P_head @ phi) / (1.0 - nu2)   # (a_{L-1},...,a_1)
    a = R[::-1]                         # (a_1,...,a_{L-1})

    lags = rec[-1: -L: -1]
    if lags.size != a.size:
        return np.nan
    return float(np.dot(a, lags))

def build_rlssa_history(returns):
    """
    For each month t, compute an RLSSA forecast using the last LOOKBACK_YEARS returns
    (ending at t-1). Index = forecast months (MS); columns = tickers.
    """
    start = (START_DATE + relativedelta(years=LOOKBACK_YEARS)).replace(day=1)
    end   = FINAL_END.replace(day=1)
    dates = pd.date_range(start, end, freq='MS')
    rlssa_df = pd.DataFrame(index=dates, columns=returns.keys(), dtype=float)

    for dt in dates:
        lb0 = dt - relativedelta(years=LOOKBACK_YEARS)
        lb1 = dt - relativedelta(months=1)
        for tkr, series in returns.items():
            window = series.loc[lb0:lb1].values
            rlssa_df.at[dt, tkr] = compute_rlssa(window, L=SSA_WINDOW, q=SSA_COMPS)
    return rlssa_df

# -----------------------------------------------------------------------------
# 4) CONTRACT FINDER
# -----------------------------------------------------------------------------
def find_contract(ticker: str, year: int, month: int):
    ROOT_DIR = Path().resolve().parent.parent / "Complete Data"
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

# -----------------------------------------------------------------------------
# 5) PREP & RLSSA TABLE
# -----------------------------------------------------------------------------
base = Path().resolve().parent.parent / "Complete Data"

# RLSSA modeling on LOG monthly returns (input series for forecasting)
log_returns      = load_monthly_returns(base / "All_Monthly_Log_Return_Data")
# EWMA volatility computed on SIMPLE monthly returns (realized % series)
simple_returns   = load_monthly_returns(base / "All_Monthly_Return_Data")

rlssa_score = build_rlssa_history(log_returns)
print(rlssa_score)
tickers   = list(simple_returns)

# First possible forecast date (after first 10y window)
initial_lb_end       = (START_DATE + relativedelta(years=LOOKBACK_YEARS) - pd.offsets.MonthEnd(1))
first_trade_forecast = initial_lb_end + pd.offsets.MonthBegin(1)

# -----------------------------------------------------------------------------
# Helper: realized monthly return from daily bars (consistent with daily compounding)
# -----------------------------------------------------------------------------
def realized_month_return_from_daily(mdf: pd.DataFrame, is_short: bool) -> float:
    """
    From month-sliced daily DataFrame (with 'open','close'), compute realized month return.
    Long: compound daily simple returns; Short: r_short = 1/(1+r_long) - 1 per day.
    """
    if mdf is None or mdf.empty:
        return np.nan
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
        step = (1.0 / (1.0 + r_long) - 1.0) if is_short else r_long
        factor *= (1.0 + step)
        prev = c
    return factor - 1.0

def ewma_vol(series: pd.Series, lb0: datetime, lb1: datetime, lam: float, min_obs: int) -> float:
    """
    EWMA volatility on monthly *simple* returns over [lb0, lb1].
    Returns sigma (float) or np.nan if insufficient data.
    """
    win = series.loc[lb0:lb1].dropna()
    if len(win) < min_obs:
        return np.nan
    alpha = 1.0 - lam
    ewma_var = (win**2).ewm(alpha=alpha, adjust=False).mean().iloc[-1]
    sigma = float(np.sqrt(ewma_var)) if np.isfinite(ewma_var) else np.nan
    return sigma if (np.isfinite(sigma) and sigma > 0) else np.nan

# -----------------------------------------------------------------------------
# 6) MAIN LOOP (RLSSA forecast; EWMA scaling on same 10y slice)
# -----------------------------------------------------------------------------
records = []
cur     = initial_lb_end + pd.offsets.MonthBegin(0)

while cur <= FINAL_END:
    if cur < first_trade_forecast:
        cur += relativedelta(months=1)
        continue

    # 1) RLSSA forecasts (per ticker) at forecast month 'cur'
    current_f = rlssa_score.loc[cur]

    # 2) EWMA vol over the SAME 10-year slice
    lb0 = (cur - relativedelta(years=LOOKBACK_YEARS)).to_pydatetime()
    lb1 = (cur - relativedelta(months=1)).to_pydatetime()

    # 3) Build list of adjusted scores
    candidates = []
    for tkr in tickers:
        sc = current_f.get(tkr, np.nan)
        if not np.isfinite(sc):
            continue

        if USE_EWMA_SCALE:
            sigma = ewma_vol(simple_returns[tkr], lb0, lb1, EWMA_LAMBDA, MIN_OBS_FOR_VOL)
            if not np.isfinite(sigma):
                continue
            score = sc / sigma
        else:
            score = sc

        candidates.append((tkr, sc, score))

    # 4) Rank & pick by adjusted 'score'
    K = max(1, NUM_SELECT)
    if MODE == "Long":
        picks = [t for (t, sc, s) in sorted(candidates, key=lambda x: x[2], reverse=True)[:K]]
    elif MODE == "Short":
        picks = [f"-{t}" for (t, sc, s) in sorted(candidates, key=lambda x: x[2])[:K]]
    else:  # LongShort
        half = max(1, K // 2)
        rank_sorted = sorted(candidates, key=lambda x: x[2])
        shorts_pick = [f"-{t}" for (t, sc, s) in rank_sorted[:half]]
        longs_pick  = [t for (t, sc, s) in rank_sorted[::-1][:half]]
        picks = longs_pick + shorts_pick

    # 5) Fetch contracts & compute realized monthly returns from DAILY bars
    daily, contrib = {}, []
    comb = 0.0
    n = max(1, len(picks))
    for sig in picks:
        tkr = sig.lstrip('-')
        _, mdf = find_contract(tkr, cur.year, cur.month)
        daily[sig] = mdf

        r_month = realized_month_return_from_daily(mdf, is_short=sig.startswith('-'))
        w = r_month / n
        comb += w
        contrib.append(f"{sig}:{r_month:+.2%}→{w:+.2%}")

    records.append({
        'forecast': cur,
        'signals':  picks,
        'contribs': contrib,
        'combined': comb,
        'daily_dfs': daily
    })

    if cur == DEBUG_DATE:
        print(f"[DEBUG] {cur.date()} | picks={picks}")

    cur += relativedelta(months=1)

# -----------------------------------------------------------------------------
# 7) TABULATE & PLOT
# -----------------------------------------------------------------------------
hist_df = pd.DataFrame([{
    'forecast': rec['forecast'],
    'signals':  rec['signals'],
    'contribs': rec['contribs'],
    'combined_ret': rec['combined']
} for rec in records])

pd.set_option('display.precision', 20)
print(hist_df.to_string(index=False))

tot_return = 1.0
for r in hist_df['combined_ret']:
    tot_return *= (1 + r)
print(f"\nTotal return for {len(records)} months: {tot_return-1:.2%}")

# -----------------------------------------------------------------------------
# 8) PERFORMANCE & PLOT (no costs when not trading)
# -----------------------------------------------------------------------------
vc_nc = vc_wc = START_VALUE
dates, nc, wc, overall_return, cur_return = [], [], [], [], []

for rec in records:
    fc = rec['forecast']
    if not (PLOT_START <= fc <= PLOT_END):
        continue

    daily = rec['daily_dfs']
    if not daily:
        d = (fc) + pd.offsets.MonthEnd(0)
        dates += [d]; nc += [vc_nc]; wc += [vc_wc]
        overall_return += [vc_nc / START_VALUE]; cur_return += [0.0]
        continue

    # entry cost once at month start (before first day’s returns)
    vc_wc *= (1 - ENTRY_COST)

    all_df = pd.concat([mdf.assign(signal=s) for s, mdf in daily.items()]).sort_values('Date')

    # --- BUY-AND-HOLD WITHIN MONTH (no daily rebalancing) ---
    prevs   = {s: None for s in daily}
    n_legs  = len(daily)
    leg_nc  = {s: vc_nc / n_legs for s in daily}   # allocate capital at month start
    leg_wc  = {s: vc_wc / n_legs for s in daily}

    prev_port_nc = vc_nc
    prev_port_wc = vc_wc

    for d, grp in all_df.groupby('Date'):
        for r in grp.itertuples():
            sig   = r.signal
            prev  = prevs[sig]
            close = r.close
            if prev is None:
                open_  = r.open
                r_long = (close / open_) - 1.0
            else:
                r_long = (close / prev) - 1.0

            step_ret = (1.0/(1.0 + r_long) - 1.0) if sig.startswith('-') else r_long

            # update leg values (no cross-leg rebal)
            leg_nc[sig] *= (1.0 + step_ret)
            leg_wc[sig] *= (1.0 + step_ret)
            prevs[sig] = close

        # portfolio marks each day
        new_vc_nc = sum(leg_nc.values())
        new_vc_wc = sum(leg_wc.values())

        rs_nc = new_vc_nc / prev_port_nc - 1.0
        prev_port_nc = new_vc_nc
        prev_port_wc = new_vc_wc
        vc_nc, vc_wc = new_vc_nc, new_vc_wc

        dates.append(d)
        nc.append(vc_nc)
        wc.append(vc_wc)
        overall_return.append(vc_nc / START_VALUE)
        cur_return.append(rs_nc)

perf = pd.DataFrame(
    {'Date': dates, 'NoCosts': nc, 'WithCosts': wc,
     'Tot_Return': overall_return, 'cur_return': cur_return}
).set_index('Date')

final_nc    = perf['NoCosts'].iloc[-1]
final_wc    = perf['WithCosts'].iloc[-1]
tot_nc      = (final_nc/START_VALUE - 1)*100
tot_wc      = (final_wc/START_VALUE - 1)*100
print(tot_nc)

scale_tag = "EWMA94" if USE_EWMA_SCALE else "RAW"
title_str = f"RLSSA_{scale_tag}_{MODE}_Portfolio_{NUM_SELECT}_LB_{LOOKBACK_YEARS}Y.png"

output_dir = Path("plots/RLSSA_Plots")
output_dir.mkdir(exist_ok=True)

plt.figure(figsize=(10,6))
plt.plot(perf.index, perf['NoCosts'],  label=f'No Costs (Total: {tot_nc:.2f}%)')
plt.plot(perf.index, perf['WithCosts'], label=f'With Costs (Total: {tot_wc:.2f}%)')
plt.xlabel('Date'); plt.ylabel('Portfolio Value (CHF)')
plt.title(f'RLSSA {NUM_SELECT} {MODE} Portfolio — {scale_tag} scaling — LB {LOOKBACK_YEARS}y')
plt.legend(); plt.grid(True)
plt.xlim(PLOT_START, PLOT_END)
plt.tight_layout()
#plt.show()

save_path = output_dir / title_str
plt.savefig(save_path, dpi=300)
