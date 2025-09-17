import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------------------
# 1) PARAMETERS
# ---------------------------------------------------------------------
START_DATE      = datetime(2001, 1, 1)
LOOKBACK_YEARS  = 10
FINAL_END       = datetime(2024, 12, 31)

NUM_SELECT      = 1
MODE            = "Long"   # "Long", "Short", or "LongShort"

SSA_WINDOW      = 12
SSA_COMPS       = 2

# EWMA vol-scaling for ranking
USE_EWMA_SCALE  = False
EWMA_LAMBDA     = 0.94          # monthly decay; alpha = 1 - lambda
MIN_OBS_FOR_VOL = 12            # min months needed to compute EWMA

ENTRY_COST      = 0.0025         # entry cost only
START_VALUE     = 1000.0
PLOT_START, PLOT_END = datetime(2016, 1, 1), datetime(2024, 12, 31)

VOLUME_THRESHOLD = 1000
DEBUG_DATE       = datetime(2016, 1, 1)

# Data paths
ROOT_DIR    = Path().resolve().parent.parent / "Complete Data"

# ---------------------------------------------------------------------
# 2) LOAD MONTHLY RETURNS
# ---------------------------------------------------------------------
def load_monthly_returns(root_dir: Path) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for path in sorted(root_dir.glob("*_Monthly_Revenues.csv")):
        ticker = path.stem.replace("_Monthly_Revenues", "")
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df[['year','month']].assign(day=1))
        df['return'] = pd.to_numeric(df['return'], errors='coerce')
        out[ticker] = df.set_index('date')['return'].sort_index()
    return out

# ---------------------------------------------------------------------
# 3) SSA CORE
# ---------------------------------------------------------------------
def compute_ssa(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    N = len(x); L = int(SSA_WINDOW); r = int(SSA_COMPS)
    if N < max(L, 3) or L <= 1 or L >= N or r < 1: return np.nan
    if np.isnan(x).any(): return np.nan

    K = N - L + 1
    X = np.column_stack([x[i:i+L] for i in range(K)])  # L x K

    S = X @ X.T
    eigvals, eigvecs = np.linalg.eigh(S)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
    eps = 1e-12
    pos = eigvals > eps
    if not np.any(pos): return np.nan
    eigvals = eigvals[pos]; eigvecs = eigvecs[:, pos]

    r_eff = int(min(r, eigvals.size))
    U = eigvecs[:, :r_eff]
    sigma = np.sqrt(eigvals[:r_eff])
    V = (X.T @ U) / sigma

    Xr = (U * sigma) @ V.T

    rec = np.zeros(N); cnt = np.zeros(N)
    for i in range(L):
        for j in range(K):
            rec[i + j] += Xr[i, j]
            cnt[i + j] += 1.0
    if not np.all(cnt > 0): return np.nan
    rec /= cnt

    P_head = U[:-1, :]
    pi = U[-1, :]
    nu2 = float(np.dot(pi, pi))
    if 1.0 - nu2 <= 1e-10: return np.nan

    R = (P_head @ pi) / (1.0 - nu2)
    a = R[::-1]

    lags = rec[-1: -L: -1]
    if lags.size != a.size: return np.nan
    return float(np.dot(a, lags))

def build_ssa_history(returns):
    start = (START_DATE + relativedelta(years=LOOKBACK_YEARS)).replace(day=1)
    end   = FINAL_END.replace(day=1)
    dates = pd.date_range(start, end, freq='MS')
    ssa_df = pd.DataFrame(index=dates, columns=returns.keys(), dtype=float)
    for dt in dates:
        lb0 = dt - relativedelta(years=LOOKBACK_YEARS)
        lb1 = dt - relativedelta(months=1)
        for tkr, series in returns.items():
            ssa_df.at[dt, tkr] = compute_ssa(series.loc[lb0:lb1].values)
    return ssa_df

# ---------------------------------------------------------------------
# 4) CONTRACT FINDER
# ---------------------------------------------------------------------
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

        if df["Date"].max() < mend: continue

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

# ---------------------------------------------------------------------
# 5) PREP
# ---------------------------------------------------------------------
base = ROOT_DIR
# SSA is fit on log returns (if your log file is log); realized/EWMA on simple returns
log_returns = load_monthly_returns(base / "All_Monthly_Log_Return_Data")
returns     = load_monthly_returns(base / "All_Monthly_Return_Data")

ssa_score = build_ssa_history(log_returns)
print(f"ssa_score {ssa_score}")
tickers   = list(returns)

initial_lb_end       = (START_DATE + relativedelta(years=LOOKBACK_YEARS) - pd.offsets.MonthEnd(1))
first_trade_forecast = initial_lb_end + pd.offsets.MonthBegin(1)

# ---------------------------------------------------------------------
# Helper: realized monthly return from daily bars with exact short math
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# 6) MAIN LOOP (SSA + EWMA scaling)
# ---------------------------------------------------------------------
records = []
cur = initial_lb_end + pd.offsets.MonthBegin(0)

while cur <= FINAL_END:
    if cur < first_trade_forecast:
        cur += relativedelta(months=1)
        continue

    current_ssa = ssa_score.loc[cur]

    # rank by score = SSA forecast / EWMA sigma over same 10y slice
    lb0 = (cur - relativedelta(years=LOOKBACK_YEARS)).to_pydatetime()
    lb1 = (cur - relativedelta(months=1)).to_pydatetime()

    candidates = []
    for tkr in tickers:
        sc = current_ssa.get(tkr, np.nan)
        if not np.isfinite(sc): continue

        if USE_EWMA_SCALE:
            r_win = returns[tkr].loc[lb0:lb1].dropna()
            if len(r_win) < MIN_OBS_FOR_VOL: continue
            alpha = 1.0 - EWMA_LAMBDA
            ewma_var = (r_win**2).ewm(alpha=alpha, adjust=False).mean().iloc[-1]
            sigma = float(np.sqrt(ewma_var)) if np.isfinite(ewma_var) else np.nan
            if not (np.isfinite(sigma) and sigma > 0): continue
            score = sc / sigma
        else:
            score = sc

        candidates.append((tkr, sc, score))

    # choose by score
    K = NUM_SELECT
    rank_long  = sorted([r for r in candidates if r[2] > 0], key=lambda x: x[2], reverse=True)
    rank_short = sorted([r for r in candidates if r[2] < 0], key=lambda x: x[2])

    if MODE == "Long":
        picks = [t for (t, sc, s) in rank_long[:K]]
    elif MODE == "Short":
        picks = [f"-{t}" for (t, sc, s) in rank_short[:K]]
    else:  # LongShort
        half = max(1, K // 2)
        longs_pick  = [t for (t, sc, s) in rank_long[:half]]
        shorts_pick = [f"-{t}" for (t, sc, s) in rank_short[:half]]
        picks = longs_pick + shorts_pick

    # realized monthly returns from DAILY bars (exact short math)
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

    cur += relativedelta(months=1)

# ---------------------------------------------------------------------
# 7) TABULATE & PRINT
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# 8) DAILY-COMPOUNDED PERFORMANCE (ENTRY COST ONLY)
# ---------------------------------------------------------------------
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

final_nc = perf['NoCosts'].iloc[-1]
final_wc = perf['WithCosts'].iloc[-1]
tot_nc   = (final_nc/START_VALUE - 1)*100
tot_wc   = (final_wc/START_VALUE - 1)*100
print(tot_nc)

scale_tag = "EWMA94" if USE_EWMA_SCALE else "RAW"
title_str = f"SSA_{scale_tag}_{MODE}_Portfolio_{NUM_SELECT}_LB_{LOOKBACK_YEARS}Y.png"

output_dir = Path("plots/SSA_Plots"); output_dir.mkdir(exist_ok=True)
output_dir = Path("plots/Additional_Plots"); output_dir.mkdir(exist_ok=True)

plt.figure(figsize=(10,6))
plt.plot(perf.index, perf['NoCosts'],  label=f'No Costs (Total: {tot_nc:.2f}%)')
plt.plot(perf.index, perf['WithCosts'], label=f'With Costs (Total: {tot_wc:.2f}%)')
plt.xlabel('Date'); plt.ylabel('Portfolio Value (CHF)')
plt.title(f'SSA {NUM_SELECT} {MODE} Portfolio — {scale_tag} scaling — LB {LOOKBACK_YEARS}y')
plt.legend(); plt.grid(True)
plt.xlim(PLOT_START, PLOT_END)
plt.tight_layout()
#plt.show()

save_path = output_dir / title_str
plt.savefig(save_path, dpi=300)
