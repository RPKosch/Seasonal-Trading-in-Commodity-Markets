import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy.stats import norm

# -----------------------------------------------------------------------------
# 1) PARAMETERS
# -----------------------------------------------------------------------------
START_DATE      = datetime(2001, 1, 1)
LOOKBACK_YEARS  = 10   # only used to build SSA history
CALIB_YEARS     = 5
FINAL_END       = datetime(2024, 12, 31)

NUM_SELECT      = 1
STRICT_SEL      = False
MODE            = "Short"   # "Long", "Short", or "LongShort"
SIG_LEVEL       = 1

SSA_WINDOW      = 12
SSA_COMPS       = 2

ENTRY_COST      = EXIT_COST = 0.0025
START_VALUE     = 1000.0
PLOT_START, PLOT_END = datetime(2016, 1, 1), datetime(2024, 12, 31)

VOLUME_THRESHOLD = 1000
DEBUG_DATE      = datetime(2016, 1, 1)

# Data paths
ROOT_DIR    = Path().resolve().parent.parent / "Complete Data"
MONTHLY_DIR = ROOT_DIR / "All_Monthly_Return_Data"

# -----------------------------------------------------------------------------
# 2) BUILD SSA HISTORY
# -----------------------------------------------------------------------------
def load_monthly_returns(root_dir: Path) -> dict[str, pd.Series]:
    """
    Loads all “*_Monthly_Revenues.csv” files from root_dir,
    parses year/month → datetime, coerces returns to numeric,
    and returns a dict ticker → pd.Series indexed by date.
    """
    out: dict[str, pd.Series] = {}
    for path in sorted(root_dir.glob("*_Monthly_Revenues.csv")):
        ticker = path.stem.replace("_Monthly_Revenues", "")
        df = pd.read_csv(path)

        # build a proper datetime index
        df['date'] = pd.to_datetime(df[['year','month']].assign(day=1))

        # coerce the return column
        df['return'] = pd.to_numeric(df['return'], errors='coerce')

        # extract a Series indexed by date
        series = df.set_index('date')['return'].sort_index()

        out[ticker] = series

    return out

def compute_ssa(x: np.ndarray) -> float:
    """
    Basic SSA on a 1D series x of length N using globals:
      SSA_WINDOW = L (1 < L < N), SSA_COMPS = r (number of leading components)
    Returns the SSA one-step forecast \hat{x}_{N+1} based on the grouped reconstruction
    using the first r eigentriples, as in Golyandina & Korobeynikov (2012).

    If inputs are insufficient or numerically ill-posed, returns np.nan.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    L = int(SSA_WINDOW)
    r = int(SSA_COMPS)

    # basic guards
    if N < max(L, 3) or L <= 1 or L >= N or r < 1:
        return np.nan
    if np.isnan(x).any():
        # SSA assumes a contiguous series; handle imputation upstream if needed
        return np.nan

    K = N - L + 1

    # 1) Embedding (trajectory matrix, Hankel)
    X = np.column_stack([x[i:i+L] for i in range(K)])  # shape L x K

    # 2) Decomposition via eigen of S = X X^T
    S = X @ X.T                                        # L x L, symmetric psd
    eigvals, eigvecs = np.linalg.eigh(S)               # ascending order
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # keep only positive, non-negligible singular values
    eps = 1e-12
    pos = eigvals > eps
    if not np.any(pos):
        return np.nan
    eigvals = eigvals[pos]
    eigvecs = eigvecs[:, pos]

    r_eff = int(min(r, eigvals.size))                  # selected components
    U = eigvecs[:, :r_eff]                             # L x r
    sigma = np.sqrt(eigvals[:r_eff])                   # r

    # Right singular vectors V = X^T U / sigma
    V = (X.T @ U) / sigma                              # K x r  (broadcast)

    # 3) Reconstruct grouped X_r = sum sigma_i U_i V_i^T
    # Scale columns of U by sigma, then multiply by V^T
    Xr = (U * sigma) @ V.T                             # L x K

    # 4) Diagonal averaging (Hankelization) -> length-N reconstructed series
    rec = np.zeros(N, dtype=float)
    cnt = np.zeros(N, dtype=float)
    # Anti-diagonals: indices s = i + j with i in [0..L-1], j in [0..K-1]
    for i in range(L):
        for j in range(K):
            rec[i + j] += Xr[i, j]
            cnt[i + j] += 1.0
    valid = cnt > 0
    if not np.all(valid):
        return np.nan
    rec /= cnt

    # 5) Recurrent SSA one-step forecast using selected eigenvectors
    #    P_i = U_i, split into head (first L-1) and last coord pi_i
    P_head = U[:-1, :]                 # (L-1) x r
    pi = U[-1, :]                      # r
    nu2 = float(np.dot(pi, pi))        # sum pi_i^2

    # guard: (1 - nu^2) must not be ~0
    if 1.0 - nu2 <= 1e-10:
        return np.nan

    # R = (a_{L-1}, ..., a_1)^T = (1/(1 - nu^2)) * sum_i pi_i * P_i_head
    # which is matrix-vector product P_head @ pi
    R = (P_head @ pi) / (1.0 - nu2)    # length L-1
    a = R[::-1]                        # (a_1, ..., a_{L-1})

    # one-step forecast: \hat x_{N+1} = sum_{j=1}^{L-1} a_j * \tilde x_{N+1-j}
    lags = rec[-1: -L: -1]             # [rec[N-1], rec[N-2], ..., rec[N-L+1]] length L-1
    if lags.size != a.size:
        return np.nan
    x_hat_next = float(np.dot(a, lags))
    return x_hat_next

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
# -----------------------------------------------------------------------------
# 3) CONTRACT FINDER
# -----------------------------------------------------------------------------
def find_contract(ticker: str, year: int, month: int):
    root    = Path().resolve().parent.parent / "Complete Data" / f"{ticker}_Historic_Data"
    m0      = datetime(year, month, 1)
    mend    = m0 + relativedelta(months=1) - timedelta(days=1)
    pattern = re.compile(rf"^{ticker}[_-](\d{{4}})-(\d{{2}})\.csv$")

    candidates = []
    earliest_first_date = None

    if not root.exists():
        print(f"  ✖ {year}-{month:02d} {ticker}: directory not found: {root}")
        return None, None

    for p in root.iterdir():
        if not p.is_file():
            continue
        mobj = pattern.match(p.name)
        if not mobj:
            continue

        fy, fm = int(mobj.group(1)), int(mobj.group(2))
        lag = (fy - year) * 12 + (fm - month)
        if lag < 2:
            continue

        try:
            df = pd.read_csv(p, parse_dates=["Date"])
        except Exception as e:
            print(f"  • skipped {p.name}: {e}")
            continue

        if not df.empty:
            fmin = df["Date"].min()
            if earliest_first_date is None or fmin < earliest_first_date:
                earliest_first_date = fmin

        # Must trade through month-end + 15 days
        if df["Date"].max() < mend + timedelta(days=15):
            continue

        # In-month slice
        mdf = df[(df["Date"] >= m0) & (df["Date"] <= mend)]
        if mdf.empty:
            continue

        # Volume checks
        if "volume" not in mdf.columns:
            print(f"  • rejected {year}-{month:02d} {ticker} file {p.name}: no 'volume' column.")
            continue

        vol = pd.to_numeric(mdf["volume"], errors="coerce")
        avg_vol = float(vol.mean(skipna=True))
        if pd.isna(avg_vol) or avg_vol < VOLUME_THRESHOLD:
            # Uncomment if you want verbose reason:
            # print(f"  • rejected {year}-{month:02d} {ticker} {p.name}: avg vol {avg_vol:.0f} < {VOLUME_THRESHOLD}.")
            continue

        # Candidate accepted (sort by date for consistent first/last rows)
        candidates.append((lag, mdf.sort_values("Date"), p.name, avg_vol))

    if not candidates:
        if earliest_first_date is not None and earliest_first_date > mend:
            print(
                f"  ✖ {year}-{month:02d} {ticker}: No usable contract — "
                f"earliest file starts {earliest_first_date.date()} > month-end {mend.date()}."
            )
        else:
            print(
                f"  ✖ {year}-{month:02d} {ticker}: No contract met criteria "
                f"(lag≥2, trades through {(mend + timedelta(days=15)).date()}, "
                f"in-month data, avg volume≥{VOLUME_THRESHOLD})."
            )
        return None, None

    # Pick the closest acceptable front-month
    _, best_mdf, best_name, avg_vol = min(candidates, key=lambda x: x[0])
    return ticker, best_mdf

# -----------------------------------------------------------------------------
# 4) PREP & SSA TABLE
# -----------------------------------------------------------------------------
base = Path().resolve().parent.parent / "Complete Data"

log_returns    = load_monthly_returns(base / "All_Monthly_Log_Return_Data")
returns = load_monthly_returns(base / "All_Monthly_Return_Data")

ssa_score = build_ssa_history(log_returns)
print(ssa_score)
tickers   = list(returns)

# first possible forecast date
initial_lb_end       = (START_DATE + relativedelta(years=LOOKBACK_YEARS)
                        - pd.offsets.MonthEnd(1))
first_trade_forecast = (initial_lb_end
                        + pd.offsets.MonthBegin(1)
                        + relativedelta(years=CALIB_YEARS))

# -----------------------------------------------------------------------------
# 5) MAIN LOOP (using only ssa_score & calibration slice)
# -----------------------------------------------------------------------------
records = []
cur     = initial_lb_end + pd.offsets.MonthBegin(0)

while cur <= FINAL_END:
    if cur < first_trade_forecast:
        cur += relativedelta(months=1)
        continue

    # 1) get current SSA vector
    current_ssa = ssa_score.loc[cur]

    # 2) slice calibration block
    cal_start = cur - relativedelta(years=CALIB_YEARS)
    cal_end   = cur - relativedelta(months=1)
    calib_block = ssa_score.loc[cal_start:cal_end]

    # 3) compute mean & std per ticker
    means = calib_block.mean()
    stds  = calib_block.std(ddof=1)

    # 4) compute raw list with z & p-values
    raw = []
    for t in tickers:
        sc = current_ssa[t]
        if np.isnan(sc) or stds[t] <= 0:
            continue
        z = (sc - means[t]) / stds[t]
        p = 2*(1 - norm.cdf(abs(z)))
        raw.append((t, sc, z, p))

    # debug
    if cur == DEBUG_DATE:
        print(f"[DEBUG] SSA/P on {DEBUG_DATE.date()}:")
        print(calib_block)
        print(pd.DataFrame(raw, columns=['Tkr','SSA','Z','P']).set_index('Tkr'))

    # 5) select signals
    sigL = sorted([r for r in raw if r[1] > 0 and r[3] <= SIG_LEVEL], key=lambda x: x[1], reverse=True)
    sigS = sorted([r for r in raw if r[1] < 0 and r[3] <= SIG_LEVEL], key=lambda x: x[1])

    # the “rest” sorted by score
    restL = sorted([r for r in raw if r not in sigL], key=lambda x: x[1], reverse=True)
    restS = sorted([r for r in raw if r not in sigS], key=lambda x: x[1])

    picks = []

    if MODE == "Long":
        # if strict and not enough signals, nothing
        if STRICT_SEL and len(sigL) < NUM_SELECT:
            picks = []
        else:
            # take up to NUM_SELECT from the sigL (if any), then pad from restL
            pool = sigL + restL
            picks = [t for t, _, _, _ in pool[:NUM_SELECT]]

    elif MODE == "Short":
        if STRICT_SEL and len(sigS) < NUM_SELECT:
            picks = []
        else:
            pool = sigS + restS
            picks = [f"-{t}" for t, _, _, _ in pool[:NUM_SELECT]]

    else:  # LongShort
        half = NUM_SELECT // 2
        longs_enough = len(sigL) >= half
        shorts_enough = len(sigS) >= half
        if STRICT_SEL and not (longs_enough and shorts_enough):
            picks = []
        else:
            longs_pool = sigL + restL
            shorts_pool = sigS + restS
            longs_pick = [t for t, _, _, _ in longs_pool[:half]]
            shorts_pick = [f"-{t}" for t, _, _, _ in shorts_pool[:half]]
            picks = longs_pick + shorts_pick

    # 6) fetch contracts & record
    daily, contrib = {}, []
    comb = 0.0; n = max(1, len(picks))
    for sig in picks:
        tkr = sig.lstrip('-')
        _, mdf = find_contract(tkr, cur.year, cur.month)
        daily[sig] = mdf
        r = returns[tkr].get(datetime(cur.year, cur.month, 1), np.nan)
        w = (1/(1+r)-1 if sig.startswith('-') else r)/n
        comb += w
        contrib.append(f"{sig}:{r:.2%}→{w:.2%}")

    records.append({
        'forecast': cur,
        'signals':  picks,
        'contribs': contrib,
        'combined': comb,
        'daily_dfs': daily
    })

    cur += relativedelta(months=1)

# -----------------------------------------------------------------------------
# 6) TABULATE & PLOT
# -----------------------------------------------------------------------------
hist_df = pd.DataFrame([{
    'forecast': rec['forecast'],
    'signals':  rec['signals'],
    'contribs': rec['contribs'],
    'combined_ret': rec['combined']
} for rec in records])

pd.set_option('display.precision', 20)
print(hist_df.to_string(index=False))

tot_return = 1
for r in hist_df['combined_ret']:
    tot_return *= (1 + r)
print(f"\nTotal return for {len(records)} months: {tot_return-1:.2%}")

# -----------------------------------------------------------------------------
# 6) PERFORMANCE & PLOT (no costs when not trading)
# -----------------------------------------------------------------------------
vc_nc = vc_wc = START_VALUE
dates, nc, wc, overall_return, cur_return = [], [], [], [], []
Overall_Return = 1.0
for rec in records:
    fc = rec['forecast']
    if not (PLOT_START <= fc <= PLOT_END):
        print(f"Forcast: {fc}")
        continue

    daily = rec['daily_dfs']
    if not daily:
        d = (fc) + pd.offsets.MonthEnd(0)
        dates += [d]; nc += [vc_nc]; wc += [vc_wc]; overall_return += [Overall_Return]; cur_return += [0.0]
    else:
        vc_wc *= (1 - ENTRY_COST)
        all_df = pd.concat([mdf.assign(signal=s) for s,mdf in daily.items()])\
                   .sort_values('Date')
        prevs  = {s:None for s in daily}
        for d, grp in all_df.groupby('Date'):
            rs = 0.0
            for r in grp.itertuples():
                sig = r.signal
                prev = prevs[sig]
                close = r.close
                # first bar: use open→close
                if prev is None:
                    open_ = r.open
                    if sig.startswith('-'):
                        # short: entry price is the open, exit is close
                        ret = open_ / close - 1
                    else:
                        # long
                        ret = close / open_ - 1
                else:
                    # subsequent bars: prev is last-close
                    if sig.startswith('-'):
                        # short: profit = prev / close - 1
                        ret = prev / close - 1
                    else:
                        # long: profit = close / prev - 1
                        ret = close / prev - 1

                rs += ret / len(daily)
                prevs[sig] = close

            # now compound
            vc_nc *= (1 + rs)
            vc_wc *= (1 + rs)
            Overall_Return *= (1 + rs)
            dates.append(d)
            nc.append(vc_nc)
            wc.append(vc_wc)
            overall_return.append(Overall_Return)
            cur_return.append(rs)

print(len(dates), len(nc), len(wc), len(overall_return), len(cur_return))

perf = pd.DataFrame({'Date':dates,'NoCosts':nc,'WithCosts':wc, 'Tot_Return': overall_return, 'cur_return': cur_return})\
           .set_index('Date')

#print(perf.to_string(index=True))

final_nc    = perf['NoCosts'].iloc[-1]
final_wc    = perf['WithCosts'].iloc[-1]
tot_nc      = (final_nc/START_VALUE - 1)*100
tot_wc      = (final_wc/START_VALUE - 1)*100

print(tot_nc)

title_str = f"SSA_{MODE}_Portfolio_{NUM_SELECT}_A_&_LB_{LOOKBACK_YEARS}Y_SL_{SIG_LEVEL}_SS_{STRICT_SEL}.png"
output_dir = Path("plots/SSA_Plots")
output_dir.mkdir(exist_ok=True)

plt.figure(figsize=(10,6))
plt.plot(perf.index, perf['NoCosts'],
         label=f'No Costs (Total: {tot_nc:.2f}%)')
plt.plot(perf.index, perf['WithCosts'],
         label=f'With Costs (Total: {tot_wc:.2f}%)')
plt.xlabel('Date'); plt.ylabel('Portfolio Value (CHF)')
plt.title(f'SSA {MODE} Portfolio with {NUM_SELECT} Assets & Lookback of {LOOKBACK_YEARS} Years & SL {SIG_LEVEL} & SS {STRICT_SEL}')
plt.legend(); plt.grid(True)
plt.xlim(PLOT_START, PLOT_END)
plt.tight_layout()
#plt.show()

save_path = output_dir / title_str
plt.savefig(save_path, dpi=300)

