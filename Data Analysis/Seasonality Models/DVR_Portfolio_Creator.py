import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
from decimal import Decimal, getcontext

# -----------------------------------------------------------------------------
# 1) PARAMETERS
# -----------------------------------------------------------------------------
START_DATE      = datetime(2001, 1, 1)
LOOKBACK_YEARS  = 10   # only used to build SSA history
FINAL_END       = datetime(2024, 12, 31)

START_VALUE      = 1000.0
ENTRY_COST       = 0.0025
LOOKBACK_YEARS   = 10      # or None for full history
NUM_SELECT       = 1
STRICT_SEL       = False
MODE             = "Long"   # "Long", "Short", or "LongShort"
SIG_LEVEL        = 1

PLOT_START, PLOT_END = datetime(2011, 1, 1), datetime(2024, 12, 31)

VOLUME_THRESHOLD = 1000
DEBUG = True
DEBUG_DATE = datetime(2020, 5, 1) #+ pd.offsets.MonthEnd(0)

# -----------------------------------------------------------------------------
# 3) CONTRACT‑FINDER
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

def newey_west_lags(T: int) -> int:
    """Rule of thumb: L = floor(0.75 * T^(1/3)), at least 1."""
    return max(1, int(np.floor(0.75 * (T ** (1/3)))))

# -----------------------------------------------------------------------------
# 4) LOAD RETURNS
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

base = Path().resolve().parent.parent / "Complete Data"

returns    = load_monthly_returns(base / "All_Monthly_Log_Return_Data")
simple_returns = load_monthly_returns(base / "All_Monthly_Return_Data")

# first possible forecast date
initial_lb_end       = (START_DATE + relativedelta(years=LOOKBACK_YEARS)
                        - pd.offsets.MonthEnd(1))
first_trade_forecast = (initial_lb_end
                        + pd.offsets.MonthBegin(1))
print(f"first_trade_forecast:  {first_trade_forecast}")

# -----------------------------------------------------------------------------
# 5) BUILD SELECTION HISTORY VIA DUMMY REGRESSION (WITH avg_m‑BASED RANKING)
# -----------------------------------------------------------------------------
records = []
current     = initial_lb_end + pd.offsets.MonthBegin(0)
print(current)

while current <= FINAL_END:
    if current < first_trade_forecast:
        current += relativedelta(months=1)
        continue
    # next forecast month
    fc = current
    nm = fc.month

    # collect stats for every ticker
    stats = []
    for tkr, series in returns.items():
        # slice lookback & require ≥12 obs
        df = series.loc[:(current-relativedelta(months=1))].to_frame('return')
        if LOOKBACK_YEARS is not None:
            cutoff = current - relativedelta(years=LOOKBACK_YEARS)
            df = df[df.index >= cutoff]
        if fc == DEBUG_DATE and DEBUG:
            print(f"\n[DEBUG] Regression input table for ticker {tkr!r} at {DEBUG_DATE.date()}:")
            print(df)

        df = df.copy()
        df['month'] = df.index.month
        df['D']     = (df['month'] == nm).astype(float)

        # OLS with Newey–West (HAC) robust standard errors
        X = sm.add_constant(df['D'])
        L = newey_west_lags(len(df))  # auto-select NW lag based on window length
        print(f"HUHU LEnght of Newwey : {L} and {len(df)}")
        ols = sm.OLS(df['return'], X)
        model = ols.fit(cov_type='HAC', cov_kwds={'maxlags': L})  # adjust lag length as needed

        beta  = model.params['D']
        pval  = model.pvalues['D']
        avg_m = df.loc[df['month'] == nm, 'return'].mean()

        stats.append((tkr, beta, pval, avg_m))

    longs_elig = [(b, t) for (t, b, p, m) in stats if p <= SIG_LEVEL and b > 0]
    shorts_elig = [(b, t) for (t, b, p, m) in stats if p <= SIG_LEVEL and b < 0]

    # everyone else goes to the "rest" buckets, carrying their beta as the key
    long_elig_names = {t for _, t in longs_elig}
    short_elig_names = {t for _, t in shorts_elig}

    rest_longs = [(b, t) for (t, b, p, m) in stats if t not in long_elig_names]
    rest_shorts = [(b, t) for (t, b, p, m) in stats if t not in short_elig_names]

    # --- sort by BETA magnitude/sign direction ---
    longs_elig.sort(key=lambda x: x[0], reverse=True)  # highest positive beta first
    rest_longs.sort(key=lambda x: x[0], reverse=True)

    shorts_elig.sort(key=lambda x: x[0])  # most negative beta first
    rest_shorts.sort(key=lambda x: x[0])

    # build final picks
    selected = []
    n_select = NUM_SELECT

    if MODE == "Long":
        if not (STRICT_SEL and len(longs_elig) < n_select):
            # take up to n_select from elig, then pad from rest
            pool = longs_elig + rest_longs
            selected = [tkr for _,tkr in pool[:n_select]]

    elif MODE == "Short":
        if not (STRICT_SEL and len(shorts_elig) < n_select):
            pool = shorts_elig + rest_shorts
            # mark shorts with a leading '-'
            selected = [f"-{tkr}" for _,tkr in pool[:n_select]]

    else:  # LongShort
        half = n_select // 2
        longs_ok  = not (STRICT_SEL and len(longs_elig) < half)
        shorts_ok = not (STRICT_SEL and len(shorts_elig) < half)
        if longs_ok and shorts_ok:
            longs_pool  = longs_elig[:half]  + rest_longs[:max(0, half-len(longs_elig))]
            shorts_pool = shorts_elig[:half] + rest_shorts[:max(0, half-len(shorts_elig))]
            selected = [tkr for _,tkr in longs_pool] + [f"-{tkr}" for _,tkr in shorts_pool]

    # debug on DEBUG_DATE
    if fc == DEBUG_DATE and DEBUG:
        print(f"\n[DEBUG] Stats @ {DEBUG_DATE.date()}:\n",
              pd.DataFrame(stats, columns=['tkr','beta','pval','avg_m']))
        print(f"[DEBUG] Long‐eligible sorted by avg_m:\n", longs_elig)
        print(f"[DEBUG] Rest‐long sorted by avg_m:\n", rest_longs)
        print(f"[DEBUG] Selected signals:\n", selected)

    # fetch contracts and compute contributions
    daily_dfs, contribs, tot_return = {}, [], []
    comb = 0.0
    month_ret = 1.0
    n    = max(1, len(selected))
    for sig in selected:
        tkr, mdf = find_contract(sig.lstrip('-'), fc.year, fc.month)
        daily_dfs[sig] = mdf
        r = simple_returns[sig.lstrip('-')].get(datetime(fc.year, fc.month, 1), np.nan)
        if sig.startswith('-'):
            r = 1 / (1 + r) - 1  # exact short return
        w = r / n
        comb += w
        contribs.append(f"{sig}:{r:.2%}→{w:.2%}")

    records.append({
        'analysis_end':   current,
        'forecast':       fc,
        'signals':        selected,
        'contribs':       contribs,
        'combined_ret':   comb,
        'daily_dfs':      daily_dfs
    })

    current += pd.DateOffset(months=1)

# -----------------------------------------------------------------------------
# 6) DISPLAY SELECTION HISTORY
# -----------------------------------------------------------------------------
hist_df = pd.DataFrame([{
    'forecast': rec['forecast'],
    'signals':  rec['signals'],
    'contribs': rec['contribs'],
    'combined_ret': rec['combined_ret']
} for rec in records])

pd.set_option('display.precision', 20)
print(hist_df.to_string(index=False))

tot_return = 1
for r in hist_df['combined_ret']:
    tot_return *= (1 + r)
print(f"\nTotal return for {len(records)} months: {tot_return-1:.2%}")
# -----------------------------------------------------------------------------
# 7) DAILY‑COMPOUNDED PERFORMANCE
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
        # no trades this month
        d = fc + pd.offsets.MonthEnd(0)
        dates.append(d)
        nc.append(vc_nc)
        wc.append(vc_wc)
        overall_return.append(Overall_Return)
        cur_return.append(0.0)
        continue

    # apply entry cost
    vc_wc *= (1 - ENTRY_COST)

    # stitch all daily bars together
    all_df = (
        pd.concat([mdf.assign(signal=s) for s, mdf in daily.items()])
          .sort_values('Date')
    )
    # remember last‐seen price for each signal
    prevs = {sig: None for sig in daily}

    # step through each trading day
    for d, grp in all_df.groupby('Date'):
        rs = 0.0
        for row in grp.itertuples():
            sig   = row.signal
            prev  = prevs[sig]
            open_ = row.open
            close = row.close

            # 1) calculate the *long* return for this bar
            if prev is None:
                # first bar of the month for this signal: Open → Close
                r_long = (close / open_) - 1.0
            else:
                # subsequent bars: prev Close → Close
                r_long = (close / prev) - 1.0

            # 2) if it's a short, invert into exact futures short P&L
            if sig.startswith('-'):
                # exact short return:
                r = 1.0 / (1.0 + r_long) - 1.0
                #r = -r_long
            else:
                r = r_long

            # 3) accumulate average across all signals
            rs += r / len(daily)

            # update last price
            prevs[sig] = close

        # 4) compound—both with and without costs
        vc_nc *= (1.0 + rs)
        vc_wc *= (1.0 + rs)
        Overall_Return *= (1.0 + rs)

        # 5) record
        dates.append(d)
        nc.append(vc_nc)
        wc.append(vc_wc)
        overall_return.append(Overall_Return)
        cur_return.append(rs)


print(len(dates), len(nc), len(wc), len(overall_return), len(cur_return))

perf = pd.DataFrame({'Date':dates,'NoCosts':nc,'WithCosts':wc, 'Tot_Return': overall_return, 'cur_return': cur_return})\
           .set_index('Date')
#perf = pd.DataFrame({'Date':dates,'NoCosts':nc,'WithCosts':wc})\
#           .set_index('Date')

if (DEBUG):
    print(perf.to_string(index=True))
    print(perf.loc[DEBUG_DATE - pd.offsets.Day(2) : DEBUG_DATE + pd.offsets.MonthEnd(0)])

# -----------------------------------------------------------------------------
# 8) PLOT PERFORMANCE
# -----------------------------------------------------------------------------
final_nc    = perf['NoCosts'].iloc[-1]
final_wc    = perf['WithCosts'].iloc[-1]
tot_nc      = (final_nc/START_VALUE - 1)*100
tot_wc      = (final_wc/START_VALUE - 1)*100

print(tot_nc)

title_str = f"DVR_{MODE}_Portfolio_{NUM_SELECT}_A_&_LB_{LOOKBACK_YEARS}Y_SL_{SIG_LEVEL}_SS_{STRICT_SEL}.png"
output_dir = Path("plots/DVR_Plots")
output_dir.mkdir(exist_ok=True)

plt.figure(figsize=(10,6))
plt.plot(perf.index, perf['NoCosts'],
         label=f'No Costs (Total: {tot_nc:.2f}%)')
plt.plot(perf.index, perf['WithCosts'],
         label=f'With Costs (Total: {tot_wc:.2f}%)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (CHF)')
plt.title(f'DVR {MODE} Portfolio with {NUM_SELECT} Assets & Lookback of {LOOKBACK_YEARS} Years & SL {SIG_LEVEL} & SS {STRICT_SEL}')
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.legend()
plt.grid(True)
plt.xlim(PLOT_START, PLOT_END)
plt.tight_layout()
#plt.show()

save_path = output_dir / title_str
plt.savefig(save_path, dpi=300)
