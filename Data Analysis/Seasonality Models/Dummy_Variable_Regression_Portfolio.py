import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates

# -----------------------------------------------------------------------------
# 1) PARAMETERS
# -----------------------------------------------------------------------------
START_YEAR, START_MONTH             = 2001, 1
INITIAL_END_YEAR, INITIAL_END_MONTH = 2010, 12
FINAL_END_YEAR, FINAL_END_MONTH     = 2024, 12

START_VALUE      = 1000.0
ENTRY_COST       = 0.0025
EXIT_COST        = 0.0025
LOOKBACK_YEARS   = None       # or None for full history
NUM_SELECT       = 1
STRICT_SELECTION = True
MODE             = "Short"   # "Long", "Short", or "LongShort"
SIG_LEVEL        = 0.05

PLOT_START_YEAR, PLOT_START_MONTH = 2011, 1
PLOT_END_YEAR,   PLOT_END_MONTH   = 2024, 12

DEBUG = True
DEBUG_DATE = datetime(2024, 10, 1) #+ pd.offsets.MonthEnd(0)

# -----------------------------------------------------------------------------
# 2) DATE ENDPOINTS
# -----------------------------------------------------------------------------
start_date  = datetime(START_YEAR, START_MONTH, 1)
initial_end = (datetime(INITIAL_END_YEAR, INITIAL_END_MONTH, 1)
               + pd.offsets.MonthEnd(0))
final_end   = (datetime(FINAL_END_YEAR, FINAL_END_MONTH, 1)
               + pd.offsets.MonthEnd(0))
plot_start  = datetime(PLOT_START_YEAR, PLOT_START_MONTH, 1)
plot_end    = (datetime(PLOT_END_YEAR, PLOT_END_MONTH, 1)
               + pd.offsets.MonthEnd(0))

# -----------------------------------------------------------------------------
# 3) CONTRACT‑FINDER
# -----------------------------------------------------------------------------
def find_contract(ticker: str, year: int, month: int):
    root    = Path().resolve().parent.parent / "Complete Data" / f"{ticker}_Historic_Data"
    m0      = datetime(year, month, 1)
    mend    = m0 + relativedelta(months=1) - timedelta(days=1)
    pattern = re.compile(rf"^{ticker}[_-](\d{{4}})-(\d{{2}})\.csv$")
    candidates = []
    for p in root.iterdir():
        m = pattern.match(p.name)
        if not m: continue
        fy, fm = int(m.group(1)), int(m.group(2))
        diff = (fy - year)*12 + (fm - month)
        if diff < 2: continue
        df = pd.read_csv(p, parse_dates=["Date"])
        if df.Date.max() < mend + timedelta(days=15): continue
        mdf = df[(df.Date>=m0)&(df.Date<=mend)]
        if mdf.empty: continue
        candidates.append((diff, mdf.sort_values("Date")))
    if not candidates:
        return None, None
    _, mdf = min(candidates, key=lambda x: x[0])
    return ticker, mdf

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

# -----------------------------------------------------------------------------
# 5) BUILD SELECTION HISTORY VIA DUMMY REGRESSION
# -----------------------------------------------------------------------------
history = []
current = initial_end

while current < final_end:
    # next forecast month
    fc = current + pd.offsets.MonthBegin(1)
    nm = fc.month

    candidates = []
    for tkr, series in returns.items():
        # slice lookback
        df = series.loc[:current].to_frame(name='return')
        if LOOKBACK_YEARS is not None:
            cutoff = current - relativedelta(years=LOOKBACK_YEARS)
            df = df[df.index >= cutoff]
        if len(df) < 12:
            continue

        df = df.copy()
        df['month'] = df.index.month
        df['D'] = (df['month'] == nm).astype(float)

        # OLS regression with HAC(1)
        X     = sm.add_constant(df['D'])
        model = sm.OLS(df['return'], X).fit(cov_type='HAC', cov_kwds={'maxlags':1})
        beta  = model.params['D']
        pval  = model.pvalues['D']
        avg_m = df.loc[df['month'] == nm, 'return'].mean()

        # debug: dump this ticker’s regression on DEBUG_DATE
        if fc == DEBUG_DATE and DEBUG == True:
            print(f"\n[DEBUG] {tkr} @ {DEBUG_DATE.date()}")
            print("Data sample:\n", df)
            print(model.summary())

        # select if significant and directionally consistent
        if pval <= SIG_LEVEL and beta * avg_m > 0:
            sig = tkr if beta > 0 else f"-{tkr}"
            candidates.append((abs(beta), sig, tkr))

    # split longs/shorts and sort by |beta|
    longs  = [(b,s,t) for b,s,t in candidates if not s.startswith('-')]
    shorts = [(b,s,t) for b,s,t in candidates if s.startswith('-')]

    selected = []
    if MODE == "Long":
        longs.sort(key=lambda x: x[0], reverse=True)
        if not (STRICT_SELECTION and len(longs) < NUM_SELECT):
            selected = [s for _,s,_ in longs[:NUM_SELECT]]
    elif MODE == "Short":
        shorts.sort(key=lambda x: x[0], reverse=True)
        if not (STRICT_SELECTION and len(shorts) < NUM_SELECT):
            selected = [s for _,s,_ in shorts[:NUM_SELECT]]
    else:  # LongShort
        half = NUM_SELECT // 2
        longs.sort(key=lambda x: x[0], reverse=True)
        shorts.sort(key=lambda x: x[0], reverse=True)
        if not (STRICT_SELECTION and (len(longs) < half or len(shorts) < half)):
            selected = [s for _,s,_ in longs[:half]] + [s for _,s,_ in shorts[:half]]

    # debug: list all candidates and final picks on DEBUG_DATE
    if fc == DEBUG_DATE:
        print(f"[DEBUG] All candidates @ {DEBUG_DATE.date()}:\n", pd.DataFrame(candidates, columns=['|β|','signal','ticker']))
        print(f"[DEBUG] Selected signals @ {DEBUG_DATE.date()}:", selected)
        print()

    # fetch contracts and compute contributions
    daily_dfs, contribs = {}, []
    comb = 0.0
    n    = max(1, len(selected))
    for sig in selected:
        tkr, mdf = find_contract(sig.lstrip('-'), fc.year, fc.month)
        daily_dfs[sig] = mdf
        r = simple_returns[sig.lstrip('-')].get(datetime(fc.year, fc.month, 1), np.nan)
        w = (-r if sig.startswith('-') else r) / n
        comb += w
        contribs.append(f"{sig}:{r:.2%}→{w:.2%}")

    history.append({
        'analysis_end':   current,
        'forecast_month': fc,
        'signals':        selected,
        'contribs':       contribs,
        'combined_ret':   comb,
        'daily_dfs':      daily_dfs
    })

    current += pd.DateOffset(months=1)

hist_df = pd.DataFrame(history)[0:-1]

# -----------------------------------------------------------------------------
# 6) DISPLAY SELECTION HISTORY
# -----------------------------------------------------------------------------
print(hist_df[[
    'analysis_end','forecast_month','signals','contribs','combined_ret'
]].to_string(index=False))

# -----------------------------------------------------------------------------
# 7) DAILY‑COMPOUNDED PERFORMANCE
# -----------------------------------------------------------------------------
vc_nc = vc_wc = START_VALUE
dates, vals_nc, vals_wc = [], [], []

for _, row in hist_df.iterrows():
    fc = row['forecast_month']
    if not (plot_start <= fc <= plot_end):
        continue

    daily_dfs = row['daily_dfs']

    # handle months with no positions
    if not daily_dfs:
        m0   = fc
        mend = fc + pd.offsets.MonthEnd(0)
        dates += [m0, mend]
        vals_nc += [vc_nc, vc_nc]
        vals_wc += [vc_wc, vc_wc]
        continue

    vc_wc *= (1 - ENTRY_COST)

    # build and concat daily Price DataFrames
    df_list = []; prevs = {}
    for sig, mdf in daily_dfs.items():
        tmp = mdf.copy()
        tmp['signal'] = sig
        df_list.append(tmp.set_index('Date'))
        prevs[sig] = None

    df_all = pd.concat(df_list).sort_index()

    if DEBUG and not df_all.loc[DEBUG_DATE - pd.offsets.Day(2): DEBUG_DATE + pd.offsets.MonthEnd(0)].empty:
        print(df_all.loc[DEBUG_DATE - pd.offsets.Day(2): DEBUG_DATE + pd.offsets.MonthEnd(0)])

    for date, grp in df_all.groupby(level=0):
        ret_sum = 0.0
        for r in grp.itertuples():
            sig = r.signal
            prev = prevs[sig]
            rt   = (r.close/r.open - 1) if prev is None else (r.close/prev - 1)
            if sig.startswith('-'):
                rt = -rt
            ret_sum += rt / len(daily_dfs)
            prevs[sig] = r.close

        vc_nc *= (1 + ret_sum)
        vc_wc *= (1 + ret_sum)
        dates.append(date)
        vals_nc.append(vc_nc)
        vals_wc.append(vc_wc)

    vc_wc *= (1 - EXIT_COST)

perf = pd.DataFrame({
    'Date': dates,
    'NoCosts': vals_nc,
    'WithCosts': vals_wc
}).set_index('Date')

if (DEBUG):
    print(perf.loc[DEBUG_DATE - pd.offsets.Day(2) : DEBUG_DATE + pd.offsets.MonthEnd(0)])

# -----------------------------------------------------------------------------
# 8) PLOT PERFORMANCE
# -----------------------------------------------------------------------------
initial_val = perf['NoCosts'].iloc[0]
final_nc    = perf['NoCosts'].iloc[-1]
final_wc    = perf['WithCosts'].iloc[-1]
tot_nc      = (final_nc/initial_val - 1)*100
tot_wc      = (final_wc/initial_val - 1)*100

title_str = f"DVR_{MODE}_Portfolio_{NUM_SELECT}_Assets_&_Lookback_{LOOKBACK_YEARS}Y_SL_{SIG_LEVEL}.png"
output_dir = Path("plots/DVR_Plots")
output_dir.mkdir(exist_ok=True)

plt.figure(figsize=(10,6))
plt.plot(perf.index, perf['NoCosts'],
         label=f'No Costs (Total: {tot_nc:.2f}%)')
plt.plot(perf.index, perf['WithCosts'],
         label=f'With Costs (Total: {tot_wc:.2f}%)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (CHF)')
plt.title(f'DVR {MODE} Portfolio with {NUM_SELECT} Assets & Lookback of {LOOKBACK_YEARS} Years & SL {SIG_LEVEL}')
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.legend()
plt.grid(True)
plt.xlim(plot_start, plot_end)
plt.tight_layout()
#plt.show()

save_path = output_dir / title_str
plt.savefig(save_path, dpi=300)
