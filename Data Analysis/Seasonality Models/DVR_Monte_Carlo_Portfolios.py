import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from decimal import Decimal
from collections import defaultdict
from dataclasses import dataclass
from scipy import stats as st
import sys, io
from contextlib import redirect_stdout
import warnings

# =============================== #
# 0) GLOBAL SWITCHES & NUMPY/WARNING SETTINGS
# =============================== #
GENERATE_METRICS_CORR_CSV = False
GENERATE_METRICS_CORR_HEATMAP = False  # requires matplotlib

SIGNIF_TWO_SIDED_ALPHA = 0.01

# =============================== #
# 1) PARAMETERS & WINDOWS
# =============================== #
START_DATE = datetime(2001, 1, 1)
FINAL_END = datetime(2024, 12, 31)

LOOKBACK_YEARS = 10  # DVR lookback

# Portfolio & costs
START_VALUE = 1000.0
TRADING_COST = 0.00086  # per side (buy OR sell)

# DVR params
SIG_LEVEL = 0.1  # 1.0 => greedy by Z (no p filter)

# Risk-normalisation / EWMA
EWMA_LAMBDA = 0.97

# IO
ROOT_DIR = Path().resolve().parent.parent / "Complete Data"
OUT_DIR_MC = Path().resolve() / "Outputs_MC" / f"DVR_MC_p≤{SIG_LEVEL}_classic_vs_risk"

# Monte Carlo (MEB)
MC_RUNS = 100
RNG_SEED = 42
MEB_TRIM = 0.1
SAVE_SERIES = False
SAVE_TICKER_CHOICES_CSV = True
ZERO_NOISE = False  # True => use original historical path (no MEB bootstrap)

# =============================== #
# 2) DATE RANGES
# =============================== #
FINAL_SIM_START = datetime(2016, 1, 1)
FINAL_SIM_END = FINAL_END
print(f"Apply   : {FINAL_SIM_START.date()} -> {FINAL_SIM_END.date()}")

# =============================== #
# 3) HELPERS
# =============================== #
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings(
    "ignore",
    message="Degrees of freedom <= 0 for slice",
    category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in scalar divide",
    category=RuntimeWarning
)


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return f"{x:.{nd}f}"


def _fmt_p(p, alpha=SIGNIF_TWO_SIDED_ALPHA):
    if p is None or not np.isfinite(p):
        return ""
    star = "*" if p < alpha else ""
    return f"{p:.4f}{star}"


def _sig_conclusion(t, p, left_label, right_label, alpha=SIGNIF_TWO_SIDED_ALPHA):
    if (t is None) or (p is None) or (not np.isfinite(t)) or (not np.isfinite(p)) or p >= alpha:
        return "ns"
    return f"{right_label} > {left_label}" if t > 0 else f"{left_label} > {right_label}"


def rule(ch="─", n=96) -> str:
    return ch * n


def section(title: str, ch="═", n=96):
    print("\n" + rule(ch, n))
    print(title)
    print(rule(ch, n) + "\n")


def build_equal_weight_benchmark(simple_rets_dict: dict[str, pd.Series],
                                 start_dt: pd.Timestamp,
                                 end_dt: pd.Timestamp) -> pd.Series:
    """
    Passive monthly benchmark (gross, long): equally-weighted average across all tickers' simple returns
    available in a given month. Index is the month grid [start_dt..end_dt].
    """
    idx = pd.date_range(start_dt, end_dt, freq='MS')
    df = pd.DataFrame({t: s.reindex(idx) for t, s in simple_rets_dict.items()})
    return df.mean(axis=1, skipna=True).rename("benchmark")


def apply_roundtrip_cost(simple_ret: float | np.ndarray, side_cost: float) -> float | np.ndarray:
    """
    Apply one full roundtrip trading cost (buy + sell) to a simple monthly return.

    side_cost is PER SIDE (e.g. 0.0025 = 0.25% on buy, 0.25% on sell).

    For scalar or array-like:
        net = (1 - side_cost)^2 * (1 + simple_ret) - 1
    """
    return (1.0 - side_cost) ** 2 * (1.0 + simple_ret) - 1.0


def long_benchmark_net(bench_gross_long: pd.Series, entry_cost: float) -> pd.Series:
    """
    Convert a gross, LONG benchmark to NET-of-cost using one full monthly roundtrip:
        r_net = (1 - entry_cost)^2 * (1 + r) - 1
    where entry_cost is PER SIDE (buy OR sell).
    """
    b = bench_gross_long.astype(float)
    b = apply_roundtrip_cost(b, entry_cost)
    return b.rename(f"{bench_gross_long.name or 'benchmark'}_long_net")


def compute_beta(port: pd.Series, bench: pd.Series) -> float:
    a = pd.concat([bench, port], axis=1, join="inner").dropna()
    if a.shape[0] < 3:
        return np.nan
    m = a.iloc[:, 0].values
    b = a.iloc[:, 1].values
    var_m = np.var(m, ddof=1)
    if var_m == 0 or not np.isfinite(var_m):
        return np.nan
    cov = np.cov(b, m, ddof=1)[0, 1]
    return cov / var_m


def treynor_ratio_series(port: pd.Series, bench: pd.Series) -> float:
    """Raw Treynor (annualized)."""
    beta = compute_beta(port, bench)
    if not np.isfinite(beta) or beta == 0:
        return np.nan
    mu_m = float(np.nanmean(port.values)) if len(port) else np.nan
    return 12.0 * (mu_m / beta)


def treynor_ratio_series_absbeta(port: pd.Series, bench: pd.Series) -> float:
    """
    Treynor (annualized) using |beta| so 'higher is better' for long and short.
    """
    beta = compute_beta(port, bench)
    if not np.isfinite(beta) or beta == 0:
        return np.nan
    mu_m = float(np.nanmean(port.values)) if len(port) else np.nan
    return 12.0 * (mu_m / abs(beta))


def information_ratio_series(port: pd.Series, bench: pd.Series) -> float:
    a = pd.concat([port, bench], axis=1, join="inner").dropna()
    if a.shape[0] < 3:
        return np.nan
    active = a.iloc[:, 0] - a.iloc[:, 1]
    std = float(np.nanstd(active.values, ddof=1))
    if std == 0 or not np.isfinite(std):
        return np.nan
    return float(np.nanmean(active.values)) / std * np.sqrt(12)


def mean_excess_return_series(port: pd.Series, bench: pd.Series) -> float:
    a = pd.concat([port, bench], axis=1, join="inner")
    if a.shape[0] == 0:
        return np.nan
    active = a.iloc[:, 0] - a.iloc[:, 1]
    return float(np.nanmean(active.values))


def load_returns(root_dir: Path) -> dict[str, pd.Series]:
    """
    Load *_Monthly_Revenues.csv as Series(date->return). Assumes columns year, month, return.
    """
    out: dict[str, pd.Series] = {}
    for f in sorted(root_dir.glob("*_Monthly_Revenues.csv")):
        ticker = f.stem.replace("_Monthly_Revenues", "")
        df = (
            pd.read_csv(f)
            .assign(
                date=lambda d: pd.to_datetime(d[['year', 'month']].assign(day=1)),
                rtn=lambda d: pd.to_numeric(d['return'], errors='coerce')
            )
            .set_index('date')['rtn']
            .sort_index()
        )
        out[ticker] = df
    return out


def nav_from_returns_on_grid(returns: pd.Series,
                             start_value: float,
                             full_index: pd.DatetimeIndex) -> pd.Series:
    cur = float(start_value)
    out_vals, out_idx = [], []
    for dt in full_index:
        r = returns.get(dt, np.nan)
        step = 0.0 if pd.isna(r) else float(r)
        cur *= (1.0 + step)
        out_vals.append(cur)
        out_idx.append(dt)
    return pd.Series(out_vals, index=out_idx, name="nav")


def sharpe_ratio(returns: list[Decimal] | np.ndarray) -> float:
    arr = np.array([float(r) for r in returns], dtype=float)
    if arr.size < 2:
        return np.nan
    std = arr.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return np.nan
    return arr.mean() / std * np.sqrt(12)


def sortino_ratio(returns: list[Decimal] | np.ndarray) -> float:
    arr = np.array([float(r) for r in returns], dtype=float)
    if arr.size == 0:
        return np.nan
    neg = arr[arr < 0]
    if neg.size < 2:
        return np.nan
    std_neg = neg.std(ddof=1)
    if std_neg == 0 or not np.isfinite(std_neg):
        return np.nan
    return arr.mean() / std_neg * np.sqrt(12)


def calmar_ratio(returns: list[Decimal] | np.ndarray) -> float:
    arr = np.array([float(r) for r in returns], dtype=float)
    if arr.size == 0:
        return np.nan
    cum = np.cumprod(1 + arr)
    if cum.size == 0:
        return np.nan
    years = len(arr) / 12.0
    if years <= 0:
        return np.nan
    cagr = cum[-1] ** (1 / years) - 1
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    mdd = dd.min() if dd.size else np.nan  # negative
    if not np.isfinite(mdd) or mdd == 0:
        return np.nan
    return cagr / abs(mdd)


def newey_west_lags(T: int) -> int:
    return max(1, int(np.floor(0.75 * (T ** (1 / 3)))))


def dvr_stats(monthly_series: pd.Series, forecast_month: pd.Timestamp,
              lookback_years: int | None) -> tuple[float, float, float]:
    """
    OLS with month dummy over lookback window (Newey-West HAC).
    Returns (beta, pval, zscore).
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
    df['D'] = (df['month'] == nm).astype(float)

    X = sm.add_constant(df['D'])
    L = newey_west_lags(len(df))
    try:
        model = sm.OLS(df['return'], X).fit(cov_type='HAC', cov_kwds={'maxlags': L})
        beta = float(model.params.get('D', np.nan))
        pval = float(model.pvalues.get('D', np.nan))
        zscore = float(model.tvalues.get('D', np.nan))
    except Exception:
        beta, pval, zscore = (np.nan, np.nan, np.nan)
    return (beta, pval, zscore)


def _rank_greedy(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    orderL = df.sort_values('z', ascending=False).index.tolist()
    orderS = df.sort_values('z', ascending=True).index.tolist()
    return orderL, orderS


# =============================== #
# 4) LOAD DATA
# =============================== #
base = ROOT_DIR


def load_all():
    log_rets = load_returns(base / "All_Monthly_Log_Return_Data")
    simple_rets = load_returns(base / "All_Monthly_Return_Data")
    return log_rets, simple_rets, list(log_rets.keys())


log_rets, simple_rets, tickers = load_all()


# =============================== #
# 5) MONTE CARLO LAYER — Maximum Entropy Bootstrap (MEB)
# =============================== #

def maximum_entropy_bootstrap_1d(x: np.ndarray,
                                 rng: np.random.Generator,
                                 trim_prop: float = MEB_TRIM) -> np.ndarray:
    """
    One Maximum Entropy Bootstrap (MEB) replicate of a 1D array x.

    Tail extension m_trm is based on a trimmed mean of absolute
    consecutive differences in TIME ORDER, as in Vinod's meboot.
    """
    x = np.asarray(x, dtype=float)
    T = x.size
    if T < 3:
        return x.copy()

    # 1) Sort values and keep the permutation (time indices)
    order = np.argsort(x)
    x_sorted = x[order]

    # 2) Internal interval boundaries z_1,...,z_{T-1}
    mid = 0.5 * (x_sorted[:-1] + x_sorted[1:])  # length T-1

    # 2′) Tail extension m_trm from ABSOLUTE consecutive differences in TIME ORDER
    diffs = np.diff(x)                      # time-order differences
    diffs = np.abs(diffs)                   # absolute
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        m_trm = 0.0
    else:
        n = diffs.size
        k = int(np.floor(trim_prop * n))    # e.g. 10% trimming in each tail
        if 2 * k >= n:
            m_trm = float(diffs.mean())
        else:
            diffs_sorted = np.sort(diffs)
            m_trm = float(diffs_sorted[k:n - k].mean())

    # 3) Full boundaries z_0,...,z_T
    z = np.empty(T + 1, dtype=float)
    z[0] = x_sorted[0] - m_trm
    z[1:T] = mid
    z[T] = x_sorted[-1] + m_trm

    # 4) Draw uniforms, map each to its interval, and sample linearly
    u = rng.uniform(0.0, 1.0, size=T)
    u.sort()
    x_star_sorted = np.empty(T, dtype=float)

    for j in range(T):
        v = T * u[j] - j
        v = float(np.clip(v, 0.0, 1.0))
        width = z[j + 1] - z[j]
        x_star_sorted[j] = z[j] + v * width

    # 5) Restore original time order
    x_star = np.empty(T, dtype=float)
    x_star[order] = x_star_sorted

    # Simple mean-preserving adjustment (global shift)
    mu_orig = float(np.mean(x))
    mu_star = float(np.mean(x_star))
    if np.isfinite(mu_orig) and np.isfinite(mu_star):
        x_star += (mu_orig - mu_star)

    return x_star

def maximum_entropy_bootstrap_series(s: pd.Series,
                                     rng: np.random.Generator,
                                     trim_prop: float = 0.1) -> pd.Series:
    """
    Apply one MEB replicate to a pandas Series (e.g. log returns).
    """
    s = s.dropna().sort_index().astype(float)
    if s.size < 3:
        return s.copy()

    arr_star = maximum_entropy_bootstrap_1d(s.values, rng=rng, trim_prop=trim_prop)
    out = pd.Series(arr_star, index=s.index, name=s.name)
    return out


def simulate_log_returns_meb(log_rets_dict: dict[str, pd.Series],
                             rng: np.random.Generator,
                             no_bootstrap: bool = False,
                             trim_prop: float = 0.1) -> dict[str, pd.Series]:
    """
    Monte Carlo layer based on Maximum Entropy Bootstrap (MEB).
    """
    sim: dict[str, pd.Series] = {}
    for tkr, s in log_rets_dict.items():
        s = s.dropna().sort_index().astype(float)
        if s.empty:
            continue
        if no_bootstrap:
            sim[tkr] = s.copy()
        else:
            sim[tkr] = maximum_entropy_bootstrap_series(
                s, rng=rng, trim_prop=trim_prop
            )
            sim[tkr].name = tkr
    return sim


def log_to_simple_dict(log_rets_dict: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """
    Convert LOG monthly returns to SIMPLE monthly returns.
    """
    return {tkr: (np.exp(s.astype(float)) - 1.0).rename(tkr)
            for tkr, s in log_rets_dict.items()}


def generate_simulated_simple_returns_for_run(
        rng: np.random.Generator,
        zero_noise: bool,
        trim_prop: float = 0.1,
):
    """
    For a given Monte Carlo run (MEB version).
    """
    log_sim = simulate_log_returns_meb(
        log_rets, rng=rng, no_bootstrap=zero_noise, trim_prop=trim_prop
    )
    simple_sim = log_to_simple_dict(log_sim)
    return simple_sim, log_sim


# =============================== #
# 6) RISK-AWARE NORMALISATION (EWMA VOL + RUNNING MEAN)
# =============================== #

def ewma_vol_forecast_series(log_returns: pd.Series,
                             lam: float = EWMA_LAMBDA) -> pd.Series:
    """
    EWMA volatility forecast sigma_t using only r_0,...,r_{t-1}.

    Implementation detail:

      - Use the first N=12 observations to initialise the variance
        as their average squared return.
      - Thereafter, apply the standard RiskMetrics recursion

            sigma_t^2 = lam * sigma_{t-1}^2 + (1-lam) * r_{t-1}^2

        so that sigma_t never uses information from r_t or later.
    """
    s = log_returns.dropna().sort_index().astype(float)
    N = s.size
    if N == 0:
        return s.copy()

    r = s.values
    sig = np.empty(N, dtype=float)

    if N >= 12:
        sigma2_prev = float(np.mean(r[:12] ** 2))
    else:
        sigma2_prev = float(r[0] ** 2)

    # sigma_0 based on initial variance
    sig[0] = np.sqrt(max(sigma2_prev, 1e-12))

    for t in range(1, N):
        sigma2 = lam * sigma2_prev + (1.0 - lam) * float(r[t - 1]) ** 2
        sig[t] = np.sqrt(max(sigma2, 1e-12))
        sigma2_prev = sigma2

    return pd.Series(sig, index=s.index, name=f"{s.name}_ewma_vol")


def risk_adjusted_normalised_series(log_returns: pd.Series,
                                    lam: float = EWMA_LAMBDA,
                                    lookback_years: int | None = LOOKBACK_YEARS) -> pd.Series:
    """
    y_t = (r_t - μ_t) / σ_t

    Logic:
      1) Compute EWMA volatility σ_t for the whole series (one σ_t per month t).
      2) For the mean μ_t, with lookback_years (e.g. 10 years = 120 months):

         - If N < window (lookback period), use the full-sample mean for all t.
         - If N >= window:
             * For the first `window` months (t = 1..window):
                 μ_t is the SAME constant = mean of the first `window` returns.
             * From month t = window+1 onward:
                 μ_t is a rolling mean over the LAST `window` returns
                 (i.e. at t, mean of r_{t-window+1}, ..., r_t).

      So the mean is constant on the first block, then a sliding window afterwards.
    """
    s = log_returns.dropna().sort_index().astype(float)
    N = s.size
    if N == 0:
        return s.copy()

    # 1) EWMA volatility forecast σ_t for each t (uses only past returns internally)
    sig_series = ewma_vol_forecast_series(s, lam=lam)
    sig = sig_series.values
    sig_safe = np.where(sig <= 0.0, np.nan, sig)

    # 2) Mean μ_t
    if lookback_years is None:
        # No explicit lookback: use one global mean for all t
        mu_val = float(s.mean())
        mu_vals = np.full(N, mu_val, dtype=float)

    else:
        window = int(lookback_years * 12)
        if window <= 0:
            mu_val = float(s.mean())
            mu_vals = np.full(N, mu_val, dtype=float)
        elif N < window:
            # Series shorter than lookback: just use full-sample mean for all t
            mu_val = float(s.mean())
            mu_vals = np.full(N, mu_val, dtype=float)
        else:
            # Rolling mean over EXACTLY `window` observations
            # rolling_mean[t] is mean of s[t-window+1 : t+1] for t >= window-1
            rolling_mean = s.rolling(window=window, min_periods=window).mean()

            mu_vals = np.empty(N, dtype=float)

            # First `window` months all use the same mean = mean of first `window` returns
            base_mean = float(rolling_mean.iloc[window - 1])  # mean of s[0:window]
            mu_vals[:window] = base_mean

            # From t = window onward, use the rolling window mean
            # (indices window .. N-1 correspond to months window+1 .. N in 1-based terms)
            mu_vals[window:] = rolling_mean.iloc[window:].to_numpy()

    # 3) Normalised series y_t
    y_vals = (s.values - mu_vals) / sig_safe
    return pd.Series(y_vals, index=s.index, name=f"{s.name}_risknorm")

def build_risk_adjusted_log_dict(log_rets_dict: dict[str, pd.Series],
                                 lam: float = EWMA_LAMBDA,
                                 lookback_years: int | None = LOOKBACK_YEARS) -> dict[str, pd.Series]:
    """
    Apply risk-adjusted normalisation contract by contract.
    """
    out: dict[str, pd.Series] = {}
    for tkr, s in log_rets_dict.items():
        out[tkr] = risk_adjusted_normalised_series(s, lam=lam, lookback_years=lookback_years)
    return out


# =============================== #
# 7) DVR RANKINGS ON CLASSIC VS RISK-ADJUSTED SERIES
# =============================== #
def build_rankings_from_log_rets(log_rets_sim: dict[str, pd.Series],
                                 lookback_years: int) -> tuple[dict, dict]:
    """
    Build monthly DVR rankings for LONG and SHORT directions based on
    (possibly transformed) log-return series.
    Rankings are only needed from FINAL_SIM_START onward.
    """
    tickers_sim = list(log_rets_sim)
    long_rankings, short_rankings = {}, {}
    cur = FINAL_SIM_START
    while cur <= FINAL_END:
        stats_rows = []
        for t in tickers_sim:
            beta, pval, z = dvr_stats(log_rets_sim[t], cur, lookback_years)
            if not np.isfinite(z):
                continue
            stats_rows.append({'ticker': t, 'beta': beta, 'pval': pval, 'z': z})

        if not stats_rows:
            long_rankings[cur] = []
            short_rankings[cur] = []
            cur += relativedelta(months=1)
            continue

        dfm = pd.DataFrame(stats_rows).set_index('ticker')

        if float(SIG_LEVEL) >= 0.999999:
            orderL, orderS = _rank_greedy(dfm)
        else:
            df_valid = dfm[np.isfinite(dfm['beta']) & np.isfinite(dfm['pval'])]
            if df_valid.empty:
                long_rankings[cur] = []
                short_rankings[cur] = []
                cur += relativedelta(months=1)
                continue

            df_long = df_valid[(df_valid['pval'] <= float(SIG_LEVEL)) & (df_valid['beta'] > 0)]
            df_short = df_valid[(df_valid['pval'] <= float(SIG_LEVEL)) & (df_valid['beta'] < 0)]

            orderL = df_long.sort_values('beta', ascending=False).index.tolist()
            orderS = df_short.sort_values('beta', ascending=True).index.tolist()

        long_rankings[cur] = orderL
        short_rankings[cur] = orderS
        cur += relativedelta(months=1)

    return long_rankings, short_rankings


# =============================== #
# 8) TOP-1 MONTHLY SERIES
# =============================== #
def monthly_top1_returns(rankings: dict[pd.Timestamp, list[str]],
                         simple_rets_dict: dict[str, pd.Series],
                         *,
                         direction: str,
                         start_dt: pd.Timestamp,
                         end_dt: pd.Timestamp,
                         entry_cost: float,
                         return_tickers: bool = False) -> pd.Series | tuple[pd.Series, pd.Series]:
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
                # turn long simple return into short simple return
                r = (1.0 / (1.0 + r)) - 1.0
            r_wc = apply_roundtrip_cost(r, entry_cost)
        out_ret.append(r_wc)
        idx.append(dt)
    ret_ser = pd.Series(out_ret, index=idx, name='top1_return')
    if return_tickers:
        tkr_ser = pd.Series(out_tkr, index=idx, name='top1_ticker')
        return ret_ser, tkr_ser
    return ret_ser


def max_drawdown_from_monthly(returns: pd.Series) -> float:
    r = pd.Series(returns, dtype=float).dropna()
    if r.empty:
        return np.nan
    cum = (1.0 + r.values).cumprod()
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    return float(dd.min())  # negative (higher is better)


def df_to_string_centered(df: pd.DataFrame, index: bool = False) -> str:
    """
    Pretty-print a DataFrame with centered columns.
    """
    df_str = df.copy()
    if not index:
        df_str = df_str.reset_index(drop=True)
    df_str = df_str.astype(str)
    cols = list(df_str.columns)
    rows = df_str.values.tolist()
    widths = []
    for i, c in enumerate(cols):
        col_cells = [r[i] for r in rows] if rows else []
        widths.append(max(len(str(c)), *(len(x) for x in col_cells)) if col_cells else len(str(c)))
    header = "  ".join(str(col).center(widths[i]) for i, col in enumerate(cols))
    body_lines = []
    for r in rows:
        body_lines.append("  ".join(str(r[i]).center(widths[i]) for i in range(len(cols))))
    return header + "\n" + "\n".join(body_lines)


def build_monthly_top5_table(df_choices: pd.DataFrame, direction: str, n_runs: int, top_k: int = 5) -> pd.DataFrame:
    sub = df_choices[df_choices['direction'] == direction].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=['Date'] + [f'C{i}' for i in range(1, top_k + 1)] + [f'R{i}' for i in range(1, top_k + 1)]
        )
    sub['date'] = pd.to_datetime(sub['date'])
    dates = pd.date_range(FINAL_SIM_START, FINAL_SIM_END, freq='MS')
    rows = []
    for dt in dates:
        dsel = sub[sub['date'] == dt]
        classic_counts = dsel['classic_ticker'].value_counts()
        risk_counts = dsel['risk_ticker'].value_counts()

        def fmt_top(series):
            series = series.head(top_k)
            labels = []
            for tkr, cnt in series.items():
                pct = 100.0 * float(cnt) / float(n_runs)
                labels.append(f"[{tkr} | {pct:.1f}%]")
            while len(labels) < top_k:
                labels.append("")
            return labels

        C = fmt_top(classic_counts)
        R = fmt_top(risk_counts)
        rows.append({
            'Date': dt.date().isoformat(),
            **{f'C{i + 1}': C[i] for i in range(top_k)},
            **{f'R{i + 1}': R[i] for i in range(top_k)},
        })
    return pd.DataFrame(rows)


# =============================== #
# 9) METRICS & WINDOWS
# =============================== #
@dataclass
class Perf:
    cum_ret: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    mdd: float  # negative by construction (higher is better)
    stdev: float  # monthly stdev


def perf_from_monthly(returns: pd.Series) -> Perf:
    arr = returns.astype(float).values
    if arr.size == 0:
        return Perf(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    sr = sharpe_ratio(arr)
    sor = sortino_ratio(arr)
    cal = calmar_ratio(arr)
    cum = float(np.prod(1.0 + arr) - 1.0)
    years = len(arr) / 12.0
    cagr = (np.prod(1.0 + arr) ** (1 / years) - 1.0) if years > 0 else np.nan
    cum_curve = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(cum_curve)
    dd = cum_curve / peak - 1.0
    mdd = float(np.min(dd)) if dd.size else np.nan  # negative
    sd = float(np.std(arr, ddof=1)) if arr.size >= 2 else np.nan
    return Perf(cum, cagr, sr, sor, cal, mdd, sd)


def three_fixed_windows() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Three non-overlapping 3-year windows covering the full apply period:
    [FINAL_SIM_START .. FINAL_SIM_END], with the final window ending exactly
    at FINAL_SIM_END (e.g. 2016–2018, 2019–2021, 2022–2024).
    """
    wins: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur_start = pd.Timestamp(FINAL_SIM_START)

    for i in range(3):
        if i < 2:
            # 3-year window ending at end-of-month
            w_end = cur_start + relativedelta(years=3) - pd.offsets.MonthEnd(1)
        else:
            # force inclusion of FINAL_SIM_END in the last window
            w_end = pd.Timestamp(FINAL_SIM_END)
        wins.append((cur_start, w_end))
        cur_start = w_end + pd.offsets.MonthBegin(1)

    return wins


def compute_metrics_correlation(df_metrics: pd.DataFrame,
                                metric_cols: list[str]) -> pd.DataFrame:
    """
    Safe correlation: drops metric columns with < 2 finite observations
    and suppresses numpy 'degrees of freedom' warnings.
    """
    m = df_metrics[metric_cols].astype(float)

    # keep only columns with at least 2 valid values
    valid_cols = [c for c in metric_cols if np.isfinite(m[c]).sum() >= 2]
    if len(valid_cols) < 2:
        return pd.DataFrame()

    m = m[valid_cols]

    with np.errstate(invalid='ignore', divide='ignore'):
        corr = m.corr(method='pearson')

    return corr


def save_correlation_csv(corr_df: pd.DataFrame, out_path: Path) -> None:
    corr_df.to_csv(out_path)


def save_correlation_heatmap(corr_df: pd.DataFrame, out_path_pdf: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_df.values, vmin=-1, vmax=1, cmap='coolwarm')
    metric_cols = list(corr_df.columns)
    ax.set_xticks(range(len(metric_cols)))
    ax.set_yticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=45, ha='right')
    ax.set_yticklabels(metric_cols)
    for i in range(len(metric_cols)):
        for j in range(len(metric_cols)):
            val = corr_df.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation of Performance Metrics")
    fig.tight_layout()
    fig.savefig(out_path_pdf)
    plt.close(fig)


def paired_two_sided(left: np.ndarray, right: np.ndarray):
    """
    Paired two-sided t-test: tests H0: mean(right - left) = 0.
    Returns (t, p_two_sided, n).
    """
    mask = np.isfinite(left) & np.isfinite(right)
    n = int(mask.sum())
    if n < 3:
        return None, None, n
    t, p = st.ttest_rel(right[mask], left[mask], alternative='two-sided')
    return float(t), float(p), n


def one_sample_two_sided(x: np.ndarray, popmean: float = 0.0):
    """
    One-sample two-sided t-test: tests H0: mean(x) = popmean.
    Returns (t, p_two_sided, n).
    """
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    x = x[mask]
    n = int(x.size)
    if n < 3:
        return None, None, n
    t, p = st.ttest_1samp(x, popmean, alternative='two-sided')
    return float(t), float(p), n


def print_sig_line_generic(metric_name: str,
                           left_label: str, right_label: str,
                           t, p, n: int):
    if n < 3:
        print(f"    {right_label} vs {left_label} — {metric_name:<11}: insufficient valid pairs.")
        return
    concl = _sig_conclusion(t, p, left_label, right_label, alpha=SIGNIF_TWO_SIDED_ALPHA)
    print(f"    {right_label} vs {left_label} — {metric_name:<11}: "
          f"n={n:4d} | t={_fmt(t):>6} | p(two-sided)={_fmt_p(p, SIGNIF_TWO_SIDED_ALPHA):>8} -> {concl}")


# =============================== #
# 10) MONTE CARLO: CLASSIC VS RISK
# =============================== #
def run_monte_carlo_both(n_runs=MC_RUNS, save_series=SAVE_SERIES, zero_noise=ZERO_NOISE):
    OUT_DIR_MC.mkdir(parents=True, exist_ok=True)
    # dedicated subfolder for all correlation outputs
    corr_dir = OUT_DIR_MC / "correlation_matrices"
    corr_dir.mkdir(parents=True, exist_ok=True)

    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st_ in self.streams:
                st_.write(s)
            return len(s)

        def flush(self):
            for st_ in self.streams:
                try:
                    st_.flush()
                except Exception:
                    pass

    tee = Tee(sys.__stdout__, buf)

    metrics_rows: list[dict] = []
    choice_records = []

    with redirect_stdout(tee):
        rng = np.random.default_rng(RNG_SEED)
        t0 = time.time()
        section(
            f"Starting Monte Carlo (MEB) — BOTH directions (classic vs risk-adjusted DVR) | "
            f"runs={n_runs:,} | zero_noise={zero_noise}"
        )

        month_grid = pd.date_range(FINAL_SIM_START, FINAL_SIM_END, freq='MS')
        splits = three_fixed_windows()

        STRATS_BY_DIR = {
            'long': ('classic', 'risk', 'weighted'),
            'short': ('classic', 'risk')
        }

        full_metrics = {
            'long': {'classic': [], 'risk': [], 'weighted': []},
            'short': {'classic': [], 'risk': []}
        }
        sub_metrics = {
            'long': {'classic': defaultdict(list),
                     'risk': defaultdict(list),
                     'weighted': defaultdict(list)},
            'short': {'classic': defaultdict(list),
                      'risk': defaultdict(list)}
        }
        pos_counts = {
            'long': {'classic': 0, 'risk': 0, 'weighted': 0},
            'short': {'classic': 0, 'risk': 0}
        }
        neg_counts = {
            'long': {'classic': 0, 'risk': 0, 'weighted': 0},
            'short': {'classic': 0, 'risk': 0}
        }

        for run in range(1, n_runs + 1):
            # 1) simulate log returns (MEB)
            simple_sim, log_sim = generate_simulated_simple_returns_for_run(
                rng, zero_noise
            )

            # 2) classic vs risk-adjusted log returns
            log_sim_classic = log_sim
            log_sim_risk = build_risk_adjusted_log_dict(
                log_sim, lam=EWMA_LAMBDA, lookback_years=LOOKBACK_YEARS
            )

            # 3) DVR rankings
            L_rank_classic, S_rank_classic = build_rankings_from_log_rets(
                log_sim_classic, LOOKBACK_YEARS
            )
            L_rank_risk, S_rank_risk = build_rankings_from_log_rets(
                log_sim_risk, LOOKBACK_YEARS
            )

            # 4) Long-only EW benchmarks (gross long) on apply window
            bench_apply_gross_long = build_equal_weight_benchmark(
                simple_sim, FINAL_SIM_START, FINAL_SIM_END
            )
            bench_apply_long_net = long_benchmark_net(
                bench_apply_gross_long, TRADING_COST
            )

            weighted_long_series = bench_apply_long_net.copy().rename("equal_weight_long_net")

            for direction in ('long', 'short'):
                if direction == 'long':
                    rank_classic = L_rank_classic
                    rank_risk = L_rank_risk
                else:
                    rank_classic = S_rank_classic
                    rank_risk = S_rank_risk

                # Top-1 Classic & Risk (NET) with direction-aware P&L
                top1_classic, tkr_classic = monthly_top1_returns(
                    rank_classic, simple_sim, direction=direction,
                    start_dt=FINAL_SIM_START, end_dt=FINAL_SIM_END,
                    entry_cost=TRADING_COST, return_tickers=True
                )
                top1_risk, tkr_risk = monthly_top1_returns(
                    rank_risk, simple_sim, direction=direction,
                    start_dt=FINAL_SIM_START, end_dt=FINAL_SIM_END,
                    entry_cost=TRADING_COST, return_tickers=True
                )

                # NAVs (Top-1 only) for display context
                nav_classic = nav_from_returns_on_grid(
                    top1_classic, START_VALUE, month_grid
                )
                nav_risk = nav_from_returns_on_grid(
                    top1_risk, START_VALUE, month_grid
                )

                # Record choices for Top-5 tables
                for dt in month_grid:
                    choice_records.append({
                        'direction': direction,
                        'run': run,
                        'date': dt.date().isoformat(),
                        'classic_ticker': tkr_classic.get(dt, np.nan)
                        if not tkr_classic.empty else np.nan,
                        'risk_ticker': tkr_risk.get(dt, np.nan)
                        if not tkr_risk.empty else np.nan,
                        'classic_value': float(nav_classic.get(dt, np.nan)),
                        'risk_value': float(nav_risk.get(dt, np.nan)),
                    })

                # Save series (per direction)
                if save_series:
                    ddir = OUT_DIR_MC / f"{direction}_run_{run:03d}"
                    ddir.mkdir(parents=True, exist_ok=True)
                    top1_classic.to_csv(ddir / "top1_classic_monthly.csv")
                    top1_risk.to_csv(ddir / "top1_risk_monthly.csv")
                    if direction == 'long':
                        weighted_long_series.to_csv(
                            ddir / "weighted_equal_monthly.csv"
                        )

                # Full-period metrics (port-only stats)
                pc = perf_from_monthly(top1_classic)
                pr = perf_from_monthly(top1_risk)
                full_metrics[direction]['classic'].append(pc)
                full_metrics[direction]['risk'].append(pr)
                pos_counts[direction]['classic'] += int(pc.cum_ret > 0)
                neg_counts[direction]['classic'] += int(pc.cum_ret <= 0)
                pos_counts[direction]['risk'] += int(pr.cum_ret > 0)
                neg_counts[direction]['risk'] += int(pr.cum_ret <= 0)

                if direction == 'long':
                    pw = perf_from_monthly(weighted_long_series)
                    full_metrics['long']['weighted'].append(pw)
                    pos_counts['long']['weighted'] += int(pw.cum_ret > 0)
                    neg_counts['long']['weighted'] += int(pw.cum_ret <= 0)

                # ---- Rows for correlation / metrics CSV
                metrics_rows.append({
                    'run': run, 'direction': direction, 'strategy': 'classic',
                    'sharpe': pc.sharpe, 'sortino': pc.sortino,
                    'treynor': treynor_ratio_series_absbeta(
                        top1_classic, bench_apply_long_net
                    ),
                    'treynor_raw': treynor_ratio_series(
                        top1_classic, bench_apply_long_net
                    ),
                    'information': information_ratio_series(
                        top1_classic, bench_apply_long_net
                    ),
                    'calmar': pc.calmar, 'cum_ret': pc.cum_ret,
                    'mdd': pc.mdd if np.isfinite(pc.mdd) else np.nan,
                    'mdd_mag': abs(pc.mdd) if np.isfinite(pc.mdd) else np.nan,
                    'excess_ret_m': mean_excess_return_series(
                        top1_classic, bench_apply_long_net
                    ),
                })
                metrics_rows.append({
                    'run': run, 'direction': direction, 'strategy': 'risk',
                    'sharpe': pr.sharpe, 'sortino': pr.sortino,
                    'treynor': treynor_ratio_series_absbeta(
                        top1_risk, bench_apply_long_net
                    ),
                    'treynor_raw': treynor_ratio_series(
                        top1_risk, bench_apply_long_net
                    ),
                    'information': information_ratio_series(
                        top1_risk, bench_apply_long_net
                    ),
                    'calmar': pr.calmar, 'cum_ret': pr.cum_ret,
                    'mdd': pr.mdd if np.isfinite(pr.mdd) else np.nan,
                    'mdd_mag': abs(pr.mdd) if np.isfinite(pr.mdd) else np.nan,
                    'excess_ret_m': mean_excess_return_series(
                        top1_risk, bench_apply_long_net
                    ),
                })
                if direction == 'long':
                    metrics_rows.append({
                        'run': run, 'direction': 'long', 'strategy': 'weighted',
                        'sharpe': pw.sharpe, 'sortino': pw.sortino,
                        'treynor': treynor_ratio_series_absbeta(
                            weighted_long_series, bench_apply_long_net
                        ),
                        'treynor_raw': treynor_ratio_series(
                            weighted_long_series, bench_apply_long_net
                        ),
                        'information': information_ratio_series(
                            weighted_long_series, bench_apply_long_net
                        ),
                        'calmar': pw.calmar, 'cum_ret': pw.cum_ret,
                        'mdd': pw.mdd if np.isfinite(pw.mdd) else np.nan,
                        'mdd_mag': abs(pw.mdd) if np.isfinite(pw.mdd) else np.nan,
                        'excess_ret_m': mean_excess_return_series(
                            weighted_long_series, bench_apply_long_net
                        ),
                    })

                # ---- Subperiods (three 3y windows)
                for i, (s, e) in enumerate(splits, start=1):
                    sc = top1_classic[
                        (top1_classic.index >= s) & (top1_classic.index <= e)
                    ]
                    sr_ = top1_risk[
                        (top1_risk.index >= s) & (top1_risk.index <= e)
                    ]
                    sub_metrics[direction]['classic'][i].append(
                        perf_from_monthly(sc)
                    )
                    sub_metrics[direction]['risk'][i].append(
                        perf_from_monthly(sr_)
                    )

                    if direction == 'long':
                        sw = weighted_long_series[
                            (weighted_long_series.index >= s)
                            & (weighted_long_series.index <= e)
                        ]
                        sub_metrics['long']['weighted'][i].append(
                            perf_from_monthly(sw)
                        )

            if (run % 10) == 0:
                elapsed = time.time() - t0
                print(
                    f"  Progress: completed {run:>4}/{n_runs:<4} runs | "
                    f"elapsed {time.strftime('%H:%M:%S', time.gmtime(elapsed))}"
                )

        # === Top-5 selection tables ===
        if SAVE_TICKER_CHOICES_CSV and choice_records:
            df_choices = pd.DataFrame(choice_records)
            section("Top-5 selection share tables (per month)")
            for direction in ('long', 'short'):
                tbl = build_monthly_top5_table(df_choices, direction, n_runs, top_k=5)
                out_path = OUT_DIR_MC / f"top5_selections_table_{direction}.csv"
                tbl.to_csv(out_path, index=False)
                print(
                    f"Direction: {direction.upper()} — Top-5 per method "
                    f"(Classic first 5 columns, Risk-adjusted next 5)"
                )
                print(df_to_string_centered(tbl.head(12), index=False) + "\n")
                print(f"Saved full table to: {out_path}\n")

        # === Safe stats helpers ===
        def _finite(x) -> np.ndarray:
            arr = np.asarray(x, dtype=float)
            return arr[np.isfinite(arr)]

        def safe_mean(x) -> float:
            v = _finite(x)
            return float(v.mean()) if v.size > 0 else np.nan

        def safe_median(x) -> float:
            v = _finite(x)
            return float(np.median(v)) if v.size > 0 else np.nan

        def safe_std(x, ddof=1) -> float:
            v = _finite(x)
            return float(v.std(ddof=ddof)) if v.size > ddof else np.nan

        def agg(perf_list: list[Perf]) -> dict[str, float]:
            if not perf_list:
                keys = ['cum_ret', 'cagr', 'sharpe',
                        'sortino', 'calmar', 'mdd', 'stdev']
                return ({f"{k}_mean": np.nan for k in keys}
                        | {f"{k}_median": np.nan for k in keys}
                        | {f"{k}_std": np.nan for k in keys})
            out = {}
            for k in ['cum_ret', 'cagr', 'sharpe',
                      'sortino', 'calmar', 'mdd', 'stdev']:
                x = np.array([getattr(p, k) for p in perf_list], dtype=float)
                out[f'{k}_mean'] = safe_mean(x)
                out[f'{k}_median'] = safe_median(x)
                out[f'{k}_std'] = safe_std(x, ddof=1)
            return out

        df_metrics_all = pd.DataFrame(metrics_rows) if metrics_rows else None

        # === Print LONG then SHORT summaries, subperiods & significance
        for direction in ('long', 'short'):
            n_runs_local = n_runs
            strats = STRATS_BY_DIR[direction]

            # Aggregates
            agg_full = {s: agg(full_metrics[direction][s]) for s in strats}

            section(f"MONTE CARLO SUMMARY — Direction: {direction.upper()}")

            print("Config")
            print(f"  Lookback: {LOOKBACK_YEARS}y")
            print(
                f"  Apply   : {FINAL_SIM_START.date()} → {FINAL_SIM_END.date()}"
            )
            print(
                f"  SIG     : {SIG_LEVEL}   | Entry cost={TRADING_COST:.4f}"
            )
            print(f"  EWMA λ  : {EWMA_LAMBDA}\n")

            # Outcome counts
            print("Outcome counts (Top-1 over apply)")
            hdr = f"  {'Strategy':<12} {'Pos(+)':>7} {'Neg(-)':>7}"
            print(hdr)
            print(f"  {'-' * 12} {'-' * 7:>7} {'-' * 7:>7}")
            print(
                f"  {'Classic':<12} {pos_counts[direction]['classic']:>7} "
                f"{neg_counts[direction]['classic']:>7}"
            )
            print(
                f"  {'Risk-adj':<12} {pos_counts[direction]['risk']:>7} "
                f"{neg_counts[direction]['risk']:>7}"
            )
            if 'weighted' in strats:
                print(
                    f"  {'Weighted':<12} {pos_counts[direction]['weighted']:>7} "
                    f"{neg_counts[direction]['weighted']:>7}"
                )
            print("")

            # ========= Full-period summary tables =========
            def pretty_full(d, name):
                print(
                    f"{name} — Full Apply Period (across {n_runs_local} runs)"
                )

                def fmt4(v):
                    return "" if v is None or not np.isfinite(v) else f"{v:.4f}"

                def fmt3(v):
                    return "" if v is None or not np.isfinite(v) else f"{v:.3f}"

                rows = []
                # Only: Return, CAGR, Sharpe, MDD
                rows.append({
                    "Metric": "Return",
                    "Mean": fmt4(d['cum_ret_mean']),
                    "Median": fmt4(d['cum_ret_median']),
                    "SD": fmt4(d['cum_ret_std']),
                })
                rows.append({
                    "Metric": "CAGR",
                    "Mean": fmt4(d['cagr_mean']),
                    "Median": fmt4(d['cagr_median']),
                    "SD": fmt4(d['cagr_std']),
                })
                rows.append({
                    "Metric": "Sharpe",
                    "Mean": fmt3(d['sharpe_mean']),
                    "Median": fmt3(d['sharpe_median']),
                    "SD": fmt3(d['sharpe_std']),
                })
                rows.append({
                    "Metric": "MDD",
                    "Mean": fmt3(d['mdd_mean']),
                    "Median": fmt3(d['mdd_median']),
                    "SD": fmt3(d['mdd_std']),
                })

                df_tab = pd.DataFrame(rows, columns=["Metric", "Mean", "Median", "SD"])
                print(df_to_string_centered(df_tab, index=False))
                print("")

            pretty_full(agg_full['classic'], "CLASSIC")
            pretty_full(agg_full['risk'], "RISK-ADJUSTED")
            if 'weighted' in strats:
                pretty_full(agg_full['weighted'], "WEIGHTED")

            # ========= Subperiods — Three fixed 3-year windows =========
            print("Subperiods — Three fixed 3-year windows")
            for i, (s, e) in enumerate(splits, start=1):
                ac = agg(sub_metrics[direction]['classic'][i])
                ar = agg(sub_metrics[direction]['risk'][i])

                print(f"\n  Window {i}: {s.date()} → {e.date()}")
                if direction == 'short':
                    print(
                        f"    {'Metric':<14} {'Classic':>12} {'Risk-adj':>12}"
                    )
                    print(
                        f"    {'-' * 14:<14} {'-' * 12:>12} {'-' * 12:>12}"
                    )

                    def line2(label, f, g, fmt=".4f"):
                        def fmt1(x):
                            return ("{:" + fmt + "}").format(float(x)) if np.isfinite(x) else "nan"
                        print(
                            f"    {label:<14} {fmt1(f):>12} {fmt1(g):>12}"
                        )

                    # Only: Return, CAGR, Sharpe, MDD
                    line2("Return mean", ac['cum_ret_mean'], ar['cum_ret_mean'])
                    line2("CAGR mean", ac['cagr_mean'], ar['cagr_mean'])
                    line2("Sharpe mean", ac['sharpe_mean'], ar['sharpe_mean'], fmt=".3f")
                    line2("MDD mean", ac['mdd_mean'], ar['mdd_mean'], fmt=".3f")
                else:
                    aw = agg(sub_metrics['long']['weighted'][i])
                    print(
                        f"    {'Metric':<14} {'Classic':>12} {'Risk-adj':>12} {'Weighted':>12}"
                    )
                    print(
                        f"    {'-' * 14:<14} {'-' * 12:>12} {'-' * 12:>12} {'-' * 12:>12}"
                    )

                    def line3(label, f, g, w, fmt=".4f"):
                        def fmt1(x):
                            return ("{:" + fmt + "}").format(float(x)) if np.isfinite(x) else "nan"
                        print(
                            f"    {label:<14} {fmt1(f):>12} {fmt1(g):>12} {fmt1(w):>12}"
                        )

                    # Only: Return, CAGR, Sharpe, MDD
                    line3("Return mean", ac['cum_ret_mean'],
                          ar['cum_ret_mean'], aw['cum_ret_mean'])
                    line3("CAGR mean", ac['cagr_mean'],
                          ar['cagr_mean'], aw['cagr_mean'])
                    line3("Sharpe mean", ac['sharpe_mean'],
                          ar['sharpe_mean'], aw['sharpe_mean'], fmt=".3f")
                    line3("MDD mean", ac['mdd_mean'],
                          ar['mdd_mean'], aw['mdd_mean'], fmt=".3f")

            # ===================== Significance — Paired, two-sided =====================
            section(
                f"Significance — Paired, two-sided (α = {SIGNIF_TWO_SIDED_ALPHA:.3f})"
            )

            # Build arrays for paired tests
            c_list = full_metrics[direction]['classic']
            r_list = full_metrics[direction]['risk']
            arr_ = lambda attr, L: np.array(
                [getattr(p, attr) for p in L], dtype=float
            )

            def sig_pair_block(left_label, right_label,
                               L_list, R_list):
                """
                Compact significance table for two-sided paired tests:
                  Return, CAGR, Sharpe, MDD.
                """
                print(
                    f"[ {left_label.upper()}  →  {right_label.upper()} ]  "
                    f"(two-sided α={SIGNIF_TWO_SIDED_ALPHA:.3f})"
                )

                rows = []

                def add_row(metric_name, t, p, n):
                    rows.append({
                        "Metric": metric_name,
                        "n": ("" if n is None else f"{n:d}"),
                        "t": _fmt(t),
                        "p(two-sided)": _fmt_p(
                            p, SIGNIF_TWO_SIDED_ALPHA
                        ),
                        "Conclusion": _sig_conclusion(
                            t, p, left_label, right_label,
                            alpha=SIGNIF_TWO_SIDED_ALPHA
                        ),
                    })

                # Port-only metrics: Return, CAGR, Sharpe, MDD
                for mname, attr in [
                    ("Return", "cum_ret"),
                    ("CAGR", "cagr"),
                    ("Sharpe", "sharpe"),
                    ("MDD", "mdd"),
                ]:
                    t, p, n = paired_two_sided(
                        arr_(attr, L_list), arr_(attr, R_list)
                    )
                    add_row(mname, t, p, n)

                df_out = pd.DataFrame(
                    rows,
                    columns=["Metric", "n", "t", "p(two-sided)", "Conclusion"],
                )
                print(df_to_string_centered(df_out, index=False))
                print("")

            if direction == 'long':
                # Long-side comparisons, including the weighted long benchmark
                w_list = full_metrics['long']['weighted']

                sig_pair_block("classic long", "risk long",
                               c_list, r_list)
                sig_pair_block("classic long", "weighted long",
                               c_list, w_list)
                sig_pair_block("risk long", "weighted long",
                               r_list, w_list)

            if direction == 'short':
                # For shorts, benchmark against the SAME weighted long portfolio
                w_list = full_metrics['long']['weighted']

                # 1) Classic vs risk on the short side
                sig_pair_block("classic short", "risk short",
                               c_list, r_list)

                # 2) Short classic vs long weighted benchmark
                sig_pair_block("classic short", "weighted long",
                               c_list, w_list)

                # 3) Short risk vs long weighted benchmark
                sig_pair_block("risk short", "weighted long",
                               r_list, w_list)

            print(rule("─"))
            print("")

        total_elapsed = time.time() - t0
        section("Completed")
        print(
            f"Total runtime: {time.strftime('%H:%M:%S', time.gmtime(total_elapsed))}\n"
        )

        # === Save metrics table + correlations ===
        if metrics_rows:
            df_metrics = pd.DataFrame(metrics_rows)
            # annualized excess return
            df_metrics['excess_ret_ann'] = 12.0 * df_metrics['excess_ret_m']
            df_metrics.to_csv(
                OUT_DIR_MC / "performance_metrics_by_portfolio.csv",
                index=False,
            )

            metric_cols = [
                "sharpe", "sortino", "treynor", "information",
                "calmar", "cum_ret", "mdd", "excess_ret_ann",
            ]

            # Per-direction Sharpe <-> CumRet
            for d in ["long", "short"]:
                sub = df_metrics[df_metrics["direction"] == d]
                if sub.shape[0] < 2:
                    continue
                with np.errstate(invalid='ignore', divide='ignore'):
                    corr = sub[metric_cols].astype(float).corr()
                if ("sharpe" in corr.index) and ("cum_ret" in corr.columns):
                    corr.to_csv(corr_dir / f"metrics_correlation_matrix_{d}.csv")

            # Per-direction & per-strategy
            for d in ["long", "short"]:
                strategies = (
                    ["classic", "risk", "weighted"]
                    if d == "long" else ["classic", "risk"]
                )
                for s in strategies:
                    sub = df_metrics[
                        (df_metrics["direction"] == d)
                        & (df_metrics["strategy"] == s)
                    ]
                    if sub.shape[0] < 2:
                        continue
                    with np.errstate(invalid='ignore', divide='ignore'):
                        corr = sub[metric_cols].astype(float).corr()
                    if ("sharpe" in corr.index) and ("cum_ret" in corr.columns):
                        corr.to_csv(
                            corr_dir
                            / f"metrics_correlation_matrix_{d}_{s}.csv"
                        )

            # Global correlation matrix (all directions/strategies)
            if GENERATE_METRICS_CORR_CSV or GENERATE_METRICS_CORR_HEATMAP:
                corr = compute_metrics_correlation(df_metrics, metric_cols)
                if not corr.empty:
                    if GENERATE_METRICS_CORR_CSV:
                        save_correlation_csv(
                            corr, corr_dir / "metrics_correlation_matrix.csv"
                        )
                    if GENERATE_METRICS_CORR_HEATMAP:
                        save_correlation_heatmap(
                            corr, corr_dir / "metrics_correlation_heatmap.pdf"
                        )

    # === Write terminal printout to file ===
    with open(OUT_DIR_MC / "console_report.txt", "w", encoding="utf-8") as f:
        f.write(buf.getvalue())


# =============================== #
# 11) RUN
# =============================== #
if __name__ == "__main__":
    run_monte_carlo_both(n_runs=MC_RUNS, save_series=SAVE_SERIES, zero_noise=ZERO_NOISE)
