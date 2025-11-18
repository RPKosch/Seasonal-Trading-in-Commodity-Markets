import sys
import math
import argparse
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import calendar
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, breaks_cusumolsresid, het_arch

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 200)

# ------------------------- logging -------------------------

def init_logger(reports_root: Path):
    reports_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = reports_root / f"run_{ts}.log"

    class Tee:
        def __init__(self, path):
            self.f = open(path, "w", encoding="utf-8")
        def write(self, msg):
            sys.__stdout__.write(msg)
            self.f.write(msg)
        def flush(self):
            sys.__stdout__.flush()
            self.f.flush()

    sys.stdout = Tee(logfile)
    sys.stderr = sys.stdout
    return logfile

# ------------------------- helpers -------------------------

def stars_from_p(p) -> str:
    """Significance stars for a two-sided p-value: * p<0.10, ** p<0.05, *** p<0.01"""
    try:
        p = float(p)
    except Exception:
        return ""
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""

def coerce_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[^\d\-\+\.,eE]", "", regex=True)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def as_float(x) -> float:
    try:
        return float(np.asarray(x).ravel()[0])
    except Exception:
        return float("nan")

def df_to_md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except Exception:
        return df.to_string(index=False)

def month_label(m: int) -> str:
    return f"{m:02d}-{calendar.month_abbr[m]}"

def save_debug_dump(ticker: str, df_t: pd.DataFrame, X: pd.DataFrame, y: pd.Series, reports_root: Path):
    # design snapshots
    X_out = X.copy()
    X_out.insert(0, "orig_index", X.index)
    X_out.to_csv(reports_root / f"{ticker}_design_X.csv", index=False)
    y_out = pd.DataFrame({"orig_index": y.index, "y_return": y.values})
    y_out.to_csv(reports_root / f"{ticker}_design_y.csv", index=False)

# ------------------------- IO -------------------------

def load_and_normalize(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"ticker", "year", "month", "return"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{fp.name}: missing required columns: {missing}")

    keep = [c for c in ["ticker","year","month","avg_vol","return",
                        "start_date","start_value","end_date","end_value","contract_file"]
            if c in df.columns]
    df = df[keep].copy()

    df["year"] = coerce_numeric(df["year"]).astype("Int64")
    df["month"] = coerce_numeric(df["month"]).astype("Int64")
    df["return"] = coerce_numeric(df["return"])
    if "avg_vol" in df.columns:
        df["avg_vol"] = coerce_numeric(df["avg_vol"])

    df["date"] = pd.to_datetime(
        dict(year=df["year"].astype("float").astype("Int64"),
             month=df["month"].astype("float").astype("Int64"),
             day=1),
        errors="coerce"
    )
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)
    return df

def arch_lm_test(resid: np.ndarray, lags: int = 12) -> Dict[str, float]:
    """
    Engle's ARCH LM test on residuals. Small p-values indicate conditional heteroskedasticity
    (volatility clustering). 'lags' is the number of ARCH lags to test; for monthly data 12 is sensible.
    """
    r = np.asarray(resid, dtype=float).ravel()
    try:
        lm, lm_p, fstat, f_p = het_arch(r, nlags=lags)
        return dict(arch_lm=as_float(lm), arch_lm_p=as_float(lm_p),
                    arch_f=as_float(fstat), arch_fp=as_float(f_p),
                    arch_lags=int(lags))
    except Exception as e:
        return dict(arch_error=str(e))


# ------------------------- tests & designs -------------------------

def adf_kpss_tests(x: pd.Series) -> Dict[str, float]:
    xv = coerce_numeric(x).dropna().values
    out = {}
    if len(xv) < 12:
        return {"note": "too few observations for reliable unit-root tests"}

    # ADF
    adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(xv, autolag="AIC")
    out.update(dict(
        adf_stat=as_float(adf_stat), adf_p=as_float(adf_p),
        adf_lags=int(adf_lags), adf_nobs=int(adf_nobs),
        adf_crit=adf_crit, adf_sig=stars_from_p(adf_p)
    ))
    # KPSS
    try:
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(xv, regression="c", nlags="auto")
        out.update(dict(
            kpss_stat=as_float(kpss_stat), kpss_p=as_float(kpss_p),
            kpss_lags=int(kpss_lags), kpss_crit=kpss_crit, kpss_sig=stars_from_p(kpss_p)
        ))
    except Exception as e:
        out.update(dict(kpss_error=str(e)))
    return out

def build_baseline_design(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Intercept + 11 month dummies (January omitted) used for diagnostics residuals and condition number."""
    y = coerce_numeric(df["return"])
    M = pd.get_dummies(df["month"].astype("Int64"), prefix="m", drop_first=True, dtype=float)  # m_2..m_12
    X = sm.add_constant(M, has_constant="add").astype(float)
    mask = y.replace([np.inf, -np.inf], np.nan).notna()
    mask &= X.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    return X.loc[mask].astype(float), y.loc[mask].astype(float)

def build_effects_design(df: pd.DataFrame, ref_month: int = 12) -> pd.DataFrame:
    """Effects coding (sum-to-zero w.r.t intercept)."""
    m = df["month"].astype("Int64")
    n = len(m)
    cols = []
    E = np.zeros((n, 0), dtype=float)
    for j in range(1, 13):
        if j == ref_month:
            continue
        col = np.where(m == j, 1.0, np.where(m == ref_month, -1.0, 0.0))
        E = np.column_stack([E, col])
        cols.append(f"e_{j}")
    X = pd.DataFrame(E, columns=cols, index=df.index)
    X = sm.add_constant(X, has_constant="add")
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    return X

def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """Compute VIFs for each column EXCEPT the intercept."""
    Xv = X.copy()
    for c in list(Xv.columns):
        if c.lower() in ("const", "intercept"):
            Xv = Xv.drop(columns=[c])
            break
    Xv = Xv.replace([np.inf, -np.inf], np.nan).dropna()
    if Xv.shape[1] == 0 or Xv.shape[0] <= Xv.shape[1]:
        return pd.DataFrame({"variable": Xv.columns, "VIF": [np.nan]*len(Xv.columns)})
    arr = np.asarray(Xv.values, dtype=float)
    out = []
    for i, c in enumerate(Xv.columns):
        try:
            v = variance_inflation_factor(arr, i)
        except Exception:
            v = np.nan
        out.append((c, v))
    return pd.DataFrame(out, columns=["variable","VIF"])

def bp_white_tests(resid: np.ndarray, X: pd.DataFrame) -> Dict[str, float]:
    r = np.asarray(resid, dtype=float)
    Xa = np.asarray(X, dtype=float)
    lm, lmp, fval, fp = het_breuschpagan(r, Xa)
    lm_w, lmp_w, f_w, fp_w = het_white(r, Xa)
    return dict(bp_lm=as_float(lm), bp_p=as_float(lmp), bp_f=as_float(fval), bp_fp=as_float(fp),
                white_lm=as_float(lm_w), white_p=as_float(lmp_w), white_f=as_float(f_w), white_fp=as_float(fp_w))

def cusum_test(resid: np.ndarray, X: pd.DataFrame) -> Dict[str, float]:
    r = np.asarray(resid, dtype=float).ravel()
    k = int(X.shape[1]) if X is not None else 0
    try:
        stat, pval, _ = breaks_cusumolsresid(r, ddof=k)
        return dict(cusum_stat=float(stat), cusum_p=float(pval))
    except Exception as e:
        return dict(cusum_error=str(e))

def chow_search(y: pd.Series, X: pd.DataFrame, min_seg: int = 24) -> Dict[str, float]:
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)
    n = len(y); k = X.shape[1]
    if n < (2*min_seg + 1):
        return {"note": "sample too small for rolling Chow with given min_seg"}
    model_full = sm.OLS(y.values, X.values).fit()
    RSS_full = float(np.sum(np.asarray(model_full.resid, dtype=float)**2))
    best = dict(F=-np.inf, p=np.nan, break_at=np.nan, n_left=np.nan, n_right=np.nan)
    for b in range(min_seg, n - min_seg):
        try:
            y1, X1 = y.iloc[:b].values, X.iloc[:b, :].values
            y2, X2 = y.iloc[b:].values, X.iloc[b:, :].values
            m1 = sm.OLS(y1, X1).fit()
            m2 = sm.OLS(y2, X2).fit()
            RSS1 = float(np.sum(np.asarray(m1.resid, dtype=float)**2))
            RSS2 = float(np.sum(np.asarray(m2.resid, dtype=float)**2))
            df1 = k
            df2 = (len(y1) + len(y2) - 2*k)
            if df2 <= 0:
                continue
            F = ((RSS_full - (RSS1 + RSS2)) / df1) / ((RSS1 + RSS2) / df2)
            p = stats.f.sf(F, df1, df2)
            F = as_float(F); p = as_float(p)
            if math.isfinite(F) and F > best["F"]:
                best = dict(F=F, p=p, break_at=int(b), n_left=int(len(y1)), n_right=int(len(y2)))
        except Exception:
            continue
    return best

def summarize_series(x: pd.Series) -> pd.DataFrame:
    x = coerce_numeric(x).dropna()
    if len(x) == 0:
        return pd.DataFrame([{"count":0,"mean":np.nan,"std":np.nan,"skew":np.nan,"kurtosis_excess":np.nan,
                              "min":np.nan,"q25":np.nan,"median":np.nan,"q75":np.nan,"max":np.nan}])
    qs = x.quantile([0.25, 0.5, 0.75])
    return pd.DataFrame({
        "count":[len(x)], "mean":[x.mean()], "std":[x.std(ddof=1)],
        "skew":[stats.skew(x, bias=False)], "kurtosis_excess":[stats.kurtosis(x, fisher=True, bias=False)],
        "min":[x.min()], "q25":[qs.loc[0.25]], "median":[qs.loc[0.5]], "q75":[qs.loc[0.75]], "max":[x.max()]
    })

def month_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("month")["return"]
    n = g.size(); m = g.mean(); s = g.std(ddof=1)
    n_np, m_np, s_np = n.to_numpy(), m.to_numpy(), s.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = s_np / np.sqrt(n_np)
        t_np = np.where(denom > 0, m_np / denom, np.nan)
    dfree = np.maximum(n_np - 1, 1)
    p_np = 2 * stats.t.sf(np.abs(t_np), df=dfree)
    tbl = pd.DataFrame({
        "month": n.index.to_numpy().astype(int),
        "n": n_np, "mean": m_np, "std": s_np,
        "t_stat": t_np, "p_value": p_np
    })
    tbl["sig"] = [stars_from_p(p) for p in tbl["p_value"]]
    tbl["label"] = tbl["month"].apply(month_label)
    return tbl.sort_values("month").reset_index(drop=True)

def month_vs_rest_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each month m, test returns in month m against all non-m months (two-sample Welch t-test).
    Reports n_m, n_rest, mean_m, mean_rest, difference, t_stat, p_value, and significance stars.
    """
    df = df.copy()
    df["return"] = coerce_numeric(df["return"])
    out_rows = []
    for m in range(1, 13):
        m_mask = (df["month"].astype("Int64") == m)
        x = df.loc[m_mask, "return"].dropna().to_numpy()
        y = df.loc[~m_mask, "return"].dropna().to_numpy()
        if len(x) == 0 or len(y) == 0:
            t_stat, p_val = np.nan, np.nan
            mean_x, mean_y = np.nan, np.nan
        else:
            t_stat, p_val = stats.ttest_ind(x, y, equal_var=False, alternative="two-sided")
            mean_x, mean_y = float(np.mean(x)), float(np.mean(y))
        out_rows.append({
            "month": m,
            "label": month_label(m),
            "n_month": int(len(x)),
            "n_rest": int(len(y)),
            "mean_month": mean_x,
            "mean_rest": mean_y,
            "diff": (mean_x - mean_y) if (np.isfinite(mean_x) and np.isfinite(mean_y)) else np.nan,
            "t_stat": as_float(t_stat),
            "p_value": as_float(p_val),
            "sig": stars_from_p(p_val)
        })
    tbl = pd.DataFrame(out_rows).sort_values("month").reset_index(drop=True)
    return tbl[["label","n_month","n_rest","mean_month","mean_rest","diff","t_stat","p_value","sig"]]

def outlier_tables(model, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    infl = OLSInfluence(model)
    stud = infl.resid_studentized_external
    lev = infl.hat_matrix_diag
    cooks = infl.cooks_distance[0]
    k = model.df_model + 1
    n = int(model.nobs)
    if hasattr(model.model.data, "row_labels") and (model.model.data.row_labels is not None):
        idx = model.model.data.row_labels
        dates = df.loc[idx, "date"].values
        years = df.loc[idx, "year"].values
        months = df.loc[idx, "month"].values
        rets = df.loc[idx, "return"].values
    else:
        dates = df["date"].iloc[:n].values
        years = df["year"].iloc[:n].values
        months = df["month"].iloc[:n].values
        rets = df["return"].iloc[:n].values
    flags = pd.DataFrame({
        "date": dates, "year": years, "month": months, "return": rets,
        "stud_resid": stud, "leverage": lev, "cooks_d": cooks,
        "flag_studentized>|3|": np.abs(stud) > 3,
        "flag_leverage>2k/n": lev > (2*k/n),
        "flag_cooks>4/n": cooks > (4/n),
    }).sort_values("cooks_d", ascending=False)
    r = coerce_numeric(df["return"])
    med = np.nanmedian(r); mad = np.nanmedian(np.abs(r - med))
    madz = np.zeros_like(r) if (mad == 0 or np.isnan(mad)) else 0.6745 * (r - med) / mad
    uni = pd.DataFrame({
        "date": df["date"].values, "year": df["year"].values, "month": df["month"].values,
        "return": r.values, "MAD_z": madz, "flag_|MADz|>3.5": np.abs(madz) > 3.5,
    }).sort_values("MAD_z", key=lambda s: pd.Series(s).abs(), ascending=False)
    return flags, uni

# ------------------------- per-ticker -------------------------

def analyze_ticker(df_t: pd.DataFrame, outdir: Path, ticker: str):
    print(f"  [info] {ticker}: rows={len(df_t)}, date_range={df_t['date'].min()}..{df_t['date'].max()}")
    df_t = df_t.sort_values("date").copy()

    # EDA
    desc = summarize_series(df_t["return"])
    # months = month_table(df_t)  # kept in case you need it elsewhere
    months_vs_rest = month_vs_rest_table(df_t)

    # Stationarity
    print("  [info] ADF/KPSS...")
    utests = adf_kpss_tests(df_t["return"])

    # Baseline design (for residuals and condition number)
    print("  [info] building baseline design...")
    X, y = build_baseline_design(df_t)
    print(f"  [info] design: X.shape={X.shape}, y.shape={y.shape}")
    if len(y) < 15 or X.shape[1] < 2:
        raise ValueError("Not enough clean observations after NA/inf filtering to fit OLS.")

    # Snapshot before fitting
    #save_debug_dump(ticker, df_t, X, y, outdir)

    print("  [info] fitting baseline OLS...")
    model = sm.OLS(y.values, X.values).fit()
    model_df = sm.OLS(y, X).fit()  # labeled

    # Condition number (baseline design)
    cond_num = float(np.linalg.cond(X.values))

    # Heteroskedasticity & stability
    resid = model.resid
    het = bp_white_tests(resid, X)
    arch = arch_lm_test(resid, lags=12)
    cus = cusum_test(resid, X)

    # Structural break (Chow)
    print("  [info] rolling Chow...")
    chow = chow_search(pd.Series(y.values), pd.DataFrame(X.values, columns=X.columns), min_seg=24)
    if "break_at" in chow and math.isfinite(as_float(chow.get("break_at", np.nan))):
        idx_in_clean = int(chow["break_at"])
        orig_idx = y.index[idx_in_clean]
        chow_date = df_t.loc[orig_idx, "date"]
    else:
        chow_date = None

    # Outliers
    print("  [info] outliers...")
    reg_outliers, uni_outliers = outlier_tables(model_df, df_t)

    # ---------- Markdown report ----------
    md = []
    md.append(f"# {ticker} — Monthly Diagnostics\n")

    md.append("## Data\n")
    md.append(f"- Observations: **{len(df_t)}**  \n- Date range: **{df_t['date'].min().date()} — {df_t['date'].max().date()}**")

    avg_vol = float(df_t["avg_vol"].mean(skipna=True))
    min_vol = float(df_t["avg_vol"].min(skipna=True))
    md.append(f"- Average monthly volume: **{avg_vol:,.0f}**  \n- Minimum monthly volume: **{min_vol:,.0f}**")

    md.append("\n\n## Exploratory Analysis\n")
    md.append("**Summary of monthly returns**\n" + df_to_md(desc) + "\n")

    md.append("\n**Month vs. Rest-of-Year (Welch two-sample t-tests)**\n")
    md.append(
        df_to_md(
            months_vs_rest.rename(columns={
                "label": "month",
                "n_month": "n(m)",
                "n_rest": "n(rest)",
                "mean_month": "mean(m)",
                "mean_rest": "mean(rest)",
                "diff": "mean(m) − mean(rest)"
            })
        ) + "\n"
    )
    md.append(
        "_Interpretation:_ For each row, we test whether the average return in that month differs from "
        "the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s "
        "average return is significantly different from the rest of the year.\n"
    )

    # Stationarity with stars + critical values
    md.append("\n## Stationarity\n")
    if "note" in utests:
        md.append(f"- {utests['note']}\n")
    else:
        md.append(f"- **ADF**: stat = {utests['adf_stat']:.3f}, p = {utests['adf_p']:.4f} {utests['adf_sig']}, "
                  f"lags used = {utests['adf_lags']}, n = {utests['adf_nobs']}  \n"
                  f"  critical values: 1%={utests['adf_crit']['1%']:.3f}, 5%={utests['adf_crit']['5%']:.3f}, 10%={utests['adf_crit']['10%']:.3f}")
        if "kpss_stat" in utests:
            md.append(f"\n- **KPSS (level)**: stat = {utests['kpss_stat']:.3f}, p = {utests['kpss_p']:.4f} {utests['kpss_sig']}, "
                      f"lags used = {utests['kpss_lags']}  \n"
                      f"  critical values: 10%={float(utests['kpss_crit']['10%']):.3f}, 5%={float(utests['kpss_crit']['5%']):.3f}, "
                      f"2.5%={float(utests['kpss_crit']['2.5%']):.3f}, 1%={float(utests['kpss_crit']['1%']):.3f}\n")
        joint = ("ADF rejects & KPSS does not reject → returns look stationary."
                 if (utests.get("adf_p",1)<0.05 and utests.get("kpss_p",1)>0.05)
                 else "Mixed or non-stationary signals; interpret with caution.")
        md.append(f"- **Joint read**: {joint}\n")

    # Homoskedasticity
    md.append("\n## Homoskedasticity (constant error variance)\n")
    md.append(f"- **Breusch–Pagan**: LM p = {het['bp_p']:.4f}, F p = {het['bp_fp']:.4f}  \n"
              f"- **White**: LM p = {het['white_p']:.4f}, F p = {het['white_fp']:.4f}\n"
              "  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** "
              "(here: month dummies).\n")

    # ARCH LM (time-varying volatility)
    if "arch_lm" in arch:
        md.append(
            f"- **ARCH LM (lags={arch['arch_lags']})**: LM p = {arch['arch_lm_p']:.4f}, F p = {arch['arch_fp']:.4f}  \n"
            "  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** "
            "(volatility clustering) in residuals.\n")
    else:
        md.append(f"- **ARCH LM**: not available ({arch.get('arch_error', 'n/a')}).\n")

    # Clear conclusion combining BP/White + ARCH
    bp_min_p = np.nanmin([het.get("bp_p", np.nan), het.get("bp_fp", np.nan)])
    white_min_p = np.nanmin([het.get("white_p", np.nan), het.get("white_fp", np.nan)])
    arch_min_p = np.nanmin([arch.get("arch_lm_p", np.nan), arch.get("arch_fp", np.nan)])

    tests_p = [bp_min_p, white_min_p, arch_min_p]
    any_hetero = any(np.isfinite(p) and p < 0.05 for p in tests_p)
    all_large = all(np.isfinite(p) and p > 0.10 for p in tests_p if np.isfinite(p))

    if any_hetero:
        homo_concl = ("**Conclusion:** Evidence of **heteroskedasticity**. "
                      "Report **robust standard errors** (HC3; and **HAC/Newey–West** if serial correlation is present).")
    elif all_large:
        homo_concl = ("**Conclusion:** Tests **do not suggest heteroskedasticity** at conventional levels. "
                      "A homoskedastic assumption appears reasonable; **robust SEs** are still a conservative default in finance.")
    else:
        homo_concl = ("**Conclusion:** **Mixed/ambiguous** evidence (some tests borderline). "
                      "Prefer **robust/HAC standard errors** to be safe.")

    md.append(homo_concl + "\n")

    # Structural breaks
    md.append("\n## Structural Breaks\n")
    # CUSUM explanation + result
    if "cusum_p" in cus and np.isfinite(cus["cusum_p"]):
        md.append(
            f"- **CUSUM (parameter stability test)**: p = {cus['cusum_p']:.4f}. "
            "CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate "
            "**time-varying coefficients** (instability) rather than constant effects over the sample.\n"
        )
        if cus["cusum_p"] < 0.05:
            md.append("  **Interpretation:** Reject stability — parameters likely drift over time.\n")
        elif cus["cusum_p"] > 0.10:
            md.append("  **Interpretation:** No evidence against stability — parameters appear stable.\n")
        else:
            md.append("  **Interpretation:** Borderline evidence — stability is uncertain.\n")
    else:
        md.append(f"- **CUSUM**: not available ({cus.get('cusum_error', 'n/a')}).\n")

    # Rolling Chow explanation + result
    if math.isfinite(as_float(chow.get("F", np.nan))):
        md.append(
            f"- **Chow (rolling break search)**: F = {as_float(chow['F']):.2f}, "
            f"p = {as_float(chow['p']):.4f}, break at index "
            f"{int(chow['break_at']) if math.isfinite(as_float(chow.get('break_at', np.nan))) else 'n/a'} "
            f"({chow_date.date() if chow_date is not None else 'n/a'}), "
            f"left n = {as_float(chow.get('n_left', np.nan)):.0f}, right n = {as_float(chow.get('n_right', np.nan)):.0f}\n"
        )
        md.append(
            "  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a "
            "**two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** "
            "at (or near) the reported split.\n"
        )
        if as_float(chow.get("p", np.nan)) < 0.05:
            md.append("  **Interpretation:** Break suggested at the reported date; parameter shifts before/after that point.\n")
        elif as_float(chow.get("p", np.nan)) > 0.10:
            md.append("  **Interpretation:** No strong evidence of a single dominant break.\n")
        else:
            md.append("  **Interpretation:** Weak or borderline evidence of a break.\n")
    else:
        md.append("- **Chow**: not enough data for rolling test or no valid breakpoint found.\n")

    # Outliers
    md.append("\n## Outliers\n")
    md.append("**Regression influence (top 10 by Cook’s D)**\n" + df_to_md(reg_outliers.head(10)) + "\n")
    md.append("\n**Univariate return outliers (MAD-z)**\n" + df_to_md(uni_outliers.head(10)) + "\n")

    (outdir / f"{ticker}_diagnostics.md").write_text("\n".join(md), encoding="utf-8")

# ------------------------- main -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default=None, help="Run only this ticker (e.g., CC)")
    args = parser.parse_args()

    project_root = Path().resolve().parent.parent
    data_root = project_root / "Complete Data" / "All_Monthly_Return_Data"
    reports_root = Path().resolve() / "Reports"

    #logfile = init_logger(reports_root)
    #print(f"[info] logging to: {logfile}")

    reports_root = Path().resolve() / "Reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    # adjust pattern as needed
    files = sorted(data_root.glob("*_Monthly_Revenues.csv"))
    print(files)
    if not files:
        print(f"No *_Monthly_Revenues.csv files found under {data_root}")
        sys.exit(1)

    print(f"Found {len(files)} file(s). Processing...")
    frames = []
    for fp in files:
        try:
            print(f"[load] {fp.name}")
            df = load_and_normalize(fp)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] {fp.name}: {e}\n{traceback.format_exc()}")

    if not frames:
        print("No valid files loaded.")
        sys.exit(1)

    all_df = pd.concat(frames, ignore_index=True)
    tickers = sorted(all_df["ticker"].dropna().unique())
    print(f"Tickers detected: {', '.join(map(str, tickers))}")

    if args.ticker:
        tickers = [t for t in tickers if str(t).upper() == args.ticker.upper()]
        if not tickers:
            print(f"[error] requested ticker {args.ticker} not found.")
            sys.exit(1)

    for t in tickers:
        try:
            print(f"- Analyzing {t} ...")
            analyze_ticker(all_df[all_df["ticker"] == t].copy(), outdir=reports_root, ticker=t)
        except Exception as e:
            print(f"[ERROR] {t}: {e}\n{traceback.format_exc()}")

    print(f"Done. See reports in: {reports_root}")

if __name__ == "__main__":
    main()
