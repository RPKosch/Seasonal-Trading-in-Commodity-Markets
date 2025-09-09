# -*- coding: utf-8 -*-
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import OLSInfluence

# ========= User settings =========
INPUT_DIR = Path().resolve().parent.parent / "Complete Data" / "All_Monthly_Return_Data"
OUTPUT_DIR = Path().resolve().parent / "Seasonality Models" / "plots" / "Tickers Returns"

# Inclusive date filters (YYYY-MM) or None for full range
START_YM = None           # e.g. "2001-01"
END_YM   = None           # e.g. "2025-07"

# Overlay controls
OUTLIER_TOP_K = 3                  # take up to K from the Cook's-D top 10 (after threshold)
OUTLIER_STUD_RESID_THRESH = 3.0    # only keep outliers with |stud_resid| >= threshold
PLOT_OUTLIERS = True
PLOT_CHOW_BREAK = True
CHOW_MIN_SEG = 24                  # min months on each side of a possible break
CHOW_P_THRESHOLD = 0.05            # <<< only plot breaks with p < this threshold

# Show plots interactively after saving
SHOW_PLOTS = False
# =================================


# ---------------------- helpers ----------------------
def coerce_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[^\d\-\+\.,eE]", "", regex=True)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def parse_year_month(df: pd.DataFrame) -> pd.Series:
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    dates = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1), errors="coerce")
    return dates

def rebase_to_100(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    base = series.iloc[0]
    if pd.isna(base) or base == 0:
        return series * np.nan
    return 100.0 * (series / base)

def compute_stats(index_series: pd.Series) -> dict:
    out = {"total_return": np.nan, "cagr": np.nan, "max_drawdown": np.nan, "months": len(index_series)}
    if len(index_series) < 1 or index_series.isna().all():
        return out
    start_val = index_series.iloc[0]
    end_val = index_series.iloc[-1]
    months = len(index_series)
    if pd.notna(start_val) and start_val > 0 and pd.notna(end_val):
        out["total_return"] = (end_val / start_val) - 1.0
        out["cagr"] = (end_val / start_val) ** (12.0 / months) - 1.0
    rolling_peak = index_series.cummax()
    dd = index_series / rolling_peak - 1.0
    out["max_drawdown"] = float(dd.min()) if not dd.isna().all() else np.nan
    return out


# ---------------------- modeling pieces ----------------------
def build_baseline_design(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Intercept + 11 month dummies (Jan omitted). Keep original index to map back to dates."""
    y = coerce_numeric(df["return"])
    M = pd.get_dummies(df["month"].astype("Int64"), prefix="m", drop_first=True, dtype=float)  # m_2..m_12
    X = sm.add_constant(M, has_constant="add").astype(float)
    mask = y.replace([np.inf, -np.inf], np.nan).notna()
    mask &= X.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    return X.loc[mask].astype(float), y.loc[mask].astype(float)

def cooks_top(df_t: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Top-10 by Cook's D with stud_resid etc."""
    model_labeled = sm.OLS(y, X).fit()
    infl = OLSInfluence(model_labeled)
    rows = pd.DataFrame({
        "orig_idx": y.index,
        "stud_resid": infl.resid_studentized_external,
        "leverage": infl.hat_matrix_diag,
        "cooks_d": infl.cooks_distance[0],
    }).sort_values("cooks_d", ascending=False).head(10)

    out = rows.merge(
        df_t.loc[:, ["date", "year", "month", "return"]],
        left_on="orig_idx", right_index=True, how="left"
    ).drop(columns=["orig_idx"])

    for c in ["return", "stud_resid", "leverage", "cooks_d"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def chow_search(y: pd.Series, X: pd.DataFrame, min_seg: int = 24) -> dict:
    """Rolling Chow: best F over admissible splits."""
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)
    n = len(y); k = X.shape[1]
    out = {"F": np.nan, "p": np.nan, "break_at": np.nan, "n_left": np.nan, "n_right": np.nan}
    if n < (2*min_seg + 1):
        return out

    model_full = sm.OLS(y.values, X.values).fit()
    RSS_full = float(np.sum(np.asarray(model_full.resid, dtype=float)**2))
    bestF, best = -np.inf, out.copy()

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
            if np.isfinite(F) and F > bestF:
                bestF = F
                best = {"F": float(F), "p": float(p), "break_at": int(b),
                        "n_left": int(len(y1)), "n_right": int(len(y2))}
        except Exception:
            continue
    return best


# ---------------------- smart legend placement ----------------------
def _pick_legend_loc(dates: pd.Series,
                     values: pd.Series,
                     chow_date: Optional[pd.Timestamp]) -> Tuple[str, Optional[Tuple[float, float]], bool]:
    """
    Heuristic corner choice to avoid covering data/Chow line.
    Returns: (loc, bbox_to_anchor, outside_flag)
    - If no clean corner, place legend outside on the right.
    """
    dates = pd.to_datetime(dates)
    y = pd.to_numeric(values, errors="coerce")
    y = y[np.isfinite(y)]
    if len(dates) == 0 or len(y) == 0:
        return "upper right", None, False

    n = len(dates)
    left_cut = dates.iloc[max(0, int(0.2*n)-1)]
    right_cut = dates.iloc[min(n-1, int(0.8*n))]

    left_band = values[dates <= left_cut]
    right_band = values[dates >= right_cut]

    y_min, y_max = float(np.nanmin(values)), float(np.nanmax(values))
    rng = max(1e-9, y_max - y_min)
    top_thr = y_min + 0.85 * rng
    bot_thr = y_min + 0.15 * rng

    def band_stats(band):
        b = pd.to_numeric(band, errors="coerce")
        return float(np.nanmin(b)) if len(b) else np.inf, float(np.nanmax(b)) if len(b) else -np.inf

    lb_min, lb_max = band_stats(left_band)
    rb_min, rb_max = band_stats(right_band)

    top_left_ok = lb_max < top_thr
    top_right_ok = rb_max < top_thr
    bottom_left_ok = lb_min > bot_thr
    bottom_right_ok = rb_min > bot_thr

    order = ["upper right", "upper left", "lower right", "lower left"]
    if chow_date is not None and dates.max() > dates.min():
        frac = (chow_date - dates.min()) / (dates.max() - dates.min())
        frac = float(frac)
        if frac >= 0.75:
            order = ["upper left", "lower left", "upper right", "lower right"]
        elif frac <= 0.25:
            order = ["upper right", "lower right", "upper left", "lower left"]

    ok_map = {
        "upper right": top_right_ok,
        "upper left": top_left_ok,
        "lower right": bottom_right_ok,
        "lower left": bottom_left_ok,
    }

    for loc in order:
        if ok_map.get(loc, False):
            return loc, None, False

    return "upper left", (1.02, 1.0), True


# ---------------------- plotting per ticker ----------------------
def plot_ticker(csv_path: Path,
                start_ym: Optional[str],
                end_ym: Optional[str]) -> None:
    ticker = csv_path.stem.split("_")[0]

    # Load & prep
    df = pd.read_csv(csv_path)
    if "return" not in df.columns or "year" not in df.columns or "month" not in df.columns:
        print(f"[WARN] Missing required columns in {csv_path.name}. Skipping.")
        return
    df["date"] = parse_year_month(df)
    df = df.sort_values("date").reset_index(drop=True)

    if start_ym is not None:
        df = df[df["date"] >= pd.to_datetime(start_ym + "-01")]
    if end_ym is not None:
        df = df[df["date"] <= pd.to_datetime(end_ym + "-01")]
    if df.empty:
        print(f"[INFO] No data in range for {ticker}. Skipping.")
        return

    df["return"] = coerce_numeric(df["return"])

    # Buy & hold index
    gross = (1.0 + df["return"]).cumprod()
    index_100 = rebase_to_100(gross)

    stats_ = compute_stats(index_100)
    start_label = df["date"].dt.strftime("%Y-%m").iloc[0]
    end_label = df["date"].dt.strftime("%Y-%m").iloc[-1]
    ttl_ret_pct = f"{stats_['total_return']*100:,.2f}%" if pd.notna(stats_["total_return"]) else "n/a"
    cagr_pct = f"{stats_['cagr']*100:,.2f}%" if pd.notna(stats_["cagr"]) else "n/a"
    mdd_pct = f"{stats_['max_drawdown']*100:,.2f}%" if pd.notna(stats_["max_drawdown"]) else "n/a"

    # Diagnostics we need
    X, y = build_baseline_design(df)

    # Outliers
    reg_outliers = None
    if PLOT_OUTLIERS and len(y) >= 15:
        reg_outliers = cooks_top(df, X, y)
        reg_outliers = reg_outliers.loc[reg_outliers["stud_resid"].abs() >= OUTLIER_STUD_RESID_THRESH]
        reg_outliers = reg_outliers.head(OUTLIER_TOP_K)
        print(f"[INFO] {ticker}: {len(reg_outliers)} outliers pass |stud_resid| >= {OUTLIER_STUD_RESID_THRESH} (top-K={OUTLIER_TOP_K}).")

    # Chow break (only accept if p < CHOW_P_THRESHOLD)
    chow_date = None
    if PLOT_CHOW_BREAK and len(y) >= 2*CHOW_MIN_SEG + 1:
        chow = chow_search(y, X, min_seg=CHOW_MIN_SEG)
        if np.isfinite(chow.get("F", np.nan)) and np.isfinite(chow.get("break_at", np.nan)) and np.isfinite(chow.get("p", np.nan)):
            clean_break_idx = int(chow["break_at"])
            orig_idx = y.index[clean_break_idx]
            candidate_date = df.loc[orig_idx, "date"]
            if chow["p"] < CHOW_P_THRESHOLD:
                chow_date = candidate_date
                print(f"[INFO] {ticker}: Chow break accepted at {candidate_date.date()}, F={chow['F']:.2f}, p={chow['p']:.4f} (< {CHOW_P_THRESHOLD})")
            else:
                print(f"[INFO] {ticker}: Chow break rejected (p={chow['p']:.4f} ≥ {CHOW_P_THRESHOLD}); candidate {candidate_date.date()} not plotted.")
        else:
            print(f"[INFO] {ticker}: no valid Chow break found.")

    # Legend labels
    outlier_label_pos = f"Outliers (Cook's D top {OUTLIER_TOP_K}, |t*|≥{OUTLIER_STUD_RESID_THRESH}) (+)"
    outlier_label_neg = f"Outliers (Cook's D top {OUTLIER_TOP_K}, |t*|≥{OUTLIER_STUD_RESID_THRESH}) (−)"
    chow_label = f"Chow break (p<{CHOW_P_THRESHOLD:g})"

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6.8))
    ax.plot(df["date"], index_100, linewidth=2.0, label=f"{ticker} index (base=100)")

    # Chow line (no text annotation)
    if chow_date is not None and df["date"].min() <= chow_date <= df["date"].max():
        ax.axvline(chow_date, linestyle="--", linewidth=2, color="black", zorder=5, label=chow_label)

    # Outlier dots (no annotations)
    if reg_outliers is not None and not reg_outliers.empty:
        vis = reg_outliers[(reg_outliers["date"] >= df["date"].min()) & (reg_outliers["date"] <= df["date"].max())].copy()
        if not vis.empty:
            merged = pd.merge(
                vis[["date", "stud_resid"]],
                pd.DataFrame({"date": df["date"], "index100": index_100}),
                on="date", how="inner"
            )
            if not merged.empty:
                pos = merged[merged["stud_resid"] >= 0]
                neg = merged[merged["stud_resid"] < 0]
                if not pos.empty:
                    ax.scatter(pos["date"], pos["index100"], s=80, marker="o",
                               color="red", edgecolors="none", zorder=6, label=outlier_label_pos)
                if not neg.empty:
                    ax.scatter(neg["date"], neg["index100"], s=80, marker="o",
                               color="orange", edgecolors="none", zorder=6, label=outlier_label_neg)

    # Smart legend placement
    loc, bba, outside = _pick_legend_loc(df["date"], index_100, chow_date)
    if outside:
        ax.legend(loc=loc, bbox_to_anchor=bba, frameon=False)
    else:
        ax.legend(loc=loc, frameon=False)

    # Titles & layout
    ax.set_title(f"{ticker} Buy-and-Hold Performance\n{start_label} to {end_label}", fontsize=14, pad=10)
    subtitle = f"Total return {ttl_ret_pct}   |   CAGR {cagr_pct}   |   Max drawdown {mdd_pct}   |   Months {stats_['months']}"
    fig.suptitle(subtitle, y=0.94, fontsize=10)

    ax.set_xlabel("")
    ax.set_ylabel("Index (base 100)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    # Save (ensure outside legend isn't clipped)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{ticker}_long_hold_{start_label}_to_{end_label}.png"
    out_path = OUTPUT_DIR / fname
    save_kwargs = {"dpi": 200}
    if outside:
        save_kwargs["bbox_inches"] = "tight"
    plt.savefig(out_path, **save_kwargs)
    print(f"[OK] Saved {out_path}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# ---------------------- main ----------------------
def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder does not exist: {INPUT_DIR}")

    files = sorted(INPUT_DIR.glob("*_Monthly_Revenues.csv"))
    if not files:
        print(f"[INFO] No '*_Monthly_Revenues.csv' files found in {INPUT_DIR}")
        return

    for csv_path in files:
        try:
            plot_ticker(csv_path, START_YM, END_YM)
        except Exception as e:
            print(f"[ERROR] Failed on {csv_path.name}: {e}")

if __name__ == "__main__":
    main()
