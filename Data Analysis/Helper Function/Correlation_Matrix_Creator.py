#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a colored correlation matrix across all tickers' monthly log returns.

- Scans:   ../Complete Data/All_Monthly_Log_Return_Data/*_Monthly_Revenues.csv
- Expects: columns at least {ticker, year, month, return}
- Output:  Reports/Correlation/
           - correlations_<method>.csv
           - overlaps.csv  (pairwise # of overlapping months used)
           - corr_heatmap_<method>.png/.pdf  (optionally clustered)

Usage examples:
  python correlation_matrix.py
  python correlation_matrix.py --method spearman --min-overlap 24 --cluster
  python correlation_matrix.py --start 2005-01 --end 2020-12 --annot
"""

import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# seaborn only for a clean heatmap
import seaborn as sns

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 200)

# ------------------------- helpers -------------------------

def coerce_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[^\d\-\+\.,eE]", "", regex=True)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def load_and_normalize(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"ticker", "year", "month", "return"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{fp.name}: missing {missing}")
    keep = [c for c in ["ticker","year","month","return"] if c in df.columns]
    df = df[keep].copy()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["year"] = coerce_numeric(df["year"]).astype("Int64")
    df["month"] = coerce_numeric(df["month"]).astype("Int64")
    df["return"] = coerce_numeric(df["return"])
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1), errors="coerce")
    df = df.dropna(subset=["ticker","date","return"])
    return df[["ticker","date","return"]].sort_values(["ticker","date"])

def make_wide_returns(df_all: pd.DataFrame, start: str|None, end: str|None) -> pd.DataFrame:
    # Optional date filter
    if start:
        start_dt = pd.to_datetime(start + "-01") if len(start) == 7 else pd.to_datetime(start)
        df_all = df_all[df_all["date"] >= start_dt]
    if end:
        end_dt = pd.to_datetime(end + "-01") if len(end) == 7 else pd.to_datetime(end)
        df_all = df_all[df_all["date"] <= end_dt]
    # Pivot to wide (date × ticker)
    wide = df_all.pivot_table(index="date", columns="ticker", values="return", aggfunc="first").sort_index()
    return wide

def pairwise_corr_with_min_overlap(wide: pd.DataFrame, method: str = "pearson", min_overlap: int = 12
                                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise correlation with pairwise deletion and min-overlap rule.
    Returns (corr_df, overlap_counts_df).
    """
    tickers = list(wide.columns)
    n = len(tickers)
    corr = pd.DataFrame(np.nan, index=tickers, columns=tickers, dtype=float)
    counts = pd.DataFrame(0, index=tickers, columns=tickers, dtype=int)

    for i in range(n):
        xi = wide.iloc[:, i]
        for j in range(i, n):
            xj = wide.iloc[:, j]
            both = xi.notna() & xj.notna()
            k = int(both.sum())
            counts.iat[i, j] = counts.iat[j, i] = k
            if k >= min_overlap:
                if method == "spearman":
                    rho = xi[both].rank().corr(xj[both].rank())
                else:
                    rho = xi[both].corr(xj[both])
                corr.iat[i, j] = corr.iat[j, i] = float(rho)
            else:
                corr.iat[i, j] = corr.iat[j, i] = np.nan
    np.fill_diagonal(corr.values, 1.0)
    np.fill_diagonal(counts.values, wide.notna().sum().to_numpy())
    return corr, counts

def cluster_order(corr: pd.DataFrame) -> list[str]:
    """
    Order tickers by hierarchical clustering of distances (1 - corr)/2.
    NaNs off-diagonal are treated as neutral distance 1.0 (corr=0).
    """
    C = corr.copy()
    tickers = list(C.index)
    # distance in [0,1]: 0 ~ perfectly correlated, 1 ~ zero corr; negative corrs >1 but we cap at [0, 2]
    D = 1.0 - C
    # Handle NaNs: neutral distance
    D = D.fillna(1.0)
    # Ensure zeros on diagonal
    np.fill_diagonal(D.values, 0.0)
    # Convert to condensed distance for linkage
    # If negative correlations are common, you may want distance = (1 - corr)/2 to map [-1,1] -> [1,0]
    Dc = squareform(((1 - C).fillna(0.0)).clip(lower=-1, upper=2).to_numpy(), checks=False)
    Z = linkage(Dc, method="average")
    order_idx = leaves_list(Z)
    return [tickers[i] for i in order_idx]

def draw_heatmap(corr: pd.DataFrame, title: str, out_png: Path, annot: bool = False):
    # dynamic figure size
    m = corr.shape[0]
    base = max(6, min(24, int(0.5 * m)))  # keep reasonable bounds
    fig, ax = plt.subplots(figsize=(base, base))
    # a nice diverging map centered at zero
    sns.heatmap(
        corr, ax=ax, cmap="coolwarm", vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.5, linecolor="white",
        annot=annot, fmt=".2f", cbar_kws={"shrink": 0.8}
    )
    ax.set_title(title, pad=12)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["pearson","spearman"], default="pearson",
                    help="Correlation method (default: pearson)")
    ap.add_argument("--min-overlap", type=int, default=12,
                    help="Minimum overlapping months required for a pairwise correlation (default: 12)")
    ap.add_argument("--start", type=str, default=None, help="Start date filter (YYYY-MM or YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default=None, help="End date filter (YYYY-MM or YYYY-MM-DD)")
    ap.add_argument("--cluster", action="store_true", help="Cluster tickers to group similar series together")
    ap.add_argument("--annot", action="store_true", help="Annotate heatmap cells with correlation numbers")
    args = ap.parse_args()

    # Paths (match your other scripts)
    project_root = Path().resolve().parent.parent
    data_root = project_root / "Complete Data" / "All_Monthly_Log_Return_Data"
    out_root = Path().resolve() / "Reports" / "Correlation"
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(data_root.glob("*_Monthly_Revenues.csv"))
    if not files:
        print(f"[error] no *_Monthly_Revenues.csv under {data_root}")
        return

    frames = []
    for fp in files:
        try:
            frames.append(load_and_normalize(fp))
        except Exception as e:
            print(f"[WARN] {fp.name}: {e}")

    if not frames:
        print("[error] no valid files loaded.")
        return

    all_df = pd.concat(frames, ignore_index=True)
    wide = make_wide_returns(all_df, start=args.start, end=args.end)

    # Drop columns with no data after filtering
    wide = wide.dropna(axis=1, how="all")
    if wide.shape[1] < 2:
        print("[error] need at least two tickers with data to compute correlations.")
        return

    corr, counts = pairwise_corr_with_min_overlap(wide, method=args.method, min_overlap=args.min_overlap)

    # Optional clustering
    if args.cluster:
        order = cluster_order(corr.fillna(0.0))
        corr = corr.loc[order, order]

    # Save CSVs
    corr_csv = out_root / f"correlations_{args.method}.csv"
    counts_csv = out_root / "overlaps.csv"
    corr.to_csv(corr_csv, float_format="%.6f")
    counts.to_csv(counts_csv)

    # Draw heatmap
    date_span = ""
    if args.start or args.end:
        date_span = f" [{args.start or wide.index.min().date()} — {args.end or wide.index.max().date()}]"

    title = f"Ticker Correlations ({args.method.capitalize()}){date_span}"
    out_png = out_root / f"corr_heatmap_{args.method}{'_clustered' if args.cluster else ''}.png"
    draw_heatmap(corr, title, out_png, annot=args.annot)

    # A quick console summary
    print(f"[info] saved correlation CSV: {corr_csv}")
    print(f"[info] saved overlap counts CSV: {counts_csv}")
    print(f"[info] saved heatmap: {out_png} and {out_png.with_suffix('.pdf')}")
    print(f"[info] tickers included: {', '.join(corr.index)}")

if __name__ == "__main__":
    main()
