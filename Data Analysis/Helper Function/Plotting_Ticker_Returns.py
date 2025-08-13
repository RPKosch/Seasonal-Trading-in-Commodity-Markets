import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= User settings =========
INPUT_DIR = base = Path().resolve().parent.parent / "Complete Data" / "All_Monthly_Return_Data"
OUTPUT_DIR = Path().resolve().parent / "Seasonality Models" / "plots" / "Tickers Returns"

# Set inclusive start and end (YYYY-MM). Use None to take min/max available.
START_YM = None            # e.g. "2001-01"
END_YM   = None            # e.g. "2025-07"

# Show plots interactively after saving
SHOW_PLOTS = False
# =================================


def parse_year_month(df: pd.DataFrame) -> pd.Series:
    """Create a monthly period date from 'year' and 'month'."""
    # Ensure ints
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    # First day of month as Timestamp
    dates = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1), errors="coerce")
    return dates


def rebase_to_100(series: pd.Series) -> pd.Series:
    """Rebase a positive series to start at 100."""
    if series.empty:
        return series
    base = series.iloc[0]
    if pd.isna(base) or base == 0:
        return series * np.nan
    return 100.0 * (series / base)


def compute_stats(index_series: pd.Series) -> dict:
    """Compute total return, CAGR, and max drawdown from a 100-based index series."""
    out = {"total_return": np.nan, "cagr": np.nan, "max_drawdown": np.nan, "months": len(index_series)}
    if len(index_series) < 1 or index_series.isna().all():
        return out

    start_val = index_series.iloc[0]
    end_val = index_series.iloc[-1]
    months = len(index_series)
    if pd.notna(start_val) and start_val > 0 and pd.notna(end_val):
        total_return = (end_val / start_val) - 1.0
        out["total_return"] = total_return
        if months >= 12:
            out["cagr"] = (end_val / start_val) ** (12.0 / months) - 1.0
        else:
            # Annualize even if <12 months for completeness
            out["cagr"] = (end_val / start_val) ** (12.0 / months) - 1.0

    # Max drawdown
    rolling_peak = index_series.cummax()
    dd = index_series / rolling_peak - 1.0
    out["max_drawdown"] = float(dd.min()) if not dd.isna().all() else np.nan
    return out


def plot_ticker(csv_path: Path, start_ym: str | None, end_ym: str | None) -> None:
    """Read a single ticker CSV and save its buy-and-hold plot."""
    # Infer ticker from file name like "CC_Monthly_Revenues.csv"
    ticker = csv_path.stem.split("_")[0]

    # Read
    df = pd.read_csv(csv_path)
    # Expected columns: ticker,year,month,return,contract,start_date,start_value,end_date,end_value
    if "return" not in df.columns:
        print(f"[WARN] Missing 'return' column in {csv_path.name}. Skipping.")
        return

    # Build a date column and sort
    df["date"] = parse_year_month(df)
    df = df.sort_values("date").reset_index(drop=True)

    # Filter by date range
    if start_ym is not None:
        start_ts = pd.to_datetime(start_ym + "-01")
        df = df[df["date"] >= start_ts]
    if end_ym is not None:
        end_ts = pd.to_datetime(end_ym + "-01")
        df = df[df["date"] <= end_ts]

    if df.empty:
        print(f"[INFO] No data in range for {ticker}. Skipping.")
        return

    # Ensure numeric returns
    df["return"] = pd.to_numeric(df["return"], errors="coerce")

    # Compute cumulative gross and rebase to 100 at first available month
    # Arithmetic monthly returns: index_t = index_{t-1} * (1 + r_t)
    gross = (1.0 + df["return"]).cumprod()
    index_100 = rebase_to_100(gross)

    stats = compute_stats(index_100)

    # Pretty labels
    start_label = df["date"].dt.strftime("%Y-%m").iloc[0]
    end_label = df["date"].dt.strftime("%Y-%m").iloc[-1]
    ttl_ret_pct = f"{stats['total_return']*100:,.2f}%" if pd.notna(stats["total_return"]) else "n/a"
    cagr_pct = f"{stats['cagr']*100:,.2f}%" if pd.notna(stats["cagr"]) else "n/a"
    mdd_pct = f"{stats['max_drawdown']*100:,.2f}%" if pd.notna(stats["max_drawdown"]) else "n/a"

    # Plot
    plt.figure(figsize=(11, 6.5))
    plt.plot(df["date"], index_100, linewidth=2.0, label=f"{ticker} index (base=100)")
    plt.title(f"{ticker} Buy-and-Hold Performance\n{start_label} to {end_label}", fontsize=14, pad=10)
    subtitle = f"Total return {ttl_ret_pct}   |   CAGR {cagr_pct}   |   Max drawdown {mdd_pct}   |   Months {stats['months']}"
    plt.suptitle(subtitle, y=0.94, fontsize=10)

    plt.xlabel("")
    plt.ylabel("Index (base 100)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{ticker}_long_hold_{start_label}_to_{end_label}.png"
    out_path = OUTPUT_DIR / fname
    plt.savefig(out_path, dpi=200)
    print(f"[OK] Saved {out_path}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


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

