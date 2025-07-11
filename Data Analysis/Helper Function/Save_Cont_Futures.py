import yfinance as yf
import pandas as pd

def fetch_and_save_monthly_close(
    ticker: str,
    start_date: str,
    end_date: str,
    output_csv: str
) -> None:
    """
    Fetch monthly Close prices for one ticker and save a clean two-column CSV.
    """
    # 1) Download monthly data
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1mo",
        progress=False,
        auto_adjust=False
    )

    # 2) Extract the Close series (handle flat or multi-level columns)
    # If df['Close'] returns a DataFrame (multi-index), take the first column.
    close_obj = df["Close"]
    if isinstance(close_obj, pd.DataFrame):
        # e.g. columns like ('Close','CL=F')
        close_s = close_obj.iloc[:, 0]
    else:
        # Series
        close_s = close_obj

    # 3) Build a one-column DataFrame named after the ticker
    out = close_s.to_frame(name=ticker)
    out.index.name = "Date"

    # 4) Save to CSV with ISO dates and single header row
    out.to_csv(output_csv, date_format="%Y-%m-%d")

    print(f"✅ Saved {len(out)} rows of monthly '{ticker}' Close prices to '{output_csv}'")

if __name__ == "__main__":
    fetch_and_save_monthly_close(
        ticker="CL=F",
        start_date="2000-01-01",
        end_date=pd.Timestamp.today().strftime("%Y-%m-%d"),
        output_csv="crude_monthly_close.csv"
    )
