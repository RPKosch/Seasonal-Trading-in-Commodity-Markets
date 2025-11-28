# File: Complete Data/Clean_Zero_Volume.py

import pandas as pd
from pathlib import Path

# === CONFIGURATION ===
TICKERS     = ["CC", "CF", "CO", "CP", "CT", "ZW", "GD", "HE", "HO", "LE",
               "NG", "PA", "PL", "ZS", "SU", "SV", "ZC"]

def delete_zero_volume_files(base_dir: Path) -> int:
    """
    For each ticker’s Historic_Data folder, delete any CSV where
    the 'volume' column is all zeros. Returns the count of deleted files.
    """
    deleted_count = 0

    for ticker in TICKERS:
        hist_dir = base_dir / f"{ticker}_Historic_Data"
        if not hist_dir.exists():
            continue

        for csv_path in hist_dir.glob(f"{ticker}_????-??.csv"):
            try:
                df = pd.read_csv(csv_path, usecols=["volume"])
            except Exception:
                # skip unreadable files
                continue

            # delete if all zeros or no volume column
            if "volume" not in df.columns or df["volume"].eq(0).all():
                print(f"Deleting {csv_path.name} (all zero volume)")
                csv_path.unlink()
                deleted_count += 1

    return deleted_count

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    base_dir     = project_root / "Complete Data"
    total_deleted = delete_zero_volume_files(base_dir)
    print(f"\nTotal files deleted: {total_deleted}")
