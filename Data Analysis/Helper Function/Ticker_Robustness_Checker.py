from pathlib import Path
import re
import numpy as np
import pandas as pd
from pathlib import Path
import re
import numpy as np
import pandas as pd

BASE_DIR = Path().resolve().parent / "Seasonality Models" / "Monte_Carlo_Outputs"

# Subfolders to process under BASE_DIR
MODELS = [
   "DVR_MC_p≤0.1_classic_vs_risk_adjusted",
   "SSA_MC_classic_vs_risk_adjusted",
   "RLSSA_MC_classic_vs_risk_adjusted",
]

CELL_RE = re.compile(r"\[\s*(?P<ticker>[^\|\]\[]+?)\s*\|\s*(?P<pct>\d+(?:\.\d+)?)\s*%\s*\]")

def parse_top_cell(cell) -> tuple[str | None, float]:
   """Parse a cell like '[TICK | 37.5%]' -> ('TICK', 37.5)."""
   if cell is None:
       return None, np.nan
   s = str(cell).strip()
   if not s:
       return None, np.nan
   m = CELL_RE.search(s)
   if not m:
       return None, np.nan
   return m.group("ticker").strip(), float(m.group("pct"))


def first_nonempty_top(row: pd.Series, prefix: str) -> tuple[str | None, float]:
   """Return first non-empty among B1..B5 (or G1..G5) as (ticker, pct)."""
   for k in range(1, 6):
       t, p = parse_top_cell(row.get(f"{prefix}{k}"))
       if t is not None and np.isfinite(p):
           return t, p
   return None, np.nan


def detect_prefixes(df: pd.DataFrame) -> tuple[str, str]:
   """
   Detect which pair of prefixes to use for 'left' and 'right' strategies.
   Supports:
     - B*/G*  (Baseline vs GPS)
     - C*/R*  (Classic vs Risk-adjusted)
   """
   cols = {str(c).strip() for c in df.columns}


   if {"B1", "G1"}.issubset(cols):
       return "B", "G"
   if {"C1", "R1"}.issubset(cols):
       return "C", "R"


   raise ValueError(
       f"Could not detect prefixes: expected B1/G1 or C1/R1, "
       f"but got columns: {sorted(cols)}"
   )


def compute_summary(df: pd.DataFrame) -> dict[str, float]:
   """
   Compute Avg_B1_Pct, Avg_G1_Pct, and SameTopRate for a Top-5 selections table.


   - Works with:
       * B1..B5 / G1..G5   (Baseline vs GPS)
       * C1..C5 / R1..R5   (Classic vs Risk-adjusted)
   - Output keys stay the same for backward compatibility:
       * Avg_B1_Pct: 'left' model (B* or C*)
       * Avg_G1_Pct: 'right' model (G* or R*)
   """
   df = df.copy()
   df.columns = [str(c).strip() for c in df.columns]
   if "Date" not in df.columns:
       raise ValueError("Input CSV must contain a 'Date' column.")


   left_prefix, right_prefix = detect_prefixes(df)


   b_pcts, g_pcts, same_flags = [], [], []


   for _, row in df.iterrows():
       # LEFT side (B* or C*)
       b_t, b_p = parse_top_cell(row.get(f"{left_prefix}1"))
       if b_t is None or not np.isfinite(b_p):
           b_t, b_p = first_nonempty_top(row, left_prefix)


       # RIGHT side (G* or R*)
       g_t, g_p = parse_top_cell(row.get(f"{right_prefix}1"))
       if g_t is None or not np.isfinite(g_p):
           g_t, g_p = first_nonempty_top(row, right_prefix)


       b_pcts.append(b_p)
       g_pcts.append(g_p)


       same_flags.append(
           int(b_t is not None and g_t is not None and (b_t == g_t))
       )


   # Convert to arrays so we can check "all NaN" safely (no warnings)
   b_arr = np.asarray(b_pcts, dtype=float)
   g_arr = np.asarray(g_pcts, dtype=float)


   avg_b = float(np.nanmean(b_arr)) if np.isfinite(b_arr).any() else np.nan
   avg_g = float(np.nanmean(g_arr)) if np.isfinite(g_arr).any() else np.nan


   # same_flags is 0/1 only → plain mean is fine
   match_rate = float(np.mean(same_flags)) if same_flags else np.nan  # in [0,1]


   return {"Avg_B1_Pct": avg_b, "Avg_G1_Pct": avg_g, "SameTopRate": match_rate}


def process_model(model_name: str) -> pd.DataFrame | None:
   """Compute and write overall robustness CSV for one model folder."""
   model_dir = BASE_DIR / model_name
   long_file = model_dir / "top5_selections_table_long.csv"
   short_file = model_dir / "top5_selections_table_short.csv"


   if not long_file.exists() or not short_file.exists():
       print(f"[WARN] Missing long/short table(s) for {model_name}. Skipping.")
       return None


   df_long = pd.read_csv(long_file)
   df_short = pd.read_csv(short_file)


   long_summary  = compute_summary(df_long)
   short_summary = compute_summary(df_short)


   overall = pd.DataFrame([
       {"Model": model_name, "Side": "LONG",  **long_summary},
       {"Model": model_name, "Side": "SHORT", **short_summary},
   ])


   # 🔹 Rename the columns from generic B/G to Classic / Risk-Adjusted
   overall = overall.rename(columns={
       "Avg_B1_Pct": "Avg_Classic_Pct",
       "Avg_G1_Pct": "Avg_Risk_Adjusted_Pct",
   })

   return overall

def main():
   BASE_DIR.mkdir(parents=True, exist_ok=True)


   # Run for each model and collect a combined summary too
   combined_rows = []
   for m in MODELS:
       res = process_model(m)
       if res is not None:
           combined_rows.append(res)


   # Optional combined file across all models
   if combined_rows:
       combined = pd.concat(combined_rows, ignore_index=True)
       combined_path = BASE_DIR / "Top1_Robustness_Test.csv"
       combined.to_csv(combined_path, index=False)
       print(f"Wrote combined summary: {combined_path}")


if __name__ == "__main__":
   main()
