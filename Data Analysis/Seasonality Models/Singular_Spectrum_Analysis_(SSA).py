import numpy as np
import pandas as pd
from pathlib import Path

# --- User parameters ---
L          = 12  # SSA window length
J          = 2   # Number of components
start_date = "1999-01-01"
end_date   = "2025-04-15"
K_SELECT   = 2   # Number of tickers to long/short

# --- Auto-detect tickers based on filenames ---
project_root = Path().resolve().parent.parent
data_dir     = project_root / "Complete Data" / "All_Monthly_Return_Data"
paths = list(data_dir.glob("*_Monthly_Revenues.csv"))
ASSET_LIST = [p.stem.replace("_Monthly_Revenues", "") for p in paths]
# ASSET_LIST = ["CC", "CF", "CO"]

def ssa_score(series: pd.Series, L: int, J: int):
    x = series.values.astype(float)
    N = len(x)
    if L > N:
        return np.full(N, np.nan), np.nan

    # 1. Hankel embedding
    K = N - L + 1
    X = np.column_stack([x[i:i+L] for i in range(K)])
    # 2. SVD
    S = X @ X.T
    λ, U = np.linalg.eigh(S)
    idx = np.argsort(λ)[::-1]
    λ, U = λ[idx], U[:, idx]
    # 3. Reconstruct top J components
    Xj = sum(
        np.sqrt(λ[m]) * np.outer(
            U[:, m],
            (X.T @ U[:, m]) / np.sqrt(λ[m])
        )
        for m in range(J)
    )
    # 4. Diagonal averaging (Hankelization)
    rec = np.zeros(N)
    counts = np.zeros(N)
    for i in range(L):
        for j in range(K):
            rec[i+j]    += Xj[i, j]
            counts[i+j] += 1
    rec /= counts
    # 5. Final score = mean of last L reconstructed values
    score = rec[-L:].mean() if N >= L else np.nan

    return rec, score

# --- Compute final SSA scores automatically for all detected tickers ---
scores = {}
for ticker in ASSET_LIST:
    path = data_dir / f"{ticker}_Monthly_Revenues.csv"
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df = df.set_index('date').sort_index()
    series = df.loc[start_date:end_date, 'return'].astype(float)

    # Compute one SSA score
    _, final_score = ssa_score(series, L, J)
    scores[ticker] = final_score

# --- Sort and select top/bottom tickers ---
final_scores = pd.Series(scores, name="SSA_Seasonality_Score").to_frame()
sorted_scores = final_scores.sort_values("SSA_Seasonality_Score", ascending=False)

long_tickers  = sorted_scores.head(K_SELECT).index.tolist()
short_tickers = sorted_scores.tail(K_SELECT).index.tolist()

# --- Display results ---
print("Detected tickers:", ASSET_LIST)
print("Seasonality Scores:\n", sorted_scores, "\n")
print(f"Long: {long_tickers}")
print(f"Short: {short_tickers}")
