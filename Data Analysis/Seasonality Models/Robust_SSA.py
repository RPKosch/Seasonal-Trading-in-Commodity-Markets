import numpy as np
import pandas as pd
from pathlib import Path
import scipy.linalg as la

# --- User parameters ---
L = 12  # SSA window length
q = 2  # Robust rank (number of components)
start_date = "1999-01-01"
end_date = "2025-04-15"
K_SELECT = 2  # Number of tickers to long/short

# --- Auto-detect tickers ---
project_root = Path().resolve().parent.parent
data_dir = project_root / "Complete Data" / "All_Monthly_Return_Data"
paths = list(data_dir.glob("*_Monthly_Revenues.csv"))
ASSET_LIST = [p.stem.replace("_Monthly_Revenues", "") for p in paths]


def robust_low_rank(X, q, max_iter=10, eps=1e-6):
    """
    Compute an L1-robust low-rank approximation of X via IRLS.
    Returns U (L×q), V (K×q) such that UV^T ≈ X minimizing L1 error.
    """
    L, K = X.shape
    # initialize U, V via classical SVD
    U0, s0, V0t = la.svd(X, full_matrices=False)
    U = U0[:, :q] * np.sqrt(s0[:q])
    V = (V0t[:q, :].T) * np.sqrt(s0[:q])

    for _ in range(max_iter):
        R = X - U @ V.T
        W = 1.0 / (np.abs(R) + eps)  # weights ~ 1/|residual|
        # solve weighted least-squares: minimize sum_ij W_ij*(X_ij - [UV^T]_ij)^2
        # Equivalent to SVD of (sqrt(W) * X)
        Xw = np.sqrt(W) * X
        Uw, sw, Vwt = la.svd(Xw, full_matrices=False)
        U = Uw[:, :q] * np.sqrt(sw[:q])
        V = (Vwt[:q, :].T) * np.sqrt(sw[:q])
    return U, V


def rlssa_score(series, L, q):
    x = series.values.astype(float)
    N = len(x)
    if L > N:
        return np.nan

    # 1. Hankel embedding
    K = N - L + 1
    X = np.column_stack([x[i:i + L] for i in range(K)])
    # 2. Robust low-rank decomposition (L1)
    U, V = robust_low_rank(X, q)
    # extract singular values for grouping
    # reconstruct signal matrix X_rob
    X_rob = U @ V.T

    # 3. Diagonal averaging
    rec = np.zeros(N)
    counts = np.zeros(N)
    for i in range(L):
        for j in range(K):
            rec[i + j] += X_rob[i, j]
            counts[i + j] += 1
    rec /= counts

    # 4. Seasonal score: mean of last L points
    score = rec[-L:].mean()
    return score


# --- Compute RLSSA scores ---
scores = {}
for ticker in ASSET_LIST:
    path = data_dir / f"{ticker}_Monthly_Revenues.csv"
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df = df.set_index('date').sort_index()
    series = df.loc[start_date:end_date, 'return'].astype(float)
    scores[ticker] = rlssa_score(series, L, q)

# --- Select top/bottom assets ---
final_scores = pd.Series(scores, name="RLSSA_Score").to_frame()
sorted_scores = final_scores.sort_values("RLSSA_Score", ascending=False)
long_tickers = sorted_scores.head(K_SELECT).index.tolist()
short_tickers = sorted_scores.tail(K_SELECT).index.tolist()

# Display
print("Detected tickers:", ASSET_LIST)
print("RLSSA Scores:\n", sorted_scores, "\n")
print(f"Long:  {long_tickers}")
print(f"Short: {short_tickers}")
