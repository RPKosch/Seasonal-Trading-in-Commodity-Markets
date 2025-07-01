import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ssa_score_paper(series: pd.Series, L: int, J: int):
    """
    Compute the SSA seasonal score per Tang et al. (2024):
      1) Embedding into trajectory matrix X (L×K)
      2) Eigendecomp of XX^T to get eigentriples (λ_i, U_i, V_i)
      3) Group top J components: X^(J) = sum_{i=1}^J X_i
      4) Diagonal averaging to reconstruct series
      5) Compute score = average of last K values of reconstructed series
    """
    # 0) Prepare
    x = series.values
    N = len(x)
    if L > N//2:
        raise ValueError(f"Choose L ≤ N/2 (got L={L}, N={N})")
    K = N - L + 1

    # 1) Embedding: build trajectory matrix X ∈ ℝ^{L×K}
    X = np.column_stack([x[i:i+L] for i in range(K)])  # each column is X_i

    # 2) Eigendecomposition of S = X X^T
    S = X @ X.T                    # L×L
    λ, U = np.linalg.eigh(S)       # ascending order
    idx = np.argsort(λ)[::-1]      # descending
    λ, U = λ[idx], U[:, idx]       # reorder
    d = np.sum(λ > 1e-12)          # number of nonzero eigs

    # build V_i for i=1..d:  V_i = X^T U_i / sqrt(λ_i)
    V = np.zeros((K, d))
    for i in range(d):
        V[:, i] = (X.T @ U[:, i]) / np.sqrt(λ[i])

    # 3) Reconstruct elementary matrices X_i = sqrt(λ_i) U_i V_i^T
    #    Then group the first J:
    XJ = np.zeros_like(X)
    for i in range(min(J, d)):
        XJ += np.sqrt(λ[i]) * np.outer(U[:, i], V[:, i])

    # 4) Diagonal averaging (Hankelization) of XJ into a length-(L+K-1) series
    N_rec = L + K - 1
    recon = np.zeros(N_rec)
    counts = np.zeros(N_rec)
    for i in range(L):
        for j in range(K):
            recon[i+j]  += XJ[i, j]
            counts[i+j] += 1
    recon /= counts
    recon = recon[:N]  # truncate back to length N

    # 5) Seasonal score: average of final K reconstructed values
    score = recon[-K:].mean()
    return score, recon

if __name__ == "__main__":
    # --- Load your clean two‐column CSV: Date,CL=F ---
    df = pd.read_csv(
        "crude_monthly_close.csv",
        index_col="Date",
        parse_dates=True,
        infer_datetime_format=True
    )
    series = pd.to_numeric(df["CL=F"], errors="coerce").dropna()

    # --- Set SSA parameters ---
    L = 12  # window length (≤ N/2)
    J = 2    # number of leading components

    # --- Run ---
    score, recon = ssa_score_paper(series, L, J)

    # --- Output ---
    print(f"SSA Seasonal Score for CL=F: {score:.4f}\n")
    recon_s = pd.Series(recon, index=series.index)
    print("Reconstructed seasonal component (last 5 points):")
    print(recon_s.tail())
    print(recon_s)
    print(score)

    # recon_s is your reconstructed Series with a DatetimeIndex
    recon_s.index = pd.to_datetime(recon_s.index)
    # extract month number 1–12
    by_month = recon_s.groupby(recon_s.index.month).mean()
    print(by_month)

    # 1) Demean to center on zero
    seasonal_dev = by_month - by_month.mean()

    # 2) (Optional) Standardize so that variance = 1
    seasonal_z = (seasonal_dev - seasonal_dev.mean()) / seasonal_dev.std()

    # 3) Visualize
    plt.figure(figsize=(8, 4))
    plt.bar(seasonal_dev.index, seasonal_dev.values, color='tab:blue')
    plt.axhline(0, color='k', lw=1)
    plt.xticks(range(1, 13))
    plt.xlabel("Month")
    plt.ylabel("Seasonal deviation (price units)")
    plt.title("Oil SSA seasonal profile (demeaned)")
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.bar(seasonal_z.index, seasonal_z.values, color='tab:orange')
    plt.axhline(0, color='k', lw=1)
    plt.xticks(range(1, 13))
    plt.xlabel("Month")
    plt.ylabel("Seasonal z-score")
    plt.title("Oil SSA seasonal profile (standardized)")
    plt.show()
