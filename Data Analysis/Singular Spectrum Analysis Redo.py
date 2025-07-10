import numpy as np
import pandas as pd

def ssa_forecast(series: pd.Series, L: int, J: int, h: int):
    """
    SSA with Recurrent Forecasting:
      1) Embed series of length N into X (L×K)
      2) Eigendecompose S = XX^T, keep top-J eigentriples (λ, U)
      3) Reconstruct X_J and diagonal‐average → reconstructed series rec (length N)
      4) Compute recurrence coefficients a[1..L-1] via
           a = (1/den) * sum_{m=1..J} (U[L-1,m] * U[0:L-1,m])
         where den = sum_{m=1..J} U[L-1,m]^2
      5) Forecast h steps:
           for t in 1..h:
             x[N+t] = sum_{i=1..L-1} a[i-1] * x[N+t-i]
    Returns: forecast array (h,), score=float
    """
    x = series.values.astype(float)
    N = len(x)
    if L > N:
        raise ValueError(f"L ({L}) must be ≤ series length ({N})")
    K = N - L + 1

    # 1) Trajectory matrix
    X = np.column_stack([x[i:i+L] for i in range(K)])  # L×K

    # 2) Eigendecomposition
    S = X @ X.T
    λ, U = np.linalg.eigh(S)
    idx = np.argsort(λ)[::-1]      # descending
    λ, U = λ[idx], U[:, idx]
    # keep only first J
    Uj = U[:, :J]

    # 3) Reconstruction & diagonal‐averaging
    Xj = np.zeros_like(X)
    for m in range(J):
        Xj += np.sqrt(λ[m]) * np.outer(U[:, m], (X.T @ U[:, m]) / np.sqrt(λ[m]))
    # hankelization:
    rec = np.zeros(N)
    counts = np.zeros(N)
    for i in range(L):
        for j in range(K):
            rec[i+j]  += Xj[i,j]
            counts[i+j] += 1
    rec /= counts

    # 4) Recurrence coeffs a[0..L-2]
    #    Uj[L-1, m] are the last elements of each eigenvector
    nu = Uj[-1, :]                   # shape (J,)
    mu = Uj[:-1, :]                  # shape (L-1, J)
    den = np.sum(nu**2)
    a = (mu @ nu) / den              # shape (L-1,)

    # 5) Forecasting
    extended = list(rec.copy())
    for t in range(h):
        past = extended[-(L-1):]    # last L-1 reconstructed+forecasted
        next_val = np.dot(a, past[::-1])
        extended.append(next_val)
    forecast = np.array(extended[N:])

    score = forecast.mean()
    return forecast, score

if __name__ == "__main__":
    # --- Load data ---
    df = pd.read_csv(
        "crude_monthly_close.csv",
        index_col="Date",
        parse_dates=True
    )
    series = pd.to_numeric(df["CL=F"], errors="coerce").dropna()

    # --- Parameters ---
    L = 12    # embedding window
    J = 2     # number of components to keep
    h = 12    # forecast horizon

    # --- Run SSA + Forecast ---
    forecast_vals, forecast_score = ssa_forecast(series, L, J, h)

    # --- Prepare future dates ---
    last_date = series.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(),
        periods=h,
        freq="MS"
    )

    # --- Print the next 12 months and their forecasts ---
    print("\nForecast for the next 12 months:\n")
    for dt, val in zip(future_dates, forecast_vals):
        print(f"{dt.strftime('%Y-%m')} → {val:.2f}")

    # --- Print the SSA seasonal score of that forecast ---
    print(f"\nSSA seasonal score (mean of next {h} months): {forecast_score:.4f}\n")
