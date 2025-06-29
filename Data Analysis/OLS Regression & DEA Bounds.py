import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ------------------- User parameters -------------------
np.random.seed(0)
M = 100
max_score = 1.5   # hard ceiling on any predicted score
w_true = 0.7      # underlying true mixture

# ------------------- Synthetic data -------------------
SR = np.random.uniform(0.5, 2.0, size=M)
TR = np.random.uniform(0.2, 1.0, size=M)
E_true = w_true*SR + (1-w_true)*TR + np.random.normal(scale=0.01, size=M)

# ------------------- 1) OLS fit to E_true -------------------
X = np.vstack([SR, TR]).T
u_ols, v_ols = np.linalg.lstsq(X, E_true, rcond=None)[0]
pred_ols = u_ols*SR + v_ols*TR

# ------------------- 2) Constrained SLSQP fit -------------------
def objective(w):
    u, v = w
    return np.sum((u*SR + v*TR - E_true)**2)

# constraints: u*SR[i] + v*TR[i] <= max_score
cons = [{'type': 'ineq', 'fun': lambda w, i=i: max_score - (w[0]*SR[i] + w[1]*TR[i])}
        for i in range(M)]
bounds = [(0, None), (0, None)]
res = minimize(objective, x0=[u_ols, v_ols], method='SLSQP', bounds=bounds, constraints=cons)
u_con, v_con = res.x
pred_con = u_con*SR + v_con*TR

# ------------------- 3) Report weights & table -------------------
print("=== Weights ===")
print(f"OLS:        u={u_ols:.4f}, v={v_ols:.4f}")
print(f"Constrained u={u_con:.4f}, v={v_con:.4f}\n")

df = pd.DataFrame({
    'SR': SR,
    'TR': TR,
    'E_true': E_true,
    'pred_OLS': pred_ols,
    'pred_Constrained': pred_con
})
print(df.head(10))

# ------------------- 4) Surface plots -------------------
srg = np.linspace(SR.min(), SR.max(), 100)
trg = np.linspace(TR.min(), TR.max(), 100)
SRg, TRg = np.meshgrid(srg, trg)
Z_true = w_true*SRg + (1-w_true)*TRg
Z_ols  = u_ols*SRg + v_ols*TRg
Z_con  = u_con*SRg + v_con*TRg

fig, axes = plt.subplots(1, 3, figsize=(15,5), sharey=True)
for ax, Z, title in zip(axes, [Z_true, Z_ols, Z_con], ['True', 'OLS', 'Constrained']):
    cs = ax.contourf(SRg, TRg, Z, levels=20, cmap='viridis')
    ax.scatter(SR, TR, c=(E_true if title=='True' else (pred_ols if title=='OLS' else pred_con)), edgecolor='k')
    ax.set_title(f"{title} surface")
    ax.set_xlabel("SR")
    if ax is axes[0]:
        ax.set_ylabel("TR")
    fig.colorbar(cs, ax=ax)
plt.tight_layout()
plt.show()

# ------------------- 5) Scatter + regression line plots -------------------
fig, axs = plt.subplots(1, 2, figsize=(12,5))
# OLS
axs[0].scatter(E_true, pred_ols, color='blue', label='Data')
m_ols, b_ols = np.polyfit(E_true, pred_ols, 1)
x_line = np.array([E_true.min(), E_true.max()])
axs[0].plot(x_line, m_ols*x_line + b_ols, 'r-', label=f'y={m_ols:.2f}x+{b_ols:.2f}')
axs[0].plot(x_line, x_line, 'k--', label='Ideal')
axs[0].set_title('OLS vs True')
axs[0].set_xlabel('E_true'); axs[0].set_ylabel('pred_OLS')
axs[0].legend(); axs[0].grid(True)

# Constrained
axs[1].scatter(E_true, pred_con, color='green', label='Data')
m_con, b_con = np.polyfit(E_true, pred_con, 1)
axs[1].plot(x_line, m_con*x_line + b_con, 'r-', label=f'y={m_con:.2f}x+{b_con:.2f}')
axs[1].plot(x_line, x_line, 'k--', label='Ideal')
axs[1].set_title('Constrained vs True')
axs[1].set_xlabel('E_true'); axs[1].set_ylabel('pred_Constrained')
axs[1].legend(); axs[1].grid(True)

plt.tight_layout()
plt.show()
