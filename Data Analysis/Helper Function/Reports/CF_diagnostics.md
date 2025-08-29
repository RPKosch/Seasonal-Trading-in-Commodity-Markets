# CF — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **15,248**  
- Minimum monthly volume: **4,063**


## Exploratory Analysis

**Summary of monthly returns**
 count      mean      std     skew  kurtosis_excess       min       q25   median      q75      max
   292 -0.001702 0.086806 0.406606         0.949945 -0.260128 -0.057193 -0.01045 0.042984 0.350435


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.017429   -0.003493              0.020922  1.195591 0.241500    
02-Feb    25      267  0.003082   -0.002150              0.005231  0.240113 0.812058    
03-Mar    25      267 -0.022294    0.000226             -0.022520 -1.268428 0.214754    
04-Apr    25      267  0.008383   -0.002646              0.011029  0.674194 0.505333    
05-May    24      268 -0.016465   -0.000380             -0.016085 -0.743775 0.463703    
06-Jun    24      268 -0.014061   -0.000595             -0.013466 -0.712731 0.482110    
07-Jul    24      268 -0.003780   -0.001516             -0.002264 -0.134530 0.893931    
08-Aug    24      268  0.000145   -0.001867              0.002012  0.136366 0.892425    
09-Sep    24      268 -0.013972   -0.000603             -0.013369 -0.753462 0.457515    
10-Oct    24      268 -0.014217   -0.000581             -0.013636 -0.662580 0.513365    
11-Nov    24      268  0.033760   -0.004877              0.038638  1.958940 0.060679   *
12-Dec    24      268  0.001010   -0.001945              0.002955  0.183733 0.855497    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -18.511, p = 0.0000 ***, lags used = 0, n = 291  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.254, p = 0.1000 , lags used = 4  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.8121, F p = 0.8197  
- **White**: LM p = 0.8121, F p = 0.8197
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.9957, F p = 0.9961  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Tests **do not suggest heteroskedasticity** at conventional levels. A homoskedastic assumption appears reasonable; **robust SEs** are still a conservative default in finance.


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.2464. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.71, p = 0.0639, break at index 234 (2020-07-01), left n = 234, right n = 58

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** Weak or borderline evidence of a break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2014-02-01  2014      2  0.350435    4.189901  0.040000 0.057553                  True               False            True
2020-01-01  2020      1 -0.232742   -2.972970  0.040000 0.029854                 False               False            True
2008-03-01  2008      3 -0.260128   -2.822063  0.040000 0.026982                 False               False            True
2004-05-01  2004      5  0.211378    2.702752  0.041667 0.025884                 False               False            True
2024-11-01  2024     11  0.258708    2.667528  0.041667 0.025230                 False               False            True
2002-03-01  2002      3  0.207102    2.719234  0.040000 0.025101                 False               False            True
2010-06-01  2010      6  0.201743    2.556503  0.041667 0.023221                 False               False            True
2011-09-01  2011      9 -0.226720   -2.519471  0.041667 0.022568                 False               False            True
2002-10-01  2002     10  0.195294    2.480289  0.041667 0.021887                 False               False            True
2022-10-01  2022     10 -0.217161   -2.400900  0.041667 0.020536                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2014-02-01  2014      2  0.350435  4.833884             True
2024-11-01  2024     11  0.258708  3.605244             True
2008-03-01  2008      3 -0.260128 -3.344311            False
2004-11-01  2004     11  0.229061  3.208143            False
2020-01-01  2020      1 -0.232742 -2.977489            False
2004-05-01  2004      5  0.211378  2.971280            False
2002-03-01  2002      3  0.207102  2.914001            False
2011-09-01  2011      9 -0.226720 -2.896824            False
2010-06-01  2010      6  0.201743  2.842228            False
2022-10-01  2022     10 -0.217161 -2.768787            False
