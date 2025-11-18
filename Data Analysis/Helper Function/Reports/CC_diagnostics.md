# CC — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **13,575**  
- Minimum monthly volume: **2,857**


## Exploratory Analysis

**Summary of monthly returns**
 count     mean      std     skew  kurtosis_excess       min      q25   median      q75      max
   292 0.012854 0.095482 1.136568         5.116534 -0.254618 -0.04547 0.008392 0.061227 0.595752


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.029462    0.011299              0.018163  0.872157 0.390483    
02-Feb    25      267  0.026000    0.011623              0.014376  0.669610 0.508607    
03-Mar    25      267  0.009019    0.013213             -0.004194 -0.135368 0.893382    
04-Apr    25      267  0.018165    0.012357              0.005809  0.334876 0.739996    
05-May    24      268 -0.015968    0.015435             -0.031403 -1.808354 0.080815   *
06-Jun    24      268  0.015997    0.012573              0.003424  0.215946 0.830441    
07-Jul    24      268  0.012160    0.012916             -0.000757 -0.040996 0.967585    
08-Aug    24      268  0.016679    0.012512              0.004168  0.240862 0.811331    
09-Sep    24      268  0.006313    0.013440             -0.007127 -0.390565 0.699009    
10-Oct    24      268 -0.023127    0.016076             -0.039204 -2.487579 0.018432  **
11-Nov    24      268  0.041190    0.010317              0.030874  1.244094 0.224733    
12-Dec    24      268  0.017058    0.012478              0.004581  0.248445 0.805572    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -8.144, p = 0.0000 ***, lags used = 3, n = 288  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.236, p = 0.1000 , lags used = 6  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.3617, F p = 0.3664  
- **White**: LM p = 0.3617, F p = 0.3664
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.0119, F p = 0.0102  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Evidence of **heteroskedasticity**. Report **robust standard errors** (HC3; and **HAC/Newey–West** if serial correlation is present).


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.2787. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 2.41, p = 0.0055, break at index 267 (2023-04-01), left n = 267, right n = 25

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** Break suggested at the reported date; parameter shifts before/after that point.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2024-03-01  2024      3  0.595752    6.728697  0.040000 0.135742                  True               False            True
2024-11-01  2024     11  0.367925    3.556612  0.041667 0.044001                  True               False            True
2001-01-01  2001      1  0.328125    3.236076  0.040000 0.035172                  True               False            True
2001-11-01  2001     11  0.307617    2.878365  0.041667 0.029257                 False               False            True
2024-02-01  2024      2  0.270531    2.633310  0.040000 0.023578                 False               False            True
2003-05-01  2003      5 -0.254618   -2.570743  0.041667 0.023474                 False               False            True
2024-12-01  2024     12  0.241229    2.411413  0.041667 0.020712                 False               False            True
2003-08-01  2003      8  0.231469    2.308524  0.041667 0.019015                 False               False            True
2004-07-01  2004      7  0.222140    2.255881  0.041667 0.018173                 False               False            True
2011-03-01  2011      3 -0.201083   -2.255213  0.040000 0.017406                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2024-03-01  2024      3  0.595752  7.427008             True
2024-11-01  2024     11  0.367925  4.546192             True
2001-01-01  2001      1  0.328125  4.042937             True
2001-11-01  2001     11  0.307617  3.783621             True
2003-05-01  2003      5 -0.254618 -3.325702            False
2024-02-01  2024      2  0.270531  3.314681            False
2024-12-01  2024     12  0.241229  2.944159            False
2003-08-01  2003      8  0.231469  2.820740            False
2004-07-01  2004      7  0.222140  2.702779            False
2008-10-01  2008     10 -0.202718 -2.669444            False
