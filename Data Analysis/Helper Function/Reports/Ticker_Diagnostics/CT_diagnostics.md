# CT — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **13,834**  
- Minimum monthly volume: **1,061**


## Exploratory Analysis

**Summary of monthly returns**
 count      mean      std     skew  kurtosis_excess       min       q25   median      q75      max
   292 -0.000977 0.081777 0.071142         0.681106 -0.244779 -0.049144 0.003941 0.044302 0.252632


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.013497   -0.002332              0.015829  1.294412 0.203956    
02-Feb    25      267  0.016663   -0.002629              0.019291  1.034212 0.309961    
03-Mar    25      267 -0.013581    0.000203             -0.013784 -0.796060 0.432567    
04-Apr    25      267 -0.005517   -0.000552             -0.004965 -0.280582 0.781071    
05-May    24      268 -0.031666    0.001771             -0.033437 -1.981034 0.057638   *
06-Jun    24      268 -0.008103   -0.000339             -0.007765 -0.433907 0.667800    
07-Jul    24      268 -0.011093   -0.000071             -0.011021 -0.723606 0.475102    
08-Aug    24      268  0.008854   -0.001857              0.010711  0.630986 0.533244    
09-Sep    24      268 -0.014417    0.000227             -0.014643 -0.669039 0.509497    
10-Oct    24      268 -0.002512   -0.000839             -0.001673 -0.081292 0.935836    
11-Nov    24      268  0.008884   -0.001860              0.010744  0.644552 0.524502    
12-Dec    24      268  0.026643   -0.003450              0.030093  2.167902 0.038083  **

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -6.929, p = 0.0000 ***, lags used = 8, n = 283  
  critical values: 1%=-3.454, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.106, p = 0.1000 , lags used = 4  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.4061, F p = 0.4117  
- **White**: LM p = 0.4061, F p = 0.4117
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.0006, F p = 0.0004  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Evidence of **heteroskedasticity**. Report **robust standard errors** (HC3; and **HAC/Newey–West** if serial correlation is present).


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.3334. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.36, p = 0.1850, break at index 25 (2003-02-01), left n = 25, right n = 267

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** No strong evidence of a single dominant break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2001-11-01  2001     11  0.252632    3.088828  0.041667 0.033545                  True               False            True
2008-10-01  2008     10 -0.236511   -2.961325  0.041667 0.030916                 False               False            True
2010-10-01  2010     10  0.229244    2.932070  0.041667 0.030326                 False               False            True
2022-09-01  2022      9 -0.244779   -2.913889  0.041667 0.029962                 False               False            True
2010-12-01  2010     12  0.234106    2.616726  0.041667 0.024301                 False               False            True
2004-08-01  2004      8  0.213393    2.578956  0.041667 0.023621                 False               False            True
2010-09-01  2010      9  0.182367    2.478972  0.041667 0.021864                 False               False            True
2022-06-01  2022      6 -0.192813   -2.323835  0.041667 0.019263                 False               False            True
2011-04-01  2011      4 -0.182852   -2.227404  0.040000 0.016987                 False               False            True
2009-04-01  2009      4  0.168566    2.185833  0.040000 0.016369                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2022-09-01  2022      9 -0.244779 -3.569709             True
2001-11-01  2001     11  0.252632  3.569284             True
2008-10-01  2008     10 -0.236511 -3.451047            False
2010-12-01  2010     12  0.234106  3.303399            False
2010-10-01  2010     10  0.229244  3.233623            False
2004-08-01  2004      8  0.213393  3.006117            False
2012-05-01  2012      5 -0.199664 -2.922214            False
2022-06-01  2022      6 -0.192813 -2.823885            False
2011-04-01  2011      4 -0.182852 -2.680922            False
2020-03-01  2020      3 -0.181265 -2.658139            False
