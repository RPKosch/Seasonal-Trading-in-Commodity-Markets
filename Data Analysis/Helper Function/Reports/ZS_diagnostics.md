# ZS — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **71,053**  
- Minimum monthly volume: **4,537**


## Exploratory Analysis

**Summary of monthly returns**
 count     mean      std      skew  kurtosis_excess       min      q25   median     q75      max
   292 0.006369 0.070101 -0.015356         0.718662 -0.219524 -0.03738 0.002086 0.04642 0.211885


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267 -0.003835    0.007325             -0.011159 -0.976307 0.336044    
02-Feb    25      267  0.030567    0.004103              0.026463  1.630370 0.114385    
03-Mar    25      267 -0.003954    0.007336             -0.011290 -0.830234 0.413038    
04-Apr    25      267  0.021351    0.004966              0.016384  1.428486 0.162618    
05-May    24      268 -0.001246    0.007051             -0.008297 -0.538325 0.594767    
06-Jun    24      268  0.021616    0.005004              0.016612  0.973600 0.339199    
07-Jul    24      268 -0.015191    0.008300             -0.023491 -1.407600 0.170987    
08-Aug    24      268 -0.002879    0.007197             -0.010076 -0.662238 0.513411    
09-Sep    24      268 -0.022493    0.008954             -0.031447 -1.630009 0.115517    
10-Oct    24      268  0.021529    0.005012              0.016517  1.127144 0.269425    
11-Nov    24      268  0.006786    0.006332              0.000454  0.040968 0.967574    
12-Dec    24      268  0.023403    0.004844              0.018560  1.491641 0.146318    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -7.642, p = 0.0000 ***, lags used = 8, n = 283  
  critical values: 1%=-3.454, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.117, p = 0.1000 , lags used = 3  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.1389, F p = 0.1377  
- **White**: LM p = 0.1389, F p = 0.1377
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.0444, F p = 0.0416  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Evidence of **heteroskedasticity**. Report **robust standard errors** (HC3; and **HAC/Newey–West** if serial correlation is present).


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.6314. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.06, p = 0.3963, break at index 174 (2015-07-01), left n = 174, right n = 118

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** No strong evidence of a single dominant break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2008-03-01  2008      3 -0.219524   -3.218935  0.040000 0.034814                  True               False            True
2004-05-01  2004      5 -0.199016   -2.947134  0.041667 0.030629                 False               False            True
2003-09-01  2003      9  0.172727    2.907988  0.041667 0.029844                 False               False            True
2018-06-01  2018      6 -0.156944   -2.653247  0.041667 0.024968                 False               False            True
2005-02-01  2005      2  0.211885    2.692894  0.040000 0.024630                 False               False            True
2008-09-01  2008      9 -0.197389   -2.597462  0.041667 0.023953                 False               False            True
2011-09-01  2011      9 -0.189691   -2.480556  0.041667 0.021891                 False               False            True
2023-06-01  2023      6  0.185368    2.428351  0.041667 0.020998                 False               False            True
2004-07-01  2004      7 -0.176934   -2.397933  0.041667 0.020486                 False               False            True
2003-08-01  2003      8  0.157171    2.372329  0.041667 0.020060                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2008-03-01  2008      3 -0.219524 -3.635650             True
2005-02-01  2005      2  0.211885  3.441889            False
2004-05-01  2004      5 -0.199016 -3.299201            False
2008-09-01  2008      9 -0.197389 -3.272504            False
2011-09-01  2011      9 -0.189691 -3.146215            False
2008-02-01  2008      2  0.188781  3.062860            False
2023-06-01  2023      6  0.185368  3.006858            False
2004-07-01  2004      7 -0.176934 -2.936934            False
2003-10-01  2003     10  0.174890  2.834955            False
2008-06-01  2008      6  0.173059  2.804928            False
