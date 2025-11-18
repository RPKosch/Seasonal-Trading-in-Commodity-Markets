# ZC — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **117,352**  
- Minimum monthly volume: **11,093**


## Exploratory Analysis

**Summary of monthly returns**
 count     mean      std     skew  kurtosis_excess       min       q25    median      q75      max
   292 -0.00136 0.076361 0.302746         0.770075 -0.226501 -0.043887 -0.007558 0.044903 0.255253


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.008778   -0.002309              0.011088  0.817451 0.419890    
02-Feb    25      267  0.012340   -0.002643              0.014983  1.206282 0.236309    
03-Mar    25      267 -0.008477   -0.000694             -0.007783 -0.589460 0.559742    
04-Apr    25      267  0.007376   -0.002178              0.009554  0.716611 0.478907    
05-May    24      268 -0.000729   -0.001417              0.000687  0.048724 0.961472    
06-Jun    24      268 -0.030900    0.001285             -0.032185 -1.443091 0.161442    
07-Jul    24      268 -0.026068    0.000852             -0.026920 -1.253017 0.221731    
08-Aug    24      268 -0.001756   -0.001325             -0.000431 -0.027882 0.977955    
09-Sep    24      268 -0.019468    0.000261             -0.019730 -1.070321 0.294258    
10-Oct    24      268  0.028110   -0.003999              0.032110  1.759887 0.090104   *
11-Nov    24      268 -0.010930   -0.000503             -0.010426 -0.783136 0.439663    
12-Dec    24      268  0.024342   -0.003662              0.028003  2.102479 0.044001  **

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -10.753, p = 0.0000 ***, lags used = 1, n = 290  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.049, p = 0.1000 , lags used = 2  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.0316, F p = 0.0293  
- **White**: LM p = 0.0316, F p = 0.0293
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.0017, F p = 0.0012  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Evidence of **heteroskedasticity**. Report **robust standard errors** (HC3; and **HAC/Newey–West** if serial correlation is present).


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.9609. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.30, p = 0.2177, break at index 174 (2015-07-01), left n = 174, right n = 118

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** No strong evidence of a single dominant break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2012-07-01  2012      7  0.255253    3.891527  0.041667 0.052231                  True               False            True
2008-06-01  2008      6  0.207941    3.279154  0.041667 0.037648                  True               False            True
2012-06-01  2012      6  0.195435    3.101365  0.041667 0.033809                  True               False            True
2015-06-01  2015      6  0.184561    2.947607  0.041667 0.030638                 False               False            True
2008-10-01  2008     10 -0.187247   -2.946137  0.041667 0.030609                 False               False            True
2011-09-01  2011      9 -0.226501   -2.828921  0.041667 0.028288                 False               False            True
2021-04-01  2021      4  0.210881    2.776899  0.040000 0.026148                 False               False            True
2006-10-01  2006     10  0.218424    2.594699  0.041667 0.023904                 False               False            True
2019-05-01  2019      5  0.177931    2.432346  0.041667 0.021066                 False               False            True
2006-11-01  2006     11  0.167414    2.427942  0.041667 0.020991                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2012-07-01  2012      7  0.255253  4.089844             True
2006-10-01  2006     10  0.218424  3.516707             True
2011-09-01  2011      9 -0.226501 -3.407193            False
2021-04-01  2021      4  0.210881  3.399335            False
2008-06-01  2008      6  0.207941  3.353579            False
2012-06-01  2012      6  0.195435  3.158962            False
2009-06-01  2009      6 -0.201577 -3.019316            False
2015-06-01  2015      6  0.184561  2.989746            False
2008-07-01  2008      7 -0.198499 -2.971428            False
2019-05-01  2019      5  0.177931  2.886564            False
