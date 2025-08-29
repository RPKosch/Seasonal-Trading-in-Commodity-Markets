# CT — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **13,834**  
- Minimum monthly volume: **1,061**


## Exploratory Analysis

**Summary of monthly returns**
 count      mean      std      skew  kurtosis_excess       min       q25   median      q75      max
   292 -0.004344 0.082475 -0.257379         0.767716 -0.280745 -0.050392 0.003933 0.043348 0.225247


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.012005   -0.005875              0.017881  1.500148 0.142230    
02-Feb    25      267  0.012805   -0.005950              0.018756  1.026294 0.313522    
03-Mar    25      267 -0.017196   -0.003141             -0.014055 -0.779916 0.441936    
04-Apr    25      267 -0.009058   -0.003903             -0.005155 -0.287751 0.775636    
05-May    24      268 -0.035381   -0.001565             -0.033816 -1.937105 0.063142   *
06-Jun    24      268 -0.011646   -0.003691             -0.007955 -0.435705 0.666526    
07-Jul    24      268 -0.013639   -0.003512             -0.010127 -0.647316 0.522565    
08-Aug    24      268  0.005936   -0.005265              0.011201  0.678698 0.502896    
09-Sep    24      268 -0.020054   -0.002938             -0.017116 -0.756242 0.456491    
10-Oct    24      268 -0.007263   -0.004083             -0.003180 -0.150807 0.881304    
11-Nov    24      268  0.006161   -0.005285              0.011447  0.720292 0.477208    
12-Dec    24      268  0.024530   -0.006930              0.031461  2.362224 0.024501  **

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -6.970, p = 0.0000 ***, lags used = 8, n = 283  
  critical values: 1%=-3.454, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.122, p = 0.1000 , lags used = 4  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.2374, F p = 0.2389  
- **White**: LM p = 0.2374, F p = 0.2389
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.0029, F p = 0.0021  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Evidence of **heteroskedasticity**. Report **robust standard errors** (HC3; and **HAC/Newey–West** if serial correlation is present).


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.2895. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.24, p = 0.2555, break at index 24 (2003-01-01), left n = 24, right n = 268

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** No strong evidence of a single dominant break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2008-10-01  2008     10 -0.269856   -3.311237  0.041667 0.038361                  True               False            True
2022-09-01  2022      9 -0.280745   -3.286305  0.041667 0.037807                  True               False            True
2001-11-01  2001     11  0.225247    2.746254  0.041667 0.026702                 False               False            True
2010-10-01  2010     10  0.206400    2.676510  0.041667 0.025396                 False               False            True
2022-06-01  2022      6 -0.214200   -2.534076  0.041667 0.022825                 False               False            True
2011-04-01  2011      4 -0.201936   -2.408289  0.040000 0.019799                 False               False            True
2010-09-01  2010      9  0.167518    2.342794  0.041667 0.019573                 False               False            True
2004-08-01  2004      8  0.193420    2.341676  0.041667 0.019554                 False               False            True
2012-05-01  2012      5 -0.222724   -2.339875  0.041667 0.019525                 False               False            True
2010-12-01  2010     12  0.210347    2.320442  0.041667 0.019208                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2022-09-01  2022      9 -0.280745 -4.078142             True
2008-10-01  2008     10 -0.269856 -3.922166             True
2012-05-01  2012      5 -0.222724 -3.246974            False
2001-11-01  2001     11  0.225247  3.170411            False
2022-06-01  2022      6 -0.214200 -3.124867            False
2010-12-01  2010     12  0.210347  2.956965            False
2011-04-01  2011      4 -0.201936 -2.949168            False
2020-03-01  2020      3 -0.199995 -2.921366            False
2010-10-01  2010     10  0.206400  2.900420            False
2008-09-01  2008      9 -0.193611 -2.829919            False
