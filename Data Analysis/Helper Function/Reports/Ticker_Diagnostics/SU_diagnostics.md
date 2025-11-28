# SU — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **48,129**  
- Minimum monthly volume: **8,101**


## Exploratory Analysis

**Summary of monthly returns**
 count     mean      std     skew  kurtosis_excess       min       q25   median      q75      max
   292 0.002385 0.087176 0.121505         0.732349 -0.297034 -0.056782 0.001032 0.053261 0.303581


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.025051    0.000262              0.024788  1.255133 0.219871    
02-Feb    25      267 -0.000422    0.002648             -0.003069 -0.203936 0.839709    
03-Mar    25      267 -0.055249    0.007781             -0.063030 -2.909145 0.007185 ***
04-Apr    25      267 -0.003324    0.002919             -0.006243 -0.350448 0.728537    
05-May    24      268 -0.013050    0.003767             -0.016817 -1.125371 0.269233    
06-Jun    24      268  0.029577   -0.000050              0.029627  1.673294 0.105489    
07-Jul    24      268  0.020321    0.000779              0.019542  0.936732 0.357447    
08-Aug    24      268 -0.012457    0.003714             -0.016171 -0.796270 0.432966    
09-Sep    24      268  0.016965    0.001079              0.015886  0.898699 0.376517    
10-Oct    24      268  0.014789    0.001274              0.013516  0.752118 0.458333    
11-Nov    24      268  0.003019    0.002328              0.000691  0.050700 0.959875    
12-Dec    24      268  0.005208    0.002132              0.003076  0.160416 0.873748    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -15.387, p = 0.0000 ***, lags used = 0, n = 291  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.082, p = 0.1000 , lags used = 3  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.5547, F p = 0.5628  
- **White**: LM p = 0.5547, F p = 0.5628
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.0231, F p = 0.0207  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Evidence of **heteroskedasticity**. Report **robust standard errors** (HC3; and **HAC/Newey–West** if serial correlation is present).


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.5335. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.65, p = 0.0772, break at index 243 (2021-04-01), left n = 243, right n = 49

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** Weak or borderline evidence of a break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2009-08-01  2009      8  0.303581    3.846902  0.041667 0.051100                  True               False            True
2010-03-01  2010      3 -0.297034   -2.908632  0.040000 0.028613                 False               False            True
2023-12-01  2023     12 -0.213002   -2.620007  0.041667 0.024361                 False               False            True
2010-10-01  2010     10  0.227139    2.547975  0.041667 0.023070                 False               False            True
2020-03-01  2020      3 -0.268772   -2.560116  0.040000 0.022315                 False               False            True
2010-07-01  2010      7  0.226959    2.477922  0.041667 0.021846                 False               False            True
2006-01-01  2006      1  0.234247    2.507049  0.040000 0.021420                 False               False            True
2023-04-01  2023      4  0.203196    2.474278  0.040000 0.020875                 False               False            True
2006-08-01  2006      8 -0.210702   -2.375201  0.041667 0.020107                 False               False            True
2010-09-01  2010      9  0.212810    2.345860  0.041667 0.019623                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2009-08-01  2009      8  0.303581  3.682588             True
2010-03-01  2010      3 -0.297034 -3.628014             True
2020-03-01  2020      3 -0.268772 -3.284013            False
2006-01-01  2006      1  0.234247  2.838659            False
2010-10-01  2010     10  0.227139  2.752143            False
2010-07-01  2010      7  0.226959  2.749959            False
2023-12-01  2023     12 -0.213002 -2.605188            False
2010-09-01  2010      9  0.212810  2.577735            False
2006-08-01  2006      8 -0.210702 -2.577198            False
2008-03-01  2008      3 -0.202048 -2.471856            False
