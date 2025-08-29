# SU — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **48,129**  
- Minimum monthly volume: **8,101**


## Exploratory Analysis

**Summary of monthly returns**
 count      mean      std      skew  kurtosis_excess       min       q25   median      q75      max
   292 -0.001415 0.087578 -0.251002         1.081029 -0.352447 -0.058458 0.001031 0.051891 0.265115


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.020596   -0.003475              0.024072  1.243517 0.223965    
02-Feb    25      267 -0.002809   -0.001284             -0.001525 -0.100589 0.920512    
03-Mar    25      267 -0.063150    0.004366             -0.067516 -2.831394 0.008774 ***
04-Apr    25      267 -0.006717   -0.000918             -0.005799 -0.330683 0.743241    
05-May    24      268 -0.015461   -0.000157             -0.015304 -1.002025 0.324319    
06-Jun    24      268  0.025984   -0.003868              0.029852  1.703289 0.099601   *
07-Jul    24      268  0.015591   -0.002937              0.018528  0.900840 0.375827    
08-Aug    24      268 -0.016858   -0.000032             -0.016827 -0.844298 0.406020    
09-Sep    24      268  0.013695   -0.002768              0.016463  0.950127 0.350132    
10-Oct    24      268  0.011515   -0.002572              0.014088  0.813968 0.422489    
11-Nov    24      268  0.001263   -0.001654              0.002917  0.216843 0.829668    
12-Dec    24      268  0.001310   -0.001659              0.002969  0.154721 0.878191    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -15.401, p = 0.0000 ***, lags used = 0, n = 291  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.070, p = 0.1000 , lags used = 4  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.2672, F p = 0.2694  
- **White**: LM p = 0.2672, F p = 0.2694
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.0129, F p = 0.0111  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Evidence of **heteroskedasticity**. Report **robust standard errors** (HC3; and **HAC/Newey–West** if serial correlation is present).


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.7231. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.75, p = 0.0567, break at index 254 (2022-03-01), left n = 254, right n = 38

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** Weak or borderline evidence of a break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2010-03-01  2010      3 -0.352447   -3.494899  0.040000 0.040778                  True               False            True
2009-08-01  2009      8  0.265115    3.405792  0.041667 0.040494                  True               False            True
2020-03-01  2020      3 -0.313030   -3.002076  0.040000 0.030423                  True               False            True
2023-12-01  2023     12 -0.239529   -2.892748  0.041667 0.029541                 False               False            True
2006-08-01  2006      8 -0.236612   -2.632877  0.041667 0.024595                 False               False            True
2010-10-01  2010     10  0.204685    2.307867  0.041667 0.019004                 False               False            True
2010-07-01  2010      7  0.204539    2.256498  0.041667 0.018183                 False               False            True
2023-04-01  2023      4  0.184982    2.287928  0.040000 0.017905                 False               False            True
2006-01-01  2006      1  0.210461    2.265637  0.040000 0.017564                 False               False            True
2010-09-01  2010      9  0.192940    2.138667  0.041667 0.016363                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2010-03-01  2010      3 -0.352447 -4.220797             True
2020-03-01  2020      3 -0.313030 -3.750131             True
2009-08-01  2009      8  0.265115  3.153364            False
2023-12-01  2023     12 -0.239529 -2.872479            False
2006-08-01  2006      8 -0.236612 -2.837640            False
2008-03-01  2008      3 -0.225707 -2.707423            False
2006-01-01  2006      1  0.210461  2.500750            False
2010-10-01  2010     10  0.204685  2.431785            False
2010-07-01  2010      7  0.204539  2.430039            False
2010-09-01  2010      9  0.192940  2.291538            False
