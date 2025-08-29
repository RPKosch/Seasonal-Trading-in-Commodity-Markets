# NG — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **71,479**  
- Minimum monthly volume: **9,241**


## Exploratory Analysis

**Summary of monthly returns**
 count      mean      std    skew  kurtosis_excess       min       q25   median      q75      max
   292 -0.027198 0.142902 -0.0955         0.694115 -0.424135 -0.113515 -0.02002 0.059802 0.420514


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267 -0.056545   -0.024450             -0.032096 -0.948793 0.351009    
02-Feb    25      267 -0.014683   -0.028370              0.013686  0.448199 0.657404    
03-Mar    25      267 -0.004019   -0.029368              0.025349  0.853061 0.400676    
04-Apr    25      267  0.006883   -0.030389              0.037272  1.658661 0.106458    
05-May    24      268 -0.017920   -0.028029              0.010108  0.484018 0.631441    
06-Jun    24      268 -0.025973   -0.027307              0.001334  0.043809 0.965376    
07-Jul    24      268 -0.019709   -0.027868              0.008159  0.253245 0.802016    
08-Aug    24      268 -0.030840   -0.026872             -0.003968 -0.107860 0.914945    
09-Sep    24      268 -0.009248   -0.028805              0.019557  0.595135 0.556791    
10-Oct    24      268 -0.025766   -0.027326              0.001560  0.067186 0.946858    
11-Nov    24      268 -0.053058   -0.024882             -0.028176 -0.942184 0.354307    
12-Dec    24      268 -0.077178   -0.022722             -0.054456 -1.459715 0.156546    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -4.797, p = 0.0001 ***, lags used = 10, n = 281  
  critical values: 1%=-3.454, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.052, p = 0.1000 , lags used = 4  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.2756, F p = 0.2780  
- **White**: LM p = 0.2756, F p = 0.2780
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.0008, F p = 0.0005  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Evidence of **heteroskedasticity**. Report **robust standard errors** (HC3; and **HAC/Newey–West** if serial correlation is present).


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.9115. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.73, p = 0.0613, break at index 47 (2004-12-01), left n = 47, right n = 245

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** Weak or borderline evidence of a break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2003-02-01  2003      2  0.420514    3.138028  0.040000 0.033144                  True               False            True
2022-07-01  2022      7  0.385076    2.914429  0.041667 0.029973                 False               False            True
2022-06-01  2022      6 -0.419380   -2.830111  0.041667 0.028311                 False               False            True
2005-08-01  2005      8  0.360468    2.814581  0.041667 0.028010                 False               False            True
2018-11-01  2018     11  0.327493    2.735105  0.041667 0.026491                 False               False            True
2008-07-01  2008      7 -0.386013   -2.630122  0.041667 0.024545                 False               False            True
2022-01-01  2022      1  0.315841    2.672524  0.040000 0.024268                 False               False            True
2006-09-01  2006      9 -0.359340   -2.511024  0.041667 0.022420                 False               False            True
2022-12-01  2022     12 -0.424135   -2.488042  0.041667 0.022021                 False               False            True
2023-01-01  2023      1 -0.400235   -2.461911  0.040000 0.020672                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2003-02-01  2003      2  0.420514  3.464211            False
2022-07-01  2022      7  0.385076  3.185541            False
2022-12-01  2022     12 -0.424135 -3.177829            False
2022-06-01  2022      6 -0.419380 -3.140433            False
2005-08-01  2005      8  0.360468  2.992026            False
2023-01-01  2023      1 -0.400235 -2.989889            False
2001-01-01  2001      1 -0.387728 -2.891535            False
2008-07-01  2008      7 -0.386013 -2.878052            False
2018-12-01  2018     12 -0.382528 -2.850647            False
2018-11-01  2018     11  0.327493  2.732721            False
