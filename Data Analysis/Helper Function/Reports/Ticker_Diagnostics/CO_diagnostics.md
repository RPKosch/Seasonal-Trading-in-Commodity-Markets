# CO — Monthly Diagnostics

## Data

- Observations: **292**  
- Date range: **2001-01-01 — 2025-04-01**
- Average monthly volume: **225,595**  
- Minimum monthly volume: **48,893**


## Exploratory Analysis

**Summary of monthly returns**
 count     mean      std     skew  kurtosis_excess       min       q25   median      q75      max
   292 0.003978 0.101769 0.078817         6.300238 -0.533804 -0.053646 0.009981 0.062558 0.610254


**Month vs. Rest-of-Year (Welch two-sample t-tests)**

 month  n(m)  n(rest)   mean(m)  mean(rest)  mean(m) − mean(rest)    t_stat  p_value sig
01-Jan    25      267  0.007070    0.003688              0.003382  0.176705 0.860922    
02-Feb    25      267  0.029990    0.001542              0.028448  1.681561 0.102276    
03-Mar    25      267  0.001095    0.004248             -0.003153 -0.119244 0.905973    
04-Apr    25      267  0.010216    0.003394              0.006822  0.332100 0.742191    
05-May    24      268  0.022577    0.002312              0.020265  0.595994 0.556672    
06-Jun    24      268  0.025792    0.002024              0.023767  1.655848 0.106562    
07-Jul    24      268  0.004947    0.003891              0.001057  0.055443 0.956166    
08-Aug    24      268 -0.002219    0.004533             -0.006752 -0.464821 0.644925    
09-Sep    24      268 -0.008866    0.005128             -0.013994 -0.717374 0.478964    
10-Oct    24      268 -0.020466    0.006167             -0.026633 -1.125225 0.270603    
11-Nov    24      268 -0.030148    0.007034             -0.037182 -1.627240 0.115417    
12-Dec    24      268  0.006393    0.003761              0.002632  0.123274 0.902788    

_Interpretation:_ For each row, we test whether the average return in that month differs from the average return in all the other months. Small p-values (e.g., < 0.05) indicate the month’s average return is significantly different from the rest of the year.


## Stationarity

- **ADF**: stat = -14.251, p = 0.0000 ***, lags used = 0, n = 291  
  critical values: 1%=-3.453, 5%=-2.872, 10%=-2.572

- **KPSS (level)**: stat = 0.133, p = 0.1000 , lags used = 0  
  critical values: 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739

- **Joint read**: ADF rejects & KPSS does not reject → returns look stationary.


## Homoskedasticity (constant error variance)

- **Breusch–Pagan**: LM p = 0.3680, F p = 0.3728  
- **White**: LM p = 0.3680, F p = 0.3728
  *Interpretation:* **small p-values (<0.05)** suggest **heteroskedasticity conditional on the regressors** (here: month dummies).

- **ARCH LM (lags=12)**: LM p = 0.0000, F p = 0.0000  
  *Interpretation:* **small p-values (<0.05)** indicate **conditional heteroskedasticity** (volatility clustering) in residuals.

**Conclusion:** Evidence of **heteroskedasticity**. Report **robust standard errors** (HC3; and **HAC/Newey–West** if serial correlation is present).


## Structural Breaks

- **CUSUM (parameter stability test)**: p = 0.4143. CUSUM tracks cumulative sums of recursive residuals; **small p-values (< 0.05)** indicate **time-varying coefficients** (instability) rather than constant effects over the sample.

  **Interpretation:** No evidence against stability — parameters appear stable.

- **Chow (rolling break search)**: F = 1.45, p = 0.1419, break at index 221 (2019-06-01), left n = 221, right n = 71

  *What it does:* For each admissible split point, the test compares a **single-regime fit** to a **two-regime fit** (pre-break vs post-break). **Small p-values (< 0.05)** indicate a **structural break** at (or near) the reported split.

  **Interpretation:** No strong evidence of a single dominant break.


## Outliers

**Regression influence (top 10 by Cook’s D)**
      date  year  month    return  stud_resid  leverage  cooks_d  flag_studentized>|3|  flag_leverage>2k/n  flag_cooks>4/n
2020-05-01  2020      5  0.610254    6.257168  0.041667 0.124845                  True               False            True
2020-03-01  2020      3 -0.533804   -5.622441  0.040000 0.098946                  True               False            True
2008-10-01  2008     10 -0.331921   -3.160512  0.041667 0.035066                  True               False            True
2020-11-01  2020     11  0.277183    3.117202  0.041667 0.034143                  True               False            True
2009-05-01  2009      5  0.277157    2.568156  0.041667 0.023428                 False               False            True
2020-04-01  2020      4 -0.220844   -2.323988  0.040000 0.018463                 False               False            True
2015-07-01  2015      7 -0.205932   -2.119466  0.041667 0.016075                 False               False            True
2008-12-01  2008     12 -0.198850   -2.061951  0.041667 0.015228                 False               False            True
2011-10-01  2011     10  0.183816    2.052147  0.041667 0.015085                 False               False            True
2014-12-01  2014     12 -0.195682   -2.029644  0.041667 0.014761                 False               False            True


**Univariate return outliers (MAD-z)**
      date  year  month    return     MAD_z  flag_|MADz|>3.5
2020-05-01  2020      5  0.610254  6.911991             True
2020-03-01  2020      3 -0.533804 -6.261550             True
2008-10-01  2008     10 -0.331921 -3.936922             True
2020-11-01  2020     11  0.277183  3.076762            False
2009-05-01  2009      5  0.277157  3.076463            False
2020-04-01  2020      4 -0.220844 -2.657892            False
2018-11-01  2018     11 -0.216823 -2.611595            False
2015-07-01  2015      7 -0.205932 -2.486187            False
2008-11-01  2008     11 -0.199559 -2.412802            False
2008-12-01  2008     12 -0.198850 -2.404645            False
