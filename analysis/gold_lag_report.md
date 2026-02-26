# Gold Market Lag Diagnostics

Overlap window: 2016-12-20 — 2026-02-24 UTC (2211 trading days).

## Stationarity
Series | ADF stat | ADF p | KPSS stat | KPSS p
---|---|---|---|---
COMEX close | 5.324 | 1.0000 | 5.321 | 0.0100
Shanghai close | 5.437 | 1.0000 | 5.526 | 0.0100
COMEX returns | -12.793 | 0.0000 | 0.584 | 0.0241
Shanghai returns | -10.264 | 0.0000 | 0.830 | 0.0100

## Cointegration
Engle–Granger stat -2.110, p-value 0.4714.
Johansen traces [36.69412777496878, 16.381945635534112] vs 95% critical [15.4943, 3.8415].

## Cross-Correlation (COMEX lead = positive lag)
Lag | Corr
---|---
-10 | -0.013
-9 | -0.015
-8 | 0.049
-7 | 0.008
-6 | -0.041
-5 | -0.052
-4 | -0.066
-3 | 0.051
-2 | -0.019
-1 | -0.060
0 | 0.337
1 | 0.627
2 | -0.034
3 | -0.030
4 | -0.001
5 | 0.009
6 | -0.027
7 | -0.051
8 | -0.042
9 | 0.010
10 | -0.007

## ARDL (Shanghai returns on COMEX returns)
AIC -16402.791, BIC -16362.889.
COMEX lag coefficients: {
  "comex_ret.L0": 0.2841111856103579,
  "comex_ret.L1": 0.6541491713012925,
  "comex_ret.L2": 0.22635144257946338
}

## VAR & Granger Causality
VAR lag 6, AIC -19.458, BIC -19.391.
COMEX→Shanghai F=333.040 (p=0.0000); Shanghai→COMEX F=4.608 (p=0.0001).
Lagwise Granger SSR F-tests: {
  "1": {
    "ssr_ftest_stat": NaN,
    "ssr_ftest_pvalue": NaN
  },
  "2": {
    "ssr_ftest_stat": NaN,
    "ssr_ftest_pvalue": NaN
  },
  "3": {
    "ssr_ftest_stat": NaN,
    "ssr_ftest_pvalue": NaN
  },
  "4": {
    "ssr_ftest_stat": NaN,
    "ssr_ftest_pvalue": NaN
  },
  "5": {
    "ssr_ftest_stat": NaN,
    "ssr_ftest_pvalue": NaN
  }
}