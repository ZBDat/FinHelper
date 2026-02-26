# Gold Market Lag Diagnostics

Overlap window: 2016-12-20 — 2026-02-24 UTC (2211 trading days).

## Stationarity
Series | ADF stat | ADF p | KPSS stat | KPSS p
---|---|---|---|---
COMEX log returns | -12.853 | 0.0000 | 0.417 | 0.0697
Shanghai log returns | -10.252 | 0.0000 | 0.843 | 0.0100

## Cointegration
Not evaluated because inputs are already log-differenced.

## Cross-Correlation (COMEX lead = positive lag)
Lag | Corr
---|---
-10 | 0.009
-9 | -0.033
-8 | 0.034
-7 | 0.009
-6 | -0.044
-5 | -0.059
-4 | -0.068
-3 | 0.054
-2 | -0.022
-1 | -0.061
0 | 0.288
1 | 0.644
2 | -0.040
3 | -0.030
4 | -0.005
5 | 0.012
6 | -0.039
7 | -0.043
8 | -0.053
9 | 0.022
10 | -0.007

## ARDL (Shanghai returns on COMEX returns)
AIC -16262.000, BIC -16222.098.
COMEX lag coefficients: {
  "comex_ret.L0": 0.24494204107667406,
  "comex_ret.L1": 0.6493982966163545,
  "comex_ret.L2": 0.1727647979281782
}

## VAR & Granger Causality
VAR lag 6, AIC -19.426, BIC -19.359.
COMEX→Shanghai F=324.053 (p=0.0000); Shanghai→COMEX F=3.842 (p=0.0008).
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