"""COMEX vs Shanghai gold price lead-lag diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "gold_lag_report_logdiff.md"


@dataclass
class StationarityResult:
    name: str
    adf_stat: float
    adf_pvalue: float
    kpss_stat: float
    kpss_pvalue: float


def load_logdiff_series(path: Path, tz: str, close_hour: int) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    if "log_return" not in df.columns:
        raise ValueError(f"Expected 'log_return' in {path}")
    df = df.dropna(subset=["log_return"]).copy()
    local_ts = pd.to_datetime(df["date"]) + pd.to_timedelta(close_hour, unit="h")
    local_ts = local_ts.dt.tz_localize(tz).dt.tz_convert("UTC").dt.normalize()
    series = pd.Series(df["log_return"].astype(float).values, index=local_ts)
    series = series.tz_localize(None)
    daily = series.groupby(level=0).sum().sort_index()
    return daily


def build_panel() -> pd.DataFrame:
    comex = load_logdiff_series(ROOT / "comex_gold_logdiff.csv", "America/New_York", 17)
    sh = load_logdiff_series(ROOT / "shanghai_gold_9999_logdiff.csv", "Asia/Shanghai", 15)
    panel = pd.concat({"comex_ret": comex, "shanghai_ret": sh}, axis=1, join="inner").dropna()
    panel = panel.sort_index()
    return panel


def stationarity_tests(series_map: Dict[str, pd.Series]) -> List[StationarityResult]:
    results: List[StationarityResult] = []
    for name, series in series_map.items():
        cleaned = series.dropna()
        if len(cleaned) < 20:
            continue
        adf_stat, adf_pvalue, *_ = adfuller(cleaned, autolag="AIC")
        try:
            kpss_stat, kpss_pvalue, *_ = kpss(cleaned, regression="c", nlags="auto")
        except ValueError:
            kpss_stat, kpss_pvalue = np.nan, np.nan
        results.append(
            StationarityResult(name, float(adf_stat), float(adf_pvalue), float(kpss_stat), float(kpss_pvalue))
        )
    return results


def cointegration_tests(_: pd.DataFrame) -> Dict[str, object]:
    raise RuntimeError("Cointegration tests are not applicable for log-differenced data")


def cross_correlation(returns: pd.DataFrame, max_lag: int = 10) -> Dict[int, float]:
    corrs: Dict[int, float] = {}
    c = returns["comex_ret"]
    s = returns["shanghai_ret"]
    for lag in range(-max_lag, max_lag + 1):
        aligned = pd.concat([c.shift(lag), s], axis=1).dropna()
        corrs[lag] = float(aligned.corr().iloc[0, 1]) if not aligned.empty else np.nan
    return corrs


def ardl_summary(returns: pd.DataFrame, lags: int = 2) -> Dict[str, float]:
    model = ARDL(returns["shanghai_ret"], lags, returns[["comex_ret"]], order=lags)
    res = model.fit()
    coeffs = {
        k: float(v)
        for k, v in res.params.items()
        if "comex_ret" in k
    }
    return {"aic": float(res.aic), "bic": float(res.bic), **coeffs}


def var_and_granger(returns: pd.DataFrame, max_lag: int = 10) -> Dict[str, object]:
    selector = VAR(returns).select_order(max_lag)
    lag = selector.selected_orders.get("aic") or selector.selected_orders.get("bic") or 2
    var_model = VAR(returns).fit(int(lag))
    gc_forward = var_model.test_causality("shanghai_ret", ["comex_ret"], kind="f")
    gc_reverse = var_model.test_causality("comex_ret", ["shanghai_ret"], kind="f")
    lagwise_stats: Dict[int, Dict[str, float]] = {}
    lagwise = grangercausalitytests(returns[["shanghai_ret", "comex_ret"]], maxlag=5, verbose=False)
    for lag_i, tests in lagwise.items():
        ssr_ftest = tests[0]
        stat = float(ssr_ftest[0]) if isinstance(ssr_ftest, tuple) else float("nan")
        pvalue = float(ssr_ftest[1]) if isinstance(ssr_ftest, tuple) else float("nan")
        lagwise_stats[int(lag_i)] = {
            "ssr_ftest_stat": stat,
            "ssr_ftest_pvalue": pvalue,
        }
    return {
        "var_lag": int(lag),
        "var_aic": float(var_model.aic),
        "var_bic": float(var_model.bic),
        "gc_comex_to_shanghai": {
            "f": float(gc_forward.test_statistic),
            "pvalue": float(gc_forward.pvalue),
        },
        "gc_shanghai_to_comex": {
            "f": float(gc_reverse.test_statistic),
            "pvalue": float(gc_reverse.pvalue),
        },
        "lagwise": lagwise_stats,
    }


def write_report(payload: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# Gold Market Lag Diagnostics")
    lines.append("")
    overlap = payload["overlap"]
    lines.append(
        f"Overlap window: {overlap['start']} — {overlap['end']} UTC ({overlap['n_obs']} trading days)."
    )
    lines.append("")

    lines.append("## Stationarity")
    lines.append("Series | ADF stat | ADF p | KPSS stat | KPSS p")
    lines.append("---|---|---|---|---")
    for res in payload["stationarity"]:
        lines.append(
            f"{res.name} | {res.adf_stat:.3f} | {res.adf_pvalue:.4f} | "
            f"{res.kpss_stat:.3f} | {res.kpss_pvalue:.4f}"
        )
    lines.append("")

    lines.append("## Cointegration")
    lines.append("Not evaluated because inputs are already log-differenced.")
    lines.append("")

    lines.append("## Cross-Correlation (COMEX lead = positive lag)")
    lines.append("Lag | Corr")
    lines.append("---|---")
    for lag, val in payload["cross_correlations"].items():
        lines.append(f"{lag} | {val:.3f}" if np.isfinite(val) else f"{lag} | NaN")
    lines.append("")

    lines.append("## ARDL (Shanghai returns on COMEX returns)")
    ardl = payload["ardl"]
    lines.append(f"AIC {ardl['aic']:.3f}, BIC {ardl['bic']:.3f}.")
    coeffs = {k: v for k, v in ardl.items() if k not in {"aic", "bic"}}
    lines.append("COMEX lag coefficients: " + json.dumps(coeffs, indent=2))
    lines.append("")

    lines.append("## VAR & Granger Causality")
    var = payload["var"]
    lines.append(
        f"VAR lag {var['var_lag']}, AIC {var['var_aic']:.3f}, BIC {var['var_bic']:.3f}."
    )
    lines.append(
        f"COMEX→Shanghai F={var['gc_comex_to_shanghai']['f']:.3f} (p={var['gc_comex_to_shanghai']['pvalue']:.4f}); "
        f"Shanghai→COMEX F={var['gc_shanghai_to_comex']['f']:.3f} (p={var['gc_shanghai_to_comex']['pvalue']:.4f})."
    )
    lines.append("Lagwise Granger SSR F-tests: " + json.dumps(var["lagwise"], indent=2))

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    panel = build_panel()
    if panel.empty:
        raise RuntimeError("No overlapping data between series")
    returns = panel[["comex_ret", "shanghai_ret"]]
    overlap = {
        "start": returns.index.min().strftime("%Y-%m-%d"),
        "end": returns.index.max().strftime("%Y-%m-%d"),
        "n_obs": int(len(returns)),
    }

    payload = {
        "overlap": overlap,
        "stationarity": stationarity_tests(
            {
                "COMEX log returns": returns["comex_ret"],
                "Shanghai log returns": returns["shanghai_ret"],
            }
        ),
        "cointegration": None,
        "cross_correlations": cross_correlation(returns),
        "ardl": ardl_summary(returns),
        "var": var_and_granger(returns),
    }

    write_report(payload)


if __name__ == "__main__":
    main()
