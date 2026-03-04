"""Simple preparation script to build aligned features and label for transformer.

This is a lightweight helper to satisfy the current test expectations:
- Reads COMEX and Shanghai CSVs from a given raw directory.
- Aligns by date (UTC-normalized to midnight) using provided close columns.
- Produces log_close_sh and a label column = next-day log_close_sh.
- Drops rows with missing data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from analysis.generate_logdiff_data import compute_technical_indicators
except Exception:
    compute_technical_indicators = None  # type: ignore


def _load_price(path: Path, tz: str, close_hour: int) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["close"])
    ts = pd.to_datetime(df["date"]) + pd.to_timedelta(close_hour, unit="h")
    ts = ts.dt.tz_localize(tz).dt.tz_convert("UTC").dt.normalize()
    values = df["close"].astype(float).to_numpy()
    series = pd.Series(values, index=ts)
    series = series.tz_localize(None)
    grouped = series.groupby(level=0).mean().sort_index()
    return pd.Series(grouped.values, index=grouped.index, name="close")


def _build_with_indicators(comex_path: Path, sh_path: Path) -> pd.DataFrame:
    if compute_technical_indicators is None:
        raise RuntimeError("compute_technical_indicators unavailable; set include_indicators=False")

    comex_ind = compute_technical_indicators(comex_path)
    sh_ind = compute_technical_indicators(sh_path)

    comex_series = _load_price(comex_path, tz="America/New_York", close_hour=17)
    sh_series = _load_price(sh_path, tz="Asia/Shanghai", close_hour=15)

    comex_log = pd.Series(np.log(comex_series.to_numpy(dtype=float)), index=comex_series.index, name="log_close_comex")
    sh_log = pd.Series(np.log(sh_series.to_numpy(dtype=float)), index=sh_series.index, name="log_close_sh")

    base = pd.concat([comex_log, sh_log], axis=1, join="inner").sort_index()

    comex_ind = comex_ind.set_index(pd.to_datetime(comex_ind["date"])).sort_index().drop(columns=["date"], errors="ignore")
    sh_ind = sh_ind.set_index(pd.to_datetime(sh_ind["date"])).sort_index().drop(columns=["date"], errors="ignore")
    comex_ind = comex_ind.add_prefix("comex_")
    sh_ind = sh_ind.add_prefix("sh_")

    df = base.join(comex_ind, how="inner").join(sh_ind, how="inner")
    df = df.dropna().sort_index()
    df["label"] = df["log_close_sh"].shift(-1)
    df = df.dropna()
    return df


def _build_minimal(comex_path: Path, sh_path: Path) -> pd.DataFrame:
    comex = _load_price(comex_path, tz="America/New_York", close_hour=17)
    sh = _load_price(sh_path, tz="Asia/Shanghai", close_hour=15)

    comex_log = pd.Series(np.log(comex.to_numpy(dtype=float)), index=comex.index, name="log_close_comex")
    sh_log = pd.Series(np.log(sh.to_numpy(dtype=float)), index=sh.index, name="log_close_sh")
    df = pd.concat([comex_log, sh_log], axis=1, join="inner")
    df = df.dropna().sort_index()
    df["label"] = df["log_close_sh"].shift(-1)
    df = df.dropna()
    return df


def run_prep(raw_dir: Path, out_path: Optional[Path] = None, include_indicators: bool = False) -> pd.DataFrame:
    """Prepare feature table with optional technical indicators and label = next-day log_close_sh.

    Default keeps indicators off to remain robust for very short synthetic datasets (tests).
    """

    raw_dir = Path(raw_dir)
    comex_path = raw_dir / "comex_gold.csv"
    sh_path = raw_dir / "shanghai_gold_9999.csv"

    if include_indicators and compute_technical_indicators is not None:
        df = _build_with_indicators(comex_path, sh_path)
        if len(df) < 3:
            # Fallback when rolling-window indicators wipe out short samples
            df = _build_minimal(comex_path, sh_path)
    else:
        df = _build_minimal(comex_path, sh_path)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".parquet":
            try:
                df.to_parquet(out_path, index=True)
            except Exception:
                # Fallback to csv if parquet engine is unavailable
                out_path = out_path.with_suffix(".csv")
                df.to_csv(out_path, index=True)
        else:
            df.to_csv(out_path, index=True)

    return df


if __name__ == "__main__":
    run_prep(Path("."), out_path=Path("data/gold_features.csv"))
