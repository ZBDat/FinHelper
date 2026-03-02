"""Create log-differenced CSVs for COMEX and Shanghai gold series."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def compute_technical_indicators(
    input_csv: Path,
    kdj_period: int = 9,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    rsi_period: int = 14,
    wr_period: int = 14,
    dmi_period: int = 14,
    cci_period: int = 14,
    bias_period: int = 6,
) -> pd.DataFrame:
    """Compute technical indicators from non-differenced OHLC trade data CSV."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required = {"date", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {sorted(required)} in {input_csv}")
    df["date"] = pd.to_datetime(df["date"])

    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)

    rolling_low = low.rolling(window=kdj_period, min_periods=kdj_period).min()
    rolling_high = high.rolling(window=kdj_period, min_periods=kdj_period).max()
    raw_stochastic_value = (close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan) * 100
    out["kdj_k"] = raw_stochastic_value.ewm(alpha=1 / 3, adjust=False).mean()
    out["kdj_d"] = out["kdj_k"].ewm(alpha=1 / 3, adjust=False).mean()
    out["kdj_j"] = 3 * out["kdj_k"] - 2 * out["kdj_d"]

    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    out["macd_dif"] = ema_fast - ema_slow
    out["macd_dea"] = out["macd_dif"].ewm(span=macd_signal, adjust=False).mean()
    out["macd_hist"] = 2 * (out["macd_dif"] - out["macd_dea"])

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    hh = high.rolling(window=wr_period, min_periods=wr_period).max()
    ll = low.rolling(window=wr_period, min_periods=wr_period).min()
    out["wr"] = (hh - close) / (hh - ll).replace(0, np.nan) * 100

    prev_close = close.shift(1)
    true_range = np.maximum(np.maximum(high - low, (high - prev_close).abs()), (low - prev_close).abs())
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = true_range.rolling(window=dmi_period, min_periods=dmi_period).mean()
    plus_di = 100 * plus_dm.rolling(window=dmi_period, min_periods=dmi_period).mean() / atr
    minus_di = 100 * minus_dm.rolling(window=dmi_period, min_periods=dmi_period).mean() / atr
    out["dmi_pdi"] = plus_di
    out["dmi_mdi"] = minus_di
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    out["dmi_adx"] = dx.rolling(window=dmi_period, min_periods=dmi_period).mean()

    typical_price = (high + low + close) / 3
    ma_tp = typical_price.rolling(window=cci_period, min_periods=cci_period).mean()
    mean_dev = (typical_price - ma_tp).abs().rolling(window=cci_period, min_periods=cci_period).mean()
    out["cci"] = (typical_price - ma_tp) / (0.015 * mean_dev.replace(0, np.nan))

    ma_close = close.rolling(window=bias_period, min_periods=bias_period).mean()
    out["bias"] = (close - ma_close) / ma_close.replace(0, np.nan) * 100
    return out


def compute_logdiff(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv, parse_dates=["date"])
    if "close" not in df.columns:
        raise ValueError(f"Expected 'close' column in {input_csv}")

    df = df.dropna(subset=["close"]).copy()
    df["close"] = df["close"].astype(float)
    df["log_price"] = np.log(df["close"])
    df["log_return"] = df["log_price"].diff()
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote log-differenced data to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate log-diff CSVs for gold datasets")
    parser.add_argument(
        "--comex",
        type=Path,
        default=ROOT / "comex_gold.csv",
        help="Path to original COMEX CSV",
    )
    parser.add_argument(
        "--shanghai",
        type=Path,
        default=ROOT / "shanghai_gold_9999.csv",
        help="Path to original Shanghai CSV",
    )
    parser.add_argument(
        "--comex-out",
        type=Path,
        default=ROOT / "comex_gold_logdiff.csv",
        help="Output path for COMEX log-diff CSV",
    )
    parser.add_argument(
        "--shanghai-out",
        type=Path,
        default=ROOT / "shanghai_gold_9999_logdiff.csv",
        help="Output path for Shanghai log-diff CSV",
    )
    args = parser.parse_args()

    compute_logdiff(args.comex, args.comex_out)
    compute_logdiff(args.shanghai, args.shanghai_out)


if __name__ == "__main__":
    main()
