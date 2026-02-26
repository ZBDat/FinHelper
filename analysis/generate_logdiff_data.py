"""Create log-differenced CSVs for COMEX and Shanghai gold series."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


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
