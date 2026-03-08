from pathlib import Path

import pandas as pd
import pytest

from analysis.ui_mlp_service import predict_with_bundle, train_from_uploaded_data


def _make_rows(scale: float, n: int = 32):
    """Build synthetic OHLC rows; `scale` separates COMEX/Shanghai levels for testing."""
    rows = []
    for i in range(n):
        base = (100 + i) * scale
        rows.append(
            {
                "date": f"2024-01-{(i % 28) + 1:02d}",
                "open": base + 0.1,
                "high": base + 0.6,
                "low": base - 0.4,
                "close": base,
            }
        )
    return rows


def test_train_and_predict_from_uploaded_data(tmp_path: Path) -> None:
    datasets = [
        {"name": "shanghai.csv", "data": _make_rows(1.2)},
        {"name": "comex.csv", "data": _make_rows(1.0)},
    ]
    params = {
        "seq_len": 2,
        "train_frac": 0.7,
        "val_frac": 0.1,
        "epochs": 4,
        "patience": 2,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "hidden": 32,
        "seed": 42,
        "include_indicators": False,
    }
    save_path = tmp_path / "model.pth"

    bundle = train_from_uploaded_data(datasets=datasets, params=params, save_path=save_path)

    assert save_path.exists()
    assert "test_metrics" in bundle.report
    assert "setup" in bundle.report
    assert bundle.seq_len == 2

    pred = predict_with_bundle(bundle, datasets)
    assert "predicted_close" in pred
    assert pred["predicted_close"] > 0


def test_train_fails_on_missing_required_columns(tmp_path: Path) -> None:
    datasets = [{"name": "broken.csv", "data": [{"date": "2024-01-01", "open": 1.0, "close": 1.0}]}]
    with pytest.raises(ValueError, match="missing required columns"):
        train_from_uploaded_data(datasets=datasets, params={"epochs": 1}, save_path=tmp_path / "m.pth")


def test_predict_uses_latest_available_date(tmp_path: Path) -> None:
    rows = _make_rows(1.1, n=24)
    rows[-1]["date"] = "2024-02-24"
    datasets = [{"name": "shanghai.csv", "data": rows}]
    params = {
        "seq_len": 2,
        "train_frac": 0.7,
        "val_frac": 0.1,
        "epochs": 3,
        "patience": 2,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "hidden": 32,
        "seed": 42,
        "include_indicators": False,
    }

    bundle = train_from_uploaded_data(datasets=datasets, params=params, save_path=tmp_path / "model.pth")
    pred = predict_with_bundle(bundle, datasets)

    assert pred["based_on_date"] == "2024-02-24"
    expected_next_business_day = (pd.Timestamp("2024-02-24") + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")
    assert pred["prediction_for_date"] == expected_next_business_day
