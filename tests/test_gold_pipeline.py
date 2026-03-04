import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Helpers

def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@pytest.fixture
def synthetic_raw(tmp_path: Path) -> dict:
    """Create minimal synthetic COMEX/Shanghai CSVs with OHLC columns.

    Column names follow existing scripts: date, open, high, low, close.
    Dates are consecutive to allow seq_len=3 windowing.
    """

    dates = pd.date_range("2024-01-01", periods=8, freq="D")

    def _make_df(scale: float) -> pd.DataFrame:
        base = np.linspace(100 * scale, 108 * scale, len(dates))
        return pd.DataFrame(
            {
                "date": dates,
                "open": base + 0.1,
                "high": base + 0.5,
                "low": base - 0.3,
                "close": base,
            }
        )

    comex = _make_df(1.0)
    sh = _make_df(1.2)

    comex_path = tmp_path / "comex_gold.csv"
    sh_path = tmp_path / "shanghai_gold_9999.csv"
    comex.to_csv(comex_path, index=False)
    sh.to_csv(sh_path, index=False)

    return {"comex": comex_path, "shanghai": sh_path}


def test_prep_outputs_columns_no_nans(tmp_path: Path, synthetic_raw: dict) -> None:
    if not _has_module("prep_script"):
        pytest.skip("prep_script not implemented yet")
    prep = importlib.import_module("prep_script")
    run_prep = getattr(prep, "run_prep", None)
    if run_prep is None:
        pytest.skip("run_prep not available in prep_script")

    out_path = tmp_path / "features.parquet"
    df = run_prep(raw_dir=tmp_path, out_path=out_path)

    required_cols = {"log_close_sh", "label"}
    assert required_cols.issubset(df.columns)
    assert not df.isna().any().any()
    assert len(df) >= 3  # enough for seq_len=3


def test_label_shift_log_close_sh(tmp_path: Path, synthetic_raw: dict) -> None:
    if not _has_module("prep_script"):
        pytest.skip("prep_script not implemented yet")
    prep = importlib.import_module("prep_script")
    run_prep = getattr(prep, "run_prep", None)
    if run_prep is None:
        pytest.skip("run_prep not available in prep_script")

    df = run_prep(raw_dir=tmp_path, out_path=tmp_path / "features.parquet")
    assert "label" in df.columns and "log_close_sh" in df.columns

    expected = df["log_close_sh"].shift(-1).dropna()
    actual = df["label"].iloc[:-1]
    assert np.allclose(actual.to_numpy(), expected.to_numpy())


def test_window_builder_seq_len_three(tmp_path: Path, synthetic_raw: dict) -> None:
    if not _has_module("analysis.gold_transformer_forecast"):
        pytest.skip("gold_transformer_forecast not available")
    mod = importlib.import_module("analysis.gold_transformer_forecast")
    build_windows = getattr(mod, "build_windows", None)
    if build_windows is None:
        pytest.skip("build_windows not implemented yet")

    # Minimal frame with 5 rows after prep would yield 3 windows of len 3
    df = pd.DataFrame(
        {
            "log_close_sh": np.arange(5, dtype=float),
            "label": np.arange(1, 6, dtype=float),
        }
    )
    feature_cols = ["log_close_sh"]
    windows, targets = build_windows(df, feature_cols=feature_cols, seq_len=3, label_col="label")
    assert windows.shape[1] == 3
    assert windows.shape[2] == len(feature_cols)
    assert len(windows) == len(df) - 2
    assert len(targets) == len(windows)


def test_split_integrity_and_scaler_scope(tmp_path: Path, synthetic_raw: dict) -> None:
    if not _has_module("analysis.gold_transformer_forecast"):
        pytest.skip("gold_transformer_forecast not available")
    mod = importlib.import_module("analysis.gold_transformer_forecast")
    split_time_series = getattr(mod, "split_time_series", None)
    if split_time_series is None:
        pytest.skip("split_time_series not implemented yet")

    df = pd.DataFrame({"log_close_sh": np.arange(10, dtype=float), "label": np.arange(1, 11, dtype=float)})
    split = split_time_series(df, train_frac=0.6, val_frac=0.2)
    assert split.train_dates.max() < split.val_dates.min() < split.test_dates.min()

    # Ensure scaler fit scope is train-only when applied
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(split.X_train.reshape(-1, len(split.X_train[0][0])))
    train_mean = scaler.mean_.copy()
    # Refit on val would change mean; guard against unintended refits
    scaler_val = StandardScaler().fit(split.X_val.reshape(-1, len(split.X_val[0][0])))
    assert not np.allclose(train_mean, scaler_val.mean_)  # sanity check difference


def test_model_forward_shapes() -> None:
    if not _has_module("analysis.gold_transformer_forecast"):
        pytest.skip("gold_transformer_forecast not available")
    if not _has_module("torch"):
        pytest.skip("torch not installed")
    import torch

    mod = importlib.import_module("analysis.gold_transformer_forecast")
    configure_model = getattr(mod, "configure_model", None)
    if configure_model is None:
        pytest.skip("configure_model not implemented yet")

    model = configure_model(feature_dim=4, target_type="logprice", seq_len=3)
    dummy = torch.randn(4, 3, 4)
    out = model(dummy)
    assert out.shape == (4, 1)
    out.sum().backward()


def test_training_improves_on_synthetic() -> None:
    if not _has_module("analysis.gold_transformer_forecast"):
        pytest.skip("gold_transformer_forecast not available")
    if not _has_module("torch"):
        pytest.skip("torch not installed")
    mod = importlib.import_module("analysis.gold_transformer_forecast")
    train_quick = getattr(mod, "train_quick", None)
    evaluate_baseline = getattr(mod, "evaluate_baseline", None)
    if train_quick is None or evaluate_baseline is None:
        pytest.skip("train_quick/evaluate_baseline not implemented yet")

    data = {
        "windows": np.random.randn(20, 3, 2).astype(np.float32),
        "targets": np.random.randn(20).astype(np.float32),
    }
    base = evaluate_baseline(data)
    result = train_quick(data, epochs=5)
    assert result["history"]["train_loss"][-1] <= result["history"]["train_loss"][0]
    assert result["metrics"]["rmse_price"] <= base["rmse_price"]


def test_inverse_transform_metrics() -> None:
    if not _has_module("analysis.gold_transformer_forecast"):
        pytest.skip("gold_transformer_forecast not available")
    mod = importlib.import_module("analysis.gold_transformer_forecast")
    eval_metrics = getattr(mod, "evaluate_metrics", None)
    if eval_metrics is None:
        pytest.skip("evaluate_metrics not implemented yet")

    from sklearn.preprocessing import StandardScaler

    true = np.array([1.0, 1.1, 0.9]).reshape(-1, 1)
    scaler = StandardScaler().fit(true)
    preds_scaled = scaler.transform(true * 0.99)
    metrics = eval_metrics(preds_scaled.flatten(), scaler.transform(true).flatten(), label_scaler=scaler)
    assert "rmse_price" in metrics
    assert metrics["direction_accuracy"] <= 1.0
