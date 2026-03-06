from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from prep_script import run_prep
from analysis.gold_mlp_forecast import FlatDataset, FlattenMLP, split_flat, train_mlp
from analysis.gold_transformer_forecast import build_windows, evaluate_baseline, evaluate_metrics, set_seed


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close"]
# Keep the compressed second layer expressive enough when hidden is configured small.
MIN_SECOND_HIDDEN = 8


@dataclass
class TrainedBundle:
    model: torch.nn.Module
    scaler_x: StandardScaler
    scaler_y: StandardScaler
    feature_cols: List[str]
    seq_len: int
    include_indicators: bool
    report: Dict[str, Any]
    model_path: str


def _to_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        raise ValueError("Imported data is empty and cannot be used for training")

    frame = pd.DataFrame(rows)
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Imported data is missing required columns: {', '.join(missing)}")

    out = frame[REQUIRED_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna().sort_values("date")
    # Need enough rows to pass windowing plus non-empty train/val/test splits.
    if len(out) < 6:
        raise ValueError("Too few valid data points; at least 6 rows are required")
    return out


def _prepare_raw_files(datasets: List[Dict[str, Any]], raw_dir: Path) -> None:
    if not datasets:
        raise ValueError("Please import trade data before training")

    shanghai_rows = datasets[0].get("data", [])
    comex_rows = datasets[1].get("data", []) if len(datasets) > 1 else shanghai_rows

    shanghai_df = _to_frame(shanghai_rows)
    comex_df = _to_frame(comex_rows)

    comex_df.to_csv(raw_dir / "comex_gold.csv", index=False)
    shanghai_df.to_csv(raw_dir / "shanghai_gold_9999.csv", index=False)


def train_from_uploaded_data(
    datasets: List[Dict[str, Any]],
    params: Dict[str, Any],
    save_path: Path,
) -> TrainedBundle:
    seq_len = int(params.get("seq_len", 2))
    hidden = int(params.get("hidden", 128))
    dropout = float(params.get("dropout", 0.1))
    train_frac = float(params.get("train_frac", 0.7))
    val_frac = float(params.get("val_frac", 0.1))
    batch_size = int(params.get("batch_size", 64))
    epochs = int(params.get("epochs", 100))
    patience = int(params.get("patience", 20))
    lr = float(params.get("lr", 1e-3))
    weight_decay = float(params.get("weight_decay", 1e-4))
    include_indicators = bool(params.get("include_indicators", False))
    seed = int(params.get("seed", 42))

    set_seed(seed)

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir = Path(tmp_dir)
        _prepare_raw_files(datasets, raw_dir)
        panel = run_prep(raw_dir=raw_dir, out_path=None, include_indicators=include_indicators)

    if "label" not in panel.columns:
        raise ValueError("Prepared dataset is missing required 'label' column")

    feature_cols = [col for col in panel.columns if col != "label"]
    windows, targets = build_windows(panel, feature_cols=feature_cols, seq_len=seq_len, label_col="label")
    flat = windows.reshape(len(windows), -1)

    X_train, y_train, X_val, y_val, X_test, y_test = split_flat(flat, targets, train_frac=train_frac, val_frac=val_frac)

    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_val_scaled = scaler_x.transform(X_val)
    X_test_scaled = scaler_x.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    train_loader = DataLoader(FlatDataset(X_train_scaled, y_train_scaled), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FlatDataset(X_val_scaled, y_val_scaled), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(FlatDataset(X_test_scaled, y_test_scaled), batch_size=batch_size, shuffle=False)

    model = FlattenMLP(
        input_dim=seq_len * len(feature_cols),
        hidden=[hidden, max(hidden // 2, MIN_SECOND_HIDDEN)],
        dropout=dropout,
    )
    training = train_mlp(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
    )
    model = training["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    preds_scaled: List[np.ndarray] = []
    trues_scaled: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds_scaled.append(pred)
            trues_scaled.append(yb.numpy())

    preds_scaled_np = np.concatenate(preds_scaled)
    trues_scaled_np = np.concatenate(trues_scaled)
    test_metrics = evaluate_metrics(preds_scaled_np, trues_scaled_np, label_scaler=scaler_y)
    baseline = evaluate_baseline({"targets": y_test})

    report: Dict[str, Any] = {
        "setup": {
            "seq_len": seq_len,
            "hidden": hidden,
            "dropout": dropout,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "lr": lr,
            "weight_decay": weight_decay,
            "include_indicators": include_indicators,
            "seed": seed,
        },
        "test_metrics": test_metrics,
        "baseline_metrics": baseline,
        "history": training.get("history", {}),
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler_x": scaler_x,
            "scaler_y": scaler_y,
            "feature_cols": feature_cols,
            "seq_len": seq_len,
            "include_indicators": include_indicators,
            "hidden": hidden,
            "dropout": dropout,
        },
        save_path,
    )

    return TrainedBundle(
        model=model,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        feature_cols=feature_cols,
        seq_len=seq_len,
        include_indicators=include_indicators,
        report=report,
        model_path=str(save_path),
    )


def predict_with_bundle(bundle: TrainedBundle, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir = Path(tmp_dir)
        _prepare_raw_files(datasets, raw_dir)
        panel = run_prep(raw_dir=raw_dir, out_path=None, include_indicators=bundle.include_indicators)

    feature_cols = bundle.feature_cols
    if "label" not in panel.columns:
        raise ValueError("Prediction failed: prepared dataset is missing required 'label' column")

    missing = [col for col in feature_cols if col not in panel.columns]
    if missing:
        raise ValueError(f"Prediction failed: missing feature columns from training: {', '.join(missing)}")

    windows, _ = build_windows(panel, feature_cols=feature_cols, seq_len=bundle.seq_len, label_col="label")
    last_window = windows[-1].reshape(1, -1)
    last_scaled = bundle.scaler_x.transform(last_window)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle.model = bundle.model.to(device)
    bundle.model.eval()
    with torch.no_grad():
        pred_scaled = bundle.model(torch.tensor(last_scaled, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1, 1)

    pred_log_close = float(bundle.scaler_y.inverse_transform(pred_scaled).ravel()[0])
    pred_close = float(np.exp(pred_log_close))
    latest_date = pd.to_datetime(panel.index[-1])
    next_date = latest_date + pd.tseries.offsets.BDay(1)

    return {
        "predicted_log_close": pred_log_close,
        "predicted_close": pred_close,
        "prediction_for_date": next_date.strftime("%Y-%m-%d"),
        "based_on_date": latest_date.strftime("%Y-%m-%d"),
    }
