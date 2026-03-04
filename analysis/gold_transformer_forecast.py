"""Transformer forecaster for Shanghai gold log price with COMEX/Shanghai features."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import sys


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "analysis" / "gold_transformer_report.md"
PLOT_PATH = ROOT / "analysis" / "gold_transformer_loss.png"

# Ensure project root is on sys.path for script execution (prep_script import)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_price_series(path: Path, price_col: str, tz: str, close_hour: int) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=[price_col])
    local_ts = pd.to_datetime(df["date"]) + pd.to_timedelta(close_hour, unit="h")
    local_ts = local_ts.dt.tz_localize(tz).dt.tz_convert("UTC").dt.normalize()
    series = pd.Series(df[price_col].astype(float).values, index=local_ts)
    series = series.tz_localize(None)
    daily = series.groupby(level=0).mean().sort_index()
    daily.name = price_col
    return pd.Series(daily)


def build_panel_basic(include_indicators: bool = True) -> pd.DataFrame:
    """Build panel from raw CSVs with optional technical indicators."""
    comex_path = ROOT / "comex_gold.csv"
    sh_path = ROOT / "shanghai_gold_9999.csv"

    if include_indicators:
        # Use prep_script logic to align on UTC-normalized dates and add indicators
        from prep_script import run_prep

        df = run_prep(raw_dir=ROOT, out_path=None, include_indicators=True)
        return df

    comex = load_price_series(comex_path, "close", "America/New_York", 17)
    sh = load_price_series(sh_path, "close", "Asia/Shanghai", 15)
    panel = pd.concat({"comex": comex, "shanghai": sh}, axis=1, join="inner").dropna()
    panel = panel.sort_index()
    panel["comex_log"] = np.log(panel["comex"])
    panel["shanghai_log"] = np.log(panel["shanghai"])
    panel["comex_ret"] = panel["comex_log"].diff()
    panel["shanghai_ret"] = panel["shanghai_log"].diff()
    panel = panel.dropna()
    panel["label"] = panel["shanghai_log"].shift(-1)
    panel = panel.dropna()
    return panel


def build_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    label_col: str = "label",
) -> Tuple[np.ndarray, np.ndarray]:
    if len(df) < seq_len + 1:
        raise ValueError("Not enough rows to build windows for the requested seq_len")
    values = df[feature_cols].to_numpy(dtype=float)
    labels = df[label_col].to_numpy(dtype=float)
    Xs: List[np.ndarray] = []
    ys: List[float] = []
    # Inclusive window of length seq_len ending at position i
    for i in range(seq_len - 1, len(df)):
        Xs.append(values[i - seq_len + 1 : i + 1, :])
        ys.append(labels[i])
    return np.stack(Xs), np.array(ys)


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_dates: pd.Index
    val_dates: pd.Index
    test_dates: pd.Index


def split_time_series(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    label_col: str = "label",
    seq_len: int = 3,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> SplitData:
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != label_col]
    dates = df.index
    X, y = build_windows(df, feature_cols, seq_len=seq_len, label_col=label_col)
    y_dates = dates[seq_len:]
    n = len(X)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    if train_end == 0 or val_end == train_end or val_end >= n:
        raise ValueError("Invalid split ratios for available samples")
    return SplitData(
        X[:train_end],
        y[:train_end],
        X[train_end:val_end],
        y[train_end:val_end],
        X[val_end:],
        y[val_end:],
        y_dates[:train_end],
        y_dates[train_end:val_end],
        y_dates[val_end:],
    )


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.X)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return self.X[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len]
        return x + pe


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        seq_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        input_dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.input_dropout = nn.Dropout(input_dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.input_dropout(h)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        h_last = h[:, -1, :]
        out = self.head(h_last)
        return out.view(-1, 1)


def configure_model(
    feature_dim: int,
    target_type: str,
    seq_len: int,
    d_model: int = 64,
    nhead: int = 4,
    layers: int = 2,
    dim_ff: int = 256,
    dropout: float = 0.1,
    input_dropout: float = 0.1,
) -> nn.Module:
    del target_type  # unused; kept for signature compatibility
    return TransformerRegressor(
        feature_dim=feature_dim,
        seq_len=seq_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=layers,
        dim_feedforward=dim_ff,
        dropout=dropout,
        input_dropout=input_dropout,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 300,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}
    no_improve = 0

    for n in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)
        if n % 5 == 0 or n == 1:
            print(f"Epoch {n}/{epochs} - Train Loss: {train_loss:.6f}", end="")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * len(xb)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"model": model, "history": history, "best_val": best_val}


def evaluate_metrics(
    preds_scaled: np.ndarray,
    trues_scaled: np.ndarray,
    label_scaler: Optional[StandardScaler] = None,
) -> Dict[str, float]:
    preds = np.asarray(preds_scaled).reshape(-1, 1)
    trues = np.asarray(trues_scaled).reshape(-1, 1)
    if label_scaler is not None:
        preds = label_scaler.inverse_transform(preds)
        trues = label_scaler.inverse_transform(trues)
    preds_flat = preds.ravel()
    trues_flat = trues.ravel()

    mse = mean_squared_error(trues_flat, preds_flat)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(trues_flat, preds_flat)
    try:
        r2 = r2_score(trues_flat, preds_flat)
    except ValueError:
        r2 = float("nan")
    direction_acc = float(
        np.mean(
            np.sign(np.diff(preds_flat, prepend=preds_flat[0]))
            == np.sign(np.diff(trues_flat, prepend=trues_flat[0]))
        )
    )
    return {
        "rmse_price": float(rmse),
        "mae_price": float(mae),
        "mse_price": float(mse),
        "r2": float(r2),
        "direction_accuracy": direction_acc,
    }


def evaluate_baseline(data: Any) -> Dict[str, float]:
    targets = data["targets"] if isinstance(data, dict) else np.asarray(data)
    preds = np.zeros_like(targets)
    mse = mean_squared_error(targets, preds)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(targets, preds)
    return {"rmse_price": float(rmse), "mae_price": float(mae), "mse_price": float(mse)}


def train_quick(data: Dict[str, np.ndarray], epochs: int = 5) -> Dict[str, object]:
    X = data["windows"]
    y = data["targets"]

    # Standardize targets to ease optimization
    from sklearn.preprocessing import StandardScaler

    y_scaler = StandardScaler().fit(y.reshape(-1, 1))
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).ravel()

    model = TransformerRegressor(feature_dim=X.shape[2], seq_len=X.shape[1])
    ds = SequenceDataset(X, y_scaled)
    # Simple split to avoid overfitting baseline; ensure at least 1 sample in val
    split = max(1, int(0.8 * len(ds))) if len(ds) > 1 else 1
    train_indices = list(range(split))
    val_indices = list(range(split, len(ds))) if split < len(ds) else list(range(len(ds)))
    train_loader = DataLoader(torch.utils.data.Subset(ds, train_indices), batch_size=min(16, len(train_indices)), shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(ds, val_indices), batch_size=min(16, len(val_indices)), shuffle=False)
    out = train_model(model, train_loader, val_loader, epochs=max(epochs, 8), patience=max(epochs, 8))

    preds_list: List[np.ndarray] = []
    trues_list: List[np.ndarray] = []
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        for xb, yb in DataLoader(ds, batch_size=min(16, len(ds)), shuffle=False):
            xb = xb.to(device)
            preds_list.append(model(xb).cpu().numpy())
            trues_list.append(yb.numpy())
    preds_scaled = np.concatenate(preds_list)
    trues_scaled = np.concatenate(trues_list)
    metrics = evaluate_metrics(preds_scaled, trues_scaled, label_scaler=y_scaler)
    return {"history": out["history"], "metrics": metrics}


def plot_loss(history: Dict[str, list]) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(history.get("train_loss", []), label="Train loss")
    plt.plot(history.get("val_loss", []), label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Transformer Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


def write_report(payload: Dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def run_cli(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if args.data_path:
        df = pd.read_parquet(args.data_path) if args.data_path.suffix == ".parquet" else pd.read_csv(args.data_path)
        if "date" in df.columns:
            df.index = pd.to_datetime(df["date"])
        else:
            df.index = pd.to_datetime(df.index)
    else:
        df = build_panel_basic(include_indicators=args.include_indicators)

    if "label" not in df.columns:
        raise ValueError("Prepared data must contain 'label' column for next-day log price")

    feature_cols = [c for c in df.columns if c not in {"label"}]
    split = split_time_series(
        df,
        feature_cols=feature_cols,
        label_col="label",
        seq_len=args.seq_len,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    if len(split.X_test) == 0:
        raise RuntimeError("Not enough samples for the requested split ratios")

    scaler_x = StandardScaler()
    X_train_np = np.asarray(split.X_train, dtype=float)
    X_val_np = np.asarray(split.X_val, dtype=float)
    X_test_np = np.asarray(split.X_test, dtype=float)
    X_train_scaled = scaler_x.fit_transform(X_train_np.reshape(-1, X_train_np.shape[2])).reshape(X_train_np.shape)
    X_val_scaled = scaler_x.transform(X_val_np.reshape(-1, X_val_np.shape[2])).reshape(X_val_np.shape)
    X_test_scaled = scaler_x.transform(X_test_np.reshape(-1, X_test_np.shape[2])).reshape(X_test_np.shape)

    scaler_y = StandardScaler()
    y_train_np = np.asarray(split.y_train, dtype=float)
    y_val_np = np.asarray(split.y_val, dtype=float)
    y_test_np = np.asarray(split.y_test, dtype=float)
    y_train_scaled = scaler_y.fit_transform(y_train_np.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val_np.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test_np.reshape(-1, 1)).ravel()

    train_loader = DataLoader(SequenceDataset(X_train_scaled, y_train_scaled), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val_scaled, y_val_scaled), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SequenceDataset(X_test_scaled, y_test_scaled), batch_size=args.batch_size, shuffle=False)

    model = configure_model(
        feature_dim=len(feature_cols),
        target_type=args.target,
        seq_len=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        input_dropout=args.input_dropout,
    )
    training = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    model = training["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds_scaled, trues_scaled = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds_scaled.append(pred)
            trues_scaled.append(yb.numpy())
    preds_scaled = np.concatenate(preds_scaled)
    trues_scaled = np.concatenate(trues_scaled)

    test_metrics = evaluate_metrics(preds_scaled, trues_scaled, label_scaler=scaler_y)
    baseline = evaluate_baseline({"targets": split.y_test})

    setup = {
        "start_date": str(df.index.min().date()) if len(df) else "",
        "end_date": str(df.index.max().date()) if len(df) else "",
        "n_samples": len(split.X_train) + len(split.X_val) + len(split.X_test),
        "train_size": len(split.X_train),
        "val_size": len(split.X_val),
        "test_size": len(split.X_test),
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "layers": args.layers,
        "dim_ff": args.dim_ff,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    write_report({"setup": setup, "test_metrics": test_metrics, "baseline_metrics": baseline})
    plot_loss(training.get("history", {}))
    print(json.dumps({"test_metrics": test_metrics, "baseline_metrics": baseline}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transformer-based COMEX→Shanghai forecaster (log price)")
    parser.add_argument("--data-path", type=Path, default=None, help="Path to prepared feature table (parquet/csv)")
    parser.add_argument("--target", type=str, choices=["logprice", "price", "pricediff"], default="logprice")
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dim-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--input-dropout", type=float, default=0.1, help="Dropout applied to input projection")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-indicators", dest="include_indicators", action="store_true", default=False, help="Include technical indicators")
    parser.add_argument("--no-include-indicators", dest="include_indicators", action="store_false", help="Exclude technical indicators")
    return parser.parse_args()


if __name__ == "__main__":
    run_cli(parse_args())
