"""Train a small Transformer to predict Shanghai gold returns from COMEX history."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "analysis" / "gold_transformer_report.md"
PLOT_PATH = ROOT / "analysis" / "gold_transformer_loss.png"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_price_series(path: Path, price_col: str, tz: str, close_hour: int) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=[price_col])
    local_ts = pd.to_datetime(df["date"]) + pd.to_timedelta(close_hour, unit="h")
    local_ts = local_ts.dt.tz_localize(tz).dt.tz_convert("UTC").dt.normalize()
    series = pd.Series(df[price_col].astype(float).values, index=local_ts)
    series = series.tz_localize(None)
    daily = series.groupby(level=0).mean().sort_index()
    return daily


def build_panel() -> pd.DataFrame:
    comex = load_price_series(ROOT / "comex_gold.csv", "close", "America/New_York", 17)
    sh = load_price_series(ROOT / "shanghai_gold_9999.csv", "close", "Asia/Shanghai", 15)
    panel = pd.concat({"comex": comex, "shanghai": sh}, axis=1, join="inner").dropna()
    panel = panel.sort_index()
    panel["comex_log"] = np.log(panel["comex"])
    panel["shanghai_log"] = np.log(panel["shanghai"])
    panel["comex_ret"] = panel["comex_log"].diff()
    panel["shanghai_ret"] = panel["shanghai_log"].diff()
    panel = panel.dropna()
    return panel


def create_sequences(
    returns: pd.DataFrame,
    seq_len: int,
    feature_col: str = "comex_ret",
    target_col: str = "shanghai_ret",
) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    feats = returns[feature_col].values
    targets = returns[target_col].values
    dates = returns.index
    Xs, ys, y_dates = [], [], []
    for i in range(seq_len, len(returns)):
        Xs.append(feats[i - seq_len : i].reshape(seq_len, 1))
        ys.append(targets[i])
        y_dates.append(dates[i])
    return np.stack(Xs), np.array(ys), pd.Index(y_dates)


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
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.Index,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> SplitData:
    n = len(X)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return SplitData(
        X[:train_end],
        y[:train_end],
        X[train_end:val_end],
        y[train_end:val_end],
        X[val_end:],
        y[val_end:],
        dates[:train_end],
        dates[train_end:val_end],
        dates[val_end:],
    )


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class ComexTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
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
        h = self.pos_encoding(h)
        h = self.encoder(h)
        h_last = h[:, -1, :]
        out = self.head(h_last)
        return out.squeeze(-1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 300,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

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


def evaluate_model(model: nn.Module, loader: DataLoader, scaler_y: StandardScaler) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    preds_real = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
    trues_real = scaler_y.inverse_transform(trues.reshape(-1, 1)).ravel()

    mse = mean_squared_error(trues_real, preds_real)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(trues_real, preds_real)
    r2 = r2_score(trues_real, preds_real)
    direction_acc = float(np.mean(np.sign(trues_real) == np.sign(preds_real)))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "direction_accuracy": direction_acc,
        "preds": preds_real.tolist(),
        "trues": trues_real.tolist(),
    }


def baseline_metrics(y_true: np.ndarray) -> Dict[str, float]:
    y_pred = np.zeros_like(y_true)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    direction_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "direction_accuracy": direction_acc,
    }


def write_report(context: Dict[str, object]) -> None:
    lines = ["# COMEX→Shanghai Transformer Forecast", ""]
    setup = context["setup"]
    lines.append(
        f"Data window: {setup['start_date']} — {setup['end_date']} UTC, total sequences {setup['n_samples']}."
    )
    lines.append(
        f"Train/Val/Test sizes: {setup['train_size']}/{setup['val_size']}/{setup['test_size']}."
    )
    lines.append(
        f"Sequence length {setup['seq_len']} days, horizon 1 day, features: COMEX log returns."
    )
    lines.append("")

    lines.append("## Model & Training")
    lines.append("- Architecture: Transformer encoder (d_model={d_model}, nhead={nhead}, layers={layers}).".format(**setup))
    lines.append("- Optimizer: Adam lr {lr}, weight_decay {weight_decay}, epochs {epochs}, patience {patience}.".format(**setup))
    lines.append("")

    lines.append("## Test Metrics")
    test = context["test_metrics"]
    lines.append(
        f"MSE {test['mse']:.6e}, RMSE {test['rmse']:.6e}, MAE {test['mae']:.6e}, R² {test['r2']:.4f}, Direction accuracy {test['direction_accuracy']:.3f}."
    )
    baseline = context["baseline_metrics"]
    lines.append(
        f"Baseline (zero-return) → MSE {baseline['mse']:.6e}, RMSE {baseline['rmse']:.6e}, MAE {baseline['mae']:.6e}, R² {baseline['r2']:.4f}, Direction {baseline['direction_accuracy']:.3f}."
    )
    lines.append("")

    lines.append("## Notes")
    lines.append(f"- Device used: {setup['device']}")
    lines.append("- Inputs/outputs scaled with StandardScaler fit on train split.")
    lines.append("- Loss curve saved to `analysis/gold_transformer_loss.png`.")
    lines.append("- Script command: `python analysis/gold_transformer_forecast.py`.")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    panel = build_panel()
    returns = panel[["comex_ret", "shanghai_ret"]].dropna()
    X, y, dates = create_sequences(returns, seq_len=args.seq_len)
    split = split_time_series(X, y, dates, train_frac=args.train_frac, val_frac=args.val_frac)

    if len(split.X_test) == 0:
        raise RuntimeError("Not enough samples for the requested split ratios")

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(split.X_train.reshape(-1, 1)).reshape(split.X_train.shape)
    y_train_scaled = scaler_y.fit_transform(split.y_train.reshape(-1, 1)).ravel()

    X_val_scaled = scaler_x.transform(split.X_val.reshape(-1, 1)).reshape(split.X_val.shape)
    y_val_scaled = scaler_y.transform(split.y_val.reshape(-1, 1)).ravel()
    X_test_scaled = scaler_x.transform(split.X_test.reshape(-1, 1)).reshape(split.X_test.shape)
    y_test_scaled = scaler_y.transform(split.y_test.reshape(-1, 1)).ravel()

    train_loader = DataLoader(SequenceDataset(X_train_scaled, y_train_scaled), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val_scaled, y_val_scaled), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SequenceDataset(X_test_scaled, y_test_scaled), batch_size=args.batch_size, shuffle=False)

    model = ComexTransformer(
        seq_len=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
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
    trained_model = training["model"]

    test_metrics = evaluate_model(trained_model, test_loader, scaler_y)
    baseline = baseline_metrics(split.y_test)

    setup = {
        "start_date": str(returns.index[0].date()),
        "end_date": str(returns.index[-1].date()),
        "n_samples": len(X),
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
    plot_loss(training["history"])

    summary = {
        "test_metrics": test_metrics,
        "baseline_metrics": baseline,
        "history": training["history"],
    }
    print(json.dumps(summary, indent=2))


def plot_loss(history: Dict[str, list]) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Transformer Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transformer-based COMEX→Shanghai forecaster")
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dim-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
