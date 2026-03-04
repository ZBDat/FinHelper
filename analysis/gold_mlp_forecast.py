"""MLP forecaster for next-day Shanghai gold log price using flattened short sequences.

Default sequence length is 2 with flattened features per window.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local helpers
from analysis.gold_transformer_forecast import (
    build_panel_basic,
    build_windows,
    evaluate_baseline,
    evaluate_metrics,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]


class FlattenMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int] = [128, 64], dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        dims = [input_dim] + hidden
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1)


class FlatDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.X)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return self.X[idx], self.y[idx]


def prepare_data(seq_len: int = 2, include_indicators: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = build_panel_basic(include_indicators=include_indicators)
    if "label" not in df.columns:
        raise ValueError("Prepared data must contain 'label'")
    feature_cols = [c for c in df.columns if c != "label"]
    windows, targets = build_windows(df, feature_cols=feature_cols, seq_len=seq_len, label_col="label")
    # Flatten windows: (N, seq_len * feature_dim)
    flat = windows.reshape(len(windows), -1)
    return flat, targets, feature_cols


def split_flat(X: np.ndarray, y: np.ndarray, train_frac: float = 0.7, val_frac: float = 0.1):
    n = len(X)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    if train_end == 0 or val_end == train_end or val_end >= n:
        raise ValueError("Invalid split ratios for available samples")
    return (
        X[:train_end],
        y[:train_end],
        X[train_end:val_end],
        y[train_end:val_end],
        X[val_end:],
        y[val_end:],
    )


def train_mlp(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}
    no_improve = 0

    for _ in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            preds = model(xb)
            loss = crit(preds, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = crit(preds, yb)
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


def run_cli(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    X, y, feature_cols = prepare_data(seq_len=args.seq_len, include_indicators=args.include_indicators)
    X_train, y_train, X_val, y_val, X_test, y_test = split_flat(X, y, train_frac=args.train_frac, val_frac=args.val_frac)

    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_val_scaled = scaler_x.transform(X_val)
    X_test_scaled = scaler_x.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    train_loader = DataLoader(FlatDataset(X_train_scaled, y_train_scaled), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FlatDataset(X_val_scaled, y_val_scaled), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(FlatDataset(X_test_scaled, y_test_scaled), batch_size=args.batch_size, shuffle=False)

    model = FlattenMLP(input_dim=args.seq_len * len(feature_cols), hidden=[args.hidden, args.hidden // 2], dropout=args.dropout)
    training = train_mlp(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
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
    baseline = evaluate_baseline({"targets": y_test})

    report = {
        "setup": {
            "seq_len": args.seq_len,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "include_indicators": args.include_indicators,
        },
        "test_metrics": test_metrics,
        "baseline_metrics": baseline,
    }
    out_path = ROOT / "analysis" / "gold_mlp_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "scaler_x": scaler_x,
                "scaler_y": scaler_y,
                "feature_cols": feature_cols,
                "seq_len": args.seq_len,
                "include_indicators": args.include_indicators,
            },
            save_path,
        )
    # Plot loss curve
    hist = training.get("history", {})
    plt.figure(figsize=(6, 4))
    plt.plot(hist.get("train_loss", []), label="Train loss")
    plt.plot(hist.get("val_loss", []), label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Training Loss (seq_len=2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "analysis" / "gold_mlp_loss.png", dpi=150)
    plt.close()
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLP forecaster for Shanghai gold (flattened windows)")
    parser.add_argument("--seq-len", type=int, default=2, help="Window length (flattened)")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--include-indicators", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=Path, default=None, help="Path to save trained model state")
    return parser.parse_args()


if __name__ == "__main__":
    run_cli(parse_args())
