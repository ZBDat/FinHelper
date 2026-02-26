# COMEX→Shanghai Transformer Forecast

Data window: 2016-12-20 — 2026-02-24 UTC, total sequences 2206.
Train/Val/Test sizes: 1544/220/442.
Sequence length 5 days, horizon 1 day, features: COMEX log returns.

## Model & Training
- Architecture: Transformer encoder (d_model=32, nhead=4, layers=2).
- Optimizer: Adam lr 0.001, weight_decay 0.0001, epochs 300, patience 300.

## Test Metrics
MSE 1.512155e-04, RMSE 1.229697e-02, MAE 7.107423e-03, R² 0.1500, Direction accuracy 0.706.
Baseline (zero-return) → MSE 1.804692e-04, RMSE 1.343388e-02, MAE 8.379753e-03, R² -0.0144, Direction 0.000.

## Notes
- Device used: cuda
- Inputs/outputs scaled with StandardScaler fit on train split.
- Loss curve saved to `analysis/gold_transformer_loss.png`.
- Script command: `python analysis/gold_transformer_forecast.py`.