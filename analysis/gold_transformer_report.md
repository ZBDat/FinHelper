{
  "setup": {
    "start_date": "2017-01-26",
    "end_date": "2026-02-13",
    "n_samples": 2182,
    "train_size": 1527,
    "val_size": 218,
    "test_size": 437,
    "seq_len": 3,
    "d_model": 60,
    "nhead": 5,
    "layers": 1,
    "dim_ff": 256,
    "dropout": 0.2,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "epochs": 100,
    "patience": 100,
    "device": "cuda"
  },
  "test_metrics": {
    "rmse_price": 0.5196939084683844,
    "mae_price": 0.4717978239059448,
    "mse_price": 0.2700817584991455,
    "r2": -5.158025741577148,
    "direction_accuracy": 0.43935926773455375
  },
  "baseline_metrics": {
    "rmse_price": 6.5820665192088015,
    "mae_price": 6.578734009235542,
    "mse_price": 43.323599663289464
  }
}