# Datathon 2026 — Round 1: Sales Forecasting

Daily Revenue and COGS forecasts for a Vietnamese fashion e-commerce
retailer over the **2023-01-01 → 2024-07-01** horizon.

## Approach

A two-stage model.

**Shape model.** A LightGBM regressor trained on a year-mean-normalized
target so it learns within-year shape and seasonality, decoupled from
the long-run growth trend and the 2020 dip. Sample weights
exponentially favor recent years (`exp(0.10 * (year - 2022))`). Six
expanding-window time-series CV folds at six-month strides cover
2020-2022; the final model is refit on all training data using
`median(best_iter) * 1.15` rounds.

**Global multiplier.** A single scalar correction is fit from post-2019
out-of-fold residuals. It absorbs the fact that the post-recovery
growth slope is steeper than a log-linear projection from earlier
years assumes. The multiplier combines two yearly residuals -
`shape_error = actual / OOF` and `scale_error = actual / walk_forward_proj` -
and is averaged with linearly increasing recency weights, then clipped
to `[0.8, 2.5]`.

Features come exclusively from the supplied CSVs and fall into four
groups:

- Calendar and event flags (day-of-week, cyclical encodings, Tet
  windows with signed `days_to_tet`, mega-sale dates 9.9 / 10.10 /
  11.11 / 12.12, Black Friday, Vietnamese public holidays).
- Promotion calendar derived from `promotions.csv`.
- Daily operational signals from web traffic, orders, order_items,
  returns, reviews, customer signups, payments, and shipments,
  collapsed to `(month, dow)` seasonal profiles using only training
  rows so test rows never see future operational data.
- Long-range sales lags of 365, 730, 1095, 1460 days plus rolling
  365-day means at the same lags. The minimum lag of 365 days exceeds
  the 18-month test horizon, so these are all known at prediction time.

No raw test-period operational features and no test-period Revenue or
COGS values enter the model.

## Repository layout

```
.
├── main.py                       # entry point
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py                 # paths, constants, LGB hyperparameters, calendar tables
│   ├── features.py               # calendar, event, and promotion feature builders
│   ├── build_master.py           # raw CSVs -> master_full.parquet
│   ├── model_shape.py            # LightGBM Revenue and COGS shape models
│   ├── model_multiplier.py       # global scale correction submodel
│   └── predict.py                # writes submission CSVs
├── dataset/                      # raw competition CSVs (not committed)
└── output/                       # generated artifacts (created on first run)
```

## How to reproduce

1. Create the environment:

   ```bash
   pip install -r requirements.txt
   ```

2. Place the 14 raw CSVs into `./dataset/`.

   ```
   dataset/
   ├── customers.csv
   ├── geography.csv
   ├── inventory.csv
   ├── order_items.csv
   ├── orders.csv
   ├── payments.csv
   ├── products.csv
   ├── promotions.csv
   ├── returns.csv
   ├── reviews.csv
   ├── sales.csv
   ├── sample_submission.csv
   ├── shipments.csv
   └── web_traffic.csv
   ```

3. Run the pipeline:

   ```bash
   python main.py
   ```

The pipeline writes the following to `./output/`:

| File | Purpose |
| --- | --- |
| `submission.csv` | Final Kaggle submission (with multiplier). |
| `submission_with_multiplier.csv` | Identical to `submission.csv`, kept for traceability. |
| `submission_B1b.csv` | Shape-only baseline (multiplier off). |
| `master_full.parquet`, `master_train.parquet`, `master_test.parquet` | Cached feature frames. |
| `b1b_oof_revenue.csv`, `b1b_oof_cogs.csv` | Out-of-fold predictions on the 2022 holdout. |
| `b1b_shap_revenue.csv`, `b1b_shap_cogs.csv` | SHAP, gain, and split importances. |
| `b1b_lgb_revenue.txt`, `b1b_lgb_cogs.txt` | Saved LightGBM models. |

`seed=42` is pinned in `LGB_PARAMS` and SHAP sampling uses
`random_state=0`, so results are reproducible bit-for-bit.

## Compliance with competition constraints

- **No external data.** All features are built from the supplied CSVs.
  Tet and fixed-holiday dates are calendar facts encoded in
  `src/config.py`, not external datasets.
- **No test-period Revenue or COGS as input.** Lag features use a
  minimum lag of 365 days; raw daily operational features are kept
  with a `_raw` suffix and explicitly dropped in `src/model_shape.py`
  before training; target-mean scalers for 2023 and 2024 use a
  log-linear projection fit on 2020-2022 only.
- **Reproducibility.** Random seeds are pinned and the entire pipeline
  is one command.
- **Explainability.** SHAP values plus gain and split importances are
  exported per target.
