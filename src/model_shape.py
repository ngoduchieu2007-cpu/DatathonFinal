"""
LightGBM shape models for daily Revenue and COGS.

The target is divided by its year's mean (log-linear projection for
2023-2024) so the model learns within-year shape without being
distorted by the long-run growth trend or the 2020 dip. Sample
weights exponentially favor recent years. Time-series CV uses six
expanding folds at six-month strides over 2020-2022. The final model
is refit on all training data using a slightly inflated round budget
(median best-iter * 1.15). SHAP values plus gain and split importance
are written for explainability.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from .config import DATA_DIR, OUTPUT_DIR, TRAIN_END, HOLDOUT_START, LGB_PARAMS


# -------------------------------------------------------------------- #
# Helpers
# -------------------------------------------------------------------- #
def _make_yearly_mean_and_proj(sales: pd.DataFrame):
    """Yearly means on observed data; log-linear projection for 2023-2024
    fitted on 2020-2022 (the post-recovery slope)."""
    yearly_mean = (sales.assign(year=sales["Date"].dt.year)
                        .groupby("year")[["Revenue", "COGS"]].mean())
    recov = yearly_mean.loc[2020:2022]
    yrs   = recov.index.values.astype(float)
    br_rev,  ar_rev  = np.polyfit(yrs, np.log(recov["Revenue"].values), 1)
    br_cogs, ar_cogs = np.polyfit(yrs, np.log(recov["COGS"].values),    1)
    proj = pd.DataFrame({
        "year": [2023, 2024],
        "Revenue_proj": [np.exp(ar_rev  + br_rev  * y) for y in [2023, 2024]],
        "COGS_proj":    [np.exp(ar_cogs + br_cogs * y) for y in [2023, 2024]],
    }).set_index("year")
    return yearly_mean, proj, br_rev, br_cogs


def _make_scaler(dates, target, yearly_mean, proj):
    s = []
    for d in dates:
        y = d.year
        if y in yearly_mean.index:
            s.append(yearly_mean.loc[y, target])
        else:
            s.append(proj.loc[y, f"{target}_proj"])
    return np.asarray(s)


def _make_folds(dates):
    """Expanding-window time-series CV: 6 folds, 6-month strides."""
    folds = []
    for end in pd.date_range("2020-07-01", "2022-07-01", freq="6MS"):
        vstart = end
        vend   = end + pd.DateOffset(months=6) - pd.Timedelta(days=1)
        tr = np.where(dates < vstart)[0]
        va = np.where((dates >= vstart) & (dates <= vend))[0]
        if len(va) > 30:
            folds.append((tr, va))
    return folds


def _keep_feature(c: str) -> bool:
    """Drop targets, raw daily aux features (test rows are NaN), and the
    yearly trend columns (the model already sees them through the
    target normalization)."""
    if c in ("Date", "Revenue", "COGS"):
        return False
    if c.endswith("_raw"):
        return False
    if c in ("trend_rev_year", "trend_cogs_year",
             "trend_rev_per_day", "trend_cogs_per_day"):
        return False
    return True


# -------------------------------------------------------------------- #
# Main entry point
# -------------------------------------------------------------------- #
def train_shape_models(data_dir: "Path" = DATA_DIR,
                       output_dir: "Path" = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_parquet(output_dir / "master_full.parquet")
    sales  = pd.read_csv(data_dir / "sales.csv", parse_dates=["Date"])

    yearly_mean, proj, br_rev, br_cogs = _make_yearly_mean_and_proj(sales)
    print("Yearly means:")
    print(yearly_mean.round(0))
    print(f"\nRecovery slope rev:  {(np.exp(br_rev)-1)*100:+.2f}%/yr")
    print(f"Recovery slope cogs: {(np.exp(br_cogs)-1)*100:+.2f}%/yr")
    print("Projected:")
    print(proj.round(0))

    # Defensive merge so Revenue/COGS are guaranteed clean
    master = master.merge(sales, on="Date", how="left", suffixes=("", "_drop"))
    for c in [c for c in master.columns if c.endswith("_drop")]:
        master = master.drop(columns=c)

    train = (master[master["Date"] <= TRAIN_END]
             .dropna(subset=["Revenue", "COGS"])
             .reset_index(drop=True))
    test  = master[master["Date"] > TRAIN_END].reset_index(drop=True)

    FEATURES = [c for c in train.columns if _keep_feature(c)]
    print(f"\n#features: {len(FEATURES)}")

    for c in FEATURES:
        if train[c].dtype == bool:
            train[c] = train[c].astype(int)
        if c in test.columns and test[c].dtype == bool:
            test[c] = test[c].astype(int)

    train_dates = pd.to_datetime(train["Date"].values)
    test_dates  = pd.to_datetime(test["Date"].values)

    scale_rev_train  = _make_scaler(train_dates, "Revenue", yearly_mean, proj)
    scale_cogs_train = _make_scaler(train_dates, "COGS",    yearly_mean, proj)
    scale_rev_test   = _make_scaler(test_dates,  "Revenue", yearly_mean, proj)
    scale_cogs_test  = _make_scaler(test_dates,  "COGS",    yearly_mean, proj)

    train["rev_norm"]  = train["Revenue"].values / scale_rev_train
    train["cogs_norm"] = train["COGS"].values    / scale_cogs_train

    years    = train["Date"].dt.year.values
    sample_w = np.exp(0.10 * (years - years.max()))

    folds = _make_folds(train_dates)
    print(f"#CV folds: {len(folds)}")
    X_tr = train[FEATURES]
    X_te = test[FEATURES]

    def _train_target(name, target_norm_col, scale_train, scale_test):
        print(f"\n========== B1b LGB {name} (normalized) ==========")
        y_norm = train[target_norm_col].values
        y_real = train[name].values
        cv_preds_real = np.full(len(y_norm), np.nan)
        rounds = []

        for k, (tr, va) in enumerate(folds, 1):
            dtr = lgb.Dataset(X_tr.iloc[tr], y_norm[tr], weight=sample_w[tr])
            dva = lgb.Dataset(X_tr.iloc[va], y_norm[va], weight=sample_w[va],
                              reference=dtr)
            m = lgb.train(LGB_PARAMS, dtr, 5000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(200, verbose=False),
                                     lgb.log_evaluation(0)])
            p_norm = m.predict(X_tr.iloc[va])
            p_real = np.maximum(p_norm * scale_train[va], 0)
            cv_preds_real[va] = p_real
            rmse = np.sqrt(np.mean((p_real - y_real[va]) ** 2))
            mae  = np.mean(np.abs(p_real - y_real[va]))
            print(f"  fold {k}: RMSE={rmse:,.0f}  MAE={mae:,.0f}  iters={m.best_iteration}")
            rounds.append(m.best_iteration)

        final_rounds = int(np.median(rounds) * 1.15)
        print(f"  refit full -> {final_rounds} rounds")
        dall  = lgb.Dataset(X_tr, y_norm, weight=sample_w)
        final = lgb.train(LGB_PARAMS, dall, num_boost_round=final_rounds)
        test_pred = np.maximum(final.predict(X_te) * scale_test, 0)

        # Save OOF on the 2022 holdout window for downstream reporting
        holdout = (train["Date"] >= HOLDOUT_START)
        pd.DataFrame({"Date":     train["Date"][holdout].values,
                      "y":        y_real[holdout],
                      "lgb_pred": cv_preds_real[holdout]}).to_csv(
            output_dir / f"b1b_oof_{name.lower()}.csv", index=False)

        # SHAP + gain + split importance for explainability
        try:
            import shap
            sample = X_tr.sample(min(1500, len(X_tr)), random_state=0)
            sv = shap.TreeExplainer(final).shap_values(sample)
            (pd.DataFrame({"feature":       FEATURES,
                           "shap_mean_abs": np.abs(sv).mean(axis=0),
                           "gain":          final.feature_importance("gain"),
                           "split":         final.feature_importance("split")})
               .sort_values("shap_mean_abs", ascending=False)
               .to_csv(output_dir / f"b1b_shap_{name.lower()}.csv", index=False))
        except Exception:
            pass

        final.save_model(str(output_dir / f"b1b_lgb_{name.lower()}.txt"))
        return test_pred, cv_preds_real

    rev_pred,  oof_rev  = _train_target("Revenue", "rev_norm",  scale_rev_train,  scale_rev_test)
    cogs_pred, oof_cogs = _train_target("COGS",    "cogs_norm", scale_cogs_train, scale_cogs_test)

    return {
        "test_dates":  test["Date"].values,
        "rev_pred":    rev_pred,
        "cogs_pred":   cogs_pred,
        "oof_rev":     oof_rev,
        "oof_cogs":    oof_cogs,
        "train":       train,
        "yearly_mean": yearly_mean,
    }


if __name__ == "__main__":
    train_shape_models()
