"""
Global multiplier submodel.

The shape model captures within-year seasonality but its absolute scale
drifts because the post-2020 recovery slope is steeper than the
log-linear projection assumes. We correct this with a single scalar
applied uniformly to the test window.

For each year y >= 2019 we compute two residuals from out-of-fold data:

    shape_error(y) = actual_yearly_mean(y) / oof_yearly_mean(y)
    scale_error(y) = actual_yearly_mean(y) / walk_forward_proj(y)
    combined(y)    = shape_error(y) * scale_error(y)

The walk-forward projection at year y is a log-linear fit on the three
preceding years' yearly means - it deliberately reproduces the same
information set the shape model sees at training time. The global
multiplier is a recency-weighted average of post-2019 combined values,
clipped for safety.
"""
import numpy as np
import pandas as pd


def _walk_forward_proj(series: pd.Series, year: int) -> float:
    """Log-linear extrapolation from the three years preceding `year`."""
    fit_years = [y for y in range(year - 3, year) if y in series.index]
    if len(fit_years) < 2:
        return np.nan
    means = series.loc[fit_years].values
    b, a = np.polyfit(fit_years, np.log(means), 1)
    return float(np.exp(a + b * year))


def fit_global_multiplier(train: pd.DataFrame,
                          oof_rev: np.ndarray,
                          oof_cogs: np.ndarray,
                          yearly_mean: pd.DataFrame):
    print("\n\n========== Global Multiplier Submodel ==========")

    # Walk-forward projections for 2015-2022, plus the 2020-2022 fit
    # used for the 2023/2024 horizon.
    wf_rev  = {y: _walk_forward_proj(yearly_mean["Revenue"], y) for y in range(2015, 2023)}
    wf_cogs = {y: _walk_forward_proj(yearly_mean["COGS"],    y) for y in range(2015, 2023)}
    for Y in [2023, 2024]:
        fy = [2020, 2021, 2022]
        b, a = np.polyfit(fy, np.log(yearly_mean.loc[fy, "Revenue"].values), 1)
        wf_rev[Y]  = float(np.exp(a + b * Y))
        b, a = np.polyfit(fy, np.log(yearly_mean.loc[fy, "COGS"].values), 1)
        wf_cogs[Y] = float(np.exp(a + b * Y))

    oof_df = pd.DataFrame({
        "Date":     train["Date"].values,
        "Revenue":  train["Revenue"].values,
        "COGS":     train["COGS"].values,
        "oof_rev":  oof_rev,
        "oof_cogs": oof_cogs,
    }).dropna(subset=["oof_rev"])
    oof_df["year"] = oof_df["Date"].dt.year

    yearly_mult = (oof_df.groupby("year")
                         .agg(actual_rev=("Revenue", "mean"),
                              pred_rev=("oof_rev", "mean"),
                              actual_cogs=("COGS", "mean"),
                              pred_cogs=("oof_cogs", "mean"))
                         .reset_index())

    yearly_mult["shape_err_rev"]  = yearly_mult["actual_rev"]  / yearly_mult["pred_rev"]
    yearly_mult["shape_err_cogs"] = yearly_mult["actual_cogs"] / yearly_mult["pred_cogs"]
    yearly_mult["scale_err_rev"]  = (yearly_mult["year"].map(yearly_mean["Revenue"].to_dict())
                                      / yearly_mult["year"].map(wf_rev))
    yearly_mult["scale_err_cogs"] = (yearly_mult["year"].map(yearly_mean["COGS"].to_dict())
                                      / yearly_mult["year"].map(wf_cogs))
    yearly_mult["mult_rev"]  = (yearly_mult["shape_err_rev"]
                                * yearly_mult["scale_err_rev"]).clip(0.5, 3.0)
    yearly_mult["mult_cogs"] = (yearly_mult["shape_err_cogs"]
                                * yearly_mult["scale_err_cogs"]).clip(0.5, 3.0)

    yearly_mult = yearly_mult[yearly_mult["year"] >= 2019].reset_index(drop=True)

    print(f"\nPer-year combined multipliers (post-2019):")
    print(f"  {'Year':>4}  {'shape_err':>10}  {'scale_err':>10}  {'combined':>10}")
    for _, r in yearly_mult.iterrows():
        print(f"  {int(r['year']):>4}  {r['shape_err_rev']:>10.4f}  "
              f"{r['scale_err_rev']:>10.4f}  {r['mult_rev']:>10.4f}")

    # Linearly increasing recency weights: oldest=1, newest=n
    n_years = len(yearly_mult)
    recency_weights = np.arange(1, n_years + 1, dtype=float)

    g_rev  = float(np.average(yearly_mult["mult_rev"].values,  weights=recency_weights))
    g_cogs = float(np.average(yearly_mult["mult_cogs"].values, weights=recency_weights))
    g_rev  = float(np.clip(g_rev,  0.8, 2.5))
    g_cogs = float(np.clip(g_cogs, 0.8, 2.5))

    print(f"\nGlobal multiplier (recency-weighted mean of post-2019 years):")
    print(f"  rev={g_rev:.4f}  cogs={g_cogs:.4f}")
    return g_rev, g_cogs
