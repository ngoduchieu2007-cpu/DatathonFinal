"""
Build the daily master frame consumed by the forecasting models.

Pipeline
--------
1. Load raw CSVs.
2. Generate calendar, event, and promotion features over the full
   2012-07-04 to 2024-07-01 horizon.
3. Aggregate every operational table to daily level and derive
   (month, dow) seasonal profiles using only the train window.
   Test rows therefore receive seasonal averages and never see raw
   future operational signals.
4. Add a log-linear yearly trend fitted on 2013-2022 totals.
5. Append safe long-range sales lags (>= 365 days), so the 18-month
   test horizon is never reached.
6. Persist master_full / master_train / master_test parquets.
"""
import numpy as np
import pandas as pd

from .config import DATA_DIR, OUTPUT_DIR, TRAIN_END, TEST_END
from .features import (
    add_calendar_features, add_event_features,
    compute_promo_features, daily_agg,
)


def build_master(data_dir: "Path" = DATA_DIR,
                 output_dir: "Path" = OUTPUT_DIR) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    sales = pd.read_csv(data_dir / "sales.csv", parse_dates=["Date"])
    sub   = pd.read_csv(data_dir / "sample_submission.csv", parse_dates=["Date"])
    all_dates = pd.date_range(sales["Date"].min(), sub["Date"].max(), freq="D")
    M = pd.DataFrame({"Date": all_dates})

    # --- Calendar / event / promo flags ---------------------------------
    M = add_calendar_features(M, anchor_date=sales["Date"].min())
    M = add_event_features(M)

    promos = pd.read_csv(data_dir / "promotions.csv",
                         parse_dates=["start_date", "end_date"])
    M = pd.concat(
        [M.reset_index(drop=True),
         compute_promo_features(M["Date"], promos)],
        axis=1,
    )
    print("promo features done")

    # --- Daily operational aggregates ----------------------------------
    orders = pd.read_csv(data_dir / "orders.csv", parse_dates=["order_date"])


    wt = pd.read_csv(data_dir / "web_traffic.csv", parse_dates=["date"])
    wt_d = daily_agg(wt, "date", dict(
        sessions=("sessions", "sum"),
        unique_visitors=("unique_visitors", "sum"),
        page_views=("page_views", "sum"),
        bounce_rate=("bounce_rate", "mean"),
        avg_session_sec=("avg_session_duration_sec", "mean"),
    ))

    od = daily_agg(orders, "order_date", dict(
        n_orders=("order_id", "count"),
        n_unique_cust=("customer_id", "nunique"),
    ))

    oi = pd.read_csv(data_dir / "order_items.csv")
    oi = oi.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    oi["gross"] = oi["quantity"] * oi["unit_price"]
    oid = daily_agg(oi, "order_date", dict(
        total_qty=("quantity", "sum"),
        total_gross=("gross", "sum"),
        total_disc=("discount_amount", "sum"),
        avg_unit_price=("unit_price", "mean"),
        n_lines=("product_id", "count"),
        n_with_promo=("promo_id", lambda s: s.notna().sum()),
    ))
    oid["disc_rate"]   = oid["total_disc"] / oid["total_gross"].replace(0, np.nan)
    oid["promo_share"] = oid["n_with_promo"] / oid["n_lines"].replace(0, np.nan)

    prod = pd.read_csv(data_dir / "products.csv")
    oi_cat = oi.merge(prod[["product_id", "category", "segment"]],
                      on="product_id", how="left")

    cat_mix = (oi_cat.groupby(["order_date", "category"])["gross"].sum()
                     .unstack(fill_value=0))
    cat_mix = cat_mix.div(cat_mix.sum(axis=1).replace(0, np.nan), axis=0)
    cat_mix = (cat_mix.add_prefix("cat_share_")
                      .reset_index()
                      .rename(columns={"order_date": "Date"}))

    seg_mix = (oi_cat.groupby(["order_date", "segment"])["gross"].sum()
                     .unstack(fill_value=0))
    seg_mix = seg_mix.div(seg_mix.sum(axis=1).replace(0, np.nan), axis=0)
    seg_mix = (seg_mix.add_prefix("seg_share_")
                      .reset_index()
                      .rename(columns={"order_date": "Date"}))

    ret = pd.read_csv(data_dir / "returns.csv", parse_dates=["return_date"])
    ret_d = daily_agg(ret, "return_date", dict(
        n_returns=("return_id", "count"),
        refund_amt=("refund_amount", "sum"),
    ))

    rev = pd.read_csv(data_dir / "reviews.csv", parse_dates=["review_date"])
    rev_d = daily_agg(rev, "review_date", dict(
        n_reviews=("review_id", "count"),
        avg_rating=("rating", "mean"),
    ))

    cust = pd.read_csv(data_dir / "customers.csv", parse_dates=["signup_date"])
    sign_d = (cust.groupby("signup_date").size()
                  .reset_index(name="n_signups")
                  .rename(columns={"signup_date": "Date"}))

    pay = pd.read_csv(data_dir / "payments.csv")
    pay = pay.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    pay_d = daily_agg(pay, "order_date", dict(
        avg_installments=("installments", "mean"),
        avg_payment=("payment_value", "mean"),
    ))

    ship = pd.read_csv(data_dir / "shipments.csv",
                       parse_dates=["ship_date", "delivery_date"])
    ship["lead_time"] = (ship["delivery_date"] - ship["ship_date"]).dt.days
    ship = ship.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    ship_d = daily_agg(ship, "order_date", dict(
        avg_lead_time=("lead_time", "mean"),
        avg_ship_fee=("shipping_fee", "mean"),
    ))

    aux_tables = [wt_d, od, oid, cat_mix, seg_mix, ret_d, rev_d, sign_d, pay_d, ship_d]
    aux = M[["Date"]].copy()
    for t in aux_tables:
        aux = aux.merge(t, on="Date", how="left")

    # --- (month, dow) profile from train rows only ---------------------
    aux_t = aux[aux["Date"] <= TRAIN_END].copy()
    aux_t["month"] = aux_t["Date"].dt.month
    aux_t["dow"]   = aux_t["Date"].dt.dayofweek
    profile_cols = [c for c in aux.columns if c != "Date"]
    profile = (aux_t.groupby(["month", "dow"])[profile_cols]
                    .mean()
                    .add_suffix("_prof")
                    .reset_index())
    M = M.merge(profile, on=["month", "dow"], how="left")

    # Raw daily values are kept under a *_raw suffix for diagnostics but
    # are explicitly dropped before model training.
    aux_raw = aux.rename(columns={c: f"{c}_raw" for c in profile_cols})
    M = M.merge(aux_raw, on="Date", how="left")
    print("aux profiles merged")

    # --- Yearly log-linear trend ---------------------------------------
    yearly = (sales.assign(year=sales["Date"].dt.year)
                   .groupby("year")[["Revenue", "COGS"]].sum())
    ftrain = yearly.loc[2013:2022]
    yrs = ftrain.index.values.astype(float)
    b_r, a_r = np.polyfit(yrs, np.log(ftrain["Revenue"].values), 1)
    b_c, a_c = np.polyfit(yrs, np.log(ftrain["COGS"].values),    1)
    M["trend_rev_year"]    = np.exp(a_r + b_r * M["year"])
    M["trend_cogs_year"]   = np.exp(a_c + b_c * M["year"])
    M["trend_rev_per_day"]  = M["trend_rev_year"]  / 365.0
    M["trend_cogs_per_day"] = M["trend_cogs_year"] / 365.0

    # --- Long-range sales lags (>= 365 days) ---------------------------
    sx = sales.set_index("Date").sort_index()
    sx["rev_roll365"]  = sx["Revenue"].rolling(365, min_periods=180).mean()
    sx["cogs_roll365"] = sx["COGS"].rolling(365, min_periods=180).mean()

    for lag in [365, 730, 1095, 1460]:
        shifted = sx[["Revenue", "COGS"]].copy()
        shifted.index = shifted.index + pd.Timedelta(days=lag)
        shifted = shifted.rename(columns={"Revenue": f"rev_lag{lag}",
                                          "COGS":    f"cogs_lag{lag}"})
        M = M.merge(shifted, left_on="Date", right_index=True, how="left")

    for lag in [365, 730, 1095]:
        shifted = sx[["rev_roll365", "cogs_roll365"]].copy()
        shifted.index = shifted.index + pd.Timedelta(days=lag)
        shifted = shifted.rename(columns={
            "rev_roll365":  f"rev_roll365_lag{lag}",
            "cogs_roll365": f"cogs_roll365_lag{lag}",
        })
        M = M.merge(shifted, left_on="Date", right_index=True, how="left")

    # Attach actual targets (NaN on test rows)
    M = M.merge(sales, on="Date", how="left")

    train = M[M["Date"] <= TRAIN_END].reset_index(drop=True)
    test  = M[(M["Date"] > TRAIN_END) & (M["Date"] <= TEST_END)].reset_index(drop=True)

    M.to_parquet(output_dir / "master_full.parquet", index=False)
    train.to_parquet(output_dir / "master_train.parquet", index=False)
    test.to_parquet(output_dir / "master_test.parquet",   index=False)

    n_feat = len([c for c in M.columns if c not in ("Date", "Revenue", "COGS")])
    print(f"\nTrain: {train.shape}  ({train['Date'].min().date()} -> {train['Date'].max().date()})")
    print(f"Test : {test.shape}  ({test['Date'].min().date()} -> {test['Date'].max().date()})")
    print(f"Total feature cols: {n_feat}")
    return M


if __name__ == "__main__":
    build_master()
