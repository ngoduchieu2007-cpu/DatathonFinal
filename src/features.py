"""
Feature builders that depend only on the calendar or on the promotion
schedule. Each function below produces values that are knowable on
arbitrary future dates without touching test-period targets.
"""
import numpy as np
import pandas as pd

from .config import TET_DATES, FIXED_EVENTS


def add_calendar_features(M: pd.DataFrame, anchor_date: pd.Timestamp) -> pd.DataFrame:
    """Standard date parts plus cyclical (sin, cos) encodings."""
    d = M["Date"]
    M["year"]    = d.dt.year
    M["month"]   = d.dt.month
    M["day"]     = d.dt.day
    M["dow"]     = d.dt.dayofweek
    M["doy"]     = d.dt.dayofyear
    M["woy"]     = d.dt.isocalendar().week.astype(int)
    M["quarter"] = d.dt.quarter
    M["is_weekend"]      = (M["dow"] >= 5).astype(int)
    M["is_month_start"]  = d.dt.is_month_start.astype(int)
    M["is_month_end"]    = d.dt.is_month_end.astype(int)
    M["is_quarter_end"]  = d.dt.is_quarter_end.astype(int)
    M["days_since_2012"] = (d - anchor_date).dt.days
    for col, period in [("dow", 7), ("doy", 366), ("month", 12)]:
        M[f"{col}_sin"] = np.sin(2 * np.pi * M[col] / period)
        M[f"{col}_cos"] = np.cos(2 * np.pi * M[col] / period)
    return M


def add_event_features(M: pd.DataFrame) -> pd.DataFrame:
    """Tet windows, fixed local holidays, and Black Friday."""
    def _days_to_tet(date):
        diffs = [(t - date).days for t in TET_DATES.values()]
        diffs = [x for x in diffs if abs(x) <= 60]
        return min(diffs, key=abs) if diffs else 60

    M["days_to_tet"]   = M["Date"].apply(_days_to_tet)
    M["is_tet_window"] = (M["days_to_tet"].abs() <= 7).astype(int)
    M["is_pre_tet"]    = ((M["days_to_tet"] > 0) & (M["days_to_tet"] <= 14)).astype(int)
    M["is_post_tet"]   = ((M["days_to_tet"] < 0) & (M["days_to_tet"] >= -7)).astype(int)

    for name, (mo, da) in FIXED_EVENTS.items():
        M[f"is_{name}"] = ((M["month"] == mo) & (M["day"] == da)).astype(int)

    def _is_black_friday(dt):
        if dt.month != 11:
            return 0
        last = pd.Timestamp(year=dt.year, month=11, day=30)
        bf = last - pd.Timedelta(days=(last.weekday() - 4) % 7)
        return int(dt.normalize() == bf)

    M["is_black_friday"] = M["Date"].apply(_is_black_friday)
    return M


def compute_promo_features(dates: pd.Series, promos: pd.DataFrame) -> pd.DataFrame:
    """Summarise promotions whose [start_date, end_date] interval covers each date."""
    rows = []
    for dt in dates:
        active = promos[(promos["start_date"] <= dt) & (promos["end_date"] >= dt)]
        if len(active):
            pct = active.loc[active["promo_type"] == "percentage", "discount_value"]
            fxd = active.loc[active["promo_type"] == "fixed",      "discount_value"]
            rows.append({
                "n_active_promos": len(active),
                "max_promo_pct":   pct.max()  if len(pct) else 0,
                "mean_promo_pct":  pct.mean() if len(pct) else 0,
                "max_promo_fixed": fxd.max()  if len(fxd) else 0,
                "n_stackable":     int(active["stackable_flag"].sum()),
            })
        else:
            rows.append({"n_active_promos": 0, "max_promo_pct": 0,
                         "mean_promo_pct": 0, "max_promo_fixed": 0, "n_stackable": 0})
    return pd.DataFrame(rows)


def daily_agg(df: pd.DataFrame, date_col: str, aggs: dict) -> pd.DataFrame:
    """groupby(date_col).agg(...) and rename the date column to 'Date'."""
    return (df.groupby(date_col)
              .agg(**aggs)
              .reset_index()
              .rename(columns={date_col: "Date"}))
