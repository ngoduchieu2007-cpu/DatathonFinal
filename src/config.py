"""Paths, constants, LightGBM hyperparameters, and calendar tables."""
from pathlib import Path
import pandas as pd

ROOT       = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT / "dataset"
OUTPUT_DIR = ROOT / "output"

TRAIN_END     = pd.Timestamp("2022-12-31")
TEST_END      = pd.Timestamp("2024-07-01")
HOLDOUT_START = pd.Timestamp("2022-01-01")
SEED          = 42

LGB_PARAMS = dict(
    objective="regression", metric="rmse",
    learning_rate=0.025, num_leaves=63, min_data_in_leaf=30,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    lambda_l2=2.0, lambda_l1=0.3, verbose=-1, seed=SEED, n_jobs=-1,
)

# Vietnamese Lunar New Year (Tet), day 1 of each year. Used to derive
# signed days-to-Tet and Tet-window flags. These dates are calendar
# facts, not external data.
TET_DATES = {
    2012: "2012-01-23", 2013: "2013-02-10", 2014: "2014-01-31", 2015: "2015-02-19",
    2016: "2016-02-08", 2017: "2017-01-28", 2018: "2018-02-16", 2019: "2019-02-05",
    2020: "2020-01-25", 2021: "2021-02-12", 2022: "2022-02-01", 2023: "2023-01-22",
    2024: "2024-02-10",
}
TET_DATES = {k: pd.Timestamp(v) for k, v in TET_DATES.items()}

# Fixed-date Vietnamese events plus pan-Asia mega-sale dates that drive
# fashion e-commerce revenue (9.9, 10.10, 11.11, 12.12).
FIXED_EVENTS = {
    "newyear":          (1, 1),
    "valentine":        (2, 14),
    "intl_women":       (3, 8),
    "reunif":           (4, 30),
    "labor":            (5, 1),
    "natl":             (9, 2),
    "mid_autumn_proxy": (9, 15),
    "viet_women":       (10, 20),
    "sale_99":          (9, 9),
    "sale_1010":        (10, 10),
    "sale_1111":        (11, 11),
    "sale_1212":        (12, 12),
    "christmas":        (12, 25),
    "newyear_eve":      (12, 31),
}
