"""
End-to-end pipeline runner for Datathon 2026, Round 1.

Stages
------
1. build_master       : merge raw CSVs and engineer features.
2. train_shape_models : fit normalized LightGBM models for Revenue and COGS.
3. fit_global_multiplier : derive the global scale correction.
4. write_submissions  : emit submission CSVs (and SHAP / OOF reports).

All artifacts are written to ./output/.
"""
from src.config import OUTPUT_DIR
from src.build_master import build_master
from src.model_shape import train_shape_models
from src.model_multiplier import fit_global_multiplier
from src.predict import write_submissions


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    build_master()

    shape = train_shape_models()

    mult_rev, mult_cogs = fit_global_multiplier(
        train=shape["train"],
        oof_rev=shape["oof_rev"],
        oof_cogs=shape["oof_cogs"],
        yearly_mean=shape["yearly_mean"],
    )

    write_submissions(
        test_dates=shape["test_dates"],
        rev_pred=shape["rev_pred"],
        cogs_pred=shape["cogs_pred"],
        mult_rev=mult_rev,
        mult_cogs=mult_cogs,
    )


if __name__ == "__main__":
    main()
