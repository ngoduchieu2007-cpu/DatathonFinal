"""Combine shape predictions and the global multiplier into Kaggle submissions."""
import numpy as np
import pandas as pd

from .config import DATA_DIR, OUTPUT_DIR


def write_submissions(test_dates,
                      rev_pred: np.ndarray,
                      cogs_pred: np.ndarray,
                      mult_rev: float,
                      mult_cogs: float,
                      data_dir: "Path" = DATA_DIR,
                      output_dir: "Path" = OUTPUT_DIR) -> pd.DataFrame:
    """Write the multiplier-adjusted Kaggle submission and the shape-only baseline.

    Both files preserve the row order of sample_submission.csv.
    """
    sub_template = pd.read_csv(data_dir / "sample_submission.csv", parse_dates=["Date"])

    # ---- Shape-only baseline ------------------------------------------
    base = pd.DataFrame({"Date": test_dates,
                         "Revenue": rev_pred,
                         "COGS":    cogs_pred})
    sub_base = sub_template[["Date"]].merge(base, on="Date", how="left")
    assert sub_base["Revenue"].notna().all()
    sub_base["Date"]    = sub_base["Date"].dt.strftime("%Y-%m-%d")
    sub_base["Revenue"] = sub_base["Revenue"].round(2)
    sub_base["COGS"]    = sub_base["COGS"].round(2)
    sub_base.to_csv(output_dir / "submission_B1b.csv", index=False)
    print(f"\nSaved submission_B1b.csv ({len(sub_base)} rows)")

    # ---- Final submission (with multiplier) ---------------------------
    rev_final  = np.maximum(rev_pred  * mult_rev,  0)
    cogs_final = np.maximum(cogs_pred * mult_cogs, 0)

    final = pd.DataFrame({"Date": test_dates,
                          "Revenue": rev_final,
                          "COGS":    cogs_final})
    sub_final = sub_template[["Date"]].merge(final, on="Date", how="left")
    assert sub_final["Revenue"].notna().all()
    sub_final["Date"]    = sub_final["Date"].dt.strftime("%Y-%m-%d")
    sub_final["Revenue"] = sub_final["Revenue"].round(2)
    sub_final["COGS"]    = sub_final["COGS"].round(2)

    # Two filenames for convenience: submission.csv is the canonical
    # Kaggle upload; submission_with_multiplier.csv keeps the original
    # naming used in the experimentation notebook.
    sub_final.to_csv(output_dir / "submission_with_multiplier.csv", index=False)
    sub_final.to_csv(output_dir / "submission.csv", index=False)

    print(f"Saved submission.csv ({len(sub_final)} rows)")
    print(sub_final[["Revenue", "COGS"]].describe().round(0))
    return sub_final
