"""Model selection from the recorded comparison. Run: python -m training.select_model

PRIMARY METRIC: CV observation-level Top-1 accuracy (mean over the 5 trip-grouped folds, all 80 training trips).
  Why: PathSnap's job is to say which road segment each GPS point is on; Top-1 is exactly that decision per
  observation (reference arc missing from the candidate set counts as wrong). It is computed on held-out trips,
  never on training data. Tie-break: lower CV observation-level log loss (probability quality matters because the
  HMM uses these probabilities as emission scores).
ROBUSTNESS CHECK: the same ranking restricted to the reliable trips (the 2 trips whose reference route does not
  match the GPS track are removed from the *evaluation*), plus rankings by log loss and by student-test Top-1.
Output: results/model_selection.json
"""
import json

import pandas as pd

from src.config import RESULTS_DIR


def main():
    c = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    c["rank_primary"] = c["cv_top1_mean"].rank(ascending=False, method="min")
    ranked = c.sort_values(["cv_top1_mean", "cv_obs_log_loss_mean"], ascending=[False, True])
    best = ranked.iloc[0]["model"]
    views = {
        "primary: CV Top-1 (all 80 trips)": ("cv_top1_mean", False),
        "check: CV Top-1 (reliable trips only)": ("cv_rel_top1_mean", False),
        "check: CV observation log loss (lower better)": ("cv_obs_log_loss_mean", True),
        "check: CV candidate macro F1": ("cv_cand_macro_f1_mean", False),
        "check: student-test Top-1": ("student_top1", False),
        "check: student-test observation log loss (lower better)": ("student_obs_log_loss", True),
    }
    order = {k: list(c.sort_values(col, ascending=asc)["model"]) for k, (col, asc) in views.items()}
    out = {"selected_model": best,
           "selection_rule": "highest CV observation-level Top-1; tie-break lowest CV observation log loss",
           "top1_cv_std_of_selected": float(c.loc[c.model == best, "cv_top1_std"].iloc[0]),
           "rankings": order}
    (RESULTS_DIR / "model_selection.json").write_text(json.dumps(out, indent=2))
    for k, v in order.items():
        print(f"{k}: {' > '.join(v)}")
    print("\nSELECTED:", best)


if __name__ == "__main__":
    main()
