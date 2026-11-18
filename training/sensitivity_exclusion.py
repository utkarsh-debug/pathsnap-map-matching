"""Sensitivity of the tuned model to the 2 training trips whose reference route does not match the GPS track.
Fits the tuned model on (a) all 80 training trips (the main setting, as the PDF specifies) and (b) the 78 reliable
trips, and evaluates both on the 10 student-test trips. Run: python -m training.sensitivity_exclusion
Writes results/sensitivity_exclusion.csv"""
import json
import time

import pandas as pd

from src.config import RESULTS_DIR
from src.features import FEATURES
from training.bayes_opt import build
from training.common import evaluate, load_split, unreliable_trips


def main():
    bo = json.loads((RESULTS_DIR / "bo_best_params.json").read_text())
    name, params = bo["model"], bo["best_params"]
    tr, st = load_split("train"), load_split("student_test")
    bad = unreliable_trips()
    rows = []
    for label, d in (("all 80 trips (main setting)", tr), (f"78 reliable trips (excluding {sorted(bad)})", tr[~tr.trip.isin(bad)])):
        t0 = time.time(); m = build(name, params).fit(d[FEATURES].values, d["y"].values); fit_s = time.time() - t0
        s = evaluate(st, m.predict_proba(st[FEATURES].values)[:, 1])
        rows.append(dict(model=f"{name}_tuned", training_set=label, train_trips=d.trip.nunique(), train_rows=len(d),
                         student_top1=s["top1"], student_obs_macro_f1=s["obs_macro_f1"], student_cand_macro_f1=s["cand_macro_f1"],
                         student_obs_log_loss=s["obs_log_loss"], student_cand_log_loss=s["cand_log_loss"],
                         student_route_continuity=s["route_continuity"], train_time_s=fit_s))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "sensitivity_exclusion.csv", index=False)
    print(out.round(4).T.to_string())


if __name__ == "__main__":
    main()
