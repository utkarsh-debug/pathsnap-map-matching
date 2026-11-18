"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected model (XGBoost).

Objective : maximise mean macro F1 over the SAME 5 trip-grouped CV folds used in the model comparison
            (78 usable training trips). Per-trial log loss / Top-1 / continuity are also recorded.
Data      : training trips only for search; student-test trips only for the final tuned-vs-untuned check.
Run       : python -m training.bayes_opt [n_trials]
Outputs   : results/bo_trials.csv, results/bo_best_params.json, results/bo_search_space.json,
            results/tuned_vs_untuned.csv, results/figures/bo_*.png, data/processed/models/XGBoost_tuned.*
"""
import json
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from src.config import FIG_DIR, PROCESSED_DIR, RESULTS_DIR, SEED
from src.features import FEATURES
from training.common import evaluate, load_split

N_SPLITS = 5
SEARCH_SPACE = {
    "n_estimators": "int [100, 600]",
    "learning_rate": "float log-uniform [0.01, 0.3]",
    "max_depth": "int [3, 10]",
    "min_child_weight": "float log-uniform [1, 50]",
    "subsample": "float uniform [0.5, 1.0]",
    "colsample_bytree": "float uniform [0.5, 1.0]",
    "reg_lambda": "float log-uniform [0.01, 50]",
    "reg_alpha": "float log-uniform [0.001, 10]",
    "gamma": "float uniform [0, 5]",
}
UNTUNED = dict(n_estimators=300, learning_rate=0.05, max_depth=6)


def make(params):
    return xgb.XGBClassifier(tree_method="hist", random_state=SEED, n_jobs=-1, eval_metric="logloss", **params)


def cv_eval(tr, folds, params):
    rows = []
    for a, b in folds:
        dtr, dva = tr.iloc[a], tr.iloc[b]
        m = make(params).fit(dtr[FEATURES].values, dtr["y"].values)
        rows.append(evaluate(dva, m.predict_proba(dva[FEATURES].values)[:, 1]))
    return pd.DataFrame(rows)


def main(n_trials):
    tr = load_split("train")
    st = load_split("student_test", usable_only=False)
    folds = list(GroupKFold(n_splits=N_SPLITS).split(tr, groups=tr["trip"].values))

    def objective(trial):
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 50, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.01, 50, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 0.001, 10, log=True),
            gamma=trial.suggest_float("gamma", 0, 5),
        )
        t0 = time.time()
        r = cv_eval(tr, folds, p)
        trial.set_user_attr("cv_log_loss", float(r.log_loss.mean()))
        trial.set_user_attr("cv_top1", float(r.top1.mean()))
        trial.set_user_attr("cv_route_continuity", float(r.route_continuity.mean()))
        trial.set_user_attr("cv_macro_f1_std", float(r.macro_f1.std()))
        trial.set_user_attr("seconds", time.time() - t0)
        print(f"trial {trial.number:02d} F1={r.macro_f1.mean():.4f} ll={r.log_loss.mean():.4f} "
              f"top1={r.top1.mean():.4f} {time.time()-t0:.0f}s {p}", flush=True)
        return float(r.macro_f1.mean())

    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="xgb_pathsnap")
    # the untuned config is trial 0, so the search is compared against the baseline on equal terms
    study.enqueue_trial({**UNTUNED, "min_child_weight": 1.0, "subsample": 1.0, "colsample_bytree": 1.0,
                         "reg_lambda": 1.0, "reg_alpha": 0.001, "gamma": 0.0})
    t_start = time.time()
    study.optimize(objective, n_trials=n_trials)
    search_s = time.time() - t_start

    trials = study.trials_dataframe()
    trials.to_csv(RESULTS_DIR / "bo_trials.csv", index=False)
    best = study.best_params
    (RESULTS_DIR / "bo_best_params.json").write_text(json.dumps(
        {"best_params": best, "best_cv_macro_f1": study.best_value, "n_trials": n_trials,
         "sampler": "TPE (seed 42, 10 random start-up trials)", "search_seconds": search_s}, indent=2))
    (RESULTS_DIR / "bo_search_space.json").write_text(json.dumps(
        {"objective": "maximise mean macro F1 over 5-fold GroupKFold (groups=trip) on 78 usable training trips",
         "space": SEARCH_SPACE, "untuned_baseline": UNTUNED}, indent=2))

    # tuned vs untuned: identical folds, identical metrics, identical student-test partition
    out = []
    for label, params in (("untuned", UNTUNED), ("tuned", best)):
        r = cv_eval(tr, folds, params)
        t0 = time.time(); m = make(params).fit(tr[FEATURES].values, tr["y"].values); fit_s = time.time() - t0
        t0 = time.time(); p = m.predict_proba(st[FEATURES].values)[:, 1]; inf_s = time.time() - t0
        s = evaluate(st, p)
        out.append(dict(model=f"XGBoost_{label}", cv_macro_f1_mean=r.macro_f1.mean(), cv_macro_f1_std=r.macro_f1.std(),
                        cv_top1=r.top1.mean(), cv_log_loss=r.log_loss.mean(),
                        student_top1=s["top1"], student_macro_f1=s["macro_f1"], student_log_loss=s["log_loss"],
                        student_route_continuity=s["route_continuity"], train_time_s=fit_s,
                        infer_time_s=inf_s, infer_ms_per_obs=inf_s / s["n_obs"] * 1000, params=json.dumps(params)))
        if label == "tuned":
            joblib.dump(m, PROCESSED_DIR / "models" / "XGBoost_tuned.joblib")
            pd.DataFrame({"trip": st.trip.values, "obs": st.obs.values, "arc": st.arc.values,
                          "y": st.y.values, "p": p}).to_parquet(PROCESSED_DIR / "models" / "preds_XGBoost_tuned.parquet")
    cmp = pd.DataFrame(out)
    cmp.to_csv(RESULTS_DIR / "tuned_vs_untuned.csv", index=False)
    print(cmp.drop(columns="params").round(4).T.to_string())

    done = trials[trials.state == "COMPLETE"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(done.number, done.value, "o", alpha=0.6, label="trial")
    ax[0].plot(done.number, done.value.cummax(), "-", color="red", label="best so far")
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("CV macro F1"); ax[0].set_title("Optuna optimisation history"); ax[0].legend()
    try:
        imp = optuna.importance.get_param_importances(study)
        ax[1].barh(list(imp)[::-1], list(imp.values())[::-1]); ax[1].set_title("Hyperparameter importance (fANOVA)")
    except Exception as e:
        ax[1].text(0.1, 0.5, f"importance unavailable: {e}")
    plt.tight_layout(); plt.savefig(FIG_DIR / "bo_history_importance.png", dpi=140); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, (c, t) in zip(ax, (("cv_macro_f1_mean", "CV macro F1"), ("student_macro_f1", "Student macro F1"),
                              ("student_top1", "Student Top-1"))):
        v = cmp[c].values; a.bar(["untuned", "tuned"], v, color=["tab:gray", "tab:green"]); a.set_title(t)
        a.set_ylim(min(v) - 0.01, max(v) + 0.01)
        for i, x in enumerate(v):
            a.text(i, x, f"{x:.4f}", ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(FIG_DIR / "bo_tuned_vs_untuned.png", dpi=140); plt.close()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
