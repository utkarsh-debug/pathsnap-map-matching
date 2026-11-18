"""Bayesian hyperparameter optimisation (Optuna TPE) of the model chosen in results/model_selection.json.

Objective : maximise mean CV observation-level Top-1 (the primary selection metric) over the SAME 5 trip-grouped
            CV folds as the model comparison, on ALL 80 training trips. Other metrics are recorded per trial.
Data      : training trips only for the search; the 10 student-test trips are used once, for the final
            tuned-vs-untuned check. No evaluator data is touched.
Run       : python -m training.bayes_opt [n_trials]
Outputs   : results/bo_trials.csv, bo_best_params.json, bo_search_space.json, tuned_vs_untuned.csv,
            figures/bo_*.png, data/processed/models/<Model>_tuned.joblib + preds_<Model>_tuned.parquet
"""
import json
import sys
import time

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from src.config import FIG_DIR, PROCESSED_DIR, RESULTS_DIR, SEED
from src.features import FEATURES
from training.common import evaluate, load_split, unreliable_trips
from training.models import MODELS

N_SPLITS = 5
BAD = unreliable_trips()

# name -> (search space description, suggest function, builder for tuned params)
SPACES = {
    "XGBoost": {
        "n_estimators": "int [100, 600]", "learning_rate": "float log-uniform [0.01, 0.3]",
        "max_depth": "int [3, 10]", "min_child_weight": "float log-uniform [1, 50]",
        "subsample": "float uniform [0.5, 1.0]", "colsample_bytree": "float uniform [0.5, 1.0]",
        "reg_lambda": "float log-uniform [0.01, 50]", "reg_alpha": "float log-uniform [0.001, 10]",
        "gamma": "float uniform [0, 5]"},
    "LightGBM": {
        "n_estimators": "int [100, 600]", "learning_rate": "float log-uniform [0.01, 0.3]",
        "num_leaves": "int [15, 255]", "min_child_samples": "int log-uniform [5, 200]",
        "subsample": "float uniform [0.5, 1.0] (subsample_freq=1)", "colsample_bytree": "float uniform [0.5, 1.0]",
        "reg_lambda": "float log-uniform [0.001, 50]", "reg_alpha": "float log-uniform [0.001, 10]"},
    "RandomForest": {
        "n_estimators": "int [80, 300]", "max_depth": "int [6, 30]",
        "min_samples_leaf": "int log-uniform [1, 40]", "max_features": "float uniform [0.3, 1.0]"},
}


# Why each range was chosen (written to results/bo_search_space.json; PDF asks for a justified search space)
RATIONALE = {
    "n_estimators": "number of boosting rounds/trees; the untuned model uses 300, so the range brackets it from under- to over-fitted",
    "learning_rate": "log scale because its effect is multiplicative; 0.05 (untuned) sits inside; small values need more trees",
    "max_depth": "tree depth controls interaction order; 6 (untuned) is inside; >10 overfits on ~1.9M rows of noisy labels",
    "min_child_weight": "minimum hessian per leaf: larger values regularise noisy-label regions (e.g. the 2 unreliable trips)",
    "subsample": "row subsampling per tree (decorrelates trees); below 0.5 loses too much signal",
    "colsample_bytree": "feature subsampling per tree; useful because several of the 19 features are correlated (distance family)",
    "reg_lambda": "L2 leaf-weight penalty, log scale; default is 1",
    "reg_alpha": "L1 leaf-weight penalty, log scale; small default effect, allowed to grow for sparsity",
    "gamma": "minimum loss reduction to split; prunes splits that only fit label noise",
    "num_leaves": "main complexity control for leaf-wise LightGBM; 31 (untuned) is inside",
    "min_child_samples": "minimum rows per leaf, log scale; regularises against noisy labels",
    "max_features": "fraction of features tried per split (random-forest decorrelation)",
    "min_samples_leaf": "minimum rows per leaf, log scale; regularises against noisy labels",
}


def suggest(name, t):
    if name == "XGBoost":
        return dict(n_estimators=t.suggest_int("n_estimators", 100, 600),
                    learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    max_depth=t.suggest_int("max_depth", 3, 10),
                    min_child_weight=t.suggest_float("min_child_weight", 1, 50, log=True),
                    subsample=t.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 0.01, 50, log=True),
                    reg_alpha=t.suggest_float("reg_alpha", 0.001, 10, log=True),
                    gamma=t.suggest_float("gamma", 0, 5))
    if name == "LightGBM":
        return dict(n_estimators=t.suggest_int("n_estimators", 100, 600),
                    learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    num_leaves=t.suggest_int("num_leaves", 15, 255),
                    min_child_samples=t.suggest_int("min_child_samples", 5, 200, log=True),
                    subsample=t.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 0.001, 50, log=True),
                    reg_alpha=t.suggest_float("reg_alpha", 0.001, 10, log=True))
    if name == "RandomForest":
        return dict(n_estimators=t.suggest_int("n_estimators", 80, 300),
                    max_depth=t.suggest_int("max_depth", 6, 30),
                    min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 40, log=True),
                    max_features=t.suggest_float("max_features", 0.3, 1.0))
    raise NotImplementedError(f"no search space defined for {name}")


def build(name, params):
    if params is None:                       # the untuned model, exactly as in the comparison
        return MODELS[name]["make"]()
    if name == "XGBoost":
        return xgb.XGBClassifier(tree_method="hist", random_state=SEED, n_jobs=-1, eval_metric="logloss", **params)
    if name == "LightGBM":
        return lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, subsample_freq=1, **params)
    if name == "RandomForest":
        return RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
    raise NotImplementedError(name)


def cv_eval(name, tr, folds, params):
    rows = []
    for a, b in folds:
        dtr, dva = tr.iloc[a], tr.iloc[b]
        m = build(name, params).fit(dtr[FEATURES].values, dtr["y"].values)
        p = m.predict_proba(dva[FEATURES].values)[:, 1]
        r = evaluate(dva, p)
        keep = ~dva["trip"].isin(BAD).values
        r["rel_top1"] = r["top1"] if keep.all() else evaluate(dva[keep], p[keep])["top1"]
        rows.append(r)
    return pd.DataFrame(rows)


def main(n_trials):
    name = json.loads((RESULTS_DIR / "model_selection.json").read_text())["selected_model"]
    print("model selected for tuning:", name, flush=True)
    tr = load_split("train")
    st = load_split("student_test")
    folds = list(GroupKFold(n_splits=N_SPLITS).split(tr, groups=tr["trip"].values))

    def objective(trial):
        p = suggest(name, trial)
        t0 = time.time()
        r = cv_eval(name, tr, folds, p)
        for k, c in (("cv_top1_std", "top1"), ):
            trial.set_user_attr(k, float(r[c].std()))
        trial.set_user_attr("cv_rel_top1", float(r.rel_top1.mean()))
        trial.set_user_attr("cv_obs_macro_f1", float(r.obs_macro_f1.mean()))
        trial.set_user_attr("cv_obs_log_loss", float(r.obs_log_loss.mean()))
        trial.set_user_attr("cv_cand_macro_f1", float(r.cand_macro_f1.mean()))
        trial.set_user_attr("cv_cand_log_loss", float(r.cand_log_loss.mean()))
        trial.set_user_attr("cv_route_continuity", float(r.route_continuity.mean()))
        trial.set_user_attr("seconds", time.time() - t0)
        print(f"trial {trial.number:02d} top1={r.top1.mean():.4f} rel_top1={r.rel_top1.mean():.4f} "
              f"obsLL={r.obs_log_loss.mean():.4f} candF1={r.cand_macro_f1.mean():.4f} {time.time()-t0:.0f}s {p}", flush=True)
        return float(r.top1.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
                                study_name=f"{name}_pathsnap")
    t_start = time.time()
    study.optimize(objective, n_trials=n_trials)
    search_s = time.time() - t_start

    trials = study.trials_dataframe()
    trials.to_csv(RESULTS_DIR / "bo_trials.csv", index=False)
    best = study.best_params
    (RESULTS_DIR / "bo_best_params.json").write_text(json.dumps(
        {"model": name, "best_params": best, "best_cv_top1": study.best_value, "n_trials": n_trials,
         "sampler": f"TPE (seed {SEED}, 10 random start-up trials)", "search_seconds": search_s}, indent=2))
    (RESULTS_DIR / "bo_search_space.json").write_text(json.dumps(
        {"model": name, "objective": "maximise mean CV observation-level Top-1 over 5-fold GroupKFold (groups=trip) "
                                     "on all 80 training trips", "space": SPACES[name],
         "justification": {k: RATIONALE.get(k, "") for k in SPACES[name]},
         "n_trials": n_trials, "sampler": "TPE", "seed": SEED,
         "untuned_baseline_config": MODELS[name]["config"]}, indent=2))

    # tuned vs untuned: identical folds, metrics and partitions
    out = []
    for label, params in (("untuned", None), ("tuned", best)):
        r = cv_eval(name, tr, folds, params)
        t0 = time.time(); m = build(name, params).fit(tr[FEATURES].values, tr["y"].values); fit_s = time.time() - t0
        t0 = time.time(); p = m.predict_proba(st[FEATURES].values)[:, 1]; inf_s = time.time() - t0
        s = evaluate(st, p)
        out.append(dict(
            model=f"{name}_{label}", cv_top1=r.top1.mean(), cv_top1_std=r.top1.std(), cv_rel_top1=r.rel_top1.mean(),
            cv_obs_macro_f1=r.obs_macro_f1.mean(), cv_cand_macro_f1=r.cand_macro_f1.mean(),
            cv_obs_log_loss=r.obs_log_loss.mean(), cv_cand_log_loss=r.cand_log_loss.mean(),
            student_top1=s["top1"], student_obs_macro_f1=s["obs_macro_f1"], student_cand_macro_f1=s["cand_macro_f1"],
            student_obs_log_loss=s["obs_log_loss"], student_cand_log_loss=s["cand_log_loss"],
            student_route_continuity=s["route_continuity"], train_time_s=fit_s, infer_time_s=inf_s,
            infer_ms_per_obs=inf_s / s["n_obs"] * 1000, params=json.dumps(params) if params else MODELS[name]["config"]))
        if label == "tuned":
            joblib.dump(m, PROCESSED_DIR / "models" / f"{name}_tuned.joblib")
            pd.DataFrame({"trip": st.trip.values, "obs": st.obs.values, "arc": st.arc.values,
                          "y": st.y.values, "p": p}).to_parquet(PROCESSED_DIR / "models" / f"preds_{name}_tuned.parquet")
    cmp = pd.DataFrame(out)
    cmp.to_csv(RESULTS_DIR / "tuned_vs_untuned.csv", index=False)
    print(cmp.drop(columns="params").round(4).T.to_string())

    done = trials[trials.state == "COMPLETE"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(done.number, done.value, "o", alpha=0.6, label="trial")
    ax[0].plot(done.number, done.value.cummax(), "-", color="red", label="best so far")
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("CV Top-1"); ax[0].set_title(f"Optuna history ({name})"); ax[0].legend()
    try:
        imp = optuna.importance.get_param_importances(study)
        ax[1].barh(list(imp)[::-1], list(imp.values())[::-1]); ax[1].set_title("Hyperparameter importance (fANOVA)")
    except Exception as e:
        ax[1].text(0.1, 0.5, f"importance unavailable: {e}")
    plt.tight_layout(); plt.savefig(FIG_DIR / "bo_history_importance.png", dpi=140); plt.close()

    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    for a, (c, t) in zip(ax, (("cv_top1", "CV Top-1 (primary)"), ("student_top1", "Student Top-1"),
                              ("student_obs_macro_f1", "Student obs. macro F1"),
                              ("student_obs_log_loss", "Student obs. log loss (lower better)"))):
        v = cmp[c].values; a.bar(["untuned", "tuned"], v, color=["tab:gray", "tab:green"]); a.set_title(t)
        lo, hi = min(v), max(v); pad = max((hi - lo) * 0.5, 0.005)
        a.set_ylim(lo - pad, hi + pad)
        for i, xv in enumerate(v):
            a.text(i, xv, f"{xv:.4f}", ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(FIG_DIR / "bo_tuned_vs_untuned.png", dpi=140); plt.close()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
