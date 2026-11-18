"""Train + compare models. Run: python -m training.train_models [Model ...]
Group-aware (by trip) 5-fold CV on the usable training trips, then refit on all of them and evaluate
on the 10 student-test trips. Writes results/model_comparison.csv, results/cv_folds.csv,
results/histories/*.csv, and data/processed/models/*.joblib + preds_*.parquet."""
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from src.config import PROCESSED_DIR, RESULTS_DIR, SEED
from src.features import FEATURES
from training.common import evaluate, load_split, sample_obs
from training.models import MODELS

MODEL_DIR = PROCESSED_DIR / "models"
HIST_DIR = RESULTS_DIR / "histories"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
HIST_DIR.mkdir(exist_ok=True, parents=True)
N_SPLITS = 5


def fit(name, X, y, Xv=None, yv=None):
    spec = MODELS[name]
    m = spec["make"]()
    t0 = time.time()
    if name in ("LightGBM", "XGBoost") and Xv is not None:
        if name == "LightGBM":
            m.fit(X, y, eval_set=[(X, y), (Xv, yv)], eval_names=["train", "val"], eval_metric="binary_logloss")
        else:
            m.fit(X, y, eval_set=[(X, y), (Xv, yv)], verbose=False)
    else:
        m.fit(X, y)
    return m, time.time() - t0


def history(name, m):
    if name == "LightGBM":
        r = m.evals_result_
        return pd.DataFrame({"iteration": range(1, len(r["train"]["binary_logloss"]) + 1),
                             "train_loss": r["train"]["binary_logloss"], "val_loss": r["val"]["binary_logloss"]})
    if name == "XGBoost":
        r = m.evals_result()
        return pd.DataFrame({"iteration": range(1, len(r["validation_0"]["logloss"]) + 1),
                             "train_loss": r["validation_0"]["logloss"], "val_loss": r["validation_1"]["logloss"]})
    return None


def mlp_history(Xtr, ytr, Xv, yv, epochs=30):
    """Real epoch-wise train/validation log-loss for the MLP via partial_fit."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xv)
    net = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3,
                        batch_size=512, random_state=SEED)
    rng = np.random.default_rng(SEED)
    rows = []
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(A))
        for s in range(0, len(A), 512):
            b = idx[s:s + 512]
            net.partial_fit(A[b], ytr[b], classes=[0, 1])
        rows.append(dict(epoch=ep, train_loss=log_loss(ytr, np.clip(net.predict_proba(A)[:, 1], 1e-7, 1 - 1e-7), labels=[0, 1]),
                         val_loss=log_loss(yv, np.clip(net.predict_proba(B)[:, 1], 1e-7, 1 - 1e-7), labels=[0, 1])))
    return pd.DataFrame(rows)


def main(names):
    tr = load_split("train")
    st = load_split("student_test", usable_only=False)
    groups = tr["trip"].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    folds = list(gkf.split(tr, groups=groups))
    print(f"train rows {len(tr)} trips {tr.trip.nunique()} | student rows {len(st)} trips {st.trip.nunique()}", flush=True)

    comp_path = RESULTS_DIR / "model_comparison.csv"
    fold_path = RESULTS_DIR / "cv_folds.csv"
    comp = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()
    foldrows = pd.read_csv(fold_path) if fold_path.exists() else pd.DataFrame()

    for name in names:
        spec = MODELS[name]
        print(f"\n=== {name} ({spec['family']}) ===", flush=True)
        fm = []
        for k, (a, b) in enumerate(folds):
            dtr = sample_obs(tr.iloc[a], spec["train_frac"], SEED + k)
            dva = tr.iloc[b]
            Xtr, ytr = dtr[FEATURES].values, dtr["y"].values
            Xv, yv = dva[FEATURES].values, dva["y"].values
            m, tt = fit(name, Xtr, ytr, Xv, yv)
            t0 = time.time(); p = m.predict_proba(Xv)[:, 1]; ti = time.time() - t0
            r = evaluate(dva, p)
            r.update(model=name, fold=k, train_time_s=tt, infer_time_s=ti, train_rows=len(dtr))
            # overfitting evidence: score on a slice of the training data of this fold
            ptr = m.predict_proba(Xtr[:200000])[:, 1]
            r["train_log_loss"] = log_loss(ytr[:200000], np.clip(ptr, 1e-7, 1 - 1e-7), labels=[0, 1])
            fm.append(r)
            print(f"  fold{k}: F1={r['macro_f1']:.4f} top1={r['top1']:.4f} ll={r['log_loss']:.4f} "
                  f"(train ll {r['train_log_loss']:.4f}) fit {tt:.0f}s", flush=True)
            if k == 0:
                h = history(name, m)
                if name == "MLP":
                    h = mlp_history(Xtr, ytr, Xv, yv)
                if h is not None:
                    h.to_csv(HIST_DIR / f"{name}_history.csv", index=False)
        fdf = pd.DataFrame(fm)

        # final refit on all usable training trips -> student test
        dall = sample_obs(tr, spec["train_frac"], SEED)
        m, tt = fit(name, dall[FEATURES].values, dall["y"].values)
        t0 = time.time(); p = m.predict_proba(st[FEATURES].values)[:, 1]; ti = time.time() - t0
        s = evaluate(st, p)
        print(f"  STUDENT: F1={s['macro_f1']:.4f} top1={s['top1']:.4f} ll={s['log_loss']:.4f} "
              f"cont={s['route_continuity']:.4f} fit {tt:.0f}s infer {ti:.1f}s", flush=True)
        joblib.dump(m, MODEL_DIR / f"{name}.joblib")
        pd.DataFrame({"trip": st.trip.values, "obs": st.obs.values, "arc": st.arc.values,
                      "y": st.y.values, "p": p}).to_parquet(MODEL_DIR / f"preds_{name}.parquet")

        row = dict(model=name, family=spec["family"], config=spec["config"], train_frac=spec["train_frac"],
                   cv_macro_f1_mean=fdf.macro_f1.mean(), cv_macro_f1_std=fdf.macro_f1.std(),
                   cv_top1_mean=fdf.top1.mean(), cv_log_loss_mean=fdf.log_loss.mean(),
                   cv_route_continuity_mean=fdf.route_continuity.mean(),
                   cv_train_log_loss_mean=fdf.train_log_loss.mean(),
                   student_macro_f1=s["macro_f1"], student_macro_precision=s["macro_precision"],
                   student_macro_recall=s["macro_recall"], student_accuracy=s["accuracy"],
                   student_top1=s["top1"], student_log_loss=s["log_loss"],
                   student_route_continuity=s["route_continuity"],
                   train_time_s=tt, infer_time_s=ti, infer_ms_per_obs=ti / s["n_obs"] * 1000, seed=SEED)
        comp = comp[comp.model != name] if len(comp) else comp
        comp = pd.concat([comp, pd.DataFrame([row])], ignore_index=True)
        foldrows = foldrows[foldrows.model != name] if len(foldrows) else foldrows
        foldrows = pd.concat([foldrows, fdf], ignore_index=True)
        comp.to_csv(comp_path, index=False)
        foldrows.to_csv(fold_path, index=False)

    # confusion matrices come from the saved student-test predictions (viz/plots_models.py)
    (RESULTS_DIR / "logs" / "train_models_config.json").write_text(
        json.dumps({n: {"family": MODELS[n]["family"], "config": MODELS[n]["config"],
                        "train_frac": MODELS[n]["train_frac"]} for n in MODELS}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
