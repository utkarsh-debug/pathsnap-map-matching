"""Train + compare models. Run: python -m training.train_models [Model ...]
Group-aware (by trip) 5-fold CV on ALL 80 training trips, then refit on all 80 and evaluate on the 10
student-test trips. Every model uses 100% of the training rows. Seed 42.
Writes results/model_comparison.csv, results/cv_folds.csv, results/histories/*.csv,
data/processed/models/*.joblib + preds_*.parquet."""
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
from training.common import evaluate, load_split, sample_obs, unreliable_trips
from training.models import MODELS

BAD = unreliable_trips()

MODEL_DIR = PROCESSED_DIR / "models"
HIST_DIR = RESULTS_DIR / "histories"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
HIST_DIR.mkdir(exist_ok=True, parents=True)
N_SPLITS = 5
TRAIN_EVAL_FRAC = 0.03      # share of the in-fold training observations used to measure train-side metrics


def fit(name, X, y, Xv=None, yv=None):
    m = MODELS[name]["make"]()
    t0 = time.time()
    if name == "LightGBM" and Xv is not None:
        m.fit(X, y, eval_set=[(X, y), (Xv, yv)], eval_names=["train", "val"], eval_metric="binary_logloss")
    elif name == "XGBoost" and Xv is not None:
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
    """Real epoch-wise train/validation log-loss for the MLP via partial_fit (CV fold 0)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xv)
    net = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3,
                        batch_size=512, random_state=SEED)
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(A), size=min(len(A), 300000), replace=False)    # train-loss estimate on a fixed subset
    rows = []
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(A))
        for s in range(0, len(A), 512):
            b = idx[s:s + 512]
            net.partial_fit(A[b], ytr[b], classes=[0, 1])
        ll = lambda yy, pp: log_loss(yy, np.clip(pp, 1e-7, 1 - 1e-7), labels=[0, 1])
        rows.append(dict(epoch=ep, train_loss=ll(ytr[sub], net.predict_proba(A[sub])[:, 1]),
                         val_loss=ll(yv, net.predict_proba(B)[:, 1])))
    return pd.DataFrame(rows)


def main(names):
    tr = load_split("train")
    st = load_split("student_test")
    folds = list(GroupKFold(n_splits=N_SPLITS).split(tr, groups=tr["trip"].values))
    print(f"train rows {len(tr)} trips {tr.trip.nunique()} | student rows {len(st)} trips {st.trip.nunique()}", flush=True)

    comp_path, fold_path = RESULTS_DIR / "model_comparison.csv", RESULTS_DIR / "cv_folds.csv"
    comp = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()
    foldrows = pd.read_csv(fold_path) if fold_path.exists() else pd.DataFrame()

    for name in names:
        spec = MODELS[name]
        print(f"\n=== {name} ({spec['family']}) ===", flush=True)
        fm = []
        for k, (a, b) in enumerate(folds):
            dtr, dva = tr.iloc[a], tr.iloc[b]
            Xtr, ytr = dtr[FEATURES].values, dtr["y"].values
            Xv, yv = dva[FEATURES].values, dva["y"].values
            m, tt = fit(name, Xtr, ytr, Xv, yv)
            t0 = time.time(); p = m.predict_proba(Xv)[:, 1]; ti = time.time() - t0
            r = evaluate(dva, p)
            # second view: same fold, excluding the trips whose reference route does not match the track
            keep = ~dva["trip"].isin(BAD).values
            if keep.all():
                rr = r
            else:
                rr = evaluate(dva[keep], p[keep])
            r.update({f"rel_{c}": rr[c] for c in ("top1", "obs_macro_f1", "obs_log_loss", "cand_macro_f1", "cand_log_loss")})
            r.update(model=name, fold=k, train_time_s=tt, infer_time_s=ti, train_rows=len(dtr))
            # train-side metrics on a sample of the in-fold training observations (overfitting evidence)
            dts = sample_obs(dtr, TRAIN_EVAL_FRAC, SEED + k)
            rt = evaluate(dts, m.predict_proba(dts[FEATURES].values)[:, 1])
            r.update(train_top1=rt["top1"], train_obs_macro_f1=rt["obs_macro_f1"],
                     train_cand_macro_f1=rt["cand_macro_f1"], train_cand_log_loss=rt["cand_log_loss"],
                     train_obs_log_loss=rt["obs_log_loss"])
            fm.append(r)
            print(f"  fold{k}: top1={r['top1']:.4f} (reliable-trips {r['rel_top1']:.4f}, train {rt['top1']:.4f}) obsF1={r['obs_macro_f1']:.4f} "
                  f"obsLL={r['obs_log_loss']:.4f} candF1={r['cand_macro_f1']:.4f} fit {tt:.0f}s pred {ti:.0f}s", flush=True)
            if k == 0:
                h = mlp_history(Xtr, ytr, Xv, yv) if name == "MLP" else history(name, m)
                if h is not None:
                    h.to_csv(HIST_DIR / f"{name}_history.csv", index=False)
        fdf = pd.DataFrame(fm)

        # final refit on all 80 training trips -> student-test trips
        m, tt = fit(name, tr[FEATURES].values, tr["y"].values)
        t0 = time.time(); p = m.predict_proba(st[FEATURES].values)[:, 1]; ti = time.time() - t0
        s = evaluate(st, p)
        print(f"  STUDENT: top1={s['top1']:.4f} obsF1={s['obs_macro_f1']:.4f} obsLL={s['obs_log_loss']:.4f} "
              f"candF1={s['cand_macro_f1']:.4f} candLL={s['cand_log_loss']:.4f} cont={s['route_continuity']:.4f} "
              f"fit {tt:.0f}s infer {ti:.1f}s", flush=True)
        joblib.dump(m, MODEL_DIR / f"{name}.joblib")
        pd.DataFrame({"trip": st.trip.values, "obs": st.obs.values, "arc": st.arc.values,
                      "y": st.y.values, "p": p}).to_parquet(MODEL_DIR / f"preds_{name}.parquet")

        mean = lambda c: float(fdf[c].mean())
        row = dict(model=name, family=spec["family"], config=spec["config"], train_frac=spec["train_frac"],
                   cv_top1_mean=mean("top1"), cv_top1_std=float(fdf.top1.std()),
                   cv_obs_macro_f1_mean=mean("obs_macro_f1"), cv_obs_macro_f1_std=float(fdf.obs_macro_f1.std()),
                   cv_obs_macro_precision_mean=mean("obs_macro_precision"), cv_obs_macro_recall_mean=mean("obs_macro_recall"),
                   cv_obs_log_loss_mean=mean("obs_log_loss"),
                   cv_cand_macro_f1_mean=mean("cand_macro_f1"), cv_cand_macro_f1_std=float(fdf.cand_macro_f1.std()),
                   cv_cand_macro_precision_mean=mean("cand_macro_precision"), cv_cand_macro_recall_mean=mean("cand_macro_recall"),
                   cv_cand_log_loss_mean=mean("cand_log_loss"), cv_route_continuity_mean=mean("route_continuity"),
                   cv_rel_top1_mean=mean("rel_top1"), cv_rel_top1_std=float(fdf.rel_top1.std()),
                   cv_rel_obs_macro_f1_mean=mean("rel_obs_macro_f1"), cv_rel_obs_log_loss_mean=mean("rel_obs_log_loss"),
                   cv_rel_cand_macro_f1_mean=mean("rel_cand_macro_f1"), cv_rel_cand_log_loss_mean=mean("rel_cand_log_loss"),
                   cv_train_top1_mean=mean("train_top1"), cv_train_cand_log_loss_mean=mean("train_cand_log_loss"),
                   cv_train_obs_log_loss_mean=mean("train_obs_log_loss"),
                   student_top1=s["top1"], student_obs_macro_f1=s["obs_macro_f1"],
                   student_obs_macro_precision=s["obs_macro_precision"], student_obs_macro_recall=s["obs_macro_recall"],
                   student_obs_log_loss=s["obs_log_loss"], student_ref_in_candidates=s["ref_in_candidates"],
                   student_cand_macro_f1=s["cand_macro_f1"], student_cand_macro_precision=s["cand_macro_precision"],
                   student_cand_macro_recall=s["cand_macro_recall"], student_cand_accuracy=s["cand_accuracy"],
                   student_cand_log_loss=s["cand_log_loss"], student_route_continuity=s["route_continuity"],
                   train_time_s=tt, infer_time_s=ti, infer_ms_per_obs=ti / s["n_obs"] * 1000, seed=SEED)
        comp = comp[comp.model != name] if len(comp) else comp
        comp = pd.concat([comp, pd.DataFrame([row])], ignore_index=True)
        foldrows = foldrows[foldrows.model != name] if len(foldrows) else foldrows
        foldrows = pd.concat([foldrows, fdf], ignore_index=True)
        comp.to_csv(comp_path, index=False)
        foldrows.to_csv(fold_path, index=False)

    (RESULTS_DIR / "logs" / "train_models_config.json").write_text(
        json.dumps({n: {"family": MODELS[n]["family"], "config": MODELS[n]["config"],
                        "train_frac": MODELS[n]["train_frac"]} for n in MODELS}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
