"""Tune the HMM decoder and compare classifier-only vs classifier+HMM.
Run: python -m hmm.run_hmm [n_trials] [model_tag]     (model_tag default XGBoost_tuned)

Decoder parameters are tuned on TRAINING trips only (out-of-fold probabilities, so the classifier never
scored a trip it was trained on); the 10 student-test trips are used for the development-stage comparison.
Writes results/hmm_params.json, hmm_tuning_trials.csv, hmm_comparison_per_trip.csv, hmm_comparison.csv."""
import json
import sys
import time

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from hmm.viterbi import HMMParams, TripGraph, candidates_from_frame, viterbi_decode
from src.config import PROCESSED_DIR, RESULTS_DIR, SEED
from src.features import FEATURES
from src.io import load_trip
from training.common import arcs_of, load_split

MODEL_DIR = PROCESSED_DIR / "models"
TUNE_TRIPS = 20


def oof_probabilities(tr, params):
    """Out-of-fold probabilities for the training trips (same GroupKFold as model comparison)."""
    path = MODEL_DIR / "oof_XGBoost_tuned.parquet"
    if path.exists():
        return pd.read_parquet(path)
    parts = []
    for a, b in GroupKFold(n_splits=5).split(tr, groups=tr["trip"].values):
        m = xgb.XGBClassifier(tree_method="hist", random_state=SEED, n_jobs=-1, eval_metric="logloss", **params)
        m.fit(tr.iloc[a][FEATURES].values, tr.iloc[a]["y"].values)
        d = tr.iloc[b][["trip", "obs", "arc", "rank", "frac", "y"]].copy()
        d["p"] = m.predict_proba(tr.iloc[b][FEATURES].values)[:, 1]
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(path)
    return out


class TripData:
    def __init__(self, part, trip, cands, labels):
        t = load_trip(part, trip, with_route=False)
        self.trip = trip
        self.graph = TripGraph(t.nodes[["x", "y"]].values, t.arcs.u.values, t.arcs.v.values)
        self.xy = t.track[["x", "y"]].values
        self.t = t.track["t"].values
        self.n = len(self.xy)
        self.cands = candidates_from_frame(cands, self.n)
        self.label = labels.sort_values("obs")["label_arc"].values
        # classifier-only prediction = argmax probability per observation
        self.clf = np.full(self.n, -1, dtype=np.int64)
        for i, (a, f, p) in enumerate(self.cands):
            if len(a):
                self.clf[i] = a[int(np.argmax(p))]
        self.ref_set = set(self.label.tolist())


def continuity(pred, graph):
    ok = tot = 0
    a = pred[pred >= 0]
    if len(a) < 2:
        return 0, 0
    good = (a[:-1] == a[1:]) | (graph.v[a[:-1]] == graph.u[a[1:]])
    return int(good.sum()), len(good)


def metrics(pred, d):
    ok, tot = continuity(pred, d.graph)
    ps, rs = set(pred[pred >= 0].tolist()), d.ref_set
    inter = len(ps & rs)
    prec, rec = inter / max(len(ps), 1), inter / max(len(rs), 1)
    return dict(top1=float((pred == d.label).mean()), cont_ok=ok, cont_tot=tot,
                arc_precision=prec, arc_recall=rec,
                arc_f1=2 * prec * rec / max(prec + rec, 1e-9))


def evaluate_set(datas, params):
    rows = []
    for d in datas:
        cands = [(a.copy(), f.copy(), p.copy()) for a, f, p in d.cands]   # decoder mutates nothing, copy to be safe
        t0 = time.time()
        pred, breaks = viterbi_decode(d.graph, d.xy, d.t, cands, params)
        m = metrics(pred, d)
        m.update(trip=d.trip, n_obs=d.n, n_breaks=len(breaks), decode_s=time.time() - t0)
        c = metrics(d.clf, d)
        m.update({f"clf_{k}": v for k, v in c.items()})
        rows.append(m)
    return pd.DataFrame(rows)


def summarise(df):
    n = df.n_obs.sum()
    out = dict(trips=len(df), obs=int(n),
               clf_top1=(df.clf_top1 * df.n_obs).sum() / n, hmm_top1=(df.top1 * df.n_obs).sum() / n,
               clf_continuity=df.clf_cont_ok.sum() / max(df.clf_cont_tot.sum(), 1),
               hmm_continuity=df.cont_ok.sum() / max(df.cont_tot.sum(), 1),
               clf_arc_f1=df.clf_arc_f1.mean(), hmm_arc_f1=df.arc_f1.mean(),
               hmm_better_trips=int((df.top1 > df.clf_top1).sum()), hmm_worse_trips=int((df.top1 < df.clf_top1).sum()),
               hmm_equal_trips=int((df.top1 == df.clf_top1).sum()))
    return out


def main(n_trials=30, tag="XGBoost_tuned"):
    tr = load_split("train")
    st = load_split("student_test", usable_only=False)
    best_clf = json.loads((RESULTS_DIR / "bo_best_params.json").read_text())["best_params"]
    oof = oof_probabilities(tr, best_clf)

    lab_tr = pd.read_parquet(PROCESSED_DIR / "train_labels.parquet")
    lab_st = pd.read_parquet(PROCESSED_DIR / "student_test_labels.parquet")
    st_pred = pd.read_parquet(MODEL_DIR / f"preds_{tag}.parquet").merge(
        st[["trip", "obs", "arc", "rank", "frac"]], on=["trip", "obs", "arc"])

    rng = np.random.default_rng(SEED)
    trips_tr = sorted(oof.trip.unique())
    tune_ids = list(rng.choice(trips_tr, size=min(TUNE_TRIPS, len(trips_tr)), replace=False))
    print("tuning trips:", tune_ids, flush=True)
    tune = [TripData("train", t, oof[oof.trip == t], lab_tr[lab_tr.trip == t]) for t in tune_ids]

    def objective(trial):
        p = HMMParams(alpha=trial.suggest_float("alpha", 0.2, 3.0, log=True),
                      beta=trial.suggest_float("beta", 3.0, 100.0, log=True),
                      unreach_penalty=trial.suggest_float("unreach_penalty", 2.0, 40.0),
                      back_penalty=trial.suggest_float("back_penalty", 0.0, 20.0),
                      max_dt=trial.suggest_float("max_dt", 3.0, 30.0))
        df = evaluate_set(tune, p)
        s = summarise(df)
        trial.set_user_attr("clf_top1", s["clf_top1"]); trial.set_user_attr("hmm_continuity", s["hmm_continuity"])
        print(f"trial {trial.number:02d} hmm_top1={s['hmm_top1']:.4f} (clf {s['clf_top1']:.4f}) "
              f"cont={s['hmm_continuity']:.4f} {p}", flush=True)
        return s["hmm_top1"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial(dict(alpha=1.0, beta=30.0, unreach_penalty=15.0, back_penalty=5.0, max_dt=10.0))
    study.optimize(objective, n_trials=n_trials)
    study.trials_dataframe().to_csv(RESULTS_DIR / "hmm_tuning_trials.csv", index=False)
    best = study.best_params
    (RESULTS_DIR / "hmm_params.json").write_text(json.dumps(
        {"best_params": best, "tuning_objective": "mean pointwise Top-1 of HMM output on 20 training trips (OOF probabilities)",
         "tuning_trips": tune_ids, "n_trials": n_trials,
         "fixed": {"max_jump": 250.0, "search_radius": 600.0}}, indent=2))
    params = HMMParams(**best)

    # full comparison: all usable training trips (OOF) and the 10 student-test trips (final-model probabilities)
    res = {}
    for name, part, preds, labels in (("train_oof", "train", oof, lab_tr), ("student_test", "student_test", st_pred, lab_st)):
        ids = sorted(preds.trip.unique())
        datas = [TripData(part, t, preds[preds.trip == t], labels[labels.trip == t]) for t in ids]
        df = evaluate_set(datas, params)
        df.insert(0, "partition", name)
        res[name] = df
    per = pd.concat(res.values(), ignore_index=True)
    per.to_csv(RESULTS_DIR / "hmm_comparison_per_trip.csv", index=False)
    summ = pd.DataFrame([dict(partition=k, **summarise(v)) for k, v in res.items()])
    summ.to_csv(RESULTS_DIR / "hmm_comparison.csv", index=False)
    print(summ.round(4).T.to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30, sys.argv[2] if len(sys.argv) > 2 else "XGBoost_tuned")
