"""Tune the HMM decoder using training-trip out-of-fold predictions.

Run after `python -m src.build_dataset` has created the train and student-test
feature parquet files. The selected parameters are optimized on five-fold
GroupKFold out-of-fold predictions for the training trips. Student-test trips
are used only for the final classifier-only versus HMM comparison. Evaluator
data is never loaded by this script.

Outputs:
  results/hmm_trials.csv
  results/hmm_params.json
  results/hmm_comparison.csv
  api/artifacts/hmm_params.json
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from hmm.viterbi import HMMParams, TripGraph, candidates_from_frame, viterbi_decode
from src.candidates import arc_geometries, label_points
from src.config import ARTIFACT_DIR, DATA_DIR, PROCESSED_DIR, RESULTS_DIR, SEED
from src.features import FEATURES
from src.io import load_trip


N_SPLITS = 5
FIXED_PARAMS = {
    "max_jump": 250.0,
    "search_radius": 600.0,
}


@dataclass
class TripContext:
    trip_id: str
    graph: TripGraph
    track_xy: np.ndarray
    timestamps: np.ndarray
    candidates: list
    labels: np.ndarray
    classifier_prediction: np.ndarray
    candidate_miss_rate: float


def _pipeline_config() -> dict:
    path = ARTIFACT_DIR / "pipeline_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; export the selected model artifact first.")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("features") != FEATURES:
        raise ValueError("API artifact feature list does not match src/features.py")
    if not str(cfg.get("model_name", "")).startswith("XGBoost"):
        raise ValueError("HMM tuning currently expects the selected XGBoost model")
    candidate_cfg = cfg.get("candidates", {})
    if int(candidate_cfg.get("k", -1)) != 15 or float(candidate_cfg.get("radius_m", -1)) != 80.0:
        raise ValueError("tuning expects the checked-in 15-candidate, 80 m training feature configuration")
    return cfg


def _xgb_for_fold(cfg: dict) -> xgb.XGBClassifier:
    params = cfg.get("hyperparameters") or {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
    }
    return xgb.XGBClassifier(tree_method="hist", random_state=SEED, n_jobs=-1,
                             eval_metric="logloss", **params)


def _check_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_dirs = [part for part in ("train", "student_test") if not (DATA_DIR / part).is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            f"Raw partition(s) missing under {DATA_DIR}: {', '.join(missing_dirs)}. "
            "Set PATHSNAP_DATA to the dataset root containing train/ and student_test/."
        )
    missing_tables = [part for part in ("train", "student_test")
                      if not (PROCESSED_DIR / f"{part}_features.parquet").exists()]
    if missing_tables:
        raise FileNotFoundError(
            f"Missing processed feature table(s): {', '.join(missing_tables)}. "
            "Run `python -m src.build_dataset` with PATHSNAP_DATA set first."
        )
    train = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    student = pd.read_parquet(PROCESSED_DIR / "student_test_features.parquet")
    if train.trip.nunique() < N_SPLITS:
        raise ValueError(f"Need at least {N_SPLITS} training trips for GroupKFold")
    return train.sort_values(["trip", "obs", "rank"]).reset_index(drop=True), \
        student.sort_values(["trip", "obs", "rank"]).reset_index(drop=True)


def _out_of_fold_probabilities(train: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Fit the selected classifier on each training fold and predict held-out trips."""
    groups = train["trip"].astype(str).values
    folds = GroupKFold(n_splits=N_SPLITS).split(train, groups=groups)
    probabilities = np.full(len(train), np.nan, dtype=np.float32)
    for fold, (fit_rows, held_rows) in enumerate(folds):
        model = _xgb_for_fold(cfg)
        model.fit(train.iloc[fit_rows][FEATURES].values, train.iloc[fit_rows]["y"].values)
        probabilities[held_rows] = model.predict_proba(train.iloc[held_rows][FEATURES].values)[:, 1]
        print(f"OOF model fold {fold + 1}/{N_SPLITS}: "
              f"{train.iloc[held_rows].trip.nunique()} held-out trips", flush=True)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Some training candidate rows did not receive an out-of-fold prediction")
    return probabilities


def _context_from_frame(partition: str, trip_id: str, frame: pd.DataFrame,
                        probabilities: np.ndarray) -> TripContext:
    trip = load_trip(partition, trip_id, DATA_DIR, with_route=True)
    if trip.route is None:
        raise ValueError(f"{partition}/{trip_id} has no reference route")
    if len(frame) != len(probabilities):
        raise ValueError(f"candidate probability length mismatch for trip {trip_id}")
    candidates_frame = frame[["obs", "arc", "frac", "rank"]].copy()
    candidates_frame["p"] = probabilities
    candidates = candidates_from_frame(candidates_frame, len(trip.track))

    _, coords = arc_geometries(trip)
    labels, _ = label_points(trip, coords)
    classifier_prediction = np.full(len(trip.track), -1, dtype=np.int64)
    if len(candidates_frame):
        ranked = candidates_frame.assign(p=probabilities)
        best_rows = ranked.loc[ranked.groupby("obs", sort=False)["p"].idxmax()]
        obs = best_rows["obs"].to_numpy(dtype=np.int64)
        classifier_prediction[obs] = best_rows["arc"].to_numpy(dtype=np.int64)

    hits = np.zeros(len(trip.track), dtype=bool)
    for obs, (arcs, _fractions, _probabilities) in enumerate(candidates):
        hits[obs] = np.any(arcs == int(labels[obs]))
    graph = TripGraph(trip.nodes[["x", "y"]].to_numpy(), trip.arcs.u.values, trip.arcs.v.values)
    return TripContext(
        trip_id=str(trip_id), graph=graph,
        track_xy=trip.track[["x", "y"]].to_numpy(dtype=float),
        timestamps=trip.track["t"].to_numpy(dtype=float), candidates=candidates,
        labels=np.asarray(labels, dtype=np.int64), classifier_prediction=classifier_prediction,
        candidate_miss_rate=float(1.0 - hits.mean()) if len(hits) else float("nan"),
    )


def _training_contexts(train: pd.DataFrame, oof_probabilities: np.ndarray) -> list[TripContext]:
    with_probabilities = train.copy()
    with_probabilities["_p"] = oof_probabilities
    contexts = []
    for trip_id, frame in with_probabilities.groupby("trip", sort=True):
        contexts.append(_context_from_frame("train", str(trip_id), frame,
                                            frame["_p"].to_numpy(dtype=float)))
    return contexts


def _student_contexts(student: pd.DataFrame, cfg: dict) -> list[TripContext]:
    model_path = ARTIFACT_DIR / cfg["model_file"]
    if not model_path.exists():
        raise FileNotFoundError(f"Missing selected model artifact: {model_path}")
    model = joblib.load(model_path)
    probabilities = model.predict_proba(student[FEATURES].values)[:, 1]
    with_probabilities = student.copy()
    with_probabilities["_p"] = probabilities
    contexts = []
    for trip_id, frame in with_probabilities.groupby("trip", sort=True):
        contexts.append(_context_from_frame("student_test", str(trip_id), frame,
                                            frame["_p"].to_numpy(dtype=float)))
    return contexts


def _trip_metrics(context: TripContext, prediction: np.ndarray) -> dict:
    labels = context.labels
    prediction = np.asarray(prediction, dtype=np.int64)
    if len(prediction) != len(labels):
        raise ValueError(f"decoder returned the wrong number of states for trip {context.trip_id}")
    correct = int((prediction == labels).sum())
    predicted_arcs = set(prediction[prediction >= 0].tolist())
    reference_arcs = set(labels.tolist())
    intersection = len(predicted_arcs & reference_arcs)
    precision = intersection / max(len(predicted_arcs), 1)
    recall = intersection / max(len(reference_arcs), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    valid_arcs = prediction[prediction >= 0]
    continuity_pairs = max(len(valid_arcs) - 1, 0)
    continuity_good = 0
    if continuity_pairs:
        continuity_good = int(((valid_arcs[:-1] == valid_arcs[1:]) |
                               (context.graph.v[valid_arcs[:-1]] == context.graph.u[valid_arcs[1:]])).sum())
    return {
        "n_points": len(labels), "correct": correct,
        "exact_sequence": int(np.array_equal(prediction, labels)),
        "continuity_good": continuity_good, "continuity_pairs": continuity_pairs,
        "arc_precision": precision, "arc_recall": recall, "arc_f1": f1,
    }


def _evaluate(contexts: list[TripContext], params: HMMParams | None) -> dict:
    totals = {"n_points": 0, "correct": 0, "exact_sequences": 0,
              "continuity_good": 0, "continuity_pairs": 0, "candidate_misses": 0,
              "decoder_breaks": 0}
    trip_arc_metrics = []
    for context in contexts:
        if params is None:
            prediction = context.classifier_prediction
            breaks = []
        else:
            prediction, breaks = viterbi_decode(context.graph, context.track_xy, context.timestamps,
                                                context.candidates, params)
        row = _trip_metrics(context, prediction)
        totals["n_points"] += row["n_points"]
        totals["correct"] += row["correct"]
        totals["exact_sequences"] += row["exact_sequence"]
        totals["continuity_good"] += row["continuity_good"]
        totals["continuity_pairs"] += row["continuity_pairs"]
        totals["candidate_misses"] += context.candidate_miss_rate * row["n_points"]
        totals["decoder_breaks"] += len(breaks)
        trip_arc_metrics.append((row["arc_precision"], row["arc_recall"], row["arc_f1"]))

    n_trips = len(contexts)
    n_points = max(totals["n_points"], 1)
    n_pairs = max(totals["continuity_pairs"], 1)
    arc_metrics = np.asarray(trip_arc_metrics, dtype=float)
    return {
        "n_trips": n_trips,
        "n_points": totals["n_points"],
        "top1_accuracy": totals["correct"] / n_points,
        "trip_exact_sequence_accuracy": totals["exact_sequences"] / max(n_trips, 1),
        "route_continuity": totals["continuity_good"] / n_pairs,
        "route_arc_precision": float(arc_metrics[:, 0].mean()) if len(arc_metrics) else float("nan"),
        "route_arc_recall": float(arc_metrics[:, 1].mean()) if len(arc_metrics) else float("nan"),
        "route_arc_f1": float(arc_metrics[:, 2].mean()) if len(arc_metrics) else float("nan"),
        "candidate_miss_rate": totals["candidate_misses"] / n_points,
        "decoder_breaks_per_trip": totals["decoder_breaks"] / max(n_trips, 1),
    }


def _suggest(trial: optuna.Trial) -> dict:
    return {
        "alpha": trial.suggest_float("alpha", 0.2, 3.0, log=True),
        "beta": trial.suggest_float("beta", 3.0, 100.0, log=True),
        "unreach_penalty": trial.suggest_float("unreach_penalty", 2.0, 40.0),
        "back_penalty": trial.suggest_float("back_penalty", 0.0, 20.0),
        "max_dt": trial.suggest_float("max_dt", 3.0, 30.0),
    }


def main(n_trials: int = 40):
    cfg = _pipeline_config()
    train, student = _check_inputs()
    print(f"data root: {DATA_DIR}", flush=True)
    print(f"training: {train.trip.nunique()} trips / {len(train):,} candidate rows", flush=True)

    start = time.perf_counter()
    oof_probabilities = _out_of_fold_probabilities(train, cfg)
    train_contexts = _training_contexts(train, oof_probabilities)
    student_contexts = _student_contexts(student, cfg)
    print(f"prepared {len(train_contexts)} OOF training and {len(student_contexts)} student-test trips "
          f"in {time.perf_counter() - start:.1f}s", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
                                study_name="pathsnap_hmm_viterbi")

    def objective(trial: optuna.Trial) -> float:
        params = HMMParams(**_suggest(trial), **FIXED_PARAMS)
        metrics = _evaluate(train_contexts, params)
        trial.set_user_attr("route_continuity", metrics["route_continuity"])
        trial.set_user_attr("trip_exact_sequence_accuracy", metrics["trip_exact_sequence_accuracy"])
        trial.set_user_attr("route_arc_f1", metrics["route_arc_f1"])
        trial.set_user_attr("decoder_breaks_per_trip", metrics["decoder_breaks_per_trip"])
        print(f"trial {trial.number + 1}/{n_trials}: CV Top-1={metrics['top1_accuracy']:.4f}, "
              f"continuity={metrics['route_continuity']:.4f}, "
              f"exact trips={metrics['trip_exact_sequence_accuracy']:.4f}", flush=True)
        return float(metrics["top1_accuracy"])

    study.optimize(objective, n_trials=n_trials)
    best_params = dict(study.best_params)
    best_hmm = HMMParams(**best_params, **FIXED_PARAMS)
    train_classifier = _evaluate(train_contexts, None)
    train_hmm = _evaluate(train_contexts, best_hmm)
    student_classifier = _evaluate(student_contexts, None)
    student_hmm = _evaluate(student_contexts, best_hmm)

    payload = {
        "model": cfg["model_name"],
        "status": "tuned_on_training_oof",
        "objective": "maximize pooled observation-level Top-1 on 5-fold GroupKFold out-of-fold training predictions",
        "best_params": best_params,
        "fixed": FIXED_PARAMS,
        "best_cv_top1": float(study.best_value),
        "n_trials": int(n_trials),
        "sampler": f"TPE (seed {SEED}, 10 startup trials)",
        "training_seconds": round(time.perf_counter() - start, 2),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "hmm_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "hmm_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    study.trials_dataframe().to_csv(RESULTS_DIR / "hmm_trials.csv", index=False)

    comparison = []
    for partition, method, metrics in (
        ("train_oof", "classifier_only", train_classifier),
        ("train_oof", "hmm_viterbi", train_hmm),
        ("student_test", "classifier_only", student_classifier),
        ("student_test", "hmm_viterbi", student_hmm),
    ):
        comparison.append({"partition": partition, "method": method, **metrics})
    comparison_frame = pd.DataFrame(comparison)
    comparison_frame.to_csv(RESULTS_DIR / "hmm_comparison.csv", index=False)
    print(comparison_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)
    print(f"saved tuned parameters to {RESULTS_DIR / 'hmm_params.json'} and {ARTIFACT_DIR / 'hmm_params.json'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
