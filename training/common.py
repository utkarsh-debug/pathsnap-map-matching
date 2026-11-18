"""Shared data loading and metrics for model experiments.

Two metric families are reported (the PDFs ask for both):
  candidate-level : every (GPS observation, candidate arc) row is a binary example "is this the reference arc".
                    -> cand_macro_f1 / precision / recall / accuracy / log loss
  observation-level: one decision per GPS observation (the highest-scoring candidate vs the reference arc).
                    -> top1, obs_macro_f1 / precision / recall (classes = reference road segments),
                       obs_log_loss (-log of the normalised probability of the reference arc; only for
                       observations whose reference arc is inside the candidate set, coverage reported),
                       route_continuity.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

from src.config import DATA_DIR, PROCESSED_DIR, RESULTS_DIR


def load_split(part: str, usable_only: bool = False) -> pd.DataFrame:
    """All trips of a partition. The PDF says to use all 80 training trips, so usable_only defaults to False;
    usable_only=True (train only) drops the 2 trips whose reference route does not match the GPS track
    (used only for the sensitivity analysis, see training/sensitivity_exclusion.py)."""
    df = pd.read_parquet(PROCESSED_DIR / f"{part}_features.parquet")
    if usable_only and part == "train":
        st = pd.read_csv(RESULTS_DIR / "candidate_stats.csv", dtype={"trip": str})
        ok = set(st[(st.partition == "train") & st.usable_for_training].trip)
        df = df[df.trip.isin(ok)]
    return df.sort_values(["trip", "obs", "rank"]).reset_index(drop=True)


def unreliable_trips() -> set:
    """Training trips whose reference route does not describe the GPS track (>50% of reference arcs outside
    the candidate set, see results/candidate_stats.csv). Used ONLY to report a second, noise-free view of
    the CV metrics and for the sensitivity analysis; they are still part of training and the main CV."""
    st = pd.read_csv(RESULTS_DIR / "candidate_stats.csv", dtype={"trip": str})
    return set(st[(st.partition == "train") & ~st.usable_for_training].trip)


_ARCS = {}


def arcs_of(trip: str):
    if trip not in _ARCS:
        for part in ("train", "student_test", "evaluator_test"):
            p = DATA_DIR / part / trip / f"{trip}.arcs"
            if p.exists():
                a = pd.read_csv(p, sep=r"\s+", header=None).values
                _ARCS[trip] = (a[:, 0], a[:, 1])
                break
    return _ARCS[trip]


def top1_frame(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Highest-scoring candidate per observation: trip, obs, arc, label_arc, y."""
    d = df[["trip", "obs", "arc", "label_arc", "y"]].copy()
    d["p"] = p
    idx = d.groupby(["trip", "obs"], sort=False)["p"].idxmax()
    return d.loc[idx].reset_index(drop=True)


def route_continuity(pred: pd.DataFrame) -> float:
    """Share of consecutive predicted observations whose arcs are identical or connected
    (head node of arc_i == tail node of arc_i+1)."""
    ok = tot = 0
    for trip, g in pred.groupby("trip", sort=False):
        u, v = arcs_of(trip)
        a = g.sort_values("obs")["arc"].values
        if len(a) < 2:
            continue
        good = (a[:-1] == a[1:]) | (v[a[:-1]] == u[a[1:]])
        ok += int(good.sum()); tot += len(good)
    return ok / max(tot, 1)


def _macro_prf(true_key, pred_key):
    """Macro precision/recall/F1 over the union of reference and predicted segments (classes are
    (trip, arc) pairs because arc ids are only unique inside one trip's network)."""
    classes = np.union1d(true_key, pred_key)
    t = np.searchsorted(classes, true_key)
    pr = np.searchsorted(classes, pred_key)
    k = len(classes)
    true_cnt = np.bincount(t, minlength=k).astype(float)
    pred_cnt = np.bincount(pr, minlength=k).astype(float)
    tp = np.bincount(t[t == pr], minlength=k).astype(float)
    prec = np.divide(tp, pred_cnt, out=np.zeros(k), where=pred_cnt > 0)
    rec = np.divide(tp, true_cnt, out=np.zeros(k), where=true_cnt > 0)
    f1 = np.divide(2 * tp, pred_cnt + true_cnt, out=np.zeros(k), where=(pred_cnt + true_cnt) > 0)
    return prec.mean(), rec.mean(), f1.mean()


def evaluate(df: pd.DataFrame, p: np.ndarray) -> dict:
    y = df["y"].values
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    yhat = (p >= 0.5).astype(int)
    out = dict(
        cand_macro_f1=f1_score(y, yhat, average="macro"),
        cand_macro_precision=precision_score(y, yhat, average="macro", zero_division=0),
        cand_macro_recall=recall_score(y, yhat, average="macro", zero_division=0),
        cand_accuracy=float((y == yhat).mean()),
        cand_log_loss=log_loss(y, p, labels=[0, 1]),
    )
    top = top1_frame(df, p)
    out["top1"] = float((top["arc"] == top["label_arc"]).mean())   # misses count as wrong
    tcode = pd.factorize(top["trip"])[0].astype(np.int64) * 10 ** 8
    prec, rec, f1 = _macro_prf(tcode + top["label_arc"].values, tcode + top["arc"].values)
    out.update(obs_macro_precision=prec, obs_macro_recall=rec, obs_macro_f1=f1)
    # observation-level log loss on the probability of the reference arc (normalised over the candidates)
    d = df[["trip", "obs", "y"]].copy()
    d["p"] = p
    q = d["p"] / d.groupby(["trip", "obs"], sort=False)["p"].transform("sum")
    ref = d["y"].values == 1
    out["obs_log_loss"] = float(-np.log(np.clip(q[ref].values, 1e-7, 1.0)).mean()) if ref.any() else float("nan")
    out["ref_in_candidates"] = float(ref.sum() / len(top))          # coverage of the log loss (1 - miss rate)
    out["route_continuity"] = route_continuity(top)
    out["n_obs"] = len(top)
    out["n_rows"] = len(df)
    return out


def sample_obs(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Subsample whole observations (keeps all candidates of a sampled GPS point)."""
    if frac >= 1:
        return df
    key = df["trip"] + "_" + df["obs"].astype(str)
    uniq = key.unique()
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(uniq, size=int(len(uniq) * frac), replace=False))
    return df[key.isin(keep)]
