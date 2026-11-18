"""Shared data loading and metrics for model experiments."""
import numpy as np
import pandas as pd

from src.config import DATA_DIR, PROCESSED_DIR, RESULTS_DIR
from src.features import FEATURES
from src.io import list_trips
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def load_split(part: str, usable_only: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / f"{part}_features.parquet")
    if usable_only and part == "train":
        st = pd.read_csv(RESULTS_DIR / "candidate_stats.csv", dtype={"trip": str})
        ok = set(st[(st.partition == "train") & st.usable_for_training].trip)
        df = df[df.trip.isin(ok)]
    return df.sort_values(["trip", "obs", "rank"]).reset_index(drop=True)


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


def predicted_arcs(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Top-1 candidate per observation. Returns trip, obs, arc, y (is-it-the-reference-arc)."""
    d = df[["trip", "obs", "arc", "y"]].copy()
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


def evaluate(df: pd.DataFrame, p: np.ndarray) -> dict:
    y = df["y"].values
    p = np.clip(p, 1e-7, 1 - 1e-7)
    yhat = (p >= 0.5).astype(int)
    pred = predicted_arcs(df, p)
    return dict(
        macro_f1=f1_score(y, yhat, average="macro"),
        macro_precision=precision_score(y, yhat, average="macro", zero_division=0),
        macro_recall=recall_score(y, yhat, average="macro", zero_division=0),
        accuracy=float((y == yhat).mean()),
        log_loss=log_loss(y, p, labels=[0, 1]),
        top1=float(pred["y"].mean()),                       # observation-level Top-1 (misses count wrong)
        route_continuity=route_continuity(pred),
        n_obs=len(pred), n_rows=len(df),
    )


def sample_obs(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Subsample whole observations (keeps all candidates of a sampled GPS point)."""
    if frac >= 1:
        return df
    key = df["trip"] + "_" + df["obs"].astype(str)
    uniq = key.unique()
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(uniq, size=int(len(uniq) * frac), replace=False))
    return df[key.isin(keep)]
