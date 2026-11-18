"""Inference pipeline: trip folder -> candidates -> features -> classifier -> (HMM/Viterbi) -> route.

The pipeline works ONLY from .track / .nodes / .arcs. The reference route (.route) is never loaded here
(load_trip(..., with_route=False)); it is used afterwards, server-side only, by api/scoring.py.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from src.candidates import build_index
from src.features import FEATURES, build_features
from src.io import load_trip
from .config import ARTIFACT_DIR, DATA_ROOT

log = logging.getLogger("pathsnap.api")


def _load_hmm():
    """HMM/Viterbi decoder is optional: if the module or its tuned parameters are missing the API
    falls back to classifier-only predictions and says so in every response."""
    try:
        from hmm.viterbi import HMMParams, TripGraph, candidates_from_frame, viterbi_decode
    except Exception as e:                                   # noqa: BLE001
        log.warning("HMM module not available (%s): classifier-only mode", e)
        return None
    p = ARTIFACT_DIR / "hmm_params.json"
    if not p.exists():
        log.warning("%s missing: classifier-only mode", p.name)
        return None
    cfg = json.loads(p.read_text())
    params = HMMParams(**cfg["best_params"], **cfg.get("fixed", {}))
    return dict(params=params, TripGraph=TripGraph, candidates_from_frame=candidates_from_frame,
                viterbi_decode=viterbi_decode)


@dataclass
class Internals:
    """Server-side only: never serialised into a response."""
    trip: object
    feats: object
    clf_pred: np.ndarray
    final_pred: np.ndarray
    hmm_pred: np.ndarray | None
    timings: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self):
        cfg = json.loads((ARTIFACT_DIR / "pipeline_config.json").read_text())
        self.cfg = cfg
        self.model = joblib.load(ARTIFACT_DIR / cfg["model_file"])
        self.k, self.radius = cfg["candidates"]["k"], cfg["candidates"]["radius_m"]
        assert cfg["features"] == FEATURES, "feature list in artifacts differs from src/features.py"
        self.hmm = _load_hmm()
        self.data_root = DATA_ROOT
        log.info("pipeline ready: model=%s decoder=%s data=%s", cfg["model_name"],
                 "HMM/Viterbi" if self.hmm else "classifier_only", self.data_root)

    # ---- trips ----
    def list_trips(self):
        out = []
        for d in sorted(p for p in self.data_root.iterdir() if p.is_dir()):
            tf = d / f"{d.name}.track"
            if tf.exists() and (d / f"{d.name}.nodes").exists() and (d / f"{d.name}.arcs").exists():
                with open(tf, "rb") as f:
                    n = sum(1 for _ in f)
                out.append({"trip_id": d.name, "n_points": n})
        return out

    def trip_exists(self, trip_id: str) -> bool:
        return any(t["trip_id"] == trip_id for t in self._trip_cache())

    def _trip_cache(self):
        if not hasattr(self, "_tc"):
            self._tc = self.list_trips()
        return self._tc

    # ---- prediction ----
    def run(self, trip_id: str):
        t0 = time.perf_counter()
        trip = load_trip(self.data_root.name, trip_id, self.data_root.parent, with_route=False)
        t_load = time.perf_counter()
        idx = build_index(trip)
        feats = build_features(trip, idx, k=self.k, radius=self.radius)
        t_feat = time.perf_counter()
        p = self.model.predict_proba(feats[FEATURES].values)[:, 1]
        feats["p"] = p
        n = len(trip.track)
        # classifier-only: highest-scoring candidate per observation
        best = feats.loc[feats.groupby("obs")["p"].idxmax()]
        clf = np.full(n, -1, dtype=np.int64)
        clf[best["obs"].values] = best["arc"].values
        t_clf = time.perf_counter()
        hmm_pred, final = None, clf.copy()
        if self.hmm:
            h = self.hmm
            graph = h["TripGraph"](trip.nodes[["x", "y"]].values, trip.arcs.u.values, trip.arcs.v.values)
            cands = h["candidates_from_frame"](feats[["obs", "arc", "frac", "p", "rank"]], n)
            hmm_pred, _breaks = h["viterbi_decode"](graph, trip.track[["x", "y"]].values,
                                                    trip.track["t"].values, cands, h["params"])
            final = np.where(hmm_pred >= 0, hmm_pred, clf)
        t_end = time.perf_counter()
        timings = {"load_ms": (t_load - t0) * 1000, "candidates_features_ms": (t_feat - t_load) * 1000,
                   "classifier_ms": (t_clf - t_feat) * 1000, "hmm_ms": (t_end - t_clf) * 1000,
                   "total_ms": (t_end - t0) * 1000}
        public = {
            "trip_id": trip_id,
            "n_points": n,
            "model": self.cfg["model_name"],
            "decoder": "hmm_viterbi" if self.hmm else "classifier_only",
            "gps": np.round(trip.track[["lat", "lon"]].values, 6).tolist(),
            "predicted_arcs": final.tolist(),
            "predicted_route": route_polyline(final, trip),
            "timings_ms": {k: round(v, 1) for k, v in timings.items()},
        }
        return public, Internals(trip, feats, clf, final, hmm_pred, timings)


def route_polyline(arcs, trip):
    """[lat, lon] polyline of the predicted road segments in travel order (consecutive duplicates merged)."""
    seq = [int(a) for a in arcs if a >= 0]
    seq = [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]
    lon, lat = trip.nodes["lon"].values, trip.nodes["lat"].values
    u, v = trip.arcs.u.values, trip.arcs.v.values
    pts = []
    for a in seq:
        for node in (u[a], v[a]):
            pt = [round(float(lat[node]), 6), round(float(lon[node]), 6)]
            if not pts or pts[-1] != pt:
                pts.append(pt)
    return pts
