"""Server-side scoring against the reference route (.route).

This is the ONLY place the reference data is read. It runs after the prediction is finished, its output
is metrics only (never the reference arcs), and nothing here feeds back into the model or the decoder.
"""
import numpy as np

from src.candidates import arc_geometries, label_points
from src.io import load_trip
from .config import DATA_ROOT


def _continuity(pred, u, v):
    a = pred[pred >= 0]
    if len(a) < 2:
        return None
    return float(((a[:-1] == a[1:]) | (v[a[:-1]] == u[a[1:]])).mean())


def score(trip_id: str, internals) -> dict | None:
    ref_trip = load_trip(DATA_ROOT.name, trip_id, DATA_ROOT.parent, with_route=True)
    if ref_trip.route is None:
        return None
    _, coords = arc_geometries(ref_trip)
    labels, _ = label_points(ref_trip, coords)          # reference arc per GPS observation
    n = len(labels)
    u, v = ref_trip.arcs.u.values, ref_trip.arcs.v.values
    clf, final, hmm = internals.clf_pred, internals.final_pred, internals.hmm_pred

    # candidate miss rate: reference arc absent from the generated candidate set
    big = 10 ** 8
    cand_keys = internals.feats["obs"].values.astype(np.int64) * big + internals.feats["arc"].values
    ref_keys = np.arange(n, dtype=np.int64) * big + labels
    miss = float((~np.isin(ref_keys, cand_keys)).mean())

    def arc_set_prf(pred):
        ps, rs = set(pred[pred >= 0].tolist()), set(labels.tolist())
        inter = len(ps & rs)
        prec, rec = inter / max(len(ps), 1), inter / max(len(rs), 1)
        return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)

    prec, rec, f1 = arc_set_prf(final)
    out = {
        "n_points": n,
        "top1_accuracy": float((final == labels).mean()),
        "top1_accuracy_classifier_only": float((clf == labels).mean()),
        "top1_accuracy_hmm": None if hmm is None else float((np.where(hmm >= 0, hmm, clf) == labels).mean()),
        "route_continuity": _continuity(final, u, v),
        "route_continuity_classifier_only": _continuity(clf, u, v),
        "route_arc_precision": prec, "route_arc_recall": rec, "route_arc_f1": f1,
        "candidate_miss_rate": miss,
    }
    return {k: (round(x, 4) if isinstance(x, float) else x) for k, x in out.items()}
