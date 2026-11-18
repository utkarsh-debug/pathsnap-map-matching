"""Per-candidate feature engineering. A row = (GPS observation, candidate arc).

Features (all computed from the trip's own track + network, no learned transforms, so no leakage):
  dist, dist_rel, rank, n_cands          : distance to arc, relative to the nearest candidate, rank
  frac                                   : position of the projection along the arc (0..1)
  arc_len, out_deg_v, in_deg_u           : arc length and connectivity of its end nodes
  cos_head, moving                       : cosine between track heading and arc direction (separates
                                           the two directions of a two-way road), moving flag
  speed, dt_prev, dt_next                : speed (m/s) and timing context
  step_prev, step_next                   : distances to neighbouring GPS points
  dist_prev_same, dist_next_same         : distance of the SAME arc to the previous/next observation
                                           (sequence context; 'absent' filled with ABSENT)
  on_prev_best, on_next_best             : arc is (or is graph-adjacent to) nearest arc of neighbour
"""
import numpy as np
import pandas as pd

from .candidates import build_index, generate_candidates

ABSENT = 200.0
FEATURES = [
    "dist", "dist_rel", "rank", "n_cands", "frac", "arc_len", "out_deg_v", "in_deg_u",
    "cos_head", "moving", "speed", "dt_prev", "dt_next", "step_prev", "step_next",
    "dist_prev_same", "dist_next_same", "adj_prev_best", "adj_next_best",
]


def _track_motion(trip):
    x, y, t = trip.track.x.values, trip.track.y.values, trip.track.t.values
    n = len(x)
    step = np.hypot(np.diff(x), np.diff(y))
    dt = np.diff(t)
    step_prev = np.r_[0.0, step]
    step_next = np.r_[step, 0.0]
    dt_prev = np.r_[1.0, dt]
    dt_next = np.r_[dt, 1.0]
    # heading from central difference (clamped at ends)
    i0 = np.clip(np.arange(n) - 1, 0, n - 1)
    i1 = np.clip(np.arange(n) + 1, 0, n - 1)
    hx, hy = x[i1] - x[i0], y[i1] - y[i0]
    hn = np.hypot(hx, hy)
    span = np.maximum(t[i1] - t[i0], 1.0)
    speed = hn / span
    return dict(hx=hx, hy=hy, hn=hn, speed=speed, dt_prev=dt_prev, dt_next=dt_next,
                step_prev=step_prev, step_next=step_next)


def build_features(trip, index=None, k=15, radius=80.0):
    tree, geoms, coords = index or build_index(trip)
    c = generate_candidates(trip, (tree, geoms, coords), k=k, radius=radius)
    m = _track_motion(trip)
    obs = c["obs"].values
    arc = c["arc"].values

    a, b = coords[arc, 0], coords[arc, 1]
    ad = b - a
    arc_len = np.hypot(ad[:, 0], ad[:, 1])
    hn = m["hn"][obs]
    moving = (hn > 2.0).astype(float)
    cos = (m["hx"][obs] * ad[:, 0] + m["hy"][obs] * ad[:, 1]) / np.maximum(hn * arc_len, 1e-9)
    c["cos_head"] = np.where(moving > 0, cos, 0.0)
    c["moving"] = moving
    c["arc_len"] = arc_len
    nn = len(trip.nodes)
    out_deg = np.bincount(trip.arcs.u.values, minlength=nn)
    in_deg = np.bincount(trip.arcs.v.values, minlength=nn)
    c["out_deg_v"] = out_deg[trip.arcs.v.values[arc]]
    c["in_deg_u"] = in_deg[trip.arcs.u.values[arc]]
    for key in ("speed", "dt_prev", "dt_next", "step_prev", "step_next"):
        c[key] = m[key][obs]
    c["n_cands"] = c.groupby("obs")["arc"].transform("size")
    c["dist_rel"] = c["dist"] - c.groupby("obs")["dist"].transform("min")

    # sequence context: same arc's distance at neighbouring observations
    key = c[["obs", "arc", "dist"]]
    prev = key.assign(obs=key["obs"] + 1).rename(columns={"dist": "dist_prev_same"})
    nxt = key.assign(obs=key["obs"] - 1).rename(columns={"dist": "dist_next_same"})
    c = c.merge(prev, on=["obs", "arc"], how="left").merge(nxt, on=["obs", "arc"], how="left")
    c["dist_prev_same"] = c["dist_prev_same"].fillna(ABSENT)
    c["dist_next_same"] = c["dist_next_same"].fillna(ABSENT)

    # adjacency to the nearest candidate of the neighbouring observations
    best = c[c["rank"] == 0].set_index("obs")["arc"]
    nobs = len(trip.track)
    best_arr = best.reindex(range(nobs)).fillna(-1).astype(int).values
    au, av = trip.arcs.u.values, trip.arcs.v.values

    def adj(my_arc, other_arc):
        ok = other_arc >= 0
        o = np.where(ok, other_arc, 0)
        same = my_arc == o
        fwd = av[o] == au[my_arc]       # other -> mine
        bwd = av[my_arc] == au[o]       # mine -> other
        return (ok & (same | fwd | bwd)).astype(float)

    prev_best = best_arr[np.clip(c["obs"].values - 1, 0, nobs - 1)]
    next_best = best_arr[np.clip(c["obs"].values + 1, 0, nobs - 1)]
    c["adj_prev_best"] = adj(c["arc"].values, prev_best)
    c["adj_next_best"] = adj(c["arc"].values, next_best)
    return c
