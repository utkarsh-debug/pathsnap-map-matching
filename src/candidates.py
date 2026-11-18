"""Candidate road-segment generation and reference-label alignment.

Hidden state / candidate = a directed arc (arc_id = row index of .arcs).
For every GPS observation we take the K nearest arcs within RADIUS metres.
"""
import numpy as np
import shapely
from shapely import STRtree

K_DEFAULT = 15
RADIUS_DEFAULT = 80.0   # metres


def arc_geometries(trip):
    n = trip.nodes
    u, v = trip.arcs.u.values, trip.arcs.v.values
    x, y = n.x.values, n.y.values
    coords = np.stack([np.stack([x[u], y[u]], 1), np.stack([x[v], y[v]], 1)], 1)  # (A,2,2)
    return shapely.linestrings(coords), coords


def build_index(trip):
    geoms, coords = arc_geometries(trip)
    return STRtree(geoms), geoms, coords


def _project(px, py, a, b):
    """Closest-point projection of points onto segments a->b. Returns dist, frac t in [0,1]."""
    d = b - a
    L2 = (d ** 2).sum(1)
    L2s = np.where(L2 == 0, 1, L2)
    t = np.clip(((px - a[:, 0]) * d[:, 0] + (py - a[:, 1]) * d[:, 1]) / L2s, 0, 1)
    cx, cy = a[:, 0] + t * d[:, 0], a[:, 1] + t * d[:, 1]
    return np.hypot(px - cx, py - cy), t


def generate_candidates(trip, index=None, k=K_DEFAULT, radius=RADIUS_DEFAULT):
    """Return DataFrame-like dict arrays: obs (point index), arc, dist, frac, rank.
    Always keeps at least the nearest arc of each point (even beyond radius)."""
    import pandas as pd
    tree, geoms, coords = index or build_index(trip)
    px, py = trip.track.x.values, trip.track.y.values
    pts = shapely.points(px, py)
    pi, ai = tree.query(pts, predicate="dwithin", distance=radius)
    # make sure every point has >=1 candidate
    have = np.zeros(len(px), bool); have[pi] = True
    miss = np.where(~have)[0]
    if len(miss):
        near = tree.query_nearest(pts[miss], all_matches=False)
        pi = np.r_[pi, miss[near[0]]]
        ai = np.r_[ai, near[1]]
    a, b = coords[ai, 0], coords[ai, 1]
    dist, frac = _project(px[pi], py[pi], a, b)
    df = pd.DataFrame({"obs": pi, "arc": ai, "dist": dist, "frac": frac})
    df = df.sort_values(["obs", "dist"], kind="stable")
    df["rank"] = df.groupby("obs").cumcount()
    df = df[df["rank"] < k].reset_index(drop=True)
    return df


def label_points(trip, coords, band=40.0, back_penalty=200.0):
    """Align GPS observations to the (untimed) reference route.

    The .route gives an ordered arc sequence without timestamps. We assign each point a route
    position j(i) minimising sum of point-to-arc distances subject to j non-decreasing
    (soft: moving backwards costs back_penalty). Returns arc_id per observation and the route
    position per observation. This is our assumption: reference route = approximate truth.
    """
    route = trip.route
    px, py = trip.track.x.values, trip.track.y.values
    # distances of every point to route arcs, only within a band of the best match
    ra = coords[route]                         # (m,2,2)
    geoms = shapely.linestrings(ra)
    rtree = STRtree(geoms)
    pts = shapely.points(px, py)
    pi, ji = rtree.query(pts, predicate="dwithin", distance=band * 2)
    have = np.zeros(len(px), bool); have[pi] = True
    miss = np.where(~have)[0]
    if len(miss):
        near = rtree.query_nearest(pts[miss], all_matches=False)
        pi = np.r_[pi, miss[near[0]]]; ji = np.r_[ji, near[1]]
    d, _ = _project(px[pi], py[pi], ra[ji, 0], ra[ji, 1])
    order = np.lexsort((ji, pi))
    pi, ji, d = pi[order], ji[order], d[order]
    starts = np.searchsorted(pi, np.arange(len(px) + 1))

    n = len(px)
    back = []            # backpointers per point
    prev_j = prev_c = None
    states_j, states_cost, states_bp = [], [], []
    for i in range(n):
        s, e = starts[i], starts[i + 1]
        j, c = ji[s:e], d[s:e]
        if prev_j is None:
            cost = c.copy(); bp = np.full(len(j), -1)
        else:
            # best previous state with position <= j  (prefix-min over sorted prev positions)
            pm = np.minimum.accumulate(prev_c)
            pa = np.arange(len(prev_c))
            # argmin prefix index
            pidx = np.maximum.accumulate(np.where(prev_c == pm, pa, 0))
            pos = np.searchsorted(prev_j, j, side="right") - 1
            ok = pos >= 0
            cost = np.empty(len(j)); bp = np.empty(len(j), int)
            cost[ok] = c[ok] + pm[pos[ok]]
            bp[ok] = pidx[pos[ok]]
            # no earlier-or-equal state: allow a backward jump with penalty
            g = np.argmin(prev_c)
            cost[~ok] = c[~ok] + prev_c[g] + back_penalty
            bp[~ok] = g
            # also allow staying with a backward jump when cheaper than the monotone move
            alt = c + prev_c[g] + back_penalty
            use = alt < cost
            cost[use] = alt[use]; bp[use] = g
        states_j.append(j); states_cost.append(cost); states_bp.append(bp)
        prev_j, prev_c = j, cost
    # backtrack
    jpos = np.empty(n, int)
    k = int(np.argmin(states_cost[-1]))
    for i in range(n - 1, -1, -1):
        jpos[i] = states_j[i][k]
        k = states_bp[i][k] if i > 0 else k
    return route[jpos], jpos
