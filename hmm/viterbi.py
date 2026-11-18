"""HMM / Viterbi route reconstruction on top of classifier probabilities.

Hidden state   : the road segment (directed arc, arc_id) the vehicle is on at observation i,
                 restricted to that observation's candidate set (K nearest arcs, see src/candidates.py).
                 State index -> arc_id correspondence is kept explicitly (cand arrays), so the decoded state
                 sequence maps straight back to road segments.
Emission       : log P(obs_i | arc) = alpha * log q_i(arc),  q_i = classifier probability normalised over the
                 candidates of observation i (the classifier already encodes distance, heading, speed context).
Transition     : Newson-Krumm style. For consecutive observations with candidate arcs a -> b
                     d_net = route distance along the network from the projected point on a to that on b
                     d_gps = straight-line distance between the two GPS fixes
                     log P(b | a) = -|d_net - d_gps| / beta
                 same arc moving forward: d_net = (frac_b - frac_a) * len ; moving backwards on the same arc
                 is penalised by `back_penalty`. Unreachable pairs (no path within the search radius) get the
                 finite log-penalty `unreach_penalty` instead of -inf, so the decoder can always recover.
Gaps           : if the time gap exceeds `max_dt` seconds or the GPS jump exceeds `max_jump` metres the step is a
                 "break": the transition matrix is made uninformative (all zero) and the chain effectively
                 restarts from the emission scores, instead of forcing an unrealistic connection.
Decoder        : standard Viterbi recurrence in log-space
                     delta_i(b) = e_i(b) + max_a [ delta_{i-1}(a) + T_{i-1,i}(a,b) ]
                 with back-pointers; the best final state is traced back.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


@dataclass
class HMMParams:
    alpha: float = 1.0            # emission weight
    beta: float = 30.0            # transition scale (metres)
    unreach_penalty: float = 15.0 # log-penalty for unreachable arc pairs
    back_penalty: float = 5.0     # log-penalty for moving backwards along the same arc
    max_dt: float = 10.0          # seconds; larger -> break
    max_jump: float = 250.0       # metres; larger -> break
    search_radius: float = 600.0  # metres, bounded Dijkstra radius


class TripGraph:
    """Directed road graph of one trip with a cache of bounded shortest-path searches."""

    def __init__(self, nodes_xy, arcs_u, arcs_v):
        self.u, self.v = np.asarray(arcs_u), np.asarray(arcs_v)
        d = nodes_xy[self.v] - nodes_xy[self.u]
        self.arc_len = np.hypot(d[:, 0], d[:, 1])
        n = len(nodes_xy)
        self.n = n
        w = np.maximum(self.arc_len, 1e-3)   # zero-length arcs would be dropped by csr otherwise
        self.csr = csr_matrix((w, (self.u, self.v)), shape=(n, n))
        self._cache = {}

    def reach(self, src, radius):
        """Sorted (nodes, dists) reachable from src within radius."""
        key = (int(src), radius)
        hit = self._cache.get(key)
        if hit is None:
            row = dijkstra(self.csr, directed=True, indices=int(src), limit=radius)
            nodes = np.flatnonzero(np.isfinite(row))
            hit = (nodes, row[nodes])
            self._cache[key] = hit
        return hit

    def sp_matrix(self, src_nodes, dst_nodes, radius):
        """Shortest path length src_nodes[i] -> dst_nodes[j] (inf if beyond radius)."""
        out = np.full((len(src_nodes), len(dst_nodes)), np.inf)
        for i, s in enumerate(src_nodes):
            nodes, dist = self.reach(s, radius)
            pos = np.searchsorted(nodes, dst_nodes)
            pos = np.clip(pos, 0, len(nodes) - 1)
            ok = nodes[pos] == dst_nodes
            out[i, ok] = dist[pos[ok]]
        return out


def viterbi_decode(graph, track_xy, track_t, obs_cands, params=HMMParams()):
    """obs_cands: list over observations of (arc_ids, frac, prob) arrays (may be empty).
    Returns predicted arc per observation (-1 where an observation has no candidate) and
    a list of break indices (steps where the chain was restarted)."""
    n = len(obs_cands)
    pred = np.full(n, -1, dtype=np.int64)
    breaks = []
    prev_idx = None                       # last observation index that has candidates
    delta = None
    back = {}                             # i -> argmax pointers into previous candidate list
    for i in range(n):
        arcs, frac, prob = obs_cands[i]
        if len(arcs) == 0:
            breaks.append(i)              # missing candidates: skip this point, chain continues across it
            continue
        q = prob / max(prob.sum(), 1e-12)
        em = params.alpha * np.log(np.clip(q, 1e-9, 1.0))
        if prev_idx is None:
            delta = em
        else:
            pa, pf, _ = obs_cands[prev_idx]
            dt = track_t[i] - track_t[prev_idx]
            dgps = float(np.hypot(*(track_xy[i] - track_xy[prev_idx])))
            if dt > params.max_dt or dgps > params.max_jump:
                T = np.zeros((len(pa), len(arcs)))
                breaks.append(i)
            else:
                sp = graph.sp_matrix(graph.v[pa], graph.u[arcs], params.search_radius)
                dnet = (1.0 - pf)[:, None] * graph.arc_len[pa][:, None] + sp + (frac * graph.arc_len[arcs])[None, :]
                same = pa[:, None] == arcs[None, :]
                fwd = (frac[None, :] >= pf[:, None])
                d_same = (frac[None, :] - pf[:, None]) * graph.arc_len[arcs][None, :]
                dnet = np.where(same & fwd, d_same, dnet)
                T = -np.abs(dnet - dgps) / params.beta
                T = np.where(same & ~fwd, T - params.back_penalty, T)
                T = np.where(np.isfinite(dnet), T, -params.unreach_penalty)
            cand = delta[:, None] + T
            bp = np.argmax(cand, axis=0)
            delta = em + cand[bp, np.arange(len(arcs))]
            back[i] = (prev_idx, bp)
        prev_idx = i
        obs_cands[i] = (arcs, frac, prob)
        # remember delta for the trace-back only through back pointers; store final choice later
    # trace back from best final state
    if prev_idx is None:
        return pred, breaks
    k = int(np.argmax(delta))
    i = prev_idx
    while True:
        pred[i] = obs_cands[i][0][k]
        if i not in back:
            break
        j, bp = back[i]
        k = int(bp[k])
        i = j
    return pred, breaks


def candidates_from_frame(df, n_obs):
    """Group a candidate frame (obs, arc, frac, p) into per-observation arrays."""
    out = [(np.empty(0, int), np.empty(0), np.empty(0)) for _ in range(n_obs)]
    df = df.sort_values(["obs", "rank"] if "rank" in df else ["obs"])
    obs = df["obs"].values
    bounds = np.flatnonzero(np.r_[True, obs[1:] != obs[:-1], True])
    arcs, frac, p = df["arc"].values, df["frac"].values, df["p"].values
    for s, e in zip(bounds[:-1], bounds[1:]):
        out[obs[s]] = (arcs[s:e].astype(np.int64), frac[s:e].astype(float), p[s:e].astype(float))
    return out


def stitch_route(graph, pred_arcs):
    """Turn the per-point arc sequence into a connected road route: consecutive duplicates are merged
    and disconnected neighbours are joined by the shortest path between them (if one exists nearby).
    Returns (route_arcs, n_disconnected_joins)."""
    seq = [int(a) for a in pred_arcs if a >= 0]
    dedup = [a for k, a in enumerate(seq) if k == 0 or a != seq[k - 1]]
    route, gaps = [], 0
    for k, a in enumerate(dedup):
        if k == 0:
            route.append(a); continue
        b = route[-1]
        if graph.v[b] == graph.u[a]:
            route.append(a); continue
        # shortest path v_b -> u_a, returned as arcs
        src, dst = int(graph.v[b]), int(graph.u[a])
        if src == dst:
            route.append(a); continue
        dist, pred = dijkstra(graph.csr, directed=True, indices=src, return_predecessors=True, limit=2000.0)
        if not np.isfinite(dist[dst]):
            gaps += 1; route.append(a); continue
        path = [dst]
        while path[-1] != src:
            path.append(int(pred[path[-1]]))
        path = path[::-1]
        # nodes -> arcs via lookup
        for x, y in zip(path[:-1], path[1:]):
            lo, hi = graph.csr.indptr[x], graph.csr.indptr[x + 1]
            cols = graph.csr.indices[lo:hi]
            # recover an arc id with u==x, v==y
            m = np.flatnonzero((graph.u == x) & (graph.v == y))
            if len(m):
                route.append(int(m[0]))
        route.append(a)
    return route, gaps
