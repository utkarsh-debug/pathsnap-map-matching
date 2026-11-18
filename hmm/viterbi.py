"""Candidate-state HMM and Viterbi route decoder used by the FastAPI pipeline.

Each state is a directed road arc from the per-observation candidate set. Model
probabilities supply the emissions. Transitions use directed shortest-path
distance between candidate positions and the GPS displacement between points.
Long timestamp gaps and candidate-empty observations split the sequence so that
the decoder still returns one result slot per GPS observation.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.special import logsumexp


@dataclass(frozen=True)
class HMMParams:
    """Decoder settings. `best_params` in the JSON artifact can override these defaults."""

    emission_power: float = 1.0
    transition_weight: float = 1.0
    transition_sigma_m: float = 25.0
    transition_sigma_per_s: float = 2.0
    max_gap_s: float = 30.0
    max_speed_mps: float = 60.0
    max_transition_distance_m: float = 5000.0
    max_states_per_obs: int = 15

    def __post_init__(self):
        if self.emission_power <= 0:
            raise ValueError("emission_power must be positive")
        if self.transition_weight <= 0:
            raise ValueError("transition_weight must be positive")
        if self.transition_sigma_m <= 0 or self.transition_sigma_per_s < 0:
            raise ValueError("transition sigma values must be positive (per-second may be zero)")
        if self.max_gap_s <= 0 or self.max_speed_mps <= 0 or self.max_transition_distance_m <= 0:
            raise ValueError("gap, speed, and transition-distance limits must be positive")
        if self.max_states_per_obs < 1:
            raise ValueError("max_states_per_obs must be at least one")


@dataclass(frozen=True)
class Candidate:
    """A candidate directed arc and its projection/probability at one observation."""

    arc: int
    frac: float
    probability: float
    rank: int = 0


class TripGraph:
    """Directed road graph with metric arc lengths and bounded shortest-path lookup."""

    def __init__(self, nodes_xy, arc_u, arc_v, distance_cache_size: int = 12):
        xy = np.asarray(nodes_xy, dtype=float)
        self.u = np.asarray(arc_u, dtype=np.int64).reshape(-1)
        self.v = np.asarray(arc_v, dtype=np.int64).reshape(-1)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("nodes_xy must have shape (n_nodes, 2)")
        if len(self.u) != len(self.v):
            raise ValueError("arc_u and arc_v must have the same length")
        if not len(xy) or (self.u.size and (self.u.min() < 0 or self.v.min() < 0 or
                                             self.u.max() >= len(xy) or self.v.max() >= len(xy))):
            raise ValueError("road arcs refer to an invalid node")

        self.n_nodes = len(xy)
        self.n_arcs = len(self.u)
        delta = xy[self.v] - xy[self.u]
        self.arc_lengths = np.hypot(delta[:, 0], delta[:, 1])

        # scipy sums duplicate sparse entries. Keep the shortest edge for each
        # node pair so duplicate arcs do not inflate the network distance.
        if self.n_arcs:
            edge_keys = self.u * self.n_nodes + self.v
            unique_keys, inverse = np.unique(edge_keys, return_inverse=True)
            edge_lengths = np.full(len(unique_keys), np.inf, dtype=float)
            np.minimum.at(edge_lengths, inverse, self.arc_lengths)
            edge_lengths[edge_lengths == 0] = 1e-6
            rows = unique_keys // self.n_nodes
            cols = unique_keys % self.n_nodes
            self._adjacency = csr_matrix((edge_lengths, (rows, cols)),
                                         shape=(self.n_nodes, self.n_nodes))
        else:
            self._adjacency = csr_matrix((self.n_nodes, self.n_nodes), dtype=float)

        self._distance_cache_size = max(1, int(distance_cache_size))
        self._distance_cache: OrderedDict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def _distances_from(self, source_node: int, cutoff_m: float) -> tuple[np.ndarray, np.ndarray]:
        source = int(source_node)
        cutoff = float(cutoff_m)
        key = (source, round(cutoff, 3))
        cached = self._distance_cache.get(key)
        if cached is not None:
            self._distance_cache.move_to_end(key)
            return cached

        if source < 0 or source >= self.n_nodes:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        distances = dijkstra(self._adjacency, directed=True, indices=source, limit=cutoff)
        reachable = np.flatnonzero(np.isfinite(distances)).astype(np.int32)
        values = distances[reachable].astype(np.float32)
        result = (reachable, values)
        self._distance_cache[key] = result
        if len(self._distance_cache) > self._distance_cache_size:
            self._distance_cache.popitem(last=False)
        return result

    def _node_distance(self, source_node: int, target_node: int, cutoff_m: float) -> float:
        nodes, distances = self._distances_from(source_node, cutoff_m)
        pos = int(np.searchsorted(nodes, int(target_node)))
        if pos >= len(nodes) or int(nodes[pos]) != int(target_node):
            return float("inf")
        return float(distances[pos])

    def distance_between_candidates(self, previous: Candidate, current: Candidate,
                                    cutoff_m: float) -> float | None:
        """Directed route distance between projected positions on two candidate arcs."""
        if not (0 <= previous.arc < self.n_arcs and 0 <= current.arc < self.n_arcs):
            return None
        previous_length = float(self.arc_lengths[previous.arc])
        current_length = float(self.arc_lengths[current.arc])
        if previous.arc == current.arc:
            distance = abs(current.frac - previous.frac) * previous_length
            return distance if distance <= cutoff_m else None

        from_node = int(self.v[previous.arc])
        to_node = int(self.u[current.arc])
        offset = (1.0 - previous.frac) * previous_length
        approach = current.frac * current_length
        remaining = cutoff_m - offset - approach
        if remaining < 0:
            return None
        if from_node == to_node:
            middle = 0.0
        else:
            middle = self._node_distance(from_node, to_node, cutoff_m)
        if not np.isfinite(middle) or middle > remaining:
            return None
        return float(offset + middle + approach)


def candidates_from_frame(frame, n_obs: int) -> list[list[Candidate]]:
    """Convert a feature frame into one candidate list per observation.

    The API passes columns `obs`, `arc`, `frac`, `p`, and `rank`. Duplicate arc
    entries are collapsed, retaining the candidate with the highest probability.
    """
    if n_obs < 0:
        raise ValueError("n_obs cannot be negative")
    by_obs: list[dict[int, Candidate]] = [dict() for _ in range(n_obs)]
    if frame is None or len(frame) == 0:
        return [[] for _ in range(n_obs)]

    required = {"obs", "arc", "frac", "p", "rank"}
    if not required.issubset(frame.columns):
        raise ValueError(f"candidate frame is missing columns: {sorted(required - set(frame.columns))}")

    for obs, arc, frac, probability, rank in frame[["obs", "arc", "frac", "p", "rank"]].itertuples(index=False, name=None):
        try:
            obs_i, arc_i = int(obs), int(arc)
        except (TypeError, ValueError, OverflowError):
            continue
        if not 0 <= obs_i < n_obs or arc_i < 0:
            continue
        frac_f = float(frac) if np.isfinite(frac) else 0.5
        prob_f = float(probability) if np.isfinite(probability) else 0.0
        try:
            rank_i = int(rank)
        except (TypeError, ValueError, OverflowError):
            rank_i = 0
        candidate = Candidate(arc_i, float(np.clip(frac_f, 0.0, 1.0)),
                              float(np.clip(prob_f, 0.0, 1.0)), rank_i)
        old = by_obs[obs_i].get(arc_i)
        if old is None or candidate.probability > old.probability:
            by_obs[obs_i][arc_i] = candidate
    return [sorted(group.values(), key=lambda item: (item.rank, -item.probability)) for group in by_obs]


def _emission_log_scores(states: list[Candidate], params: HMMParams) -> np.ndarray:
    probabilities = np.asarray([state.probability for state in states], dtype=float)
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0:
        probabilities.fill(1.0 / len(states))
    else:
        probabilities /= total
    return params.emission_power * np.log(probabilities)


def _transition_matrix(previous_states: list[Candidate], current_states: list[Candidate],
                       graph: TripGraph, gps_distance_m: float, dt_s: float,
                       params: HMMParams) -> np.ndarray:
    transitions = np.full((len(previous_states), len(current_states)), -np.inf, dtype=float)
    sigma = params.transition_sigma_m + params.transition_sigma_per_s * dt_s
    max_network_distance = min(params.max_transition_distance_m,
                               params.max_speed_mps * dt_s + 4.0 * sigma)
    # A small distance bucket reuses bounded Dijkstra results during tuning.
    max_network_distance = min(params.max_transition_distance_m,
                               float(np.ceil(max_network_distance / 25.0) * 25.0))
    for i, previous in enumerate(previous_states):
        for j, current in enumerate(current_states):
            route_distance = graph.distance_between_candidates(previous, current, max_network_distance)
            if route_distance is None:
                continue
            if route_distance > params.max_speed_mps * dt_s + 4.0 * sigma:
                continue
            error = (route_distance - gps_distance_m) / sigma
            transitions[i, j] = -0.5 * params.transition_weight * error * error
        reachable = np.isfinite(transitions[i])
        if reachable.any():
            transitions[i, reachable] -= logsumexp(transitions[i, reachable])
    return transitions


def viterbi_decode(graph: TripGraph, track_xy, timestamps, candidates: Iterable[Iterable[Candidate]],
                   params: HMMParams) -> tuple[np.ndarray, list[int]]:
    """Decode an arc for each observation and return indices where a new segment starts.

    Unreachable or speed-infeasible transitions break the sequence and restart
    Viterbi at that observation. Missing candidate rows produce -1 for that point;
    all later points are still decoded. The caller can fall back to its
    classifier-only result for any -1 value.
    """
    xy = np.asarray(track_xy, dtype=float)
    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("track_xy must have shape (n_observations, 2)")
    if len(xy) != len(times):
        raise ValueError("track_xy and timestamps must have the same length")
    raw_candidates = list(candidates)
    if len(raw_candidates) != len(times):
        raise ValueError("candidate list must have one entry per observation")

    states_by_obs: list[list[Candidate]] = []
    for row in raw_candidates:
        states = [c for c in row if isinstance(c, Candidate) and 0 <= c.arc < graph.n_arcs]
        if len(states) > params.max_states_per_obs:
            states = sorted(states, key=lambda item: item.probability, reverse=True)[:params.max_states_per_obs]
        states_by_obs.append(states)

    n_obs = len(times)
    prediction = np.full(n_obs, -1, dtype=np.int64)
    breaks: list[int] = []
    i = 0
    while i < n_obs:
        if not states_by_obs[i]:
            i += 1
            if i < n_obs:
                breaks.append(i)
            continue

        segment_start = i
        scores = _emission_log_scores(states_by_obs[i], params)
        back_pointers: list[np.ndarray] = []
        j = i + 1
        while j < n_obs and states_by_obs[j]:
            dt = float(times[j] - times[j - 1])
            if not np.isfinite(dt) or dt <= 0 or dt > params.max_gap_s:
                break
            gps_distance = float(np.linalg.norm(xy[j] - xy[j - 1]))
            transitions = _transition_matrix(states_by_obs[j - 1], states_by_obs[j], graph,
                                             gps_distance, dt, params)
            combined = scores[:, None] + transitions
            best_previous = np.argmax(combined, axis=0)
            best_scores = combined[best_previous, np.arange(combined.shape[1])]
            if not np.isfinite(best_scores).any():
                break
            back_pointers.append(best_previous)
            scores = best_scores + _emission_log_scores(states_by_obs[j], params)
            j += 1

        state_index = int(np.argmax(scores))
        segment_indices = [0] * (j - segment_start)
        segment_indices[-1] = state_index
        for offset in range(len(segment_indices) - 1, 0, -1):
            segment_indices[offset - 1] = int(back_pointers[offset - 1][segment_indices[offset]])
        for offset, chosen in enumerate(segment_indices):
            prediction[segment_start + offset] = states_by_obs[segment_start + offset][chosen].arc

        if j < n_obs:
            breaks.append(j)
        i = j

    return prediction, sorted(set(index for index in breaks if 0 < index < n_obs))
