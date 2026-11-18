"""Candidate-state HMM and Viterbi route decoder used by training and the API.

Each observation is represented by the candidate arc IDs, projected fractions,
and classifier probabilities in its candidate row. The decoder uses the same
parameter names and transition rules as ``results/hmm_params.json``.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


@dataclass(frozen=True)
class HMMParams:
    """Viterbi settings; ``best_params`` and ``fixed`` are API artifact fields."""

    alpha: float = 1.0
    beta: float = 30.0
    unreach_penalty: float = 15.0
    back_penalty: float = 5.0
    max_dt: float = 10.0
    max_jump: float = 250.0
    search_radius: float = 600.0

    def __post_init__(self):
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.unreach_penalty < 0 or self.back_penalty < 0:
            raise ValueError("transition penalties must be non-negative")
        if self.max_dt <= 0 or self.max_jump <= 0 or self.search_radius <= 0:
            raise ValueError("max_dt, max_jump, and search_radius must be positive")


@dataclass(frozen=True)
class Candidate:
    """A candidate directed arc and its projection/probability at one observation."""

    arc: int
    frac: float
    probability: float
    rank: int = 0


class TripGraph:
    """Directed road graph with cached bounded shortest-path lookups."""

    def __init__(self, nodes_xy, arc_u, arc_v, distance_cache_size: int = 64):
        xy = np.asarray(nodes_xy, dtype=float)
        self.u = np.asarray(arc_u, dtype=np.int64).reshape(-1)
        self.v = np.asarray(arc_v, dtype=np.int64).reshape(-1)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("nodes_xy must have shape (n_nodes, 2)")
        if not len(xy):
            raise ValueError("road graph must contain at least one node")
        if len(self.u) != len(self.v):
            raise ValueError("arc_u and arc_v must have the same length")
        if self.u.size and (self.u.min() < 0 or self.v.min() < 0 or
                            self.u.max() >= len(xy) or self.v.max() >= len(xy)):
            raise ValueError("road arcs refer to an invalid node")

        self.n_nodes = len(xy)
        self.n_arcs = len(self.u)
        delta = xy[self.v] - xy[self.u]
        self.arc_len = np.hypot(delta[:, 0], delta[:, 1])
        # Keep the old attribute name for callers that inspect graph lengths.
        self.arc_lengths = self.arc_len

        self.edge_arc: dict[tuple[int, int], int] = {}
        for arc, (source, target) in enumerate(zip(self.u, self.v)):
            key = (int(source), int(target))
            previous = self.edge_arc.get(key)
            if previous is None or self.arc_len[arc] < self.arc_len[previous]:
                self.edge_arc[key] = arc

        if self.edge_arc:
            pairs = list(self.edge_arc)
            weights = [max(float(self.arc_len[self.edge_arc[pair]]), 1e-6) for pair in pairs]
            rows = [pair[0] for pair in pairs]
            cols = [pair[1] for pair in pairs]
            self.csr = csr_matrix((weights, (rows, cols)), shape=(self.n_nodes, self.n_nodes))
        else:
            self.csr = csr_matrix((self.n_nodes, self.n_nodes), dtype=float)

        self._distance_cache_size = max(1, int(distance_cache_size))
        self._cache: OrderedDict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def reach(self, source_node: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """Sorted nodes and distances reachable from ``source_node`` within radius."""
        source, cutoff = int(source_node), float(radius)
        key = (source, cutoff)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        if not 0 <= source < self.n_nodes:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)

        distances = dijkstra(self.csr, directed=True, indices=source, limit=cutoff)
        reachable = np.flatnonzero(np.isfinite(distances)).astype(np.int64)
        result = (reachable, distances[reachable].astype(float))
        self._cache[key] = result
        if len(self._cache) > self._distance_cache_size:
            self._cache.popitem(last=False)
        return result

    def sp_matrix(self, source_nodes, target_nodes, radius: float) -> np.ndarray:
        """Return directed shortest distances for every source/target pair.

        Distances beyond ``radius`` are ``inf``. This matrix interface is shared
        by the experiment workflow and the API decoder.
        """
        sources = np.asarray(source_nodes, dtype=np.int64).reshape(-1)
        targets = np.asarray(target_nodes, dtype=np.int64).reshape(-1)
        out = np.full((len(sources), len(targets)), np.inf, dtype=float)
        if not len(targets):
            return out
        for i, source in enumerate(sources):
            nodes, distances = self.reach(int(source), radius)
            if not len(nodes):
                continue
            positions = np.searchsorted(nodes, targets)
            in_range = positions < len(nodes)
            safe_positions = np.minimum(positions, len(nodes) - 1)
            found = in_range & (nodes[safe_positions] == targets)
            out[i, found] = distances[safe_positions[found]]
        return out


def _empty_candidate_row() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.empty(0, dtype=np.int64), np.empty(0, dtype=float),
            np.empty(0, dtype=float))


def candidates_from_frame(frame, n_obs: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Group ``obs, arc, frac, p, rank`` rows into per-observation arrays.

    Duplicate arc candidates are collapsed to the row with the highest model
    probability. Each result row is ``(arc_ids, fractions, probabilities)``.
    """
    if n_obs < 0:
        raise ValueError("n_obs cannot be negative")
    grouped: list[dict[int, tuple[float, float, int]]] = [dict() for _ in range(n_obs)]
    if frame is None or len(frame) == 0:
        return [_empty_candidate_row() for _ in range(n_obs)]

    required = {"obs", "arc", "frac", "p"}
    if not required.issubset(frame.columns):
        raise ValueError(f"candidate frame is missing columns: {sorted(required - set(frame.columns))}")

    columns = ["obs", "arc", "frac", "p"]
    has_rank = "rank" in frame.columns
    if has_rank:
        columns.append("rank")
    for values in frame[columns].itertuples(index=False, name=None):
        obs, arc, frac, probability = values[:4]
        rank = values[4] if has_rank else 0
        try:
            obs_i, arc_i = int(obs), int(arc)
        except (TypeError, ValueError, OverflowError):
            continue
        if not 0 <= obs_i < n_obs or arc_i < 0:
            continue
        try:
            frac_f = float(frac)
        except (TypeError, ValueError, OverflowError):
            frac_f = 0.5
        try:
            probability_f = float(probability)
        except (TypeError, ValueError, OverflowError):
            probability_f = 0.0
        try:
            rank_i = int(rank)
        except (TypeError, ValueError, OverflowError):
            rank_i = 0
        if not np.isfinite(frac_f):
            frac_f = 0.5
        if not np.isfinite(probability_f):
            probability_f = 0.0
        candidate = (float(np.clip(frac_f, 0.0, 1.0)),
                     float(np.clip(probability_f, 0.0, 1.0)), rank_i)
        previous = grouped[obs_i].get(arc_i)
        if previous is None or candidate[1] > previous[1]:
            grouped[obs_i][arc_i] = candidate

    result = []
    for row in grouped:
        ordered = sorted(row.items(), key=lambda item: (item[1][2], -item[1][1]))
        arcs = np.asarray([arc for arc, _ in ordered], dtype=np.int64)
        fractions = np.asarray([value[0] for _, value in ordered], dtype=float)
        probabilities = np.asarray([value[1] for _, value in ordered], dtype=float)
        result.append((arcs, fractions, probabilities))
    return result


def _candidate_arrays(row, n_arcs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize the public triple-array format and accept legacy Candidate rows."""
    if row is None:
        return _empty_candidate_row()
    if isinstance(row, tuple) and len(row) == 3:
        arcs, fractions, probabilities = row
        arcs = np.asarray(arcs, dtype=np.int64).reshape(-1)
        fractions = np.asarray(fractions, dtype=float).reshape(-1)
        probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    else:
        states = list(row)
        if not states:
            return _empty_candidate_row()
        if not all(isinstance(state, Candidate) for state in states):
            raise ValueError("candidate rows must be (arc_ids, fractions, probabilities) arrays")
        arcs = np.asarray([state.arc for state in states], dtype=np.int64)
        fractions = np.asarray([state.frac for state in states], dtype=float)
        probabilities = np.asarray([state.probability for state in states], dtype=float)

    if not (len(arcs) == len(fractions) == len(probabilities)):
        raise ValueError("candidate arc, fraction, and probability arrays must have equal lengths")
    valid = ((arcs >= 0) & (arcs < n_arcs) & np.isfinite(fractions) &
             np.isfinite(probabilities))
    arcs = arcs[valid]
    fractions = np.clip(fractions[valid], 0.0, 1.0)
    probabilities = np.clip(probabilities[valid], 0.0, 1.0)
    return arcs, fractions, probabilities


def _emission_log_scores(probabilities: np.ndarray, params: HMMParams) -> np.ndarray:
    if not len(probabilities):
        return np.empty(0, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0:
        probabilities = np.full(len(probabilities), 1.0 / len(probabilities), dtype=float)
    else:
        probabilities = probabilities / total
    return params.alpha * np.log(probabilities)


def _transition_matrix(previous, current, graph: TripGraph, gps_distance: float,
                       params: HMMParams) -> np.ndarray:
    previous_arcs, previous_fracs, _ = previous
    current_arcs, current_fracs, _ = current
    shortest = graph.sp_matrix(graph.v[previous_arcs], graph.u[current_arcs],
                                params.search_radius)
    network_distance = ((1.0 - previous_fracs)[:, None] * graph.arc_len[previous_arcs, None] +
                        shortest +
                        current_fracs[None, :] * graph.arc_len[current_arcs][None, :])

    same_arc = previous_arcs[:, None] == current_arcs[None, :]
    same_arc_distance = ((current_fracs[None, :] - previous_fracs[:, None]) *
                         graph.arc_len[previous_arcs][:, None])
    same_forward = same_arc & (current_fracs[None, :] >= previous_fracs[:, None])
    same_backward = same_arc & ~same_forward
    network_distance = np.where(same_forward, same_arc_distance, network_distance)

    transitions = -np.abs(network_distance - gps_distance) / params.beta
    transitions = np.where(same_backward, transitions - params.back_penalty, transitions)
    return np.where(np.isfinite(network_distance), transitions, -params.unreach_penalty)


def viterbi_decode(graph: TripGraph, track_xy, timestamps,
                   candidates: Iterable[object], params: HMMParams | None = None
                   ) -> tuple[np.ndarray, list[int]]:
    """Decode one arc per observation and return indices where segments restart.

    Candidate rows use ``(arc_ids, fractions, probabilities)`` arrays. Empty
    rows remain ``-1`` and split the sequence. Time gaps and GPS jumps beyond
    the configured thresholds also start a fresh Viterbi segment.
    """
    if params is None:
        params = HMMParams()
    xy = np.asarray(track_xy, dtype=float)
    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("track_xy must have shape (n_observations, 2)")
    if len(xy) != len(times):
        raise ValueError("track_xy and timestamps must have the same length")
    raw_candidates = list(candidates)
    if len(raw_candidates) != len(times):
        raise ValueError("candidate list must have one row per observation")

    rows = [_candidate_arrays(row, graph.n_arcs) for row in raw_candidates]
    prediction = np.full(len(times), -1, dtype=np.int64)
    breaks: list[int] = []
    segment_obs: list[int] = []
    segment_arcs: list[np.ndarray] = []
    back_pointers: list[np.ndarray] = []
    scores: np.ndarray | None = None
    pending_gap = False

    def finish_segment():
        nonlocal scores
        if not segment_obs:
            scores = None
            return
        state = int(np.argmax(scores))
        for offset in range(len(segment_obs) - 1, -1, -1):
            prediction[segment_obs[offset]] = segment_arcs[offset][state]
            if offset:
                state = int(back_pointers[offset - 1][state])
        segment_obs.clear()
        segment_arcs.clear()
        back_pointers.clear()
        scores = None

    for index, row in enumerate(rows):
        arcs, fractions, probabilities = row
        if not len(arcs):
            finish_segment()
            pending_gap = True
            continue

        emissions = _emission_log_scores(probabilities, params)
        if pending_gap:
            breaks.append(index)
            pending_gap = False

        if segment_obs:
            previous_index = segment_obs[-1]
            dt = float(times[index] - times[previous_index])
            gps_distance = float(np.linalg.norm(xy[index] - xy[previous_index]))
            if (not np.isfinite(dt) or dt <= 0 or dt > params.max_dt or
                    not np.isfinite(gps_distance) or gps_distance > params.max_jump):
                finish_segment()
                breaks.append(index)
            else:
                previous_row = rows[previous_index]
                transitions = _transition_matrix(previous_row, row, graph, gps_distance, params)
                combined = scores[:, None] + transitions
                best_previous = np.argmax(combined, axis=0)
                scores = emissions + combined[best_previous, np.arange(len(arcs))]
                back_pointers.append(best_previous)

        if not segment_obs:
            scores = emissions.copy()
        segment_obs.append(index)
        segment_arcs.append(arcs)

    finish_segment()
    return prediction, sorted(set(index for index in breaks if 0 < index < len(times)))


def stitch_route(graph: TripGraph, pred_arcs: Iterable[int], max_path_m: float = 2000.0
                 ) -> tuple[list[int], int]:
    """Collapse repeated arcs and connect adjacent predictions where possible.

    Returns the route arc IDs and the number of joins for which no connecting
    network path was found within ``max_path_m``.
    """
    sequence = [int(arc) for arc in pred_arcs if 0 <= int(arc) < graph.n_arcs]
    deduplicated = [arc for i, arc in enumerate(sequence)
                    if i == 0 or arc != sequence[i - 1]]
    route: list[int] = []
    gaps = 0
    for arc in deduplicated:
        if not route:
            route.append(arc)
            continue
        previous = route[-1]
        source, target = int(graph.v[previous]), int(graph.u[arc])
        if source == target:
            route.append(arc)
            continue

        distances, predecessors = dijkstra(graph.csr, directed=True, indices=source,
                                           return_predecessors=True, limit=max_path_m)
        if not np.isfinite(distances[target]):
            gaps += 1
            route.append(arc)
            continue

        path_nodes = [target]
        while path_nodes[-1] != source:
            predecessor = int(predecessors[path_nodes[-1]])
            if predecessor < 0:
                break
            path_nodes.append(predecessor)
        path_nodes.reverse()
        if not path_nodes or path_nodes[0] != source:
            gaps += 1
            route.append(arc)
            continue
        for left, right in zip(path_nodes[:-1], path_nodes[1:]):
            path_arc = graph.edge_arc.get((int(left), int(right)))
            if path_arc is not None:
                route.append(path_arc)
        route.append(arc)
    return route, gaps
