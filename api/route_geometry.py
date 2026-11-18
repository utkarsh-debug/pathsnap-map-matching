"""Display-only route reconstruction. Never changes per-observation predictions or scores."""
import numpy as np
from scipy.sparse.csgraph import dijkstra


def build_route_geometry(predicted_arcs, trip, graph, breaks=(), max_dt=10.,
                         max_jump=250., max_path_m=600.):
    """Join arcs by directed road paths; otherwise split rather than invent straight lines.

    Intermediate connectors have a conservative bound min(600 m, 50 + 3*GPS step).
    This is an explicit display heuristic, not a learned threshold or accuracy improvement.
    No reference route is inspected. Time/jump gaps and decoder restarts always split.
    """
    arcs = np.asarray(predicted_arcs, dtype=np.int64)
    xy = trip.track[["x", "y"]].to_numpy()
    times = trip.track["t"].to_numpy()
    if len(arcs) != len(times):
        raise ValueError("Expected one predicted arc per observation")
    breaks = set(breaks)
    pieces, current, gap_reasons = [], [], {}
    previous, previous_i = None, None
    inserted = 0

    def split(reason):
        nonlocal current
        if current:
            pieces.append(current)
        current = []
        gap_reasons[reason] = gap_reasons.get(reason, 0) + 1

    for i, arc in enumerate(arcs):
        arc = int(arc)
        if arc < 0 or arc >= len(graph.u):
            split("missing_prediction")
            previous, previous_i = None, None
            continue
        if previous is not None:
            dt = float(times[i] - times[previous_i])
            step = float(np.linalg.norm(xy[i] - xy[previous_i]))
            if (i in breaks or not np.isfinite(dt) or dt <= 0 or dt > max_dt
                    or not np.isfinite(step) or step > max_jump):
                split("observation_gap")
            elif arc == previous:
                previous_i = i
                continue
            elif graph.v[previous] != graph.u[arc]:
                source, target = int(graph.v[previous]), int(graph.u[arc])
                limit = min(max_path_m, 50. + 3. * step)
                distances, predecessors = dijkstra(
                    graph.csr, directed=True, indices=source,
                    return_predecessors=True, limit=limit)
                if np.isfinite(distances[target]):
                    nodes = [target]
                    while nodes[-1] != source:
                        node = int(predecessors[nodes[-1]])
                        if node < 0:
                            break
                        nodes.append(node)
                    if nodes[-1] == source:
                        nodes.reverse()
                        connectors = [graph.edge_arc[(a, b)] for a, b in zip(nodes, nodes[1:])]
                        current.extend(connectors)
                        inserted += len(connectors)
                    else:
                        split("unreachable_or_implausible")
                else:
                    split("unreachable_or_implausible")
        current.append(arc)
        previous, previous_i = arc, i
    if current:
        pieces.append(current)

    lon, lat = trip.nodes.lon.to_numpy(), trip.nodes.lat.to_numpy()
    segments = []
    for piece in pieces:
        points = []
        for arc in piece:
            for node in (graph.u[arc], graph.v[arc]):
                point = [round(float(lat[node]), 6), round(float(lon[node]), 6)]
                if not points or points[-1] != point:
                    points.append(point)
        segments.append(points)
    # A legacy client gets one continuous segment, never falsely joined disconnected pieces.
    longest = max(segments, key=len, default=[])
    return {
        "predicted_route": longest,
        "predicted_route_segments": segments,
        "route_geometry": {
            "segment_count": len(segments), "inserted_connector_arcs": inserted,
            "break_reasons": gap_reasons,
            "legacy_route_is_partial": len(segments) > 1,
            "policy": "bounded_directed_shortest_paths_and_explicit_gaps",
            "max_connector_m": max_path_m,
        },
    }
