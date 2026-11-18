"""Generate a Folium map for one evaluator trip.

Run from the repository root with:
    python -m viz.folium_map --trip-id 00000015 --data-dir PATH_TO_EVALUATOR_TEST
"""
import argparse
import inspect
import json
import os
from pathlib import Path

import folium
from folium import FeatureGroup
from folium.plugins import MarkerCluster


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRIP_ID = "00000015"
DEFAULT_DATA_DIR = Path(
    os.environ.get("PATHSNAP_API_DATA", ROOT.parent / "Dataset" / "evaluator_test")
)


def _initialize_pipeline(data_dir: Path):
    """Create the existing API pipeline, disabling only an incompatible optional HMM artifact."""
    os.environ["PATHSNAP_API_DATA"] = str(data_dir)

    import api.pipeline as pipeline_module
    from hmm.viterbi import HMMParams

    hmm_path = pipeline_module.ARTIFACT_DIR / "hmm_params.json"
    if hmm_path.exists():
        hmm_config = json.loads(hmm_path.read_text(encoding="utf-8"))
        configured = {
            **hmm_config.get("best_params", {}),
            **hmm_config.get("fixed", {}),
        }
        supported = set(inspect.signature(HMMParams).parameters)
        unsupported = sorted(set(configured) - supported)
        if unsupported:
            print(
                "WARNING: HMM artifact parameters are incompatible with the current "
                f"decoder ({', '.join(unsupported)}); generating the prediction with "
                "PathSnap's existing classifier-only mode."
            )
            original_loader = pipeline_module._load_hmm
            pipeline_module._load_hmm = lambda: None
            try:
                return pipeline_module.Pipeline()
            finally:
                pipeline_module._load_hmm = original_loader

    return pipeline_module.Pipeline()


def _split_arc_coordinates(arc_ids, arcs_u, arcs_v, nodes):
    """Return one coordinate line per connected run; negative IDs force a break."""
    arc_runs = []
    current_run = []
    previous_arc = None

    def finish_run():
        if current_run:
            node_ids = [int(arcs_u[current_run[0]])]
            node_ids.extend(int(arcs_v[arc_id]) for arc_id in current_run)
            arc_runs.append(
                [(float(nodes.iloc[node]["lat"]), float(nodes.iloc[node]["lon"]))
                 for node in node_ids]
            )
            current_run.clear()

    for raw_arc_id in arc_ids:
        arc_id = int(raw_arc_id)
        if arc_id < 0:
            finish_run()
            previous_arc = None
            continue
        if arc_id >= len(arcs_u):
            raise ValueError(f"Predicted/reference arc ID {arc_id} is outside the trip network")
        if (previous_arc is not None and arc_id != previous_arc
                and int(arcs_v[previous_arc]) != int(arcs_u[arc_id])):
            finish_run()
        if not current_run or arc_id != previous_arc:
            current_run.append(arc_id)
        previous_arc = arc_id

    finish_run()
    return arc_runs


def _add_legend(map_object):
    legend = """
    <div style="
        position: fixed; bottom: 28px; left: 28px; z-index: 9999;
        background: white; border: 1px solid #777; border-radius: 4px;
        padding: 8px 11px; font-size: 13px; line-height: 1.7;">
      <strong>PathSnap map layers</strong><br>
      <span style="color:#536878;">&#9632;</span> Road network<br>
      <span style="color:#168aad;">&#9679;</span> GPS observations<br>
      <span style="color:#238b45;">&#9632;</span> Reference route<br>
      <span style="color:#d73027;">&#9632;</span> Predicted route
    </div>
    """
    map_object.get_root().html.add_child(folium.Element(legend))


def create_map(trip, predicted_arcs, decoder, output_path: Path):
    track = trip.track
    nodes = trip.nodes
    arcs_u = trip.arcs["u"].to_numpy()
    arcs_v = trip.arcs["v"].to_numpy()
    gps_locations = list(zip(track["lat"].astype(float), track["lon"].astype(float)))
    center = [float(track["lat"].mean()), float(track["lon"].mean())]

    map_object = folium.Map(
        location=center,
        zoom_start=14,
        control_scale=True,
        tiles="OpenStreetMap",
    )

    road_layer = FeatureGroup(name=f"Road network ({len(arcs_u)} directed arcs)")
    for arc_id, (start, end) in enumerate(zip(arcs_u, arcs_v)):
        coordinates = [
            (float(nodes.iloc[int(start)]["lat"]), float(nodes.iloc[int(start)]["lon"])),
            (float(nodes.iloc[int(end)]["lat"]), float(nodes.iloc[int(end)]["lon"])),
        ]
        folium.PolyLine(
            coordinates,
            color="#536878",
            weight=1,
            opacity=0.55,
            tooltip=f"Road arc {arc_id}",
        ).add_to(road_layer)
    road_layer.add_to(map_object)

    gps_layer = FeatureGroup(name=f"GPS trajectory ({len(gps_locations)} observations)")
    if len(gps_locations) > 1:
        folium.PolyLine(
            gps_locations,
            color="#168aad",
            weight=2,
            opacity=0.65,
            tooltip="GPS trajectory in observation order",
        ).add_to(gps_layer)
    gps_markers = MarkerCluster(name="GPS point markers")
    for index, (location, row) in enumerate(zip(gps_locations, track.itertuples())):
        time_value = getattr(row, "t")
        folium.CircleMarker(
            location=location,
            radius=3,
            color="#168aad",
            fill=True,
            fill_color="#168aad",
            fill_opacity=0.8,
            tooltip=f"GPS point {index} | t={time_value}",
        ).add_to(gps_markers)
    gps_markers.add_to(gps_layer)
    if gps_locations:
        folium.Marker(
            gps_locations[0],
            tooltip="GPS start",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(gps_layer)
        folium.Marker(
            gps_locations[-1],
            tooltip="GPS end",
            icon=folium.Icon(color="red", icon="stop"),
        ).add_to(gps_layer)
    gps_layer.add_to(map_object)

    if trip.route is not None:
        reference_layer = FeatureGroup(name=f"Reference route ({len(trip.route)} arcs)")
        reference_runs = _split_arc_coordinates(trip.route, arcs_u, arcs_v, nodes)
        for segment_index, coordinates in enumerate(reference_runs, start=1):
            folium.PolyLine(
                coordinates,
                color="#238b45",
                weight=5,
                opacity=0.85,
                tooltip=f"Reference route segment {segment_index}",
            ).add_to(reference_layer)
        reference_layer.add_to(map_object)

    predicted_layer = FeatureGroup(
        name=f"Predicted route ({decoder}; {len(predicted_arcs)} observations)"
    )
    predicted_runs = _split_arc_coordinates(predicted_arcs, arcs_u, arcs_v, nodes)
    for segment_index, coordinates in enumerate(predicted_runs, start=1):
        folium.PolyLine(
            coordinates,
            color="#d73027",
            weight=5,
            opacity=0.9,
            tooltip=f"Predicted route segment {segment_index}",
        ).add_to(predicted_layer)
    predicted_layer.add_to(map_object)

    bounds = [
        [float(track["lat"].min()), float(track["lon"].min())],
        [float(track["lat"].max()), float(track["lon"].max())],
    ]
    map_object.fit_bounds(bounds, padding=(20, 20))
    _add_legend(map_object)
    folium.LayerControl(collapsed=False).add_to(map_object)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_object.save(str(output_path))
    return len(predicted_runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip-id", default=DEFAULT_TRIP_ID, help="Evaluator trip directory ID")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing evaluator trip subdirectories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="HTML output path (default: results/maps/evaluator_test_<trip-id>.html)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        parser.error(f"evaluator data directory does not exist: {data_dir}")

    pipeline = _initialize_pipeline(data_dir)
    prediction, internals = pipeline.run(args.trip_id)
    from src.io import load_trip

    trip = load_trip(data_dir.name, args.trip_id, data_dir.parent, with_route=True)
    output_path = args.output or ROOT / "results" / "maps" / f"evaluator_test_{args.trip_id}.html"
    output_path = output_path.expanduser().resolve()
    segment_count = create_map(
        trip,
        internals.final_pred,
        prediction["decoder"],
        output_path,
    )

    print(f"Trip: {args.trip_id} ({len(trip.track)} GPS observations)")
    print(f"Reference route: {'available' if trip.route is not None else 'not available'}")
    print(f"Predicted route: {segment_count} connected segment(s)")
    print(f"HTML map saved to: {output_path}")


if __name__ == "__main__":
    main()
