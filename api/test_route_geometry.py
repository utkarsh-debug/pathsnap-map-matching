"""Synthetic route-geometry tests; no evaluator data."""
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from hmm.viterbi import TripGraph
from api.route_geometry import build_route_geometry


class GeometryTests(unittest.TestCase):
    def make(self, xy, edges, track_xy=None, times=None):
        xy = np.asarray(xy, dtype=float)
        if track_xy is None:
            track_xy = xy[:2]
        track_xy = np.asarray(track_xy, dtype=float)
        graph = TripGraph(xy, np.array([a for a, b in edges]), np.array([b for a, b in edges]))
        trip = SimpleNamespace(
            nodes=pd.DataFrame({"lon": xy[:, 0]/100000, "lat": xy[:, 1]/100000}),
            track=pd.DataFrame({"x": track_xy[:, 0], "y": track_xy[:, 1],
                                "t": np.arange(len(track_xy)) if times is None else times}))
        return trip, graph

    def test_short_connector_is_inserted(self):
        trip, graph = self.make([[0,0],[10,0],[20,0],[30,0]], [(0,1),(1,2),(2,3)])
        pred = np.array([0,2])
        result = build_route_geometry(pred, trip, graph)
        self.assertEqual(result["route_geometry"]["inserted_connector_arcs"], 1)
        self.assertEqual(len(result["predicted_route_segments"]), 1)
        self.assertEqual(len(result["predicted_route"]), 4)
        np.testing.assert_array_equal(pred, [0,2])

    def test_disconnection_splits(self):
        trip, graph = self.make([[0,0],[10,0],[20,0],[30,0]], [(0,1),(2,3)])
        r = build_route_geometry([0,1], trip, graph)
        self.assertEqual(len(r["predicted_route_segments"]), 2)
        self.assertTrue(r["route_geometry"]["legacy_route_is_partial"])
        self.assertIn(r["predicted_route"], r["predicted_route_segments"])

    def test_wrong_direction_is_not_connected(self):
        trip, graph = self.make([[0,0],[10,0],[20,0],[30,0]], [(0,1),(2,1),(2,3)])
        self.assertEqual(len(build_route_geometry([0,2], trip, graph)["predicted_route_segments"]), 2)

    def test_long_detour_is_rejected(self):
        trip, graph = self.make([[0,0],[10,0],[10,500],[20,0],[30,0]],
                                [(0,1),(1,2),(2,3),(3,4)])
        self.assertEqual(len(build_route_geometry([0,3], trip, graph)["predicted_route_segments"]), 2)

    def test_time_gap_and_decoder_restart_split_even_same_arc(self):
        trip, graph = self.make([[0,0],[10,0]], [(0,1)], times=[0,100])
        self.assertEqual(len(build_route_geometry([0,0], trip, graph)["predicted_route_segments"]), 2)
        trip.track["t"] = [0,1]
        self.assertEqual(len(build_route_geometry([0,0], trip, graph, breaks=[1])["predicted_route_segments"]), 2)

    def test_missing_observation_splits(self):
        trip, graph = self.make([[0,0],[10,0],[20,0]], [(0,1),(1,2)], track_xy=[[0,0],[5,0],[10,0]])
        self.assertEqual(len(build_route_geometry([0,-1,1], trip, graph)["predicted_route_segments"]), 2)

    def test_empty_predictions(self):
        trip, graph = self.make([[0,0],[10,0]], [(0,1)], track_xy=np.empty((0,2)))
        self.assertEqual(build_route_geometry([], trip, graph)["predicted_route_segments"], [])


if __name__ == "__main__":
    unittest.main()
