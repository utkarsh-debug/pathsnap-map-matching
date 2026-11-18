"""Deterministic decoder edge-case tests: python -m unittest hmm.test_viterbi."""
import unittest
from itertools import product
import numpy as np
import pandas as pd
from hmm.viterbi import TripGraph, HMMParams, viterbi_decode, candidates_from_frame, stitch_route


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.g = TripGraph(np.array([[0., 0.], [10., 0.], [20., 0.], [0., 100.], [10., 100.]]),
                           np.array([0, 1, 3]), np.array([1, 2, 4]))

    def test_connected_choice(self):
        c = [(np.array([0, 2]), np.array([.5, .5]), np.array([.9, .1])),
             (np.array([1, 2]), np.array([.5, .6]), np.array([.4, .6]))]
        p, _ = viterbi_decode(self.g, np.array([[5., 0.], [15., 0.]]), np.array([0., 1.]), c,
                              HMMParams(beta=3))
        np.testing.assert_array_equal(p, [0, 1])

    def test_gap_restarts(self):
        c = [(np.array([0, 2]), np.array([.5, .5]), np.array([.9, .1])),
             (np.array([1, 2]), np.array([.5, .6]), np.array([.1, .9]))]
        p, breaks = viterbi_decode(self.g, np.array([[5., 0.], [15., 100.]]),
                                   np.array([0., 100.]), c)
        np.testing.assert_array_equal(p, [0, 2])
        self.assertIn(1, breaks)

    def test_missing_candidates(self):
        empty = (np.array([], dtype=int), np.array([]), np.array([]))
        p, _ = viterbi_decode(self.g, np.zeros((2, 2)), np.arange(2),
                              [empty, (np.array([0]), np.array([.5]), np.array([1.]))])
        np.testing.assert_array_equal(p, [-1, 0])

    def test_all_empty_frame(self):
        self.assertEqual(len(candidates_from_frame(pd.DataFrame(), 3)), 3)

    def test_stitch_and_disconnection(self):
        route, gaps = stitch_route(self.g, [0, 0, 1, 2])
        self.assertEqual(route, [0, 1, 2])
        self.assertEqual(gaps, 1)

    def test_matches_exhaustive_search(self):
        """Check the dynamic program against all eight state paths of a tiny example."""
        c = [(np.array([0, 2]), np.array([.2, .2]), np.array([.8, .2])),
             (np.array([0, 2]), np.array([.5, .5]), np.array([.3, .7])),
             (np.array([1, 2]), np.array([.1, .8]), np.array([.6, .4]))]
        xy = np.array([[2., 0.], [5., 0.], [11., 0.]])
        params = HMMParams(beta=10)
        scored = []
        for path in product(range(2), repeat=3):
            score = sum(np.log(c[i][2][s]) for i, s in enumerate(path))
            for i in range(1, 3):
                a, b = c[i-1][0][path[i-1]], c[i][0][path[i]]
                fa, fb = c[i-1][1][path[i-1]], c[i][1][path[i]]
                sp = self.g.sp_matrix([self.g.v[a]], [self.g.u[b]], params.search_radius)[0, 0]
                distance = (1-fa)*self.g.arc_len[a] + sp + fb*self.g.arc_len[b]
                if a == b and fb >= fa:
                    distance = (fb-fa)*self.g.arc_len[a]
                transition = -abs(distance - np.linalg.norm(xy[i]-xy[i-1]))/params.beta
                if a == b and fb < fa:
                    transition -= params.back_penalty
                if not np.isfinite(distance):
                    transition = -params.unreach_penalty
                score += transition
            scored.append((score, path))
        best = max(scored)[1]
        pred, _ = viterbi_decode(self.g, xy, np.arange(3), c, params)
        np.testing.assert_array_equal(pred, [c[i][0][s] for i, s in enumerate(best)])


if __name__ == "__main__":
    unittest.main()
