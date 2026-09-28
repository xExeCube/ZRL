import numpy as np
from neighborsMirror import NeighborSearch
from fieldMirror import triangular_lattice


def test_counts_match_brute_force():
    rng = np.random.default_rng(0)
    pos = rng.random((400, 2)) * 5.0
    ns = NeighborSearch(pos)
    r = 0.5
    counts = ns.counts(r)
    for i in (0, 17, 199, 250, 399):
        d = np.hypot(pos[:, 0] - pos[i, 0], pos[:, 1] - pos[i, 1])
        brute = int(np.count_nonzero(d <= r)) - 1
        assert brute == int(counts[i])


def test_nearest_dist_on_lattice():
    pts = triangular_lattice((3.0, 3.0), 0.2)
    ns = NeighborSearch(pts)
    assert abs(float(np.median(ns.nearest_dist())) - 0.2) < 1e-6


def test_pairs_exclude_self():
    rng = np.random.default_rng(1)
    pos = rng.random((50, 2))
    ns = NeighborSearch(pos)
    pairs = ns.pairs_within(0.4)
    for i, nb in enumerate(pairs):
        assert i not in set(nb.tolist())
