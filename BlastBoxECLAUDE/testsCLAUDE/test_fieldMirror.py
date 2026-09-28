import numpy as np
from scipy.spatial import cKDTree
from configMirror import open_box
from fieldMirror import triangular_lattice, build_field


def test_lattice_nearest_neighbour_equals_spacing():
    pts = triangular_lattice((2.0, 2.0), 0.1)
    d, _ = cKDTree(pts).query(pts, k=2)
    assert abs(d[:, 1].min() - 0.1) < 1e-6


def test_build_all_inside_fluid():
    s = open_box()
    w = s.make_world()
    field, info = build_field(s, w)
    assert field.n > 0
    assert bool(np.all(w.clearance(field.pos) > 0))
    assert info["placed"] == field.n
    assert info["requested"] >= info["placed"]


def test_zone_temperature_applied():
    s = open_box()
    w = s.make_world()
    field, _ = build_field(s, w)
    assert np.any(field.temperature > s.ambient_temperature + 1.0)


def test_build_is_deterministic():
    s = open_box()
    w = s.make_world()
    a, _ = build_field(s, w)
    b, _ = build_field(s, w)
    assert a.n == b.n
    assert np.array_equal(a.pos, b.pos)
