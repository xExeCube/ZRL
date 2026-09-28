import numpy as np
from configMirror import Box, two_rooms, open_box
from spaceMirror import sdf_box, World


def test_sdf_box_sign_and_distance():
    b = Box(0, 0, 2, 2)
    assert sdf_box([[1, 1]], b)[0] < 0            # centre is inside -> negative
    d = sdf_box([[4, 1]], b)[0]                    # 2 m right of the right face
    assert abs(d - 2.0) < 1e-9


def test_clearance_positive_in_open_room():
    s = open_box()
    w = s.make_world()
    W, H = s.world_size
    assert w.clearance([[W / 2, H / 2]])[0] > 0


def test_obstacle_excluded_doorway_open():
    s = two_rooms()
    w = s.make_world()
    W, H = s.world_size
    assert not w.inside([[W / 2, 1.0]])[0]         # inside a wall slab
    assert w.inside([[W / 2, 4.0]])[0]             # in the doorway gap


def test_normal_points_inward_from_left_wall():
    s = open_box()
    w = s.make_world()
    n = w.normal([[0.05, 4.0]])[0]                 # just inside the left wall
    assert n[0] > 0                                # inward normal is +x


def test_open_box_area_and_perimeter():
    w = open_box().make_world()
    assert abs(w.fluid_area() - 14.0 * 8.0) < 1e-12
    assert abs(w.wall_perimeter() - 2 * (14.0 + 8.0)) < 1e-12


def test_two_rooms_area_subtracts_the_wall_slabs():
    w = two_rooms().make_world()
    # two slabs, each 0.30 wide: one 3.4 tall, one (8 - 4.6) = 3.4 tall
    assert abs(w.fluid_area() - (112.0 - 0.30 * 3.4 - 0.30 * 3.4)) < 1e-12


def test_two_rooms_perimeter_does_not_double_count_flush_edges():
    """Each slab sits flush against a domain edge. That stretch is one wall, not
    two: the slab's flush edge is skipped and the domain edge loses its length."""
    w = two_rooms().make_world()
    domain = (14.0 - 0.30) + (14.0 - 0.30) + 8.0 + 8.0     # 43.40
    slabs = 2 * (3.4 + 3.4 + 0.30)                          # 14.20
    assert abs(w.wall_perimeter() - (domain + slabs)) < 1e-9
    assert abs(w.wall_perimeter() - 57.60) < 1e-9
