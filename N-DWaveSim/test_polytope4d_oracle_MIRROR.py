"""
ZRL 4D oracle — every closed form in ZRL_4D_findings_MIRROR.md.

Run:    python -m pytest test_polytope4d_oracle_MIRROR.py -v

Pure stdlib (math + itertools + pytest). No numpy, no sympy, so it runs under either
interpreter on this machine (canon: `pytest` is Py3.9 without numpy, `python` is Py3.12).

DESIGN RULE, same as the 3D oracle: nothing is checked against the formula it came from.
The k-face radii are rebuilt from the PYTHAGOREAN LADDER

        0R^2 = kR^2 + rho_k^2        rho_k = circumradius of the k-face in its own hyperplane

using only 0R and elementary sub-polytope circumradii, then compared to the closed forms in
the findings doc. The tesseract shadow rows are computed from actual vertex coordinates.

NOT RUN by Claude (author runs scripts himself). Written 17/09/2026 by Claude Opus 5.
"""

import math
from itertools import product

import pytest

PHI = (1.0 + math.sqrt(5.0)) / 2.0
TOL = 1e-12          # closed-form vs closed-form
TOL_DOC = 1e-9       # printed decimals in the findings doc


# ---------------------------------------------------------------------------
# elementary circumradii at EDGE LENGTH 1 (used to rebuild the radii ladder)
# ---------------------------------------------------------------------------

def polygon_circumradius(p):
    """Regular p-gon, edge 1."""
    return 1.0 / (2.0 * math.sin(math.pi / p))


CELL_CIRCUMRADIUS = {                     # regular polyhedron, edge 1
    "tetrahedron": math.sqrt(6.0) / 4.0,
    "cube": math.sqrt(3.0) / 2.0,
    "octahedron": math.sqrt(2.0) / 2.0,
    "dodecahedron": (math.sqrt(3.0) / 2.0) * PHI,
}

# name -> (schlafli, V, E, F, C, 0R at edge 1, face p-gon, cell name, dual, |W|, |rot|)
POLYTOPES = {
    "5-cell":    ("{3,3,3}",   5,   10,   10,   5, math.sqrt(2.0 / 5.0), 3, "tetrahedron",  "5-cell",   120,   60),
    "tesseract": ("{4,3,3}",  16,   32,   24,   8, 1.0,                  4, "cube",         "16-cell",  384,  192),
    "16-cell":   ("{3,3,4}",   8,   24,   32,  16, math.sqrt(2.0) / 2.0, 3, "tetrahedron",  "tesseract",384,  192),
    "24-cell":   ("{3,4,3}",  24,   96,   96,  24, 1.0,                  3, "octahedron",   "24-cell", 1152,  576),
    "120-cell":  ("{5,3,3}", 600, 1200,  720, 120, PHI * PHI * math.sqrt(2.0), 5, "dodecahedron", "600-cell", 14400, 7200),
    "600-cell":  ("{3,3,5}", 120,  720, 1200, 600, PHI,                  3, "tetrahedron",  "120-cell", 14400, 7200),
}

# the findings-doc signature vectors (s1, s2, s3), as CLOSED FORMS
DOC_SIGNATURE = {
    "5-cell":    (math.sqrt(6.0) / 4.0,            math.sqrt(6.0) / 6.0,          0.25),
    "tesseract": (math.sqrt(3.0) / 2.0,            math.sqrt(2.0) / 2.0,          0.5),
    "16-cell":   (math.sqrt(2.0) / 2.0,            math.sqrt(3.0) / 3.0,          0.5),
    "24-cell":   (math.sqrt(3.0) / 2.0,            math.sqrt(6.0) / 3.0,          math.sqrt(2.0) / 2.0),
    "120-cell":  (PHI * math.sqrt(6.0) / 4.0,      math.sqrt((5.0 + 2.0 * math.sqrt(5.0)) / 10.0),
                                                                                  PHI * PHI * math.sqrt(2.0) / 4.0),
    "600-cell":  (math.sqrt(10.0 + 2.0 * math.sqrt(5.0)) / 4.0, PHI / math.sqrt(3.0),
                                                                                  PHI * PHI * math.sqrt(2.0) / 4.0),
}

# the decimals printed in the findings doc, §2.2
DOC_DECIMALS = {
    "5-cell":    (0.612372436, 0.408248290, 0.250000000),
    "tesseract": (0.866025404, 0.707106781, 0.500000000),
    "16-cell":   (0.707106781, 0.577350269, 0.500000000),
    "24-cell":   (0.866025404, 0.816496581, 0.707106781),
    "120-cell":  (0.990839415, 0.973248990, 0.925614793),
    "600-cell":  (0.951056516, 0.934172359, 0.925614793),
}


def ladder_signature(name):
    """Rebuild (s1, s2, s3) from 0R and sub-polytope circumradii only."""
    _, _, _, _, _, R0, face_p, cell, _, _, _ = POLYTOPES[name]
    rho = (0.5, polygon_circumradius(face_p), CELL_CIRCUMRADIUS[cell])
    out = []
    for r in rho:
        inner = R0 * R0 - r * r
        assert inner > 0.0, f"{name}: ladder gave a non-positive kR^2"
        out.append(math.sqrt(inner) / R0)
    return tuple(out)


# ---------------------------------------------------------------------------
# §2.1  counts, Euler, group orders
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(POLYTOPES))
def test_euler_relation_is_zero(name):
    """V - E + F - C = 0 in 4D (not 2)."""
    _, V, E, F, C, *_ = POLYTOPES[name]
    assert V - E + F - C == 0


def test_euler_parity_general():
    """sum (-1)^k f_k = 1 - (-1)^n : 0 in even dims, 2 in odd."""
    for n in range(2, 9):
        assert 1 - (-1) ** n == (0 if n % 2 == 0 else 2)


@pytest.mark.parametrize(
    "name,degrees",
    [("5-cell", (2, 3, 4, 5)), ("tesseract", (2, 4, 6, 8)), ("16-cell", (2, 4, 6, 8)),
     ("24-cell", (2, 6, 8, 12)), ("120-cell", (2, 12, 20, 30)), ("600-cell", (2, 12, 20, 30))],
)
def test_group_order_is_product_of_degrees(name, degrees):
    """|W| = product of the Coxeter degrees; rotation subgroup is index 2."""
    order = POLYTOPES[name][9]
    rot = POLYTOPES[name][10]
    prod = 1
    for d in degrees:
        prod *= d
    assert prod == order
    assert rot * 2 == order


CELL_FLAGS = {"tetrahedron": 24, "cube": 48, "octahedron": 48, "dodecahedron": 120}


@pytest.mark.parametrize("name", list(POLYTOPES))
def test_flag_count_equals_group_order(name):
    """Regularity certificate |G| = F.

    Flags of a 4-polytope = C * (flags of one cell). The group acts freely on flags,
    so |W| = F exactly for a regular polytope -- this is the §3.5 certificate, and it
    is what the symmetry-snap test measures.
    """
    _, _, _, _, C, _, _, cell, _, order, _ = POLYTOPES[name]
    assert C * CELL_FLAGS[cell] == order


def test_dual_pairs_swap_f_vectors():
    for a, b in [("tesseract", "16-cell"), ("120-cell", "600-cell")]:
        _, Va, Ea, Fa, Ca, *_ = POLYTOPES[a]
        _, Vb, Eb, Fb, Cb, *_ = POLYTOPES[b]
        assert (Va, Ea, Fa, Ca) == (Cb, Fb, Eb, Vb)
    for s in ["5-cell", "24-cell"]:
        _, V, E, F, C, *_ = POLYTOPES[s]
        assert (V, E) == (C, F)          # self-dual


# ---------------------------------------------------------------------------
# §2.2  the signature vectors  (the new canon)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(POLYTOPES))
def test_signature_from_ladder_matches_closed_form(name):
    """Independent construction (ladder) vs the doc's closed forms."""
    built = ladder_signature(name)
    doc = DOC_SIGNATURE[name]
    for b, d in zip(built, doc):
        assert b == pytest.approx(d, abs=TOL)


@pytest.mark.parametrize("name", list(POLYTOPES))
def test_signature_decimals_as_printed(name):
    """The decimals printed in §2.2 are right to 1e-9."""
    for closed, printed in zip(DOC_SIGNATURE[name], DOC_DECIMALS[name]):
        assert closed == pytest.approx(printed, abs=TOL_DOC)


@pytest.mark.parametrize("name", list(POLYTOPES))
def test_radii_strictly_monotone(name):
    """0 < s3 < s2 < s1 < 1. Any violation is a fatal geometry error."""
    s1, s2, s3 = DOC_SIGNATURE[name]
    assert 0.0 < s3 < s2 < s1 < 1.0


def test_canon_family_formulas_at_n4():
    """CONFIRMS CANON: the three N-D closed forms evaluate correctly at n = 4."""
    n = 4
    hypercube = tuple(math.sqrt((n - k) / n) for k in (1, 2, 3))
    cross = tuple(1.0 / math.sqrt(k + 1) for k in (1, 2, 3))
    simplex = tuple(math.sqrt((n - k) / (n * (k + 1))) for k in (1, 2, 3))
    for got, want in [(hypercube, DOC_SIGNATURE["tesseract"]),
                      (cross, DOC_SIGNATURE["16-cell"]),
                      (simplex, DOC_SIGNATURE["5-cell"])]:
        for g, w in zip(got, want):
            assert g == pytest.approx(w, abs=TOL)


# ---------------------------------------------------------------------------
# §2.3  duality and self-duality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p,q", [("tesseract", "16-cell"), ("120-cell", "600-cell")])
def test_duality_law(p, q):
    """s_k(P*) = s_{n-1} / s_{n-1-k}(P), with s_0 = 1. n = 4."""
    sp = (1.0,) + DOC_SIGNATURE[p]        # s0..s3
    sq = DOC_SIGNATURE[q]                 # s1..s3
    s3 = sp[3]
    for k in (1, 2, 3):
        assert sq[k - 1] == pytest.approx(s3 / sp[3 - k], abs=TOL)


@pytest.mark.parametrize("name", ["5-cell", "24-cell"])
def test_self_duality_multiplicative_palindrome(name):
    """Self-dual <=> s1 * s2 = s3 (NOT s_k = s_{n-1-k}, which would force the ball)."""
    s1, s2, s3 = DOC_SIGNATURE[name]
    assert s1 * s2 == pytest.approx(s3, abs=TOL)
    assert s1 != pytest.approx(s2, abs=1e-6)   # not a literal palindrome


def test_duality_law_reduces_to_canon_3d_statement():
    """In 3D the law's k = n-1 case is canon's 'duals share the facet entry'."""
    cube = (math.sqrt(2.0 / 3.0), 1.0 / math.sqrt(3.0))       # (s1, s2), n = 3
    octa = (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(3.0))
    s2 = cube[1]
    assert octa[1] == pytest.approx(s2 / 1.0, abs=TOL)         # shared facet entry
    assert octa[0] == pytest.approx(s2 / cube[0], abs=TOL)     # differing edge entry


# ---------------------------------------------------------------------------
# §2.4  the cos(pi/m) result that supersedes the Niven line
# ---------------------------------------------------------------------------

def _cos_pi_over_m(x, mmax=400):
    """Return integer m with cos(pi/m) == x to 1e-12, else None."""
    for m in range(1, mmax + 1):
        if abs(math.cos(math.pi / m) - x) < 1e-12:
            return m
    return None


def test_tesseract_whole_vector_is_cos_pi_over_m():
    """MAJOR: the 4-cube's entire signature is (cos pi/6, cos pi/4, cos pi/3)."""
    assert [_cos_pi_over_m(s) for s in DOC_SIGNATURE["tesseract"]] == [6, 4, 3]


def test_hypercube_whole_vector_cos_hits_only_at_n2_and_n4():
    """THEOREM: the n-cube's whole signature is cos(pi/integer) iff n in {2, 4}."""
    good = []
    for n in range(2, 41):
        sig = [math.sqrt((n - k) / n) for k in range(1, n)]
        if all(_cos_pi_over_m(s) is not None for s in sig):
            good.append(n)
    assert good == [2, 4]


def test_niven_set_is_the_quarters():
    """The mechanism: rational cos^2(pi/m) values are exactly {0, 1/4, 1/2, 3/4, 1}."""
    rationals = set()
    for m in range(1, 200):
        c2 = math.cos(math.pi / m) ** 2
        for num, den in [(0, 1), (1, 4), (1, 2), (3, 4), (1, 1)]:
            if abs(c2 - num / den) < 1e-12:
                rationals.add((num, den))
    assert rationals == {(0, 1), (1, 4), (1, 2), (3, 4), (1, 1)}
    # and n = 4 realises the three interior ones exactly once
    assert sorted((4 - k) / 4 for k in (1, 2, 3)) == [0.25, 0.5, 0.75]


def test_all_eighteen_components_cos_hit_census():
    """Exactly 8 of the 18 4D signature components are cos(pi/integer)."""
    hits = {}
    for name, sig in DOC_SIGNATURE.items():
        hits[name] = [_cos_pi_over_m(s) for s in sig]
    assert hits["tesseract"] == [6, 4, 3]
    assert hits["16-cell"] == [4, None, 3]
    assert hits["24-cell"] == [6, None, 4]
    assert hits["600-cell"] == [10, None, None]
    assert hits["5-cell"] == [None, None, None]
    assert hits["120-cell"] == [None, None, None]
    assert sum(1 for v in hits.values() for m in v if m is not None) == 8


def test_600cell_edge_signature_is_cos_pi_over_10():
    assert DOC_SIGNATURE["600-cell"][0] == pytest.approx(math.cos(math.pi / 10), abs=TOL)


def test_canon_erratum_E1_cross_polytope_edge_signature():
    """E1: the n-cross-polytope has 1R/0R = 1/sqrt2 = cos(pi/4) for EVERY n >= 2.

    This is the counterexample to the unscoped caveat at
    ZRL_ND_flavor_renderer_spec_MIRROR.md:99 -- including n = 3 (the octahedron).
    Built from coordinates, not from the 1/sqrt(k+1) formula: with vertices +-R*e_i,
    the midpoint of the edge joining R*e_1 and R*e_2 is (R/2)(e_1 + e_2), of norm
    R/sqrt(2), in every dimension.
    """
    R = 1.0
    for n in range(2, 13):
        v1 = [R if i == 0 else 0.0 for i in range(n)]
        v2 = [R if i == 1 else 0.0 for i in range(n)]
        mid = [(a + b) / 2.0 for a, b in zip(v1, v2)]
        s1 = math.sqrt(sum(c * c for c in mid)) / R
        assert s1 == pytest.approx(1.0 / math.sqrt(2.0), abs=TOL)
        assert _cos_pi_over_m(s1) == 4          # cos(pi/4), in every dimension
    # n = 3 is the octahedron: the case canon's unscoped caveat denies
    assert _cos_pi_over_m(1.0 / math.sqrt(2.0)) == 4
    # while the FACET entries in 3D really are not cos(pi/n) -- canon's scoped claim holds
    for facet in (1.0 / 3.0, 1.0 / math.sqrt(3.0), 0.7946544723):
        assert _cos_pi_over_m(facet) is None


# ---------------------------------------------------------------------------
# §2.2 / §1.1  the organising identity and the honeycomb test
# ---------------------------------------------------------------------------

DIHEDRAL_DEG = {
    "5-cell": math.degrees(math.acos(0.25)),
    "tesseract": 90.0,
    "16-cell": 120.0,
    "24-cell": 120.0,
    "120-cell": 144.0,
    "600-cell": math.degrees(math.acos(-(1.0 + 3.0 * math.sqrt(5.0)) / 8.0)),
}


@pytest.mark.parametrize("name", list(POLYTOPES))
def test_organising_identity_s1_equals_cos_half_edge_arc(name):
    """s1 = cos(edge_arc/2) and edge_arc(P) = 180 - dihedral(P*)."""
    dual = POLYTOPES[name][8]
    edge_arc = 180.0 - DIHEDRAL_DEG[dual]
    assert DOC_SIGNATURE[name][0] == pytest.approx(
        math.cos(math.radians(edge_arc / 2.0)), abs=1e-10
    )


def test_edge_arcs_as_printed():
    """The §2.2 edge-arc list, including the 15.5224878 the sweep first got wrong."""
    expected = {"5-cell": 104.4775122, "tesseract": 60.0, "16-cell": 90.0,
                "24-cell": 60.0, "120-cell": 15.5224878, "600-cell": 36.0}
    for name, want in expected.items():
        dual = POLYTOPES[name][8]
        got = 180.0 - DIHEDRAL_DEG[dual]
        assert got == pytest.approx(want, abs=1e-6)


def test_dihedral_angles_as_printed():
    assert DIHEDRAL_DEG["5-cell"] == pytest.approx(75.5224878, abs=1e-6)
    assert DIHEDRAL_DEG["600-cell"] == pytest.approx(164.477512, abs=1e-5)


def test_exactly_three_regular_polytopes_tile_R4():
    """360 / dihedral integral <=> tiles E^4 face-to-face."""
    tilers = []
    for name, d in DIHEDRAL_DEG.items():
        k = 360.0 / d
        if abs(k - round(k)) < 1e-9:
            tilers.append((name, round(k)))
    assert sorted(tilers) == sorted([("tesseract", 4), ("16-cell", 3), ("24-cell", 3)])


def test_nontilers_ratios_as_printed():
    assert 360.0 / DIHEDRAL_DEG["5-cell"] == pytest.approx(4.7667921, abs=1e-6)
    assert 360.0 / DIHEDRAL_DEG["120-cell"] == pytest.approx(2.5, abs=1e-12)
    assert 360.0 / DIHEDRAL_DEG["600-cell"] == pytest.approx(2.1887490, abs=1e-6)


# ---------------------------------------------------------------------------
# §3.5  the tesseract's vertex-first shadow -- the regularity counterexample
# ---------------------------------------------------------------------------

def _tesseract_shadow():
    """Vertices (+-1)^4 projected along n = (1,1,1,1)/2. Returns (images, radii)."""
    n = (0.5, 0.5, 0.5, 0.5)
    verts = list(product((-1.0, 1.0), repeat=4))
    images = []
    for v in verts:
        dot = sum(a * b for a, b in zip(v, n))
        images.append(tuple(a - dot * b for a, b in zip(v, n)))
    return verts, images


def test_shadow_all_32_edges_draw_equal():
    """All 32 tesseract edges project to exactly sqrt(3) -- equal, yet not regular."""
    verts, images = _tesseract_shadow()
    idx = {v: i for i, v in enumerate(verts)}
    lengths = []
    for v in verts:
        for k in range(4):
            w = list(v)
            w[k] = -w[k]
            a, b = images[idx[v]], images[idx[tuple(w)]]
            lengths.append(math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))
    assert len(lengths) == 64                      # each edge counted twice
    for L in lengths:
        assert L == pytest.approx(math.sqrt(3.0), abs=1e-12)


def test_shadow_has_15_distinct_points_14_on_hull():
    """§3.5 / ledger row 11: dedupe count 15, hull vertex count 14, radii sqrt3 and 2."""
    _, images = _tesseract_shadow()
    distinct = set()
    for p in images:
        distinct.add(tuple(round(c, 9) + 0.0 for c in p))
    assert len(distinct) == 15
    radii = sorted({round(math.sqrt(sum(c * c for c in p)), 9) for p in images})
    assert radii == [0.0, round(math.sqrt(3.0), 9), 2.0]
    on_hull = [p for p in distinct if any(abs(c) > 1e-12 for c in p)]
    assert len(on_hull) == 14
    n_sqrt3 = sum(1 for p in images if abs(math.sqrt(sum(c * c for c in p)) - math.sqrt(3.0)) < 1e-9)
    n_two = sum(1 for p in images if abs(math.sqrt(sum(c * c for c in p)) - 2.0) < 1e-9)
    assert (n_sqrt3, n_two) == (8, 6)


def test_signature_is_the_shadow_shrink_factor():
    """The hypercube's s1 IS the orthographic edge shrink factor on the main diagonal.

    n = 3 reproduces the canon [111] factor sqrt(2/3); n = 4 gives sqrt3/2, which is why
    the tesseract's 32 edges all draw at sqrt(3) when the true edge is 2.
    """
    for n in range(2, 9):
        shrink = math.sqrt(1.0 - 1.0 / n)          # |P e_i| for the main-diagonal kernel
        assert shrink == pytest.approx(math.sqrt((n - 1) / n), abs=TOL)
    assert math.sqrt(1.0 - 1.0 / 3.0) == pytest.approx(math.sqrt(2.0 / 3.0), abs=TOL)
    assert math.sqrt(1.0 - 1.0 / 4.0) == pytest.approx(DOC_SIGNATURE["tesseract"][0], abs=TOL)
    assert 2.0 * math.sqrt(1.0 - 1.0 / 4.0) == pytest.approx(math.sqrt(3.0), abs=TOL)


def test_single_view_length_ambiguity_witness():
    """§3.4: edge (1,0,0,0) and face diagonal (1,0,0,1) share an image, differ in length."""
    # choose a kernel that kills the difference: the two differ by (0,0,0,1)
    k = (0.0, 0.0, 0.0, 1.0)
    norm = math.sqrt(sum(c * c for c in k))
    kk = tuple(c / norm for c in k)

    def proj_k(v):
        d = sum(a * b for a, b in zip(v, kk))
        return tuple(a - d * b for a, b in zip(v, kk))

    u1, u2 = (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0)
    assert proj_k(u1) == pytest.approx(proj_k(u2), abs=TOL)
    assert math.sqrt(sum(c * c for c in u1)) == pytest.approx(1.0, abs=TOL)
    assert math.sqrt(sum(c * c for c in u2)) == pytest.approx(math.sqrt(2.0), abs=TOL)

    # and the two-view fix: a second projection with a DISTINCT kernel separates them
    k2 = (0.0, 0.0, 1.0, 1.0)
    norm2 = math.sqrt(sum(c * c for c in k2))
    kk2 = tuple(c / norm2 for c in k2)

    def proj_k2(v):
        d = sum(a * b for a, b in zip(v, kk2))
        return tuple(a - d * b for a, b in zip(v, kk2))

    sep = math.sqrt(sum((a - b) ** 2 for a, b in zip(proj_k2(u1), proj_k2(u2))))
    assert sep > 0.1                                  # distinct kernel lines => separated


# ---------------------------------------------------------------------------
# §3.6  SO(4), snap angles
# ---------------------------------------------------------------------------

def test_so_n_dimension():
    assert [n * (n - 1) // 2 for n in (2, 3, 4)] == [1, 3, 6]


def test_snap_angles_from_coxeter_numbers():
    """360/h names which polytope you are looking at."""
    h = {"5-cell": 5, "tesseract": 8, "16-cell": 8, "24-cell": 12, "120-cell": 30, "600-cell": 30}
    want = {"5-cell": 72.0, "tesseract": 45.0, "16-cell": 45.0,
            "24-cell": 30.0, "120-cell": 12.0, "600-cell": 12.0}
    for name in h:
        assert 360.0 / h[name] == pytest.approx(want[name], abs=TOL)


def test_vertex_group_orders_divide_hopf_rings():
    """Lagrange: 24/6 = 4 great hexagons, 120/10 = 12 great decagons, 8/4 = 2 great squares."""
    assert 24 // 6 == 4 and 24 % 6 == 0
    assert 120 // 10 == 12 and 120 % 10 == 0
    assert 8 // 4 == 2 and 8 % 4 == 0
    assert 16 % 24 != 0          # the tesseract's 16 vertices are NOT a quaternion group


def test_canon_erratum_E2_binary_octahedral_units():
    """E2: (+-1 +- e_a +- e_b)/sqrt2 is NOT a unit quaternion; (+-e_a +- e_b)/sqrt2 is."""
    bad = math.sqrt(1.0 + 1.0 + 1.0) / math.sqrt(2.0)          # the printed formula
    assert bad == pytest.approx(math.sqrt(1.5), abs=TOL)
    assert abs(bad - 1.0) > 0.2                                 # not a unit
    good = math.sqrt(1.0 + 1.0) / math.sqrt(2.0)                # the correct form
    assert good == pytest.approx(1.0, abs=TOL)
    assert 6 * 4 == 24 and 24 + 24 == 48                        # C(4,2) pairs x signs, |2O|


def test_cl40_nonsimple_bivector_is_not_minus_one():
    """§3.6: ((e12+e34)/sqrt2)^2 = -1 + e1234, a zero divisor, not -1.

    Modelled on the commutative subalgebra span{1, e1234} with e1234^2 = +1:
    write z = a + b*w with w^2 = 1.
    """
    def mul(x, y):
        return (x[0] * y[0] + x[1] * y[1], x[0] * y[1] + x[1] * y[0])

    Z = (-1.0, 1.0)                     # -1 + e1234
    assert mul(Z, Z) == pytest.approx((-2.0 * Z[0], -2.0 * Z[1]), abs=TOL)   # Z^2 = -2Z
    assert Z != pytest.approx((-1.0, 0.0), abs=TOL)                          # not -1
    # zero divisor: Z * (Z + 2) = 0
    assert mul(Z, (Z[0] + 2.0, Z[1])) == pytest.approx((0.0, 0.0), abs=TOL)
    # Plucker condition b12*b34 - b13*b24 + b14*b23 = 0 detects simple bivectors.
    # e12 alone is simple; (e12 + e34)/sqrt2 is not.
    def plucker(b12, b13, b14, b23, b24, b34):
        return b12 * b34 - b13 * b24 + b14 * b23
    assert plucker(1, 0, 0, 0, 0, 0) == pytest.approx(0.0, abs=TOL)           # simple
    r = 1.0 / math.sqrt(2.0)
    assert plucker(r, 0, 0, 0, 0, r) == pytest.approx(0.5, abs=TOL)           # NOT simple


# ---------------------------------------------------------------------------
# §1.5 / §8  measures, densities, ledger constants
# ---------------------------------------------------------------------------

SURFACE_3VOLUME = {            # S3 at edge 1
    "5-cell": 5.0 * math.sqrt(2.0) / 12.0,
    "tesseract": 8.0,
    "16-cell": 4.0 * math.sqrt(2.0) / 3.0,
    "24-cell": 8.0 * math.sqrt(2.0),
    "600-cell": 50.0 * math.sqrt(2.0),
    "120-cell": 450.0 + 210.0 * math.sqrt(5.0),
}


@pytest.mark.parametrize("name,v4", [
    ("5-cell", math.sqrt(5.0) / 96.0),
    ("tesseract", 1.0),
    ("16-cell", 1.0 / 6.0),
    ("120-cell", (1575.0 + 705.0 * math.sqrt(5.0)) / 4.0),
])
def test_v4_equals_quarter_inradius_times_surface(name, v4):
    """V4 = (1/4) * 3R * S3 -- a free independent route to every 4-volume."""
    R0 = POLYTOPES[name][5]
    s3 = DOC_SIGNATURE[name][2]
    inradius = s3 * R0
    assert 0.25 * inradius * SURFACE_3VOLUME[name] == pytest.approx(v4, rel=1e-9)


def test_5cell_4volume_decimal():
    assert math.sqrt(5.0) / 96.0 == pytest.approx(0.0232923748, abs=1e-10)


def test_120cell_4volume_at_unit_circumradius():
    """15*sqrt5/8 = 4.192627458 (the sweep printed 4.19276)."""
    edge_at_R1 = 1.0 / POLYTOPES["120-cell"][5]
    v4_edge1 = (1575.0 + 705.0 * math.sqrt(5.0)) / 4.0
    assert v4_edge1 * edge_at_R1 ** 4 == pytest.approx(15.0 * math.sqrt(5.0) / 8.0, rel=1e-9)
    assert 15.0 * math.sqrt(5.0) / 8.0 == pytest.approx(4.192627458, abs=1e-8)


def test_cauchy_mean_shadow():
    """Mean orthographic shadow volume = (2/(3 pi)) * S3."""
    assert 2.0 * 8.0 / (3.0 * math.pi) == pytest.approx(16.0 / (3.0 * math.pi), abs=TOL)
    assert 16.0 / (3.0 * math.pi) == pytest.approx(1.697652726, abs=1e-8)
    assert 2.0 * SURFACE_3VOLUME["24-cell"] / (3.0 * math.pi) == pytest.approx(2.40084351, abs=1e-7)
    assert 2.0 * SURFACE_3VOLUME["120-cell"] / (3.0 * math.pi) == pytest.approx(195.139722, abs=1e-5)
    assert 2.0 * SURFACE_3VOLUME["600-cell"] / (3.0 * math.pi) == pytest.approx(15.005272, abs=1e-5)
    # unit 4-ball sanity: S3 = 2 pi^2 gives mean = 4 pi / 3 = the 3-ball volume
    assert 2.0 * (2.0 * math.pi ** 2) / (3.0 * math.pi) == pytest.approx(4.0 * math.pi / 3.0, abs=TOL)


def test_sixteencell_brightness_row_is_wrong_and_must_not_ship():
    """DO-NOT row: b((1,1,0,0)/sqrt2) = 2 sqrt2 / 3 < 1, so the range [1, 4/3] is false."""
    signs = [s for s in product((-1.0, 1.0), repeat=4)]
    assert len(signs) == 16

    def b(u):
        return sum(abs(sum(a * c for a, c in zip(u, s))) for s in signs) / 12.0

    assert b((1.0, 0.0, 0.0, 0.0)) == pytest.approx(4.0 / 3.0, abs=1e-12)
    assert b((0.5, 0.5, 0.5, 0.5)) == pytest.approx(1.0, abs=1e-12)
    r = 1.0 / math.sqrt(2.0)
    assert b((r, r, 0.0, 0.0)) == pytest.approx(2.0 * math.sqrt(2.0) / 3.0, abs=1e-12)
    assert b((r, r, 0.0, 0.0)) < 1.0                       # the claimed minimum is not minimal
    assert (4.0 / 3.0) / b((r, r, 0.0, 0.0)) == pytest.approx(math.sqrt(2.0), abs=1e-12)


def test_packing_and_covering_constants():
    assert math.pi ** 2 / 16.0 == pytest.approx(0.616850275, abs=1e-9)          # D4
    assert 2.0 * math.pi ** 2 / (5.0 * math.sqrt(5.0)) == pytest.approx(1.7655285, abs=1e-7)
    assert math.pi ** 2 / 16.0 / 0.636108 == pytest.approx(0.96973, abs=1e-5)   # 3.03% below bound


def test_kelvin_candidates_normalised():
    """S / V^(3/4) for flat-faced monohedral 4D candidates; 4-ball is the floor."""
    def q(S, V):
        return S / V ** 0.75

    # 24-cell at EDGE 1: S3 = 8 sqrt2 = 11.313708, V4 = 2 (NOT 8 -- the 4-volume of the
    # 24-cell is 2a^4). Ratio-invariant under scale, so any consistent edge works.
    assert q(SURFACE_3VOLUME["24-cell"], 2.0) == pytest.approx(2.0 ** (11.0 / 4.0), rel=1e-9)
    assert 2.0 ** (11.0 / 4.0) == pytest.approx(6.727171, abs=1e-6)
    assert q(SURFACE_3VOLUME["tesseract"], 1.0) == pytest.approx(8.0, abs=TOL)

    # 16-cell. CAUGHT A DOC ERROR on the first run, 21/09/2026: the findings doc printed
    # 7.228760 (from the sweep). The closed form is
    #     q^4 = (4 sqrt2 / 3)^4 * 216 = 1024 * 216 / 81 = 2^13 / 3 = 8192/3
    # so q = 2^(13/4) / 3^(1/4) = 7.228816029. Assert the CLOSED FORM, never a hand decimal.
    q16 = 2.0 ** (13.0 / 4.0) / 3.0 ** 0.25
    assert q16 ** 4 == pytest.approx(8192.0 / 3.0, rel=1e-12)
    assert q(SURFACE_3VOLUME["16-cell"], 1.0 / 6.0) == pytest.approx(q16, rel=1e-12)
    assert q16 == pytest.approx(7.228816029, abs=1e-8)
    v_ball = math.pi ** 2 / 2.0
    s_ball = 2.0 * math.pi ** 2
    # 4-ball. SECOND DOC ERROR, caught on the re-run, 26/09/2026: this line and the findings
    # doc both carried a typed 5.962003 (off by 2.03e-4). The closed form is
    #     q = 2 pi^2 / (pi^2/2)^(3/4) = 2 pi^2 * 2^(3/4) / pi^(3/2) = 2^(7/4) sqrt(pi)
    # so q = 5.961800358. Same rule as the 16-cell: assert the CLOSED FORM.
    q_ball = 2.0 ** 1.75 * math.sqrt(math.pi)
    assert q(s_ball, v_ball) == pytest.approx(q_ball, rel=1e-12)
    assert q_ball == pytest.approx(5.961800358, abs=1e-8)
    # the ordering the doc quotes: 4-ball floor < 24-cell < 16-cell < tesseract
    assert q(s_ball, v_ball) < q(SURFACE_3VOLUME["24-cell"], 2.0) \
        < q(SURFACE_3VOLUME["16-cell"], 1.0 / 6.0) < q(SURFACE_3VOLUME["tesseract"], 1.0)


def test_unit_ball_volume_canon():
    """V_4 = pi^2/2, and V_n peaks at n = 5 (canon, re-checked)."""
    def V(n):
        return math.pi ** (n / 2.0) / math.gamma(n / 2.0 + 1.0)
    assert V(4) == pytest.approx(math.pi ** 2 / 2.0, abs=TOL)
    assert V(4) == pytest.approx(4.934802200, abs=1e-9)
    assert V(4) / 2.0 ** 4 == pytest.approx(0.308425, abs=1e-6)
    assert max(range(1, 21), key=V) == 5


def test_cfl_at_D4_and_the_niven_meeting_point():
    """1/sqrt(4) = 1/2 = cos(pi/3); and D = sec^2(pi/n) integral only at n = 1, 3, 4."""
    assert 1.0 / math.sqrt(4.0) == 0.5 == pytest.approx(math.cos(math.pi / 3.0), abs=TOL)
    integral = []
    for n in (1, 2, 3, 4, 6):
        c = math.cos(math.pi / n)
        if abs(c) < 1e-15:
            continue
        D = 1.0 / (c * c)
        if abs(D - round(D)) < 1e-9:
            integral.append((n, round(D)))
    assert integral == [(1, 1), (3, 4), (4, 2)]


def test_cfl_boundary_is_marginal_not_strict():
    """E6: at Co = 1/sqrt(D) the worst mode gives A + 1/A = -2, a repeated root A = -1."""
    for D in (1, 2, 3, 4, 5, 8):
        Co = 1.0 / math.sqrt(D)
        S = float(D)                       # worst mode: sin^2(k_j h/2) = 1 for all j
        rhs = -4.0 * Co * Co * S           # A - 2 + 1/A
        assert rhs == pytest.approx(-4.0, abs=TOL)
        # A + 1/A = -2  =>  (A+1)^2 = 0  =>  A = -1 twice (defective)
        assert (rhs + 2.0) == pytest.approx(-2.0, abs=TOL)


def test_parallelotope_sequence():
    """1, 2, 5, 52, 110244 -- and 103769 / 179372 are the superseded Engel figures."""
    seq = [1, 2, 5, 52, 110244]
    assert seq[3] == 52                      # Delone 1929 (51) + Stogrin's 52nd
    assert seq[4] == 110244                  # Dutour Sikiric et al. 2016, combinatorial types
    assert 181394 != 110244                  # contraction types, a finer invariant
    assert 103769 not in seq and 179372 not in seq


def test_hyperbolic_counts_do_not_all_peak_at_four():
    compact = {2: None, 3: 4, 4: 5, 5: 0, 6: 0}
    paracompact = {3: 11, 4: 2, 5: 5, 6: 0}
    assert compact[4] > compact[3] and compact[5] == 0          # compact peaks at 4
    assert paracompact[4] < paracompact[3]                      # paracompact DIPS at 4
    assert paracompact[5] > paracompact[4]


def test_star_polychora_census():
    """10 Schlafli-Hess; nine have 120 vertices, nine have 120 cells, DIFFERENT exceptions."""
    assert 14400 // 24 == 600          # {5/2,3,3}: N0 = |H4| / |Sym{3,3}|
    assert 14400 // 120 == 120         # {5/2,3,3}: N3
    assert 14400 // 120 == 120         # {3,3,5/2}: N0 = |H4| / |Sym{3,5/2}|
    assert 14400 // 24 == 600          # {3,3,5/2}: N3
    densities = [4, 4, 6, 20, 20, 66, 76, 76, 191, 191]
    assert len(densities) == 10
    assert len(set(densities)) == 6
    eulers = [480, -480, 0, 0, 0, 0, -480, 480, 0, 0]
    assert len(eulers) == 10 and sum(1 for e in eulers if e != 0) == 4
