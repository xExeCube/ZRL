"""
ZRL — N-D Flavor Renderer, tier 0: pytest oracle for every closed form in
ZRL_ND_flavor_renderer_spec_MIRROR.md §2, plus an independent numpy port of the
cube-sphere mesh used by flavor_renderer_3d_MIRROR.html (so the mesh design
decisions — exact extremal vertices, symmetric split, O(h²) chordal error — are
tested here before the browser ledger is trusted).

Run:  pytest test_flavor3d_oracle_MIRROR.py -v
Deps: numpy, pytest (no scipy, no sympy).
"""
import itertools
import math

import numpy as np
import pytest

SQ2, SQ3 = math.sqrt(2), math.sqrt(3)
SQ23, SQ13 = math.sqrt(2 / 3), math.sqrt(1 / 3)
PHI = (1 + math.sqrt(5)) / 2
P_INF = 64                      # spec §2.3: p ≥ 64 → exact max
RNG = np.random.default_rng(20260915)
CUBE_NORMALS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
DIAGS = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], float) / SQ3


# ---------------------------------------------------------------- helpers
def rand_dirs(k):
    v = RNG.normal(size=(k, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def sym26():
    out = [np.array(c, float) for c in itertools.product((-1, 0, 1), repeat=3) if any(c)]
    return np.array([c / np.linalg.norm(c) for c in out])


def norm_p(U, p):
    """Stable p-norm of each row of U: ‖u‖_∞ · (Σ (|u_i|/‖u‖_∞)^p)^(1/p); exact max for p ≥ P_INF."""
    U = np.abs(np.asarray(U, float))
    m = U.max(axis=1)
    if p >= P_INF:
        return m
    return m * (np.sum((U / m[:, None]) ** p, axis=1)) ** (1 / p)


def r_flavor(U, p, a):
    return a / norm_p(U, p)


def sig_face(p):            # ₂R/₀R for the p-ball (§2.3)
    return 3 ** (-abs(0.5 - (0 if p >= P_INF else 1 / p)))


def sig_edge(p):            # ₁R/₀R for the p-ball (derived in the app; not in spec §2.3)
    q = 0 if p >= P_INF else 1 / p
    return (2 / 3) ** (0.5 - q) if p >= 2 else 2 ** (0.5 - q)


def proj111(v):
    eu = np.array([1, -1, 0]) / SQ2
    ev = np.array([1, 1, -2]) / math.sqrt(6)
    v = np.asarray(v, float)
    return np.stack([v @ eu, v @ ev], axis=-1)


def polygon_stats(pts):
    r = np.linalg.norm(pts, axis=1)
    ang = np.sort(np.degrees(np.arctan2(pts[:, 1], pts[:, 0])))
    gaps = (np.roll(ang, -1) - ang) % 360
    return r.min(), r.max(), gaps.min(), gaps.max()


# ---------------------------------------------------------------- §2.1
def test_max_over_normals_is_linf():
    u = rand_dirs(200_000)
    r1 = 0.5 / (u @ CUBE_NORMALS.T).max(axis=1)
    r2 = 0.5 / np.abs(u).max(axis=1)
    assert np.max(np.abs(r1 - r2)) < 1e-12


# ---------------------------------------------------------------- §2.2 k-face radii, n = 3, 4, 5
def kface_ratios_hypercube(n):
    verts = np.array(list(itertools.product((-1, 1), repeat=n)), float)
    R0 = np.linalg.norm(verts[0])
    out = []
    for k in range(n):
        # the k-face whose last n−k coordinates are fixed at +1: 2^k vertices, centroid from the vertex list
        face = verts[np.all(verts[:, k:] == 1, axis=1)]
        assert len(face) == 2 ** k
        out.append(np.linalg.norm(face.mean(axis=0)) / R0)
    return np.array(out)


def kface_ratios_cross(n):
    V = np.vstack([np.eye(n), -np.eye(n)])                # vertices ±e_i, ₀R = 1
    out = []
    for k in range(n):
        face = V[:k + 1]                                  # e_0..e_k: no antipodal pair → a k-face
        out.append(np.linalg.norm(face.mean(axis=0)) / np.linalg.norm(V[0]))
    return np.array(out)


def kface_ratios_simplex(n):
    V = np.eye(n + 1)                                     # regular n-simplex in R^{n+1}
    c = V.mean(axis=0)
    R0 = np.linalg.norm(V[0] - c)
    return np.array([np.linalg.norm(V[:k + 1].mean(axis=0) - c) / R0 for k in range(n)])


@pytest.mark.parametrize("n", [3, 4, 5])
def test_kface_radii_closed_forms(n):
    k = np.arange(n)
    assert np.allclose(kface_ratios_hypercube(n), np.sqrt((n - k) / n), atol=1e-12)
    assert np.allclose(kface_ratios_cross(n), 1 / np.sqrt(k + 1), atol=1e-12)
    assert np.allclose(kface_ratios_simplex(n), np.sqrt((n - k) / (n * (k + 1))), atol=1e-12)


def test_3d_signature_table():
    cube = kface_ratios_hypercube(3)
    octa = kface_ratios_cross(3)
    tet = kface_ratios_simplex(3)
    assert abs(cube[1] - SQ23) < 1e-12 and abs(cube[2] - SQ13) < 1e-12
    assert abs(octa[1] - SQ2 / 2) < 1e-12 and abs(octa[2] - SQ13) < 1e-12
    assert abs(tet[1] - SQ13) < 1e-12 and abs(tet[2] - 1 / 3) < 1e-12
    # dual pair shares the facet entry, differs in the edge entry
    assert abs(cube[2] - octa[2]) < 1e-12 and abs(cube[1] - octa[1]) > 0.1
    # canon identities: √(2/3) = [111]-projected edge factor, 1/√3 = C_max at D = 3
    assert abs(cube[1] - math.sqrt(2 / 3)) < 1e-15 and abs(cube[2] - 1 / math.sqrt(3)) < 1e-15
    # no integer n has cos(π/n) = 1/√3
    n_eff = math.pi / math.acos(SQ13)
    assert abs(n_eff - round(n_eff)) > 0.1


# ---------------------------------------------------------------- §2.3 p-norm family
@pytest.mark.parametrize("p", [1, 1.5, 2, 3, 4, 8, 32, 64, 1e6])
def test_lp_signature(p):
    u = np.vstack([rand_dirs(200_000), sym26()])
    r = r_flavor(u, p, 1.0)
    ratio = r.min() / r.max()
    assert abs(ratio - sig_face(p)) < 1e-9
    e = r_flavor(np.array([[1, 1, 0]]) / SQ2, p, 1.0)[0]
    assert abs(e / r.max() - sig_edge(p)) < 1e-9


def test_lp_signature_spec_values():
    assert abs(sig_face(1.5) - 0.832683) < 1e-6
    assert abs(sig_face(4) - 0.759836) < 1e-6
    # spec §2.3 prints 0.662342 for p = 8 (its sampled ratio); the closed form 3^(−3/8) = 0.6623378 — spec erratum
    assert abs(sig_face(8) - 0.662338) < 1e-6
    assert abs(sig_face(1) - SQ13) < 1e-12 and abs(sig_face(64) - SQ13) < 1e-12


def test_naive_power_underflows_at_large_p():
    u = np.vstack([rand_dirs(1000), sym26()])
    p = 1e6
    with np.errstate(under="ignore", divide="ignore", invalid="ignore"):
        naive = (np.sum(np.abs(u) ** p, axis=1)) ** (1 / p)      # every |u_i|^p underflows to 0 unless |u_i| = 1
    # |u_i|^1e6 underflows to exactly 0 only for |u_i| < 1 − ~7.4e-4 (2^-1074 = e^-744); stay clear of the axes
    generic = np.all(np.abs(u) < 0.999, axis=1)
    assert np.all(naive[generic] == 0)                           # spec §2.3: the naive form returns 0
    stable = norm_p(u, p)
    assert np.all(np.isfinite(stable)) and np.all(stable > 0)
    assert abs((1 / stable).min() / (1 / stable).max() - SQ13) < 1e-9


# ---------------------------------------------------------------- §2.4 octant pieces
def test_cube_octant_pieces():
    a = 0.5                                       # edge 1
    # patch = three quarter-faces, each a square of side a: area from the corner coordinates
    quarter = lambda o, e1, e2: np.linalg.norm(np.cross(e1 - o, e2 - o))
    faces = [(np.array([a, 0, 0]), np.array([a, a, 0]), np.array([a, 0, a])),
             (np.array([0, a, 0]), np.array([0, a, a]), np.array([a, a, 0])),
             (np.array([0, 0, a]), np.array([a, 0, a]), np.array([0, a, a]))]
    patch = sum(quarter(*f) for f in faces)
    assert abs(patch - 0.75) < 1e-15 and abs(8 * patch - 6) < 1e-15
    # carrier loop = the midline hexagon face-centre → edge-midpoint (NOT cube half-edges)
    loop = np.array([[a, 0, 0], [a, a, 0], [0, a, 0], [0, a, a], [0, 0, a], [a, 0, a]])
    L = sum(np.linalg.norm(loop[(i + 1) % 6] - loop[i]) for i in range(6))
    assert abs(L - 3) < 1e-15
    # the cube's own half-edges from the corner (a,a,a) are interior spokes, not loop edges
    corner = np.array([a, a, a])
    for tip in loop[1::2]:
        assert abs(np.linalg.norm(tip - corner) - a) < 1e-15


def spherical_excess(u1, u2, u3):
    num = abs(np.dot(u1, np.cross(u2, u3)))
    den = 1 + np.dot(u1, u2) + np.dot(u2, u3) + np.dot(u3, u1)
    return 2 * math.atan2(num, den)


def test_sphere_octant_pieces():
    a = 1.0
    # octant area by geodesic subdivision of the octant triangle (exact at any level)
    m = 12
    tri = []
    for i in range(m):
        for j in range(m - i):
            P = lambda x, y: np.array([x, y, m - x - y], float) / np.linalg.norm([x, y, m - x - y])
            tri.append((P(i, j), P(i + 1, j), P(i, j + 1)))
            if i + j < m - 1:
                tri.append((P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)))
    area = sum(spherical_excess(*t) for t in tri) * a * a
    assert abs(area - math.pi / 2) < 1e-12
    assert abs(8 * area - 4 * math.pi) < 1e-11
    # loop: three quarter great circles, chord-summed (converges O(h²) to 3πa/2)
    t = np.linspace(0, math.pi / 2, 20001)
    arc = lambda P: np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))
    z = 0 * t
    L = arc(np.stack([np.cos(t), np.sin(t), z], 1)) + arc(np.stack([z, np.cos(t), np.sin(t)], 1)) + arc(np.stack([np.sin(t), z, np.cos(t)], 1))
    assert abs(L - 1.5 * math.pi * a) < 1e-8
    # Gauss–Bonnet: corner angle at (1,0,0) between the tangents of the two arcs meeting there
    t1 = np.array([0, 1, 0.0])                    # d/dt (cos t, sin t, 0) at t = 0
    t3 = -np.array([0, 0, -1.0])                  # arriving arc (sin t, 0, cos t) at t = π/2, reversed
    corner = math.acos(np.dot(t1, t3))
    assert abs(3 * corner - math.pi - math.pi / 2) < 1e-15 and abs(area - math.pi / 2) < 1e-12


# ---------------------------------------------------------------- §2.5 [111] projection
def test_projection_111_octant_hexagon_and_whole_cube():
    a = 0.5
    loop = np.array([[a, 0, 0], [a, a, 0], [0, a, 0], [0, a, a], [0, 0, a], [a, 0, a]])
    rmin, rmax, gmin, gmax = polygon_stats(proj111(loop))
    assert abs(rmin - a * SQ23) < 1e-12 and abs(rmax - a * SQ23) < 1e-12
    assert abs(gmin - 60) < 1e-9 and abs(gmax - 60) < 1e-9
    assert abs(a * SQ23 - 0.408248) < 1e-6
    corners = np.array(list(itertools.product((-a, a), repeat=3)))
    P = proj111(corners)
    r = np.linalg.norm(P, axis=1)
    assert np.sum(r < 1e-12) == 2                            # the (1,1,1) pair collapses to the centre
    outer = P[r > 1e-12]
    rmin, rmax, gmin, gmax = polygon_stats(outer)
    assert abs(rmin - 2 * a * SQ23) < 1e-12 and abs(rmax - 2 * a * SQ23) < 1e-12
    assert abs(gmin - 60) < 1e-9 and abs(gmax - 60) < 1e-9
    assert abs(2 * a * SQ23 - 0.816497) < 1e-6                 # parent §2.7 hexagon side / cube edge
    assert abs(math.degrees(math.acos(SQ13)) - 54.7356) < 1e-4


# ---------------------------------------------------------------- §2.6 V_n table
def V_n(n):
    return math.pi ** (n / 2) / math.gamma(n / 2 + 1)


def test_Vn_table():
    V = {n: V_n(n) for n in range(1, 13)}
    for n, val in {1: 2, 2: 3.141593, 3: 4.188790, 4: 4.934802, 5: 5.263789, 6: 5.167713,
                   7: 4.724766, 8: 4.058712, 10: 2.550164}.items():
        assert abs(V[n] - val) < 1e-6
    assert max(V, key=V.get) == 5
    assert abs(V[10] / 2 ** 10 - 0.002490) < 1e-6                 # the parent's erratum, confirmed
    ratios = [V[n] / 2 ** n for n in range(1, 13)]
    assert all(ratios[i + 1] < ratios[i] for i in range(11))     # monotone decrease (separate statement)
    assert abs(V[3] / 8 - 0.523599) < 1e-6 and abs(V[2] / 4 - 0.785398) < 1e-6


# ---------------------------------------------------------------- §2.7 groups
def platonic_vertices():
    cyc = lambda pts: [np.roll(p, s) for p in pts for s in range(3)]
    tet = [np.array(v, float) for v in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]]
    cube = [np.array(v, float) for v in itertools.product((-1, 1), repeat=3)]
    octa = [s * e for e in np.eye(3) for s in (-1, 1)]
    ico = cyc([np.array([0, s1, s2 * PHI]) for s1 in (-1, 1) for s2 in (-1, 1)])
    dod = cube + cyc([np.array([0, s1 / PHI, s2 * PHI]) for s1 in (-1, 1) for s2 in (-1, 1)])
    return {"tetrahedron": tet, "cube": cube, "octahedron": octa, "icosahedron": ico, "dodecahedron": dod}


def closed_under_negation(V):
    S = {tuple(np.round(v, 9)) for v in V}
    return all(tuple(np.round(-v, 9)) in S for v in V)


def test_only_the_tetrahedron_lacks_inversion():
    P = platonic_vertices()
    assert not closed_under_negation(P["tetrahedron"])
    for name in ("cube", "octahedron", "icosahedron", "dodecahedron"):
        assert closed_under_negation(P[name]), name
    # the tetrahedron does keep the V4 sub-part: π rotations about the three axes
    S = {tuple(v) for v in P["tetrahedron"]}
    for ax in range(3):
        for v in P["tetrahedron"]:
            w = -v.copy(); w[ax] = v[ax]
            assert tuple(w) in S


def test_group_orders_and_bits():
    assert abs(math.log2(8) - 3) < 1e-15
    assert abs(math.log2(24) - 4.585) < 1e-3
    assert abs(math.log2(48) - 5.585) < 1e-3


# ---------------------------------------------------------------- §2.8 the Dehn wall (Niven route)
def test_tetrahedron_dihedral_is_niven_irrational():
    dihedral = math.acos(1 / 3)
    assert abs(math.degrees(dihedral) - 70.528779) < 1e-6
    # same number as the angle between adjacent cube body diagonals
    assert abs(np.dot(DIAGS[0], DIAGS[1]) - 1 / 3) < 1e-15
    # Niven: cos(θ) rational with θ a rational multiple of π ⇒ cos θ ∈ {0, ±1/2, ±1}
    assert 1 / 3 not in (0, 0.5, -0.5, 1, -1)
    # the cube's dihedral is π/2 — Dehn invariant 0 (cos = 0 IS in Niven's set)
    assert math.cos(math.pi / 2) < 1e-15


# ---------------------------------------------------------------- §2.9 regular sections of the cube
def cube_section(n, c):
    """Polygon of {n·x = c} ∩ cube [−1,1]³, ordered by angle in the plane."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    V = np.array(list(itertools.product((-1, 1), repeat=3)), float)
    pts = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(V[i] != V[j]) != 1:
                continue
            si, sj = n @ V[i] - c, n @ V[j] - c
            if si * sj < 0:
                t = si / (si - sj)
                pts.append(V[i] + t * (V[j] - V[i]))
    pts = np.array(pts)
    if len(pts) < 3:
        return pts
    ctr = pts.mean(axis=0)
    e1 = pts[0] - ctr; e1 -= n * (e1 @ n); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    ang = np.arctan2((pts - ctr) @ e2, (pts - ctr) @ e1)
    return pts[np.argsort(ang)]


def is_regular(poly, tol=1e-9):
    k = len(poly)
    sides = [np.linalg.norm(poly[(i + 1) % k] - poly[i]) for i in range(k)]
    angs = []
    for i in range(k):
        u = poly[i - 1] - poly[i]; v = poly[(i + 1) % k] - poly[i]
        angs.append(math.acos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))
    return max(sides) - min(sides) < tol and max(angs) - min(angs) < tol


def test_cube_regular_sections_are_3_4_6():
    hexagon = cube_section([1, 1, 1], 0.0)
    assert len(hexagon) == 6 and is_regular(hexagon)
    triangle = cube_section([1, 1, 1], 2.0 / SQ3)          # x + y + z = 2 cuts the corner (1,1,1)
    assert len(triangle) == 3 and is_regular(triangle)
    square = cube_section([0, 0, 1], 0.0)
    assert len(square) == 4 and is_regular(square)


def test_pentagonal_sections_have_parallel_edges_hence_never_regular():
    found = 0
    for normal in ([1, 0.6, 0.3], [1, 0.8, 0.5], [0.9, 0.7, 0.4]):
        for c in np.linspace(-1.2, 1.2, 241):
            poly = cube_section(normal, c)
            if len(poly) != 5:
                continue
            found += 1
            edges = [poly[(i + 1) % 5] - poly[i] for i in range(5)]
            edges = [e / np.linalg.norm(e) for e in edges]
            par = sum(1 for i in range(5) for j in range(i + 1, 5) if abs(abs(edges[i] @ edges[j]) - 1) < 1e-9)
            assert par >= 2                                    # two parallel face-pairs among five faces
            assert not is_regular(poly, tol=1e-6)
    assert found > 0


# ---------------------------------------------------------------- rotors: Cl(3,0) port with the e31 storage trap
def popcount(x):
    return bin(x).count("1")


def blade_sign(a, b):
    s, x = 0, a >> 1
    while x:
        s += popcount(x & b); x >>= 1
    return -1 if s & 1 else 1


def gmul(A, B):
    out = np.zeros(8)
    for a in range(8):
        if A[a] == 0: continue
        for b in range(8):
            if B[b] == 0: continue
            out[a ^ b] += A[a] * B[b] * blade_sign(a, b)
    return out


def grev(A):
    return np.array([A[m] * (-1 if popcount(m) in (2, 3) else 1) for m in range(8)])


def rotor(n, th):
    c, s = math.cos(th / 2), math.sin(th / 2)
    R = np.zeros(8); R[0] = c; R[6] = -s * n[0]; R[5] = s * n[1]; R[3] = -s * n[2]   # mask 5 = e13 = −e31
    return R


def sandwich(R, v):
    V = np.zeros(8); V[1], V[2], V[4] = v
    out = gmul(gmul(R, V), grev(R))
    return np.array([out[1], out[2], out[4]]), np.hypot.reduce([out[0], out[3], out[5], out[6], out[7]])


def rodrigues(n, th, v):
    return v * math.cos(th) + np.cross(n, v) * math.sin(th) + n * (n @ v) * (1 - math.cos(th))


def quat_rotate(q, v):                      # q = (x, y, z, w); v' = v + 2 w (q×v) + 2 q×(q×v)
    x, w = q[:3], q[3]
    t = 2 * np.cross(x, v)
    return v + w * t + np.cross(x, t)


def test_rotor_sandwich_equals_rodrigues_and_quaternion():
    for _ in range(50):
        n = RNG.normal(size=3); n /= np.linalg.norm(n)
        th = RNG.uniform(0, 2 * math.pi); v = RNG.uniform(-1, 1, 3)
        R = rotor(n, th)
        sw, impure = sandwich(R, v)
        assert np.linalg.norm(sw - rodrigues(n, th, v)) < 1e-12 and impure < 1e-12
        q = np.array([-R[6], R[5], -R[3], R[0]])         # rotorToQuat in the app
        assert abs(np.linalg.norm(q) - 1) < 1e-12
        assert np.linalg.norm(sw - quat_rotate(q, v)) < 1e-12
        RR = gmul(R, grev(R)); assert abs(RR[0] - 1) < 1e-12 and np.linalg.norm(RR[1:]) < 1e-12


def test_isomer_rotors_land_on_z():
    z = np.array([0, 0, 1.0])
    for d in ([1, 1, 1], [1, 1, 0]):
        d = np.array(d, float); d /= np.linalg.norm(d)
        n = np.cross(d, z); n /= np.linalg.norm(n)
        sw, _ = sandwich(rotor(n, math.acos(d @ z)), d)
        assert np.linalg.norm(sw - z) < 1e-12


# ---------------------------------------------------------------- §7 sampling
def fib_dirs(K):
    i = np.arange(K); z = 1 - (2 * i + 1) / K; r = np.sqrt(np.maximum(0, 1 - z * z)); ph = i * math.pi * (3 - math.sqrt(5))
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], axis=1)


def nn_spacing_cv(P):
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    return nn.std() / nn.mean()


def test_fibonacci_sphere_uniformity_vs_latlong():
    assert nn_spacing_cv(fib_dirs(2000)) < 0.02
    th = np.linspace(0, math.pi, 41)[1:-1]; ph = np.linspace(0, 2 * math.pi, 51)[:-1]
    T, Pp = np.meshgrid(th, ph, indexing="ij")
    LL = np.stack([np.sin(T) * np.cos(Pp), np.sin(T) * np.sin(Pp), np.cos(T)], axis=-1).reshape(-1, 3)
    assert nn_spacing_cv(LL) > 0.25


def test_kepler_gap():
    assert abs(math.pi / math.sqrt(18) - 0.740480) < 1e-6
    assert abs(1 - math.pi / math.sqrt(18) - 0.259520) < 1e-6


# ---------------------------------------------------------------- the cube-sphere mesh (numpy port of the app's buildMesh)
def cube_sphere(N, p, a):
    assert N % 2 == 0
    h = N // 2
    verts, pos, cp = {}, [], []

    def vid(c):
        key = tuple(c)
        if key in verts:
            return verts[key]
        i = len(pos); verts[key] = i
        u = np.array(c, float); u /= np.linalg.norm(u)
        r = r_flavor(u[None, :], p, a)[0]
        pos.append(r * u); cp.append(list(c))
        return i

    tris = []
    for f in range(6):
        ax, sg = f >> 1, (-1 if f & 1 else 1)
        b1, b2 = (ax + 1) % 3, (ax + 2) % 3

        def P(i, j):
            c = [0, 0, 0]; c[ax] = sg * h; c[b1] = -h + i; c[b2] = -h + j
            return vid(c)

        for i in range(N):
            for j in range(N):
                v00, v10, v11, v01 = P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)
                d1 = np.linalg.norm((pos[v00] + pos[v11]) / 2); d2 = np.linalg.norm((pos[v10] + pos[v01]) / 2)
                if abs(d1 - d2) > 1e-12 * a:
                    use1 = d1 > d2
                else:
                    cw = lambda x: abs(cp[x][b1]) + abs(cp[x][b2])
                    use1 = max(cw(v00), cw(v11)) >= max(cw(v10), cw(v01))
                tris += [(v00, v10, v11), (v00, v11, v01)] if use1 else [(v10, v11, v01), (v10, v01, v00)]
    pos, cp, tris = np.array(pos), np.array(cp), np.array(tris)
    A, B, C = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(B - A, C - A), A + B + C) < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return pos, cp, tris


def integrals(pos, tris):
    A, B, C = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    V = np.einsum("ij,ij->i", A, np.cross(B, C)).sum() / 6
    S = (np.linalg.norm(np.cross(B - A, C - A), axis=1) / 2).sum()
    E = {tuple(sorted(e)) for t in tris for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    return V, S, len(pos), len(E), len(tris)


def octant(cp, tris):
    return tris[np.all(cp[tris] >= 0, axis=(1, 2))]


def boundary_edges(tris):
    cnt = {}
    for t in tris:
        for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = tuple(sorted(e)); cnt[k] = cnt.get(k, 0) + 1
    return [k for k, c in cnt.items() if c == 1]


def test_mesh_euler_and_counts():
    pos, cp, tris = cube_sphere(8, 2, 1.0)
    V, S, nV, nE, nF = integrals(pos, tris)
    assert nV == 6 * 64 + 2 and nF == 12 * 64 and nE == 18 * 64 and nV - nE + nF == 2


def test_mesh_cube_is_exact():
    pos, cp, tris = cube_sphere(8, math.inf, 0.5)
    V, S, *_ = integrals(pos, tris)
    assert abs(V - 1) < 1e-12 and abs(S - 6) < 1e-12
    r = np.linalg.norm(pos, axis=1)
    assert abs(r.max() - 0.5 * SQ3) < 1e-12 and abs(r.min() - 0.5) < 1e-12
    edge_ids = [i for i in range(len(pos)) if sorted(np.abs(cp[i])) == [0, 4, 4]]
    assert len(edge_ids) == 12 and np.ptp(r[edge_ids]) < 1e-12 and abs(r[edge_ids][0] - 0.5 * SQ2) < 1e-12
    T = octant(cp, tris)
    Vo, So, *_ = integrals(pos, T)
    assert abs(Vo - 1 / 8) < 1e-12 and abs(So - 0.75) < 1e-12
    be = boundary_edges(T)
    L = sum(np.linalg.norm(pos[i] - pos[j]) for i, j in be)
    assert abs(L - 3) < 1e-12
    bverts = {v for e in be for v in e}
    for c in ([4, 0, 0], [4, 4, 0], [0, 4, 0], [0, 4, 4], [0, 0, 4], [4, 0, 4]):
        assert any(np.array_equal(cp[v], c) for v in bverts)
    corner = next(v for v in range(len(pos)) if np.array_equal(cp[v], [4, 4, 4]))
    assert corner not in bverts                              # the corner is interior to the patch


def test_mesh_octahedron_is_exact():
    pos, cp, tris = cube_sphere(8, 1, 1.0)
    V, S, *_ = integrals(pos, tris)
    assert abs(V - 4 / 3) < 1e-12 and abs(S - 4 * SQ3) < 1e-12
    T = octant(cp, tris)
    Vo, So, *_ = integrals(pos, T)
    assert abs(So - SQ3 / 2) < 1e-12
    L = sum(np.linalg.norm(pos[i] - pos[j]) for i, j in boundary_edges(T))
    assert abs(L - 3 * SQ2) < 1e-12


@pytest.mark.parametrize("p", [2, 1.5, 4])
def test_mesh_octants_are_exact_mirror_images(p):
    pos, cp, tris = cube_sphere(8, p, 1.0)
    V, S, *_ = integrals(pos, tris)
    T = octant(cp, tris)
    Vo, So, *_ = integrals(pos, T)
    assert abs(Vo / V - 1 / 8) < 1e-12 and abs(8 * So / S - 1) < 1e-12


def test_mesh_sphere_chordal_error_is_second_order_and_within_ledger_tolerance():
    errs = {}
    for N in (8, 16, 32):
        pos, cp, tris = cube_sphere(N, 2, 1.0)
        V, S, *_ = integrals(pos, tris)
        errs[N] = (abs(V - 4 * math.pi / 3) / (4 * math.pi / 3), abs(S - 4 * math.pi) / (4 * math.pi))
        assert errs[N][0] < 3 / N ** 2 and errs[N][1] < 3 / N ** 2       # the app's curved tolerance 3/N²
    for k in (0, 1):
        order = math.log2(errs[8][k] / errs[16][k])
        assert 1.8 < order < 2.2
    # geodesic-arc loop length and Gauss–Bonnet on the drawn vertices are exact
    pos, cp, tris = cube_sphere(16, 2, 1.0)
    T = octant(cp, tris)
    arc = sum(math.acos(np.clip(pos[i] @ pos[j], -1, 1)) for i, j in boundary_edges(T))
    assert abs(arc - 1.5 * math.pi) < 1e-9
    exc = sum(spherical_excess(pos[a] / np.linalg.norm(pos[a]), pos[b] / np.linalg.norm(pos[b]), pos[c] / np.linalg.norm(pos[c])) for a, b, c in T)
    assert abs(exc - math.pi / 2) < 1e-9


def test_mesh_extremal_directions_are_vertices_so_signature_is_exact():
    for p in (1.5, 2, 4, 8):
        pos, cp, tris = cube_sphere(8, p, 1.0)
        r = np.linalg.norm(pos, axis=1)
        assert abs(r.min() / r.max() - sig_face(p)) < 1e-12
        edge_ids = [i for i in range(len(pos)) if sorted(np.abs(cp[i])) == [0, 4, 4]]
        assert abs(r[edge_ids].mean() / r.max() - sig_edge(p)) < 1e-12
