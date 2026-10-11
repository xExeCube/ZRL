"""ZRL 2D wave membrane -- Taichi port: the physics LEDGER.

A value-for-value mirror of the web app's ledger
(FormConstantsCLAUDE/FlavorRenderersCLAUDE/wave_membrane_MIRROR.html, the merged build of 28/09/2026,
section 3, lines ~1687-2487): fmt / fe / st, symmetryResidual, mirrorSymmetric, mirrorResidual,
symRotResidual, frontFree, frontRadii, srcCentred, coneWall, coneMeasure, energyStats, buildLedger,
and what they call (clearanceFrom / CLEAR, CONE_WALL, coneNorm, reachInradius, stencilSymbol,
shapeArea / idealArea / circleClipArea, the probes pA / pB, MODEP, HARM). The edge-alignment rows
7c / 7c-bis come from alignment_MIRROR.al_ledger_rows (checked against the JS there).

Rows are Row tuples whose first five fields are (name, status, value, note, tag):
  name    the row's label (plain unicode: the web app's HTML entities decoded, tags dropped)
  status  PASS / FAIL / NA / stated  (the web app's badge folded into four classes: 'RED OK' is a
          PASS -- a run past the CFL limit that DID diverge --, 'not yet' a FAIL; 'measured',
          'stated' and 'visual' are informational -> 'stated')
  value   a compact measured value ('' when the row is n/a)
  note    the web app's detail text, word for word (numbers from this run)
  tag     the web app's last column (the tier / reference)
and then key (a stable id), badge (the web app's own status text), dsg (the label is a design
statement -- the web app's 'dsg' style), num (the primary measured number or None), port (a
port-only remark: solids, the 'unlicensed' plate, precision, the recorder) and fixes (the ids of
the FIXES that changed this row).

Checked against the web app's own buildLedger() by crosscheck_ledger_MIRROR.py (Node harness,
ToolsCLAUDE/js_ledger_dump_MIRROR.mjs). NotesCLAUDE/ledger_MIRROR.md has the row-by-row decisions.

Port-only situations (the JS has no case for them; decided row by row, 30/09/2026):
  - SOLIDS (solids_MIRROR.py): a row whose premise is the pristine flavor domain reports NA with the
    reason, instead of a false FAIL -- the drum spectrum, the eigenmode and dispersion rows, the
    probe rows that assume the only reflector is the edge (boundary behaviour, the 2D wake). The
    symmetry rows test their premise directly: the free-cell mask must be invariant under the
    orbit (checked cell by cell, exact) and the source cell must be a fixed point -- then they are
    graded as usual, so a symmetric arrangement of solids is still a solver test. Exact counts
    (conformance, staircase) count the DOMAIN mask (solids excluded: they do not change how the
    flavor shape is rasterised). Energy, CFL, linearity, the light cone and the front anisotropy
    stay graded: pinned walls conserve the energy, and CLEAR / CONE_WALL are measured on the
    free-cell mask, so the front and cone rows see the solids as walls.
  - 'unlicensed' (the 2:1 plate): the JS logic has no case in mirrorSymmetric (it parses the shape
    name as an n-gon) -- the mirror premise is the exact mask check instead; the staircase row uses
    the plate's closed-form area W H - (4 - pi) r^2. Every other row's JS logic already handles it
    (not 8-fold, not a lattice rotation, no closed-form spectrum, not the square's mode).
  - the RECORDER switched off (m.record = False) for some steps: the history rows (HIST_ROWS: energy,
    boundary, wake, dispersion, linearity) are n/a with the reason instead of grading a gapped series
    (the history is complete iff it has one entry per step, capped).
  - WALLS and the TRIANGULAR-LATTICE MODES (05/10/2026, NotesCLAUDE/walls_MIRROR.md sections 9 and 13):
    rigid solids and a rigid edge conserve the energy like pinned ones (the energy row stays graded,
    its port text says so); the symmetry rows also check the TYPE map (left pinned / right rigid is
    not symmetric); the masked square with a rigid edge has cos.cos modes, so the eigenmode rows are
    n/a; the 'array' shape is the full square on the square lattice (every row as for it) and the
    whole rhombic array on the triangular one: conformance counts (N-2)^2 (N^2 on the torus) lattice
    points, the staircase is exact, the mirror holds on the torus only, the 6-fold orbit holds on the
    torus (its period lattice is 6-fold) and is checked with the wrap there, as are the light cone
    (the hexagonal norm's nearest image) and the front clearance (the seam); Mur on the triangular
    array falls back to the clamped ring (the rows say so: edge-kind). Every web-app configuration
    keeps the web app's logic.
  - f32 builds (Vulkan, --f32): the exact-identity tolerances are the web app's f64 ones scaled to
    single precision (TOL_F32, measured); the cone's underflow threshold is 1e-30 instead of 1e-290.
  - ABSORBING BOUNDARIES (09-10/10/2026, NotesCLAUDE/absorbing_MIRROR.md): Mur 2nd order, the CPML and
    sponge frames of the full array and a masked shape's OPEN edge (a sponge outside the drawn edge) are
    port-only. The rows describe the OPEN domain (the drawn shape, the array minus a frame): the recorder
    sums it only, the counts (conformance, staircase) count it, the front clearance stops at the layer.
    Rows whose premise is a closed, lossless domain go n/a with the reason (energy, the drum spectrum, the
    eigenmode and dispersion rows); the boundary row reads "absorbing edge: little comes back", the wake
    and linearity rows accept them as the absorbing boundary; the symmetry rows check their premise on
    the layer's damping map cell by cell (a quantised distance ramp need not be symmetric). One extra row,
    'eopen' ("energy inside the open domain does not grow"), is added for these boundaries only, so every
    web-app configuration keeps the web app's row list.

FIXES (applied unless build_ledger(..., compat=True)): JS ledger faults found while porting. The
cross-check runs compat=True for value-for-value agreement, and compat=False to list what the fixes
change (every such row carries the fix ids in Row.fixes).
"""
import math
import os
import sys
import textwrap
import time
import weakref
from typing import NamedTuple, Optional, Tuple

import numpy as np
import taichi as ti

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import membrane_core_MIRROR as mc          # noqa: E402
import alignment_MIRROR as al               # noqa: E402

PI = math.pi
TAU = 2 * PI
S3H = math.sqrt(3) / 2
SQRT2 = math.sqrt(2)                        # Math.SQRT2 (the same double)
SQRT1_2 = math.sqrt(0.5)                    # Math.SQRT1_2

PASS, FAIL, NA, STATED = 'PASS', 'FAIL', 'NA', 'stated'
BADGE_STATUS = {'PASS': PASS, 'FAIL': FAIL, 'n/a': NA, 'RED OK': PASS, 'not yet': FAIL,
                'measured': STATED, 'stated': STATED, 'visual': STATED}

# f32 builds: the web app's tolerances assume f64. Measured 30/09/2026 (crosscheck_ledger --f32,
# CUDA f32, see NotesCLAUDE/ledger_MIRROR.md) -- the largest f32 residuals stay well below these.
TOL_F32 = {'sym': 1e-4, 'disp': 1e-5}
UNDERFLOW = {True: (1e-290, 'float64 (< 1e-290)'), False: (1e-30, 'float32 (< 1e-30)')}

FIXES = {
    'edge-kind': 'The BC control acts on the ARRAY RING only (the web app says so itself: every masked '
                 'domain is Dirichlet on the mask edge, the triangular lattice always clamps its ring). '
                 'The web app graded "absorbing edge: little comes back", excused the energy row and '
                 'required "the ABSORBING boundary" for the wake by S.bc alone, so a masked hexagon '
                 'with Mur selected read FAIL (the wave does come back from its clamped wall), and a '
                 'closed masked cavity was never energy-tested. Here the edge is what bounds the '
                 'domain: Mur only where the free cells reach the ring (sq lattice).',
    'tri-diagonal': 'Front anisotropy on the TRIANGULAR lattice: the index diagonal (1,1) is 30 deg '
                    'from the x axis and sqrt3 long, not sqrt2; the web app sampled r/sqrt2 steps and '
                    'labelled them dd*sqrt2, so the ratio read sqrt(2/3) = 0.82 of the true value '
                    '(measured 30/09/2026, tri hexagon at step 50: 0.8081 -> 0.9897).',
    'sqfull-side': 'Staircase row on the FULL-GRID square: the text quoted the masked square\'s R '
                   '(e.g. "the side is 157 whole cells" at N = 161) while the domain is the whole '
                   'N x N array.',
    'lens-text': 'Refraction row: the lens medium is 0.55 c, not "the slab (c/2 there)".',
    'mirror-live': 'Mirror row: when the run is not live (before step 6, or a diverged / zero field) the '
                   'web app said "needs a pulse, impulse or vertex source" even when the source was one, and '
                   '"source y != 0" for a vertex source that ignores y; here it says why.',
    'cfl-nan': 'CFL row past the limit: once the field is ALL NaN, record()\'s max skips NaN and '
               'S.amax reads 0, so the row said "not yet" (red) after a divergence; here a non-finite '
               'energy also counts as diverged.',
    'cfl-damped': 'CFL row with damping (01/10/2026): the damping term -d (u - u_prev) of the update lowers the limit to '
                  'C_max(d) = C_max(0) sqrt(1 - d/2) (0.7036 on sq5 at d = 0.02); the web app graded Co against '
                  'C_max(0), so a run between the two blew up under a "stable" PASS row (a false FAIL once it '
                  'diverged). Here the row uses the damped limit (membrane_core co_limit()).',
    'tri-energy': ("Energy row on the TRIANGULAR lattice (05/10/2026): record()'s energy sums the forward "
                   "differences along the index axes (1,0) and (0,1) only -- the SQUARE lattice's functional. On "
                   "the triangular lattice that is not the conserved quantity: it sloshes with the direction of "
                   "the field's gradients (a directed packet, the rhombic array's anisotropic modes), while "
                   "the solver conserves its exact discrete energy to ~1e-15. The port also records the third "
                   "link direction (1,-1) (membrane_core rec_fold, m.hist['Etri']) and grades "
                   "E = kin/(2 Co^2) + (1/3)(gx^2 + gy^2 + gz^2), the triangular operator's own Dirichlet "
                   "form -(u, L u) / 2."),
}


class Row(NamedTuple):
    name: str
    status: str
    value: str
    note: str
    tag: str
    key: str = ''
    badge: str = ''
    dsg: bool = False
    num: Optional[float] = None
    port: str = ''
    fixes: Tuple[str, ...] = ()


def _row(key, name, badge, note, tag, value='', num=None, dsg=False, port='', fixes=()):
    return Row(name, BADGE_STATUS[badge], value if badge != 'n/a' else '', note, tag, key, badge, bool(dsg),
               num, port, tuple(fixes))


# ---- the web app's number formatting ----------------------------------------------------------
fmt = al.fmt                                # fmt(x, d=6): em dash for null / non-finite, else toFixed


def fe(x):
    """fe(x): em dash for null / non-finite, else toExponential(2)."""
    if x is None or not math.isfinite(x):
        return '—'
    return al.js_to_exponential(x, 2)


def st(r, t):
    """st(r, t): PASS iff r <= t (NaN -> FAIL, as in JS)."""
    return 'PASS' if r <= t else 'FAIL'


s_ = al.js_num_str                          # `${x}` for a number
jround = mc.js_round
jhypot = al.js_hypot


# ---- per-Membrane geometry cache ----------------------------------------------------------------
# Everything here depends on the mask, the source cells and the shape, not on the field: CLEAR,
# CONE_WALL, the boundary set they scan, the cell counts, the areas, the mask symmetry checks. It is
# recomputed when the key changes (a reseed builds new mask arrays; update_solids edits in place
# but bumps the scene's version and n_solid).
_GEO = weakref.WeakKeyDictionary()


def _geo_key(m):
    p = m.p
    sc = m.scene
    return (m.N, p.lattice, p.stencil, p.shape, p.rot, p.rad, p.rtheta, p.bc, p.edge, p.src, p.sx, p.sy, p.sigma,
            p.vtx_inset, id(m.mask_np), id(m.domain_mask_np), None if sc is None else sc.version,
            m.n_solid, tuple(m.src[:2]), tuple(tuple(v[:2]) for v in m.vtx),
            id(getattr(m, 'abs_np', None)))          # 10/10/2026: an absorbing layer's codes (the open domain)


class _Geo(dict):
    def get_or(self, name, fn):
        if name not in self:
            self[name] = fn()
        return self[name]


def _geo(m):
    k = _geo_key(m)
    hit = _GEO.get(m)
    if hit is None or hit[0] != k:
        hit = (k, _Geo())
        _GEO[m] = hit
    return hit[1]


def _boundary_cells(M):
    """Masked-out cells with a free cell among their 8 index neighbours: (I, J) arrays.
    The nearest masked-out cell to a free cell p -- by physical distance (clearanceFrom) and by
    every cone norm (coneWall) -- lies in this set: from any masked-out q != p, one of the index
    steps toward p ((sa,0), (0,sb), (sa,sb), sa/sb the signs of p - q) stays in the array and is
    strictly nearer on both lattices (checked for the triangular metric a^2+ab+b^2 case by case),
    so a nearest q cannot have all its 8 neighbours masked out. That makes the scan O(perimeter)
    instead of O(N^2) per source (12 vertex sources at N = 8193)."""
    free = M != 0
    n = free.shape[0]
    d = np.zeros_like(free)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            a0, a1 = max(0, di), n + min(0, di)
            b0, b1 = max(0, dj), n + min(0, dj)
            d[a0 - di:a1 - di, b0 - dj:b1 - dj] |= free[a0:a1, b0:b1]
    return np.nonzero(d & ~free)


def _cell_xy_np(m, I, J):
    c = m.c()
    da, db = I.astype(np.float64) - c, J.astype(np.float64) - c
    return (da + db * 0.5, db * S3H) if m.tri() else (da, db)


def _open_free(m):
    """The free cells of the OPEN domain (10/10/2026): mask_np without an absorbing layer's cells (Membrane.open_mask),
    mask_np itself without a layer."""
    f = getattr(m, 'open_mask', None)
    return f() if f is not None else m.mask_np


def _clearance(m, G, i, j):
    """The web app's clearanceFrom(i, j): physical distance to the nearest masked-out cell (the free
    mask: solids count as walls), the array edge too on the full-grid square; baseHalf if none.
    10/10/2026: with an absorbing layer, the nearest cell outside the OPEN domain (the layer counts as the
    wall: inside it the front is damped, so it is no longer the free front these rows measure)."""
    n = m.N
    best = math.inf
    if m.sq_full():
        best = min(i, j, n - 1 - i, n - 1 - j)
    elif _tri_torus(m):
        # the rhombic torus (05/10/2026): as the full square's array edge, the seam -- the lines of
        # constant i and j are S3H apart, so the physical distance to the nearest is S3H x the index
        best = S3H * min(i, j, n - 1 - i, n - 1 - j)
    if m.mask_np[i, j] == 0:
        return 0.0
    if getattr(m, 'abs_np', None) is not None:
        BI, BJ = G.get_or('bnd_open', lambda: _boundary_cells(_open_free(m)))
    else:
        BI, BJ = G.get_or('bnd', lambda: _boundary_cells(m.mask_np))
    if BI.size:
        px, py = m.cell_xy(i, j)
        X, Y = _cell_xy_np(m, BI, BJ)
        d = float(al.js_hypot_np(X - px, Y - py).min())
        if d < best:
            best = d
    return best if math.isfinite(best) else m.base_half()


def _full(m):
    f = getattr(m, 'full_array', None)
    return bool(f()) if f is not None else m.sq_full()


def _torus(m):
    f = getattr(m, 'torus', None)
    return bool(f()) if f is not None else (m.p.bc == 'periodic' and not m.tri())


def _tri_torus(m):
    """The rhombic torus (the triangular 'array' with bc periodic, 05/10/2026)."""
    return m.tri() and _torus(m)


def _wrap_cone(m):
    """The light cone wraps: the full square's torus (the web app) or the rhombic torus."""
    return _torus(m) and _full(m)


def _clear(m, G):
    """CLEAR (recacheSource): the first front to hit a wall."""
    def f():
        if m.p.src == 'vtx' and m.vtx:
            return min(_clearance(m, G, v[0], v[1]) for v in m.vtx)
        return _clearance(m, G, m.src[0], m.src[1])
    return G.get_or('clear', f)


def cone_norm(m, da, db):
    """coneNorm: the lattice norm whose unit ball is the reach polytope (index offsets)."""
    if m.tri():
        return np.maximum(np.maximum(np.abs(da), np.abs(db)), np.abs(da + db))
    if m.p.stencil == 9:
        return np.maximum(np.abs(da), np.abs(db))
    return np.abs(da) + np.abs(db)


def _cone_wall(m, G):
    """CONE_WALL (coneWall, set at seed): the cone-norm distance from the source to the first clamped
    cell -- masked out (solids included) or on the array ring. The ring's minimum is
    min(si, sj, n-1-si, n-1-sj) for all three norms (one axis offset, the other free)."""
    def f():
        n, p = m.N, m.p
        si, sj = int(m.src[0]), int(m.src[1])
        if _wrap_cone(m):
            return (n - 1) // 2              # the cone meets itself (the rhombic torus' periods are N long in the hex norm too)
        if m.mask_np[si, sj] == 0:
            return 0
        best = min(si, sj, n - 1 - si, n - 1 - sj)
        BI, BJ = G.get_or('bnd', lambda: _boundary_cells(m.mask_np))
        if BI.size:
            best = min(best, int(cone_norm(m, BI - si, BJ - sj).min()))
        return best
    return G.get_or('cone_wall', f)


# ---- exact mask symmetry checks (the premise of the symmetry rows with solids / the plate) -------
def _mask_sym8(M):
    return bool(np.array_equal(M, M[::-1, :]) and np.array_equal(M, M[:, ::-1]) and np.array_equal(M, M.T))


def _mirror_ij(m, i, j):
    """The mirror y -> -y as an index map: (i, 2c-j) on the square lattice, (a,b) -> (a+b, -b) on
    the triangular one."""
    c = (m.N - 1) // 2
    return (i + (j - c), 2 * c - j) if m.tri() else (i, 2 * c - j)


def _type_in(m):
    """The solid TYPE map inside the domain (0 free / outside, 1 pinned, 2 rigid), or None when no
    solid is rigid: then the free-cell mask decides the symmetry alone (05/10/2026)."""
    T = getattr(m, 'solid_np', None)
    if T is None:
        return None
    T = np.where(m.domain_mask_np != 0, T, 0).astype(np.uint8)
    return T if (T == 2).any() else None


def _tri_maps(m):
    c = (m.N - 1) // 2
    I, J = np.meshgrid(np.arange(m.N), np.arange(m.N), indexing='ij')
    return c, I - c, J - c


def _mask_mirror(m, M):
    n = m.N
    if m.tri() and _tri_torus(m):
        c, A, B = _tri_maps(m)                # (a, b) -> (a + b, -b), mod n on the torus
        return bool(np.array_equal(M, M[(c + A + B) % n, (c - B) % n]))
    if not m.tri():
        return bool(np.array_equal(M, M[:, ::-1]))
    c = (n - 1) // 2
    for j in range(n):
        s, j2 = j - c, 2 * c - j
        col, col2 = M[:, j], M[:, j2]
        # M[i, j] must equal M[i + s, j2]; a free cell whose partner leaves the array breaks it
        i0, i1 = max(0, -s), min(n, n - s)
        if not np.array_equal(col[i0:i1], col2[i0 + s:i1 + s]):
            return False
        if col[:i0].any() or col[i1:].any():
            return False
    return True


def _mask_rot(m, M, per):
    """Invariance of the free mask under (a,b) -> (-b, a+b) applied `per` times (60 deg each):
    M[i, j] = M[2c - j, i + j - c] for one application -- a column of M against a shifted row."""
    n = m.N
    c = (n - 1) // 2
    if _tri_torus(m):                         # mod n on the rhombic torus (its period lattice is 6-fold)
        _, A, B = _tri_maps(m)
        P, Q = A.copy(), B.copy()
        for _ in range(per):
            P, Q = -Q, P + Q
        return bool(np.array_equal(M, M[(c + P) % n, (c + Q) % n]))
    R = M
    for _ in range(per):                      # R := R o rot (the rotated mask, then compare)
        out = np.zeros_like(R)
        for j in range(n):
            i2 = 2 * c - j                    # row of the partner
            s = j - c                         # partner column = i + s
            i0, i1 = max(0, -s), min(n, n - s)
            out[i0:i1, j] = R[i2, i0 + s:i1 + s] if 0 <= i2 < n else 0
        R = out
    return bool(np.array_equal(R, M))


# ---- the context of one ledger build ------------------------------------------------------------
class _Ctx:
    def __init__(self, m, compat):
        self.m, self.p, self.n = m, m.p, m.N
        self.tri = m.tri()
        self.compat = compat
        self.fp64 = bool(mc.FP64)
        self.G = _geo(m)
        self.solids = bool(m.has_solids)
        self.unl = m.p.shape == 'unlicensed'
        # 05/10/2026: the 'array' shape, the triangular modes, rigid walls
        self.arr = m.p.shape == 'array'
        self.full = _full(m)
        self.torus = _torus(m)
        self.tri_torus = self.tri and self.torus
        bi = m.bc_info() if hasattr(m, 'bc_info') else {'fallback': False, 'reason': None}
        self.bc_fallback, self.bc_reason = bool(bi.get('fallback')), bi.get('reason')
        T = getattr(m, 'solid_np', None)
        self.rigid_solids = bool(self.solids and T is not None and ((T == 2) & (m.domain_mask_np != 0)).any())
        self.edge_rigid = getattr(m.p, 'edge', 'pinned') == 'rigid' and not self.full
        self.rigid_ring = self.tri and self.arr and not self.torus and m.p.bc == 'neumann'
        # 10/10/2026: the absorbing boundaries. absorbing = what absorbs ('mur' / 'mur2' / 'cpml' / 'sponge' on the
        # full array, 'open' for a masked shape's sponge edge) or None; port_abs = a port-only one (every row
        # but the web app's Mur takes the port's logic, in both modes: the web app has no case for them)
        f = getattr(m, 'absorbing', None)
        self.absorbing = f() if f is not None else None
        self.port_abs = self.absorbing in ('mur2', 'cpml', 'sponge', 'open')
        f = getattr(m, 'abs_layer', None)
        self.layer = f() if f is not None else None
        self.abs_np = getattr(m, 'abs_np', None)
        self._U = None
        self.step = int(m.step_n)
        self.drops = int(m.drops)
        self.hist = m.hist
        if m.record:
            self.amax = float(m.amax)
        else:                                  # the recorder is off: S.amax of the current state
            self.amax = float(m.stats()[1])
        E = m.hist.get('E') or []
        self.E_last = E[-1] if E else None
        # the per-step history is complete iff it holds one entry per step (capped): the recorder
        # appends once per step and seed() clears it (mode: plus the seeded step-0 value). A shorter
        # history means m.record was False for some steps -- a stale or gapped series the history
        # rows must not grade (port-only: the web app always records).
        cap = mc.HIST_CAP
        self.hist_gap = (len(E) != min(self.step, cap['E'])
                         or (m.p.src == 'mode' and len(m.hist.get('mode') or []) != min(self.step + 1, cap['mode'])))
        self.live = self.step > 5 and math.isfinite(self.amax) and self.amax > 1e-12
        self.c = (self.n - 1) // 2

    def fix(self, fid):
        return not self.compat

    def U(self):
        """The current level U1 as [i, j] (one device -> host copy per build, only when a row needs
        the field)."""
        if self._U is None:
            self._U = self.m.level(1)
        return self._U

    def co_limit(self):
        """The web app's undamped C_max (the light cone / mode rows' geometry, and the CFL row in
        compat mode); the CFL row uses the damped limit under the 'cfl-damped' fix."""
        return self.m.co_limit(damped=False)

    def src_centred(self):
        return abs(self.p.sx) < 1e-9 and abs(self.p.sy) < 1e-9

    def fits(self):
        return self.G.get_or('fits', lambda: al.fits_inside(self.m, self.m.shape_poly()))

    def cells_domain(self):
        """The domain's cell count (solids included); 10/10/2026: the OPEN domain's (an absorbing layer excluded)."""
        def f():
            g = getattr(self.m, 'open_domain_np', None)
            return int(np.count_nonzero(g() if g is not None else self.m.domain_mask_np))
        return self.G.get_or('cells_dom', f)

    def frame_Ld(self):
        """An absorbing FRAME's index depth (the first open row), else 0 (10/10/2026)."""
        lay = self.layer
        return int(lay['Ld']) if (lay is not None and lay.get('where') == 'frame') else 0

    def abs_text(self):
        """The absorbing boundary in words (10/10/2026)."""
        lay, a = self.layer, self.absorbing
        if a == 'open':
            return f'the open edge (a sponge {fmt(lay["L"], 0)} cells wide outside the drawn edge)'
        if a == 'cpml':
            return f'the CPML frame ({lay["Lc"]} cells, {lay["level"]})'
        if a == 'sponge':
            return f'the sponge frame ({fmt(lay["L"], 0)} cells)'
        if a == 'mur2':
            return 'the Mur ring, 2nd order'
        return 'the Mur ring'

    def layer_sym(self, kind, fold=0):
        """The symmetry rows' premise for an absorbing LAYER (10/10/2026): its damping map (the mask codes) is
        invariant under the orbit, checked cell by cell (a quantised distance ramp of a rotated n-gon need not
        be). True without a layer. kind: 'sym8' | 'mirror' | 'rot' (fold)."""
        A = self.abs_np
        if A is None:
            return True
        def f():
            if kind == 'sym8':
                return _mask_sym8(A)
            if kind == 'mirror':
                return _mask_mirror(self.m, A)
            return _mask_rot(self.m, A, 6 // fold)
        return self.G.get_or(f'layersym_{kind}{fold}', f)

    def tol(self, f64, kind):
        return f64 if self.fp64 else TOL_F32[kind]

    def edge(self):
        """What bounds the domain (FIX edge-kind): 'clamped' (the mask wall / the tri ring),
        p.bc for the full-grid square, 'mixed-<bc>' when a masked domain's free cells reach the
        square lattice's ring (part flavor wall, part array edge). 05/10/2026: 'rigid' for a masked
        domain with edge = 'rigid' (it reflects and does not absorb, as the clamped one); the
        triangular 'array': the bc it runs (dirichlet / neumann = the rigid ring / periodic; Mur falls
        back to dirichlet). 10/10/2026: 'open' for a masked shape's absorbing edge; the full square the bc it
        RUNS (bc_effective: mur2 / cpml / sponge, or a fallback), a masked shape that reaches the ring
        'mixed-' + the bc it runs (mur2 / cpml / sponge fall back to dirichlet there)."""
        def f():
            m, p = self.m, self.p
            eff = m.bc_effective() if hasattr(m, 'bc_effective') else p.bc
            if self.tri and self.arr:
                return eff
            if self.absorbing == 'open':
                return 'open'
            if self.tri:
                return 'rigid' if self.edge_rigid else 'clamped'
            if m.sq_full():
                return eff
            M = m.domain_mask_np
            ring = M[0, :].any() or M[-1, :].any() or M[:, 0].any() or M[:, -1].any()
            if not ring:
                return 'rigid' if self.edge_rigid else 'clamped'
            return 'mixed-' + eff
        return self.G.get_or('edge', f)

    def mur_port(self, what):
        """The port remark when Mur is selected but no Mur ring bounds the domain (FIX edge-kind)."""
        if self.tri and self.arr:
            return (self.bc_reason or mc.TRI_MUR_REASON) + ' (FIX edge-kind)'
        kind = 'rigid' if self.edge_rigid else 'clamped'
        return what.replace('clamped', kind)

    def wall_port(self):
        """What the walls are, for the rows that grade a closed domain (energy, boundary)."""
        parts = []
        if self.rigid_solids or self.edge_rigid or self.rigid_ring:
            parts.append('pinned and rigid walls both conserve the energy (a rigid wall does no work: its links are '
                         'removed); a rigid wall reflects upright (R = +1)')
        if self.tri_torus:
            parts.append("the rhombic torus has no wall; its recorder sums every cell with wrapped differences (the "
                         "square torus keeps the web app's interior-only sum)")
        return ' · '.join(parts)


# ---- the residuals ----------------------------------------------------------------------------
def _jsmax(a):
    """Math.max folded over an array: NaN anywhere -> NaN; empty -> 0 (the JS starts at 0)."""
    if a.size == 0:
        return 0.0
    return float(np.max(a))


def _jmax(a, b):
    """Math.max(a, b): NaN if either is NaN (Python's max() would drop it)."""
    return math.nan if (a != a or b != b) else (a if a >= b else b)


def _jmin(a, b):
    return math.nan if (a != a or b != b) else (a if a <= b else b)


@np.errstate(invalid='ignore')           # inf - inf in a diverged run: NaN, as in the JS
def symmetry_residual(U):
    """symmetryResidual(): the octant orbit, sampled on even offsets of the first quadrant."""
    n = U.shape[0]
    if n % 2 == 0:
        return None
    c = (n - 1) // 2
    Q = U[c:, c:][::2, ::2]
    A = U[c::-1, c:][::2, ::2]
    B = U[c:, c::-1][::2, ::2]
    C = U[c::-1, c::-1][::2, ::2]
    worst = 0.0
    for P in (A, B, C, Q.T, A.T, B.T, C.T):
        worst = _jmax(worst, _jsmax(np.abs(Q - P)))
    return worst


@np.errstate(invalid='ignore')           # inf - inf in a diverged run: NaN, as in the JS
def mirror_residual(m, U):
    """mirrorResidual(): |u(x, +y) - u(x, -y)| on odd row offsets and even columns / a-offsets."""
    n = m.N
    if n % 2 == 0:
        return None
    c = (n - 1) // 2
    if not m.tri():
        return _jsmax(np.abs(U[::2, c + 1::2] - U[::2, c - 1::-2]))
    b = np.arange(1, c + 1, 2)[:, None]
    a = np.arange(-c, c + 1, 2)[None, :]
    i, j, i2, j2 = c + a, c + b, c + a + b, c - b
    if _tri_torus(m):
        i2 = i2 % n                          # the rhombic torus: every pair, mod n (05/10/2026)
    ok = (i >= 0) & (i < n) & (i2 >= 0) & (i2 < n)
    i, j, i2, j2 = (np.broadcast_to(x, ok.shape)[ok] for x in (i, j, i2, j2))
    return _jsmax(np.abs(U[i, j] - U[i2, j2]))


@np.errstate(invalid='ignore')           # inf - inf in a diverged run: NaN, as in the JS
def sym_rot_residual(m, U, fold):
    """symRotResidual(fold): the triangular lattice's own 6- (or 3-) fold orbit, (a,b) -> (-b, a+b)."""
    n = m.N
    c = (n - 1) // 2
    per = 6 // fold
    b, a = np.meshgrid(np.arange(-c, c + 1, 2), np.arange(-c, c + 1, 2), indexing='ij')
    a, b = a.ravel(), b.ravel()
    v = U[c + a, c + b]
    p, q = a.copy(), b.copy()
    worst = 0.0
    for hop in range(1, fold):
        for _ in range(per):
            p, q = -q, p + q
        i2, j2 = c + p, c + q
        if _tri_torus(m):
            i2, j2 = i2 % n, j2 % n          # the rhombic torus: its period lattice is 6-fold (05/10/2026)
        inr = (i2 >= 0) & (i2 < n) & (j2 >= 0) & (j2 < n)
        if np.any(np.abs(v[~inr]) > 1e-12):
            return None
        worst = _jmax(worst, _jsmax(np.abs(v[inr] - U[i2[inr], j2[inr]])))
    return worst


def energy_stats(E):
    """energyStats(): drift (normalised slope of a linear fit) and spread over the last 1500."""
    if len(E) < 1200:
        return None
    E = E[-1500:]
    mean = 0.0
    for e in E:
        mean += e
    mean /= len(E)
    if mean <= 1e-12:                         # NaN passes on, as in the JS (drift NaN -> FAIL)
        return None
    lo, hi = math.inf, -math.inf
    for e in E:
        lo = _jmin(lo, e)
        hi = _jmax(hi, e)
    sx = sy = sxx = sxy = 0.0
    for i, y in enumerate(E):
        sx += i
        sy += y
        sxx += i * i
        sxy += i * y
    mm = len(E)
    slope = (mm * sxy - sx * sy) / (mm * sxx - sx * sx)
    return {'mean': mean, 'spread': (hi - lo) / mean, 'drift': abs(slope * mm / mean)}


def stencil_symbol(m, kx, ky):
    if m.tri():
        s = 0.0
        for a in (0, PI / 3, 2 * PI / 3):
            p = kx * math.cos(a) + ky * math.sin(a)
            s += math.sin(p / 2) ** 2
        return -(8 / 3) * s
    if m.p.stencil == 9:
        return (2 / 3) * (2 * math.cos(kx) + 2 * math.cos(ky) - 4) + (1 / 6) * (4 * math.cos(kx) * math.cos(ky) - 4)
    return -4 * (math.sin(kx / 2) ** 2 + math.sin(ky / 2) ** 2)


# ---- device reductions (the field never leaves the GPU) ------------------------------------------
# Measured 30/09/2026 (CUDA f64, ScratchCLAUDE/LedgerCLAUDE/perf_MIRROR.py) with the numpy versions above:
# copying the current level to the host alone cost 3.5 ms at N = 1025, 12 at 2049, 45 at 4097, 180-220 at
# 8193; the octant orbit 530 ms and the triangular 6-fold orbit 3.8 s at N = 8193 (strided transposes /
# gathers), the mode correlation 1.1 s. So every O(N^2) reduction runs here as a kernel -- max |a - b|
# over an orbit, the cone's support, the mode correlation -- with the core's two-level pattern (a
# thread walks a run of rows serially, then ONE atomic per thread into one of RSLOT slots; atomics of
# millions of threads into one address are slow, 29/09), and only the RSLOT x 8 slot table comes back.
# max |a - b| is exact in any order, so these agree BIT FOR BIT with the numpy references (checked by
# crosscheck_ledger_MIRROR.py's port-only section); the sums (cone energy share, correlation) differ in
# the last bits from the JS's sequential order, which the ledger prints to 2 and 9 decimals.
# NaN: Math.max propagates NaN; atomic_max drops it, so a NaN is flagged in its own column. The test is
# written with ORDERED compares -- `d >= 0` for d = |a - b|, `not (v == 0)` for "v !== 0" -- because on
# Vulkan `d != d` compiled as an ordered compare and never saw a NaN (measured 30/09/2026: a diverged f32
# run read 0.0 / inf where the numpy reference read NaN).
# Sums accumulate in the core's ACC_DT (f64 on CUDA / CPU even in an f32 build; f32 on Vulkan, whose f64
# accumulators crashed on 29/09): with f32 slots the (3,2) mode's correlation read 1.00000006.
RSLOT = 64
RTHREADS = 65536
DEVICE = True                               # False: the numpy references on a host copy of the field
_OUT = {}


def _out_buf():
    b = _OUT.get(mc.FP64)
    if b is None:
        b = ti.ndarray(dtype=mc.ACC_DT, shape=(RSLOT + 1, 8))
        _OUT[mc.FP64] = b
    return b


@ti.func
def _vhypot(x, y):
    """V8's Math.hypot(x, y) (see alignment_MIRROR.js_hypot)."""
    ax, ay = ti.abs(x), ti.abs(y)
    mm = ti.max(ax, ay)
    r = 0.0
    if mm > 0:
        a, b = ax / mm, ay / mm
        r = ti.sqrt(a * a + b * b) * mm
    return r


@ti.func
def _cnorm(da, db, kind):
    r = 0
    if kind == 2:
        r = ti.max(ti.max(ti.abs(da), ti.abs(db)), ti.abs(da + db))
    elif kind == 1:
        r = ti.max(ti.abs(da), ti.abs(db))
    else:
        r = ti.abs(da) + ti.abs(db)
    return r


@ti.kernel
def k_orbit(U: ti.types.ndarray(ndim=3), s: ti.i32, mode: ti.template(), per: ti.i32, fold: ti.i32,
            OUT: ti.types.ndarray(ndim=2), wrap: ti.i32):
    """max |u(p) - u(g p)| over an orbit, sampled as the web app samples it. mode 0: the octant orbit
    (symmetryResidual); 1: mirror, square lattice; 2: mirror, triangular lattice (a,b) -> (a+b, -b);
    3: the triangular rotation (a,b) -> (-b, a+b) applied `per` times per hop, fold-1 hops
    (symRotResidual; column 2 flags a partner off the array with |v| > 1e-12 -> null). wrap = 1 (the
    rhombic torus, 05/10/2026): modes 2 and 3 take the partner mod n, so every pair is compared."""
    n = U.shape[1]
    c = (n - 1) // 2
    for k in range(RSLOT):
        for q in ti.static(range(3)):
            OUT[k, q] = 0.0
    A = c + 1                                   # the serial coordinate
    B = c + 1                                   # the parallel one
    if ti.static(mode == 0):
        A = c // 2 + 1
        B = c // 2 + 1
    elif ti.static(mode == 1 or mode == 2):
        B = (c + 1) // 2
    G = ti.max(1, ti.min(A, RTHREADS // ti.max(B, 1)))
    gs = (A + G - 1) // G
    for g, kb in ti.ndrange(G, B):
        w = 0.0
        nan = 0.0
        nul = 0.0
        for ka in range(g * gs, ti.min(A, (g + 1) * gs)):
            if ti.static(mode == 0):
                i = 2 * ka
                j = 2 * kb
                a = U[s, c + i, c + j]
                for q in ti.static(range(7)):
                    sa, sb, sw = ti.static([(-1, 1, 0), (1, -1, 0), (-1, -1, 0), (1, 1, 1), (-1, 1, 1),
                                            (1, -1, 1), (-1, -1, 1)][q])
                    b = 0.0
                    if ti.static(sw):
                        b = U[s, c + sa * j, c + sb * i]
                    else:
                        b = U[s, c + sa * i, c + sb * j]
                    d = ti.abs(a - b)
                    if d >= 0:
                        w = ti.max(w, d)
                    else:
                        nan = 1.0
            elif ti.static(mode == 1):
                i = 2 * ka
                dd = 2 * kb + 1
                d = ti.abs(U[s, i, c + dd] - U[s, i, c - dd])
                if d >= 0:
                    w = ti.max(w, d)
                else:
                    nan = 1.0
            elif ti.static(mode == 2):
                a = -c + 2 * ka
                b = 2 * kb + 1
                i2 = c + a + b
                if wrap:
                    i2 = ((i2 % n) + n) % n
                if i2 >= 0 and i2 < n:
                    d = ti.abs(U[s, c + a, c + b] - U[s, i2, c - b])
                    if d >= 0:
                        w = ti.max(w, d)
                    else:
                        nan = 1.0
            else:
                a = -c + 2 * ka
                b = -c + 2 * kb
                v = U[s, c + a, c + b]
                p = a
                q = b
                for hop in range(1, fold):
                    for t in range(per):
                        np_ = -q
                        q = p + q
                        p = np_
                    i2 = c + p
                    j2 = c + q
                    if wrap:
                        i2 = ((i2 % n) + n) % n
                        j2 = ((j2 % n) + n) % n
                    if i2 < 0 or i2 >= n or j2 < 0 or j2 >= n:
                        if ti.abs(v) > 1e-12:
                            nul = 1.0
                    else:
                        d = ti.abs(v - U[s, i2, j2])
                        if d >= 0:
                            w = ti.max(w, d)
                        else:
                            nan = 1.0
        slot = (g * 7 + kb) % RSLOT
        ti.atomic_max(OUT[slot, 0], ti.cast(w, mc.ACC_DT))
        ti.atomic_max(OUT[slot, 1], ti.cast(nan, mc.ACC_DT))
        ti.atomic_max(OUT[slot, 2], ti.cast(nul, mc.ACC_DT))


@ti.func
def _cnorm_w(da, db, kind, n, h, wrap):
    """The cone norm of an index offset; wrap: the torus' nearest image -- each axis into [-h, h]
    (exact for the square lattice's norms), and on the triangular lattice (kind 2) also the six
    neighbouring images (+-n, 0), (0, +-n), (+-n, -+n), whose hexagonal norm can be smaller (05/10/2026)."""
    r = 0
    if wrap:
        a = (da + n + h) % n - h
        b = (db + n + h) % n - h
        r = _cnorm(a, b, kind)
        if kind == 2:
            r = ti.min(r, _cnorm(a - n, b, kind))
            r = ti.min(r, _cnorm(a + n, b, kind))
            r = ti.min(r, _cnorm(a, b - n, kind))
            r = ti.min(r, _cnorm(a, b + n, kind))
            r = ti.min(r, _cnorm(a - n, b + n, kind))
            r = ti.min(r, _cnorm(a + n, b - n, kind))
    else:
        r = _cnorm(da, db, kind)
    return r


@ti.kernel
def k_cone(U: ti.types.ndarray(ndim=3), s: ti.i32, si: ti.i32, sj: ti.i32, mstep: ti.i32, kind: ti.i32,
           wrap: ti.i32, tri: ti.i32, rdisk: float, OUT: ti.types.ndarray(ndim=2)):
    """coneMeasure(): columns 0..3 = support radius, cells outside the cone, sum u^2, sum u^2 inside the
    disk; OUT[RSLOT, 0] = the folded radius; columns 4, 5 = max |u| on that ring and its NaN flag."""
    n = U.shape[1]
    h = n // 2
    cf = (n - 1) / 2
    for k in range(RSLOT + 1):
        for q in ti.static(range(8)):
            OUT[k, q] = 0.0
    G = ti.max(1, ti.min(n, RTHREADS // n))
    gs = (n + G - 1) // G
    # physical offsets computed exactly as the JS does: cellXY(i, j) - cellXY(si, sj)
    ys = (sj - cf) * 1.0
    xs = (si - cf) * 1.0
    if tri:
        ys = (sj - cf) * 0.8660254037844386
        xs = (si - cf) + (sj - cf) * 0.5
    for g, j in ti.ndrange(G, n):
        rmax = 0.0
        cnt = 0.0
        ea = ti.cast(0.0, mc.ACC_DT)
        ei = ti.cast(0.0, mc.ACC_DT)
        for i in range(g * gs, ti.min(n, (g + 1) * gs)):
            v = U[s, i, j]
            if not (v == 0):                    # NaN counts, as "v !== 0" does in the JS
                da = i - si
                db = j - sj
                r = ti.cast(_cnorm_w(da, db, kind, n, h, wrap), float)
                rmax = ti.max(rmax, r)
                if r > mstep:
                    cnt += 1.0
                e = ti.cast(v * v, mc.ACC_DT)
                ea += e
                x = (i - cf) * 1.0
                y = (j - cf) * 1.0
                if tri:
                    x = (i - cf) + (j - cf) * 0.5
                    y = (j - cf) * 0.8660254037844386
                if _vhypot(x - xs, y - ys) <= rdisk:
                    ei += e
        slot = (g * 7 + j) % RSLOT
        ti.atomic_max(OUT[slot, 0], ti.cast(rmax, mc.ACC_DT))
        ti.atomic_add(OUT[slot, 1], ti.cast(cnt, mc.ACC_DT))
        ti.atomic_add(OUT[slot, 2], ea)
        ti.atomic_add(OUT[slot, 3], ei)
    for _ in range(1):
        r = ti.cast(0.0, mc.ACC_DT)
        for k in range(RSLOT):
            r = ti.max(r, OUT[k, 0])
        OUT[RSLOT, 0] = r
    for g, j in ti.ndrange(G, n):
        rout = ti.cast(OUT[RSLOT, 0], float)
        w = 0.0
        nan = 0.0
        for i in range(g * gs, ti.min(n, (g + 1) * gs)):
            da = i - si
            db = j - sj
            if ti.cast(_cnorm_w(da, db, kind, n, h, wrap), float) == rout:
                a = ti.abs(U[s, i, j])
                if a >= 0:
                    w = ti.max(w, a)
                else:
                    nan = 1.0
        slot = (g * 7 + j) % RSLOT
        ti.atomic_max(OUT[slot, 4], ti.cast(w, mc.ACC_DT))
        ti.atomic_max(OUT[slot, 5], ti.cast(nan, mc.ACC_DT))


@ti.kernel
def k_modecorr(U: ti.types.ndarray(ndim=3), s: ti.i32, M: ti.types.ndarray(ndim=2), Si: ti.types.ndarray(ndim=1),
               Sj: ti.types.ndarray(ndim=1), OUT: ti.types.ndarray(ndim=2)):
    # Si / Sj are f64 host arrays (only called where ACC_DT is f64)
    """The eigenmode row's sums over the masked interior: num = sum a b, da = sum a^2, db = sum b^2,
    a = Si[i] Sj[j] (host-computed sines), b = u."""
    n = U.shape[1]
    for k in range(RSLOT):
        for q in ti.static(range(3)):
            OUT[k, q] = 0.0
    A = n - 2
    G = ti.max(1, ti.min(A, RTHREADS // A))
    gs = (A + G - 1) // G
    for g, j in ti.ndrange(G, (1, n - 1)):
        num = ti.cast(0.0, mc.ACC_DT)
        da = ti.cast(0.0, mc.ACC_DT)
        db = ti.cast(0.0, mc.ACC_DT)
        for i in range(1 + g * gs, ti.min(n - 1, 1 + (g + 1) * gs)):
            if M[i, j] != 0:
                a = ti.cast(Si[i], mc.ACC_DT) * ti.cast(Sj[j], mc.ACC_DT)
                b = ti.cast(U[s, i, j], mc.ACC_DT)
                num += a * b
                da += a * a
                db += b * b
        slot = (g * 7 + j) % RSLOT
        ti.atomic_add(OUT[slot, 0], num)
        ti.atomic_add(OUT[slot, 1], da)
        ti.atomic_add(OUT[slot, 2], db)


@ti.kernel
def k_gather(U: ti.types.ndarray(ndim=3), s: ti.i32, I: ti.types.ndarray(ndim=1), J: ti.types.ndarray(ndim=1),
             V: ti.types.ndarray(ndim=1)):
    for k in range(I.shape[0]):
        V[k] = U[s, I[k], J[k]]


def _cur(m):
    return (m.slot + 1) % 3


def orbit_residual(x, kind, fold=6):
    """symmetryResidual / mirrorResidual / symRotResidual on the device (numpy fallback: DEVICE False)."""
    m = x.m
    if not DEVICE:
        U = x.U()
        return (symmetry_residual(U) if kind == 'sym8' else mirror_residual(m, U) if kind == 'mirror'
                else sym_rot_residual(m, U, fold))
    OUT = _out_buf()
    mode = {'sym8': 0, 'mirror': 2 if x.tri else 1, 'rot': 3}[kind]
    k_orbit(m.U, _cur(m), mode, (6 // fold) if kind == 'rot' else 1, fold, OUT, 1 if x.tri_torus else 0)
    o = OUT.to_numpy()[:RSLOT].astype(np.float64)
    if kind == 'rot' and o[:, 2].max() > 0:
        return None
    if o[:, 1].max() > 0:
        return math.nan
    return float(o[:, 0].max())


def gather(x, I, J):
    """Values of the current level at cells (I[k], J[k]) -- a few hundred cells, no whole-field copy."""
    m = x.m
    if not DEVICE:
        U = x.U()
        return U[np.asarray(I, dtype=np.int64), np.asarray(J, dtype=np.int64)].astype(np.float64)
    I = np.ascontiguousarray(I, dtype=np.int32)
    J = np.ascontiguousarray(J, dtype=np.int32)
    V = np.zeros(max(1, I.size), dtype=m.fdt)
    if I.size:
        k_gather(m.U, _cur(m), I, J, V)
    return V[:I.size].astype(np.float64)


# ---- geometry the staircase row needs -----------------------------------------------------------
def _unl_area(m):
    hw, hh, rc = m.plate_dims()
    return (2 * hw) * (2 * hh) - (4 - PI) * rc * rc


def circle_clip_area(R, h):
    if R <= h:
        return PI * R * R
    if R >= h * SQRT2:
        return 4 * h * h
    x1, x2 = math.sqrt(R * R - h * h), min(R, h)
    F = lambda x: R * R * math.asin(min(1, max(-1, x / R))) + x * math.sqrt(max(0, R * R - x * x))
    return 4 * h * x1 + 2 * (F(x2) - F(x1))


def shape_area(ctx, R):
    m, p = ctx.m, ctx.p
    if m.sq_full():
        Ld = ctx.frame_Ld()                    # 10/10/2026: an absorbing frame: the open domain is (N - 2 Ld)^2
        return (m.N - 2 * Ld) ** 2
    P = m.shape_poly()
    if ctx.fits():
        if p.shape == 'circle':
            return PI * R * R
        if p.shape == 'square':
            return 4 * R * R
        if ctx.unl:
            return _unl_area(m)
        return al.poly_area(P)
    if p.shape == 'circle' and not ctx.tri:
        return circle_clip_area(R, m.N / 2)
    return al.poly_area(m.clip_convex(P, m.clip_region()))


def ideal_area(ctx):
    m, p = ctx.m, ctx.p
    if p.shape == 'circle':
        return PI * m.dom_R() * m.dom_R()
    if ctx.unl and ctx.fits():
        return _unl_area(m)
    return al.poly_area(m.shape_poly())


# ---- the rows ---------------------------------------------------------------------------------
_SOL = 'solid objects are present'


def _row_cfl(x):
    m, p = x.m, x.p
    L = x.co_limit()
    Lname = '√(2/3)' if x.tri else ('√3/2' if p.stencil == 9 else '1/√2')
    Lwho = 'triangular, 6-neighbour' if x.tri else f'{p.stencil}-point'
    fixes = ()
    if p.damp > 0 and x.fix('cfl-damped'):
        Ld = x.m.co_limit()
        fixes = ('cfl-damped',)                 # the limit (and the row's text) changes
        L, Lname = Ld, f'{Lname}·√(1−d/2)'
        Lwho += f', damping d = {fmt(p.damp, 4)}'
    over = p.Co > L
    diverged = not math.isfinite(x.amax) or x.amax > 1e6
    if over and not diverged and x.fix('cfl-nan') and x.E_last is not None and not math.isfinite(x.E_last):
        diverged, fixes = True, fixes + ('cfl-nan',)
    if x.step < 40:
        badge, detail, val = 'n/a', 'run a few dozen steps first', ''
    elif not over:
        badge = 'FAIL' if diverged else 'PASS'
        detail = f'Co = {fmt(p.Co, 4)} ≤ {Lname} = {fmt(L, 6)} ({Lwho}) · bounded, max|u| = {fe(x.amax)}'
        val = f'max|u| {fe(x.amax)}'
    else:
        badge = 'RED OK' if diverged else 'not yet'
        detail = f'Co = {fmt(p.Co, 4)} > {Lname} — must diverge; max|u| = {fe(x.amax)} after {x.step} steps'
        val = f'max|u| {fe(x.amax)}'
    name = (f'CFL: above {Lname} it MUST blow up (it cannot be stable)' if over
            else f'CFL: the limit belongs to the LATTICE+STENCIL — {Lwho} → {Lname}')
    return _row('cfl', name, badge, detail, 'proof §5', val, x.amax, dsg=over, fixes=fixes)


def _row_sym8(x):
    m, p, n = x.m, x.p, x.n
    nn = int(p.shape[1:]) if (p.shape[:1] == 'n' and p.shape[1:].isdigit()) else 0
    sym8 = not x.tri and (p.shape in ('circle', 'square', 'array') or
                          (nn % 4 == 0 and nn > 0 and abs(p.rot / (PI / nn) - jround(p.rot / (PI / nn))) < 1e-9))
    on_axis = x.src_centred()
    # 10/10/2026: the port's absorbing boundaries are symmetric BCs too (Mur 2nd order's ring rule and the
    # frames are 8-fold symmetric; the layer's damping map is checked cell by cell below)
    centred = (p.src in ('pulse', 'impulse') and p.medium == 'uniform' and sym8 and on_axis
               and (p.bc in ('dirichlet', 'neumann', 'periodic') or x.port_abs))
    port = ''
    sol_ok = True
    if centred and x.port_abs and not x.layer_sym('sym8'):
        sol_ok = False
        port = (f'{x.abs_text()}: its damping map (the distance ramp, quantised per cell) is not 8-fold symmetric '
                '(checked cell by cell), so this orbit does not apply')
    elif centred and x.port_abs:
        port = f'{x.abs_text()}: an 8-fold symmetric boundary (its damping map checked cell by cell)'
    if centred and x.solids and sol_ok:
        c = x.c
        mask_ok = x.G.get_or('mask8', lambda: _mask_sym8(m.mask_np) and (_type_in(m) is None or _mask_sym8(_type_in(m))))
        at_c = (int(m.src[0]), int(m.src[1])) == (c, c)
        sol_ok = mask_ok and at_c
        port = (('solids: the free-cell mask is 8-fold symmetric (checked cell by cell) and the source '
                 'is on the centre cell, so the orbit still applies') if sol_ok else
                (f'{_SOL}: the free-cell mask (or the pinned / rigid types) is not 8-fold symmetric (checked cell '
                 'by cell), so this orbit does not apply') if not mask_ok else
                (f'{_SOL}: the source snapped out of a solid to cell ({int(m.src[0])}, {int(m.src[1])}), '
                 'off the centre, so the field is correctly not 8-fold symmetric'))
    good = centred and sol_ok
    r = orbit_residual(x, 'sym8') if good else None
    tol = x.tol(1e-12, 'sym')
    if good and r is not None and x.live:
        badge = st(r, tol)
        detail = f'max asymmetry {fe(r)} over the octant orbit · any solver bug breaks this first'
    else:
        badge = 'n/a'
        detail = ('needs an odd grid N for a true centre cell' if n % 2 == 0
                  else f'the {p.shape} domain is not 8-fold symmetric, so this orbit does not apply' if not sym8
                  else (f'the source is offset to ({fmt(p.sx, 3)}, {fmt(p.sy, 3)}) — the field is correctly NOT '
                        '8-fold symmetric; press "centre" to restore the orbit') if not on_axis
                  else 'needs a centred source, uniform medium, symmetric BC' if not centred
                  else port if not sol_ok
                  else 'not running yet')
    if not x.fp64 and good:
        port = (port + ' · ' if port else '') + f'f32 build: tolerance {fe(tol)} (the web app\'s 1e-12 is f64)'
    return _row('sym8', 'centred source keeps 8-fold symmetry (exact, machine precision)', badge, detail, 'solver',
                fe(r) if r is not None else '', r, port=port if (sol_ok or badge != 'n/a') else '')


def _row_sym6(x):
    m, p = x.m, x.p
    fold = 6 if p.shape in ('circle', 'n6', 'n12') else 3 if p.shape == 'n3' else 0
    if x.arr:
        fold = 6 if x.tri_torus else 0        # the rhombic torus' period lattice is 6-fold; the array is 2-fold
    whole = x.fits()
    good = (bool(fold) and whole and x.src_centred() and p.medium == 'uniform' and not x.drops
            and p.src in ('pulse', 'impulse') and x.live)
    port = ''
    sol_ok = True
    if good and x.port_abs:
        # 10/10/2026: the absorbing layer's damping map must be invariant under the orbit too (cell by cell)
        if x.layer_sym('rot', fold):
            port = f'{x.abs_text()}: a {fold}-fold symmetric boundary (its damping map checked cell by cell)'
        else:
            sol_ok = False
            port = (f'{x.abs_text()}: its damping map (the distance ramp, quantised per cell) is not {fold}-fold '
                    'symmetric (checked cell by cell), so this orbit does not apply')
    if good and x.solids and sol_ok:
        c = x.c
        mask_ok = x.G.get_or(f'maskrot{fold}', lambda: _mask_rot(m, m.mask_np, 6 // fold) and
                             (_type_in(m) is None or _mask_rot(m, _type_in(m), 6 // fold)))
        at_c = (int(m.src[0]), int(m.src[1])) == (c, c)
        sol_ok = mask_ok and at_c
        port = ((f'solids: the free-cell mask is {fold}-fold symmetric (checked cell by cell) and the source '
                 'is on the centre cell, so the orbit still applies') if sol_ok else
                (f'{_SOL}: the free-cell mask (or the pinned / rigid types) is not {fold}-fold symmetric (checked '
                 'cell by cell), so this orbit does not apply') if not mask_ok else
                (f'{_SOL}: the source snapped out of a solid to cell ({int(m.src[0])}, {int(m.src[1])}), '
                 'off the centre'))
    good = good and sol_ok
    r = orbit_residual(x, 'rot', fold) if good else None
    tol = x.tol(1e-12, 'sym')
    if good and r is not None:
        badge = st(r, tol)
        detail = (f'max |u(p) − u(rot p)| = {fe(r)} over the {fold}-fold orbit · the rotation is (a,b) → '
                  '(−b, a+b) in LATTICE coordinates — integer, order exactly 6, and it permutes the six '
                  'neighbours in one cycle')
    else:
        badge = 'n/a'
        detail = (('the rhombic array is only 2-fold (point-) symmetric — 3- and 6-fold orbits need the hexagon, '
                   'the triangle, the circle or the rhombic torus (bc periodic)') if (x.arr and not fold)
                  else (f'the {p.shape} domain is 4- or 2-fold, and neither is a rotation of the triangular lattice '
                   '— only 3- and 6-fold orbits exist here') if not fold
                  else ('the shape overflows the rhombic array, and the clip is not rotationally symmetric — '
                        'lower the radius until it fits') if not whole
                  else 'the source is offset, so the field is correctly not rotationally symmetric'
                  if not x.src_centred()
                  else port if not sol_ok
                  else 'needs a centred pulse or impulse in a uniform medium')
    if x.tri_torus and good:
        port = (port + ' · ' if port else '') + ('the rhombic torus: its period lattice is 6-fold too, so the orbit '
                                                 'is taken mod N and holds after the wave wraps')
    if not x.fp64 and good:
        port = (port + ' · ' if port else '') + f'f32 build: tolerance {fe(tol)} (the web app\'s 1e-12 is f64)'
    return _row('sym6', f'triangular lattice: centred source keeps {fold or 6}-fold symmetry (exact)', badge, detail,
                'tier C', fe(r) if r is not None else '', r, port=port if (sol_ok or badge != 'n/a') else '')


def _row_conform(x):
    m, p = x.m, x.p
    rot_deg = p.rot * 180 / PI
    at30 = m.at_conf_rot() and x.fits()
    cells = x.cells_domain()
    exactN, form = None, ''
    R = m.dom_R()
    rh60 = p.shape == 'rhomb' and abs(p.rtheta - PI / 3) < 1e-6
    if p.shape == 'n6':
        L = jround(R)
        if abs(R - L) < 0.26 and at30:
            exactN, form = 3 * L * L + 3 * L + 1, f'3L²+3L+1, L = {L}'
    elif rh60:
        s = jround(2 * R / math.sqrt(3))
        if at30:
            exactN, form = (s + 1) * (s + 1), f'(s+1)², s = {s}'
    elif p.shape == 'n3':
        L = jround(R * math.sqrt(3))
        if at30 and L % 3 == 0:
            exactN, form = (L + 1) * (L + 2) // 2, f'(L+1)(L+2)/2, L = {L}'
    elif x.arr:
        # the whole rhombic array (port only): every lattice point but the ring, or all N^2 on the torus;
        # 10/10/2026: with an absorbing frame the OPEN domain, the array less the frame's Ld rows per side
        Ld = x.frame_Ld()
        exactN, form = ((m.N * m.N, f'N², the rhombic torus, N = {m.N}') if x.tri_torus
                        else ((m.N - 2 * Ld) ** 2, f'(N−2·{Ld})², the rhombic array less its {Ld}-row absorbing frame, '
                              f'N = {m.N}') if Ld
                        else ((m.N - 2) ** 2, f'(N−2)², the rhombic array less its ring, N = {m.N}'))
    if exactN is not None:
        badge = 'PASS' if cells == exactN else 'FAIL'
        detail = (f'{cells} cells vs the exact lattice-point count {exactN} ({form}) — zero staircase, the '
                  'boundary runs along lattice rows')
    elif p.shape in ('n6', 'n3') or rh60:
        badge = 'n/a'
        detail = (f'{{3,4,6}} tilers conform at rotation 30° or 90° (currently {fmt(rot_deg, 2)}°)'
                  + (' with side L a multiple of 3 — otherwise the centroid is not a lattice site'
                     if p.shape == 'n3' else '')
                  + (' with an EVEN side — its vertices are ±(s/2)(e₁±e₂)' if p.shape == 'rhomb' else '')
                  + ' and an integer side; press "snap conforming"')
    else:
        badge = 'n/a'
        detail = ('only the triangle, hexagon and rhombus(60°) tile the triangular lattice — the '
                  'crystallographic restriction allows n = 3,4,6 only, so {5,8,12,∞} never conform')
    port = ''
    if x.unl:
        port = ('the Unlicensed plate is a rounded rectangle: its corner arcs are not lattice lines, so it '
                'never conforms')
    if x.arr and exactN is not None:
        port = 'the "array" shape (port only): the domain is the triangular lattice\'s own rhombic array'
    if x.solids and exactN is not None:
        port = (port + ' · ' if port else '') + 'counts the DOMAIN mask (solids excluded: they do not change how the shape is rasterised)'
    return _row('conform', ('conforming tiler: the mask IS the shape’s lattice points' if exactN is not None
                            else 'conformance: this pair is not lattice-conforming'), badge, detail, 'tier C',
                f'{cells} / {exactN}' if exactN is not None else '', cells if exactN is not None else None,
                dsg=exactN is None, port=port)


def _mirror_symmetric_js(x):
    m, p = x.m, x.p
    if x.arr:                                  # port only: the square / the rhombic torus are, the rhombus is not
        return x.tri_torus or not x.tri
    if x.tri and not x.fits():
        return False
    if p.shape in ('square', 'circle'):
        return True
    q = PI / 2 if p.shape == 'rhomb' else (PI / int(p.shape[1:]) if p.shape[1:].isdigit() else math.nan)
    t = p.rot / q
    return abs(t - jround(t)) < 1e-9 if math.isfinite(t) else False


def _row_mirror(x):
    m, p = x.m, x.p
    sy0 = abs(p.sy) < 1e-9
    src_ok = sy0 if p.src in ('pulse', 'impulse') else (p.src == 'vtx')
    port = ''
    maskgeo = x.solids or x.unl                  # the premise is checked on the mask itself
    if maskgeo:
        mask_ok = x.G.get_or('maskmirror', lambda: _mask_mirror(m, m.mask_np) and
                             (_type_in(m) is None or _mask_mirror(m, _type_in(m))))
        geo_shape = mask_ok and not (x.tri and not x.fits()) and (_mirror_symmetric_js(x) if x.arr else True)
    else:
        geo_shape = _mirror_symmetric_js(x)
    # 10/10/2026: an absorbing layer's damping map must be mirror-symmetric too (cell by cell); the port's
    # absorbing boundaries are otherwise mirror-symmetric (a masked shape with bc 'mur' and an open edge never
    # reaches the Mur ring)
    layer_ok = (not x.port_abs) or x.layer_sym('mirror')
    geo = geo_shape and p.medium == 'uniform' and (p.bc != 'mur' or x.absorbing == 'open') and layer_ok
    cell_ok = True
    if maskgeo and src_ok:
        c = x.c
        if p.src in ('pulse', 'impulse'):
            cell_ok = int(m.src[1]) == c
        else:
            S = {(int(v[0]), int(v[1])) for v in m.vtx}
            cell_ok = S == {_mirror_ij(m, i, j) for (i, j) in S}
    good = src_ok and geo and x.live and not x.drops and cell_ok
    r = orbit_residual(x, 'mirror') if good else None
    tol = x.tol(1e-12, 'sym')
    fixes = ()
    if good and r is not None:
        badge = st(r, tol)
        detail = (f'max |u(x, +y) − u(x, −y)| = {fe(r)} · a row swap, so it is exact on the grid · the triangle '
                  'and pentagon have NO 8-fold orbit, so this is their only exact solver test')
    else:
        badge = 'n/a'
        if not geo:
            if p.medium != 'uniform':
                detail = 'the slab and lens are not mirror-symmetric about x'
            elif x.arr and x.tri and not x.tri_torus:
                detail = ('the triangular array is a 60° rhombus — point-symmetric but not mirror-symmetric about x '
                          '(the rhombic torus is: bc periodic)')
            elif x.tri and not x.fits():
                detail = ('the triangular ARRAY is a rhombus — point-symmetric but not mirror-symmetric — so a '
                          'shape it clips loses the mirror. Reduce the radius until the shape fits inside the array.')
            elif geo_shape and not layer_ok:
                detail = (f'{x.abs_text()}: its damping map (the distance ramp, quantised per cell) is not '
                          'mirror-symmetric about x (checked cell by cell)')
            elif p.bc == 'mur' and x.absorbing != 'open':
                detail = 'first-order Mur is not exactly mirror-symmetric off-axis'
            elif maskgeo:
                detail = ((f'{_SOL}: ' if x.solids else '') + 'the free-cell mask is not mirror-symmetric about x '
                          '(checked cell by cell' + (': the plate’s lettering, or its rotation' if x.unl else '')
                          + (', the pinned / rigid types included' if x.rigid_solids else '') + ')')
            else:
                detail = (f'rot = {fmt(p.rot * 180 / PI, 1)}° is not a multiple of 180°/n, so the domain itself is '
                          'not mirror-symmetric about x')
        elif x.drops:
            detail = f'{x.drops} manual pulse(s) dropped — reset to restore a clean invariant'
        else:
            sy_txt = f'source y = {fmt(p.sy, 3)} ≠ 0 — move it back onto the x-axis (source x is free)'
            # the web app: any source with sy != 0 gets the sy message, anything else "needs a pulse,
            # impulse or vertex source" -- also a pulse that simply has not run 6 steps yet
            js = sy_txt if not sy0 else 'needs a pulse, impulse or vertex source'
            if not cell_ok:
                fx = (f'{_SOL}: the source cells are not mirror images (they snap out of solids)'
                      if x.solids else 'the vertex source cells are not mirror images')
            elif not src_ok:
                fx = sy_txt if (not sy0 and p.src in ('pulse', 'impulse')) else 'needs a pulse, impulse or vertex source'
            elif x.step <= 5:
                fx = 'not running yet'
            elif not math.isfinite(x.amax) or (x.E_last is not None and not math.isfinite(x.E_last)):
                fx = 'the run diverged: the field is not finite'
            else:
                fx = 'the field is zero (max|u| ≤ 1e-12)'
            if not cell_ok or x.compat or fx == js:
                detail = fx if not cell_ok else js
            else:
                detail, fixes = fx, ('mirror-live',)
    if maskgeo and badge != 'n/a':
        port = ('the mirror premise was checked on the free-cell mask (cell by cell)'
                + ('; solids included' if x.solids else '') + ('; their pinned / rigid types too' if x.rigid_solids else ''))
    if x.tri_torus and badge != 'n/a':
        port = (port + ' · ' if port else '') + 'the rhombic torus: the mirror is taken mod N'
    if x.port_abs and badge != 'n/a':
        port = (port + ' · ' if port else '') + (f'{x.abs_text()}: a mirror-symmetric boundary'
                                                 + (' (its damping map checked cell by cell)' if x.abs_np is not None
                                                    else ''))
    if not x.fp64 and good:
        port = (port + ' · ' if port else '') + f'f32 build: tolerance {fe(tol)} (the web app\'s 1e-12 is f64)'
    return _row('mirror', 'source on the x-axis keeps MIRROR symmetry (exact — works for odd n)', badge, detail,
                'solver', fe(r) if r is not None else '', r, port=port, fixes=fixes)


def _row_energy(x):
    m, p = x.m, x.p
    E_web, E_tri = x.hist.get('E') or [], x.hist.get('Etri') or []
    # FIX tri-energy (05/10/2026): on the triangular lattice grade the triangular operator's own
    # energy (the third link direction recorded too); the web app's series is the square functional
    use_tri = x.tri and x.fix('tri-energy') and len(E_tri) == len(E_web)
    es_web = energy_stats(E_web)
    es = energy_stats(E_tri) if use_tri else es_web
    fixes = ()
    if x.port_abs:
        # 10/10/2026: a port-only absorbing boundary (both modes: the web app has no case for it) removes the
        # energy by design; the 'eopen' row grades instead that the open domain's energy does not grow
        undamped = False
    elif x.fix('edge-kind'):
        e = x.edge()
        undamped = p.damp < 1e-9 and e not in ('mur', 'mixed-mur')      # no Mur part: a closed edge
        if (undamped != (p.damp < 1e-9 and p.bc in ('dirichlet', 'periodic', 'neumann'))
                and p.medium == 'uniform' and not x.drops
                and p.src not in ('cont', 'slit', 'beam') and not (p.src == 'vtx' and p.vtx_drive)):
            fixes = ('edge-kind',)
    else:
        undamped = p.damp < 1e-9 and p.bc in ('dirichlet', 'periodic', 'neumann')
    uniform = p.medium == 'uniform'
    closed = p.src not in ('cont', 'slit', 'beam') and not (p.src == 'vtx' and p.vtx_drive) and not x.drops
    # 01/10/2026 (beam phase): the recorder's energy is the SQUARE lattice's (forward differences
    # along the index axes). On the triangular lattice it is not the conserved quantity, and for a
    # DIRECTED packet it changes whenever the packet turns (a wall): 5.4% drift at 1200 steps on n6
    # while the solver is exact (a pulse, being isotropic, stays at 0.7%). Not graded there.
    def render(es, tri_packet):
        ok = es is not None and undamped and uniform and closed and x.live and not tri_packet
        badge = st(es['drift'], 0.02) if ok else 'n/a'
        detail = _energy_detail(x, es, tri_packet, uniform, closed, undamped)
        return ok, badge, detail
    ok, badge, detail = render(es, x.tri and p.src == 'packet' and not use_tri)
    if use_tri and render(es_web, p.src == 'packet') != (ok, badge, detail):
        fixes = fixes + ('tri-energy',)
    port = ''
    if x.solids and ok and not x.rigid_solids:
        port = 'solids are pinned (u = 0) walls: the same conserved energy as a Dirichlet domain'
    wp = x.wall_port() if ok else ''
    if wp:
        port = (port + ' · ' if port else '') + wp
    if x.tri and x.arr and x.bc_fallback and ok:
        port = (port + ' · ' if port else '') + (x.bc_reason or mc.TRI_MUR_REASON)
    if x.port_abs:
        port = (port + ' · ' if port else '') + (f'{x.abs_text()} absorbs: the energy leaves by design (the recorder sums '
                                                 'the open domain only); see "energy inside the open domain does not '
                                                 'grow"')
    if use_tri and ok:
        port = (port + ' · ' if port else '') + ("graded on the triangular operator's own energy: the third link "
                                                 "direction is recorded too (FIX tri-energy)")
    return _row('energy', 'energy bounded, no secular drift (leapfrog conserves a MODIFIED energy)', badge, detail,
                'scheme', f'drift {fe(es["drift"])}' if es else '', es['drift'] if es else None, port=port,
                fixes=fixes)


EOPEN_TOL = 1.05          # no growth: the later half's max may exceed the earlier half's by the functional's wobble
EOPEN_EMPTY = 1e-8        # below this share of the run's peak the open domain is empty (rounding, a static mode)


def _row_eopen(x):
    """Port-only (10/10/2026), for the port's absorbing boundaries only: with no drive the energy inside the OPEN
    domain (the recorder sums it only) does not grow -- the layer removes energy and returns at most its small
    reflection. Graded on the recorded history window (the newest HIST_CAP['E'] steps): the maximum over its
    later half <= EOPEN_TOL x the maximum over its earlier half, or the earlier half already holds under
    EOPEN_EMPTY of the run's peak (Membrane.hist_emax). Not "never above the start": the recorded functional
    kin/(2 Co^2) + pot/2 lags the conserved leapfrog energy by O(dt) -- released from rest it reads half the
    potential and rises 10-23% within ~10 steps (sigma 3, Co 0.5-0.8) also in a closed drum, where the halves stay
    within 0.96-1.03 (ScratchCLAUDE/Round5CLAUDE/AbsorbCLAUDE/probe_eopen_MIRROR.log). n/a for a driven source,
    dropped pulses or a non-uniform medium (the functional assumes uniform c)."""
    p = x.p
    E = x.hist.get('E') or []
    name = 'energy inside the open domain does not grow (absorbing boundary, no drive)'
    driven = p.src in ('cont', 'slit', 'beam') or (p.src == 'vtx' and p.vtx_drive)
    why = ('a driven source injects energy every step' if driven
           else f'{x.drops} manual pulse(s) dropped — each one adds energy; reset' if x.drops
           else 'variable medium: the recorded functional assumes uniform c' if p.medium != 'uniform'
           else 'run a few dozen steps first' if len(E) < 40
           else '')
    port = f'{x.abs_text()}; the recorder sums the open domain only (the layer is excluded)'
    if why:
        return _row('eopen', name, 'n/a', why, 'port', port=port)
    Ea = np.asarray(E, dtype=np.float64)
    h = len(Ea) // 2
    e1, e2 = float(np.max(Ea[:h])), float(np.max(Ea[h:]))
    peak = getattr(x.m, 'hist_emax', None)
    peak = float(np.max(Ea)) if peak is None else float(peak)
    s0 = x.step - len(Ea) + 1                    # the window's first step
    fin = math.isfinite(e1) and math.isfinite(e2) and math.isfinite(peak) and peak > 0
    empty = fin and e1 <= EOPEN_EMPTY * peak
    r = e2 / e1 if (fin and e1 > 0) else math.nan
    badge = 'PASS' if (fin and (empty or r <= EOPEN_TOL)) else 'FAIL'
    detail = ((f'max E over steps {s0 + h}–{x.step} / max over {s0}–{s0 + h - 1} = {fe(r)} (≤ {fmt(EOPEN_TOL, 2)}: '
               'the recorded functional wobbles by a few % as kinetic and potential energy trade)'
               + (f' · the open domain is empty (≤ {fe(EOPEN_EMPTY)} of its peak)' if empty else '')
               + f' · now {fe(Ea[-1] / peak)} of the peak: the rest left through {x.abs_text()}') if fin
              else 'the recorded energy is not finite: the run diverged')
    return _row('eopen', name, badge, detail, 'port', f'late/early {fe(r)}' if math.isfinite(r) else '',
                r if math.isfinite(r) else None, port=port)


def _energy_detail(x, es, tri_packet, uniform, closed, undamped):
    p = x.p
    if tri_packet:
        detail = ('directed packet on the triangular lattice: the recorded energy uses the square lattice’s '
                  'differences, which is not conserved for a directional wave there (it changes when the '
                  'packet turns at a wall) -- not graded; use a pulse for this row')
    elif not uniform:
        detail = ('variable medium: this functional assumes uniform c, so it is not the conserved quantity '
                  'here (the correct one is c-weighted)')
    elif x.drops:
        detail = (f'{x.drops} manual pulse(s) dropped — each one ADDS energy, so drift is correct here; reset '
                  'for a closed system')
    elif not closed:
        detail = 'a driven source injects energy every step — growth is correct here, not drift'
    elif not undamped:
        detail = ('damping is on, so energy SHOULD decay' if p.damp >= 1e-9 else
                  'absorbing edge removes energy by design')
        if x.port_abs and p.damp < 1e-9:
            detail = f'absorbing edge removes energy by design ({x.abs_text()})'
    elif es is not None:
        detail = (f'drift {fe(es["drift"])} of the mean over the window · oscillation ±{fmt(100 * es["spread"] / 2, 2)}% '
                  'is EXPECTED (O(Δt²)), not an error')
    else:
        detail = ('run to ~1200 steps: the window must span many oscillation periods before a slope fit can '
                  'tell drift from the standing-wave beat')
    return detail


def _row_recip(x):
    return _row('recip', 'reciprocity: swapping source and receiver gives the same signal', 'n/a',
                'press "reciprocity" — not yet wired in this build', 'linear')


def _row_bc(x):
    m, p = x.m, x.p
    P = x.hist.get('pA') or []
    fixes = ()
    port = ''
    if x.solids and p.src in ('pulse', 'impulse'):
        return _row('bc', 'boundary behaviour at the probe', 'n/a',
                    f'{_SOL}: they reflect the wave too, so the edge\'s own reflection cannot be isolated at the '
                    'probe', 'BC')
    if len(P) > 300 and p.src in ('pulse', 'impulse'):
        q = min(len(P), 1200)
        early = 0.0
        for i in range(min(140, q)):
            early = _jmax(early, abs(P[i]))
        late = 0.0
        for i in range(math.floor(q * 0.55), q):
            late = _jmax(late, abs(P[i]))
        ratio = late / early if early > 1e-12 else math.nan
        absorbing = p.bc == 'mur'
        if x.port_abs:
            # 10/10/2026: the port's absorbing boundaries (both modes: the web app has no case for them)
            name = 'absorbing edge: little comes back'
            if p.src == 'impulse':
                # the impulse's grid-scale part travels at ~zero group velocity and reaches the probe late whatever the
                # edge does (the web app's Mur1 reads FAIL here for that reason: late/early 0.28 with Mur2 at N 161)
                return _row('bc', name, 'n/a', 'needs a pulse: an impulse\'s grid-scale tail reaches the probe late '
                            'whatever the edge does (its group velocity is ~0)', 'BC',
                            port='a port-only absorbing boundary (NotesCLAUDE/absorbing_MIRROR.md)')
            if math.isfinite(ratio):
                return _row('bc', name, st(ratio, 0.15),
                            f'late/early amplitude at the probe = {fe(ratio)} · {x.abs_text()}', 'BC',
                            f'late/early {fe(ratio)}', ratio,
                            port='a port-only absorbing boundary (NotesCLAUDE/absorbing_MIRROR.md)')
            return _row('bc', name, 'n/a', 'run longer', 'BC')
        if x.fix('edge-kind'):
            e = x.edge()
            if absorbing and e != 'mur':
                fixes = ('edge-kind',)
                if e.startswith('mixed'):
                    return _row('bc', 'boundary behaviour at the probe', 'n/a',
                                'the boundary is part Mur ring, part clamped flavor wall: neither "little comes '
                                'back" nor a pure drum applies', 'BC', fixes=fixes)
                absorbing = False
                port = x.mur_port('Mur is selected, but it acts on the array ring only: this domain\'s wall is the clamped '
                                  + ('triangular ring' if x.tri else 'mask edge'))
        name = 'absorbing edge: little comes back' if absorbing else 'fixed edge: the wave DOES come back'
        if not absorbing and (x.rigid_ring or x.edge_rigid or x.tri_torus) and not x.compat:
            extra = ('the rhombic torus: the wave comes back by wrapping around the seam, not by reflection'
                     if x.tri_torus else 'the edge is RIGID: the echo comes back upright (R = +1), not inverted')
            port = (port + ' · ' if port else '') + extra
        if math.isfinite(ratio):
            badge = st(ratio, 0.15) if absorbing else ('PASS' if ratio > 0.05 else 'FAIL')
            detail = (f'late/early amplitude at the probe = {fe(ratio)} '
                      + (f'· Mur k = (Co−1)/(Co+1) = {fmt((p.Co - 1) / (p.Co + 1), 4)}; 1st order leaks off-axis'
                         if absorbing else '· a drum is supposed to ring'))
        else:
            badge, detail = 'n/a', 'run longer'
        return _row('bc', name, badge, detail, 'BC', f'late/early {fe(ratio)}' if math.isfinite(ratio) else '',
                    ratio if math.isfinite(ratio) else None, port=port, fixes=fixes)
    return _row('bc', 'boundary behaviour at the probe', 'n/a', 'pulse/impulse source, run longer', 'BC', port=port)


def front_free(x):
    return x.step * x.p.Co < _clear(x.m, x.G) - 4


def front_radii(x):
    """frontRadii(): the front's radius along +x and along the index diagonal, from the source."""
    if not front_free(x):
        return None
    m, p, n = x.m, x.p, x.n
    ci, cj = int(m.src[0]), int(m.src[1])
    M = m.mask_np
    CLEAR = _clear(m, x.G)
    tri_fix = x.tri and x.fix('tri-diagonal')
    dstep = math.sqrt(3) if tri_fix else SQRT2
    ba, bav, bd, bdv = 0, -1.0, 0.0, -1.0
    rmax = min(math.floor(CLEAR) - 2, math.ceil(x.step * p.Co) + 6)
    R = range(3, max(3, rmax))
    DD = [jround(r / dstep) for r in R]
    I = [ci + r for r in R] + [ci + dd for dd in DD]
    J = [cj] * len(R) + [cj + dd for dd in DD]
    ok = [i < n for i in I]                  # (the cells the JS reads; the rest are never used)
    V = gather(x, [i if o else 0 for i, o in zip(I, ok)], [j if o and j < n else 0 for j, o in zip(J, ok)])
    for k, r in enumerate(R):
        if ci + r < n and M[ci + r, cj]:
            av = abs(float(V[k]))
            if av > bav:
                bav, ba = av, r
        dd = DD[k]
        if ci + dd < n - 1 and cj + dd < n - 1 and M[ci + dd, cj + dd]:
            dv = abs(float(V[len(R) + k]))
            if dv > bdv:
                bdv, bd = dv, dd * dstep
    if bav > 1e-9 and bdv > 1e-9 and ba > 0:
        return {'ax': ba, 'dg': bd, 'ratio': bd / ba, 'fix': tri_fix}
    return None


def _row_aniso(x):
    p = x.p
    f = front_radii(x)
    centred = p.src in ('pulse', 'impulse') and p.medium == 'uniform'
    fixes = ('tri-diagonal',) if (f and f['fix'] and centred and x.live) else ()
    if f and centred and x.live:
        badge = 'measured'
        detail = (f'diagonal/axial front radius = {fmt(f["ratio"], 4)} (axis {fmt(f["ax"], 1)}, diag {fmt(f["dg"], 1)}) · '
                  'driven by POINTS PER WAVELENGTH, not by Co — measured: sweeping Co 0.1→0.7 moves it by 5e-5, '
                  'while coarsening the grid moves it by 1e-1. The Co² term cancels to leading order. Switching '
                  'to the 9-point stencil buys 145–2500×')
    else:
        badge = 'n/a'
        detail = ('single point source (pulse or impulse), uniform medium' if not centred
                  else (f'the front has already reached the wall (step {x.step}; the source has only '
                        f'{fmt(_clear(x.m, x.G), 1)} cells of clearance) — reset, move the source away from the wall, '
                        'or use the absorbing boundary; after reflection there is no single front to measure')
                  if not front_free(x) else 'not running yet')
    port = ''
    if x.solids:
        port = 'the clearance counts solids as walls: the front is measured only while no solid has been reached'
    if fixes:
        port = (port + ' · ' if port else '') + ('triangular lattice: the index diagonal is 30° and √3 long '
                                                 '(FIX tri-diagonal)')
    return _row('aniso', 'grid anisotropy: the front is NOT a circle (it cannot be)', badge, detail, 'canon',
                f'ratio {fmt(f["ratio"], 4)}' if (f and centred and x.live) else '',
                f['ratio'] if (f and centred and x.live) else None, dsg=True, port=port, fixes=fixes)


def _row_wake(x):
    m, p = x.m, x.p
    P = x.hist.get('pA') or []
    mur = p.bc == 'mur'
    fixes = ()
    rest = p.src in ('pulse', 'impulse') and p.medium == 'uniform' and not x.drops
    if x.port_abs:
        mur = True                              # 10/10/2026: any port absorbing boundary is "the ABSORBING boundary"
    elif x.fix('edge-kind') and mur and x.edge() != 'mur':
        mur = False
        fixes = ('edge-kind',) if rest else ()
    ok_src = rest and mur
    c = jround((x.n - 1) / 2)
    dprobe = jhypot(c + m.pA() - m.src[0], c - m.src[1])
    co = max(p.Co, 1e-9)
    t_front = dprobe / co
    w0 = math.ceil(t_front + 4 * p.sigma / co)
    w1 = math.ceil(w0 + 3 * t_front)
    if x.solids and ok_src:
        return _row('wake', '2D leaves a WAKE (it cannot be sharp)', 'n/a',
                    f'{_SOL}: they reflect the front back to the probe, so the tail is not the free-space wake',
                    'Hadamard', dsg=True)
    if ok_src and len(P) > w1:
        pk = 0.0
        for v in P:
            pk = _jmax(pk, abs(v))
        tail = 0.0
        for i in range(w0, min(w1, len(P))):
            tail = _jmax(tail, abs(P[i]))
        rel = tail / pk if pk > 1e-12 else math.nan
        fin = math.isfinite(rel)
        return _row('wake', '2D leaves a WAKE after the front (it cannot be sharp)', 'measured' if fin else 'n/a',
                    (f'tail/peak = {fe(rel)} in steps {w0}–{w1} (just after the front at ~{jround(t_front)}) · the '
                     'wake decays as a power law ~t⁻², so it is largest here · stated: an identical 3D run gives '
                     '3e-5, 11369× smaller') if fin else '', 'Hadamard',
                    f'tail/peak {fe(rel)}' if fin else '', rel if fin else None, dsg=True,
                    port=f'the absorbing boundary is {x.abs_text()}' if x.port_abs else '')
    detail = ('needs a centred pulse with the ABSORBING boundary (else reflections mask it)' if not ok_src
              else f'run to at least step {w1} (the window opens just after the front)')
    port = ''
    if fixes and p.src in ('pulse', 'impulse'):
        port = x.mur_port('Mur is selected, but it acts on the array ring only; this domain\'s wall is clamped '
                          '(FIX edge-kind)')
    return _row('wake', '2D leaves a WAKE (it cannot be sharp)', 'n/a', detail, 'Hadamard', dsg=True, port=port,
                fixes=fixes)


def _row_stair(x):
    m, p = x.m, x.p
    R = m.dom_R()
    cells = x.cells_domain()
    covered = cells * (S3H if x.tri else 1)
    exact = covered if (x.arr and x.tri) else shape_area(x, R)
    rel = abs(covered - exact) / exact
    sq = (p.shape in ('square', 'array') and not x.tri) or (x.arr and x.tri)
    quantised = x.full or abs(2 * R - (2 * math.floor(R) + 1)) < 1e-9
    aligned = sq and quantised
    ideal = exact if (x.arr and x.tri) else ideal_area(x)
    lost = 1 - exact / ideal if ideal > 0 else 0
    clipped = lost > 1e-9
    fixes = ()
    if clipped:
        name, badge = 'partial flavor: the shape no longer fits the grid', 'measured'
    elif aligned:
        name, badge = 'grid-aligned domain: the mask IS the shape, exactly', st(rel, 1e-12)
    elif sq:
        name, badge = 'grid-aligned square is no longer exact — the R snap has broken', 'FAIL'
    else:
        name, badge = 'staircase: a square grid cannot represent this domain exactly', 'measured'
    detail = (f'mask {cells} cells' + (f' × √3/2 = {fmt(covered, 1)}' if x.tri else '')
              + f' vs exact {fmt(exact, 1)} → relative error {fe(rel)}')
    if clipped:
        detail += (f' · {fmt(100 * lost, 1)}% of the flavor shape lies outside the grid, so the domain is shape ∩ '
                   'square — the boundary is part flavor wall, part grid edge. The comparison above is against the '
                   'CLIPPED shape, so this is still staircase error and not the missing corners.')
    elif aligned:
        if x.frame_Ld():
            # 10/10/2026: an absorbing frame (both lattices): the open domain is the array less the frame
            Ld = x.frame_Ld()
            detail += (f' · the open domain is the whole array less its {Ld}-row absorbing frame '
                       f'({m.N - 2 * Ld} × {m.N - 2 * Ld} lattice points): its edges are lattice rows, so there is no '
                       'staircase and no quantisation')
        elif x.arr and x.tri:
            detail += (f' · the domain is the whole rhombic array ({m.N} × {m.N} lattice points'
                       + (', the torus' if x.tri_torus else ' less its ring') + '): its walls are the array\'s '
                       'lattice rows, so there is no staircase and no quantisation')
        elif m.sq_full() and x.fix('sqfull-side'):
            detail += (f' · the domain is the whole {m.N} × {m.N} array: its walls are the array edge, so there '
                       'is no staircase and no quantisation')
            fixes = ('sqfull-side',)
        else:
            detail += (f' · R = {fmt(R, 1)} cells, so 2R = {fmt(2 * R, 0)} is an odd integer and the side is '
                       f'{2 * math.floor(R) + 1} whole cells — no staircase, no quantisation')
    elif sq:
        detail += f' · 2R = {fmt(2 * R, 3)} is not an odd integer, so the wall lands between cells'
    elif x.unl:
        detail += (f' · the plate\'s straight sides can run along lattice rows, its corner arcs cannot · '
                   f'R = {fmt(R, 1)} cells')
    else:
        detail += (' · falls as ~1/N; the conforming lattice for a tiler {3,4,6} would be exact, a non-tiler '
                   '{5,∞} has none · shrinking the radius at fixed N makes this WORSE: '
                   f'R = {fmt(R, 1)} cells')
    port = ''
    if x.solids:
        port = 'counts the DOMAIN mask (solids excluded: they are objects inside it, not its staircase)'
    if x.unl:
        port = (port + ' · ' if port else '') + 'the plate\'s exact area is W·H − (4 − π)r² (closed form)'
    if x.arr and x.tri:
        port = (port + ' · ' if port else '') + ('the "array" shape (port only): the mask is the lattice itself, so the '
                                                 'exact area is the cell count × √3/2')
    if x.layer is not None:
        port = (port + ' · ' if port else '') + (f'counts the OPEN domain ({x.abs_text()} excluded: its cells are not '
                                                 'part of the drawn shape)')
    return _row('stair', name, badge, detail, 'tier A', f'rel err {fe(rel)}', rel,
                dsg=(clipped or not (aligned or sq)), port=port, fixes=fixes)


def _rows_align(x):
    out = []
    for lab, status, txt, tier in al.al_ledger_rows(x.m):
        key = 'align' if lab.startswith('edge alignment: each') else 'straight'
        port = 'measured on the DOMAIN mask (solids are not domain edges)' if x.solids else ''
        out.append(_row(key, lab, status, txt, tier, port=port))
    return out


def cone_measure(x):
    """coneMeasure() on the device (k_cone); cone_measure_np is the numpy reference."""
    if not DEVICE:
        return cone_measure_np(x)
    m, p = x.m, x.p
    OUT = _out_buf()
    kind = 2 if x.tri else (1 if p.stencil == 9 else 0)
    wrap = 1 if _wrap_cone(m) else 0
    k_cone(m.U, _cur(m), int(m.src[0]), int(m.src[1]), x.step, kind, wrap, 1 if x.tri else 0,
           p.Co * x.step + 1.5, OUT)
    o = OUT.to_numpy().astype(np.float64)
    e_all, e_in = float(o[:RSLOT, 2].sum()), float(o[:RSLOT, 3].sum())
    ring = math.nan if o[:RSLOT, 5].max() > 0 else float(o[:RSLOT, 4].max())
    return {'m': x.step, 'rOut': int(o[RSLOT, 0]), 'outside': int(round(o[:RSLOT, 1].sum())),
            'frac': e_in / e_all if e_all > 0 else math.nan, 'ringMax': ring}


def cone_measure_np(x):
    """coneMeasure(): support radius (cone norm) of the nonzero field, cells outside the cone of
    radius m, the energy share inside the physical disk Co*m + 1.5, and the largest |u| on the
    outermost ring. In chunks of columns (~4M cells each): at N = 8193 a late impulse covers
    ~20M cells, and whole-grid index arrays would hold ~1 GB of temporaries."""
    m, p, n = x.m, x.p, x.n
    si, sj = int(m.src[0]), int(m.src[1])
    mstep = x.step
    wrap = _wrap_cone(m)
    h = n // 2
    mi = (lambda d: ((d % n) + n + h) % n - h) if wrap else (lambda d: d)   # d in (-n, n): JS % == numpy %

    def cone_norm_w(da, db):
        r = cone_norm(m, da, db)
        if wrap and x.tri:                   # the rhombic torus: the hexagonal norm's nearest image
            for sa, sb in ((-n, 0), (n, 0), (0, -n), (0, n), (-n, n), (n, -n)):
                r = np.minimum(r, cone_norm(m, da + sa, db + sb))
        return r
    U = x.U()
    sx, sy = m.cell_xy(si, sj)
    r_disk = p.Co * mstep + 1.5
    CH = max(1, (1 << 22) // n)
    r_out, outside, e_all, e_in = 0, 0, 0.0, 0.0
    for a0 in range(0, n, CH):
        I, J = np.nonzero(U[a0:a0 + CH] != 0)        # NaN != 0 counts, as in the JS
        if not I.size:
            continue
        I = I + a0
        V = U[I, J].astype(np.float64)
        r = cone_norm_w(mi(I - si), mi(J - sj))
        r_out = max(r_out, int(r.max()))
        outside += int(np.count_nonzero(r > mstep))
        e = V * V
        e_all += float(e.sum())
        X, Y = _cell_xy_np(m, I, J)
        e_in += float(e[al.js_hypot_np(X - sx, Y - sy) <= r_disk].sum())
    # the ring cone_norm == r_out (zeros included) lies inside the index box of half-size r_out
    # (every cone norm is >= max(|a|, |b|)); with the periodic wrap, the whole grid
    if wrap:
        i0, i1, j0, j1 = 0, n, 0, n
    else:
        i0, i1, j0, j1 = max(0, si - r_out), min(n, si + r_out + 1), max(0, sj - r_out), min(n, sj + r_out + 1)
    ring_max = 0.0
    Jb = np.arange(j0, j1)
    for a0 in range(i0, i1, CH):
        a1 = min(i1, a0 + CH)
        Ib = np.arange(a0, a1)
        ring = cone_norm_w(mi(Ib[:, None] - si), mi(Jb[None, :] - sj)) == r_out
        ring_max = _jmax(ring_max, _jsmax(np.abs(U[a0:a1, j0:j1][ring]).astype(np.float64)))
    return {'m': mstep, 'rOut': r_out, 'outside': outside, 'frac': e_in / e_all if e_all > 0 else math.nan,
            'ringMax': ring_max}


def _row_cone(x):
    m, p = x.m, x.p
    imp = p.src == 'impulse'
    lim = x.co_limit()
    ri = S3H if x.tri else (1 if p.stencil == 9 else SQRT1_2)
    reach = 'hexagon' if x.tri else ('square' if p.stencil == 9 else 'diamond')
    geom = (f'the wave’s disk (radius Co·m) fits the {reach} iff Co ≤ its inradius {fmt(ri, 4)} (necessary); the sharp '
            f'limit is {fmt(lim, 4)}'
            + (' — they COINCIDE here' if abs(ri - lim) < 1e-9 else
               f' — a gap of {fmt(ri - lim, 4)}: this lattice’s worst mode cannot put every neighbour in antiphase, so '
               'its cone overstates the speed'))
    name = f'lattice light cone: after m ticks the field is EXACTLY zero outside the {reach} of radius m'
    if not imp or x.step < 1:
        return _row('cone', name, 'n/a',
                    ('needs the IMPULSE source — a one-cell start; a gaussian already covers the grid' if not imp
                     else 'step the solver') + ' · ' + geom, 'causal')
    C = cone_measure(x)
    wall = _cone_wall(m, x.G)
    mm = C['m']
    clear = mm < wall
    uthr, utxt = UNDERFLOW[x.fp64]
    under = C['rOut'] < mm and clear and C['ringMax'] < uthr
    ok = C['outside'] == 0 and ((not clear) or C['rOut'] == mm or under)
    detail = (f'm = {mm} ticks: support radius {C["rOut"]} (cone norm), {C["outside"]} nonzero cells outside it'
              + ((' — the cone grows exactly one bond per tick: u₀ = h·F' if C['rOut'] == mm else
                  f' — the tip values underflowed {utxt}, not physics' if under else '') if clear
                 else f' — the cone reached the wall at m = {s_(wall)}; containment still holds')
              + f' · the wave runs at c = Co·u₀: {fmt(100 * C["frac"], 2)}% of Σu² lies inside the disk of radius '
                f'Co·m + 1.5 = {fmt(p.Co * mm + 1.5, 1)} · {geom}')
    port = 'the wall distance counts solids (they clamp like the mask edge)' if x.solids else ''
    return _row('cone', name, 'PASS' if ok else 'FAIL', detail, 'causal',
                f'radius {C["rOut"]} at m = {mm}, {C["outside"]} outside', float(C['outside']), port=port)


def _row_spectra(x):
    p = x.p
    if x.port_abs:
        # 10/10/2026: an absorbing boundary opens the domain: no standing modes, no drum spectrum
        return _row('spectra', 'stated — the drum spectrum', 'n/a',
                    f'an absorbing boundary ({x.abs_text()}): the domain is an open window with no standing modes, '
                    'so it has no drum spectrum', 'spectra', dsg=True)
    if x.solids:
        return _row('spectra', 'stated — the drum spectrum', 'n/a',
                    f'{_SOL}: the closed-form spectra are for the EMPTY domain; with objects inside it the spectrum '
                    'is numerical', 'spectra', dsg=True)
    if p.shape == 'circle':
        return _row('spectra', 'stated — circular drum overtones are IRRATIONAL (not harmonic)', 'stated',
                    'j₀₁ = 2.404826 · ratios 1.593341, 2.135549, 2.295417, 2.917295 — why a drum reads as unpitched '
                    'next to a string', 'Bessel', dsg=True)
    if p.shape in ('square', 'n4') or (x.arr and not x.tri):
        return _row('spectra', 'stated — square drum modes: f ∝ √(m²+n²)', 'stated',
                    'f₁₂/f₁₁ = √(5/2) = 1.581139 · f₂₂/f₁₁ = 2 exactly', 'Dirichlet', dsg=True)
    if p.shape == 'n3':
        return _row('spectra', 'stated — the EQUILATERAL TRIANGLE has an exact spectrum too (Lamé)', 'stated',
                    'f ∝ √(m²+mn+n²) — one of the very few non-separable domains with a closed form, so it is the '
                    'most valuable test case here · and the one the staircase damages most', 'Lamé', dsg=True)
    return _row('spectra', 'stated — no closed-form spectrum for this domain', 'stated',
                'only the square, the circle (Bessel) and the equilateral triangle (Lamé) have exact spectra in 2D; '
                'the rest are numerical', 'spectra', dsg=True)


def mode_sums(x, mm, mn, z, L):
    """(sum a b, sum a^2, sum b^2) over the masked interior, a = sin(mm pi (i-z)/L) sin(mn pi (j-z)/L)."""
    m, n = x.m, x.n
    ii = np.arange(0, n, dtype=np.float64)
    Si = np.sin(mm * PI * (ii - z) / L)
    Sj = np.sin(mn * PI * (ii - z) / L)
    if DEVICE and mc.ACC_DT == ti.f64:          # f32 slots: the host copy with f64 sums instead
        OUT = _out_buf()
        k_modecorr(m.U, _cur(m), m.M, Si, Sj, OUT)
        o = OUT.to_numpy()[:RSLOT].astype(np.float64)
        return float(o[:, 0].sum()), float(o[:, 1].sum()), float(o[:, 2].sum())
    U = x.U()
    A = Si[1:n - 1, None] * Sj[None, 1:n - 1]
    B = U[1:n - 1, 1:n - 1].astype(np.float64)
    Mk = m.mask_np[1:n - 1, 1:n - 1] != 0
    return (float(np.sum(A * B, where=Mk)), float(np.sum(A * A, where=Mk)), float(np.sum(B * B, where=Mk)))


def _rows_mode(x):
    m, p, n = x.m, x.p, x.n
    out = []
    sq_dir = (not x.tri and p.shape in ('square', 'array') and (p.bc == 'dirichlet' if m.sq_full() else not x.edge_rigid))
    # 10/10/2026: an absorbing boundary has no standing modes (the seeded sin.sin leaves through the open edge)
    ab_txt = (f'an absorbing boundary ({x.abs_text()}): the domain has no standing modes, so sin·sin is not an '
              'eigenmode here (it leaves through the open edge)') if x.port_abs else ''
    if x.port_abs:
        sq_dir = False
    mm, mn = m.mode_mn()
    mtag = f'({mm},{mn})'
    z, L = m.mode_box()
    sol = x.solids and p.src == 'mode'
    if p.src == 'mode' and sq_dir and p.medium == 'uniform' and x.live and not sol:
        num, da, db = mode_sums(x, mm, mn, z, L)
        corr = abs(num / math.sqrt(da * db)) if (da > 0 and db > 0) else math.nan
        fin = math.isfinite(corr)
        out.append(_row('mode', f'the {mtag} eigenmode keeps its SHAPE (only its amplitude oscillates)',
                        st(abs(corr - 1), 2e-3) if fin else 'n/a',
                        (f'|correlation with sin·sin| = {fmt(corr, 9)} after {x.step} steps · box L = {L} cells'
                         + ('' if m.sq_full() else ' (masked square)')) if fin else '', 'Dirichlet',
                        f'corr {fmt(corr, 9)}' if fin else '', corr if fin else None))
    else:
        detail = ('sin·sin is the eigenmode of a square on a SQUARE lattice; the triangular Laplacian has different '
                  'eigenfunctions, so this is not a test here' if x.tri
                  else 'the sin·sin mode is the SQUARE’s eigenmode — it is not one for this domain, so seeding it '
                       'here is not a test of anything' if p.shape not in ('square', 'array')
                  else ('the masked square has a RIGID edge: its eigenmodes are cos·cos, not sin·sin, so seeding sin·sin '
                        'is not a test here') if (x.edge_rigid and not m.sq_full())
                  else f'{_SOL}: sin·sin is not an eigenmode of a domain with objects inside it' if sol
                  else ab_txt if ab_txt
                  else 'select the "drum mode (m,n)" source on a square with a fixed edge')
        out.append(_row('mode', f'the {mtag} eigenmode keeps its shape', 'n/a', detail, 'Dirichlet'))

    P = x.hist.get('mode') or []
    if p.src == 'mode' and sq_dir and p.medium == 'uniform' and p.damp < 1e-12 and len(P) >= 5 and not sol:
        kx, ky = mm * PI / L, mn * PI / L
        kk = jhypot(kx, ky)
        co = p.Co
        pk = 0.0
        for v in P:
            pk = _jmax(pk, abs(v))
        cs = []
        for t in range(1, len(P) - 1):
            if abs(P[t]) > 0.5 * pk:
                cs.append((P[t + 1] + P[t - 1]) / (2 * P[t]))
        cs.sort()
        c_meas = cs[len(cs) >> 1] if cs else math.nan
        c_pred = 1 + 0.5 * co * co * stencil_symbol(m, kx, ky)
        w_meas = math.acos(max(-1, min(1, c_meas))) / co if math.isfinite(c_meas) else math.nan
        w_pred = math.acos(max(-1, min(1, c_pred))) / co
        ppw = TAU / kk
        err_meas, err_pred = w_meas / kk - 1, w_pred / kk - 1
        fin = math.isfinite(c_meas)
        tol = x.tol(1e-9, 'disp')
        d = abs(c_meas - c_pred)
        port = '' if x.fp64 else f'f32 build: tolerance {fe(tol)} (the web app\'s 1e-9 is f64)'
        out.append(_row('disp', f'dispersion, MEASURED: the {mtag} mode’s frequency at {fmt(ppw, 1)} points per wavelength',
                        st(d, tol) if fin else 'n/a',
                        (f'cos(ωΔt) from the field’s own recurrence = {fmt(c_meas, 12)} vs the dispersion relation '
                         f'{fmt(c_pred, 12)} (Δ {fe(d)}) · phase speed error {al.js_to_fixed(100 * err_meas, 4)}% measured, '
                         f'{al.js_to_fixed(100 * err_pred, 4)}% predicted · direction '
                         f'{fmt(math.atan2(ky, kx) * 180 / PI, 1)}°, Co/limit {fmt(co / x.co_limit(), 3)} — raise m,n to '
                         'lower the PPW and watch the error grow as 1/PPW²; raise Co toward the limit and watch it SHRINK')
                        if fin else 'run a few steps (the recurrence needs three consecutive samples at an antinode)',
                        'dispersion', f'Δ {fe(d)}' if fin else '', d if fin else None, port=port if fin else ''))
    else:
        detail = ('select the drum mode (m,n) source: an exact eigenmode lets the field’s own history measure ω with '
                  'no fitting — raise m,n to lower the points per wavelength' if p.src != 'mode'
                  else ('sin·sin is not an eigenmode of the triangular Laplacian' if x.tri
                        else ab_txt if ab_txt
                        else 'needs the square domain with a fixed edge') if not sq_dir
                  else f'{_SOL}: sin·sin is not an eigenmode of a domain with objects inside it' if sol
                  else 'damping changes the recurrence — set damping to 0' if p.damp >= 1e-12
                  else 'needs a uniform medium' if p.medium != 'uniform' else 'run a few steps')
        out.append(_row('disp', 'dispersion, MEASURED: an exact mode’s frequency vs the dispersion relation', 'n/a',
                        detail, 'dispersion'))
    return out


def _row_linear(x):
    m, p = x.m, x.p
    driven = p.src in ('cont', 'slit', 'beam') or (p.src == 'vtx' and p.vtx_drive)
    K = m.k_max()
    harm = list(m.harm)
    in_band = {int(h[0]) for h in harm}
    per = 1 / max(p.freq, 1e-9)
    want = math.ceil(per * 8)
    P = x.hist.get('pA') or []
    fixes = ()
    can_settle = p.bc == 'mur' or p.damp > 1e-6
    if x.port_abs:
        can_settle = True                       # 10/10/2026: the port's absorbing boundaries let a drive settle
    elif x.fix('edge-kind'):
        e = x.edge()
        cs = e == 'mur' or e == 'mixed-mur' or p.damp > 1e-6
        if cs != can_settle and driven:
            fixes = ('edge-kind',)
        can_settle = cs
    settled = can_settle and x.step * p.Co > 4 * m.inner_R()
    res = None
    if driven and settled and x.live and len(P) > want + 40:
        Pa = np.asarray(P, dtype=np.float64)
        whole = math.floor((len(P) - 20) / per) * per
        half = max(per * 2, math.floor(whole / 2 / per) * per)

        def ratio(a, b):
            t = np.arange(a, b, dtype=np.float64)
            seg = Pa[a:b]
            iM = oM = 0.0
            for k in range(1, K + 7):
                ph = TAU * p.freq * k * t
                if t.size:
                    sr, si = float(np.sum(seg * np.sin(ph))), float(np.sum(seg * np.cos(ph)))
                    v = 2 * jhypot(sr, si) / t.size
                else:
                    v = 0.0
                if k in in_band:
                    iM = _jmax(iM, v)
                else:
                    oM = _jmax(oM, v)
            return {'iM': iM, 'oM': oM, 'rel': oM / iM if iM > 1e-12 else math.nan}
        end = len(P)
        mid = end - math.floor(half)
        start = max(0, end - 2 * math.floor(half))
        w1, w2 = ratio(start, mid), ratio(mid, end)
        res = {'rel': w2['rel'], 'prev': w1['rel'],
               'falling': math.isfinite(w1['rel']) and math.isfinite(w2['rel']) and w2['rel'] < w1['rel'] * 0.98}
    if res and math.isfinite(res['rel']):
        badge = 'PASS' if (res['rel'] < 2e-3 or res['falling']) else 'FAIL'
        detail = (f'out-of-band / in-band at the probe = {fe(res["rel"])}, down from {fe(res["prev"])} over the '
                  'previous window — ' + ('FALLING, so it is the decaying switch-on transient, not mixing'
                                          if res['falling'] else 'not falling')
                  + f' · {p.wave} drive injects {len(harm)} harmonic(s) ({", ".join(s_(h[0]) for h in harm)})')
    else:
        badge = 'n/a'
        detail = ('needs a driven source (continuous, two-slit, or driven vertices)' if not driven
                  else ('an undamped closed cavity never settles — its own modes ring for ever at frequencies '
                        'unrelated to the drive, which is real out-of-band content, not mixing. Use the absorbing '
                        'edge or add damping.') if not can_settle
                  else (f'waiting for the switch-on transient to leave (step {x.step}, needs '
                        f'~{math.ceil(4 * m.inner_R() / max(p.Co, 1e-9))})') if not settled
                  else f'run to ~{want} steps so the lock-in spans several drive periods')
    port = ''
    if fixes and not can_settle:
        port = x.mur_port('Mur is selected, but it acts on the array ring only; this domain\'s wall is clamped (FIX edge-kind)')
    return _row('linear', 'a LINEAR scheme cannot create a harmonic that was not injected', badge, detail, 'linearity',
                f'out/in {fe(res["rel"])}' if (res and math.isfinite(res['rel'])) else '',
                res['rel'] if (res and math.isfinite(res['rel'])) else None, port=port, fixes=fixes)


def _row_refraction(x):
    p = x.p
    if p.medium == 'uniform':
        return _row('refraction', 'refraction: a slower medium bends the front', 'n/a', 'select a non-uniform medium',
                    'media')
    fixes = ()
    name = 'refraction: the front slows in the slab (c/2 there)'
    detail = ("per-cell Co is halved in the slow region — Berglund's two-Courant-number trick; a quantitative Snell "
              'check is not yet wired')
    if p.medium == 'lens' and x.fix('lens-text'):
        name = 'refraction: the front slows in the lens (0.55 c there)'
        detail = ("per-cell Co is 0.55 of the reference in the lens — Berglund's two-Courant-number trick; a "
                  'quantitative Snell check is not yet wired')
        fixes = ('lens-text',)
    return _row('refraction', name, 'visual', detail, 'media', dsg=True, fixes=fixes)


# ---- the ledger -------------------------------------------------------------------------------
def build_ledger(m, compat=False, profile=None):
    """The web app's buildLedger() for Membrane m: a list of Row, in the web app's order
    (CFL, 8-fold, [tri: 6-fold, conformance], mirror, energy, reciprocity, boundary, anisotropy,
    wake, staircase, edge alignment x2, light cone, drum spectrum, eigenmode, dispersion,
    linearity, refraction). compat=True turns the FIXES off (the web app's exact logic).
    profile: an optional dict that receives per-row build times in ms."""
    x = _Ctx(m, compat)
    rows = []
    steps = [_row_cfl, _row_sym8]
    if x.tri:
        steps += [_row_sym6, _row_conform]
    steps += [_row_mirror, _row_energy]
    if x.port_abs:
        steps += [_row_eopen]                   # 10/10/2026: the port's absorbing boundaries only
    steps += [_row_recip, _row_bc, _row_aniso, _row_wake, _row_stair, _rows_align,
              _row_cone, _row_spectra, _rows_mode, _row_linear, _row_refraction]
    for f in steps:
        t0 = time.perf_counter()
        r = f(x)
        if profile is not None:
            profile[f.__name__] = profile.get(f.__name__, 0.0) + (time.perf_counter() - t0) * 1e3
        rows.extend(r if isinstance(r, list) else [r])
    if x.hist_gap:
        rows = [_gap_row(r) if r.key in HIST_ROWS else r for r in rows]
    return rows


# The rows graded on the recorder's per-step history (m.hist): which series each reads.
HIST_ROWS = {'energy': 'E', 'bc': 'pA', 'wake': 'pA', 'disp': 'mode', 'linear': 'pA', 'eopen': 'E'}
_GAP = ('the recorder was off (m.record = False) for some steps of this run, so the per-step history has a gap '
        'and this row is not graded -- reset (or reseed) with the recorder on')


def _gap_row(r):
    """A history row when the history is incomplete: a graded row becomes n/a with the reason; an n/a
    row keeps its own reason (its premise) and gets the recorder remark."""
    port = _GAP if not r.port else r.port + ' · ' + _GAP
    if r.status == NA:
        return r._replace(port=port)
    return r._replace(status=NA, badge='n/a', value='', num=None, note=_GAP, port=port)


def summary(rows):
    """{status: count}."""
    out = {PASS: 0, FAIL: 0, NA: 0, STATED: 0}
    for r in rows:
        out[r.status] += 1
    return out


def format_rows(rows, width=110, notes=True):
    """Plain-text rendering (CLI, logs)."""
    lines = []
    for r in rows:
        lines.append(f'[{r.badge:>8}] {r.name}' + (f'   = {r.value}' if r.value else ''))
        if notes:
            for piece in textwrap.wrap(r.note, width - 12) or ['']:
                lines.append(' ' * 12 + piece)
            if r.port:
                for piece in textwrap.wrap('port: ' + r.port, width - 12):
                    lines.append(' ' * 12 + piece)
            if r.fixes:
                lines.append(' ' * 12 + 'fixed (vs the web app): ' + ', '.join(r.fixes))
    return '\n'.join(lines)


# ---- GGUI panel ---------------------------------------------------------------------------------
# GGUI draws with ImGui's default font (Latin-1 only): anything else prints as '?'.
_GLYPH = {'—': '--', '–': '-', '→': '->', '≤': '<=', '≥': '>=', '≠': '!=', '√': 'sqrt', '∝': '~',
          '∩': ' cap ', '∞': 'inf', 'Σ': 'sum ', 'Δ': 'd', 'ω': 'w', 'π': 'pi', 'λ': 'lambda', 'σ': 'sigma',
          '−': '-', '’': "'", '‘': "'", '“': '"', '”': '"', '…': '...', '₀': '0', '₁': '1', '₂': '2',
          '⁻': '^-', '✓': 'OK', '✗': 'X', '≅': '~=', 'α': 'alpha'}
COL = {PASS: (0.35, 0.90, 0.45), FAIL: (0.98, 0.35, 0.35), NA: (0.55, 0.55, 0.60), STATED: (0.45, 0.65, 1.00),
       'head': (0.72, 0.62, 1.00), 'dim': (0.62, 0.62, 0.68), 'text': (0.86, 0.86, 0.88)}
TAG = {PASS: 'PASS', FAIL: 'FAIL', NA: 'n/a ', STATED: 'info'}


def gui_text(s):
    s = s.replace('⁻²', '^-2')
    return ''.join(_GLYPH.get(ch, ch if ord(ch) < 256 else '?') for ch in s)


class LedgerPanel:
    """Per-viewer ledger state: the cached rows (rebuilt only on refresh), the display mode, the
    auto-refresh flag and interval.

    Measured 30/09/2026 (CUDA f64, RTX 5070 Ti; ScratchCLAUDE/LedgerCLAUDE/perf_MIRROR.py): a WARM
    rebuild (the device reductions, no field copy) costs 1-6 ms at every N from 161 to 8193; the first
    build after a reset or a geometry change is COLD (boundary cells, mask symmetry checks, the
    edge-alignment rows): <= 131 ms at N = 2049, 0.2-0.5 s at 4097, 1.0-1.9 s at 8193. So auto-refresh
    (default every 0.5 s) is on by default up to N = 2049 and off above it, where the ledger is built on
    demand ('refresh ledger') -- a cold build would stall the frame after every reset. The interval has
    a floor of 20x the last WARM build time (<= ~5% of the frame time whatever the grid size); a cold
    build does not raise the floor."""

    AUTO_MAX_N = 2049

    def __init__(self, N=0):
        self.rows = None
        self.lines = None
        self.compact = True
        self.auto = N <= self.AUTO_MAX_N
        self.auto_user = False                 # the user set auto-refresh: stop following N
        self.interval = 0.5
        self.compat = False
        self.wrap = 70
        self.t_built = -1e9
        self.ms = 0.0
        self.ms_warm = 0.0
        self.cold = False
        self.key = None
        self.step = -1
        self.error = ''

    def _key(self, m):
        p = m.p
        return (id(m), m.N, tuple(sorted(p.__dict__.items())), id(m.mask_np), m.n_solid,
                None if m.scene is None else m.scene.version)

    def stale(self, m):
        return self.rows is None or self._key(m) != self.key or self.step != m.step_n

    def refresh(self, m):
        hit = _GEO.get(m)
        self.cold = hit is None or hit[0] != _geo_key(m)
        t0 = time.perf_counter()
        try:
            self.rows = build_ledger(m, compat=self.compat)
            self.error = ''
        except Exception as e:                  # a ledger fault must never take the viewer down
            self.error = f'ledger error: {type(e).__name__}: {e}'
            self.rows = self.rows or []
        self.ms = (time.perf_counter() - t0) * 1e3
        if not self.cold:
            self.ms_warm = self.ms
        self.t_built = time.perf_counter()
        self.key = self._key(m)
        self.step = m.step_n
        self.lines = None

    def due(self, m):
        if not self.auto or not self.stale(m):
            return False
        return time.perf_counter() - self.t_built >= max(self.interval, 20 * self.ms_warm / 1e3)

    def render_lines(self):
        if self.lines is not None:
            return self.lines
        out = []
        for r in self.rows or []:
            head = f'{TAG[r.status]} {gui_text(r.name)}'
            if self.compact:
                if r.value:
                    head += '  = ' + gui_text(r.value)
                out.append((head if len(head) <= self.wrap + 20 else head[:self.wrap + 17] + '...', COL[r.status]))
            else:
                out += [(piece, COL[r.status]) for piece in textwrap.wrap(head, self.wrap + 20)]
                if r.note:
                    out += [('    ' + piece, COL['text']) for piece in textwrap.wrap(gui_text(r.note), self.wrap)]
                if r.port:
                    out += [('    ' + piece, COL['dim']) for piece in textwrap.wrap('port: ' + gui_text(r.port), self.wrap)]
                if r.fixes:
                    out.append(('    fixed vs the web app: ' + ', '.join(r.fixes), COL['dim']))
        self.lines = out
        return out


def gui_section(w, viewer):
    """Draw the ledger inside an open GGUI sub_window `w`. viewer has .m (Membrane) and .reset().
    Rows are cached: they are rebuilt on 'refresh', or by auto-refresh when the run has moved on
    and the interval (floor: 20x the last warm build time) has passed -- never every frame. Above
    N = LedgerPanel.AUTO_MAX_N it is built on demand only (unless the user turns auto-refresh on)."""
    st_ = getattr(viewer, 'ledger_panel', None)
    m = viewer.m
    if st_ is None:
        st_ = LedgerPanel(m.N)
        viewer.ledger_panel = st_
    if not st_.auto_user:
        st_.auto = m.N <= st_.AUTO_MAX_N        # the default follows the grid size (a new N, a new m)
    if st_.due(m):
        st_.refresh(m)
    s = summary(st_.rows or [])
    w.text(f'physics ledger   {s[PASS]} pass  {s[FAIL]} fail  {s[NA]} n/a  {s[STATED]} info', color=COL['head'])
    if st_.rows is None:
        w.text(f'not built yet: N = {m.N} > {st_.AUTO_MAX_N}, press "refresh ledger" (the first build '
               'takes ~1-2 s at N = 8193)', color=COL['dim'])
    else:
        age = time.perf_counter() - st_.t_built
        w.text(f'built at step {st_.step} in {st_.ms:.0f} ms{" (cold)" if st_.cold else ""}, {age:.1f} s ago'
               + ('  (stale)' if st_.stale(m) else ''), color=COL['dim'])
    if st_.error:
        w.text(gui_text(st_.error), color=COL[FAIL])
    if w.button('refresh ledger'):
        st_.refresh(m)
    if hasattr(viewer, 'reset') and w.button('reset run (reseed, then rebuild)'):
        viewer.reset()                          # e.g. after the recorder was off: a complete history
        st_.refresh(viewer.m)
    a = w.checkbox('auto-refresh', st_.auto)
    if a != st_.auto:
        st_.auto, st_.auto_user = a, True
    if st_.auto:
        st_.interval = w.slider_float('interval (s)', st_.interval, 0.25, 10.0)
    if w.button('expanded view' if st_.compact else 'compact view'):
        st_.compact = not st_.compact
        st_.lines = None
    c = w.checkbox('web-app logic (no fixes)', st_.compat)
    if c != st_.compat:
        st_.compat = c
        st_.refresh(m)
    for piece, col in st_.render_lines():
        w.text(piece, color=col)


# ---- CLI: print the ledger for one configuration --------------------------------------------------
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='print the physics ledger for one configuration (headless)')
    ap.add_argument('--arch', default='cuda')
    ap.add_argument('--f32', action='store_true')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--compat', action='store_true', help='the web app\'s exact logic (no FIXES)')
    ap.add_argument('--brief', action='store_true')
    ap.add_argument('params', nargs='*', help='Params fields as key=value (e.g. N=161 src=impulse shape=n6)')
    a = ap.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    mc.init_taichi(a.arch, fp64=not a.f32, quiet=True)
    kw = {}
    for kv in a.params:
        k, v = kv.split('=', 1)
        d = getattr(mc.Params(), k)
        kw[k] = (v.lower() in ('1', 'true', 'yes')) if isinstance(d, bool) else type(d)(v)
    m = mc.Membrane(mc.Params(**kw))
    m.step(a.steps)
    prof = {}
    t0 = time.perf_counter()
    rows = build_ledger(m, compat=a.compat, profile=prof)
    print(format_rows(rows, notes=not a.brief))
    print(f'\n{summary(rows)}  built in {(time.perf_counter() - t0) * 1e3:.1f} ms  '
          + ' '.join(f'{k[1:]}={v:.1f}' for k, v in prof.items() if v > 0.5))
