"""ZRL 2D wave membrane -- Taichi port: the GRID <-> EDGE ALIGNMENT panel.

A value-for-value mirror of the web app's alignment code
(FormConstantsCLAUDE/FlavorRenderersCLAUDE/wave_membrane_MIRROR.html, the merged build of 28/09/2026):
  - section 1c, the pure part (ALIGN-BEGIN .. ALIGN-END): alMod180, alGcdI, alGcdF, alEdgeDirs0,
    alLatticeDir, alClass, alNearDir, alCount, alAngles, alBest, alSymPeriod, alCanon;
  - section 1d, the same applied to the live state: alLat, alRotDeg, alDirs, alList, alEdges;
  - alVectors / alMeasure (the period of each edge's boundary cells, MEASURED from the mask);
  - alStencilNB / alEdgeReport / alRep (where the zeroed nodes sit beyond each drawn edge);
  - the panel: alNothingLeft, alBestOf, alFlavorPick, the NOTE text (alNoteRender), the snap list
    (alRefresh), alSetRot, alStep, alOptimal, alFlavor;
  - the two ledger rows the alignment feeds (7c period test, 7c-bis straightness test).
Every function takes the core's Membrane `m` and reads m.p, m.N, m.tri(), m.mask_np, m.dom_R(),
m.shape_poly(), m.clip_region(), m.clip_convex(). Python names are snake_case mirrors of the JS.

Checked against the web app's own functions by crosscheck_alignment_MIRROR.py (Node harness).

Exactness notes (measured 29/09/2026 on this machine, Node 20.11.1 vs Python 3.12):
  - V8's Math.hypot(x, y) is sqrt((x/m)^2 + (y/m)^2) * m with m = max(|x|, |y|) (0 of 200000
    random pairs differ from that formula); Python's math.hypot differs from it in the last bit
    for 794 of 2000 pairs. js_hypot below is the V8 formula, so lengths, frames and distances
    are the web app's to the bit.
  - atan2/cos/sin/tan/acos differ from V8's fdlibm in the last bit for 1-17% of arguments
    (C runtime vs fdlibm). Every decision that uses them has a tolerance (1e-7 deg or more), and
    the cross-check found no config where that changed an outcome (see NotesCLAUDE).
  - JS toFixed rounds the EXACT binary value half-up (0.125.toFixed(2) = '0.13'); Python's
    format rounds half-even ('0.12'). js_to_fixed reproduces the JS rule with Decimal.

Verified (30/09/2026, CPU f64): full sweep 80055/80055 comparisons over 2295 configs; extended
sweep (near-aligned angles in 0.05-degree steps, clipping-threshold radii, rhombus 20..90 degrees)
82380/82380 over 2370 configs; 'unlicensed' invariants on the core's real plate 51266/51266 over
816 configs (crosscheck_alignment_MIRROR.py --unl --big). See NotesCLAUDE/alignment_MIRROR.md.

Deviations (documented in NotesCLAUDE/alignment_MIRROR.md):
  - the 'unlicensed' domain (a 2:1 rounded rectangle, added in the port) does not exist in the
    JS. Here it is aligned EXACTLY like a rectangle: straight edge directions {0, 90} + rotation;
    the rounded corners are arcs, not edges (they take part in assigning mask cells to edges, so
    arc cells do not pollute a side's measurement, and are then dropped from every list). Its NOTE
    has two extra lines (sizes; the direction rule). The scans read m.domain_mask_np, so the
    plate's lettering and screws (interior solids) are never taken for domain edges.
  - unknown shapes get no alignment model (empty lists, a one-line note), never an exception.
  - al_set_rot(..., snap=0.05) reproduces the HTML slider's 0.05-degree snap (the default, used by
    the cross-check); the GGUI panel passes snap=None because the port's rotation slider is
    continuous, so an aligned angle that is not a multiple of 0.05 (a rhombus at an arbitrary
    theta) lands exactly instead of being reported as unreachable.
"""
import math
import re
import textwrap
from decimal import Decimal, ROUND_HALF_UP, localcontext
from functools import lru_cache

import numpy as np

PI = math.pi
S3H = math.sqrt(3) / 2

# ---- JavaScript arithmetic and number formatting, reproduced exactly ------------------------


def js_round(x):
    """Math.round: halves go toward +infinity."""
    return math.floor(x + 0.5)


def js_hypot(x, y):
    """V8's Math.hypot for two arguments (verified bitwise, see the module docstring)."""
    x, y = abs(x), abs(y)
    m = x if x > y else y
    if m == 0:
        return 0.0
    if m == math.inf:
        return math.inf
    a, b = x / m, y / m
    return math.sqrt(a * a + b * b) * m


def js_hypot_np(x, y):
    """js_hypot, elementwise (same operations, same order)."""
    x, y = np.abs(x), np.abs(y)
    m = np.maximum(x, y)
    with np.errstate(invalid='ignore', divide='ignore'):
        a, b = x / m, y / m
        r = np.sqrt(a * a + b * b) * m
    return np.where(m == 0, 0.0, r)


def js_num_str(x):
    """String(x) / template-literal ${x} (ECMA-262 Number::toString): the shortest round-trip
    digits (Python's repr has the same digits), laid out by the JS rules -- plain decimals for
    1e-7 <= |x| < 1e21, else 'de+n' (Python's repr switches to exponents below 1e-4 and writes
    'e-05'; fixed 30/09/2026)."""
    if isinstance(x, bool):
        return 'true' if x else 'false'
    if isinstance(x, int):
        return str(x)
    if x != x:
        return 'NaN'
    if x in (math.inf, -math.inf):
        return 'Infinity' if x > 0 else '-Infinity'
    if x == 0:
        return '0'
    sign, dg, ex = Decimal(repr(abs(float(x)))).normalize().as_tuple()
    s = ''.join(map(str, dg))
    k = len(s)
    n = ex + k                                       # value = 0.s * 10^n
    if k <= n <= 21:
        out = s + '0' * (n - k)
    elif 0 < n <= 21:
        out = s[:n] + '.' + s[n:]
    elif -6 < n <= 0:
        out = '0.' + '0' * (-n) + s
    else:
        e = n - 1
        out = (s if k == 1 else s[0] + '.' + s[1:]) + 'e' + ('+' if e >= 0 else '-') + str(abs(e))
    return ('-' if x < 0 else '') + out


def js_to_fixed(x, d):
    """Number.prototype.toFixed: the exact binary value, rounded half-up; -0 prints as '0'."""
    if x != x:
        return 'NaN'
    if abs(x) >= 1e21:
        return js_num_str(x)
    neg = x < 0
    with localcontext() as ctx:
        ctx.prec = 200
        s = str(Decimal(abs(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
    return ('-' if neg else '') + s


def js_to_exponential(x, d):
    """Number.prototype.toExponential(d) (finite x)."""
    if x == 0:
        return '0' + ('.' + '0' * d if d else '') + 'e+0'
    neg = x < 0
    with localcontext() as ctx:
        ctx.prec = 200
        D = Decimal(abs(x))
        e = D.adjusted()
        q = D.scaleb(-e).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
        if q >= 10:
            e += 1
            q = D.scaleb(-e).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return ('-' if neg else '') + str(q) + 'e' + ('+' if e >= 0 else '-') + str(abs(e))


def fmt(x, d=6):
    """The web app's fmt(): an em dash for null / non-finite, else toFixed(d)."""
    if x is None or not math.isfinite(x):
        return '—'
    return js_to_fixed(x, d)


def js_max(*xs):
    """Math.max over numbers (an empty list gives -Infinity)."""
    return max(xs) if xs else -math.inf


# ---- 1c. the pure section (ALIGN-BEGIN .. ALIGN-END) --------------------------------------
AL_ROWS = {'sq': [0, 90], 'tri': [0, 60, 120]}
AL_DIAG = {'sq': [45, 135], 'tri': [30, 90, 150]}
AL_DPER = {'sq': math.sqrt(2), 'tri': math.sqrt(3)}
AL_LROT = {'sq': 90, 'tri': 60}                      # the lattice's own rotation period
AL_NAME = {'sq': 'square', 'tri': 'triangular'}
# the web app's edge colours (AL_COL, 0..1 RGB) -- row / diag / rational / irrational
AL_COL = {'row': (0.0, 0.620, 0.525), 'diag': (0.282, 0.373, 0.780), 'rat': (0.784, 0.659, 0.0),
          'irr': (0.886, 0.133, 0.067)}
NB_TRI = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
UNL = 'unlicensed'
_NGON = re.compile(r'^n\d+$')


def al_mod180(x):
    return math.fmod(math.fmod(x, 180) + 180, 180)


def al_gcd_i(a, b):
    a, b = abs(int(a)), abs(int(b))
    while b:
        a, b = b, a % b
    return a


def al_gcd_f(a, b):
    """gcd of two angles in degrees (tolerant)."""
    a, b = abs(a), abs(b)
    for _ in range(64):
        if not b > 1e-7:
            break
        t = math.fmod(a, b)
        a = b
        b = 0 if (t < 1e-7 or b - t < 1e-7) else t
    return a


def al_edge_dirs0(shape, rtheta_rad):
    """Edge DIRECTIONS (degrees, mod 180) of the unrotated flavor polygon, in shapePoly order.
    'unlicensed' (port only): a rectangle's two directions, 0 and 90; its arcs are not edges."""
    if shape == 'circle':
        return []
    if shape == 'square':
        return [0, 90, 0, 90]
    if shape == UNL:
        return [0, 90, 0, 90]
    if shape == 'rhomb':
        h = rtheta_rad * 90 / PI                     # edges at rot -+ theta/2
        return [al_mod180(-h), al_mod180(h), al_mod180(-h), al_mod180(h)]
    if not _NGON.match(str(shape)):
        return []                                    # unknown shape: no alignment model
    n = int(shape[1:])
    # shapePoly's edge k joins the vertices at pi/n + 2pi k/n and pi/n + 2pi(k+1)/n, so its
    # outward normal is at 360(k+1)/n and its direction 90 degrees further on
    return [al_mod180(360 * (k + 1) / n + 90) for k in range(n)]


@lru_cache(maxsize=65536)
def al_lattice_dir(phi, lat, M=24):
    """The primitive lattice direction (a, b) along angle phi, if phi IS one (else None)."""
    M = M or 24
    p = al_mod180(phi) * PI / 180
    c, s = math.cos(p), math.sin(p)
    if abs(s) < 1e-12:
        return {'a': 1, 'b': 0, 'd': 1}
    for b in range(1, M + 1):
        aReal = b * (math.sqrt(3) / 2 * c / s - 0.5) if lat == 'tri' else b * c / s
        a = js_round(aReal)
        if abs(aReal - a) > 1e-7 * b:
            continue
        g = al_gcd_i(a, b)
        A, B = a // g if a >= 0 else -((-a) // g), b // g
        d = math.sqrt(A * A + A * B + B * B) if lat == 'tri' else math.sqrt(A * A + B * B)
        return {'a': A, 'b': B, 'd': d}
    return None


@lru_cache(maxsize=65536)
def _al_class(phi, lat):
    r = al_lattice_dir(phi, lat, 24)
    if not r:
        return {'cls': 'irr'}
    cls = 'row' if r['d'] < 1 + 1e-9 else ('diag' if abs(r['d'] - AL_DPER[lat]) < 1e-9 else 'rat')
    return {'cls': cls, 'a': r['a'], 'b': r['b'], 'd': r['d']}


def al_class(phi, lat):
    return dict(_al_class(phi, lat))


@lru_cache(maxsize=65536)
def _al_near_dir(phi, lat, length):
    sp = math.sqrt(3) / 2 if lat == 'tri' else 1
    best = None
    for b in range(0, 17):
        for a in range(-16, 17):
            if not a and not b:
                continue
            if al_gcd_i(a, b) != 1:
                continue
            vx = a + b / 2 if lat == 'tri' else a
            vy = b * math.sqrt(3) / 2 if lat == 'tri' else b
            ang = math.atan2(vy, vx) * 180 / PI
            dA = abs(al_mod180(ang) - al_mod180(phi))
            dA = min(dA, 180 - dA)
            if length * math.sin(dA * PI / 180) >= sp / 2:
                continue
            d = js_hypot(vx, vy)
            if not best or d < best['d'] - 1e-9:
                best = {'a': a, 'b': b, 'd': d, 'dA': dA}
    return best


def al_near_dir(phi, lat, length):
    """For an edge that is NOT a lattice direction: the shortest-period lattice direction whose
    staircase it follows over its whole length (drift < half a row spacing), or None."""
    r = _al_near_dir(phi, lat, length)
    return dict(r) if r else None


def al_count(dirs0, rot_deg, lat):
    rows = diags = 0
    for p in dirs0:
        c = _al_class(p + rot_deg, lat)['cls']
        if c == 'row':
            rows += 1
        elif c == 'diag':
            diags += 1
    return {'rows': rows, 'diags': diags, 'n': len(dirs0)}


@lru_cache(maxsize=4096)
def _al_angles(dirs0, lat, lo, hi):
    cand = []
    for p in dirs0:
        for al in AL_ROWS[lat] + AL_DIAG[lat]:
            r = al_mod180(al - p)
            for q in (r - 180, r, r + 180):
                qq = js_round(q * 1e9) / 1e9
                if lo - 1e-9 <= qq <= hi + 1e-9:
                    cand.append(float(min(hi, max(lo, qq))))
    cand.sort()
    out = []
    for r in cand:
        if out and abs(out[-1]['rot'] - r) < 1e-7:
            continue
        c = al_count(dirs0, r, lat)
        if c['rows'] + c['diags']:
            out.append({'rot': r, 'rows': c['rows'], 'diags': c['diags'], 'n': c['n']})
    return tuple(out)


def al_angles(dirs0, lat, lo, hi):
    """Every rotation in [lo, hi] (degrees) at which >= 1 edge is a row or a diag, exactly."""
    return [dict(e) for e in _al_angles(tuple(dirs0), lat, lo, hi)]


def _al_key(e, mode):
    return (e['rows'] + e['diags'], e['rows']) if mode == 'smooth' else (e['rows'], e['diags'])


def al_best(lst, mode):
    """'row' ranks straight rows first, 'smooth' rows+diags first; the smallest angle wins a tie."""
    best = None
    for e in lst:
        if best is None:
            best = e
            continue
        if _al_key(e, mode) > _al_key(best, mode):
            best = e
    return best


def al_sym_period(shape, lat):
    """Rotations that differ by gcd(shape period, lattice period), or mirrored, are congruent.
    'unlicensed' (port only): a rectangle's period, 180."""
    if shape in ('circle', 'square'):
        return 0
    if shape in ('rhomb', UNL):
        P = 180
    elif _NGON.match(str(shape)):
        P = 360 / int(shape[1:])
    else:
        return 0
    return al_gcd_f(P, AL_LROT[lat])


def al_canon(rot, g):
    if not g:
        return 0
    r = math.fmod(math.fmod(rot, g) + g, g)
    c = min(r, g - r)
    return js_round(c * 1e6) / 1e6


# ---- geometry helpers the core may not carry ---------------------------------------------
def poly_area(P):
    s = 0.0
    for k in range(len(P)):
        a, b = P[k], P[(k + 1) % len(P)]
        s += a[0] * b[1] - b[0] * a[1]
    return abs(s) / 2


def fits_inside(m, P):
    """The web app's fitsInside: every vertex of P inside the array's clip region."""
    C = m.clip_region()
    for e in range(len(C)):
        a, b = C[e], C[(e + 1) % len(C)]
        nx, ny = -(b[1] - a[1]), (b[0] - a[0])
        for p in P:
            if nx * (p[0] - a[0]) + ny * (p[1] - a[1]) < -1e-9:
                return False
    return True


def _clip_convex(m, P, C):
    f = getattr(m, 'clip_convex', None)
    if f is not None:
        return f(P, C)
    raise AttributeError('Membrane.clip_convex missing')


def _model(m):
    """'none' (circle), 'js' (a web-app flavor), 'unl' (port-only rounded rectangle), 'unknown'."""
    s = m.p.shape
    if s == 'circle':
        return 'none'
    if s in ('square', 'rhomb') or _NGON.match(str(s)):
        return 'js'
    if s == UNL:
        return 'unl'
    return 'unknown'


def _shape_poly(m):
    try:
        P = [tuple(map(float, q)) for q in m.shape_poly()]
    except Exception:                               # a shape the core cannot outline yet
        return []
    if _model(m) == 'unl' and len(P) >= 3:          # the JS polygons are all CCW; make sure
        s = sum(P[k][0] * P[(k + 1) % len(P)][1] - P[(k + 1) % len(P)][0] * P[k][1]
                for k in range(len(P)))
        if s < 0:
            P = P[::-1]
    return P


def _mask_ji(m):
    """The DOMAIN mask as the JS stores it, [j, i]. If the core keeps solids in the mask, a
    domain-only copy (domain_mask_np) is preferred: interior solids are not domain edges."""
    M = getattr(m, 'domain_mask_np', None)
    if M is None:
        M = m.mask_np
    return np.ascontiguousarray(np.asarray(M).T != 0)


def _on_line(p, a, b):
    ex, ey = b[0] - a[0], b[1] - a[1]
    L = js_hypot(ex, ey) or 1
    return abs(ex * (p[1] - a[1]) - ey * (p[0] - a[0])) / L < 1e-6


def _unl_sides(m, P):
    """The straight sides of the 'unlicensed' outline: polygon edges along rot or rot + 90
    (to 1e-6 degree; the arcs' chords are never exactly there), with their exact direction."""
    rot = m.p.rot * 180 / PI
    out = []
    for e in range(len(P)):
        p0, p1 = P[e], P[(e + 1) % len(P)]
        if js_hypot(p1[0] - p0[0], p1[1] - p0[1]) < 1e-6:
            continue
        ang = al_mod180(math.atan2(p1[1] - p0[1], p1[0] - p0[0]) * 180 / PI - rot)
        if min(ang, 180 - ang) < 1e-6:
            out.append((p0, p1, 0))
        elif abs(ang - 90) < 1e-6:
            out.append((p0, p1, 90))
    return out


# ---- 1d. alignment applied to the LIVE state ---------------------------------------------
def al_lat(m):
    return 'tri' if m.tri() else 'sq'


def al_rot_deg(m):
    return 0 if m.p.shape == 'square' else m.p.rot * 180 / PI


def al_dirs(m):
    return [] if m.p.shape == 'circle' else al_edge_dirs0(m.p.shape, m.p.rtheta)


def al_aligned(e, cls='row'):
    return e['rows'] + e['diags'] if cls == 'smooth' else e['rows']


def al_list(m, cls='row', lat=None):
    """Every rotation in [0, 90] with >= 1 edge aligned under the chosen definition."""
    lat = lat or al_lat(m)
    d = al_dirs(m)
    if not d:
        return []
    if m.p.shape == 'square':
        c = al_count(d, 0, lat)
        L = [{'rot': 0, 'rows': c['rows'], 'diags': c['diags'], 'n': c['n']}]
    else:
        L = al_angles(d, lat, 0, 90)
    return [e for e in L if al_aligned(e, cls) > 0]


def _al_edges_all(m):
    """alEdges with the 'unlicensed' arcs kept in place (flag 'arc'); JS shapes have none."""
    mod = _model(m)
    if mod in ('none', 'unknown'):
        return []
    lat, rot, d0 = al_lat(m), al_rot_deg(m), al_dirs(m)
    P, C = _shape_poly(m), m.clip_region()
    if not P:
        return []
    Q, out = _clip_convex(m, P, C), []
    sides = _unl_sides(m, P) if mod == 'unl' else None
    for k in range(len(Q)):
        a, b = Q[k], Q[(k + 1) % len(Q)]
        a, b = (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))
        ln = js_hypot(b[0] - a[0], b[1] - a[1])
        if ln < 1e-6:
            continue
        phi, grid, arc = None, False, False
        if sides is None:
            for e in range(len(P)):
                if phi is not None:
                    break
                p0, p1 = P[e], P[(e + 1) % len(P)]
                if _on_line(a, p0, p1) and _on_line(b, p0, p1):
                    phi = d0[e] + rot
        else:
            for p0, p1, dd in sides:
                if _on_line(a, p0, p1) and _on_line(b, p0, p1):
                    phi = dd + rot
                    break
        for e in range(len(C)):
            if phi is not None:
                break
            c0, c1 = C[e], C[(e + 1) % len(C)]
            if _on_line(a, c0, c1) and _on_line(b, c0, c1):
                phi = math.atan2(c1[1] - c0[1], c1[0] - c0[0]) * 180 / PI
                grid = True
        if phi is None:
            phi = math.atan2(b[1] - a[1], b[0] - a[0]) * 180 / PI
            arc = sides is not None
        c = _al_class(phi, lat)
        out.append({'a': a, 'b': b, 'len': ln, 'phi': al_mod180(phi), 'grid': grid, 'cls': c['cls'],
                    'da': c.get('a'), 'db': c.get('b'), 'd': c.get('d'),
                    'near': al_near_dir(phi, lat, ln) if c['cls'] == 'irr' else None, 'arc': arc})
    return out


def al_edges(m):
    """The domain's ACTUAL edges (the clipped polygon, grid-edge pieces included), classed.
    Keys as in the JS: a, b, len, phi, grid, cls, da, db, d, near (+ 'arc', always False here)."""
    return [e for e in _al_edges_all(m) if not e['arc']]


# ---- the measured period (alVectors / alMeasure) ------------------------------------------
_AL_VEC = {}


def al_vectors(lat):
    """Primitive lattice vectors, shortest first (stable order on ties, as V8's sort)."""
    if lat in _AL_VEC:
        return _AL_VEC[lat]
    out = []
    for b in range(0, 25):
        for a in range(-24, 25):
            if b == 0 and a <= 0:
                continue
            if al_gcd_i(a, b) != 1:
                continue
            x = a + b / 2 if lat == 'tri' else a
            y = b * S3H if lat == 'tri' else b
            out.append({'a': a, 'b': b, 'x': x, 'y': y, 'len': js_hypot(x, y)})
    out.sort(key=lambda v: v['len'])
    _AL_VEC[lat] = out
    return out


def _cell_xy_np(m, I, J):
    c = (m.N - 1) / 2
    da, db = I - c, J - c
    if m.tri():
        return da + db * 0.5, db * S3H
    return da.astype(np.float64), db.astype(np.float64)


def _dom_mask(m):
    M = getattr(m, 'domain_mask_np', None)
    return getattr(m, 'mask_np', None) if M is None else M


def _geom_params(m):
    p = m.p
    return (p.shape, p.lattice, p.rot, p.rad, p.rtheta, m.N, p.stencil)


def _geom_key(m):
    return _geom_params(m) + (id(_dom_mask(m)),)


_CACHE = {}


def _cached(name, m, fn):
    """Mask scans are cached per (shape, lattice, rot, rad, rtheta, N, stencil) AND mask content.
    The core builds a NEW domain array on every reseed and on every set_co (build_domain), and
    never edits one in place (update_solids edits a copy), so: the same object is a hit at once;
    a different object with the same geometry is compared by content (np.array_equal: ~8 ms at
    N = 4097, against a ~1-4 s rescan -- a Co slider drag no longer rescans, 30/09/2026); a
    different geometry always rescans. The entry keeps a reference to the mask it was computed
    from (replaced by the newest equal one, so at most one old array stays alive)."""
    k = (name,) + _geom_params(m)
    M = _dom_mask(m)
    hit = _CACHE.get(name)
    if hit is not None and hit[0] == k:
        if hit[1] is M:
            return hit[2]
        old = hit[1]
        if M is not None and old is not None and old.shape == M.shape and np.array_equal(old, M):
            for nm, h in list(_CACHE.items()):       # one comparison serves every scan of that mask
                if h[1] is old and h[0][1:] == k[1:]:
                    _CACHE[nm] = (h[0], M, h[2])
            return hit[2]
    v = fn(m)
    _CACHE[name] = (k, M, v)
    return v


def _al_measure(m):
    E = _al_edges_all(m)
    n, lat = m.N, al_lat(m)
    if not E:
        return {'edges': []}
    inside = _mask_ji(m)
    nb = NB_TRI if lat == 'tri' else [(1, 0), (-1, 0), (0, 1), (0, -1)]
    # boundary = inside the mask with a lattice neighbour outside it (or off the array)
    pad = np.zeros((n + 2, n + 2), dtype=bool)
    pad[1:-1, 1:-1] = inside
    outnb = np.zeros((n, n), dtype=bool)
    for di, dj in nb:
        outnb |= ~pad[1 + dj:n + 1 + dj, 1 + di:n + 1 + di]
    isB = inside & outnb
    Jb, Ib = np.nonzero(isB)                         # k = j*n + i order, as the JS loop
    mm = len(E)
    F = []
    for q, e in enumerate(E):
        ux, uy = (e['b'][0] - e['a'][0]) / e['len'], (e['b'][1] - e['a'][1]) / e['len']

        def ang(p, o, dx, dy):
            vx, vy = p[0] - o[0], p[1] - o[1]
            L = js_hypot(vx, vy) or 1
            return math.acos(max(-1, min(1, (vx * dx + vy * dy) / L)))
        aA = ang(E[(q + mm - 1) % mm]['a'], e['a'], ux, uy)
        aB = ang(E[(q + 1) % mm]['b'], e['b'], -ux, -uy)
        mg = lambda al: max(3, 2.5 / math.tan(max(al, 1e-3) / 2))
        F.append({'ux': ux, 'uy': uy, 'lo': mg(aA), 'hi': e['len'] - mg(aB)})
    # assign each boundary cell to its nearest edge (first minimum wins, as the strict '<')
    px, py = _cell_xy_np(m, Ib, Jb)
    D = np.empty((len(Ib), mm))
    Sv = np.empty((len(Ib), mm))
    for q, e in enumerate(E):
        f = F[q]
        rx, ry = px - e['a'][0], py - e['a'][1]
        s = rx * f['ux'] + ry * f['uy']
        t = -rx * f['uy'] + ry * f['ux']
        d = np.where(s < 0, js_hypot_np(rx, ry),
                     np.where(s > e['len'], js_hypot_np(px - e['b'][0], py - e['b'][1]), np.abs(t)))
        D[:, q], Sv[:, q] = d, s
    if len(Ib):
        best = np.argmin(D, axis=1)
        bd = D[np.arange(len(Ib)), best]
        bs = Sv[np.arange(len(Ib)), best]
    else:
        best = bd = bs = np.zeros(0)
    vecs, out = al_vectors(lat), []
    for q, e in enumerate(E):
        if e['arc']:
            continue
        f = F[q]
        win = f['hi'] - f['lo']
        sel = (best == q) & (bd < 2)
        ci, cj, cs = Ib[sel], Jb[sel], bs[sel]
        w = (cs >= f['lo']) & (cs <= f['hi'])
        ci, cj, cs = ci[w], cj[w], cs[w]
        meas = None
        if win > 0:
            for v in vecs:
                if v['len'] > win / 2:
                    break                            # needs two periods inside the window
                pr = v['x'] * f['ux'] + v['y'] * f['uy']
                sg = -1 if pr < 0 else 1
                da, db, ps = sg * v['a'], sg * v['b'], abs(pr)
                k = ~(cs + ps > f['hi'])
                ii, jj = ci[k] + da, cj[k] + db
                inb = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < n)
                if not inb.all():
                    continue
                if not isB[jj, ii].all():
                    continue
                tested = int(k.sum())
                if tested >= 4:
                    # the JS reports the UNSIGNED vector (v.a, v.b), not the shift it applied
                    meas = {'a': v['a'], 'b': v['b'], 'd': v['len'], 'tested': tested}
                    break
        cl = e['cls']
        pred = (1 if cl == 'row' else AL_DPER[lat] if cl == 'diag' else e['d'] if cl == 'rat'
                else (e['near']['d'] if e['near'] else None))
        r = {k2: v2 for k2, v2 in e.items() if k2 != 'arc'}
        r.update({'meas': meas, 'pred': pred, 'win': win, 'nCells': int(len(cs))})
        out.append(r)
    return {'edges': out}


def al_measure(m):
    """Per edge, the SHORTEST lattice translation mapping its boundary cells onto boundary
    cells (exact, integer shifts) against the period its class predicts. Cached per mask."""
    return _cached('measure', m, _al_measure)


# ---- where the zeroed nodes sit (alStencilNB / alEdgeReport / alRep) ----------------------
def al_diag_name(lat):
    return 'zigzag' if lat == 'tri' else 'diagonal'


def al_stencil_nb(m):
    if m.tri():
        return list(NB_TRI)
    if m.p.stencil == 9:
        return [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    return [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _al_edge_report(m):
    lat, rep = m.p.lattice, {'edges': []}
    mod = _model(m)
    if mod in ('none', 'unknown'):
        return rep
    P0 = _shape_poly(m)
    if not P0:
        return rep
    C = m.clip_region()
    P = _clip_convex(m, P0, C)
    mm = len(P)
    sides = _unl_sides(m, P0) if mod == 'unl' else None

    def on_clip(x, y):
        for e in range(len(C)):
            a, b = C[e], C[(e + 1) % len(C)]
            L = js_hypot(b[0] - a[0], b[1] - a[1])
            if abs((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])) / L < 1e-6:
                return True
        return False
    sq_full = m.sq_full()
    E = []
    for k in range(mm):
        a, b = P[k], P[(k + 1) % mm]
        a, b = (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))
        dx, dy = b[0] - a[0], b[1] - a[1]
        ln = js_hypot(dx, dy)
        if ln < 1e-6:
            continue
        nx, ny = dy / ln, -dx / ln                   # outward: every polygon here is CCW
        phi = math.atan2(dy, dx) * 180 / PI
        cl = _al_class(phi, lat)
        arc = False
        if sides is not None:                        # 'unlicensed': a side, the array edge, or an arc
            arc = not (any(_on_line(a, p0, p1) and _on_line(b, p0, p1) for p0, p1, _ in sides) or
                       any(_on_line(a, C[e], C[(e + 1) % len(C)]) and _on_line(b, C[e], C[(e + 1) % len(C)])
                           for e in range(len(C))))
        E.append({'a': a, 'b': b, 'len': ln, 'nx': nx, 'ny': ny, 'off': nx * a[0] + ny * a[1],
                  'normal': math.fmod(math.atan2(ny, nx) * 180 / PI + 360, 360), 'cls': cl['cls'],
                  'la': cl.get('a'), 'lb': cl.get('b'), 'd': cl.get('d'),
                  'near': al_near_dir(phi, lat, ln) if cl['cls'] == 'irr' else None,
                  'arr': sq_full or on_clip((a[0] + b[0]) / 2, (a[1] + b[1]) / 2),
                  'cnt': 0, 'lo': math.inf, 'hi': -math.inf, 'sum': 0.0, 'arc': arc})
    if not sq_full and E:
        n = m.N
        inside = _mask_ji(m)
        pad = np.zeros((n + 2, n + 2), dtype=bool)
        pad[1:-1, 1:-1] = inside
        touch = np.zeros((n, n), dtype=bool)
        for di, dj in al_stencil_nb(m):
            touch |= pad[1 + dj:n + 1 + dj, 1 + di:n + 1 + di]
        J, I = np.nonzero(touch & ~inside)            # k order
        x, y = _cell_xy_np(m, I, J)
        corner = np.zeros(len(I), dtype=bool)
        D = np.empty((len(I), len(E)))
        for e, q in enumerate(E):
            ux, uy = q['b'][0] - q['a'][0], q['b'][1] - q['a'][1]
            corner |= (js_hypot_np(x - q['a'][0], y - q['a'][1]) < 1.5) | \
                      (js_hypot_np(x - q['b'][0], y - q['b'][1]) < 1.5)
            t = ((x - q['a'][0]) * ux + (y - q['a'][1]) * uy) / (q['len'] * q['len'])
            t = np.maximum(0, np.minimum(1, t))
            D[:, e] = js_hypot_np(x - q['a'][0] - t * ux, y - q['a'][1] - t * uy)
        if len(I):
            best = np.argmin(D, axis=1)
            keep = ~corner
            for e, q in enumerate(E):
                sel = keep & (best == e)
                if not sel.any():
                    continue
                s = q['nx'] * x[sel] + q['ny'] * y[sel] - q['off']
                q['cnt'] = int(sel.sum())
                q['sum'] = float(np.cumsum(s)[-1])    # sequential, as the JS loop adds
                q['lo'], q['hi'] = float(s.min()), float(s.max())
    for q in E:
        q['mean'] = q['sum'] / q['cnt'] if q['cnt'] else math.nan
        q['spread'] = q['hi'] - q['lo'] if q['cnt'] else math.nan
        q['measStraight'] = (q['spread'] < 1e-6) if (q['cnt'] >= 4 and not q['arr']) else None
        # ONE line of zeroed nodes: a row always; the square lattice's 45-degree edge too, but
        # only with the 5-point stencil -- the 9-point one also reads the next line out
        q['predStraight'] = q['cls'] == 'row' or (lat == 'sq' and q['cls'] == 'diag' and m.p.stencil != 9)
    rep['edges'] = [{k: v for k, v in q.items() if k != 'arc'} for q in E if not q['arc']]
    return rep


def al_edge_report(m):
    """alEdgeReport / alRep: per clipped edge, the zeroed nodes' distance beyond the drawn edge
    line (cnt, lo, hi, mean, spread), measured vs predicted straightness. Cached per mask."""
    return _cached('report', m, _al_edge_report)


al_rep = al_edge_report


# ---- the panel ----------------------------------------------------------------------------
def al_nothing_left(m):
    """Every flavor edge clipped away (radius > 1): the domain is the array, nothing to align."""
    if m.p.shape == 'circle' or _model(m) == 'unknown':
        return False
    P = _shape_poly(m)
    return bool(P) and not fits_inside(m, P) and not any(not e['grid'] for e in al_edges(m))


def al_best_of(m, lat, cls='row'):
    L = al_list(m, cls, lat)
    return al_best(L, cls) if L else None


def al_flavor_pick(m, cls='row'):
    """The lattice that aligns the most edges; on a tie the CURRENT lattice stays."""
    key = lambda e: _al_key(e, cls) if e else (-1, -1)
    cur = al_lat(m)
    oth = 'tri' if cur == 'sq' else 'sq'
    bc, bo = al_best_of(m, cur, cls), al_best_of(m, oth, cls)
    kc, ko = key(bc), key(bo)
    if ko > kc:
        return {'lat': oth, 'best': bo, 'tie': False, 'other': bc}
    return {'lat': cur, 'best': bc, 'tie': ko == kc, 'other': bo}


def al_note_lines(m, cls='row', L=None, edge_col=False):
    """The panel's NOTE (alNoteRender) as [(line, colour key or None)], unicode, no HTML.
    Colour keys: 'row' / 'diag' / 'rat' / 'irr' (AL_COL), 'hi', 'accent', 'link', 'dim'."""
    p = m.p
    lat, d = al_lat(m), al_dirs(m)
    n = len(d)
    if p.shape == 'circle':
        return [('the ∞ flavor has no edges — every boundary point faces a different direction, '
                 'so no rotation and no lattice aligns it', 'link')]
    if _model(m) == 'unknown':
        return [(f"shape '{p.shape}': no alignment model — only the web app's flavors and "
                 f"'unlicensed' are classified", 'link')]
    if L is None:
        L = al_list(m, cls)
    rot = al_rot_deg(m)
    c = al_count(d, rot, lat)
    k = al_aligned(c, cls)
    P = _shape_poly(m)
    fitsA = fits_inside(m, P)
    E = al_edges(m)
    surv = len([e for e in E if not e['grid']])
    all_ = k == n and fitsA
    seen = {}
    for e in E:
        if e['grid']:
            continue
        key = js_round(e['phi'] * 1e6)
        if key not in seen:
            seen[key] = {'e': e, 'count': 0}
        seen[key]['count'] += 1
    dirs = []
    for v in seen.values():
        e, count = v['e'], v['count']
        if e['cls'] == 'row':
            what = 'row'
        elif e['cls'] == 'diag':
            what = f"diag, d = {fmt(e['d'], 3)} → silent for λ > {fmt(2 * e['d'], 2)}"
        elif e['cls'] == 'rat':
            what = (f"rational ({e['da']},{e['db']}), d = {fmt(e['d'], 3)} → silent only for "
                    f"λ > {fmt(2 * e['d'], 2)}")
        else:
            nr = e['near']
            what = 'irrational' + (f" — over its length it follows ({nr['a']},{nr['b']}), "
                                   f"d = {fmt(nr['d'], 2)}; leaks weakly at every λ" if nr else '')
        dirs.append((f"{fmt(e['phi'], 2)}° {what}" + (f' ×{count}' if count > 1 else ''), e['cls']))
    g = al_sym_period(p.shape, lat)
    whole = fitsA
    best = al_best(L, cls) if L else None
    pick = al_flavor_pick(m, cls)
    lam = p.Co / max(p.freq, 1e-9)
    out = []
    if p.shape == 'square':
        out.append(('the exact square IGNORES rotation (fixed at 0°) — '
                    '"square, rotatable" is the n = 4 polygon that turns', 'dim'))
    out.append((f'rot {fmt(rot, 2)}° on the {AL_NAME[lat]} lattice: {k}/{n} edges aligned '
                f"({c['rows']} row, {c['diags']} diag)" + (' — ALL' if all_ else ''),
                'hi' if all_ else 'accent' if k else 'link'))
    if dirs:
        out.extend(dirs)
    else:
        out.append(('', None))                       # the JS joins an empty list, then <br>
    if not fitsA:
        out.append((f'the array clips this shape: {surv} flavor edge piece(s) survive, the rest of the '
                    'wall is the array edge — the counts above are for the WHOLE polygon' if surv else
                    'every flavor edge is clipped away at this radius — the domain is the array '
                    'itself, so there is nothing to align', 'link'))
    if p.shape == 'n5':
        out.append(('a pentagon gets at most ONE edge on a direction of either lattice (tan 36° and '
                    'tan 72° are in neither Q nor √3·Q); on ANY lattice at most 3 of 5', 'dim'))
    if p.shape == UNL:
        hw, hh, rc = m.plate_dims()
        R = m.dom_R() or 1
        out.append((f'unlicensed (port only): a rounded 2:1 rectangle, aligned as a rectangle — its 4 '
                    f'straight sides (long {fmt(2 * (hw - rc) / R, 3)} R, short {fmt(2 * (hh - rc) / R, 3)} R) '
                    f'are the edges; the corner arcs (radius {fmt(rc / R, 3)} R) are not edges', 'dim'))
    RP = al_rep(m)
    EE = [e for e in RP['edges'] if not e['arr']]
    if EE:
        parts = []
        for i, e in enumerate(EE):
            if not e['cnt']:
                v = '(on the array edge)'
            elif e['spread'] < 1e-6:
                v = '+' + js_to_fixed(e['mean'], 3)
            else:
                v = '+' + js_to_fixed(e['lo'], 2) + '…+' + js_to_fixed(e['hi'], 2)
            parts.append(f'e{i + 1} {v}')
        out.append(('zeroed nodes beyond each drawn edge (measured): ' + ' · '.join(parts), None))
        per = lambda e: (e['near']['d'] if e['near'] else math.inf) if e['cls'] == 'irr' else e['d']
        silent = len([e for e in EE if lam > 2 * per(e)])
        noPer = len([e for e in EE if not math.isfinite(per(e))])
        out.append((f'{silent}/{len(EE)} edges have every side beam evanescent at the drive λ '
                    f'(λ > 2d)' + (f'; {noPer} follow no lattice period over their length and leak '
                                         f'weakly at every λ' if noPer else ''), None))
        worst = js_max(0, *[x for x in map(per, EE) if math.isfinite(x)])
        if worst > 0 and p.src in ('pulse', 'vtx'):
            x = math.pow(PI * p.sigma / worst, 2)
            frac = (1 + x) * math.exp(-x)
            fs = js_to_exponential(frac, 1) if frac < 1e-4 else js_to_fixed(100 * frac, 2) + '%'
            out.append((f'pulse σ = {js_to_fixed(p.sigma, 1)}: {fs} of its energy can open the side '
                        f'beams of the worst PERIODIC edge (d = {js_to_fixed(worst, 2)})', None))
    if L:
        out.append((f"{len(L)} aligned angle{'s' if len(L) > 1 else ''} in 0–90°: "
                    + ', '.join(fmt(e['rot'], 2) for e in L), None))
    else:
        out.append(('no rotation aligns an edge under this definition', None))
    if g and whole and len(L) > 1:
        out.append((f'rotations {fmt(g, 2)}° apart, or mirrored, give the SAME mask '
                    '(lattice turn × shape turn)', 'dim'))
    if best:
        out.append((f"optimal angle: {fmt(best['rot'], 2)}° — {al_aligned(best, cls)}/{n}"
                    + (f" ({best['rows']} row + {best['diags']} diag)" if cls == 'smooth' else ''), 'hi'))
    if pick['best']:
        pb = pick['best']
        s = (f"optimal grid flavor: {AL_NAME[pick['lat']]} @ {fmt(pb['rot'], 2)}° "
             f"({al_aligned(pb, cls)}/{n})")
        if pick['tie']:
            s += (f' — tie: the other lattice also reaches {al_aligned(pb, cls)}/{n}, '
                  'so the current one stays')
        elif pick['other']:
            s += (f" vs {al_aligned(pick['other'], cls)}/{n} on the "
                  f"{AL_NAME['tri' if pick['lat'] == 'sq' else 'sq']}")
        out.append((s, 'hi'))
    if _NGON.match(str(p.shape)):
        out.append((f'rule: edge k lies along lattice direction α iff rot ≡ α − 90° '
                    f'− 360°k/{n} (mod 180°) · rows α = '
                    f"{', '.join(map(str, AL_ROWS[lat]))}°, diags {', '.join(map(str, AL_DIAG[lat]))}°",
                    'dim'))
    if p.shape == UNL:
        out.append((f'rule: the long sides lie along lattice direction α iff rot ≡ α (mod 180°), the short '
                    f"sides iff rot ≡ α − 90° · rows α = {', '.join(map(str, AL_ROWS[lat]))}°, "
                    f"diags {', '.join(map(str, AL_DIAG[lat]))}°", 'dim'))
    out.append((f'drive λ = Co/f = {fmt(lam, 1)} cells', 'dim'))
    if edge_col:
        out.append(('row · diag · rational · irrational', None))
    return out


def al_note(m, cls='row', L=None, edge_col=False):
    """alNoteRender's text: lines joined by newlines (HTML entities as unicode)."""
    return '\n'.join(t for t, _ in al_note_lines(m, cls, L, edge_col))


def al_snap_options(m, cls='row'):
    """alRefresh's snap list: {'header', 'entries': [{rot, rows, diags, n, label, congruent_to}],
    'selected': index into entries of the current rotation, or -1}."""
    L = al_list(m, cls)
    cur, lat = al_rot_deg(m), al_lat(m)
    g = al_sym_period(m.p.shape, lat)
    P = _shape_poly(m)
    whole = fits_inside(m, P) if P else False
    first, entries, sel = {}, [], -1
    for idx, e in enumerate(L):
        cl = al_canon(e['rot'], g)
        if cl not in first:
            first[cl] = e['rot']
        f0 = first[cl]
        if abs(e['rot'] - cur) < 1e-6:
            sel = idx
        k = al_aligned(e, cls)
        cong = f0 if (g and whole and abs(f0 - e['rot']) > 1e-6) else None
        lab = (f"{fmt(e['rot'], 2)}° — {e['rows']} row"
               + (f" + {e['diags']} diag" if e['diags'] else '') + f" / {e['n']}"
               + (' · ALL' if (k == e['n'] and whole) else '')
               + (f' (≅ {fmt(f0, 2)}°)' if cong is not None else ''))
        entries.append({'rot': e['rot'], 'rows': e['rows'], 'diags': e['diags'], 'n': e['n'],
                        'label': lab, 'congruent_to': cong})
    if m.p.shape == 'circle':
        head = '— the circle has no edges —'
    elif L:
        head = f"— {len(L)} aligned angle{'s' if len(L) > 1 else ''} (0–90°) —"
    else:
        head = '— no aligned angle —'
    return {'header': head, 'entries': entries, 'selected': sel}


def _slider(v, mn, mx, st):
    """The HTML range input's clamp + step snap, as the Node harness emulates it."""
    x = min(mx, max(mn, v))
    x = mn + js_round((x - mn) / st) * st
    return float(js_to_fixed(x, 10))


def al_set_rot(m, deg, snap=0.05):
    """Set m.p.rot (radians) to `deg` degrees. Returns (ok, message or None).
    snap=0.05 reproduces the web app's slider (range 0..90, step 0.05): the rotation is set to
    the snapped value even when it cannot land exactly (ok False), as the JS does. snap=None sets
    the exact angle (the port's continuous slider). The caller resets the membrane."""
    if m.p.shape == 'square':
        return False, 'the exact square is fixed at 0° — use "square, rotatable"'
    if snap:
        v = _slider(deg, 0, 90, snap)
        m.p.rot = v * PI / 180
        ok = abs(v - deg) < 1e-9
        return ok, (None if ok else f'slider could not land on {js_num_str(deg)}° exactly (step {js_num_str(snap)})')
    m.p.rot = deg * PI / 180
    return True, None


_NOTHING = 'every flavor edge is clipped away at this radius — nothing to align'


def _result(m, before, msgs):
    p = m.p
    after = {'rot': p.rot, 'lattice': p.lattice, 'Co': p.Co}
    fields = {k: v for k, v in after.items() if v != before[k]}
    return {'changed': bool(fields), 'fields': fields, 'msg': msgs[-1] if msgs else None, 'msgs': msgs}


def _state(m):
    return {'rot': m.p.rot, 'lattice': m.p.lattice, 'Co': m.p.Co}


def al_step(m, direction, cls='row', snap=0.05):
    """alStep: to the next (+1) / previous (-1) aligned angle, cyclic. Sets m.p.rot; returns
    {'changed', 'fields' (what changed), 'msg' (the flash), 'msgs'}. The caller resets."""
    before, msgs = _state(m), []
    if al_nothing_left(m):
        return _result(m, before, [_NOTHING])
    L = al_list(m, cls)
    if not L:
        return _result(m, before, ['the circle has no edges to align' if m.p.shape == 'circle'
                                   else 'no aligned angle'])
    cur = al_rot_deg(m)
    e = None
    if direction > 0:
        e = next((x for x in L if x['rot'] > cur + 1e-6), None) or L[0]
    else:
        for x in reversed(L):
            if x['rot'] < cur - 1e-6:
                e = x
                break
        e = e or L[-1]
    ok, msg = al_set_rot(m, e['rot'], snap)
    if msg:
        msgs.append(msg)
    if ok:
        msgs.append(f"aligned: {fmt(e['rot'], 2)}° — {al_aligned(e, cls)}/{e['n']} edges")
    return _result(m, before, msgs)


def al_optimal(m, cls='row', snap=0.05):
    """alOptimal: the best aligned angle for the current lattice (see al_best)."""
    before, msgs = _state(m), []
    if al_nothing_left(m):
        return _result(m, before, [_NOTHING])
    L = al_list(m, cls)
    if not L:
        return _result(m, before, ['the circle has no edges to align' if m.p.shape == 'circle'
                                   else 'no aligned angle'])
    b = al_best(L, cls)
    ok, msg = al_set_rot(m, b['rot'], snap)
    if msg:
        msgs.append(msg)
    if ok:
        msgs.append(f"optimal angle {fmt(b['rot'], 2)}° — {al_aligned(b, cls)}/{b['n']} edges")
    return _result(m, before, msgs)


def al_flavor(m, cls='row', snap=0.05):
    """alFlavor: switch to the lattice that aligns the most edges (m.p.lattice; Co is lowered to
    0.95 of the new limit if it would be unstable, as the web app's lattice control does) and
    set its best angle. The caller resets."""
    before, msgs = _state(m), []
    if al_nothing_left(m):
        return _result(m, before, [_NOTHING])
    if not al_dirs(m):
        return _result(m, before, ['the circle has no edges — nothing to align'
                                   if m.p.shape == 'circle' else 'this shape has no edges to align'])
    pk = al_flavor_pick(m, cls)
    if pk['lat'] != al_lat(m):
        m.p.lattice = pk['lat']
        if m.p.Co > m.co_limit():
            m.p.Co = _slider(js_round(m.co_limit() * 0.95 * 200) / 200, 0.05, 1.00, 0.005)
    if pk['best'] and m.p.shape != 'square':
        ok, msg = al_set_rot(m, pk['best']['rot'], snap)
        if msg:
            msgs.append(msg)
    if al_nothing_left(m):
        msgs.append(f"switched to the {AL_NAME[pk['lat']]} lattice, but at this radius its array clips "
                    'every flavor edge away — lower the radius')
        return _result(m, before, msgs)
    b = pk['best']
    msgs.append(f"optimal grid flavor: {AL_NAME[pk['lat']]} @ {fmt(b['rot'] if b else 0, 2)}° — "
                f"{al_aligned(b, cls) if b else 0}/{len(al_dirs(m))} edges" + (' (tie: kept)' if pk['tie'] else ''))
    return _result(m, before, msgs)


# ---- the two ledger rows the alignment feeds ----------------------------------------------
LEDGER_7C = 'edge alignment: each edge’s staircase repeats with its lattice period'


def al_ledger_rows(m):
    """Ledger rows 7c and 7c-bis as (label, status, text, tier); status 'PASS' / 'FAIL' / 'n/a'."""
    rows = []
    E = al_measure(m)['edges']
    if not E:
        rows.append((LEDGER_7C, 'n/a',
                     (f"shape '{m.p.shape}' has no alignment model" if _model(m) == 'unknown' else
                      'the ∞ flavor has no edges — every boundary point faces a different direction, '
                      'so there is nothing to align'), 'align'))
    else:
        G, checked, bad, short = {}, 0, 0, 0
        for e in E:
            key = ('g' if e['grid'] else 'f') + str(js_round(e['phi'] * 1e6))
            if key not in G:
                G[key] = {'e': e, 'count': 0, 'meas': []}
            G[key]['count'] += 1
            G[key]['meas'].append(e['meas'])
            if e['cls'] == 'irr':
                continue
            if not e['meas']:
                short += 1
                continue
            checked += 1
            if abs(e['meas']['d'] - e['pred']) > 1e-9:
                bad += 1
        cn = lambda e: {'row': 'row', 'diag': 'diag', 'irr': 'irr'}.get(e['cls'], f"({e['da']},{e['db']})")
        parts = []
        for v in G.values():
            e, count = v['e'], v['count']
            ms = [x for x in v['meas'] if x]
            md = ms[0]['d'] if ms else None
            same = all(abs(x['d'] - md) < 1e-9 for x in ms)
            mtxt = 'too short to measure' if md is None else f"period {fmt(md, 3)}" + ('' if same else ' (varies)')
            if e['cls'] == 'irr':
                verdict = '' if md is None else (f" — follows ({ms[0]['a']},{ms[0]['b']}) over this length"
                                                 + (f"; drift estimate ({e['near']['a']},{e['near']['b']}) "
                                                    f"{fmt(e['near']['d'], 3)}" if e['near'] else ''))
            else:
                verdict = '' if md is None else (' ✓' if abs(md - e['pred']) < 1e-9
                                                 else f" ✗ (predicted {fmt(e['pred'], 3)})")
            parts.append(f"{fmt(e['phi'], 2)}° {'grid edge ' if e['grid'] else ''}{cn(e)} ×{count}: "
                         f"{mtxt}{verdict}")
        txt = (' · '.join(parts)
               + (f' — {checked} exact-direction edge(s) checked, shortest integer lattice shift that maps '
                  'each edge’s boundary cells onto boundary cells' if checked else
                  ' — no edge long enough to hold two periods clear of its vertices')
               + (f' · {short} edge(s) too short to measure' if short else '')
               + ' · row = a straight line of nodes (period 1); diag = the shortest zigzag, acts smooth '
                 'for λ > 2d; irrational = no exact period')
        rows.append((LEDGER_7C + ' (measured from the mask)', ('FAIL' if bad else 'PASS') if checked else 'n/a',
                     txt, 'align'))
    RE = al_rep(m)['edges']
    E2 = [e for e in RE if not e['arr']]
    judged = [e for e in E2 if e['measStraight'] is not None]
    badj = [e for e in judged if e['measStraight'] != e['predStraight']]
    nRow = len([e for e in E2 if e['cls'] == 'row'])
    nDg = len([e for e in E2 if e['cls'] == 'diag'])
    straight = len([e for e in judged if e['measStraight']])
    if judged:
        txt = (f'{nRow} of {len(E2)} edges on lattice rows, {nDg} on the {al_diag_name(m.p.lattice)}, '
               f'{len(E2) - nRow - nDg} other · measured from the mask: {straight} straight (every '
               'zeroed node at one distance from the drawn edge)'
               + (' · DISAGREE on edge(s) ' + ', '.join(str(next(i for i, x in enumerate(E2) if x is e) + 1)
                                                           for e in badj)
                  if badj else ' · rule and mask agree on every edge'))
    elif m.p.shape == 'circle':
        txt = 'the circle has no edges'
    elif m.sq_full():
        txt = 'the full-grid square: its walls ARE the array edge, governed by the boundary control'
    elif (lambda P: bool(P) and not fits_inside(m, P))(_shape_poly(m)) and not any(not e['arr'] for e in RE):
        txt = 'every flavor edge is clipped away at this radius: the wall is the array edge'
    else:
        txt = 'the edges are too short to measure'
    rows.append(('edge alignment: the direction rule predicts exactly which edges are ONE straight line of nodes',
                 (('FAIL' if len(badj) > 0 else 'PASS') if judged else 'n/a'), txt, 'tier C'))
    return rows


# ---- straightness report and edge colouring for the viewer ---------------------------------
def al_level(cls):
    """The port's per-edge LEVEL (the JS has no such letter): A = a row (one straight line of
    nodes, period 1), B = the shortest diagonal/zigzag (acts smooth for lambda >= 4),
    C = any other lattice direction or an irrational one (a grating / a weak leak)."""
    return 'A' if cls == 'row' else 'B' if cls == 'diag' else 'C'


def al_straightness(m, cls='row'):
    """Per edge that is not on the array edge: aligned (under cls), predicted and measured
    straightness, level, the zeroed nodes' offset. A list of dicts plus one text line each."""
    out = []
    for i, e in enumerate([e for e in al_rep(m)['edges'] if not e['arr']]):
        aligned = e['cls'] == 'row' or (cls == 'smooth' and e['cls'] == 'diag')
        ms = e['measStraight']
        if not e['cnt']:
            off = 'on the array edge'
        elif e['spread'] < 1e-6:
            off = '+' + js_to_fixed(e['mean'], 3)
        else:
            off = '+' + js_to_fixed(e['lo'], 2) + '..+' + js_to_fixed(e['hi'], 2)
        verdict = ('' if ms is None else (' OK' if ms == e['predStraight'] else ' DISAGREE'))
        txt = (f"e{i + 1} {fmt(math.fmod(e['normal'] + 270, 180), 2)}° {e['cls']} [{al_level(e['cls'])}] "
               f"aligned {'yes' if aligned else 'no'} · straight: rule {'yes' if e['predStraight'] else 'no'}, "
               f"mask {'-' if ms is None else 'yes' if ms else 'no'}{verdict} · {off}")
        out.append({'i': i + 1, 'cls': e['cls'], 'level': al_level(e['cls']), 'aligned': aligned,
                    'pred': e['predStraight'], 'meas': ms, 'offset': off, 'text': txt})
    return out


def al_edge_lines(m):
    """The web app's edge colouring (buildEdgeCol): [(a, b, rgb)] in cell offsets from the centre."""
    return [(e['a'], e['b'], AL_COL[e['cls']]) for e in al_edges(m)]


# ---- GGUI panel ---------------------------------------------------------------------------
# GGUI draws with ImGui's default font, whose glyphs cover 0x20..0xFF only (Latin-1): degree,
# times and middle dot survive, dashes, arrows and Greek letters would print as '?'.
_GLYPH = {'—': '--', '–': '-', '→': '->', 'λ': 'lambda', 'σ': 'sigma',
          '∞': 'inf', '≅': '~=', '≡': '==', 'α': 'alpha', '√': 'sqrt',
          '−': '-', '…': '...', '’': "'", '✓': 'OK', '✗': 'X'}
_COLKEY = {'hi': (0.72, 0.62, 1.0), 'accent': (0.85, 0.55, 1.0), 'link': (0.95, 0.45, 0.45),
           'dim': (0.55, 0.55, 0.60), None: (0.85, 0.85, 0.85)}


def gui_text(s):
    return ''.join(_GLYPH.get(ch, ch if ord(ch) < 256 else '?') for ch in s)


class AlignPanel:
    """Per-viewer panel state: the class filter, the pending snap index, the edge-colour flag,
    and the cached panel data (recomputed only when its key changes)."""

    def __init__(self):
        self.cls = 'row'
        self.edge_col = False
        self.snap_i = -1
        self.wrap = 64
        self._key = None
        self._data = None

    def key(self, m):
        p = m.p
        return _geom_key(m) + (self.cls, self.edge_col, p.Co, p.freq, p.sigma, p.src)

    def _lines(self, D):
        """The note, the straightness report and the two ledger verdicts, wrapped once per key
        (textwrap every frame cost ~1.5 ms at 60 lines)."""
        col = lambda k: _COLKEY.get(k, AL_COL.get(k, _COLKEY[None]))
        out = []
        for line, ck in D['note']:
            out += [(piece, col(ck)) for piece in (textwrap.wrap(gui_text(line), self.wrap) or [''])]
        if D['straight']:
            out.append(('straightness, per edge (level A row / B diag / C other):', _COLKEY['dim']))
            for s in D['straight']:
                out += [(piece, AL_COL[s['cls']]) for piece in textwrap.wrap(gui_text(s['text']), self.wrap)]
        for lab, status, txt, tier in D['ledger']:
            out.append((gui_text(f'[{status}] ' + lab.replace('edge alignment: ', '')),
                        (0.4, 0.9, 0.5) if status == 'PASS' else (0.95, 0.4, 0.4) if status == 'FAIL'
                        else _COLKEY['dim']))
        return out

    def data(self, m):
        k = self.key(m)
        if k != self._key:
            L = al_list(m, self.cls)
            snap = al_snap_options(m, self.cls)
            self._data = {'note': al_note_lines(m, self.cls, L, self.edge_col), 'snap': snap,
                          'straight': al_straightness(m, self.cls), 'ledger': al_ledger_rows(m)}
            self._data['lines'] = self._lines(self._data)
            self.snap_i = snap['selected'] if snap['selected'] >= 0 else (0 if snap['entries'] else -1)
            self._key = k
        return self._data


def _flash(viewer, msg):
    if not msg:
        return
    f = getattr(viewer, 'flash', None)
    if callable(f):
        f(gui_text(msg))
    else:
        print(gui_text(msg))


def _apply(viewer, res):
    _flash(viewer, res['msg'])
    if res['changed']:
        viewer.reset()


def gui_section(w, viewer):
    """Draw the alignment panel inside an open GGUI sub_window `w`. viewer has .m (Membrane) and
    .reset(); optional .flash(msg). Heavy work is cached, so calling it every frame is cheap."""
    st = getattr(viewer, 'align_panel', None)
    if st is None:
        st = AlignPanel()
        viewer.align_panel = st
    m = viewer.m
    D = st.data(m)
    w.text('grid <-> edge alignment', color=_COLKEY['hi'])
    if w.button('aligned = row' if st.cls == 'row' else 'aligned = row or shortest diagonal'):
        st.cls = 'smooth' if st.cls == 'row' else 'row'
        D = st.data(m)
    snap = D['snap']
    E = snap['entries']
    if E:
        i = min(max(st.snap_i, 0), len(E) - 1)
        w.text(gui_text(f"snap {i + 1}/{len(E)}: {E[i]['label']}" + ('  (current)' if i == snap['selected'] else '')))
        if w.button('< snap'):
            st.snap_i = (i - 1) % len(E)
        if w.button('snap >'):
            st.snap_i = (i + 1) % len(E)
        if w.button('apply snap'):
            ok, msg = al_set_rot(m, E[i]['rot'], snap=None)
            _flash(viewer, msg or f"rotation {fmt(E[i]['rot'], 2)}°")
            viewer.reset()
    else:
        w.text(gui_text(snap['header']))
    if w.button('< prev aligned'):
        _apply(viewer, al_step(m, -1, st.cls, snap=None))
    if w.button('next aligned >'):
        _apply(viewer, al_step(m, +1, st.cls, snap=None))
    if w.button('optimal angle'):
        _apply(viewer, al_optimal(m, st.cls, snap=None))
    if w.button('optimal grid flavor'):
        _apply(viewer, al_flavor(m, st.cls, snap=None))
    st.edge_col = w.checkbox('colour the domain edges by alignment', st.edge_col)
    for piece, c in st.data(m)['lines']:
        w.text(piece, color=c)
