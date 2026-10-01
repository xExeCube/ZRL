"""Cross-check: the port's alignment panel (alignment_MIRROR.py) against the web app's OWN code.

The web app's module script runs headlessly under Node (ToolsCLAUDE/js_align_dump_MIRROR.mjs,
through js_harness_MIRROR.mjs); the port runs the same configurations on the core's Membrane.
Compared, config by config and function by function:
  pure        alClass / alLatticeDir / alNearDir on a grid of angles, alAngles, alGcdF, alCanon
  live        alRotDeg, alDirs, alCount, alSymPeriod, fitsInside, alNothingLeft
  edges       alEdges (clipped edges, classes, near directions)
  measure     alMeasure (the measured period of each edge's boundary cells, from the mask)
  report      alEdgeReport / alRep (the zeroed nodes' distance beyond each drawn edge)
  per class   alList, alCanon per entry, alBestOf(sq / tri), alFlavorPick,
              the NOTE (alNoteRender) and the snap list (alRefresh), HTML -> text,
              alStep(+1), alStep(-1), alOptimal, alFlavor: rotation, lattice, Co and flash text
  ledger      rows 7c (period, measured) and 7c-bis (straightness), HTML -> text
Angles and lengths to 1e-9 (relative for large values), counts / classes / booleans / strings exact
(edge pieces shorter than 1e-3 cells: float fields at 1e-13 * N / len, see diff_edges).

    python crosscheck_alignment_MIRROR.py [--quick | --ext] [--reuse] [--arch cpu|cuda] [--app PATH]
    python crosscheck_alignment_MIRROR.py --unl [--big]     'unlicensed' invariants (no JS)

Results 30/09/2026 (CPU f64, Node 20.11.1): full 80055/80055 over 2295 configs (JS 265 s, port 73 s);
--ext 82380/82380 over 2370 configs (21 sliver fields differ at 1e-9 only); --quick 27882/27882
(also on CUDA); --unl --big 51266/51266 checks over 816 configs.
"""
import argparse, html, json, math, os, re, subprocess, sys, time
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import membrane_core_MIRROR as mc
import alignment_MIRROR as al

D = math.pi / 180
ROTS = [0, 5, 7.5, 12, 15, 18, 22.5, 30, 36, 45, 54, 60, 72, 75, 90]
RADS = [0.5, 1.0, 1.5]
RTHS = [60, 45, 30]
NS = [161, 257]
SHAPES = ['square', 'n4', 'circle', 'n3', 'n5', 'n6', 'n8', 'n12', 'rhomb']


def base(shape, lat, rot, rad, rth, N, **kw):
    p = dict(shape=shape, lattice=lat, rot=rot * D, rad=rad, rtheta=rth * D, N=N, stencil=5, Co=0.5,
             freq=0.12, sigma=3.0, src='pulse')
    p.update(kw)
    return p


def configs(quick=False):
    C = []
    rots = [0, 7.5, 12, 22.5, 45, 72] if quick else ROTS
    ns = [161] if quick else NS
    for N in ns:
        for shape in SHAPES:
            for lat in ('sq', 'tri'):
                for rot in rots:
                    for rad in RADS:
                        for th in (RTHS if shape == 'rhomb' else [60]):
                            C.append({'name': f'{shape}_{lat}_r{rot}_R{rad}_t{th}_N{N}',
                                      'p': base(shape, lat, rot, rad, th, N)})
    # the 9-point stencil changes which zeroed nodes the report sees (and the 45-degree rule)
    for shape in SHAPES:
        for rot in rots:
            for th in (RTHS if shape == 'rhomb' else [60]):
                C.append({'name': f'{shape}_sq9_r{rot}_t{th}', 'p': base(shape, 'sq', rot, 1.0, th, 161, stencil=9)})
    # the other readout inputs: Co above the square limit (alFlavor lowers it), sources, drive
    # frequency, the edge-colour legend
    for shape in ('n5', 'n6', 'rhomb', 'n12', 'square'):
        for lat in ('sq', 'tri'):
            for rot in (0, 15, 30):
                C.append({'name': f'{shape}_{lat}_r{rot}_Co75', 'p': base(shape, lat, rot, 1.0, 60, 161, Co=0.75)})
                C.append({'name': f'{shape}_{lat}_r{rot}_cont', 'edgeCol': True,
                          'p': base(shape, lat, rot, 0.8, 50, 161, src='cont', freq=0.03)})
                C.append({'name': f'{shape}_{lat}_r{rot}_vtx', 'p': base(shape, lat, rot, 1.2, 70, 161, src='vtx', sigma=1.5)})
                C.append({'name': f'{shape}_{lat}_r{rot}_imp', 'p': base(shape, lat, rot, 0.7, 60, 161, src='impulse')})
                C.append({'name': f'{shape}_{lat}_r{rot}_s8', 'p': base(shape, lat, rot, 0.9, 60, 161, sigma=8.0, freq=0.25)})
    return C


EXT_SHAPES = [('square', 60), ('n3', 60), ('n4', 60), ('n5', 60), ('n6', 60), ('n8', 60), ('n12', 60),
              ('rhomb', 60), ('rhomb', 45), ('rhomb', 30), ('rhomb', 72.5)]
EXT_NEAR = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]      # degrees, the HTML slider's step
EXT_RAD_OFF = [-2e-2, -1e-3, -1e-9, 1e-9, 1e-3, 2e-2]
EXT_RTH = [20, 25, 30, 33.33, 35, 40, 45, 50, 55, 60, 65, 70, 71.1, 75, 80, 85, 89.95, 90]


def _threshold(m, pred, lo=0.3, hi=4.0):
    """The radius where pred(m) turns True (pred monotone in rad), by bisection, or None."""
    m.p.rad = hi
    if not pred(m):
        return None
    m.p.rad = lo
    if pred(m):
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        m.p.rad = mid
        lo, hi = (lo, mid) if pred(m) else (mid, hi)
    return hi


def configs_ext(m):
    """The extended sweep (30/09/2026): rotations in 0.05-degree steps around every aligned angle
    (both classes, both lattices, the 9-point stencil too), radii just below / above the two
    clipping thresholds (the polygon stops fitting; every flavor edge is clipped away), and
    rhombus angles 20..90 degrees (33.33 / 71.1 / 89.95 put aligned angles off the 0.05 grid).
    The thresholds are found on the port's geometry (fits_inside / al_nothing_left only read
    shape_poly and clip_region, so no reseed is needed while bisecting)."""
    C = []
    for shape, th in EXT_SHAPES:
        for lat in ('sq', 'tri'):
            m.p.shape, m.p.lattice, m.p.rtheta, m.p.rad, m.p.rot, m.p.stencil = shape, lat, th * D, 1.0, 0.0, 5
            if shape == 'square':
                angs = [0.0]
            else:
                angs = sorted({e['rot'] for cls in ('row', 'smooth') for e in al.al_list(m, cls)})
            for a0 in angs:
                for off in EXT_NEAR:
                    r = round(a0 + off, 9)
                    if not 0 <= r <= 90:
                        continue
                    C.append({'name': f'near_{shape}_{lat}_t{th}_r{r}', 'p': base(shape, lat, r, 1.0, th, 161)})
                    if lat == 'sq' and abs(off) < 0.06:
                        C.append({'name': f'near9_{shape}_t{th}_r{r}',
                                  'p': base(shape, lat, r, 1.0, th, 161, stencil=9)})
    for shape, th in EXT_SHAPES:
        if th == 72.5:
            continue
        for lat in ('sq', 'tri'):
            for rot in ((0,) if shape == 'square' else (0, 7.5, 12, 30, 45)):
                m.p.shape, m.p.lattice, m.p.rtheta, m.p.rot, m.p.stencil = shape, lat, th * D, rot * D, 5
                for tname, pred in (('fit', lambda mm: not al.fits_inside(mm, mm.shape_poly())),
                                    ('none', al.al_nothing_left)):
                    r0 = _threshold(m, pred)
                    if r0 is None:
                        continue
                    for off in EXT_RAD_OFF:
                        C.append({'name': f'rad_{tname}_{shape}_{lat}_t{th}_r{rot}_R{r0 + off!r}',
                                  'p': base(shape, lat, rot, r0 + off, th, 161)})
    for th in EXT_RTH:
        for lat in ('sq', 'tri'):
            for rot in (0, 10, 22.5, 45, 60, 80):
                for rad in (0.8, 1.3):
                    C.append({'name': f'rth_{lat}_t{th}_r{rot}_R{rad}', 'p': base('rhomb', lat, rot, rad, th, 161)})
            for rot in (0, 22.5):
                C.append({'name': f'rth_{lat}_t{th}_r{rot}_N257', 'p': base('rhomb', lat, rot, 1.0, th, 257)})
    m.p.rad, m.p.rot, m.p.stencil = 1.0, 0.0, 5
    return C


# ---- 'unlicensed' (port only: the JS has no such shape, so it is checked by invariants) -------
UNL_ROTS = [0, 0.05, 5, 7.5, 12, 14.95, 15, 22.5, 26.56505117707799, 29.95, 30, 30.05, 33.69006752597979,
            36.86989764584402, 44.95, 45, 45.05, 53.13010235415598, 59.95, 60, 60.05, 63.43494882292201,
            75, 89.95, 90]
UNL_RADS = [0.5, 0.8, 0.89, 1.0, 1.2, 1.6, 2.2, 3.0]


def _seg_dist(px, py, a, b):
    """Distance from points (arrays) to the segment ab."""
    import numpy as np
    ux, uy = b[0] - a[0], b[1] - a[1]
    L2 = ux * ux + uy * uy or 1.0
    t = np.clip(((px - a[0]) * ux + (py - a[1]) * uy) / L2, 0, 1)
    return np.hypot(px - a[0] - t * ux, py - a[1] - t * uy)


def unl_check_one(m, bad, where, solids_cmp=False):
    """Every invariant the 'unlicensed' alignment model promises, on the core's real plate.
    Appends failures to bad; returns the number of checks made."""
    import numpy as np
    n = 0

    def ck(ok, what):
        nonlocal n
        n += 1
        if not ok:
            bad.append(f'{where}: {what}')
    p, lat = m.p, al.al_lat(m)
    rot = p.rot * 180 / math.pi
    hw, hh, rc = m.plate_dims()
    P = [tuple(map(float, q)) for q in m.shape_poly()]
    C = m.clip_region()
    # (1) the model: 4 straight sides of the right lengths, directions rot / rot + 90, CCW outline
    sides = al._unl_sides(m, P)
    ck(len(sides) == 4, f'{len(sides)} straight sides')
    for p0, p1, dd in sides:
        ln = al.js_hypot(p1[0] - p0[0], p1[1] - p0[1])
        want = 2 * (hw - rc) if dd == 0 else 2 * (hh - rc)
        ck(abs(ln - want) < 1e-9 * max(1, want), f'side {dd} length {ln} != {want}')
    ck(al.poly_area(P) > 0 and sum(P[k][0] * P[(k + 1) % len(P)][1] - P[(k + 1) % len(P)][0] * P[k][1]
                                   for k in range(len(P))) > 0, 'outline not CCW')
    ck(al.al_dirs(m) == [0, 90, 0, 90], f'dirs {al.al_dirs(m)}')
    ck(al.al_sym_period(p.shape, lat) == (90 if lat == 'sq' else 60), 'sym period')
    # (2) the clipped outline split into side pieces / array-edge pieces / arc pieces, none lost
    A = al._al_edges_all(m)
    cen = [(sx * (hw - rc), sy * (hh - rc)) for sx, sy in ((1, -1), (1, 1), (-1, 1), (-1, -1))]
    co, si = math.cos(p.rot), math.sin(p.rot)
    cen = [(x * co - y * si, x * si + y * co) for x, y in cen]
    on_clip = lambda q: any(al._on_line(q, C[e], C[(e + 1) % len(C)]) for e in range(len(C)))
    on_arc = lambda q: any(abs(al.js_hypot(q[0] - c[0], q[1] - c[1]) - rc) < 1e-6 for c in cen)
    for e in A:
        if e['arc']:
            ck(all(on_arc(q) or on_clip(q) for q in (e['a'], e['b'])), f"arc piece off the arcs {e['a']} {e['b']}")
        elif e['grid']:
            ck(on_clip(e['a']) and on_clip(e['b']), 'grid piece off the array edge')
        else:
            hit = [dd for p0, p1, dd in sides if al._on_line(e['a'], p0, p1) and al._on_line(e['b'], p0, p1)]
            ck(len(hit) == 1, 'side piece on no side')
            if hit:
                d = abs(al.al_mod180(e['phi'] - (hit[0] + rot)))
                ck(min(d, 180 - d) < 1e-9, f"side piece phi {e['phi']} vs {hit[0] + rot}")
    E = al.al_edges(m)
    fits = al.fits_inside(m, P)
    if fits:
        ck(len(E) == 4 and not any(e['grid'] for e in E), f'fits but {len(E)} edges')
        ck(len([e for e in A if e['arc']]) == 4 * mc.PLATE_ARC_SEG, 'arc piece count')
    ck(al.al_nothing_left(m) == (not fits and not any(not e['grid'] for e in E)), 'nothing-left rule')
    # (3) al_count / al_list on a rectangle: rows at rot = 0, 90 (sq) and 0, 30, 60, 90 (tri)
    c = al.al_count(al.al_dirs(m), al.al_rot_deg(m), lat)
    cls = [al._al_class(dd + rot, lat)['cls'] for dd in (0, 90, 0, 90)]
    ck(c['rows'] == cls.count('row') and c['diags'] == cls.count('diag'), 'al_count')
    L = [e['rot'] for e in al.al_list(m, 'row')]
    ck(L == ([0.0, 90.0] if lat == 'sq' else [0.0, 30.0, 60.0, 90.0]), f'row list {L}')
    L = [e['rot'] for e in al.al_list(m, 'smooth')]
    ck(L == ([0.0, 45.0, 90.0] if lat == 'sq' else [0.0, 30.0, 60.0, 90.0]), f'smooth list {L}')
    # (4) the mask and the outline agree (shape, size AND rotation): every boundary cell of the
    # domain mask, and every zeroed node touching it, is within one lattice spacing (+ the chord
    # sagitta) of the clipped outline
    Q = m.clip_convex(P, C)
    Mj = al._mask_ji(m)
    N = m.N
    pad = np.zeros((N + 2, N + 2), dtype=bool)
    pad[1:-1, 1:-1] = Mj
    nb = al.NB_TRI if lat == 'tri' else [(1, 0), (-1, 0), (0, 1), (0, -1)]
    outn = np.zeros((N, N), dtype=bool)
    inn = np.zeros((N, N), dtype=bool)
    for di, dj in nb:
        outn |= ~pad[1 + dj:N + 1 + dj, 1 + di:N + 1 + di]
        inn |= pad[1 + dj:N + 1 + dj, 1 + di:N + 1 + di]
    sag = rc * (1 - math.cos(math.pi / 4 / mc.PLATE_ARC_SEG)) + 1e-9
    for sel, what in ((Mj & outn, 'boundary cell'), (~Mj & inn, 'zeroed node')):
        J, I = np.nonzero(sel)
        x, y = al._cell_xy_np(m, I, J)
        d = np.full(len(I), np.inf)
        for k in range(len(Q)):
            d = np.minimum(d, _seg_dist(x, y, Q[k], Q[(k + 1) % len(Q)]))
        dm = float(d.max()) if len(I) else 0.0
        ck(dm <= 1 + sag, f'{what} {dm:.4f} from the outline')
    # (5) the rules agree with the mask: ledger rows never FAIL, every judged edge agrees
    rows = al.al_ledger_rows(m)
    ck(rows[0][1] != 'FAIL' and rows[1][1] != 'FAIL', f'ledger {rows[0][1]} / {rows[1][1]}')
    for e in al.al_rep(m)['edges']:
        if e['measStraight'] is not None:
            ck(e['measStraight'] == e['predStraight'], f"straightness {e['cls']} meas {e['measStraight']}")
    for e in al.al_measure(m)['edges']:
        if e['cls'] != 'irr' and e['meas']:
            ck(abs(e['meas']['d'] - e['pred']) < 1e-9, f"period {e['cls']} {e['meas']['d']} vs {e['pred']}")
    # (6) the panel runs end to end, and says what it is
    note = al.al_note(m, 'row')
    ck('unlicensed (port only)' in note, 'note line')
    al.al_snap_options(m, 'smooth'); al.al_straightness(m, 'smooth')
    for fn in (lambda: al.al_step(m, 1), lambda: al.al_step(m, -1), lambda: al.al_optimal(m),
               lambda: al.al_flavor(m, 'smooth')):
        save = (p.rot, p.lattice, p.Co)
        fn()
        p.rot, p.lattice, p.Co = save
    # (7) interior solids (the plate's lettering and screws) are not domain edges
    if solids_cmp and m.has_solids:
        scan = lambda: json.dumps([al._al_measure(m), al._al_edge_report(m)], sort_keys=True)  # NaN == NaN here
        with_s = scan()
        sc = m.scene
        m.attach_scene(None)
        ck(not m.has_solids, 'solids still present after attach_scene(None)')
        ck(with_s == scan(), 'solids change the edge scans')
        m.attach_scene(sc)
    return n


def unl_checks(m, big=False):
    bad, n, k = [], 0, 0
    t0 = time.time()
    Ns = [161, 257] + ([1025] if big else [])
    for N in Ns:
        for lat, st in (('sq', 5), ('sq', 9), ('tri', 5)):
            rots = UNL_ROTS if N == 161 else UNL_ROTS[::3]
            for rot in rots:
                for rad in (UNL_RADS if N == 161 else UNL_RADS[::2]):
                    m.p.shape, m.p.lattice, m.p.stencil, m.p.rot, m.p.rad, m.p.N = 'unlicensed', lat, st, rot * D, rad, N
                    m.reseed()
                    n += unl_check_one(m, bad, f'N{N} {lat}{st} rot {rot} rad {rad}', solids_cmp=(k % 7 == 0))
                    k += 1
    # the scan cache: a set_co (new mask object, same content) is a hit; a geometry change read
    # before the reseed (a stale mask) must not survive the reseed
    m.p.shape, m.p.lattice, m.p.stencil, m.p.rot, m.p.rad, m.p.N = 'unlicensed', 'sq', 5, 12 * D, 1.0, 257
    m.reseed()
    r1, M0 = al.al_rep(m), m.domain_mask_np
    m.set_co(m.p.Co * 0.9)
    n += 2
    if m.domain_mask_np is M0:
        bad.append('cache: set_co kept the mask object (the content path is not exercised)')
    if al.al_rep(m) is not r1:
        bad.append('cache: set_co (same geometry, equal mask) rescanned')
    m.p.rot = 30 * D
    stale = al.al_rep(m)                       # the geometry says 30 degrees, the mask is still 12
    m.reseed()
    fresh = al.al_rep(m)
    n += 1
    if fresh is stale or json.dumps(fresh, sort_keys=True) != json.dumps(al._al_edge_report(m), sort_keys=True):
        bad.append('cache: a result from a stale mask survived the reseed')
    print(f"'unlicensed' invariants: {n - len(bad)}/{n} checks pass over {k} configs ({time.time() - t0:.1f} s)")
    for s in bad[:20]:
        print('   ', s)
    return not bad


def pure_inputs():
    phis = [k * 0.25 for k in range(0, 721)] + [1.3, 10.3, 17.3, 33.69006752597979, 26.56505117707799,
                                                 40.89339464913091, 19.106605350869096, 173.1, 91.7]
    angles = []
    for shape in ('n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n12', 'n16', 'square'):
        for lat in ('sq', 'tri'):
            angles.append([al.al_edge_dirs0(shape, 60 * D), lat, 0, 90])
    for th in (20, 30, 33.5, 45, 60, 72.5, 90):
        for lat in ('sq', 'tri'):
            angles.append([al.al_edge_dirs0('rhomb', th * D), lat, 0, 90])
            angles.append([al.al_edge_dirs0('rhomb', th * D), lat, -45, 180])
    gcd = [[a, b] for a in (180, 120, 72, 51.42857142857143, 45, 40, 36, 30, 22.5, 360 / 7)
           for b in (90, 60, 12.5)]
    canon = [[r, g] for r in (0, 6, 12, 18, 29.999999, 45, 54, 72, 89.75, -7.5) for g in (0, 12, 15, 30, 45, 60, 90)]
    return {'phis': phis, 'lens': [3, 12, 50, 200], 'angles': angles, 'gcd': gcd, 'canon': canon}


# ---- comparison ---------------------------------------------------------------------------
def h2t(s):
    """HTML -> the text the port writes: <br> is a newline, tags dropped, entities decoded."""
    s = re.sub(r'<br\s*/?>', '\n', s)
    s = re.sub(r'<[^>]+>', '', s)
    return html.unescape(s)


def num(x):
    if isinstance(x, str) and x in ('Infinity', '-Infinity', 'NaN'):
        return float(x.replace('Infinity', 'inf'))
    return x


def diff(js, py, path='', tol=1e-9):
    """First difference between a JS value (from JSON) and a port value, or None."""
    js = num(js)
    if js is None or py is None:
        if js is None and py is None:
            return None
        if isinstance(py, float) and math.isnan(py) and js is None:
            return None
        return f'{path}: js={js!r} py={py!r}'
    if isinstance(js, bool) or isinstance(py, bool):
        return None if js == py and type(js) is type(py) else f'{path}: js={js!r} py={py!r}'
    if isinstance(js, (int, float)) and isinstance(py, (int, float)):
        if math.isnan(js) and math.isnan(py):
            return None
        if math.isinf(js) or math.isinf(py):
            return None if js == py else f'{path}: js={js!r} py={py!r}'
        return None if abs(js - py) <= tol * max(1.0, abs(js)) else f'{path}: js={js!r} py={py!r}'
    if isinstance(js, dict) and isinstance(py, dict):
        for k in set(js) | set(py):
            if k == 'arc':
                continue
            r = diff(js.get(k), py.get(k), f'{path}.{k}', tol)
            if r:
                return r
        return None
    if isinstance(js, list) and isinstance(py, (list, tuple)):
        if len(js) != len(py):
            return f'{path}: length js={len(js)} py={len(py)}'
        for i, (a, b) in enumerate(zip(js, py)):
            r = diff(a, b, f'{path}[{i}]', tol)
            if r:
                return r
        return None
    if isinstance(js, str) and isinstance(py, str):
        if js == py:
            return None
        i = next((k for k in range(min(len(js), len(py))) if js[k] != py[k]), min(len(js), len(py)))
        return f'{path}: strings differ at {i}: js=...{js[max(0, i - 40):i + 60]!r} py=...{py[max(0, i - 40):i + 60]!r}'
    return f'{path}: types js={type(js).__name__} py={type(py).__name__}'


SLIVER = 1e-3        # cells
SLIVERS = []         # (where, strict difference) for every sliver edge compared loosely


def diff_edges(js, py, N, where, path=''):
    """diff() for a per-edge list, with a per-edge tolerance. An edge piece shorter than SLIVER
    cells (the clipper leaves ~1.1e-6-cell pieces where a vertex just crosses the array edge,
    e.g. at the nothing-left radius - 1e-9) has a direction fixed by two vertices ~1e-6 apart
    whose coordinates (~N/2) carry last-bit cos/sin differences (C runtime vs V8's fdlibm,
    ~1e-14 cells): its unit normal, window and near-direction angle then agree to only
    ~1e-14 * N / len (measured 30/09/2026: 1e-9..4e-8 at len 1.1e-6). Those float fields get
    tol = max(1e-9, 1e-13 * N / len); counts, classes, booleans and strings stay exact."""
    if not isinstance(js, list) or not isinstance(py, (list, tuple)) or len(js) != len(py):
        return diff(js, py, path)
    for i, (a, b) in enumerate(zip(js, py)):
        ln = b.get('len', 1.0) if isinstance(b, dict) else 1.0
        if ln < SLIVER:
            strict = diff(a, b, f'{path}[{i}]')
            r = diff(a, b, f'{path}[{i}]', max(1e-9, 1e-13 * N / ln))
            if strict:
                SLIVERS.append((where, f'len {ln:.3e}: {strict}'))
        else:
            r = diff(a, b, f'{path}[{i}]')
        if r:
            return r
    return None


class Tally:
    def __init__(self):
        self.n = OrderedDict()
        self.bad = OrderedDict()

    def add(self, name, d, where):
        self.n.setdefault(name, [0, 0])
        self.n[name][0] += 1
        if d is None:
            self.n[name][1] += 1
        else:
            self.bad.setdefault(name, []).append(f'{where}: {d}')


def parse_snap(s):
    opts = re.findall(r'<option value="([^"]*)"([^>]*)>([\s\S]*?)</option>', s)
    head = h2t(opts[0][2]) if opts else ''
    ent = [{'rot': float(v), 'label': h2t(t)} for v, a, t in opts[1:]]
    sel = next((i - 1 for i, (v, a, t) in enumerate(opts) if 'selected' in a), -1)
    return head, ent, sel


def run_ops(m, cls, q, T, where):
    p = m.p
    for op, fn in (('next', lambda: al.al_step(m, +1, cls)), ('prev', lambda: al.al_step(m, -1, cls)),
                   ('opt', lambda: al.al_optimal(m, cls)), ('flav', lambda: al.al_flavor(m, cls))):
        save = (p.rot, p.lattice, p.Co)
        r = fn()
        js = q['ops'][op]
        got = {'rot': p.rot, 'lattice': p.lattice, 'Co': p.Co, 'flash': r['msgs']}
        exp = {'rot': js['rot'], 'lattice': js['lattice'], 'Co': js['Co'], 'flash': [h2t(x) for x in js['flash']]}
        T.add({'next': 'al_step(+1)', 'prev': 'al_step(-1)', 'opt': 'al_optimal', 'flav': 'al_flavor'}[op],
              diff(exp, got, op, 1e-12), where)
        p.rot, p.lattice, p.Co = save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='cpu')
    ap.add_argument('--quick', action='store_true', help='N = 161 and 6 rotations only')
    ap.add_argument('--ext', action='store_true', help='the extended sweep instead (near-aligned angles, '
                    'clipping radii, rhombus angles 20-90)')
    ap.add_argument('--unl', action='store_true', help="instead: the 'unlicensed' invariants on the core's "
                    'real plate (no JS: the web app has no such shape)')
    ap.add_argument('--big', action='store_true', help='with --unl: N = 1025 too')
    ap.add_argument('--reuse', action='store_true', help='reuse the last JS dump if present')
    ap.add_argument('--app', default=None, help='path to wave_membrane_MIRROR.html')
    ap.add_argument('--show', type=int, default=4, help='mismatches printed per function')
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # the console is cp1252
    scratch = os.path.join(HERE, 'ScratchCLAUDE')
    os.makedirs(scratch, exist_ok=True)
    tag = 'ext' if a.ext else 'quick' if a.quick else 'full'
    cfg_path = os.path.join(scratch, f'align_configs_{tag}_MIRROR.json')
    out_path = os.path.join(scratch, f'align_js_{tag}_MIRROR.jsonl')
    mc.init_taichi(a.arch, quiet=True)
    m = mc.Membrane(mc.Params(N=161))
    if a.unl:
        return 0 if unl_checks(m, big=a.big) else 1
    C = configs_ext(m) if a.ext else configs(a.quick)
    cfg = {'pure': pure_inputs(), 'configs': C}
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f)
    t0 = time.time()
    if not (a.reuse and os.path.exists(out_path)):
        cmd = ['node', os.path.join(HERE, 'ToolsCLAUDE', 'js_align_dump_MIRROR.mjs'), cfg_path, out_path]
        if a.app:
            cmd.append(a.app)
        subprocess.run(cmd, check=True)
    print(f'JS dump: {time.time() - t0:.1f} s, {len(C)} configs')
    with open(out_path, encoding='utf-8') as f:
        lines = [json.loads(s) for s in f]
    T = Tally()
    # ---- pure ----
    P, jp = cfg['pure'], lines[0]
    assert jp.get('pure')
    k = kn = 0
    for lat in ('sq', 'tri'):
        for phi in P['phis']:
            w = f'{lat} phi={phi}'
            T.add('al_class', diff(jp['cls'][k], al.al_class(phi, lat)), w)
            T.add('al_lattice_dir', diff(jp['ldir'][k], al.al_lattice_dir(phi, lat, 24)), w)
            k += 1
            for ln in P['lens']:
                T.add('al_near_dir', diff(jp['near'][kn], al.al_near_dir(phi, lat, ln)), f'{w} len={ln}')
                kn += 1
    for i, (dirs, lat, lo, hi) in enumerate(P['angles']):
        T.add('al_angles', diff(jp['angles'][i], al.al_angles(dirs, lat, lo, hi)), f'{lat} {dirs} [{lo},{hi}]')
    for i, (x, y) in enumerate(P['gcd']):
        T.add('al_gcd_f', diff(jp['gcd'][i], al.al_gcd_f(x, y)), f'{x},{y}')
    for i, (r, g) in enumerate(P['canon']):
        T.add('al_canon', diff(jp['canon'][i], al.al_canon(r, g)), f'{r},{g}')
    # ---- live ----
    t1 = time.time()
    wraps = 0
    for c, js in zip(C, lines[1:]):
        assert c['name'] == js['name'], (c['name'], js['name'])
        w = c['name']
        for kk, v in c['p'].items():
            setattr(m.p, kk, v)
        m.reseed()
        lat, d, rot = al.al_lat(m), al.al_dirs(m), al.al_rot_deg(m)
        g = al.al_sym_period(m.p.shape, lat)
        T.add('al_rot_deg / al_lat', diff([js['rotDeg'], js['lat']], [rot, lat]), w)
        T.add('al_dirs', diff(js['dirs'], d), w)
        T.add('al_count', diff(js['count'], al.al_count(d, rot, lat)), w)
        T.add('al_sym_period', diff(js['symPeriod'], g), w)
        T.add('fits_inside', diff(js['fits'], al.fits_inside(m, m.shape_poly())), w)
        T.add('al_nothing_left', diff(js['nothingLeft'], al.al_nothing_left(m)), w)
        T.add('al_edges', diff(js['edges'], al.al_edges(m)), w)
        T.add('al_measure', diff_edges(js['meas'], al.al_measure(m)['edges'], m.N, w), w)
        rep = al.al_rep(m)['edges']
        # a normal is an angle mod 360: an outward normal at -1e-16 degrees reads 0 or 359.9999999999996
        # depending on the last bit of a polygon vertex (the core's cos/sin vs V8's fdlibm)
        wrap = [i for i, (jr, pr) in enumerate(zip(js['rep'], rep))
                if abs(abs(num(jr['normal']) - pr['normal']) - 360) < 1e-9]
        rep = [dict(pr, normal=num(js['rep'][i]['normal'])) if i in wrap else pr for i, pr in enumerate(rep)]
        wraps += len(wrap)
        T.add('al_edge_report', diff_edges(js['rep'], rep, m.N, w), w)
        rows = al.al_ledger_rows(m)
        jrows = [[h2t(x) for x in r] for r in js['ledger']]
        T.add('ledger 7c (period)', diff(jrows[0] if jrows else None, list(rows[0])), w)
        T.add('ledger 7c-bis (straightness)', diff(jrows[1] if len(jrows) > 1 else None, list(rows[1])), w)
        for cls in ('row', 'smooth'):
            q = js['cls'][cls]
            wc = f'{w} [{cls}]'
            L = al.al_list(m, cls)
            T.add(f'al_list', diff(q['list'], L), wc)
            T.add('al_canon (per entry)', diff(q['canon'], [al.al_canon(e['rot'], g) for e in L]), wc)
            T.add('al_best_of', diff([q['bestSq'], q['bestTri']], [al.al_best_of(m, 'sq', cls), al.al_best_of(m, 'tri', cls)]), wc)
            T.add('al_flavor_pick', diff(q['pick'], al.al_flavor_pick(m, cls)), wc)
            T.add('NOTE text (al_note)', diff(h2t(q['note']), al.al_note(m, cls, edge_col=bool(c.get('edgeCol')))), wc)
            head, ent, sel = parse_snap(q['snap'])
            so = al.al_snap_options(m, cls)
            T.add('snap list (al_snap_options)', diff([head, ent, sel], [so['header'], [{'rot': e['rot'], 'label': e['label']}
                                                                               for e in so['entries']], so['selected']]), wc)
            run_ops(m, cls, q, T, wc)
    print(f'port: {time.time() - t1:.1f} s')
    tot = sum(v[0] for v in T.n.values())
    ok = sum(v[1] for v in T.n.values())
    print(f'\n{"function":34s} {"agree":>14s}')
    for name, (n, g) in T.n.items():
        print(f'{name:34s} {g:6d}/{n:<6d}' + ('' if g == n else '   <-- DIFFER'))
    print(f'(edge normals compared mod 360: {wraps} read 0 in the JS and 360 - eps here, or back)')
    print(f'(sliver edges < {SLIVER} cells compared at 1e-13*N/len: {len(SLIVERS)} differ at the strict 1e-9'
          + (f', in {len({w for w, _ in SLIVERS})} configs; some shown below)' if SLIVERS else ')'))
    for w, s in sorted(SLIVERS, key=lambda x: x[1])[:a.show]:
        print('   ', w, s[:200])
    print(f'\nTOTAL {ok}/{tot} comparisons agree over {len(C)} configs (+ the pure sweep)')
    for name, lst in T.bad.items():
        print(f'\n{name}: {len(lst)} mismatches, first {min(a.show, len(lst))}:')
        for s in lst[:a.show]:
            print('   ', s[:600])
    return 0 if ok == tot else 1


if __name__ == '__main__':
    sys.exit(main())
