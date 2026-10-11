"""Tests for round 4c (05/10/2026): the port's vertex-source defaults and the source helpers the UI shows.
Design and numbers: NotesCLAUDE/vertex_choppiness_MIRROR.md section 7 (and README, Deviations).

    python defaults_tests_MIRROR.py [--arch cuda|vulkan|cpu] [--f32] [--quick] [--only 1,3,...]

 1  defaults: Params.vtx_inset = 'wall', Params.vtx_drive_bump = False; the wall inset cuts at most 0.012%
    of the vertex pulses at the app's defaults (N 513, sigma 3), the web app's radial inset 1.2-27%
    (the diagnosis' clip column, reproduced)
 2  driven vertices start from rest: an all-zero seed; vtx_drive_bump=True seeds exactly the undriven
    vertex field (the web app) and shows the one-cell dent at step 1; the default run equals the
    diagnosis' 'seeded bump removed' variant bit for bit
 3  source_note(): the clearance equals the ledger's (and so the web app's) CLEAR on 14 configurations;
    the clip line, the hint, the capped corners, the circle, driven-at-rest; the cache (per reseed /
    mask change, never per frame) and its cost
 4  drive_warnings(): hard drives under 8 cells per wavelength in cells mode only
 5  phase_lag(): the predicted lag of the pulse's fronts per domain crossing, against the lag MEASURED on
    the lattice (a centred pulse on the torus vs the exact continuum solution, per direction)
"""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import membrane_core_MIRROR as mc
import solids_MIRROR as so
import ledger_MIRROR as lg

D = math.pi / 180
FAILS = []


def ok(name, cond, det=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('  | ' + det if det else ''), flush=True)
    if not cond:
        FAILS.append(name)


def mk(**kw):
    prm = dict(N=513, src='vtx')
    prm.update(kw)
    return mc.Membrane(mc.Params(**prm))


# ---- 1 ---------------------------------------------------------------------------------------------
# (lattice, shape, rot deg, the diagnosis' radial clip %) -- vertex_choppiness_MIRROR.md table 4.2
# (s1b_invariant_all_MIRROR.py / s1_reproduce_MIRROR.py, N 513, sigma 3, Co 0.5, the 04/10 snapshot)
DIAG = [('sq', 5, 'square', 0, 9.4), ('sq', 5, 'n3', 7, 25.6), ('sq', 5, 'n4', 7, 13.5), ('sq', 5, 'n6', 0, 4.5),
        ('sq', 5, 'n12', 0, 1.7), ('sq', 5, 'rhomb', 30, 12.8), ('sq', 9, 'n5', 0, 6.6), ('tri', 5, 'n3', 30, 15.0),
        ('tri', 5, 'n3', 7, 27.1), ('tri', 5, 'n6', 30, 2.2), ('tri', 5, 'rhomb', 7, 15.9), ('sq', 5, 'unlicensed', 0, 1.2)]


def t1_defaults(a):
    p = mc.Params()
    ok('1a Params defaults: vtx_inset wall, vtx_drive_bump False; VTX_INSETS = [radial, wall]',
       p.vtx_inset == 'wall' and p.vtx_drive_bump is False and mc.VTX_INSETS == ['radial', 'wall'],
       f'vtx_inset {p.vtx_inset!r}, vtx_drive_bump {p.vtx_drive_bump!r}')
    rows, bad = [], []
    for lat, st, shape, rot, want in (DIAG[:6] if a.quick else DIAG):
        kw = dict(lattice=lat, stencil=st, shape=shape, rot=rot * D)
        if shape == 'unlicensed' and lat == 'tri':
            kw['rad'] = 0.89
        mw = mk(**kw)
        mr = mk(vtx_inset='radial', **kw)
        nw, nr = mw.source_note(), mr.source_note()
        cw, cr = 100 * mw.clip_frac, 100 * mr.clip_frac
        rows.append(f'{lat}{st if lat == "sq" else ""} {shape} {rot}: radial {cr:.1f}% (diag {want}) wall {cw:.4f}% '
                    f'clear {nw["clearance_sigma"]:.2f}/{nr["clearance_sigma"]:.2f} sigma')
        if not (abs(cr - want) <= 0.051 and cw <= 0.012 and nw['clearance_sigma'] >= 2.6 and not nw['clip_warn']
                and nr['clip_warn'] == (cr > 1) and nw['capped'] == 0):
            bad.append(rows[-1])
    ok(f'1b wall inset (the default) clip <= 0.012%, clearance >= 2.6 sigma; radial reproduces the diagnosis\' '
       f'clip column ({len(rows)} configs)', not bad, '; '.join(bad) if bad else ' | '.join(rows[:4]) + ' ...')


# ---- 2 ---------------------------------------------------------------------------------------------
def t2_driven(a):
    # the HARD drive (vtx_soft=False): this section tests the 05/10 rule for the web app's imposed drive; the
    # port's default became the soft drive on 09/10/2026 (decisions_tests_MIRROR.py section 3, with the bump flag)
    kw = dict(N=257, shape='n6', rot=7 * D, src='vtx', vtx_drive=True, freq=0.03125, vtx_soft=False)
    m0 = mc.Membrane(mc.Params(**kw))                                  # the port default: from rest
    u0 = m0.field()
    note = m0.source_note()
    ok('2a driven vertices start from rest: seed all zero, peak 0; clip fraction still computed (the web app\'s), '
       'not shown', float(np.abs(u0).max()) == 0.0 and m0.amax == 0.0 and m0.clip_frac is not None
       and note['clip'] is None and note['driven_rest'] and not m0.vtx_bumps(),
       f'{len(m0.vtx)} vertices, clip_frac {m0.clip_frac:.2e}, note lines {len(note["lines"])}')
    mb = mc.Membrane(mc.Params(vtx_drive_bump=True, **kw))            # the web app
    mu = mc.Membrane(mc.Params(**{**kw, 'vtx_drive': False}))          # undriven vertices
    same = np.array_equal(mb.field(), mu.field())
    mb.step(1)
    u1 = mb.level(1)
    deps = []
    for (i, j, *_r) in mb.vtx:
        nb = 0.25 * (u1[i - 1, j] + u1[i + 1, j] + u1[i, j - 1] + u1[i, j + 1])
        deps.append(nb - u1[i, j])
    ok('2b vtx_drive_bump=True: the seed equals the undriven vertex seed exactly (the web app); after step 1 '
       'the hard drive dents every bump centre to drive(0) = 0', same and min(deps) > 0.8
       and all(u1[v[0], v[1]] == 0.0 for v in mb.vtx), f'dent depth (neighbour mean - centre) {min(deps):.4f}..{max(deps):.4f}')
    # the diagnosis' variant 'lam16_noseed' (s4b_driven_vertices_MIRROR.py): the levels zeroed after construction
    steps = 150 if a.quick else 600
    mz = mc.Membrane(mc.Params(vtx_drive_bump=True, **kw))
    z = np.zeros((mz.N, mz.N))
    mz.set_levels(z, z)
    mz.amax = 0.0
    m0.step(steps)
    mz.step(steps)
    eq = np.array_equal(m0.field(), mz.field()) and m0.hist['pA'] == mz.hist['pA'] and m0.hist['pB'] == mz.hist['pB']
    ok(f'2c the default run == the diagnosis\' "seeded bump removed" variant, bit for bit ({steps} steps, '
       'field and probe histories)', eq, f'max|u| {float(np.abs(m0.field()).max()):.4g}')


# ---- 3 ---------------------------------------------------------------------------------------------
def t3_note(a):
    cfgs = [('sq5 n3 rot 7 vtx radial', dict(shape='n3', rot=7 * D, vtx_inset='radial')),
            ('sq5 n3 rot 7 vtx wall', dict(shape='n3', rot=7 * D)),
            ('sq9 n5 vtx wall', dict(shape='n5', stencil=9)),
            ('tri n6 rot 30 vtx radial', dict(lattice='tri', shape='n6', rot=30 * D, vtx_inset='radial')),
            ('tri rhomb rot 7 vtx wall', dict(lattice='tri', shape='rhomb', rot=7 * D)),
            ('sq5 full square vtx radial', dict(shape='square', vtx_inset='radial')),
            ('sq5 full square pulse off-centre', dict(shape='square', src='pulse', sx=0.93, sy=-0.4)),
            ('sq5 circle impulse', dict(shape='circle', rad=0.8, src='impulse', sx=0.5)),
            ('sq5 n6 cont', dict(shape='n6', src='cont', sx=-0.6, sy=0.2)),
            ('tri array torus pulse', dict(lattice='tri', shape='array', bc='periodic', src='pulse', sx=0.7, sy=-0.9)),
            ('tri array rigid ring vtx', dict(lattice='tri', shape='array', bc='neumann')),
            ('sq5 plate vtx radial', dict(shape='unlicensed', vtx_inset='radial')),
            ('tri plate pulse', dict(lattice='tri', shape='unlicensed', rad=0.89, src='pulse', sy=-0.9))]
    bad, shown = [], []
    for name, kw in cfgs:
        m = mk(N=257, **kw)
        c_led = lg._clear(m, lg._geo(m))
        n = m.source_note()
        if n['clearance'] is None or abs(n['clearance'] - c_led) > 1e-9 * max(1.0, c_led):
            bad.append(f'{name}: note {n["clearance"]} vs ledger {c_led}')
        shown.append(f'{name} {n["clearance"]:.2f}')
    # with solids: a disk beside a vertex lowers the clearance; the ledger agrees
    m = mk(N=257, shape='n6', rot=7 * D)
    before = m.source_note()['clearance']
    v = m.vtx[0]
    x, y = so.cell_to_grid(m, v[0], v[1])
    rr = math.hypot(x, y)
    g = so.cells_to_grid(m, 5.0)                     # a 1.5-cell disk 5 cells inward of the vertex
    sc = m.ensure_scene()
    sc.add_circle(x * (1 - g / rr), y * (1 - g / rr), so.cells_to_grid(m, 1.5))
    m.update_solids(None)
    after = m.source_note()
    c_led = lg._clear(m, lg._geo(m))
    if not (after['clearance'] < before and abs(after['clearance'] - c_led) <= 1e-9 * max(1.0, c_led)):
        bad.append(f'solids: {before} -> {after["clearance"]} vs ledger {c_led}')
    shown.append(f'n6 + disk {before:.2f} -> {after["clearance"]:.2f}')
    ok(f'3a clearance == the ledger\'s CLEAR (crosscheck_ledger compares that with the web app\'s) on '
       f'{len(cfgs) + 1} configurations, solids included', not bad, '; '.join(bad) if bad else ', '.join(shown[:6]) + ' ...')

    # the lines: the web app's texts
    m = mk(shape='n3', rot=7 * D, vtx_inset='radial')
    n = m.source_note()
    L = [t for t, _ in n['lines']]
    hint = [t for t, lv in n['lines'] if 'try vertex inset: wall' in t and lv == 'warn']
    ok('3b radial inset: amber clip line with the hint "the cut is a sharp edge the grid cannot resolve; try vertex '
       'inset: wall"; clearance in amber', bool(hint) and 'the cut is a sharp edge the grid cannot resolve; try vertex '
       'inset: wall' in hint[0] and n['clearance_ok'] is False and n['lines'][0][1] == 'warn', ' / '.join(L))
    m = mk(shape='n3', rot=7 * D)
    n = m.source_note()
    ok('3c wall inset: clip line in the normal colour, no hint; the clearance line ok (>= 2.9 sigma) or info, '
       'never amber with nothing cut',
       n['clip'] is not None and not n['clip_warn'] and not any('try' in t for t, _ in n['lines'])
       and n['lines'][0][1] == ('ok' if n['clearance_ok'] else 'info'),
       ' / '.join(t for t, _ in n['lines']))
    m = mk(shape='square', src='pulse', sx=0.98, sy=0.0)
    n = m.source_note()
    ok('3d a pulse cut by the wall: the hint without the vertex-inset advice', n['clip_warn'] and
       any('cannot resolve' in t and 'try' not in t for t, _ in n['lines']), ' / '.join(t for t, _ in n['lines']))
    m = mc.Membrane(mc.Params(N=41, shape='n3', src='vtx'))
    n = m.source_note()
    ok('3e the 3-sigma inset capped on a tiny grid (N 41 triangle): capped count and the amber line',
       n['capped'] == m.vtx_capped > 0 and any('capped at' in t and lv == 'warn' for t, lv in n['lines']),
       f'capped {n["capped"]}: ' + ' / '.join(t for t, _ in n['lines']))
    m = mk(shape='circle')
    n = m.source_note()
    ok('3f the circle: no vertex sources, the web app\'s advice', n['n_sources'] == 0 and n['clearance'] is None
       and n['clip'] is None and any('no vertices' in t for t, _ in n['lines']), ' / '.join(t for t, _ in n['lines']))
    m = mk(shape='n6', src='vtx', vtx_drive=True)
    n = m.source_note()
    ok('3g driven vertices from rest: clearance in cells only, the rest line, no clip line',
       n['driven_rest'] and n['clip'] is None and n['clearance_sigma'] is None and
       any('start from rest' in t for t, _ in n['lines']), ' / '.join(t for t, _ in n['lines']))
    m = mk(shape='n6', src='mode')
    n = m.source_note()
    ok('3h a source with no point (mode): no clearance, no lines', n['clearance'] is None and not n['lines'], f'{n}')

    # the cache and the cost
    N = 2049 if a.quick else 4097
    m = mk(N=N, shape='n6', rot=7 * D)
    t0 = time.perf_counter()
    n1 = m.source_note()
    t1 = time.perf_counter()
    for _ in range(100):
        n2 = m.source_note()
    t2 = time.perf_counter()
    m.p.sigma = 4.0
    m.reseed()
    n3 = m.source_note()
    mp = mk(N=N, shape='n6', rot=7 * D, src='pulse')
    t3 = time.perf_counter()
    mp.source_note()
    t4 = time.perf_counter()
    first, per_us, centred = (t1 - t0) * 1e3, (t2 - t1) / 100 * 1e6, (t4 - t3) * 1e3
    ok(f'3i cached: the same dict until a reseed or mask change; cost at N = {N}: vertex sources (first call) and '
       'a repeat call', n2 is n1 and n3 is not n1 and abs(n3['clearance_sigma'] * 4.0 - n3['clearance']) < 1e-9
       and per_us < 50 and first < 200,
       f'vertex first call {first:.1f} ms, repeat {per_us:.1f} us; centred pulse first call {centred:.0f} ms')


# ---- 4 ---------------------------------------------------------------------------------------------
def t4_warn(a):
    # freq 0.12 set explicitly: the default became 0.03125 (lambda 16) on 09/10/2026
    cases = [('cont, freq 0.12 Co 0.5', dict(src='cont', freq=0.12), 1),
             ('slit, freq 0.12', dict(src='slit', freq=0.12), 1),
             ('driven vertices, freq 0.12', dict(src='vtx', vtx_drive=True, freq=0.12), 1),
             ('driven vertices (hard), freq 0.12', dict(src='vtx', vtx_drive=True, vtx_soft=False, freq=0.12), 1),
             ('cont at the default freq (0.03125, lambda 16)', dict(src='cont'), 0),
             ('undriven vertices', dict(src='vtx'), 0),
             ('pulse', dict(src='pulse'), 0),
             ('beam (its own check)', dict(src='beam'), 0),
             ('cont, freq 0.03125 (lambda 16)', dict(src='cont', freq=0.03125), 0),
             ('cont, freq 0.0625 (lambda 8, the limit)', dict(src='cont', freq=0.0625), 0),
             ('cont, freq 0.07 (lambda 7.1)', dict(src='cont', freq=0.07), 1),
             ('cont, domain mode, lam_f small (scale_warnings has it)', dict(src='cont', scale='domain', lam_f=0.02), 0)]
    bad, txt = [], ''
    for name, kw, want in cases:
        m = mk(N=161, **kw)
        w = m.drive_warnings()
        if len(w) != want:
            bad.append(f'{name}: {w}')
        if want and not txt:
            txt = w[0]
    m = mk(N=161, src='cont', scale='domain', lam_f=0.02)
    sw = m.scale_warnings()
    ok('4 drive_warnings: hard drives under 8 cells per wavelength in cells mode; nothing for soft or point '
       'sources; domain mode left to scale_warnings', not bad and len(sw) == 1 and txt.startswith('continuous source: lambda 4.2 cells < 8'),
       '; '.join(bad) if bad else f'{txt!r}; domain mode scale_warnings: {sw}')


# ---- 5 ---------------------------------------------------------------------------------------------
def continuum_profile(sigma, ct, r):
    """The exact (continuum) solution of a gaussian exp(-r^2 / 2 sigma^2) released at rest, at radius r after
    travelling ct cells: u = int_0^inf sigma^2 exp(-k^2 sigma^2 / 2) cos(k ct) J0(k r) k dk (the 2D Hankel
    transform; free space -- on the torus the images are > 0.3 N away and their gaussian tails ~e^-500)."""
    from scipy.special import j0
    k = np.linspace(0.0, 9.0 / sigma, 24001)
    wk = sigma * sigma * np.exp(-0.5 * (k * sigma) ** 2) * np.cos(k * ct) * k
    out = np.empty(len(r))
    for a0 in range(0, len(r), 200):
        rr = r[a0:a0 + 200]
        out[a0:a0 + 200] = np.trapezoid(wk[None, :] * j0(np.outer(rr, k)), k, axis=1)
    return out


def measured_lag(lat, st, Co, sigma, N, crossings, dirs):
    """Run a centred gaussian on the torus (square lattice: the full square, periodic; triangular: the
    rhombic torus) for about `crossings` x r_in of travel (kept under 0.4 N, so the images stay far), and
    fit each ray's profile against the EXACT continuum solution (continuum_profile) at t = (n + 1/2) Co
    -- a run started at rest (u_prev = u) is the continuum solution half a step later. The radial shift s
    with lattice(r) = exact(r + s) that fits best in a +-4 sigma window around the front is the lag.
    Returns ({direction deg: lag per crossing}, crossings travelled, the Membrane)."""
    kw = dict(N=N, Co=Co, src='pulse', sigma=sigma, bc='periodic', lattice=lat, stencil=st)
    if lat == 'tri':
        kw['shape'] = 'array'
    m = mc.Membrane(mc.Params(**kw))
    m.record = False
    r_in = m.inner_R()
    n = int(round(crossings * r_in / Co))
    m.step(n)
    u = m.field().astype(np.float64)
    ct = (n + 0.5) * Co
    cr = ct / r_in
    rf = np.arange(ct - 8 * sigma - 10, ct + 8 * sigma + 10, 0.02)
    uf = continuum_profile(sigma, ct, rf)
    i0, j0 = m.src[:2]
    out = {}
    for deg, (di, dj) in dirs.items():
        step = math.hypot(di + 0.5 * dj, dj * mc.S3H) if lat == 'tri' else math.hypot(di, dj)
        ks = np.arange(1, int((ct + 6 * sigma) / step) + 2)
        prof = u[(i0 + ks * di) % N, (j0 + ks * dj) % N]
        r = ks * step
        pk = int(np.argmax(np.abs(prof)))
        win = (r > r[pk] - 4 * sigma) & (r < r[pk] + 4 * sigma)
        rw, target = r[win], prof[win]

        def miss(sh):
            return float(((target - np.interp(rw + sh, rf, uf)) ** 2).sum())
        lo, hi = -5.0, 5.0
        for _ in range(4):
            ss = np.linspace(lo, hi, 41)
            e = [miss(x) for x in ss]
            kk = int(np.argmin(e))
            lo, hi = ss[max(0, kk - 1)], ss[min(len(ss) - 1, kk + 1)]
        out[deg] = 0.5 * (lo + hi) / cr
    return out, cr, m


def t5_phase(a):
    cases = [('sq', 5, 0.5, 3.0, {0: (1, 0), 45: (1, 1), 26.57: (2, 1)}),
             ('sq', 5, 0.69, 3.0, {0: (1, 0), 45: (1, 1)}),
             ('sq', 9, 0.5, 3.0, {0: (1, 0), 45: (1, 1)}),
             ('tri', 5, 0.5, 3.0, {0: (1, 0), 30: (1, 1), 60: (0, 1), 90: (-1, 2)}),
             ('tri', 5, 0.8, 3.0, {0: (1, 0), 30: (1, 1)}),
             ('sq', 5, 0.5, 6.0, {0: (1, 0), 45: (1, 1)})]
    if a.quick:
        cases = cases[:1] + cases[3:4]
    N, cr = (257, 0.7) if a.quick else (1025, 0.7)
    rows, bad = [], []
    for lat, st, Co, sig, dirs in cases:
        meas, cr, m = measured_lag(lat, st, Co, sig, N, cr, dirs)
        for deg, lag in meas.items():
            pred = m.phase_lag(theta_deg=deg)['lag']
            rel = abs(lag - pred) / max(abs(pred), 1e-9)
            rows.append(f'{lat}{st if lat == "sq" else ""} Co {Co} sigma {sig:g} {deg:g} deg: measured {lag:.3f} '
                        f'predicted {pred:.3f} cells/crossing ({100 * rel:.0f}%)')
            if rel > 0.15 and abs(lag - pred) > 0.02:
                bad.append(rows[-1])
    for r in rows:
        print('    ' + r)
    # the readout at the app's defaults, its cost and its cache
    m = mc.Membrane(mc.Params(N=513, src='pulse'))
    t0 = time.perf_counter()
    q = m.phase_lag()
    t1 = time.perf_counter()
    q2 = m.phase_lag()
    t2 = time.perf_counter()
    m.p.sigma = 6.0
    q6 = m.phase_lag()
    print(f'    app defaults (sq5 N 513 sigma 3 Co 0.5, r_in {q["r_in"]:.0f}): lag {q["lag_min"]:.2f}-{q["lag_max"]:.2f} cells '
          f'per crossing (dirs {q["dir_min"]:g} / {q["dir_max"]:g} deg), {q["crossings_to_sigma"]:.1f} crossings to one '
          f'sigma; sigma 6: {q6["lag_max"]:.3f} ({q["lag_max"] / q6["lag_max"]:.2f}x less); first call '
          f'{(t1 - t0) * 1e3:.1f} ms, cached {(t2 - t1) * 1e6:.1f} us')
    if not (q2 is q and (t1 - t0) < 0.1 and q6 is not q and 3.5 < q['lag_max'] / q6['lag_max'] < 4.5
            and any('domain mode / larger N / Co nearer the limit' in t for t, _ in q['lines'])):
        bad.append('readout: cache / cost / sigma scaling / hint')
    ok(f'5 phase_lag() vs the lag measured on the lattice (N {N}, {cr:.2f} crossings, {len(rows)} directions): '
       'within 15% (or 0.02 cells)', not bad, '; '.join(bad) if bad else f'{len(rows)} directions agree')


TESTS = {1: t1_defaults, 2: t2_driven, 3: t3_note, 4: t4_warn, 5: t5_phase}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='cuda')
    ap.add_argument('--f32', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    f32 = a.f32 or a.arch == 'vulkan'
    mc.init_taichi(a.arch, fp64=not f32, quiet=True)
    only = {int(x) for x in a.only.split(',') if x} if a.only else set(TESTS)
    t0 = time.perf_counter()
    for k, fn in TESTS.items():
        if k in only:
            fn(a)
    print(f'\n{"ALL PASS" if not FAILS else str(len(FAILS)) + " FAIL: " + "; ".join(FAILS)} '
          f'({a.arch} {"f32" if f32 else "f64"}, {time.perf_counter() - t0:.0f} s)')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
