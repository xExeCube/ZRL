"""Tests of the round-5 decisions (09/10/2026, the user's decisions on three items left open at v1.0;
NotesCLAUDE/v1_decisions_MIRROR.md).

  1  sq9 + Neumann + a shape clipped by the array (a port deviation since 09/10/2026): the exterior ring cell
     of a pinned edge blocks the sq9 diagonal into the ring like a pinned ring solid (_word's ghost rule).
     1a every enumerated configuration (shapes clipped at several radii and rotations, both edges, with and
        without solids) has an exactly symmetric operator; the v1.0 core's is asymmetric on the pinned edge;
        every configuration the v1.0 core had symmetric keeps its operator entry for entry
     1b the step map's spectral radius <= 1 (dense, N 25, Co 0.5 and 0.865)
     1c the exact leapfrog energy over 20,000 steps
     1d the review's reproducer (N 41, Co 0.865, 200,000 steps) stays bounded (v1.0: 6.7e21)
     1e every unaffected configuration is bit-identical to the v1.0 snapshot (fields, peak / probe histories,
        the walls state); the affected one runs the walls kernel and differs
  2  the default drive frequency 0.03125 (lambda 16 cells at Co 0.5): the value, no drive warning at the
     defaults, the default only a value (bit-identical to v1.0 with the same freq set; freq=0.12 gives v1.0's
     default run), domain mode's lam_f, the band limit
  3  soft driven vertices (Params.vtx_soft, default True) and the hard toggle
     3a defaults and the readouts; 3b the hard path bit-identical to v1.0; 3c transparency (zero drive: soft
     cells change nothing, hard ones reflect); 3d the soft / hard amplitude ratio (free space and the
     defaults); 3e no spurious growth; 3f the bump flag with soft vertices; 3g the harmonics and gains

    python decisions_tests_MIRROR.py [--arch cuda|vulkan|cpu] [--f32] [--quick] [--only 1,2,3] [--long]

Never bare pytest. --long runs the full 200,000 steps in 1d (the default runs 200,000 on f64, 40,000 in f32).
*Generated from scratch -- Claude Opus 5.5 -- 09/10/2026*
"""
import argparse
import importlib.util
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import membrane_core_MIRROR as mc          # noqa: E402

D = math.pi / 180
FAILS = []
V10_PATH = os.path.join(HERE, 'ScratchCLAUDE', 'Round5CLAUDE', 'BeforeCLAUDE', 'membrane_core_MIRROR.py')


def ok(name, cond, det=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('  | ' + det if det else ''), flush=True)
    if not cond:
        FAILS.append(name)


def acc_tol():
    import taichi as ti
    return 1e-6 if mc.ACC_DT == ti.f32 else 1e-14


def load_v10():
    """The v1.0 core (ScratchCLAUDE/Round5CLAUDE/BeforeCLAUDE, never edited), sharing this process's Taichi."""
    spec = importlib.util.spec_from_file_location('membrane_core_v10_MIRROR', V10_PATH)
    B = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    spec.loader.exec_module(B)
    sys.path[:] = saved
    B.MASK_DT, B.SQ9_CONSISTENT, B.FP64, B.ACC_DT, B.WORD_DT = mc.MASK_DT, mc.SQ9_CONSISTENT, mc.FP64, mc.ACC_DT, mc.WORD_DT
    return B


def mk(mod, **prm):
    """A Membrane of module mod with the fields that module knows (the v1.0 core has no vtx_soft)."""
    fld = mod.Params.__dataclass_fields__
    return mod.Membrane(mod.Params(**{k: v for k, v in prm.items() if k in fld}))


def hist_pack(m):
    h = m.hist
    return np.array(h['E']), np.array(h['amax'] + h['pA'] + h['pB'] + h['mode'])


def solids_map(N, kind, seed=0):
    """A deterministic type map: pinned / rigid cells ON the ring rows and 1-2 cells in, and a few blocks."""
    if kind == 'none':
        return None
    rng = np.random.default_rng(N * 7 + seed + (1 if kind == 'pinned' else 2 if kind == 'rigid' else 3))
    T = np.zeros((N, N), np.uint8)
    pick = lambda: (1 if kind == 'pinned' else 2 if kind == 'rigid' else int(rng.integers(1, 3)))
    for _ in range(max(4, N // 6)):
        side = int(rng.integers(0, 4))
        t = int(rng.integers(1, N - 1))
        d = int(rng.integers(0, 3))
        i, j = [(d, t), (N - 1 - d, t), (t, d), (t, N - 1 - d)][side]
        T[i, j] = pick()
    for _ in range(3):
        i, j = (int(v) for v in rng.integers(N // 4, 3 * N // 4, 2))
        T[i:i + 2, j:j + 2] = pick()
    return T


def ring_touch(m):
    d = m.domain_mask_np
    return bool(d[0, :].any() or d[-1, :].any() or d[:, 0].any() or d[:, -1].any())


def energy_run(m, steps, every=1000):
    L = m.operator()
    E0 = m.leapfrog_energy(L)
    worst, done = 0.0, 0
    while done < steps:
        k = min(every, steps - done)
        m.step(k)
        done += k
        worst = max(worst, abs(m.leapfrog_energy(L) - E0) / abs(E0))
    return worst, E0


def step_map_rho(m):
    L = m.operator()
    unk = m.unknowns().ravel()
    A = L[unk][:, unk].toarray()
    n = A.shape[0]
    cc = m.cc_np.ravel()[unk]
    G = np.block([[2 * np.eye(n) + cc[:, None] * A, -np.eye(n)], [np.eye(n), np.zeros((n, n))]])
    return float(np.abs(np.linalg.eigvals(G)).max())


# ---- 1 ---------------------------------------------------------------------------------------------------
NEU_SHAPES = ['circle', 'n3', 'n4', 'n5', 'n6', 'n8', 'n12', 'rhomb', 'unlicensed']


def t1_neu_clip(a, B):
    f32 = not mc.FP64
    # 1a the enumeration
    Ns = (41,) if a.quick else (41, 97)
    rads = (1.05, 1.3, 1.6) if a.quick else (1.02, 1.05, 1.2, 1.3, 1.45, 1.6, 2.0)
    rots = (0.0, 22.5) if a.quick else (0.0, 7.0, 17.3, 30.0, 45.0)
    sks = ('none', 'mixed') if a.quick else ('none', 'pinned', 'rigid', 'mixed')
    n_cfg = n_asym_now = n_asym_v10 = n_asym_v10_rigid = 0
    n_same, n_sym10, worst_now, worst_v10 = 0, 0, 0.0, 0.0
    changed_sym, t0 = [], time.perf_counter()
    for N in Ns:
        for shape in NEU_SHAPES:
            for rad in rads:
                for rot in (rots if shape != 'circle' else (0.0,)):
                    for edge in ('pinned', 'rigid'):
                        for sk in sks:
                            prm = dict(N=N, lattice='sq', stencil=9, bc='neumann', shape=shape, rad=rad, rot=rot * D,
                                       edge=edge, src='impulse', Co=0.5)
                            ms = []
                            for mod in (mc, B):
                                m = mk(mod, **prm)
                                if not ring_touch(m):
                                    break
                                T = solids_map(m.N, sk)
                                if T is not None:
                                    m.set_solid_np(T)
                                ms.append(m)
                            if len(ms) < 2:
                                continue
                            Ln, Lo = ms[0].operator(), ms[1].operator()
                            an = float(abs(Ln - Ln.T).max()) if Ln.nnz else 0.0
                            ao = float(abs(Lo - Lo.T).max()) if Lo.nnz else 0.0
                            n_cfg += 1
                            n_asym_now += an > 0
                            n_asym_v10 += ao > 0
                            n_asym_v10_rigid += (ao > 0) and edge == 'rigid'
                            worst_now, worst_v10 = max(worst_now, an), max(worst_v10, ao)
                            if ao == 0:
                                n_sym10 += 1
                                same = (Ln != Lo).nnz == 0
                                n_same += same
                                if not same:
                                    changed_sym.append(f'N{N} {shape} r{rad} rot{rot} {edge} {sk}')
    ok('1a sq9 Neumann, shapes clipped by the array: every operator exactly symmetric; the v1.0 core asymmetric on '
       'the pinned edge only; the configurations v1.0 had symmetric keep their operator entry for entry',
       n_cfg > 100 and n_asym_now == 0 and n_asym_v10 > 0 and n_asym_v10_rigid == 0 and n_same == n_sym10,
       f'{n_cfg} configs (N {Ns}, {len(NEU_SHAPES)} shapes, radii {rads}, rotations {rots} deg, pinned / rigid edge, '
       f'solids {sks}; {time.perf_counter() - t0:.0f} s): asymmetric now {n_asym_now} (max|L - L^T| {worst_now:.2g}), '
       f'v1.0 {n_asym_v10} (max {worst_v10:.3g}; rigid edge {n_asym_v10_rigid}); v1.0-symmetric {n_sym10}, operator '
       f'unchanged {n_same}' + (f'; CHANGED: {changed_sym[:4]}' if changed_sym else ''))

    # 1b spectral radius of the step map
    cases = [('circle r 1.3, pinned edge', dict(shape='circle', rad=1.3), None),
             ('octagon r 1.25, pinned edge', dict(shape='n8', rad=1.25), None),
             ('pentagon r 1.3 at 17.3 deg, pinned edge', dict(shape='n5', rad=1.3, rot=17.3 * D), None),
             ('rhombus r 1.6 at 30 deg, pinned edge', dict(shape='rhomb', rad=1.6, rot=30 * D), None),
             ('circle r 1.3, pinned edge, mixed solids', dict(shape='circle', rad=1.3), 'mixed'),
             ('circle r 1.3, rigid edge', dict(shape='circle', rad=1.3, edge='rigid'), None),
             ('n12 r 1.45, rigid edge, mixed solids', dict(shape='n12', rad=1.45, edge='rigid'), 'mixed')]
    bad, det = [], []
    for label, prm, sk in cases:
        for co in (0.5, 0.865):
            res = []
            for mod in (mc, B):
                m = mk(mod, N=25, lattice='sq', stencil=9, bc='neumann', src='impulse', Co=co, **prm)
                T = solids_map(m.N, sk) if sk else None
                if T is not None:
                    m.set_solid_np(T)
                res.append(step_map_rho(m))
            # a fully Neumann domain (rigid edge) has a zero mode: a double root at 1, 1 + ~1e-8 numerically
            if res[0] > 1 + 1e-7:
                bad.append(f'{label} Co {co}: {res[0]:.10f}')
            det.append(f'{label} Co {co}: {res[0]:.9f} (v1.0 {res[1]:.6f})')
    ok('1b the step map\'s spectral radius <= 1 (dense, N 25; within 1e-7: the rigid edge\'s zero mode is a double '
       'root)', not bad, '; '.join(bad) if bad else '; '.join(det))

    # 1c the exact energy
    steps = 4000 if a.quick else 20000
    tol_e = 1e-4 if f32 else 1e-12
    cases = [('circle r 1.3, pinned edge, Co 0.865', dict(shape='circle', rad=1.3, Co=0.865), None),
             ('octagon r 1.25, pinned edge, Co 0.865', dict(shape='n8', rad=1.25, Co=0.865), None),
             ('pentagon r 1.3 at 17.3 deg, mixed solids, Co 0.8', dict(shape='n5', rad=1.3, rot=17.3 * D, Co=0.8), 'mixed'),
             ('Unlicensed plate r 1.3 (pinned lettering), Co 0.6', dict(shape='unlicensed', rad=1.3, Co=0.6), None),
             ('circle r 1.3, rigid edge, Co 0.865', dict(shape='circle', rad=1.3, Co=0.865, edge='rigid'), None)]
    bad, det = [], []
    for label, prm, sk in cases:
        res = []
        for mod in (mc, B):
            m = mk(mod, N=61, lattice='sq', stencil=9, bc='neumann', src='pulse', sigma=3, sx=0.5, sy=0.3, **prm)
            m.record = False
            T = solids_map(m.N, sk) if sk else None
            if T is not None:
                m.set_solid_np(T)
                m.reseed()
            # E0 after two steps: the seed's gaussian tail sits on the ring cells, which hold a ghost (or 0) only
            # from step 1 (inherited: the seed is masked, not ring-clamped); measured from step 0 every sq9
            # configuration here, the full square too, reads ~1e-9 (dec_energy_co_probe_MIRROR.py), from step 2 ~1e-15
            m.step(2)
            drift, _ = energy_run(m, steps)
            res.append((drift, float(np.abs(m.field()).max())))
        if res[0][0] > tol_e:
            bad.append(f'{label}: {res[0][0]:.2e}')
        det.append(f'{label}: {res[0][0]:.1e} (v1.0 {res[1][0]:.1e}, max|u| {res[1][1]:.3g})')
    ok(f'1c the exact leapfrog energy over {steps} steps (N 61, tol {tol_e:.0e})', not bad,
       '; '.join(bad) if bad else '; '.join(det))

    # 1d the review's reproducer
    total = 200000 if (a.long or not f32) else 40000
    if a.quick and not a.long:
        total = 40000
    out = []
    good = True
    for shape, rad in (('circle', 1.3), ('n8', 1.25)):
        m = mk(mc, N=41, lattice='sq', stencil=9, shape=shape, rad=rad, bc='neumann', src='pulse', sigma=3, Co=0.865)
        m.record = False
        u0 = float(np.abs(m.field()).max())
        mx = []
        for _ in range(10):
            m.step(total // 10)
            mx.append(float(np.abs(m.field()).max()))
        good &= max(mx) <= 2 * u0 and np.isfinite(mx).all() and m.walls_active()
        out.append(f'{shape} r {rad}: max|u| {u0:.2f} at 0, every {total // 10} steps ' + ' '.join(f'{v:.3g}' for v in mx))
    ok(f'1d the review\'s reproducer (rv_neu_clip_growth: N 41, Co 0.865, pinned edge) stays bounded over {total} steps '
       '(v1.0: 6.7e21 at 200,000 for the circle)', good, '; '.join(out))

    # 1e parity with v1.0 where nothing is affected, and the affected configuration differs
    steps = 150 if a.quick else 400
    objs = solids_map(161, 'mixed', seed=5)
    cases = [('sq9 Neumann full square, corner pulse', dict(stencil=9, bc='neumann', sx=-0.8, sy=-0.8, sigma=2), None),
             ('sq9 Neumann circle r 0.8 (inside the array)', dict(stencil=9, bc='neumann', shape='circle', rad=0.8, sx=0.3), None),
             ('sq9 Neumann pentagon r 1 at 17.3 deg', dict(stencil=9, bc='neumann', shape='n5', rot=17.3 * D, sx=0.3), None),
             ('sq9 Dirichlet circle r 1.3', dict(stencil=9, shape='circle', rad=1.3, sx=0.5), None),
             ('sq9 Mur circle r 1.3', dict(stencil=9, bc='mur', shape='circle', rad=1.3, sx=0.5), None),
             ('sq9 periodic circle r 1.3', dict(stencil=9, bc='periodic', shape='circle', rad=1.3, sx=0.7), None),
             ('sq5 Neumann circle r 1.3', dict(bc='neumann', shape='circle', rad=1.3, sx=0.6), None),
             ('sq5 Neumann octagon r 1.25 + solids', dict(bc='neumann', shape='n8', rad=1.25, sx=0.6), objs),
             ('sq9 Neumann circle r 1.3, rigid edge', dict(stencil=9, bc='neumann', shape='circle', rad=1.3, edge='rigid', sx=0.6), None),
             ('sq9 Neumann circle r 1.3, rigid edge + solids', dict(stencil=9, bc='neumann', shape='circle', rad=1.3, edge='rigid', sx=0.6), objs),
             ('sq9 Neumann full square + ring solids', dict(stencil=9, bc='neumann', sx=0.6), objs),
             ('tri circle r 1.3, bc neumann', dict(lattice='tri', bc='neumann', shape='circle', rad=1.3, sx=0.5), None),
             ('sq9 Neumann square r 0.998 (the full square)', dict(stencil=9, bc='neumann', shape='square', rad=0.998, sx=0.5), None)]
    nbad, det = 0, []
    for label, prm, T in cases:
        res = []
        for mod in (mc, B):
            m = mk(mod, **{'N': 161, 'src': 'pulse', 'sigma': 3, **prm})
            if T is not None:
                m.set_solid_np(T)
                m.reseed()
            m.step(steps)
            E, P = hist_pack(m)
            res.append((m.field(), E, P, m.walls_active(), m.W is not None))
        (u1, E1, P1, w1, k1), (u2, E2, P2, w2, k2) = res
        dE = float(np.abs(E1 - E2).max() / max(np.abs(E2).max(), 1e-300)) if len(E1) == len(E2) else 1.0
        good = np.array_equal(u1, u2) and np.array_equal(P1, P2) and dE <= acc_tol() and w1 == w2
        nbad += not good
        if not good:
            det.append(f'{label}: field {"=" if np.array_equal(u1, u2) else "DIFF"}, hist '
                       f'{"=" if np.array_equal(P1, P2) else "DIFF"}, dE {dE:.1e}, walls {w1}/{w2}')
    ok(f'1e unaffected configurations bit-identical to the v1.0 core ({len(cases)} configs, N 161, {steps} steps: '
       'fields, peak / probe histories, walls state; the energy history to the atomics\' rounding)', nbad == 0,
       '; '.join(det) if det else f'{len(cases)}/{len(cases)} bit-identical')
    res = []
    for mod in (mc, B):
        m = mk(mod, N=161, stencil=9, bc='neumann', shape='circle', rad=1.3, src='pulse', sigma=3, sx=0.85, sy=0.3)
        m.step(steps)
        res.append((m.field(), m.walls_active(), m.n_wall_cells if mod is mc else 0))
    d = float(np.abs(res[0][0] - res[1][0]).max() / np.abs(res[1][0]).max())
    ok('1e the affected configuration (sq9 Neumann circle r 1.3, pinned edge, no solids) runs the walls kernel and '
       'differs from v1.0', res[0][1] and not res[1][1] and d > 1e-6,
       f'{res[0][2]} cells with a blocked link; max|u - u_v10| / max|u| = {d:.2e} after {steps} steps')


# ---- 2 ---------------------------------------------------------------------------------------------------
def t2_freq(a, B):
    p = mc.Params()
    ok('2a Params.freq default 0.03125: lambda = Co / freq = 16 cells at the default Co 0.5; the web app\'s 0.12 is '
       '4.17 cells', p.freq == 0.03125 and p.Co / p.freq == 16.0 and p.Co == 0.5,
       f'freq {p.freq}, Co {p.Co}, lambda {p.Co / p.freq} cells')
    bad = []
    for src, kw in (('cont', {}), ('slit', {}), ('vtx', dict(vtx_drive=True)), ('vtx', dict(vtx_drive=True, vtx_soft=False))):
        m = mc.Membrane(mc.Params(src=src, **kw))
        if m.drive_warnings():
            bad.append(f'{src} {kw}: {m.drive_warnings()}')
        m2 = mc.Membrane(mc.Params(src=src, freq=0.12, **kw))
        if len(m2.drive_warnings()) != 1:
            bad.append(f'{src} {kw} at 0.12: {m2.drive_warnings()}')
    ok('2a no drive-wavelength warning at the defaults (cont, slit, soft and hard driven vertices); at freq 0.12 it '
       'is back', not bad, '; '.join(bad) if bad else 'none at 0.03125; one each at 0.12')
    m = mc.Membrane(mc.Params(src='cont', wave='saw'))
    ok('2b the band limit at the default: k_max = floor(Co / (8 freq)) = 2 harmonics of the saw', m.k_max() == 2
       and [h for h, _ in m.harm] == [1, 2], f'k_max {m.k_max()}, harmonics {[h for h, _ in m.harm]}')
    # the default is a value: the same runs as v1.0 with that value set
    steps = 150 if a.quick else 400
    nbad, det = 0, []
    cases = [('cont', dict(src='cont', sx=0.2)), ('slit square wave', dict(src='slit', wave='square')),
             ('beam', dict(src='beam', beam_angle=30.0)), ('tri cont saw', dict(lattice='tri', shape='n6', src='cont', wave='saw')),
             ('sq9 hard vertices', dict(stencil=9, shape='n5', src='vtx', vtx_drive=True, vtx_soft=False))]
    for label, prm in cases:
        for f_new, f_old in ((None, 0.03125), (0.12, None)):
            pn = dict(N=161, **prm)
            if f_new is not None:
                pn['freq'] = f_new
            po = dict(N=161, **prm)
            if f_old is not None:
                po['freq'] = f_old
            m1, m2 = mk(mc, **pn), mk(B, **po)
            m1.step(steps)
            m2.step(steps)
            E1, P1 = hist_pack(m1)
            E2, P2 = hist_pack(m2)
            good = np.array_equal(m1.field(), m2.field()) and np.array_equal(P1, P2)
            nbad += not good
            if not good:
                det.append(f'{label} (new {f_new}, v1.0 {f_old}): DIFFERS')
    ok('2c the default is only a value: the new default run equals v1.0 with freq=0.03125 set, and freq=0.12 set '
       'equals v1.0\'s default run, bit for bit (cont, slit, beam, tri, sq9 hard vertices)', nbad == 0,
       '; '.join(det) if det else f'{2 * len(cases)}/{2 * len(cases)} bit-identical, {steps} steps')
    m = mc.Membrane(mc.Params(N=257, src='cont', scale='domain'))
    lam = m.p.lam_f * m.inner_R()
    ok('2d domain mode takes lam_f from the default: lambda = lam_f r_in = 16 cells at the starting N; freq = Co / lambda',
       abs(lam - 16.0) < 1e-9 and abs(m.p.freq - 0.03125) < 1e-12, f'lam_f {m.p.lam_f:.6f}, r_in {m.inner_R()}, '
       f'lambda {lam:.6f} cells, freq {m.p.freq}')


# ---- 3 ---------------------------------------------------------------------------------------------------
LATS = (('sq5', dict(lattice='sq', stencil=5)), ('sq9', dict(lattice='sq', stencil=9)), ('tri', dict(lattice='tri')))


def one_cell(m, i, j):
    """Drive one cell only (a lone vertex), for the free-space measurement."""
    import taichi as ti
    m.drive_cells = [(i, j)]
    m.nsrc = 1
    m.sc = ti.ndarray(dtype=ti.i32, shape=(1, 2))
    m.sc.from_numpy(np.array([[i, j]], dtype=np.int32))
    m._soft = (None, None)


def set_probes(m, pts):
    """The recorder's three probe cells (pA, pB, mode) -> pts: their values each step without a host read."""
    flat = [v for pt in pts for v in pt]
    m.prb.from_numpy(np.array(flat, dtype=np.int32))


def lockin(series, freq, t0):
    t = t0 + np.arange(len(series))
    z = np.sum(np.asarray(series) * np.exp(-2j * math.pi * freq * t)) * 2 / len(series)
    return abs(z), math.degrees(math.atan2(z.imag, z.real))


def probe_run(m, pts, steps):
    """Run steps with the recorder on pts (3 cells); return the three series (pA, pB, mode)."""
    m.record = True
    set_probes(m, pts)
    h0 = len(m.hist['pA'])
    m.step(steps)
    return [np.array(m.hist['pA'][h0:]), np.array(m.hist['pB'][h0:]),
            np.array(m.hist['mode'][h0:]) if m.p.src == 'mode' else None]


def t3_soft(a, B):
    f32 = not mc.FP64
    p = mc.Params()
    m = mc.Membrane(mc.Params(src='vtx', vtx_drive=True))
    n = m.source_note()
    ok('3a defaults: vtx_soft True (VTX_DRIVES soft / hard); a default driven-vertex run takes the soft path; the source '
       'note says "driven soft" and drive = soft', p.vtx_soft is True and mc.VTX_DRIVES == ['soft', 'hard']
       and m.vtx_soft_on() and n['drive'] == 'soft' and 'driven soft' in n['lines'][0][0],
       f'{n["lines"][0][0]!r}, drive {n["drive"]!r}')
    mh = mc.Membrane(mc.Params(src='vtx', vtx_drive=True, vtx_soft=False))
    nh = mh.source_note()
    mu = mc.Membrane(mc.Params(src='vtx'))
    ok('3a the hard toggle: vtx_soft=False runs the imposed drive; undriven vertices have drive None',
       not mh.vtx_soft_on() and nh['drive'] == 'hard' and 'driven hard' in nh['lines'][0][0]
       and mu.source_note()['drive'] is None and not mu.vtx_soft_on(), f'{nh["lines"][0][0]!r}')

    # 3b the hard path = v1.0
    steps = 150 if a.quick else 400
    cases = [('sq5 octagon r 1.4 saw (crosscheck_js)', dict(shape='n8', rot=22.5 * D, rad=1.4, vtx_inset='wall', wave='saw',
                                                            freq=0.03, damp=0.002)),
             ('sq9 pentagon sine', dict(stencil=9, shape='n5', rot=17 * D, freq=0.05)),
             ('tri triangle radial + bumps (crosscheck_js)', dict(lattice='tri', shape='n3', rot=30 * D, wave='tri', freq=0.05,
                                                                  vtx_inset='radial', vtx_drive_bump=True)),
             ('sq5 square, freq 0.03125 (the new default, set for both cores)', dict(shape='square', rad=0.9, freq=0.03125))]
    nbad, det = 0, []
    for label, prm in cases:
        res = []
        for mod in (mc, B):
            m = mk(mod, N=161, src='vtx', vtx_drive=True, vtx_soft=False, **prm)
            m.step(steps)
            res.append((m.field(),) + hist_pack(m))
        (u1, E1, P1), (u2, E2, P2) = res
        dE = float(np.abs(E1 - E2).max() / max(np.abs(E2).max(), 1e-300))
        good = np.array_equal(u1, u2) and np.array_equal(P1, P2) and dE <= acc_tol()
        nbad += not good
        if not good:
            det.append(f'{label}: DIFFERS (dE {dE:.1e})')
    ok('3b hard driven vertices (vtx_soft=False) bit-identical to the v1.0 core (fields, histories)', nbad == 0,
       '; '.join(det) if det else f'{len(cases)}/{len(cases)}, N 161, {steps} steps')

    # 3c transparency
    steps = 300 if a.quick else 420
    det, good = [], True
    for kind, lat in LATS:
        res = {}
        for mode in ('none', 'soft', 'hard'):
            prm = dict(N=161, src='vtx', vtx_drive=(mode != 'none'), vtx_soft=(mode == 'soft'), shape='square',
                       rad=0.9, **lat)
            m = mc.Membrane(mc.Params(**prm))
            m.record = False
            X, Y = m._grid_xy()
            g = np.exp(-(X * X + Y * Y) / (2 * 3.0 ** 2))
            g[m.mask_np == 0] = 0.0
            m.set_levels(g, g)
            m.harm = [(1, 0.0)]                  # zero drive amplitude
            m.step(steps)
            res[mode] = m.field()
            if mode == 'soft':
                on = m.vtx_soft_on()
        pk = float(np.abs(res['none']).max())
        ds = float(np.abs(res['soft'] - res['none']).max()) / pk
        dh = float(np.abs(res['hard'] - res['none']).max()) / pk
        g_ = on and ds == 0.0 and dh > 1e-2
        good &= g_
        det.append(f'{kind}: soft {ds:.1e}, hard {dh:.2e}')
    ok(f'3c transparency: a pulse crossing driven vertices at ZERO drive amplitude: soft cells change nothing (bit '
       f'for bit), hard ones reflect it (max|u - u_undriven| / peak after {steps} steps)', good, '; '.join(det))

    # 3d the amplitude ratio, free space: one cell at the centre, lambda 16 (and 8), r = 3 lambda
    lams = (16.0,) if a.quick else (16.0, 8.0)
    det, good = [], True
    N = 1025
    for lam in lams:
        per = lam / 0.5
        r = int(round(3 * lam))
        t0 = int(math.ceil((r / 0.5 + 6 * per) / per) * per)
        t1 = t0 + int(per * int((2 * 500 - 2 * r) / 0.5 / per - 7))
        for kind, lat in LATS:
            amps = {}
            for soft in (False, True):
                shape = 'array' if kind == 'tri' else 'square'
                m = mc.Membrane(mc.Params(N=N, Co=0.5, src='vtx', vtx_drive=True, vtx_soft=soft, freq=0.5 / lam,
                                          shape=shape, **lat))
                c = (m.N - 1) // 2
                one_cell(m, c, c)
                d = int(round(r / math.sqrt(2)))
                pts = [(c + r, c), (c, c + r) if m.tri() else (c + d, c + d), (c, c)]
                m.record = False
                m.step(t0)
                s = probe_run(m, pts, t1 - t0)
                amps[soft] = (lockin(s[0], m.p.freq, t0 + 1), lockin(s[1], m.p.freq, t0 + 1))
                if soft:
                    gain = m.soft_tables()['gain'][0][0]
            rx = amps[True][0][0] / amps[False][0][0]
            rd = amps[True][1][0] / amps[False][1][0]
            ph = ((amps[True][0][1] - amps[False][0][1] + 180) % 360) - 180
            tol = 0.02 if not f32 else 0.03
            good &= abs(rx - 1) < tol and abs(rd - 1) < tol
            det.append(f'{kind} lambda {lam:g}: {rx:.4f} / {rd:.4f} (gain {gain:.4f}, phase {ph:+.1f} deg)')
    ok('3d free space (one driven cell, N 1025, r = 3 lambda, lock-in over whole periods before any echo): soft / hard '
       'amplitude within 2% on the axis and the diagonal (tri: 60 deg)', good, '; '.join(det))
    # at the defaults: the square, N 513, lambda 16, the wall inset (measured, documented; "comparable")
    det, good = [], True
    for kind, lat in LATS:
        amps = {}
        for soft in (False, True):
            m = mc.Membrane(mc.Params(N=513, src='vtx', vtx_drive=True, vtx_soft=soft, **lat))
            v = m.drive_cells[0]
            x0, y0 = m.cell_xy(*v)
            L = math.hypot(x0, y0)
            fi, fj = m.xy_to_cell(x0 - x0 / L * 48, y0 - y0 / L * 48)
            c = (m.N - 1) // 2
            pts = [(int(round(fi)), int(round(fj))), (c, c), (c, c)]
            m.record = False
            m.step(192)
            s = probe_run(m, pts, 1472 - 192)
            amps[soft] = (lockin(s[0][:576], m.p.freq, 193)[0], lockin(s[1][576:], m.p.freq, 769)[0])
        r1 = amps[True][0] / amps[False][0]
        r2 = amps[True][1] / amps[False][1]
        good &= 0.6 < r1 < 1.4 and 0.6 < r2 < 1.4
        det.append(f'{kind}: {r1:.3f} at 3 lambda from a vertex, {r2:.3f} at the centre')
    ok('3d at the defaults (square, N 513, Co 0.5, lambda 16, wall inset 9 cells from both walls): soft / hard amplitude '
       '(comparable: within 0.6-1.4; the hard cell also re-radiates its corner\'s echoes)', good, '; '.join(det))

    # 3e no spurious growth
    steps = 4000 if a.quick else 20000
    tol_e = 1e-4 if f32 else 1e-12
    det, good = [], True
    for kind, lat in LATS:
        m = mc.Membrane(mc.Params(N=101, src='vtx', vtx_drive=True, vtx_soft=True, shape='n5', rot=17 * D, Co=0.6, **lat))
        m.record = False
        X, Y = m._grid_xy()
        g = np.exp(-((X - 8) ** 2 + (Y + 5) ** 2) / (2 * 3.0 ** 2))
        g[m.mask_np == 0] = 0.0
        m.set_levels(g, g)
        m.harm = [(1, 0.0)]
        drift, _ = energy_run(m, steps)
        good &= drift <= tol_e and m.vtx_soft_on()
        det.append(f'{kind} zero-drive energy {drift:.1e}')
    for kind, lat in LATS:
        m = mc.Membrane(mc.Params(N=161, src='vtx', vtx_drive=True, vtx_soft=True, shape='n6', damp=0.01, **lat))
        m.record = False
        per = int(round(1 / m.p.freq))
        m.step(3000)
        w = []
        for _ in range(2):
            mx = 0.0
            for _ in range(8):
                m.step(per // 2)
                mx = max(mx, float(np.abs(m.field()).max()))
            w.append(mx)
        rel = abs(w[1] - w[0]) / w[0]
        good &= rel < 0.01 and np.isfinite(w).all()
        det.append(f'{kind} damped steady state {w[0]:.4g} -> {w[1]:.4g}')
    for kind, lat in LATS[:2]:
        m = mc.Membrane(mc.Params(N=161, src='vtx', vtx_drive=True, vtx_soft=True, bc='mur', Co=0.7, **lat))
        m.record = False
        mx = []
        for _ in range(4):
            m.step(2500)
            mx.append(float(np.abs(m.field()).max()))
        good &= max(mx) < 5 * mx[0] and np.isfinite(mx).all()
        det.append(f'{kind} Mur full square Co 0.7 driven: ' + ' '.join(f'{v:.3g}' for v in mx))
    ok(f'3e no spurious growth: soft vertices at zero drive keep the exact energy over {steps} steps (tol {tol_e:.0e}); a '
       'damped driven run reaches a steady amplitude (<1% between windows); driven with Mur stays bounded', good,
       '; '.join(det))

    # 3f the bump flag with soft vertices
    det, good = [], True
    for kind, lat in LATS:
        kw = dict(N=161, shape='n6', rot=7 * D, src='vtx', **lat)
        ms = mc.Membrane(mc.Params(vtx_drive=True, vtx_soft=True, vtx_drive_bump=True, **kw))
        mh = mc.Membrane(mc.Params(vtx_drive=True, vtx_soft=False, vtx_drive_bump=True, **kw))
        mu = mc.Membrane(mc.Params(vtx_drive=False, **kw))
        same_seed = np.array_equal(ms.field(), mu.field())
        for m in (ms, mh, mu):
            m.step(1)
        v = ms.drive_cells[0]
        dent_h = float(mu.field()[v] - mh.field()[v])
        same1 = np.array_equal(ms.field(), mu.field())       # the drive is sin(0) = 0 at step 0: nothing added
        ms.step(40)
        mu.step(40)
        moved = float(np.abs(ms.field() - mu.field()).max())
        good &= same_seed and same1 and dent_h > 0.5 and moved > 0
        det.append(f'{kind}: seed = undriven {same_seed}, step 1 = undriven {same1}, hard dent {dent_h:.2f}')
    ok('3f vtx_drive_bump with SOFT vertices seeds the web app\'s bumps and adds the drive on top: no dent (the hard '
       'drive dents each bump centre at step 1)', good, '; '.join(det))

    # 3g the harmonics and gains
    m = mc.Membrane(mc.Params(N=161, src='vtx', vtx_drive=True, wave='saw', freq=0.02, medium='slab', shape='n4', rad=1.2))
    t = m.soft_tables()
    hh = [h for h, _ in m.harm]
    want = np.array([[a_ * mc.soft_gain('sq5', c2, 2 * math.pi * m.p.freq * h) for h, a_ in m.harm] for c2 in t['co2']])
    dv = m.soft_drive(37)
    ref = np.array([sum(a_ * mc.soft_gain('sq5', c2, 2 * math.pi * m.p.freq * h) * math.sin(2 * math.pi * m.p.freq * h * 37)
                        for h, a_ in m.harm) for c2 in t['co2']])
    ok('3g the soft drive keeps every harmonic of the hard one (band limit k_max), each scaled by soft_gain at its own '
       'frequency and the vertex cell\'s Co^2 (a slab: two media)', list(t['hh']) == hh and np.allclose(t['amp'], want, rtol=1e-12)
       and np.allclose(dv, ref, rtol=1e-12, atol=1e-15) and len(set(t['co2'])) == 2,
       f'{len(hh)} harmonics, Co^2 at the vertices {sorted(set(round(c, 4) for c in t["co2"]))}, gains of h 1: '
       f'{sorted(set(round(float(g), 4) for g in t["gain"][:, 0]))}')
    bad = []
    for kind, gw in (('sq5', 0.4987), ('sq9', 0.4690), ('tri', 0.5446)):
        g = mc.soft_gain(kind, 0.25, 2 * math.pi * 0.03125)
        if abs(g - gw) > 5e-5:
            bad.append(f'{kind} {g:.5f}')
    g00 = mc.lattice_g00('sq5', 1e-5)
    ok('3g soft_gain at Co 0.5, lambda 16: 0.4987 (sq5), 0.4690 (sq9), 0.5446 (tri); Im g00 -> 1/4 at small E on sq5',
       not bad and abs(g00.imag - 0.25) < 1e-4, '; '.join(bad) if bad else f'Im g00(1e-5) = {g00.imag:.6f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='cuda')
    ap.add_argument('--f32', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--long', action='store_true')
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    f32 = a.f32 or a.arch == 'vulkan'
    mc.init_taichi(a.arch, fp64=not f32, quiet=True)
    B = load_v10()
    T0 = time.perf_counter()
    only = {s.strip() for s in a.only.split(',') if s.strip()}
    run = lambda k: not only or k in only
    if run('1'):
        t1_neu_clip(a, B)
    if run('2'):
        t2_freq(a, B)
    if run('3'):
        t3_soft(a, B)
    print(f'\n{"ALL PASS" if not FAILS else str(len(FAILS)) + " FAILED: " + "; ".join(FAILS)} '
          f'({a.arch} {"f32" if f32 else "f64"}, {time.perf_counter() - T0:.0f} s)')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
