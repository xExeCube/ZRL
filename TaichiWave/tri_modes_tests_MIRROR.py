"""Tests for the triangular-lattice boundary modes (round 4b, 05/10/2026): the 'array' domain shape,
the triangular array's clamped ring (bc dirichlet), RIGID ring (bc neumann, link removal), RHOMBIC
TORUS (bc periodic: the six neighbours taken mod N) and the Mur fallback. The design and every number
are in NotesCLAUDE/walls_MIRROR.md section 13.

    python tri_modes_tests_MIRROR.py [--arch cuda|vulkan|cpu] [--f32] [--quick] [--only 1,4,...]

 1  exact discrete energy on the rhombic torus (plain; lens medium + pinned/rigid solids across the seam)
 2  translation invariance on the torus: a seed shifted by a lattice vector gives the exactly shifted field
 3  6-fold symmetry of a centred pulse (before it wraps, and after: the torus' period lattice is 6-fold too)
 4  the rigid ring: exact energy (with solids on the ring); R = +1 reflection against the 1D chain, the
    clamped ring's R = -1 for contrast
 5  stability at 0.995 C_max (torus, rigid ring, torus with 25% random solids), noise start
 6  'array' on the square lattice == 'square' at rad 1, bit for bit (4 bcs, both stencils, sources, solids)
 7  painting across the torus seam / on the rigid ring: incremental == full rebuild
 8  parity: every web-app configuration on the triangular lattice (any bc) and the square lattice is
    bit-identical to the core before this round (ScratchCLAUDE/Round4CLAUDE/TriModesCLAUDE)
 9  helpers: operator() vs an independent numpy operator, kernel == operator stepping, set_co rescale on
    the torus, bc_info / Mur fallback, clip fraction, vertices, source placement, mode note, recorder,
    beam / packet, the bc switch without a reseed, the recorder's triangular energy series (Etri)
10  alignment and ledger on the new modes (no crash; 'array' fully aligned; no FAIL row)
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
import membrane_core_MIRROR as mc
import solids_MIRROR as so

D = math.pi / 180
FAILS = []
PIN, RIG = 1, 2
S3H = mc.S3H


def ok(name, cond, det=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('  | ' + det if det else ''), flush=True)
    if not cond:
        FAILS.append(name)


def acc_tol():
    import taichi as ti
    return 1e-6 if mc.ACC_DT == ti.f32 else 1e-14


def tri_array(N, bc, **kw):
    prm = dict(N=N, lattice='tri', shape='array', bc=bc, src='impulse')
    prm.update(kw)
    m = mc.Membrane(mc.Params(**prm))
    m.record = False
    return m


def hex_r2(N, i0, j0, wrap):
    """a^2 + ab + b^2 (the physical distance squared, an exact integer) of the index offsets (a, b)
    from (i0, j0); wrap: the minimum over the torus images (a + kN, b + lN), k, l in {-1, 0, 1} --
    invariant under the 60-degree rotation, since the period lattice N Z^2 is."""
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    a, b = I - i0, J - j0
    if not wrap:
        return a * a + a * b + b * b
    best = None
    for k in (-1, 0, 1):
        for l in (-1, 0, 1):
            aa, bb = a + k * N, b + l * N
            r = aa * aa + aa * bb + bb * bb
            best = r if best is None else np.minimum(best, r)
    return best


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


def seam_solids(N, seed=3):
    """Pinned and rigid objects that touch / cross every seam of the array (rows 0 and N-1, columns 0
    and N-1, the corner), plus a one-cell diagonal across the corner."""
    S = np.zeros((N, N), np.uint8)
    S[0:3, 30:42] = RIG
    S[N - 3:N, 30:42] = RIG                 # with the above: one object across the i seam
    S[40:52, 0:2] = PIN
    S[40:52, N - 2:N] = PIN                 # across the j seam
    S[N - 4:N, N - 4:N] = RIG
    S[0:3, 0:3] = RIG                       # around the corner (all four corners meet on the torus)
    for k in range(10):
        S[(N - 5 + k) % N, (5 - k) % N] = RIG     # a one-cell (1,-1) diagonal across the corner seam
    S[60:66, 60:63] = PIN
    S[20:24, 70:76] = RIG
    return S


# ---------------------------------------------------------------------------------------------
def t1_torus_energy(a, tol_e):
    """1. The rhombic torus conserves the exact leapfrog energy (20,000 steps at 0.95 C_max)."""
    steps = 4000 if a.quick else 20000
    N = 97
    co = 0.95 * mc.CO_LIMIT_T
    for label, extra, solids in (('plain, pulse across the corner seam', dict(), False),
                                 ('lens medium + pinned/rigid solids across every seam', dict(medium='lens'), True)):
        m = tri_array(N, 'periodic', Co=co, **extra)
        if solids:
            m.set_solid_np(seam_solids(m.N))
        n = m.N
        u = np.exp(-hex_r2(n, 2, n - 3, True) / (2 * 9.0)) * m.unknowns()
        m.set_levels(u, u)
        drift, E0 = energy_run(m, steps)
        ok(f'1. rhombic torus conserves the exact energy: {label}',
           drift <= tol_e and E0 > 0 and m.tri_torus() and (m.walls_active() == solids),
           f'N={n}, Co = 0.95 C_max, {steps} steps: max|E - E0|/E0 = {drift:.1e} (tol {tol_e:.0e}); '
           f'walls kernel {"ON" if m.walls_active() else "off"} ({m.n_wall_cells} non-plain cells)')


def t2_translation(a):
    """2. Translation invariance: on the torus a seed shifted by a lattice vector (di, dj) gives the
    EXACTLY shifted field (bit for bit), plain and with solids shifted along (the walls path, words
    wrapping across the seam)."""
    N = 129
    K = 150 if a.quick else 600
    shifts = ((37, -55), (64, 64), (-1, 1))
    for walls in (False, True):
        det, good = [], True
        for di, dj in shifts:
            res = []
            for k in range(2):
                m = tri_array(N, 'periodic', Co=0.6)
                n = m.N
                S = seam_solids(n) if walls else None
                u = np.exp(-hex_r2(n, n - 10, 7, True) / (2 * 6.25))
                u += 0.3 * np.exp(-hex_r2(n, 50, 90, True) / (2 * 2.0))
                if walls:
                    u = u * (S == 0)
                if k == 1:
                    u = np.roll(u, (di, dj), (0, 1))
                    if walls:
                        S = np.roll(S, (di, dj), (0, 1))
                if walls:
                    m.set_solid_np(S)
                m.set_levels(u, u)
                m.step(K)
                res.append(m.field())
            same = np.array_equal(np.roll(res[0], (di, dj), (0, 1)), res[1])
            dmax = float(np.abs(np.roll(res[0], (di, dj), (0, 1)) - res[1]).max())
            good &= same and float(np.abs(res[0]).max()) > 1e-3
            det.append(f'({di},{dj}) {"bit-identical" if same else f"differs {dmax:.1e}"}')
        ok(f'2. translation invariance on the torus: {"walls path (solids shifted along)" if walls else "plain"}',
           good, f'N={N}, {K} steps (the pulses wrap several times): ' + '; '.join(det))


def t3_sixfold(a, tol):
    """3. A centred pulse on the rhombic torus keeps the lattice's 60-degree rotation (a, b) -> (-b, a+b)
    exactly up to rounding: before it reaches the seam, and after (the torus' period lattice is itself
    6-fold symmetric, so the symmetry survives the wrap)."""
    N = 161
    m = tri_array(N, 'periodic', Co=0.6)
    n = m.N
    c = (n - 1) // 2
    u = np.exp(-hex_r2(n, c, c, True) / (2 * 9.0))
    m.set_levels(u, u)
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    A, B = I - c, J - c
    det, good = [], True
    done = 0
    for K, tag in ((60, 'before the seam'), (300 if a.quick else 1500, 'after many wraps')):
        m.step(K - done)
        done = K
        U = m.field().astype(np.float64)
        worst = 0.0
        p, q = A.copy(), B.copy()
        for _ in range(5):
            p, q = -q, p + q
            worst = max(worst, float(np.abs(U - U[(c + p) % n, (c + q) % n]).max()))
        rel = worst / float(np.abs(U).max())
        reach = 0.6 * K + 12
        g = rel <= tol
        good &= g
        det.append(f'step {K} ({tag}, front ~{reach:.0f} of {c * S3H:.0f} cells to the seam): max|u - u(rot^k)| / '
                   f'max|u| = {rel:.1e}')
    ok('3. 6-fold symmetry of a centred pulse on the rhombic torus', good, f'N={n}, tol {tol:.0e}: ' + '; '.join(det))


def t4_rigid_ring(a, tol_e, tol_img):
    """4. The RIGID ring (bc neumann): exact energy, with and without solids on the ring; a plane
    pulse reflects with R = +1 exactly as the 1D chain with the link-removal ghost; the clamped ring
    (bc dirichlet) reflects with R = -1 (the ghost holds 0)."""
    steps = 4000 if a.quick else 20000
    co = 0.95 * mc.CO_LIMIT_T
    for label, solids in (('pulse at the acute corner', False), ('lens + pinned/rigid solids on the ring', True)):
        m = tri_array(97, 'neumann', Co=co, medium='lens' if solids else 'uniform')
        n = m.N
        if solids:
            S = seam_solids(n)
            m.set_solid_np(S)
        u = np.exp(-hex_r2(n, 8, 8, False) / (2 * 9.0)) * m.unknowns()
        m.set_levels(u, u)
        drift, E0 = energy_run(m, steps)
        ok(f'4. rigid ring conserves the exact energy: {label}', drift <= tol_e and m.walls_active() and E0 > 0,
           f'N={n}, Co = 0.95 C_max, {steps} steps: max|E - E0|/E0 = {drift:.1e} (tol {tol_e:.0e}); '
           f'{m.n_wall_cells} cells with dropped links')
    # reflection of a plane pulse: the field in a band of columns that the side rings cannot reach in
    # T steps (one cell per step) equals the 1D chain u(j); a rigid ring cell's links are dropped
    # (the ghost reads u itself), a clamped ring reads 0
    N, T, sig, j0 = 257, 100, 3.0, 30
    det, good = [], True
    for bc, sgn in (('neumann', 1.0), ('dirichlet', -1.0)):
        m = tri_array(N, bc, Co=0.5)
        n = m.N
        jj = np.arange(n, dtype=np.float64)
        f = np.exp(-((jj - j0) * S3H) ** 2 / (2 * sig * sig))
        f[0] = f[-1] = 0.0
        U0 = np.broadcast_to(f[None, :], (n, n)) * m.unknowns()
        m.set_levels(U0, U0)
        m.step(T)
        got = m.field().astype(np.float64)
        cc = float(m.cc_np[n // 2, n // 2])
        dt = np.float32 if not mc.FP64 else np.float64

        def chain(rigid, pad=0):
            g = np.concatenate([np.zeros(pad), f]).astype(dt)
            up, u = g.copy(), g.copy()
            two3, cc_ = dt(2.0 / 3.0), dt(cc)
            for _ in range(T):
                um, uc, upp = u[:-2], u[1:-1], u[2:]
                if rigid and pad == 0:
                    um = um.copy()
                    um[0] = uc[0]                                  # links (0,-1), (1,-1) into the ring dropped
                    upp = upp.copy()
                    upp[-1] = uc[-1]
                s = uc + uc + um + upp + um + upp
                new = np.zeros_like(u)
                new[1:-1] = (dt(2) * uc - up[1:-1] + cc_ * two3 * (s - dt(6) * uc))
                up, u = u, new
            return u[pad:].astype(np.float64)
        ref = chain(bc == 'neumann')
        free = chain(False, pad=400)                               # no wall within reach: the free chain
        band = slice(T + 2, n - 2 - T)
        rel = float(np.abs(got[band, :] - ref[None, :]).max() / np.abs(ref).max())
        refl = (ref - free)[1:60]
        k = int(np.argmax(np.abs(refl)))
        got_sgn = 1.0 if refl[k] > 0 else -1.0
        g = rel <= tol_img and got_sgn == sgn
        good &= g
        det.append(f'{"rigid" if bc == "neumann" else "clamped"} ring: |2D - 1D chain| / max = {rel:.1e} over '
                   f'{n - 4 - 2 * T} columns, reflected peak {refl[k]:+.4f} (R {"+" if got_sgn > 0 else "-"}1)')
    ok('4. plane-pulse reflection off the ring = the 1D chain: rigid R = +1, clamped R = -1', good,
       f'N={N}, sigma {sig} at row {j0}, {T} steps (tol {tol_img:.0e}): ' + '; '.join(det))


def t5_cfl(a):
    """5. Stable at 0.995 C_max: torus, rigid ring, torus with 25% random pinned/rigid solids."""
    steps = 4000 if a.quick else 20000
    rng = np.random.default_rng(5)
    co = 0.995 * mc.CO_LIMIT_T
    for label, bc, solids in (('rhombic torus', 'periodic', False), ('rigid ring', 'neumann', False),
                              ('torus, 25% random solids', 'periodic', True),
                              ('rigid ring, 25% random solids', 'neumann', True)):
        m = tri_array(129, bc, Co=co)
        n = m.N
        if solids:
            S = np.where(rng.random((n, n)) < 0.25, rng.integers(1, 3, (n, n)), 0).astype(np.uint8)
            m.set_solid_np(S)
        u = rng.standard_normal((n, n)) * m.unknowns()
        m.set_levels(u, u)
        u0 = float(np.abs(u).max())
        drift, _ = energy_run(m, steps, every=steps // 4)
        um = float(np.abs(m.field()).max())
        ok(f'5. stable at 0.995 C_max: {label}', np.isfinite(um) and um < 10 * u0 and drift < 1e-6,
           f'N={n}, Co {co:.4f}, {steps} steps, noise start: max|u| {u0:.2f} -> {um:.2f}, energy drift {drift:.1e}; '
           f'{m.n_wall_cells} non-plain cells')


def _hist_pack(m):
    h = m.hist
    return np.array(h['E']), np.array(h['amax'] + h['pA'] + h['pB'] + h['mode'])


def t6_sq_array(a):
    """6. 'array' on the square lattice IS the full square ('square' at rad 1): fields, histories,
    source and vertex cells, clip fraction, geometry, link words -- bit for bit."""
    steps = 150 if a.quick else 400

    def objs():
        sc = so.SolidScene()
        sc.add(so.SolidObject('rect', x=0.3, y=0.2, w=0.2, h=0.1, wall='rigid'))
        sc.add(so.SolidObject('circle', x=-0.4, y=-0.3, r=0.12))
        sc.add(so.SolidObject('rect', x=0.97, y=-0.5, w=0.1, h=0.3, wall='rigid'))     # on the ring
        sc.add_stroke([(-0.9, 0.99), (0.2, 0.6)], 0.01, wall='rigid')
        return sc
    cases = [('sq5 Dirichlet pulse', dict(src='pulse', sx=0.3), None),
             ('sq9 Neumann corner pulse', dict(stencil=9, bc='neumann', src='pulse', sx=-0.8, sy=-0.8, sigma=2), None),
             ('sq5 Mur continuous', dict(bc='mur', src='cont', freq=0.06, sx=0.2), None),
             ('sq9 torus pulse', dict(stencil=9, bc='periodic', src='pulse', sx=0.6, Co=0.8), None),
             ('sq5 mode (3,2)', dict(src='mode', mm=3, mn=2), None),
             ('sq5 vertices, wall inset', dict(src='vtx', vtx_inset='wall', sigma=2), None),
             ('sq9 beam', dict(stencil=9, src='beam', beam_angle=30.0, freq=0.08), None),
             ('sq5 packet, torus', dict(bc='periodic', src='packet', beam_angle=-60.0, sx=0.5), None),
             ('sq9 Neumann + pinned/rigid solids, edge rigid', dict(stencil=9, bc='neumann', src='pulse', edge='rigid'), objs),
             ('sq5 torus + solids', dict(bc='periodic', src='pulse', sx=0.9), objs)]
    for label, prm, mk in cases:
        res = []
        for shape, rad, rot in (('array', 0.6, 0.7), ('square', 1.0, 0.0)):
            # 'array' ignores rad and rot: give it other values on purpose
            m = mc.Membrane(mc.Params(N=161, shape=shape, rad=rad, rot=rot, **prm))
            if mk is not None:
                m.attach_scene(mk())
                m.reseed()
            geo = (m.sq_full(), m.dom_R(), m.inner_R(), tuple(m.shape_poly()), m.mode_box(), tuple(m.src),
                   tuple(map(tuple, m.vtx)), m.clip_frac, m.bc_info()['effective'])
            m.step(steps)
            E, P = _hist_pack(m)
            W = m.W.to_numpy() if m.W is not None else None
            res.append((m.field(), E, P, geo, W, m.walls_active()))
        (u1, E1, P1, g1, W1, a1), (u2, E2, P2, g2, W2, a2) = res
        fs, ps, gs = np.array_equal(u1, u2), np.array_equal(P1, P2), g1 == g2
        ws = (W1 is None and W2 is None) or (W1 is not None and W2 is not None and np.array_equal(W1, W2))
        dE = float(np.abs(E1 - E2).max() / max(np.abs(E2).max(), 1e-300)) if len(E1) == len(E2) and len(E2) else 1.0
        ok(f"6. sq 'array' == 'square' rad 1: {label}", fs and ps and gs and ws and dE <= acc_tol() and a1 == a2,
           f'N=161, {steps} steps: field {"bit-identical" if fs else "DIFFERS"}, peak/probe histories '
           f'{"bit-identical" if ps else "DIFFER"}, energy history {dE:.1e}, geometry/source/vertices/clip '
           f'{"equal" if gs else "DIFFER"}, link words {"equal" if ws else "DIFFER"} (walls {a1})')


def t7_paint_seam(a):
    """7. Painting on the rhombic torus (objects and strokes across the seams, both wall types, an
    eraser, a pinned -> rigid edit) and on the rigid ring: incremental == the full rebuild (type map,
    device mask, classes, words, counters); solids hold 0 after stepping."""
    for label, bc in (('rhombic torus', 'periodic'), ('rigid ring', 'neumann'), ('clamped ring', 'dirichlet')):
        m = mc.Membrane(mc.Params(N=257, lattice='tri', shape='array', bc=bc, src='pulse', sigma=3, sx=0.1))
        sc = so.SolidScene()
        m.attach_scene(sc)
        m.step(40)
        bad, nup = [], 0

        def check(tag):
            nonlocal nup
            nup += 1
            ref = sc.rasterize_full(m)
            ref = np.zeros((m.N, m.N), np.uint8) if ref is None else ref
            cur = np.zeros((m.N, m.N), np.uint8) if m.solid_np is None else m.solid_np
            n_t = int(np.count_nonzero(ref != cur))
            n_m = int(np.count_nonzero(m.M.to_numpy().astype(np.uint8) != (m.domain_mask_np & (ref == 0))))
            if m.W is None:
                if n_t or n_m:
                    bad.append(f'{tag}: type {n_t} mask {n_m}')
                return
            Wi, Ci = m.W.to_numpy().copy(), m.C.to_numpy().copy()
            cnt_i = (m.n_wall_cells, m.n_blk4)
            m._walls_full()
            n_w = int(np.count_nonzero(Wi != m.W.to_numpy()))
            n_c = int(np.count_nonzero(Ci != m.C.to_numpy()))
            cnt_f = (m.n_wall_cells, m.n_blk4)
            if n_t or n_m or n_w or n_c or cnt_i != cnt_f:
                bad.append(f'{tag}: type {n_t} mask {n_m} class {n_c} word {n_w} counts {cnt_i} vs {cnt_f}')
        r1 = so.cells_to_grid(m, 1.0)
        # grid units: y = +-0.99 are the rows next to the j seam; x = -+1.15 at y = 0 the columns next
        # to the i seam (the rhombus spans x in [-1.73, 1.73] grid units at its long diagonal)
        m.update_solids(sc.add(so.SolidObject('circle', x=0.3, y=0.2, r=0.08, wall='rigid'))[1])
        check('rigid circle')
        m.update_solids(sc.add(so.SolidObject('rect', x=0.0, y=-0.99, w=0.2, h=0.06, wall='pinned'))[1])
        check('pinned rect on the j seam')
        m.update_solids(sc.add(so.SolidObject('rect', x=-0.2, y=0.99, w=0.2, h=0.06, wall='rigid'))[1])
        check('rigid rect on the other side of the j seam')
        m.update_solids(sc.add(so.SolidObject('ngon', x=-1.15, y=0.0, r=0.1, n=5, wall='rigid'))[1])
        check('rigid pentagon on the i seam')
        m.update_solids(sc.add(so.SolidObject('rect', x=1.15, y=0.0, w=0.08, h=0.2, wall='pinned'))[1])
        check('pinned rect on the other side of the i seam')
        for wall, pts in (('rigid', [(-1.5, -0.99), (0.3, 0.1), (1.6, 0.99), (0.6, -0.9)]),
                          ('pinned', [(-1.6, -0.97), (-0.5, 0.97), (1.5, 0.95)])):
            m.update_solids(sc.begin_stroke(*pts[0], r1, wall=wall))
            check(f'{wall} stroke start')
            for q in pts[1:]:
                m.update_solids(sc.extend_stroke(*q))
                check(f'{wall} stroke segment')
            sc.end_stroke()
        m.update_solids(sc.begin_stroke(-0.2, -0.3, so.cells_to_grid(m, 4.0), erase=True))
        m.update_solids(sc.extend_stroke(0.5, 0.5))
        check('eraser')
        sc.end_stroke()
        idx, bb = sc.add(so.SolidObject('rect', x=0.0, y=-0.6, w=0.3, h=0.05, wall='pinned'))
        m.update_solids(bb)
        m.update_solids(sc.edit(idx, wall='rigid'))
        check('object switched pinned -> rigid')
        m.step(40)
        solid = m.solid_np != 0 if m.solid_np is not None else np.zeros((m.N, m.N), bool)
        U = m.U.to_numpy()
        zero = float(np.abs(U[:, solid]).max()) if solid.any() else 0.0
        fin = bool(np.isfinite(U).all())
        ok(f'7. incremental painting == full rebuild: tri array, {label}', not bad and zero == 0.0 and fin and m.walls_active(),
           f'N={m.N}: {nup} updates compared (type map, device mask, classes, words, counters): '
           + ('all equal' if not bad else '; '.join(bad[:4])) + f'; solids after 40 more steps max|u| {zero:.1e}; '
           f'{int((m.solid_np == 1).sum())} pinned + {int((m.solid_np == 2).sum())} rigid cells')


def load_before():
    path = os.path.join(HERE, 'ScratchCLAUDE', 'Round4CLAUDE', 'TriModesCLAUDE', 'membrane_core_before_MIRROR.py')
    spec = importlib.util.spec_from_file_location('membrane_core_before_trimodes_MIRROR', path)
    B = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    spec.loader.exec_module(B)
    sys.path[:] = saved
    B.MASK_DT, B.SQ9_CONSISTENT, B.FP64, B.ACC_DT, B.WORD_DT = mc.MASK_DT, mc.SQ9_CONSISTENT, mc.FP64, mc.ACC_DT, mc.WORD_DT
    return B


def t8_parity(a):
    """8. Nothing the web app can make changes: every triangular configuration of crosscheck_js with
    every bc (the web app ignores bc on the triangular lattice; so does the port for its shapes), the
    triangular Unlicensed plate, rigid edges and solids, and square-lattice configurations, against the
    core of 04/10/2026 (before this round): fields, peak / probe histories, source cells, vertices,
    clip fraction bit for bit; the energy history to the atomics' rounding."""
    B = load_before()
    steps = 120 if a.quick else 300
    tri_js = [dict(shape='n6', rot=30 * D, Co=0.8),
              dict(shape='n3', rot=30 * D, src='vtx', vtx_drive=True, wave='tri', freq=0.05, vtx_inset='radial',
                   vtx_drive_bump=True),
              dict(shape='rhomb', rot=30 * D, rad=1.5, src='impulse'),
              dict(shape='circle', rad=1.3, medium='lens', sx=0.5),
              dict(shape='n12', src='cont', wave='saw', freq=0.05),
              dict(shape='n5', rot=12 * D, src='slit', freq=0.06),
              dict(shape='square', rad=0.9, medium='slab', src='vtx', sigma=1.5, vtx_inset='radial')]
    # the vertex defaults changed after 04/10 (05/10/2026: inset 'wall', driven vertices from rest): the
    # vertex configurations set the 04/10 values explicitly, and a field the 04/10 core does not have
    # (vtx_drive_bump) is passed to the current core only
    bcs = ('dirichlet', 'mur', 'neumann', 'periodic')
    cases = []
    for k, prm in enumerate(tri_js):
        for bc in (bcs if not a.quick else ('periodic', 'neumann')):
            cases.append((f'tri js#{k + 1} {prm["shape"]} bc {bc}', dict(lattice='tri', bc=bc, **prm), None))

    def objs():
        sc = so.SolidScene()
        sc.add(so.SolidObject('rect', x=0.3, y=0.2, w=0.2, h=0.1, wall='rigid'))
        sc.add(so.SolidObject('circle', x=-0.4, y=-0.3, r=0.12))
        sc.add_stroke([(-0.9, 0.99), (0.2, 0.6)], 0.01, wall='rigid')
        return sc
    cases += [('tri Unlicensed r 0.85, bc periodic', dict(lattice='tri', shape='unlicensed', rad=0.85, sy=-0.9, bc='periodic'), None),
              ('tri hexagon, edge rigid, bc neumann', dict(lattice='tri', shape='n6', rot=30 * D, edge='rigid', bc='neumann'), None),
              ('tri circle r 1.3, edge rigid, bc periodic', dict(lattice='tri', shape='circle', rad=1.3, edge='rigid', bc='periodic'), None),
              ('tri circle r 1.2, solids, bc periodic', dict(lattice='tri', shape='circle', rad=1.2, bc='periodic', sx=0.2), objs),
              ('sq5 full Dirichlet pulse', dict(sx=0.3), None),
              ('sq9 full Neumann corner', dict(stencil=9, bc='neumann', sx=-0.8, sy=-0.8, sigma=2), None),
              ('sq5 circle r 1.3 torus', dict(shape='circle', rad=1.3, bc='periodic', sx=0.7), None),
              ('sq9 circle Mur, driven', dict(shape='circle', rad=0.8, stencil=9, bc='mur', src='cont', freq=0.06), None),
              ('sq9 full torus + solids', dict(stencil=9, bc='periodic', sx=0.5), objs),
              ('sq5 full Neumann + solids, edge rigid', dict(bc='neumann', edge='rigid'), objs),
              ('sq9 pentagon, edge rigid + solids', dict(stencil=9, shape='n5', rot=17 * D, edge='rigid'), objs),
              ('sq5 masked square r 0.7 mode', dict(shape='square', rad=0.7, src='mode', mm=2, mn=5), None)]
    nbad = 0
    for label, prm, mk in cases:
        res = []
        # the vertex cells are compared for every source (recache_source places them always), so every
        # case runs the 04/10 inset (05/10/2026: the port's default became 'wall')
        # and (09/10/2026) the drive frequency and the hard vertex drive of 04/10 (the port's defaults became 0.03125
        # and soft); vtx_soft is passed to the current core only (the field filter below)
        prm = {'vtx_inset': 'radial', 'freq': 0.12, 'vtx_soft': False, **prm}
        for mod in (mc, B):
            fld = mod.Params.__dataclass_fields__
            m = mod.Membrane(mod.Params(N=161, **{k: v for k, v in prm.items() if k in fld}))
            if mk is not None:
                m.attach_scene(mk())
                m.reseed()
            geo = (tuple(m.src[:2]), tuple(tuple(v[:2]) for v in m.vtx), m.clip_frac)
            m.step(steps)
            E, P = _hist_pack(m)
            res.append((m.field(), E, P, geo, m.walls_active()))
        (u1, E1, P1, g1, w1), (u2, E2, P2, g2, w2) = res
        fs, ps, gs = np.array_equal(u1, u2), np.array_equal(P1, P2), g1 == g2
        dE = float(np.abs(E1 - E2).max() / max(np.abs(E2).max(), 1e-300)) if len(E1) == len(E2) and len(E2) else 1.0
        good = fs and ps and gs and dE <= acc_tol() and w1 == w2
        nbad += not good
        if not good or a.verbose:
            ok(f'8. parity with the 04/10 core: {label}', good,
               f'field {"bit-identical" if fs else "DIFFERS"}, histories {"bit-identical" if ps else "DIFFER"}, '
               f'energy {dE:.1e}, source/vertices/clip {"equal" if gs else "DIFFER"}, walls {w1}/{w2}')
    ok(f'8. parity with the 04/10 core: {len(cases)} configurations (every web-app tri config x every bc, '
       'the tri plate, rigid edges, solids, square lattice)', nbad == 0,
       f'N=161, {steps} steps each: {len(cases) - nbad}/{len(cases)} bit-identical (fields, peak/probe histories, '
       f'source cells, vertices, clip fraction; energy history <= {acc_tol():.0e})')


def numpy_tri_L(N, mode):
    """An independent triangular operator (2/3)(sum of the 6 neighbours - 6u) over the whole grid:
    'torus' wraps (every cell an unknown); 'clamped' / 'rigid': unknowns = the interior, the ring
    reads 0 (clamped) or its links are dropped (rigid)."""
    import scipy.sparse as sp
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    I, J = I.ravel(), J.ravel()
    if mode != 'torus':
        keep = (I > 0) & (J > 0) & (I < N - 1) & (J < N - 1)
        I, J = I[keep], J[keep]
    rows, cols, vals = [], [], []
    diag = np.zeros(N * N)
    for di, dj in mc.TRI_LINKS:
        ni, nj = I + di, J + dj
        if mode == 'torus':
            ni, nj = ni % N, nj % N
            act = np.ones(I.size, bool)
            inner = act
        else:
            inner = (ni > 0) & (nj > 0) & (ni < N - 1) & (nj < N - 1)
            act = inner if mode == 'rigid' else np.ones(I.size, bool)
        diag[(I * N + J)[act]] -= 2.0 / 3.0
        sel = act & inner
        rows.append((I * N + J)[sel])
        cols.append((ni * N + nj)[sel])
        vals.append(np.full(int(sel.sum()), 2.0 / 3.0))
    L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N * N, N * N))
    return (L + sp.diags(diag)).tocsr()


def t9_helpers(a, tol_op):
    """9. The helpers and every shape- or ring-dependent piece of the core."""
    # (a) operator() == an independent numpy operator; (b) 32 kernel steps == 32 operator steps
    N = 65
    for bc, mode in (('periodic', 'torus'), ('neumann', 'rigid'), ('dirichlet', 'clamped'), ('mur', 'clamped')):
        m = tri_array(N, bc, Co=0.6)
        n = m.N
        Lm, Lr = m.operator(), numpy_tri_L(n, mode)
        dL = abs(Lm - Lr).max()
        unk = m.unknowns()
        u0 = np.exp(-hex_r2(n, 5, n - 6, mode == 'torus') / 8.0) * unk
        co2 = float(m.cc_np[n // 2, n // 2])
        u1 = u0 + 0.5 * co2 * (Lr @ u0.ravel()).reshape(n, n)
        A, Bv = u0.ravel().copy(), u1.ravel().copy()
        for _ in range(32):
            A, Bv = Bv, 2 * Bv - A + co2 * (Lr @ Bv)
        m.set_levels(u0, u1)
        m.step(32)
        rel = float(np.abs(m.field().astype(np.float64).ravel() - Bv).max() / np.abs(Bv).max())
        ok(f'9a. operator() == numpy operator, kernel == operator stepping: tri array bc {bc}',
           dL == 0.0 and rel <= tol_op,
           f'N={n}: max|L - L_ref| {dL:.1e}; 32 steps max|u - u_ref|/max {rel:.1e} (tol {tol_op:.0e}); '
           f'unknowns {int(unk.sum())}, walls {m.walls_active()}')
    # (c) set_co(rescale=True) on the torus (plain and walls path) == the formula with operator()
    for walls in (False, True):
        m = tri_array(97, 'periodic', Co=0.5)
        n = m.N
        if walls:
            m.set_solid_np(seam_solids(n))
        u = np.exp(-hex_r2(n, 3, n - 4, True) / 18.0) * m.unknowns()
        m.set_levels(u, u)
        m.step(57)
        L = m.operator()
        up, uc = m.level(0).astype(np.float64), m.level(1).astype(np.float64)
        cc_old = m.cc_np.copy()
        r = 0.6 / 0.5
        want = uc - r * (uc - up) + 0.5 * (r * r - r) * cc_old * (L @ uc.ravel()).reshape(n, n)
        want *= m.unknowns()
        m.set_co(0.6, rescale=True)
        got = m.level(0).astype(np.float64)
        rel = float(np.abs(got - want).max() / np.abs(want).max())
        ok(f'9c. set_co(rescale) on the rhombic torus: {"walls path" if walls else "plain"}', rel <= tol_op,
           f'N={n}: max|u_prev - formula|/max {rel:.1e} (tol {tol_op:.0e}); the seam cells included')
    # (d) bc_info and the Mur fallback: tri array + mur runs exactly the clamped ring
    ma = tri_array(97, 'mur', src='pulse', sx=0.6, Co=0.6)
    mb = tri_array(97, 'dirichlet', src='pulse', sx=0.6, Co=0.6)
    ma.step(300)
    mb.step(300)
    bi = ma.bc_info()
    mi = mc.Membrane(mc.Params(N=97, lattice='tri', shape='circle', bc='periodic')).bc_info()
    sqi = mc.Membrane(mc.Params(N=97, shape='array', bc='mur')).bc_info()
    good = (np.array_equal(ma.field(), mb.field()) and bi['fallback'] and bi['effective'] == 'dirichlet'
            and bi['reason'] == mc.TRI_MUR_REASON and 'mur' not in bi['options'] and not bi['edge_applies']
            and mi['fallback'] and mi['reason'] == mc.TRI_MASKED_REASON and mi['options'] == ['dirichlet']
            and mi['edge_applies'] and not sqi['fallback'] and sqi['options'] == mc.BCS)
    ok('9d. bc_info: the Mur fallback on the triangular array runs the clamped ring bit for bit', good,
       f'tri array mur: effective {bi["effective"]}, fallback {bi["fallback"]}, options {bi["options"]}, ring '
       f'"{bi["ring"]}"; tri circle + periodic: fallback {mi["fallback"]}, options {mi["options"]}; sq array mur: '
       f'fallback {sqi["fallback"]}')
    # (e) clip fraction: the torus' ring is not cut, the clamped / rigid ring is
    cl = {}
    for bc in ('periodic', 'neumann', 'dirichlet'):
        m = mc.Membrane(mc.Params(N=97, lattice='tri', shape='array', bc=bc, src='pulse', sigma=4, sx=0.0, sy=-0.97))
        cl[bc] = m.clip_frac
    good = 0 < cl['periodic'] < cl['neumann'] and cl['neumann'] == cl['dirichlet']
    ok('9e. pulse clip fraction: torus ring kept, clamped / rigid ring cut', good,
       f'pulse sigma 4 at sy = -0.97: torus {cl["periodic"]:.4f} (only what leaves the array), rigid '
       f'{cl["neumann"]:.4f}, clamped {cl["dirichlet"]:.4f}')
    # (f) vertices: the 4 rhombus corners, both inset rules
    det, good = [], True
    for mode in ('radial', 'wall'):
        m = mc.Membrane(mc.Params(N=161, lattice='tri', shape='array', src='vtx', vtx_inset=mode, sigma=3))
        P = m.shape_poly()
        V = [m.cell_xy(v[0], v[1]) for v in m.vtx]
        inset = 3 * 3.0 if mode == 'wall' else max(2, math.ceil(1.5 * 3.0))
        dist = []
        for (px, py), (vx, vy) in zip(P, V):
            dist.append(math.hypot(vx - px, vy - py))
        # wall rule: inset / sin(half the corner angle): 2 inset at the 60-degree corners (0 and 2),
        # inset / sin 60 at the 120-degree ones; radial: inset along the ray to the centre
        want = ([2 * inset, inset / S3H, 2 * inset, inset / S3H] if mode == 'wall' else [inset] * 4)
        err = max(abs(d - w) for d, w in zip(dist, want))
        g = len(V) == 4 and err <= 1.0 and m.vtx_capped == 0 and all(m.mask_np[v[0], v[1]] for v in m.vtx)
        good &= g
        det.append(f'{mode}: corner -> vertex {", ".join(f"{d:.2f}" for d in dist)} (want {", ".join(f"{w:.2f}" for w in want)}, '
                   f'snap <= 1 cell)')
    ok('9f. vertex sources at the 4 rhombus corners, both inset rules', good, '; '.join(det))
    # (g) source placement in fractions of r_in = base_half; scale mode uses the same r_in
    m = mc.Membrane(mc.Params(N=161, lattice='tri', shape='array', bc='neumann', src='pulse', sx=1.0, sy=0.0))
    x, y = m.cell_xy(*m.src[:2])
    s1 = (not m.src_moved and abs(x - m.inner_R()) <= 1.0 and abs(y) <= 1.0)
    m.p.sx, m.p.sy = 0.0, 0.99
    m.recache_source()
    x2, y2 = m.cell_xy(*m.src[:2])
    s2 = m.src_snap == 'none' and m.mask_np[m.src[0], m.src[1]] == 1 and abs(y2 - 0.99 * m.inner_R()) <= 1.0
    m.set_scale('domain')
    sf = m.p.sigma_f
    s3 = abs(sf * m.inner_R() - m.p.sigma) < 1e-12 and abs(m.inner_R() - m.base_half()) < 1e-12
    ok('9g. source placement and scale mode use r_in = the array inradius', s1 and s2 and s3,
       f'r_in {m.inner_R():.3f} = base_half; sx = 1 -> ({x:.2f}, {y:.2f}); sy = 0.99 -> y {y2:.2f} (snap {m.src_snap}); '
       f'scale domain sigma_f {sf:.4f}')
    # (h) the mode source: index-space sin.sin, n/a as an eigenmode on the triangular array
    m = mc.Membrane(mc.Params(N=97, lattice='tri', shape='array', src='mode', mm=2, mn=3))
    note = m.mode_note()
    ms = mc.Membrane(mc.Params(N=97, shape='array', src='mode', mm=2, mn=3))
    m.step(50)
    ok('9h. mode source: exact on the square array, n/a (with the reason) on the triangular one',
       note is not None and 'not an eigenmode' in note and ms.mode_note() is None and m.mode_box() == (0, m.N - 1)
       and np.isfinite(m.field()).all(),
       f'tri: "{note}"; sq array: {ms.mode_note()}; tri mode_box {m.mode_box()}')
    # (i) the recorder: stats() == the recorder's row on the torus and the rigid ring; a constant field
    # inside the rigid ring stores no potential, the clamped ring does
    res = {}
    for bc in ('periodic', 'neumann', 'dirichlet'):
        m = mc.Membrane(mc.Params(N=97, lattice='tri', shape='array', bc=bc, src='impulse', Co=0.5))
        u = np.where(m.unknowns(), 1.0, 0.0)
        if bc == 'periodic':
            u = u + 0.5 * np.exp(-hex_r2(m.N, 3, 3, True) / 8.0)
        m.set_levels(u, u)
        m.step(8)
        E_rec = m.hist['E'][-1]
        E_st, _ = m.stats()
        res[bc] = (E_rec, E_st)
    same = all(abs(r[0] - r[1]) <= acc_tol() * max(abs(r[1]), 1.0) for r in res.values())
    good = same and res['neumann'][0] <= 1e-20 and res['dirichlet'][0] > 1.0
    ok('9i. recorder: stats() == recorder; a constant field in the rigid ring stores no potential', good,
       f'E after 8 steps: torus {res["periodic"][0]:.4g}, rigid ring {res["neumann"][0]:.1e}, clamped ring '
       f'{res["dirichlet"][0]:.4g}; stats() == recorder: {same}')
    # (l) the triangular energy series (m.hist['Etri'], the ledger's 'tri-energy'): the recorder's value ==
    # numpy's kin/(2Co^2) + (gx^2 + gy^2 + gz^2)/3 from the last two levels -- over EVERY cell with wrapped
    # differences on the rhombic torus, over the interior (the ring reading its zeros) on the clamped ring;
    # on the square lattice there is no Etri
    det, good = [], True
    for bc in ('periodic', 'dirichlet'):
        m = mc.Membrane(mc.Params(N=97, lattice='tri', shape='array', bc=bc, src='pulse', sx=0.6, sy=-0.8, Co=0.6))
        m.step(150)
        up, u = m.level(0).astype(np.float64), m.level(1).astype(np.float64)
        gx = np.roll(u, -1, 0) - u
        gy = np.roll(u, -1, 1) - u
        gz = np.roll(np.roll(u, -1, 0), 1, 1) - u
        dk = (u - up) ** 2
        dp = (gx * gx + gy * gy + gz * gz) / 3.0
        if bc != 'periodic':
            dk, dp = dk[1:-1, 1:-1], dp[1:-1, 1:-1]
        want = dk.sum() / (2 * m.p.Co ** 2) + dp.sum()
        got = m.hist['Etri'][-1]
        rel = abs(got - want) / want
        g = rel <= max(acc_tol(), 1e-12 if mc.FP64 else 1e-5) and len(m.hist['Etri']) == len(m.hist['E'])
        good &= g
        det.append(f'{bc}: recorder {got:.6g} vs numpy {want:.6g} (rel {rel:.1e})')
    sq = mc.Membrane(mc.Params(N=65, src='pulse'))
    sq.step(20)
    good &= not sq.hist['Etri']
    ok("9l. the triangular energy series Etri (whole torus on the rhombic torus)", good,
       '; '.join(det) + f'; square lattice Etri entries {len(sq.hist["Etri"])}')
    # (j) beam and packet on the torus and the rigid ring: they run, stay finite, the aperture stays inside
    det, good = [], True
    for bc in ('periodic', 'neumann'):
        for src in ('beam', 'packet'):
            m = mc.Membrane(mc.Params(N=161, lattice='tri', shape='array', bc=bc, src=src, beam_angle=20.0,
                                      freq=0.08, sx=-0.5))
            m.step(200)
            fin = bool(np.isfinite(m.field()).all()) and float(np.abs(m.field()).max()) > 1e-6
            cells = m.beam['cells'] if (src == 'beam' and m.beam) else np.zeros((0, 2), int)
            inside = bool(((cells >= 1) & (cells <= m.N - 2)).all())
            g = fin and inside
            good &= g
            det.append(f'{bc} {src}: finite {fin}' + (f', {len(cells)} aperture cells inside the array' if src == 'beam' else ''))
    ok('9j. beam and packet on the torus and the rigid ring (no aperture wrap)', good, '; '.join(det))
    # (k) bc changed without a reseed: the step rebuilds the triangular array's domain
    m = mc.Membrane(mc.Params(N=97, lattice='tri', shape='array', bc='periodic', src='pulse', sy=-0.9, sigma=4))
    m.step(40)
    ring_before = int(m.domain_mask_np[0, :].sum())
    m.p.bc = 'neumann'
    m.step(40)
    U = m.U.to_numpy()
    ringz = float(np.abs(U[:, 0, :]).max() + np.abs(U[:, :, 0]).max())
    good = ring_before == m.N and int(m.domain_mask_np[0, :].sum()) == 0 and ringz == 0.0 and m.walls_active()
    ok('9k. bc switched without a reseed (torus -> rigid ring): the domain follows', good,
       f'ring cells in the domain {ring_before} -> {int(m.domain_mask_np[0, :].sum())}, ring |u| {ringz:.1e}, '
       f'walls {m.walls_active()}')


def t10_panels(a):
    """10. The alignment model and the ledger on the new modes: no exception; 'array' is aligned on
    every edge (rows) on both lattices; no ledger row FAILs."""
    import alignment_MIRROR as al
    import ledger_MIRROR as lg
    det, good = [], True
    for lat in ('sq', 'tri'):
        m = mc.Membrane(mc.Params(N=97, lattice=lat, shape='array'))
        E = al.al_edges(m)
        L = al.al_list(m)
        note = al.al_note(m)
        rows = al.al_ledger_rows(m)
        n_row = sum(e['cls'] == 'row' for e in E)
        g = len(E) == 4 and n_row == 4 and L and L[0]['rows'] == 4 and all(r[1] != 'FAIL' for r in rows)
        good &= bool(g)
        det.append(f'{lat}: {n_row}/4 edges rows, list {[(e["rot"], e["rows"]) for e in L]}, rows '
                   f'{[r[1] for r in rows]}')
    ok("10a. alignment: 'array' has 4 row edges on both lattices", good, '; '.join(det))
    steps = 300 if a.quick else 1300
    det, good = [], True
    cfgs = [('tri array torus pulse', dict(lattice='tri', shape='array', bc='periodic', src='pulse')),
            ('tri array torus impulse', dict(lattice='tri', shape='array', bc='periodic', src='impulse')),
            ('tri array rigid ring pulse', dict(lattice='tri', shape='array', bc='neumann', src='pulse')),
            ('tri array clamped impulse', dict(lattice='tri', shape='array', src='impulse')),
            ('tri array mur fallback pulse', dict(lattice='tri', shape='array', bc='mur', src='pulse')),
            ('tri array mode', dict(lattice='tri', shape='array', src='mode', mm=2, mn=2)),
            ('sq array torus mode', dict(shape='array', bc='periodic', src='mode')),
            ('tri hexagon rigid edge', dict(lattice='tri', shape='n6', rot=30 * D, edge='rigid', src='pulse')),
            ('sq circle rigid edge mur', dict(shape='circle', rad=0.8, edge='rigid', bc='mur', src='pulse'))]
    for label, prm in cfgs:
        m = mc.Membrane(mc.Params(N=97, **prm))
        m.step(steps)
        rows = lg.build_ledger(m)
        fails = [r.key for r in rows if r.status == lg.FAIL]
        g = not fails
        good &= g
        det.append(f'{label}: {lg.summary(rows)}' + (f' FAIL {fails}' if fails else ''))
    ok('10b. ledger on the new modes: no exception, no FAIL row', good, '; '.join(det))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='cuda')
    ap.add_argument('--f32', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    f32 = a.f32 or a.arch == 'vulkan'
    mc.init_taichi(a.arch, fp64=not f32, quiet=True)
    T0 = time.perf_counter()
    only = {s.strip() for s in a.only.split(',') if s.strip()}
    run = lambda k: not only or k in only
    tol_e = 1e-4 if f32 else 1e-13
    if run('1'):
        t1_torus_energy(a, tol_e)
    if run('2'):
        t2_translation(a)
    if run('3'):
        t3_sixfold(a, 1e-4 if f32 else 1e-12)
    if run('4'):
        t4_rigid_ring(a, tol_e, 1e-5 if f32 else 1e-12)
    if run('5'):
        t5_cfl(a)
    if run('6'):
        t6_sq_array(a)
    if run('7'):
        t7_paint_seam(a)
    if run('8'):
        t8_parity(a)
    if run('9'):
        t9_helpers(a, 1e-5 if f32 else 1e-13)
    if run('10'):
        t10_panels(a)
    print(f'\n{"ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: " + ", ".join(FAILS)} '
          f'({a.arch} {"f32" if f32 else "f64"}, {time.perf_counter() - T0:.1f} s)')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
