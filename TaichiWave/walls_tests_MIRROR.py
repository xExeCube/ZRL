"""Tests for the walls (round 4, 04/10/2026): rigid solids by link removal, corner blocking of the
sq9 diagonals and the triangular links (for pinned solids too), and the rigid domain edge
(Params.edge). The design and every number are in NotesCLAUDE/walls_MIRROR.md; the tests follow
NotesCLAUDE/rigid_vs_pinned_walls_MIRROR.md section 5.5 (tests 1-4 and 6-8; 5 = rule R1 is not
implemented) plus the build's own checks.

    python walls_tests_MIRROR.py [--arch cuda|vulkan|cpu] [--f32] [--quick] [--only 1,4,...]

The reference operators come from ResearchCLAUDE/WallsCLAUDE/walls_lib_MIRROR.py (sparse matrices,
built independently of the Taichi kernels). In f32 (Vulkan) the tolerances are rounding-level
for f32; the exact checks (zeros, bit-identity, incremental == full) are exact on every backend.
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

RES = os.path.join(HERE, 'ResearchCLAUDE', 'WallsCLAUDE')
sys.path.insert(0, RES)
import walls_lib_MIRROR as WL           # noqa: E402  (the research reference operators)
import energy_leak_MIRROR as EL         # noqa: E402  (the hard-case scene)
import circle_eigs_MIRROR as CE         # noqa: E402  (disc geometry, Bessel errors)

D = math.pi / 180
FAILS = []
LATS = (('sq', 5), ('sq', 9), ('tri', 5))
PIN, RIG = 1, 2


def acc_tol():
    """Relative noise of the recorder's sums: f64 accumulators reorder their atomics from run to
    run (~5e-16); Vulkan's are f32 (mc.ACC_DT, the README's two-level sum, ~2e-7)."""
    import taichi as ti
    return 1e-6 if mc.ACC_DT == ti.f32 else 1e-14


def ok(name, cond, det=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('  | ' + det if det else ''), flush=True)
    if not cond:
        FAILS.append(name)


def lname(lat, st):
    return f'{lat}{st}' if lat == 'sq' else 'tri'


def wl_lat(lat, st):
    return WL.TRI if lat == 'tri' else (WL.SQ9 if st == 9 else WL.SQ5)


def all_levels(m):
    U = m.U.to_numpy()
    return [U[(m.slot + k) % 3] for k in (1, 0, 2)]


def gauss_on(m, x0, y0, sig):
    X, Y = m._grid_xy()
    return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sig * sig))


def energy_run(m, steps, every=1000):
    """max |E - E0| / |E0| of the exact leapfrog energy over `steps` steps (operator built once)."""
    L = m.operator()
    E0 = m.leapfrog_energy(L)
    worst = 0.0
    done = 0
    while done < steps:
        k = min(every, steps - done)
        m.step(k)
        done += k
        worst = max(worst, abs(m.leapfrog_energy(L) - E0) / abs(E0))
    return worst, E0


def ext_xx_links(m):
    """Active free-free links whose two common neighbours are both domain exterior (the X-X
    exception of edge = 'pinned'), from the device classes -- 0 means walls_lib's all-blocking
    rule and the core's rule give the same operator."""
    if m.C is None:
        return 0
    C = m.C.to_numpy().astype(np.int64) & 3
    unk = m.unknowns()
    tri = m.tri()
    nine = (not tri) and m.p.stencil == 9
    lat = mc.LAT_TRI if tri else mc.LAT_SQ
    N = m.N
    cnt = 0
    for k, d in enumerate(mc.lat_links(lat, nine)):
        if mc.w_zbit(lat, nine, k) < 0:
            continue
        c1, c2 = mc.link_commons(lat, k)
        I, J = np.nonzero(unk)
        ni, nj = I + d[0], J + d[1]
        ok_ = (ni >= 0) & (nj >= 0) & (ni < N) & (nj < N)
        I, J, ni, nj = I[ok_], J[ok_], ni[ok_], nj[ok_]
        f2 = unk[ni, nj]
        x1 = C[I + c1[0], J + c1[1]] == mc.CL_X
        x2 = C[I + c2[0], J + c2[1]] == mc.CL_X
        cnt += int((f2 & x1 & x2).sum())
    return cnt


# ---------------------------------------------------------------------------------------------
def t1_operator(a, tol):
    """1. The kernel + link words == walls_lib.build_L (mixed pinned/rigid hard-case scene)."""
    s = EL.scene(97)
    T = EL.mixed_types(s).astype(np.uint8)                 # left objects pinned, right rigid, ring pinned
    cases = []
    for lat, st in LATS:
        if lat == 'sq':
            cases.append((lat, st, 'full square, Dirichlet ring = pinned', dict(), 'pinned'))
            cases.append((lat, st, 'circle, rigid edge', dict(shape='circle', rad=0.97), 'rigid'))
        else:
            cases.append((lat, st, 'circle, rigid edge', dict(shape='circle', rad=1.0), 'rigid'))
            cases.append((lat, st, 'hexagon at 30 deg, rigid edge', dict(shape='n6', rot=30 * D), 'rigid'))
    for lat, st, label, extra, edge in cases:
        lw = wl_lat(lat, st)
        co = 0.9 * lw.co_max
        m = mc.Membrane(mc.Params(N=97, lattice=lat, stencil=st, src='impulse', Co=co, edge=edge, **extra))
        m.record = False
        m.set_solid_np(T)
        n = m.N
        Tl = T.astype(np.int8).copy()
        ext = m.domain_mask_np == 0
        if lat == 'tri':
            ext = ext.copy()
            ext[0, :] = ext[-1, :] = ext[:, 0] = ext[:, -1] = True
        Tl[ext] = RIG if edge == 'rigid' else PIN
        L = WL.build_L(Tl, lw, block='both')
        free = Tl == WL.FREE
        same_free = np.array_equal(free, m.unknowns())
        Lm = m.operator()
        idx = np.flatnonzero(free.ravel())
        dL = abs((Lm - L)[idx][:, idx]).max() if idx.size else 0.0
        X, Y = WL.grid_xy(lw, n, n, (n - 1) // 2 + 3, (n - 1) // 2 - 5)
        u0 = np.exp(-(X * X + Y * Y) / 18) * free
        u1 = u0 + 0.5 * co * co * (L @ u0.ravel()).reshape(n, n)
        A, B = u0.ravel().copy(), u1.ravel().copy()
        for _ in range(32):
            A, B = B, 2 * B - A + co * co * (L @ B)
        ref = B.reshape(n, n)
        m.set_levels(u0, u1)
        m.step(32)
        got = m.field().astype(np.float64)
        rel = float(np.abs(got - ref).max() / np.abs(ref).max())
        ok(f'1. operator == walls_lib.build_L: {lname(lat, st)} {label}',
           same_free and dL < 1e-12 and rel <= tol and m.walls_active(),
           f'N={n}, Co {co:.3f}: free cells {"=" if same_free else "!="}; max|L - L_ref| {dL:.1e}; '
           f'32 steps: max|u - u_ref| / max|u_ref| = {rel:.1e} (tol {tol:.0e}); '
           f'{m.n_wall_cells} cells with non-plain links')


def t2_energy(a, tol_e):
    """2. The exact leapfrog energy is conserved with pinned + rigid objects mixed (and with
    corner blocking, a medium, a Neumann ring and a torus seam)."""
    steps = 4000 if a.quick else 20000
    s = EL.scene(97)
    T = EL.mixed_types(s).astype(np.uint8)
    Tin = T.copy()
    Tin[0, :] = Tin[-1, :] = Tin[:, 0] = Tin[:, -1] = 0          # objects only (the ring is the bc's)
    Tring = Tin.copy()                                         # objects that touch / cross the ring
    Tring[0:4, 30:42] = RIG
    Tring[93:97, 50:58] = PIN
    Tring[40:50, 0:3] = RIG
    Tring[60:64, 94:97] = RIG
    Tseam = Tring.copy()
    Tseam[0:2, 70:80] = PIN
    Tseam[95:97, 70:80] = PIN
    for k in range(10):
        Tseam[(90 + k) % 97, (5 + k) % 97] = RIG                # a one-cell diagonal across the corner seam
    cases = [('sq5 full Dirichlet', 'sq', 5, dict(), T),
             ('sq9 full Dirichlet', 'sq', 9, dict(), T),
             ('tri circle (edge pinned)', 'tri', 5, dict(shape='circle', rad=1.0), T),
             ('sq9 full Neumann + slab medium, solids on the ring', 'sq', 9, dict(bc='neumann', medium='slab'), Tring),
             ('sq5 full Neumann + lens, solids on the ring', 'sq', 5, dict(bc='neumann', medium='lens'), Tring),
             ('sq9 torus (periodic), solids across the seam', 'sq', 9, dict(bc='periodic'), Tseam),
             ('tri hexagon, edge rigid + lens', 'tri', 5, dict(shape='n6', rot=30 * D, edge='rigid', medium='lens'), T)]
    for label, lat, st, extra, TT in cases:
        lw = wl_lat(lat, st)
        m = mc.Membrane(mc.Params(N=97, lattice=lat, stencil=st, src='impulse', Co=0.95 * lw.co_max, **extra))
        m.record = False
        m.set_solid_np(TT)
        u = gauss_on(m, 6.0, -4.0, 3.0) * m.unknowns()
        m.set_levels(u, u)
        drift, E0 = energy_run(m, steps)
        ok(f'2. energy conserved: {label}', drift <= tol_e and m.walls_active() and E0 > 0,
           f'N={m.N}, Co = 0.95 C_max, {steps} steps: max|E - E0|/E0 = {drift:.1e} (tol {tol_e:.0e}); '
           f'{int((TT == PIN).sum())} pinned + {int((TT == RIG).sum())} rigid cells; '
           f'{m.n_wall_cells} cells with non-plain links')


def t3_leak(a):
    """3. Exactly 0.0 through one-cell straight and diagonal walls, both types, every lattice."""
    steps = 600 if a.quick else 1500
    for lat, st in LATS:
        for typ, tname in ((PIN, 'pinned'), (RIG, 'rigid')):
            det, good = [], True
            for wall in ('straight', 'diagonal', 'anti-diagonal'):
                sxy = {'straight': (0.2, -0.45), 'diagonal': (0.45, -0.45), 'anti-diagonal': (-0.45, -0.45)}[wall]
                prm = dict(N=257, lattice=lat, stencil=st, src='cont', freq=0.05, sx=sxy[0], sy=sxy[1], Co=0.5)
                if lat == 'tri':
                    prm.update(shape='n6', rot=30 * D)
                m = mc.Membrane(mc.Params(**prm))
                m.record = False
                N = m.N
                c = (N - 1) // 2
                I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
                line = {'straight': J == c, 'diagonal': I == J, 'anti-diagonal': I + J == N - 1}[wall]
                far = {'straight': J > c, 'diagonal': J > I, 'anti-diagonal': I + J > N - 1}[wall]
                m.set_solid_np(np.where(line, typ, 0).astype(np.uint8))
                m.reseed()
                far &= m.mask_np == 1
                adj = np.zeros_like(line)
                for di, dj in m.neighbour_offsets():
                    adj |= np.roll(np.roll(line, di, 0), dj, 1)
                hug = adj & ~far & (m.mask_np == 1)                  # free cells touching the wall, source side
                src_far = bool(far[m.src[0], m.src[1]])
                peak, pnear = 0.0, 0.0
                for _ in range(5):
                    m.step(steps // 5)
                    for L_ in all_levels(m):
                        peak = max(peak, float(np.abs(L_[far]).max()))
                    pnear = max(pnear, float(np.abs(m.field()[hug]).max()))
                g = peak == 0.0 and not src_far and pnear > 1e-6
                good &= g
                det.append(f'{wall} {peak:.1e} (near the wall {pnear:.1e})')
            ok(f'3. no leak through one-cell walls: {lname(lat, st)} {tname}', good,
               f'N=257, driven cell, {steps} steps, max|u| beyond the wall over 3 levels: ' + '; '.join(det))


def t4_reflection(a, tol):
    """4. An axis wall reflects a plane pulse exactly like the image solution: rigid = even mirror
    on the RASTERISATION line (half-way between the last free and the first solid row), R = +1;
    pinned = odd mirror on the solid row's CENTRES, R = -1."""
    for lat, st in LATS:
        N = 513
        prm = dict(N=N, lattice=lat, stencil=st, src='impulse', Co=0.5)
        if lat == 'tri':
            prm.update(shape='circle', rad=1.0)
        c = (N - 1) // 2
        jw = c + 20
        h = mc.S3H if lat == 'tri' else 1.0
        sig, T = 3.0, 100
        det, good = [], True
        for typ, tname, sgn, yw in ((RIG, 'rigid', 1.0, (jw - c - 0.5) * h), (PIN, 'pinned', -1.0, (jw - c) * h)):
            runs = {}
            for kind in ('wall', 'image', 'free'):
                m = mc.Membrane(mc.Params(**prm))
                m.record = False
                if kind == 'wall':
                    S = np.zeros((N, N), np.uint8)
                    S[:, jw:] = typ
                    m.set_solid_np(S)
                X, Y = m._grid_xy()
                y0 = -5.0 * h
                u = np.exp(-(Y - y0) ** 2 / (2 * sig * sig))
                if kind != 'free':
                    # the walled run starts from the image run's data on its free side (on tri the
                    # image pulse's tail there is ~5e-12, not negligible: rows are 0.866 apart)
                    u = u + sgn * np.exp(-(Y - (2 * yw - y0)) ** 2 / (2 * sig * sig))
                u = u * (m.mask_np != 0)
                m.set_levels(u, u)
                m.step(T)
                runs[kind] = m.field().astype(np.float64)
                if kind == 'wall':
                    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
                    reg = (np.abs(X) <= 100) & (J < jw) & (J > c - 80) & (m.mask_np != 0)
            dAB = float(np.abs(runs['wall'] - runs['image'])[reg].max() / np.abs(runs['image'][reg]).max())
            refl = (runs['wall'] - runs['free'])[reg]
            k = int(np.argmax(np.abs(refl)))
            got_sgn = 1.0 if refl[k] > 0 else -1.0
            g = dAB <= tol and got_sgn == sgn
            good &= g
            det.append(f'{tname}: |wall - image| {dAB:.1e}, reflected peak {refl[k]:+.4f} (R {"+" if got_sgn > 0 else "-"}1)')
        ok(f'4. axis-wall reflection = image solution, sign and line: {lname(lat, st)}', good,
           f'N={N}, plane pulse sigma {sig} at 5 rows from the wall row, {T} steps, compared within |x| <= 100: '
           + '; '.join(det))


def t6_cfl(a):
    """6. Stable at 0.995 C_max with dense random solids (random pinned / rigid, 25% of the cells)."""
    steps = 4000 if a.quick else 20000
    rng = np.random.default_rng(11)
    for lat, st in LATS:
        prm = dict(N=129, lattice=lat, stencil=st, src='impulse')
        if lat == 'tri':
            prm.update(shape='n6', rot=30 * D)
        m0 = mc.Membrane(mc.Params(**prm))
        co = 0.995 * m0.co_limit()
        m = mc.Membrane(mc.Params(Co=co, **prm))
        m.record = False
        N = m.N
        S = np.where(rng.random((N, N)) < 0.25, rng.integers(1, 3, (N, N)), 0).astype(np.uint8)
        m.set_solid_np(S)
        u = rng.standard_normal((N, N)) * m.unknowns()
        m.set_levels(u, u)
        u0max = float(np.abs(u).max())
        drift, E0 = energy_run(m, steps, every=steps // 4)
        umax = float(np.abs(m.field()).max())
        ok(f'6. stable at 0.995 C_max, random solids: {lname(lat, st)}',
           np.isfinite(umax) and umax < 10 * u0max and drift < 1e-6,
           f'N={N}, Co {co:.4f}, {int((S == PIN).sum())} pinned + {int((S == RIG).sum())} rigid cells, '
           f'{m.n_wall_cells} non-plain; noise start: max|u| {u0max:.2f} -> {umax:.2f} after {steps} steps, '
           f'energy drift {drift:.1e}')


def t7_circle(a):
    """7. Disc eigenfrequencies: the core's operator (from the device words) has the eigenvalues
    of walls_lib's, and R |err| lies in the research note's section 3.8 bands (a regression
    guard, not an accuracy target)."""
    import scipy.sparse.linalg as sla
    c4 = {k: CE.c4_of(v) for k, v in WL.LATS.items()}
    # R |err| upper bounds of the 3.8 table (mode 1; cell-centred / offset), + 0.02
    hi = {('sq5', 'P'): (0.33, 0.38), ('sq9', 'P'): (0.36, 0.42), ('tri', 'P'): (0.35, 0.36),
          ('sq5', 'R'): (0.06, 0.14), ('sq9', 'R'): (0.07, 0.14), ('tri', 'R'): (0.11, 0.11)}
    radii = (16,) if a.quick else (16, 32)
    for lat, st in LATS:
        ln = lname(lat, st)
        lw = wl_lat(lat, st)
        det, good = [], True
        for typ, rule, rn in ((PIN, 'Pcb', 'P'), (RIG, 'R0', 'R')):
            for oi, off in enumerate(((0.0, 0.0), (0.31, 0.17))):
                for R in radii:
                    solid, _, _ = CE.disk(lw, R, off)
                    n = solid.shape[0]
                    prm = dict(N=n, lattice=lat, stencil=st, src='impulse')
                    if lat == 'tri':
                        prm.update(shape='circle', rad=1.0)
                    m = mc.Membrane(mc.Params(**prm))
                    m.set_solid_np(np.where(solid, typ, 0).astype(np.uint8))
                    free = ~solid
                    if not np.array_equal(free, m.unknowns()):
                        good = False
                        det.append(f'{rn} R={R}: free cells differ')
                        continue
                    k8 = CE.low_eigs(m.operator(), free, k=8, sigma=-1e-3)
                    kr = CE.low_eigs(WL.op_for(solid, lw, rule), free, k=8, sigma=-1e-3)
                    same = float(np.abs(k8 - kr).max())
                    if rn == 'P':
                        e1 = CE.errs_dir(k8, R, c4[ln])[0]
                    else:
                        e1 = CE.errs_neu(k8, R, c4[ln])[0]
                    re = R * abs(e1)
                    g = same < 1e-9 and re <= hi[(ln, rn)][oi] + 0.02
                    good &= g
                    det.append(f'{rn} {"off" if oi else "ctr"} R={R}: R|err| {re:.3f}{"" if g else " (!)"}')
        ok(f'7. disc eigenfrequencies in the 3.8 bands: {ln}', good, '; '.join(det))


def t8_cavity(a, tol, tol_long):
    """8. A closed rigid cavity (a ring with a counter) keeps the DC of a seed at rest.
    Its mean is a ZERO MODE: nothing restores it, so the rounding noise of every step is
    integrated twice (a random walk of the mean velocity) and the drift grows ~ n^1.5. No
    floating-point leapfrog keeps it to 1e-14 for long: walls_lib's numpy run of this very cavity
    drifts 9.3e-13 (sq5), 1.2e-9 (sq9), 9.0e-10 (tri) in 5000 steps (04/10/2026, both with
    systematic weight-sum residues). The rounding walk predicts ~1e-14 after 100 steps for 64 cells
    (~8 eps of the sum per step, integrated twice): so <= tol (3e-14, f64) after 100 steps and
    <= tol_long (1e-10) over 5000 steps. The requested 1e-14 sits AT that noise (measured 5.6e-15 to
    1.3e-14 after 100 steps, 04/10/2026; see NotesCLAUDE/walls_MIRROR.md)."""
    steps = 2000 if a.quick else 5000
    for lat, st in LATS:
        prm = dict(N=97, lattice=lat, stencil=st, src='impulse', Co=0.5)
        if lat == 'tri':
            prm.update(shape='circle', rad=1.0)
        m = mc.Membrane(mc.Params(**prm))
        m.record = False
        N = m.N
        S = np.zeros((N, N), np.uint8)
        S[40:56, 40:56] = RIG
        S[44:52, 44:52] = 0                                   # the counter: 8 x 8 free cells
        S[20:24, 60:70] = PIN                                 # a pinned object outside
        m.set_solid_np(S)
        cav = np.zeros((N, N), bool)
        cav[44:52, 44:52] = True
        X, Y = m._grid_xy()
        cx, cy = m.cell_xy(47, 48)
        u = np.where(cav, 1.0 + 0.5 * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / 4), 0.0)
        u += gauss_on(m, -20.0, -15.0, 3.0) * m.unknowns()
        m.set_levels(u, u)
        dc0 = float(u[cav].sum())
        m.step(100)
        short = abs(float(m.field().astype(np.float64)[cav].sum()) - dc0) / abs(dc0)
        worst = short
        done = 100
        for k in range(5):
            m.step((k + 1) * steps // 5 - done)
            done = (k + 1) * steps // 5
            worst = max(worst, abs(float(m.field().astype(np.float64)[cav].sum()) - dc0) / abs(dc0))
        moved = float(np.abs(m.field()[cav] - u[cav]).max())
        ok(f'8. rigid cavity keeps its DC: {lname(lat, st)}', short <= tol and worst <= tol_long and moved > 1e-3,
           f'N={N}, 64-cell counter inside a rigid ring, seed at rest (mean {dc0 / 64:.4f}): '
           f'|sum u - sum u0| / sum u0 = {short:.1e} after 100 steps (tol {tol:.0e}), max {worst:.1e} over '
           f'{steps} steps (tol {tol_long:.0e}, the rounding walk of the zero mode); the bump moved by {moved:.2e}')


def t9_incremental(a):
    """Incremental painting (objects and strokes of both types, erasers, across the array ring /
    the torus seam / the domain edge) == the full rebuild: type map, mask, class codes and link
    words, and the counters."""
    cases = [('sq9 Dirichlet', dict(stencil=9)), ('sq5 Neumann', dict(bc='neumann')),
             ('sq9 Neumann', dict(stencil=9, bc='neumann')), ('sq9 torus', dict(stencil=9, bc='periodic')),
             ('tri hexagon', dict(lattice='tri', shape='n6', rot=30 * D)),
             ('tri circle, edge rigid', dict(lattice='tri', shape='circle', rad=0.9, edge='rigid')),
             ('sq9 circle, edge rigid', dict(stencil=9, shape='circle', rad=0.9, edge='rigid'))]
    for label, extra in cases:
        m = mc.Membrane(mc.Params(N=257, src='pulse', sigma=3, sx=0.1, **extra))
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
        m.update_solids(sc.add(so.SolidObject('circle', x=0.3, y=0.2, r=0.08, wall='rigid'))[1])
        check('rigid circle')
        m.update_solids(sc.add(so.SolidObject('rect', x=0.95, y=-0.3, w=0.2, h=0.15, wall='pinned'))[1])
        check('pinned rect on the ring')
        m.update_solids(sc.add(so.SolidObject('ngon', x=-0.99, y=0.5, r=0.1, n=5, wall='rigid'))[1])
        check('rigid pentagon on the ring')
        for wall, pts in (('rigid', [(-0.6, -0.99), (0.2, 0.1), (0.99, 0.99), (0.4, -0.7)]),
                          ('pinned', [(-0.99, 0.2), (0.6, -0.2), (0.1, 0.99)])):
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
        zero = max(float(np.abs(L_[solid]).max()) for L_ in all_levels(m)) if solid.any() else 0.0
        ok(f'incremental painting == full rebuild: {label}', not bad and zero == 0.0 and m.walls_active(),
           f'N={m.N}: {nup} updates compared (type map, device mask, classes, words, counters): '
           + ('all equal' if not bad else '; '.join(bad[:4])) + f'; solids after 40 more steps max|u| {zero:.1e}; '
           f'{int((m.solid_np == 1).sum())} pinned + {int((m.solid_np == 2).sum())} rigid cells')


def load_before():
    path = os.path.join(HERE, 'ScratchCLAUDE', 'Round4CLAUDE', 'BeforeCLAUDE', 'membrane_core_MIRROR.py')
    spec = importlib.util.spec_from_file_location('membrane_core_before_MIRROR', path)
    B = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    spec.loader.exec_module(B)                # its 'import beam_MIRROR' finds the loaded module
    sys.path[:] = saved
    B.MASK_DT, B.SQ9_CONSISTENT, B.FP64, B.ACC_DT = mc.MASK_DT, mc.SQ9_CONSISTENT, mc.FP64, mc.ACC_DT
    return B


def t10_parity(a):
    """The web app's path is untouched: no solids, pinned solids that block no link, and
    edge = 'rigid' on the full square (no mask edge) give fields, peak and probe histories
    bit-identical to the core of 03/10/2026 (ScratchCLAUDE/Round4CLAUDE/BeforeCLAUDE). The energy
    history is compared to 1e-14 relative: its sum goes through GPU/CPU atomics, whose order varies
    from run to run -- the SAME core differs from itself by up to 5.5e-16 (measured 04/10/2026)."""
    B = load_before()
    steps = 200 if a.quick else 400

    def objs():
        sc = so.SolidScene()
        sc.add_rect(0.3, 0.2, 0.2, 0.1)
        sc.add_circle(-0.4, -0.3, 0.12)
        sc.add_rect(-0.2, 0.5, 0.3, 0.06)
        return sc
    cases = [('sq5 full Dirichlet', dict(src='pulse', sx=0.2), None, None),
             ('sq9 full Neumann corner', dict(src='pulse', bc='neumann', stencil=9, sx=-0.8, sy=-0.8, sigma=2), None, None),
             ('sq9 circle Mur, driven', dict(shape='circle', rad=0.8, stencil=9, bc='mur', src='cont', freq=0.06), None, None),
             ('sq5 n5 slit', dict(shape='n5', rot=12 * D, src='slit', freq=0.06), None, None),
             ('tri hexagon pulse', dict(lattice='tri', shape='n6', rot=30 * D, src='pulse'), None, None),
             ('tri circle lens', dict(lattice='tri', shape='circle', rad=1.3, medium='lens', src='pulse', sx=0.4), None, None),
             ('sq9 full, pinned objects', dict(stencil=9, src='pulse', sx=0.1), objs, None),
             ('tri hexagon, pinned objects', dict(lattice='tri', shape='n6', rot=30 * D, src='pulse', sx=0.1), objs, None),
             ('sq5 Unlicensed plate (pinned)', dict(shape='unlicensed', src='pulse', sy=-0.9), None, None),
             ('sq9 full, edge rigid (no mask edge)', dict(stencil=9, src='pulse', sx=0.3), None, 'rigid')]
    for label, prm, mk, edge in cases:
        res = []
        for mod in (mc, B):
            m = mod.Membrane(mod.Params(N=257, **prm))
            if edge is not None and mod is mc:
                m.set_edge(edge)
            if mk is not None:
                m.attach_scene(mk())
                m.reseed()
            m.step(steps)
            h = m.hist
            res.append((m.field(), np.array(h['E']), np.array(h['amax'] + h['pA'] + h['pB']),
                        m.walls_active() if mod is mc else None, m.has_solids))
        (u1, E1, P1, act, hs1), (u2, E2, P2, _, hs2) = res
        fsame = np.array_equal(u1, u2)
        psame = np.array_equal(P1, P2)
        dE = float(np.abs(E1 - E2).max() / np.abs(E2).max()) if len(E1) == len(E2) and len(E2) else 1.0
        ok(f'parity with the 03/10 core: {label}', fsame and psame and dE <= acc_tol() and len(E1) == steps
           and not act and hs1 == hs2,
           f'N=257, {steps} steps: field {"bit-identical" if fsame else "DIFFERS"}, peak + probe histories '
           f'{"bit-identical" if psame else "DIFFER"}, energy history max rel diff {dE:.1e} ({len(E1)} rows); '
           f'walls kernel {"ON" if act else "off"}; solids {hs1}')


def t11_edge(a, tol_e):
    """edge = 'rigid' on masked shapes: the exact energy is conserved, the field differs from the
    pinned edge, and the plate (pinned lettering) mixes with it."""
    steps = 4000 if a.quick else 20000
    cases = [('sq5 circle', dict(shape='circle', rad=0.8)),
             ('sq9 pentagon at 17 deg', dict(stencil=9, shape='n5', rot=17 * D)),
             ('sq9 square r 0.7', dict(stencil=9, shape='square', rad=0.7)),
             ('tri circle', dict(lattice='tri', shape='circle', rad=0.9)),
             ('tri hexagon', dict(lattice='tri', shape='n6', rot=30 * D)),
             ('tri triangle at 30 deg', dict(lattice='tri', shape='n3', rot=30 * D)),
             ('sq9 Unlicensed, pinned plate', dict(stencil=9, shape='unlicensed', sy=-0.9)),
             ('tri Unlicensed r 0.85, rigid plate', dict(lattice='tri', shape='unlicensed', rad=0.85, sy=-0.9))]
    for label, extra in cases:
        prm = dict(N=161, src='pulse', sigma=3, Co=0.6, **extra)
        mp = mc.Membrane(mc.Params(**prm))
        mr = mc.Membrane(mc.Params(edge='rigid', **prm))
        if 'rigid plate' in label:
            mr.scene.set_preset(plate_wall='rigid')
            mr.update_solids(None)
        mr.record = False
        drift, E0 = energy_run(mr, steps, every=steps // 4)
        mp.step(steps)
        diff = float(np.abs(mp.field() - mr.field()).max())
        ok(f'edge rigid conserves energy: {label}', drift <= tol_e and mr.walls_active() and diff > 1e-3,
           f'N={mr.N}, {steps} steps: max|E - E0|/E0 = {drift:.1e} (tol {tol_e:.0e}); '
           f'{mr.n_wall_cells} non-plain cells; max|u_rigid - u_pinned| {diff:.2e}')


def t12_recorder(a):
    """The recorder on the walls path: a constant field in a closed RIGID box stores no potential
    (E = 0 exactly: the forward differences into the wall are dropped links), a pinned box does;
    stats() equals the recorder's last row."""
    for lat, st in LATS:
        res = {}
        for typ, tname in ((RIG, 'rigid'), (PIN, 'pinned')):
            prm = dict(N=97, lattice=lat, stencil=st, src='impulse', Co=0.5)
            if lat == 'tri':
                prm.update(shape='circle', rad=1.0)
            m = mc.Membrane(mc.Params(**prm))
            N = m.N
            S = np.zeros((N, N), np.uint8)
            S[30:62, 30:62] = typ
            S[33:59, 33:59] = 0
            m.set_solid_np(S)
            box = np.zeros((N, N), bool)
            box[33:59, 33:59] = True
            u = np.where(box, 1.0, 0.0)
            m.set_levels(u, u)
            m.step(8)
            E_rec = m.hist['E'][-1]
            E_st, _ = m.stats()
            res[tname] = (E_rec, E_st, float(np.abs(m.field()[box] - 1.0).max()))
        (er, sr, dr), (ep, sp_, dp) = res['rigid'], res['pinned']
        # sq9 in f64 keeps the web app's operation order, (2/3)e + (1/6)dg - (10/3)u, which leaves
        # -4.4e-16 u on a constant field (measured 04/10/2026), so the rigid box's E is ~1e-29, not 0
        same = abs(sr - er) <= acc_tol() * abs(ep) and abs(sp_ - ep) <= acc_tol() * abs(ep)
        ok(f'recorder on the walls path: {lname(lat, st)}', er <= 1e-20 * ep and dr < 1e-12 and ep > 1.0 and same,
           f'constant 1 in a 26 x 26 box after 8 steps: rigid box E = {er:.1e} (field moved {dr:.1e}), '
           f'pinned box E = {ep:.3g}; stats() == recorder: {same}')


def t13_main(a):
    """main_region() under corner blocking: a counter sealed ONLY by blocked links is not part of
    the main region (sq9: a diamond of one-cell diagonal lines; tri: a box whose vertical sides
    are 'dotted' -- a solid cell every second row -- sealed by blocked (1,0) links, the refined
    labelling path), checked against the connected components of the operator's own links; a
    source requested on the wall snaps to the main region, never into the counter."""
    import scipy.sparse.csgraph as csg
    for lat, st in (('sq', 9), ('tri', 5)):
        for typ, tname in ((PIN, 'pinned'), (RIG, 'rigid')):
            prm = dict(N=129, lattice=lat, stencil=st, src='cont', freq=0.05)
            if lat == 'tri':
                prm.update(shape='circle', rad=1.0)
            m = mc.Membrane(mc.Params(**prm))
            N = m.N
            c = (N - 1) // 2
            S = np.zeros((N, N), np.uint8)
            inside = np.zeros((N, N), bool)
            if lat == 'sq':
                r = 12
                I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
                d1 = np.abs(I - c) + np.abs(J - c)
                S[d1 == r] = typ
                inside = d1 < r
            else:
                # full solid rows at j = c -+ 10 (top, bottom); the vertical sides x = -+8 are
                # "dotted": a solid cell only in the rows with j - c even, i = c + x - (j - c)/2. In
                # the rows between, the (1,0) link from x = 7.5 to 8.5 passes between two solid
                # cells sqrt(3) apart (its common neighbours) -- blocked, and nothing else crosses
                jlo, jhi = c - 10, c + 10
                for j in range(jlo, jhi + 1):
                    if (j - c) % 2 == 0:
                        for xs in (-8, 8):
                            S[c + xs - (j - c) // 2, j] = typ
                X, Y = m._grid_xy()
                I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
                S[((J == jlo) | (J == jhi)) & (np.abs(X) <= 8 + 1e-9)] = typ
                inside = (J > jlo) & (J < jhi) & (np.abs(X) < 8 - 1e-9)
            m.set_solid_np(S)
            inside &= m.mask_np != 0
            main = m.main_region()
            L = m.operator()
            unk = m.unknowns()
            idx = np.flatnonzero(unk.ravel())
            A = L[idx][:, idx].copy()
            A.setdiag(0)
            A.eliminate_zeros()
            ncomp, lab = csg.connected_components(A, directed=False)
            big = np.bincount(lab).argmax()
            ref = np.zeros(N * N, bool)
            ref[idx[lab == big]] = True
            ref = ref.reshape(N, N)
            same = main is not None and np.array_equal(main, ref)
            sealed = inside.sum() > 20 and not (main & inside).any()
            # the same counter without blocking would have joined the plain 8 / 6-neighbour labelling
            from scipy import ndimage
            lab8, _ = ndimage.label(unk, structure=m.stencil_structure())
            joined = len(np.unique(lab8[inside])) == 1 and lab8[inside][0] == np.bincount(lab8[unk]).argmax()
            # a source requested ON the wall next to the counter
            wi, wj = np.argwhere(S != 0)[0]
            x, y = m.cell_xy(wi, wj)
            ir = m.inner_R()
            m.p.sx, m.p.sy = x / ir, y / ir
            m.recache_source()
            snap_ok = bool(main[m.src[0], m.src[1]]) and m.src_moved
            path = 'refined' if (lat == 'tri' and m.n_blk4 > 0) else '4-neighbour + corner merge'
            ok(f'main region under corner blocking: {lname(lat, st)} {tname}', same and sealed and snap_ok,
               f'N={N}: {int(inside.sum())}-cell counter sealed only by blocked links (plain labelling joins it: '
               f'{joined}); main region == operator components: {same} ({path}, blk4 {m.n_blk4}); '
               f'source on the wall -> {m.src[:2]} in the main region: {snap_ok}')


def t14_misc(a):
    """Odds and ends: a source requested in a rigid solid moves out; the Co change with
    rescale on the walls path; Mur + rigid objects; set_edge without a reset."""
    m = mc.Membrane(mc.Params(N=161, stencil=9, src='cont', sx=0.3, sy=0.1, freq=0.05))
    want = m.src[:2]
    sc = so.SolidScene()
    sc.add(so.SolidObject('circle', *so.cell_to_grid(m, *want), r=0.06, wall='rigid'))
    m.attach_scene(sc)
    moved = m.src_moved and m.mask_np[m.src[0], m.src[1]] == 1 and m.src[:2] != want
    m.step(200)
    m.set_co(0.7, rescale=True)
    m.step(200)
    zero = max(float(np.abs(L_[m.solid_np != 0]).max()) for L_ in all_levels(m))
    fin = bool(np.isfinite(m.field()).all())
    mm = mc.Membrane(mc.Params(N=161, stencil=9, bc='mur', src='pulse', sx=0.4))
    sc2 = so.SolidScene()
    sc2.add(so.SolidObject('rect', x=-0.3, y=0.2, w=0.3, h=0.04, rot_deg=45, wall='rigid'))
    sc2.add_stroke([(-0.9, -0.95), (0.9, -0.99)], so.cells_to_grid(mm, 1.0), wall='rigid')
    mm.attach_scene(sc2)
    mm.step(1500)
    zm = max(float(np.abs(L_[mm.solid_np != 0]).max()) for L_ in all_levels(mm))
    me = mc.Membrane(mc.Params(N=161, shape='circle', rad=0.8, src='pulse', sx=0.3))
    me.step(100)
    u_before = me.field()
    me.set_edge('rigid')
    kept = np.array_equal(u_before, me.field()) and me.walls_active()
    me.set_edge('pinned')
    back = not me.walls_active() and me.W is None
    ok('rigid snap, Co rescale, Mur + rigid, set_edge', moved and zero == 0.0 and fin and zm == 0.0 and kept and back,
       f'source in a rigid disc moved {want} -> {m.src[:2]}; after set_co(0.7, rescale) 200 steps: solids max|u| '
       f'{zero:.1e}, finite {fin}; Mur + rigid stroke on the ring 1500 steps: solids {zm:.1e}; set_edge kept the '
       f'field: {kept}, back to pinned frees the words: {back}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='cuda')
    ap.add_argument('--f32', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    f32 = a.f32 or a.arch == 'vulkan'
    mc.init_taichi(a.arch, fp64=not f32, quiet=True)
    T0 = time.perf_counter()
    only = {s.strip() for s in a.only.split(',') if s.strip()}
    run = lambda k: not only or k in only
    tol_op = 1e-5 if f32 else 1e-13
    tol_e = 1e-4 if f32 else 1e-13
    tol_img = 1e-5 if f32 else 1e-12
    tol_dc = 3e-5 if f32 else 3e-14
    if run('1'):
        t1_operator(a, tol_op)
    if run('2'):
        t2_energy(a, tol_e)
    if run('3'):
        t3_leak(a)
    if run('4'):
        t4_reflection(a, tol_img)
    if run('6'):
        t6_cfl(a)
    if run('7'):
        t7_circle(a)
    if run('8'):
        t8_cavity(a, tol_dc, 3e-2 if f32 else 1e-10)
    if run('9'):
        t9_incremental(a)
    if run('10'):
        t10_parity(a)
    if run('11'):
        t11_edge(a, 1e-4 if f32 else 1e-13)
    if run('12'):
        t12_recorder(a)
    if run('13'):
        t13_main(a)
    if run('14'):
        t14_misc(a)
    print(f'\n{"ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: " + ", ".join(FAILS)} '
          f'({a.arch} {"f32" if f32 else "f64"}, {time.perf_counter() - T0:.1f} s)')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
