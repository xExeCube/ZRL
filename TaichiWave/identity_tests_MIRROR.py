"""Identity tests for the Taichi port: properties that hold EXACTLY (or to a closed form),
checked at grid sizes the web app cannot reach. Complements crosscheck_js_MIRROR.py, which
checks agreement with the web app at the web app's own sizes.

    python identity_tests_MIRROR.py [--arch cuda|vulkan|cpu] [--f32] [--big N]
"""
import argparse, math, sys, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import membrane_core_MIRROR as mc

D = math.pi / 180
FAILS = []


def ok(name, cond, det=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('  | ' + det if det else ''))
    if not cond:
        FAILS.append(name)


def offsets(m):
    """Lattice offsets (a, b) of every cell from the centre, shape (N, N)."""
    c = (m.N - 1) // 2
    I, J = np.meshgrid(np.arange(m.N) - c, np.arange(m.N) - c, indexing='ij')
    return I, J


def lap_np(u, m):
    """The solver's Laplacian in numpy (Dirichlet ring = zero padding)."""
    p = np.pad(u, 1)
    l, r, d, t = p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
    if m.tri():
        return (2 / 3) * (l + r + d + t + p[2:, :-2] + p[:-2, 2:] - 6 * u)
    if m.p.stencil == 9:
        dg = p[:-2, :-2] + p[2:, :-2] + p[:-2, 2:] + p[2:, 2:]
        return (2 / 3) * (l + r + d + t - 4 * u) + (1 / 6) * (dg - 4 * u)
    return l + r + d + t - 4 * u


def lap_full(u, nine):
    """The square-lattice Laplacian on the interior [1:-1, 1:-1], reading the array's OWN ring
    (Neumann ghost cells), in the web app's operation order."""
    c = u[1:-1, 1:-1]
    l, r, d, t = u[:-2, 1:-1], u[2:, 1:-1], u[1:-1, :-2], u[1:-1, 2:]
    if nine:
        e = l + r + d + t
        dg = u[:-2, :-2] + u[2:, :-2] + u[:-2, 2:] + u[2:, 2:]
        return (2.0 / 3.0) * e + (1.0 / 6.0) * dg - (10.0 / 3.0) * c
    return l + r + d + t - 4 * c


def np_neumann_run(seed, cc, nine, K, old_corner, keep_prev=False):
    """K steps of the web app's full-square Neumann solver in numpy, with its three rotating
    buffers (U2 starts as zeros, clearField). old_corner=True reproduces the sequential ring
    loop: (0,0) takes (1,0) from the buffer BEFORE this step's ring writes (an older level);
    False: every corner = its diagonal interior neighbour of this step (the port fix)."""
    A, Bf, C = seed.astype(np.float64).copy(), seed.astype(np.float64).copy(), np.zeros_like(seed, dtype=np.float64)
    for _ in range(K):
        C[1:-1, 1:-1] = (2 * Bf[1:-1, 1:-1] - A[1:-1, 1:-1] + cc * lap_full(Bf, nine)) - 0.0 * (Bf[1:-1, 1:-1] - A[1:-1, 1:-1])
        stale = C[1, 0]
        C[1:-1, 0] = C[1:-1, 1]
        C[1:-1, -1] = C[1:-1, -2]
        C[0, 1:-1] = C[1, 1:-1]
        C[-1, 1:-1] = C[-2, 1:-1]
        C[0, 0] = stale if old_corner else C[1, 1]
        C[0, -1] = C[1, -2]
        C[-1, 0] = C[-2, 1]
        C[-1, -1] = C[-2, -2]
        A, Bf, C = Bf, C, A
    return (A, Bf) if keep_prev else Bf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='cuda')
    ap.add_argument('--f32', action='store_true')
    ap.add_argument('--big', type=int, default=1025, help='grid size for the large-N tests')
    a = ap.parse_args()
    mc.init_taichi(a.arch, fp64=not a.f32)
    f32 = a.f32
    B = a.big | 1
    T0 = time.perf_counter()

    # 1. LIGHT CONE. One tick moves information one bond: after m ticks an impulse is
    #    confined to the reach polytope of radius m -- EXACT zeros outside, not small values.
    for lat, st, name, dist in (('sq', 5, 'diamond |a|+|b|', lambda A, Bb: np.abs(A) + np.abs(Bb)),
                                ('sq', 9, 'square max(|a|,|b|)', lambda A, Bb: np.maximum(np.abs(A), np.abs(Bb))),
                                ('tri', 5, 'hexagon max(|a|,|b|,|a+b|)',
                                 lambda A, Bb: np.maximum(np.maximum(np.abs(A), np.abs(Bb)), np.abs(A + Bb)))):
        m = mc.Membrane(mc.Params(N=B, lattice=lat, stencil=st, src='impulse', shape='n6' if lat == 'tri' else 'square',
                                  rot=30 * D if lat == 'tri' else 0, Co=0.5))
        # the leading edge after m ticks is ~Co^(2m): 1e-62 at m = 200, far below f32's
        # smallest number, so in f32 the rim itself underflows to 0 -- test a younger cone
        K = min(20 if f32 else 200, (B - 1) // 4)
        m.step(K)
        u = m.field()
        A_, Bb = offsets(m)
        dd = dist(A_, Bb)
        outside = np.abs(u[dd > K]).max()
        rim = np.abs(u[dd == K]).max()
        ok(f'light cone {lat}{st if lat == "sq" else ""}: {name} <= m', outside == 0 and rim > 0,
           f'N={m.N}, m={K}: max|u| outside = {outside:.1e} (exactly 0), on the rim = {rim:.2e}')

    # 2. CFL FLIP. Stable just below the lattice's C_max, unstable just above.
    for lat, st, cmax in (('sq', 5, mc.CO_LIMIT_5), ('sq', 9, mc.CO_LIMIT_9), ('tri', 5, mc.CO_LIMIT_T)):
        res = []
        for f in (0.995, 1.01):
            m = mc.Membrane(mc.Params(N=161, lattice=lat, stencil=st, src='impulse', Co=cmax * f,
                                      shape='n6' if lat == 'tri' else 'square'))
            m.step(3000)
            x = np.abs(m.field()).max()
            res.append(x if np.isfinite(x) else math.inf)   # overflowed to inf/nan = diverged
        ok(f'CFL flip {lat}{st if lat == "sq" else ""} at C_max = {cmax:.4f}', res[0] < 1 and res[1] > 1e6,
           f'max|u| after 3000 steps: {res[0]:.2e} at 0.995 C_max, {res[1]:.2e} at 1.01 C_max')

    # 3. SYMMETRY of a centred pulse: 8-fold on the square lattice, 6-fold on the triangular
    m = mc.Membrane(mc.Params(N=B, src='pulse', sigma=4))
    m.step(B // 3)
    u = m.field()
    s8 = max(np.abs(u - u.T).max(), np.abs(u - u[::-1, :]).max(), np.abs(u - u[:, ::-1]).max()) / np.abs(u).max()
    ok('8-fold symmetry, square lattice, full grid', s8 < (1e-5 if f32 else 1e-12), f'N={m.N}: residual {s8:.1e}')
    m = mc.Membrane(mc.Params(N=B, lattice='tri', shape='n6', rot=30 * D, src='pulse', sigma=4))
    m.step(B // 3)
    u = m.field()
    A_, Bb = offsets(m)
    c = (m.N - 1) // 2
    inm = m.mask_np.astype(bool)
    A2, B2 = -Bb, A_ + Bb                                  # 60-degree turn in lattice coords
    val = np.abs(A2) <= c
    val &= np.abs(B2) <= c
    rot = np.zeros_like(u)
    rot[val] = u[(A2[val] + c), (B2[val] + c)]
    s6 = np.abs((u - rot)[inm & val]).max() / np.abs(u).max()
    ok('6-fold symmetry, triangular lattice, hexagon at 30 deg', s6 < (1e-5 if f32 else 1e-12),
       f'N={m.N}: residual {s6:.1e}')

    # 4. ENERGY. Leapfrog conserves E = sum (du)^2 / Co_k^2 - <u_new, L u_old> EXACTLY (to
    #    rounding) -- also in an inhomogeneous medium, where the weight is the local Co^2.
    for label, prm in (('uniform square, Dirichlet', dict(src='pulse', sigma=5)),
                       ('circle + lens (inhomogeneous)', dict(shape='circle', rad=0.9, medium='lens', src='pulse', sigma=5, sx=0.3)),
                       ('triangular hexagon', dict(lattice='tri', shape='n6', src='pulse', sigma=5))):
        m = mc.Membrane(mc.Params(N=B, **prm))
        msk = m.mask_np.astype(bool)
        cc = np.where(msk, m.cc_np, 1.0)
        def energy():
            u0, u1 = m.level(0).astype(np.float64), m.level(1).astype(np.float64)
            du = np.where(msk, u1 - u0, 0.0)
            return (du * du / cc).sum() - (u1 * np.where(msk, lap_np(u0, m), 0.0)).sum()
        m.step(1)
        E0 = energy()
        m.step(2000)
        E1 = energy()
        drift = abs(E1 - E0) / abs(E0)
        ok(f'energy conserved: {label}', drift < (1e-4 if f32 else 1e-11), f'N={m.N}, 2000 steps: drift {drift:.1e}')

    # 5. DRUM DISPERSION. A seeded (m,n) mode of the full square evolves as
    #    cos(n w) + B sin(n w),  cos w = 1 - Co^2 lam/2,  lam = 4 sin^2(m pi/2L) + 4 sin^2(n pi/2L),
    #    B = (cos w - 1)/sin w (seeded at rest) -- the lattice's exact dispersion relation.
    for st in (5, 9):
        for mm, nn in ((1, 1), (7, 3), (40, 25)):
            m = mc.Membrane(mc.Params(N=B, src='mode', mm=mm, mn=nn, stencil=st, Co=0.6))
            L = m.N - 1
            kx, ky = mm * math.pi / L, nn * math.pi / L
            if st == 9:
                lam = -((2 / 3) * (2 * math.cos(kx) + 2 * math.cos(ky) - 4) + (1 / 6) * (4 * math.cos(kx) * math.cos(ky) - 4))
            else:
                lam = 4 * math.sin(kx / 2) ** 2 + 4 * math.sin(ky / 2) ** 2
            w = math.acos(1 - 0.36 * lam / 2)
            Bc = (math.cos(w) - 1) / math.sin(w)
            ai, aj = round(L / (2 * mm)), round(L / (2 * nn))
            u00 = m.field()[ai, aj]
            worst = 0.0
            for n in range(1, 401):
                m.step(1)
                want = u00 * (math.cos(n * w) + Bc * math.sin(n * w))
                worst = max(worst, abs(m.value(ai, aj) - want))
            # f32: the (1,1) mode's curvature is 1.9e-5 of its amplitude at N = 1025, so each
            # step's rounding (6e-8 of u, x4 neighbours) is ~1% of the signal, and it adds up
            # COHERENTLY (smooth field -> correlated rounding): 4e-5 after 400 steps, measured
            ok(f'drum dispersion sq{st} mode ({mm},{nn})', worst < (1e-4 if f32 else 1e-11),
               f'N={m.N}: 400 steps vs the closed form, worst {worst:.1e} (w = {w:.6f} rad/step)')

    # 6. CONFORMANCE. At 30 deg on the triangular lattice the {3,4,6} tilers hold EXACTLY the
    #    lattice-point count of the ideal shape.
    for shape, rad in (('n6', 0.37), ('n6', 0.91), ('n3', 0.6), ('n3', 1.0), ('rhomb', 0.55), ('rhomb', 0.97)):
        m = mc.Membrane(mc.Params(N=B, lattice='tri', shape=shape, rad=rad, rot=30 * D))
        R = m.dom_R()
        if shape == 'n6':                                  # centred hexagonal number
            s = round(R); want = 3 * s * s + 3 * s + 1
        elif shape == 'n3':                                # triangular number, side a multiple of 3
            s = round(R * math.sqrt(3)); want = (s + 1) * (s + 2) // 2
        else:                                              # 60-degree rhombus, even side
            s = round(2 * R / math.sqrt(3)); want = (s + 1) ** 2
        got = m.domain_cells()
        ok(f'conformance tri {shape} radius {rad}', got == want, f'N={m.N}, side {s}: {got} cells, closed form {want}')

    # 7. TIME REFLECTION. Changing Co mid-run is a temporal boundary: a backward wave appears.
    #    With the 2nd-order time-level rescale (set_co(rescale=True)) it does not.
    NB = 801
    def run(switch, rescale=False, total=None):
        m = mc.Membrane(mc.Params(N=NB, src='pulse', sigma=3, Co=0.5))
        m.step(300)
        if switch:
            m.set_co(0.7, rescale=rescale)
            m.step(100)
        else:
            m.step(total - 300)
        return m.field()
    uA = run(False, total=440)                       # physical time 220 at Co 0.5 throughout
    uB = run(True)                                   # 300 x 0.5 + 100 x 0.7 = 220 as well
    uC = run(True, rescale=True)
    c = (NB - 1) // 2
    X, Y = np.meshgrid(np.arange(NB) - c, np.arange(NB) - c, indexing='ij')
    r = np.hypot(X, Y)
    inner, outer = (r > 50) & (r < 110), (r > 190) & (r < 250)
    fw = np.abs(uA[outer]).max()
    bwB = np.abs((uB - uA)[inner]).max()
    bwC = np.abs((uC - uA)[inner]).max()
    ok('time reflection: a backward ring appears when Co jumps 0.5 -> 0.7', 0.1 < bwB / fw < 0.5,
       f'backward/forward = {bwB / fw:.3f} (1D long-wave B/F = 0.167; x1.66 for the 2D ring '
       f'spreading: converging vs diverging -> ~0.28)')
    ok('time reflection: the 2nd-order rescale removes it', bwC < 0.05 * bwB,
       f'backward with rescale = {bwC / bwB:.1e} of without')

    # 8. NEUMANN CORNERS (the port fix of 29/09/2026). Every Neumann corner now equals its
    #    diagonal interior neighbour of THIS step; the web app's sequential ring loop left (0,0)
    #    holding (1,0) from an older buffer. A numpy stepper in the web app's operation order runs
    #    both rules: the NEW one must agree with the port, and the OLD one shows what it cost.
    #    (a) 8-fold symmetry of a centred pulse, full square, Neumann, sq5 and sq9;
    #    (b) the leapfrog energy with ghost cells: exactly conserved only if the operator on the
    #        interior unknowns is symmetric, which the new corner rule makes it (for sq9; sq5
    #        never reads a corner).
    #    N = 257 and 900 steps: the front (0.5 cell/step) must reach the corners (181 cells
    #    away) and come back, or the corner rule cannot show at all (at N = 1025, 341 steps, it
    #    did not: 0 cells differed).
    NS = 257
    for st in (5, 9):
        m = mc.Membrane(mc.Params(N=NS, src='pulse', sigma=4, bc='neumann', stencil=st, Co=0.5))
        K = 900
        u_seed = m.field().astype(np.float64)
        m.step(K)
        u = m.field().astype(np.float64)
        sym = lambda v: max(np.abs(v - v.T).max(), np.abs(v - v[::-1, :]).max(), np.abs(v - v[:, ::-1]).max()) / np.abs(v).max()
        s_new = sym(u)
        ref_new = np_neumann_run(u_seed, 0.25, st == 9, K, old_corner=False)
        ref_old = np_neumann_run(u_seed, 0.25, st == 9, K, old_corner=True)
        d_ref = np.abs(u - ref_new).max() / np.abs(u).max()
        s_old = sym(ref_old)
        n_diff = int((np.abs(ref_old - ref_new) > 1e-12 * np.abs(ref_new).max()).sum())
        ok(f'Neumann corner: 8-fold symmetry sq{st}, centred pulse', s_new < (1e-5 if f32 else 1e-12)
           and d_ref < (1e-4 if f32 else 1e-12),
           f'N={m.N}, {K} steps: residual {s_new:.1e} (old corner rule, numpy: {s_old:.1e}, '
           f'{n_diff} cells differ from the new rule); port vs numpy new rule {d_ref:.1e}')
    for st in (5, 9):
        NE = 513
        m = mc.Membrane(mc.Params(N=NE, src='pulse', sigma=3, bc='neumann', stencil=st, Co=0.5, sx=-0.8, sy=-0.7))
        u_seed = m.field().astype(np.float64)
        def e_ghost(u0, u1, nine):
            du = (u1 - u0)[1:-1, 1:-1]
            return (du * du).sum() / 0.25 - (u1[1:-1, 1:-1] * lap_full(u0, nine)).sum()
        m.step(2)                                          # both stored levels written by the solver
        E0 = e_ghost(m.level(0).astype(np.float64), m.level(1).astype(np.float64), st == 9)
        m.step(2000)
        E1 = e_ghost(m.level(0).astype(np.float64), m.level(1).astype(np.float64), st == 9)
        drift = abs(E1 - E0) / abs(E0)
        # the old rule, numpy (same seed, same energy)
        lv = np_neumann_run(u_seed, 0.25, st == 9, 2002, old_corner=True, keep_prev=True)
        lv2 = np_neumann_run(u_seed, 0.25, st == 9, 2, old_corner=True, keep_prev=True)
        E0o, E1o = e_ghost(lv2[0], lv2[1], st == 9), e_ghost(lv[0], lv[1], st == 9)
        drift_old = abs(E1o - E0o) / abs(E0o)
        ok(f'Neumann corner: ghost-cell energy conserved sq{st}', drift < (1e-4 if f32 else 1e-11),
           f'N={NE}, pulse near the (0,0) corner, 2000 steps: drift {drift:.1e} '
           f'(old corner rule, numpy: {drift_old:.1e})')

    # 9. THE RECORDER READS ONLY: the field with the per-step recorder on is bit-identical to
    #    the field with it off (on every backend: it is a separate pass after the step).
    for lat, st in (('sq', 9), ('tri', 5)):
        prm = dict(N=513, src='pulse', sigma=4, lattice=lat, stencil=st, bc='mur',
                   shape='n6' if lat == 'tri' else 'square', sx=0.3)
        m1 = mc.Membrane(mc.Params(**prm))
        m2 = mc.Membrane(mc.Params(**prm))
        m2.record = False
        m1.step(300)
        m2.step(300)
        same = np.array_equal(m1.field(), m2.field()) and np.array_equal(m1.level(0), m2.level(0))
        ok(f'recorder never changes the field ({lat}{st if lat == "sq" else ""})', same and len(m1.hist['E']) == 300
           and not m2.hist['E'],
           f'300 steps: fields {"bit-identical" if same else "DIFFER"}; history {len(m1.hist["E"])} vs {len(m2.hist["E"])} entries')

    print(f'\n{"ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: " + ", ".join(FAILS)} '
          f'({a.arch} {"f32" if f32 else "f64"}, {time.perf_counter() - T0:.1f} s)')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
