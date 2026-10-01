"""Walls research (29/09/2026): energy conservation, CFL radius, thin-wall leaks, cavities.

  A. discrete energy drift over 20000 leapfrog steps, per lattice x rule, on a scene with every
     hard case (fin, diagonal wall, single cell, block, closed ring with a counter, notch) and
     on a MIXED scene (half the objects pinned, half rigid); plus lam_max(-L) vs the
     infinite-lattice value (the CFL radius), also on a random 15%-solid scene;
  B. leak through one-cell walls: straight, the two array diagonals, and brush-rasterised lines
     at 0..90 deg for brush radii 0.5 .. 1.0 (the hand-drawn-wall case);
  C. enclosed cavities: zero modes (connected components) per rule.

    python energy_leak_MIRROR.py
"""
import sys, os, math, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walls_lib_MIRROR as W

LAM_INF = {'sq5': 8.0, 'sq9': 16 / 3, 'tri': 6.0}      # sup of the symbol = 4 / Co_max^2


def scene(n=96):
    s = np.zeros((n, n), bool)
    s[0, :] = s[-1, :] = s[:, 0] = s[:, -1] = True
    s[20, 10:40] = True                                   # 1-cell fin
    for k in range(40):
        s[44 + k, 10 + k] = True                           # 1-cell diagonal (8-connected) wall
    s[16, 70] = True                                      # single cell
    s[60:63, 60:63] = True                                # 3x3 block
    s[72:86, 16:30] = True; s[76:82, 20:26] = False       # closed ring with a counter
    s[10:18, 80:88] = True; s[12:16, 84:88] = False       # notch (concave)
    return s


def mixed_types(s):
    """Left half of the objects pinned, right half rigid; the ring pinned."""
    n = s.shape[0]
    T = np.where(s, W.PIN, W.FREE).astype(np.int8)
    J = np.arange(n)[None, :].repeat(n, 0)
    T[(s) & (J >= n // 2)] = W.RIG
    T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = W.PIN
    return T


def gauss(lat, n, ci, cj, sig):
    X, Y = W.grid_xy(lat, n, n, ci, cj)
    return np.exp(-(X * X + Y * Y) / (2 * sig * sig))


def part_A():
    print('=== A. energy drift over 20000 steps (Co = 0.95 Co_max), and the CFL radius ===')
    s = scene()
    n = s.shape[0]
    rng = np.random.default_rng(7)
    rnd = rng.random((64, 64)) < 0.15
    rnd[0, :] = rnd[-1, :] = rnd[:, 0] = rnd[:, -1] = True
    for latn in ('sq5', 'sq9', 'tri'):
        lat = W.LATS[latn]
        co = 0.95 * lat.co_max
        for rule in ('P', 'Pcb', 'R0nb', 'R0', 'R1', 'mixed'):
            if rule == 'mixed':
                L = W.build_L(mixed_types(s), lat, block='both', transfer=True)
                free = ~s
            else:
                L = W.op_for(s, lat, rule)
                free = ~s
            u = (gauss(lat, n, 48, 48, 3.0) * free).ravel()
            u0 = u.copy()
            u1 = u.copy()
            # a start with zero velocity: u^1 = u^0 + (Co^2/2) L u^0
            u1 = u0 + 0.5 * co * co * (L @ u0)
            E0 = W.energy(L, u0, u1, co)
            Es = []
            W.run(L, u0, u1, co, 20000, every=1000, cb=lambda k, a, b: Es.append(W.energy(L, a, b, co)))
            drift = max(abs(e - E0) for e in Es) / E0
            lm = W.lam_max(L, free)
            lmr = W.lam_max(W.op_for(rnd, lat, rule) if rule != 'mixed' else
                            W.build_L(mixed_types(rnd), lat, block='both', transfer=True), ~rnd)
            print(f'  {latn} {rule:5}: max |E-E0|/E0 = {drift:.1e}   sym err {W.sym_err(L):.0e}   '
                  f'lam_max/lam_inf = {lm / LAM_INF[latn]:.4f} (scene), {lmr / LAM_INF[latn]:.4f} (random 15%)')


def split_leak(lat, solid, sideB, steps=500, co=None, rule='P', src=None):
    n = solid.shape[0]
    co = co or 0.5
    L = W.op_for(solid, lat, rule)
    # the start is EXACTLY zero on side B (a Gaussian's tail would otherwise put ~1e-30 there)
    u0 = (gauss(lat, n, src[0], src[1], 2.5) * (~solid & ~sideB)).ravel()
    u1 = u0 + 0.5 * co * co * (L @ u0)
    B = sideB.ravel() & ~solid.ravel()
    worst = [0.0]

    def cb(k, a, b):
        tot = float(b @ b)
        worst[0] = max(worst[0], float(b[B] @ b[B]) / tot)
    W.run(L, u0, u1, co, steps, every=5, cb=cb)
    return worst[0]


def brush_line(lat, n, ang_deg, r):
    """Cells whose centre is within r (physical) of the line through the box centre at ang."""
    c = (n - 1) / 2
    X, Y = W.grid_xy(lat, n, n, c, c)
    a = math.radians(ang_deg)
    d = -math.sin(a) * X + math.cos(a) * Y
    return np.abs(d) <= r, d


def part_B():
    print('\n=== B. leak through one-cell walls: max over 500 steps of sum_B u^2 / sum u^2 ===')
    print('   ("0" = exactly zero: no link crosses the wall)')
    n = 81
    ring = np.zeros((n, n), bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    cases = {
        'straight i = 40': (I == 40, I > 40, (20, 40)),
        'array diagonal i = j': (I == J, I < J, (55, 25)),
        'anti-diagonal i + j = n-1': (I + J == n - 1, I + J > n - 1, (25, 25)),
    }
    for latn in ('sq5', 'sq9', 'tri'):
        lat = W.LATS[latn]
        for name, (wall, sideB, src) in cases.items():
            solid = ring | wall
            res = []
            for rule in ('P', 'Pcb', 'R0nb', 'R0', 'R1'):
                lk = split_leak(lat, solid, sideB, rule=rule, src=src)
                res.append(f'{rule} {("0" if lk == 0 else f"{lk:.2e}"):>8}')
            print(f'  {latn} {name:26}: ' + '  '.join(res))

    print('\n   brush-rasterised straight lines through the centre, radius r (physical cells):')
    print('   worst leak over angles 0,5,...,90 (tri: 0..60 covers every direction by symmetry, 0..90 run)')
    angs = list(range(0, 91, 5))
    for latn in ('sq5', 'sq9', 'tri'):
        lat = W.LATS[latn]
        for rule in ('P', 'Pcb', 'R0nb', 'R0'):
            row = []
            for r in (0.5, 0.6, 0.71, 0.8, 1.0):
                worst, wa = 0.0, None
                for a in angs:
                    wall, d = brush_line(lat, n, a, r)
                    solid = ring | wall
                    sideB = d > 0
                    # source on side A, 12 cells from the line, near the middle
                    c = (n - 1) / 2
                    sx, sy = c + 12 * math.sin(math.radians(a)), c - 12 * math.cos(math.radians(a))
                    # physical -> index
                    X, Y = W.grid_xy(lat, n, n, c, c)
                    k = np.argmin((X - (sx - c)) ** 2 + (Y - (sy - c)) ** 2)
                    si, sj = np.unravel_index(k, X.shape)
                    lk = split_leak(lat, solid, sideB, steps=300, rule=rule, src=(si, sj))
                    if lk > worst:
                        worst, wa = lk, a
                row.append(f'r={r}: {("0" if worst == 0 else f"{worst:.1e}@{wa}"):>11}')
            print(f'  {latn} {rule:4}: ' + '  '.join(row))


def part_C():
    print('\n=== C. enclosed cavities: zero modes = connected components with no pinned link ===')
    s = scene()
    for latn in ('sq5', 'sq9', 'tri'):
        lat = W.LATS[latn]
        for rule in ('P', 'R0', 'R0nb'):
            L = W.op_for(s, lat, rule)
            k = np.flatnonzero(~s.ravel())
            Lf = L[k][:, k]
            A = Lf.copy(); A.setdiag(0); A.eliminate_zeros()
            ncomp, lab = csg.connected_components(A, directed=False)
            rowsum = np.asarray(Lf.sum(axis=1)).ravel()
            zero = sum(1 for c in range(ncomp) if abs(rowsum[lab == c]).max() < 1e-12)
            sizes = sorted(np.bincount(lab), reverse=True)
            print(f'  {latn} {rule:4}: {ncomp} components (sizes {sizes[:4]}), {zero} with a zero mode')


if __name__ == '__main__':
    t0 = time.perf_counter()
    part_A(); print(f'  [{time.perf_counter() - t0:.0f} s]')
    part_B(); print(f'  [{time.perf_counter() - t0:.0f} s]')
    part_C(); print(f'  [{time.perf_counter() - t0:.0f} s]')
