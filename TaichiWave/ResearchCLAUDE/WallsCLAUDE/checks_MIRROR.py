"""Walls research: sanity checks of walls_lib_MIRROR (29/09/2026).

1. rule P reproduces the port's pinned stencils (zero padding) exactly;
2. every rule's operator is symmetric on a scene with every hard case;
3. R1 on a sq9 room with a rigid ring == the full-square ghost-copy Neumann ring
   (ring cell = its interior neighbour, corner = the diagonal cell u(1,1));
4. tangential/normal second moments of the boundary row at a flat wall for R0 and R1.

    python checks_MIRROR.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walls_lib_MIRROR as W

rng = np.random.default_rng(1)
FAIL = []


def ok(name, cond, det=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('  | ' + det if det else ''))
    if not cond:
        FAIL.append(name)


def lap_np(u, lat):
    p = np.pad(u, 1)
    l, r, d, t = p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
    if lat == 'tri':
        return (2 / 3) * (l + r + d + t + p[2:, :-2] + p[:-2, 2:] - 6 * u)
    if lat == 'sq9':
        dg = p[:-2, :-2] + p[2:, :-2] + p[:-2, 2:] + p[2:, 2:]
        return (2 / 3) * (l + r + d + t - 4 * u) + (1 / 6) * (dg - 4 * u)
    return l + r + d + t - 4 * u


def scene(n=48):
    """Every hard case in one box: ring wall, 1-cell straight fin, 1-cell diagonal wall,
    single cell, 3x3 block, closed ring with a counter (cavity), a concave notch."""
    s = np.zeros((n, n), bool)
    s[0, :] = s[-1, :] = s[:, 0] = s[:, -1] = True
    s[10, 5:20] = True                          # fin
    for k in range(20):                         # diagonal 8-connected wall
        s[22 + k, 5 + k] = True
    s[8, 35] = True                             # single cell
    s[30:33, 30:33] = True                      # 3x3
    s[36:43, 8:15] = True; s[38:41, 10:13] = False   # ring with a counter
    s[5:9, 40:44] = True; s[6:8, 42:44] = False       # notch
    return s


def main():
    # 1. P == port stencil with zero padding
    for lat in ('sq5', 'sq9', 'tri'):
        n = 40
        solid = rng.random((n, n)) < 0.15
        u = rng.standard_normal((n, n)); u[solid] = 0
        L = W.op_for(solid, W.LATS[lat], 'P')
        ref = lap_np(u, lat); ref[solid] = 0
        err = np.abs(L @ u.ravel() - ref.ravel()).max()
        ok(f'P == port stencil ({lat})', err < 1e-13, f'max diff {err:.1e}')

    # 2. symmetry of every rule on the hard-case scene
    s = scene()
    for lat in ('sq5', 'sq9', 'tri'):
        for rule in W.RULES:
            L = W.op_for(s, W.LATS[lat], rule)
            e = W.sym_err(L)
            ok(f'symmetric {lat} {rule}', e < 1e-15, f'max |L-L^T| = {e:.1e}')

    # 3. R1 sq9 room == ghost-copy Neumann ring, corner = u(1,1)
    n = 30
    s = np.zeros((n, n), bool); s[0, :] = s[-1, :] = s[:, 0] = s[:, -1] = True
    L = W.op_for(s, W.SQ9, 'R1')
    u = rng.standard_normal((n, n))
    g = u.copy()
    g[0, 1:-1] = u[1, 1:-1]; g[-1, 1:-1] = u[-2, 1:-1]; g[1:-1, 0] = u[1:-1, 1]; g[1:-1, -1] = u[1:-1, -2]
    g[0, 0] = u[1, 1]; g[-1, 0] = u[-2, 1]; g[0, -1] = u[1, -2]; g[-1, -1] = u[-2, -2]
    ref = np.zeros_like(u)
    # direct stencil on the ghost-filled array
    c = g[1:-1, 1:-1]
    e = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    dg = g[:-2, :-2] + g[2:, :-2] + g[:-2, 2:] + g[2:, 2:]
    ref[1:-1, 1:-1] = (2 / 3) * (e - 4 * c) + (1 / 6) * (dg - 4 * c)
    uu = u.copy(); uu[s] = 0
    err = np.abs(L @ uu.ravel() - ref.ravel()).max()
    ok('R1 (sq9, rigid ring) == ghost-copy Neumann with corner u(1,1)', err < 1e-13, f'max diff {err:.1e}')
    L0 = W.op_for(s, W.SQ9, 'R0')
    err0 = np.abs(L0 @ uu.ravel() - ref.ravel()).max()
    print(f'     (R0 differs from the ghost ring by {err0:.2f} on random data: link removal alone is NOT the mirror for sq9)')

    # 4. second moments of the boundary row next to a flat wall (u = x^2 along, y^2 across)
    for lat in ('sq5', 'sq9', 'tri'):
        LT = W.LATS[lat]
        n = 40
        s = np.zeros((n, n), bool); s[:, :3] = True         # wall = rows j < 3 (lattice rows)
        I, J = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing='ij')
        X, Y = LT.xy(I, J)
        yw = LT.xy(0.0, 2.5)[1]                             # the half-way line under row j = 3
        for rule in ('R0', 'R1'):
            L = W.op_for(s, LT, rule)
            f_t = (X - 20) ** 2 * 0.5                       # tangential: lap should be 1
            f_n = (Y - yw) ** 2 * 0.5                       # normal, du/dn = 0 at the wall: lap 1
            lt = (L @ np.where(s, 0, f_t).ravel()).reshape(n, n)[20, 3]
            ln = (L @ np.where(s, 0, f_n).ravel()).reshape(n, n)[20, 3]
            li = (L @ np.where(s, 0, f_t).ravel()).reshape(n, n)[20, 10]
            print(f'     {lat} {rule}: boundary-row lap of x^2/2 = {lt:.4f}, of (y-y_w)^2/2 = {ln:.4f} '
                  f'(interior {li:.4f}; exact 1)')
    print('\nFAILS:', FAIL if FAIL else 'none')


if __name__ == '__main__':
    main()
