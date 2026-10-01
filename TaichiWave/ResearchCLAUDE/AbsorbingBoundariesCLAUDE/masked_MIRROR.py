"""The 'absorbing wall inside a masked shape': a damping ramp by distance to the mask edge.

Masked domains in the app are Dirichlet on the mask edge (cells outside are held at 0), and the
edge of a circle or a rotated polygon is a staircase. A sponge needs no normal direction: every
cell gets d = dmax * ((L - dist)/L)^p for dist < L, dist = the distance from its centre to the
nearest OUTSIDE cell centre (physical coordinates; a k-d tree, so it works for any mask).

Lattices: 'sq' (5-point) and 'tri' (6-neighbour, x = i + j/2, y = j sqrt3/2, neighbours
(i+-1,j), (i,j+-1), (i+1,j-1), (i-1,j+1); lap = (2/3)(sum - 6u), as the core).
Shapes: 'circle' and 'hex' (edges normal to 0, 60, 120 deg -- lattice rows on 'tri').

Measurement: a Gaussian pulse (both levels equal) at the centre; the test shape has physical
radius Rphys plus an L-cell sponge; the reference is the same shape Rphys + E with no sponge,
big enough that nothing returns in the window. At the end, sqrt(E_refl/E0) with E_refl the
leapfrog energy of (test - reference) over the physical cells (dist >= L in the test) and E0 the
pulse's energy. The window runs until the front is at Rphys + 2*40 + 60 cells, so even the
reflection from the bottom of a 40-cell sponge is back inside when it is measured.
"""
import math
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

S3H = math.sqrt(3) / 2


def coords(lat, N):
    i = np.arange(N)[:, None].astype(np.float64)
    j = np.arange(N)[None, :].astype(np.float64)
    if lat == 'tri':
        return i + 0.5 * j, np.broadcast_to(j * S3H, (N, N)).copy()
    return np.broadcast_to(i, (N, N)).copy(), np.broadcast_to(j, (N, N)).copy()


def grid_size(lat, R):
    # tri: |j| <= R/(sqrt3/2) = 1.155 R and |i| <= R + |j|/2 = 1.577 R around the centre cell
    return int(2 * math.ceil(1.6 * R) + 12) if lat == 'tri' else int(2 * math.ceil(R) + 7)


def centre(lat, N):
    """Physical centre: the middle cell."""
    X, Y = coords(lat, N)
    c = N // 2
    return X[c, c], Y[c, c]


def shape_mask(lat, shape, N, R):
    X, Y = coords(lat, N)
    cx, cy = centre(lat, N)
    dx, dy = X - cx, Y - cy
    if shape == 'circle':
        inside = dx * dx + dy * dy <= R * R
    else:
        h = np.zeros_like(dx)
        for k in range(6):
            a = k * math.pi / 3
            h = np.maximum(h, dx * math.cos(a) + dy * math.sin(a))
        inside = h <= R
    inside[0, :] = inside[-1, :] = inside[:, 0] = inside[:, -1] = False
    return inside


def dist_to_edge(lat, M):
    X, Y = coords(lat, M.shape[0])
    out = np.stack([X[~M], Y[~M]], 1)
    tree = cKDTree(out)
    d = np.zeros(M.shape)
    d[M] = tree.query(np.stack([X[M], Y[M]], 1))[0]
    return d


def lap(lat, u):
    L = np.zeros_like(u)
    c = u[1:-1, 1:-1]
    s = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
    if lat == 'tri':
        s = s + u[2:, :-2] + u[:-2, 2:]
        L[1:-1, 1:-1] = (2.0 / 3.0) * (s - 6 * c)
    else:
        L[1:-1, 1:-1] = s - 4 * c
    return L


def energy(lat, d1, d0, Co, reg):
    kin = np.sum(((d1 - d0) ** 2)[reg]) / (Co * Co)
    w = 2.0 / 3.0 if lat == 'tri' else 1.0
    pairs = [((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
             ((slice(None), slice(1, None)), (slice(None), slice(None, -1)))]
    if lat == 'tri':
        pairs.append(((slice(1, None), slice(None, -1)), (slice(None, -1), slice(1, None))))
    pot = 0.0
    for A_, B_ in pairs:
        e = reg[A_] & reg[B_]
        pot += w * np.sum(((d1[A_] - d1[B_]) * (d0[A_] - d0[B_]))[e])
    return kin + pot


def run(lat, shape, R, Co, T, s, d=None):
    N = grid_size(lat, R)
    M = shape_mask(lat, shape, N, R)
    X, Y = coords(lat, N)
    cx, cy = centre(lat, N)
    g = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s)) * M
    up, u = g.copy(), g.copy()
    C2 = Co * Co
    E0 = None
    for n in range(T):
        un = 2 * u - up + C2 * lap(lat, u)
        if d is not None:
            un -= d * (u - up)
        un[~M] = 0.0
        up, u = u, un
        if n == 0:
            E0 = energy(lat, u, up, Co, M)
    return dict(N=N, M=M, X=X, Y=Y, cx=cx, cy=cy, u=u, up=up, E0=E0)


def sponge_reflection(lat, shape, Rphys, L, dmax, p, Co=0.5, s=3.0, ref=None):
    r_end = Rphys + 2 * 40 + 60
    T = int(round(r_end / Co))
    Rt = Rphys + L
    N = grid_size(lat, Rt)
    M = shape_mask(lat, shape, N, Rt)
    dist = dist_to_edge(lat, M)
    d = np.where(M & (dist < L), dmax * np.clip((L - dist) / max(L, 1e-9), 0, 1) ** p, 0.0) if L > 0 else None
    t = run(lat, shape, Rt, Co, T, s, d)
    if ref is None:
        E = int(0.5 * r_end) + 20
        ref = run(lat, shape, Rphys + L + E, Co, T, s)       # same pulse, nothing returns
    # map the reference onto the test grid through physical coordinates (same lattice, same
    # centre cell => an integer offset)
    off = (ref['N'] // 2) - (N // 2)
    ru = ref['u'][off:off + N, off:off + N]
    rup = ref['up'][off:off + N, off:off + N]
    reg = M & (dist >= L) if L > 0 else M.copy()
    Er = energy(lat, t['u'] - ru, t['up'] - rup, Co, reg)
    return math.sqrt(max(Er, 0.0) / ref['E0']), ref


if __name__ == '__main__':
    import json
    HERE = os.path.dirname(os.path.abspath(__file__))
    res = {}
    for lat, shape in (('sq', 'circle'), ('tri', 'hex'), ('tri', 'circle')):
        ref = None
        for L, dmax, p in [(0, 0, 0), (10, None, 2), (20, None, 2), (40, None, 2)]:
            if L == 0:
                r, ref = sponge_reflection(lat, shape, 100, 0, 0, 0, ref=ref)
                res['%s %s L0 (Dirichlet mask)' % (lat, shape)] = r
                print(lat, shape, 'L0', '%.3e' % r, flush=True)
                continue
            for dm in (0.1, 0.2, 0.35, 0.5):
                r, ref = sponge_reflection(lat, shape, 100, L, dm, p, ref=ref)
                res['%s %s L%d d%.2f p%d' % (lat, shape, L, dm, p)] = r
                print(lat, shape, L, dm, p, '%.3e' % r, flush=True)
    with open(os.path.join(HERE, 'results_masked_MIRROR.json'), 'w') as f:
        json.dump(res, f, indent=1)
