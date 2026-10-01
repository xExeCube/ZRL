"""A LOCAL absorbing rule on a staircase mask edge, applied along the shape's normal -- the
alternative to the distance-ramp sponge (masked_MIRROR.py) for the flavor shapes.

Boundary cells = mask cells with at least one lattice neighbour outside the mask. They are not
updated by the interior stencil; instead (transport along the outward normal n, speed c):
  'T1'   u_b^{n+1} = f * u^n(x_b - c dt n)                         (Liao order 1 / upwind Mur)
  'T2'   u_b^{n+1} = f * [2 u^n(x_b - c dt n) - f2 u^{n-1}(x_b - 2 c dt n)]   (Liao order 2,
         direct interpolation)
The off-grid values are interpolated at level n / n-1, which are complete, so the rule is
explicit and order-free (one parallel pass on the GPU). Interpolation: bilinear on 'sq',
barycentric on the equilateral triangles of 'tri'. f is the 2D cylindrical-spreading factor
sqrt((r - c dt)/r) for a circle (a Bayliss-Turkel B1-like correction; f2 likewise for 2 c dt);
f = 1 on flat polygon faces. n is the analytic normal (radial for the circle, the nearest face
for the hexagon); a generic mask would take it from the gradient of a smoothed distance field.

Measurement as masked_MIRROR.py: Gaussian pulse, reference in a shape big enough that nothing
returns, sqrt(E_refl / E0) over the physical cells (the boundary ring excluded) when the front
is at r_end. Pulses: centred (every ray hits the circle at normal incidence) and off-centre at
0.5 R (incidence up to 30 deg on the circle).      python masked_normal_MIRROR.py
Written 30/09/2026.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import masked_MIRROR as MM  # noqa: E402

S3H = math.sqrt(3) / 2


def neighbours(lat):
    nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if lat == 'tri':
        nb += [(1, -1), (-1, 1)]
    return nb


def boundary_cells(lat, M):
    B = np.zeros_like(M)
    for di, dj in neighbours(lat):
        B |= M & ~np.roll(np.roll(M, -di, 0), -dj, 1)
    return B


def normals(shape, X, Y, cx, cy):
    dx, dy = X - cx, Y - cy
    if shape == 'circle':
        r = np.hypot(dx, dy) + 1e-300
        return dx / r, dy / r, r
    best = np.full(dx.shape, -1e30)
    nx, ny = np.zeros_like(dx), np.zeros_like(dy)
    for k in range(6):
        a = k * math.pi / 3
        h = dx * math.cos(a) + dy * math.sin(a)
        sel = h > best
        best = np.where(sel, h, best)
        nx = np.where(sel, math.cos(a), nx)
        ny = np.where(sel, math.sin(a), ny)
    return nx, ny, np.full(dx.shape, np.inf)


def interp_weights(lat, px, py):
    """(index arrays, weights) of the 4 (sq) / 3 (tri) cells around the points (px, py)."""
    if lat == 'sq':
        i0, j0 = np.floor(px).astype(int), np.floor(py).astype(int)
        a, b = px - i0, py - j0
        idx = [(i0, j0), (i0 + 1, j0), (i0, j0 + 1), (i0 + 1, j0 + 1)]
        w = [(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b]
        return idx, w
    jf = py / S3H
    i_f = px - 0.5 * jf
    i0, j0 = np.floor(i_f).astype(int), np.floor(jf).astype(int)
    a, b = i_f - i0, jf - j0
    lo = (a + b) <= 1
    # lower triangle (0,0) (1,0) (0,1); upper (1,1) (0,1) (1,0) -- the (1,0)-(0,1) diagonal is a link
    idx = [(np.where(lo, i0, i0 + 1), np.where(lo, j0, j0 + 1)), (i0 + 1, j0), (i0, j0 + 1)]
    w = [np.where(lo, 1 - a - b, a + b - 1), np.where(lo, a, 1 - b), np.where(lo, b, 1 - a)]
    return idx, w


class Rule:
    def __init__(self, lat, shape, M, X, Y, cx, cy, Co, rule):
        self.rule = rule
        B = boundary_cells(lat, M)
        self.B = B
        self.bi, self.bj = np.nonzero(B)
        nx, ny, r = normals(shape, X[B], Y[B], cx, cy)
        self.p1 = interp_weights(lat, X[B] - Co * nx, Y[B] - Co * ny)
        self.p2 = interp_weights(lat, X[B] - 2 * Co * nx, Y[B] - 2 * Co * ny)
        if shape == 'circle':
            self.f1 = np.sqrt(np.clip((r - Co) / r, 0, 1))
            self.f2 = np.sqrt(np.clip((r - 2 * Co) / (r - Co), 0, 1))
        else:
            self.f1 = self.f2 = 1.0

    @staticmethod
    def at(field, p):
        idx, w = p
        return sum(wk * field[ik, jk] for (ik, jk), wk in zip(idx, w))

    def apply(self, un, u, up):
        if self.rule == 'T1':
            un[self.bi, self.bj] = self.f1 * self.at(u, self.p1)
        elif self.rule == 'T2':
            un[self.bi, self.bj] = self.f1 * (2 * self.at(u, self.p1) - self.f2 * self.at(up, self.p2))


def run(lat, shape, R, Co, T, s, src, rule=None):
    N = MM.grid_size(lat, R)
    M = MM.shape_mask(lat, shape, N, R)
    X, Y = MM.coords(lat, N)
    cx, cy = MM.centre(lat, N)
    g = np.exp(-((X - cx - src[0]) ** 2 + (Y - cy - src[1]) ** 2) / (2 * s * s)) * M
    up, u = g.copy(), g.copy()
    C2 = Co * Co
    rl = Rule(lat, shape, M, X, Y, cx, cy, Co, rule) if rule else None
    E0 = None
    for n in range(T):
        un = 2 * u - up + C2 * MM.lap(lat, u)
        if rl is not None:
            rl.apply(un, u, up)
        un[~M] = 0.0
        up, u = u, un
        if n == 0:
            E0 = MM.energy(lat, u, up, Co, M)
    return dict(N=N, M=M, u=u, up=up, E0=E0, B=boundary_cells(lat, M))


def reflection(lat, shape, Rphys, rule, src_frac, Co=0.5, s=3.0, ref=None):
    r_end = Rphys + 140
    T = int(round(r_end / Co))
    src = (src_frac * Rphys, 0.0)
    t = run(lat, shape, Rphys, Co, T, s, src, rule)
    if ref is None:
        ref = run(lat, shape, Rphys + int(0.5 * r_end) + 90, Co, T, s, src)
    off = ref['N'] // 2 - t['N'] // 2
    ru = ref['u'][off:off + t['N'], off:off + t['N']]
    rup = ref['up'][off:off + t['N'], off:off + t['N']]
    reg = t['M'] & ~t['B']
    Er = MM.energy(lat, t['u'] - ru, t['up'] - rup, Co, reg)
    return math.sqrt(max(Er, 0.0) / ref['E0']), ref


def noise(lat, shape, R, rule, Co, steps=20000, seed=5):
    rng = np.random.default_rng(seed)
    N = MM.grid_size(lat, R)
    M = MM.shape_mask(lat, shape, N, R)
    X, Y = MM.coords(lat, N)
    cx, cy = MM.centre(lat, N)
    rl = Rule(lat, shape, M, X, Y, cx, cy, Co, rule)
    u = rng.uniform(-1, 1, (N, N)) * M
    up = u.copy()
    out = {}
    for n in range(1, steps + 1):
        un = 2 * u - up + Co * Co * MM.lap(lat, u)
        rl.apply(un, u, up)
        un[~M] = 0.0
        up, u = u, un
        if n in (1000, 5000, 10000, 20000):
            out[n] = float(np.max(np.abs(u)))
            if not np.isfinite(out[n]):
                break
    return out


if __name__ == '__main__':
    res = {'date': '30/09/2026', 'reflection': {}, 'noise': {}}
    for lat, shape in (('sq', 'circle'), ('tri', 'circle'), ('tri', 'hex')):
        for sf in (0.0, 0.5):
            ref = None
            for rule in (None, 'T1', 'T2'):
                r, ref = reflection(lat, shape, 100, rule, sf, ref=ref)
                k = '%s %s src%.1fR %s' % (lat, shape, sf, rule or 'dirichlet')
                res['reflection'][k] = r
                print(k, '%.3e' % r, flush=True)
    for lat, shape in (('sq', 'circle'), ('tri', 'hex')):
        for rule in ('T1', 'T2'):
            # CFL limits: sq5 1/sqrt2 = 0.7071, tri sqrt(2/3) = 0.8165 (the web app's CO_LIMIT_T)
            for Co in ((0.5, 0.7) if lat == 'sq' else (0.5, 0.8)):
                k = '%s %s %s Co%.2f' % (lat, shape, rule, Co)
                res['noise'][k] = noise(lat, shape, 40, rule, Co)
                print(k, res['noise'][k], flush=True)
    with open(os.path.join(HERE, 'results_masked_normal_MIRROR.json'), 'w') as f:
        json.dump(res, f, indent=1)
