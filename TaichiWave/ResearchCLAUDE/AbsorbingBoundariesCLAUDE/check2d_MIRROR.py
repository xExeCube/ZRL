"""Check that the 1D reduction in abc1d_MIRROR.py is exact: run a genuine 2D grid, periodic in y
with Ny = 32 so that ky = 2 pi / 32 fits (lambda = 16 at theta = 30 deg), with the wall rule
applied to every row using the real tangential second difference, and compare it with the 1D
run times cos(ky j).       python check2d_MIRROR.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abc1d_MIRROR as A  # noqa: E402


def run2d(g, left, Ny):
    Nx, Co, ky = g['Nx'], g['Co'], g['ky']
    C2 = Co * Co
    a0, a1 = A.packet(Nx, g['i0'], g['sig'], g['kx'], g['w'], g['vg'])
    cy = np.cos(ky * np.arange(Ny))[None, :]
    kind = left[0]
    hist = max(3, A.rule_depth(left[1])[1] + 1) if kind == 'rule' else 3
    H = [np.zeros((Nx, Ny)) for _ in range(hist)]
    H[2][:] = a0[:, None] * cy
    H[1][:] = a1[:, None] * cy
    if kind == 'cpml':
        s1 = A.Sim1D(Nx, Co, ky, left)          # borrow its coefficient profiles
        m1 = np.zeros((Nx - 1, Ny))
        m2 = np.zeros((Nx, Ny))
    for _ in range(g['T']):
        u, up, un = H[1], H[2], H[0]
        lapy = np.roll(u, 1, 1) + np.roll(u, -1, 1) - 2 * u
        if kind == 'cpml':
            gx = u[1:] - u[:-1]
            m1 = s1.bf_[:, None] * m1 + s1.af_[:, None] * gx
            q = gx + m1
            dq = np.zeros_like(u)
            dq[1:-1] = q[1:] - q[:-1]
            m2 = s1.bc_[:, None] * m2 + s1.ac_[:, None] * dq
            lapx = dq + m2
        else:
            lapx = np.zeros_like(u)
            lapx[1:-1] = u[:-2] + u[2:] - 2 * u[1:-1]
        un[:] = 2 * u - up + C2 * (lapx + lapy)
        un[-1] = 0.0
        if kind == 'rule':
            acc = 0.0
            c00 = 1.0
            for a, b, c0, c1 in left[1]:
                if a == 0 and b == 0:
                    c00 = c0
                    continue
                row = H[b][a]
                acc = acc + c0 * row + c1 * (np.roll(row, 1) + np.roll(row, -1) - 2 * row)
            un[0] = -acc / c00
        else:
            un[0] = 0.0
        H.insert(0, H.pop())
    return H[1], cy


def main():
    Co = 0.5
    g = A.geometry(Co, 16, 30)
    assert abs(g['ky'] - 2 * math.pi / 32) < 1e-12
    cases = {'mur2': (('rule', A.rule_mur2(Co)), 1),
             'higdon3(0,30,60)e.005': (('rule', A.rule_higdon(Co, [0, 30, 60], 0.005)), 1),
             'cpml10': (('cpml', 10, 1.0, 2, 0.0), 10)}
    for name, (left, L) in cases.items():
        u2, cy = run2d(g, left, 32)
        s = A.Sim1D(g['Nx'], Co, g['ky'], left)
        a0, a1 = A.packet(g['Nx'], g['i0'], g['sig'], g['kx'], g['w'], g['vg'])
        s.set(a0, a1)
        for _ in range(g['T']):
            s.step()
        dev = np.max(np.abs(u2 - s.u()[:, None] * cy))
        print('%-24s max |u_2D - u_1D cos(ky j)| = %.2e   (max |u| = %.2e)' % (name, dev, np.max(np.abs(u2))))


if __name__ == '__main__':
    main()
