"""Leapfrog dispersion for the three 2D stencils, at EQUAL POINT DENSITY.

sin(w dt/2) = (c dt/2) sqrt(-L(k)),  L = the stencil's Laplacian symbol.
Phase-speed error and its spread over directions (anisotropy) at a given number of
POINTS PER WAVELENGTH, with each lattice at the same number of points per unit area:
square spacing 1, triangular spacing h_t = sqrt(2/sqrt3) = 1.07457 (cell area sqrt3/2 h_t^2 = 1).
Co is set to the same FRACTION of each lattice's own limit (README: anisotropy is set by
points per wavelength, not by Co), so the comparison is at equal stability margin.
"""
import numpy as np

S3 = np.sqrt(3)
HT = np.sqrt(2 / S3)                       # triangular spacing at unit point density

def symbol(st, kx, ky):
    if st == 'sq5':
        return -4 * (np.sin(kx / 2) ** 2 + np.sin(ky / 2) ** 2)
    if st == 'sq9':
        return (2/3) * (2*np.cos(kx) + 2*np.cos(ky) - 4) + (1/6) * (4*np.cos(kx)*np.cos(ky) - 4)
    # tri: 3 bond directions of length HT at 0, 60, 120 deg; lap = (2/3)/h^2 sum_m (u_m - u0)
    s = 0
    for a in (0, np.pi / 3, 2 * np.pi / 3):
        s = s + np.sin(HT * (kx * np.cos(a) + ky * np.sin(a)) / 2) ** 2
    return -(8 / 3) * s / HT ** 2

LIM = {'sq5': 1 / np.sqrt(2), 'sq9': S3 / 2, 'tri': np.sqrt(2 / 3)}
SPACING = {'sq5': 1.0, 'sq9': 1.0, 'tri': HT}

def phase_speed(st, ppw, frac=0.5, nang=721):
    """c = 1; returns v_phase / c over directions for wavelength ppw (in unit-density points)"""
    h = SPACING[st]; dt = frac * LIM[st] * h        # Co = c dt / h = frac * limit
    k = 2 * np.pi / ppw
    th = np.linspace(0, np.pi, nang)
    kx, ky = k * np.cos(th), k * np.sin(th)
    L = symbol(st, kx, ky)
    w = 2 / dt * np.arcsin(np.clip(dt / 2 * np.sqrt(-L), -1, 1))
    return w / k, dt

print(f'{"stencil":6} {"PPW":>4} {"mean |v/c-1|":>13} {"anisotropy":>11} {"dt_max (h/c)":>13} {"steps/unit t":>13}')
for ppw in (6, 8, 12, 16):
    for st in ('sq5', 'sq9', 'tri'):
        v, dt = phase_speed(st, ppw)
        dtmax = LIM[st] * SPACING[st]
        print(f'{st:6} {ppw:4d} {np.mean(np.abs(v - 1)):13.3e} {v.max() / v.min() - 1:11.3e} '
              f'{dtmax:13.5f} {1 / dtmax:13.5f}')
    print()
# check: the tri symbol is isotropic to leading order (-k^2) and at 4th order
k = 0.05; th = np.linspace(0, np.pi, 361)
for st in ('sq5', 'sq9', 'tri'):
    L = symbol(st, k*np.cos(th), k*np.sin(th))
    print(f'{st}: -L/k^2 at k=0.05 ranges {(-L/k**2).min():.9f} .. {(-L/k**2).max():.9f}')
