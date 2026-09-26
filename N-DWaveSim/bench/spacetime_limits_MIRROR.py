"""The time x distance grid: its light cone, its CFL limits, and what accuracy costs.

Part 1 -- the N-D generalisation of "C_max = c = 1" (ZRL canon, R^4 list).
  One tick moves information one BOND. After m ticks the field of a one-cell impulse is
  confined to the REACH POLYTOPE of radius m: the convex hull of the stencil's neighbour
  offsets, scaled by m. That is the lattice's own light cone, speed u0 = 1 bond per tick
  (= h*F, frame rate times spacing). A wave of speed c fills a BALL of radius c*t, and the
  ball must fit inside the polytope (Courant-Friedrichs-Lewy 1928, domain of dependence):
        c <= u0 * inradius(reach polytope)          -- NECESSARY
  The sharp limit is von Neumann's: Co_max = 2 / sqrt(max_k |L(k)|), L the stencil symbol.
  For the (2D+1)-point cubic stencil the reach polytope is the cross-polytope, inradius
  1/sqrt(D), and the two coincide (the canon's "signature" identification). Where do they not?

Part 2 -- points per wavelength (PPW): the phase-speed error of leapfrog + each stencil,
  its dependence on Co, the PPW needed for a target accuracy, and the cost of that accuracy
  per simulated time at equal wall-clock work per cell (measured JS rates, bench/sweep).
"""
import itertools, math
import numpy as np
from scipy.spatial import ConvexHull

S3 = math.sqrt(3)

# ---------------- stencils: neighbour offsets (bond length 1) and Laplacian weights --------
def cubic(D):   # (2D+1)-point
    return [tuple(s * (i == j) for j in range(D)) for i in range(D) for s in (1, -1)], None

def sq9():
    edge = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    diag = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    return edge + diag, {**{e: 2/3 for e in edge}, **{d: 1/6 for d in diag}}

def tri():
    return [(math.cos(a), math.sin(a)) for a in np.arange(6) * np.pi / 3], None

def bcc():
    return [tuple(np.array(s) / S3) for s in itertools.product((1, -1), repeat=3)], None

def fcc():
    out = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        for si, sj in itertools.product((1, -1), repeat=2):
            v = [0, 0, 0]; v[i] = si; v[j] = sj; out.append(tuple(np.array(v) / math.sqrt(2)))
    return out, None

def symbol_fn(offs, weights):
    """L(k) = sum_m w_m (cos(k.d_m) - 1), normalised so L ~ -|k|^2 (second-order Laplacian)"""
    D = np.array(offs, float); dim = D.shape[1]
    if weights is None:
        w = np.full(len(D), 2 * dim / len(D))     # sum d(x)d = (Z/D) I  ->  coefficient 2D/Z
    else:
        w = np.array([weights[tuple(int(round(x)) for x in d)] for d in offs])
    return lambda K: (w[None, :] * (np.cos(K @ D.T) - 1)).sum(1)

def von_neumann(offs, weights, dim, n=41):
    # scan a box that contains the Brillouin zone (reciprocal vectors are at most 2pi/|a|
    # with |a| >= the bond length 1 for these lattices -> a box of half-width 2pi covers it),
    # then POLISH the best candidates with a local minimiser: the extremal k of BCC
    # (pi*sqrt3, 0, 0) and of the triangular K point are not on any rational grid, and the
    # raw scan read 0.578367 / 0.816881 instead of 1/sqrt3 / sqrt(2/3)
    from scipy.optimize import minimize
    f = symbol_fn(offs, weights)
    g = np.linspace(-2 * np.pi, 2 * np.pi, n)
    K = np.array(list(itertools.product(g, repeat=dim)))
    L = f(K)
    best = L.min()
    for k0 in K[np.argsort(L)[:12]]:
        r = minimize(lambda k: f(k[None, :])[0], k0, method='Nelder-Mead',
                     options={'xatol': 1e-12, 'fatol': 1e-15, 'maxiter': 20000})
        best = min(best, r.fun)
    return 2 / math.sqrt(-best)

def reach_inradius(offs):
    P = np.array(offs, float)
    if P.shape[1] == 1: return float(np.abs(P).min())
    H = ConvexHull(P)
    return float((-H.equations[:, -1]).min())     # facet offsets: n.x + c = 0, |n| = 1

cases = [('1D  (3-point)', *cubic(1), 1), ('2D square 5-point', *cubic(2), 2),
         ('2D square 9-point', *sq9(), 2), ('2D triangular 6', *tri(), 2),
         ('3D SC 7-point', *cubic(3), 3), ('3D BCC 8', *bcc(), 3), ('3D FCC 12', *fcc(), 3),
         ('4D tesseractic 9-point', *cubic(4), 4)]
print('PART 1 -- the lattice light cone vs the sharp CFL limit (u0 = 1 bond per tick)')
print(f'{"stencil":24}{"reach polytope":>16}{"inradius (necessary)":>22}{"von Neumann (sharp)":>21}   coincide?')
for name, offs, wts, dim in cases:
    ri = reach_inradius(offs)
    vn = von_neumann(offs, wts, dim, n=41 if dim <= 2 else (25 if dim == 3 else 13))
    poly = {1: 'segment', 2: {4: 'diamond', 8: 'square', 6: 'hexagon'}.get(len(offs), '?'),
            3: {6: 'octahedron', 8: 'cube', 12: 'cuboctahedron'}.get(len(offs), '?'), 4: '16-cell'}[dim]
    print(f'{name:24}{poly:>16}{ri:22.6f}{vn:21.6f}   {"YES" if abs(ri - vn) < 2e-3 else "no (gap %.4f)" % (ri - vn)}')

# ---------------- Part 2: PPW accuracy -------------------------------------------------------
LIM = {'sq5': 1 / math.sqrt(2), 'sq9': S3 / 2, 'tri': math.sqrt(2 / 3), 'sq24': None}
SYM = {'sq5': symbol_fn(*cubic(2)), 'sq9': symbol_fn(*sq9()), 'tri': symbol_fn(*tri())}
def sym_sq24(K):   # 4th-order square Laplacian: (-1/12, 4/3, -5/2, 4/3, -1/12) per axis
    return sum((-1/6) * np.cos(2 * K[:, i]) + (8/3) * np.cos(K[:, i]) - 5/2 for i in range(2))
SYM['sq24'] = sym_sq24
LIM['sq24'] = 2 / math.sqrt(-sym_sq24(np.array([[np.pi, np.pi]])).min())
CELL_AREA = {'sq5': 1.0, 'sq9': 1.0, 'tri': S3 / 2, 'sq24': 1.0}
# measured JS cell-updates/s at N = 1281 (bench/sweep_MIRROR.csv); (2,4) assumed 1.5x the
# 5-point work (9 reads vs 5) -- NOT measured, marked below
WORK = {'sq5': 1.0, 'sq9': 193 / 114, 'tri': 193 / 170, 'sq24': 1.5}

def phase_err(st, ppw, frac, nang=360):
    dt = frac * LIM[st]                      # h = 1, c = 1: Co = dt
    k = 2 * np.pi / ppw; th = np.linspace(0, np.pi, nang, endpoint=False)
    K = np.stack([k * np.cos(th), k * np.sin(th)], 1)
    L = SYM[st](K)
    w = 2 / dt * np.arcsin(np.clip(dt / 2 * np.sqrt(-L), -1, 1))
    v = w / k
    return np.abs(v - 1).mean(), np.abs(v - 1).max(), v.max() / v.min() - 1

print('\nPART 2a -- mean phase-speed error vs Co (as a fraction of each limit), PPW = 8')
print(f'{"stencil":8}' + ''.join(f'{f:>11.2f}' for f in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)))
for st in ('sq5', 'sq9', 'tri', 'sq24'):
    print(f'{st:8}' + ''.join(f'{phase_err(st, 8, f)[0]:11.3e}' for f in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)))
v_diag = phase_err('sq5', 8, 0.999999)[:2]
th = np.pi / 4; k = 2 * np.pi / 8; dt = 0.999999 / math.sqrt(2)
w = 2 / dt * np.arcsin(dt / 2 * math.sqrt(-SYM['sq5'](np.array([[k * math.cos(th), k * math.sin(th)]]))[0]))
print(f'  (5-point at the limit, along the DIAGONAL: v/c - 1 = {w / k - 1:.2e} -- dispersion-free there)')

def ppw_needed(st, frac, tol, worst=False):
    lo, hi = 2.05, 400.0
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        e = phase_err(st, mid, frac)[1 if worst else 0]
        lo, hi = (mid, hi) if e > tol else (lo, mid)
    return hi

print('\nPART 2b -- PPW needed, and the cost of that accuracy per simulated time')
print('cost ~ (points per lambda^2) x (steps per unit time) x (work per update)')
print('     = PPW^2/A_cell x PPW/Co x W       (relative to square 5-point at Co = 0.5 limit, 1%)')
ref = None
rows = []
for tol in (1e-2, 1e-3):
    for st in ('sq5', 'sq9', 'tri', 'sq24'):
        for frac in (0.5, 0.95):
            p = ppw_needed(st, frac, tol)
            cost = p ** 3 / CELL_AREA[st] / (frac * LIM[st]) * WORK[st]
            rows.append((tol, st, frac, p, cost))
ref = [r for r in rows if r[0] == 1e-2 and r[1] == 'sq5' and r[2] == 0.5][0][4]
print(f'{"target":>8}{"stencil":>8}{"Co/lim":>8}{"PPW":>8}{"rel. cost":>11}')
for tol, st, frac, p, cost in rows:
    print(f'{tol:8.0e}{st:>8}{frac:8.2f}{p:8.2f}{cost / ref:11.3f}' + ('   (work assumed)' if st == 'sq24' else ''))
print(f'\n(2,4) square limit: Co <= {LIM["sq24"]:.6f}')

def group_err(st, ppw, frac, nang=360):
    """|v_group/c - 1| mean and worst over directions (central difference in |k|)"""
    dt = frac * LIM[st]; k = 2 * np.pi / ppw; dk = k * 1e-4
    th = np.linspace(0, np.pi, nang, endpoint=False)
    def w(kk):
        K = np.stack([kk * np.cos(th), kk * np.sin(th)], 1)
        return 2 / dt * np.arcsin(np.clip(dt / 2 * np.sqrt(-SYM[st](K)), -1, 1))
    vg = (w(k + dk) - w(k - dk)) / (2 * dk)
    return np.abs(vg - 1).mean(), np.abs(vg - 1).max()

print('\nPART 2c -- the best configurations at PPW = 8: mean / WORST-direction phase error,')
print('anisotropy, and the GROUP-speed error (what a pulse front feels)')
print(f'{"stencil":8}{"Co/lim":>7}{"phase mean":>12}{"phase worst":>12}{"anisotropy":>12}{"group mean":>12}{"group worst":>12}')
for st, fr in (('sq5', 0.5), ('sq5', 0.95), ('sq9', 0.95), ('tri', 0.5), ('tri', 0.95), ('sq24', 0.25)):
    pm, pw_, an = phase_err(st, 8, fr); gm, gw = group_err(st, 8, fr)
    print(f'{st:8}{fr:7.2f}{pm:12.3e}{pw_:12.3e}{an:12.3e}{gm:12.3e}{gw:12.3e}')
print('\nPPW needed for a WORST-direction phase error of 1% / 0.1%:')
for st, fr in (('sq5', 0.5), ('sq5', 0.95), ('sq9', 0.95), ('tri', 0.5), ('tri', 0.95), ('sq24', 0.25)):
    print(f'  {st:5} Co/lim {fr:.2f}:  {ppw_needed(st, fr, 1e-2, worst=True):6.2f}  /  {ppw_needed(st, fr, 1e-3, worst=True):6.2f}')
