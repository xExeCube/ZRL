"""Long-wave speed through a tiling of density cells (spec §3.5): measured against c0 / sqrt(mean m).

The medium is the apps' own form, u+ - 2u + u- = CC_i (L u)_i with CC_i = (Co f_i)^2, read as a node mass
M_i = 1 / f_i^2. With the density reading "mass = density count" (f = 1/sqrt(m)), M_i = m_i.
For a mass-only variation the long-wave limit is the mass average: c_eff = c0 / sqrt(mean M).

Measured on a periodic box (sq5, sq9 square torus; tri rhombic torus) holding whole tiling periods:
the lowest non-zero eigenvalues of  -L u = lam M u  against the uniform box's (M = 1):
c_eff / c0 = sqrt(lam / lam0). Two box sizes per case, to show the approach to the long-wave limit.

The density m_i is the number of circumcircle discs covering node i, read three ways:
  'open'  d < R (strict), 'closed' d <= R, 'area' the covered area of the node's own square or hexagon
  (supersampled 8 x 8). The tiling's vertices are nodes and sit exactly on several circles, so 'open'
  and 'closed' differ there.

Run: python effective_speed_MIRROR.py      (numpy + scipy, ~1 min)
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

S3 = np.sqrt(3.0)

def operator(kind, N):
    """-L on the N x N torus as a sparse matrix (the apps' stencils)."""
    idx = np.arange(N * N).reshape(N, N)
    if kind == 'sq5':
        nb = [((1, 0), 1.0), ((-1, 0), 1.0), ((0, 1), 1.0), ((0, -1), 1.0)]
    elif kind == 'sq9':
        nb = [((1, 0), 2 / 3), ((-1, 0), 2 / 3), ((0, 1), 2 / 3), ((0, -1), 2 / 3),
              ((1, 1), 1 / 6), ((1, -1), 1 / 6), ((-1, 1), 1 / 6), ((-1, -1), 1 / 6)]
    else:  # tri: (a, b) lattice coordinates, six neighbours, weight 2/3
        nb = [((1, 0), 2 / 3), ((-1, 0), 2 / 3), ((0, 1), 2 / 3), ((0, -1), 2 / 3), ((1, -1), 2 / 3), ((-1, 1), 2 / 3)]
    rows, cols, vals = [], [], []
    diag = sum(w for _, w in nb)
    for (da, db), w in nb:
        rows.append(idx.ravel()); cols.append(np.roll(np.roll(idx, -da, 0), -db, 1).ravel()); vals.append(np.full(N * N, -w))
    rows.append(idx.ravel()); cols.append(idx.ravel()); vals.append(np.full(N * N, diag))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N * N, N * N))

def positions(kind, N, sub=None):
    a, b = np.meshgrid(np.arange(N, dtype=float), np.arange(N, dtype=float), indexing='ij')
    if sub is not None:  # offsets inside the node's own cell (square, or the hexagon's rhombus approximation)
        a, b = a[..., None] + sub[0], b[..., None] + sub[1]
    if kind == 'tri':
        return a + b / 2, b * S3 / 2
    return a, b

def tiling(flavor, size, kind):
    """(centres in one period, period vectors, R, box period in nodes) for a tiling whose vertices are nodes."""
    if kind == 'tri':
        e1, e2 = np.array([1.0, 0.0]), np.array([0.5, S3 / 2])
        if flavor == 'hexagon':      # side R = size; centre superlattice R(1,1), R(-1,2); box N = 3R q
            R = size; A1, A2 = R * (e1 + e2), R * (2 * e2 - e1); C = [np.zeros(2)]; per = 3 * size
        else:                        # triangle side s = size; R = s / sqrt3; box N = s q
            R = size / S3; A1, A2 = size * e1, size * e2; C = [(A1 + A2) / 3, 2 * (A1 + A2) / 3]; per = size
    else:
        if flavor == 'square-grid':  # grid isomer, side s = size, R = s / sqrt2
            R = size / np.sqrt(2); A1, A2 = np.array([size, 0.0]), np.array([0.0, size]); C = [np.array([size / 2] * 2)]; per = size
        else:                        # ZRL isomer (vertex on the x-axis), R = size; centres (R,R), (R,-R) lattice
            R = size; A1, A2 = np.array([R, R]), np.array([R, -R]); C = [np.zeros(2)]; per = 2 * size
    return C, A1, A2, R, per

def density(kind, flavor, size, N, rule):
    C, A1, A2, R, per = tiling(flavor, size, kind)
    # superlattice indices that cover the box (plus two cells of margin): invert [A1 A2] at the box corners
    X0, Y0 = positions(kind, N)
    corners = np.array([[X0.min(), Y0.min()], [X0.max(), Y0.min()], [X0.min(), Y0.max()], [X0.max(), Y0.max()]])
    ij = np.linalg.solve(np.column_stack([A1, A2]), corners.T)
    lo_i, hi_i = int(np.floor(ij[0].min())) - 3, int(np.ceil(ij[0].max())) + 3
    lo_j, hi_j = int(np.floor(ij[1].min())) - 3, int(np.ceil(ij[1].max())) + 3
    P = np.array([c + i * A1 + j * A2 for c in C for i in range(lo_i, hi_i + 1) for j in range(lo_j, hi_j + 1)])
    if rule == 'area':
        g = (np.arange(8) + 0.5) / 8 - 0.5
        sa, sb = np.meshgrid(g, g, indexing='ij'); sub = (sa.ravel(), sb.ravel())
        X, Y = positions(kind, N, sub)
    else:
        X, Y = positions(kind, N); X, Y = X[..., None], Y[..., None]
    m = np.zeros(X.shape)
    lo = np.array([X.min() - 2 * R, Y.min() - 2 * R]); hi = np.array([X.max() + 2 * R, Y.max() + 2 * R])
    P = P[(P[:, 0] > lo[0]) & (P[:, 0] < hi[0]) & (P[:, 1] > lo[1]) & (P[:, 1] < hi[1])]
    for p in P:
        d2 = (X - p[0]) ** 2 + (Y - p[1]) ** 2
        if rule == 'open': m += d2 < R * R * (1 - 1e-12)
        else: m += d2 <= R * R * (1 + 1e-12)
    return m.mean(-1) if rule == 'area' else m[..., 0], R, per

def lowest(A, M, k=10):
    w = eigsh(A, k=k, M=M, sigma=-1e-6, which='LM', return_eigenvectors=False)
    w = np.sort(w); return w[w > 1e-9][0]

if __name__ == '__main__':
    cases = [('sq5', 'square-grid', 12), ('sq9', 'square-grid', 12), ('sq5', 'square-zrl', 6),
             ('tri', 'hexagon', 4), ('tri', 'triangle', 12)]
    print('c_eff / c0 measured (lowest Bloch mode) vs 1/sqrt(mean m) predicted; m = circumcircle count, M = m')
    for kind, flavor, size in cases:
        for rule in ('open', 'closed', 'area'):
            line, means = [], []
            for q in (6, 12):
                C, A1, A2, R, per = tiling(flavor, size, kind)
                N = per * q
                m, R, per = density(kind, flavor, size, N, rule)
                A = operator(kind, N)
                lam0 = lowest(A, sp.identity(N * N, format='csr'))
                lam = lowest(A, sp.diags(m.ravel()).tocsr())
                meas, pred = np.sqrt(lam / lam0), 1 / np.sqrt(m.mean())
                means.append(m.mean())
                line.append(f'N {N:>3} ({q} periods): mean m {m.mean():.4f} measured {meas:.5f} predicted {pred:.5f} diff {meas - pred:+.1e}')
            assert abs(means[0] - means[1]) < 1e-12, 'the box must hold whole periods: mean m differs between sizes'
            print(f'{kind} {flavor:<11} R = {R:6.3f} nodes, rule {rule:<6}: ' + ' | '.join(line))
