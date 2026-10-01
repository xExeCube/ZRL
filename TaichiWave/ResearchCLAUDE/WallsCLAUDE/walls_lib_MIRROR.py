"""Walls research (29/09/2026): shared numpy/scipy prototype of the lattice edge rules.

Everything here is a PROTOTYPE for NotesCLAUDE/rigid_vs_pinned_walls_MIRROR.md; nothing in the
Taichi port imports it. The operators are written as explicit sparse matrices so that the
properties that matter (symmetry, spectrum, CFL radius, energy) can be checked directly.

Lattices (the port's stencils, h = 1, lap ~ del^2):
  sq5  lap = sum_4 (u_j - u_i)                                       Co_max = 1/sqrt(2)
  sq9  lap = (2/3) sum_edges (u_j - u_i) + (1/6) sum_diag (u_j - u_i)  Co_max = sqrt(3)/2
  tri  lap = (2/3) sum_6 (u_j - u_i), axial offsets as membrane_core   Co_max = sqrt(2/3)
Each is a graph Laplacian: lap_i = sum_links w_ij (u_j - u_i).

Cell types:  FREE = 0 (fluid), PIN = 1 (u = 0, Dirichlet), RIG = 2 (rigid, Neumann).

Edge rules, per link i -> j from a FREE cell i:
  j FREE     : w (u_j - u_i)                      (unless the link is corner-BLOCKED, below)
  j PIN      : w (0 - u_i)                        (the current rule: keep the link, value 0)
  j RIG      : dropped                            (link removal)
  BLOCKED    : a link between two FREE cells whose two common neighbours are both solid
               (sq9 diagonals: the two edge cells it passes between; tri: the two triangle
               apexes). It crosses a wall through a corner-to-corner gap. Treated as a link to
               a PIN cell if either blocking cell is PIN, else dropped. Symmetric by construction:
               both endpoints see the same pair of common neighbours.
  TRANSFER   : (rule R1) a dropped RIG link whose common neighbours are one FREE (k) and one
               RIG gives tau*w to the link i-k (only if i-k is itself an active, unblocked
               link), then the transfers are symmetrised
               W = (T + T^T)/2. tau restores the tangential second moment of the stencil at a
               flat wall: sq9 tau = 1 (this IS the even ghost mirror), tri tau = 1/4.
"""
import math
import numpy as np
import scipy.sparse as sp

S3H = math.sqrt(3) / 2
FREE, PIN, RIG = 0, 1, 2


class Lattice:
    def __init__(self, name, links, A, co_max, tau):
        self.name = name
        self.links = links            # list of (d, w, commons) ; d = (di, dj)
        self.A = np.array(A, float)   # index -> physical: x = A @ (i, j)
        self.co_max = co_max
        self.tau = tau
        self.area = abs(np.linalg.det(self.A))

    def xy(self, I, J):
        return self.A[0, 0] * I + self.A[0, 1] * J, self.A[1, 0] * I + self.A[1, 1] * J


def _mk_sq5():
    L = [((1, 0), 1.0, []), ((-1, 0), 1.0, []), ((0, 1), 1.0, []), ((0, -1), 1.0, [])]
    return Lattice('sq5', L, [[1, 0], [0, 1]], 1 / math.sqrt(2), 0.0)


def _mk_sq9():
    L = [((1, 0), 2 / 3, []), ((-1, 0), 2 / 3, []), ((0, 1), 2 / 3, []), ((0, -1), 2 / 3, [])]
    for a in (1, -1):
        for b in (1, -1):
            L.append(((a, b), 1 / 6, [(a, 0), (0, b)]))
    return Lattice('sq9', L, [[1, 0], [0, 1]], math.sqrt(3) / 2, 1.0)


def _mk_tri():
    offs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    L = []
    for d in offs:
        com = [c for c in offs if (d[0] - c[0], d[1] - c[1]) in offs]
        assert len(com) == 2
        L.append((d, 2 / 3, com))
    return Lattice('tri', L, [[1, 0.5], [0, S3H]], math.sqrt(2 / 3), 0.25)


SQ5, SQ9, TRI = _mk_sq5(), _mk_sq9(), _mk_tri()
LATS = {'sq5': SQ5, 'sq9': SQ9, 'tri': TRI}


def shift(a, d, fill):
    """b[i, j] = a[i + di, j + dj] (fill outside the array)."""
    di, dj = d
    n0, n1 = a.shape
    b = np.full_like(a, fill)
    si0, si1 = max(0, -di), min(n0, n0 - di)
    sj0, sj1 = max(0, -dj), min(n1, n1 - dj)
    b[si0:si1, sj0:sj1] = a[si0 + di:si1 + di, sj0 + dj:sj1 + dj]
    return b


def build_L(T, lat, block='both', transfer=False, return_parts=False):
    """Sparse operator on the FULL grid (rows of solid cells are empty). T: int array of cell
    types. Outside the array counts as PIN. block: 'both' | 'none' | 'either'."""
    n0, n1 = T.shape
    NN = n0 * n1
    idx = np.arange(NN).reshape(n0, n1)
    free = T == FREE
    rows, cols, vals = [], [], []
    diag = np.zeros((n0, n1))
    tr_r, tr_c, tr_v = [], [], []
    stats = dict(pinned_links=0, dropped_links=0, blocked=0, transfers=0)
    # active (free-free, unblocked) links per offset: a transfer may only reinforce one of
    # these -- found 29/09: on the tri lattice a transfer next to a corner of the ring
    # revived the blocked link across a one-cell diagonal wall (a 21% leak).
    active = {}
    for d, w, com in lat.links:
        a = free & (shift(T, d, PIN) == FREE)
        if com and block != 'none':
            s1, s2 = shift(T, com[0], PIN) != FREE, shift(T, com[1], PIN) != FREE
            a &= ~((s1 & s2) if block == 'both' else (s1 | s2))
        active[d] = a
    for d, w, com in lat.links:
        Tn = shift(T, d, PIN)
        jdx = shift(idx, d, -1)
        blk = np.zeros_like(free)
        blk_pin = np.zeros_like(free)
        if com and block != 'none':
            c1, c2 = shift(T, com[0], PIN), shift(T, com[1], PIN)
            s1, s2 = c1 != FREE, c2 != FREE
            blk = (s1 & s2) if block == 'both' else (s1 | s2)
            blk_pin = (c1 == PIN) | (c2 == PIN)
        nfree = Tn == FREE
        act = free & nfree & ~blk
        rows.append(idx[act]); cols.append(jdx[act]); vals.append(np.full(act.sum(), w))
        diag[act] -= w
        pin = free & ((Tn == PIN) | (nfree & blk & blk_pin))
        diag[pin] -= w
        stats['pinned_links'] += int(pin.sum())
        dropped = free & ((Tn == RIG) | (nfree & blk & ~blk_pin))
        stats['dropped_links'] += int(dropped.sum())
        stats['blocked'] += int((free & nfree & blk).sum())
        if transfer and com:
            # only links dropped because the ENDPOINT is rigid transfer (a blocked free-free
            # link has no mirror image); exactly one common neighbour free, the other solid
            c1, c2 = shift(T, com[0], PIN), shift(T, com[1], PIN)
            base = free & (Tn == RIG)
            for cf, cs, co in ((c1, c2, com[0]), (c2, c1, com[1])):
                m = base & (cf == FREE) & (cs == RIG) & active[tuple(co)]
                kdx = shift(idx, co, -1)
                tr_r.append(idx[m]); tr_c.append(kdx[m]); tr_v.append(np.full(m.sum(), lat.tau * w))
                stats['transfers'] += int(m.sum())
    L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(NN, NN))
    L = L + sp.diags(diag.ravel())
    if transfer and tr_r:
        Tm = sp.csr_matrix((np.concatenate(tr_v), (np.concatenate(tr_r), np.concatenate(tr_c))), shape=(NN, NN))
        W = 0.5 * (Tm + Tm.T)
        L = L + W - sp.diags(np.asarray(W.sum(axis=1)).ravel())
    L = L.tocsr()
    L.eliminate_zeros()
    if return_parts:
        return L, stats
    return L


RULES = {  # name -> (solid type, block, transfer)
    'P':   (PIN, 'none', False),     # the current app rule
    'Pcb': (PIN, 'both', False),     # pinned + corner blocking
    'R0':  (RIG, 'both', False),     # rigid, link removal (+ corner blocking)
    'R0nb': (RIG, 'none', False),    # rigid, link removal, NO corner blocking
    'R1':  (RIG, 'both', True),      # rigid, link removal + tangential-moment transfer
}


def op_for(solid, lat, rule):
    """solid: bool array (True = wall). Returns the full-grid operator for the named rule."""
    typ, block, tr = RULES[rule]
    T = np.where(solid, typ, FREE).astype(np.int8)
    return build_L(T, lat, block=block, transfer=tr)


def sym_err(L):
    D = (L - L.T).tocsr()
    return 0.0 if D.nnz == 0 else float(np.abs(D.data).max())


def lam_max(L, free_mask=None):
    """Largest |eigenvalue| of -L over the free cells (the CFL radius: stable iff
    Co^2 * lam_max < 4)."""
    import scipy.sparse.linalg as sla
    if free_mask is not None:
        k = np.flatnonzero(free_mask.ravel())
        L = L[k][:, k]
    v = sla.eigsh(-L, k=1, which='LA', return_eigenvectors=False, tol=1e-10)
    return float(v[0])


def energy(L, u0, u1, co):
    """Leapfrog's conserved energy for symmetric L between levels u0 (n) and u1 (n+1)."""
    du = u1 - u0
    return float(du @ du) / (co * co) - float(u1 @ (L @ u0))


def run(L, u_prev, u_cur, co, steps, every=0, cb=None):
    c2 = co * co
    a, b = u_prev.copy(), u_cur.copy()
    for s in range(steps):
        c = 2 * b - a + c2 * (L @ b)
        a, b = b, c
        if every and cb is not None and (s + 1) % every == 0:
            cb(s + 1, a, b)
    return a, b


def grid_xy(lat, n0, n1, ci, cj):
    I, J = np.meshgrid(np.arange(n0) - ci, np.arange(n1) - cj, indexing='ij')
    return lat.xy(I.astype(float), J.astype(float))
