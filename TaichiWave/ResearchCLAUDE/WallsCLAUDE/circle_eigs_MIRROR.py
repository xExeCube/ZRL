"""Walls research (29/09/2026): staircase accuracy -- the lowest eigenfrequencies of a circular
domain of radius R (cells) vs the exact Bessel values, as R grows.

Operator eigenvalues L v = -Lam v, kappa = sqrt(Lam). The time step plays no part (leapfrog
maps omega <-> Lam one-to-one), so this isolates the SPACE error. The lattice's own bulk
dispersion is removed to leading order with the angle-averaged 4th-order symbol term
(<Lam(k)>_theta = k^2 + c4 k^4: sq5 c4 = -1/16, sq9 and tri: computed below), so what is left
is the BOUNDARY error. Mask: a cell is free iff its centre is at distance < R.

Rules: P (pinned = Dirichlet), R0, R1 (rigid = Neumann), and on sq5 three reference methods:
  RG  ghost mirror along the true normal (bilinear interpolation, NOT symmetric);
  CF  finite-volume cut cells for Neumann (face apertures + volume fractions, V-weighted
      symmetric), with a volume-fraction floor Vmin against the small-cell CFL collapse;
  GB  symmetric Dirichlet with the link fraction theta (the Gibou et al. 2002 treatment),
      with a floor theta_min.

Exact: Dirichlet j01 = 2.404826, j11 = 3.831706; Neumann j'11 = 1.841184, j'21 = 3.054237,
j'02 = 3.831706 (= j11).

    python circle_eigs_MIRROR.py
"""
import sys, os, math, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walls_lib_MIRROR as W

J01, J11 = 2.404825557695773, 3.831705970207512
JP11, JP21, JP02 = 1.841183780989569, 3.054236928227140, 3.831705970207512


def c4_of(lat):
    """Angle-averaged 4th-order coefficient of the symbol: <Lam(k)> = k^2 + c4 k^4 + ..."""
    k = 1e-2
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    vals = []
    for t in th:
        kv = k * np.array([math.cos(t), math.sin(t)])
        vals.append(sum(w * (1 - math.cos(kv @ (lat.A @ np.array(d)))) for d, w, c in lat.links))
    return (np.mean(vals) - k * k) / k ** 4


def k_equiv(kappa, c4):
    # solve k^2 + c4 k^4 = kappa^2 for k (small-k branch)
    lam = kappa * kappa
    return np.sqrt((-1 + np.sqrt(np.maximum(1 + 4 * c4 * lam, 0))) / (2 * c4)) if c4 != 0 else kappa


def disk(lat, R, off=(0.0, 0.0)):
    n = int(2 * R / (W.S3H if lat.name == 'tri' else 1) + 8) | 1
    if lat.name == 'tri':
        n = int(2 * R / W.S3H + 2 * R + 8) | 1
    c = (n - 1) // 2
    X, Y = W.grid_xy(lat, n, n, c, c)
    X, Y = X - off[0], Y - off[1]
    solid = X * X + Y * Y >= R * R
    return solid, X, Y


def low_eigs(L, free, k=8, sigma=-1e-3, M=None):
    idx = np.flatnonzero(free.ravel())
    Lf = (-L)[idx][:, idx].tocsc()
    if M is not None:
        vals = sla.eigsh(Lf, k=k, M=M, sigma=sigma, which='LM', return_eigenvectors=False)
    else:
        vals = sla.eigsh(Lf, k=k, sigma=sigma, which='LM', return_eigenvectors=False)
    return np.sort(np.sqrt(np.maximum(vals, 0)))


def errs_dir(kap, R, c4):
    ke = k_equiv(kap, c4)
    return (ke[0] * R - J01) / J01, (0.5 * (ke[1] + ke[2]) * R - J11) / J11, J01 / ke[0] - R


def errs_neu(kap, R, c4):
    ke = k_equiv(kap, c4)
    return ((0.5 * (ke[1] + ke[2]) * R - JP11) / JP11, (0.5 * (ke[3] + ke[4]) * R - JP21) / JP21,
            (ke[5] * R - JP02) / JP02, JP11 / (0.5 * (ke[1] + ke[2])) - R)


# ---------------------------------------------------------------- sq5 reference methods ----
def op_ghost_normal(solid, X, Y, R):
    """sq5: link to a solid cell j reads the bilinear interpolation of u at the mirror image of
    j's centre across the circle (true normal). Not symmetric."""
    n = solid.shape[0]
    idx = np.arange(n * n).reshape(n, n)
    x0, y0 = X[0, 0], Y[0, 0]
    rows, cols, vals = [], [], []
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if solid[i, j]:
                continue
            me = idx[i, j]
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                rows.append(me); cols.append(me); vals.append(-1.0)
                if not solid[a, b]:
                    rows.append(me); cols.append(idx[a, b]); vals.append(1.0)
                    continue
                xj, yj = X[a, b], Y[a, b]
                r = math.hypot(xj, yj)
                mx, my = xj * (2 * R / r - 1), yj * (2 * R / r - 1)
                fi, fj = mx - x0, my - y0
                i0, j0 = int(math.floor(fi)), int(math.floor(fj))
                tx, ty = fi - i0, fj - j0
                ws = [((i0, j0), (1 - tx) * (1 - ty)), ((i0 + 1, j0), tx * (1 - ty)),
                      ((i0, j0 + 1), (1 - tx) * ty), ((i0 + 1, j0 + 1), tx * ty)]
                ws = [(c, w) for c, w in ws if not solid[c]]
                sw = sum(w for c, w in ws)
                if sw < 1e-12:               # no free cell around the mirror point: fall back to u_i
                    ws, sw = [((i, j), 1.0)], 1.0
                for c, w in ws:
                    rows.append(me); cols.append(idx[c]); vals.append(w / sw)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n * n, n * n))


def cutcell(R, off=(0.0, 0.0), sub=24):
    """sq5 finite-volume cut cells in a disk: returns (K, V, cellmask) with lap = V^-1 K."""
    n = int(2 * R + 8) | 1
    c = (n - 1) // 2
    X, Y = W.grid_xy(W.SQ5, n, n, c, c)
    X, Y = X - off[0], Y - off[1]
    s = (np.arange(sub) + 0.5) / sub - 0.5
    V = np.zeros((n, n))
    for a in s:
        for b in s:
            V += ((X + a) ** 2 + (Y + b) ** 2 < R * R)
    V /= sub * sub

    def aperture(xf, y0, y1):
        """fraction of the segment x = xf, y in [y0, y1] inside the disk"""
        h2 = R * R - xf * xf
        h = np.sqrt(np.maximum(h2, 0))
        lo, hi = np.maximum(y0, -h), np.minimum(y1, h)
        return np.where(h2 > 0, np.clip(hi - lo, 0, None) / (y1 - y0), 0.0)
    Ax = aperture(X + 0.5, Y - 0.5, Y + 0.5)          # face between (i,j) and (i+1,j)
    Ay = aperture(Y + 0.5, X - 0.5, X + 0.5)          # face between (i,j) and (i,j+1) (symmetric)
    cell = V > 1e-9
    idx = np.arange(n * n).reshape(n, n)
    rows, cols, vals = [], [], []
    for A, d in ((Ax, (1, 0)), (Ay, (0, 1))):
        a = A[:-1, :-1] if d == (1, 0) else A[:-1, :-1]
        I, J = np.nonzero((A > 0) & cell & np.roll(cell, -1, axis=0 if d == (1, 0) else 1))
        I2, J2 = I + d[0], J + d[1]
        ok = (I2 < n) & (J2 < n)
        I, J, I2, J2 = I[ok], J[ok], I2[ok], J2[ok]
        wv = A[I, J]
        p, q = idx[I, J], idx[I2, J2]
        rows += [p, q, p, q]; cols += [q, p, p, q]; vals += [wv, wv, -wv, -wv]
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n))
    return K, V, cell


def gibou(solid, X, Y, R, th_min):
    """sq5 symmetric Dirichlet: link to the outside across the circle at fraction theta
    becomes -u_i / theta (theta floored at th_min)."""
    n = solid.shape[0]
    L = W.op_for(solid, W.SQ5, 'R0nb').tolil()        # links to free cells only
    diag = np.zeros((n, n))
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        Sn = W.shift(solid, (di, dj), True)
        Xn, Yn = W.shift(X, (di, dj), 0.0), W.shift(Y, (di, dj), 0.0)
        m = ~solid & Sn
        # |x_i + t (x_j - x_i)| = R, t in (0, 1]
        dx, dy = Xn - X, Yn - Y
        a = dx * dx + dy * dy
        b = 2 * (X * dx + Y * dy)
        cc = X * X + Y * Y - R * R
        t = (-b + np.sqrt(np.maximum(b * b - 4 * a * cc, 0))) / (2 * a)
        t = np.clip(t, th_min, 1.0)
        diag[m] -= 1.0 / t[m]
    return (L.tocsr() + sp.diags(diag.ravel())).tocsr()


def main():
    t0 = time.perf_counter()
    c4 = {k: c4_of(v) for k, v in W.LATS.items()}
    print('bulk 4th-order coefficients c4:', {k: round(v, 5) for k, v in c4.items()})
    radii = (8, 11, 16, 23, 32, 45, 64, 90, 128)
    offs = {'cell-centred': (0.0, 0.0), 'offset (0.31, 0.17)': (0.31, 0.17)}
    results = {}
    for oname, off in offs.items():
        print(f'\n=== centre {oname} ===')
        for latn in ('sq5', 'sq9', 'tri'):
            lat = W.LATS[latn]
            for rule in ('P', 'R0', 'R1'):
                if rule == 'R1' and latn == 'sq5':
                    continue
                print(f'  {latn} {rule}:')
                for R in radii:
                    solid, X, Y = disk(lat, R, off)
                    L = W.op_for(solid, lat, rule)
                    kap = low_eigs(L, ~solid, k=8, sigma=-1e-3)
                    if rule == 'P':
                        e1, e2, dR = errs_dir(kap, R, c4[latn])
                        print(f'    R={R:4}: j01 err {e1:+.2e}  j11 err {e2:+.2e}  R_eff-R = {dR:+.3f}')
                        results[(oname, latn, rule, R)] = (e1, e2, dR)
                    else:
                        e1, e2, e3, dR = errs_neu(kap, R, c4[latn])
                        print(f"    R={R:4}: j'11 err {e1:+.2e}  j'21 err {e2:+.2e}  j'02 err {e3:+.2e}  R_eff-R = {dR:+.3f}")
                        results[(oname, latn, rule, R)] = (e1, e2, dR)
                print(f'      [{time.perf_counter() - t0:.0f} s]')

    # 30/09: one pooled log-log fit misled (sq9 R0 mode 1 gave p = 0.15) because the error changes
    # sign between radii. Reported instead: the fit per centre, whether the error keeps one sign,
    # and R*|err| (constant for a first-order method, falling for a second-order one).
    print('\n=== convergence: fitted p (|err| ~ R^-p over R >= 16, per centre) and R*|err| at R = 16, 45, 128 ===')
    for latn in ('sq5', 'sq9', 'tri'):
        for rule in ('P', 'R0', 'R1'):
            if rule == 'R1' and latn == 'sq5':
                continue
            for mi, mname in ((0, 'mode 1'), (1, 'mode 2')):
                parts = []
                for oname in offs:
                    rs = [R for R in radii if R >= 16]
                    es = [results[(oname, latn, rule, R)][mi] for R in rs]
                    p = -np.polyfit(np.log(rs), np.log(np.abs(es) + 1e-300), 1)[0]
                    sgn = 'one sign' if (all(e > 0 for e in es) or all(e < 0 for e in es)) else 'sign changes'
                    re = ' '.join(f'{R * abs(results[(oname, latn, rule, R)][mi]):.3f}' for R in (16, 45, 128))
                    parts.append(f'{oname[:6]}: p {p:.2f} ({sgn}), R|err| {re}')
                print(f'  {latn} {rule:3} {mname}: ' + '   '.join(parts))
    return results


def sq5_references():
    t0 = time.perf_counter()
    c4 = c4_of(W.SQ5)
    print('\n=== sq5 reference methods (cell-centred and offset centres) ===')
    for off in ((0.0, 0.0), (0.31, 0.17)):
        print(f'  centre offset {off}')
        for R in (8, 16, 32, 64):
            solid, X, Y = disk(W.SQ5, R, off)
            # RG: ghost along the true normal, non-symmetric
            L = op_ghost_normal(solid, X, Y, R)
            idx = np.flatnonzero(~solid.ravel())
            Lf = (-L)[idx][:, idx].tocsc()
            ev = sla.eigs(Lf, k=8, sigma=-1e-3, which='LM', return_eigenvectors=False)
            ev = ev[np.argsort(ev.real)]
            kap = np.sqrt(np.maximum(ev.real, 0))
            e1, e2, e3, dR = errs_neu(kap, R, c4)
            lmx = sla.eigs(Lf, k=4, which='LM', return_eigenvectors=False)
            asym = abs(Lf - Lf.T).max()
            print(f"    R={R:3} RG: j'11 err {e1:+.2e}  j'21 err {e2:+.2e}  max|Im| low {np.abs(ev.imag).max():.1e}  "
                  f"|lam|max/8 = {np.abs(lmx).max() / 8:.3f}  max|L-L^T| = {asym:.2f}")
            # CF: cut cells with V floors
            K, V, cell = cutcell(R, off)
            idc = np.flatnonzero(cell.ravel())
            Kf = (-K)[idc][:, idc].tocsc()
            for vmin in (0.0, 0.25, 0.5):
                Vf = np.maximum(V.ravel()[idc], vmin) if vmin > 0 else V.ravel()[idc]
                M = sp.diags(Vf).tocsc()
                ev = np.sort(sla.eigsh(Kf, k=8, M=M, sigma=-1e-3, which='LM', return_eigenvectors=False))
                kap = np.sqrt(np.maximum(ev, 0))
                e1, e2, e3, dR = errs_neu(kap, R, c4)
                lmax = sla.eigsh(Kf, k=1, M=M, which='LM', return_eigenvectors=False)[0]
                print(f"    R={R:3} CF Vmin={vmin:4}: j'11 err {e1:+.2e}  j'21 err {e2:+.2e}  j'02 err {e3:+.2e}  "
                      f"lam_max/8 = {lmax / 8:.2f} (min V {V.ravel()[idc].min():.1e})")
            for thm in (1e-3, 0.25, 0.5):
                L = gibou(solid, X, Y, R, thm)
                kap = low_eigs(L, ~solid, k=4, sigma=0.0)
                e1, e2, dR = errs_dir(kap, R, c4)
                lmax = W.lam_max(L, ~solid)
                print(f'    R={R:3} GB th_min={thm:5}: j01 err {e1:+.2e}  j11 err {e2:+.2e}  lam_max/8 = {lmax / 8:.2f}')
        print(f'      [{time.perf_counter() - t0:.0f} s]')

    # RG in the time domain: is it stable at Co = 0.5?
    R = 24
    solid, X, Y = disk(W.SQ5, R, (0.31, 0.17))
    L = op_ghost_normal(solid, X, Y, R)
    n = solid.shape[0]
    c = (n - 1) // 2
    Xg, Yg = W.grid_xy(W.SQ5, n, n, c + 6, c - 4)
    u0 = (np.exp(-(Xg ** 2 + Yg ** 2) / 8) * ~solid).ravel()
    co = 0.5
    u1 = u0 + 0.5 * co * co * (L @ u0)
    peaks = []
    W.run(L, u0, u1, co, 20000, every=2000, cb=lambda k, a, b: peaks.append(np.abs(b).max()))
    print('  RG time domain (R = 24, Co = 0.5): max|u| every 2000 steps:', ' '.join(f'{p:.2e}' for p in peaks))
    L0 = W.op_for(solid, W.SQ5, 'R0')
    peaks = []
    W.run(L0, u0, u0 + 0.5 * co * co * (L0 @ u0), co, 20000, every=2000, cb=lambda k, a, b: peaks.append(np.abs(b).max()))
    print('  R0 same run                          :', ' '.join(f'{p:.2e}' for p in peaks))


if __name__ == '__main__':
    main()
    sq5_references()
