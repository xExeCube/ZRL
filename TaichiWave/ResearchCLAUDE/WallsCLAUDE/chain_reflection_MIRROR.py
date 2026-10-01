"""Walls research (29/09/2026): EXACT plane-wave reflection off an infinite straight wall on
the lattice, by Bloch reduction to a 1D chain.

A half-space wall along the lattice vector T = (q, p) is invariant under translation by T, and
so is a plane wave with a fixed tangential Bloch phase beta per T. Every cell (i, j) then
carries w(n) e^{i beta t}: n = p*i - q*j indexes the lattice lines parallel to the wall (they
are d_n = cell_area / |A T| apart along the normal), t is the position along T. The 2D problem
becomes a 1D chain in n with long-range complex hopping -- exact, including the staircase's
fine structure (grating orders, if any, live in the same chain).

A narrow-band pulse (8% bandwidth in the normal wavenumber) is launched toward the wall with
the lattice + leapfrog dispersion (so it is purely incident). The reflected pulse is
Fourier-analysed at the mirrored wavenumber. Reported:
  |R|  and  delta: R = s * exp(2 i k_n (D_line + delta)), s = -1 (pinned) or +1 (rigid);
  D_line = the RASTERISATION line, half-way between the last free and the first solid lattice
  line. delta > 0 means the wall acts deeper inside the solid (in physical cells).
For impedance walls (rule 'Z', zeta = Z/(rho c)) the complex R is referenced to D_line.

    python chain_reflection_MIRROR.py
"""
import sys, os, math, time
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walls_lib_MIRROR as W

RULE_DEF = {'P': (W.PIN, 'none', False), 'Pcb': (W.PIN, 'both', False), 'R0': (W.RIG, 'both', False),
            'R0nb': (W.RIG, 'none', False), 'R1': (W.RIG, 'both', True), 'Z': (W.RIG, 'both', False),
            'ZM': (W.PIN, 'none', False)}
# 'Z'  : impedance as a local damping term on the boundary cell (links to the wall dropped,
#        g = Co * sum(w * link normal length) / (2 zeta), u+ = (... + g u-)/(1 + g)).
# 'ZM' : impedance as a GHOST level with its own update, the generalised Mur formula
#        g^{n+1} = i^n + k (i^{n+1} - g^n),  k = (zeta Co - 1)/(zeta Co + 1)
#        (zeta = 1 is exactly Mur's first-order ABC). Axis-aligned walls only here.


class Chain:
    def __init__(self, lat, T, nw, rule, beta=0.0, co=0.5, zeta=None):
        self.lat, self.T, self.nw, self.beta, self.co, self.rule = lat, np.array(T), nw, beta, co, rule
        q, p = T
        self.p, self.q = p, q
        r1 = min(((a, b) for a in range(-12, 13) for b in range(-12, 13) if p * a - q * b == 1),
                 key=lambda ab: (abs(ab[0]) + abs(ab[1]), ab))
        self.r1 = np.array(r1)
        self.AT = lat.A @ self.T
        self.Ar1 = lat.A @ self.r1
        self.dn = lat.area / np.linalg.norm(self.AT)
        gn = np.linalg.solve(lat.A.T, np.array([p, -q], float))       # grad n, physical
        self.nin = gn / np.linalg.norm(gn)                            # normal INTO the wall
        self.tan = self.AT / np.linalg.norm(self.AT)
        typ, block, transfer = RULE_DEF[rule]

        def lev(d):
            d = np.array(d)
            dn = p * d[0] - q * d[1]
            rem = d - dn * self.r1
            tt = rem @ self.T / (self.T @ self.T)
            assert np.allclose(rem, round(tt) * self.T)
            return int(dn), int(round(tt))

        def typ_at(m):
            return W.FREE if 0 <= m < nw else (typ if m >= nw else W.PIN)

        levs = {tuple(d): (lev(d), [lev(c) for c in com]) for d, w, com in lat.links}
        by_lev = {v[0]: v[1] for v in levs.values()}

        def link_active(n, m, dl):
            # the transfer target link must be free-free and unblocked (as in walls_lib)
            if typ_at(m) != W.FREE:
                return False
            cl = by_lev[dl]
            if cl and block != 'none':
                return not (typ_at(n + cl[0][0]) != W.FREE and typ_at(n + cl[1][0]) != W.FREE)
            return True

        rows, cols, vals = [], [], []
        diag = np.zeros(nw, complex)
        g = np.zeros(nw)
        tr = {}
        for d, w, com in lat.links:
            dn, dt = lev(d)
            ph = np.exp(1j * beta * dt)
            cl = [lev(c) for c in com]
            for n in range(nw):
                m = n + dn
                tn = typ_at(m)
                blk = blk_pin = False
                if com and block != 'none':
                    t1, t2 = typ_at(n + cl[0][0]), typ_at(n + cl[1][0])
                    blk = t1 != W.FREE and t2 != W.FREE
                    blk_pin = t1 == W.PIN or t2 == W.PIN
                if tn == W.FREE and not blk:
                    rows.append(n); cols.append(m); vals.append(w * ph); diag[n] -= w
                elif tn == W.PIN or (tn == W.FREE and blk and blk_pin):
                    diag[n] -= w
                else:                                         # dropped
                    if rule == 'Z' and tn == W.RIG:
                        g[n] += co * w * dn * self.dn / (2 * zeta)   # w * (normal length of link)
                    if transfer and com and tn == W.RIG:
                        t1, t2 = typ_at(n + cl[0][0]), typ_at(n + cl[1][0])
                        for tf, ts, (cdn, cdt) in ((t1, t2, cl[0]), (t2, t1, cl[1])):
                            if tf == W.FREE and ts == W.RIG and link_active(n, n + cdn, (cdn, cdt)):
                                key = (n, n + cdn)
                                ph0, ab0 = tr.get(key, (0, 0))
                                tr[key] = (ph0 + lat.tau * w * np.exp(1j * beta * cdt), ab0 + lat.tau * w)
        L = sp.csr_matrix((np.array(vals, complex), (rows, cols)), shape=(nw, nw)) + sp.diags(diag)
        if tr:
            # the diagonal term is the sum of the link WEIGHTS, not the row sum of the phased
            # entries (found 29/09: using the phased row sum cancelled every same-level
            # transfer, so R1 == R0 at oblique incidence)
            ks = list(tr)
            ij = ([x[0] for x in ks], [x[1] for x in ks])
            Tm = sp.csr_matrix((np.array([tr[x][0] for x in ks]), ij), shape=(nw, nw))
            Ta = sp.csr_matrix((np.array([tr[x][1] for x in ks], float), ij), shape=(nw, nw))
            Wt = 0.5 * (Tm + Tm.conj().T)
            Wa = 0.5 * (Ta + Ta.T)
            L = L + Wt - sp.diags(np.asarray(Wa.sum(axis=1)).ravel())
        self.L = L.tocsr()
        self.g = g

    def Lam(self, k):
        return sum(w * (1 - math.cos(k @ (self.lat.A @ np.array(d)))) for d, w, c in self.lat.links)

    def omega(self, k):
        return 2 * math.asin(min(1.0, self.co * math.sqrt(max(self.Lam(k), 0.0)) / 2))

    def k_of(self, kappa):
        return np.linalg.solve(np.array([self.Ar1, self.AT]), np.array([kappa, self.beta]))


def measure(lat, T, rule, lam, theta_deg=0.0, co=0.5, zeta=None, rel_bw=0.08):
    """Complex R (referenced to the rasterisation line) at physical wavelength lam (cells)
    and incidence angle theta_deg from the wall normal."""
    probe = Chain(lat, T, 4, rule, co=co, zeta=zeta)
    kmag = 2 * math.pi / lam
    th = math.radians(theta_deg)
    k = kmag * (math.cos(th) * probe.nin + math.sin(th) * probe.tan)
    beta = k @ probe.AT
    kn = k @ probe.nin
    kap_span = 2 * kn * probe.dn                       # kappa - kappa_r
    sk = rel_bw * kn * probe.dn
    sig_n = 1 / sk
    nw = int(16 * sig_n + 120)
    ghost = rule == 'ZM'
    ch = Chain(lat, T, nw + (1 if ghost else 0), rule, beta=beta, co=co, zeta=zeta)
    if ghost:
        kz = (zeta * co - 1) / (zeta * co + 1)
    kappa0 = k @ ch.Ar1
    kr = k - 2 * kn * ch.nin
    kappa_r = kr @ ch.Ar1
    if abs(beta) > 1e-12:
        assert abs(ch.Lam(kr) - ch.Lam(k)) < 1e-9, 'oblique incidence needs a wall that is a lattice mirror'
    e = 1e-6
    vg = (ch.omega(ch.k_of(kappa0 + e)) - ch.omega(ch.k_of(kappa0 - e))) / (2 * e)
    assert vg > 0
    n = np.arange(nw + (1 if ghost else 0))
    nc = nw - 8 * sig_n - 30
    m = np.arange(-nw // 2, nw // 2)
    kap = kappa0 + 2 * math.pi * m / nw
    sel = np.abs(kap - kappa0) < 6 * sk
    a = np.exp(-((kap[sel] - kappa0) / sk) ** 2 / 2)
    om = np.array([ch.omega(ch.k_of(x)) for x in kap[sel]])
    E = np.exp(1j * np.outer(n - nc, kap[sel]))
    w0 = E @ a
    wm1 = E @ (a * np.exp(1j * om))
    if ghost:
        w0[nw] = 0
        wm1[nw] = 0
    steps = int(round(2 * (nw - nc) / vg))
    c2 = co * co
    A_, B_ = wm1, w0
    g = ch.g
    for s in range(steps):
        C_ = 2 * B_ - A_ + c2 * (ch.L @ B_)
        if g.any():
            C_ = (C_ + g * A_) / (1 + g)
        if ghost:
            C_[nw] = B_[nw - 1] + kz * (C_[nw - 1] - B_[nw])
        A_, B_ = B_, C_
    Fi = np.sum(w0[:nw] * np.exp(-1j * kappa0 * (n[:nw] - nc)))
    Ff = np.sum(B_[:nw] * np.exp(-1j * kappa_r * (n[:nw] - nc)))
    R = Ff / (Fi * np.exp(-1j * ch.omega(k) * steps))
    # re-reference from n = nc to the rasterisation line n = nw - 1/2
    R_line = R * np.exp(1j * (kappa0 - kappa_r) * (nc - (nw - 0.5)))
    return R_line, kap_span, ch


def delta_of(R_line, kap_span, dn, s):
    """Effective wall offset (physical cells) from R = s exp(i kap_span (n_line + delta_n))."""
    dphi = np.angle(R_line / s)
    return dphi / kap_span * dn, math.degrees(dphi)


def main():
    t0 = time.perf_counter()
    co = 0.5
    print('=== 1. Normal incidence on straight walls and staircases (Co = 0.5) ===')
    print('delta = effective wall offset beyond the rasterisation line, physical cells;')
    print('dphi = reflection phase error vs a wall ON the rasterisation line, degrees.\n')
    walls = [('sq5', (1, 0)), ('sq5', (1, 1)), ('sq5', (2, 1)), ('sq5', (3, 1)), ('sq5', (3, 2)),
             ('sq9', (1, 0)), ('sq9', (1, 1)), ('sq9', (2, 1)), ('sq9', (3, 2)),
             ('tri', (1, 0)), ('tri', (1, 1)), ('tri', (2, 1)), ('tri', (3, 1))]
    lams = (64, 32, 16, 8, 5)
    print(f"{'lattice':6} {'T':8} {'angle':>6} {'rule':5} " + ' '.join(f'{"lam=" + str(l):>17}' for l in lams))
    for latn, T in walls:
        lat = W.LATS[latn]
        AT = lat.A @ np.array(T)
        ang = math.degrees(math.atan2(AT[1], AT[0]))
        for rule, s in (('P', -1), ('R0', 1), ('R1', 1)):
            if rule == 'R1' and latn == 'sq5':
                continue
            cells = []
            for lam in lams:
                R, span, ch = measure(lat, T, rule, lam, 0.0, co, rel_bw=0.02 if lam <= 6 else 0.08)
                dl, dph = delta_of(R, span, ch.dn, s)
                cells.append(f'|R|{abs(R):.4f} d{dl:+.3f}')
            print(f'{latn:6} {str(T):8} {ang:6.1f} {rule:5} ' + ' '.join(f'{c:>17}' for c in cells))
    print(f'  [{time.perf_counter() - t0:.0f} s]')

    print('\n=== 2. Oblique incidence on lattice-mirror walls (sq axis, sq 45 deg, tri row, tri 30 deg) ===')
    print('phase error dphi (deg) vs the ideal wall ON the rasterisation line; |R| = 1 unless shown.\n')
    angs = (0, 30, 60, 80)
    for latn, T in (('sq5', (1, 0)), ('sq5', (1, 1)), ('sq9', (1, 0)), ('sq9', (1, 1)), ('tri', (1, 0)), ('tri', (1, 1))):
        lat = W.LATS[latn]
        for rule, s in (('P', -1), ('R0', 1), ('R1', 1)):
            if rule == 'R1' and latn == 'sq5':
                continue
            for lam in (32, 8):
                cells = []
                for th in angs:
                    R, span, ch = measure(lat, T, rule, lam, th, co)
                    dl, dph = delta_of(R, span, ch.dn, s)
                    cells.append(f'{dph:+7.2f}' + ('' if abs(abs(R) - 1) < 1e-3 else f'(|R|{abs(R):.3f})'))
                print(f'{latn} T={str(T):7} {rule:3} lam={lam:3}: ' + '  '.join(f'th{a}:{c}' for a, c in zip(angs, cells)))
    print(f'  [{time.perf_counter() - t0:.0f} s]')

    print('\n=== 3. Impedance (Robin) walls on an axis wall: R vs zeta and angle; Z = damping form, ZM = generalised-Mur ghost ===')
    print('expected (continuum) R = (zeta cos th - 1)/(zeta cos th + 1), real\n')
    for latn in ('sq5', 'sq9', 'tri'):
        lat = W.LATS[latn]
        for zeta in (0.25, 1.0, 2.0, 5.0):
            for lam in (32, 8):
                cells = []
                for th in (0, 30, 60):
                    ex = (zeta * math.cos(math.radians(th)) - 1) / (zeta * math.cos(math.radians(th)) + 1)
                    Rz, span, ch = measure(lat, (1, 0), 'Z', lam, th, co, zeta=zeta)
                    Rm, span, ch = measure(lat, (1, 0), 'ZM', lam, th, co, zeta=zeta)
                    cells.append(f'th{th}: exp {ex:+.3f} | Z {Rz.real:+.3f}{Rz.imag:+.3f}i | ZM {Rm.real:+.3f}{Rm.imag:+.3f}i')
                print(f'{latn} zeta={zeta:4} lam={lam:3}: ' + '   '.join(cells))
    print(f'  [{time.perf_counter() - t0:.0f} s]')


if __name__ == '__main__':
    main()
