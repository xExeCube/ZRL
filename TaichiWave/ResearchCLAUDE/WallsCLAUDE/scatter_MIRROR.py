"""Walls research (30/09/2026): scattering by small obstacles, pinned vs rigid, on the lattice.

Method (time domain, exact difference field):
  * a narrow-band plane pulse (10% bandwidth) built from EXACT lattice + leapfrog plane waves
    (so it is purely forward-going on this lattice) is run twice: through the empty lattice
    (incident run) and with the obstacle (total run). The difference d = total - incident is the
    scattered field exactly -- lattice dispersion and the edge disturbances of the finite box are
    identical in both runs and cancel.
  * the pulse starts 8 lambda (5 envelope sigmas) before the obstacle and runs until it is
    8 lambda past it; the box is big enough that nothing scattered reaches the box edge.
  * scattered energy E_s = sum over cells farther than r_ex from the obstacle of the leapfrog
    energy density of d, (d1 - d0)^2 / Co^2 - d1 (L0 d0) (L0 = empty lattice). The incident
    energy per unit wavefront width is the same density summed over a central strip at t = 0,
    divided by the strip width. Their ratio is the SCATTERING WIDTH sigma (physical cells):
    the width of incident wavefront whose energy the obstacle scatters.
  * continuum references: a disc of the same area, Dirichlet and Neumann, exact series
    sigma = (4/k) sum_n |J_n(ka)/H_n(ka)|^2   and   (4/k) sum_n |J_n'(ka)/H_n'(ka)|^2.

Parts:
  1. single cell and 3x3 block, lambda = 4 .. 32, rules P, R0, R1 (R1 == R0 for a single cell:
     no dropped link has a free and a rigid common neighbour);
  2. an 11 x 11 square rotated 30 and 45 deg vs the axis-aligned square hit at -30 and -45 deg (the same physical
     scene rotated): total sigma and the angular pattern, pinned vs rigid.

    python scatter_MIRROR.py sq5|sq9|tri [part]
"""
import sys, os, math, time
import numpy as np
import scipy.special as ss
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walls_lib_MIRROR as W

CO = 0.5
BW = 0.10


def lam_sym(lat, k):
    return sum(w * (1 - math.cos(k @ (lat.A @ np.array(d)))) for d, w, c in lat.links)


def omega(lat, k):
    return 2 * math.asin(min(1.0, CO * math.sqrt(max(lam_sym(lat, k), 0.0)) / 2))


def box(lat, H):
    """Index box whose physical inscribed disc (about the box centre) has radius >= H."""
    if lat.name == 'tri':
        n = int(math.ceil(2 * H / W.S3H)) + 8
    else:
        n = int(math.ceil(2 * H)) + 8
    n |= 1
    c = (n - 1) // 2
    X, Y = W.grid_xy(lat, n, n, c, c)
    return n, X, Y


def incident(lat, X, Y, lam, theta, s0):
    """u^0, u^{-1} of a forward plane pulse along direction theta, envelope centred at s = s0
    (s = d . x), carrier wavelength lam. Exact superposition of lattice plane waves."""
    d = np.array([math.cos(theta), math.sin(theta)])
    s = d[0] * X + d[1] * Y
    k0 = 2 * math.pi / lam
    sk = BW * k0
    span = (s.max() - s.min()) * 2 + 1
    dk = 2 * math.pi / span
    ks = k0 + dk * np.arange(-int(5 * sk / dk), int(5 * sk / dk) + 1)
    u0 = np.zeros_like(X)
    um = np.zeros_like(X)
    for kk in ks:
        a = math.exp(-((kk - k0) / sk) ** 2 / 2)
        om = omega(lat, kk * d)
        ph = kk * (s - s0)
        u0 += a * np.cos(ph)
        um += a * np.cos(ph + om)          # one step earlier: phase + omega
    sc = 1.0 / np.abs(u0).max()
    return u0 * sc, um * sc


def local_energy(L0, a, b):
    """leapfrog energy density between levels a (n) and b (n+1), per cell."""
    return (b - a) ** 2 / (CO * CO) - b * (L0 @ a)


def run_pair(L, a, b, steps):
    c2 = CO * CO
    for _ in range(steps):
        a, b = b, 2 * b - a + c2 * (L @ b)
    return a, b


def sigma_cont(ka, kind, k):
    nmax = int(ka + 12)
    tot = 0.0
    for nn in range(-nmax, nmax + 1):
        if kind == 'D':
            r = ss.jv(nn, ka) / ss.hankel1(nn, ka)
        else:
            r = ss.jvp(nn, ka) / ss.h1vp(nn, ka)
        tot += abs(r) ** 2
    return 4 / k * tot


class Setup:
    """One (lattice, lambda, incident angle) setup: box, incident run, energy normalisation."""

    def __init__(self, lat, lam, theta=0.0, size=3.0):
        self.lat, self.lam, self.theta = lat, lam, theta
        # envelope sigma = 1/(BW k) = 1.59 lam, 5 sigma = 8 lam: start 8 lam + 10 before the
        # obstacle, stop 8 lam + 10 past it. The scattered front is then at ~D, so a box of
        # inscribed radius D + 30 keeps it clear of the edge (checked: 'edge' column).
        self.D = 16 * lam + 20 + size             # travel of the envelope centre
        H = self.D + 30
        self.H = H
        self.n, self.X, self.Y = box(lat, H)
        self.R = np.hypot(self.X, self.Y)
        self.PHI = np.arctan2(self.Y, self.X)
        u0, um = incident(lat, self.X, self.Y, lam, theta, -(8 * lam + 10 + size / 2))
        empty = np.zeros((self.n, self.n), bool)
        empty[0, :] = empty[-1, :] = empty[:, 0] = empty[:, -1] = True
        self.empty = empty
        self.L0 = W.op_for(empty, lat, 'P')
        self.u0 = (u0 * ~empty).ravel()
        self.um = (um * ~empty).ravel()
        # group velocity along d, in physical cells per step
        d = np.array([math.cos(theta), math.sin(theta)])
        k0 = 2 * math.pi / lam
        e = 1e-5
        vg = (omega(lat, (k0 + e) * d) - omega(lat, (k0 - e) * d)) / (2 * e)
        self.steps = int(round(self.D / vg))
        # incident energy per unit wavefront width: central strip |t| < Wc
        t = -math.sin(theta) * self.X + math.cos(theta) * self.Y
        Wc = 0.25 * H
        e0 = local_energy(self.L0, self.um, self.u0).reshape(self.n, self.n)
        self.e_per_width = float(e0[np.abs(t) < Wc].sum()) / (2 * Wc)
        t0 = time.perf_counter()
        self.inc = run_pair(self.L0, self.um, self.u0, self.steps)
        self.t_run = time.perf_counter() - t0

    def scatter(self, solid, rule, r_ex):
        # the box ring stays PINNED in every run (as in the incident run); only the obstacle
        # takes the rule's wall type -- found 30/09: op_for(solid | ring) made the ring rigid
        # too under R0, and the ring's own pinned-vs-rigid difference swamped the scattering.
        typ, block, tr = W.RULES[rule]
        T = np.where(solid, typ, W.FREE).astype(np.int8)
        T[self.empty] = W.PIN
        L = W.build_L(T, self.lat, block=block, transfer=tr)
        keep = (~solid).ravel()
        a, b = run_pair(L, self.um * keep, self.u0 * keep, self.steps)
        da, db = a - self.inc[0], b - self.inc[1]
        e = local_energy(self.L0, da, db).reshape(self.n, self.n)
        far = self.R > r_ex
        # the scattered ring must be well inside the box: check the energy near the box edge
        edge = self.R > 0.9 * self.H
        sig = float(e[far].sum()) / self.e_per_width
        return sig, e, float(np.abs(e[edge]).sum()) / max(float(e[far].sum()), 1e-300)


def obstacle(lat, X, Y, kind, ang=0.0, side=12.0):
    if kind == 'cell':
        return (np.abs(X) < 1e-9) & (np.abs(Y) < 1e-9)
    if kind == 'block3':
        # the 3x3 index block around the centre (on tri: a 3x3 rhombus of cells)
        n = X.shape[0]
        c = (n - 1) // 2
        s = np.zeros_like(X, bool)
        s[c - 1:c + 2, c - 1:c + 2] = True
        return s
    if kind == 'square':
        a = math.radians(ang)
        xr = math.cos(a) * X + math.sin(a) * Y
        yr = -math.sin(a) * X + math.cos(a) * Y
        return (np.abs(xr) < side / 2) & (np.abs(yr) < side / 2)
    raise ValueError(kind)


def part1(latn):
    lat = W.LATS[latn]
    rules = ('P', 'R0') if latn == 'sq5' else ('P', 'R0', 'R1')
    print(f'=== 1. {latn}: scattering width sigma (physical cells) vs wavelength, Co = {CO} ===')
    print('   disc refs: same-area disc, continuum Dirichlet (D) and Neumann (N); edge = energy fraction near the box edge (must be ~0)')
    res = {}
    for lam in (4, 6, 8, 12, 16, 24, 32):
        t0 = time.perf_counter()
        st = Setup(lat, lam, 0.0, size=3.0)
        k = 2 * math.pi / lam
        for kind in ('cell', 'block3'):
            solid = obstacle(lat, st.X, st.Y, kind)
            for rule in rules:
                res[(kind, lam, rule)] = st.scatter(solid, rule, r_ex=8.0)[::2]
        print(f'    [lam={lam}: n={st.n}, {st.steps} steps, {time.perf_counter() - t0:.0f} s]', flush=True)
    for kind in ('cell', 'block3'):
        area = lat.area * (1 if kind == 'cell' else 9)
        a_eq = math.sqrt(area / math.pi)
        print(f'  obstacle {kind} (area {area:.3f}, equal-area disc radius {a_eq:.3f}):')
        for lam in (4, 6, 8, 12, 16, 24, 32):
            k = 2 * math.pi / lam
            cells = []
            for rule in rules:
                sig, edge = res[(kind, lam, rule)]
                cells.append(f'{rule} {sig:9.5f}' + (f' (edge {edge:.0e})' if edge > 1e-6 else ''))
            sd, sn = sigma_cont(k * a_eq, 'D', k), sigma_cont(k * a_eq, 'N', k)
            print(f'    lam={lam:3} ka={k * a_eq:5.3f}: ' + '  '.join(cells) +
                  f'   | disc D {sd:8.4f}  N {sn:9.5f}   P/R0 = {res[(kind, lam, "P")][0] / res[(kind, lam, "R0")][0]:8.1f}')
        # long-wave power law sigma ~ lam^-p over lam = 16..32
        for rule in rules:
            p = -math.log(res[(kind, 32, rule)][0] / res[(kind, 16, rule)][0]) / math.log(2)
            print(f'      {rule}: sigma ~ lam^{-p:+.2f} between lam 16 and 32')


def pattern(e, PHI, far, nb=72):
    idx = ((PHI[far] + math.pi) / (2 * math.pi) * nb).astype(int) % nb
    return np.bincount(idx, weights=e[far], minlength=nb)


def part2(latn, side=11.0):
    """30/09 rerun: the first version used side 12 with a strict |x| < 6 test, so the AXIS
    square rasterised to 11 x 11 = 121 cells while the rotated one had 145 -- a 20% area
    mismatch that alone explained most of the 'rotated rigid square scatters 38-54% more'.
    Side 11 puts no cell centre on the axis square's edge (121 cells); the cell counts of both
    scenes are printed so a mismatch is visible."""
    lat = W.LATS[latn]
    rules = ('P', 'R0') + (() if latn == 'sq5' else ('R1',))
    print(f'\n=== 2. {latn}: {side:.0f}x{side:.0f} square rotated by ang (incident 0) vs axis square (incident -ang) ===')
    print('   the two scenes are the same physical scene rotated; differences = staircase + lattice anisotropy.')
    print('   pattern diff = || pat_B(phi) - pat_A(phi - ang) ||_1 / || pat_A ||_1 over 72 bins of 5 deg')
    for lam in (8, 16):
        stB = Setup(lat, lam, 0.0, size=18.0)
        k = 2 * math.pi / lam
        a_eq = side / math.sqrt(math.pi)
        print(f'  lam={lam}: equal-area disc, continuum: D {sigma_cont(k * a_eq, "D", k):.2f}  N {sigma_cont(k * a_eq, "N", k):.2f}')
        for ang in (30.0, 45.0):
            stA = Setup(lat, lam, -math.radians(ang), size=18.0)
            solB = obstacle(lat, stB.X, stB.Y, 'square', ang, side)
            solA = obstacle(lat, stA.X, stA.Y, 'square', 0.0, side)
            sh = int(round(ang / 5))
            for rule in rules:
                sA, eA, _ = stA.scatter(solA, rule, r_ex=20.0)
                sB, eB, _ = stB.scatter(solB, rule, r_ex=20.0)
                pA = pattern(eA, stA.PHI, stA.R > 20.0)
                pB = pattern(eB, stB.PHI, stB.R > 20.0)
                diff = np.abs(pB - np.roll(pA, sh)).sum() / np.abs(pA).sum()
                print(f'  lam={lam:3} rot {ang:4.0f} {rule:3}: sigma axis {sA:8.3f}  rotated {sB:8.3f}  (ratio {sB / sA:.4f})  '
                      f'pattern diff {diff:.3f}   cells axis {int(solA.sum())} rotated {int(solB.sum())}', flush=True)
            # reference: a DISC in both setups (identical rasterisation; only the incident
            # direction vs the lattice differs)
            rd = side / math.sqrt(math.pi)
            dA, dB = stA.R < rd, stB.R < rd
            for rule in ('P', 'R0'):
                sA, eA, _ = stA.scatter(dA, rule, r_ex=20.0)
                sB, eB, _ = stB.scatter(dB, rule, r_ex=20.0)
                pA = pattern(eA, stA.PHI, stA.R > 20.0)
                pB = pattern(eB, stB.PHI, stB.R > 20.0)
                diff = np.abs(pB - np.roll(pA, sh)).sum() / np.abs(pA).sum()
                print(f'  lam={lam:3} rot {ang:4.0f} {rule:3} DISC reference: sigma inc -{ang:.0f} {sA:8.3f}  inc 0 {sB:8.3f}  '
                      f'(ratio {sB / sA:.4f})  pattern diff {diff:.3f}   cells {int(dA.sum())}', flush=True)


if __name__ == '__main__':
    latn = sys.argv[1]
    part = sys.argv[2] if len(sys.argv) > 2 else 'all'
    t0 = time.perf_counter()
    if part in ('all', '1'):
        part1(latn)
    if part in ('all', '2'):
        part2(latn)
    print(f'  [{time.perf_counter() - t0:.0f} s]')
