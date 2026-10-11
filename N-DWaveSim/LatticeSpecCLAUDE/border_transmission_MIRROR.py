"""Translucent one-node borders and density steps on the apps' grids (spec §3.6, §3.7).

The apps' step is u+ - 2u + u- = CC_i (L u)_i, i.e. M_i (u+ - 2u + u-) = Co^2 (L_K u)_i with node mass M_i = Co^2/CC_i
and link weights K (all 1 in the apps today). Two one-node borders:
  mass border  a line of nodes with mass 1 + s   (a line at speed factor f: s = 1/f^2 - 1; pinned is s -> inf)
  link border  the links crossing the line get weight 1/(1 + s)   (rigid, the apps' link removal, is s -> inf)
Predicted on the 1D grid (derived in spec §3.6): |t|^2 = 1 / (1 + X^2), X = s tan(k/2), k = 2 pi / PPW. Co drops out.
On sq5 with a border along a grid axis and tangential wavenumber ky (same derivation):
  mass  X = s (2 sin^2(kx/2) + 2 sin^2(ky/2)) / sin(kx)      link  X = s tan(kx/2)
which become s k / (2 cos th) and s k cos th / 2 in the continuum.

Parts:
  1. 1D: the closed form against a direct frequency-domain solve and a time-domain packet (Co 0.5 and 0.9).
  2. 2D oblique incidence, time-domain packets on sq5, sq9 and tri (border along a grid row), PPW 8 and 16,
     angles 0, 30, 60 degrees, against the sq5 formula. sq9 and tri have no closed form here (measured only).
  3. 1D density steps: mass-only (r = (f - 1)/(f + 1) predicted) and impedance-matched (mass x 1/f, links x f;
     r = 0 predicted at normal incidence in the continuum).
  4. The exact discrete reflection of the 1D mass step (two unknowns), against part 3's packets.

Run: python border_transmission_MIRROR.py      (numpy only, ~2 min)
"""
import numpy as np

# ---------- 1. 1D ----------
def solve_1d(k, s, kind):
    """Frequency-domain r, t for one border on the infinite 1D grid (two unknowns)."""
    E = np.exp(1j * k); W = 4 * np.sin(k / 2) ** 2           # W = 4 sin^2(w/2) / Co^2 from the background dispersion
    if kind == 'mass':   # u_j = e^{ikj} + r e^{-ikj} (j <= 0), t e^{ikj} (j >= 0); node 0 has mass 1 + s
        A = np.array([[1, -1], [-E, -W * (1 + s) - E + 2]], complex); b = np.array([-1, 1 / E], complex)
    else:                # u_j as above for j <= 0, t e^{ikj} for j >= 1; link 0-1 has weight ka
        ka = 1 / (1 + s)
        A = np.array([[-W + ka - E + 1, -ka * E], [-ka, -W * E - E ** 2 + E + ka * E]], complex)
        b = np.array([W - ka + 1 / E - 1, ka], complex)
    r, t = np.linalg.solve(A, b); return r, t

def run_1d(M, K, k0, Co, x0, sig, T):
    """Leapfrog M (u+ - 2u + u-) = Co^2 (L_K u) on a line; returns the initial and final fields."""
    n = len(M); x = np.arange(n)
    w0 = 2 * np.arcsin(Co * np.sin(k0 / 2)); vg = Co ** 2 * np.sin(k0) / np.sin(w0)
    u = np.exp(-((x - x0) ** 2) / (2 * sig ** 2)) * np.cos(k0 * (x - x0))
    up = np.exp(-((x - x0 + vg) ** 2) / (2 * sig ** 2)) * np.cos(k0 * (x - x0) + w0)
    u0 = u.copy()
    for _ in range(T):
        f = K * (u[1:] - u[:-1]); Lu = np.zeros(n); Lu[:-1] += f; Lu[1:] -= f
        u, up = 2 * u - up + Co ** 2 * Lu / M, u
    return u0, u, vg

def td_1d(kind, s, ppw, Co):
    n = 8000; mid = n // 2; k0 = 2 * np.pi / ppw; sig = 6 * ppw; x0 = mid - 10 * sig
    M = np.ones(n); K = np.ones(n - 1)
    if kind == 'mass': M[mid] = 1 + s
    else: K[mid] = 1 / (1 + s)
    _, _, vg = run_1d(M, K, k0, Co, x0, sig, 0)
    u0, u, _ = run_1d(M, K, k0, Co, x0, sig, int(20 * sig / vg))
    E0 = (u0 ** 2).sum()
    return (u[mid + 1:] ** 2).sum() / E0, (u[:mid] ** 2).sum() / E0

# ---------- 2. 2D oblique ----------
STENCILS = {   # offsets (across, along) in index space, weight; physical position of an index offset
    'sq5': [((1, 0), 1.0), ((-1, 0), 1.0), ((0, 1), 1.0), ((0, -1), 1.0)],
    'sq9': [((1, 0), 2 / 3), ((-1, 0), 2 / 3), ((0, 1), 2 / 3), ((0, -1), 2 / 3),
            ((1, 1), 1 / 6), ((1, -1), 1 / 6), ((-1, 1), 1 / 6), ((-1, -1), 1 / 6)],
    # tri in (b, a) order: axis 0 = row index b (across the border row), axis 1 = a (along it)
    'tri': [((0, 1), 2 / 3), ((0, -1), 2 / 3), ((1, 0), 2 / 3), ((-1, 0), 2 / 3), ((1, -1), 2 / 3), ((-1, 1), 2 / 3)],
}
def phys(kind, d):
    """Physical (along, across) displacement of an index offset d = (across, along)."""
    if kind == 'tri':
        db, da = d; return np.array([da + db / 2, db * np.sqrt(3) / 2])
    return np.array([d[1], d[0]])

def td_2d(kind, mode, s, ppw, theta_deg, Co=0.5):
    """Plane-wave packet at angle theta from the border's normal; transmitted and reflected sum u^2 fractions."""
    st = STENCILS[kind]; k = 2 * np.pi / ppw; th = np.radians(theta_deg)
    pitch = phys(kind, (0, 1))[0]                         # node spacing along the border
    # tangential wavenumber, quantised to the periodic strip
    if theta_deg == 0: Ny, kt = 1, 0.0
    else:
        kt_want = k * np.sin(th); Ny = max(2, int(round(2 * np.pi / (kt_want * pitch))))
        kt = 2 * np.pi / (Ny * pitch)
    kn = np.sqrt(k * k - kt * kt); Kvec = np.array([kt, kn])
    across = phys(kind, (1, 0))                          # physical step of one row
    # discrete dispersion and group velocity across rows (per step, in rows)
    sym = lambda K: sum(w * (np.cos(K @ phys(kind, d)) - 1) for d, w in st)
    w0 = 2 * np.arcsin(Co * np.sqrt(-sym(Kvec)) / 2)
    eps = 1e-6; grad = np.array([(sym(Kvec + eps * e) - sym(Kvec - eps * e)) / (2 * eps) for e in np.eye(2)])
    vg = -Co ** 2 * grad / (2 * np.sin(w0))               # physical group velocity
    vrow = vg[1] / across[1]                              # rows per step
    sig = 4 * ppw / across[1]; nx = int(24 * sig) + 200; mid = nx // 2; x0 = mid - 7 * sig
    b = np.arange(nx)[:, None]; a = np.arange(Ny)[None, :]
    X = a + (b / 2 if kind == 'tri' else 0) * 1.0
    Y = b * across[1] + 0.0 * a
    phase = Kvec[0] * X + Kvec[1] * Y
    env = lambda bb: np.exp(-((bb - x0) ** 2) / (2 * sig ** 2))
    u = env(b) * np.cos(phase - Kvec[1] * x0 * across[1]); up = env(b + vrow) * np.cos(phase - Kvec[1] * x0 * across[1] + w0)
    # the along-row drift of the envelope is absent (the envelope is uniform along the row)
    M = np.ones((nx, Ny)); Kw = {d: np.ones((nx, Ny)) for d, _ in st}
    if mode == 'mass': M[mid, :] = 1 + s
    else:
        for d, _ in st:   # a link crosses the border if it joins row <= mid to row >= mid + 1
            if d[0] == 1: Kw[d][mid, :] = 1 / (1 + s)
            if d[0] == -1: Kw[d][mid + 1, :] = 1 / (1 + s)
    u0 = u.copy(); T = int(14 * sig / vrow)
    for _ in range(T):
        Lu = np.zeros_like(u)
        for d, w in st:
            Lu += w * Kw[d] * (np.roll(np.roll(u, -d[0], 0), -d[1], 1) - u)
        u, up = 2 * u - up + Co ** 2 * Lu / M, u
    E0 = (u0 ** 2).sum()
    return (u[mid + 1:] ** 2).sum() / E0, (u[:mid] ** 2).sum() / E0, kn, kt, np.degrees(np.arctan2(kt, kn))

def pred_sq5(mode, s, kn, kt):
    X = s * (2 * np.sin(kn / 2) ** 2 + 2 * np.sin(kt / 2) ** 2) / np.sin(kn) if mode == 'mass' else s * np.tan(kn / 2)
    return 1 / (1 + X * X)

# ---------- 3. 1D density steps ----------
def step_1d(f, ppw, matched, Co=0.5):
    n = 8000; mid = n // 2; k0 = 2 * np.pi / ppw; sig = 6 * ppw; x0 = mid - 10 * sig
    M = np.ones(n); K = np.ones(n - 1)
    if matched: M[mid:] = 1 / f; K[mid:] = f; K[mid - 1] = (1 + f) / 2
    else: M[mid:] = 1 / f ** 2
    _, _, vg = run_1d(M, K, k0, Co, x0, sig, 0)
    u0, u, _ = run_1d(M, K, k0, Co, x0, sig, int(14 * sig / vg))
    return (u[:mid] ** 2).sum() / (u0 ** 2).sum()

def step_exact(f, ppw):
    """Exact |r|^2 of the 1D mass step: nodes j < 0 mass 1, j >= 0 mass 1/f^2 (as in step_1d)."""
    k1 = 2 * np.pi / ppw; M2 = 1 / f ** 2; W = 4 * np.sin(k1 / 2) ** 2
    k2 = np.arccos(1 - W * M2 / 2)                      # M2 W = 4 sin^2(k2/2) on the slow side
    E1, E2 = np.exp(1j * k1), np.exp(1j * k2)
    # 1 + r = t ; node 0 (mass M2): r E1 + t (E2 - 2 + W M2) = -1/E1
    A = np.array([[1, -1], [E1, E2 - 2 + W * M2]], complex); b = np.array([-1, -1 / E1], complex)
    r, t = np.linalg.solve(A, b); return abs(r) ** 2

if __name__ == '__main__':
    print('1. 1D, s = 1 (mass: f = 0.7071; link: weight 0.5)')
    for ppw in (32, 16, 8, 4):
        k = 2 * np.pi / ppw; pred = 1 / (1 + np.tan(k / 2) ** 2)
        rm, tm = solve_1d(k, 1.0, 'mass'); rl, tl = solve_1d(k, 1.0, 'link')
        line = (f'   PPW {ppw:>2}: predicted |t|^2 {pred:.6f} | solve mass {abs(tm) ** 2:.6f} link {abs(tl) ** 2:.6f} '
                f'| arg r mass {np.degrees(np.angle(rm)):6.1f} link {np.degrees(np.angle(rl)):6.1f} deg')
        if ppw in (16, 8):
            for Co in (0.5, 0.9):
                tt, rr = td_1d('mass', 1.0, ppw, Co); tl2, rl2 = td_1d('link', 1.0, ppw, Co)
                line += f' | packet Co {Co}: mass {tt:.4f} (+r {rr:.4f}) link {tl2:.4f} (+r {rl2:.4f})'
        print(line)
    print('   s for 50% power at PPW 16:', round(1 / np.tan(np.pi / 16), 4), '-> mass f =', round(1 / np.sqrt(1 + 1 / np.tan(np.pi / 16)), 4))
    print('2. 2D oblique, s = 1, Co 0.5: transmitted sum u^2 (reflected) vs the sq5 formula')
    for kind in ('sq5', 'sq9', 'tri'):
        for mode in ('mass', 'link'):
            for ppw in (16, 8):
                cells = []
                for ang in (0, 30, 60):
                    t2, r2, kn, kt, act = td_2d(kind, mode, 1.0, ppw, ang)
                    cells.append(f'{act:4.1f} deg: {t2:.4f} ({r2:.4f}) pred {pred_sq5(mode, 1.0, kn, kt):.4f}')
                print(f'   {kind} {mode:<4} PPW {ppw:>2}: ' + ' | '.join(cells))
    print('3. 1D density steps, Co 0.5: reflected power |r|^2')
    for f in (0.7071, 0.55):
        pred = ((f - 1) / (f + 1)) ** 2
        cells = [f'PPW {ppw}: mass-only {step_1d(f, ppw, False):.5f}, matched {step_1d(f, ppw, True):.2e}' for ppw in (16, 8)]
        print(f'   f = {f}: predicted mass-only {pred:.5f} | ' + ' | '.join(cells))
    print('4. 1D mass step, exact discrete |r|^2, PPW 64 / 32 / 16 / 8')
    for f in (0.7071, 0.55):
        print(f'   f = {f}: ' + ' / '.join(f'{step_exact(f, ppw):.5f}' for ppw in (64, 32, 16, 8)))
