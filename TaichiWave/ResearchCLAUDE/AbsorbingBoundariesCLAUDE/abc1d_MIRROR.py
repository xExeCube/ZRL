"""Absorbing boundaries for the ZRL membrane -- the 1D-REDUCED reflection bench (numpy).

What is measured: the amplitude reflection coefficient |R| of one straight wall of the app's
square 5-point leapfrog lattice, versus incidence angle and wavelength.

Why a 1D reduction is EXACT here: on a grid periodic in y, a field u[i] * cos(ky*j) stays of
that form under the 5-point update, with the y-part of the Laplacian replaced by its symbol
    u[i,j-1] - 2u[i,j] + u[i,j+1]  ->  m * u[i],   m = 2 cos(ky) - 2.
Every boundary rule tested here is linear, shift-invariant along the wall and even in y, so it
reduces the same way (Mur2's tangential second difference becomes m times the value). A single
ky therefore fixes the angle EXACTLY; only kx carries the packet's spectral width. The 2D check
(check2d_MIRROR.py) runs the same cases on a real 2D periodic-y grid and agrees to rounding.

Method: a Gaussian wave packet with carrier (kx, ky) on the lattice's own dispersion relation
    sin^2(w/2) = Co^2 (sin^2(kx/2) + sin^2(ky/2))      (w per step, k per cell)
travels toward the wall at i = 0. The same packet runs in a REFERENCE domain extended to the
left far enough that nothing returns to the physical cells in the time window. When the
reflected packet is back at the launch point (T = 2*i0/vg steps), R = ||u_test - u_ref|| over
the physical cells divided by the incident packet's norm at launch. The packet is
sigma_x = 3*lambda/cos(theta) wide, so the angle spread stays ~5% of cos(theta).

Units: h = 1, c = 1, dt = Co; so a damping rate sigma in 1/time multiplies dt = Co.
"""
import math
import numpy as np

TAU = 2 * math.pi


# ---- the packet --------------------------------------------------------------------------------
def dispersion(Co, kx, ky):
    s = Co * Co * (math.sin(kx / 2) ** 2 + math.sin(ky / 2) ** 2)
    w = 2 * math.asin(math.sqrt(s))
    vg = Co * Co * math.sin(kx) / math.sin(w)          # d w / d kx, cells per step
    return w, vg


def packet(Nx, i0, sig, kx, w, vg, shift=0):
    """Two time levels of a packet moving toward -x: u = env(i - i0 + vg n) cos(kx i + w n)."""
    i = np.arange(Nx, dtype=np.float64) + shift
    u0 = np.exp(-0.5 * ((i - i0) / sig) ** 2) * np.cos(kx * i)
    u1 = np.exp(-0.5 * ((i - i0 + vg) / sig) ** 2) * np.cos(kx * i + w)
    return u0, u1


# ---- boundary rules as polynomials in S (one cell into the interior) and B (one step back) ------
# A rule is a list of terms (a, b, c0, c1): c0*u[a]^(n+1-b) + c1*(D_tt u)[a]^(n+1-b), summed = 0,
# with D_tt the tangential second difference along the wall (m * u in the 1D reduction).
# The boundary node is then u[0]^(n+1) = -(sum of the other terms) / c0(0, 0).
def rule_mur1(Co):
    k = (Co - 1) / (Co + 1)
    return [(0, 0, 1.0, 0.0), (1, 1, -1.0, 0.0), (1, 0, -k, 0.0), (0, 1, k, 0.0)]


def rule_mur2(Co):
    """Mur (1981) 2nd order = Engquist-Majda A2 (u_xt - u_tt/c + (c/2) u_yy = 0) centred at
    (x = h/2, t = n dt), in the form printed in Taflove & Hagness:
      u0' = -u1'' + k (u1' + u0'') + 2/(Co+1) (u0 + u1) + Co^2/(2(Co+1)) (Dtt u0 + Dtt u1)
    (' = n+1, '' = n-1)."""
    k = (Co - 1) / (Co + 1)
    g = 2 / (Co + 1)
    e = Co * Co / (2 * (Co + 1))
    return [(0, 0, 1.0, 0.0), (1, 2, 1.0, 0.0), (1, 0, -k, 0.0), (0, 2, -k, 0.0),
            (0, 1, -g, -e), (1, 1, -g, -e)]


def _poly_mul(P, Q):
    R = {}
    for (a1, b1), c1 in P.items():
        for (a2, b2), c2 in Q.items():
            R[(a1 + a2, b1 + b2)] = R.get((a1 + a2, b1 + b2), 0.0) + c1 * c2
    return R


def rule_higdon(Co, angles_deg, eps=0.0):
    """Higdon (1986/1987): prod_j (cos(a_j) d/dt - c d/dx) u = 0, each factor discretised with
    the box scheme (d/dt averaged over the two nodes, d/dx averaged over the two levels):
      2 dt * factor = cos a (1 - B)(1 + S) - Co (S - 1)(1 + B).
    One factor with a = 0 is exactly Mur1. Order p needs p+1 time levels and p+1 cells.
    eps > 0 adds a damping term to every factor, (cos a d/dt - c d/dx + eps), averaged over the
    four nodes; it removes the zero-frequency (static-ramp / drift) solutions that the undamped
    product admits (measured 29/09: undamped order 3 grows linearly in time at ky = 0)."""
    P = {(0, 0): 1.0}
    e = 0.5 * eps * Co
    for a in angles_deg:
        ca = math.cos(math.radians(a))
        F = {(0, 0): ca + Co + e, (1, 0): ca - Co + e, (0, 1): -ca + Co + e, (1, 1): -ca - Co + e}
        P = _poly_mul(P, F)
    return [(a, b, c, 0.0) for (a, b), c in P.items()]


def rule_liao(Co, order, ca=1.0, composed=True):
    """Liao's multi-transmitting formula (Liao et al. 1984): extrapolate along the outgoing
    characteristic, u0^(n+1) = sum_j (-1)^(j+1) C(N, j) u(x = j*ca*dt, t = (n+1-j) dt).
    The off-grid values come from quadratic interpolation through nodes 0, 1, 2.
    composed=True (Liao's form, as far as I recall it): the value at j*ca*dt is the j-fold
      COMPOSITION of the one-step interpolation, so the rule is exactly (1 - B*Phi(S))^N with
      Phi(S) = w0 + w1 S + w2 S^2, reaching nodes 0..2N.
    composed=False: interpolate directly at j*ca*dt from the 3 nearest nodes. Measured 29/09:
      this is INCONSISTENT for N = 3 (|R| -> 0.143 as lambda -> infinity, normal incidence)
      and only first order for N = 2: the interpolation error (O(k^3)) is not raised to the
      N-th power, while the outgoing-wave residual P(-) shrinks like k^N."""
    def w(s):
        return ((s - 1) * (s - 2) / 2, -s * (s - 2), s * (s - 1) / 2)
    if composed:
        w0, w1, w2 = w(ca * Co)
        F = {(0, 0): 1.0, (0, 1): -w0, (1, 1): -w1, (2, 1): -w2}
        P = {(0, 0): 1.0}
        for _ in range(order):
            P = _poly_mul(P, F)
        return [(a, b, c, 0.0) for (a, b), c in P.items() if c != 0.0]
    terms = [(0, 0, 1.0, 0.0)]
    for j in range(1, order + 1):
        s = j * ca * Co
        base = 0 if s <= 2 else int(math.floor(s)) - 1     # nodes 0,1,2 while they bracket s
        nodes = [base, base + 1, base + 2]
        coef = (-1) ** (j + 1) * math.comb(order, j)
        for a in nodes:
            others = [x for x in nodes if x != a]
            wa = (s - others[0]) * (s - others[1]) / ((a - others[0]) * (a - others[1]))
            terms.append((a, j, -coef * wa, 0.0))
    return terms


def rule_depth(rule):
    return max(t[0] for t in rule), max(t[1] for t in rule)


def rule_theory(rule, Co, kx, ky):
    """|R| of a discrete rule for the exact lattice plane wave: R = -P(e^{+ikx}) / P(e^{-ikx})."""
    w, _ = dispersion(Co, kx, ky)
    m = 2 * math.cos(ky) - 2
    z = complex(math.cos(w), math.sin(w))
    def P(sgn):
        return sum((c0 + c1 * m) * complex(math.cos(sgn * kx * a), math.sin(sgn * kx * a)) * z ** (-b)
                   for a, b, c0, c1 in rule)
    return abs(P(+1) / P(-1))


# ---- the 1D-reduced solver ------------------------------------------------------------------------
class Sim1D:
    """u[i] for i = 0..Nx-1; node Nx-1 is Dirichlet (far away). `left` picks the left end:
      ('dir',)                    u[0] = 0
      ('rule', terms)             a local ABC at node 0 (Mur1/Mur2/Higdon/Liao)
      ('sponge', L, dmax, p, form, end)   damping d(i) = dmax ((L-i)/L)^p on 0 < i < L, in the
                                  app's form  - d (u - u_prev)  (form 'app') or the centred
                                  form (form 'ctr'); end = 'dir' or 'mur1' at node 0
      ('cerjan', L, a)            Cerjan et al. 1985: both levels multiplied by exp(-(a(L-i))^2)
      ('cpml', L, fac, p, alpha)  convolutional PML with memory variables (Roden-Gedney recursion)
      ('gs', L, fac, p)           Grote-Sim (2010) second-order PML with auxiliary fields
    """

    def __init__(self, Nx, Co, ky, left):
        self.Nx, self.Co, self.left = Nx, Co, left
        self.m = 2 * math.cos(ky) - 2
        kind = left[0]
        self.hist = 3
        if kind == 'rule':
            _, bmax = rule_depth(left[1])
            self.hist = max(3, bmax + 1)
        self.H = [np.zeros(Nx) for _ in range(self.hist)]   # H[0] = u^{n+1} (scratch), H[1] = u^n, ...
        dt = Co
        if kind in ('sponge',):
            L, dmax, p = left[1], left[2], left[3]
            i = np.arange(Nx)
            self.d = np.where(i < L, dmax * np.clip((L - i) / L, 0, 1) ** p, 0.0)
        if kind == 'cerjan':
            L, a = left[1], left[2]
            i = np.arange(Nx)
            self.G = np.where(i < L, np.exp(-(a * (L - i)) ** 2), 1.0)
        if kind in ('cpml', 'gs'):
            L, fac, p = left[1], left[2], left[3]
            R0 = 1e-6
            smax = fac * (p + 1) * math.log(1 / R0) / (2 * L)      # 1/time (c = h = 1)
            ic = np.arange(Nx, dtype=np.float64)
            icf = np.arange(Nx - 1, dtype=np.float64) + 0.5          # face i+1/2
            self.sc = smax * np.clip((L - ic) / L, 0, 1) ** p
            self.sf = smax * np.clip((L - icf) / L, 0, 1) ** p
            if kind == 'cpml':
                al = left[4]
                bc_, bf_ = np.exp(-(self.sc + al) * dt), np.exp(-(self.sf + al) * dt)
                self.bc_, self.bf_ = bc_, bf_
                self.ac_ = np.where(self.sc > 0, self.sc / (self.sc + al + 1e-300) * (bc_ - 1), 0.0)
                self.af_ = np.where(self.sf > 0, self.sf / (self.sf + al + 1e-300) * (bf_ - 1), 0.0)
                self.m1 = np.zeros(Nx - 1)
                self.m2 = np.zeros(Nx)
            else:
                self.phi = np.zeros(Nx - 1)     # at faces, time n-1/2
                self.psi = np.zeros(Nx)         # at cells, time n-1/2

    def set(self, u_prev, u_cur):
        self.H[2][:] = u_prev
        self.H[1][:] = u_cur

    def step(self):
        Co, m, H = self.Co, self.m, self.H
        C2 = Co * Co
        u, up, un = H[1], H[2], H[0]
        kind = self.left[0]
        lap = np.zeros_like(u)
        lap[1:-1] = u[:-2] + u[2:] - 2 * u[1:-1] + m * u[1:-1]
        if kind == 'cpml':
            g = u[1:] - u[:-1]
            self.m1 = self.bf_ * self.m1 + self.af_ * g
            q = g + self.m1
            dq = np.zeros_like(u)
            dq[1:-1] = q[1:] - q[:-1]
            self.m2 = self.bc_ * self.m2 + self.ac_ * dq
            lap[1:-1] = dq[1:-1] + self.m2[1:-1] + m * u[1:-1]
            un[:] = 2 * u - up + C2 * lap
        elif kind == 'gs':
            dt = Co
            sf, sc = self.sf, self.sc
            g = u[1:] - u[:-1]
            phi_old, psi_old = self.phi.copy(), self.psi.copy()
            self.phi = ((1 - 0.5 * dt * sf) * self.phi - dt * sf * g) / (1 + 0.5 * dt * sf)
            self.psi = self.psi + dt * sc * m * u
            ph = 0.5 * (self.phi + phi_old)
            ps = 0.5 * (self.psi + psi_old)
            extra = np.zeros_like(u)
            extra[1:-1] = ph[1:] - ph[:-1] + ps[1:-1]
            s = sc
            un[:] = (2 * u - up * (1 - 0.5 * dt * s) + C2 * (lap + extra)) / (1 + 0.5 * dt * s)
        elif kind == 'sponge' and self.left[4] == 'ctr':
            s = self.d                     # d = gamma*dt: u_tt + gamma u_t, centred
            un[:] = (2 * u - up * (1 - 0.5 * s) + C2 * lap) / (1 + 0.5 * s)
        elif kind == 'sponge':
            un[:] = (2 * u - up + C2 * lap) - self.d * (u - up)
        else:
            un[:] = 2 * u - up + C2 * lap
        un[-1] = 0.0
        # the left end
        if kind == 'rule':
            terms = self.left[1]
            acc = 0.0
            c00 = None
            for a, b, c0, c1 in terms:
                if a == 0 and b == 0:
                    c00 = c0
                    continue
                acc += (c0 + c1 * m) * H[b][a]
            un[0] = -acc / c00
        elif kind == 'sponge' and self.left[5] == 'mur1':
            k = (Co - 1) / (Co + 1)
            un[0] = u[1] + k * (un[1] - u[0])
        else:
            un[0] = 0.0
        if kind == 'cerjan':
            un *= self.G
            u *= self.G
        # rotate history
        last = H.pop()
        H.insert(0, last)          # new scratch = the oldest array
        # now H[1] = u^{n+1}

    def u(self):
        return self.H[1]


LMAX = 40       # every test domain has room for a 40-cell layer, so one reference serves all methods


def geometry(Co, lam, theta_deg, sigfac=3.0):
    k = TAU / lam
    th = math.radians(theta_deg)
    kx, ky = k * math.cos(th), k * math.sin(th)
    w, vg = dispersion(Co, kx, ky)
    sig = sigfac * lam / max(math.cos(th), 1e-9)
    i0 = int(LMAX + 5 * sig + 10)
    Nx = int(i0 + 5 * sig + 20)
    T = int(round(2 * i0 / vg))
    E = int(5 * sig + LMAX + 50 + 2 * vg)    # reference extension (see the module docstring)
    return dict(Co=Co, lam=lam, th=theta_deg, kx=kx, ky=ky, w=w, vg=vg, sig=sig, i0=i0, Nx=Nx, T=T, E=E)


def run_reference(g):
    """The reference run: returns (u_ref over the test domain's cells at T, incident norm)."""
    Nx, E = g['Nx'], g['E']
    ref = Sim1D(Nx + E, g['Co'], g['ky'], ('dir',))
    b0, b1 = packet(Nx + E, g['i0'], g['sig'], g['kx'], g['w'], g['vg'], shift=-E)  # same phase per cell
    ref.set(b0, b1)
    inc = float(np.linalg.norm(b1))   # the incident packet (the reference's own packet reaches its
    for _ in range(g['T']):           # far wall near T, so ||u_ref(T)|| is NOT a safe normaliser)
        ref.step()
    return ref.u()[E:E + Nx].copy(), inc


def run_test(g, left, L, ref):
    """|R| of the left end `left`; its layer (if any) occupies cells [0, L)."""
    test = Sim1D(g['Nx'], g['Co'], g['ky'], left)
    a0, a1 = packet(g['Nx'], g['i0'], g['sig'], g['kx'], g['w'], g['vg'])
    test.set(a0, a1)
    for _ in range(g['T']):
        test.step()
    ur, inc = ref
    u = test.u()
    if not np.all(np.isfinite(u)):
        return float('inf')
    return float(np.linalg.norm(u[L:] - ur[L:]) / inc)


def reflection(left, Co, lam, theta_deg, L=1):
    g = geometry(Co, lam, theta_deg)
    return run_test(g, left, L, run_reference(g))


def growth_check(left, Co, ky, Nx=400, steps=20000, seed=1):
    """Random initial data in a 1D-reduced cell; max|u| at a few times (instability check for one
    ky -- the 2D box check covers all wavenumbers at once)."""
    rng = np.random.default_rng(seed)
    s = Sim1D(Nx, Co, ky, left)
    u0 = rng.uniform(-1, 1, Nx)
    u0[0] = u0[-1] = 0
    s.set(u0, u0)
    out = {}
    for n in range(1, steps + 1):
        s.step()
        if n in (1000, 5000, 10000, 20000):
            out[n] = float(np.max(np.abs(s.u())))
    return out
