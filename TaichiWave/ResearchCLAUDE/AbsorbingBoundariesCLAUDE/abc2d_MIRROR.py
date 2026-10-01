"""Absorbing boundaries for the ZRL membrane -- 2D prototypes on the square 5-point lattice (numpy).

The app's update, u' = 2u - u_prev + Co^2 lap(u) - d (u - u_prev), on an N x N box whose outer
ring (or outer L-cell frame) carries the absorbing boundary. Indexing [i, j] as in the core.

Left-end specs mirror abc1d_MIRROR.py:
  ('dir',)                          u = 0 on the ring
  ('rule', terms, corner)           a local ABC on all four walls (terms from abc1d_MIRROR),
                                    corner in {'zero' (the app's Mur), 'diag' (Mur1 along the
                                    diagonal), 'avg' (the normal rule along both edge lines,
                                    tangential terms dropped, averaged)}
  ('sponge', L, dmax, p, form, end) per-cell damping d = dmax s^p, s = normalised depth into the
                                    L-cell frame, s = max(sx, sy) -- i.e. by distance to the
                                    nearest edge, the rule that generalises to masked shapes
  ('cpml', L, fac, p, alpha)        convolutional PML (memory variables), both axes, corners
                                    get both
"""
import math
import numpy as np

import abc1d_MIRROR as A


def _depth(N, L, pos):
    """Normalised depth (0 inside, 1 at the outer ring) of positions pos (cell i or face i+1/2)."""
    if L <= 0:
        return np.zeros_like(pos)
    s = np.maximum(np.maximum(L - pos, pos - (N - 1 - L)), 0.0) / L
    return np.clip(s, 0, 1)


class Box2D:
    def __init__(self, N, Co, spec, damp=0.0):
        self.N, self.Co, self.spec = N, Co, spec
        kind = spec[0]
        hist = 3
        if kind == 'rule':
            hist = max(3, A.rule_depth(spec[1])[1] + 1)
        self.H = [np.zeros((N, N)) for _ in range(hist)]
        self.damp = damp
        ic = np.arange(N, dtype=np.float64)
        if kind == 'sponge':
            L, dmax, p = spec[1], spec[2], spec[3]
            s = np.maximum(_depth(N, L, ic)[:, None], _depth(N, L, ic)[None, :])
            self.d = dmax * s ** p
        if kind == 'cpml':
            L, fac, p, al = spec[1], spec[2], spec[3], spec[4]
            smax = fac * (p + 1) * math.log(1e6) / (2 * L)
            dt = Co
            icf = np.arange(N - 1, dtype=np.float64) + 0.5
            sc = smax * _depth(N, L, ic) ** p
            sf = smax * _depth(N, L, icf) ** p

            def coefs(s):
                b = np.exp(-(s + al) * dt)
                a = np.where(s > 0, s / (s + al + 1e-300) * (b - 1), 0.0)
                return b, a
            self.bc, self.ac = coefs(sc)
            self.bf, self.af = coefs(sf)
            self.mx1 = np.zeros((N - 1, N))
            self.my1 = np.zeros((N, N - 1))
            self.mx2 = np.zeros((N, N))
            self.my2 = np.zeros((N, N))

    def set(self, u_prev, u_cur):
        self.H[2][:] = u_prev
        self.H[1][:] = u_cur

    def u(self):
        return self.H[1]

    def uprev(self):
        return self.H[2]

    # -- walls for local rules ------------------------------------------------------------------
    def _line(self, b, side, a, sl=slice(1, -1)):
        H, N = self.H, self.N
        if side == 'x0':
            return H[b][a, sl]
        if side == 'x1':
            return H[b][N - 1 - a, sl]
        if side == 'y0':
            return H[b][sl, a]
        return H[b][sl, N - 1 - a]

    def _dtt(self, b, side, a):
        full = self._line(b, side, a, slice(None))
        return full[:-2] - 2 * full[1:-1] + full[2:]

    def _apply_rule(self, terms):
        un = self.H[0]
        N = self.N
        for side in ('x0', 'x1', 'y0', 'y1'):
            acc = 0.0
            c00 = 1.0
            for a, b, c0, c1 in terms:
                if a == 0 and b == 0:
                    c00 = c0
                    continue
                acc = acc + c0 * self._line(b, side, a)
                if c1 != 0.0:
                    acc = acc + c1 * self._dtt(b, side, a)
            val = -acc / c00
            if side == 'x0':
                un[0, 1:-1] = val
            elif side == 'x1':
                un[N - 1, 1:-1] = val
            elif side == 'y0':
                un[1:-1, 0] = val
            else:
                un[1:-1, N - 1] = val

    def _corners(self, terms, mode):
        un, H, N, Co = self.H[0], self.H, self.N, self.Co
        cs = [(0, 0, 1, 1), (N - 1, 0, -1, 1), (0, N - 1, 1, -1), (N - 1, N - 1, -1, -1)]
        for ci, cj, si, sj in cs:
            if mode == 'zero':
                un[ci, cj] = 0.0
            elif mode == 'diag':
                kd = (Co - math.sqrt(2)) / (Co + math.sqrt(2))
                un[ci, cj] = H[1][ci + si, cj + sj] + kd * (un[ci + si, cj + sj] - H[1][ci, cj])
            else:   # 'avg': the normal rule along the edge row and along the edge column
                vals = []
                for dirn in ('i', 'j'):
                    acc, c00 = 0.0, 1.0
                    for a, b, c0, c1 in terms:
                        if a == 0 and b == 0:
                            c00 = c0
                            continue
                        v = H[b][ci + si * a, cj] if dirn == 'i' else H[b][ci, cj + sj * a]
                        acc += c0 * v
                    vals.append(-acc / c00)
                un[ci, cj] = 0.5 * (vals[0] + vals[1])

    def step(self):
        H, Co, N = self.H, self.Co, self.N
        C2 = Co * Co
        u, up, un = H[1], H[2], H[0]
        kind = self.spec[0]
        if kind == 'cpml':
            gx = u[1:, :] - u[:-1, :]
            self.mx1 = self.bf[:, None] * self.mx1 + self.af[:, None] * gx
            qx = gx + self.mx1
            gy = u[:, 1:] - u[:, :-1]
            self.my1 = self.bf[None, :] * self.my1 + self.af[None, :] * gy
            qy = gy + self.my1
            dqx = np.zeros_like(u)
            dqy = np.zeros_like(u)
            dqx[1:-1, :] = qx[1:, :] - qx[:-1, :]
            dqy[:, 1:-1] = qy[:, 1:] - qy[:, :-1]
            self.mx2 = self.bc[:, None] * self.mx2 + self.ac[:, None] * dqx
            self.my2 = self.bc[None, :] * self.my2 + self.ac[None, :] * dqy
            lap = dqx + self.mx2 + dqy + self.my2
            un[1:-1, 1:-1] = 2 * u[1:-1, 1:-1] - up[1:-1, 1:-1] + C2 * lap[1:-1, 1:-1]
        else:
            lap = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * u[1:-1, 1:-1]
            ui, upi = u[1:-1, 1:-1], up[1:-1, 1:-1]
            if kind == 'sponge' and self.spec[4] == 'ctr':
                s = self.d[1:-1, 1:-1]
                un[1:-1, 1:-1] = (2 * ui - upi * (1 - 0.5 * s) + C2 * lap) / (1 + 0.5 * s)
            elif kind == 'sponge':
                un[1:-1, 1:-1] = (2 * ui - upi + C2 * lap) - self.d[1:-1, 1:-1] * (ui - upi)
            else:
                un[1:-1, 1:-1] = (2 * ui - upi + C2 * lap) - self.damp * (ui - upi)
        # the ring
        if kind == 'rule':
            self._apply_rule(self.spec[1])
            self._corners(self.spec[1], self.spec[2])
        elif kind == 'sponge' and self.spec[5] == 'mur1':
            self._apply_rule(A.rule_mur1(Co))
            self._corners(A.rule_mur1(Co), 'zero')
        else:
            un[0, :] = 0.0
            un[-1, :] = 0.0
            un[:, 0] = 0.0
            un[:, -1] = 0.0
        last = H.pop()
        H.insert(0, last)


def layer_of(spec):
    return spec[1] if spec[0] in ('sponge', 'cpml') else 1


def energy(d1, d0, Co, region):
    """Leapfrog energy of a field pair (d0 = level n, d1 = level n+1) restricted to `region`:
    sum (d1-d0)^2/Co^2 + sum over edges inside the region of grad(d1).grad(d0)."""
    kin = np.sum(((d1 - d0) ** 2)[region]) / (Co * Co)
    ex = region[1:, :] & region[:-1, :]
    ey = region[:, 1:] & region[:, :-1]
    pot = np.sum(((d1[1:, :] - d1[:-1, :]) * (d0[1:, :] - d0[:-1, :]))[ex])
    pot += np.sum(((d1[:, 1:] - d1[:, :-1]) * (d0[:, 1:] - d0[:, :-1]))[ey])
    return kin + pot


def gauss(N, ci, cj, s):
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    return np.exp(-((i - ci) ** 2 + (j - cj) ** 2) / (2 * s * s))


def pulse_reflection(spec, Co, Nphys, src, r_end, s=3.0, ref_cache=None):
    """A Gaussian pulse (both levels equal, as the app seeds it) at phys cell src; run until the
    front has travelled r_end cells; return sqrt(E_refl / E0), with E_refl the energy of
    (test - reference) over the physical cells and E0 the pulse's energy at launch."""
    T = int(round(r_end / Co))
    L = layer_of(spec)
    Nt = Nphys + 2 * L
    E = int(0.5 * r_end) + 30
    Nr = Nphys + 2 * E
    key = (Co, Nphys, src, r_end, s)
    if ref_cache is not None and key in ref_cache:
        ref_u1, ref_u0, E0 = ref_cache[key]
    else:
        ref = Box2D(Nr, Co, ('dir',))
        g = gauss(Nr, src[0] + E, src[1] + E, s)
        ref.set(g, g)
        allr = np.ones((Nr, Nr), bool)
        for n in range(T):
            ref.step()
            if n == 0:
                E0 = energy(ref.u(), ref.uprev(), Co, allr)
        ref_u1 = ref.u()[E:E + Nphys, E:E + Nphys].copy()
        ref_u0 = ref.uprev()[E:E + Nphys, E:E + Nphys].copy()
        if ref_cache is not None:
            ref_cache[key] = (ref_u1, ref_u0, E0)
    test = Box2D(Nt, Co, spec)
    g = gauss(Nt, src[0] + L, src[1] + L, s)
    test.set(g, g)
    for n in range(T):
        test.step()
    d1 = test.u()[L:L + Nphys, L:L + Nphys] - ref_u1
    d0 = test.uprev()[L:L + Nphys, L:L + Nphys] - ref_u0
    reg = np.ones((Nphys, Nphys), bool)
    reg[0, :] = reg[-1, :] = reg[:, 0] = reg[:, -1] = False     # the rule methods' ring is not physical
    Er = energy(d1, d0, Co, reg)
    return math.sqrt(max(Er, 0.0) / E0)


def noise_run(spec, Co, Nphys=100, steps=20000, seed=3, marks=(1000, 5000, 10000, 20000)):
    """White-noise initial data (both levels equal) in the physical cells: max|u| and energy
    over the physical cells at the marks. A stable absorbing boundary lets both fall or level
    off (standing near-checkerboard modes have ~zero group velocity and barely reach a wall)."""
    L = layer_of(spec)
    Nt = Nphys + 2 * L
    rng = np.random.default_rng(seed)
    b = Box2D(Nt, Co, spec)
    u0 = np.zeros((Nt, Nt))
    u0[L + 1:L + Nphys - 1, L + 1:L + Nphys - 1] = rng.uniform(-1, 1, (Nphys - 2, Nphys - 2))
    b.set(u0, u0)
    reg = np.zeros((Nt, Nt), bool)
    reg[L:L + Nphys, L:L + Nphys] = True
    E0 = energy(u0, u0, Co, reg)
    out = {}
    for n in range(1, steps + 1):
        b.step()
        if n in marks:
            u = b.u()
            if not np.all(np.isfinite(u)):
                out[n] = (float('inf'), float('inf'))
                break
            out[n] = (float(np.max(np.abs(u))), float(energy(u, b.uprev(), Co, reg) / E0))
        if n > 1000 and not np.isfinite(b.u()[Nt // 2, Nt // 2]):
            out[n] = (float('inf'), float('inf'))
            break
    return out
