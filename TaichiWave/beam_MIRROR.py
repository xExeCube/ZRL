"""Directional sources for the membrane: the lattice's discrete dispersion, the beam aperture
and the directed wave packet (pure numpy; membrane_core_MIRROR.py calls into it).

Generated from scratch -- Claude Opus 5.5 -- 01/10/2026. Design and measurements:
NotesCLAUDE/beam_MIRROR.md.

Lattice symbols (physical wavevector K = (kx, ky), spacing 1). The Laplacians of the core are
L u -> -Lambda(K) u for a plane wave, with
    sq5: Lambda = 4 sin^2(kx/2) + 4 sin^2(ky/2)
    sq9: Lambda = 10/3 - (4/3)(cos kx + cos ky) - (2/3) cos kx cos ky
    tri: Lambda = (2/3) (6 - 2 sum_{e in e1,e2,e3} cos(K.e)),  e1 = (1,0), e2 = (1/2, s3/2), e3 = (-1/2, s3/2)
and the leapfrog (undamped) gives the discrete dispersion  4 sin^2(omega/2) = Co^2 Lambda(K)
(omega per time step, Co^2 = the per-cell CC). Group velocity: grad omega = Co^2 grad Lambda / (2 sin omega).
"""
import math

import numpy as np

S3H = math.sqrt(3) / 2
TRI_E = ((1.0, 0.0), (0.5, S3H), (-0.5, S3H))
# nearest-neighbour vectors (physical): the beam's back aperture is the front one shifted by one
NEIGH = {
    'sq5': ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)),
    'sq9': ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)),
    'tri': ((1.0, 0.0), (0.5, S3H), (-0.5, S3H), (-1.0, 0.0), (-0.5, -S3H), (0.5, -S3H)),
}
APERTURE_TRUNC = 2.5        # the aperture is cut at |s| <= 2.5 w (amplitude e^-6.25 = 0.2%)
# The back aperture: 'slab' (default) = its own slab d behind the front one, every cell phased by its
# own position -- the coarse-grained sheets are then exactly d apart ALONG the beam, and the
# backward lobe cancels to second order in its angle (~29-30 dB at w0 = 8, lambda = 12 on all three
# lattices, beam_tests_MIRROR.py). 'shift' = the front cells translated by one neighbour vector v:
# an exact copy, but v has a component ACROSS the beam except on the lattice axes, which leaves a
# first-order residue (17.6 dB at 45 deg on sq). Measured 01/10/2026.
BACK_MODE = 'slab'
RAMP_PERIODS = 4.0          # raised-cosine switch-on of the beam, in periods of the fundamental


def symbol(kind, kx, ky):
    kx = np.asarray(kx, dtype=np.float64)
    ky = np.asarray(ky, dtype=np.float64)
    if kind == 'sq5':
        return 4 * np.sin(kx / 2) ** 2 + 4 * np.sin(ky / 2) ** 2
    if kind == 'sq9':
        cx, cy = np.cos(kx), np.cos(ky)
        return 10.0 / 3.0 - (4.0 / 3.0) * (cx + cy) - (2.0 / 3.0) * cx * cy
    s = 0.0
    for ex, ey in TRI_E:
        s = s + np.cos(kx * ex + ky * ey)
    return (2.0 / 3.0) * (6 - 2 * s)


def symbol_grad(kind, kx, ky, h=1e-6):
    gx = (symbol(kind, kx + h, ky) - symbol(kind, kx - h, ky)) / (2 * h)
    gy = (symbol(kind, kx, ky + h) - symbol(kind, kx, ky - h)) / (2 * h)
    return gx, gy


def omega_of(kind, co2, kx, ky):
    """Discrete angular frequency (rad/step) of the plane wave K; NaN past the CFL edge."""
    a = np.sqrt(np.maximum(co2 * symbol(kind, kx, ky), 0.0)) / 2
    return np.where(a <= 1, 2 * np.arcsin(np.minimum(a, 1.0)), np.nan)


def group_velocity(kind, co2, kx, ky):
    w = omega_of(kind, co2, kx, ky)
    gx, gy = symbol_grad(kind, kx, ky)
    f = co2 / (2 * np.sin(w))
    return gx * f, gy * f


def k_along(kind, co2, omega, th):
    """|K| of the plane wave of frequency omega with K along angle th (discrete dispersion);
    None past the lattice's cut-off in that direction."""
    tgt = 4 * math.sin(omega / 2) ** 2 / co2
    c, s = math.cos(th), math.sin(th)
    # first zone crossing along the ray: K.e = pi for the most aligned neighbour vector
    emax = max(abs(c * ex + s * ey) for ex, ey in NEIGH[kind])
    k_hi = 1.5 * math.pi / emax
    ks = np.linspace(0, k_hi, 1537)
    lam = symbol(kind, ks * c, ks * s)
    idx = np.nonzero(lam >= tgt)[0]
    if tgt <= 0:
        return 0.0
    if len(idx) == 0:
        return None
    hi = float(ks[idx[0]])
    lo = float(ks[max(0, idx[0] - 1)])
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(symbol(kind, mid * c, mid * s)) < tgt:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def group_angle(kind, co2, k, th):
    gx, gy = symbol_grad(kind, k * math.cos(th), k * math.sin(th))
    return math.atan2(float(gy), float(gx))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def aim(kind, co2, theta, omega=None, k=None, group=True):
    """The wavevector angle th_k whose GROUP velocity points along theta (rad). On an
    anisotropic lattice phase and energy directions differ (except on the symmetry axes);
    group=False returns theta itself (phase fronts perpendicular to theta). Either omega
    (beam: fixed frequency) or k (packet: fixed wavelength) is held."""
    if not group:
        return theta

    def gang(tk):
        kk = k if k is not None else k_along(kind, co2, omega, tk)
        if kk is None or kk <= 0:
            return None
        return group_angle(kind, co2, kk, tk)

    tk = theta
    for _ in range(30):
        g = gang(tk)
        if g is None:
            return theta
        f = _wrap(g - theta)
        if abs(f) < 1e-9:                       # (the numeric gradient's noise floor: on a symmetry
            break                               #  axis the request is kept exactly)
        h = 1e-5
        g2 = gang(tk + h)
        if g2 is None:
            return theta
        dfd = _wrap(g2 - g) / h
        if abs(dfd) < 1e-6:
            break
        step = max(-0.2, min(0.2, f / dfd))
        tk -= step
    if abs(_wrap(tk - theta)) > 0.6:            # pathological (near the cut-off): do not aim
        return theta
    return tk


def back_shift(kind, th):
    """(vx, vy, d): the nearest-neighbour vector most opposed to the direction th and its
    along-beam length d = -n.v > 0. The front aperture is the slab |zeta| < d/2 (one cell
    thick in the sense that its copies shifted by v tile the lattice), the back aperture is
    that slab shifted by v -- disjoint by construction."""
    c, s = math.cos(th), math.sin(th)
    best = max(NEIGH[kind], key=lambda v: -(c * v[0] + s * v[1]))
    return best[0], best[1], -(c * best[0] + s * best[1])


def sheet_gain(kind, co2, k, th, d):
    """The source strength S per cell that makes a planar sheet (slab of thickness d) radiate a
    plane wave of unit amplitude on each side: S = Co^2 dLambda/dk / d (exact for an infinite
    sheet normal to a lattice axis; derived in NotesCLAUDE/beam_MIRROR.md)."""
    h = 1e-6
    c, s = math.cos(th), math.sin(th)
    dl = (float(symbol(kind, (k + h) * c, (k + h) * s)) - float(symbol(kind, (k - h) * c, (k - h) * s))) / (2 * h)
    return co2 * dl / d


def beam_tables(kind, co2, omega, harm, th_k, X, Y, x0, y0, width, one_sided, ok_a, ok_b_fn):
    """The beam's driven cells and their per-harmonic amplitude and phase lag.

    X, Y: physical coordinates of candidate cells (1D arrays, same order as the caller's index
    arrays); ok_a: bool mask of candidates that may be driven (in the domain, not solid, not on
    the array ring). ok_b_fn(sel) -> (idx_b, ok_b): for the selected front cells, the index of
    their back partner (the cell at +v) in the caller's numbering and whether it may be driven.

    Returns dict(front=idx, back=idx, amp=(nh, n), lag=(nh, n), info) where idx are caller
    indices of the driven cells in the order of the amp/lag columns (front first, then back)."""
    c, s = math.cos(th_k), math.sin(th_k)
    vx, vy, d = back_shift(kind, th_k)
    zeta = (X - x0) * c + (Y - y0) * s
    tr = -(X - x0) * s + (Y - y0) * c
    # quantised for the slab tests: on the lattice axes whole rows of cells sit EXACTLY on a slab
    # edge, and cos/sin round-off (cos(pi) = -1, sin(pi) = 1.2e-16) would sort them at random
    # (measured: tri at 180 deg fell from 29.6 to 18.4 dB before this)
    zq, tq = np.round(zeta * 1e9) / 1e9, np.round(tr * 1e9) / 1e9
    d = round(d * 1e9) / 1e9
    slab = (zq >= -d / 2) & (zq < d / 2) & (np.abs(tq) <= APERTURE_TRUNC * width)
    front_all = np.nonzero(slab)[0]
    front = front_all[ok_a[front_all]]
    if one_sided and BACK_MODE == 'slab':
        bslab = (zq >= -1.5 * d) & (zq < -d / 2) & (np.abs(tq) <= APERTURE_TRUNC * width)
        ball = np.nonzero(bslab)[0]
        back = ball[ok_a[ball]]
        n_cut_b = len(ball) - len(back)
        fb = back
    elif one_sided:
        idx_b, ok_b = ok_b_fn(front)
        back = idx_b[ok_b]
        fb = front[ok_b]                        # the front partners of the driven back cells
        n_cut_b = len(front) - len(back)
    else:
        back = np.zeros(0, dtype=np.int64)
        fb = back
        n_cut_b = 0
    nh = len(harm)
    nf, nb = len(front), len(back)
    amp = np.zeros((nh, nf + nb))
    lag = np.zeros((nh, nf + nb))
    info = dict(back_mode=BACK_MODE if one_sided else 'none', d=d, v=(vx, vy), k=[], lam=None, ok=True, warn='', gain=[],
                n_front=nf, n_back=nb, n_clipped=int(len(front_all) - nf) + int(n_cut_b))
    for hi, (h, a) in enumerate(harm):
        kh = k_along(kind, co2, h * omega, th_k)
        info['k'].append(kh)
        if kh is None or kh <= 0:
            if hi == 0:
                info['ok'] = False
                info['warn'] = 'frequency above the lattice cut-off in this direction'
            continue
        g = sheet_gain(kind, co2, kh, th_k, d)
        if one_sided:
            g = g / max(abs(2 * math.sin(kh * d)), 1e-9)       # forward gain |1 - e^{2ikd}|
        info['gain'].append(g)
        gs = np.exp(-(tr / width) ** 2)
        amp[hi, :nf] = a * g * gs[front]
        lag[hi, :nf] = kh * zeta[front]
        if nb:
            # the back cell emits the INVERTED front signal DELAYED by the travel time d / v_p
            amp[hi, nf:] = a * g * gs[fb]
            if BACK_MODE == 'slab':
                lag[hi, nf:] = kh * zeta[fb] + 2 * kh * d + math.pi
            else:
                lag[hi, nf:] = kh * zeta[fb] + kh * d + math.pi
    if info['k'] and info['k'][0]:
        info['lam'] = 2 * math.pi / info['k'][0]
    return dict(front=front, back=back, amp=amp, lag=lag, info=info)


def beam_drive(amp, lag, harm_h, omega, n0, K, ramp_n):
    """(K, ncells) drive values for steps n0 .. n0+K-1: ramp(n) sum_h amp sin(h omega n - lag)."""
    n = (n0 + np.arange(K, dtype=np.float64))[:, None]
    out = np.zeros((K, amp.shape[1]))
    for hi, h in enumerate(harm_h):
        out += amp[hi][None, :] * np.sin(h * omega * n - lag[hi][None, :])
    if ramp_n > 0:
        r = np.where(n < ramp_n, 0.5 * (1 - np.cos(np.pi * np.minimum(n, ramp_n) / ramp_n)), 1.0)
        out *= r
    return out


def packet_levels(kind, co2, x0, y0, th_k, k, width, length, X, Y, tri, group=True):
    """(u0, u_prev) of a one-way wave packet on an index box: u0 = exp(-(zeta/L)^2 - (s/w)^2) cos(k zeta);
    u_prev = the same packet one step earlier, built mode by mode: u0 is split into its
    plane waves (an FFT of the box), only those with K.n > 0 are kept (doubled; the real part of
    the sum is u0 again) and each is moved back by its own discrete omega(K). The packet then
    travels one way under the lattice's own dispersion, with no backward part beyond what the
    box truncation and the K.n = 0 line leave. X, Y: physical coordinates, shape (ni, nj) of
    the box in index order; tri says how index wavenumbers map to physical ones."""
    c, s = math.cos(th_k), math.sin(th_k)
    zeta = (X - x0) * c + (Y - y0) * s
    tr = -(X - x0) * s + (Y - y0) * c
    u0 = np.exp(-(zeta / length) ** 2 - (tr / width) ** 2) * np.cos(k * zeta)
    ni, nj = u0.shape
    a = 2 * np.pi * np.fft.fftfreq(ni)[:, None]
    b = 2 * np.pi * np.fft.fftfreq(nj)[None, :]
    if tri:
        kx, ky = a + 0 * b, (2 * b - a) / math.sqrt(3)
    else:
        kx, ky = a + 0 * b, b + 0 * a
    U0 = np.fft.fft2(u0)
    kn = kx * c + ky * s
    # the K.n = 0 line (weight 1) within a tolerance, so a mirror-symmetric packet stays exactly symmetric
    tol = 1e-7 * np.hypot(kx, ky)
    w = np.where(kn > tol, 2.0, np.where(kn < -tol, 0.0, 1.0))
    om = omega_of(kind, co2, kx, ky)
    om = np.where(np.isfinite(om), om, 0.0)
    um = np.real(np.fft.ifft2(U0 * w * np.exp(1j * om)))
    # the packet's mean group velocity: its plane waves' v_g weighted by their energy |a(K)|^2 --
    # what its energy centroid moves with (slower along n than the central v_g, because the
    # off-axis waves of a finite-width packet go partly sideways)
    pw = np.abs(U0 * w) ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        gx, gy = group_velocity(kind, co2, kx, ky)
    good = np.isfinite(gx) & np.isfinite(gy) & (pw > 0)
    tot = pw[good].sum()
    vmean = (float((pw * np.where(good, gx, 0)).sum() / tot), float((pw * np.where(good, gy, 0)).sum() / tot))
    return u0, um, vmean
