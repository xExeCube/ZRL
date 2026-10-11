"""Scale mode tests (02/10/2026, NotesCLAUDE/scale_MIRROR.md).

'domain' mode gives every size as a fraction of the domain's inradius r_in, so a larger N should show
the SAME picture with less grid error. Checked here:
  1. 'cells' mode (the default) is bit-identical to the core before the scale mode (15 configs, the
     reference fields ScratchCLAUDE/ScaleCLAUDE/ref_cells_<arch>_MIRROR.npz made by make_ref_MIRROR.py
     from the pre-edit core), and the fraction fields are ignored in 'cells' mode.
  2. the mode switch, N and Co changes, the warnings and the crossing clock.
  3. CONVERGENCE: a centred pulse (sigma = 0.05 r_in) on sq5, sq9 and tri at N = 257, 513, 1025, 2049;
     the field at T = 0.5 crossings, sampled at fixed fractional positions (a common 181 x 181 grid on
     [-0.9, 0.9]^2, cubic-spline resampling, cubic Lagrange in time); the difference between successive
     N should fall ~4x per doubling (second order).
  4. AMPLITUDE: the peak over time of u at the fractional point r = 0.5 (0 and 45 deg) is independent of
     N in 'domain' mode (spread reported), while in 'cells' mode (sigma = 3 cells) it falls ~1/sqrt(2)
     per doubling.

Usage: python scale_tests_MIRROR.py [--arch cuda|cpu] [--quick]   (never bare pytest)
"""
import argparse
import math
import os
import sys
import time

import numpy as np
from scipy.ndimage import map_coordinates

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'ScratchCLAUDE', 'ScaleCLAUDE'))
import membrane_core_MIRROR as mc

RESULTS = []
LATS = (('sq5', dict(lattice='sq', stencil=5)), ('sq9', dict(lattice='sq', stencil=9)), ('tri', dict(lattice='tri')))
SIGMA_F = 0.05          # pulse sigma / r_in
T_SNAP = 0.5            # snapshot time (crossings, r_in / c)
R0 = 0.5                # the amplitude probe's fractional radius
CO = 0.5


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok), detail))
    print(f'[{"PASS" if ok else "FAIL"}] {name}' + (f' -- {detail}' if detail else ''), flush=True)


def frac_to_index(m, X, Y):
    """Fractional position (units of r_in, about the array centre) -> fractional index (i, j)."""
    r, c = m.inner_R(), m.c()
    x, y = np.asarray(X) * r, np.asarray(Y) * r
    if m.tri():
        db = y / mc.S3H
        return c + x - 0.5 * db, c + db
    return c + x, c + y


def lag4(t):
    """Cubic Lagrange weights on nodes -1, 0, 1, 2 at t."""
    return np.array([-t * (t - 1) * (t - 2) / 6, (t + 1) * (t - 1) * (t - 2) / 2,
                     -(t + 1) * t * (t - 2) / 2, (t + 1) * t * (t - 1) / 6])


def patch_value(U, fi, fj):
    """u at the fractional index (fi, fj), cubic Lagrange on the 4 x 4 cells around it."""
    i0, j0 = math.floor(fi), math.floor(fj)
    wi, wj = lag4(fi - i0), lag4(fj - j0)
    return float(wi @ U[i0 - 1:i0 + 3, j0 - 1:j0 + 3] @ wj)


def domain_membrane(N, lat, **kw):
    p = mc.Params(N=N, Co=CO, src='pulse', scale='domain', sigma_f=SIGMA_F, **lat)
    for k, v in kw.items():
        setattr(p, k, v)
    m = mc.Membrane(p)
    m.record = False
    return m


def snapshot(m, T, offset=0.5):
    """The field at T crossings as a function of fractional position: cubic Lagrange in time over 4
    levels. offset = 0.5: a run started at rest (u_prev = u) is the continuum solution that is at rest
    half a step EARLIER, so level n samples t = (n + 1/2) dt (NotesCLAUDE/scale_MIRROR.md section 3)."""
    s = T * m.inner_R() / m.p.Co - offset
    n0 = math.floor(s)
    m.step(n0)
    lv = [m.level(0), m.level(1)]
    for _ in range(2):
        m.step(1)
        lv.append(m.level(1))
    w = lag4(s - n0)
    return sum(wk * u for wk, u in zip(w, lv))


def peak_series(m, pts, t0=0.3, t1=0.75):
    """max over time of |u| at fractional points pts (list of (X, Y)), sampled every step from t0 to t1
    crossings, refined by a parabola through the three samples around the maximum."""
    r, co = m.inner_R(), m.p.Co
    m.step(int(t0 * r / co))
    idx = [frac_to_index(m, X, Y) for X, Y in pts]
    ser = [[] for _ in pts]
    for _ in range(int((t1 - t0) * r / co)):
        m.step(1)
        U = m.level(1)
        for q, (fi, fj) in enumerate(idx):
            ser[q].append(patch_value(U, float(fi), float(fj)))
    out = []
    for s in ser:
        a = np.abs(np.array(s))
        k = int(np.argmax(a))
        if 0 < k < len(a) - 1:
            y0, y1, y2 = a[k - 1], a[k], a[k + 1]
            den = y0 - 2 * y1 + y2
            out.append(y1 - (y0 - y2) ** 2 / (8 * den) if den < 0 else y1)
        else:
            out.append(a[k])
    return out


# ---- 1. 'cells' mode unchanged ------------------------------------------------------------------
def t_cells_identity(arch):
    path = os.path.join(HERE, 'ScratchCLAUDE', 'ScaleCLAUDE', f'ref_cells_{"cpu" if arch in ("cpu", "x64") else arch}_MIRROR.npz')
    if not os.path.exists(path):
        check('cells mode bit-identical to the pre-scale core', False, f'no reference {path}')
        return
    import make_ref_MIRROR as mr
    ref = np.load(path)
    f = mr.ref_fields()
    bad = [k for k in ref.files if not np.array_equal(ref[k], f[k])]
    check('cells mode bit-identical to the pre-scale core (15 configs x 2 levels, 300 steps)', not bad,
          f'differ: {bad}' if bad else f'{len(ref.files)} configs, max|u| up to {max(np.abs(ref[k]).max() for k in ref.files):.3g}')
    # the fractions are ignored in cells mode
    m1 = mc.Membrane(mc.Params(N=257))
    m2 = mc.Membrane(mc.Params(N=257, sigma_f=0.2, lam_f=0.3, bw_f=0.1, pkl_f=0.1, pklam_f=0.1))
    m1.step(200)
    m2.step(200)
    check('cells mode ignores the fraction fields', np.array_equal(m1.level(1), m2.level(1)) and m2.p.sigma == 3.0)


# ---- 2. switching, N, Co, warnings, clock ---------------------------------------------------------
def t_logic():
    p = mc.Params(N=257, src='beam', sigma=4.0, freq=0.07, beam_width=10.0, pk_len=12.0, pk_lambda=20.0)
    m = mc.Membrane(p)
    m.step(50)
    before = (p.sigma, p.freq, p.beam_width, p.pk_len, p.pk_lambda)
    u_before = m.level(1)
    m.set_scale('domain')
    r = m.inner_R()
    same = before == (p.sigma, p.freq, p.beam_width, p.pk_len, p.pk_lambda)
    fr_ok = (abs(p.sigma_f - 4.0 / r) < 1e-15 and abs(p.lam_f - (p.Co / 0.07) / r) < 1e-15
             and abs(p.bw_f * r - 10.0) < 1e-12)
    check('entering domain mode keeps every cell size exactly and the field untouched (no jump)',
          same and fr_ok and np.array_equal(u_before, m.level(1)) and m.step_n == 50,
          f'r_in {r}, sigma_f {p.sigma_f:.5f}, lam_f {p.lam_f:.5f}')
    p.N = 1025
    m.reseed()
    r2 = m.inner_R()
    check('domain mode: N 257 -> 1025 keeps every fraction (sizes scale with r_in)',
          abs(p.sigma / r2 - 4.0 / r) < 1e-14 and abs(p.beam_width / r2 - 10.0 / r) < 1e-14
          and abs((p.Co / p.freq) / r2 - (0.5 / 0.07) / r) < 1e-12,
          f'sigma {p.sigma:.4f}, beam width {p.beam_width:.4f}, wavelength {p.Co / p.freq:.4f} cells')
    lam = p.Co / p.freq
    m.set_co(0.3)
    check('domain mode: a Co change keeps the wavelength (freq = Co / lambda)', abs(p.Co / p.freq - lam) < 1e-9,
          f'freq {p.freq:.6f} cyc/step at Co 0.3')
    sig = p.sigma
    m.set_scale('cells')
    p.N = 257
    m.reseed()
    check('back to cells mode: the cell sizes stay put (N change no longer scales them)', p.sigma == sig)
    # warnings
    q = mc.Params(N=49, src='pulse', scale='domain', sigma_f=0.02)
    mw = mc.Membrane(q)
    w_small = mw.scale_warnings()
    q.N = 513
    mw.reseed()
    w_big = mw.scale_warnings()
    q.scale = 'cells'
    q.N = 49
    mw.reseed()
    check('warnings: sigma under 2 cells at small N in domain mode only',
          len(w_small) == 1 and not w_big and not mw.scale_warnings(), f'"{w_small[0] if w_small else ""}"')
    qb = mc.Params(N=129, src='beam', scale='domain', lam_f=0.1, bw_f=0.02)
    mb = mc.Membrane(qb)
    wb = mb.scale_warnings()
    check('warnings: beam wavelength under 8 cells and width under 2 cells', len(wb) == 2, ' | '.join(wb))
    # the crossing clock
    mt = mc.Membrane(mc.Params(N=257, scale='domain', sigma_f=0.05))
    mt.step(512)
    check('crossings() = steps * Co / r_in', abs(mt.crossings() - 512 * 0.5 / mt.inner_R()) < 1e-12,
          f'{mt.crossings():.6f} crossings after 512 steps at N = 257')


# ---- 3. convergence ---------------------------------------------------------------------------------
def t_convergence(NS):
    g = np.linspace(-0.9, 0.9, 181)
    X, Y = np.meshgrid(g, g, indexing='ij')
    lines = []
    for name, lat in LATS:
        fields, fields_no = [], []
        for N in NS:
            for off, store in ((0.5, fields), (0.0, fields_no)):
                m = domain_membrane(N, lat)
                U = snapshot(m, T_SNAP, offset=off)
                I, J = frac_to_index(m, X, Y)
                store.append(map_coordinates(U, [I, J], order=3, mode='nearest'))
        scale = np.abs(fields[-1]).max()
        d = [np.abs(fields[k + 1] - fields[k]).max() / scale for k in range(len(NS) - 1)]
        d2 = [math.sqrt(np.mean((fields[k + 1] - fields[k]) ** 2)) / scale for k in range(len(NS) - 1)]
        dn = [np.abs(fields_no[k + 1] - fields_no[k]).max() / scale for k in range(len(NS) - 1)]
        rat = [d[k] / d[k + 1] for k in range(len(d) - 1)]
        rat2 = [d2[k] / d2[k + 1] for k in range(len(d2) - 1)]
        ratn = [dn[k] / dn[k + 1] for k in range(len(dn) - 1)]
        lines.append((name, d, rat, d2, rat2, dn, ratn))
        check(f'{name}: domain-mode field converges at second order (max-norm ratio per doubling ~4)',
              all(3.0 < x < 5.5 for x in rat) and all(3.0 < x < 5.5 for x in rat2),
              'max diff/peak ' + ', '.join(f'{x:.3e}' for x in d) + '; ratios ' + ', '.join(f'{x:.2f}' for x in rat)
              + '; RMS ratios ' + ', '.join(f'{x:.2f}' for x in rat2)
              + '; without the half-step offset: ratios ' + ', '.join(f'{x:.2f}' for x in ratn))
    return lines


# ---- 4. amplitude independent of N -----------------------------------------------------------------
def t_amplitude(NS):
    pts = [(R0, 0.0), (R0 / math.sqrt(2), R0 / math.sqrt(2))]
    rows = []
    for name, lat in LATS:
        pk = [peak_series(domain_membrane(N, lat), pts) for N in NS]
        a0 = [x[0] for x in pk]
        a45 = [x[1] for x in pk]
        spread = lambda a: (max(a) - min(a)) / np.mean(a)
        last = lambda a: abs(a[-1] - a[-2]) / a[-1]
        rows.append((name, a0, a45))
        check(f'{name}: domain-mode peak at r = {R0} r_in independent of N (spread < 1%)',
              spread(a0) < 0.01 and spread(a45) < 0.01,
              '0 deg ' + ', '.join(f'{x:.5f}' for x in a0) + f' (spread {100 * spread(a0):.3f}%, last doubling {100 * last(a0):.4f}%)'
              + '; 45 deg ' + ', '.join(f'{x:.5f}' for x in a45) + f' (spread {100 * spread(a45):.3f}%)')
    # cells mode for contrast: sigma = 3 cells
    pc = []
    for N in NS:
        m = mc.Membrane(mc.Params(N=N, Co=CO, src='pulse'))
        m.record = False
        pc.append(peak_series(m, pts[:1])[0])
    rat = [pc[k + 1] / pc[k] for k in range(len(pc) - 1)]
    check('cells mode (sigma = 3 cells) for contrast: the peak falls ~1/sqrt(2) per doubling',
          all(0.55 < x < 0.8 for x in rat),
          'peaks ' + ', '.join(f'{x:.5f}' for x in pc) + '; ratios ' + ', '.join(f'{x:.4f}' for x in rat))
    return rows, pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='cuda')
    ap.add_argument('--quick', action='store_true', help='N up to 1025 only')
    a = ap.parse_args()
    mc.init_taichi(a.arch, fp64=True, quiet=True)
    NS = [257, 513, 1025] if a.quick else [257, 513, 1025, 2049]
    t0 = time.perf_counter()
    t_cells_identity(a.arch)
    t_logic()
    t_convergence(NS)
    t_amplitude(NS)
    nf = sum(1 for r in RESULTS if not r[1])
    print(f'\n{len(RESULTS) - nf}/{len(RESULTS)} passed ({time.perf_counter() - t0:.0f} s, arch {a.arch}, N {NS})')
    sys.exit(1 if nf else 0)


if __name__ == '__main__':
    main()
