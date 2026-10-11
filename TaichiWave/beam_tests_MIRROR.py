"""Measurements and checks of the directional sources ('beam', 'packet') on sq5, sq9 and tri.

    python beam_tests_MIRROR.py [--arch cuda|cpu|vulkan] [--quick] [--json out.json]

Measures (NotesCLAUDE/beam_MIRROR.md has the tables):
  directivity   forward / backward energy, one-sided vs two-sided
  width         far-field width vs the Gaussian-beam law w(z) = w0 sqrt(1 + (z/zR)^2), zR = pi w0^2 / lambda
  steering      measured energy direction vs requested, angles 0..90, several wavelengths, both aims
  packet        backward residue and group velocity vs the discrete dispersion prediction
  stability     no NaN / growth at Co = 0.99 C_max
  obstacles     a solid in the path (reflection), an aperture crossing a wall / a solid
  cost          ms per step vs the point sources

Generated from scratch -- Claude Opus 5.5 -- 01/10/2026.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import membrane_core_MIRROR as mc      # noqa: E402
import beam_MIRROR as bm               # noqa: E402

LATS = (('sq5', 'sq', 5), ('sq9', 'sq', 9), ('tri', 'tri', 5))
RES = {'checks': [], 'tables': {}}
FAILS = []


def ok(name, cond, det=''):
    RES['checks'].append((name, bool(cond), det))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {det}")
    if not cond:
        FAILS.append(name)


def make(lat, st, **kw):
    prm = dict(N=601, lattice=lat, stencil=st, Co=0.5, bc='dirichlet', src='beam', freq=0.5 / 12,
               beam_width=8.0, beam_angle=0.0)
    prm.update(kw)
    m = mc.Membrane(mc.Params(**prm))
    m.record = False
    return m


def geom(m):
    X, Y = m._grid_xy()
    x0, y0 = m.cell_xy(*m.src[:2])
    return X - x0, Y - y0


def mean_sq(m, T, P):
    """Run T steps, return 2 <u^2> over the last P steps (the local intensity A^2)."""
    m.step(T - P)
    acc = np.zeros((m.N, m.N))
    for _ in range(P):
        m.step(1)
        u = m.field().astype(np.float64)
        acc += u * u
    return 2 * acc / P


def run_beam_intensity(lat, st, lam, ang, w0, onesided=True, group=True, T=520, N=601, extra=None):
    co = 0.5
    if lat == 'tri' and N == 601:
        N = 701                                 # the tri array is only N s3/2 tall: keep the same room
    m = make(lat, st, N=N, freq=co / lam, beam_angle=ang, beam_width=w0, beam_onesided=onesided,
             beam_group=group, **(extra or {}))
    P = max(4, int(round(1 / m.p.freq)))
    I = mean_sq(m, T, P)
    return m, I


def directivity(m, I, ang):
    X, Y = geom(m)
    th = math.radians(ang)
    z = X * math.cos(th) + Y * math.sin(th)
    f = I[z > 3].sum()
    b = I[z < -3].sum()
    return f / max(b, 1e-300)


# ---------------------------------------------------------------------------------------------
def t_directivity():
    print('directivity (forward/backward energy, lambda 12, w0 8, T 520, Co 0.5)')
    tab = []
    for name, lat, st in LATS:
        for ang in (0, 20, 30, 45, 60):
            r1 = directivity(*run_beam_intensity(lat, st, 12, ang, 8, True)[:2], ang)
            r2 = directivity(*run_beam_intensity(lat, st, 12, ang, 8, False)[:2], ang)
            r3 = directivity(*run_beam_intensity(lat, st, 12, ang, 16, True)[:2], ang)
            tab.append((name, ang, r1, r2, r3))
            print(f'    {name} {ang:3d} deg: one-sided {10 * math.log10(r1):6.1f} dB   two-sided {10 * math.log10(r2):6.1f} dB'
                  f'   one-sided w0 16: {10 * math.log10(r3):6.1f} dB')
    RES['tables']['directivity'] = tab
    ok('one-sided forward/backward >= 26 dB (w0 8) on every lattice and angle', min(t[2] for t in tab) >= 10 ** 2.6,
       f'min {10 * math.log10(min(t[2] for t in tab)):.1f} dB')
    ok('one-sided forward/backward >= 31 dB with w0 16', min(t[4] for t in tab) >= 10 ** 3.1,
       f'min {10 * math.log10(min(t[4] for t in tab)):.1f} dB')
    ok('two-sided is ~symmetric (|ratio| within 3 dB of 0)',
       all(abs(10 * math.log10(t[3])) < 3 for t in tab),
       f'range {min(10 * math.log10(t[3]) for t in tab):.1f}..{max(10 * math.log10(t[3]) for t in tab):.1f} dB')


def beam_width_profile(m, I, ang, zs):
    X, Y = geom(m)
    thk = math.radians(m.beam['info']['theta_k'])
    th = math.radians(ang)
    out = []
    for z in zs:
        # cells near the point z along the ENERGY direction, measured across it
        cx, cy = z * math.cos(th), z * math.sin(th)
        zz = (X - cx) * math.cos(th) + (Y - cy) * math.sin(th)
        ss = -(X - cx) * math.sin(th) + (Y - cy) * math.cos(th)
        sel = (np.abs(zz) <= 0.75) & (np.abs(ss) < 0.45 * m.N)
        wgt = I[sel]
        s = ss[sel]
        mu = (wgt * s).sum() / wgt.sum()
        out.append(2 * math.sqrt(max((wgt * (s - mu) ** 2).sum() / wgt.sum(), 0)))
    return out


def t_width():
    W0, LAM = 16.0, 12.0
    print('beam width vs the Gaussian-beam law w0 sqrt(1 + (z/zR)^2) (lambda 12, w0 16: zR = %.1f, '
          'half-angle lambda/(pi w0) = %.3f rad; N 801, 800 steps)' % (math.pi * W0 * W0 / LAM, LAM / (math.pi * W0)))
    tab = []
    zs = (50, 100, 150, 200, 250)
    for name, lat, st in LATS:
        for ang in (0, 30, 45):
            m, I = run_beam_intensity(lat, st, LAM, ang, W0, True, T=800, N=801 if lat == 'sq' else 921)
            lam = m.beam['info']['lam']
            ws = beam_width_profile(m, I, ang, zs)
            zr = math.pi * W0 * W0 / lam
            pred = [W0 * math.sqrt(1 + (z / zr) ** 2) for z in zs]
            th_fit = math.sqrt(max(ws[-1] ** 2 - W0 ** 2, 0)) / zs[-1]
            err = max(abs(w / q - 1) for w, q in zip(ws, pred))
            tab.append((name, ang, lam, ws, pred, th_fit, lam / (math.pi * W0), err))
            print(f'    {name} {ang:2d} deg lam {lam:.2f}: w(z) ' + ' '.join(f'{w:.1f}' for w in ws)
                  + '  law ' + ' '.join(f'{w:.1f}' for w in pred)
                  + f'  half-angle {th_fit:.3f} vs {lam / (math.pi * W0):.3f}  worst {100 * err:.1f}%')
    RES['tables']['width'] = tab
    err = max(t[7] for t in tab)
    ok('beam width follows the Gaussian-beam law within 10% (z = 100..250)', max(max(abs(w / q - 1) for w, q in zip(t[3][1:], t[4][1:])) for t in tab) < 0.10,
       f'worst {100 * max(max(abs(w / q - 1) for w, q in zip(t[3][1:], t[4][1:])) for t in tab):.1f}% (z = 50 included: {100 * err:.1f}%)')
    e2 = max(abs(t[5] / t[6] - 1) for t in tab)
    ok('far-field half-angle within 10% of lambda/(pi w0)', e2 < 0.10, f'worst {100 * e2:.1f}%')


def steer_angle(m, I, ang, r0, r1):
    X, Y = geom(m)
    R = np.hypot(X, Y)
    phi = np.arctan2(Y, X)
    th = math.radians(ang)
    dphi = (phi - th + math.pi) % (2 * math.pi) - math.pi
    sel = (R >= r0) & (R <= r1) & (np.abs(dphi) < math.radians(45))
    w = I[sel]
    return math.degrees(th + math.atan2((w * np.sin(dphi[sel])).sum(), (w * np.cos(dphi[sel])).sum()))


def t_steering(quick):
    lams = (6, 12) if quick else (6, 8, 12, 24)
    angs = (0, 20, 45, 70, 90) if quick else (0, 10, 20, 30, 45, 60, 75, 90)
    print('steering: measured energy direction - requested (deg), annulus 100..200 cells, w0 = 1.5 lambda')
    tab = []
    for name, lat, st in LATS:
        for lam in lams:
            for group in (True, False):
                if not group and lam not in (6, 12):
                    continue
                errs = []
                for ang in angs:
                    m, I = run_beam_intensity(lat, st, lam, ang, 1.5 * lam, True, group)
                    errs.append(steer_angle(m, I, ang, 100, 200) - ang)
                tab.append((name, lam, group, list(angs), errs))
                print(f'    {name} lam {lam:2d} {"energy" if group else "phase "}-aim: '
                      + ' '.join(f'{a}:{e:+.2f}' for a, e in zip(angs, errs)))
    RES['tables']['steering'] = tab
    g = [max(abs(e) for e in t[4]) for t in tab if t[2] and t[1] >= 8]
    ok('energy-aimed beam within 1 deg of the request (lambda >= 8)', not g or max(g) < 1.0,
       f'worst {max(g):.2f} deg' if g else 'skipped (quick)')
    g6 = [max(abs(e) for e in t[4]) for t in tab if t[2] and t[1] == 6]
    ok('energy-aimed beam within 2 deg at lambda 6', max(g6) < 2.0, f'worst {max(g6):.2f} deg')


def envelope(u, up, om):
    return u * u + ((u - up) / (2 * math.sin(om / 2))) ** 2


def t_packet():
    print('packet: backward residue and group velocity (lambda 16, L 12, w 16, N 601, 400 steps)')
    tab = []
    for name, lat, st in LATS:
        for ang in (0, 30, 45):
            m = make(lat, st, src='packet', pk_lambda=16, pk_len=12, beam_width=16, beam_angle=ang)
            pi_ = m.packet_info
            om = pi_['omega']
            X, Y = geom(m)
            th = math.radians(ang)
            z = X * math.cos(th) + Y * math.sin(th)
            ts, cs = [], []
            E0 = None
            for t in range(0, 401, 50):
                if t:
                    m.step(50)
                u, up = m.level(1).astype(np.float64), m.level(0).astype(np.float64)
                E = envelope(u, up, om)
                if E0 is None:
                    E0 = E.sum()
                ts.append(t)
                cs.append(((E * X).sum() / E.sum(), (E * Y).sum() / E.sum()))
            cs = np.array(cs)
            sel = np.array(ts) >= 100
            vx = np.polyfit(np.array(ts)[sel], cs[sel, 0], 1)[0]
            vy = np.polyfit(np.array(ts)[sel], cs[sel, 1], 1)[0]
            vpred = math.hypot(*pi_['vg_mean'])
            vcen = math.hypot(*pi_['vg'])
            vmeas = math.hypot(vx, vy)
            dirm = math.degrees(math.atan2(vy, vx))
            back = E[z < 0.5 * vpred * 400].sum() / E.sum()
            tab.append((name, ang, vmeas, vpred, dirm - ang, back, vcen))
            print(f'    {name} {ang:2d} deg: v_g measured {vmeas:.4f} predicted {vpred:.4f} (central K {vcen:.4f}) '
                  f'({100 * (vmeas / vpred - 1):+.2f}%), direction err {dirm - ang:+.2f} deg, backward residue {back:.2e}')
    # the naive alternative: the same packet started at rest (u_prev = u0) splits in two
    m = make('sq', 5, src='packet', pk_lambda=16, pk_len=12, beam_width=16)
    u0 = m.level(1)
    m.set_levels(u0, u0)
    m.step(400)
    u, up = m.level(1).astype(np.float64), m.level(0).astype(np.float64)
    E = envelope(u, up, m.packet_info['omega'])
    X, Y = geom(m)
    rest = E[X < 0.5 * math.hypot(*m.packet_info['vg']) * 400].sum() / E.sum()
    print(f'    reference: the same packet started at rest leaves {rest:.3f} behind')
    RES['tables']['packet'] = tab + [('rest-start sq5', 0, None, None, None, rest)]
    ok('packet speed within 0.3% of the discrete prediction (spectrum-weighted v_g)', max(abs(t[2] / t[3] - 1) for t in tab) < 0.003,
       f'worst {100 * max(abs(t[2] / t[3] - 1) for t in tab):.2f}%')
    ok('packet backward residue < 1e-3 of its energy', max(t[5] for t in tab) < 1e-3,
       f'worst {max(t[5] for t in tab):.1e} (at-rest start: {rest:.2f})')
    ok('packet direction within 0.5 deg', max(abs(t[4]) for t in tab) < 0.5, f'worst {max(abs(t[4]) for t in tab):.2f} deg')


def t_stability():
    print('stability at Co = 0.99 C_max (3000 steps, N 257)')
    for name, lat, st in LATS:
        for src in ('beam', 'packet'):
            for bc in (('dirichlet', 'mur') if lat == 'sq' else ('dirichlet',)):
                m = make(lat, st, N=257, src=src, freq=0.03, bc=bc, beam_angle=37, pk_lambda=10, beam_width=6)
                m.set_co(0.99 * m.co_limit())
                m.reseed()
                mx = []
                for _ in range(6):
                    m.step(500)
                    mx.append(float(np.abs(m.field()).max()))
                fin = all(np.isfinite(v) for v in mx)
                ok(f'{name} {src} {bc} Co {m.p.Co:.4f}: finite, bounded', fin and max(mx) < 200,
                   'peak per 500 steps ' + ' '.join(f'{v:.2f}' for v in mx))


def t_obstacles():
    import solids_MIRROR as so
    print('obstacles: a solid in the path; an aperture crossing the domain wall or a solid')
    for name, lat, st in LATS:
        base = run_beam_intensity(lat, st, 12, 0, 8, True, T=600, N=401)
        m0, I0 = base
        sc = so.SolidScene()
        sc.add_rect(0.5, 0.0, 0.05, 0.5)              # grid units: a 10 x 100 cell slab, 100 cells ahead
        m = make(lat, st, N=401, freq=0.5 / 12)
        m.attach_scene(sc)
        m.reseed()
        P = int(round(1 / m.p.freq))
        I = mean_sq(m, 600, P)
        X, Y = geom(m)
        b0, b1 = I0[X < -3].sum(), I[X < -3].sum()
        solid_ok = bool((np.abs(m.field())[m.solid_np.astype(bool)] == 0).all())
        ok(f'{name}: a pinned slab ahead reflects (backward energy x{b1 / max(b0, 1e-300):.0f}), solid stays 0',
           b1 > 50 * b0 and solid_ok and np.isfinite(I).all())
        # aperture across the circle's wall: source near the edge, beam along the wall's tangent
        m = make(lat, st, N=257, shape='circle', sx=0.9, sy=0.0, beam_angle=90, beam_width=10)
        m.step(400)
        inf = m.beam['info']
        cells = m.beam['cells']
        inside = bool(all(m.mask_np[i, j] for i, j in cells))
        ok(f'{name}: aperture crossing the circle wall is clipped ({inf["n_clipped"]} cells dropped), no crash',
           inf['n_clipped'] > 0 and inside and np.isfinite(m.field()).all())
        # aperture through a solid
        sc = so.SolidScene()
        sc.add_circle(0.0, 0.08, 0.04)
        m = make(lat, st, N=257, beam_angle=0, beam_width=10)
        m.attach_scene(sc)
        m.reseed()
        m.step(300)
        cells = m.beam['cells']
        inside = bool(all(m.mask_np[i, j] for i, j in cells))
        sol = m.solid_np.astype(bool)
        ok(f'{name}: aperture through a solid skips its cells ({m.beam["info"]["n_clipped"]} dropped), solid stays 0',
           inside and m.beam['info']['n_clipped'] > 0 and (np.abs(m.field())[sol] == 0).all())


def t_cost():
    import taichi as ti
    print('cost per step (ms), record off / on')
    tab = []
    for N in (1025, 2049):
        for name, lat, st in (('sq5', 'sq', 5), ('tri', 'tri', 5)):
            for src, w in (('pulse', 8), ('cont', 8), ('beam', 8), ('beam', 64), ('packet', 8)):
                row = [N, name, src, w]
                for rec in (False, True):
                    m = make(lat, st, N=N, src=src, beam_width=w, freq=0.04)
                    m.record = rec
                    m.step(64)
                    ti.sync()
                    t = time.perf_counter()
                    m.step(800)
                    ti.sync()
                    row.append(1e3 * (time.perf_counter() - t) / 800)
                nb = m.nbeam if src == 'beam' else ''
                tab.append(row + [nb])
                print(f'    N {N} {name} {src:6s} w {w:2d} cells {nb!s:>4}: {row[4]:.3f} / {row[5]:.3f}')
    RES['tables']['cost'] = tab


def t_api():
    print('API: types in the cycle, existing defaults unchanged, two-sided, rebuild on Co change')
    ok('beam and packet are in SOURCES (after the web app\'s six)', mc.SOURCES[:6] == ['impulse', 'pulse', 'cont', 'slit', 'mode', 'vtx']
       and mc.SOURCES[6:] == ['beam', 'packet'])
    m = make('sq', 5, N=201)
    n1 = m.nbeam
    m.p.beam_onesided = False
    m.step(1)
    ok('two-sided drives only the front aperture', m.nbeam == m.beam['info']['n_front'] and m.nbeam < n1,
       f'{n1} -> {m.nbeam}')
    k1 = m.beam['info']['k'][0]
    m.set_co(0.6)
    m.step(1)
    ok('Co change rebuilds the beam tables (k follows the dispersion)', m.beam['info']['k'][0] != k1)
    g = m.beam_geometry()
    ok('beam_geometry gives source, aperture ends, arrow tip', g is not None and len(g) == 4)
    m.p.src = 'pulse'
    ok('beam_geometry is None for point sources', m.beam_geometry() is None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='cuda')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--json', default=os.path.join(HERE, 'ScratchCLAUDE', 'BeamCLAUDE', 'beam_measure_MIRROR.json'))
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    mc.init_taichi(a.arch, quiet=True)
    t0 = time.time()
    tests = [('api', t_api), ('directivity', t_directivity), ('width', t_width),
             ('steering', lambda: t_steering(a.quick)), ('packet', t_packet), ('stability', t_stability),
             ('obstacles', t_obstacles), ('cost', t_cost)]
    for nm, fn in tests:
        if a.only and nm not in a.only.split(','):
            continue
        fn()
    npass = sum(1 for c in RES['checks'] if c[1])
    print(f'\n{npass}/{len(RES["checks"])} checks pass ({time.time() - t0:.0f} s)')
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(RES, f, indent=1, default=float)
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
