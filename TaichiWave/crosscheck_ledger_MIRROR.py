"""Cross-check: the port's physics ledger (ledger_MIRROR.py) against the web app's OWN buildLedger().

For each configuration the web app's module script runs headlessly under Node
(ToolsCLAUDE/js_ledger_dump_MIRROR.mjs, through js_harness_MIRROR.mjs WITH its documented PATCHES --
the Neumann corner fix of 29/09/2026 -- as crosscheck_js_MIRROR.py does), steps to each checkpoint
(dropping pulses where the config says), calls buildLedger() and keeps the rows it writes into
#led. The port runs the same configuration to the same checkpoints and builds its ledger twice:
  compat=True   the web app's exact logic -- must agree row by row:
                  the row count and order, every label, the status badge EXACTLY, every tier tag;
                  the detail text word for word, with its numbers compared as follows:
                    - equal strings agree;
                    - otherwise |a - b| <= one unit of the last printed digit (a value computed from
                      fields that differ in the last bits can round to the neighbouring display value),
                      or |a - b| <= 1e-6 |a| (sums in another order);
                    - both |a|, |b| < 1e-13: machine-noise residuals (8-fold / mirror / 6-fold
                      asymmetry ~1e-17..1e-15, the dispersion recurrence's |cos - cos|): only the
                      noise floor is reproducible, not the noise itself (the seeds differ by one ulp:
                      numpy exp vs V8 Math.exp -- crosscheck_js_MIRROR.py).
  compat=False  with the FIXES: every row that differs from the web app must name the fix that
                changed it (Row.fixes); anything else is an unexplained mismatch.
The module state the rows read (step, S.amax, CLEAR, CONE_WALL, the source cell) is compared too.

PORT-ONLY section (after the comparison; --port-only runs it alone, --no-port skips it): solids, the
'unlicensed' plate, the recorder switched off, the geometry cache under incremental painting. The web
app has no case for these, so each check states the status the row's premise implies (a symmetric
arrangement of solids keeps the orbit rows graded; an asymmetric one makes them n/a with the reason;
no row may FAIL in these runs; removing every solid gives the never-solid ledger exactly; ...).

    python crosscheck_ledger_MIRROR.py [--arch cuda|cpu|vulkan] [--f32] [--only name,name] [--quick]
                                       [--jsonl saved_web_app_dump.jsonl] [--port-only | --no-port]
"""
import argparse
import html
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import membrane_core_MIRROR as mc          # noqa: E402
import ledger_MIRROR as lg                  # noqa: E402

D = math.pi / 180
CP = (0, 50, 300, 1200)


def configs():
    C = []

    def add(name, cps=CP, drops=(), same_seed=False, noise=None, **p):
        # noise: 'always' / 'inexact' -- a run past the CFL limit, whose numbers are amplified rounding
        # noise: from the port's own seed ('always'), or only off CPU f64 ('inexact': an impulse seed
        # or the web app's own seed, which CPU f64 steps bit-identically)
        C.append({'name': name, 'checkpoints': list(cps), 'drops': list(drops), 'p': p, 'same_seed': same_seed,
                  'noise': noise, 'chaotic': noise == 'always'})
    # square lattice, full grid: the four boundaries, centred and offset sources
    add('sq5_full_dir_pulse', N=161, src='pulse')
    add('sq5_full_dir_impulse', N=161, src='impulse')
    add('sq5_full_neu_impulse', N=161, src='impulse', bc='neumann')
    add('sq5_full_mur_pulse', N=161, src='pulse', bc='mur')
    add('sq5_full_mur_impulse_off', N=161, src='impulse', bc='mur', sx=0.5, sy=-0.3, Co=0.6)
    add('sq5_full_per_pulse', N=161, src='pulse', bc='periodic')
    add('sq5_full_per_impulse', N=161, src='impulse', bc='periodic')
    add('sq9_full_dir_impulse', N=161, src='impulse', stencil=9, Co=0.8)
    add('sq9_full_neu_pulse_corner', N=161, src='pulse', bc='neumann', stencil=9, sx=-0.8, sy=-0.8, sigma=2)
    add('sq9_full_mur_pulse', N=161, src='pulse', bc='mur', stencil=9)
    add('sq5_full_mur_pulse_s2_off', N=161, src='pulse', bc='mur', sigma=2, sx=0.25)
    add('sq5_full_dir_drops', N=161, src='pulse', drops=(100,))
    add('sq5_full_mur_cont', N=161, src='cont', bc='mur', freq=0.05)
    # drum modes: the eigenmode and dispersion rows
    add('sq5_full_mode11', N=161, src='mode')
    add('sq9_full_mode32', N=161, src='mode', stencil=9, mm=3, mn=2)
    add('sq5_sqr07_mode25', N=161, src='mode', rad=0.7, mm=2, mn=5)
    add('sq5_full_mode73_damp', N=161, src='mode', mm=7, mn=3, damp=0.001)
    add('sq5_full_mode_slab', N=161, src='mode', medium='slab')
    add('sq5_full_mode_neu', N=161, src='mode', bc='neumann')
    # masked shapes, media, sources
    add('sq5_circle_lens', N=161, shape='circle', rad=0.8, medium='lens', sigma=2.5)
    add('sq5_circle_impulse', N=161, shape='circle', rad=0.9, src='impulse')
    add('sq5_n8_pulse', N=161, shape='n8', rot=22.5 * D)
    add('sq9_n4_pulse', N=161, shape='n4', stencil=9, rad=0.8)
    add('sq5_n12_mur_cont_tri', N=161, shape='n12', bc='mur', src='cont', wave='tri', freq=0.04)
    add('sq5_n6_slab_cont', N=161, shape='n6', medium='slab', src='cont', freq=0.08, sx=-0.2)
    add('sq5_n6_mur_pulse', N=161, shape='n6', bc='mur')
    add('sq5_n6_vtx_rot0', N=161, shape='n6', src='vtx')
    add('sq5_n5_vtxwall_clip', N=161, shape='n5', rot=17.3 * D, rad=1.2, src='vtx', vtx_inset='wall')
    add('sq5_n5_vtx_rot0', N=161, shape='n5', src='vtx')
    add('sq5_n3_pulse_x', N=161, shape='n3', sx=0.3)
    add('sq5_n4_slit_square_damp', N=161, shape='n4', rot=45 * D, src='slit', wave='square', freq=0.05, damp=0.01)
    add('sq5_n8_vtxdrive_saw', N=161, shape='n8', rot=22.5 * D, rad=1.4, src='vtx', vtx_drive=True,
        vtx_inset='wall', wave='saw', freq=0.03, damp=0.002)
    add('sq5_rhomb_neu', N=161, shape='rhomb', rtheta=40 * D, rot=17 * D, bc='neumann', sy=0.4)
    add('sq5_rhomb_rot0_impulse', N=161, shape='rhomb', rtheta=50 * D, src='impulse')
    add('sq5_sqr05_neu_pulse', N=161, shape='square', rad=0.5, bc='neumann', sx=0.6, sy=0.6)
    # past the CFL limit
    add('sq5_unstable_Co075', N=161, src='pulse', Co=0.75, noise='always')
    # past the limit the checkerboard grows ~6.9x per step FROM ROUNDING NOISE, so the port's own
    # seed (numpy exp, one ulp from V8's Math.exp) gives other numbers; started from the web app's
    # seed the CPU f64 stepping is bit-identical (crosscheck_js_MIRROR.py), so the ledger must be too
    add('sq5_unstable_Co075_js_seed', N=161, src='pulse', Co=0.75, same_seed=True, noise='inexact')
    add('sq9_Co09_over', N=161, src='impulse', stencil=9, Co=0.9, cps=(0, 50, 300), noise='inexact')
    # triangular lattice (always masked, clamped ring)
    add('tri_n6_conf_pulse', N=161, lattice='tri', shape='n6', rot=30 * D, Co=0.8)
    add('tri_n3_conf_impulse', N=161, lattice='tri', shape='n3', rot=30 * D, src='impulse')
    add('tri_rhomb60_imp', N=161, lattice='tri', shape='rhomb', rot=30 * D, rad=0.8, src='impulse')
    add('tri_circle_pulse', N=161, lattice='tri', shape='circle', rad=0.9)
    add('tri_circle_lens_off', N=161, lattice='tri', shape='circle', rad=1.3, medium='lens', sx=0.5)
    add('tri_n12_cont_saw', N=161, lattice='tri', shape='n12', src='cont', wave='saw', freq=0.05)
    add('tri_n5_slit', N=161, lattice='tri', shape='n5', rot=12 * D, src='slit', freq=0.06)
    add('tri_square_slab_vtx', N=161, lattice='tri', shape='square', rad=0.9, medium='slab', src='vtx', sigma=1.5)
    add('tri_n6_vtxwall_conf', N=161, lattice='tri', shape='n6', rot=30 * D, src='vtx', vtx_inset='wall')
    add('tri_n6_rot0_mur', N=161, lattice='tri', shape='n6', bc='mur')
    add('tri_Co085_over', N=161, lattice='tri', shape='n6', rot=30 * D, Co=0.85, src='impulse', cps=(0, 50, 300),
        noise='inexact')
    # another size
    add('sq5_n6_N257', N=257, shape='n6', rot=10 * D, sigma=4, cps=(0, 50, 300))
    return C


def js_keys(p):
    ren = {'vtx_drive': 'vtxDrive', 'vtx_inset': 'vtxInset'}
    return {ren.get(k, k): v for k, v in p.items()}


# ---- the web app's HTML -> the port's plain text ----------------------------------------------
_TAG = re.compile(r'<[^>]+>')


def html_text(h):
    return html.unescape(_TAG.sub('', h)).replace('\xa0', ' ')


def js_row(r):
    label, status, detail, tier = r
    return {'name': html_text(label), 'dsg': '<span class="dsg">' in label, 'badge': html_text(status),
            'note': html_text(detail), 'tag': html_text(tier)}


_NUM = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:e[-+]?\d+)?')


def _unit(tok):
    """One unit of the last printed digit of a number token."""
    mant, _, ex = tok.partition('e')
    d = len(mant.split('.')[1]) if '.' in mant else 0
    return 10.0 ** (-d) * (10.0 ** int(ex) if ex else 1.0)


def text_agree(a, b, floor=1e-13, rtol=1e-6):
    """(True, '') if a and b agree word for word with their numbers within tolerance, else
    (False, the first difference)."""
    if a == b:
        return True, ''
    ta, tb = _NUM.split(a), _NUM.split(b)
    na, nb = _NUM.findall(a), _NUM.findall(b)
    if len(na) != len(nb) or ta != tb:
        for i, (x, y) in enumerate(zip(ta, tb)):
            if x != y:
                return False, f'text: {x!r} vs {y!r}'
        return False, f'{len(na)} vs {len(nb)} numbers'
    for x, y in zip(na, nb):
        if x == y:
            continue
        fx, fy = float(x), float(y)
        if abs(fx) < floor and abs(fy) < floor:
            continue
        u = max(_unit(x), _unit(y))
        if abs(fx - fy) <= 1.0000001 * u or abs(fx - fy) <= rtol * max(abs(fx), abs(fy)):
            continue
        return False, f'number: {x} vs {y}'
    return True, ''


def row_diff(j, p, floor=1e-13, rtol=1e-6):
    """Differences between a web-app row j (dict) and a port Row p: a list of strings."""
    out = []
    if j['badge'] != p.badge:
        out.append(f'status {j["badge"]!r} vs {p.badge!r}')
    ok, why = text_agree(j['name'], p.name, floor, rtol)
    if not ok:
        out.append('label ' + why)
    if j['dsg'] != p.dsg:
        out.append(f'dsg {j["dsg"]} vs {p.dsg}')
    ok, why = text_agree(j['note'], p.note, floor, rtol)
    if not ok:
        out.append('detail ' + why)
    if j['tag'] != p.tag:
        out.append(f'tag {j["tag"]!r} vs {p.tag!r}')
    return out


def visible_same(a, b):
    """The rows show the same thing: label, status, detail, tier, style."""
    return (a.name, a.badge, a.note, a.tag, a.dsg) == (b.name, b.badge, b.note, b.tag, b.dsg)


def run_port(c, seed_dir=None):
    """The port through the same checkpoints: {checkpoint: (rows_compat, rows_fixed, dbg, ms)}."""
    base = mc.Params()
    m = mc.Membrane(mc.Params(**{**base.__dict__, **c['p']}))
    if c.get('same_seed') and seed_dir:
        n = m.N
        u0 = np.fromfile(os.path.join(seed_dir, c['name'] + '_u0_MIRROR.bin'), dtype=np.float64).reshape(n, n).T
        m.seed()
        m.set_levels(u0, u0)
    drops, done, step, out = set(c['drops']), set(), 0, {}
    for cp in sorted(c['checkpoints']):
        while step < cp:
            if step in drops and step not in done:
                m.drop_pulse()
                done.add(step)
            nxt = min([cp] + [d for d in drops if step < d <= cp])
            m.step(nxt - step)
            step = nxt
        if step in drops and step not in done:
            m.drop_pulse()
            done.add(step)
        t0 = time.perf_counter()
        rc = lg.build_ledger(m, compat=True)
        ms = (time.perf_counter() - t0) * 1e3
        rf = lg.build_ledger(m, compat=False)
        G = lg._geo(m)
        dbg = {'step': m.step_n, 'amax': m.amax, 'drops': m.drops, 'CLEAR': lg._clear(m, G),
               'CONE_WALL': lg._cone_wall(m, G), 'SRC': [int(m.src[0]), int(m.src[1])],
               'nE': len(m.hist['E']), 'nA': len(m.hist['pA']), 'nM': len(m.hist['mode'])}
        out[cp] = (rc, rf, dbg, ms)
    return out


def dbg_diff(jd, pd, rtol=1e-9):
    out = []
    for k in ('step', 'drops', 'CONE_WALL', 'SRC', 'nE', 'nA', 'nM'):
        if jd.get(k) != pd.get(k):
            out.append(f'{k} {jd.get(k)} vs {pd.get(k)}')
    for k in ('CLEAR', 'amax'):
        a, b = jd.get(k), pd.get(k)
        a = float(a) if a is not None else math.nan
        tol = rtol if k == 'amax' else 1e-9
        if not (a == b or (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol * max(abs(a), abs(b), 1e-300)):
            out.append(f'{k} {a!r} vs {b!r}')
    return out


# ---- port-only checks (30/09/2026) ----------------------------------------------------------------
def _disks(n, centres, r2):
    c = (n - 1) // 2
    I, J = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing='ij')
    S = np.zeros((n, n), np.uint8)
    for a, b in centres:
        S |= ((I - a) ** 2 + (J - b) ** 2 <= r2)
    return S


def port_checks(verbose=False):
    """[(name, ok, message)]."""
    out = []
    N = 161
    SOLN = 'solid objects are present'

    def by(rows):
        return {r.key: r for r in rows}

    def expect(name, rows, want, no_fail=True):
        R = by(rows)
        bad = []
        for key, (status, sub) in want.items():
            r = R.get(key)
            if r is None:
                bad.append(f'{key}: missing')
                continue
            if r.status != status:
                bad.append(f'{key}: {r.status} (want {status}) -- {r.note[:140]}')
            elif sub and sub not in r.note and sub not in r.port:
                bad.append(f'{key}: note lacks {sub!r}: {r.note[:140]}')
        if no_fail:
            bad += [f'{r.key}: unexpected FAIL -- {r.note[:140]}' for r in rows
                    if r.status == lg.FAIL and r.key not in want]
        out.append((name, not bad, '; '.join(bad) if bad else 'ok'))
        if verbose:
            print(f'--- {name}')
            print(lg.format_rows(rows, notes=False))

    def mk(steps=1300, solid=None, **kw):
        m = mc.Membrane(mc.Params(N=N, **kw))
        if solid is not None:
            m.set_solid_np(solid)
            m.reseed()
        m.step(steps)
        return m

    # A. an 8-fold symmetric arrangement: the orbit rows stay graded (a solver test with solids)
    sym = _disks(N, [(30, 0), (0, 30), (-30, 0), (0, -30), (22, 22), (-22, 22), (22, -22), (-22, -22)], 36)
    m = mk(solid=sym, src='pulse')
    expect('8-fold symmetric solids, centred pulse', lg.build_ledger(m),
           {'sym8': (lg.PASS, ''), 'mirror': (lg.PASS, ''), 'energy': (lg.PASS, 'pinned'),
            'spectra': (lg.NA, SOLN), 'bc': (lg.NA, SOLN), 'stair': (lg.PASS, 'DOMAIN mask')})
    # B. an asymmetric block: the orbit rows are n/a with the reason, never a false FAIL
    blk = np.zeros((N, N), np.uint8)
    blk[100:110, 40:70] = 1
    m = mk(solid=blk, src='pulse')
    expect('asymmetric block, centred pulse', lg.build_ledger(m),
           {'sym8': (lg.NA, 'not 8-fold symmetric'), 'mirror': (lg.NA, 'not mirror-symmetric'),
            'energy': (lg.PASS, 'pinned'), 'bc': (lg.NA, SOLN), 'spectra': (lg.NA, SOLN)})
    # C. a drum mode with an object inside: sin.sin is no longer an eigenmode
    m = mk(solid=blk, src='mode', mm=3, mn=2)
    expect('asymmetric block, mode (3,2)', lg.build_ledger(m),
           {'mode': (lg.NA, 'not an eigenmode'), 'disp': (lg.NA, 'not an eigenmode'), 'spectra': (lg.NA, SOLN)})
    # D. a solid over the centre: the source snaps out, off the orbit's fixed point
    m = mk(solid=_disks(N, [(0, 0)], 9), src='pulse')
    expect('solid over the centre (source snaps out)', lg.build_ledger(m),
           {'sym8': (lg.NA, 'snapped'), 'energy': (lg.PASS, '')})
    # E. triangular hexagon, impulse, one circle ON the x axis: mirror graded, 6-fold n/a, cone graded
    m = mc.Membrane(mc.Params(N=N, lattice='tri', shape='n6', rot=30 * D, src='impulse'))
    sc = m.ensure_scene()
    sc.add_circle(0.5, 0.0, 0.08)
    m.update_solids(None)
    m.reseed()
    m.step(1300)
    expect('tri hexagon, impulse, circle on the x axis', lg.build_ledger(m),
           {'sym6': (lg.NA, 'not 6-fold'), 'mirror': (lg.PASS, ''), 'cone': (lg.PASS, 'solids'),
            'conform': (lg.PASS, 'DOMAIN mask'), 'energy': (lg.PASS, '')})
    # F. centred pulse, Mur, solids: the wake row cannot isolate the free-space tail
    m = mk(solid=blk, src='pulse', bc='mur', steps=600)
    expect('Mur + block, centred pulse (wake)', lg.build_ledger(m),
           {'wake': (lg.NA, SOLN), 'bc': (lg.NA, SOLN), 'energy': (lg.NA, '')})
    # G. a driven source with solids (linearity stays graded: pinned walls are linear)
    m = mk(solid=blk, src='cont', bc='mur', freq=0.05, steps=1300)
    expect('Mur + block, continuous drive (linearity)', lg.build_ledger(m), {})
    # H. the Unlicensed plate with its lettering and screws
    m = mc.Membrane(mc.Params(N=N, shape='unlicensed', src='pulse', sy=-0.9))
    m.step(1300)
    expect('unlicensed plate (lettering), pulse sy=-0.9', lg.build_ledger(m),
           {'sym8': (lg.NA, ''), 'mirror': (lg.NA, 'lettering'), 'stair': (lg.STATED, 'closed form'),
            'spectra': (lg.NA, SOLN), 'energy': (lg.PASS, 'pinned')})
    # I. the bare plate (preset off): mirror-symmetric, graded
    m = mc.Membrane(mc.Params(N=N, shape='unlicensed', src='pulse'))
    m.scene.set_preset(plate_text=False, plate_screws=False)
    m.update_solids(None)
    m.reseed()
    m.step(1300)
    rows = lg.build_ledger(m)
    expect('bare unlicensed plate, centred pulse', rows,
           {'mirror': (lg.PASS, ''), 'spectra': (lg.STATED, ''), 'bc': (lg.PASS, ''), 'energy': (lg.PASS, '')})
    # J. the bare plate rotated 30 deg: not mirror-symmetric about x
    m = mc.Membrane(mc.Params(N=N, shape='unlicensed', src='pulse', rot=30 * D))
    m.scene.set_preset(plate_text=False, plate_screws=False)
    m.update_solids(None)
    m.reseed()
    m.step(300)
    expect('bare plate rotated 30 deg', lg.build_ledger(m), {'mirror': (lg.NA, 'not mirror-symmetric')})
    # K. the plate on the triangular lattice (radius 0.89 fits), impulse
    m = mc.Membrane(mc.Params(N=N, lattice='tri', shape='unlicensed', rad=0.89, src='impulse', sy=-0.9))
    m.step(300)
    expect('unlicensed plate, tri lattice r=0.89, impulse', lg.build_ledger(m), {'cone': (lg.PASS, '')})
    # L. removing every solid returns the exact web-app path: the ledger equals a never-solid run's
    a = mk(src='impulse', steps=300)
    b = mc.Membrane(mc.Params(N=N, src='impulse'))
    b.set_solid_np(blk)
    b.set_solid_np(np.zeros((N, N), np.uint8))
    b.reseed()
    b.step(300)
    ra, rb = lg.build_ledger(a), lg.build_ledger(b)
    same = ra == rb
    out.append(('solids removed -> identical ledger', same, 'ok' if same else
                '; '.join(f'{x.key}: {x.note[:80]} vs {y.note[:80]}' for x, y in zip(ra, rb) if x != y)))
    # M. the geometry cache follows incremental solid edits (update_solids with a box, in place)
    m = mc.Membrane(mc.Params(N=N, src='pulse'))
    m.step(20)
    r0 = lg.build_ledger(m)
    sc = m.ensure_scene()
    bb = sc.begin_stroke(0.3, 0.1, 0.02)
    m.update_solids(bb)
    m.update_solids(sc.extend_stroke(0.3, 0.3))
    sc.end_stroke()
    r1 = lg.build_ledger(m)
    lg._GEO.clear()
    r2 = lg.build_ledger(m)
    ok = r1 == r2 and by(r1)['sym8'].status == lg.NA and by(r0)['sym8'].status != lg.NA
    out.append(('geometry cache follows a painted stroke', ok,
                'ok' if ok else f'sym8 {by(r0)["sym8"].status} -> {by(r1)["sym8"].status}; cached == fresh: {r1 == r2}'))
    # N. the recorder switched off for part of the run: history rows are n/a, not graded on a stale prefix
    m = mc.Membrane(mc.Params(N=N, src='pulse'))
    m.step(600)
    m.record = False
    m.step(700)
    m.record = True
    expect('recorder off for part of the run', lg.build_ledger(m),
           {'energy': (lg.NA, 'recorder'), 'bc': (lg.NA, 'recorder'), 'cfl': (lg.PASS, '')})
    m = mc.Membrane(mc.Params(N=N, src='mode', mm=2, mn=3))
    m.record = False
    m.step(1300)
    expect('recorder off from the start, mode (2,3)', lg.build_ledger(m),
           {'energy': (lg.NA, 'recorder'), 'disp': (lg.NA, 'recorder'), 'mode': (lg.PASS, '')})
    return out


def run_port_checks(verbose=False):
    res = port_checks(verbose)
    print('\n==== port-only checks (solids, the unlicensed plate, the recorder, the cache) ====')
    for name, ok, msg in res:
        print(f'{"ok  " if ok else "FAIL"} {name}: {msg}')
    n_ok = sum(ok for _, ok, _ in res)
    print(f'{n_ok}/{len(res)} port-only checks pass')
    return n_ok == len(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='cuda')
    ap.add_argument('--f32', action='store_true')
    ap.add_argument('--only', default='')
    ap.add_argument('--quick', action='store_true', help='checkpoints capped at 300 steps')
    ap.add_argument('--jsonl', default='', help='reuse a previous web-app dump (skip Node)')
    ap.add_argument('--app', default=None)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--port-only', action='store_true', help='only the port-only checks (no web app)')
    ap.add_argument('--no-port', action='store_true', help='skip the port-only checks')
    a = ap.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if a.port_only:
        mc.init_taichi(a.arch, fp64=not a.f32, quiet=True)
        ok = run_port_checks(a.verbose)
        print('RESULT: ' + ('PASS' if ok else 'FAIL') + ' (port-only)')
        return 0 if ok else 1

    C = configs()
    if a.only:
        keep = set(a.only.split(','))
        C = [c for c in C if c['name'] in keep]
    if a.quick:
        for c in C:
            c['checkpoints'] = [k for k in c['checkpoints'] if k <= 300]
    base = mc.Params()
    if a.jsonl:
        out_path = a.jsonl
        tmp = os.path.dirname(os.path.abspath(a.jsonl))
    else:
        tmp = tempfile.mkdtemp(prefix='zrl_ledger_')
        jc = [{'name': c['name'], 'checkpoints': c['checkpoints'], 'drops': c['drops'],
               'seed_out': os.path.join(tmp, c['name'] + '_u0_MIRROR.bin') if c['same_seed'] else None,
               'p': js_keys({**base.__dict__, **c['p']})} for c in C]
        cfg_path = os.path.join(tmp, 'configs.json')
        with open(cfg_path, 'w') as f:
            json.dump(jc, f)
        out_path = os.path.join(tmp, 'ledger_js.jsonl')
        cmd = ['node', os.path.join(HERE, 'ToolsCLAUDE', 'js_ledger_dump_MIRROR.mjs'), cfg_path, out_path]
        if a.app:
            cmd.append(a.app)
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        print(f'web app (Node) ran {len(C)} configs in {time.perf_counter() - t0:.1f} s  ({out_path})')
    J = {}
    for line in open(out_path, encoding='utf-8'):
        d = json.loads(line)
        J[(d['name'], d['checkpoint'])] = d
    patches = next(iter(J.values()))['patches'] if J else []
    print(f'patches applied to the web app: {", ".join(patches) if patches else "NONE (ZRL_JS_NOPATCH=1)"}')

    mc.init_taichi(a.arch, fp64=not a.f32, quiet=True)
    # only CPU f64 steps bit-identically to the web app: CUDA fuses a*b + c into one rounding (FMA)
    # and f32 rounds ~1e9x coarser, so there the js_seed twin of the unstable run is noise-driven too
    exact = a.arch == 'cpu' and not a.f32
    for c in C:
        if c['noise'] == 'inexact' and not exact:
            c['chaotic'] = True
    # f32: the numbers of this run are f32-rounded (residuals ~1e-7 where f64 gives ~1e-17), so the
    # text is compared with an f32 noise floor and a looser relative tolerance; statuses stay exact
    floor, rtol = (1e-4, 1e-3) if a.f32 else (1e-13, 1e-6)
    worst = {}
    n_rows = n_status = n_text = n_led = 0
    n_fixdiff = n_unexpl = 0
    bad_dbg = 0
    fixes_seen = {}
    status_fail = []
    text_fail = []
    chaotic = []
    spurious = []
    f32_unrep = []
    f32_text = []
    unexpl = []
    t_port = []
    for c in C:
        R = run_port(c, tmp)
        for cp in sorted(c['checkpoints']):
            jd = J.get((c['name'], cp))
            if jd is None:
                print(f'MISSING web-app ledger {c["name"]} @ {cp}')
                continue
            rc, rf, pdbg, ms = R[cp]
            t_port.append(ms)
            n_led += 1
            jr = [js_row(r) for r in jd['rows']]
            dd = dbg_diff(jd['dbg'], pdbg, 1e-3 if a.f32 else 1e-9)
            # f32 cannot hold the web app's value: a run past the CFL limit reaches 1e72 by step 300 in
            # f64, but overflows f32 (3.4e38) and then turns to inf/NaN -- the rows it feeds cannot agree
            j_amax = jd['dbg'].get('amax')
            unrep = a.f32 and isinstance(j_amax, (int, float)) and abs(j_amax) > 3.4e38
            if dd:
                bad_dbg += 1
                print(f'  state {c["name"]} @ {cp}: ' + '; '.join(dd))
            if len(jr) != len(rc):
                print(f'FAIL {c["name"]} @ {cp}: {len(jr)} web-app rows vs {len(rc)} port rows')
                status_fail.append((c['name'], cp, 'row count'))
                continue
            for j, p, pf in zip(jr, rc, rf):
                n_rows += 1
                diffs = row_diff(j, p, floor, rtol)
                if p.key in ('sym8', 'sym6', 'mirror', 'disp') and p.num is not None and p.status == lg.PASS:
                    if p.num > worst.get(p.key, (-1,))[0]:
                        worst[p.key] = (p.num, f'{c["name"]}@{cp}')
                st_ok = j['badge'] == p.badge
                if not st_ok and unrep:
                    f32_unrep.append((c['name'], cp, p.key, diffs))
                    print(f'  expected (beyond f32 range) {c["name"]} @ {cp} [{p.key}]: ' + '; '.join(diffs))
                    n_status += 1
                    n_text += 1
                    continue
                n_status += st_ok
                n_text += not diffs
                if not st_ok:
                    status_fail.append((c['name'], cp, p.key))
                if diffs and a.f32 and st_ok:
                    n_text += 1
                    f32_text.append((c['name'], cp, p.key, diffs))
                    print(f'  f32 text (informational) {c["name"]} @ {cp} [{p.key}]: ' + '; '.join(diffs))
                elif diffs and c['chaotic'] and st_ok:
                    n_text += 1
                    chaotic.append((c['name'], cp, p.key, diffs))
                    print(f'  expected (noise-amplified) {c["name"]} @ {cp} [{p.key}]: ' + '; '.join(diffs))
                elif diffs:
                    text_fail.append((c['name'], cp, p.key, diffs))
                    print(f'  DIFF {c["name"]} @ {cp} [{p.key}]: ' + '; '.join(diffs))
                    if a.verbose:
                        print(f'      web app: {j["badge"]} | {j["name"]} | {j["note"]}')
                        print(f'      port:    {p.badge} | {p.name} | {p.note}')
                # compat=False: a row the FIXES changed must name its fix; a row they did not change is
                # the compat row, already compared above
                if pf.fixes and visible_same(pf, p):
                    spurious.append((c['name'], cp, pf.key, pf.fixes))
                if not visible_same(pf, p):
                    fd = row_diff(j, pf, floor, rtol)
                    if pf.fixes:
                        n_fixdiff += 1
                        for fx in pf.fixes:
                            fixes_seen.setdefault(fx, []).append(f'{c["name"]}@{cp}[{pf.key}]: ' + '; '.join(fd))
                    else:
                        n_unexpl += 1
                        unexpl.append((c['name'], cp, pf.key, fd))
        print(f'{c["name"]}: checkpoints {",".join(map(str, c["checkpoints"]))} done')

    print('\n==== summary ====')
    print(f'{len(C)} configurations, {n_led} ledgers, {n_rows} rows (compat=True, the web app\'s logic)')
    print(f'status agreement: {n_status}/{n_rows} ({100 * n_status / max(1, n_rows):.2f}%)')
    print(f'full-row agreement (label, status, detail with numbers in tolerance, tier): '
          f'{n_text}/{n_rows} ({100 * n_text / max(1, n_rows):.2f}%)')
    print(f'  of which {len(chaotic)} rows of a run past the CFL limit agree in status only: its checkerboard grows '
          '~6.9x per step from ROUNDING NOISE (the port’s seed differs from the web app’s by one ulp: numpy exp '
          'vs Math.exp; off CPU f64, CUDA’s FMA and f32 rounding differ too); on CPU f64 the runs started from '
          'the same seed (impulse, _js_seed) must agree in full')
    if a.f32:
        print(f'  f32: {len(f32_unrep)} rows exempt (the web app’s S.amax > 3.4e38, beyond f32), {len(f32_text)} rows '
              'with the same status but f32-rounded numbers beyond the f32 text tolerance (listed above)')
    print(f'module state (step, S.amax, CLEAR, CONE_WALL, SRC, history lengths) differing: {bad_dbg} ledgers '
          '(a state line is printed for each; S.amax of the noise-driven run is expected to)')
    print(f'with FIXES (compat=False): {n_fixdiff} rows show something else than the web app’s logic, each '
          f'naming its fix in Row.fixes; {n_unexpl} changed without naming a fix')
    for fx, lst in fixes_seen.items():
        print(f'  fix {fx}: {len(lst)} rows, e.g. {lst[0][:220]}')
        if a.verbose:
            for x in lst:
                print(f'      {x[:300]}')
    print(f'rows naming a fix that changed nothing visible: {len(spurious)}' + (f', e.g. {spurious[:3]}' if spurious else ''))
    for u in unexpl[:20]:
        print(f'  UNEXPLAINED {u}')
    print('largest PASSing exact-identity residuals: ' + ', '.join(f'{k} {v[0]:.2e} ({v[1]})' for k, v in worst.items()))
    if t_port:
        t_port.sort()
        print(f'port ledger build (compat=True): median {t_port[len(t_port) // 2]:.1f} ms, max {t_port[-1]:.1f} ms '
              f'(N <= 257, arch {a.arch}{" f32" if a.f32 else ""})')
    ok = n_status == n_rows and not n_unexpl and not text_fail
    port_ok = True if a.no_port else run_port_checks(a.verbose)
    print('RESULT: ' + ('PASS' if (ok and port_ok) else 'FAIL') + f' ({len(status_fail)} status mismatches, '
          f'{len(text_fail)} text mismatches' + ('' if a.no_port else
                                                 f', port-only checks {"pass" if port_ok else "FAIL"}') + ')')
    ok = ok and port_ok
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
