"""Draft of the port-only section of crosscheck_ledger_MIRROR.py (solids, the 'unlicensed' plate,
the recorder, the geometry cache). The web app has no case for any of these, so there is nothing
to compare against: each check states the expected status from the row's premise."""
import os, sys, math
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import membrane_core_MIRROR as mc
import ledger_MIRROR as lg

D = math.pi / 180


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


if __name__ == '__main__':
    mc.init_taichi(sys.argv[1] if len(sys.argv) > 1 else 'cuda', fp64='f32' not in sys.argv, quiet=True)
    res = port_checks(verbose='-v' in sys.argv)
    for name, ok, msg in res:
        print(f'{"ok  " if ok else "FAIL"} {name}: {msg}')
    print(f'{sum(ok for _, ok, _ in res)}/{len(res)} port-only checks pass')
