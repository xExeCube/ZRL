"""Cross-check: the Taichi port against the web app's OWN solver, config by config.

For each configuration the web app's module script is run headlessly under Node
(ToolsCLAUDE/js_dump_MIRROR.mjs), and the port is run with the same parameters. Compared:
  mask          must be IDENTICAL (cell for cell)
  per-cell Co^2 max |diff|
  seeded field  max |diff|  (t = 0: source placement, gaussian, mode shape, mask)
  field at K    max |diff| / max |u|  after K steps (the whole update: stencil, BC, media,
                drive, slit, damping)
  vertex cells  identical list;  pulse clip fraction;  peak |u|
  histories     record()'s per-step ENERGY (relative), PROBE / PROBE_B / MODEP (like fields),
                their lengths (the web app's caps: 4000 / 600) and S.amax

The web app runs WITH the documented port deviations (PATCHES in ToolsCLAUDE/js_harness_MIRROR.mjs,
e.g. the Neumann corner fix of 29/09/2026). ZRL_JS_NOPATCH=1 compares against the unpatched web
app instead, to show what each patch changes.

    python crosscheck_js_MIRROR.py [--arch cuda|vulkan|cpu] [--f32] [--steps K]
"""
import argparse, json, math, os, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import membrane_core_MIRROR as mc

D = math.pi / 180


def configs(K):
    C = []
    def add(name, steps=K, **p):
        C.append({'name': name, 'steps': steps, 'p': p})
    # square lattice, full grid: the four boundary conditions
    add('sq5_full_dir_pulse', N=161, src='pulse')
    add('sq5_full_neu_impulse', N=161, src='impulse', bc='neumann')
    add('sq5_full_mur_pulse_off', N=161, src='pulse', bc='mur', sx=0.3, sy=-0.2, Co=0.6)
    add('sq5_full_per_pulse', N=161, src='pulse', bc='periodic', sx=0.5)
    add('sq9_full_per_pulse', N=161, src='pulse', bc='periodic', stencil=9, Co=0.8)
    add('sq9_full_mode32', N=161, src='mode', stencil=9, mm=3, mn=2)
    # the 9-point stencil reads the ring CORNER through its diagonal, so the Neumann corner's
    # loop-order value (two steps old at (0,0)) feeds back into the field only here
    add('sq9_full_neu_corner', N=161, src='pulse', bc='neumann', stencil=9, sx=-0.8, sy=-0.8, sigma=2)
    add('sq9_full_mur_corner', N=161, src='pulse', bc='mur', stencil=9, sx=0.8, sy=-0.8, sigma=2)
    add('sq5_sqr07_mode', N=161, src='mode', rad=0.7, mm=2, mn=5)
    # masked shapes, media, sources
    add('sq9_circle_lens', N=161, shape='circle', rad=0.8, medium='lens', stencil=9, sigma=2.5)
    add('sq5_n5_vtxwall', N=161, shape='n5', rot=17.3 * D, rad=1.2, src='vtx', vtx_inset='wall')
    add('sq5_n6_slab_cont', N=161, shape='n6', medium='slab', src='cont', freq=0.08, sx=-0.2)
    add('sq5_n4_slit_square', N=161, shape='n4', rot=45 * D, src='slit', wave='square', freq=0.05)
    add('sq5_n8_vtxdrive_saw', N=161, shape='n8', rot=22.5 * D, rad=1.4, src='vtx', vtx_drive=True,
        vtx_inset='wall', wave='saw', freq=0.03, damp=0.002)
    add('sq5_rhomb_neu', N=161, shape='rhomb', rtheta=40 * D, rot=17 * D, bc='neumann', sy=0.4)
    add('sq5_n12_mur_cont_tri', N=161, shape='n12', bc='mur', src='cont', wave='tri', freq=0.04)
    add('sq5_sqr05_neu_pulse', N=161, shape='square', rad=0.5, bc='neumann', sx=0.6, sy=0.6)
    # triangular lattice (always masked, clamped ring)
    add('tri_n6_conf_pulse', N=161, lattice='tri', shape='n6', rot=30 * D, Co=0.8)
    add('tri_n3_conf_vtxdrive', N=161, lattice='tri', shape='n3', rot=30 * D, src='vtx',
        vtx_drive=True, wave='tri', freq=0.05)
    add('tri_rhomb60_imp', N=161, lattice='tri', shape='rhomb', rot=30 * D, rad=1.5, src='impulse')
    add('tri_circle_lens_off', N=161, lattice='tri', shape='circle', rad=1.3, medium='lens', sx=0.5)
    add('tri_n12_cont_saw', N=161, lattice='tri', shape='n12', src='cont', wave='saw', freq=0.05)
    add('tri_n5_slit', N=161, lattice='tri', shape='n5', rot=12 * D, src='slit', freq=0.06)
    add('tri_square_slab_vtx', N=161, lattice='tri', shape='square', rad=0.9, medium='slab',
        src='vtx', sigma=1.5)
    # another size, and a statically unstable Co (the growth must match too, for a while)
    add('sq5_n6_N257', N=257, shape='n6', rot=10 * D, src='pulse', sigma=4)
    add('sq5_unstable_Co075', steps=60, N=161, src='pulse', Co=0.75)
    # the history caps (added 30/09/2026): ENERGY / PROBE / PROBE_B keep the newest 4000 entries,
    # MODEP 600 (its step-0 seed value long gone); a small grid keeps the web app's run short
    add('sq5_mode_caps', steps=4100, N=49, src='mode', mm=2, mn=3, Co=0.6)
    return C


def js_keys(p):
    """Params field names -> the web app's S keys."""
    ren = {'vtx_drive': 'vtxDrive', 'vtx_inset': 'vtxInset'}
    return {ren.get(k, k): v for k, v in p.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='cuda')
    ap.add_argument('--f32', action='store_true')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--app', default=None, help='path to wave_membrane_MIRROR.html')
    a = ap.parse_args()

    C = configs(a.steps)
    out = tempfile.mkdtemp(prefix='zrl_xcheck_')
    # the web app's S needs EVERY key reset per config, or one config's setting leaks
    # into the next: start each from the defaults
    base = mc.Params()
    jc = []
    for c in C:
        p = {**{k: v for k, v in base.__dict__.items()}, **c['p']}
        jc.append({'name': c['name'], 'steps': c['steps'], 'p': js_keys(p)})
    cfg_path = os.path.join(out, 'configs.json')
    with open(cfg_path, 'w') as f:
        json.dump(jc, f)
    cmd = ['node', os.path.join(HERE, 'ToolsCLAUDE', 'js_dump_MIRROR.mjs'), cfg_path, out]
    if a.app:
        cmd.append(a.app)
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    patched = json.load(open(os.path.join(out, C[0]['name'] + '.json')))['patches']
    print(f'web app (Node) ran {len(C)} configs in {time.perf_counter() - t0:.1f} s; '
          f'patches applied: {", ".join(patched) if patched else "NONE (ZRL_JS_NOPATCH=1)"}')

    mc.init_taichi(a.arch, fp64=not a.f32)
    tol = 1e-4 if a.f32 else 1e-9
    tol_E = 1e-4 if a.f32 else 1e-10          # energy history, relative to its largest value
    fails = 0
    failed = []
    ok = lambda name, cond, det='': print(('PASS ' if cond else 'FAIL ') + name + ('  | ' + det if det else ''))
    for c in C:
        name = c['name']
        meta = json.load(open(os.path.join(out, name + '.json')))
        n = meta['N']
        rd = lambda suf, dt: np.fromfile(os.path.join(out, f'{name}_{suf}.bin'), dtype=dt).reshape(n, n).T
        jm, jcc, ju0, juK = rd('mask', np.uint8), rd('cc', np.float64), rd('u0', np.float64), rd('uK', np.float64)
        rh = lambda suf: np.fromfile(os.path.join(out, f'{name}_{suf}.bin'), dtype=np.float64)
        jh = {'E': rh('hE'), 'pA': rh('hA'), 'pB': rh('hB'), 'mode': rh('hM')}
        m = mc.Membrane(mc.Params(**{**base.__dict__, **c['p']}))
        tm, tcc, tu0 = m.mask_np, m.cc_np, m.field()
        m.step(c['steps'])
        tuK = m.field()
        th = {k: np.array(v, dtype=np.float64) for k, v in m.hist.items()}
        t_amax = m.amax
        E, amax = m.stats()
        scale = max(np.abs(juK).max(), 1e-300)
        d_cc = np.abs(tcc - jcc).max()
        d_u0 = np.abs(tu0 - ju0).max()
        d_uK = np.abs(tuK - juK).max() / scale
        # the same K steps started from the web app's OWN seed: isolates the stepping from
        # the seeding (numpy's exp and JS's Math.exp can differ in the last bit)
        m.seed()
        m.set_levels(ju0, ju0)
        m.step(c['steps'])
        d_same = np.abs(m.field() - juK).max() / scale
        th_same = {k: np.array(v, dtype=np.float64) for k, v in m.hist.items()}
        E, amax = m.stats()
        unstable = c['name'].startswith('sq5_unstable')
        if unstable:
            # past the CFL limit the checkerboard grows ~6.9x per step FROM ROUNDING NOISE, so
            # a last-bit difference anywhere is amplified without bound. The seed (numpy exp vs
            # Math.exp) is one such difference; on CUDA the other is FMA: the GPU compiler fuses
            # a*b + c into one rounding. So: on the CPU the stepping must be bit-identical from
            # the same seed; on a GPU both runs must blow up by the same factor (within 5%).
            # In f32 the rounding noise that seeds it is ~1e9 times larger, so the run must
            # blow up SOONER than the web app's f64 one -- agreement there would be the fault.
            if a.f32:
                d_uK = 0.0 if amax > 100 * meta['amax'] else 1.0
            else:
                d_uK = d_same if a.arch == 'cpu' else abs(math.log(amax / meta['amax'])) * tol / 0.05
        # record()'s histories. The stepping is compared from the port's OWN seed (like du(K));
        # past the CFL limit only the CPU's same-seed run is comparable (see above), and on a
        # GPU / in f32 the unstable run's histories are not compared at all.
        hist_note, hist_ok = '', True
        H = th_same if (unstable and a.arch == 'cpu' and not a.f32) else th
        if unstable and (a.arch != 'cpu' or a.f32):
            hist_note = 'hist skipped (unstable)'
        else:
            lens_ok = all(len(H[k]) == len(jh[k]) for k in jh)
            if lens_ok:
                pscale = max(scale, np.abs(ju0).max())
                dE = np.abs(H['E'] - jh['E']).max() / max(np.abs(jh['E']).max(), 1e-300) if len(jh['E']) else 0.0
                dP = max([np.abs(H[k] - jh[k]).max() / pscale for k in ('pA', 'pB', 'mode') if len(jh[k])] or [0.0])
                d_am = abs(t_amax - meta['amax']) / pscale if not unstable else 0.0
                hist_ok = dE <= tol_E and dP < tol and d_am < tol
                hist_note = f'hist n={len(H["E"])}/{len(H["mode"])} dE {dE:.1e} dprobe {dP:.1e} damax {d_am:.1e}'
            else:
                hist_ok = False
                hist_note = 'hist LENGTHS ' + ' '.join(f'{k} {len(H[k])} vs {len(jh[k])}' for k in jh)
        vtx_ok = [list(v[:2]) for v in m.vtx] == meta['vtx']
        cf_j, cf_t = meta['clipFrac'], m.clip_frac
        cf_ok = (cf_j is None and cf_t is None) or (cf_j is not None and cf_t is not None and abs(cf_j - cf_t) < 1e-12)
        good = (np.array_equal(tm, jm) and d_cc < 1e-15 and d_u0 < (1e-6 if a.f32 else 1e-12)
                and d_uK < tol and vtx_ok and cf_ok
                and abs(m.dom_R() - meta['domR']) < 1e-12
                and (unstable or abs(amax - meta['amax']) <= tol * scale) and hist_ok)
        fails += not good
        if not good:
            failed.append(name)
        ok(name, good, f'N={n} mask {"=" if np.array_equal(tm, jm) else "DIFF(%d)" % int((tm != jm).sum())} '
                       f'| dCo2 {d_cc:.1e} | du(0) {d_u0:.1e} | du({c["steps"]})/max {d_uK:.1e} '
                       f'(same seed {d_same:.1e}) '
                       f'| vtx {"=" if vtx_ok else "DIFF"} | clip {"=" if cf_ok else f"{cf_t} vs {cf_j}"} '
                       f'| max|u| {amax:.4g} | {hist_note}')
    print(f'\n{len(C) - fails}/{len(C)} configurations agree with the web app '
          f'{"+ patches " + "/".join(patched) + " " if patched else "(UNPATCHED) "}'
          f'({a.arch} {"f32" if a.f32 else "f64"}, tolerance {tol:g} of max|u|, energy history {tol_E:g})'
          + (f'\ndiffering: {", ".join(failed)}' if failed else ''))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
