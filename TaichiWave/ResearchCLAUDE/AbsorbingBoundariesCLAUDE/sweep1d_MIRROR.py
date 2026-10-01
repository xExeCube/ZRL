"""Reflection-coefficient sweep for every absorbing boundary prototype (1D-reduced, exact for a
straight wall of the square 5-point lattice; see abc1d_MIRROR.py for the method).

    python sweep1d_MIRROR.py            # all stages, ~5 min on 8 processes
Writes results_1d_MIRROR.json next to this file; tables_MIRROR.py formats it.
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import abc1d_MIRROR as A  # noqa: E402

ANGLES = (0, 15, 30, 45, 60, 75)
LAMS = (8, 16, 32)


def fixed_methods(Co):
    """name -> (left spec, layer width L)."""
    M = {
        'dirichlet': (('dir',), 1),
        'mur1': (('rule', A.rule_mur1(Co)), 1),
        'mur2': (('rule', A.rule_mur2(Co)), 1),
        'higdon2(0,45)': (('rule', A.rule_higdon(Co, [0, 45])), 1),
        'higdon2(0,60)': (('rule', A.rule_higdon(Co, [0, 60])), 1),
        'higdon2(0,45)e.005': (('rule', A.rule_higdon(Co, [0, 45], 0.005)), 1),
        'higdon2(0,45)e.02': (('rule', A.rule_higdon(Co, [0, 45], 0.02)), 1),
        'higdon3(0,30,60)': (('rule', A.rule_higdon(Co, [0, 30, 60])), 1),
        'higdon3(0,30,60)e.005': (('rule', A.rule_higdon(Co, [0, 30, 60], 0.005)), 1),
        'higdon3(0,30,60)e.02': (('rule', A.rule_higdon(Co, [0, 30, 60], 0.02)), 1),
        'higdon3(0,45,70)e.005': (('rule', A.rule_higdon(Co, [0, 45, 70], 0.005)), 1),
        'liao2': (('rule', A.rule_liao(Co, 2)), 1),
        'liao3': (('rule', A.rule_liao(Co, 3)), 1),
        'liao2 direct-interp': (('rule', A.rule_liao(Co, 2, composed=False)), 1),
        'liao3 direct-interp': (('rule', A.rule_liao(Co, 3, composed=False)), 1),
        'cerjan20(a=.015)': (('cerjan', 20, 0.015), 20),
        'cerjan40(a=.0075)': (('cerjan', 40, 0.0075), 40),
    }
    return M


def tuning_methods():
    M = {}
    for L in (10, 20, 40):
        for p in (1, 2, 3):
            for dmax in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8):
                for form in ('app', 'ctr'):
                    M['sponge%d p%d d%.2f %s' % (L, p, dmax, form)] = (('sponge', L, dmax, p, form, 'dir'), L)
    for L in (5, 10, 20):
        for p in (2, 3):
            for fac in (0.5, 1.0, 2.0):
                for al in (0.0, 0.05, 0.15):
                    M['cpml%d p%d f%.1f a%.2f' % (L, p, fac, al)] = (('cpml', L, fac, p, al), L)
    for L in (10, 20):
        for fac in (0.5, 1.0, 2.0):
            M['gs%d p2 f%.1f' % (L, fac)] = (('gs', L, fac, 2), L)
    return M


def job(args):
    Co, lam, th, names, specs = args
    g = A.geometry(Co, lam, th)
    ref = A.run_reference(g)
    out = {}
    for name, (left, L) in zip(names, specs):
        out[name] = A.run_test(g, left, L, ref)
    return Co, lam, th, out


def run_stage(Co, methods, pool, chunk=12):
    names = list(methods)
    tasks = []
    for lam in LAMS:
        for th in ANGLES:
            for c0 in range(0, len(names), chunk):
                nm = names[c0:c0 + chunk]
                tasks.append((Co, lam, th, nm, [methods[n] for n in nm]))
    tasks.sort(key=lambda t: -(t[1] / max(math.cos(math.radians(t[2])), 0.2)) ** 2)   # longest first
    res = {n: {} for n in names}
    for Co_, lam, th, out in pool.imap_unordered(job, tasks):
        for n, r in out.items():
            res[n]['%d/%d' % (lam, th)] = r
    return res


def gmean(d):
    v = [max(x, 1e-12) for x in d.values()]
    return float(np.exp(np.mean(np.log(v)))) if all(np.isfinite(v)) else float('inf')


def theory_table(Co):
    out = {}
    for name, (left, L) in fixed_methods(Co).items():
        if left[0] != 'rule':
            continue
        out[name] = {}
        for lam in LAMS:
            for th in ANGLES:
                k = A.TAU / lam
                kx, ky = k * math.cos(math.radians(th)), k * math.sin(math.radians(th))
                out[name]['%d/%d' % (lam, th)] = A.rule_theory(left[1], Co, kx, ky)
    return out


def main():
    t0 = time.time()
    results = {'meta': {'date': '29/09/2026', 'angles': ANGLES, 'lams': LAMS,
                        'method': 'abc1d_MIRROR.py: packet + reference, 1D-reduced exact'}}
    with Pool(8) as pool:
        results['Co0.5_fixed'] = run_stage(0.5, fixed_methods(0.5), pool)
        results['Co0.5_theory'] = theory_table(0.5)
        print('fixed done %.0fs' % (time.time() - t0), flush=True)
        tun = run_stage(0.5, tuning_methods(), pool)
        results['Co0.5_tuning'] = tun
        print('tuning done %.0fs' % (time.time() - t0), flush=True)
        # best tuned per family / width (geometric mean over all 18 angle x wavelength cases)
        best = {}
        for name, d in tun.items():
            fam = name.split(' ')[0] + (' ' + name.split(' ')[-1] if name.startswith('sponge') else '')
            if fam not in best or gmean(d) < gmean(tun[best[fam]]):
                best[fam] = name
        results['best'] = best
        # the best sponges once more, terminated with Mur1 instead of Dirichlet
        extra = {}
        for fam, name in best.items():
            if name.startswith('sponge'):
                left, L = tuning_methods()[name]
                extra[name + ' +mur1end'] = (left[:5] + ('mur1',), L)
        results['Co0.5_mur1end'] = run_stage(0.5, extra, pool)
        # Co = 0.7 for the fixed methods and the best tuned ones
        m7 = fixed_methods(0.7)
        tm = tuning_methods()
        for fam, name in best.items():
            m7[name] = tm[name]
        results['Co0.7'] = run_stage(0.7, m7, pool)
        results['Co0.7_theory'] = theory_table(0.7)
    results['meta']['seconds'] = time.time() - t0
    with open(os.path.join(HERE, 'results_1d_MIRROR.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print('done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
