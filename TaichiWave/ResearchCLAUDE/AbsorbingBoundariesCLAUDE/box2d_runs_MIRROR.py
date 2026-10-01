"""2D box measurements on the square 5-point lattice (numpy), for the note:
  1. pulse reflection with all four walls AND the corners (centre pulse and near-corner pulse);
  2. long-run stability from white noise, 20000 steps, Co = 0.5 and 0.7;
  3. numpy time per step relative to the plain Dirichlet step (indicative only: numpy's
     per-operation overheads are not the GPU's -- gpu_cost_MIRROR.py measures that). 30/09/2026: the run that wrote the json shared the CPU
     with 16 other numpy processes, so its numpy cost column is noise (Dirichlet slower than
     Mur1) and the note does not use it.
The tuned sponge / CPML parameters are read from results_1d_MIRROR.json ('best').
    python box2d_runs_MIRROR.py
Writes results_box2d_MIRROR.json.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import abc1d_MIRROR as A  # noqa: E402
import abc2d_MIRROR as B  # noqa: E402


def parse_tuned(name):
    """'sponge20 p2 d0.50 app' -> spec; 'cpml10 p3 f2.0 a0.00' -> spec."""
    parts = name.split(' ')
    if parts[0].startswith('sponge'):
        return ('sponge', int(parts[0][6:]), float(parts[2][1:]), int(parts[1][1:]), parts[3], 'dir')
    return ('cpml', int(parts[0][4:]), float(parts[2][1:]), int(parts[1][1:]), float(parts[3][1:]))


def methods(Co, best):
    M = {
        'dirichlet': ('dir',),
        'mur1 (app: corners 0)': ('rule', A.rule_mur1(Co), 'zero'),
        'mur2 corners 0': ('rule', A.rule_mur2(Co), 'zero'),
        'mur2 corners diag-Mur1': ('rule', A.rule_mur2(Co), 'diag'),
        'mur2 corners avg': ('rule', A.rule_mur2(Co), 'avg'),
        'higdon2(0,45) avg': ('rule', A.rule_higdon(Co, [0, 45]), 'avg'),
        'higdon2(0,45) diag': ('rule', A.rule_higdon(Co, [0, 45]), 'diag'),
        'higdon2(0,45)e.005 avg': ('rule', A.rule_higdon(Co, [0, 45], 0.005), 'avg'),
        'higdon3(0,30,60) avg': ('rule', A.rule_higdon(Co, [0, 30, 60]), 'avg'),
        'higdon3(0,30,60)e.005 avg': ('rule', A.rule_higdon(Co, [0, 30, 60], 0.005), 'avg'),
        'higdon3(0,45,70)e.005 avg': ('rule', A.rule_higdon(Co, [0, 45, 70], 0.005), 'avg'),
        'liao2 avg': ('rule', A.rule_liao(Co, 2), 'avg'),
        'liao3 avg': ('rule', A.rule_liao(Co, 3), 'avg'),
    }
    for fam in ('sponge10 app', 'sponge20 app', 'sponge40 app', 'sponge40 ctr', 'cpml5', 'cpml10', 'cpml20'):
        M[best[fam]] = parse_tuned(best[fam])
    return M


def job_pulse(args):
    name, spec, Co = args
    cache = {}
    rc = B.pulse_reflection(spec, Co, 200, (100, 100), 230, ref_cache=cache)
    rk = B.pulse_reflection(spec, Co, 200, (25, 25), 165, ref_cache=cache)
    return name, Co, rc, rk


def job_noise(args):
    name, spec, Co = args
    return name, Co, B.noise_run(spec, Co)


def cost(spec, Co, N=1000, steps=40):
    L = B.layer_of(spec)
    b = B.Box2D(N + 2 * L, Co, spec)
    g = B.gauss(N + 2 * L, N // 2, N // 2, 3.0)
    b.set(g, g)
    for _ in range(3):
        b.step()
    t = time.perf_counter()
    for _ in range(steps):
        b.step()
    return (time.perf_counter() - t) / steps


def main():
    with open(os.path.join(HERE, 'results_1d_MIRROR.json')) as f:
        best = json.load(f)['best']
    out = {'date': '29/09/2026', 'pulse': {}, 'noise': {}, 'cost_numpy_s_per_step_N1000': {}}
    t0 = time.time()
    with Pool(8) as pool:
        tasks = [(n, s, Co) for Co in (0.5, 0.7) for n, s in methods(Co, best).items()]
        for name, Co, rc, rk in pool.imap_unordered(job_pulse, tasks):
            out['pulse']['%s @Co%.1f' % (name, Co)] = {'centre': rc, 'corner': rk}
        print('pulse done %.0fs' % (time.time() - t0), flush=True)
        for name, Co, res in pool.imap_unordered(job_noise, tasks):
            out['noise']['%s @Co%.1f' % (name, Co)] = res
        print('noise done %.0fs' % (time.time() - t0), flush=True)
    base = None
    for name, spec in methods(0.5, best).items():
        c = cost(spec, 0.5)
        base = base or c
        out['cost_numpy_s_per_step_N1000'][name] = c
    with open(os.path.join(HERE, 'results_box2d_MIRROR.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
