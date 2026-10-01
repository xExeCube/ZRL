"""Long-run growth check of every local rule, one tangential wavenumber at a time (1D-reduced,
exact for a straight wall; see abc1d_MIRROR.py). Random data, 20000 steps, Co in (0.5, 0.7, 0.707),
ky in (0, pi/4, pi/2, 3pi/4, pi): prints max|u| at 1000/5000/10000/20000 steps. A rule is
flagged if max|u| at 20000 exceeds 2x its value at 1000 steps, or exceeds the initial 1.0.
The 2D box noise runs (box2d_runs_MIRROR.py) test all wavenumbers and the corners at once;
this one isolates WHICH wavenumber misbehaves.      python stability1d_MIRROR.py
Written 30/09/2026.
"""
import json
import math
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import abc1d_MIRROR as A  # noqa: E402

KYS = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi)
COS = (0.5, 0.7, 0.7071)


def rules(Co):
    return {
        'mur1': A.rule_mur1(Co),
        'mur2': A.rule_mur2(Co),
        'higdon2(0,45)': A.rule_higdon(Co, [0, 45]),
        'higdon2(0,45)e.005': A.rule_higdon(Co, [0, 45], 0.005),
        'higdon3(0,30,60)': A.rule_higdon(Co, [0, 30, 60]),
        'higdon3(0,30,60)e.005': A.rule_higdon(Co, [0, 30, 60], 0.005),
        'higdon3(0,30,60)e.02': A.rule_higdon(Co, [0, 30, 60], 0.02),
        'liao2': A.rule_liao(Co, 2),
        'liao3': A.rule_liao(Co, 3),
    }


def job(args):
    name, Co, ky = args
    return name, Co, ky, A.growth_check(('rule', rules(Co)[name]), Co, ky)


def main():
    tasks = [(n, Co, ky) for Co in COS for n in rules(Co) for ky in KYS]
    out = {}
    with Pool(8) as pool:
        for name, Co, ky, res in pool.imap_unordered(job, tasks):
            out['%s Co%.4f ky%.3f' % (name, Co, ky)] = res
    flagged = []
    for k in sorted(out):
        r = out[k]
        v = [r.get(n, float('inf')) for n in (1000, 5000, 10000, 20000)]
        bad = (not all(math.isfinite(x) for x in v)) or v[3] > 2 * v[0] or v[3] > 1.0
        if bad:
            flagged.append(k)
        print('%-40s %9.2e %9.2e %9.2e %9.2e %s' % (k, *v, 'GROWS' if bad else ''))
    print('flagged:', flagged)
    with open(os.path.join(HERE, 'results_stability1d_MIRROR.json'), 'w') as f:
        json.dump({'date': '30/09/2026', 'runs': out, 'flagged': flagged}, f, indent=1)


if __name__ == '__main__':
    main()
