"""Fact-check 01/10/2026, part 2: window dependence of the 2D centre-pulse reading (sigma 3, Co 0.5).
The note reads (test - reference) energy at r_end = 230; oblique reflections leave through the
adjacent walls from r ~ 141 on, so a single late window can undercount. Any window's reading is a
LOWER bound on the total reflected energy; the max over windows is the best lower bound.
Also: CPML10 / CPML20 with sigma 12 at (60,60) (initial data clear of the layer)."""
import os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checks_MIRROR import spec, B
def job(a):
    name, s, src, r_end = a
    return a, B.pulse_reflection(spec(name, 0.5), 0.5, 200, src, r_end, s=s)
if __name__ == '__main__':
    tasks = [(n, 3.0, (100, 100), r) for n in ('mur1', 'mur2diag', 'h2e', 'cpml10', 'cpml20a', 'sp40ctr')
             for r in (130, 150, 170, 190, 210)]
    tasks += [(n, 12.0, (60, 60), 180) for n in ('cpml10', 'cpml20a', 'mur1', 'sp40ctr')]
    with Pool(12) as p, open('checks2_out_MIRROR.txt', 'w') as f:
        for a, r in p.imap_unordered(job, tasks):
            line = '%s -> %.3e' % (a, r); print(line, flush=True); f.write(line + '\n'); f.flush()
