"""Fact-check runs, 01/10/2026 (Claude Opus 5.5), for NotesCLAUDE/absorbing_boundaries_MIRROR.md.
 box : 2D box pulse reflection vs pulse width sigma (3 = app default, 12 = app slider max) and
       window (centre pulse r_end 150 vs 230), corner pulse at (25,25)/165 and (40,40)/180.
 msk : masked distance sponge L40 d0.2 p2, app vs centred form, sigma 3/6/12, sq circle + tri hex.
 t1  : T1 noise runs, 100000 steps (R = 40), tri hex Co 0.5..0.7, sq circle 0.7 / 0.7071.
 nz  : 2D box noise 100000 steps: cpml10, cpml20a.05, mur2 diag, higdon2 eps.005 at Co 0.7 / 0.7071.
    python checks_MIRROR.py   -> prints one line per job (also written to checks_out_MIRROR.txt)."""
import math, os, sys
from multiprocessing import Pool
import numpy as np
RB = r'F:/FDrive-Storage/Claude Local/FormConstantsCLAUDE/TaichiWaveCLAUDE/ResearchCLAUDE/AbsorbingBoundariesCLAUDE'
sys.path.insert(0, RB)
import abc1d_MIRROR as A, abc2d_MIRROR as B, masked_MIRROR as MM, masked_normal_MIRROR as MN

def spec(name, Co):
    return {'mur1': ('rule', A.rule_mur1(Co), 'zero'), 'mur2diag': ('rule', A.rule_mur2(Co), 'diag'),
            'h2e': ('rule', A.rule_higdon(Co, [0, 45], 0.005), 'diag'),
            'cpml10': ('cpml', 10, 2.0, 3, 0.0), 'cpml20a': ('cpml', 20, 2.0, 3, 0.05),
            'sp40ctr': ('sponge', 40, 0.5, 3, 'ctr', 'dir')}[name]

def job(a):
    kind = a[0]
    if kind == 'box':
        _, name, s, src, r_end = a
        return a, B.pulse_reflection(spec(name, 0.5), 0.5, 200, src, r_end, s=s)
    if kind == 'msk':
        _, lat, shape, form, s = a
        R, L, dmax, p, Co = 100, 40, 0.2, 2, 0.5
        r_end = R + 140; T = int(round(r_end / Co)); Rt = R + L
        def run(Rr, d):
            N = MM.grid_size(lat, Rr); M = MM.shape_mask(lat, shape, N, Rr)
            X, Y = MM.coords(lat, N); cx, cy = MM.centre(lat, N)
            g = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s)) * M
            up, u = g.copy(), g.copy(); E0 = None
            for n in range(T):
                lp = MM.lap(lat, u)
                if d is None: un = 2 * u - up + Co * Co * lp
                elif form == 'app': un = 2 * u - up + Co * Co * lp - d * (u - up)
                else: un = (2 * u - up * (1 - 0.5 * d) + Co * Co * lp) / (1 + 0.5 * d)
                un[~M] = 0.0; up, u = u, un
                if n == 0: E0 = MM.energy(lat, u, up, Co, M)
            return N, M, u, up, E0
        N = MM.grid_size(lat, Rt); M = MM.shape_mask(lat, shape, N, Rt); dist = MM.dist_to_edge(lat, M)
        d = np.where(M & (dist < L), dmax * np.clip((L - dist) / L, 0, 1) ** p, 0.0)
        _, _, u, up, _ = run(Rt, d)
        Nr, _, ru, rup, E0 = run(Rt + int(0.5 * r_end) + 20, None)
        off = Nr // 2 - N // 2
        ru = ru[off:off + N, off:off + N]; rup = rup[off:off + N, off:off + N]
        Er = MM.energy(lat, u - ru, up - rup, Co, M & (dist >= L))
        return a, math.sqrt(max(Er, 0) / E0)
    if kind == 't1':
        _, lat, shape, Co, steps = a
        rng = np.random.default_rng(5); R = 40
        N = MM.grid_size(lat, R); M = MM.shape_mask(lat, shape, N, R)
        X, Y = MM.coords(lat, N); cx, cy = MM.centre(lat, N)
        rl = MN.Rule(lat, shape, M, X, Y, cx, cy, Co, 'T1')
        u = rng.uniform(-1, 1, (N, N)) * M; up = u.copy(); out = {}
        for n in range(1, steps + 1):
            un = 2 * u - up + Co * Co * MM.lap(lat, u); rl.apply(un, u, up); un[~M] = 0.0; up, u = u, un
            if n % 10000 == 0 or n == 5000:
                out[n] = float(np.max(np.abs(u)))
                if not np.isfinite(out[n]) or out[n] > 1e30: break
        return a, {k: '%.3g' % v for k, v in out.items()}
    if kind == 'nz':
        _, name, Co, steps = a
        marks = tuple([1000, 5000] + list(range(10000, steps + 1, 10000)))
        o = B.noise_run(spec(name, Co), Co, steps=steps, marks=marks)
        return a, {k: '%.3g' % v[0] for k, v in o.items()}

if __name__ == '__main__':
    tasks = []
    tasks += [('t1', 'tri', 'hex', Co, 100000) for Co in (0.5, 0.55, 0.6, 0.65, 0.7)]
    tasks += [('t1', 'sq', 'circle', Co, 100000) for Co in (0.7, 0.7071)]
    tasks += [('nz', n, Co, 100000) for n in ('cpml10', 'cpml20a') for Co in (0.7, 0.7071)]
    tasks += [('nz', n, 0.7, 100000) for n in ('mur2diag', 'h2e')]
    for s in (3.0, 12.0):
        for n in ('mur1', 'mur2diag', 'cpml10', 'cpml20a', 'sp40ctr'):
            tasks += [('box', n, s, (100, 100), 230), ('box', n, s, (40, 40), 180)]
    for n in ('mur1', 'cpml10', 'sp40ctr'):
        tasks += [('box', n, 3.0, (100, 100), 150), ('box', n, 3.0, (25, 25), 165)]
    for lat, shape in (('sq', 'circle'), ('tri', 'hex')):
        for form in ('app', 'ctr'):
            for s in (3.0, 6.0, 12.0):
                tasks.append(('msk', lat, shape, form, s))
    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checks_out_MIRROR.txt')
    with Pool(15) as p, open(outp, 'w') as f:
        for a, r in p.imap_unordered(job, tasks):
            line = '%s -> %s' % (a, ('%.3e' % r) if isinstance(r, float) else r)
            print(line, flush=True); f.write(line + '\n'); f.flush()
