"""Ledger build time vs N (CUDA f64 default): first build (geometry cache cold) and steady refresh."""
import os, sys, time, argparse
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, HERE)
import membrane_core_MIRROR as mc
import ledger_MIRROR as lg
ap = argparse.ArgumentParser()
ap.add_argument('--arch', default='cuda'); ap.add_argument('--f32', action='store_true')
ap.add_argument('--Ns', default='161,1025,2049,4097,8193')
a = ap.parse_args()
mc.init_taichi(a.arch, fp64=not a.f32, quiet=True)
cfgs = [('sq5 full pulse', dict(src='pulse')),
        ('sq5 full impulse', dict(src='impulse')),
        ('sq5 full mode', dict(src='mode')),
        ('sq5 n6 vtx', dict(shape='n6', src='vtx')),
        ('tri n6 conf pulse', dict(lattice='tri', shape='n6', rot=30 * mc.PI / 180)),
        ('sq5 full mur cont', dict(src='cont', bc='mur', freq=0.05))]
for N in map(int, a.Ns.split(',')):
    for name, kw in cfgs:
        t0 = time.perf_counter()
        m = mc.Membrane(mc.Params(N=N, **kw))
        t_setup = time.perf_counter() - t0
        m.step(64)
        p1 = {}; t0 = time.perf_counter(); rows = lg.build_ledger(m, profile=p1); t1 = (time.perf_counter() - t0) * 1e3
        m.step(8)
        p2 = {}; t0 = time.perf_counter(); rows = lg.build_ledger(m, profile=p2); t2 = (time.perf_counter() - t0) * 1e3
        m.step(8)
        t0 = time.perf_counter(); rows = lg.build_ledger(m); t3 = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter(); U = m.level(1); tU = (time.perf_counter() - t0) * 1e3
        top = sorted(p2.items(), key=lambda kv: -kv[1])[:3]
        print(f'N={N:5d} {name:18s} setup {t_setup:6.2f} s  first {t1:8.1f} ms  refresh {t2:8.1f} / {t3:8.1f} ms  '
              f'(level(1) alone {tU:6.1f} ms)  top: ' + ', '.join(f'{k[1:]} {v:.1f}' for k, v in top), flush=True)
        del m
