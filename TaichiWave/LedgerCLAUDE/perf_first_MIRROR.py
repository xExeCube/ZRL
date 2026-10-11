"""Where the FIRST (geometry-cold) ledger build spends its time at large N."""
import os, sys, time
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, HERE)
import membrane_core_MIRROR as mc
import ledger_MIRROR as lg
mc.init_taichi('cuda', quiet=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8193
for name, kw in [('sq5 full pulse', dict(src='pulse')), ('sq5 n6 vtx', dict(shape='n6', src='vtx')),
                 ('tri n6 conf pulse', dict(lattice='tri', shape='n6', rot=30 * mc.PI / 180)),
                 ('unlicensed pulse sy=-0.9', dict(shape='unlicensed', sy=-0.9))]:
    t0 = time.perf_counter()
    m = mc.Membrane(mc.Params(N=N, **kw))
    ts = time.perf_counter() - t0
    m.step(64)
    lg.build_ledger(mc.Membrane(mc.Params(N=65, **kw)))      # compile the kernels first
    p = {}
    t0 = time.perf_counter(); lg.build_ledger(m, profile=p); t1 = (time.perf_counter() - t0) * 1e3
    top = sorted(p.items(), key=lambda kv: -kv[1])[:5]
    print(f'N={N} {name:26s} setup {ts:5.1f} s  first build {t1:7.1f} ms: ' + ', '.join(f'{k[1:]} {v:.0f}' for k, v in top), flush=True)
    del m
