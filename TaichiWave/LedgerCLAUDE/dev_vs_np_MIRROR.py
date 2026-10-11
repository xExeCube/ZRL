"""Device reductions vs the numpy references: every row, both paths, several configs and step counts."""
import os, sys, math, time
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8')
import membrane_core_MIRROR as mc
import ledger_MIRROR as lg
arch = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
f32 = len(sys.argv) > 2 and sys.argv[2] == 'f32'
mc.init_taichi(arch, fp64=not f32, quiet=True)
D = math.pi / 180
cfgs = [dict(src='pulse'), dict(src='impulse'), dict(src='impulse', bc='periodic'), dict(src='impulse', stencil=9),
        dict(src='mode', mm=3, mn=2), dict(src='mode', rad=0.7, mm=2, mn=5), dict(shape='n8', rot=22.5 * D),
        dict(lattice='tri', shape='n6', rot=30 * D), dict(lattice='tri', shape='n3', rot=30 * D, src='impulse'),
        dict(lattice='tri', shape='circle', rad=0.9), dict(lattice='tri', shape='n6', rot=30 * D, src='vtx'),
        dict(shape='n5', src='vtx'), dict(src='impulse', sx=0.4, sy=-0.2, bc='mur'),
        dict(src='pulse', Co=0.75)]
bad = 0
for N in (161, 257):
    for kw in cfgs:
        m = mc.Membrane(mc.Params(N=N, **kw))
        for steps in (0, 40, 200, 700):
            m.step(steps - m.step_n)
            lg.DEVICE = True
            a = lg.build_ledger(m)
            lg.DEVICE = False
            b = lg.build_ledger(m)
            lg.DEVICE = True
            for ra, rb in zip(a, b):
                exact = ra.key in ('sym8', 'sym6', 'mirror', 'aniso')
                same_num = (ra.num == rb.num or (ra.num is not None and rb.num is not None and
                            (math.isnan(ra.num) and math.isnan(rb.num) or
                             (not exact and abs(ra.num - rb.num) <= 1e-12 * max(1e-300, abs(rb.num))))))
                if ra.badge != rb.badge or not same_num or (exact and ra.note != rb.note):
                    bad += 1
                    print(f'DIFF N={N} {kw} @{steps} [{ra.key}] {ra.badge} {ra.num!r} vs {rb.badge} {rb.num!r}')
                    if ra.note != rb.note:
                        print('   dev:', ra.note[:200]); print('   np: ', rb.note[:200])
print(f'device vs numpy ({arch}{" f32" if f32 else ""}): {bad} differing rows')
