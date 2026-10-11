import math, pickle, numpy as np
import gridlib_MIRROR as G, circle_bench_MIRROR as B
R = 60.0; res = pickle.load(open('res_main.pkl','rb'))
for k in [k for k in res if k[0].startswith(('unstructured','graded'))]: del res[k]
s = 1.25
gu = G.triangle_mesh(R, lambda r: s*np.ones_like(r), s, "unstructured uniform")
gg = G.triangle_mesh(R, lambda r: s*(1.0 - 0.5*np.clip((r-40)/20, 0, 1)), 0.5*s, "graded 2:1 toward edge")
for g in (gu, gg): print(f"{g.name}: n={g.n} tris={g.ntri} min/max angle {g.min_angle:.1f}/{g.max_angle:.1f}")
for sig in (3.0, 6.0):
    for g in (gu, gg):
        o = B.run(g, sig=sig); B.show(o); res[(g.name, sig)] = o
pickle.dump(res, open('res_main.pkl','wb'))

print("--- free-space control: same pulse, wall out of reach (bulk anisotropy only)")
ctrl = {}
for kind in ('sq5','sq9','tri'):
    g = G.lattice(kind, 190.0)
    for sig in (3.0, 6.0):
        o = B.run(g, sig=sig, cycles=1, exact=False, Tref=120.0, per_cycle=8, label=f"{kind} free space")
        ctrl[(kind, sig)] = o
        print(f"{kind} sig={sig}: eang after travel 15/60/120 cells: " + " ".join(f"{o['eang'][i]:.2e}" for i in (0,3,7)), flush=True)
pickle.dump(ctrl, open('res_ctrl.pkl','wb'))
