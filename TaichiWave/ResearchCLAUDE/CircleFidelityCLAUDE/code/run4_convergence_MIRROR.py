import math, pickle, numpy as np
import gridlib_MIRROR as G, circle_bench_MIRROR as B
conv = {}
builders = [("sq5 (app)", lambda R: G.lattice('sq5', R)), ("sq9 (app)", lambda R: G.lattice('sq9', R)),
            ("tri (app)", lambda R: G.lattice('tri', R)), ("sq + edge triangles", lambda R: G.sq_fitted(R)),
            ("tri + edge triangles", lambda R: G.tri_fitted(R)), ("hex shells -> circles", lambda R: G.hex_rings(R)),
            ("Penrose + edge triangles", lambda R: G.penrose_fitted(R))]
for R in (30.0, 60.0, 120.0):
    sig = 0.1*R
    for nm, mk in builders:
        g = mk(R); o = B.run(g, sig=sig, cycles=5, per_cycle=4, label=nm)
        conv[(nm, R)] = o
        print(f"R={R:5.0f} (N~{int(2*R+4)}) {nm:26s} n={g.n:6d} eang@m=1,2,5: " + " ".join(f"{o['eang'][np.argmin(abs(o['t']-m))]:.2e}" for m in (1,2,5))
              + "  etot@m=1,2,5: " + " ".join(f"{o['etot'][np.argmin(abs(o['t']-m))]:.2e}" for m in (1,2,5)), flush=True)
pickle.dump(conv, open('res_conv.pkl','wb'))
print("--- observed order: slope of log(eps) vs log(R) between R=60 and R=120 (energy; amplitude order = slope/2)")
for nm, _ in builders:
    for key in ('eang', 'etot'):
        a, b = conv[(nm, 60.0)][key], conv[(nm, 120.0)][key]
        i = lambda o: np.argmin(abs(o['t']-5))
        s = math.log(b[i(conv[(nm,120.0)])]/a[i(conv[(nm,60.0)])])/math.log(2)
        print(f"  {nm:26s} {key}: slope {s:+.2f}")
