import math, pickle, numpy as np
import gridlib_MIRROR as G, circle_bench_MIRROR as B
conv = pickle.load(open('res_conv.pkl','rb'))
names = ["sq5 (app)","sq9 (app)","tri (app)","sq + edge triangles","tri + edge triangles","hex shells -> circles","Penrose + edge triangles"]
print("time-averaged eps_ang over cycles 1-5 (sigma = 0.1 R):")
for nm in names:
    v = [conv[(nm,R)]['eang'][(conv[(nm,R)]['t']>=1)].mean() for R in (30.,60.,120.)]
    sl = math.log(v[2]/v[1])/math.log(2)
    print(f"  {nm:26s} N~64: {v[0]:.2e}  N~124: {v[1]:.2e}  N~244: {v[2]:.2e}   slope(124->244) {sl:+.2f}")
print("radius jitter, sigma = 6 cells, 5 cycles, time-averaged eps_ang:")
jit = {}
for nm, mk in [("sq5 (app)", lambda R: G.lattice('sq5',R)), ("sq9 (app)", lambda R: G.lattice('sq9',R)),
               ("tri (app)", lambda R: G.lattice('tri',R)), ("tri + edge triangles", lambda R: G.tri_fitted(R))]:
    vals = []
    for R in np.arange(55.0, 65.01, 1.0):
        o = B.run(mk(R), sig=6.0, cycles=5, per_cycle=4, label=nm); vals.append(o['eang'][o['t']>=1].mean())
    vals = np.array(vals); jit[nm] = vals
    print(f"  {nm:22s} min {vals.min():.2e}  median {np.median(vals):.2e}  max {vals.max():.2e}  (max/min {vals.max()/vals.min():.0f}x)", flush=True)
pickle.dump(jit, open('res_jit.pkl','wb'))
