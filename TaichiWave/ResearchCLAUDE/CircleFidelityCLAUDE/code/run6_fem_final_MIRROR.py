import math, pickle, numpy as np
import gridlib_MIRROR as G, circle_bench_MIRROR as B
R = 60.0; s = 1.25
res = pickle.load(open('res_main.pkl','rb'))
fem = [("sq + edge triangles", lambda R: G.sq_fitted(R)), ("tri + edge triangles", lambda R: G.tri_fitted(R)),
       ("hex shells -> circles", lambda R: G.hex_rings(R)), ("Penrose + edge triangles", lambda R: G.penrose_fitted(R)),
       ("unstructured uniform", lambda R: G.triangle_mesh(R, lambda r: s*np.ones_like(r), s, "unstructured uniform")),
       ("graded 2:1 toward edge", lambda R: G.triangle_mesh(R, lambda r: s*(1.0-0.5*np.clip((r-40)/20,0,1)), 0.5*s, "graded 2:1 toward edge"))]
for k in [k for k in res if k[0] in [f[0] for f in fem]]: del res[k]
for nm, mk in fem:
    g = mk(R)
    for sig in (3.0, 6.0):
        o = B.run(g, sig=sig, label=nm); B.show(o); res[(nm, sig)] = o
    print(f"    {nm}: n={g.n} tris={g.ntri} angles {g.min_angle:.1f}-{g.max_angle:.1f}", flush=True)
pickle.dump(res, open('res_main.pkl','wb'))
conv = pickle.load(open('res_conv.pkl','rb'))
for Rc in (30.0, 60.0, 120.0):
    for nm, mk in fem[:4]:
        o = B.run(mk(Rc), sig=0.1*Rc, cycles=5, per_cycle=4, label=nm); conv[(nm, Rc)] = o
pickle.dump(conv, open('res_conv.pkl','wb'))
names = ["sq5 (app)","sq9 (app)","tri (app)","sq + edge triangles","tri + edge triangles","hex shells -> circles","Penrose + edge triangles"]
print("time-averaged eps_ang, cycles 1-5, sigma = 0.1 R:")
for nm in names:
    v = [conv[(nm,Rc)]['eang'][(conv[(nm,Rc)]['t']>=1)].mean() for Rc in (30.,60.,120.)]
    e = [conv[(nm,Rc)]['etot'][(conv[(nm,Rc)]['t']>=1)].mean() for Rc in (30.,60.,120.)]
    print(f"  {nm:26s} eang N~64/124/244: {v[0]:.2e} {v[1]:.2e} {v[2]:.2e} (slope {math.log(v[2]/v[1])/math.log(2):+.2f})"
          f" | etot: {e[0]:.2e} {e[1]:.2e} {e[2]:.2e}")
jit = pickle.load(open('res_jit.pkl','rb')); vals = []
for Rj in np.arange(55.0, 65.01, 1.0):
    o = B.run(G.tri_fitted(Rj), sig=6.0, cycles=5, per_cycle=4); vals.append(o['eang'][o['t']>=1].mean())
jit["tri + edge triangles"] = np.array(vals); pickle.dump(jit, open('res_jit.pkl','wb'))
for nm, v in jit.items(): print(f"  jitter {nm:22s} min {v.min():.2e} median {np.median(v):.2e} max {v.max():.2e}")
