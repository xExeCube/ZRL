import math, pickle, numpy as np
import gridlib_MIRROR as G, circle_bench_MIRROR as B
R = 60.0
S3H = math.sqrt(3)/2
grids = [
  G.lattice('sq5', R), G.lattice('sq9', R), G.lattice('tri', R),
  G.lattice('tri', R, rot=math.pi/6),                               # flat-top hex cells (the other orientation)
  G.lattice('tri', R, shift=(0.5, S3H/3)),                          # circle centred on a triangle (hole), not a node
  G.lattice('sq5', R, shift=(0.5, 0.5)),                            # square lattice, cell-centred
  G.sq_fitted(R), G.tri_fitted(R), G.hex_rings(R),
  G.distmesh_circle(R, lambda r: np.ones_like(r), 1.0, name="unstructured uniform"),
  G.distmesh_circle(R, lambda r: 1.0 - 0.5*np.clip((r-40)/20, 0, 1), 0.5, name="graded 1 -> 0.5 at edge"),
]
names = ["sq5 (app)", "sq9 (app)", "tri (app, pointy-top)", "tri flat-top (rot 30)", "tri hole-centred",
         "sq5 cell-centred", "sq + edge triangles", "tri + edge triangles", "hex shells -> circles",
         "unstructured uniform", "graded 1 -> 0.5 at edge"]
res = {}
for sig in (3.0, 6.0):
    print(f"--- sigma = {sig}")
    for g, nm in zip(grids, names):
        o = B.run(g, sig=sig, label=nm); B.show(o); res[(nm, sig)] = o
pickle.dump(res, open('res_main.pkl', 'wb'))
for g, nm in zip(grids, names):
    if g.kind == 'fem': print(f"{nm:28s} triangles={g.ntri} min/max angle = {g.min_angle:.1f} / {g.max_angle:.1f} deg")
