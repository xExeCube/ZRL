import math, pickle, numpy as np
import gridlib_MIRROR as G, circle_bench_MIRROR as B
R = 60.0; res = pickle.load(open('res_main.pkl','rb'))
V = G.penrose_vertices(20.0)
# sanity: all Delaunay edges in the interior should have length 1 (rhomb sides) or short diagonals
from scipy.spatial import Delaunay
T = Delaunay(V).simplices; E = np.unique(np.sort(np.r_[T[:,[0,1]],T[:,[1,2]],T[:,[0,2]]],1),axis=0)
c = (V[E[:,0]]+V[E[:,1]])/2; Ls = np.linalg.norm(V[E[:,0]]-V[E[:,1]],axis=1)[np.hypot(*c.T) < 14]
print("interior edge lengths:", np.unique(np.round(Ls, 4)), " (1 = side, 0.618 = thin short diag, 1.1756 = thick short diag)")
g = G.penrose_fitted(R)
print(f"Penrose mesh: n={g.n} tris={g.ntri} min/max angle {g.min_angle:.1f}/{g.max_angle:.1f}")
for sig in (3.0, 6.0):
    o = B.run(g, sig=sig); B.show(o); res[(g.name, sig)] = o
pickle.dump(res, open('res_main.pkl','wb'))
