# Cross-check: my sparse lattice solver vs the app's own Taichi core (CPU, f64), circle domain.
import math, numpy as np, sys
from gridlib_MIRROR import lattice, S3H
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))   # TaichiWaveCLAUDE
import membrane_core_MIRROR as mc
mc.init_taichi(arch='cpu', fp64=True, quiet=True)
N, K, Co = 101, 300, 0.5
for lat, st in [('sq',5), ('sq',9), ('tri',6)]:
    p = mc.Params(N=N, shape='circle', lattice=lat, stencil=(9 if st==9 else 5), Co=Co, src='pulse', sigma=3.0)
    m = mc.Membrane(p); R = m.dom_R(); c = m.c()
    m.step(K); F = m.field()
    kind = {5:'sq5', 9:'sq9', 6:'tri'}[st]
    g = lattice(kind, R)
    # node (a,b) -> app cell (a+c, b+c); recover (a,b) from positions
    if kind == 'tri': b = np.rint(g.pos[:,1]/S3H); a = np.rint(g.pos[:,0] - b*0.5)
    else: a, b = np.rint(g.pos[:,0]), np.rint(g.pos[:,1])
    u = np.exp(-(g.r**2)/(2*9.0)); up = u.copy()          # app start: both levels = bump (at rest)
    for _ in range(K):
        u, up = 2*u - up + Co*Co*(g.L @ u), u
    Fa = F[(a+c).astype(int), (b+c).astype(int)]
    inside_app = int(m.mask_np.sum())
    print(f"{kind}: R={R:.4f}  nodes mine={g.n} app={inside_app}  max|diff|={np.abs(Fa-u).max():.2e}  max|u|={np.abs(u).max():.3e}")
