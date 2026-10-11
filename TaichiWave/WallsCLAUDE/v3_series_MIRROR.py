# v3 (01/10/2026): own continuum cylinder scattering widths; check the note's fitted radii + small-ka laws
import math, numpy as np
from scipy.special import jv, hankel1, jvp, h1vp
from scipy.optimize import brentq
def sig(ka, k, kind, N=40):
    s = 0.0
    for n in range(-N, N+1):
        r = jv(n, ka)/hankel1(n, ka) if kind == 'D' else jvp(n, ka)/h1vp(n, ka)
        s += abs(r)**2
    return 4/k*s
meas = {('cell','P'):{8:2.03165,16:2.62988,32:3.64291}, ('cell','R0'):{8:0.70722,16:0.09767,32:0.01197},
        ('block3','P'):{8:7.70854,16:8.91318,32:10.87168}, ('block3','R0'):{8:7.16147,16:2.90165,32:0.59737}}
for (ob, r), d in meas.items():
    kind = 'D' if r == 'P' else 'N'
    out = []
    for lam, s in d.items():
        k = 2*math.pi/lam
        a = brentq(lambda a: sig(k*a, k, kind) - s, 1e-4, 3.0)
        out.append(f'lam{lam}: a={a:.3f}')
    print(ob, r, kind, ' '.join(out))
# small-ka asymptotics
for ka in (0.1, 0.03):
    k = 1.0
    sD = sig(ka, k, 'D'); sN = sig(ka, k, 'N')
    aD = math.pi**2/(k*(math.log(ka/2)+0.5772156649)**2); aN = 3*math.pi**2/4*k**3*(ka/k)**4
    print(f'ka={ka}: D series {sD:.4g} vs pi^2/(k (ln(ka/2)+gamma)^2) {aD:.4g};  N series {sN:.4g} vs (3 pi^2/4) k^3 a^4 {aN:.4g}')
