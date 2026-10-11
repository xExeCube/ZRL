# Does a perfect (continuum) circle give identical cycles? 2D disc vs 3D ball, centred Gaussian, Dirichlet wall.
import math, numpy as np
from scipy.special import j0, j1, jn_zeros
R, sig = 60.0, 3.0
r = np.linspace(0, R, 6001); wr = r.copy()                       # 2D area weight ~ r dr
K = 400; jk = jn_zeros(0, K)
a = 2*sig**2/(R**2*j1(jk)**2)*np.exp(-0.5*(jk*sig/R)**2)
B = j0(np.outer(r, jk/R))
u2 = lambda t: B @ (a*np.cos(jk*t/R))
# 3D: v = r u obeys the 1D string on [0,R]; modes sin(k pi r/R)/r, frequencies k pi/R (harmonic)
kk = np.arange(1, K+1); rr = np.where(r==0, 1e-12, r)
v0 = r*np.exp(-r**2/(2*sig**2))
bk = np.array([2/R*np.trapezoid(v0*np.sin(k*math.pi*r/R), r) for k in kk])
S3 = np.sin(np.outer(rr, kk*math.pi/R))/rr[:,None]
u3 = lambda t: S3 @ (bk*np.cos(kk*math.pi*t/R))
n2 = lambda f, w: math.sqrt(np.trapezoid(f*f*w, r))
U20, U30 = u2(0), u3(0)
print("check t=0 reconstruction: 2D", f"{n2(U20-np.exp(-r**2/18), wr)/n2(np.exp(-r**2/18), wr):.1e}",
      " 3D", f"{n2(U30-np.exp(-r**2/18), r*r)/n2(np.exp(-r**2/18), r*r):.1e}")
print(" m | 2D: |u(2mR)-u(0)|/|u(0)|  centre u(2mR)/u(0) | 3D: |u(2mR)-u(0)|/|u(0)|")
for m in range(1, 9):
    t = 2*m*R; A, Bv = u2(t), u3(t)
    print(f" {m} |  {n2(A-U20, wr)/n2(U20, wr):8.4f}                {A[0]/U20[0]:+8.4f}     |  {n2(Bv-U30, r*r)/n2(U30, r*r):.2e}")
# inharmonicity of the radial spectrum: zeros of J_nu, nu = D/2 - 1
from scipy.optimize import brentq
from scipy.special import jv
for D, nu in [(2,0.0),(3,0.5),(4,1.0),(5,1.5)]:
    z=[]; x=0.5
    while len(z)<6:
        if jv(nu,x)*jv(nu,x+0.01)<0: z.append(brentq(lambda s: jv(nu,s), x, x+0.01))
        x+=0.01
    d=np.diff(z)/math.pi
    print(f"D={D}: radial-mode spacings / pi = " + ", ".join(f"{s:.5f}" for s in d))
