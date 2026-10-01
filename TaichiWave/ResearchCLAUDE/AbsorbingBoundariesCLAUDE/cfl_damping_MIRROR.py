"""The damping form and the CFL limit (30/09/2026). The app damps with a BACKWARD difference,
    u+ = 2u - u- + Co^2 lap u - d (u - u-),
whose amplification polynomial z^2 - (2 - d - Co^2 lam) z + (1 - d) has both roots in the unit
disc only for Co^2 lam <= 4 - 2d, i.e. Co_max(d) = Co_max(0) sqrt(1 - d/2) (sq5: sqrt(0.5 - d/4)).
The CENTRED form (1 + d/2) u+ = 2u - (1 - d/2) u- + Co^2 lap u keeps Co_max(0) for any d <= 2.
Check: white noise on a 64 x 64 torus, 3000 steps, just below / just above the predicted limit.
    python cfl_damping_MIRROR.py
"""
import numpy as np
def run(Co, d, form, steps=3000):
    rng = np.random.default_rng(1); u = rng.uniform(-1,1,(64,64)); up = u.copy()
    for _ in range(steps):
        lap = np.roll(u,1,0)+np.roll(u,-1,0)+np.roll(u,1,1)+np.roll(u,-1,1)-4*u
        if form == 'app': un = 2*u - up + Co*Co*lap - d*(u-up)
        else: un = (2*u - up*(1-0.5*d) + Co*Co*lap)/(1+0.5*d)
        up, u = u, un
    return np.abs(u).max()
for d in (0.02, 0.2, 0.5):
    lim = np.sqrt(0.5 - d/4)
    for Co in (lim - 0.005, lim + 0.005):
        print('d=%.2f limit %.4f  Co=%.4f  app %.2e  ctr %.2e' % (d, lim, Co, run(Co,d,'app'), run(Co,d,'ctr')))
