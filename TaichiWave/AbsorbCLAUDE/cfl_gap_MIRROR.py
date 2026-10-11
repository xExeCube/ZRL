"""Fact-check 01/10/2026: d = 0.02, Co between the predicted app-form limit 0.7036 and 1/sqrt2.
The original check put the 'above' point at 0.7086, which is above 1/sqrt2 too (not discriminating)."""
import numpy as np
def run(Co, d, form, steps):
    rng = np.random.default_rng(1); u = rng.uniform(-1,1,(64,64)); up = u.copy(); out=[]
    for n in range(1, steps+1):
        lap = np.roll(u,1,0)+np.roll(u,-1,0)+np.roll(u,1,1)+np.roll(u,-1,1)-4*u
        if form == 'app': un = 2*u - up + Co*Co*lap - d*(u-up)
        else: un = (2*u - up*(1-0.5*d) + Co*Co*lap)/(1+0.5*d)
        up, u = u, un
        if n in (1000, 5000, 20000): out.append('%.2e' % np.abs(u).max())
    return out
for Co in (0.7030, 0.7050, 0.7065):
    print('d=0.02 Co=%.4f app %s  ctr %s' % (Co, run(Co, 0.02, 'app', 20000), run(Co, 0.02, 'ctr', 20000)))
