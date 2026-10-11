"""Circle-fidelity benchmark, energy-normalised.
eps_ang(t): fraction of the wave's energy that is NOT radially symmetric (0 for a true circle, any t).
eps_tot(t): energy of (u - exact Bessel solution) / total energy."""
import math, time, numpy as np
from scipy.special import j0, j1, jn_zeros
from scipy.interpolate import make_lsq_spline
import gridlib_MIRROR as G

def radial_fit(r, u, w):
    o = np.argsort(r); rs, us, ws = r[o], u[o], w[o]
    inner = np.r_[np.arange(1.0, 4.0, 1.0), np.arange(4.0, rs[-1] - 0.3, 0.5)]
    t = np.r_[[rs[0]]*4, inner, [rs[-1]+1e-9]*4]
    out = np.empty_like(u); out[o] = make_lsq_spline(rs, us, t, k=3, w=ws)(rs); return out

class Exact:
    def __init__(s, r, Rref, sig):
        K = int(np.searchsorted(jn_zeros(0, 3000), 12*Rref/sig)) + 5
        s.jk = jn_zeros(0, K); s.R = Rref
        s.a = 2*sig**2/(Rref**2*j1(s.jk)**2)*np.exp(-0.5*(s.jk*sig/Rref)**2)
        s.B = j0(np.outer(np.minimum(r, Rref), s.jk/Rref))
    def __call__(s, t): return s.B @ (s.a*np.cos(s.jk*t/s.R))

def energy(g, un, um, dt):
    v = (un - um)/dt; ub = 0.5*(un + um)
    return 0.5*(g.w*v*v).sum() + 0.5*(-(g.w*ub*(g.L @ ub)).sum())

def run(g, sig=3.0, cycles=10, dt=None, per_cycle=8, exact=True, Tref=None, label=None):
    if not hasattr(g, 'dt_max'): g.eig_extremes()
    dt = dt if dt is not None else min(0.5, 0.8*g.dt_max)
    Rref = g.R_eff; T = Tref if Tref else 2*Rref
    ex = Exact(g.r, Rref, sig) if exact else None
    u = np.exp(-g.r**2/(2*sig**2)); um = u + 0.5*dt*dt*(g.L @ u)
    nsteps = int(round(cycles*T/dt))
    samp = {int(round(q*T/per_cycle/dt)): q/per_cycle for q in range(1, cycles*per_cycle+1)}
    out = dict(t=[], eang=[], etot=[])
    if exact:
        a, b = ex(0.37*T), ex(0.37*T - dt)
        aa, bb = a - radial_fit(g.r, a, g.w), b - radial_fit(g.r, b, g.w)
        floor = energy(g, aa, bb, dt)/energy(g, a, b, dt)
    else: floor = np.nan
    t0 = time.time(); E0 = None
    for n in range(1, nsteps+1):
        u, um = 2*u - um + dt*dt*(g.L @ u), u
        if n in samp:
            E = energy(g, u, um, dt); E0 = E0 or E
            ua, uma = u - radial_fit(g.r, u, g.w), um - radial_fit(g.r, um, g.w)
            out['t'].append(samp[n]); out['eang'].append(energy(g, ua, uma, dt)/E)
            if exact:
                ea, eb = u - ex(n*dt), um - ex((n-1)*dt)
                out['etot'].append(energy(g, ea, eb, dt)/E)
    for k in ('t','eang','etot'): out[k] = np.array(out[k])
    out.update(name=label or g.name, n=g.n, dt=dt, dtmax=g.dt_max, Reff=Rref, Rgeom=g.R_geom, floor=floor,
               secs=time.time()-t0, minang=getattr(g,'min_angle',np.nan), maxang=getattr(g,'max_angle',np.nan), sig=sig)
    return out

def show(o, ms=(1,2,5,10)):
    pick = lambda arr, m: arr[np.argmin(np.abs(o['t']-m))] if len(arr) else np.nan
    s = f"{o['name']:30s} n={o['n']:6d} dt={o['dt']:.3f}/{o['dtmax']:.3f} dR={o['Reff']-o['Rgeom']:+.3f} floor={o['floor']:.0e} | eang " \
        + " ".join(f"{pick(o['eang'],m):.2e}" for m in ms)
    if len(o['etot']): s += " | etot " + " ".join(f"{pick(o['etot'],m):.2e}" for m in ms)
    print(s + f"  ({o['secs']:.0f}s)", flush=True)
