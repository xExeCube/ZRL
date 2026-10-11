import math, pickle, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.tri as mtri
import gridlib_MIRROR as G, circle_bench_MIRROR as B
BG, PANEL, FG = '#000000', '#131313', '#FFFFFF'
plt.rcParams.update({'text.color':FG,'axes.labelcolor':FG,'xtick.color':FG,'ytick.color':FG,'axes.edgecolor':'#502FA5',
                     'font.size':10,'axes.facecolor':PANEL,'figure.facecolor':BG,'savefig.facecolor':BG})
res = pickle.load(open('res_main.pkl','rb')); conv = pickle.load(open('res_conv.pkl','rb')); jit = pickle.load(open('res_jit.pkl','rb'))
col = {"sq5 (app)":'#E22211', "sq9 (app)":'#F59E0B', "tri (app)":'#485FC7', "sq + edge triangles":'#EC4899',
       "tri + edge triangles":'#009E86', "hex shells -> circles":'#22D3EE', "Penrose + edge triangles":'#A78BFA',
       "unstructured uniform":'#9CA3AF', "graded 2:1 toward edge":'#FDE68A'}
fig = plt.figure(figsize=(15, 11.5))
gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.20, left=0.06, right=0.98, top=0.93, bottom=0.06, height_ratios=[1, 1.1])

# (A) continuum: 2D disc vs 3D ball
ax = fig.add_subplot(gs[0,0]); m = np.arange(1,9)
d2 = [0.9795,1.9985,1.0633,0.0152,0.9015,1.9869,1.1485,0.0589]
ax.bar(m-0.18, d2, 0.36, color='#485FC7', label='2D disc (exact Bessel series)')
ax.bar(m+0.18, np.zeros(8)+0.004, 0.36, color='#009E86', label='3D ball (exact) — 0 to machine precision')
for x, v in zip(m, d2): ax.text(x-0.18, v+0.04, f"{v:.2f}" if v > 0.1 else f"{v:.3f}", ha='center', fontsize=8)
ax.set_xlabel('round trip m  (t = 2mR)'); ax.set_ylabel('‖u(t) − u(0)‖ / ‖u(0)‖')
ax.set_title('A · No grid at all: does a perfect circle repeat each cycle?', color=FG, loc='left')
ax.legend(frameon=False, fontsize=9, loc='center right', bbox_to_anchor=(1.0, 0.62)); ax.set_ylim(0, 2.35)
ax.text(0.02, 0.97, 'Each pass: wall flip (180°) + focus Gouy shift (90° in 2D, 180° in 3D)\n2D returns only every 4th cycle, and drifts (Bessel zeros not evenly spaced)',
        transform=ax.transAxes, va='top', fontsize=8.5, color='#C8C8C8')

# (B) the error element vs cycles
ax = fig.add_subplot(gs[0,1])
for nm, c in col.items():
    o = res[(nm if nm != 'tri (app)' else 'tri (app, pointy-top)', 6.0)]; ax.semilogy(o['t'], np.maximum(o['eang'], 1e-9), color=c, lw=1.8 if 'edge' in nm or 'tri' in nm else 1.4, label=nm.replace(' (app)', '  [app]'))
ax.set_xlabel('round trips'); ax.set_ylabel('ε_ang  (energy fraction not radially symmetric)')
ax.set_title('B · Circle error element per flavor  (R = 60 cells, σ = 6, Co = 0.5)', color=FG, loc='left')
ax.legend(frameon=False, fontsize=7.5, ncol=3, loc='lower center'); ax.set_ylim(1e-9, 1e-1); ax.grid(alpha=0.15)

# (C) the noise itself at cycle 5 (sigma = 3, the app default), common colour scale
gsC = gs[1,0].subgridspec(1, 3, wspace=0.02)
fields = []
for nm, mk in [("sq5  [app]", lambda: G.lattice('sq5', 60.0)), ("tri  [app]", lambda: G.lattice('tri', 60.0)), ("tri + edge triangles", lambda: G.tri_fitted(60.0))]:
    g = mk(); g.eig_extremes(); dt = 0.5; sig = 3.0
    u = np.exp(-g.r**2/(2*sig**2)); um = u + 0.5*dt*dt*(g.L @ u)
    for n in range(int(round(5.25*2*g.R_eff/dt))): u, um = 2*u - um + dt*dt*(g.L @ u), u
    ua = u - B.radial_fit(g.r, u, g.w); fields.append((nm, g, u, ua))
vmax = np.percentile(np.abs(fields[0][3]), 99.5)
for k, (nm, g, u, ua) in enumerate(fields):
    a = fig.add_subplot(gsC[0,k]); tr = mtri.Triangulation(g.pos[:,0], g.pos[:,1])
    a.tripcolor(tr, ua, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='gouraud')
    a.add_patch(plt.Circle((0,0), 60, fill=False, ec='#485FC7', lw=0.8)); a.set_aspect('equal'); a.axis('off')
    a.set_title(nm, color=FG, fontsize=10)
    a.text(0.5, -0.04, f"rms noise {np.sqrt((g.w*ua*ua).sum()/(g.w*u*u).sum())*100:.2g}% of field", transform=a.transAxes, ha='center', fontsize=8.5, color='#C8C8C8')
fig.text(0.06, 0.485, 'C · The "noise": field minus its radial average, 5¼ round trips, σ = 3 (app default), same colour scale',
         color=FG, fontsize=11)

# (D) convergence with N
ax = fig.add_subplot(gs[1,1]); Ns = np.array([64, 124, 244])
for nm in ["sq5 (app)","sq9 (app)","tri (app)","sq + edge triangles","tri + edge triangles","hex shells -> circles","Penrose + edge triangles"]:
    v = [conv[(nm,Rc)]['eang'][conv[(nm,Rc)]['t']>=1].mean() for Rc in (30.,60.,120.)]
    ax.loglog(Ns, v, 'o-', color=col[nm], lw=1.6, ms=5, label=nm.replace(' (app)','  [app]'))
for nm in jit:
    v = jit[nm]; ax.errorbar([124*1.04], [np.median(v)], yerr=[[np.median(v)-v.min()],[v.max()-np.median(v)]], fmt='none', ecolor=col[nm], capsize=3, lw=1)
ax.set_xlabel('N (grid resolution across the circle)'); ax.set_ylabel('ε_ang, averaged over cycles 1–5')
ax.set_title('D · Same drum (σ = R/10), finer N  ·  bars: spread over R ± 5 cells', color=FG, loc='left'); ax.set_xlim(50, 300)
ax.legend(frameon=False, fontsize=8, loc='lower left'); ax.grid(alpha=0.15, which='both')
fig.suptitle('Circle fidelity of grid flavors — centred pulse, clamped wall, no damping', color=FG, fontsize=14, x=0.06, ha='left')
import os
plt.savefig('circle_grid_fidelity_MIRROR.png', dpi=85); print('saved')
