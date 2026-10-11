import sys, time, math
sys.path.insert(0, r'F:/FDrive-Storage/Claude Local/FormConstantsCLAUDE/TaichiWaveCLAUDE')
import numpy as np
import membrane_core_MIRROR as mc
mc.init_taichi('cuda', quiet=True)
for lat, st in (('sq',5),('sq',9),('tri',5)):
    for src in ('beam','packet'):
        p = mc.Params(N=301, lattice=lat, stencil=st, src=src, freq=0.04, Co=0.5, beam_angle=30, beam_width=8,
                      bc='dirichlet', pk_lambda=14, pk_len=10)
        m = mc.Membrane(p)
        t=time.time(); m.step(200); u = m.field()
        info = m.beam['info'] if src=='beam' else m.packet_info
        print(lat, st, src, 'max', float(np.abs(u).max()), 'finite', np.isfinite(u).all(), {k: (round(v,3) if isinstance(v,float) else v) for k,v in info.items() if k in ('theta_k','lam','n_front','n_back','n_clipped','vg','in_array','gain')}, round(time.time()-t,2))
