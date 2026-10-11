import sys
sys.path.insert(0, r'F:/FDrive-Storage/Claude Local/FormConstantsCLAUDE/TaichiWaveCLAUDE')
import membrane_core_MIRROR as mc, ledger_MIRROR as L
mc.init_taichi('cuda', quiet=True)
for src in ('beam', 'packet'):
    for lat, sh, bc in (('sq', 'square', 'dirichlet'), ('sq', 'circle', 'mur'), ('tri', 'n6', 'dirichlet')):
        m = mc.Membrane(mc.Params(N=257, lattice=lat, shape=sh, bc=bc, src=src, freq=0.04, beam_angle=0, sx=-0.3))
        m.step(1200)
        rows = L.build_ledger(m)
        print(src, lat, sh, bc, ' | '.join(f'{r.key}:{r.badge}' for r in rows))
