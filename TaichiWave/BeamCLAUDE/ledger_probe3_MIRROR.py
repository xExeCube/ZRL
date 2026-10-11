import sys
sys.path.insert(0, r'F:/FDrive-Storage/Claude Local/FormConstantsCLAUDE/TaichiWaveCLAUDE')
import membrane_core_MIRROR as mc, ledger_MIRROR as L
mc.init_taichi('cuda', quiet=True)
for st in (5, 9):
    for sh, ang in (('square', 0), ('square', 30), ('circle', 30), ('n6', 17)):
        m = mc.Membrane(mc.Params(N=257, lattice='sq', stencil=st, shape=sh, src='packet', beam_angle=ang, sx=-0.3))
        m.step(1500)
        r = [r for r in L.build_ledger(m) if r.key == 'energy'][0]
        print(st, sh, ang, r.badge, r.value)
