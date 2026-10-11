import sys
sys.path.insert(0, r'F:/FDrive-Storage/Claude Local/FormConstantsCLAUDE/TaichiWaveCLAUDE')
import membrane_core_MIRROR as mc, ledger_MIRROR as L
mc.init_taichi('cuda', quiet=True)
for src, kw in (('packet', {}), ('pulse', {}), ('packet', dict(pk_lambda=40, pk_len=20, beam_width=20))):
    m = mc.Membrane(mc.Params(N=257, lattice='tri', shape='n6', src=src, beam_angle=0, sx=-0.3, **kw))
    m.step(1200)
    r = [r for r in L.build_ledger(m) if r.key == 'energy'][0]
    print(src, kw, r.badge, r.value)
