import sys, math
sys.path.insert(0, r'F:/FDrive-Storage/Claude Local/FormConstantsCLAUDE/TaichiWaveCLAUDE')
import beam_tests_MIRROR as bt, beam_MIRROR as bm, membrane_core_MIRROR as mc
mc.init_taichi('cuda', quiet=True)
for mode in ('shift','slab'):
    for ramp in (2.0, 6.0):
        bm.BACK_MODE, bm.RAMP_PERIODS = mode, ramp
        out=[]
        for name, lat, st in bt.LATS:
            for ang in (0, 30, 45, 20):
                m, I = bt.run_beam_intensity(lat, st, 12, ang, 8, True)
                out.append(f'{name}{ang}:{10*math.log10(bt.directivity(m, I, ang)):.1f}')
        print(mode, ramp, ' '.join(out), flush=True)
