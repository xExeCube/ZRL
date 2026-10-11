import sys, math
sys.path.insert(0, r'F:/FDrive-Storage/Claude Local/FormConstantsCLAUDE/TaichiWaveCLAUDE')
import beam_tests_MIRROR as bt, membrane_core_MIRROR as mc
mc.init_taichi('cuda', quiet=True)
for ang in (0, 60, 120, 180, 240):
    for T, N in ((400, 701), (520, 701), (520, 1001)):
        m, I = bt.run_beam_intensity('tri', 5, 12, ang, 8, True, T=T, N=N)
        print(ang, T, N, round(10*math.log10(bt.directivity(m, I, ang)),1), 'R', round(m.inner_R(),1), m.beam['info']['n_front'], m.beam['info']['n_back'])
