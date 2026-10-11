# v3 (01/10/2026): TRUE resolution test for the rotated-vs-axis square on sq5 -- same ka (side/lam fixed
# at 11/8), twice the cells per wavelength. The 30/09 note compared side 11 at lam 8 and lam 16, which
# changes ka as well as resolution.
import sys, math, time
import numpy as np
sys.path.insert(0, r'F:/FDrive-Storage/Claude Local/FormConstantsCLAUDE/TaichiWaveCLAUDE/ResearchCLAUDE/WallsCLAUDE')
import walls_lib_MIRROR as W
import scatter_MIRROR as S
latn = sys.argv[1] if len(sys.argv) > 1 else 'sq5'
scale = int(sys.argv[2]) if len(sys.argv) > 2 else 2
lat = W.LATS[latn]
side = float(sys.argv[3]) if len(sys.argv) > 3 else 11.0 * scale
lam = 8.0 * side / 11.0           # same ka as side 11 at lam 8 (01/10: side 22 rasterised to 441 vs 485 cells; use odd sides)
rex = 20.0 * scale
t0 = time.perf_counter()
stB = S.Setup(lat, lam, 0.0, size=18.0 * scale)
for ang in (30.0, 45.0):
    stA = S.Setup(lat, lam, -math.radians(ang), size=18.0 * scale)
    solB = S.obstacle(lat, stB.X, stB.Y, 'square', ang, side)
    solA = S.obstacle(lat, stA.X, stA.Y, 'square', 0.0, side)
    for rule in ('P', 'R0'):
        sA, eA, edA = stA.scatter(solA, rule, r_ex=rex)
        sB, eB, edB = stB.scatter(solB, rule, r_ex=rex)
        print(f'{latn} side {side:.0f} lam {lam} rot {ang:.0f} {rule:3}: sigma axis {sA:8.3f} rotated {sB:8.3f} ratio {sB/sA:.4f} '
              f'cells {int(solA.sum())}/{int(solB.sum())} edge {max(edA, edB):.0e}  [{time.perf_counter()-t0:.0f} s]', flush=True)
    rd = side / math.sqrt(math.pi)
    dA, dB = stA.R < rd, stB.R < rd
    sA, _, _ = stA.scatter(dA, 'R0', r_ex=rex); sB, _, _ = stB.scatter(dB, 'R0', r_ex=rex)
    print(f'   R0 DISC reference ratio {sB/sA:.4f} (cells {int(dA.sum())})', flush=True)
