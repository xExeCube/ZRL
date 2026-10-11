# 30/09: sanity of scatter part 2 (sq5, lam 16): edge-energy share, and a 45 deg / 15 deg rotation
import sys, math
sys.path.insert(0, r'F:\FDrive-Storage\Claude Local\FormConstantsCLAUDE\TaichiWaveCLAUDE\ResearchCLAUDE\WallsCLAUDE')
import scatter_MIRROR as S, walls_lib_MIRROR as W
lat = W.SQ5
for ang in (30.0, 15.0, 45.0):
    stB = S.Setup(lat, 16, 0.0, size=18.0)
    stA = S.Setup(lat, 16, -math.radians(ang), size=18.0)
    sB = S.obstacle(lat, stB.X, stB.Y, 'square', ang)
    sA = S.obstacle(lat, stA.X, stA.Y, 'square', 0.0)
    for rule in ('P', 'R0'):
        a = stA.scatter(sA, rule, 20.0); b = stB.scatter(sB, rule, 20.0)
        print(f'rot {ang:4}: {rule:3} sigma axis {a[0]:.3f} (edge {a[2]:.1e}) rotated {b[0]:.3f} (edge {b[2]:.1e}) ratio {b[0]/a[0]:.4f}  cells A {sA.sum()} B {sB.sum()}', flush=True)
