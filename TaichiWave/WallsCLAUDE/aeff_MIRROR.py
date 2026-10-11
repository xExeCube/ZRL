# effective Dirichlet-disc radius of the pinned sq5 single cell / 3x3 block from scatter_sq5 (30/09)
import math, sys
from scipy.optimize import brentq
sys.path.insert(0, r'F:\FDrive-Storage\Claude Local\FormConstantsCLAUDE\TaichiWaveCLAUDE\ResearchCLAUDE\WallsCLAUDE')
import scatter_MIRROR as S
data = {'cell P': {8: 2.03165, 16: 2.62988, 32: 3.64291}, 'block3 P': {8: 7.70854, 16: 8.91318, 32: 10.87168},
        'cell R0 (Neumann)': {16: 0.09767, 32: 0.01197}, 'block3 R0 (Neumann)': {16: 2.90165, 32: 0.59737}}
for name, d in data.items():
    kind = 'N' if 'R0' in name else 'D'
    out = []
    for lam, sig in d.items():
        k = 2 * math.pi / lam
        a = brentq(lambda a: S.sigma_cont(k * a, kind, k) - sig, 1e-4, 20)
        out.append(f'lam {lam}: a_eff {a:.3f}')
    print(name, ' | '.join(out))
