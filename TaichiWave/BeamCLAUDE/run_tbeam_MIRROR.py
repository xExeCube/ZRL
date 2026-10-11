import sys, os
H = r'F:/FDrive-Storage/Claude Local/FormConstantsCLAUDE/TaichiWaveCLAUDE'
sys.path.insert(0, H); os.chdir(H)
import app_tests_MIRROR as T
T.mc.init_taichi('cuda', quiet=True)
os.makedirs(T.OUT, exist_ok=True)
v = T.app.Viewer(T.app.default_args(N=257, shot='tests'))
ctx = T.Ctx(v.win); v.camera = ctx.camera
T.t_beam(v, ctx)
print([r[:2] for r in T.RESULTS])
