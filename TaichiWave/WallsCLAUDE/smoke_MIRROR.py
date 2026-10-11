import sys, time, math
sys.path.insert(0, r'F:\FDrive-Storage\Claude Local\FormConstantsCLAUDE\TaichiWaveCLAUDE\ResearchCLAUDE\WallsCLAUDE')
import scatter_MIRROR as S, walls_lib_MIRROR as W
t0=time.perf_counter()
st=S.Setup(W.SQ5, 4, 0.0, size=18.0)
print('n',st.n,'steps',st.steps,'inc run',round(st.t_run,1),'s')
sol=S.obstacle(W.SQ5, st.X, st.Y,'square',0.0)
for r in ('P','R0'):
    print(r, st.scatter(sol,r,20.0)[::2])
k=2*math.pi/4
print('disc D', S.sigma_cont(k*12/math.sqrt(math.pi),'D',k))
print(time.perf_counter()-t0)
