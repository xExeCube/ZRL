"""numpy (vectorised) and numba (compiled loop, 1 thread and parallel) versions of the same update."""
import sys, time, numpy as np
n, st, secs, which = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
c = (n - 1) / 2; R = 0.45 * n
j, i = np.mgrid[0:n, 0:n]; x, y = i - c, j - c
M = (x * x + y * y <= R * R); CC = np.full((n, n), 0.25); U1 = np.where(M, np.exp(-(x*x + y*y) / 18), 0.0); U0 = U1.copy(); U2 = np.zeros_like(U1)
if which == 'numpy':
    Mi = M[1:-1, 1:-1]; CCi = CC[1:-1, 1:-1]
    def step(U0, U1, U2):
        C = U1[1:-1, 1:-1]
        if st == 5: lap = U1[1:-1, :-2] + U1[1:-1, 2:] + U1[:-2, 1:-1] + U1[2:, 1:-1] - 4 * C
        elif st == 9:
            e = U1[1:-1, :-2] + U1[1:-1, 2:] + U1[:-2, 1:-1] + U1[2:, 1:-1]
            g = U1[:-2, :-2] + U1[:-2, 2:] + U1[2:, :-2] + U1[2:, 2:]
            lap = (2/3) * e + (1/6) * g - (10/3) * C
        else:   # rows = j, cols = i: k+1-n -> (j-1, i+1), k-1+n -> (j+1, i-1)
            s = U1[1:-1, :-2] + U1[1:-1, 2:] + U1[:-2, 1:-1] + U1[2:, 1:-1] + U1[:-2, 2:] + U1[2:, :-2]
            lap = (2/3) * (s - 6 * C)
        U2[1:-1, 1:-1] = np.where(Mi, 2 * C - U0[1:-1, 1:-1] + CCi * lap, 0.0)
else:
    import numba
    par = which == 'numba_par'
    @numba.njit(parallel=par, fastmath=False, cache=False)
    def step(U0, U1, U2):
        for jj in numba.prange(1, n - 1):
            for ii in range(1, n - 1):
                if not M[jj, ii]: U2[jj, ii] = 0.0; continue
                cc = U1[jj, ii]
                if st == 5: lap = U1[jj, ii-1] + U1[jj, ii+1] + U1[jj-1, ii] + U1[jj+1, ii] - 4 * cc
                elif st == 9:
                    e = U1[jj, ii-1] + U1[jj, ii+1] + U1[jj-1, ii] + U1[jj+1, ii]
                    g = U1[jj-1, ii-1] + U1[jj-1, ii+1] + U1[jj+1, ii-1] + U1[jj+1, ii+1]
                    lap = (2/3) * e + (1/6) * g - (10/3) * cc
                else:
                    s = U1[jj, ii-1] + U1[jj, ii+1] + U1[jj-1, ii] + U1[jj+1, ii] + U1[jj-1, ii+1] + U1[jj+1, ii-1]
                    lap = (2/3) * (s - 6 * cc)
                U2[jj, ii] = 2 * cc - U0[jj, ii] + CC[jj, ii] * lap
if secs < 0:
    for _ in range(int(-secs)): step(U0, U1, U2); U0, U1, U2 = U1, U2, U0
    print(f'{float((U1*U1).sum()):.12e}'); sys.exit()
for _ in range(3): step(U0, U1, U2); U0, U1, U2 = U1, U2, U0      # warm-up / JIT
steps = 0; t0 = time.perf_counter()
while True:
    step(U0, U1, U2); U0, U1, U2 = U1, U2, U0; steps += 1
    t = time.perf_counter() - t0
    if t >= secs: break
print(f'{steps * (n - 2) ** 2 / t:.4e}')
