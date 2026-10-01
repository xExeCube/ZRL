"""GPU cost of absorbing boundaries inside a fused K-step kernel shaped like the core's k_run.

Self-contained (does not import the core, which other agents are editing): the same storage
(U[3, N, N] rotated by slot, CC per-cell Co^2, u8 mask M), the same 5-point update with the
app's damping form, K = 8 steps unrolled per launch. Variants:
  plain      Dirichlet ring, scalar damping (what the app runs today for 'dirichlet')
  mur1       + the app's parallel Mur1 ring pass (corners 0)
  rule       + a generic local-ABC ring pass: a 5-node x 3-level coefficient table plus the
             tangential term on levels n (covers Mur1, Mur2, Higdon order 2, composed Liao
             order 2 -- all fit the three stored levels) + diagonal-Mur1 corners
  sp_arr     sponge: a per-cell damping array D (same float type as the field) replaces d
  sp_code    sponge: the damping LEVEL is packed into the mask byte (0 outside, 1 plain,
             2..L+1 ramp), d = LUT[M]: zero extra memory traffic
  sp_rect    sponge: depth from the array edge, d computed arithmetically (rectangle only)
  sp_ctr     sp_code with the CENTRED damping form (u_prev (1 - d/2), then / (1 + d/2)), which
             keeps the CFL limit 1/sqrt2 (the app form lowers it to sqrt(0.5 - d/4); added 30/09)
  sp_ctr_mul the same with the division precomputed: a second LUT holds 1/(1 + d/2)
             (30/09: the f64 divide made sp_ctr x2 at N = 4097 -- consumer GPUs run f64 at a
             small fraction of the f32 rate, and a divide is many f64 operations)
  cpml       convolutional PML in an L-frame: a face pass over the frame (memory at faces),
             then the cell pass with memory at cells inside the frame (aux arrays full-size
             here for simplicity; only frame cells are touched)
    python gpu_cost_MIRROR.py --arch cuda --f32 --N 1025 4097
"""
import argparse
import math
import time

import numpy as np
import taichi as ti

V_PLAIN, V_MUR1, V_RULE, V_SPARR, V_SPCODE, V_SPRECT, V_CPML, V_SPCTR, V_SPCTRM = range(9)
NAMES = ['plain', 'mur1', 'rule', 'sp_arr', 'sp_code', 'sp_rect', 'cpml', 'sp_ctr', 'sp_ctr_mul']
NV = len(NAMES)


@ti.kernel
def k_run(U: ti.types.ndarray(ndim=3), M: ti.types.ndarray(ndim=2), CC: ti.types.ndarray(ndim=2),
          D: ti.types.ndarray(ndim=2), LUT: ti.types.ndarray(ndim=1), LUT2: ti.types.ndarray(ndim=1), RC: ti.types.ndarray(ndim=2),
          RD: ti.types.ndarray(ndim=2),
          MX1: ti.types.ndarray(ndim=2), MY1: ti.types.ndarray(ndim=2),
          MX2: ti.types.ndarray(ndim=2), MY2: ti.types.ndarray(ndim=2),
          BC: ti.types.ndarray(ndim=1), AC: ti.types.ndarray(ndim=1),
          BF: ti.types.ndarray(ndim=1), AF: ti.types.ndarray(ndim=1),
          s0: ti.i32, K: ti.template(), var: ti.template(), d: float, kmur: float, kd: float,
          L: ti.i32, dmax: float, pw: float):
    n = U.shape[1]
    for q in ti.static(range(K)):
        if ti.static(var == V_CPML):
            # face pass over the frame: x-faces i+1/2 with i < L or i >= n-1-L; y likewise
            for s, t in ti.ndrange(2 * L, n):
                b = (s0 + q + 1) % 3
                i = s if s < L else n - 2 - (s - L)
                MX1[i, t] = BF[i] * MX1[i, t] + AF[i] * (U[b, i + 1, t] - U[b, i, t])
                MY1[t, i] = BF[i] * MY1[t, i] + AF[i] * (U[b, t, i + 1] - U[b, t, i])
        for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
            a, b, c = (s0 + q) % 3, (s0 + q + 1) % 3, (s0 + q + 2) % 3
            m = M[i, j]
            if m == 0:
                U[c, i, j] = 0.0
            else:
                u = U[b, i, j]
                lap = U[b, i - 1, j] + U[b, i + 1, j] + U[b, i, j - 1] + U[b, i, j + 1] - 4 * u
                dd = d
                if ti.static(var == V_SPARR):
                    dd = D[i, j]
                elif ti.static(var == V_SPCODE):
                    dd = LUT[ti.cast(m, ti.i32)]
                elif ti.static(var == V_SPRECT):
                    dep = ti.min(ti.min(i, j), ti.min(n - 1 - i, n - 1 - j))
                    if dep < L:
                        dd = dmax * ti.pow((L - dep) / L, pw)
                elif ti.static(var == V_CPML):
                    inx = i < L or i >= n - 1 - L
                    iny = j < L or j >= n - 1 - L
                    if inx or iny:
                        qxp = U[b, i + 1, j] - u
                        qxm = u - U[b, i - 1, j]
                        qyp = U[b, i, j + 1] - u
                        qym = u - U[b, i, j - 1]
                        if inx:
                            qxp += MX1[i, j]
                            qxm += MX1[i - 1, j]
                        if iny:
                            qyp += MY1[i, j]
                            qym += MY1[i, j - 1]
                        dqx = qxp - qxm
                        dqy = qyp - qym
                        lx = dqx
                        ly = dqy
                        if inx:
                            MX2[i, j] = BC[i] * MX2[i, j] + AC[i] * dqx
                            lx += MX2[i, j]
                        if iny:
                            MY2[i, j] = BC[j] * MY2[i, j] + AC[j] * dqy
                            ly += MY2[i, j]
                        lap = lx + ly
                if ti.static(var == V_SPCTRM):
                    mi = ti.cast(m, ti.i32)
                    ia = LUT2[mi]                     # 1 / (1 + d/2); no divide in the loop
                    U[c, i, j] = (2 * u + CC[i, j] * lap) * ia - (1 - 0.5 * LUT[mi]) * ia * U[a, i, j]
                elif ti.static(var == V_SPCTR):
                    dd = LUT[ti.cast(m, ti.i32)]
                    U[c, i, j] = (2 * u - (1 - 0.5 * dd) * U[a, i, j] + CC[i, j] * lap) / (1 + 0.5 * dd)
                else:
                    U[c, i, j] = (2 * u - U[a, i, j] + CC[i, j] * lap) - dd * (u - U[a, i, j])
        if ti.static(var == V_MUR1):
            for t in range(1, n - 1):
                b, c = (s0 + q + 1) % 3, (s0 + q + 2) % 3
                U[c, t, 0] = U[b, t, 1] + kmur * (U[c, t, 1] - U[b, t, 0])
                U[c, t, n - 1] = U[b, t, n - 2] + kmur * (U[c, t, n - 2] - U[b, t, n - 1])
                U[c, 0, t] = U[b, 1, t] + kmur * (U[c, 1, t] - U[b, 0, t])
                U[c, n - 1, t] = U[b, n - 2, t] + kmur * (U[c, n - 2, t] - U[b, n - 1, t])
            for t in range(1):
                c = (s0 + q + 2) % 3
                U[c, 0, 0] = 0.0
                U[c, n - 1, 0] = 0.0
                U[c, 0, n - 1] = 0.0
                U[c, n - 1, n - 1] = 0.0
        elif ti.static(var == V_RULE):
            for side, t in ti.ndrange(4, (1, n - 1)):
                lv = ti.Vector([(s0 + q + 2) % 3, (s0 + q + 1) % 3, (s0 + q) % 3])
                acc = 0.0
                for aa in ti.static(range(5)):
                    for bb in ti.static(range(3)):
                        if ti.static(aa > 0 or bb > 0):
                            ii, jj = aa, t
                            if side == 1:
                                ii, jj = n - 1 - aa, t
                            elif side == 2:
                                ii, jj = t, aa
                            elif side == 3:
                                ii, jj = t, n - 1 - aa
                            acc += RC[aa, bb] * U[lv[bb], ii, jj]
                            if ti.static(bb == 1 and aa < 2):
                                di, dj = 0, 1
                                if side >= 2:
                                    di, dj = 1, 0
                                acc += RD[aa, bb] * (U[lv[bb], ii - di, jj - dj] - 2 * U[lv[bb], ii, jj]
                                                     + U[lv[bb], ii + di, jj + dj])
                val = -acc / RC[0, 0]
                if side == 0:
                    U[lv[0], 0, t] = val
                elif side == 1:
                    U[lv[0], n - 1, t] = val
                elif side == 2:
                    U[lv[0], t, 0] = val
                else:
                    U[lv[0], t, n - 1] = val
            for t in range(1):
                b, c = (s0 + q + 1) % 3, (s0 + q + 2) % 3
                U[c, 0, 0] = U[b, 1, 1] + kd * (U[c, 1, 1] - U[b, 0, 0])
                U[c, n - 1, 0] = U[b, n - 2, 1] + kd * (U[c, n - 2, 1] - U[b, n - 1, 0])
                U[c, 0, n - 1] = U[b, 1, n - 2] + kd * (U[c, 1, n - 2] - U[b, 0, n - 1])
                U[c, n - 1, n - 1] = U[b, n - 2, n - 2] + kd * (U[c, n - 2, n - 2] - U[b, n - 1, n - 1])
        else:
            for t in range(n):
                c = (s0 + q + 2) % 3
                U[c, t, 0] = 0.0
                U[c, t, n - 1] = 0.0
                U[c, 0, t] = 0.0
                U[c, n - 1, t] = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='cuda')
    ap.add_argument('--f32', action='store_true')
    ap.add_argument('--N', type=int, nargs='+', default=[1025, 4097])
    ap.add_argument('--L', type=int, default=20)
    ap.add_argument('--reps', type=int, default=5)
    args = ap.parse_args()
    fdt = ti.f32 if args.f32 else ti.f64
    ti.init(arch=getattr(ti, args.arch), default_fp=fdt, log_level=ti.ERROR)
    npdt = np.float32 if args.f32 else np.float64
    Co, L = 0.5, args.L
    K = 8
    res = {}
    for N in args.N:
        U = ti.ndarray(fdt, (3, N, N))
        M = ti.ndarray(ti.u8, (N, N))
        CC = ti.ndarray(fdt, (N, N))
        D = ti.ndarray(fdt, (N, N))
        LUT = ti.ndarray(fdt, (256,))
        LUT2 = ti.ndarray(fdt, (256,))
        RC = ti.ndarray(fdt, (5, 3))
        RD = ti.ndarray(fdt, (5, 3))
        MX1, MY1, MX2, MY2 = (ti.ndarray(fdt, (N, N)) for _ in range(4))
        BC, AC, BF, AF = (ti.ndarray(fdt, (N,)) for _ in range(4))
        # host set-up
        ii = np.arange(N)
        dep = np.minimum(np.minimum(ii[:, None], ii[None, :]), np.minimum(N - 1 - ii[:, None], N - 1 - ii[None, :]))
        code = np.where(dep < L, 2 + (L - 1 - np.minimum(dep, L - 1)), 1).astype(np.uint8)
        code[dep == 0] = 0
        M.from_numpy(code)
        CC.from_numpy(np.full((N, N), Co * Co, npdt))
        D.from_numpy(np.where(dep < L, 0.5 * ((L - dep) / L) ** 2, 0.0).astype(npdt))
        lut = np.zeros(256, npdt)
        lut[2:L + 2] = 0.5 * ((np.arange(L) + 1) / L) ** 2
        LUT.from_numpy(lut)
        LUT2.from_numpy((1 / (1 + 0.5 * lut)).astype(npdt))
        rc = np.zeros((5, 3), npdt)
        k = (Co - 1) / (Co + 1)
        g, e = 2 / (Co + 1), Co * Co / (2 * (Co + 1))
        rc[0, 0], rc[1, 2], rc[1, 0], rc[0, 2], rc[0, 1], rc[1, 1] = 1, 1, -k, -k, -g, -g   # Mur2
        rd = np.zeros((5, 3), npdt)
        rd[0, 1] = rd[1, 1] = -e
        RC.from_numpy(rc)
        RD.from_numpy(rd)
        smax = 2.0 * 4 * math.log(1e6) / (2 * L)
        sc = smax * np.clip(np.maximum(L - ii, ii - (N - 1 - L)) / L, 0, 1) ** 3
        sf = smax * np.clip(np.maximum(L - (ii + 0.5), (ii + 0.5) - (N - 1 - L)) / L, 0, 1) ** 3
        bcv, bfv = np.exp(-sc * Co), np.exp(-sf * Co)
        BC.from_numpy(bcv.astype(npdt))
        AC.from_numpy((bcv - 1).astype(npdt))
        BF.from_numpy(bfv.astype(npdt))
        AF.from_numpy((bfv - 1).astype(npdt))
        kd = (Co - math.sqrt(2)) / (Co + math.sqrt(2))
        steps = 800 if N <= 1025 else (200 if N <= 4097 else 80)
        # Compile every variant first, then time them INTERLEAVED (round r runs all 8 in turn)
        # and keep each one's best round: a first version (29/09) timed each variant in one
        # block and read the GPU's clock/boost and other agents' load as differences of up to
        # x2 between variants (30/09: sp_ctr "x2.22" at N=4097, sp_code "x0.87" at N=1025).
        args_of = lambda s, v: (U, M, CC, D, LUT, LUT2, RC, RD, MX1, MY1, MX2, MY2, BC, AC, BF, AF,
                                (s * K) % 3, K, v, 0.0, k, kd, L, 0.5, 2.0)
        comp = {}
        for v in range(NV):
            t0 = time.perf_counter()
            k_run(*args_of(0, v))
            ti.sync()
            comp[v] = time.perf_counter() - t0
        best = {v: 1e9 for v in range(NV)}
        fin = {}
        for r in range(args.reps):
            for v in range(NV):
                u0 = np.zeros((3, N, N), npdt)
                u0[:, N // 2, N // 2] = 1.0
                U.from_numpy(u0)
                for A_ in (MX1, MY1, MX2, MY2):
                    A_.from_numpy(np.zeros((N, N), npdt))
                ti.sync()
                t0 = time.perf_counter()
                for s in range(steps // K):
                    k_run(*args_of(s, v))
                ti.sync()
                best[v] = min(best[v], (time.perf_counter() - t0) / steps)
                fin[v] = bool(np.all(np.isfinite(U.to_numpy()[:, ::64, ::64])))
        for v in range(NV):
            res[(N, NAMES[v])] = best[v]
            print('N=%5d %-8s %8.1f us/step  (x%.2f of plain)  compile %.1fs  finite=%s'
                  % (N, NAMES[v], best[v] * 1e6, best[v] / best[0], comp[v], fin[v]), flush=True)

if __name__ == '__main__':
    main()
