"""Walls research (30/09/2026): what a mixed pinned/rigid wall costs in the fused Taichi kernel.

A PROTOTYPE, not the core: three sq9 updates of the same shape as membrane_core's k_run
(K = 8 steps unrolled per launch, U[3, N, N] rotated by slot, per-cell CC), timed at large N:

  A  'mask'  : the current rule. M[i, j] == 0 -> u = 0; else the plain 9-point stencil
               (pinned walls come for free: a masked neighbour holds 0).
  B  'word'  : a per-cell LINK WORD, precomputed on the host from the cell types:
                 bit 15      the cell is free (else u = 0)
                 bits 0..7   link k is active (pinned neighbours stay active: they hold 0)
                 bits 8..11  diagonal k is corner-blocked by a PINNED cell (-w u_i only)
               lap = sum_k b_k w_k (u_k - u_i) - sum_blocked w_d u_i
               Rigid links and rigid-blocked diagonals simply have their bit cleared.
  C  'types' : no precomputation: a per-cell type byte T (0 free, 1 pinned, 2 rigid); every
               link decides itself from the neighbour's type and, for a diagonal, the two
               edge cells it passes between (the rule of walls_lib_MIRROR.build_L, block='both').

B and C are checked against walls_lib_MIRROR's sparse operator (rule mix: pinned + rigid
objects, corner blocking) on a 96 x 96 scene, then timed.

    python linkmask_probe_MIRROR.py [--arch cuda|vulkan|cpu] [--f32] [--N 4097]
"""
import sys, os, time, argparse
import numpy as np
import taichi as ti

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walls_lib_MIRROR as W

ap = argparse.ArgumentParser()
ap.add_argument('--arch', default='cuda')
ap.add_argument('--f32', action='store_true')
ap.add_argument('--N', type=int, default=4097)
ap.add_argument('--reps', type=int, default=5)
args = ap.parse_args()
FP64 = not args.f32
ti.init(arch={'cuda': ti.cuda, 'vulkan': ti.vulkan, 'cpu': ti.cpu}[args.arch],
        default_fp=ti.f64 if FP64 else ti.f32, log_level=ti.ERROR)
VK = args.arch == 'vulkan'
MDT = ti.i32 if VK else ti.u8          # the core's Vulkan u8 fault (README) applies here too
WDT = ti.i32 if VK else ti.u16
NPF = np.float64 if FP64 else np.float32
NPM = np.int32 if VK else np.uint8
NPW = np.int32 if VK else np.uint16
K = 8
# link order: edges (-1,0) (+1,0) (0,-1) (0,+1); diagonals (-,-) (+,-) (-,+) (+,+) (the core's order)
EDGES = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGS = ((-1, -1), (1, -1), (-1, 1), (1, 1))


@ti.kernel
def k_mask(U: ti.types.ndarray(ndim=3), M: ti.types.ndarray(ndim=2), CC: ti.types.ndarray(ndim=2), s0: ti.i32):
    n = U.shape[1]
    for q in ti.static(range(K)):
        for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
            a, b, c = (s0 + q) % 3, (s0 + q + 1) % 3, (s0 + q + 2) % 3
            if M[i, j] == 0:
                U[c, i, j] = 0.0
            else:
                u = U[b, i, j]
                e = U[b, i - 1, j] + U[b, i + 1, j] + U[b, i, j - 1] + U[b, i, j + 1]
                dg = U[b, i - 1, j - 1] + U[b, i + 1, j - 1] + U[b, i - 1, j + 1] + U[b, i + 1, j + 1]
                lap = (2.0 / 3.0) * (e - 4 * u) + (1.0 / 6.0) * (dg - 4 * u)
                U[c, i, j] = 2 * u - U[a, i, j] + CC[i, j] * lap


@ti.kernel
def k_word(U: ti.types.ndarray(ndim=3), Wd: ti.types.ndarray(ndim=2), CC: ti.types.ndarray(ndim=2), s0: ti.i32):
    n = U.shape[1]
    for q in ti.static(range(K)):
        for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
            a, b, c = (s0 + q) % 3, (s0 + q + 1) % 3, (s0 + q + 2) % 3
            w = ti.cast(Wd[i, j], ti.i32)
            if (w >> 15) & 1 == 0:
                U[c, i, j] = 0.0
            else:
                u = U[b, i, j]
                e = 0.0
                ne = 0
                for k in ti.static(range(4)):
                    bit = (w >> k) & 1
                    e += ti.select(bit == 1, U[b, i + EDGES[k][0], j + EDGES[k][1]], 0.0)
                    ne += bit
                dg = 0.0
                nd = 0
                for k in ti.static(range(4)):
                    bit = (w >> (4 + k)) & 1
                    dg += ti.select(bit == 1, U[b, i + DIAGS[k][0], j + DIAGS[k][1]], 0.0)
                    nd += bit + ((w >> (8 + k)) & 1)
                lap = (2.0 / 3.0) * (e - ne * u) + (1.0 / 6.0) * (dg - nd * u)
                U[c, i, j] = 2 * u - U[a, i, j] + CC[i, j] * lap


@ti.kernel
def k_types(U: ti.types.ndarray(ndim=3), T: ti.types.ndarray(ndim=2), CC: ti.types.ndarray(ndim=2), s0: ti.i32):
    n = U.shape[1]
    for q in ti.static(range(K)):
        for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
            a, b, c = (s0 + q) % 3, (s0 + q + 1) % 3, (s0 + q + 2) % 3
            if T[i, j] != 0:
                U[c, i, j] = 0.0
            else:
                u = U[b, i, j]
                e = 0.0
                ne = 0.0
                for k in ti.static(range(4)):
                    tn = T[i + EDGES[k][0], j + EDGES[k][1]]
                    on = tn != 2                   # rigid neighbour: link dropped
                    e += ti.select(on, U[b, i + EDGES[k][0], j + EDGES[k][1]], 0.0)
                    ne += ti.select(on, 1.0, 0.0)
                dg = 0.0
                nd = 0.0
                for k in ti.static(range(4)):
                    di, dj = ti.static(DIAGS[k][0], DIAGS[k][1])
                    tn = T[i + di, j + dj]
                    t1 = T[i + di, j]
                    t2 = T[i, j + dj]
                    blk = (tn == 0) and (t1 != 0) and (t2 != 0)
                    on = (tn != 2) and not blk
                    pinblk = blk and ((t1 == 1) or (t2 == 1))
                    dg += ti.select(on, U[b, i + di, j + dj], 0.0)
                    nd += ti.select(on or pinblk, 1.0, 0.0)
                lap = (2.0 / 3.0) * (e - ne * u) + (1.0 / 6.0) * (dg - nd * u)
                U[c, i, j] = 2 * u - U[a, i, j] + CC[i, j] * lap


def link_word(T):
    """Host precompute of the link word from the type array (same rule as build_L)."""
    n = T.shape[0]
    free = T == W.FREE
    w = np.where(free, 1 << 15, 0).astype(np.int64)
    for k, d in enumerate(EDGES):
        tn = W.shift(T, d, W.PIN)
        w |= np.where(free & (tn != W.RIG), 1 << k, 0)
    for k, d in enumerate(DIAGS):
        tn = W.shift(T, d, W.PIN)
        t1 = W.shift(T, (d[0], 0), W.PIN)
        t2 = W.shift(T, (0, d[1]), W.PIN)
        blk = (tn == W.FREE) & (t1 != W.FREE) & (t2 != W.FREE)
        on = (tn != W.RIG) & ~blk
        pinblk = blk & ((t1 == W.PIN) | (t2 == W.PIN))
        w |= np.where(free & on, 1 << (4 + k), 0)
        w |= np.where(free & pinblk, 1 << (8 + k), 0)
    return w.astype(NPW)


def verify():
    import energy_leak_MIRROR as EL
    s = EL.scene(96)
    T = EL.mixed_types(s)                        # left objects pinned, right rigid, ring pinned
    L = W.build_L(T, W.SQ9, block='both', transfer=False)
    n = T.shape[0]
    co = 0.8
    X, Y = W.grid_xy(W.SQ9, n, n, 48, 48)
    u0 = np.exp(-(X * X + Y * Y) / 18) * (T == W.FREE)
    u1 = u0 + 0.5 * co * co * (L @ u0.ravel()).reshape(n, n)
    a, b = u0.ravel().copy(), u1.ravel().copy()
    for _ in range(K * 4):
        a, b = b, 2 * b - a + co * co * (L @ b)
    ref = b.reshape(n, n)
    CC = np.full((n, n), co * co, NPF)
    out = {}
    for name, kern, arr in (('word', k_word, link_word(T)), ('types', k_types, T.astype(NPM))):
        U = ti.ndarray(ti.f64 if FP64 else ti.f32, (3, n, n))
        U.from_numpy(np.stack([u0, u1, np.zeros_like(u0)]).astype(NPF))
        A = ti.ndarray(WDT if name == 'word' else MDT, (n, n))
        A.from_numpy(arr)
        C = ti.ndarray(ti.f64 if FP64 else ti.f32, (n, n))
        C.from_numpy(CC)
        s0 = 0
        for _ in range(4):
            kern(U, A, C, s0)
            s0 = (s0 + K) % 3
        got = U.to_numpy()[(s0 + 1) % 3]
        out[name] = float(np.abs(got - ref).max() / np.abs(ref).max())
    return out


def bench(N, reps):
    rng = np.random.default_rng(3)
    # a scene of the kind the app will have: 5% of the cells in solid objects, half pinned, half rigid
    T = np.zeros((N, N), np.int8)
    for _ in range(max(1, N * N // 20000)):
        i, j = rng.integers(2, N - 40, 2)
        h, w = rng.integers(3, 38, 2)
        T[i:i + h, j:j + w] = 1 if rng.random() < 0.5 else 2
    T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 1
    t0 = time.perf_counter()
    word = link_word(T)
    t_word = time.perf_counter() - t0
    U = ti.ndarray(ti.f64 if FP64 else ti.f32, (3, N, N))
    CC = ti.ndarray(ti.f64 if FP64 else ti.f32, (N, N))
    CC.fill(0.25)
    M = ti.ndarray(MDT, (N, N)); M.from_numpy((T == 0).astype(NPM))
    Tt = ti.ndarray(MDT, (N, N)); Tt.from_numpy(T.astype(NPM))
    Wd = ti.ndarray(WDT, (N, N)); Wd.from_numpy(word)
    res = {}
    for name, kern, arr in (('mask', k_mask, M), ('word', k_word, Wd), ('types', k_types, Tt)):
        kern(U, arr, CC, 0); ti.sync()          # compile + warm
        best = 1e9
        for r in range(reps):
            launches = 10
            ti.sync()
            t0 = time.perf_counter()
            for L_ in range(launches):
                kern(U, arr, CC, (L_ * K) % 3)
            ti.sync()
            best = min(best, (time.perf_counter() - t0) / (launches * K))
        res[name] = best
    return res, t_word, float((T != 0).mean())


if __name__ == '__main__':
    tag = f'{args.arch} {"f64" if FP64 else "f32"}'
    print(f'=== link-mask probe, {tag} ===')
    v = verify()
    print('  verify vs walls_lib (mixed pinned/rigid scene, 32 steps, Co 0.8): max rel diff '
          + ', '.join(f'{k} {x:.1e}' for k, x in v.items()))
    res, tw, solid = bench(args.N, args.reps)
    base = res['mask']
    print(f'  N = {args.N} ({solid * 100:.1f}% solid), best of {args.reps} x 10 launches x {K} steps:')
    for k, t in res.items():
        print(f'    {k:5}: {t * 1e3:7.3f} ms/step  ({t / args.N ** 2 * 1e9:.4f} ns/cell)  x{t / base:.3f} of mask')
    print(f'  host precompute of the link word (numpy): {tw:.2f} s')
