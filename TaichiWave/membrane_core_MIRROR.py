"""ZRL 2D wave membrane -- Taichi port, CORE (no GUI).

A value-for-value mirror of FormConstantsCLAUDE/FlavorRenderersCLAUDE/wave_membrane_MIRROR.html
(the merged build of 28/09/2026):
  - the same lattices (square, triangular), stencils (5-, 9-point; 6-neighbour triangular),
    domain shapes, radius snaps, media, boundaries, sources and drive waveforms;
  - the same update ORDER, down to the order of the neighbour sums, so a field computed here
    agrees with the web app's to floating-point rounding (checked by crosscheck_js_MIRROR.py).

Split of work: everything that runs once per reset (masks, per-cell Courant numbers, source
placement, seeding) is host-side numpy, written with the same arithmetic as the JS; everything
that runs every step is a Taichi kernel on the GPU (or CPU).

Array convention: fields are indexed [i, j] with i = column (x), j = row (y). The JS app stores
the same grid flat as k = j*N + i, so a JS array reshaped to (N, N) is [j, i] -- transpose it.

Documented deviations from the web app (29/09/2026; the cross-check applies the same changes to
the web app's code as PATCHES in ToolsCLAUDE/js_harness_MIRROR.mjs, so it still compares
value for value):
  - neumann-corner: every Neumann corner equals its diagonal interior neighbour of THIS step.
    The web app's sequential ring loop left (0,0) holding (1,0) from the rotating buffer's old
    contents -- the time level three steps back (traced 30/09/2026).
Additions with no web-app counterpart: the 'unlicensed' domain (a rounded 2:1 plate), solid
objects / painted walls (solids_MIRROR.py; the mask becomes domain AND NOT solid), and the
per-step recorder inside the fused kernel (the web app's record(): energy, peak, probes).
"""
import math
import os
import sys
from dataclasses import dataclass, asdict, replace
import numpy as np
import taichi as ti

PI = math.pi
TAU = 2 * PI
S3H = math.sqrt(3) / 2
CO_LIMIT_5 = 1 / math.sqrt(2)          # square, 5-point
CO_LIMIT_9 = math.sqrt(3) / 2          # square, 9-point isotropic
CO_LIMIT_T = math.sqrt(2 / 3)          # triangular, 6-neighbour
PPW_MIN = 8

# 'unlicensed' (appended 29/09/2026, the user's request): the license plate of Death Grips'
# "Government Plates" cover -- a rounded rectangle, width:height = 2:1, corner radius 0.065 H,
# measured from the cover. dom_R() is its HALF-WIDTH R, so W = 2R and H = R.
SHAPES = ['square', 'n4', 'circle', 'n3', 'n5', 'n6', 'n8', 'n12', 'rhomb', 'unlicensed']   # the web app's order + 1
PLATE_ASPECT = 2.0                     # W / H
PLATE_CORNER = 0.065                   # corner radius / H
PLATE_ARC_SEG = 16                     # segments per corner arc in shape_poly()
BCS = ['dirichlet', 'mur', 'neumann', 'periodic']
MEDIA = ['uniform', 'slab', 'lens']
SOURCES = ['impulse', 'pulse', 'cont', 'slit', 'mode', 'vtx']
WAVES = ['sine', 'square', 'saw', 'tri']


# ---- JavaScript arithmetic, reproduced exactly --------------------------------------------
def js_round(x):
    """Math.round: halves go toward +infinity (Python's round() is banker's rounding)."""
    return math.floor(x + 0.5)


def js_sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def sym_round(c, d):
    """Round an offset from the centre SYMMETRICALLY, so +d and -d land on mirrored cells."""
    return c + (0 if d == 0 else js_sign(d) * js_round(abs(d)))


def snap_q(v):
    """Quantise to 1e-9 so floating-point noise cannot break a rounding tie differently for
    mirror partners (the web app's snapQ)."""
    return js_round(v * 1e9) / 1e9


def js_fmod(a, m):
    return math.fmod(a, m)


@dataclass
class Params:
    """Mirrors the web app's S object (angles in RADIANS, as S.rot / S.rtheta are)."""
    N: int = 161
    Co: float = 0.5
    damp: float = 0.0
    shape: str = 'square'
    bc: str = 'dirichlet'
    medium: str = 'uniform'
    src: str = 'pulse'
    sigma: float = 3.0
    freq: float = 0.12
    stencil: int = 5
    rtheta: float = PI / 3
    rot: float = 0.0
    rad: float = 1.0
    sx: float = 0.0
    sy: float = 0.0
    vtx_drive: bool = False
    lattice: str = 'sq'
    wave: str = 'sine'
    vtx_inset: str = 'radial'
    mm: int = 1
    mn: int = 1


MASK_DT = ti.u8
FP64 = True
# the recorder's accumulators: f64 wherever the backend has it. Vulkan f64 accumulators
# crashed the process (segfault, measured 29/09/2026), so f32 there: the two-level sum below
# keeps its relative error at ~2e-7 even at N = 8193 (a naive serial f32 sum was off by 18%).
ACC_DT = ti.f64
# The 9-point weights (2/3)*4 + (1/6)*4 - 10/3 cancel exactly in f64 but NOT in f32: the
# residue acts as a small mass term and shifts low modes (measured 29/09: the (3,2) mode off
# by 5e-4 of its amplitude after 300 steps, on CUDA and Vulkan alike). In f32 the stencil is
# written in the form that cancels by construction, (2/3)(e - 4u) + (1/6)(dg - 4u); in f64 it
# keeps the web app's exact operation order, so the two stay bit-comparable.
SQ9_CONSISTENT = False


def init_taichi(arch='cuda', fp64=True, fast_math=False, quiet=False):
    """fast_math is OFF by default: it lets the compiler fuse and reorder float operations,
    which is fine for speed but breaks agreement with the web app at the last bits.
    (CUDA still fuses a*b + c into one FMA rounding regardless; see README, cross-check.)"""
    global MASK_DT, SQ9_CONSISTENT, FP64, ACC_DT
    SQ9_CONSISTENT = not fp64
    FP64 = fp64
    archs = {'cuda': ti.cuda, 'vulkan': ti.vulkan, 'cpu': ti.cpu, 'gpu': ti.gpu}
    ti.init(arch=archs[arch], default_fp=ti.f64 if fp64 else ti.f32, fast_math=fast_math,
            log_level=ti.ERROR if quiet else ti.WARN)
    # Vulkan emulates 8-bit storage inside 32-bit words, and with N*N not a multiple of 4
    # the LAST cell of a u8 mask read as 0 (measured 29/09: N=161 periodic, cell (160,160)
    # clamped from step 1 -- invisible until a wave reached that corner at step 291).
    # A 32-bit mask on Vulkan; u8 elsewhere (a quarter of the memory traffic).
    vk = ti.lang.impl.current_cfg().arch == ti.vulkan
    MASK_DT = ti.i32 if vk else ti.u8
    ACC_DT = ti.f32 if vk else ti.f64


def mask_np_dtype():
    return np.uint8 if MASK_DT == ti.u8 else np.int32


# ---- the time step: ONE fused kernel for K steps --------------------------------------------
# Taichi's launch overhead on this machine is ~170-265 us per kernel call (measured 29/09) --
# more than a whole step costs the GPU below N ~ 2000. So K steps are unrolled into ONE launch
# (ti.static): each step's loops are still separate parallel passes run in order, with the
# web app's per-cell arithmetic. The three time levels live in one array U[3, N, N] and rotate
# by slot: step q reads slot (s+q)%3 (previous) and (s+q+1)%3 (current), writes (s+q+2)%3.
# Storage is ti.ndarray, not ti.field: a field is baked into the compiled kernel, so every new
# grid (every reset at a new N, every Membrane) recompiled everything; an ndarray is not.
LAT_SQ, LAT_TRI = 0, 1
BC_DIR, BC_MUR, BC_NEU, BC_PER = 0, 1, 2, 3
BC_CODE = {'dirichlet': BC_DIR, 'mur': BC_MUR, 'neumann': BC_NEU, 'periodic': BC_PER}
SRC_NONE, SRC_CELLS, SRC_SLIT = 0, 1, 2
# steps per launch; a run of n steps is split greedily. Measured 29/09 (CUDA, N = 161, Mur):
#   K =  1: compile 0.25 s,  480 us/step     K = 16: compile  4.8 s,  92 us/step
#   K =  8: compile 1.8 s,   115 us/step     K = 32: compile 14.7 s,  73 us/step
# The residue is ~20 us per parallel pass inside the kernel (Windows' WDDM launch latency);
# 8 keeps the first run of a new lattice/boundary/source combination to ~2 s.
CHUNKS = (8, 1)
KMAX = CHUNKS[0]

# ---- the recorder (the web app's record(), run after every step) -----------------------------
# kin = sum (u_new - u_prev)^2, pot = sum of forward differences squared, amax = max |u_new|,
# all over the masked interior, plus three single-cell probes. A per-cell atomic into ONE
# address does not scale: measured 29/09/2026 (CUDA f64, one reduction pass): 3.8 ms at N=1025,
# 62 ms at N=4097, 247 ms at N=8193 (Taichi applies no thread-local reduction to an ndarray
# target; a 0-D field gets one, but then costs a flat ~1.4 ms per launch from ~290k per-thread
# atomics). So TWO levels: thread (g, j) sums a run of ~n/ng rows in column j serially
# (neighbouring threads read neighbouring memory), adds its partial into one of NSLOT slots,
# and one thread folds the slots. Measured, one pass: 0.34 ms at N=161..2049 (launch-bound),
# 0.58 ms at 4097, 2.05 ms at 8193 -- the cost of one copy pass -- relative error 3e-16 (f64).
# Inside k_run, per step (30/09/2026, CUDA f64, ScratchCLAUDE/DevCLAUDE/rec_tune_MIRROR.py):
# +0.05 ms at N=1025, +0.47 ms at 4097, +1.86 ms at 8193 (~50% of a step: it reads two f64
# levels and the mask, ~17 of the step's ~41 bytes per cell) with 65536 first-level threads;
# 131072-1048576 threads and 32-256 slots were no faster at 4097/8193 and slower at 1025.
NSLOT = 64
REC_THREADS = 65536                   # ~ng * n threads in the first level
HCAP = 1024                           # rows of the device-side history ring (6 values each)
# Launch cost (30/09/2026, Taichi 1.7.4, CUDA, ScratchCLAUDE/DevCLAUDE/argcost_MIRROR.py): ~30 us
# per ndarray argument and ~10 us per scalar argument, per kernel call. The recorder's buffers
# as three extra ndarrays + two scalars made even the recorder-OFF step ~15 us slower at N <= 1025
# (0.091 -> 0.106 ms/step). So the recorder has its own kernel (k_run_rec) sharing the step body
# (_steps) with the plain k_run, whose argument list is the original one; the reduction slots
# live in the history buffer's last NSLOT rows, and the group count is computed in the kernel.


def rec_groups(n):
    """First-level groups of the recorder's reduction (the kernels compute the same in-kernel)."""
    return max(1, min(n - 2, REC_THREADS // n))


@ti.func
def lap_sq(U: ti.template(), b, i, j, im, ip, jm, jp, nine: ti.template()):
    # sum order = the JS order: (i-1,j) (i+1,j) (i,j-1) (i,j+1); diagonals (-,-) (+,-) (-,+) (+,+)
    u = U[b, i, j]
    lap = 0.0
    if ti.static(nine):
        e = U[b, im, j] + U[b, ip, j] + U[b, i, jm] + U[b, i, jp]
        dg = U[b, im, jm] + U[b, ip, jm] + U[b, im, jp] + U[b, ip, jp]
        if ti.static(SQ9_CONSISTENT):
            lap = (2.0 / 3.0) * (e - 4 * u) + (1.0 / 6.0) * (dg - 4 * u)
        else:
            lap = (2.0 / 3.0) * e + (1.0 / 6.0) * dg - (10.0 / 3.0) * u
    else:
        lap = U[b, im, j] + U[b, ip, j] + U[b, i, jm] + U[b, i, jp] - 4 * u
    return lap


@ti.func
def rec_fold(U: ti.template(), M: ti.template(), PS: ti.template(), p0: ti.template(), b: ti.template(),
             c: ti.template()):
    """First level of the recorder's reduction (see NSLOT): thread (g, j) walks rows
    [1 + g*gsz, 1 + (g+1)*gsz) of column j. Slot s (row p0 + s of PS) collects the partials of
    ~ng*n/NSLOT threads. Template parameters: no copy statements at the kernel's top level (see
    _steps)."""
    n = U.shape[1]
    ng = ti.max(1, ti.min(n - 2, REC_THREADS // n))
    gsz = (n - 2 + ng - 1) // ng
    for g, j in ti.ndrange(ng, (1, n - 1)):
        ka = ti.cast(0.0, ACC_DT)
        pa = ti.cast(0.0, ACC_DT)
        mx = ti.cast(0.0, ACC_DT)
        i0 = 1 + g * gsz
        i1 = ti.min(n - 1, i0 + gsz)
        u = U[c, ti.min(i0, n - 1), j]
        for i in range(i0, i1):
            un = U[c, i + 1, j]             # the next row's value: this row's gx, the next row's u
            if M[i, j] != 0:
                dv = u - U[b, i, j]
                gx = un - u
                gy = U[c, i, j + 1] - u
                ka += ti.cast(dv * dv, ACC_DT)
                pa += ti.cast(gx * gx + gy * gy, ACC_DT)
                mx = ti.max(mx, ti.cast(ti.abs(u), ACC_DT))
            u = un
        s = p0 + (g * 7 + j) % NSLOT
        ti.atomic_add(PS[s, 0], ka)
        ti.atomic_add(PS[s, 1], pa)
        ti.atomic_max(PS[s, 2], mx)


@ti.func
def rec_sum(PS: ti.template(), p0, H: ti.template(), row):
    """Second level: ONE thread folds the NSLOT slots (rows p0.. of PS) into H[row, 0:3] and
    re-zeroes them, so the slots are always zero between uses (the buffers are zero-filled at
    allocation)."""
    ka = ti.cast(0.0, ACC_DT)
    pa = ti.cast(0.0, ACC_DT)
    mx = ti.cast(0.0, ACC_DT)
    for s in range(p0, p0 + NSLOT):
        ka += PS[s, 0]
        pa += PS[s, 1]
        mx = ti.max(mx, PS[s, 2])
        PS[s, 0] = 0.0
        PS[s, 1] = 0.0
        PS[s, 2] = 0.0
    H[row, 0] = ka
    H[row, 1] = pa
    H[row, 2] = mx


@ti.func
def _steps(U: ti.template(), M: ti.template(), CC: ti.template(), DRV: ti.template(), SC: ti.template(),
           HIST: ti.template(), PRB: ti.template(), s0: ti.template(), K: ti.template(), lat: ti.template(),
           nine: ti.template(), bc: ti.template(), masked: ti.template(), srck: ti.template(),
           nsrc: ti.template(), d: ti.template(), kmur: ti.template(), jw: ti.template(), jd: ti.template(),
           cA: ti.template(), cB: ti.template(), half: ti.template(), rec: ti.template(),
           slitm: ti.template(), h0: ti.template()):
    """The K steps (inlined into k_run / k_run_rec, so its loops are the kernel's top-level,
    parallel loops). HIST and PRB are read only when rec is on. EVERY parameter is a template
    (passed by reference): a by-value scalar parameter becomes a copy statement at the top of the
    kernel, i.e. one more serial task per launch -- measured 30/09/2026 on Vulkan: +40-60 us per
    launch (+5-8 us/step at N <= 1025) until the scalars were templates; CUDA showed nothing."""
    n = U.shape[1]
    for q in ti.static(range(K)):
        # -- interior -------------------------------------------------------------------------
        if ti.static(lat == LAT_TRI):
            # 6 neighbours at distance 1: lap = (2/3)(sum u_m - 6 u_0), JS order
            # (i-1,j) (i+1,j) (i,j-1) (i,j+1) (i+1,j-1) (i-1,j+1)
            for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
                a, b, c = (s0 + q) % 3, (s0 + q + 1) % 3, (s0 + q + 2) % 3
                if M[i, j] == 0:
                    U[c, i, j] = 0.0
                else:
                    u = U[b, i, j]
                    s = (U[b, i - 1, j] + U[b, i + 1, j] + U[b, i, j - 1] + U[b, i, j + 1]
                         + U[b, i + 1, j - 1] + U[b, i - 1, j + 1])
                    U[c, i, j] = (2 * u - U[a, i, j] + CC[i, j] * (2.0 / 3.0) * (s - 6 * u)) - d * (u - U[a, i, j])
            for t in range(n):                    # the triangular array edge is always clamped
                c = (s0 + q + 2) % 3
                U[c, t, 0] = 0.0
                U[c, t, n - 1] = 0.0
                U[c, 0, t] = 0.0
                U[c, n - 1, t] = 0.0
        elif ti.static(bc == BC_PER):
            # a torus has no boundary: ONE wrapped update over every cell
            for i, j in ti.ndrange(n, n):
                a, b, c = (s0 + q) % 3, (s0 + q + 1) % 3, (s0 + q + 2) % 3
                if M[i, j] == 0:
                    U[c, i, j] = 0.0
                else:
                    lap = lap_sq(U, b, i, j, (i - 1 + n) % n, (i + 1) % n, (j - 1 + n) % n, (j + 1) % n, nine)
                    u = U[b, i, j]
                    U[c, i, j] = (2 * u - U[a, i, j] + CC[i, j] * lap) - d * (u - U[a, i, j])
        else:
            for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
                a, b, c = (s0 + q) % 3, (s0 + q + 1) % 3, (s0 + q + 2) % 3
                if M[i, j] == 0:
                    U[c, i, j] = 0.0
                else:
                    lap = lap_sq(U, b, i, j, i - 1, i + 1, j - 1, j + 1, nine)
                    u = U[b, i, j]
                    U[c, i, j] = (2 * u - U[a, i, j] + CC[i, j] * lap) - d * (u - U[a, i, j])
            # -- boundary of the array ------------------------------------------------------
            if ti.static(bc == BC_DIR):
                for t in range(n):
                    c = (s0 + q + 2) % 3
                    U[c, t, 0] = 0.0
                    U[c, t, n - 1] = 0.0
                    U[c, 0, t] = 0.0
                    U[c, n - 1, t] = 0.0
            elif ti.static(bc == BC_NEU):
                # Sides first (each reads only interior cells: order-free, parallel), then the
                # corners, each = its diagonal interior neighbour of THIS step, read through the
                # side cell that already holds it: (0,0)=(1,1) (0,n-1)=(1,n-2) (n-1,0)=(n-2,1)
                # (n-1,n-1)=(n-2,n-2). That is the ghost-cell rule for which the 9-point operator
                # is symmetric, so the discrete energy is conserved (identity_tests_MIRROR.py).
                # PORT FIX 29/09/2026: the web app's ONE sequential ring loop writes (0,0) at
                # i = 0 from (1,0) BEFORE i = 1 refreshes it, i.e. from the rotating buffer's old
                # contents, the level three steps back (the other three corners were right). Through the
                # 9-point stencil's diagonal that changed the field by 8.6% of its peak near the
                # corner (pulse at the corner, 400 steps); it also broke the 8-fold symmetry and
                # energy conservation. The cross-check patches the web app the same way.
                for t in range(1, n - 1):
                    c = (s0 + q + 2) % 3
                    U[c, t, 0] = U[c, t, 1]
                    U[c, t, n - 1] = U[c, t, n - 2]
                    U[c, 0, t] = U[c, 1, t]
                    U[c, n - 1, t] = U[c, n - 2, t]
                for t in range(1):
                    c = (s0 + q + 2) % 3
                    U[c, 0, 0] = U[c, 1, 0]
                    U[c, 0, n - 1] = U[c, 1, n - 1]
                    U[c, n - 1, 0] = U[c, n - 1, 1]
                    U[c, n - 1, n - 1] = U[c, n - 2, n - 1]
            elif ti.static(bc == BC_MUR):
                # 1st-order Mur: u[0]^{n+1} = u[1]^n + k (u[1]^{n+1} - u[0]^n), k = (Co-1)/(Co+1).
                # Every ring cell reads interior cells only, so this is order-free: parallel.
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
        # -- any masked domain is Dirichlet on the mask edge (the interior pass already zeroed
        #    the interior cells outside it; this is the array ring) --------------------------
        if ti.static(masked):
            for t in range(n):
                c = (s0 + q + 2) % 3
                if M[t, 0] == 0:
                    U[c, t, 0] = 0.0
                if M[t, n - 1] == 0:
                    U[c, t, n - 1] = 0.0
                if M[0, t] == 0:
                    U[c, 0, t] = 0.0
                if M[n - 1, t] == 0:
                    U[c, n - 1, t] = 0.0
        # -- driven sources, AFTER the boundary, as in the JS ---------------------------------
        if ti.static(srck == SRC_CELLS):
            for k in range(nsrc):
                c = (s0 + q + 2) % 3
                U[c, SC[k, 0], SC[k, 1]] = DRV[q]
        elif ti.static(srck == SRC_SLIT):
            for i in range(n):                    # a rigid wall with two openings ...
                c = (s0 + q + 2) % 3
                if jw > 0 and jw < n:
                    inA = ti.abs(i - cA) < half
                    inB = ti.abs(i - cB) < half
                    if not inA and not inB:
                        U[c, i, jw] = 0.0
            for i in range(n):                    # ... driven from below
                c = (s0 + q + 2) % 3
                if jd > 0 and jd < n:
                    if ti.static(slitm):
                        # with solids present the drive line skips masked cells, so it never
                        # writes into a solid (the web app, which has no solids, drives the
                        # whole row, outside the domain too; slitm = 0 keeps that exactly)
                        if M[i, jd] != 0:
                            U[c, i, jd] = DRV[q]
                    else:
                        U[c, i, jd] = DRV[q]
        # -- the recorder: the web app's record(), after the step (reads only) ----------------
        if ti.static(rec):
            rec_fold(U, M, HIST, HCAP, (s0 + q + 1) % 3, (s0 + q + 2) % 3)
            for t in range(1):
                c = (s0 + q + 2) % 3
                row = (h0 + q) % HCAP
                rec_sum(HIST, HCAP, HIST, row)
                HIST[row, 3] = ti.cast(U[c, PRB[0], PRB[1]], ACC_DT)
                HIST[row, 4] = ti.cast(U[c, PRB[2], PRB[3]], ACC_DT)
                HIST[row, 5] = ti.cast(U[c, PRB[4], PRB[5]], ACC_DT)


@ti.kernel
def k_run(U: ti.types.ndarray(ndim=3), M: ti.types.ndarray(ndim=2), CC: ti.types.ndarray(ndim=2),
          DRV: ti.types.ndarray(ndim=1), SC: ti.types.ndarray(ndim=2),
          s0: ti.i32, K: ti.template(), lat: ti.template(), nine: ti.template(), bc: ti.template(),
          masked: ti.template(), srck: ti.template(), nsrc: ti.i32, d: float, kmur: float,
          jw: ti.i32, jd: ti.i32, cA: ti.i32, cB: ti.i32, half: ti.i32, slitm: ti.template()):
    # the solver alone: the argument list of 29/09 (+ the slitm template, ~free)
    _steps(U, M, CC, DRV, SC, U, DRV, s0, K, lat, nine, bc, masked, srck, nsrc, d, kmur,
           jw, jd, cA, cB, half, False, slitm, 0)


@ti.kernel
def k_run_rec(U: ti.types.ndarray(ndim=3), M: ti.types.ndarray(ndim=2), CC: ti.types.ndarray(ndim=2),
              DRV: ti.types.ndarray(ndim=1), SC: ti.types.ndarray(ndim=2), HIST: ti.types.ndarray(ndim=2),
              PRB: ti.types.ndarray(ndim=1),
              s0: ti.i32, K: ti.template(), lat: ti.template(), nine: ti.template(), bc: ti.template(),
              masked: ti.template(), srck: ti.template(), nsrc: ti.i32, d: float, kmur: float,
              jw: ti.i32, jd: ti.i32, cA: ti.i32, cB: ti.i32, half: ti.i32, slitm: ti.template(), h0: ti.i32):
    # the solver + the recorder after every step: HIST rows [0, HCAP) are the history ring (kin,
    # pot, amax, probe A, probe B, mode probe), rows [HCAP, HCAP + NSLOT) the reduction slots
    _steps(U, M, CC, DRV, SC, HIST, PRB, s0, K, lat, nine, bc, masked, srck, nsrc, d, kmur,
           jw, jd, cA, cB, half, True, slitm, h0)


@ti.kernel
def k_rescale(U: ti.types.ndarray(ndim=3), M: ti.types.ndarray(ndim=2), CC: ti.types.ndarray(ndim=2),
              a: ti.i32, b: ti.i32, r: float, tri: ti.i32, nine: ti.i32, wrap: ti.i32):
    # set_co(rescale=True): re-express the previous time level (slot a) at the new time step.
    # CC still holds the OLD Co^2 here (set_co rebuilds it afterwards).
    n = U.shape[1]
    for i, j in ti.ndrange(n, n):
        interior = i > 0 and j > 0 and i < n - 1 and j < n - 1
        if M[i, j] != 0 and (interior or wrap):
            im = (i - 1 + n) % n
            ip = (i + 1) % n
            jm = (j - 1 + n) % n
            jp = (j + 1) % n
            u = U[b, i, j]
            lap = 0.0
            if tri:
                lap = (2.0 / 3.0) * (U[b, im, j] + U[b, ip, j] + U[b, i, jm] + U[b, i, jp]
                                     + U[b, ip, jm] + U[b, im, jp] - 6 * u)
            elif nine:
                lap = ((2.0 / 3.0) * (U[b, im, j] + U[b, ip, j] + U[b, i, jm] + U[b, i, jp] - 4 * u)
                       + (1.0 / 6.0) * (U[b, im, jm] + U[b, ip, jm] + U[b, im, jp] + U[b, ip, jp] - 4 * u))
            else:
                lap = U[b, im, j] + U[b, ip, j] + U[b, i, jm] + U[b, i, jp] - 4 * u
            U[a, i, j] = u - r * (u - U[a, i, j]) + 0.5 * (r * r - r) * CC[i, j] * lap


@ti.kernel
def k_stats(U: ti.types.ndarray(ndim=3), M: ti.types.ndarray(ndim=2), acc: ti.types.ndarray(ndim=2),
            a: ti.i32, b: ti.i32):
    # the web app's record() sums (kinetic, gradient, peak |u| over the masked interior) of the
    # CURRENT state, on demand. Same two-level reduction as the recorder: the one-address
    # atomic version this replaces took 62 ms at N = 4097 and 247 ms at N = 8193 (29/09/2026).
    # acc row 0 = the result, rows 1..NSLOT = the slots.
    rec_fold(U, M, acc, 1, a, b)
    for t in range(1):
        rec_sum(acc, 1, acc, 0)


@ti.kernel
def k_zero_masked(U: ti.types.ndarray(ndim=3), M: ti.types.ndarray(ndim=2)):
    # every time level, wherever the mask is 0 (a cell that just became solid holds no wave)
    for i, j in ti.ndrange(U.shape[1], U.shape[2]):
        if M[i, j] == 0:
            U[0, i, j] = 0.0
            U[1, i, j] = 0.0
            U[2, i, j] = 0.0


@ti.kernel
def k_patch_mask(U: ti.types.ndarray(ndim=3), M: ti.types.ndarray(ndim=2), P: ti.types.ndarray(ndim=2),
                 i0: ti.i32, j0: ti.i32):
    # write a mask patch at (i0, j0) and zero all three time levels where it is 0: one launch
    # (and one small host->device copy of P) per painted segment, not a whole-grid upload
    for a, b in ti.ndrange(P.shape[0], P.shape[1]):
        v = P[a, b]
        M[i0 + a, j0 + b] = v
        if v == 0:
            U[0, i0 + a, j0 + b] = 0.0
            U[1, i0 + a, j0 + b] = 0.0
            U[2, i0 + a, j0 + b] = 0.0


@ti.kernel
def k_load(U: ti.types.ndarray(ndim=3), s: ti.i32, src: ti.types.ndarray(ndim=2)):
    for i, j in ti.ndrange(U.shape[1], U.shape[2]):
        U[s, i, j] = src[i, j]


@ti.kernel
def k_store(U: ti.types.ndarray(ndim=3), s: ti.i32, dst: ti.types.ndarray(ndim=2)):
    for i, j in ti.ndrange(U.shape[1], U.shape[2]):
        dst[i, j] = U[s, i, j]


@ti.kernel
def k_zero(U: ti.types.ndarray(ndim=3), s: ti.i32):
    for i, j in ti.ndrange(U.shape[1], U.shape[2]):
        U[s, i, j] = 0.0


HIST_CAP = {'E': 4000, 'amax': 4000, 'pA': 4000, 'pB': 4000, 'mode': 600}   # the web app's caps


class Membrane:
    """The solver state. Construct AFTER init_taichi().

    Masks: domain_mask_np (the shape) and solid_np (solids_MIRROR.py objects and painted walls,
    None when there are none) compose to mask_np = domain AND NOT solid, which is what the
    solver, the sources and the recorder see (and what M holds on the device)."""

    def __init__(self, params=None, scene=None):
        self.p = params or Params()
        self.N = 0
        self.U = self.M = self.CC = None
        self.slot = 0                   # U[slot] = previous level, U[slot+1] = current (mod 3)
        self.fdt = np.float64 if FP64 else np.float32
        self.acc = ti.ndarray(dtype=ACC_DT, shape=(1 + NSLOT, 6))   # stats(): result row + slots
        self.acc.fill(0)
        self.drv = ti.ndarray(dtype=float, shape=KMAX)
        self.harm = [(1, 1.0)]
        self.step_n = 0
        self.t = 0.0
        self.drops = 0
        self.vtx_capped = 0
        self.clip_frac = None
        # recorder: device ring HIST (HCAP rows: kin, pot, amax, probe A, probe B, mode) and the
        # reduction slots PS; the host history m.hist mirrors the web app's arrays
        self.record = True
        self.hist_dev = ti.ndarray(dtype=ACC_DT, shape=(HCAP + NSLOT, 6))   # ring + reduction slots
        self.hist_dev.fill(0)           # rec_sum re-zeroes the slots after use; they must START at 0
        self.prb = ti.ndarray(dtype=ti.i32, shape=6)
        self.hrow = 0                   # the next HIST row the kernel writes (a ring of HCAP rows)
        self.hist = {k: [] for k in HIST_CAP}
        self.amax = 0.0
        # solids
        self.scene = scene
        if scene is not None:
            scene.m = self
        self.solid_np = None
        self.domain_mask_np = None
        self.has_solids = False
        self.n_solid = 0
        self._solid_key = None
        self.reseed()

    # ---- geometry, exactly as the web app computes it ------------------------------------
    def tri(self):
        return self.p.lattice == 'tri'

    def c(self):
        return (self.N - 1) / 2

    def cell_xy(self, i, j):
        c = self.c()
        da, db = i - c, j - c
        return (da + db * 0.5, db * S3H) if self.tri() else (da, db)

    def xy_to_cell(self, x, y):
        c = self.c()
        if not self.tri():
            return c + x, c + y
        db = y / S3H
        da = x - db * 0.5
        return c + da, c + db

    def co_limit(self):
        return CO_LIMIT_T if self.tri() else (CO_LIMIT_9 if self.p.stencil == 9 else CO_LIMIT_5)

    def sq_full(self):
        return self.p.shape == 'square' and self.p.rad >= 0.999 and not self.tri()

    def base_half(self):
        return self.c() * (S3H if self.tri() else 1)

    def at_conf_rot(self):
        d = js_fmod(js_fmod(self.p.rot * 180 / PI - 30, 60) + 60, 60)
        return min(d, 60 - d) < 1e-6

    def dom_R(self):
        p = self.p
        R = (self.base_half() - 1.5) * p.rad
        if self.tri():
            if self.at_conf_rot():
                if p.shape == 'n6':
                    return max(2, js_round(R))
                if p.shape == 'rhomb' and abs(p.rtheta - PI / 3) < 1e-6:
                    return max(2, 2 * js_round(R / math.sqrt(3))) * math.sqrt(3) / 2
                if p.shape == 'n3':
                    L = js_round(R * math.sqrt(3))
                    L = max(3, js_round(L / 3) * 3)
                    return L / math.sqrt(3)
            return R
        return math.floor(R) + 0.5 if (p.shape == 'square' and not self.sq_full()) else R

    def inner_R(self):
        p = self.p
        R, h = self.dom_R(), self.base_half()
        if p.shape == 'square':
            ir = h if self.sq_full() else R
        elif p.shape == 'circle':
            ir = R
        elif p.shape == 'rhomb':
            B = R * math.tan(p.rtheta / 2)
            ir = R * B / math.hypot(R, B)
        elif p.shape == 'unlicensed':
            ir = R / PLATE_ASPECT                     # the plate's half-height
        else:
            ir = R * math.cos(PI / int(p.shape[1:]))
        return min(ir, h)

    def plate_dims(self):
        """'unlicensed': (half-width, half-height, corner radius) in physical cells."""
        R = self.dom_R()
        H = 2 * R / PLATE_ASPECT
        return R, H / 2, PLATE_CORNER * H

    def plate_corners(self):
        """The 4 corner-arc MIDPOINTS (rotated): the plate's 'vertices' for the vertex sources,
        in the order of shape_poly()'s corners (bottom-right, top-right, top-left, bottom-left)."""
        hw, hh, rc = self.plate_dims()
        co, si = math.cos(self.p.rot), math.sin(self.p.rot)
        d = rc * math.sqrt(0.5)
        out = []
        for sx, sy in ((1, -1), (1, 1), (-1, 1), (-1, -1)):
            x, y = sx * (hw - rc + d), sy * (hh - rc + d)
            out.append((x * co - y * si, x * si + y * co))
        return out

    def shape_poly(self):
        p = self.p
        R, P = self.dom_R(), []
        if p.shape == 'unlicensed':
            # the rounded rectangle, PLATE_ARC_SEG segments per corner (4 x 17 = 68 points, CCW)
            hw, hh, rc = self.plate_dims()
            co, si = math.cos(p.rot), math.sin(p.rot)
            for k, (sx, sy) in enumerate(((1, -1), (1, 1), (-1, 1), (-1, -1))):
                cx, cy = sx * (hw - rc), sy * (hh - rc)
                a0 = -PI / 2 + k * PI / 2
                for s in range(PLATE_ARC_SEG + 1):
                    a = a0 + (PI / 2) * s / PLATE_ARC_SEG
                    x, y = cx + rc * math.cos(a), cy + rc * math.sin(a)
                    P.append((x * co - y * si, x * si + y * co))
            return P
        if p.shape == 'square':
            r = self.c() if self.sq_full() else R
            P = [(-r, -r), (r, -r), (r, r), (-r, r)]
        elif p.shape == 'circle':
            P = [(R * math.cos(TAU * k / 256), R * math.sin(TAU * k / 256)) for k in range(256)]
        elif p.shape == 'rhomb':
            B = R * math.tan(p.rtheta / 2)
            co, si = math.cos(p.rot), math.sin(p.rot)
            P = [(x * co - y * si, x * si + y * co) for x, y in ((R, 0), (0, B), (-R, 0), (0, -B))]
        else:
            nn = int(p.shape[1:])
            for k in range(nn):
                t = p.rot + PI / nn + k * TAU / nn
                P.append((R * math.cos(t), R * math.sin(t)))
        return P

    def clip_region(self):
        N = self.N
        if not self.tri():
            h = N / 2
            return [(-h, -h), (h, -h), (h, h), (-h, h)]
        s = N / (N - 1)
        out = []
        for i, j in ((0, 0), (N - 1, 0), (N - 1, N - 1), (0, N - 1)):
            x, y = self.cell_xy(i, j)
            out.append((x * s, y * s))
        return out

    @staticmethod
    def clip_convex(P, C):
        """Sutherland-Hodgman against a CCW convex polygon (the web app's clipConvex)."""
        side = lambda p, a, nx, ny: nx * (p[0] - a[0]) + ny * (p[1] - a[1])
        out = list(P)
        for e in range(len(C)):
            a, b = C[e], C[(e + 1) % len(C)]
            nx, ny = -(b[1] - a[1]), (b[0] - a[0])
            src, out = out, []
            for k in range(len(src)):
                p, q = src[(k + len(src) - 1) % len(src)], src[k]
                sp, sq = side(p, a, nx, ny), side(q, a, nx, ny)
                def cut():
                    t = sp / (sp - sq)
                    return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
                if sq >= 0:
                    if sp < 0:
                        out.append(cut())
                    out.append(q)
                elif sp >= 0:
                    out.append(cut())
            if not out:
                return []
        return out

    # ---- allocation, mask and per-cell Courant numbers ----------------------------------
    def _alloc(self, N):
        if N % 2 == 0:
            N += 1                                    # odd: a true centre cell
        self.N = N
        self.U = ti.ndarray(dtype=float, shape=(3, N, N))
        self.M = ti.ndarray(dtype=MASK_DT, shape=(N, N))
        self.CC = ti.ndarray(dtype=float, shape=(N, N))

    def _grid_xy(self, i0=0, i1=None, j0=0, j1=None):
        """Physical offsets of the cells in the index window [i0,i1) x [j0,j1), as (x, y)
        arrays indexed [i, j] -- the web app's cellXY, vectorised."""
        c = self.c()
        i1 = self.N if i1 is None else i1
        j1 = self.N if j1 is None else j1
        I, J = np.meshgrid(np.arange(i0, i1, dtype=np.float64), np.arange(j0, j1, dtype=np.float64),
                           indexing='ij')
        da, db = I - c, J - c
        if self.tri():
            return da + db * 0.5, db * S3H
        return da, db

    def build_domain(self):
        """Mask + per-cell Co^2, in row chunks: at N = 8193 a whole-grid pass held ~3 GB of
        float64 temporaries on the host; a chunk of 512 rows holds a few hundred MB.
        The solids (if any) are composed in: mask_np = domain_mask_np AND NOT solid_np."""
        N = self.N
        dom = np.empty((N, N), dtype=np.uint8)
        self.cc_np = np.empty((N, N), dtype=np.float64)
        CH = max(1, (1 << 22) // N)
        for i0 in range(0, N, CH):
            i1 = min(N, i0 + CH)
            ins, cc = self._domain_rows(i0, i1)
            dom[i0:i1] = ins
            self.cc_np[i0:i1] = cc
        self.domain_mask_np = dom
        if self.scene is None and self.p.shape == 'unlicensed':
            self.ensure_scene()                     # the plate's lettering is a scene preset
        self._rasterize_solids()
        self._compose()
        self.M.from_numpy(self.mask_np.astype(mask_np_dtype()))
        self.CC.from_numpy(self.cc_np.astype(self.fdt))

    # ---- solids (solids_MIRROR.py supplies the scene; the core only composes) -------------
    def ensure_scene(self):
        """The attached SolidScene, creating an empty one (imported lazily: solids_MIRROR
        imports this module) if there is none. Does not rebuild anything."""
        if self.scene is None:
            here = os.path.dirname(os.path.abspath(__file__))
            if here not in sys.path:
                sys.path.insert(0, here)
            import solids_MIRROR
            self.scene = solids_MIRROR.SolidScene()
            self.scene.m = self
        return self.scene

    def attach_scene(self, scene):
        """Attach (or with None, detach) a SolidScene and apply it at once (update_solids)."""
        self.scene = scene
        if scene is not None:
            scene.m = self
        self._solid_key = None
        self.update_solids()

    def _solid_cache_key(self):
        p = self.p
        sig = self.scene.signature() if self.scene is not None else None
        return (id(self.scene), sig, self.N, p.lattice, p.shape, p.rot, p.rad, p.rtheta)

    def _rasterize_solids(self, force=False):
        """solid_np for the whole grid (None if the scene makes no solid cell). Cached on the
        geometry + scene signature, so set_co (which rebuilds the domain) does not re-render."""
        key = self._solid_cache_key()
        if not force and key == self._solid_key and (self.solid_np is None or self.solid_np.shape[0] == self.N):
            return
        self.solid_np = self.scene.rasterize_full(self) if self.scene is not None else None
        self._solid_key = key

    def _compose(self):
        if self.solid_np is not None and not self.solid_np.any():
            self.solid_np = None                        # nothing solid: the web app's exact path
        if self.solid_np is None:
            self.mask_np = self.domain_mask_np          # the same object: no solids, no copy
            self.has_solids = False
            self.n_solid = 0
        else:
            self.mask_np = self.domain_mask_np & (self.solid_np ^ 1)
            self.has_solids = True
            self.n_solid = int(np.count_nonzero(self.solid_np))

    def update_solids(self, bbox=None):
        """Apply the scene's solids to the running simulation WITHOUT a reset.

        bbox = (i0, i1, j0, j1), a half-open cell box (what SolidScene.begin_stroke /
        extend_stroke / add / edit / remove return): only that box is re-rasterised (from the
        whole scene: objects, then every stroke in order), recomposed, uploaded (ONE kernel with
        a patch of M, which also zeroes the three time levels there) -- measured 30/09/2026:
        ~3 ms per painted segment at N = 2049 and at N = 8193. The box must cover every cell
        the edit can change; an empty box does nothing. bbox = None redoes the whole grid
        (N = 8193 with the Unlicensed lettering: ~3 s). Either way every time level is zeroed
        where the new mask is 0, the sources are re-placed (they snap OUT of solids) and the
        probes re-cached; the field elsewhere and the clock are untouched. When the last solid
        cell goes, the solver returns to the web app's exact path (has_solids False)."""
        if bbox is None:
            self._rasterize_solids(force=True)
            self._compose()
            self.M.from_numpy(self.mask_np.astype(mask_np_dtype()))
            k_zero_masked(self.U, self.M)
        else:
            N = self.N
            i0, i1, j0, j1 = (max(0, int(bbox[0])), min(N, int(bbox[1])),
                              max(0, int(bbox[2])), min(N, int(bbox[3])))
            if i1 <= i0 or j1 <= j0:
                return
            reg = (self.scene.rasterize(self, i0, i1, j0, j1) if self.scene is not None
                   else np.zeros((i1 - i0, j1 - j0), dtype=np.uint8))
            if self.solid_np is None:
                if not reg.any():
                    self._solid_key = self._solid_cache_key()
                    return
                self.solid_np = np.zeros((N, N), dtype=np.uint8)
                self.n_solid = 0
            self.n_solid += int(np.count_nonzero(reg)) - int(np.count_nonzero(self.solid_np[i0:i1, j0:j1]))
            self.solid_np[i0:i1, j0:j1] = reg
            patch = self.domain_mask_np[i0:i1, j0:j1] & (reg ^ 1)
            if self.n_solid == 0:
                self.solid_np = None
                self.mask_np = self.domain_mask_np
                self.has_solids = False
            else:
                if self.mask_np is self.domain_mask_np:
                    self.mask_np = self.domain_mask_np.copy()
                self.mask_np[i0:i1, j0:j1] = patch
                self.has_solids = True
            self._solid_key = self._solid_cache_key()
            k_patch_mask(self.U, self.M, np.ascontiguousarray(patch, dtype=mask_np_dtype()), i0, j0)
        self.recache_source()

    def set_solid_np(self, solid):
        """A raw solid mask (uint8/bool [i, j], 1 = solid) applied like update_solids(None).
        For tests and bitmaps. It survives reseed() and set_co(); a new N, lattice, shape,
        rotation or radius, or any scene edit, replaces it with the scene's rasterisation."""
        self.solid_np = None if solid is None else (np.asarray(solid) != 0).astype(np.uint8)
        self._solid_key = self._solid_cache_key()
        self._compose()
        self.M.from_numpy(self.mask_np.astype(mask_np_dtype()))
        k_zero_masked(self.U, self.M)
        self.recache_source()

    def _domain_rows(self, i0, i1):
        p = self.p
        R, ir = self.dom_R(), self.inner_R()
        X, Y = self._grid_xy(i0, i1)
        if p.shape == 'unlicensed':
            # rounded rectangle in the rotated frame, the same +1e-9 slack as insideNgon
            hw, hh, rc = self.plate_dims()
            co, si = math.cos(p.rot), math.sin(p.rot)
            x = np.abs(X * co + Y * si)
            y = np.abs(-X * si + Y * co)
            ax, ay = x - (hw - rc), y - (hh - rc)
            corner = (ax > 0) & (ay > 0)
            inside = (x <= hw + 1e-9) & (y <= hh + 1e-9) & ~(corner & (np.hypot(ax, ay) > rc + 1e-9))
        elif p.shape == 'circle':
            inside = (X * X + Y * Y) <= R * R
        elif p.shape == 'rhomb':
            co, si = math.cos(p.rot), math.sin(p.rot)
            x = X * co + Y * si
            y = -X * si + Y * co
            B = R * math.tan(p.rtheta / 2)
            inside = np.abs(x) / R + np.abs(y) / max(B, 1e-9) <= 1 + 1e-9
        elif p.shape != 'square':
            nn = int(p.shape[1:])
            r = np.hypot(X, Y)
            m = TAU / nn
            th = np.arctan2(Y, X) - p.rot
            a = np.fmod(np.fmod(th + PI / nn, m) + m, m) - PI / nn
            inside = r <= R * math.cos(PI / nn) / np.cos(a) + 1e-9
            inside[r < 1e-12] = True
        elif not self.sq_full():
            inside = np.maximum(np.abs(X), np.abs(Y)) <= R
        else:
            inside = np.ones_like(X, dtype=bool)
        f = np.ones_like(X)
        slabJ, lensJ, lensR = ir * 0.16, ir * 0.19, ir * 0.36
        if p.medium == 'slab':
            f[Y > slabJ] = 0.5
        elif p.medium == 'lens':
            ddy = Y - lensJ
            f[np.sqrt(X * X + ddy * ddy) < lensR] = 0.55
        co = p.Co * f
        return inside, co * co

    def set_co(self, co, rescale=False):
        """Change Co WITHOUT a reset.

        rescale=False is the web app's slider: only the per-cell Courant numbers change, and
        the two stored time levels are kept. The medium's wave speed then jumps at one instant
        -- a TEMPORAL BOUNDARY -- and every wave splits into a forward part and a smaller
        backward ("time-reflected") part, amplitude B = (Co2 - Co1)/(2 Co2) in the long-wave
        limit (README: time reflection).

        rescale=True reads the change as a new TIME STEP for the same medium (Co = c dt/h, so
        at fixed c and h a new Co IS a new dt): the previous time level is re-expressed at the
        new spacing, to second order,
            u_prev <- u - r (u - u_prev) + ((r^2 - r)/2) Co1^2 L(u),   r = Co2/Co1,
        so the velocity AND acceleration carry over and nothing is reflected (measured in 1D:
        the backward wave drops to 0.07-0.26% of the unrescaled one)."""
        r = co / self.p.Co if self.p.Co > 0 else 1.0
        if rescale and abs(r - 1) > 1e-15:
            k_rescale(self.U, self.M, self.CC, self.slot, (self.slot + 1) % 3, r, 1 if self.tri() else 0,
                      1 if self.p.stencil == 9 else 0, 1 if (self.p.bc == 'periodic' and not self.tri()) else 0)
        self.p.Co = co
        self.harm = self.harmonics()
        self.build_domain()

    # ---- source placement ---------------------------------------------------------------
    def round_cell(self, fi, fj):
        c, N = self.c(), self.N
        oi, oj = snap_q(fi - c), snap_q(fj - c)
        if self.tri():
            db = oj
            x = snap_q((fi - c) + (fj - c) * 0.5)
            B = sym_round(0, db)
            A = js_round(x - B * 0.5)
            return max(1, min(N - 2, int(c + A))), max(1, min(N - 2, int(c + B)))
        return (max(1, min(N - 2, int(sym_round(c, oi)))), max(1, min(N - 2, int(sym_round(c, oj)))))

    def snap_inside(self, fi, fj):
        i0, j0 = self.round_cell(fi, fj)
        if self.mask_np[i0, j0]:
            return (i0, j0, False)
        x, y = self.cell_xy(i0, j0)
        t = 0.96
        while t >= 0:                               # walk back in PHYSICAL space
            ii, jj = self.round_cell(*self.xy_to_cell(x * t, y * t))
            if self.mask_np[ii, jj]:
                return (ii, jj, True)
            t -= 0.02
        if self.has_solids:
            # the walk toward the centre found no free cell (e.g. the centre sits inside a
            # solid): the nearest free cell by lattice hops. Without solids the web app's
            # fallback (the centre cell) is kept exactly.
            f = self.nearest_free(i0, j0)
            if f is not None:
                return (f[0], f[1], True)
        c = int(self.c())
        return (c, c, True)

    def neighbour_offsets(self):
        """The stencil's links: 4 (sq5), 8 (sq9) or 6 (tri) index offsets."""
        if self.tri():
            return ((-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (-1, 1))
        if self.p.stencil == 9:
            return ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))
        return ((-1, 0), (1, 0), (0, -1), (0, 1))

    def hops(self, di, dj):
        """Lattice hop count of an index offset: the breadth-first distance in the stencil's
        neighbour graph -- |a|+|b| (sq5), max(|a|,|b|) (sq9), max(|a|,|b|,|a+b|) (tri, whose
        links are (+-1,0), (0,+-1), (1,-1), (-1,1)). Shortest paths can be taken monotone in
        both indices, so the array's edges never lengthen them."""
        a, b = np.abs(di), np.abs(dj)
        if self.tri():
            return np.maximum(np.maximum(a, b), np.abs(np.asarray(di) + np.asarray(dj)))
        if self.p.stencil == 9:
            return np.maximum(a, b)
        return a + b

    def nearest_free(self, i0, j0):
        """The free cell (mask 1, inside [1, N-2]^2) nearest to (i0, j0) in lattice hops -- what
        a breadth-first search over the neighbour graph finds -- with ties at the same hop count
        broken by the smallest physical distance, then the smallest (i, j). None if there is no
        free cell. Vectorised over a growing window instead of a Python BFS (a solid disk of
        radius 1000 cells would have been ~3M Python visits): every cell within h hops lies
        within h of (i0, j0) in each index, so a window of half-size h that holds a free cell
        at <= h hops holds the nearest one."""
        N, msk = self.N, self.mask_np
        x0, y0 = self.cell_xy(i0, j0)
        h = 4
        while True:
            a0, a1 = max(1, i0 - h), min(N - 1, i0 + h + 1)
            b0, b1 = max(1, j0 - h), min(N - 1, j0 + h + 1)
            whole = a0 == 1 and b0 == 1 and a1 == N - 1 and b1 == N - 1
            if a1 > a0 and b1 > b0:
                I, J = np.nonzero(msk[a0:a1, b0:b1])
                if I.size:
                    I, J = I + a0, J + b0
                    hp = self.hops(I - i0, J - j0)
                    hmin = int(hp.min())
                    if hmin <= h or whole:
                        sel = hp == hmin
                        I, J = I[sel], J[sel]
                        da, db = (I - self.c()).astype(np.float64), (J - self.c()).astype(np.float64)
                        X, Y = (da + db * 0.5, db * S3H) if self.tri() else (da, db)
                        d2 = (X - x0) ** 2 + (Y - y0) ** 2
                        k = np.lexsort((J, I, d2))[0]
                        return int(I[k]), int(J[k])
            if whole:
                return None
            h *= 2

    def src_ij(self):
        ir = self.inner_R()
        return self.snap_inside(*self.xy_to_cell(self.p.sx * ir, self.p.sy * ir))

    def vertices(self, inset, mode):
        if self.p.shape == 'circle':
            return []
        # the plate's corners are arcs: its 4 arc midpoints are the vertex set, not the 68
        # points of the sampled outline (which would put a source on every arc segment)
        poly = self.plate_corners() if self.p.shape == 'unlicensed' else self.shape_poly()
        Q = self.clip_convex(poly, self.clip_region())
        m, out = len(Q), []
        for k in range(m):
            x, y = Q[k]
            L = math.hypot(x, y) or 1
            if mode == 'wall':
                a, b = Q[(k + m - 1) % m], Q[(k + 1) % m]
                ux, uy, vx, vy = a[0] - x, a[1] - y, b[0] - x, b[1] - y
                lu, lv = math.hypot(ux, uy) or 1, math.hypot(vx, vy) or 1
                ux, uy, vx, vy = ux / lu, uy / lu, vx / lv, vy / lv
                bx, by = ux + vx, uy + vy
                lb = math.hypot(bx, by)
                sh = math.sqrt(max(0, (1 - (ux * vx + uy * vy)) / 2))
                want = inset / sh if (lb > 1e-12 and sh > 1e-9) else inset
                if want > 0.95 * L:
                    self.vtx_capped += 1
                d = min(want, 0.95 * L)
                dx, dy = (bx / lb, by / lb) if lb > 1e-12 else (-x / L, -y / L)
                px, py = x + dx * d, y + dy * d
            else:
                f = max(0, (L - inset)) / L
                px, py = x * f, y * f
            px, py = js_round(px * 1e9) / 1e9, js_round(py * 1e9) / 1e9
            out.append(self.snap_inside(*self.xy_to_cell(px, py)))
        seen, uniq = set(), []
        for v in out:
            if (v[0], v[1]) not in seen:
                seen.add((v[0], v[1]))
                uniq.append(v)
        return uniq

    def recache_source(self):
        p = self.p
        self.src = self.src_ij()
        self.vtx_capped = 0
        self.vtx = (self.vertices(3 * p.sigma, 'wall') if p.vtx_inset == 'wall'
                    else self.vertices(max(2, math.ceil(1.5 * p.sigma)), 'radial'))
        # the driven cells, on the device: the continuous source, or every vertex in phase
        cells = ([self.src[:2]] if p.src == 'cont'
                 else [v[:2] for v in self.vtx] if (p.src == 'vtx' and p.vtx_drive) else [])
        self.nsrc = len(cells)
        self.sc = ti.ndarray(dtype=ti.i32, shape=(max(1, len(cells)), 2))
        if cells:
            self.sc.from_numpy(np.array(cells, dtype=np.int32))
        self.clip_frac = self.pulse_clip_fraction()
        # the recorder's three probe cells, clamped into the array (the web app's own indices
        # always lie inside it for N >= 11; a clamp only keeps a GPU read in bounds)
        (ai, aj), (bi, bj) = self.probe_cells()
        mi, mj = self.mode_antinode() if p.src == 'mode' else (ai, aj)
        cl = lambda v: max(0, min(self.N - 1, int(v)))
        self.prb.from_numpy(np.array([cl(ai), cl(aj), cl(bi), cl(bj), cl(mi), cl(mj)], dtype=np.int32))

    # ---- the web app's probe placements (record(), the ledger) ---------------------------
    def pA(self):
        """Offset of probe A along +x (index units): max(3, round(innerR * 0.44))."""
        return max(3, js_round(self.inner_R() * 0.44))

    def pB(self):
        """Offset of probe B along +y (index units): max(4, round(innerR * 0.60))."""
        return max(4, js_round(self.inner_R() * 0.60))

    def probe_cells(self):
        """((i, j) of probe A, (i, j) of probe B): (c + pA, c) and (c, c + pB), c = round((N-1)/2)."""
        c = js_round((self.N - 1) / 2)
        return (c + self.pA(), c), (c, c + self.pB())

    def mode_antinode(self):
        """The (m, n) mode's antinode cell [round(z + L/(2m)), round(z + L/(2n))]."""
        z, L = self.mode_box()
        mm, mn = self.mode_mn()
        return js_round(z + L / (2 * mm)), js_round(z + L / (2 * mn))

    def pulse_clip_fraction(self):
        """Share of the seeded gaussian(s), by u^2, cut off at t = 0 (outside the mask, off
        the array, or on the clamped ring) -- the web app's pulseClipFraction."""
        p = self.p
        pts = self.vtx if p.src == 'vtx' else ([self.src] if p.src == 'pulse' else [])
        if not pts:
            return None
        s2 = 2 * p.sigma * p.sigma
        Mw = math.ceil(4 * p.sigma / (S3H if self.tri() else 1)) + 2
        N, c = self.N, self.c()
        # the web app sums over the array extended by Mw, but only where r^2 < 64 s2: the
        # union of the sources' windows of that radius is the same sum
        boxes = [self._window(pi, pj, math.sqrt(64 * s2), pad=Mw) for (pi, pj, _) in pts]
        i0, i1 = min(b[0] for b in boxes), max(b[1] for b in boxes)
        j0, j1 = min(b[2] for b in boxes), max(b[3] for b in boxes)
        I, J = np.meshgrid(np.arange(i0, i1, dtype=np.float64), np.arange(j0, j1, dtype=np.float64),
                           indexing='ij')
        da, db = I - c, J - c
        X, Y = (da + db * 0.5, db * S3H) if self.tri() else (da, db)
        v = np.zeros_like(X)
        for (pi, pj, _) in pts:
            cx, cy = self.cell_xy(pi, pj)
            r2 = (X - cx) ** 2 + (Y - cy) ** 2
            v += np.where(r2 < 64 * s2, np.exp(-r2 / s2), 0.0)
        w = v * v
        off = (I < 0) | (J < 0) | (I >= N) | (J >= N)
        cut = off.copy()
        inner = ~off
        ii, jj = I[inner].astype(int), J[inner].astype(int)
        ring = (ii == 0) | (jj == 0) | (ii == N - 1) | (jj == N - 1)
        ring_cut = self.tri() or p.bc == 'dirichlet'
        cut[inner] = (self.mask_np[ii, jj] == 0) | (ring & ring_cut)
        tot = w.sum()
        return float(w[cut].sum() / tot) if tot > 0 else None

    # ---- drive --------------------------------------------------------------------------
    def k_max(self):
        return max(1, math.floor(self.p.Co / (PPW_MIN * max(self.p.freq, 1e-9))))

    def harmonics(self):
        K, out, w = self.k_max(), [], self.p.wave
        if w == 'sine':
            return [(1, 1.0)]
        for k in range(1, K + 1):
            if w == 'square':
                if k % 2:
                    out.append((k, 4 / (PI * k)))
            elif w == 'saw':
                out.append((k, (2 if k % 2 else -2) / (PI * k)))
            elif w == 'tri':
                if k % 2:
                    out.append((k, 8 * (-1 if ((k - 1) / 2) % 2 else 1) / (PI * PI * k * k)))
        if not out:
            out.append((1, 8 / (PI * PI) if w == 'tri' else 4 / PI))
        return out

    def drive(self, step):
        s = 0.0
        for k, a in self.harm:
            s += a * math.sin(TAU * self.p.freq * k * step)
        return s

    # ---- seeding ------------------------------------------------------------------------
    def mode_box(self):
        if self.sq_full():
            return 0, self.N - 1
        hw = math.floor(self.dom_R())
        return self.c() - hw - 1, 2 * hw + 2

    def mode_mn(self):
        L = self.mode_box()[1]
        return min(self.p.mm, L - 1), min(self.p.mn, L - 1)

    def _window(self, bi, bj, rphys, pad=0):
        """Index box holding the physical disk of radius rphys about cell (bi, bj). On the
        triangular lattice one row is S3H apart and a row shift adds up to half a column."""
        hj = math.ceil(rphys / S3H) + 2 if self.tri() else math.ceil(rphys) + 2
        hi = math.ceil(rphys + hj * 0.5) + 2 if self.tri() else hj
        lo, hi_ = -pad, self.N + pad
        return (max(lo, bi - hi), min(hi_, bi + hi + 1), max(lo, bj - hj), min(hi_, bj + hj + 1))

    def _bump(self, U, bi, bj, amp):
        """The web app sweeps the whole grid (addBump). Past r = 38.7 sigma, exp(-r^2/2s^2)
        underflows to EXACTLY 0 in f64 (e^-745), so a window of that radius adds the very same
        values -- and at N = 8193 it is ~10^3 times less work and memory."""
        s = self.p.sigma
        i0, i1, j0, j1 = self._window(bi, bj, math.sqrt(2 * 746) * s)
        bx, by = self.cell_xy(bi, bj)
        X, Y = self._grid_xy(i0, i1, j0, j1)
        dx, dy = X - bx, Y - by
        U[i0:i1, j0:j1] += amp * np.exp(-(dx * dx + dy * dy) / (2 * s * s))

    def seed(self):
        p, N = self.p, self.N
        U = np.zeros((N, N))
        self.drops = 0
        if p.src == 'impulse':
            U[self.src[0], self.src[1]] = 1.0
        elif p.src == 'pulse':
            self._bump(U, self.src[0], self.src[1], 1.0)
        elif p.src == 'vtx':
            for v in self.vtx:                          # sequential, as the JS accumulates
                self._bump(U, v[0], v[1], 1.0)
        elif p.src == 'mode':
            z, L = self.mode_box()
            mm, mn = self.mode_mn()
            I, J = np.meshgrid(np.arange(N, dtype=np.float64), np.arange(N, dtype=np.float64), indexing='ij')
            U = np.sin(mm * PI * (I - z) / L) * np.sin(mn * PI * (J - z) / L)
        U[self.mask_np == 0] = 0.0                      # applyMask
        self.slot = 0
        self.set_levels(U, U)                           # starts at rest
        k_zero(self.U, 2)
        self.step_n = 0
        self.t = 0.0
        # clearField: the histories restart; the mode probe gets the seeded value (step 0)
        self.hist = {k: [] for k in HIST_CAP}
        if self.record and p.src == 'mode':
            ai, aj = self.mode_antinode()
            if 0 <= ai < N and 0 <= aj < N:
                self.hist['mode'].append(float(self.fdt(U[ai, aj])))
        self.amax = float(np.abs(U).max()) if N else 0.0   # the web app sets S.amax at seed

    def reseed(self):
        """Everything the web app's reseed() does that affects the field."""
        if self.p.N % 2 == 0:
            self.p.N += 1                                # the web app's N slider: odd -> true centre
        if self.N != self.p.N:
            self._alloc(self.p.N)
        self.harm = self.harmonics()
        self.build_domain()
        self.recache_source()
        self.seed()

    def drop_pulse(self):
        """Superpose a gaussian at the source cell onto the running field (both time levels,
        so the new bump starts at rest), then applyMask -- the web app's dropPulse."""
        s = self.src_ij()
        add = np.zeros((self.N, self.N))
        self._bump(add, s[0], s[1], 1.0)
        out = self.mask_np == 0
        u0, u1 = self.level(0) + add, self.level(1) + add
        u0[out] = 0.0
        u1[out] = 0.0
        self.set_levels(u0, u1)
        self.drops += 1
        self.amax = float(np.abs(u1).max())             # dropPulse refreshes S.amax too

    # ---- the time step --------------------------------------------------------------------
    def step(self, n=1):
        """n steps, in launches of CHUNKS steps each (one kernel call per launch)."""
        p = self.p
        lat = LAT_TRI if self.tri() else LAT_SQ
        bc = BC_DIR if lat == LAT_TRI else BC_CODE[p.bc]
        nine = lat == LAT_SQ and p.stencil == 9
        # a solid can touch the array ring, which only the masked pass clamps
        masked = not self.sq_full() or self.has_solids
        slitm = self.has_solids
        rec = bool(self.record)
        srck = SRC_NONE
        jw = jd = cA = cB = half = 0
        if p.src == 'slit':
            srck = SRC_SLIT
            c = js_round(self.c())
            ir = self.inner_R()
            ox, oy = js_round(p.sx * ir), js_round(p.sy * ir)
            jw, jd = c + oy - js_round(ir * 0.40), c + oy - js_round(ir * 0.76)
            half, off = max(2, js_round(ir * 0.07)), js_round(ir * 0.24)
            cA, cB = c + ox - off, c + ox + off
        elif self.nsrc:
            srck = SRC_CELLS
        kmur = (p.Co - 1) / (p.Co + 1)
        start, pending = self.hrow, 0
        while n > 0:
            K = next(k for k in CHUNKS if k <= n)
            if rec and pending + K > HCAP:
                self._flush_hist(start, pending)
                start, pending = self.hrow, 0
            if srck != SRC_NONE:
                # the drive, host-side with the same sin() as before, for the K steps ahead
                dv = np.zeros(KMAX, dtype=self.fdt)
                for q in range(K):
                    dv[q] = self.drive(self.step_n + q)
                self.drv.from_numpy(dv)
            if rec:
                k_run_rec(self.U, self.M, self.CC, self.drv, self.sc, self.hist_dev, self.prb,
                          self.slot, K, lat, nine, bc, masked, srck,
                          self.nsrc, p.damp, kmur, jw, jd, cA, cB, half, slitm, self.hrow)
            else:
                k_run(self.U, self.M, self.CC, self.drv, self.sc, self.slot, K, lat, nine, bc, masked, srck,
                      self.nsrc, p.damp, kmur, jw, jd, cA, cB, half, slitm)
            self.slot = (self.slot + K) % 3
            for _ in range(K):                   # the web app's clock: S.t += S.Co per step
                self.t += p.Co
            self.step_n += K
            n -= K
            if rec:
                self.hrow = (self.hrow + K) % HCAP
                pending += K
        if rec and pending:
            self._flush_hist(start, pending)

    def _flush_hist(self, start, cnt):
        """Append the device ring's rows [start, start+cnt) to m.hist: ONE device->host copy
        per step() call (not per launch), since each copy is a sync (~0.4 ms, 29/09/2026)."""
        H = self.hist_dev.to_numpy()[:HCAP]
        rows = H[(start + np.arange(cnt)) % HCAP].astype(np.float64)
        co = self.p.Co
        E = rows[:, 0] / (2 * max(co * co, 1e-12)) + 0.5 * rows[:, 1]    # the web app's formula
        h = self.hist
        h['E'].extend(E.tolist())
        h['amax'].extend(rows[:, 2].tolist())
        h['pA'].extend(rows[:, 3].tolist())
        h['pB'].extend(rows[:, 4].tolist())
        if self.p.src == 'mode':
            h['mode'].extend(rows[:, 5].tolist())
        for k, cap in HIST_CAP.items():            # ENERGY.shift() etc.: keep the newest
            if len(h[k]) > cap:
                del h[k][:len(h[k]) - cap]
        self.amax = float(rows[-1, 2])

    # ---- readouts ---------------------------------------------------------------------------
    def level(self, k):
        """A time level as a numpy array [i, j]: k = 1 the current one, k = 0 the previous."""
        out = np.empty((self.N, self.N), dtype=self.fdt)
        k_store(self.U, (self.slot + k) % 3, out)
        return out

    def set_levels(self, u_prev, u_cur):
        k_load(self.U, self.slot, np.ascontiguousarray(u_prev, dtype=self.fdt))
        k_load(self.U, (self.slot + 1) % 3, np.ascontiguousarray(u_cur, dtype=self.fdt))

    def value(self, i, j):
        """One cell of the current level (a device read: fine for probes, not for sweeps)."""
        return float(self.U[(self.slot + 1) % 3, i, j])

    def stats(self):
        """(energy, peak |u|) of the CURRENT state with the web app's record() formula (one
        kernel + one sync). While the recorder runs, m.hist['E'][-1] / m.amax hold the same
        values for the last step without a kernel."""
        k_stats(self.U, self.M, self.acc, self.slot, (self.slot + 1) % 3)
        a = self.acc.to_numpy()[0].astype(np.float64)
        E = a[0] / (2 * max(self.p.Co * self.p.Co, 1e-12)) + 0.5 * a[1]
        return float(E), float(a[2])

    def field(self):
        return self.level(1)

    def domain_cells(self):
        """Free (wave-carrying) cells: the domain minus the solids."""
        return int(self.mask_np.sum())

    def solid_cells(self):
        """Solid cells inside the domain (solid cells outside it change nothing)."""
        if self.solid_np is None:
            return 0
        return int((self.domain_mask_np & self.solid_np).sum())
