"""Tests of the absorbing boundaries (09/10/2026, round 5: Mur 2nd order, CPML and the sponge; NotesCLAUDE/
absorbing_MIRROR.md). The expectations are the research's numbers (NotesCLAUDE/absorbing_boundaries_MIRROR.md,
ResearchCLAUDE/AbsorbingBoundariesCLAUDE/results_*_MIRROR.json).

  1  bc_info / edge_info: the options per lattice and shape, the fallbacks and their reasons, the layer, the cap
  2  parity: every configuration WITHOUT an absorbing boundary is bit-identical to the baseline core
     (ScratchCLAUDE/Round5CLAUDE/BeforeAbsorbCLAUDE: fields, probe and peak histories exactly; the energy history
     to 1e-14, since the recorder's slot sums are atomic adds in no fixed order); and the open domain keeps today's
     update bit for bit (a sponge with its codes cleared == the plain run; CPML and the open edge before the front
     reaches the layer)
  3  the kernel == an independent numpy reference of the same boundary (2D, all four walls and the corners, to
     rounding): CPML sq5 / sq9, std / high, a lens medium, global damping, pinned solids in the frame; Mur2 sq5 /
     sq9; the sponge frame sq5 / sq9 / tri; the open edge on the sq circle, the tri hexagon, the plate
  4  plane-wave reflection |R| (the research's 1D-exact reduction of a straight wall, with the core's own
     coefficient tables): CPML std / high, Mur2 (and its discrete theory), the default sponges, against the
     research's table entry by entry (sq5); sq9 against bounds
  5  the 2D box, a broadband pulse (sigma 3) from the centre and near a corner, in the core: CPML L10 <= ~2e-4,
     L20 <= ~2e-6, Mur2 ~8.8e-3 / 0.10, the sponge frame L40 ~1.4e-2 / 3.8e-2 (sq5, sq9; Co 0.5, 0.7)
  6  the open edge on masked shapes (sponge L40, d_max 0.2): the sq circle, the tri hexagon, the tri circle at R 100,
     ~1.6-1.7e-2 for a sigma-3 pulse
  7  stability: 20,000 steps of white noise (Co 0.5 / 0.7; sq9 0.85; tri 0.8; CPML also 0.7071)
  8  energy: with the sponge and no drive the exact leapfrog energy never increases (frame sq / tri, open circle,
     open tri hexagon, with solids of both types); the open domain's recorded energy never rises above its start
  9  composition: global damping, solids in the domain and in the layer (both types, the walls path), set_co with
     rescale, the recorder (the open domain only) and the probes, the beam aperture, vertex sources, the mode note,
     periodic, painting with a layer (incremental == full), a source snapped out of the layer, scale mode (the
     sponge's width follows the scale; CPML stays in cells), a bc / edge switch without a reseed
 10  cost per step (--bench): N 1025 / 4097 / 8193, plain vs each boundary
 11  long waves (scale mode): the box with the pulse widening with the domain (sigma 3 / 6 / 12): CPML std / high
     keep their level, the sponge following the scale (40 f cells, auto d_max 8 / L) keeps 1.4e-2

    python absorbing_tests_MIRROR.py [--arch cuda|vulkan|cpu] [--f32] [--quick] [--only 1,2,...] [--bench]

Never bare pytest. Vulkan is f32 only. *Generated from scratch -- Claude Opus 5.5 -- 09/10/2026*
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import membrane_core_MIRROR as mc          # noqa: E402

RES = os.path.join(HERE, 'ResearchCLAUDE', 'AbsorbingBoundariesCLAUDE')
BASE_PATH = os.path.join(HERE, 'ScratchCLAUDE', 'Round5CLAUDE', 'BeforeAbsorbCLAUDE', 'membrane_core_MIRROR.py')
FAILS = []
S3H = math.sqrt(3) / 2
D2R = math.pi / 180


def ok(name, cond, det=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('  | ' + det if det else ''), flush=True)
    if not cond:
        FAILS.append(name)


def f64():
    return bool(mc.FP64)


def tol(a64, a32):
    return a64 if f64() else a32


def load_base():
    """The baseline core (every .py after the decisions step), sharing this process's Taichi."""
    spec = importlib.util.spec_from_file_location('membrane_core_base_MIRROR', BASE_PATH)
    B = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    spec.loader.exec_module(B)
    sys.path[:] = saved
    B.MASK_DT, B.SQ9_CONSISTENT, B.FP64, B.ACC_DT, B.WORD_DT = mc.MASK_DT, mc.SQ9_CONSISTENT, mc.FP64, mc.ACC_DT, mc.WORD_DT
    return B


def mk(**kw):
    return mc.Membrane(mc.Params(**kw))


def gauss_xy(m, x0, y0, s):
    X, Y = m._grid_xy()
    return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * s * s))


def seed_cell(m, i, j, s):
    """A gaussian of width s at cell (i, j) on both levels (at rest), masked."""
    x0, y0 = m.cell_xy(i, j)
    g = gauss_xy(m, x0, y0, s) * (m.mask_np != 0)
    m.set_levels(g, g)
    return g


def noise_seed(m, rng):
    """White noise in the open domain's cells, both levels (at rest)."""
    u = rng.uniform(-1, 1, (m.N, m.N)) * (m.open_mask() != 0)
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0
    m.set_levels(u, u)
    return u


# ---- the numpy reference (section 3) --------------------------------------------------------------------
class Ref:
    """An independent numpy implementation of the core's update with an absorbing boundary, built from the
    Membrane's geometry (the device mask codes, CC, the layer tables) but NOT from its kernels. kind: 'plain' |
    'mur2' | 'sponge' | 'cpml'. Non-walls path only (pinned solids that block no link)."""

    def __init__(self, m):
        self.m, self.N = m, m.N
        self.tri = m.tri()
        self.nine = (not self.tri) and m.p.stencil == 9
        self.code = m._dev_mask().astype(np.int64)
        self.free = self.code != 0
        self.CC = m.cc_np.astype(np.float64)
        self.dg = float(m.p.damp)
        lay = m.abs_layer()
        bc = m.bc_effective()
        self.kind = lay['kind'] if lay is not None else ('mur2' if bc == 'mur2' else 'plain')
        N = self.N
        if self.kind == 'sponge':
            if lay['where'] == 'frame':
                # the frame from the geometry: d = d_max ((L - depth) / L)^p, depth the physical distance to the ring
                I = np.arange(N)[:, None]
                J = np.arange(N)[None, :]
                dep = np.minimum(np.minimum(I, J), np.minimum(N - 1 - I, N - 1 - J)) * (S3H if self.tri else 1.0)
                d = np.where(dep < lay['L'] - 1e-9, lay['dmax'] * np.clip((lay['L'] - dep) / lay['L'], 0, 1) ** lay['p'],
                             0.0)
                d = np.where(self.free, d, 0.0)
            else:
                # the open edge: the ramp quantised in s (the codes)
                d = np.where(self.code >= 2, lay['dmax'] * ((self.code - 1.5) / mc.SPONGE_LEVELS) ** lay['p'], 0.0)
            self.d = d
            self.lay = d > 0
            self.inv = 1.0 / (1.0 + 0.5 * d)
            self.c2 = (1.0 - 0.5 * d) * self.inv
        if self.kind == 'cpml':
            Lc = lay['Lc']
            L2 = 2 * Lc
            T = m._aux_tables(lay)[mc.AUX_TAB:].reshape(6, L2)
            idx = np.full(N, -1)
            for i in range(1, N - 1):
                idx[i] = i - 1 if i <= Lc else (Lc + (i - (N - 1 - Lc)) if i >= N - 1 - Lc else -1)
            self.fr = idx >= 0

            def arr(k, dflt):
                out = np.full(N, dflt)
                out[self.fr] = T[k][idx[self.fr]]
                return out
            self.bc_, self.ac_ = arr(0, 1.0), arr(1, 0.0)
            self.bm_, self.am_ = arr(2, 1.0), arr(3, 0.0)
            self.bp_, self.ap_ = arr(4, 1.0), arr(5, 0.0)
            z = lambda: np.zeros((N, N))
            self.mem = [z() for _ in range(6)]
        self.u = m.level(1).astype(np.float64)
        self.up = m.level(0).astype(np.float64)
        self.first = True               # Mur2's start-up step (the core's BC_MUR2S: Mur1 on the sides once)

    def _lap(self, u):
        L = np.zeros_like(u)
        I = slice(1, -1)
        c = u[I, I]
        e = u[:-2, I] + u[2:, I] + u[I, :-2] + u[I, 2:]
        if self.tri:
            L[I, I] = (2.0 / 3.0) * (e + u[2:, :-2] + u[:-2, 2:] - 6 * c)
        elif self.nine:
            dg = u[:-2, :-2] + u[2:, :-2] + u[:-2, 2:] + u[2:, 2:]
            L[I, I] = (2.0 / 3.0) * (e - 4 * c) + (1.0 / 6.0) * (dg - 4 * c)
        else:
            L[I, I] = e - 4 * c
        return L

    def _mem(self, gp, gm, axis, k0):
        """One axis' CPML recursion on whole arrays (old memory in, new memory out; the caller keeps it)."""
        if axis == 0:
            bm, am, bp, ap, bc, ac = (x[:, None] for x in (self.bm_, self.am_, self.bp_, self.ap_, self.bc_, self.ac_))
        else:
            bm, am, bp, ap, bc, ac = (x[None, :] for x in (self.bm_, self.am_, self.bp_, self.ap_, self.bc_, self.ac_))
        pm, pp, zz = self.mem[k0], self.mem[k0 + 1], self.mem[k0 + 2]
        npm = bm * pm + am * gm
        npp = bp * pp + ap * gp
        dq = (gp + npp) - (gm + npm)
        nz = bc * zz + ac * dq
        return dq + nz, (npm, npp, nz)

    def _cpml_lap(self, u):
        """sq5: D~xx u + D~yy u; sq9: D~xx(u + D~yy u / 6) + D~yy u (the exactly stretched L9), D~yy u from the y memory
        in the y frame rows (all columns: a neighbour's value is needed), the raw Dyy u elsewhere."""
        N = self.N
        I = slice(1, -1)
        fx = self.fr[:, None] & np.ones((1, N), bool)
        fy = np.ones((N, 1), bool) & self.fr[None, :]
        gp = np.zeros_like(u)
        gm = np.zeros_like(u)
        gp[:, I] = u[:, 2:] - u[:, I]
        gm[:, I] = u[:, I] - u[:, :-2]
        Dy_s, ny = self._mem(gp, gm, 1, 3)
        Dy = np.where(fy, Dy_s, gp - gm)
        Dy[0, :] = Dy[-1, :] = 0.0
        w = u + Dy / 6.0 if self.nine else u
        gp = np.zeros_like(u)
        gm = np.zeros_like(u)
        gp[I, :] = w[2:, :] - w[I, :]
        gm[I, :] = w[I, :] - w[:-2, :]
        Dx_s, nx = self._mem(gp, gm, 0, 0)
        lx = np.where(fx, Dx_s, gp - gm)
        upd = fx & self.free
        for k in range(3):
            self.mem[k][upd] = nx[k][upd]
        upd = fy & self.free
        for k in range(3):
            self.mem[3 + k][upd] = ny[k][upd]
        for k in range(6):
            self.mem[k][~self.free] = 0.0
        out = np.zeros_like(u)
        frame = fx | fy
        out[I, I] = np.where(frame, lx + Dy, self._lap(u))[I, I]
        return out

    def step(self):
        u, up, N = self.u, self.up, self.N
        lap = self._cpml_lap(u) if self.kind == 'cpml' else self._lap(u)
        un = (2 * u - up + self.CC * lap) - self.dg * (u - up)
        if self.kind == 'sponge':
            un = np.where(self.lay, (2 * u + self.CC * lap - self.dg * (u - up)) * self.inv - self.c2 * up, un)
        un[~self.free] = 0.0
        if self.kind == 'mur2':
            self._mur2(un)
        else:
            un[0, :] = un[-1, :] = un[:, 0] = un[:, -1] = 0.0
        if self.kind == 'mur2' and (~self.free).any():
            un[~self.free] = 0.0
        self.up, self.u = u, un
        self.first = False

    def _mur2(self, un):
        N, a, b = self.N, self.up, self.u
        co = np.sqrt(self.CC)
        k = (co - 1) / (co + 1)
        g = 2 / (co + 1)
        e = self.CC / (2 * (co + 1))
        t = np.arange(1, N - 1)
        for r0, r1, along_i in ((0, 1, False), (N - 1, N - 2, False), (0, 1, True), (N - 1, N - 2, True)):
            if along_i:     # the ring row j = r0: cells (t, r0)
                d0 = b[t - 1, r0] - 2 * b[t, r0] + b[t + 1, r0]
                d1 = b[t - 1, r1] - 2 * b[t, r1] + b[t + 1, r1]
                kk, gg, ee = k[t, r0], g[t, r0], e[t, r0]
                if self.first:
                    un[t, r0] = b[t, r1] + kk * (un[t, r1] - b[t, r0])
                else:
                    un[t, r0] = -a[t, r1] + kk * (un[t, r1] + a[t, r0]) + gg * (b[t, r0] + b[t, r1]) + ee * (d0 + d1)
            else:
                d0 = b[r0, t - 1] - 2 * b[r0, t] + b[r0, t + 1]
                d1 = b[r1, t - 1] - 2 * b[r1, t] + b[r1, t + 1]
                kk, gg, ee = k[r0, t], g[r0, t], e[r0, t]
                if self.first:
                    un[r0, t] = b[r1, t] + kk * (un[r1, t] - b[r0, t])
                else:
                    un[r0, t] = -a[r1, t] + kk * (un[r1, t] + a[r0, t]) + gg * (b[r0, t] + b[r1, t]) + ee * (d0 + d1)
        for ci, cj, si, sj in ((0, 0, 1, 1), (N - 1, 0, -1, 1), (0, N - 1, 1, -1), (N - 1, N - 1, -1, -1)):
            kd = (co[ci, cj] - math.sqrt(2)) / (co[ci, cj] + math.sqrt(2))
            un[ci, cj] = b[ci + si, cj + sj] + kd * (un[ci + si, cj + sj] - b[ci, cj])


# ---- section 1 ----------------------------------------------------------------------------------------------
def t1_info():
    print('-- 1. bc_info / edge_info', flush=True)
    sq = mk(N=161, shape='array', bc='cpml')
    bi = sq.bc_info()
    ok('1a. the full square: every bc acts, cpml runs its frame, the open domain is the array minus the frame',
       bi['options'] == mc.BCS and bi['effective'] == 'cpml' and bi['absorbing'] and bi['layer']['Lc'] == 10
       and 'CPML, 10 cells' in bi['ring'] and abs(sq.inner_R() - 70.0) < 1e-12 and not bi['edge_applies'],
       f'options {bi["options"]}; ring "{bi["ring"]}"; r_in {sq.inner_R()}')
    tri = mk(N=161, shape='array', lattice='tri', bc='sponge')
    bt = tri.bc_info()
    falls = {}
    for b in ('mur', 'mur2', 'cpml'):
        mt = mk(N=161, shape='array', lattice='tri', bc=b)
        falls[b] = (mt.bc_info()['effective'], mt.bc_info()['reason'] is not None, mt.abs_layer())
    ok('1b. the triangular array: dirichlet / sponge / neumann / periodic act; mur, mur2, cpml fall back to the clamped '
       'ring with a reason', bt['options'] == ['dirichlet', 'sponge', 'neumann', 'periodic'] and bt['effective'] == 'sponge'
       and bt['layer']['kind'] == 'sponge' and all(v[0] == 'dirichlet' and v[1] and v[2] is None for v in falls.values()),
       f'options {bt["options"]}; fallbacks {falls}')
    mm = mk(N=161, shape='circle', bc='cpml')
    bm = mm.bc_info()
    mo = mk(N=161, shape='circle', bc='periodic', edge='open')
    bo = mo.bc_info()
    ok('1c. a masked square-lattice shape: mur2 / cpml / sponge fall back to the clamped ring (use edge: open); periodic '
       'is exclusive with an open edge', bm['effective'] == 'dirichlet' and bm['fallback'] and 'edge: open' in bm['reason']
       and bm['options'] == ['dirichlet', 'mur', 'neumann', 'periodic'] and bo['effective'] == 'dirichlet'
       and bo['fallback'] and not mo.torus() and 'periodic' not in bo['options'],
       f'cpml on a circle -> {bm["effective"]} ("{bm["reason"][:60]}..."); periodic + open -> {bo["effective"]}')
    e1 = mo.edge_info()
    mc8 = mk(N=161, shape='circle', edge='open', rad=1.0)
    e2 = mc8.edge_info()
    mc6 = mk(N=161, shape='circle', edge='open', rad=0.5)
    e3 = mc6.edge_info()
    ef = sq.edge_info()
    ok('1d. edge_info: open = a sponge outside the drawn edge; the radius is capped so the layer fits (and says so); '
       'n/a on the full array', e1['applies'] and e1['kind'] == 'open' and e2['capped'] and abs(e2['R_open'] - 58.5) < 1e-9
       and not e3['capped'] and abs(e3['R_open'] - 39.25) < 1e-9 and not ef['applies'] and ef['options'] == []
       and any('shrank' in t for t, _ in e2['lines']),
       f'rad 1: R {e2["R_requested"]:.2f} -> {e2["R_open"]:.2f} (L {e2["layer"]["L"]:.0f}); rad 0.5: {e3["R_open"]:.2f}; '
       f'lines {[t[:40] for t, _ in e2["lines"]]}')
    small = mk(N=25, shape='array', bc='cpml')
    bs = small.bc_info()
    ok('1e. cpml on an array too small for its frame falls back with a reason', bs['effective'] == 'dirichlet'
       and 'needs N >= 29' in (bs['reason'] or ''), f'N 25: {bs["effective"]}, "{bs["reason"]}"')
    w = {}
    for n in (161, 321, 1025):
        w[n] = (mk(N=n, shape='circle', edge='open').sponge_width(), mk(N=n, shape='circle', edge='open').sponge_dmax())
    ok('1f. the sponge defaults: L 20 / d_max 0.35 under a half-size of 160, else L 40 / d_max 0.2; the mode source is '
       'n/a with an absorbing boundary', w[161] == (20.0, 0.35) and w[321] == (40.0, 0.2) and w[1025] == (40.0, 0.2)
       and 'no standing modes' in (mk(N=161, bc='mur2', src='mode').mode_note() or '')
       and 'no standing modes' in (mk(N=161, shape='circle', edge='open', src='mode').mode_note() or ''),
       f'widths {w}')


# ---- section 2 ----------------------------------------------------------------------------------------------
PARITY = [dict(), dict(bc='mur'), dict(bc='neumann', stencil=9), dict(bc='periodic'), dict(bc='mur', stencil=9, Co=0.8),
          dict(shape='circle', rad=1.3, bc='mur'), dict(shape='circle', rad=1.3, bc='neumann', stencil=9),
          dict(shape='circle', rad=1.3, bc='periodic'), dict(lattice='tri', shape='n6'),
          dict(lattice='tri', shape='array', bc='periodic'), dict(lattice='tri', shape='array', bc='neumann'),
          dict(lattice='tri', shape='array', bc='mur'), dict(lattice='tri', shape='circle', bc='periodic', rad=1.3),
          dict(shape='circle', edge='rigid', src='vtx'), dict(shape='n5', edge='rigid', stencil=9),
          dict(src='beam', beam_angle=30), dict(src='packet', lattice='tri', shape='n6'),
          dict(src='cont', medium='lens'), dict(src='slit', stencil=9), dict(src='mode', mm=2, mn=3),
          dict(shape='unlicensed'), dict(shape='unlicensed', lattice='tri'), dict(src='vtx', vtx_drive=True, shape='n5'),
          dict(src='vtx', vtx_drive=True, vtx_soft=False, shape='n3', lattice='tri'),
          dict(damp=0.01, bc='mur', stencil=9), dict(scale='domain', N=201, shape='n8'), dict(medium='slab', bc='neumann')]


def t2_parity(quick):
    print('-- 2. parity with the baseline core', flush=True)
    B = load_base()
    cfgs = PARITY[::3] if quick else PARITY
    bad = []
    t0 = time.time()
    for kw in cfgs:
        prm = dict(N=129, src='pulse', sx=0.2, sy=-0.1)
        prm.update(kw)
        for rec in (True, False):
            a = mc.Membrane(mc.Params(**prm))
            b = B.Membrane(B.Params(**prm))
            a.record = b.record = rec
            a.step(400)
            b.step(400)
            if kw is cfgs[0]:
                a.set_co(0.6, rescale=True)
                b.set_co(0.6, rescale=True)
                a.drop_pulse()
                b.drop_pulse()
                a.step(100)
                b.step(100)
            same = np.array_equal(a.level(0), b.level(0)) and np.array_equal(a.level(1), b.level(1))
            if rec:
                Ea, Eb = np.array(a.hist['E']), np.array(b.hist['E'])
                same = (same and len(Ea) == len(Eb) and np.abs(Ea - Eb).max() <= 1e-14 * max(np.abs(Eb).max(), 1e-300)
                        * (1 if f64() else 1e8)
                        and all(a.hist[k] == b.hist[k] for k in ('amax', 'pA', 'pB', 'mode')))
            sa, sb = a.stats(), b.stats()
            same = same and sa[1] == sb[1] and abs(sa[0] - sb[0]) <= tol(1e-14, 1e-6) * abs(sb[0])
            same = same and a.src == b.src and a.vtx == b.vtx and a.clip_frac == b.clip_frac
            if not same:
                bad.append(f'{kw} rec={rec}')
    ok('2a. every configuration without an absorbing boundary == the baseline core (fields and probe / peak histories '
       'exactly; energy to rounding), recorder on and off, a set_co(rescale) and a dropped pulse',
       not bad, f'{2 * len(cfgs)} runs, 400-500 steps ({time.time() - t0:.0f} s)' + (f'; DIFFER: {bad}' if bad else ''))
    # 2b. the sponge kernel with every layer code cleared computes the plain update bit for bit
    res = []
    for lat in ('sq', 'tri'):
        for nine in ((False, True) if lat == 'sq' else (False,)):
            kw = dict(N=161, lattice=lat, stencil=9 if nine else 5, shape='array', src='pulse', sx=0.3, sy=0.2, sigma=4)
            a = mk(bc='sponge', **kw)
            b = mk(bc='dirichlet', **kw)
            a.abs_np = np.zeros_like(a.abs_np)          # every code back to 1: the plain update everywhere
            a._open_cache = None
            a._upload_mask()
            a.set_levels(b.level(0), b.level(1))
            a.step(500)
            b.step(500)
            res.append((f'{lat}{9 if nine else 5}', np.array_equal(a.field(), b.field())))
    ok('2b. the sponge kernel with no layer code computes TODAY\'s update bit for bit (sq5, sq9, tri; 500 steps)',
       all(r[1] for r in res), str(res))
    # 2c. CPML / the open edge: an impulse at the centre, before the front reaches the layer, == the plain run
    res = []
    for kw_a, kw_b in ((dict(shape='array', bc='cpml'), dict(shape='array', bc='dirichlet')),
                       (dict(shape='array', bc='cpml', stencil=9), dict(shape='array', bc='dirichlet', stencil=9)),
                       (dict(shape='circle', rad=0.6, edge='open'), dict(shape='circle', rad=0.6, edge='pinned')),
                       (dict(shape='n6', rad=0.6, edge='open', lattice='tri'), dict(shape='n6', rad=0.6, lattice='tri'))):
        a = mk(N=161, src='impulse', **kw_a)
        b = mk(N=161, src='impulse', **kw_b)
        a.step(30)
        b.step(30)
        res.append(np.array_equal(a.field(), b.field()) and np.array_equal(a.level(0), b.level(0)))
    ok('2c. CPML (sq5, sq9) and the open edge (sq circle, tri hexagon): an impulse, 30 steps (the front has not reached '
       'the layer): bit-identical to the plain run', all(res), str(res))


# ---- section 3 ----------------------------------------------------------------------------------------------
def t3_reference(quick):
    print('-- 3. kernel == the numpy reference', flush=True)
    cases = [('CPML sq5 std', dict(N=61, shape='array', bc='cpml')),
             ('CPML sq5 high', dict(N=71, shape='array', bc='cpml', cpml_level='high')),
             ('CPML sq9 std', dict(N=61, shape='array', bc='cpml', stencil=9, Co=0.8)),
             ('CPML sq9 high + lens + damping', dict(N=71, shape='array', bc='cpml', stencil=9, cpml_level='high',
                                                    medium='lens', damp=0.01)),
             ('Mur2 sq5', dict(N=61, bc='mur2')),
             ('Mur2 sq9 + slab', dict(N=61, bc='mur2', stencil=9, medium='slab', Co=0.8)),
             ('sponge frame sq5', dict(N=81, bc='sponge')),
             ('sponge frame sq9 + damping', dict(N=81, bc='sponge', stencil=9, damp=0.01, Co=0.8)),
             ('sponge frame tri', dict(N=81, shape='array', lattice='tri', bc='sponge', Co=0.75)),
             ('open sq circle', dict(N=81, shape='circle', edge='open')),
             ('open tri hexagon', dict(N=91, shape='n6', edge='open', lattice='tri')),
             ('open sq plate (pinned lettering)', dict(N=121, shape='unlicensed', edge='open')),
             ('open sq9 octagon + lens', dict(N=81, shape='n8', edge='open', stencil=9, medium='lens', rot=0.2))]
    if quick:
        cases = cases[::2]
    T = 300
    worst = []
    for name, kw in cases:
        m = mk(src='pulse', sigma=2.5, sx=0.35, sy=-0.25, **kw)
        if name.startswith('CPML') and 'lens' not in name:
            # a pinned solid block inside the frame (the memory of its cells is held at 0)
            blk = np.zeros((m.N, m.N), np.uint8)
            blk[3:6, m.N // 2 - 4:m.N // 2 + 2] = 1
            m.set_solid_np(blk)
            m.reseed()
            m.set_solid_np(blk)
        assert not m.walls_active()
        r = Ref(m)
        a0 = float(np.abs(r.u).max())
        m.step(T)
        for _ in range(T):
            r.step()
        # f64: relative to the field's max at the end; f32: relative to the start's amplitude (by step 300 most of the
        # pulse has left, and f32 rounding is ~1e-7 of the START amplitude per step: Vulkan read 5e-4 of a 1e-4 field)
        scale = np.abs(r.u).max() if f64() else max(np.abs(r.u).max(), a0)
        rel = float(np.abs(m.field() - r.u).max() / max(scale, 1e-300))
        worst.append((name, rel))
    lim = tol(1e-11, 2e-5)
    ok('3. the kernel == the numpy reference (2D, the four walls and corners, 300 steps): CPML, Mur2, the sponge frame, '
       'the open edge', all(w[1] <= lim for w in worst), '; '.join(f'{n} {v:.1e}' for n, v in worst)
       + f' (tol {lim:.0e} of {"the field" if f64() else "the start amplitude"})')


# ---- section 4: the 1D-exact reduction --------------------------------------------------------------------
def _disp(nine, Co, kx, ky):
    X, Y = 4 * math.sin(kx / 2) ** 2, 4 * math.sin(ky / 2) ** 2
    lam = X + Y - (X * Y / 6.0 if nine else 0.0)
    s = Co * Co * lam / 4
    return 2 * math.asin(math.sqrt(s))


def _geom(nine, Co, lam, th):
    """The research's packet geometry (abc1d_MIRROR.geometry) on the stencil's own dispersion."""
    k = 2 * math.pi / lam
    kx, ky = k * math.cos(th * D2R), k * math.sin(th * D2R)
    w = _disp(nine, Co, kx, ky)
    # d omega / d kx: 2 sin(w) w' = Co^2 dLambda/dkx, dLambda/dkx = 2 sin kx (1 - Y/6 on sq9) (abc1d's formula on sq5)
    Y = 4 * math.sin(ky / 2) ** 2
    vg = Co * Co * math.sin(kx) * ((1 - Y / 6.0) if nine else 1.0) / math.sin(w)
    sig = 3.0 * lam / max(math.cos(th * D2R), 1e-9)
    LMAX = 40
    i0 = int(LMAX + 5 * sig + 10)
    Nx = int(i0 + 5 * sig + 20)
    T = int(round(2 * i0 / vg))
    E = int(5 * sig + LMAX + 50 + 2 * vg)
    return dict(kx=kx, ky=ky, w=w, vg=vg, sig=sig, i0=i0, Nx=Nx, T=T, E=E, m=2 * math.cos(ky) - 2)


class Sim1:
    """The 1D reduction of the core's update for u[i] cos(ky j) on a grid periodic in j (exact for a straight wall):
    Dyy -> m = 2 cos ky - 2. The wall is at i = 0 with the core's coefficients: 'dir' | 'cpml' (tables from
    Membrane._aux_tables, frame index s = i - 1) | 'mur2' | 'sponge' (d from the core's codes along a side)."""

    def __init__(self, Nx, g, Co, nine, kind, aux=None):
        self.Nx, self.m, self.Co, self.nine, self.kind = Nx, g['m'], Co, nine, kind
        self.u = np.zeros(Nx)
        self.up = np.zeros(Nx)
        self.first = True               # Mur2's start-up step (Mur1 once; the packet is far from the wall anyway)
        if kind == 'cpml':
            Lc, T = aux
            L2 = 2 * Lc
            idx = np.full(Nx, -1)
            idx[1:Lc + 1] = np.arange(Lc)
            fr = idx >= 0

            def arr(k, dflt):
                out = np.full(Nx, dflt)
                out[fr] = T[k][idx[fr]]
                return out
            self.fr = fr
            self.bc_, self.ac_, self.bm_, self.am_, self.bp_, self.ap_ = (arr(0, 1.0), arr(1, 0.0), arr(2, 1.0),
                                                                          arr(3, 0.0), arr(4, 1.0), arr(5, 0.0))
            self.pm, self.pp, self.z = np.zeros(Nx), np.zeros(Nx), np.zeros(Nx)
        if kind == 'sponge':
            d = np.zeros(Nx)
            d[:len(aux)] = aux
            self.inv = 1.0 / (1.0 + 0.5 * d)
            self.c2 = (1.0 - 0.5 * d) * self.inv
            self.lay = d > 0

    def step(self):
        u, up, m = self.u, self.up, self.m
        I = slice(1, -1)
        if self.nine:
            # L9 reduced = Dxx((1 + m/6) u) + m u: the x strip's exactly stretched form (and the plain L9 at sigma 0)
            vx = (1 + m / 6.0) * u
            ly = m * u
        else:
            vx = u
            ly = m * u
        gp = np.zeros_like(u)
        gm = np.zeros_like(u)
        gp[I] = vx[2:] - vx[I]
        gm[I] = vx[I] - vx[:-2]
        lx = gp - gm
        if self.kind == 'cpml':
            self.pm = np.where(self.fr, self.bm_ * self.pm + self.am_ * gm, 0.0)
            self.pp = np.where(self.fr, self.bp_ * self.pp + self.ap_ * gp, 0.0)
            dq = (gp + self.pp) - (gm + self.pm)
            self.z = np.where(self.fr, self.bc_ * self.z + self.ac_ * dq, 0.0)
            lx = np.where(self.fr, dq + self.z, lx)
        lap = lx + ly
        C2 = self.Co * self.Co
        un = 2 * u - up + C2 * lap
        if self.kind == 'sponge':
            un = np.where(self.lay, (2 * u + C2 * lap) * self.inv - self.c2 * up, un)
        un[-1] = 0.0
        if self.kind == 'mur2':
            Co = self.Co
            k, gg, e = (Co - 1) / (Co + 1), 2 / (Co + 1), Co * Co / (2 * (Co + 1))
            if self.first:
                un[0] = u[1] + k * (un[1] - u[0])
            else:
                un[0] = -up[1] + k * (un[1] + up[0]) + gg * (u[0] + u[1]) + e * m * (u[0] + u[1])
        else:
            un[0] = 0.0
        self.up, self.u = u, un
        self.first = False


def _packet(n, i0, sig, kx, w, vg, shift=0):
    i = np.arange(n, dtype=np.float64) + shift
    u0 = np.exp(-0.5 * ((i - i0) / sig) ** 2) * np.cos(kx * i)
    u1 = np.exp(-0.5 * ((i - i0 + vg) / sig) ** 2) * np.cos(kx * i + w)
    return u0, u1


def plane_R(nine, Co, lam, th, kind, aux, Lw, ref_cache):
    g = _geom(nine, Co, lam, th)
    key = (nine, Co, lam, th)
    if key not in ref_cache:
        r = Sim1(g['Nx'] + g['E'], g, Co, nine, 'dir')
        b0, b1 = _packet(g['Nx'] + g['E'], g['i0'], g['sig'], g['kx'], g['w'], g['vg'], shift=-g['E'])
        r.up[:], r.u[:] = b0, b1
        inc = float(np.linalg.norm(b1))
        for _ in range(g['T']):
            r.step()
        ref_cache[key] = (r.u[g['E']:].copy(), inc)
    ur, inc = ref_cache[key]
    t = Sim1(g['Nx'], g, Co, nine, kind, aux)
    a0, a1 = _packet(g['Nx'], g['i0'], g['sig'], g['kx'], g['w'], g['vg'])
    t.up[:], t.u[:] = a0, a1
    for _ in range(g['T']):
        t.step()
    return float(np.linalg.norm(t.u[Lw:] - ur[Lw:]) / inc)


def mur2_theory(nine, Co, lam, th):
    """|R| of the discrete Mur2 rule for the exact lattice plane wave: -P(e^{+ikx}) / P(e^{-ikx})."""
    g = _geom(nine, Co, lam, th)
    kx, w, m = g['kx'], g['w'], g['m']
    k, gg, e = (Co - 1) / (Co + 1), 2 / (Co + 1), Co * Co / (2 * (Co + 1))
    z = complex(math.cos(w), math.sin(w))
    # rule terms (a, b, coef): u0^{n+1} + u1^{n-1} - k u1^{n+1} - k u0^{n-1} - (g + e m)(u0^n + u1^n) = 0
    terms = [(0, 0, 1.0), (1, 2, 1.0), (1, 0, -k), (0, 2, -k), (0, 1, -(gg + e * m)), (1, 1, -(gg + e * m))]

    def P(sg):
        return sum(c * complex(math.cos(sg * kx * a), math.sin(sg * kx * a)) * z ** (-b) for a, b, c in terms)
    return abs(P(1) / P(-1))


def t4_plane(quick):
    print('-- 4. plane-wave |R| (the 1D-exact reduction, the core\'s coefficients)', flush=True)
    with open(os.path.join(RES, 'results_1d_MIRROR.json')) as f:
        R1 = json.load(f)
    angles = (0, 15, 30, 45, 60, 75)
    lams = (8, 16, 32)
    if quick:
        angles, lams = (0, 45, 75), (8, 32)
    cache = {}
    # the core's own tables
    mcp = {lv: mk(N=161, shape='array', bc='cpml', cpml_level=lv) for lv in ('std', 'high')}
    out = {}
    # 'high' is alpha 0 since 10/10/2026 (the research's alpha 0.05 absorbs long waves less: section 11)
    for lv, tag in (('std', 'cpml10 p3 f2.0 a0.00'), ('high', 'cpml20 p3 f2.0 a0.00')):
        m = mcp[lv]
        lay = m.abs_layer()
        Lc = lay['Lc']
        T = m._aux_tables(lay)[mc.AUX_TAB:].reshape(6, 2 * Lc)
        ref = R1['Co0.5_tuning'][tag]
        dev, worst = [], {}
        for th in angles:
            for lam in lams:
                v = plane_R(False, 0.5, lam, th, 'cpml', (Lc, T), Lc, cache)
                rv = ref[f'{lam}/{th}']
                dev.append(abs(v - rv) / max(rv, 1e-12))
                worst[th] = max(worst.get(th, 0.0), v)
        out[lv] = worst
        ok(f'4a. CPML {lv} (L {Lc}), sq5, Co 0.5: |R| per angle and wavelength == the research\'s table (within 3%)',
           max(dev) <= 0.03, 'worst over lambda: ' + ', '.join(f'{a} deg {worst[a]:.1e}' for a in angles)
           + f'; max deviation from the research {100 * max(dev):.2f}%')
    m = mk(N=161, bc='mur2')
    dev, dth, worst = [], [], {}
    ref = R1['Co0.5_fixed']['mur2']
    for th in angles:
        for lam in lams:
            v = plane_R(False, 0.5, lam, th, 'mur2', None, 1, cache)
            dev.append(abs(v - ref[f'{lam}/{th}']) / max(ref[f'{lam}/{th}'], 1e-12))
            dth.append(abs(v - mur2_theory(False, 0.5, lam, th)))
            worst[th] = max(worst.get(th, 0.0), v)
    ok('4b. Mur2, sq5, Co 0.5: |R| == the research\'s table (within 3%) and the discrete plane-wave theory (< 1e-3)',
       max(dev) <= 0.03 and max(dth) < 1e-3, 'worst over lambda: ' + ', '.join(f'{a} deg {worst[a]:.1e}' for a in angles)
       + f'; vs table {100 * max(dev):.2f}%, vs theory {max(dth):.1e}')
    for L, dm, tag in ((40, 0.2, 'sponge40 p2 d0.20 ctr'), (20, 0.35, 'sponge20 p2 d0.35 ctr')):
        m = mk(N=257, bc='sponge', sponge_L=L)
        c = m.N // 2
        lay = m.abs_layer()
        T = m._aux_tables(lay)
        codes = m.abs_np[:L + 1, c].astype(np.int64)
        inv = np.where(codes >= 2, T[mc.AUX_INV + codes], 1.0)
        d = 2.0 * (1.0 / inv - 1.0)                     # the core's table: inv = 1 / (1 + d/2)
        ref = R1['Co0.5_tuning'][tag]
        dev, worst = [], {}
        for th in angles:
            for lam in lams:
                v = plane_R(False, 0.5, lam, th, 'sponge', d, L, cache)
                rv = ref[f'{lam}/{th}']
                dev.append(abs(v - rv) / max(rv, 1e-12))
                worst[th] = max(worst.get(th, 0.0), v)
        ok(f'4c. sponge L {L} d_max {dm} (the default{"" if L == 40 else " for small N"}), the frame, sq5, Co 0.5: |R| == '
           'the research\'s table (within 1%: the frame\'s codes hold each row\'s exact damping)', max(dev) <= 0.01,
           'worst over lambda: ' + ', '.join(f'{a} deg {worst[a]:.1e}' for a in angles)
           + f'; max deviation {100 * max(dev):.1f}%')
    # sq9: no research table; bounds from the sq5 values
    m = mk(N=161, shape='array', bc='cpml', stencil=9)
    lay = m.abs_layer()
    T = m._aux_tables(lay)[mc.AUX_TAB:].reshape(6, 20)
    w9 = {}
    for th in angles:
        for lam in lams:
            w9[th] = max(w9.get(th, 0.0), plane_R(True, 0.5, lam, th, 'cpml', (10, T), 10, cache))
    mur9 = max(abs(plane_R(True, 0.5, lam, th, 'mur2', None, 1, cache) - mur2_theory(True, 0.5, lam, th))
               for th in angles for lam in lams)
    ok('4d. sq9: CPML std <= 2e-4 over 0-60 deg (the sq5 bound 1.2e-4 x 1.7) and <= 5e-5 at 75 deg; Mur2 == its '
       'discrete theory (< 2e-3: the packet averages R over its +-5% spread in kx)',
       max(w9[a] for a in angles if a <= 60) <= 2e-4 and w9.get(75, 0.0) <= 5e-5 and mur9 < 2e-3, 'CPML sq9 worst: ' + ', '.join(f'{a} deg {w9[a]:.1e}' for a in angles)
       + f'; Mur2 sq9 vs theory {mur9:.1e}')


# ---- section 5: the 2D box ---------------------------------------------------------------------------------
def energy_np(m, d1, d0, reg):
    """The research's leapfrog energy of a field pair over the region reg (sq5 / sq9 / tri link forms)."""
    Co = m.p.Co
    kin = np.sum(((d1 - d0) ** 2)[reg]) / (Co * Co)
    pot = 0.0
    if m.tri():
        pairs = (((slice(1, None), slice(None)), (slice(None, -1), slice(None)), 2 / 3),
                 ((slice(None), slice(1, None)), (slice(None), slice(None, -1)), 2 / 3),
                 ((slice(1, None), slice(None, -1)), (slice(None, -1), slice(1, None)), 2 / 3))
    elif m.p.stencil == 9:
        pairs = (((slice(1, None), slice(None)), (slice(None, -1), slice(None)), 2 / 3),
                 ((slice(None), slice(1, None)), (slice(None), slice(None, -1)), 2 / 3),
                 ((slice(1, None), slice(1, None)), (slice(None, -1), slice(None, -1)), 1 / 6),
                 ((slice(1, None), slice(None, -1)), (slice(None, -1), slice(1, None)), 1 / 6))
    else:
        pairs = (((slice(1, None), slice(None)), (slice(None, -1), slice(None)), 1.0),
                 ((slice(None), slice(1, None)), (slice(None), slice(None, -1)), 1.0))
    for A, B, wt in pairs:
        e = reg[A] & reg[B]
        pot += wt * np.sum(((d1[A] - d1[B]) * (d0[A] - d0[B]))[e])
    return kin + pot


def box_reflection(bc, Co, nine, src_off, r_end, s=3.0, extra=None, cache=None, Nphys=201):
    """The research's 2D box measurement in the core: a test full square whose OPEN domain is Nphys x Nphys cells
    (201: the research's; odd) (+ the layer), the reference a Dirichlet full square big enough that nothing returns; a
    gaussian pulse at rest at src_off (cells from the open domain's corner); sqrt(E_refl / E0) over the open domain
    minus its outer row."""
    extra = extra or {}
    # the layer's width (the frame's first open depth; 1 for a ring rule, whose ring is not physical, as the research)
    if bc == 'cpml':
        Ld = mc.CPML_LEVELS[extra.get('cpml_level', 'std')]['L']
    elif bc == 'sponge':
        Ld = int(math.ceil(extra['sponge_L'] - 1e-9))
    else:
        Ld = 1
    N = Nphys + 2 * Ld
    m = mk(N=N, bc=bc, shape='array', stencil=9 if nine else 5, Co=Co, src='impulse', **extra)
    assert m.N == N and (bc not in ('cpml', 'sponge') or m._frame_Ld() == Ld), (m.N, Ld, m._frame_Ld())
    T = int(round(r_end / Co))
    E = int(0.5 * r_end) + 30
    Nr = Nphys + 2 * E
    key = (Co, nine, src_off, r_end, s, Nphys)
    if cache is not None and key in cache:
        ru1, ru0, E0 = cache[key]
    else:
        r = mk(N=Nr, bc='dirichlet', shape='array', stencil=9 if nine else 5, Co=Co, src='impulse')
        seed_cell(r, E + src_off[0], E + src_off[1], s)
        r.record = False
        r.step(1)
        allr = np.ones((Nr, Nr), bool)
        E0 = energy_np(r, r.level(1).astype(np.float64), r.level(0).astype(np.float64), allr)
        r.step(T - 1)
        ru1 = r.level(1)[E:E + Nphys, E:E + Nphys].astype(np.float64)
        ru0 = r.level(0)[E:E + Nphys, E:E + Nphys].astype(np.float64)
        if cache is not None:
            cache[key] = (ru1, ru0, E0)
    seed_cell(m, Ld + src_off[0], Ld + src_off[1], s)
    m.record = False
    m.step(T)
    d1 = m.level(1)[Ld:Ld + Nphys, Ld:Ld + Nphys].astype(np.float64) - ru1
    d0 = m.level(0)[Ld:Ld + Nphys, Ld:Ld + Nphys].astype(np.float64) - ru0
    reg = np.ones((Nphys, Nphys), bool)
    reg[0, :] = reg[-1, :] = reg[:, 0] = reg[:, -1] = False
    return math.sqrt(max(energy_np(m, d1, d0, reg), 0.0) / E0)


def t5_box(quick):
    print('-- 5. the 2D box: a broadband pulse, the four walls and the corners', flush=True)
    # (bc, extra, centre bound, corner bound, research centre / corner at Co 0.5)
    cases = [('cpml', {}, 1.0e-4, 3.0e-4, '5.3e-5 / 1.8e-4'),
             ('cpml', dict(cpml_level='high'), 5e-6, 5e-6, '2.5e-6 / 1.8e-6'),
             ('mur2', {}, 1.5e-2, 0.15, '8.8e-3 / 0.10'),
             ('sponge', dict(sponge_L=40), 2.5e-2, 6e-2, '1.4e-2 / 3.8e-2 (proto_frame_MIRROR, L40 d.2 p2)'),
             ('mur', {}, 4e-2, 0.25, '2.2e-2 / 0.19 (Mur1, today)')]
    for nine in ((False,) if quick else (False, True)):
        for Co in ((0.5,) if quick else (0.5, 0.7)):
            cache = {}
            res = []
            for bc, extra, bc_c, bc_k, rs in cases:
                rc = box_reflection(bc, Co, nine, (100, 100), 230, extra=extra, cache=cache)
                rk = box_reflection(bc, Co, nine, (25, 25), 165, extra=extra, cache=cache)
                lim_c = bc_c * (1.5 if Co > 0.6 else 1.0) + (0 if f64() else 2e-5)
                lim_k = bc_k * (1.5 if Co > 0.6 else 1.0) + (0 if f64() else 2e-5)
                res.append((bc + ('-high' if extra.get('cpml_level') == 'high' else ''), rc, rk, rc <= lim_c and rk <= lim_k, rs))
            ok(f'5. the 2D box, {"sq9" if nine else "sq5"}, Co {Co}: the research\'s levels (centre / corner pulse)',
               all(r[3] for r in res), '; '.join(f'{r[0]} {r[1]:.1e} / {r[2]:.1e} (research {r[4]})' for r in res))


# ---- section 6: the open edge on masked shapes ------------------------------------------------------------
def masked_reflection(lat, shape, R=100.0, L=40, s=3.0, Co=0.5):
    """The research's masked measurement (masked_MIRROR.py) in the core: the shape at radius R (circumradius for
    the circle, apothem 100 for the hexagon) with an open edge (the sponge OUTSIDE it), the reference the same
    shape far bigger with a pinned edge; a pulse at the centre; the window until the front is at R + 2 L + 60;
    sqrt(E_refl / E0) over the open domain."""
    tri = lat == 'tri'
    Rc = R / math.cos(math.pi / 6) if shape == 'n6' else R
    r_end = R + 2 * 40 + 60
    T = int(round(r_end / Co))
    Eext = int(0.5 * r_end) + 20

    def build(Rshape, edge, extraL):
        bh_need = Rshape + 1.5 + extraL + 4
        c = math.ceil(bh_need / (S3H if tri else 1.0))
        N = 2 * c + 1
        bh = c * (S3H if tri else 1.0)
        rad = Rshape / (bh - 1.5)
        m = mk(N=N, lattice=lat, shape=shape, rad=rad, edge=edge, src='impulse', Co=Co, sponge_L=float(L), rot=0.0)
        return m
    m = build(Rc, 'open', L)
    assert abs(m.dom_R() - Rc) < 1e-9, (m.dom_R(), Rc)
    r = build(Rc + L + Eext, 'pinned', 0)
    # the reference's grid is bigger: map by the centre cell (same lattice, same centre => an integer offset)
    off = r.N // 2 - m.N // 2
    for mm_ in (m, r):
        c = mm_.N // 2
        seed_cell(mm_, c, c, s)
        mm_.record = False
    r.step(1)
    E0 = energy_np(r, r.level(1).astype(np.float64), r.level(0).astype(np.float64), r.mask_np != 0)
    r.step(T - 1)
    m.step(T)
    sl = (slice(off, off + m.N), slice(off, off + m.N))
    d1 = m.level(1).astype(np.float64) - r.level(1)[sl]
    d0 = m.level(0).astype(np.float64) - r.level(0)[sl]
    reg = m.open_mask() != 0
    return math.sqrt(max(energy_np(m, d1, d0, reg), 0.0) / E0), m


def t6_masked(quick):
    print('-- 6. the open edge on masked shapes (sponge L 40, d_max 0.2)', flush=True)
    res = []
    for lat, shape, rs in (('sq', 'circle', 1.65e-2), ('tri', 'n6', 1.60e-2), ('tri', 'circle', 1.7e-2)):
        if quick and lat == 'tri' and shape == 'circle':
            continue
        v, m = masked_reflection(lat, shape)
        res.append((f'{lat} {shape}', v, rs, m.N))
    lim = 2.5e-2 if f64() else 2.6e-2
    ok('6. open edge, R 100, a sigma-3 pulse from the centre: the research\'s ~1.6-1.7e-2 (bound 2.5e-2)',
       all(r[1] <= lim for r in res), '; '.join(f'{r[0]} {r[1]:.2e} (research {r[2]:.2e}, N {r[3]})' for r in res))


# ---- section 7: stability ---------------------------------------------------------------------------------
def t7_noise(quick):
    print('-- 7. stability: 20,000 steps of white noise', flush=True)
    steps = 4000 if quick else 20000
    rng = np.random.default_rng(7)
    cases = [('CPML sq5 std', dict(N=121, shape='array', bc='cpml'), (0.5, 0.7, 0.7071)),
             ('CPML sq5 high', dict(N=141, shape='array', bc='cpml', cpml_level='high'), (0.5, 0.7, 0.7071)),
             ('CPML sq9 std', dict(N=121, shape='array', bc='cpml', stencil=9), (0.5, 0.7, 0.85, 0.866)),
             ('CPML sq9 high', dict(N=141, shape='array', bc='cpml', stencil=9, cpml_level='high'), (0.5, 0.85)),
             ('Mur2 sq5', dict(N=101, bc='mur2'), (0.5, 0.7, 0.7071)),
             ('Mur2 sq9', dict(N=101, bc='mur2', stencil=9), (0.5, 0.7, 0.85, 0.866)),
             ('sponge frame sq5 L40', dict(N=181, bc='sponge', sponge_L=40), (0.5, 0.7, 0.7071)),
             ('sponge frame sq9 L40', dict(N=181, bc='sponge', stencil=9, sponge_L=40), (0.5, 0.85)),
             ('sponge frame tri L40', dict(N=201, bc='sponge', shape='array', lattice='tri', sponge_L=40), (0.5, 0.8)),
             ('open sq circle L40', dict(N=181, shape='circle', edge='open', sponge_L=40), (0.5, 0.7)),
             ('open tri hexagon L40', dict(N=221, shape='n6', edge='open', lattice='tri', sponge_L=40), (0.5, 0.8)),
             ('open tri circle L40', dict(N=221, shape='circle', edge='open', lattice='tri', sponge_L=40), (0.8,))]
    for name, kw, cos in cases:
        rows = []
        good = True
        for Co in cos:
            m = mk(src='impulse', Co=Co, **kw)
            m.record = False
            noise_seed(m, rng)
            a0 = float(np.abs(m.field()).max())
            marks = []
            for n in range(4):
                m.step(steps // 4)
                marks.append(float(np.abs(m.field()).max()))
            fin = all(math.isfinite(v) for v in marks)
            # no growth over the run (the last mark within 10% of the first), and below the start -- except at Co
            # within 0.1% of the lattice's limit, where the near-Nyquist modes have ~zero group velocity: they never
            # reach the boundary and stay, as in a closed box (bounded, not falling)
            at_lim = Co >= 0.999 * m.co_limit()
            g = fin and marks[-1] <= 1.1 * marks[0] and (at_lim or marks[-1] <= a0)
            good = good and g
            rows.append(f'Co {Co}{" (at the limit)" if at_lim else ""}: {a0:.2f} -> '
                        + ' / '.join(f'{v:.2e}' for v in marks))
        ok(f'7. {name}: {steps} steps of noise, bounded (no growth) and falling (below the limit)', good,
           '; '.join(rows))


# ---- section 8: energy ------------------------------------------------------------------------------------
def t8_energy(quick):
    print('-- 8. energy with the sponge (no drive)', flush=True)
    cases = [('sponge frame sq5', dict(N=121, bc='sponge', sponge_L=20)),
             ('sponge frame sq9', dict(N=121, bc='sponge', stencil=9, sponge_L=20, Co=0.8)),
             ('sponge frame tri', dict(N=121, bc='sponge', shape='array', lattice='tri', sponge_L=20, Co=0.75)),
             ('open sq circle', dict(N=121, shape='circle', edge='open', sponge_L=20)),
             ('open tri hexagon + lens', dict(N=141, shape='n6', edge='open', lattice='tri', sponge_L=20, medium='lens')),
             ('open sq circle + solids', dict(N=121, shape='circle', edge='open', sponge_L=20, stencil=9))]
    steps = 600 if quick else 1500
    for name, kw in cases:
        m = mk(src='pulse', sigma=3, sx=0.3, sy=0.1, **kw)
        if 'solids' in name:
            t = np.zeros((m.N, m.N), np.uint8)
            c = m.N // 2
            t[c - 20:c - 16, c - 10:c + 10] = 1               # pinned, in the domain
            t[c + 12:c + 16, c - 10:c + 10] = 2               # rigid, in the domain
            t[c - 3:c + 3, 2:12] = 2                          # rigid, in the layer
            t[c - 3:c + 3, m.N - 12:m.N - 2] = 1              # pinned, in the layer
            m.set_solid_np(t)
            m.reseed()
            m.set_solid_np(t)
        L = m.operator()
        E = [m.leapfrog_energy(L)]
        for _ in range(steps // 10):
            m.step(10)
            E.append(m.leapfrog_energy(L))
        E = np.array(E)
        rise = float(np.max(np.diff(E)) / E[0])
        Eh = np.array(m.hist['E'])
        h = len(Eh) // 2
        late = float(Eh[h:].max() / Eh[:h].max())
        ok(f'8. {name}: the exact leapfrog energy never rises (every 10 steps); the open domain\'s recorded energy does '
           'not grow (later-half max <= 1.05 x earlier-half max, the ledger row "eopen")',
           rise <= tol(1e-13, 2e-6) and late <= 1.05 and E[-1] < 0.5 * E[0],
           f'max rise {rise:.1e} of E0; E {E[0]:.3e} -> {E[-1]:.3e}; recorded open-domain E: later / earlier half max '
           f'{late:.2e}, max / step 1 {Eh.max() / Eh[0]:.3f} (the functional\'s start-up rise, not growth), end / max '
           f'{Eh[-1] / Eh.max():.2e}' + (f'; walls kernel {m.walls_active()}' if 'solids' in name else ''))


# ---- section 9: composition -------------------------------------------------------------------------------
def t9_compose(quick):
    print('-- 9. composition', flush=True)
    # 9a. global damping with each layer: decays, finite
    rows, good = [], True
    for name, kw in (('cpml', dict(shape='array', bc='cpml')), ('mur2', dict(bc='mur2')),
                     ('sponge', dict(bc='sponge')), ('open', dict(shape='circle', edge='open')),
                     ('tri sponge', dict(shape='array', lattice='tri', bc='sponge'))):
        m = mk(N=161, src='pulse', damp=0.02, Co=0.7, **kw)
        m.step(2000)
        u = m.field()
        g = np.isfinite(u).all() and np.abs(u).max() < 1e-3
        good = good and g
        rows.append(f'{name} {np.abs(u).max():.1e}')
    ok('9a. global damping 0.02 at Co 0.7 with every boundary: finite, decayed after 2000 steps', good, ', '.join(rows))
    # 9b. solids in the domain and in the layer, both types: CPML (walls path) bounded; the source snaps out of a solid
    m = mk(N=161, shape='array', bc='cpml', stencil=9, src='pulse', sx=0.2)
    t = np.zeros((161, 161), np.uint8)
    t[3:8, 60:100] = 2                    # rigid, in the frame
    t[60:100, 152:158] = 1                # pinned, in the frame
    t[70:90, 70:74] = 2                   # rigid, in the domain
    m.set_solid_np(t)
    m.reseed()
    m.set_solid_np(t)
    w = m.walls_active()
    m.step(3000)
    a = float(np.abs(m.field()).max())
    ok('9b. CPML sq9 with rigid and pinned solids in the frame and the domain (the walls kernel): bounded and absorbing',
       w and math.isfinite(a) and a < 1e-3, f'walls kernel {w}; max|u| after 3000 steps {a:.1e}')
    # 9c. the recorder: the open domain only (numpy over the open cells), the peak too; probes inside
    m = mk(N=161, bc='sponge', src='pulse', sx=0.3, sy=-0.2)
    m.step(150)
    u1, u0 = m.level(1).astype(np.float64), m.level(0).astype(np.float64)
    op = (m.open_mask() != 0)
    inner = op.copy()
    inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
    kin = float((((u1 - u0) ** 2)[inner]).sum())
    gx = np.zeros_like(u1)
    gy = np.zeros_like(u1)
    gx[:-1, :] = u1[1:, :] - u1[:-1, :]
    gy[:, :-1] = u1[:, 1:] - u1[:, :-1]
    pot = float(((gx ** 2 + gy ** 2)[inner]).sum())
    Enp = kin / (2 * 0.25) + 0.5 * pot
    amax = float(np.abs(u1[inner]).max())
    Eh = m.hist['E'][-1]
    (ai, aj), (bi, bj) = m.probe_cells()
    ok('9c. the recorder sums the open domain only (energy and peak == numpy over the open cells); the probes lie in '
       'the open domain; stats() agrees', abs(Eh - Enp) <= tol(1e-12, 1e-5) * Enp and abs(m.amax - amax) <= tol(0, 1e-6)
       and op[ai, aj] and op[bi, bj] and abs(m.stats()[0] - Enp) <= tol(1e-12, 1e-5) * Enp,
       f'E {Eh:.6e} vs numpy {Enp:.6e}; peak {m.amax:.4e} vs {amax:.4e}; probes ({ai},{aj}) ({bi},{bj})')
    # 9d. sources: a source requested in the layer snaps out; vertices inside the open domain; the beam aperture
    m = mk(N=161, bc='cpml', src='pulse', sx=1.1, sy=0.0)
    sn = m.src_snap
    i, j = m.src[:2]
    mv = mk(N=161, bc='sponge', src='vtx', shape='square')
    vin = all(mv.open_mask()[v[0], v[1]] for v in mv.vtx)
    mb = mk(N=161, bc='sponge', src='beam', beam_angle=0.0, sx=0.85, beam_width=12)
    cells = mb.beam['cells']
    bin_ = all(mb.open_mask()[c[0], c[1]] for c in cells)
    mo = mk(N=161, shape='circle', edge='open', src='pulse', sx=1.15)
    ok('9d. a source requested in the layer snaps into the open domain (src_snap "layer"); vertex sources and the beam '
       'aperture use open cells only', sn == 'layer' and m.open_mask()[i, j] and vin and bin_ and len(cells) > 0
       and mo.open_mask()[mo.src[0], mo.src[1]],
       f'cpml sx 1.1 -> cell ({i},{j}) [{sn}]; open circle sx 1.15 -> {mo.src[:2]} [{mo.src_snap}]; '
       f'{len(mv.vtx)} vertices in; beam {len(cells)} cells in')
    # 9e. painting with a layer: incremental == full rebuild (device mask with the layer codes, words)
    sys.path.insert(0, HERE)
    import solids_MIRROR as so
    bad = []
    for kw in (dict(shape='circle', edge='open', N=161), dict(shape='array', bc='sponge', N=161, lattice='tri')):
        m = mk(src='pulse', **kw)
        sc = so.SolidScene()
        m.attach_scene(sc)
        r1 = so.cells_to_grid(m, 1.5)
        for wall, pts in (('rigid', [(-0.95, -0.9), (0.2, 0.1), (0.98, 0.95)]), ('pinned', [(-0.98, 0.3), (0.7, -0.5)])):
            m.update_solids(sc.begin_stroke(*pts[0], r1, wall=wall))
            for q in pts[1:]:
                m.update_solids(sc.extend_stroke(*q))
            sc.end_stroke()
        dev = m.M.to_numpy().astype(np.int64)
        want = m._dev_mask().astype(np.int64)
        W1 = m.W.to_numpy().copy() if m.W is not None else None
        m.update_solids(None)
        dev2 = m.M.to_numpy().astype(np.int64)
        W2 = m.W.to_numpy() if m.W is not None else None
        if not (np.array_equal(dev, want) and np.array_equal(dev, dev2)
                and (W1 is None) == (W2 is None) and (W1 is None or np.array_equal(W1, W2))):
            bad.append(str(kw))
        m.step(200)
        if not np.isfinite(m.field()).all():
            bad.append(str(kw) + ' not finite')
    ok('9e. painting across a layer: the device mask keeps the layer codes; incremental == the full rebuild', not bad,
       'open circle, triangular sponge frame; rigid + pinned strokes through the layer' + (f'; BAD {bad}' if bad else ''))
    # 9f. set_co with and without rescale on a CPML / sponge run (the tables follow Co; the memory is kept)
    good, rows = True, []
    for kw in (dict(shape='array', bc='cpml'), dict(shape='circle', edge='open')):
        for rs in (False, True):
            m = mk(N=161, src='pulse', **kw)
            m.step(100)
            m.set_co(0.65, rescale=rs)
            k1 = m._aux_tkey
            m.step(1500)
            a = float(np.abs(m.field()).max())
            g = math.isfinite(a) and a < 2e-2 and (k1 is not None and k1[1] == 0.65)
            good = good and g
            rows.append(f'{list(kw.values())[-1]} rescale {rs}: {a:.1e}')
    ok('9f. set_co (with and without rescale) on CPML and the open edge: the tables follow Co, the run stays absorbing',
       good, ', '.join(rows))
    # 9g. scale mode: the sponge's width follows the scale (a fraction of the half-size, min 20 cells); CPML stays
    m1 = mk(N=513, shape='circle', edge='open', scale='domain', src='pulse')
    Lf = m1.p.sponge_Lf
    m2 = mk(N=1025, shape='circle', edge='open', scale='domain', src='pulse', sponge_Lf=Lf)
    m3 = mk(N=1025, shape='array', bc='cpml', scale='domain', src='pulse')
    ok('9g. scale mode domain: the sponge\'s width follows the scale (L 40 at N 513 -> 80 at N 1025, the same fraction '
       'of the half-size) with d_max 0.2 -> 0.1 (8 / L: the same damping per wave); the CPML frame stays 10 cells',
       abs(m1.sponge_width() - 40) < 1e-9 and abs(m1.sponge_dmax() - 0.2) < 1e-12 and abs(m2.sponge_dmax() - 0.1) < 1e-12
       and abs(m2.sponge_width() - 80) < 1e-9 and m3.abs_layer()['Lc'] == 10
       and abs(m2.inner_R() / m2.base_half() - m1.inner_R() / m1.base_half()) < 0.01,
       f'Lf {Lf:.4f}; L {m1.sponge_width():.1f} -> {m2.sponge_width():.1f}; r_in / half-size '
       f'{m1.inner_R() / m1.base_half():.4f} / {m2.inner_R() / m2.base_half():.4f}')
    # 9h. switching bc / edge without a reseed: step() and set_edge rebuild the domain, the field is kept and finite
    m = mk(N=161, src='pulse', shape='circle', rad=0.7)
    m.step(50)
    m.set_edge('open')
    l1 = m.abs_layer() is not None and m.abs_np is not None
    m.step(50)
    m.set_edge('pinned')
    l2 = m.abs_np is None
    m.p.bc = 'sponge'
    m.p.shape = 'array'
    m.reseed()
    m.step(20)
    m.p.bc = 'cpml'
    m.step(20)
    l3 = m.abs_layer()['kind'] == 'cpml' and m.abs_np is not None and np.isfinite(m.field()).all()
    m.p.bc = 'dirichlet'
    m.step(20)
    l4 = m.abs_np is None and m.aux is None
    ok('9h. switching: set_edge to / from open and p.bc changed between steps rebuild the layer (the field is kept)',
       l1 and l2 and l3 and l4, f'{l1} {l2} {l3} {l4}')
    # 9i. the beam, the packet and driven vertices with a layer run and stay finite; the energy falls
    rows, good = [], True
    for kw in (dict(src='beam', bc='cpml', shape='array'), dict(src='packet', bc='sponge'),
               dict(src='vtx', vtx_drive=True, shape='n6', edge='open'),
               dict(src='cont', bc='mur2', stencil=9), dict(src='slit', shape='circle', edge='open', lattice='tri')):
        m = mk(N=161, **kw)
        m.step(1500)
        u = m.field()
        g = np.isfinite(u).all() and np.abs(u).max() < 10
        good = good and g
        rows.append(f'{kw["src"]} {np.abs(u).max():.2e}')
    ok('9i. beam, packet, driven vertices, cont and slit with each kind of boundary: finite and bounded', good,
       ', '.join(rows))
    # 9j. Mur 2nd order keeps no static offset (10/10/2026): data on the ring rows at a seed, a drop_pulse, a set_co or a
    # bc switch would otherwise stay as a constant (Mur2 keeps its start residual: probe_mur2_dc_MIRROR.log, -0.099 for
    # a pulse 7 cells from the ring, -0.35 for a spike next to it); the start-up Mur1 step (BC_MUR2S) removes it
    rows, good = [], True
    for name in ('pulse sx 0.9', 'spike next to the ring', 'drop_pulse near the ring', 'set_co mid-run',
                 'bc switch mur -> mur2', 'sq9 spike next to the ring'):
        m = mk(N=161, bc='mur2', src='pulse', sx=0.9, stencil=9 if 'sq9' in name else 5)
        m.record = False
        if 'spike' in name:
            u = np.zeros((161, 161))
            u[1, 80] = 1.0
            m.set_levels(u, u)
        if name.startswith('drop'):
            m.step(300)
            m.p.sx, m.p.sy = 0.95, 0.4
            m.recache_source()
            m.drop_pulse()
        if name.startswith('set_co'):
            m.step(37)
            m.set_co(0.65)
        if name.startswith('bc switch'):
            m.p.bc = 'mur'
            m.step(45)
            m.p.bc = 'mur2'
        m.step(6000)
        I = m.field().astype(np.float64)[1:-1, 1:-1]
        g = abs(I.mean()) < 2e-3 and np.abs(I).max() < 2e-2
        good = good and g
        rows.append(f'{name}: mean {I.mean():+.1e}, max {np.abs(I).max():.1e}')
    ok('9j. Mur2 leaves no static offset after 6000 steps (|mean| < 2e-3; Mur1 leaves ~4e-4): data on the ring rows at '
       'the seed, a dropped pulse, set_co, a bc switch', good, '; '.join(rows))


# ---- section 11: long waves (scale mode) ---------------------------------------------------------------------
def t11_long(quick):
    """In scale mode 'domain' the pulse is a fraction of the domain, so in cells it widens with N. The box with the
    open domain 200 f + 1 and the pulse sigma 3 f at its centre (window 230 f): the CPML levels keep their reflection
    (std ~3-5e-5, high ~2e-6: alpha 0); the sponge with its width following the scale (40 f cells) and the auto
    d_max (8 / L above 40 cells: L d_max fixed) keeps the research's 1.4e-2 (probe_scale_refl_MIRROR.log: a fixed 40
    cells gives 5.1e-2 / 0.15 / 0.32 at f 2 / 4 / 8)."""
    print('-- 11. long waves: the box with the pulse widening as the domain grows', flush=True)
    fs = (1, 2) if quick else (1, 2, 4)
    rows, good = [], True
    for f in fs:
        Nphys, s, r_end = 200 * f + 1, 3.0 * f, 230 * f
        c = Nphys // 2
        cache = {}
        r1 = box_reflection('cpml', 0.5, False, (c, c), r_end, s=s, cache=cache, Nphys=Nphys)
        r2 = box_reflection('cpml', 0.5, False, (c, c), r_end, s=s, extra=dict(cpml_level='high'), cache=cache,
                            Nphys=Nphys)
        m = mk(N=Nphys + 80 * f, shape='array', bc='sponge', sponge_L=40.0 * f)
        dm = m.sponge_dmax()
        r3 = box_reflection('sponge', 0.5, False, (c, c), r_end, s=s, extra=dict(sponge_L=40.0 * f), cache=cache,
                            Nphys=Nphys)
        g = (r1 <= 1e-4 and r2 <= (1e-5 if f64() else 4e-5) and r3 <= 2e-2
             and abs(dm - 8.0 / (40.0 * f)) < 1e-12)
        good = good and g
        rows.append(f'sigma {3 * f:.0f}: cpml std {r1:.1e}, high {r2:.1e}, sponge L {40 * f} d_max {dm:.3f} {r3:.2e}')
    ok('11. long waves: CPML std <= 1e-4, high <= 1e-5, the sponge following the scale (auto d_max 8 / L) <= 2e-2 '
       'for sigma 3 / 6 / 12 cells', good, '; '.join(rows))


# ---- section 10: cost -------------------------------------------------------------------------------------
def t10_bench(quick):
    print('-- 10. cost per step', flush=True)
    import taichi as ti
    Ns = (1025, 4097) if quick else (1025, 4097, 8193)
    for N in Ns:
        cfg = [('plain (dirichlet)', dict(shape='array', bc='dirichlet')), ('mur', dict(shape='array', bc='mur')),
               ('mur2', dict(shape='array', bc='mur2')), ('cpml std', dict(shape='array', bc='cpml')),
               ('cpml high', dict(shape='array', bc='cpml', cpml_level='high')),
               ('sponge frame', dict(shape='array', bc='sponge')),
               ('circle pinned', dict(shape='circle')), ('circle open', dict(shape='circle', edge='open'))]
        steps = 400 if N <= 1025 else (96 if N <= 4097 else 32)
        ms = {}
        for name, kw in cfg:
            m = mk(N=N, src='pulse', **kw)
            m.record = False
            m.step(16)
            ti.sync()
            ms[name] = m
        best = {k: 1e9 for k in ms}
        for _ in range(5):
            for name, m in ms.items():
                ti.sync()
                t0 = time.perf_counter()
                m.step(steps)
                ti.sync()
                best[name] = min(best[name], (time.perf_counter() - t0) / steps)
        base = best['plain (dirichlet)']
        cb = best['circle pinned']
        rows = []
        for name in ms:
            ref = cb if name.startswith('circle') else base
            rows.append(f'{name} {1e3 * best[name]:.3f} ms (x{best[name] / ref:.2f})')
        lim = 1.6 if N <= 1025 else 1.35
        good = all(best[n] / (cb if n.startswith('circle') else base) <= lim for n in ms)
        ok(f'10. cost per step at N {N} ({"f64" if f64() else "f32"}): every boundary within x{lim} of plain', good,
           '; '.join(rows))
        del ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='cuda')
    ap.add_argument('--f32', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--only', default='')
    ap.add_argument('--bench', action='store_true')
    a = ap.parse_args()
    fp64 = not a.f32 and a.arch != 'vulkan'
    mc.init_taichi(a.arch, fp64=fp64, quiet=True)
    only = {int(x) for x in a.only.split(',') if x}
    sections = [(1, t1_info), (2, t2_parity), (3, t3_reference), (4, t4_plane), (5, t5_box), (6, t6_masked),
                (7, t7_noise), (8, t8_energy), (9, t9_compose), (11, t11_long)]
    if a.bench:
        sections.append((10, t10_bench))
    t0 = time.time()
    for k, f in sections:
        if only and k not in only:
            continue
        if k == 1:
            f()
        else:
            f(a.quick)
    prec = 'f64' if fp64 else 'f32'
    if FAILS:
        print(f'{len(FAILS)} FAILED ({a.arch} {prec}, {time.time() - t0:.0f} s): ' + '; '.join(FAILS))
        sys.exit(1)
    print(f'ALL PASS ({a.arch} {prec}, {time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
