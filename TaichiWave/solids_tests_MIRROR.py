"""Tests for the solid objects and painted walls (solids_MIRROR.py) and the core's composition
M = domain AND NOT solid. Every solid is PINNED (u = 0), so most checks here are EXACT: zeros
that must be 0.0, symmetries that must hold bit for bit, masks that must be identical.

    python solids_tests_MIRROR.py [--arch cuda|vulkan|cpu] [--f32] [--quick]

(--quick skips the N = 8193 timings.) The no-solids case is also covered by
crosscheck_js_MIRROR.py (the web app's own solver, 27 configurations).
"""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import membrane_core_MIRROR as mc
import solids_MIRROR as so

D = math.pi / 180
FAILS = []


def ok(name, cond, det=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('  | ' + det if det else ''), flush=True)
    if not cond:
        FAILS.append(name)


def shifted(a, di, dj):
    """b[i, j] = a[i - di, j - dj], zero-filled."""
    b = np.zeros_like(a)
    N0, N1 = a.shape
    si = slice(max(0, di), N0 + min(0, di))
    sj = slice(max(0, dj), N1 + min(0, dj))
    ti_ = slice(max(0, -di), N0 + min(0, -di))
    tj = slice(max(0, -dj), N1 + min(0, -dj))
    b[si, sj] = a[ti_, tj]
    return b


def flood(m, free, seed):
    """Cells reachable from seed over free cells through the stencil's links."""
    reach = np.zeros_like(free, dtype=bool)
    reach[seed] = free[seed]
    offs = m.neighbour_offsets()
    while True:
        grow = reach.copy()
        for di, dj in offs:
            grow |= shifted(reach, di, dj)
        grow &= free
        if np.array_equal(grow, reach):
            return reach
        reach = grow


def all_levels(m):
    """The three time-level slots of U as numpy arrays (current, previous, and the spare)."""
    U = m.U.to_numpy()
    return [U[(m.slot + k) % 3] for k in (1, 0, 2)]


def grid_cell(m, gx, gy):
    return m.round_cell(*so.grid_to_cell(m, gx, gy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='cuda')
    ap.add_argument('--f32', action='store_true')
    ap.add_argument('--quick', action='store_true', help='skip the N = 8193 timings')
    a = ap.parse_args()
    mc.init_taichi(a.arch, fp64=not a.f32, quiet=True)
    import taichi as ti
    T0 = time.perf_counter()
    lattices = (('sq', 5), ('sq', 9), ('tri', 5))
    lname = lambda lat, st: f'{lat}{st}' if lat == 'sq' else 'tri'

    # 1. NO SOLIDS = THE WEB APP'S PATH. An empty scene attached leaves has_solids False; a scene
    #    whose only object lies outside the domain takes the solids path (masked pass, slit rule)
    #    with no solid cell inside -- the field must still be bit-identical.
    base = dict(N=257, shape='circle', rad=0.8, src='pulse', sigma=3, sx=0.3, bc='mur')
    m0 = mc.Membrane(mc.Params(**base))
    m1 = mc.Membrane(mc.Params(**base))
    m1.attach_scene(so.SolidScene())
    m2 = mc.Membrane(mc.Params(**base))
    sc = so.SolidScene()
    sc.add_circle(0.95, 0.95, 0.03)
    m2.attach_scene(sc)
    for m in (m0, m1, m2):
        m.step(400)
    u0 = m0.field()
    same1, same2 = np.array_equal(u0, m1.field()), np.array_equal(u0, m2.field())
    ok('no solid cell in the domain: field bit-identical to no scene', same1 and same2 and not m1.has_solids
       and m2.has_solids and m2.solid_cells() == 0,
       f'empty scene: {"=" if same1 else "DIFF"} (has_solids {m1.has_solids}); object outside the domain: '
       f'{"=" if same2 else "DIFF"} (has_solids {m2.has_solids}, {m2.solid_cells()} solid cells inside)')

    # 2. A CLOSED PAINTED LOOP WITH THE MINIMUM BRUSH SEALS ITS INTERIOR EXACTLY, on every lattice,
    #    with segments at many angles (none along a lattice row).
    #    The source is a DRIVEN cell (amplitude 1, 0.05 cycles/step), not a seeded pulse: a
    #    gaussian's tail is nonzero everywhere at t = 0 (~1e-160 inside the loop, measured
    #    30/09/2026), so only a source with compact support can test for an exact 0.
    loop = [(0.05, -0.45), (0.5, -0.2), (0.45, 0.35), (-0.05, 0.5), (-0.3, 0.05), (0.05, -0.45)]
    for lat, st in lattices:
        prm = dict(N=257, lattice=lat, stencil=st, src='cont', freq=0.05, sx=-0.6, sy=-0.6, Co=0.5)
        if lat == 'tri':
            prm.update(shape='n6', rot=30 * D)
        m = mc.Membrane(mc.Params(**prm))
        sc = so.SolidScene()
        r_min = so.cells_to_grid(m, so.MIN_BRUSH)
        sc.add_stroke(loop, r_min)
        m.attach_scene(sc)
        m.reseed()
        inside = flood(m, m.mask_np.astype(bool), grid_cell(m, 0.15, 0.05))
        src_in = bool(inside[m.src[0], m.src[1]])
        m.step(2000)
        lv = all_levels(m)
        worst = max(np.abs(L[inside]).max() for L in lv)
        near = inside.copy()
        for di, dj in m.neighbour_offsets():                 # free cells hugging the wall, outside
            near |= shifted(m.solid_np.astype(bool), di, dj)
        near &= m.mask_np.astype(bool) & ~inside
        outside = np.abs(lv[0][near]).max()
        ok(f'closed loop, minimum brush ({so.brush_cells(m, r_min):.3f} cell): interior sealed, {lname(lat, st)}',
           worst == 0.0 and not src_in and inside.sum() > 1000 and outside > 1e-4,
           f'N={m.N}, {int(inside.sum())} interior cells, 2000 steps: max|u| inside = {worst:.1e} (3 levels); '
           f'outside, next to the wall: {outside:.2e}')

    # 3. A ONE-CELL DIAGONAL WALL (cells i = j) leaks through the sq9 diagonal links and the
    #    triangular (1,-1) links, not through sq5; the same line painted with the minimum brush
    #    seals all three.
    for lat, st in lattices:
        prm = dict(N=257, lattice=lat, stencil=st, src='cont', freq=0.05, sx=0.45, sy=-0.45, Co=0.5)
        if lat == 'tri':
            prm.update(shape='n6', rot=30 * D)
        res = []
        for kind in ('one-cell', 'brush'):
            m = mc.Membrane(mc.Params(**prm))
            N = m.N
            I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
            if kind == 'one-cell':
                m.set_solid_np(I == J)
            else:
                sc = so.SolidScene()
                sc.add_stroke([so.cell_to_grid(m, 0, 0), so.cell_to_grid(m, N - 1, N - 1)], so.cells_to_grid(m, so.MIN_BRUSH))
                m.attach_scene(sc)
            m.reseed()
            far = (J > I) & (m.mask_np == 1)
            src_far = bool(far[m.src[0], m.src[1]])
            peak_far = 0.0
            for _ in range(15):
                m.step(100)
                peak_far = max(peak_far, float(np.abs(m.field()[far]).max()))
            res.append((peak_far, src_far, int(m.solid_cells())))
        leak_expected = st == 9 or lat == 'tri'
        (p1, f1, n1), (p2, f2, n2) = res
        cond = (not f1 and not f2 and p2 == 0.0 and ((p1 > 1e-3) if leak_expected else (p1 == 0.0)))
        ok(f'one-cell diagonal wall {"LEAKS" if leak_expected else "seals"}, minimum brush seals: {lname(lat, st)}', cond,
           f'N={N}, driven cell (amplitude 1), 1500 steps: peak |u| beyond the wall {p1:.2e} (one-cell, {n1} cells), '
           f'{p2:.1e} (brush, {n2} cells)')

    # 4. SOLID CELLS STAY EXACTLY 0 in all three stored levels: the Unlicensed plate on both
    #    lattices, and objects of every kind on a full Neumann square (a rect across the array
    #    ring: the Neumann ring copy runs first, the mask pass then clamps it).
    cases = [('unlicensed sq9', dict(N=513, shape='unlicensed', stencil=9, src='pulse', sy=-0.9), None),
             ('unlicensed tri r0.85', dict(N=513, lattice='tri', shape='unlicensed', rad=0.85, src='pulse', sy=-0.9), None),
             ('objects, full square Neumann sq9', dict(N=513, bc='neumann', stencil=9, src='cont', sx=0.2, freq=0.05), 'objs')]
    for label, prm, extra in cases:
        m = mc.Membrane(mc.Params(**prm))
        if extra:
            sc = so.SolidScene()
            sc.add_rect(0.97, 0.1, 0.12, 0.3)                  # crosses the array ring (i = N-1)
            sc.add_circle(-0.4, 0.3, 0.1)
            sc.add_ngon(0.2, -0.5, 0.15, 5, rot_deg=12)
            sc.add_text('ZRL', -0.3, -0.3, 0.2, rot_deg=25)
            sc.add_stroke([(-0.8, 0.7), (-0.2, 0.9), (0.4, 0.6)], so.cells_to_grid(m, 1.5))
            m.attach_scene(sc)
        m.step(1000)
        solid = m.solid_np.astype(bool)
        worst = max(np.abs(L[solid]).max() for L in all_levels(m))
        E = m.hist['E'][-1]
        ok(f'solid cells stay exactly 0: {label}', worst == 0.0 and math.isfinite(E) and solid.sum() > 0,
           f'N={m.N}, {int(solid.sum())} solid cells, 1000 steps: max|u| on them {worst:.1e} (3 levels); E {E:.4g}')

    # 5. MIRROR SYMMETRY, EXACT: objects placed as mirror pairs about x = 0 (turned +-20 deg), a
    #    mirrored pair of strokes and an on-axis source, sq5 Dirichlet. The rasterisation is exact
    #    in floating point (negated coordinates give negated local coordinates), and the 5-point
    #    sum (i-1) + (i+1) + ... is exactly symmetric under i -> N-1-i.
    m = mc.Membrane(mc.Params(N=257, src='pulse', sigma=3, sx=0.0, sy=0.1))
    sc = so.SolidScene()
    for sgn in (1, -1):
        sc.add_rect(sgn * 0.35, 0.25, 0.2, 0.08, rot_deg=sgn * 20)
        sc.add_circle(sgn * 0.5, -0.5, 0.05)
        sc.add_stroke([(sgn * 0.6, 0.6), (sgn * 0.3, 0.7), (sgn * 0.1, 0.55)], so.cells_to_grid(m, 2.0))
    sc.add_circle(0.0, -0.35, 0.08)
    m.attach_scene(sc)
    m.reseed()
    mask_sym = np.array_equal(m.solid_np, m.solid_np[::-1, :])
    m.step(1500)
    u = m.field()
    res = float(np.abs(u - u[::-1, :]).max())
    ok('mirror-symmetric solids + on-axis source keep the mirror symmetry exactly (sq5 Dirichlet)',
       mask_sym and res == 0.0 and m.src[0] == (m.N - 1) // 2,
       f'N={m.N}: solid mask mirror-{"identical" if mask_sym else "DIFFERENT"} ({int(m.solid_np.sum())} cells); '
       f'1500 steps: max|u - mirror(u)| = {res:.1e} (max|u| {np.abs(u).max():.3g})')

    # 6. OBJECT AREAS: solid cells x cell area vs the analytic area, both lattices (N = 1025).
    shapes = [('circle r 0.3', so.SolidObject('circle', x=0.05, y=-0.02, r=0.3), math.pi * 0.3 ** 2, 2 * math.pi * 0.3),
              ('rect 0.5 x 0.3 at 17 deg', so.SolidObject('rect', x=-0.03, y=0.04, w=0.5, h=0.3, rot_deg=17), 0.15, 1.6),
              ('pentagon r 0.3 at 10 deg', so.SolidObject('ngon', x=0.02, y=0.01, r=0.3, n=5, rot_deg=10),
               2.5 * 0.09 * math.sin(2 * math.pi / 5), 10 * 0.3 * math.sin(math.pi / 5)),
              ('hexagon r 0.25', so.SolidObject('ngon', r=0.25, n=6), 3 * 0.0625 * math.sin(math.pi / 3), 1.5)]
    for lat in ('sq', 'tri'):
        prm = dict(N=1025, lattice=lat, src='impulse')
        if lat == 'tri':
            prm.update(shape='n6', rot=30 * D)
        m = mc.Membrane(mc.Params(**prm))
        h = m.base_half()
        cell_a = mc.S3H if lat == 'tri' else 1.0
        for label, obj, area_u, perim_u in shapes:
            sc = so.SolidScene([obj])
            m.attach_scene(sc)
            got = m.solid_cells() * cell_a
            want = area_u * h * h
            err = (got - want) / want
            # A lattice count differs from the area by a boundary term. For an edge in a generic
            # direction it averages out (measured 30/09/2026: ~1e-4 relative); an edge ALONG a
            # lattice line (the hexagon's vertical sides at x = +-110.85 on the square lattice
            # lose 0.35 cell per row per side) is off by up to half a line spacing per unit of
            # length. So the bound is 0.5 x the perimeter (in cells), and err/perimeter is shown.
            perim = perim_u * h
            ok(f'area {label}, {lat}', abs(got - want) <= 0.5 * perim,
               f'N={m.N}: {m.solid_cells()} cells x {cell_a:.4f} = {got:.1f} vs {want:.1f} (rel {err:+.1e}; '
               f'error / perimeter {(got - want) / perim:+.3f})')

    # 7. THE UNLICENSED PLATE: domain cells x cell area vs W H - (4 - pi) rc^2, W = 2R, H = R,
    #    rc = 0.065 H (the triangular array is a rhombus: at radius 1 it clips the plate's
    #    corners, so radius 0.85 there, which fits).
    for lat, rad in (('sq', 1.0), ('tri', 0.85), ('sq', 0.6)):
        m = mc.Membrane(mc.Params(N=1025, lattice=lat, shape='unlicensed', rad=rad, src='impulse'))
        R = m.dom_R()
        hw, hh, rc = m.plate_dims()
        want = (2 * hw) * (2 * hh) - (4 - math.pi) * rc * rc
        cell_a = mc.S3H if lat == 'tri' else 1.0
        got = int(m.domain_mask_np.sum()) * cell_a
        err = (got - want) / want
        perim = 4 * (hw + hh) - (8 - 2 * math.pi) * rc
        # the plate's straight sides run ALONG lattice lines (rows / columns), so the count is
        # off by up to half a line spacing per unit of side (see the area test above)
        ok(f'Unlicensed plate area, {lat} radius {rad}', abs(got - want) <= 0.5 * perim and abs(hw - 2 * hh) < 1e-12,
           f'N={m.N}: R = {R:.2f}, plate {2 * hw:.1f} x {2 * hh:.1f}, corner {rc:.2f}: {got:.1f} vs {want:.1f} '
           f'(rel {err:+.1e}, error / perimeter {(got - want) / perim:+.3f}); lettering + screws {m.solid_cells()} cells')

    # 8. update_solids MID-RUN: new solid cells are zeroed in all three levels, every other cell
    #    is untouched, the incremental mask equals a full re-rasterisation (host and device), and
    #    the run carries on with a finite energy.
    for lat, st in lattices:
        prm = dict(N=513, lattice=lat, stencil=st, src='pulse', sigma=4, bc='mur', Co=0.5)
        if lat == 'tri':
            prm.update(shape='n6', rot=30 * D)
        m = mc.Membrane(mc.Params(**prm))
        m.attach_scene(so.SolidScene())
        m.step(150)
        before = all_levels(m)
        sc = m.scene
        ever = np.zeros((m.N, m.N), dtype=bool)          # every cell solid at any point of the edit

        def upd(bb):
            m.update_solids(bb)
            if m.solid_np is not None:
                ever[:] |= m.solid_np.astype(bool)
        upd(sc.add_circle(0.25, 0.0, 0.1)[1])
        r = so.cells_to_grid(m, 2.0)
        upd(sc.begin_stroke(-0.3, -0.3, r))
        for k in range(1, 12):
            upd(sc.extend_stroke(-0.3 + 0.05 * k, -0.3 + 0.04 * k * (-1) ** k))
        sc.end_stroke()
        upd(sc.begin_stroke(-0.05, -0.1, so.cells_to_grid(m, 3.0), erase=True))   # an eraser cut
        upd(sc.extend_stroke(0.0, -0.05))
        sc.end_stroke()
        after = all_levels(m)
        solid = m.solid_np.astype(bool)
        erased = int((ever & ~solid).sum())
        zeroed = max(np.abs(L[ever]).max() for L in after)          # erased cells stay 0 until the wave returns
        untouched = all(np.array_equal(A[~ever], Bv[~ever]) for A, Bv in zip(before, after))
        inc_host, inc_dev = m.mask_np.copy(), m.M.to_numpy().astype(np.uint8)
        full = m.domain_mask_np & (sc.rasterize_full(m) ^ 1)
        same = np.array_equal(inc_host, full) and np.array_equal(inc_dev, full)
        m.step(500)
        E = np.array(m.hist['E'][-500:])
        solid_after = max(np.abs(L[solid]).max() for L in all_levels(m))
        ok(f'update_solids mid-run ({lname(lat, st)})',
           zeroed == 0.0 and untouched and same and erased > 0 and np.isfinite(E).all() and solid_after == 0.0,
           f'N={m.N}: {int(solid.sum())} cells made solid at step 150 (circle + 12-point stroke; the eraser freed {erased}): '
           f'max|u| on them {zeroed:.1e} (3 levels), other cells {"untouched" if untouched else "CHANGED"}, '
           f'incremental mask {"==" if same else "!="} full re-rasterisation; 500 more steps: '
           f'E {E[0]:.4g} -> {E[-1]:.4g}, solids {solid_after:.1e}')

    # 9. SOURCES SNAP OUT OF SOLIDS: (a) the web app's walk toward the centre, (b) when the
    #    whole walk is solid, the nearest free cell in lattice hops (checked by brute force),
    #    (c) the driven cells never sit in a solid.
    for lat, st in lattices:
        prm = dict(N=257, lattice=lat, stencil=st, src='cont', sx=0.3, sy=0.1, freq=0.05)
        if lat == 'tri':
            prm.update(shape='n6', rot=30 * D)
        m = mc.Membrane(mc.Params(**prm))
        want = m.src[:2]
        gx, gy = so.cell_to_grid(m, *want)
        sc = so.SolidScene()
        sc.add_circle(gx, gy, 0.06)
        m.attach_scene(sc)
        a_ok = m.mask_np[m.src[0], m.src[1]] == 1 and m.src[:2] != want and m.src[2]
        a_src = m.src[:2]
        sc.add_circle(0.0, 0.0, 0.6)                          # swallows the centre and the whole walk
        m.update_solids(None)
        si, sj = m.src[:2]
        I, J = np.nonzero(m.mask_np[1:-1, 1:-1])
        I, J = I + 1, J + 1
        hp = m.hops(I - want[0], J - want[1])
        b_ok = m.mask_np[si, sj] == 1 and int(m.hops(si - want[0], sj - want[1])) == int(hp.min())
        m.step(200)
        worst = max(np.abs(L[m.solid_np.astype(bool)]).max() for L in all_levels(m))
        ok(f'source snaps out of solids ({lname(lat, st)})', a_ok and b_ok and worst == 0.0,
           f'requested {want}: walk-back -> {a_src}; centre swallowed -> nearest free {(si, sj)} at '
           f'{int(m.hops(si - want[0], sj - want[1]))} hops (brute-force minimum {int(hp.min())}); '
           f'200 driven steps: max|u| in solids {worst:.1e}')
    m = mc.Membrane(mc.Params(N=257, shape='n5', rot=10 * D, src='vtx', vtx_drive=True, freq=0.05, sigma=2))
    sc = so.SolidScene()
    for v in m.vtx:
        sc.add_circle(*so.cell_to_grid(m, v[0], v[1]), 0.03)
    m.attach_scene(sc)
    free_v = all(m.mask_np[v[0], v[1]] == 1 for v in m.vtx)
    m.step(200)
    worst = max(np.abs(L[m.solid_np.astype(bool)]).max() for L in all_levels(m))
    ok('driven vertex sources snap out of solids', free_v and worst == 0.0,
       f'{len(m.vtx)} vertices, each under a solid disk: all re-placed on free cells; max|u| in solids {worst:.1e}')

    # 10. SAVE / LOAD round trip (ScenesCLAUDE/), and the text/font checks.
    m = mc.Membrane(mc.Params(N=257, shape='unlicensed', src='pulse', sy=-0.9))
    sc = m.scene
    sc.add_text('HI', 0.5, 0.0, 0.1, rot_deg=-10, variation='Bold Condensed')
    sc.add_stroke([(-0.9, -0.2), (-0.7, -0.35)], so.cells_to_grid(m, 1.5))
    sc.plate_screws = False
    m.update_solids(None)
    path = sc.save('test_roundtrip_MIRROR.json')
    sc2 = so.SolidScene.load(path)
    same_d = sc2.to_dict() == sc.to_dict()
    ras = np.array_equal(sc2.rasterize_full(m), m.solid_np)
    try:
        so.resolve_font('no_such_font_MIRROR.ttf')
        err_font = False
    except FileNotFoundError:
        err_font = True
    try:
        so.text_layout('X', variation='Ultra Wide')
        err_var = False
    except ValueError:
        err_var = True
    names = so.font_variations()
    ok('scene save/load round trip; font errors are clear', same_d and ras and err_font and err_var
       and 'SemiBold SemiCondensed' in names and os.path.basename(path).endswith('MIRROR.json'),
       f'{os.path.relpath(path, HERE)}: dict {"=" if same_d else "!="}, rasterisation {"=" if ras else "!="}; '
       f'missing font -> FileNotFoundError: {err_font}; unknown variation -> ValueError: {err_var}; '
       f'Bahnschrift variations: {len(names)}')

    # 11. TIMINGS: one painted segment (extend_stroke + update_solids, synchronised) over the
    #     Unlicensed plate, a full re-rasterisation, and the setup; target < 20 ms per segment.
    for N in ((2049,) if a.quick else (2049, 8193)):
        t0 = time.perf_counter()
        m = mc.Membrane(mc.Params(N=N, shape='unlicensed', src='pulse', sy=-0.9))
        t_setup = time.perf_counter() - t0
        sc = m.scene
        m.step(16)
        ti.sync()
        t0 = time.perf_counter()
        m.update_solids(None)
        ti.sync()
        t_full = time.perf_counter() - t0
        r = so.cells_to_grid(m, 3.0)
        m.update_solids(sc.begin_stroke(-0.8, -0.05, r))
        ts = []
        for k in range(1, 81):
            x = -0.8 + 0.008 * k                 # ~8 cells per segment at N = 2049, ~33 at 8193
            y = -0.05 + 0.3 * math.sin(k / 8)
            t0 = time.perf_counter()
            m.update_solids(sc.extend_stroke(x, y))
            ti.sync()
            ts.append(time.perf_counter() - t0)
        sc.end_stroke()
        ts = np.array(ts) * 1e3
        m.step(8)
        ti.sync()
        ok(f'painting speed N={N}', (np.median(ts) < 20) if N <= 2049 else True,
           f'extend_stroke + update_solids: median {np.median(ts):.2f} ms, p90 {np.percentile(ts, 90):.2f}, '
           f'max {ts.max():.2f} (80 segments over the lettering, brush 3 cells); full re-rasterisation '
           f'{t_full * 1e3:.0f} ms; Membrane setup with the plate {t_setup:.2f} s')
        del m

    # 12. (01/10/2026, Claude Opus 5.5) ONE LONG diagonal segment (a fast drag in a zoomed-out
    #     view): the incremental path (old mask + the new capsule, tile by tile) must equal the full
    #     re-rasterisation bit for bit (draw, then an eraser through the walls, on sq and tri, plate
    #     and plain domains), and stay interactive at large N (it took 1.3-2.4 s at N = 8193).
    eq_all, det12 = True, []
    for lat, shape, N in (('sq', 'unlicensed', 513), ('tri', 'unlicensed', 513), ('sq', 'n6', 257),
                          ('tri', 'circle', 257)):
        prm = dict(N=N, shape=shape, lattice=lat, src='pulse', sy=-0.9, rot=17 * D)
        if lat == 'tri':
            prm['rad'] = 0.85
        m = mc.Membrane(mc.Params(**prm))
        sc = m.ensure_scene()
        bad = 0
        for erase, pts in ((False, [(-0.9, -0.5), (0.85, 0.45), (-0.7, 0.4), (0.6, -0.55)]),
                           (True, [(-0.8, 0.5), (0.8, -0.45)])):
            r = so.cells_to_grid(m, 1.0 if not erase else 2.0)
            m.update_solids(sc.begin_stroke(*pts[0], r, erase))
            for q in pts[1:]:
                m.update_solids(sc.extend_stroke(*q))
                ref = sc.rasterize_full(m)
                ref = np.zeros((m.N, m.N), np.uint8) if ref is None else ref
                cur = np.zeros((m.N, m.N), np.uint8) if m.solid_np is None else m.solid_np
                bad += int(np.count_nonzero(ref != cur))
                bad += int(np.count_nonzero(m.M.to_numpy().astype(np.uint8) != (m.domain_mask_np & (ref ^ 1))))
            sc.end_stroke()
        eq_all &= bad == 0
        det12.append(f'{lat} {shape} N={N}: {bad} cells differ')
        del m
    Nl = 2049 if a.quick else 8193
    m = mc.Membrane(mc.Params(N=Nl, shape='unlicensed', src='pulse', sy=-0.9))
    sc = m.scene
    m.update_solids(sc.begin_stroke(-0.75, -0.6, so.cells_to_grid(m, 3.0)))
    ti.sync()
    t0 = time.perf_counter()
    m.update_solids(sc.extend_stroke(-0.75 + 0.9, -0.6 + 0.9))     # ~0.45 N cells at 45 deg
    ti.sync()
    t_long = (time.perf_counter() - t0) * 1e3
    sc.end_stroke()
    del m
    ok('one long diagonal segment: incremental == full re-rasterisation, and fast', eq_all and t_long < 250,
       '; '.join(det12) + f'; N={Nl} segment of ~{0.45 * Nl:.0f} cells at 45 deg over the plate: {t_long:.0f} ms (< 250)')

    print(f'\n{"ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: " + ", ".join(FAILS)} '
          f'({a.arch} {"f32" if a.f32 else "f64"}, {time.perf_counter() - T0:.1f} s)')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
