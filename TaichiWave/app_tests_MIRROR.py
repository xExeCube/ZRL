"""ZRL 2D wave membrane -- Taichi port: tests of the VIEWER (plain script, headless).

    python app_tests_MIRROR.py [--arch cuda|vulkan|cpu] [--f32] [--quick]

One offscreen GGUI window (show_window=False; GGUI allows one per process) renders every frame.
There is no OS window behind it (measured 30/09/2026), so input is driven through
app_input_MIRROR.Frame -- the same handler the live window feeds. What is checked:
  1. palette: zero 0.05 / exposure 1 / contrast 1 is the web app's colorOf 'div' (host), and the
     kernel's palette() equals the host copy;
  2. 2D view: the cell every pixel SHOWS (k_image_index, the same pixel_cell() as the image) equals
     the cell the cursor at that pixel's centre MAPS to (View2D + Membrane.xy_to_cell), every
     pixel, square and triangular, zoomed and panned;
  3. 3D view: markers rendered by GGUI at known world points land where CamPose.world_to_screen
     predicts (pixel error), and screen_to_plane inverts world_to_screen on the rest plane;
  4. orbit camera: clamps, no flip, freelook keeps the orbit state, f32 slider tolerance;
  5. input: painting through fake mouse frames in 2D and 3D (stroke points vs the intended grid
     points, cells), SHIFT erases, panel clicks do not paint, orbit/zoom/pan drags, hotkeys,
     W/A/S/D/Q/E change nothing, the text entry (typing, BackSpace, digits, Return, Escape);
  6. Unlicensed: shape cycle, source/radius defaults, lettering and screw toggles;
  7. every panel drawn in a real offscreen GGUI window in several states (no exception), and the
     frame time at N = 2049 with all panels open;
  8. painting latency (update_solids per segment) at N = 513 and 2049;
  9. contrast: mean CIE L* of domain vs background pixels, round-1 look vs now, both views;
 10. palettes and transparency (01/10/2026): every preset and custom stops, kernel == host, the zero
     level, the default == the 30/09 formula, the settings file (round trip, broken file, auto-save
     debounce), the palette editor (add / remove / colour / zero), measured alpha blending of the
     surface (2D and 3D) and of the grid lines, back-to-front order, frame cost.
 11. round 4 / v1.0 (05/10/2026, t_round4): the version (constant, title, panel header, --version, the
     --shot log line); the boundary control per lattice and shape (B cycles bc_info()'s options; the
     triangular 'array' skips Mur; a triangular web-app shape keeps its bc, with the note); the domain
     edge (set_edge keeps the field; greyed on 'array' / the full square); 'array' in the G cycle with
     radius / rotation greyed, fitted in 2D and (06/10/2026) in 3D; wall types (brush, object, plate lettering), the legend and counts (and
     their wait while edits rest), the scene round trip with wall types; rigid vs pinned pixels in 2D
     and 3D, the edge overlay (rigid line, dashed torus seam); the vertex inset labels, the driven-bump
     checkbox, the source note, the drive warning, the phase lag; the panel rectangles never overlap
     (default window and maximised) and every panel renders in a real GGUI frame in the new states.
*Generated from scratch -- Claude Opus 5.5 -- 30/09/2026; section 11 added by Claude Opus 5.5 -- 05/10/2026 (the 3D fit check 06/10/2026)*
"""
import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import taichi as ti
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import membrane_core_MIRROR as mc
import solids_MIRROR as so
import app_render_MIRROR as ar
import app_camera_MIRROR as acam
import app_input_MIRROR as ain
import membrane_app_MIRROR as app
import app_solids_MIRROR as asol

ARCH, F32 = 'cuda', False             # set by main() (t_round4 runs a --shot subprocess on the same backend)

OUT = os.path.join(HERE, 'ScratchCLAUDE', 'AppCLAUDE')
OUT4 = os.path.join(HERE, 'ScratchCLAUDE', 'Round4CLAUDE', 'AppCLAUDE')   # section 11's images (05/10/2026)
RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok), detail))
    print(f'{"PASS" if ok else "FAIL"}  {name}' + (f'   [{detail}]' if detail else ''), flush=True)


class Ctx:
    def __init__(self, win):
        self.window = ti.ui.Window('app tests', win, vsync=False, show_window=False)
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.gui = self.window.get_gui()
        self.camera = ti.ui.Camera()
        self.camera.z_near(0.01)
        self.camera.z_far(100.0)


def frame(v, ctx, path=None, gui=True, markers=()):
    v.draw(ctx.window, ctx.canvas, ctx.scene, ctx.camera, markers=markers)
    if gui and v.panel:
        v.gui(ctx.gui)
    if path:
        ctx.window.save_image(path)
        return np.asarray(Image.open(path).convert('RGB'))
    # every frame must be rendered: GGUI keeps the draw calls of unrendered frames (measured 30/09:
    # after 18 draws without a render, the next save_image showed a stale 2D image over the scene).
    # window.show() on a show_window=False window SEGFAULTS (30/09), so the readback flushes.
    ctx.window.get_image_buffer_as_numpy()
    return None


def fresh(v, **params):
    p = v.m.p
    for k, val in params.items():
        setattr(p, k, val)
    v.reset()


# ---- 1. palette --------------------------------------------------------------------------------
@ti.kernel
def k_pal(vals: ti.types.ndarray(ndim=1), LP: ti.template(), out: ti.types.ndarray(ndim=2)):
    for i in range(vals.shape[0]):
        c = ar.palette(ti.cast(vals[i], ti.f32), LP)
        for q in ti.static(range(3)):
            out[i, q] = c[q]


def t_palette(v):
    vs = np.linspace(-1.5, 1.5, 601)
    web = np.array([ar.web_palette(x) for x in vs])
    L = ar.Look(zero=0.05, exposure=1.0, contrast=1.0)
    host = ar.palette_np(vs, L)
    check('palette: zero 0.05, exposure 1, contrast 1 == web colorOf div (host)', np.abs(host - web).max() < 1e-12,
          f'max diff {np.abs(host - web).max():.1e}')
    for look in (L, ar.Look(), ar.Look(zero=0.3, exposure=1.7, contrast=2.5)):
        v.R.set_params(look, 1.0, 1.0)
        out = np.zeros((vs.size, 3), np.float32)
        k_pal(vs.astype(np.float32), v.R.LP, out)
        ref = ar.palette_np(vs, look)
        err = np.abs(out - ref).max()
        check(f'palette: kernel == host (zero {look.zero}, exposure {look.exposure}, contrast {look.contrast})',
              err < 2e-6, f'max diff {err:.1e}')
    # sign separation and monotone luminance of the default look (the web app's gradient checks)
    look = ar.Look()
    xs = np.linspace(0, 1, 257)
    Yp = ar.rel_lum(ar.palette_np(xs, look))
    Yn = ar.rel_lum(ar.palette_np(-xs, look))
    mono = bool(np.all(np.diff(Yp) > -2e-3) and np.all(np.diff(Yn) > -2e-3))
    sep = min(np.linalg.norm(ar.palette_np(x, look) - ar.palette_np(-x, look)) for x in np.linspace(0.5, 1, 65))
    check('palette (default look): luminance rises with |u| on both signs, sign separation at |u|>=1/2',
          mono and sep > 0.5, f'min separation {sep:.3f} (web app house ramp: 0.555)')


# ---- 2. 2D mapping -------------------------------------------------------------------------------
def t_map2d(v, quick):
    W, H = v.win
    cfgs = [('sq', 513, 1.0, (0.0, 0.0)), ('tri', 257, 3.7, (0.2, -0.1)), ('sq', 161, 12.5, (-0.61, 0.33)),
            ('sq', 257, 1.0, (0.0, 0.0), 0.125)]          # v3: picture centred in the panel-free strip
    if not quick:
        cfgs.append(('sq', 2049, 0.6, (0.05, 0.0)))
        cfgs.append(('tri', 1025, 1.0, (0.0, 0.0)))
    out = np.zeros((W, H, 2), np.int32)
    for lat, N, zoom, pan, *ox in cfgs:
        fresh(v, lattice=lat, N=N, shape='square' if lat == 'sq' else 'n6')
        m = v.m
        v.v2.zoom = zoom
        v.v2.ox = ox[0] if ox else 0.0
        v.v2.cx, v.v2.cy = acam.grid_to_phys(m, *pan)
        v.R.set_params(v.look, 1.0, 1.0, v.v2.kernel_cx(m, W, H), v.v2.cy, v.v2.px_size(m, H))
        ar.k_image_index(m.N, 1 if m.tri() else 0, v.R.LP, out)
        PX, PY = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5, indexing='ij')
        # the cursor at each pixel centre, through the viewer's own mapping
        x, y = v.v2.screen_to_phys(m, PX / W, PY / H, W, H)
        fi, fj = m.xy_to_cell(x, y)
        I = np.floor(np.asarray(fi) + 0.5).astype(np.int64)
        J = np.floor(np.asarray(fj) + 0.5).astype(np.int64)
        inside = (I >= 0) & (I < m.N) & (J >= 0) & (J < m.N)
        I[~inside] = -1
        J[~inside] = -1
        bad = (I != out[..., 0]) | (J != out[..., 1])
        nb = int(bad.sum())
        # a mismatch is only acceptable ON a cell boundary (f32 in the kernel vs f64 here)
        dist = np.minimum(np.abs((np.asarray(fi) + 0.5) % 1.0 - 0.0), np.abs((np.asarray(fi) + 0.5) % 1.0 - 1.0))
        distj = np.minimum(np.abs((np.asarray(fj) + 0.5) % 1.0), np.abs((np.asarray(fj) + 0.5) % 1.0 - 1.0))
        edge = np.minimum(dist, distj)
        worst = float(edge[bad].max()) if nb else 0.0
        check(f'2D mapping: cell shown == cell under the cursor, {lat} N={N} zoom {zoom} pan {pan} ox {v.v2.ox}',
              worst < 1e-4, f'{nb} of {W * H:,} pixels differ' + (f', all within {worst:.1e} cells of a cell edge' if nb else ''))
    v.v2.reset()
    v.v2.ox = 0.0


# ---- 3. 3D projection ------------------------------------------------------------------------------
MARK = [((0.0, 0.0, 0.0), (1.0, 0.0, 1.0)), ((0.5, 0.0, -0.3), (1.0, 1.0, 0.0)),
        ((-0.7, 0.1, 0.6), (0.0, 1.0, 1.0)), ((0.3, 0.25, 0.3), (1.0, 1.0, 1.0))]


def t_project3d(v, ctx):
    W, H = v.win
    worst = 0.0
    poses = [(0.0, 38.0, 2.9), (35.0, 60.0, 2.0), (-120.0, 20.0, 3.5), (170.0, 85.0, 1.5), (60.0, -30.0, 2.5),
             (-45.0, 45.0, 0.9)]
    for az, el, dist in poses:
        o = acam.OrbitCam(az, el, dist)
        pose = o.pose()
        pose.apply(ctx.camera)
        sc = ctx.scene
        sc.set_camera(ctx.camera)
        sc.ambient_light((1.0, 1.0, 1.0))
        v.R.markers3d(sc, MARK, radius=0.006)
        ctx.canvas.set_background_color((0.0, 0.0, 0.0))
        ctx.canvas.scene(sc)
        # signs as p / m: the user's naming rule has no hyphens (review, 06/10/2026)
        sg = lambda x: f'{x:+.0f}'.replace('+', 'p').replace('-', 'm')  # noqa: E731
        path = os.path.join(OUT, f'proj_marker_az{sg(az)}_el{sg(el)}_MIRROR.png')
        ctx.window.save_image(path)
        img = np.asarray(Image.open(path).convert('RGB')).astype(int)
        errs = []
        for P, c in MARK:
            want = pose.world_to_screen(P, W / H)
            if want is None or not (0.01 < want[0] < 0.99 and 0.01 < want[1] < 0.99):
                continue                                         # behind the eye or off the frame
            ref = np.array(c) * 255
            sel = np.abs(img - ref).max(axis=2) <= 40
            rows, cols = np.nonzero(sel)
            if rows.size == 0:
                errs.append(float('inf'))
                continue
            px, py = cols.mean() + 0.5, rows.mean() + 0.5        # pixel-centre coordinates
            wx, wy = want[0] * W, (1 - want[1]) * H
            errs.append(math.hypot(px - wx, py - wy))
        e = max(errs) if errs else float('inf')
        worst = max(worst, e)
        check(f'3D projection: GGUI marker pixel == world_to_screen, az {az} el {el} dist {dist}', e <= 1.0,
              f'max error {e:.3f} px over {len(errs)} markers')
    # the inverse on the rest plane
    rng = np.random.default_rng(1)
    err = 0.0
    for az, el, dist in poses:
        pose = acam.OrbitCam(az, el, dist).pose()
        for _ in range(200):
            X, Z = rng.uniform(-1.4, 1.4, 2)
            s = pose.world_to_screen((X, 0.0, Z), W / H)
            if s is None:
                continue
            hit = pose.screen_to_plane(s[0], s[1], W / H)
            if hit is not None:
                err = max(err, math.hypot(hit[0] - X, hit[1] - Z))
    check('3D unprojection: screen_to_plane(world_to_screen(P)) == P on the rest plane', err < 1e-9,
          f'max error {err:.1e} world units')
    return worst


# ---- 4. orbit camera -------------------------------------------------------------------------------
def t_orbit(v, ctx):
    o = acam.OrbitCam()
    o.el = 120
    o.clamp()
    ok1 = o.el == acam.EL_MAX
    o.el = -500
    o.clamp()
    ok1 &= o.el == -acam.EL_MAX
    o.dist = 1e-5
    o.clamp()
    ok1 &= o.dist == acam.DIST_MIN
    o.dist = 1e5
    o.clamp()
    ok1 &= o.dist == acam.DIST_MAX
    o.az = 200
    o.clamp()
    ok1 &= abs(o.az - (-160)) < 1e-12
    check('orbit: elevation / distance clamps, azimuth wraps to (-180, 180]', ok1)
    flip = 0
    for el in np.linspace(-acam.EL_MAX, acam.EL_MAX, 357):
        for az in (-179.0, -90.0, 0.0, 45.0, 180.0):
            pose = acam.OrbitCam(az, el, 2.0).pose()
            s, u, f = pose.frame()
            t = pose.target
            if not (u[1] > 0 and abs(math.dist(pose.eye, t) - 2.0) < 1e-9):
                flip += 1
    check('orbit: no flip (screen-up has +Y) and fixed target at every elevation in [-89, 89]', flip == 0,
          f'{flip} bad poses of {357 * 5}')
    # zoom limits through the key and the drag
    v.view3d = True
    v.orbit.reset()
    for _ in range(60):
        v.key('z')
    zin = v.orbit.dist
    for _ in range(120):
        v.key('z', shift=True)
    zout = v.orbit.dist
    check('orbit: Z / SHIFT+Z zoom stop at the distance limits', zin == acam.DIST_MIN and zout == acam.DIST_MAX,
          f'{zin} .. {zout}')
    v.orbit.reset()
    # freelook keeps the orbit state
    v.camera = ctx.camera
    v.orbit.az, v.orbit.el, v.orbit.dist = 33.0, 44.0, 2.2
    before = (v.orbit.az, v.orbit.el, v.orbit.dist)
    v.key('f')
    ctx.camera.position(0.3, 0.4, 0.5)                   # what WASD would do
    v.key('f')
    check('freelook toggle keeps the orbit state', (v.orbit.az, v.orbit.el, v.orbit.dist) == before and not v.freelook)
    # f32 sliders: a value that came back through f32 is not a change
    vals = [0.9, 0.22, 1.37, 2.9, 0.02, 1e-3, 88.9, -0.9, 7.123456789, 0.6]
    bad = [x for x in vals if app.Viewer.moved(float(np.float32(x)), x)]
    check('f32 slider round trip is not a change (Viewer.moved)', not bad, f'{bad}')
    v.orbit.reset()


# ---- 5. input -------------------------------------------------------------------------------------
def snapshot(v):
    p = v.m.p
    return (v.playing, v.view3d, v.wire, v.panel, dict(v.show), v.orbit.az, v.orbit.el, v.orbit.dist, v.v2.zoom,
            v.v2.cx, v.v2.cy, v.look.bg, v.look.exposure, p.shape, p.lattice, p.bc, p.src, p.medium, v.sol.draw,
            v.freelook, v.m.step_n)


def stroke_check(v, g, label):
    m = v.m
    st = m.scene.strokes[-1]
    pts = np.array(st.points)
    # every stroke point is one of the intended cursor samples (same order), to rounding
    G = np.array(g)
    d = np.sqrt(((pts[:, None, :] - G[None, :, :]) ** 2).sum(-1)).min(axis=1) * m.base_half()
    # the intended samples' cells are solid (a brush of >= 1 cell covers its own centre cell)
    miss = 0
    for gx, gy in g:
        i, j = m.round_cell(*so.grid_to_cell(m, gx, gy))
        if m.domain_mask_np[i, j] and m.solid_np[i, j] == 0:
            miss += 1
    check(f'{label}: stroke points == intended grid points, painted cells under the path',
          d.max() < 1e-7 and miss == 0 and len(pts) >= 2,
          f'{len(pts)} points, max offset {d.max():.1e} cells, {miss} path cells not solid, solid {m.n_solid:,}')


def t_input(v, ctx):
    W, H = v.win
    # 2D painting
    fresh(v, lattice='sq', N=513, shape='square', src='pulse', sx=0.0, sy=0.0, rad=1.0)
    v.view3d = False
    v.panel = True
    v.show.update(solids=True, align=False, ledger=False)
    m = v.m
    m.attach_scene(so.SolidScene())
    g, scr = app.programmatic_stroke(v, app.WALLS_TEST_PATH)
    stroke_check(v, g, '2D paint')
    # SHIFT + LMB erases
    n0 = m.n_solid
    app.programmatic_stroke(v, [(-0.3, -0.4), (-0.3, 0.2)], erase=True)
    check('2D: SHIFT+LMB paints an ERASER stroke that removes cells', m.scene.strokes[-1].erase and m.n_solid < n0,
          f'solid {n0:,} -> {m.n_solid:,}')
    # a press inside a visible panel does not paint
    ns = len(m.scene.strokes)
    x, y, w_, h_ = app.PANELS['solids']
    v.set_draw(True)
    v.handle(ain.Frame((x + w_ / 2, 1 - (y + h_ / 2)), {'LMB'}, ['LMB']))
    v.handle(ain.Frame((x + w_ / 2 - 0.3, 1 - (y + h_ / 2)), {'LMB'}, []))
    v.handle(ain.Frame((x + w_ / 2 - 0.3, 1 - (y + h_ / 2)), set(), []))
    check('a press inside a visible panel neither paints nor drags', len(m.scene.strokes) == ns and v.drag is None)
    v.set_draw(False)
    # 3D painting from an oblique orbit view
    v.view3d = True
    v.orbit.az, v.orbit.el, v.orbit.dist = 30.0, 50.0, 2.4
    g, scr = app.programmatic_stroke(v, [(0.1, -0.6), (0.6, -0.2), (0.4, 0.5)])
    stroke_check(v, g, '3D paint (az 30, el 50, dist 2.4)')
    v.orbit.az, v.orbit.el, v.orbit.dist = -140.0, 25.0, 3.1
    g, scr = app.programmatic_stroke(v, [(-0.6, 0.6), (-0.1, 0.7)])
    stroke_check(v, g, '3D paint (az -140, el 25, dist 3.1)')
    # orbit / zoom / pan drags
    v.orbit.reset()
    a0, e0, d0 = v.orbit.az, v.orbit.el, v.orbit.dist
    v.handle(ain.Frame((0.5, 0.5), {'RMB'}, ['RMB']))
    v.handle(ain.Frame((0.6, 0.45), {'RMB'}, []))
    v.handle(ain.Frame((0.6, 0.45), set(), []))
    want_az = a0 - 0.1 * app.ORBIT_DEG * W / H
    want_el = e0 + 0.05 * app.ORBIT_DEG
    check('3D: RMB drag orbits (azimuth, elevation) about the fixed target',
          abs(v.orbit.az - want_az) < 1e-9 and abs(v.orbit.el - want_el) < 1e-9 and v.orbit.dist == d0,
          f'az {a0} -> {v.orbit.az:.3f}, el {e0} -> {v.orbit.el:.3f}')
    v.handle(ain.Frame((0.5, 0.5), {'MMB'}, ['MMB']))
    v.handle(ain.Frame((0.5, 0.6), {'MMB'}, []))
    v.handle(ain.Frame((0.5, 0.6), set(), []))
    d1 = v.orbit.dist
    v.handle(ain.Frame((0.5, 0.6), {'RMB', 'Control'}, ['RMB']))
    v.handle(ain.Frame((0.5, 0.5), {'RMB', 'Control'}, []))
    v.handle(ain.Frame((0.5, 0.5), set(), []))
    check('3D: MMB drag up zooms in, CTRL+RMB drag down zooms out',
          abs(d1 - d0 / math.exp(0.1 * app.ZOOM_DRAG)) < 1e-9 and abs(v.orbit.dist - d0) < 1e-9,
          f'dist {d0} -> {d1:.4f} -> {v.orbit.dist:.4f}')
    v.view3d = False
    v.v2.reset()
    s = v.v2.px_size(m, H)
    v.handle(ain.Frame((0.5, 0.5), {'RMB'}, ['RMB']))
    v.handle(ain.Frame((0.55, 0.52), {'RMB'}, []))
    v.handle(ain.Frame((0.55, 0.52), set(), []))
    check('2D: RMB drag pans (the content follows the cursor)',
          abs(v.v2.cx + 0.05 * W * s) < 1e-9 and abs(v.v2.cy + 0.02 * H * s) < 1e-9, f'pan ({v.v2.cx:.2f}, {v.v2.cy:.2f})')
    # zoom about the cursor keeps the physical point under it
    before = v.v2.screen_to_phys(m, 0.7, 0.3, W, H)
    v.zoom(2.0, at=(0.7, 0.3))
    after = v.v2.screen_to_phys(m, 0.7, 0.3, W, H)
    check('2D: zoom keeps the point under the cursor', math.dist(before, after) < 1e-9,
          f'moved {math.dist(before, after):.1e} cells')
    v.v2.reset()
    # W/A/S/D/Q/E belong to GGUI's freelook: no app hotkey
    snap = snapshot(v)
    v.handle(ain.Frame((0.5, 0.5), set(), list('wasdqe')))
    check('keys W/A/S/D/Q/E change no app state', snapshot(v) == snap)
    # hotkeys
    v.show['align'] = v.show['ledger'] = False
    v.show['solids'] = False
    for k in 'ijk':
        v.handle(ain.Frame((0.5, 0.5), set(), [k]))
    ok = v.show['solids'] and v.show['align'] and v.show['ledger']
    v.handle(ain.Frame((0.5, 0.5), set(), ['t']))
    ok &= v.sol.draw
    v.handle(ain.Frame((0.5, 0.5), set(), ['t']))
    ok &= not v.sol.draw
    bg = v.look.bg
    v.handle(ain.Frame((0.5, 0.5), set(), ['y']))
    ok &= v.look.bg != bg
    v.look.bg = bg
    check('hotkeys I / J / K (panels), T (draw mode), Y (background)', ok)
    # text entry
    v.show['solids'] = True
    sc = m.scene
    v.sol.add(m, 'text')
    idx = v.sol.sel
    n0 = m.n_solid
    typed = ['Return'] + ['BackSpace'] * 5 + list('death') + [' ']
    v.handle(ain.Frame((0.5, 0.5), set(), typed))
    v.handle(ain.Frame((0.5, 0.5), set(), ['w', 'v', 'r']))              # typed, not hotkeys
    v.handle(ain.Frame((0.5, 0.5), set(), [], chars=['2', '0', '1', '3']))  # DigitPoller's path
    mid = v.sol.text
    v.handle(ain.Frame((0.5, 0.5), set(), ['Return']))
    check('text entry: Return opens, letters/space/BackSpace/digits type, hotkeys are suppressed, Return applies',
          mid == 'DEATH WVR2013' and sc.objects[idx].text == 'DEATH WVR2013' and v.sol.text is None
          and v.view3d is False and m.n_solid != n0, f'typed {mid!r}, object text {sc.objects[idx].text!r}')
    v.handle(ain.Frame((0.5, 0.5), set(), ['Return', 'x', 'Escape']))
    check('text entry: Escape cancels', sc.objects[idx].text == 'DEATH WVR2013' and v.sol.text is None)
    v.handle(ain.Frame((0.5, 0.5), set(), ['Return'] + ['BackSpace'] * 20 + ['Return']))
    check('text entry: an empty text keeps the old one', sc.objects[idx].text == 'DEATH WVR2013')
    # object editing through the panel methods
    bb_before = m.n_solid
    v.sol.edit(m, x=0.2, rot_deg=15.0, height=0.2)
    check('object edit (x, rotation, cap height) updates the solids', m.n_solid != bb_before)
    v.sol.delete(m)
    check('object delete', len(sc.objects) == 0 and v.sol.sel == -1)
    for kind in ('rect', 'circle', 'ngon'):
        v.sol.add(m, kind)
    check('add rect / circle / n-gon', [o.kind for o in sc.objects] == ['rect', 'circle', 'ngon'] and m.n_solid > 0)
    # undo + clear
    ns = len(sc.strokes)
    v.handle(ain.Frame((0.5, 0.5), set(), ['u']))
    check('U undoes the last stroke', len(sc.strokes) == ns - 1)
    v.sol.clear(m)
    check('clear painted walls', len(sc.strokes) == 0)
    # save / load round trip
    path = v.sol.save(m, os.path.join(OUT, 'scene_roundtrip_MIRROR.json'))
    n1 = m.n_solid
    v.sol.load(m, path)
    check('scene save / load round trip (same solid cells)', m.n_solid == n1, f'{n1:,} solid cells')
    m.attach_scene(so.SolidScene())


def t_unlicensed(v):
    fresh(v, lattice='sq', N=513, shape='square', sx=0.0, sy=0.0, rad=1.0)
    seen = []
    for _ in range(len(mc.SHAPES)):
        v.handle(ain.Frame((0.5, 0.5), set(), ['g']))
        seen.append(v.m.p.shape)
    check('G cycles every shape, including unlicensed', set(seen) == set(mc.SHAPES), f'{seen}')
    fresh(v, lattice='sq', N=513, shape='square', sx=0.0, sy=0.0, rad=1.0)
    m = v.m
    m.attach_scene(so.SolidScene())
    v.set_shape('unlicensed')
    free = m.mask_np[m.src[0], m.src[1]] == 1
    check('Unlicensed: entering moves the default source to sy = -0.9 (out of the A counter), plate drawn',
          m.p.sy == -0.9 and m.n_solid > 0 and free, f'solid {m.n_solid:,}, src {m.src[:2]}')
    n_all = m.n_solid
    v.sol.set_preset(m, plate_text=False)
    n_screws = m.n_solid
    v.sol.set_preset(m, plate_screws=False)
    n_none = m.n_solid
    v.sol.set_preset(m, plate_text=True, plate_screws=True)
    check('Unlicensed: lettering and screw toggles', n_all > n_screws > n_none == 0 and m.n_solid == n_all and not
          (n_none and m.has_solids), f'{n_all:,} / screws only {n_screws:,} / none {n_none:,}')
    v.handle(ain.Frame((0.5, 0.5), set(), ['l']))
    check('Unlicensed on the triangular lattice: radius drops to 0.89', m.p.lattice == 'tri' and m.p.rad == app.UNL_TRI_RAD
          and m.n_solid > 0, f'rad {m.p.rad}, solid {m.n_solid:,}')
    fresh(v, lattice='sq', shape='square', sx=0.0, sy=0.0, rad=1.0)


def t_gui(v, ctx, quick):
    """Every panel through a real (offscreen) GGUI window, in several states."""
    states = []
    errs = []
    fresh(v, lattice='sq', N=257, shape='circle', src='pulse')
    v.panel = True
    v.show.update(main=True, solids=True, align=True, ledger=True)
    m = v.m
    m.attach_scene(so.SolidScene())
    v.sol.add(m, 'text')
    for label, fn in (('3D default', lambda: None),
                      ('2D', lambda: setattr(v, 'view3d', False)),
                      ('typing', lambda: v.handle(ain.Frame((0.5, 0.5), set(), ['Return', 'a']))),
                      ('draw mode', lambda: (v.handle(ain.Frame((0.5, 0.5), set(), ['Escape', 't'])))),
                      ('unlicensed', lambda: v.set_shape('unlicensed')),
                      ('tri', lambda: v.handle(ain.Frame((0.5, 0.5), set(), ['l']))),
                      ('3D freelook', lambda: (setattr(v, 'view3d', True), v.set_freelook(True, ctx.camera))),
                      ('vtx source', lambda: (setattr(v.m.p, 'src', 'vtx'), v.reset())),
                      ('mode source', lambda: (setattr(v.m.p, 'src', 'mode'), setattr(v.m.p, 'shape', 'square'), v.reset())),
                      ('rhombus', lambda: (setattr(v.m.p, 'shape', 'rhomb'), v.reset()))):
        try:
            fn()
            v.step(8)
            frame(v, ctx, os.path.join(OUT, 'gui_state_MIRROR.png'))
            states.append(label)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            errs.append(f'{label}: {type(ex).__name__}: {ex}')
    check('every panel drawn in an offscreen GGUI window, 10 states, no exception', not errs,
          '; '.join(errs) if errs else ', '.join(states))
    # alignment edge colouring overlay
    try:
        v.set_freelook(False)
        fresh(v, lattice='sq', shape='n6', src='pulse', rad=0.8)
        if v.align_panel is None:
            frame(v, ctx)
        v.align_panel.edge_col = True
        n = len(v.edge_segments())
        frame(v, ctx, os.path.join(OUT, 'align_edges_3d_MIRROR.png'))
        v.view3d = False
        frame(v, ctx, os.path.join(OUT, 'align_edges_2d_MIRROR.png'))
        v.align_panel.edge_col = False
        check('alignment panel: edge colouring overlay drawn in both views', n == 6, f'{n} edges')
    except Exception as ex:
        check('alignment panel: edge colouring overlay drawn in both views', False, f'{type(ex).__name__}: {ex}')
    v.view3d = True
    v.set_draw(False)
    # frame time at N = 2049 with every panel open (no heavy recompute per frame)
    if not quick:
        fresh(v, lattice='sq', N=2049, shape='circle', src='pulse')
        v.show.update(main=True, solids=True, align=True, ledger=True)
        def empty():
            ctx.scene.set_camera(ctx.camera)
            ctx.canvas.scene(ctx.scene)
            ctx.window.get_image_buffer_as_numpy()
        for view, mesh in ((True, 1025), (True, 513), (False, 1025)):
            v.view3d = view
            v.mesh_max = mesh
            v.sync_domain(force=True)
            for _ in range(3):
                frame(v, ctx)
            ti.sync()
            T, B = [], []
            for _ in range(12):
                t0 = time.perf_counter()
                empty()                                     # the readback alone (headless flush)
                ti.sync()
                t1 = time.perf_counter()
                v.step(4)
                frame(v, ctx)
                ti.sync()
                t2 = time.perf_counter()
                B.append((t1 - t0) * 1e3)
                T.append((t2 - t1) * 1e3)
            med, base = float(np.median(T)), float(np.median(B))
            # 1025: GGUI's vertex upload dominates (~52 ms CUDA, ~70 ms Vulkan, 30/09/2026); the limit
            # catches regressions such as the per-frame index to_numpy() (~+50 ms, fixed 30/09)
            lim = 85.0 if (view and mesh > 513) else 30.0
            check(f'frame at N=2049, all panels open, {"3D mesh " + str(v.R.nd) if view else "2D"}: step x4 + draw + '
                  f'panels + render < {lim:.0f} ms', med - base < lim,
                  f'median {med:.1f} ms incl. a {base:.1f} ms headless readback -> ~{med - base:.1f} ms without it')
        v.mesh_max = app.MESH_MAX
        fresh(v, N=513)


def t_latency(v, quick):
    out = {}
    for N in ((513,) if quick else (513, 2049)):
        fresh(v, lattice='sq', N=N, shape='square', src='pulse', sx=0.0, sy=0.0, rad=1.0)
        v.view3d = False
        v.v2.reset()
        m = v.m
        m.attach_scene(so.SolidScene())
        v.sol.brush = 2.0
        v.sol.lat_ms = []
        path = [(-0.8, -0.7), (0.7, -0.5), (-0.6, 0.1), (0.75, 0.6), (-0.2, 0.8)]
        app.programmatic_stroke(v, path, per_seg=60)
        L = np.array(v.sol.lat_ms[1:])
        out[N] = (float(np.median(L)), float(np.percentile(L, 95)), float(L.max()), len(L))
        check(f'painting latency N={N}: extend_stroke + update_solids + sync per segment',
              np.median(L) < 20.0, f'median {out[N][0]:.2f} ms, p95 {out[N][1]:.2f} ms, max {out[N][2]:.2f} ms, '
                                  f'{out[N][3]} segments, brush 2 cells')
        m.attach_scene(so.SolidScene())
    return out


def t_contrast(v, ctx):
    """Mean L* of domain vs background pixels: the round-1 PNGs (ScratchCLAUDE/AppCLAUDE/before_*,
    rendered by the round-1 app copy) segmented with this renderer at the same camera, and now."""
    W, H = v.win
    rows = []
    for view in ('3d', '2d'):
        for shape in ('square', 'circle'):
            fresh(v, lattice='sq', N=513, shape=shape, src='pulse', sx=0.0, sy=0.0, rad=1.0)
            v.m.attach_scene(so.SolidScene())
            v.view3d = view == '3d'
            v.v2.reset()
            # round 1's camera: position (0, 1.6, 2.2), target 0
            v.orbit.az, v.orbit.el, v.orbit.dist = 0.0, math.degrees(math.atan2(1.6, 2.2)), math.hypot(1.6, 2.2)
            v.step(300)
            before = os.path.join(OUT, f'before_{view}_{shape}_MIRROR.png')
            L = v.look
            L.segment, L.floor = True, False               # round 1 drew no floor
            seg = frame(v, ctx, os.path.join(OUT, f'seg_r1_{view}_{shape}_MIRROR.png'), gui=False)
            L.segment, L.floor = False, True
            cls = ar.classify_seg(seg)
            if os.path.exists(before):
                old = np.asarray(Image.open(before).convert('RGB'))
                st = ar.contrast_stats(old, cls)
                rows.append((view, shape, 'round 1', st))
            now = frame(v, ctx, os.path.join(OUT, f'after_{view}_{shape}_MIRROR.png'), gui=False)
            L.segment = True
            seg2 = frame(v, ctx, os.path.join(OUT, f'seg_now_{view}_{shape}_MIRROR.png'), gui=False)
            L.segment = False
            st2 = ar.contrast_stats(now, ar.classify_seg(seg2))
            rows.append((view, shape, 'now', st2))
    v.orbit.reset()
    print('\ncontrast (mean CIE L* per pixel class; ratio = WCAG (Y1+0.05)/(Y2+0.05) domain vs background):')
    print(f'{"view":5s} {"shape":7s} {"look":8s} {"bg L*":>6s} {"outside":>8s} {"domain":>7s} {"dL dom-bg":>9s} {"ratio":>6s}')
    for view, shape, lab, st in rows:
        g = lambda k: st[k]['Lstar_mean'] if k in st else float('nan')
        print(f'{view:5s} {shape:7s} {lab:8s} {g("bg"):6.1f} {g("outside"):8.1f} {g("domain"):7.1f} '
              f'{st.get("dL_domain_bg", float("nan")):9.1f} {st.get("ratio_domain_bg", float("nan")):6.2f}')
    ok = all(st.get('dL_domain_bg', 0) > 15 for view, shape, lab, st in rows if lab == 'now')
    ok &= all(st.get('dL_outside_bg', 0) > 3 for view, shape, lab, st in rows if lab == 'now' and shape == 'circle')
    check('contrast now: domain >= 15 L* above the background, outside-domain >= 3 L* apart from it', ok)
    return rows


def t_resize(v):
    """Review v3 01/10/2026: the window can be resized / maximised. The 3D aspect must follow the real
    window at once (GGUI's projection does: ScratchCLAUDE/AppCLAUDE/probe_resize_v3_MIRROR.py measured
    GGUI marker pixels vs the app's math at 2560x1377 and 1200x1000, 0.0-0.1 cells), and the 2D image
    buffer once the size has settled (maximise/restore reuses the buffers)."""
    W0 = v.win
    img0 = v.R.img
    v.set_window_shape((2560, 1377), 0.0)
    a1 = abs(v.aspect() - 2560 / 1377) < 1e-12 and v.win == W0
    v.set_window_shape((2560, 1377), 0.1)
    a2 = v.win == W0
    v.set_window_shape((2560, 1377), 0.1 + app.RESIZE_SETTLE)
    a3 = v.win == (2560, 1377) and v.R.img.shape == (2560, 1377)
    v.set_window_shape((0, 0), 5.0)                      # minimised: ignored
    a4 = v.win_real == (2560, 1377)
    # the 3D cursor ray uses the real aspect: a point projected with it comes back
    m = v.m
    P = (0.55, 0.0, -0.4)
    sx, sy = v.pose().world_to_screen(P, 2560 / 1377)
    gt = acam.phys_to_grid(m, *acam.world_to_phys(m, P[0], P[2]))
    view = v.view3d
    v.view3d = True
    g = v.cursor_grid(sx, sy)
    e3 = math.hypot(g[0] - gt[0], g[1] - gt[1]) * m.base_half()
    v.view3d = view
    v.set_window_shape(W0, 10.0)
    v.set_window_shape(W0, 11.0)
    a5 = v.win == W0 and v.R.img is img0 and v.win_real == W0
    check('resize / maximise: 3D aspect follows at once, 2D buffer after settling, minimise ignored, restore reuses',
          a1 and a2 and a3 and a4 and a5 and e3 < 1e-6, f'{a1} {a2} {a3} {a4} {a5}, 3D cursor err {e3:.2e} cells')


def t_gain(v):
    """Review v3 01/10/2026: the auto colour gain on a decayed field (damping 0.02, N = 257) must not
    blow a dead field up to full scale (it showed |u| = 2.3e-7 of the start at gain 3.2e6)."""
    m = v.m
    keep = {k: getattr(m.p, k) for k in ('N', 'lattice', 'shape', 'src', 'sx', 'sy', 'rad', 'bc', 'medium', 'stencil', 'damp')}
    d0, cg0, ag0 = m.p.damp, v.cg, v.auto_gain
    v.cg, v.auto_gain = 1.0, True
    fresh(v, N=257, lattice='sq', shape='circle', src='pulse', sx=0.0, sy=0.0, rad=1.0, bc='dirichlet',
          medium='uniform', stencil=5, damp=0.02)
    v.step(300)
    live = v.colour_gain() * m.amax
    while m.step_n < 2000:
        v.step(100)
    dead = v.colour_gain() * m.amax
    n_dead = m.step_n
    capped = v.gain_capped
    fresh(v, **keep)
    fresh_ok = abs(v.colour_gain() - 1.0 / max(m.amax, 1e-12)) < 1e-9 and not v.gain_capped
    v.cg, v.auto_gain = cg0, ag0
    check('auto gain: a live decaying pulse is bright (step 300), a dead field fades (step 2000), reset clears the cap',
          live > 0.3 and dead < 1e-3 and capped and fresh_ok,
          f'step 300 shown at {live:.3f} of full scale, step {n_dead} at {dead:.2e}')


def t_text_switch(v):
    """Review v3 01/10/2026: switching / adding objects while typing must close the entry (it stayed
    open on the new rect: every later hotkey was typed into an invisible entry, and Return wrote the
    text onto the rect); Return with all panels hidden (H) shows the entry."""
    m = v.m
    v.sol.add(m, 'text')
    v.panel = False
    v.handle(ain.Frame((0.5, 0.5), set(), ['Return', 'a', 'b']))
    shown = v.panel and v.show['solids'] and v.sol.text is not None
    v.sol.add(m, 'rect')
    closed = v.sol.text is None
    rect = m.scene.objects[v.sol.sel]
    t_before = rect.text
    sol_show = v.show['solids']
    v.handle(ain.Frame((0.5, 0.5), set(), ['i']))           # a hotkey again (I toggles the solids panel)
    hot = v.show['solids'] != sol_show
    v.show['solids'] = sol_show
    v.sol.text = 'XY'                                        # an entry left open on a non-text object
    v.handle(ain.Frame((0.5, 0.5), set(), ['Return']))
    untouched = rect.text == t_before and v.sol.text is None
    v.panel = True
    check('text entry: Return with panels hidden shows it, adding an object closes it, Return never writes a rect',
          shown and closed and hot and untouched, f'{shown} {closed} {hot} {untouched}')


def t_nslider(v):
    """Review v3 01/10/2026: a drag of the N slider must not rebuild the run on every frame (0.9-1 s per
    change in 3D); the value applies, odd, once it rests app.N_SETTLE s."""
    m = v.m
    N0 = m.N
    resets = []
    orig = v.reset
    v.reset = lambda: (resets.append(m.p.N), orig())
    try:
        t = 0.0
        shown = m.N
        for N in (300, 420, 514, 600, 1024):              # a drag: a new value every frame
            v.n_slider(N, shown, t)
            shown = v.n_pending[0]
            t += 0.016
        mid = (len(resets), m.N)
        v.n_slider(shown, shown, t + 0.1)                  # resting, not long enough
        v.n_slider(shown, shown, t + 0.1 + app.N_SETTLE)
        applied = (m.N, v.n_pending)
    finally:
        v.reset = orig
    m.p.N = N0
    v.reset()
    check('N slider: a drag rebuilds nothing until the value rests, then applies it once (odd)',
          mid == (0, N0) and applied == (1025, None) and resets == [1025], f'mid {mid}, applied {applied}, resets {resets}')


def t_stroke_offview(v):
    """Review v3 01/10/2026: a programmatic stroke (--walls-test) whose path is partly behind the camera
    raised TypeError deep in Frame(); it must raise a clear ValueError and leave the state as it was."""
    view, orb = v.view3d, (v.orbit.az, v.orbit.el, v.orbit.dist)
    v.view3d = True
    v.orbit.az, v.orbit.el, v.orbit.dist = 0.0, 3.0, acam.DIST_MIN
    v.orbit.clamp()
    ns = len(v.m.scene.strokes) if v.m.scene is not None else 0
    pts = [(0.0, -0.95), (0.0, 0.95)]                  # crosses under the eye: the far half is visible only
    try:
        app.programmatic_stroke(v, pts)
        kind = 'no error'
    except ValueError:
        kind = 'ValueError'
    except Exception as ex:
        kind = type(ex).__name__
    ns2 = len(v.m.scene.strokes) if v.m.scene is not None else 0
    clean = ns2 == ns and not v.sol.draw and v.drag is None
    v.view3d = view
    v.orbit.az, v.orbit.el, v.orbit.dist = orb
    check('programmatic stroke behind the camera: clear ValueError, no stroke, draw mode restored',
          kind == 'ValueError' and clean, f'{kind}, strokes {ns} -> {ns2}')


def t_cells_cache(v):
    """Review v3 01/10/2026: the main panel's cell counts were re-summed every 0.5 s (36 ms at N = 8193,
    a periodic stutter); they must be recounted only after the masks or the solid count changed."""
    m = v.m
    calls = []
    orig = m.domain_cells
    m.domain_cells = lambda: (calls.append(1), orig())[1]
    try:
        v._cells = (0.0, 0, 0)
        v.cells()
        n1 = len(calls)
        v._cells = (v._cells[0] - 10.0,) + v._cells[1:]       # 10 s later, nothing changed
        v.cells()
        n2 = len(calls)
        v.sol.add(m, 'rect')                                   # the solid count changes
        v._cells = (v._cells[0] - 10.0,) + v._cells[1:]
        free, solid = v.cells()
        n3 = len(calls)
    finally:
        del m.domain_cells
    ok_vals = free == m.domain_cells() and solid == m.solid_cells()
    v.sol.delete(m)                                            # leave the scene as it was
    check('cell counts: recounted only when the masks / solid count change, values right',
          (n1, n2, n3) == (1, 1, 2) and ok_vals, f'calls {n1} {n2} {n3}')



# ---- typed slider values (01/10/2026) ------------------------------------------------------------
class FakeW:
    """A headless GGUI sub-window: sliders return a scripted value (a click) or their own value,
    buttons whose label contains a scripted piece return True; everything is recorded."""

    def __init__(self, picks=None, press=(), checks=None):
        self.picks, self.press = dict(picks or {}), list(press)
        self.sliders, self.buttons, self.texts, self.rows = {}, [], [], 0
        self.tcols, self.checks, self.cbox = [], [], dict(checks or {})   # 05/10/2026

    def _sl(self, kind, label, v, lo, hi):
        self.rows += 1
        self.sliders[label] = (v, lo, hi, kind)
        return self.picks.pop(label, v)

    def slider_float(self, label, v, lo, hi):
        return self._sl('f', label, v, lo, hi)

    def slider_int(self, label, v, lo, hi):
        return self._sl('i', label, v, lo, hi)

    def button(self, label):
        self.rows += 1
        self.buttons.append(label)
        return any(p in label for p in self.press)

    def text(self, t, color=None):
        self.rows += 1
        self.texts.append(t)
        self.tcols.append(color)

    def checkbox(self, label, v):
        self.rows += 1
        self.checks.append(label)
        return self.cbox.pop(label, v)

    colours = None                        # {label: rgb} scripted colour picks (01/10/2026)

    def color_edit_3(self, label, v):
        self.rows += 1
        self.__dict__.setdefault('colour_labels', []).append(label)
        return (self.colours or {}).pop(label, v)

    def __getattr__(self, name):          # other widgets: keep the value
        def f(*a, **k):
            self.rows += 1
            return a[1] if len(a) > 1 else False
        return f


class FakeG:
    """GGUI's get_gui() stand-in: v.gui(FakeG) runs the real panel code on FakeW windows."""

    def __init__(self, picks=None, press=None, checks=None):
        picks, press, checks = picks or {}, press or {}, checks or {}
        self.ws = {k: FakeW(picks.get(k), press.get(k, ()), checks.get(k)) for k in app.PANEL_TITLES}
        self.by_title = {t: self.ws[k] for k, t in app.PANEL_TITLES.items()}

    def sub_window(self, title, *a):
        return contextlib.nullcontext(self.by_title[title])


def gpass(v, picks=None, press=None, checks=None):
    g = FakeG(picks, press, checks)
    v.gui(g)
    return g.ws


def type_keys(v, text):
    """Keys for the entry the way the live window delivers them: digits . - from DigitPoller (chars),
    'e' as a GGUI key press."""
    for ch in text:
        if ch == 'e':
            v.handle(ain.Frame((0.5, 0.5), set(), ['e']))
        else:
            v.handle(ain.Frame((0.5, 0.5), set(), [], [ch]))


def typed_value(v, panel, label, text, other=None):
    """Arm, pick `label` with a click (value `other`), type `text`, Return; returns (picked, the click
    left the value alone, the value the slider shows after the typed value applied, flash message)."""
    ws = gpass(v, press={panel: [f'##ne_arm/{panel}']})
    old, lo, hi, kind = ws[panel].sliders[label]
    if other is None:
        other = lo if old != lo else hi
    gpass(v, picks={panel: {label: other}})
    picked = v.ne.target == (panel, label)
    after_pick = gpass(v)[panel].sliders[label][0]
    type_keys(v, text)
    v.flash_msg = ''
    v.handle(ain.Frame((0.5, 0.5), set(), ['Return']))
    msg = v.flash_msg
    gpass(v)
    ws = gpass(v)
    return picked, after_pick == old, ws[panel].sliders.get(label, (None,))[0], msg


def _ui_state(v):
    p = v.m.p
    return (p.lattice, p.shape, p.medium if hasattr(p, 'medium') else None, v.playing, v.panel, dict(v.show),
            v.freelook, v.view3d, v.orbit.az, v.orbit.el, v.orbit.dist, p.rad, v.sol.draw)


def t_typed(v, ctx):
    """User request 01/10/2026: type a slider's value. Pick mode (one button per panel), entry keys,
    clamping, no hotkey leakage, int + float sliders of every panel, the N slider's settle path."""
    fresh(v, lattice='sq', N=257, shape='circle', src='pulse', sx=0.0, sy=0.0, damp=0.0)
    m, p = v.m, v.m.p
    m.attach_scene(so.SolidScene())
    v.sol.add(m, 'rect')
    v.panel = True
    v.show.update(main=True, solids=True, align=True, ledger=True)
    v.ne.arm(False)
    v.type_buttons = False
    # 1. compactness: pick mode adds one row per panel; the per-slider buttons one row per slider
    ws = gpass(v)
    rows = {k: w.rows for k, w in ws.items()}
    nsl = {k: len(w.sliders) for k, w in ws.items()}
    v.type_buttons = True
    rows_b = {k: w.rows for k, w in gpass(v).items()}
    v.type_buttons = False
    arm_rows = all(sum('##ne_arm/' in b for b in w.buttons) == 1 for w in ws.values())
    check('typed values: pick mode = one row per panel; per-slider buttons add one row per slider',
          arm_rows and all(rows_b[k] - rows[k] == nsl[k] for k in rows),
          f'rows {rows}, sliders {nsl}, with buttons {rows_b}')
    # 2. keys: digits . - BackSpace; e; clamp; int rounding; bad number; Escape
    picked, kept, shown, msg = typed_value(v, 'main', 'source x', '-0.59')
    ok1 = picked and kept and p.sx == -0.59
    sx_before = p.sx
    _, _, _, msg_empty = typed_value(v, 'main', 'source x', '')
    ok_empty = p.sx == sx_before and 'nothing typed' in msg_empty
    gpass(v, press={'main': ['##ne_arm/main']})
    gpass(v, picks={'main': {'source x': 0.9}})
    type_keys(v, '-0.377')
    v.handle(ain.Frame((0.5, 0.5), set(), ['BackSpace']))
    ok_bs = v.ne.text == '-0.37'
    v.handle(ain.Frame((0.5, 0.5), set(), ['Return']))
    gpass(v)
    check('typed values: digits . - and BackSpace, the picking click keeps the value, empty entry keeps it',
          ok1 and ok_bs and ok_empty and p.sx == -0.37, f'sx {p.sx}, picked {picked} kept {kept}, "{msg_empty}"')
    typed_value(v, 'main', 'damping', '2e-3')
    ok_e = abs(p.damp - 0.002) < 1e-15
    _, _, _, msg_c = typed_value(v, 'main', 'damping', '5')
    ok_c = p.damp == 0.02 and 'clamped to 0.02' in msg_c
    _, _, _, msg_r = typed_value(v, 'main', 'steps / frame', '7.6')
    ok_r = v.sub == 8 and isinstance(v.sub, int) and 'rounded' in msg_r
    check('typed values: 2e-3 notation, clamped to the range (with a message), int sliders round',
          ok_e and ok_c and ok_r, f'damp {p.damp}, "{msg_c}", sub {v.sub} "{msg_r}"')
    p.damp = 0.0
    gpass(v, press={'main': ['##ne_arm/main']})
    gpass(v, picks={'main': {'radius': 0.5}})
    type_keys(v, '1e')
    v.handle(ain.Frame((0.5, 0.5), set(), ['Return']))
    bad_open = v.ne.open and 'not a number' in v.ne.msg
    st = _ui_state(v)
    v.handle(ain.Frame((0.5, 0.5), {'Shift'}, ['l', 'g', ' ', 'h', 'i', 'j', 'k', 'f', 'v', 'z', 'c', 'Up', 'Left',
                                               'Tab', 'm', 'o', 't', 'u', 'y', 'b']))
    st2 = _ui_state(v)
    text_only = v.ne.text == '1e'
    v.handle(ain.Frame((0.5, 0.5), set(), ['Escape']))
    gpass(v)
    check('typed values: a bad number keeps the entry open; no hotkey fires while typing; Escape cancels',
          bad_open and st == st2 and text_only and not v.ne.open and p.rad == 1.0 and not v.ne.armed,
          f'open {bad_open}, state same {st == st2}, text "{v.ne.text}", rad {p.rad}')
    gpass(v, press={'main': ['##ne_arm/main']})
    armed = v.ne.armed
    sh = dict(v.show)
    v.handle(ain.Frame((0.5, 0.5), set(), ['Escape']))
    off = not v.ne.armed and not v.ne.open
    gpass(v, press={'main': ['##ne_arm/main']})
    v.handle(ain.Frame((0.5, 0.5), set(), ['i']))           # a hotkey still works while only armed
    hot = v.show['solids'] != sh['solids']
    v.show.update(sh)
    v.handle(ain.Frame((0.5, 0.5), set(), ['Escape']))
    check('typed values: Escape stops pick mode; hotkeys still work while only armed', armed and off and hot,
          f'{armed} {off} {hot}')
    # 3. an int and a float slider of every panel (N separately below)
    res, miss = [], []
    for panel, w in gpass(v).items():
        kinds = {}
        for label, (old, lo, hi, kind) in w.sliders.items():
            if label != 'N (odd)' and kind not in kinds:
                kinds[kind] = label
        for kind in ('i', 'f'):
            if kind not in kinds:
                miss.append(f'{panel}:{kind}')
                continue
            label = kinds[kind]
            old, lo, hi, _ = gpass(v)[panel].sliders[label]
            for frac in (0.37, 0.63):
                want = lo + frac * (hi - lo)
                want = int(round(want)) if kind == 'i' else float(f'{want:.4g}')
                if want != old:
                    break
            picked, kept, shown, _ = typed_value(v, panel, label, ain.fmt_num(want, kind == 'i'))
            ok = picked and kept and shown is not None and abs(shown - want) <= 1e-9 * max(1.0, abs(want))
            res.append((panel, label, ok, shown, want))
    bad = [r for r in res if not r[2]]
    check('typed values: an int and a float slider in each panel take a typed value',
          bool(res) and not bad, f'{[(r[0], r[1]) for r in res]}, failed {bad}, panels without that kind {miss}')
    # 4. the N slider: the typed value goes through the settle wait, then applies once (odd)
    _, kept, shown, _ = typed_value(v, 'main', 'N (odd)', '300')
    pend = v.n_pending is not None and v.n_pending[0] == 300 and m.N == 257
    v.n_slider(300, 300, time.perf_counter() + app.N_SETTLE + 0.01)
    check('typed values: N goes through the settle wait, then applies once (odd)',
          kept and pend and m.N == 301 and v.n_pending is None, f'pending {pend}, N {m.N}')
    # 5. per-slider buttons still work; a queued value whose slider is not drawn is dropped
    v.type_buttons = True
    gpass(v, press={'main': ['##ne/main/damping']})
    opened = v.ne.target == ('main', 'damping')
    type_keys(v, '0.01')
    v.handle(ain.Frame((0.5, 0.5), set(), ['Return']))
    for _ in range(5):
        v.handle(ain.Frame((0.5, 0.5), set(), []))
    dropped = v.ne.pending is None
    gpass(v)
    v.type_buttons = False
    check('typed values: the per-slider button opens the entry; a value whose slider is not drawn is dropped',
          opened and dropped and p.damp == 0.0, f'opened {opened}, dropped {dropped}, damp {p.damp}')
    # 6. the real GGUI panels with pick mode armed and with an entry open
    errs = []
    try:
        v.ne.arm(True)
        frame(v, ctx)
        v.ne.start(('main', 'damping'), 0.0, 0.02, False, 0.0)
        v.ne.text = '0.0'
        v.ne.msg = '"x" is not a number'
        frame(v, ctx, os.path.join(OUT, 'typed_entry_MIRROR.png'))
    except Exception as ex:
        errs.append(f'{type(ex).__name__}: {ex}')
    v.ne.arm(False)
    check('typed values: real GGUI panels with pick mode armed and an entry open', not errs, f'{errs}')
    v.sol.delete(m)
    fresh(v, N=257, sx=0.0, sy=0.0, damp=0.0)


def t_src_note(v):
    """User request 01/10/2026: an amber note under source x/y when the source was moved out of a solid
    (Unlicensed plate: the A counter covers the centre)."""
    fresh(v, lattice='sq', N=513, shape='square', sx=0.0, sy=0.0, rad=1.0)
    m, p = v.m, v.m.p
    m.attach_scene(so.SolidScene())
    v.set_shape('unlicensed')
    n_default = v.src_moved_note()
    p.sx, p.sy = 0.0, 0.0
    v.reset()
    i0, j0 = m.round_cell(*m.xy_to_cell(0.0, 0.0))
    in_solid = bool(m.solid_np[i0, j0]) if m.solid_np is not None else False
    note = v.src_moved_note()
    free = m.mask_np[m.src[0], m.src[1]] == 1
    shown = any('source moved' in t for t in gpass(v)['main'].texts)
    core = getattr(m, 'src_moved', 'n/a')
    check('source-moved note: none at the default sy = -0.9; "moved out of a solid" at (0, 0) on the plate, '
          'shown in the main panel', n_default is None and in_solid and note is not None and 'out of a solid' in note
          and free and shown, f'default {n_default!r}, note {note!r}, core src_moved {core}')
    fresh(v, lattice='sq', N=257, shape='circle', sx=0.0, sy=0.0, rad=1.0)


def t_beam(v, ctx):
    """Beam phase 01/10/2026: the directional sources in the O cycle and the panel, their sliders take
    typed values (pick mode), live beam changes, the emitter overlay in 2D and 3D (and its toggle)."""
    fresh(v, lattice='sq', N=257, shape='square', src='vtx', sx=0.0, sy=0.0, damp=0.0, freq=0.04,
          beam_angle=0.0, beam_width=8.0)
    m, p = v.m, v.m.p
    seq = []
    for _ in range(3):
        v.handle(ain.Frame((0.5, 0.5), set(), ['o']))
        seq.append(p.src)
    check('beam: O cycles vtx -> beam -> packet -> impulse', seq == ['beam', 'packet', 'impulse'], f'{seq}')
    fresh(v, src='beam')
    v.panel = True
    v.show.update(main=True)
    sl = gpass(v)['main'].sliders
    want = ('beam angle (deg)', 'beam width w (cells)', 'freq (cyc/step)')
    picked, kept, shown, _ = typed_value(v, 'main', 'beam angle (deg)', '37.5')
    v.step(8)
    th_ok = abs(p.beam_angle - 37.5) < 1e-12 and abs(m.beam['info']['theta_k'] - 37.5) < 2
    typed_value(v, 'main', 'beam width w (cells)', '12')
    v.step(8)
    w_ok = p.beam_width == 12.0 and m.beam_key == m._beam_key() and m.step_n == 16
    check('beam: angle/width/freq sliders in the main panel take typed values; the beam follows live (no reset)',
          all(k in sl for k in want) and picked and kept and th_ok and w_ok,
          f'sliders {[k for k in want if k in sl]}, angle {p.beam_angle}, theta_k {m.beam["info"]["theta_k"]:.2f}, '
          f'width {p.beam_width}, step {m.step_n}')
    segs = v.beam_segments()
    v.show_emitter = False
    off = v.beam_segments() == []
    v.show_emitter = True
    errs = []
    try:
        for v3 in (False, True):
            v.view3d = v3
            frame(v, ctx, os.path.join(OUT, f'beam_overlay_{"3d" if v3 else "2d"}_MIRROR.png'))
    except Exception as ex:
        errs.append(f'{type(ex).__name__}: {ex}')
    v.view3d = False
    check('beam: emitter overlay = aperture + arrow (4 segments), toggles off, renders in 2D and 3D',
          len(segs) == 4 and off and not errs, f'{len(segs)} segs, off {off}, {errs}')
    fresh(v, src='packet', pk_len=10.0, pk_lambda=16.0)
    sl = gpass(v)['main'].sliders
    typed_value(v, 'main', 'packet wavelength (cells)', '20')
    pk_ok = (p.pk_lambda == 20.0 and m.packet_info['lam'] == 20.0 and m.step_n == 0
             and 'packet length L (cells)' in sl and 'freq (cyc/step)' not in sl)
    check('beam: packet panel (length, wavelength; no freq); a typed wavelength relaunches the packet', pk_ok,
          f'lambda {p.pk_lambda}, sliders {sorted(k for k in sl if "packet" in k or "beam" in k)}')
    fresh(v, src='pulse', beam_angle=0.0, beam_width=8.0, pk_lambda=16.0, freq=0.12)


def t_scale(v, ctx):
    """Scale mode 02/10/2026 (NotesCLAUDE/scale_MIRROR.md): the button converts without a jump, fraction
    sliders, N changes keep the fractions, crossings / s playback, warnings, auto height."""
    fresh(v, lattice='sq', N=257, shape='square', src='pulse', sx=0.0, sy=0.0, damp=0.0, Co=0.5, sigma=3.0,
          freq=0.12, scale='cells')
    m, p = v.m, v.m.p
    v.sub, v.panel = 4, True
    v.show.update(main=True)
    sl = gpass(v)['main'].sliders
    dflt = (p.scale == 'cells' and not v.auto_height and v.frame_steps() == 4 and 'steps / frame' in sl
            and 'pulse sigma' in sl and 'sigma / r_in' not in sl)
    check('scale: cells mode is the default (steps / frame, sigma in cells, no auto height)', dflt)
    v.step(40)
    u0 = m.level(1)
    gpass(v, press={'main': ['scale: cells']})
    r = m.inner_R()
    sl = gpass(v)['main'].sliders
    ok = (p.scale == 'domain' and p.sigma == 3.0 and m.step_n == 40 and np.array_equal(u0, m.level(1))
          and abs(p.sigma_f - 3.0 / r) < 1e-15 and abs(v.cps - 4 * 0.5 * app.FPS_NOMINAL / r) < 1e-12
          and 'sigma / r_in' in sl and 'crossings / s' in sl and 'steps / frame' not in sl)
    check('scale: the button enters domain mode without a jump (sizes, field, speed converted)', ok,
          f'sigma_f {p.sigma_f:.5f}, {v.cps:.4f} crossings/s, sliders {[k for k in sl if "r_in" in k or "/ s" in k]}')
    gpass(v, picks={'main': {'sigma / r_in': 0.05}})
    s_ok = abs(p.sigma - 0.05 * r) < 1e-6 and m.step_n == 0
    v.n_slider(513, 257, now=0.0)
    v.n_slider(513, 513, now=1.0)
    r2 = m.inner_R()
    n_ok = m.N == 513 and abs(p.sigma / r2 - p.sigma_f) < 1e-15
    check('scale: a sigma fraction slider resets with sigma = f r_in; N 257 -> 513 keeps the fraction', s_ok and n_ok,
          f'sigma {p.sigma:.3f} cells at N {m.N} (r_in {r2})')
    v.cps, v._step_acc = 0.25, 0.0
    tot513 = sum(v.frame_steps() for _ in range(60))
    p.N = 1025
    v.reset()
    v._step_acc = 0.0
    tot1025 = sum(v.frame_steps() for _ in range(60))
    want = [0.25 * rr / 0.5 for rr in (256, 512)]                # crossings/s x r_in / Co = steps per second
    check('scale: crossings / s playback, steps per second follow N (fractional steps carried)',
          abs(tot513 - want[0]) <= 1 and abs(tot1025 - want[1]) <= 1, f'60 frames: {tot513} / {tot1025} steps, want {want}')
    p.N, p.sigma_f = 49, 0.01
    v.reset()
    txt = gpass(v)['main'].texts
    warn = [t for t in txt if 'cells < 2' in t]
    check('scale: an under-resolved fraction shows an amber warning', bool(warn), f'{warn[:1]}')
    p.N, p.sigma_f = 257, 0.05
    v.reset()
    v.auto_height, v.hs = True, 1.0
    h0 = v.height_scale()
    v.step(256)
    h1 = v.height_scale()
    errs = []
    try:
        v.view3d = True
        frame(v, ctx, os.path.join(OUT, 'scale_domain_autoheight_3d_MIRROR.png'))
    except Exception as ex:
        errs.append(f'{type(ex).__name__}: {ex}')
    check('scale: auto height = height scale / recent peak (pulse start 1x, ring after 1 crossing larger); renders',
          abs(h0 - 1.0) < 1e-12 and h1 > 3 and abs(h1 - 1.0 / v.peak) < 1e-12 and not errs,
          f'{h0:.3f} -> {h1:.3f}, {errs}')
    gpass(v, press={'main': ['scale: domain']})
    back = p.scale == 'cells' and v.sub == max(1, round(0.25 * m.inner_R() / (0.5 * app.FPS_NOMINAL)))
    check('scale: back to cells mode (steps / frame from the speed)', back, f'sub {v.sub}')
    v.auto_height, v.sub, v.view3d = False, 4, False
    fresh(v, N=257, sigma=3.0, freq=0.12, scale='cells', sigma_f=None, lam_f=None, bw_f=None, pkl_f=None, pklam_f=None)



# ---- 11. round 4 / v1.0 (05/10/2026, Claude Opus 5.5) ---------------------------------------------
def _texts(w):
    """Every text line of a panel joined with single spaces (wrapped lines read as one)."""
    return ' '.join(t.strip() for t in w.texts)


def _col_of(w, piece):
    """The colour of the first panel text containing `piece` (None: normal text; 'missing')."""
    for t, c in zip(w.texts, w.tcols):
        if piece in t:
            return c
    return 'missing'


def _press(v, key):
    v.handle(ain.Frame((0.5, 0.5), set(), [key]))


def panel_rects(win):
    """The panels' default rectangles in pixels (x0, y0, x1, y1) for a window size."""
    W, H = win
    return {k: (x * W, y * H, (x + w) * W, (y + h) * H) for k, (x, y, w, h) in app.PANELS.items()}


def overlap_area(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def t_round4(v, ctx):
    m, p = v.m, v.m.p
    v.panel = True
    v.show.update(main=True, solids=True, align=False, ledger=False)
    v.view3d = False
    # (a) the version: one constant in the core, the title, the panel header, --version, the shot log line
    ws = gpass(v)
    head = ws['main'].texts[0] if ws['main'].texts else ''
    r1 = subprocess.run([sys.executable, os.path.join(HERE, 'membrane_app_MIRROR.py'), '--version'],
                        capture_output=True, text=True, timeout=180)
    shot = os.path.join(OUT4, 'r4_version_shot_MIRROR.png')
    r2 = subprocess.run([sys.executable, os.path.join(HERE, 'membrane_app_MIRROR.py'), '--shot', shot, '--N', '65',
                         '--steps', '2', '--view', '2d', '--arch', ARCH]
                        + (['--f32'] if F32 else []), capture_output=True, text=True, timeout=300)
    log = [ln for ln in r2.stdout.splitlines() if ln.startswith('wrote ')]
    check('v1.0: VERSION 1.0 in the core, the window title, the main panel header, --version, the --shot log line',
          mc.VERSION == '1.0' and app.VERSION == mc.VERSION and app.TITLE == 'ZRL wave membrane v1.0'
          and head.startswith('ZRL wave membrane v1.0')
          and r1.stdout.strip().splitlines()[-1:] == [f'ZRL wave membrane v1.0 ({mc.VERSION_DATE})']   # after Taichi's banner
          and bool(log) and 'ZRL wave membrane v1.0' in log[0],
          f'header {head!r}, --version {r1.stdout.strip()!r}, log {log[:1]}')
    # (b) the boundary control: B cycles the bcs that act here
    fresh(v, lattice='sq', N=129, shape='square', bc='dirichlet', src='pulse', sx=0.0, sy=0.0, rad=1.0, edge='pinned')
    seq_sq = []
    for _ in range(4):
        _press(v, 'b')
        seq_sq.append(p.bc)
    fresh(v, lattice='tri', N=129, shape='array', bc='dirichlet')
    seq_tri = []
    for _ in range(4):
        _press(v, 'b')
        seq_tri.append(p.bc)
    # 10/10/2026 (absorbing boundaries in the core): the square lattice's full square cycles mur2, cpml and sponge
    # too (mc.BCS order); the triangular array has the sponge (Mur, Mur2 and CPML skipped)
    check('boundary: B cycles every bc on the full square (dirichlet -> mur -> mur2 -> cpml -> sponge); on the '
          'triangular array dirichlet -> sponge -> neumann -> periodic (Mur, Mur2, CPML skipped)',
          seq_sq == ['mur', 'mur2', 'cpml', 'sponge']
          and seq_tri == ['sponge', 'neumann', 'periodic', 'dirichlet'], f'sq {seq_sq}, tri {seq_tri}')
    ring_ok = []
    for bc, want in (('dirichlet', 'clamped ring'), ('neumann', 'rigid ring (link removal'),
                     ('periodic', 'rhombic torus')):
        fresh(v, bc=bc)
        txt = _texts(gpass(v)['main'])
        ring_ok.append(f'boundary: {bc}' in txt and want in txt and 'mur: not on tri yet' in txt)
    fresh(v, bc='mur')
    w = gpass(v)['main']
    txt = _texts(w)
    mur_ok = ('boundary: dirichlet   (mur n/a here)' in txt and 'falls back to the clamped ring' in txt
              and _col_of(w, 'falls back') == app.AMBER and m.bc_info()['fallback'])
    _press(v, 'b')
    mur_next = p.bc
    check('boundary on the triangular array: the effective bc and the ring in words; Mur = "not on tri yet '
          '(absorbing-boundary round)", amber with the fallback; B from mur -> sponge (10/10/2026: the sponge acts '
          'there)', all(ring_ok) and mur_ok and mur_next == 'sponge', f'rings {ring_ok}, mur {mur_ok}, next {mur_next}')
    fresh(v, lattice='tri', shape='n6', bc='dirichlet', rad=0.8)
    v.flash_msg = ''
    _press(v, 'b')
    unchanged = p.bc == 'dirichlet' and 'does not apply' in v.flash_msg
    p.bc = 'periodic'
    v.reset()
    w = gpass(v)['main']
    txt = _texts(w)
    masked_ok = ('boundary: dirichlet   (periodic n/a here)' in txt and 'the bc does not apply to this shape' in txt
                 and any('does not apply to this shape' in b for b in w.buttons) and 'edge: pinned' in ' '.join(w.buttons))
    check('boundary on a triangular web-app shape: B changes nothing and says so; the panel says the bc does not '
          'apply (use the edge control) and offers the edge', unchanged and masked_ok, f'{unchanged} {masked_ok}')
    # (c) the domain edge: set_edge keeps the field; greyed on 'array' and the full square
    fresh(v, lattice='sq', N=129, shape='n6', bc='dirichlet', edge='pinned', rad=1.0, src='pulse')
    v.step(30)
    u0, n0 = m.level(1).copy(), m.step_n
    gpass(v, press={'main': ['##edge']})
    e1 = (p.edge == 'rigid', m.step_n == n0, np.array_equal(m.level(1), u0), m.walls_active())
    v.step(5)
    gpass(v, press={'solids': ['##edge']})                  # the same control in the walls panel
    e2 = p.edge == 'pinned' and not m.walls_active()
    fresh(v, shape='array')
    w = gpass(v)['main']
    e3 = 'edge: n/a - the array' in _texts(w) and not any('##edge' in b for b in w.buttons)
    fresh(v, shape='square', rad=1.0)
    w = gpass(v)['main']
    e4 = 'edge: n/a - the full square' in _texts(w).replace('\n', ' ') and not any('##edge' in b for b in w.buttons)
    check('edge: the button switches pinned <-> rigid WITHOUT a reset (field and step kept, walls kernel on/off), '
          'in both panels; greyed text on the array and the full square', all(e1) and e2 and e3 and e4,
          f'{e1} {e2} {e3} {e4}')
    # (d) 'array' in the G cycle; radius and rotation greyed there
    fresh(v, lattice='tri', shape='unlicensed', rad=0.89, sx=0.0, sy=-0.9)
    v.view3d, v.panel = False, True
    v.show.update(main=True, solids=False, align=False, ledger=False)
    v.v2.reset()
    _press(v, 'g')
    g1 = p.shape
    # the 2D view fits the whole rhombus into the strip the main panel leaves (05/10/2026)
    W, H = v.win
    v.v2.ox = v.free_centre()
    xs = [v.v2.phys_to_screen(m, x, y, W, H)[0] for x, y in m.shape_poly()]
    ys = [v.v2.phys_to_screen(m, x, y, W, H)[1] for x, y in m.shape_poly()]
    fit = (v.v2.zoom < 1.0 and min(xs) >= app.PANELS['main'][2] - 1e-9 and max(xs) <= 1.0 + 1e-9
           and min(ys) >= 0.0 and max(ys) <= 1.0)
    v.v2.zoom = 2.0
    _press(v, 'c')
    refit = abs(v.v2.zoom - v._fit_zoom) < 1e-12
    fz = v._fit_zoom
    w = gpass(v)['main']
    grey = ('radius' not in w.sliders and 'rotation (deg)' not in w.sliders and 'radius: fixed (the whole array)' in w.texts
            and 'rotation: fixed (the lattice array)' in w.texts and not any('rotation 30' in b for b in w.buttons))
    _press(v, 'g')
    g2 = p.shape
    w = gpass(v)['main']
    back = 'radius' in w.sliders and 'rotation (deg)' in w.sliders and v.v2.zoom == 1.0
    check('shape: G goes unlicensed -> array -> square; on array the radius / rotation sliders are greyed text; '
          'the 2D view fits the triangular rhombus into the panel-free strip (C refits; leaving restores zoom 1)',
          g1 == 'array' and g2 == 'square' and grey and back and fit and refit,
          f'{g1} {g2} grey {grey} back {back}, fit {fit} (zoom {fz:.3f}, x {min(xs):.3f}..{max(xs):.3f}), refit {refit}')
    # (d2) the 3D view (06/10/2026): entering the triangular array moves the orbit camera back until the
    # rhombus reaches no further from the window centre than the square does at the default distance;
    # C refits; leaving restores the default distance unless the user zoomed since; other shapes untouched
    def reach3d(pts):
        pose, asp = v.orbit.pose(), v.aspect()
        s = [pose.world_to_screen((X, 0.0, Z), asp) for X, Z in pts]
        return max(abs(2 * q[0] - 1) for q in s), max(abs(2 * q[1] - 1) for q in s)
    sq_pts = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    d_def = acam.OrbitCam.DEFAULT['dist']
    r3 = {}
    for az, el in ((0.0, 38.0), (45.0, 38.0), (-120.0, 25.0)):
        fresh(v, lattice='tri', shape='unlicensed', rad=0.89, sx=0.0, sy=-0.9)
        v.view3d = True
        v.orbit.reset()
        v.orbit.az, v.orbit.el = az, el
        rs = reach3d(sq_pts)
        _press(v, 'g')                                      # unlicensed -> array
        rt = reach3d([acam.phys_to_world(m, x, y) for x, y in m.shape_poly()])
        r3[(az, el)] = (v.orbit.dist, rs, rt, v.orbit.az == az and v.orbit.el == el)
    d1 = r3[(0.0, 38.0)][0]
    v.orbit.zoom(2.0)
    _press(v, 'c')
    refit3 = abs(v.orbit.dist - d1) < 1e-12 and (v.orbit.az, v.orbit.el) == (0.0, 38.0)
    _press(v, 'g')                                          # array -> square: back to the default distance
    left3 = p.shape == 'square' and v.orbit.dist == d_def
    v.set_shape('array')
    v.orbit.zoom(1.25)
    dz = v.orbit.dist
    v.set_shape('square')                                   # zoomed since: the user's distance stays
    kept3 = v.orbit.dist == dz
    _press(v, 'c')
    other3 = v.orbit.dist == d_def                          # C on the square: the plain default
    fit3 = all(d > d_def and ok and rt[0] <= rs[0] + 1e-9 and rt[1] <= rs[1] + 1e-9
               for d, rs, rt, ok in r3.values())
    v.view3d = False
    check('3D view: the triangular array moves the orbit camera back until the rhombus reaches no further from '
          'the window centre than the square does at the default distance (angles kept); C refits; leaving '
          'restores the default distance unless zoomed since',
          fit3 and refit3 and left3 and kept3 and other3,
          '; '.join(f'az {k[0]:.0f} el {k[1]:.0f}: dist {d:.3f}, reach x {rt[0]:.3f} <= {rs[0]:.3f}, '
                    f'y {rt[1]:.3f} <= {rs[1]:.3f}' for k, (d, rs, rt, ok) in r3.items())
          + f'; refit {refit3}, left {left3}, kept {kept3}, square C {other3}')
    v.show.update(main=True, solids=True)
    # (e) wall types: brush, object, plate lettering; legend; counts (and their wait); scene round trip
    fresh(v, lattice='sq', N=257, shape='square', bc='dirichlet', edge='pinned', src='pulse', rad=1.0, sx=0.0, sy=0.0)
    m.attach_scene(so.SolidScene())
    v.sol.brush_wall = 'pinned'
    gpass(v, press={'solids': ['##brushwall']})
    bw = v.sol.brush_wall
    v.view3d = False
    v.v2.reset()
    app.programmatic_stroke(v, [(-0.6, -0.5), (-0.3, -0.5)])
    st = m.scene.strokes[-1]
    n_rig_stroke = int(np.count_nonzero(m.solid_np == 2))
    v.sol.add(m, 'rect')
    o = m.scene.objects[v.sol.sel]
    n_pin = int(np.count_nonzero(m.solid_np == 1))
    gpass(v, press={'solids': ['##objwall']})
    n_rig = int(np.count_nonzero(m.solid_np == 2))
    tag = 'RIGID' in asol.describe(o)
    check('wall types: the brush toggle paints rigid strokes; the object toggle turns a pinned rect rigid (cells '
          'become type 2); the object list tags it RIGID', bw == 'rigid' and st.wall == 'rigid' and n_rig_stroke > 0
          and n_pin > 0 and o.wall == 'rigid' and n_rig == n_rig_stroke + n_pin and tag,
          f'brush {bw}, stroke {st.wall} ({n_rig_stroke} cells), rect {n_pin} pinned -> {o.wall}, rigid {n_rig}')
    v.sol.edit_t = v.sol.edit_prev = -1e9
    w = gpass(v)['solids']
    dom = m.domain_mask_np != 0
    want_pin = int(np.count_nonzero((m.solid_np == 1) & dom))
    want_rig = int(np.count_nonzero((m.solid_np == 2) & dom))
    leg = (_col_of(w, 'pinned solid (u = 0)') == tuple(v.look.solid) and _col_of(w, 'rigid solid (du/dn = 0)') ==
           tuple(v.look.rigid) and f'pinned cells {want_pin:,}   rigid cells {want_rig:,}' in w.texts
           and 'walls kernel ON' in _texts(w) and 'reflects inverted' in _texts(w).replace('\n', ' ')
           and 'water off a' in _texts(w))
    # the counts wait for an edit to rest (no whole-grid count per frame while dragging / painting)
    c0 = v.walls_counts()
    v.sol.edit(m, wall='pinned')                            # two edits in a row: a drag
    v.sol.edit(m, enabled=True)                             # (no geometry change)
    c1 = v.walls_counts(now=v.sol.edit_t + 0.01)
    c2 = v.walls_counts(now=v.sol.edit_t + app.EDIT_SETTLE + 0.01)
    wait_ok = c1 is c0 and c2['rigid_cells'] == want_rig - n_pin and c2['pinned_cells'] == want_pin + n_pin
    check('walls panel: legend in the pinned / rigid colours, the explainer, pinned / rigid cell counts, walls '
          'kernel ON; the counts wait EDIT_SETTLE s after an edit', leg and wait_ok,
          f'legend {leg}, counts {want_pin}/{want_rig}, stale {c1 is c0}, after {c2["pinned_cells"]}/{c2["rigid_cells"]}')
    v.sol.edit(m, wall='rigid')
    # the plate lettering
    v.set_shape('unlicensed')
    n_plate = m.n_solid
    gpass(v, press={'solids': ['##platewall']})
    pw = m.scene.plate_wall
    plate_rig = int(np.count_nonzero(m.solid_np == 2))
    # scene round trip with the wall types
    path = v.sol.save(m, os.path.join(OUT4, 'r4_scene_roundtrip_MIRROR.json'))
    before = m.solid_np.copy()
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    v.sol.load(m, path)
    sc2 = m.scene
    rt = (d.get('version') == 2 and d.get('plate_wall') == 'rigid' and [q['wall'] for q in d['objects']] == ['rigid']
          and [q['wall'] for q in d['strokes']] == ['rigid'] and sc2.plate_wall == 'rigid'
          and sc2.objects[0].wall == 'rigid' and sc2.strokes[0].wall == 'rigid' and np.array_equal(m.solid_np, before))
    check('wall types: the plate lettering toggle makes the lettering and screws rigid (same cells); a scene saved '
          'and loaded keeps every wall type (format 2) and the same type map', pw == 'rigid' and m.n_solid == n_plate
          and plate_rig == m.n_solid and rt, f'plate {pw}, rigid {plate_rig:,} of {m.n_solid:,}, round trip {rt}')
    # (f) rigid vs pinned pixels, 2D and 3D; the edge overlay
    fresh(v, lattice='sq', N=257, shape='square', src='pulse', bc='dirichlet', edge='pinned', rad=1.0, sx=0.0, sy=0.0)
    sc = so.SolidScene()
    sc.add(so.SolidObject('rect', x=-0.45, y=0.0, w=0.2, h=0.5, wall='pinned'))
    sc.add(so.SolidObject('rect', x=0.45, y=0.0, w=0.2, h=0.5, wall='rigid'))
    m.attach_scene(sc)
    v.panel = False
    v.look = ar.Look(shading=0.0)
    W, H = v.win
    v.view3d = False
    v.v2 = acam.View2D()
    img = frame(v, ctx, os.path.join(OUT4, 'r4_rigid_pinned_2d_MIRROR.png'), gui=False).astype(float)
    pts = [acam.grid_to_phys(m, gx, 0.0) for gx in (-0.45, 0.45)]

    def patch_px(img, sx, sy, r=6):
        Hh, Ww = img.shape[:2]
        cx, cy = int(sx * Ww), int((1 - sy) * Hh)
        return img[cy - r:cy + r + 1, cx - r:cx + r + 1].reshape(-1, 3)
    pin2 = patch_px(img, *v.v2.phys_to_screen(m, *pts[0], W, H))
    rig2 = patch_px(img, *v.v2.phys_to_screen(m, *pts[1], W, H))
    sol, rig = np.array(v.look.solid) * 255, np.array(v.look.rigid) * 255
    lum = lambda P: P @ np.array([0.2126, 0.7152, 0.0722])
    stripes = int(np.sum(np.abs(rig2 - 0.55 * rig).max(1) <= 3))
    ok2 = (np.abs(np.median(pin2, 0) - sol).max() <= 2 and np.abs(rig2.max(0) - rig).max() <= 2
           and 0.15 < stripes / len(rig2) < 0.35)
    v.view3d = True
    v.orbit.az, v.orbit.el, v.orbit.dist = 0.0, 89.0, 2.4
    img3 = frame(v, ctx, os.path.join(OUT4, 'r4_rigid_pinned_3d_MIRROR.png'), gui=False).astype(float)
    s3 = []
    for x, y in pts:
        X, Z = acam.phys_to_world(m, x, y)
        s3.append(v.pose().world_to_screen((X, v.look.solid_h, Z), W / H))
    pin3, rig3 = np.median(patch_px(img3, *s3[0]), 0), np.median(patch_px(img3, *s3[1]), 0)
    ok3 = np.abs(pin3 - sol).max() <= 3 and np.abs(rig3 - rig).max() <= 3
    dl = float(lum(sol) - lum(rig))
    check('rendering: pinned solids in the solid colour, rigid ones in the rigid colour (2D: with ~25% dark '
          'stripes), 2D and 3D; the two differ in lightness too', ok2 and ok3 and dl > 20,
          f'2D pinned {np.median(pin2, 0).round(0)}, rigid max {rig2.max(0).round(0)}, stripes {stripes}/{len(rig2)}; '
          f'3D pinned {pin3.round(0)} rigid {rig3.round(0)}; luma difference {dl:.0f}/255')
    m.attach_scene(so.SolidScene())
    segs = {}
    for lab, kw in (('pinned edge', dict(lattice='sq', shape='n6', edge='pinned', bc='dirichlet')),
                    ('rigid edge', dict(lattice='sq', shape='n6', edge='rigid', bc='dirichlet')),
                    ('tri rigid ring', dict(lattice='tri', shape='array', edge='pinned', bc='neumann')),
                    ('tri torus', dict(lattice='tri', shape='array', bc='periodic')),
                    ('tri clamped', dict(lattice='tri', shape='array', bc='dirichlet')),
                    ('sq torus', dict(lattice='sq', shape='square', rad=1.0, bc='periodic'))):
        fresh(v, **kw)
        segs[lab] = v.boundary_segments()
    v.show_edge = False
    off = v.boundary_segments() == []
    v.show_edge = True
    cols = lambda S: {c for _, _, c in S}
    ov = (segs['pinned edge'] == [] and len(segs['rigid edge']) == 6 and cols(segs['rigid edge']) == {tuple(v.look.rigid)}
          and len(segs['tri rigid ring']) == 4 and segs['tri clamped'] == [] and len(segs['tri torus']) > 20
          and cols(segs['tri torus']) == {ar.TORUS_COL} and len(segs['sq torus']) > 20 and off)
    errs = []
    try:
        fresh(v, lattice='tri', shape='array', bc='periodic')
        for v3 in (False, True):
            v.view3d = v3
            frame(v, ctx, os.path.join(OUT4, f'r4_torus_overlay_{"3d" if v3 else "2d"}_MIRROR.png'), gui=False)
    except Exception as ex:
        errs.append(f'{type(ex).__name__}: {ex}')
    check('edge overlay: none for a pinned edge or clamped ring; a rigid-colour line along a rigid edge (6 sides of '
          'n6) and the rigid ring; dashed violet torus seams; off with the toggle; renders in 2D and 3D',
          ov and not errs, f'{ {k: len(s_) for k, s_ in segs.items()} }, off {off}, {errs}')
    v.look = ar.Look()
    v.view3d = False
    v.panel = True
    # (g) the vertex sources and the drive: inset labels, the bump checkbox, the source note, warnings, phase lag
    fresh(v, lattice='sq', N=257, shape='n3', src='vtx', vtx_inset='wall', vtx_drive=False, vtx_drive_bump=False,
          sigma=3.0, rad=1.0, bc='dirichlet', edge='pinned', Co=0.5, freq=0.12, scale='cells')
    w = gpass(v)['main']
    g1 = (any(b.startswith('vertex inset: wall (3 sigma from both walls)') for b in w.buttons)
          and '3 vertex sources' in _texts(w) and 'seed pulses under the driven vertices (the web app)' not in w.checks
          and 'inset 3 sigma from BOTH walls' in _texts(w) and not any('clipped at t = 0' in t for t in w.texts))
    gpass(v, press={'main': ['vertex inset: wall']})
    w = gpass(v)['main']
    txt = _texts(w).replace('\n   ', ' ')
    g2 = (p.vtx_inset == 'radial' and any(b.startswith("vertex inset: radial (the web app's)") for b in w.buttons)
          and 'try vertex inset: wall' in txt and _col_of(w, 'pulse cut off at t = 0') == app.AMBER
          and 'phase lag ~' in txt)
    p.vtx_drive = True
    v.reset()
    w = gpass(v)['main']
    txt = _texts(w).replace('\n   ', ' ')
    g3 = ('seed pulses under the driven vertices (the web app)' in w.checks and 'start from rest' in txt
          and 'phase lag' not in txt and 'driven vertices: lambda 4.2 cells < 8' in txt
          and _col_of(w, 'driven vertices: lambda') == app.AMBER)
    v.step(3)
    gpass(v, checks={'main': {'seed pulses under the driven vertices (the web app)': True}})
    w = gpass(v)['main']
    g4 = p.vtx_drive_bump is True and m.step_n == 0 and 'start from rest' not in _texts(w) and 'phase lag ~' in _texts(w)
    fresh(v, src='cont', shape='square', vtx_drive=False, vtx_drive_bump=False, vtx_inset='wall', freq=0.12)
    t_cont = _texts(gpass(v)['main'])
    fresh(v, freq=0.03)
    t_cont2 = _texts(gpass(v)['main'])
    g5 = 'continuous source: lambda 4.2 cells < 8' in t_cont and 'continuous source: lambda' not in t_cont2
    fresh(v, src='pulse', freq=0.12)
    w = gpass(v)['main']
    g6 = 'phase lag ~' in _texts(w) and 'source cell (' in _texts(w) and 'pulse cut off at t = 0' in _texts(w)
    check('vertex sources: inset labels and toggle, the bump checkbox only while driven (it resets), the source note '
          '(clearance, the amber clip with the hint on radial), the drive-wavelength warning (amber), the phase lag '
          'while a gaussian is seeded', g1 and g2 and g3 and g4 and g5 and g6, f'{g1} {g2} {g3} {g4} {g5} {g6}')
    # (h) panel layout: the default rectangles never overlap, at the default size and maximised
    bad = []
    for win in (app.WIN, (2560, 1377), (1280, 720)):
        R = panel_rects(win)
        ks = list(R)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a = overlap_area(R[ks[i]], R[ks[j]])
                if a > 0:
                    bad.append((win, ks[i], ks[j], a))
            x0, y0, x1, y1 = R[ks[i]]
            if x0 < 0 or y0 < 0 or x1 > win[0] + 1e-9 or y1 > win[1] + 1e-9:
                bad.append((win, ks[i], 'outside'))
    errs = []
    try:
        fresh(v, lattice='sq', N=257, shape='n6', src='vtx', edge='rigid', vtx_inset='radial')
        v.show.update(main=True, solids=True, align=True, ledger=True)
        m.attach_scene(so.SolidScene())
        v.sol.add(m, 'rect')
        v.sol.edit(m, wall='rigid')
        for v3 in (False, True):
            v.view3d = v3
            frame(v, ctx)
            frame(v, ctx, os.path.join(OUT4, f'r4_all_panels_{"3d" if v3 else "2d"}_MIRROR.png'))
        fresh(v, lattice='tri', shape='array', bc='mur', src='mode', edge='pinned', vtx_inset='wall')
        frame(v, ctx, os.path.join(OUT4, 'r4_all_panels_tri_mur_MIRROR.png'))
    except Exception as ex:
        import traceback
        traceback.print_exc()
        errs.append(f'{type(ex).__name__}: {ex}')
    rows = {k: w_.rows for k, w_ in gpass(v).items()}
    check('panels: the four default rectangles never overlap and stay inside the window (1600x1000, maximised '
          '2560x1377, 1280x720); every panel renders in a real GGUI frame with the round-4 controls',
          not bad and not errs, f'{bad or "no overlap"}, rows per panel {rows}, {errs}')
    v.show.update(main=True, solids=False, align=False, ledger=False)
    m.attach_scene(so.SolidScene())
    fresh(v, lattice='sq', N=257, shape='circle', src='pulse', bc='dirichlet', edge='pinned', vtx_inset='wall',
          sx=0.0, sy=0.0, rad=1.0)


# ---- 10. palettes and transparency (01/10/2026, Claude Opus 5.5) --------------------------------
CUSTOM5 = [(-1.0, (0.1, 0.5, 1.0)), (-0.4, (0.2, 0.2, 0.6)), (0.0, (0.12, 0.1, 0.16)), (0.4, (0.7, 0.2, 0.6)),
           (1.0, (1.0, 0.85, 0.3))]


def _patch(img, sx, sy, r=3):
    """Median colour of a (2r+1)^2 patch around screen point (sx, sy) in [0,1] (y up)."""
    H, W = img.shape[:2]
    cx, cy = int(sx * W), int((1 - sy) * H)
    return np.median(img[cy - r:cy + r + 1, cx - r:cx + r + 1].reshape(-1, 3), axis=0)


def t_visuals(v, ctx, quick):
    p0 = v.m.p
    keep = (p0.N, p0.shape, p0.src, v.view3d, v.panel, v.orbit.az, v.orbit.el, v.orbit.dist, dict(v.show))
    xs = np.linspace(-1, 1, 801)
    # (a) the default palette is the 30/09 formula Z + (end - Z)|x|^(1/contrast), x exposure
    worst = 0.0
    for z, e, c in ((0.22, 1.0, 1.0), (0.05, 1.3, 0.6), (0.4, 0.7, 2.5)):
        L = ar.Look(zero=z, exposure=e, contrast=c)
        a = np.abs(xs) ** (1 / c)
        end = np.where(xs[:, None] >= 0, np.array(ar.POS_END), np.array(ar.NEG_END))
        ref = np.minimum((z + (end - z) * a[:, None]) * e, 1.0)
        worst = max(worst, float(np.abs(ar.palette_np(xs, L) - ref).max()))
    check('palette: default teal-red == the 30/09 formula (zero, exposure, contrast varied, host)', worst < 1e-12,
          f'max err {worst:.1e}')
    # (b) every preset + custom: kernel == host; zero level; stops hit exactly; cyclic closes
    vals = ti.ndarray(ti.f32, shape=xs.size)
    vals.from_numpy(xs.astype(np.float32))
    out = ti.ndarray(ti.f32, shape=(xs.size, 3))
    kerr, zerr, serr = 0.0, 0.0, 0.0
    for name in ar.PALETTE_ORDER:
        for e, c in ((1.0, 1.0), (1.3, 0.7), (0.8, 2.0)):
            L = ar.Look(exposure=e, contrast=c)
            if name == 'custom':
                L.custom = list(CUSTOM5)
            L.set_palette(name)
            v.R.set_params(L, 1.0, 1.0)
            k_pal(vals, v.R.LP, out)
            kerr = max(kerr, float(np.abs(out.to_numpy() - ar.palette_np(xs, L)).max()))
        L = ar.Look()
        if name == 'custom':
            L.custom = list(CUSTOM5)
        L.set_palette(name)
        zerr = max(zerr, abs(max(ar.palette_np(0.0, L)) - L.zero))
        for zl in (0.1, 0.35):                      # the zero slider scales the zero stop, hue kept
            L.zero = zl
            zerr = max(zerr, abs(max(ar.palette_np(0.0, L)) - zl))
        L.zero = ar.zero_of(L.stops_raw())
        for t, col in L.stops():                    # contrast 1: the stop positions show the stop colours
            serr = max(serr, float(np.abs(ar.palette_np(t, L) - np.minimum(col, 1)).max()))
    check('palette: kernel == host for all 6 palettes x 3 (exposure, contrast)', kerr < 2e-6, f'max err {kerr:.1e} (f32)')
    check('palette: zero level = brightness of the zero stop (preset own level, slider rescales, hue kept)',
          zerr < 1e-12, f'max err {zerr:.1e}')
    check('palette: the colour at each stop position is the stop colour (all palettes incl. custom)', serr < 1e-12,
          f'max err {serr:.1e}')
    Lc = ar.Look()
    Lc.set_palette('cyclic')
    cyc = float(np.abs(ar.palette_np(-1.0, Lc) - ar.palette_np(1.0, Lc)).max())
    cs = ar.clean_stops([(0.5, (2, -1, 0.5)), (-2.0, (0.1, 0.1, 0.1)), (0.0, (0.2, 0.2, 0.2))]
                        + [(0.01 * k, (0.5, 0.5, 0.5)) for k in range(1, 30)])
    cs_ok = (len(cs) == ar.KMAX and cs[0][0] == -1.0 and cs[-1][0] == 1.0 and any(t == 0.0 for t, _ in cs)
             and all(cs[k][0] < cs[k + 1][0] for k in range(len(cs) - 1))
             and all(0 <= x <= 1 for _, c in cs for x in c))
    check('palette: cyclic closes (x=-1 == x=+1); clean_stops sorts, clips, fills the ends, caps at KMAX',
          cyc == 0.0 and cs_ok, f'cyclic diff {cyc}, {len(cs)} stops')
    # (c) settings file: round trip, broken file, auto-save debounce
    path = os.path.join(OUT, 'look_settings_test_MIRROR.json')
    L = ar.Look(opacity=0.4, grid=0.3, grid_every=5, grid_col=(0.2, 0.9, 0.4), grid_field=True, bg_rgb=(0.1, 0.0, 0.2))
    L.custom = list(CUSTOM5)
    L.set_palette('custom')
    L.zero = 0.3
    ar.save_settings(L, path)
    L2 = ar.Look()
    msg = ar.load_settings(L2, path)
    same = (msg is None and L2.palette == 'custom' and L2.zero == 0.3 and L2.opacity == 0.4 and L2.grid == 0.3
            and L2.grid_every == 5 and L2.grid_col == (0.2, 0.9, 0.4) and L2.grid_field and L2.bg_rgb == (0.1, 0.0, 0.2)
            and np.array_equal(ar.palette_np(xs, L2), ar.palette_np(xs, L)))
    with open(path, 'w') as f:
        f.write('{"look": {"palette": ')
    L3 = ar.Look()
    bad = ar.load_settings(L3, path)
    untouched = L3 == ar.Look()
    miss = ar.load_settings(ar.Look(), path + '.none')
    check('settings file: round trip (palette, custom stops, zero, opacity, grid, page colour); a broken or '
          'missing file keeps the defaults with a message', same and bad and untouched and miss,
          f'{msg}; broken -> {bad!r}')
    v.settings_path = path
    v._saved = (v._settings_text(), 0.0)
    v.look.custom = list(CUSTOM5)
    v.look.set_palette('custom')
    w1 = v.autosave(now=100.0)
    w2 = v.autosave(now=100.3)
    w3 = v.autosave(now=100.6)
    w4 = v.autosave(now=101.0)
    L4 = ar.Look()
    ar.load_settings(L4, path)
    check('settings auto-save: written once the change has rested 0.5 s, then not again; the file loads back',
          (w1, w2, w3, w4) == (False, False, True, False) and L4.stops() == v.look.stops(),
          f'{(w1, w2, w3, w4)}')
    v.settings_path, v._saved = None, None
    # (d) the palette editor (scripted GUI) and the real GGUI panel
    v.look = ar.Look()
    v.view3d = True
    v.pal_edit = True
    g = FakeG(press={'main': ['palette: teal-red']})
    v.gui(g)
    cyc_ok = v.look.palette == 'greyscale' and v.look.zero == ar.zero_of(ar.PALETTES['greyscale'])
    v.look = ar.Look()
    n0 = len(ar.clean_stops(v.look.custom))
    gpass(v, press={'main': ['add a stop (positive side)']})
    n1 = len(v.look.custom)
    add_ok = v.look.palette == 'custom' and n1 == n0 + 1
    mid = [t for t, _ in v.look.custom if 0 < t < 1]
    g = FakeG()
    g.ws['main'].colours = {f'zero (t +0.00)##pal{[t for t, _ in v.look.custom].index(0.0)}': (0.3, 0.1, 0.2)}
    v.gui(g)
    z_ok = v.look.zero == 0.3 and np.allclose(ar.palette_np(0.0, v.look), (0.3, 0.1, 0.2))
    gpass(v, picks={'main': {'zero level': 0.15}})
    z2 = np.allclose(ar.palette_np(0.0, v.look), (0.15, 0.05, 0.1))
    k_mid = [t for t, _ in v.look.custom].index(mid[0]) if mid else -1
    gpass(v, press={'main': [f'remove stop {k_mid}##']})
    rm_ok = len(v.look.custom) == n0
    ws = gpass(v)['main']
    sl3 = set(ws.sliders)
    v.view3d = False
    sl2 = set(gpass(v)['main'].sliders)
    v.view3d = True
    lab_ok = ('surface opacity' in sl3 and 'grid lines (opacity, 0 = off)' in sl3
              and 'field opacity (over the page)' in sl2 and 'grid lines (opacity, 0 = off)' not in sl2
              and 'page colour (outside the array)' in ws.colour_labels)
    picked, kept, shown, _ = typed_value(v, 'main', 'surface opacity', '0.25')
    typed_ok = v.look.opacity == 0.25
    check('palette editor: next preset, add / remove a stop, zero colour pick sets the zero level, the zero slider '
          'rescales it, opacity / grid / page colour controls per view, typed opacity',
          cyc_ok and add_ok and z_ok and z2 and rm_ok and lab_ok and typed_ok,
          f'cycle {cyc_ok}, add {add_ok} ({n0}->{n1}), zero {z_ok}/{z2}, remove {rm_ok}, labels {lab_ok}, typed {typed_ok}')
    errs = []
    try:
        v.look = ar.Look(opacity=0.5, grid=0.4)
        v.look.custom = list(CUSTOM5)
        v.look.set_palette('custom')
        for v3 in (True, False):
            v.view3d = v3
            frame(v, ctx, gui=True)
            frame(v, ctx, os.path.join(OUT, f'palette_editor_{"3d" if v3 else "2d"}_MIRROR.png'), gui=True)
    except Exception as ex:
        errs.append(f'{type(ex).__name__}: {ex}')
    check('palette editor and transparency controls drawn in the real GGUI panel, 3D and 2D', not errs, f'{errs}')
    v.pal_edit = False
    # (e) measured blending. Flat field far from the pulse, unlit, top view: pixel = a Z + (1 - a) page
    fresh(v, shape='circle', src='pulse')
    v.step(1)
    v.panel = False
    m = v.m
    v.v2 = acam.View2D()
    errs3, errs2 = [], []
    for a in (1.0, 0.5, 0.0):
        v.look = ar.Look(shading=0.0, opacity=a)
        want = (a * 0.22 + (1 - a) * np.array(v.look.bg_col())) * 255
        v.view3d = True
        v.orbit.az, v.orbit.el, v.orbit.dist = 0.0, 89.0, 2.2
        img = frame(v, ctx, os.path.join(OUT, f'alpha3d_{a:.1f}_MIRROR.png'), gui=False).astype(float)
        sp = v.pose().world_to_screen((0.5, 0.0, 0.0), v.win[0] / v.win[1])
        errs3.append(float(np.abs(_patch(img, *sp) - want).max()))
        v.view3d = False
        img = frame(v, ctx, os.path.join(OUT, f'alpha2d_{a:.1f}_MIRROR.png'), gui=False).astype(float)
        sp = v.v2.phys_to_screen(m, 0.5 * m.c(), 0.0, *v.win)
        errs2.append(float(np.abs(_patch(img, *sp) - want).max()))
    check('opacity: measured pixel = a * surface + (1 - a) * page at a = 1, 0.5, 0 (3D real alpha; 2D the same mix)',
          max(errs3 + errs2) <= 1.5, f'3D err {[round(e, 2) for e in errs3]}, 2D err {[round(e, 2) for e in errs2]} (8-bit)')
    # grid lines: line pixel = g * grid colour + (1 - g) * surface (the double-drawn edge compensated)
    res = []
    for gop in (0.3, 0.6, 1.0):
        v.look = ar.Look(shading=0.0, opacity=1.0, grid=gop, grid_every=4, grid_col=(1.0, 1.0, 1.0))
        v.view3d = True
        img = frame(v, ctx, os.path.join(OUT, f'grid_{gop:.1f}_MIRROR.png'), gui=False).astype(int)
        sp = v.pose().world_to_screen((0.5, 0.0, 0.0), v.win[0] / v.win[1])
        H, W = img.shape[:2]
        cx, cy = int(sp[0] * W), int((1 - sp[1]) * H)
        px = img[cy - 30:cy + 30, cx - 30:cx + 30, 0].reshape(-1)
        surf = 0.22 * 255
        line = px[px > surf + 10]
        u, c = np.unique(line, return_counts=True)
        got = float(u[np.argmax(c)]) if u.size else -1
        want = gop * 255 + (1 - gop) * surf
        res.append((gop, got, round(want, 1)))
    gerr = max(abs(g_ - w_) for _, g_, w_ in res)
    check('grid lines: measured line pixel = g * grid colour + (1 - g) * surface (g = 0.3, 0.6, 1)', gerr <= 2.0,
          f'(g, got, want) {res}')
    # (f) back-to-front order: floor first, then domain triangles far -> near along the view
    fresh(v, shape='circle', src='pulse')
    v.look = ar.Look(opacity=0.5)
    v.view3d = True
    corr, floor_ok = [], True
    for az, el in ((30.0, 24.0), (-120.0, 40.0), (175.0, 10.0), (75.0, -20.0)):
        v.orbit.az, v.orbit.el, v.orbit.dist = az, el, 2.4
        frame(v, ctx, gui=False)
        R = v.R
        idx = R.stats['order']
        nfl = R.T_floor.size
        floor_ok &= np.array_equal(idx[:nfl], R.T_floor)
        P = R.pos.to_numpy()
        tri = idx[nfl:].reshape(-1, 3)
        e = np.array(v.pose().eye)
        cx, cz = P[tri, 0].mean(1), P[tri, 2].mean(1)
        dist = np.hypot(cx - e[0], cz - e[2])
        corr.append(float(np.corrcoef(np.arange(dist.size), -dist)[0, 1]))
    v.look = ar.Look()
    frame(v, ctx, gui=False)
    opaque_path = not v.R.stats['translucent']
    check('translucent surface: floor first, then far -> near (corr(draw rank, -distance) > 0.95 at 4 views); '
          'opacity 1 keeps the plain index prefix', min(corr) > 0.95 and floor_ok and opaque_path,
          f'corr {[round(c, 3) for c in corr]}')
    # (g) cost: a new view sector's sort, and the frame with translucency + grid at N = 1025 (mesh 513)
    if not quick:
        fresh(v, N=1025, shape='circle', src='pulse')
        v.view3d = True
        v.show.update(main=True)
        v.panel = True
        v.look = ar.Look(opacity=0.5)
        frame(v, ctx)
        v.R._sorted = {}
        t0 = time.perf_counter()
        v.R.sorted_idx((0.0, 1.0, -2.0))
        t_sort = (time.perf_counter() - t0) * 1e3
        meds = {}
        for lab, look in (('opaque', ar.Look()), ('opacity 0.5', ar.Look(opacity=0.5)),
                          ('opacity 0.5 + grid', ar.Look(opacity=0.5, grid=0.5))):
            v.look = look
            for _ in range(3):
                frame(v, ctx)
            T = []
            for _ in range(10):
                ti.sync()
                t1 = time.perf_counter()
                frame(v, ctx)
                ti.sync()
                T.append((time.perf_counter() - t1) * 1e3)
            meds[lab] = float(np.median(T))
        extra = meds['opacity 0.5 + grid'] - meds['opaque']
        check(f'transparency cost at N=1025, mesh {v.R.nd}: one sector sort < 150 ms, translucent + grid frame '
              f'< opaque + 15 ms', t_sort < 150 and extra < 15,
              f'sort {t_sort:.0f} ms; frame medians (incl. readback) ' +
              ', '.join(f'{k} {x:.1f} ms' for k, x in meds.items()))
    v.look = ar.Look()
    N, shape, src, v.view3d, v.panel, v.orbit.az, v.orbit.el, v.orbit.dist, show = keep
    v.show.update(show)
    v.v2 = acam.View2D()
    fresh(v, N=N, shape=shape, src=src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='cuda')
    ap.add_argument('--f32', action='store_true')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    global ARCH, F32
    ARCH, F32 = a.arch, a.f32
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(OUT4, exist_ok=True)
    mc.init_taichi(a.arch, fp64=not a.f32, quiet=True)
    v = app.Viewer(app.default_args(N=257, shot='tests'))
    ctx = Ctx(v.win)
    v.camera = ctx.camera
    t0 = time.perf_counter()
    t_palette(v)
    t_visuals(v, ctx, a.quick)
    t_map2d(v, a.quick)
    t_project3d(v, ctx)
    t_orbit(v, ctx)
    t_resize(v)
    t_gain(v)
    t_input(v, ctx)
    t_text_switch(v)
    t_nslider(v)
    t_stroke_offview(v)
    t_cells_cache(v)
    t_typed(v, ctx)
    t_src_note(v)
    t_beam(v, ctx)
    t_scale(v, ctx)
    t_round4(v, ctx)
    t_unlicensed(v)
    t_gui(v, ctx, a.quick)
    t_latency(v, a.quick)
    t_contrast(v, ctx)
    nf = sum(1 for r in RESULTS if not r[1])
    print(f'\n{len(RESULTS) - nf}/{len(RESULTS)} passed ({time.perf_counter() - t0:.0f} s, arch {a.arch}'
          f'{" f32" if a.f32 else ""})')
    sys.exit(1 if nf else 0)


if __name__ == '__main__':
    main()
