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
  9. contrast: mean CIE L* of domain vs background pixels, round-1 look vs now, both views.
*Generated from scratch -- Claude Opus 5.5 -- 30/09/2026*
"""
import argparse
import math
import os
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

OUT = os.path.join(HERE, 'ScratchCLAUDE', 'AppCLAUDE')
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
        path = os.path.join(OUT, f'proj_marker_az{az:+.0f}_el{el:+.0f}_MIRROR.png')
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
    d0, cg0, ag0 = m.p.damp, v.cg, v.auto_gain
    v.cg, v.auto_gain = 1.0, True
    m.p.damp = 0.02
    v.reset()
    v.step(300)
    live = v.colour_gain() * m.amax
    while m.step_n < 2000:
        v.step(100)
    dead = v.colour_gain() * m.amax
    n_dead = m.step_n
    capped = v.gain_capped
    m.p.damp = d0
    v.reset()
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='cuda')
    ap.add_argument('--f32', action='store_true')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    mc.init_taichi(a.arch, fp64=not a.f32, quiet=True)
    v = app.Viewer(app.default_args(N=257, shot='tests'))
    ctx = Ctx(v.win)
    v.camera = ctx.camera
    t0 = time.perf_counter()
    t_palette(v)
    t_map2d(v, a.quick)
    t_project3d(v, ctx)
    t_orbit(v, ctx)
    t_resize(v)
    t_gain(v)
    t_input(v, ctx)
    t_text_switch(v)
    t_nslider(v)
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
