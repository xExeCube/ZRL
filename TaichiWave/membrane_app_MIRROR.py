"""ZRL 2D wave membrane -- Taichi port, VIEWER (GGUI). The PRIMARY app since 29/09/2026 (the web app
is frozen).

The solver is membrane_core_MIRROR.py (cross-checked against the web app); solids/walls are
solids_MIRROR.py. This file is the window: a 3D heightfield or a 2D top view of the running field,
the control panel, the walls/objects panel, and the alignment and ledger panels. Its helpers:
app_render_MIRROR.py (colours, display kernels, mesh), app_camera_MIRROR.py (orbit camera, 2D view
mapping, ray casting), app_input_MIRROR.py (one frame of input), app_solids_MIRROR.py (painting,
objects, text entry, scenes). Controls, hotkeys and measurements: NotesCLAUDE/app_MIRROR.md.

    python membrane_app_MIRROR.py [--arch cuda|vulkan|cpu] [--f32] [--N 1025]
    python membrane_app_MIRROR.py --shot out.png --steps 400 [--view 2d] [--camera 30,40,2.5] [...]

Keys: SPACE play/pause  N one step  R reset  P pulse  V 3D/2D  X wireframe  H hide panels
      L lattice  B boundary  G domain shape  M medium  O source
      C reset view  F freelook  Z zoom in (SHIFT+Z out)  arrows orbit (3D) / pan (2D)  Y background
      T draw walls  U undo stroke  I walls/objects panel  J alignment panel  K ledger panel
      Return: type the selected text object (Return applies, Escape cancels)
Mouse (3D, orbit): RMB drag orbit, MMB drag or CTRL+RMB drag zoom, LMB drag orbit (draw mode off)
      (2D): RMB/LMB drag pan, MMB drag or CTRL+RMB drag zoom about the cursor
      draw mode: LMB paints walls, SHIFT+LMB erases
Freelook (F): GGUI's own camera keys W/A/S/D/Q/E + RMB look; no app hotkey uses those letters.
"""
import argparse
import math
import textwrap
import os
import sys
import time

import numpy as np
import taichi as ti

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import membrane_core_MIRROR as mc
import solids_MIRROR as so
import app_render_MIRROR as ar
import app_camera_MIRROR as acam
import app_input_MIRROR as ain
import app_solids_MIRROR as asol

try:
    import alignment_MIRROR as alignment
    ALIGN_ERR = None
except Exception as _ex:                       # another agent may be mid-edit: retried on toggle
    alignment, ALIGN_ERR = None, f'{type(_ex).__name__}: {_ex}'
try:
    import ledger_MIRROR as ledger
    LEDGER_ERR = None
except Exception as _ex:
    ledger, LEDGER_ERR = None, f'{type(_ex).__name__}: {_ex}'

D = math.pi / 180
MESH_MAX = 513           # the displayed 3D mesh is at most this many vertices a side (stride k); a panel
                         # setting (round 1 used 1025). Measured 30/09/2026 at N = 2049, all panels open,
                         # per frame without the present: ~52 ms (CUDA) / ~70 ms (Vulkan) at 1025 (2.1M
                         # triangles), ~14 / ~24 ms at 513 -- mostly GGUI 1.7's own per-frame copy of the
                         # vertex buffer (its copy_all_to_vbo kernel ~15 ms, the C++ upload ~20 ms at 1025).
                         # The 2D view always shows every cell.
WIN = (1600, 1000)       # window size; the 2D view's image has the same aspect ratio
TITLE = 'ZRL membrane - Taichi'
# nominal panel rectangles (x, y, w, h) as window fractions, y from the TOP (ImGui's frame). GGUI
# places a sub-window once per session; clicks inside a VISIBLE panel's rectangle never paint,
# orbit or pan (if a panel is dragged elsewhere, its old rectangle is what counts).
PANELS = {'main': (0.0, 0.0, 0.25, 1.0), 'solids': (0.75, 0.0, 0.25, 0.52),
          'align': (0.75, 0.52, 0.25, 0.48), 'ledger': (0.25, 0.62, 0.50, 0.38)}
PANEL_TITLES = {'main': 'ZRL membrane (Taichi)', 'solids': 'walls and solids  [I]',
                'align': 'grid <-> edge alignment  [J]', 'ledger': 'physics ledger  [K]'}
ORBIT_DEG = 180.0        # degrees per window height of mouse drag
ZOOM_DRAG = 3.0          # zoom factor e^(3 dy) for a vertical drag of dy window heights
KEY_ORBIT = 5.0          # degrees per arrow key
KEY_ZOOM = 1.25
AUTO_WINDOW = 64         # steps: the auto colour gain's peak window
AUTO_REL_FLOOR = 1e-3    # the auto gain never exceeds (colour gain) / (1e-3 x the run's largest |u|) (01/10/2026)
N_SETTLE = 0.35         # s: the N slider applies once its value has rested this long (01/10/2026)
RESIZE_SETTLE = 0.3      # s: a new window size must hold this long before the 2D image buffer follows (01/10/2026)
TEXT_WRAP = 54           # characters per panel line (ImGui's 7 px font in a 400 px panel)
UNL_TRI_RAD = 0.89       # the plate fits the triangular array at radius <= 0.89 (core, 30/09)


class Viewer:
    def __init__(self, a):
        self.a = a
        self.win = tuple(getattr(a, 'win', WIN))   # the 2D image buffer's size (2D mapping uses it)
        # the REAL window size (resized / maximised; read every frame by main()). The 3D view's
        # projection and cursor ray use its aspect. Review v3 01/10/2026: both used to stay at
        # 1600 x 1000 for ever, so a maximised window (2560 x 1377) painted 11-14 cells (N = 257) off
        # the cursor in 3D and drew the 2D circle 1.16x too wide.
        self.win_real = self.win
        self._resize_pending = None             # (size, first seen) of a size the image has not followed yet
        self.m = mc.Membrane(mc.Params(N=a.N, src='pulse'))
        self.playing = not a.shot
        self.view3d = a.view != '2d'
        self.wire = False
        self.panel = True                       # H: all panels
        self.show = {'main': True, 'solids': False, 'align': False, 'ledger': False}
        self.hs, self.cg = 1.0, 1.0
        self.auto_gain = True                   # colour gain / peak |u| (display only)
        self.peak = 1.0                         # the auto gain's peak hold
        self.gain_ref = (0.0, 0)                # (largest |u| of this run, step it was updated at)
        self.gain_capped = False
        self.n_pending = None                   # (N, time) of an N slider value not applied yet
        self.sub = 4
        self.mesh_max = MESH_MAX
        self.cont_time = False                  # Co changes as a new time step (no time reflection)
        self.ms_step = 0.0
        self.ms_frame = 0.0
        self.look = ar.Look()
        self.R = ar.Renderer(self.win, mc.mask_np_dtype())
        self.orbit = acam.OrbitCam()
        self.v2 = acam.View2D()
        self.freelook = False
        self.camera = None                      # the ti.ui.Camera (set by attach())
        self.sol = asol.SolidsUI()
        self.drag = None                        # {'kind', 'last', 'btn'}
        self.cursor = (0.5, 0.5)
        self.shift = False
        self.flash_msg, self.flash_t = '', 0.0
        self._dom_src = None
        self._dom_key = None
        self._cells = (0.0, 0, 0)
        self.align_panel = None                 # alignment_MIRROR.gui_section keeps its state here
        self._edge_cache = (None, [])
        self.sync_domain()

    # ---- messages -----------------------------------------------------------------------------
    def flash(self, msg):
        self.flash_msg, self.flash_t = str(msg), time.perf_counter()

    # ---- geometry that changes on reset -------------------------------------------------------
    def sync_domain(self, force=False):
        """Upload the DOMAIN mask and rebuild the mesh when the shape changed. set_co() rebuilds the
        domain array (a new object, same contents): compared first, so a Co slider drag at N = 8193
        does not re-mesh every frame. Painting changes M only, never the mesh."""
        m = self.m
        dm = m.domain_mask_np
        if not force and dm is self._dom_src:
            return
        same = (self._dom_src is not None and self._dom_src.shape == dm.shape and
                self._dom_key == (m.N, m.tri()) and np.array_equal(self._dom_src, dm))
        self._dom_src = dm
        if same and not force:
            return
        self._dom_key = (m.N, m.tri())
        k = max(1, math.ceil(m.N / self.mesh_max))
        self.R.set_domain(dm, m.N, k, m.tri())

    def reset(self):
        self.sol.end(self.m)
        self.m.reseed()
        self.peak = 1.0
        self.gain_ref = (0.0, 0)
        self.sync_domain()

    def _fixups(self, before, keep=()):
        """Defaults the core asked the app to handle (30/09/2026): on entering the Unlicensed plate
        the default source (centre) sits inside the lettering's A counter -- a sealed cavity -- so
        it moves to sy = -0.9 (below DEATH); on the triangular lattice the plate at radius 1 is
        clipped by the rhombic array, so the radius drops to 0.89."""
        p = self.m.p
        msgs = []
        if p.shape == 'unlicensed' and (before[0] != 'unlicensed' or before[1] != p.lattice):
            if before[0] != 'unlicensed' and p.sx == 0.0 and p.sy == 0.0 and not {'sx', 'sy'} & set(keep):
                p.sy = -0.9
                msgs.append('source moved to y = -0.9 (the centre is inside the A counter)')
            if p.lattice == 'tri' and p.rad > UNL_TRI_RAD and 'rad' not in keep:
                p.rad = UNL_TRI_RAD
                msgs.append(f'radius {UNL_TRI_RAD} (the plate fits the triangular array)')
        if msgs:
            self.flash('Unlicensed: ' + '; '.join(msgs))

    def cycle(self, attr, options):
        p = self.m.p
        before = (p.shape, p.lattice)
        cur = getattr(p, attr)
        setattr(p, attr, options[(options.index(cur) + 1) % len(options)])
        self._fixups(before)
        self.reset()

    def set_shape(self, shape):
        p = self.m.p
        before = (p.shape, p.lattice)
        p.shape = shape
        self._fixups(before)
        self.reset()

    def set_draw(self, on):
        self.sol.draw = bool(on)
        if on:
            self.show['solids'] = True
        else:
            self.sol.end(self.m)

    def set_freelook(self, on, camera=None):
        camera = camera or self.camera
        on = bool(on)
        if on and not self.freelook and camera is not None:
            self.orbit.pose().apply(camera)            # freelook starts from the orbit view
            camera.last_mouse_x = camera.last_mouse_y = camera.last_time = None
        self.freelook = on                             # the orbit state is kept untouched

    # ---- time stepping ------------------------------------------------------------------------
    def step(self, n):
        t0 = time.perf_counter()
        self.m.step(n)
        ti.sync()
        dt = (time.perf_counter() - t0) / n
        self.ms_step = dt * 1e3 if self.ms_step == 0 else 0.9 * self.ms_step + 0.1 * dt * 1e3

    @staticmethod
    def moved(new, old):
        # GGUI sliders are f32: a parameter of 0.9 comes back as 0.89999998. Comparing with
        # f64 precision saw a 'change' every frame and reset the run every frame.
        return abs(new - old) > 1e-5 * max(1.0, abs(old))

    # ---- camera / mapping ---------------------------------------------------------------------
    def aspect(self):
        return self.win_real[0] / self.win_real[1]

    def set_window_shape(self, shape, now=None):
        """The real window size, once per frame. The 3D aspect follows at once (GGUI's projection
        does). The 2D image buffer follows once the size has held RESIZE_SETTLE s (a drag-resize
        would otherwise allocate and compile a buffer per intermediate size); until then GGUI stretches
        the old buffer over the window, and the 2D mapping (self.win = buffer size, in screen
        fractions) stays exact for that stretched picture."""
        W, H = int(shape[0]), int(shape[1])
        if W < 16 or H < 16:                    # minimised
            return
        self.win_real = (W, H)
        if (W, H) == self.win:
            self._resize_pending = None
            return
        now = time.perf_counter() if now is None else now
        if self._resize_pending is None or self._resize_pending[0] != (W, H):
            self._resize_pending = ((W, H), now)
            return
        if now - self._resize_pending[1] >= RESIZE_SETTLE:
            self.R.resize(W, H)
            self.win = (W, H)
            self._resize_pending = None

    def pose(self):
        c = self.camera
        if self.freelook and c is not None:
            return acam.CamPose(tuple(c.curr_position), tuple(c.curr_lookat), tuple(c.curr_up), self.orbit.fov)
        return self.orbit.pose()

    def cursor_grid(self, sx, sy):
        """Cursor -> grid units (solids_MIRROR's frame), or None (3D: the ray misses the plane)."""
        m = self.m
        if self.view3d:
            hit = self.pose().screen_to_plane(sx, sy, self.aspect())
            if hit is None:
                return None
            x, y = acam.world_to_phys(m, hit[0], hit[1])
        else:
            x, y = self.v2.screen_to_phys(m, sx, sy, *self.win)
        return acam.phys_to_grid(m, x, y)

    def n_slider(self, N, shown, now=None):
        """The N slider's value this frame (see gui_main): a change starts / restarts the wait,
        a value that has rested N_SETTLE s is applied (odd)."""
        now = time.perf_counter() if now is None else now
        if N != shown:
            self.n_pending = (N, now)
        elif self.n_pending and now - self.n_pending[1] >= N_SETTLE:
            N = self.n_pending[0]
            self.n_pending = None
            if (N | 1) != self.m.N:
                self.m.p.N = N | 1
                self.reset()

    def free_centre(self):
        """The 2D picture's horizontal offset (window fractions from the centre): the middle of the
        strip the visible side panels leave free (0 with the panels hidden)."""
        if not self.panel:
            return 0.0
        left = PANELS['main'][2] if self.show['main'] else 0.0
        right = PANELS['solids'][2] if (self.show['solids'] or self.show['align']) else 0.0
        return 0.5 * (left - right)

    def over_panel(self, sx, sy):
        if not self.panel:
            return None
        yt = 1.0 - sy
        for k, on in self.show.items():
            if on:
                x, y, w, h = PANELS[k]
                if x <= sx <= x + w and y <= yt <= y + h:
                    return k
        return None

    # ---- input --------------------------------------------------------------------------------
    def handle(self, fr):
        """One frame of input (app_input_MIRROR.Frame): keys first, then the mouse."""
        self.cursor = fr.cursor
        self.shift = fr.held('Shift')
        self.v2.ox = self.free_centre()
        if self.sol.text is not None:
            self.sol.chars(fr.chars)
        for k in fr.presses:
            if k in ain.BUTTONS:
                continue
            if self.sol.text is not None and self.sol.key(self.m, k):
                continue
            self.key(k, shift=fr.held('Shift'))
        self.mouse(fr)

    def key(self, k, shift=False):
        m = self.m
        if k == ' ':
            self.playing = not self.playing
        elif k == 'n':
            self.step(1)
        elif k == 'r':
            self.reset()
        elif k == 'p':
            m.drop_pulse()
            self.peak = max(self.peak, m.amax)
        elif k == 'v':
            self.view3d = not self.view3d
            self.sol.end(m)
            self.drag = None
        elif k == 'x':
            self.wire = not self.wire
        elif k == 'h':
            self.panel = not self.panel
        elif k == 'l':
            self.cycle('lattice', ['sq', 'tri'])
        elif k == 'b':
            self.cycle('bc', mc.BCS)
        elif k == 'g':
            self.cycle('shape', mc.SHAPES)
        elif k == 'm':
            self.cycle('medium', mc.MEDIA)
        elif k == 'o':
            self.cycle('src', mc.SOURCES)
        elif k == 'c':
            self.orbit.reset()
            self.v2.reset()
            if self.freelook:
                self.set_freelook(False)
                self.set_freelook(True)
        elif k == 'f':
            self.set_freelook(not self.freelook)
        elif k == 'z':
            self.zoom(1 / KEY_ZOOM if shift else KEY_ZOOM)
        elif k in ('Left', 'Right', 'Up', 'Down'):
            dx = {'Left': -1, 'Right': 1}.get(k, 0)
            dy = {'Down': -1, 'Up': 1}.get(k, 0)
            if self.view3d:
                self.orbit.orbit(-dx * KEY_ORBIT, dy * KEY_ORBIT)
            else:
                self.v2.pan_screen(m, -0.05 * dx, -0.05 * dy, *self.win)
        elif k == 'y':
            self.look.cycle_bg()
        elif k == 't':
            self.set_draw(not self.sol.draw)
        elif k == 'u':
            self.flash(self.sol.undo(m))
        elif k == 'i':
            self.show['solids'] = not self.show['solids']
        elif k == 'j':
            self.toggle_align()
        elif k == 'k':
            self.toggle_ledger()
        elif k == 'Return':
            if self.sol.start_text(m):
                self.panel = True               # v3 01/10/2026: with H (panels off) the entry was invisible
                self.show['solids'] = True
                self.flash('typing: Return applies, Escape cancels')

    def zoom(self, f, at=None):
        if self.view3d:
            self.orbit.zoom(f)
        else:
            sx, sy = at if at is not None else (0.5, 0.5)
            self.v2.zoom_at(self.m, f, sx, sy, *self.win)

    def mouse(self, fr):
        m = self.m
        sx, sy = fr.cursor
        # a drag ends when its button is no longer held (a click inside one frame starts AND ends)
        for b in ('LMB', 'MMB', 'RMB'):
            if b in fr.presses and self.drag is None and self.over_panel(sx, sy) is None:
                self.drag = self._start_drag(b, fr)
        d = self.drag
        if d is None:
            return
        if d['kind'] == 'paint':
            g = self.cursor_grid(sx, sy)
            if g is not None:
                self.sol.move(m, g)
        else:
            dx, dy = sx - d['last'][0], sy - d['last'][1]
            if d['kind'] == 'orbit':
                self.orbit.orbit(-dx * ORBIT_DEG * self.aspect(), -dy * ORBIT_DEG)
            elif d['kind'] == 'pan':
                self.v2.pan_screen(m, dx, dy, *self.win)
            elif d['kind'] == 'zoom' and dy != 0:
                self.zoom(math.exp(ZOOM_DRAG * dy), at=d['start'])
        d['last'] = (sx, sy)
        if d['btn'] not in fr.down:
            if d['kind'] == 'paint':
                self.sol.end(m)
            self.drag = None

    def _start_drag(self, b, fr):
        sx, sy = fr.cursor
        kind = None
        if b == 'LMB':
            if self.sol.draw:
                g = self.cursor_grid(sx, sy)
                if g is None:
                    return None
                self.sol.begin(self.m, g, erase=fr.held('Shift'))
                kind = 'paint'
            elif not self.freelook:
                kind = 'orbit' if self.view3d else 'pan'
        elif b == 'MMB':
            kind = 'zoom'
        elif b == 'RMB':
            if self.view3d and self.freelook:
                return None                          # GGUI's freelook owns RMB
            kind = 'zoom' if fr.held('Control') else ('orbit' if self.view3d else 'pan')
        if kind is None:
            return None
        return {'kind': kind, 'btn': b, 'last': (sx, sy), 'start': (sx, sy)}

    # ---- panels -------------------------------------------------------------------------------
    def toggle_align(self):
        global alignment, ALIGN_ERR
        self.show['align'] = not self.show['align']
        if self.show['align'] and alignment is None:
            try:
                import alignment_MIRROR as alignment
                ALIGN_ERR = None
            except Exception as ex:
                ALIGN_ERR = f'{type(ex).__name__}: {ex}'

    def toggle_ledger(self):
        global ledger, LEDGER_ERR
        self.show['ledger'] = not self.show['ledger']
        if self.show['ledger'] and (ledger is None or not hasattr(ledger, 'gui_section')):
            try:
                import importlib
                ledger = importlib.reload(ledger) if ledger is not None else importlib.import_module('ledger_MIRROR')
                LEDGER_ERR = None
            except Exception as ex:
                LEDGER_ERR = f'{type(ex).__name__}: {ex}'

    # ---- drawing ------------------------------------------------------------------------------
    def colour_gain(self):
        if not self.auto_gain:
            return self.cg
        m = self.m
        # the peak over the last AUTO_WINDOW recorded steps: steady colours (no step-to-step
        # flicker), and a decaying pulse stays visible (measured 30/09: a centred pulse at N = 513
        # is down to ~0.1 of its start after 300 steps, and at gain 1 the ring was nearly the rest
        # colour). A 256-step window still held the pulse's early, 2-3x larger peak at step 300.
        h = m.hist['amax']
        a = max(h[-AUTO_WINDOW:]) if (m.record and h) else m.amax
        # Review v3 01/10/2026: 1 / peak alone blew a dead field up to full scale (damping 0.02, N = 257:
        # |u| = 2.3e-7 of the start at step 2000 shown at gain 3.2e6, as bright as the live pulse; in f32
        # that is round-off). The peak is floored at AUTO_REL_FLOOR x the largest |u| of this run (step
        # counter going back = a new run), so a decayed field fades out instead.
        ref, at = self.gain_ref
        if m.step_n < at:
            ref, at = 0.0, 0
        if m.record and h and m.step_n - at > AUTO_WINDOW:    # steps not seen since the last call
            ref = max(ref, max(h[-min(len(h), m.step_n - at):]))
        ref = max(ref, a, m.amax)
        self.gain_ref = (ref, m.step_n)
        floor = max(AUTO_REL_FLOOR * ref, 1e-12)
        self.gain_capped = a < floor
        self.peak = max(a, floor)
        return self.cg / self.peak

    def edge_segments(self):
        """The alignment panel's 'colour the domain edges' overlay: [(a, b, rgb)] physical."""
        st = self.align_panel
        if alignment is None or st is None or not getattr(st, 'edge_col', False):
            return []
        key = st.key(self.m)
        if self._edge_cache[0] != key:
            try:
                self._edge_cache = (key, alignment.al_edge_lines(self.m))
            except Exception as ex:
                self._edge_cache = (key, [])
                self.flash(f'edge colouring failed: {type(ex).__name__}: {ex}')
        return self._edge_cache[1]

    def brush_ring(self):
        """The brush outline at the cursor (physical points), when drawing."""
        if not self.sol.draw or self.over_panel(*self.cursor):
            return None
        g = self.cursor_grid(*self.cursor)
        if g is None:
            return None
        m = self.m
        x0, y0 = acam.grid_to_phys(m, *g)
        r = max(so.MIN_BRUSH, self.sol.brush)
        return [(x0 + r * math.cos(t * 2 * math.pi / 40), y0 + r * math.sin(t * 2 * math.pi / 40)) for t in range(41)]

    def draw(self, window, canvas, scene, camera, markers=()):
        m = self.m
        self.sync_domain()
        pxs = self.v2.px_size(m, self.win[1])
        self.v2.ox = self.free_centre()
        self.R.set_params(self.look, self.colour_gain(), self.hs, self.v2.kernel_cx(m, *self.win), self.v2.cy, pxs)
        canvas.set_background_color(ar.SEG['bg'] if self.look.segment else self.look.bg_col())
        ring = self.brush_ring()
        edges = self.edge_segments()
        if self.view3d:
            if not self.freelook:
                self.orbit.pose().apply(camera)
            scene.set_camera(camera)
            eye = self.pose().eye
            self.R.draw3d(m, self.look, scene, eye, self.wire)
            segs = []
            w = lambda p, h=0.004: (p[0] / m.c(), h, -p[1] / m.c())
            for a, b, c in edges:
                segs.append((w(a), w(b), c))
            if ring:
                # painting happens on the REST plane Y = 0 (where the cursor ray meets it). The ring
                # is drawn above the surface (so solids and crests do not hide it) but each point is
                # slid along its view ray, so it covers exactly the pixels of the Y = 0 ring
                col = (1.0, 0.35, 0.3) if (self.sol.erase or self.drag_erase()) else (0.95, 0.95, 0.35)
                hr = math.copysign(self.look.solid_h + 0.01, eye[1]) if abs(eye[1]) > 1e-6 else 0.0
                t = hr / eye[1] if abs(eye[1]) > 1e-6 else 0.0
                lift = lambda P: (P[0] + (eye[0] - P[0]) * t, hr, P[2] + (eye[2] - P[2]) * t)
                segs += [(lift(w(ring[k], 0.0)), lift(w(ring[k + 1], 0.0)), col) for k in range(len(ring) - 1)]
            self.R.lines3d(scene, segs)
            if markers:
                self.R.markers3d(scene, list(markers))
            canvas.scene(scene)
        else:
            self.R.draw2d(m, canvas)
            W, H = self.win
            s = lambda p: self.v2.phys_to_screen(m, p[0], p[1], W, H)
            segs = [(s(a), s(b), c) for a, b, c in edges]
            if ring:
                col = (1.0, 0.35, 0.3) if (self.sol.erase or self.drag_erase()) else (0.95, 0.95, 0.35)
                segs += [(s(ring[k]), s(ring[k + 1]), col) for k in range(len(ring) - 1)]
            self.R.lines2d(canvas, segs)

    def drag_erase(self):
        """The brush ring turns red when the next (or current) stroke erases."""
        sc = self.m.scene
        if self.sol.painting:
            return bool(sc is not None and sc.strokes and sc.strokes[-1].erase)
        return self.shift

    # ---- the panels ---------------------------------------------------------------------------
    def cells(self):
        """(free, solid) cell counts, recounted at most twice a second (a full-grid sum at N = 8193
        is ~15 ms, too much for every frame)."""
        t, free, solid = self._cells
        now = time.perf_counter()
        if now - t > 0.5:
            free, solid = self.m.domain_cells(), self.m.solid_cells()
            self._cells = (now, free, solid)
        return free, solid

    def gui(self, g):
        if self.show['main']:
            x, y, w_, h_ = PANELS['main']
            with g.sub_window(PANEL_TITLES['main'], x, y, w_, h_) as w:
                self.gui_main(w)
        if self.show['solids']:
            x, y, w_, h_ = PANELS['solids']
            with g.sub_window(PANEL_TITLES['solids'], x, y, w_, h_) as w:
                self.sol.gui(w, self)
        if self.show['align']:
            x, y, w_, h_ = PANELS['align']
            with g.sub_window(PANEL_TITLES['align'], x, y, w_, h_) as w:
                if alignment is not None and hasattr(alignment, 'gui_section'):
                    try:
                        if self.align_panel is None and hasattr(alignment, 'AlignPanel'):
                            self.align_panel = alignment.AlignPanel()
                            self.align_panel.wrap = TEXT_WRAP - 2     # its default 64 overflows the panel
                        alignment.gui_section(w, self)
                    except Exception as ex:
                        w.text(f'alignment panel error: {type(ex).__name__}: {ex}', color=(0.95, 0.4, 0.4))
                else:
                    w.text(f'alignment panel unavailable ({ALIGN_ERR or "no gui_section"})')
        if self.show['ledger']:
            x, y, w_, h_ = PANELS['ledger']
            with g.sub_window(PANEL_TITLES['ledger'], x, y, w_, h_) as w:
                if ledger is not None and hasattr(ledger, 'gui_section'):
                    try:
                        ledger.gui_section(w, self)
                    except Exception as ex:
                        w.text(f'ledger panel error: {type(ex).__name__}: {ex}', color=(0.95, 0.4, 0.4))
                else:
                    w.text('ledger: not available yet (ledger_MIRROR.py ' +
                           (f'import failed: {LEDGER_ERR})' if LEDGER_ERR else 'has no gui_section)'))

    def gui_main(self, w):
        m, p = self.m, self.m.p
        moved = self.moved
        w.text(f'{"playing" if self.playing else "paused"}   step {m.step_n}   t = {m.t:.1f}')
        if self.flash_msg and time.perf_counter() - self.flash_t < 8.0:
            for piece in textwrap.wrap(self.flash_msg, TEXT_WRAP)[:4]:
                w.text(piece, color=(0.95, 0.8, 0.35))
        if w.button('play / pause  [space]'):
            self.playing = not self.playing
        if w.button('reset  [R]'):
            self.reset()
        if w.button('drop pulse  [P]'):
            m.drop_pulse()
        w.text('')
        w.text(f'lattice: {p.lattice}   stencil: {p.stencil if p.lattice == "sq" else 6}-point')
        if w.button('lattice  [L]'):
            self.cycle('lattice', ['sq', 'tri'])
        if p.lattice == 'sq' and w.button('stencil 5 / 9'):
            p.stencil = 9 if p.stencil == 5 else 5
            self.reset()
        w.text(f'domain: {p.shape}   radius {p.rad:.2f}   R = {m.dom_R():.2f}')
        if w.button('shape  [G]'):
            self.cycle('shape', mc.SHAPES)
        w.text(f'boundary: {p.bc if p.lattice == "sq" else "clamped (tri)"}   medium: {p.medium}')
        if w.button('boundary  [B]'):
            self.cycle('bc', mc.BCS)
        if w.button('medium  [M]'):
            self.cycle('medium', mc.MEDIA)
        w.text(f'source: {p.src}' + (f' ({len(m.vtx)})' if p.src == 'vtx' else '') + f'   wave: {p.wave}')
        if w.button('source  [O]'):
            self.cycle('src', mc.SOURCES)
        if w.button('waveform'):
            p.wave = mc.WAVES[(mc.WAVES.index(p.wave) + 1) % len(mc.WAVES)]
            m.harm = m.harmonics()
        if p.src == 'vtx':
            vd = w.checkbox('drive the vertices', p.vtx_drive)
            if vd != p.vtx_drive:
                p.vtx_drive = vd
                m.recache_source()
            if w.button(f'vertex inset: {p.vtx_inset}'):
                self.cycle('vtx_inset', ['radial', 'wall'])
        w.text('')
        # sliders: apply only on change
        # review v3 01/10/2026: every N change rebuilds the run (measured 0.9-1.0 s per change in 3D,
        # 0.2-0.4 s in 2D: new mesh fields + kernel compiles), and a slider drag changed N on every
        # frame, so a drag froze the app for seconds. The value now applies once it rests N_SETTLE s.
        shown = self.n_pending[0] if self.n_pending else m.N
        N = w.slider_int('N (odd)', shown, 49, 8193)
        self.n_slider(N, shown)
        if self.n_pending:
            w.text(f'   N -> {self.n_pending[0] | 1} when the slider rests', color=(0.95, 0.75, 0.4))
        co = w.slider_float('Courant Co', p.Co, 0.05, 1.0)
        if moved(co, p.Co):
            m.set_co(co, rescale=self.cont_time)
        self.cont_time = w.checkbox('Co change = new time step', self.cont_time)
        w.text('   (off: a Co change reflects waves in time)', color=(0.55, 0.55, 0.60))
        w.text(f'   C_max = {m.co_limit():.4f}' + ('   UNSTABLE' if p.Co > m.co_limit() + 1e-12 else ''))
        dmp = w.slider_float('damping', p.damp, 0.0, 0.02)
        if moved(dmp, p.damp):
            p.damp = dmp
        self.sub = w.slider_int('steps / frame', self.sub, 1, 256)
        for attr, lab, lo, hi in (('rad', 'radius', 0.1, 1.5), ('sigma', 'pulse sigma', 0.8, 12.0),
                                  ('sx', 'source x', -1.0, 1.0), ('sy', 'source y', -1.0, 1.0)):
            v = w.slider_float(lab, getattr(p, attr), lo, hi)
            if moved(v, getattr(p, attr)):
                setattr(p, attr, v)
                self.reset()
        rot = w.slider_float('rotation (deg)', p.rot / D, 0.0, 90.0)
        if moved(rot, p.rot / D):
            p.rot = rot * D
            self.reset()
        if w.button('rotation 30 deg (conforming on tri)'):
            p.rot = 30 * D
            self.reset()
        if p.shape == 'rhomb':
            th = w.slider_float('rhombus (deg)', p.rtheta / D, 20.0, 90.0)
            if moved(th, p.rtheta / D):
                p.rtheta = th * D
                self.reset()
        if p.src in ('cont', 'slit') or (p.src == 'vtx' and p.vtx_drive):
            f = w.slider_float('freq (cyc/step)', p.freq, 0.005, 0.25)
            if moved(f, p.freq):
                p.freq = f
                m.harm = m.harmonics()
        if p.src == 'mode':
            mm = w.slider_int('mode m', p.mm, 1, 64)
            mn = w.slider_int('mode n', p.mn, 1, 64)
            if (mm, mn) != (p.mm, p.mn):
                p.mm, p.mn = mm, mn
                self.reset()
        # ---- view ----
        w.text('')
        w.text('view', color=(0.72, 0.62, 1.0))
        v3 = w.checkbox('3D view  [V]', self.view3d)
        if v3 != self.view3d:
            self.key('v')
        self.wire = w.checkbox('wireframe  [X]', self.wire)
        if self.view3d:
            mm = w.slider_int('mesh detail', self.mesh_max, 129, 1025)
            if mm != self.mesh_max:
                self.mesh_max = mm
                self.sync_domain(force=True)
            fl = w.checkbox('freelook  [F]  (W/A/S/D/Q/E + RMB)', self.freelook)
            if fl != self.freelook:
                self.set_freelook(fl)
            o = self.orbit
            z = w.slider_float('zoom [Z]', o.zoom_level, acam.OrbitCam.DEFAULT['dist'] / acam.DIST_MAX,
                               acam.OrbitCam.DEFAULT['dist'] / acam.DIST_MIN)
            if moved(z, o.zoom_level):
                o.zoom_level = z
            az = w.slider_float('azimuth (deg)', o.az, -180.0, 180.0)
            if moved(az, o.az):
                o.az = az
                o.clamp()
            el = w.slider_float('elevation (deg)', o.el, -acam.EL_MAX, acam.EL_MAX)
            if moved(el, o.el):
                o.el = el
                o.clamp()
        else:
            z = w.slider_float('2D zoom [Z]', self.v2.zoom, acam.ZOOM2D_MIN, acam.ZOOM2D_MAX)
            if moved(z, self.v2.zoom):
                self.zoom(z / self.v2.zoom)
        if w.button('reset view  [C]'):
            self.key('c')
        # ---- look ----
        w.text('')
        w.text('look', color=(0.72, 0.62, 1.0))
        L = self.look
        if w.button(f'background: {L.bg}  [Y]'):
            L.cycle_bg()
        for attr, lab, lo, hi in (('exposure', 'exposure', 0.3, 3.0), ('zero', 'zero level', 0.0, 0.5),
                                  ('contrast', 'contrast', 0.3, 4.0)):
            v = w.slider_float(lab, getattr(L, attr), lo, hi)
            if moved(v, getattr(L, attr)):
                setattr(L, attr, v)
        self.auto_gain = w.checkbox('auto colour gain (1 / peak |u|)', self.auto_gain)
        if self.auto_gain and self.gain_capped:
            w.text('  capped at 1000x: |u| < 1e-3 of the run peak', color=(0.95, 0.75, 0.4))
        self.cg = w.slider_float('colour gain', self.cg, 0.1, 50.0)
        self.hs = w.slider_float('height scale', self.hs, 0.0, 10.0)
        if self.view3d:
            v = w.slider_float('shading (0=unlit)', L.shading, 0.0, 1.0)
            if moved(v, L.shading):
                L.shading = v
            L.floor = w.checkbox('show the array outside the domain', L.floor)
            v = w.slider_float('solid height', L.solid_h, 0.0, 0.2)
            if moved(v, L.solid_h):
                L.solid_h = v
        else:
            L.hatch = w.checkbox('hatch outside the domain', L.hatch)
        sc = w.color_edit_3('solid colour', L.solid)
        L.solid = tuple(float(c) for c in sc)
        oc = w.color_edit_3('outside colour', L.outside_col())
        if any(moved(float(a), float(b)) for a, b in zip(oc, L.outside_col())):
            L.outside = tuple(float(c) for c in oc)
        # ---- panels ----
        w.text('')
        w.text('panels  (H hides all)', color=(0.72, 0.62, 1.0))
        if w.button(('hide' if self.show['solids'] else 'show') + ' walls and solids  [I]'):
            self.show['solids'] = not self.show['solids']
        d = w.checkbox('draw walls  [T]', self.sol.draw)
        if d != self.sol.draw:
            self.set_draw(d)
        if w.button(('hide' if self.show['align'] else 'show') + ' alignment  [J]'):
            self.toggle_align()
        if w.button(('hide' if self.show['ledger'] else 'show') + ' ledger  [K]'):
            self.toggle_ledger()
        # ---- readouts ----
        if m.record and m.hist['E']:
            E, amax = m.hist['E'][-1], m.amax          # the recorder's last step: no extra kernel
        else:
            E, amax = m.stats()
        free, solid = self.cells()
        w.text('')
        w.text(f'cells {free:,}' + (f'   solid {solid:,}' if solid else ''))
        w.text(f'E = {E:.5g}   max|u| = {amax:.4g}')
        w.text(f'{self.ms_step:.3f} ms/step ({self.ms_step * 1e6 / (m.N * m.N):.3f} ns/cell)')
        w.text(f'frame {self.ms_frame:.1f} ms   3D mesh {self.R.nd} (stride {self.R.k})')
        if m.clip_frac is not None and m.clip_frac > 1e-3:
            w.text(f'pulse clipped at t = 0: {100 * m.clip_frac:.1f}%')


# ---- programmatic strokes (--walls-test, tests) ------------------------------------------------
WALLS_TEST_PATH = [(-0.55, -0.35), (-0.2, 0.05), (0.15, -0.25), (0.5, 0.2)]   # grid units


def path_screen_points(v, pts_grid, per_seg=12):
    """Cursor positions that put the cursor over the given grid points in the CURRENT view, with
    `per_seg` samples per leg (what a mouse drag delivers)."""
    m = v.m
    v.v2.ox = v.free_centre()           # what handle() will use for these points
    out = []
    for (a, b) in zip(pts_grid, pts_grid[1:]):
        for t in range(per_seg):
            u = t / per_seg
            gx, gy = a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u
            out.append((gx, gy))
    out.append(pts_grid[-1])
    scr = []
    for gx, gy in out:
        x, y = acam.grid_to_phys(m, gx, gy)
        if v.view3d:
            X, Z = acam.phys_to_world(m, x, y)
            scr.append(v.pose().world_to_screen((X, 0.0, Z), v.aspect()))
        else:
            scr.append(v.v2.phys_to_screen(m, x, y, *v.win))
    return out, scr


def programmatic_stroke(v, pts_grid, erase=False, per_seg=12):
    """Paint through the SAME input path as the mouse: draw mode on, LMB press at the first
    point, drag through the rest, release. Returns (intended grid points, screen points)."""
    was, panel = v.sol.draw, v.panel
    v.set_draw(True)
    v.panel = False                     # a scripted stroke may start where a panel would be
    g, scr = path_screen_points(v, pts_grid, per_seg)
    mods = {'Shift'} if erase else set()
    v.handle(ain.Frame(scr[0], {'LMB'} | mods, ['LMB']))
    for s in scr[1:]:
        v.handle(ain.Frame(s, {'LMB'} | mods, []))
    v.handle(ain.Frame(scr[-1], mods, []))
    v.set_draw(was)
    v.panel = panel
    return g, scr


def default_args(**kw):
    a = argparse.Namespace(arch='cuda', f32=False, N=513, shot=None, steps=300, view='3d', hs=1.0, cg=1.0,
                           wire=False, set=[], win=WIN)
    for k, val in kw.items():
        setattr(a, k, val)
    return a


def parse_triple(s, n=3):
    vals = [float(t) for t in s.split(',')]
    if len(vals) != n:
        raise argparse.ArgumentTypeError(f'expected {n} comma-separated numbers, got {s!r}')
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='cuda')
    ap.add_argument('--f32', action='store_true')
    ap.add_argument('--N', type=int, default=513)
    ap.add_argument('--shot', default=None, help='render to this PNG without a window, then exit')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--view', default='3d', choices=['3d', '2d'])
    ap.add_argument('--hs', type=float, default=1.0, help='height scale')
    ap.add_argument('--cg', type=float, default=1.0, help='colour gain')
    ap.add_argument('--no-auto-gain', action='store_true', help='colour gain as given (not / peak |u|)')
    ap.add_argument('--wire', action='store_true')
    ap.add_argument('--set', nargs='*', default=[], help='Params overrides, e.g. shape=n6 rot_deg=30')
    # view and look
    ap.add_argument('--camera', default=None, help='orbit camera az,el,dist (degrees, degrees, world units)')
    ap.add_argument('--zoom', type=float, default=None, help='2D zoom factor (3D: use --camera)')
    ap.add_argument('--pan', default=None, help='2D pan x,y in grid units (the point at the window centre)')
    ap.add_argument('--freelook', action='store_true', help='start in freelook (GGUI camera keys)')
    ap.add_argument('--bg', default=None, choices=ar.BG_ORDER)
    ap.add_argument('--zero', type=float, default=None, help='palette zero level (0.05 = the web app)')
    ap.add_argument('--exposure', type=float, default=None)
    ap.add_argument('--contrast', type=float, default=None)
    ap.add_argument('--shading', type=float, default=None, help='3D: 0 = unlit, 1 = full headlight')
    ap.add_argument('--no-floor', action='store_true', help='3D: hide the array outside the domain')
    ap.add_argument('--no-hatch', action='store_true', help='2D: no hatch outside the domain')
    ap.add_argument('--solid-h', type=float, default=None, help='3D: height of solid cells')
    ap.add_argument('--solid-col', default=None, help='r,g,b of solid cells')
    ap.add_argument('--segment', action='store_true', help='key colours per cell class (contrast tests)')
    # solids
    ap.add_argument('--scene', default=None, help='load a solids scene (JSON; a bare name looks in ScenesCLAUDE)')
    ap.add_argument('--plate-text', type=int, default=None, choices=[0, 1])
    ap.add_argument('--plate-screws', type=int, default=None, choices=[0, 1])
    ap.add_argument('--walls-test', action='store_true',
                    help='paint a wall with a programmatic mouse stroke (in the chosen view) before stepping')
    ap.add_argument('--brush', type=float, default=None, help='brush radius in cells')
    ap.add_argument('--marker', default=None, help='3D: draw a marker particle at world x,y,z')
    ap.add_argument('--panels', default=None, help="show panels in the shot: 'main' or e.g. main,solids,align,ledger")
    # '--camera -135,60,2' would read '-135,60,2' as an option: glue such values to their flag
    argv, VALS = list(sys.argv[1:]), ('--camera', '--pan', '--marker', '--solid-col')
    for k in range(len(argv) - 2, -1, -1):
        if argv[k] in VALS and argv[k + 1][:1] == '-' and argv[k + 1][1:2].isdigit():
            argv[k:k + 2] = [argv[k] + '=' + argv[k + 1]]
    a = ap.parse_args(argv)
    a.win = WIN
    mc.init_taichi(a.arch, fp64=not a.f32, quiet=True)
    v = Viewer(a)
    p = v.m.p
    before = (p.shape, p.lattice)
    keys = set()
    for kv in a.set:
        k, val = kv.split('=')
        keys.add(k[:-4] if k.endswith('_deg') else k)
        if k.endswith('_deg'):
            setattr(p, k[:-4], float(val) * D)
        else:
            cur = getattr(p, k)
            setattr(p, k, type(cur)(val) if not isinstance(cur, bool) else val == '1')
    if a.set:
        v._fixups(before, keep=keys)                    # explicit sx/sy/rad win over the defaults
        v.reset()
    v.hs, v.cg, v.wire = a.hs, a.cg, a.wire
    v.auto_gain = not a.no_auto_gain
    L = v.look
    if a.bg:
        L.bg = a.bg
    for attr in ('zero', 'exposure', 'contrast', 'shading'):
        if getattr(a, attr) is not None:
            setattr(L, attr, getattr(a, attr))
    if a.solid_h is not None:
        L.solid_h = a.solid_h
    if a.solid_col:
        L.solid = tuple(parse_triple(a.solid_col))
    L.floor = not a.no_floor
    L.hatch = not a.no_hatch
    L.segment = a.segment
    if a.camera:
        az, el, dist = parse_triple(a.camera)
        v.orbit.az, v.orbit.el, v.orbit.dist = az, el, dist
        v.orbit.clamp()
    if a.zoom:
        v.v2.zoom = max(acam.ZOOM2D_MIN, min(acam.ZOOM2D_MAX, a.zoom))
    if a.pan:
        gx, gy = parse_triple(a.pan, 2)
        v.v2.cx, v.v2.cy = acam.grid_to_phys(v.m, gx, gy)
    if a.brush:
        v.sol.brush = a.brush
    if a.scene:
        v.sol.load(v.m, a.scene)
    if a.plate_text is not None or a.plate_screws is not None:
        v.sol.set_preset(v.m, plate_text=None if a.plate_text is None else bool(a.plate_text),
                         plate_screws=None if a.plate_screws is None else bool(a.plate_screws))
    if a.panels:
        for k in v.show:
            v.show[k] = k in [q.strip() for q in a.panels.split(',')]
    window = ti.ui.Window(TITLE, v.win, vsync=True, show_window=not a.shot)
    canvas, scene, gui = window.get_canvas(), window.get_scene(), window.get_gui()
    camera = ti.ui.Camera()
    camera.z_near(0.01)
    camera.z_far(100.0)
    v.camera = camera
    v.orbit.pose().apply(camera)
    if a.freelook:
        v.set_freelook(True)
    if a.shot:
        if a.walls_test:
            g, scr = programmatic_stroke(v, WALLS_TEST_PATH)
            back = [v.cursor_grid(*s) for s in scr]
            err = max(math.hypot(b[0] - q[0], b[1] - q[1]) for b, q in zip(back, g)) * v.m.base_half()
            print(f'walls-test ({"3D" if v.view3d else "2D"}): {len(v.m.scene.strokes[-1].points)} stroke points '
                  f'from {len(g)} cursor samples, solid cells {v.m.n_solid:,}, cursor round trip <= {err:.2e} cells, '
                  f'update median {np.median(v.sol.lat_ms):.2f} ms')
        v.step(a.steps)
        mk = []
        if a.marker:
            mk = [(tuple(parse_triple(a.marker)), (1.0, 0.0, 1.0))]
        if a.panels:
            # ImGui lays a new window out over one frame: without a first frame the panels are
            # missing from the saved image (round 1's "panel not in --shot images")
            v.draw(window, canvas, scene, camera, markers=mk)
            v.gui(gui)
            window.get_image_buffer_as_numpy()
        else:
            v.panel = False                 # a plain shot shows the field only
        v.draw(window, canvas, scene, camera, markers=mk)
        v.gui(gui)
        window.save_image(a.shot)
        cnt = v.R.n_idx if v.look.floor else v.R.n_dom
        print(f'wrote {a.shot}: N={v.m.N} {p.lattice} {p.shape} step {v.m.step_n}, '
              f'{v.ms_step:.3f} ms/step, {cnt // 3:,} triangles, solid cells {v.m.n_solid:,}')
        return
    digits = ain.DigitPoller(TITLE)
    while window.running:
        t0 = time.perf_counter()
        v.set_window_shape(window.get_window_shape(), t0)
        digits.active = v.sol.text is not None
        v.handle(ain.poll_window(window, ti.ui, digits))
        if v.view3d and v.freelook and v.sol.text is None:
            camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.RMB)
        if v.playing:
            v.step(v.sub)
        v.draw(window, canvas, scene, camera)
        if v.panel:
            v.gui(gui)
        try:
            window.show()
        except RuntimeError as ex:                       # a key GGUI cannot name (see app_input)
            ain._warn_once(str(ex))
        dt = (time.perf_counter() - t0) * 1e3
        v.ms_frame = dt if v.ms_frame == 0 else 0.9 * v.ms_frame + 0.1 * dt


if __name__ == '__main__':
    main()
