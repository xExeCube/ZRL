"""ZRL 2D wave membrane -- Taichi port, VIEWER (GGUI). The PRIMARY app since 29/09/2026 (the web app
is frozen). Version: membrane_core_MIRROR.VERSION (v1.0, 05/10/2026; CHANGELOG_MIRROR.md).

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
Typed values: "type a value" at the top of a panel, then click a slider: an entry opens under it
      (digits . - e, BackSpace, Return applies, Escape cancels; clamped to the slider's range; no
      hotkey fires while it is open). A main-panel checkbox adds a button under every slider instead.
Mouse (3D, orbit): RMB drag orbit, MMB drag or CTRL+RMB drag zoom, LMB drag orbit (draw mode off)
      (2D): RMB/LMB drag pan, MMB drag or CTRL+RMB drag zoom about the cursor
      draw mode: LMB paints walls, SHIFT+LMB erases
Freelook (F): GGUI's own camera keys W/A/S/D/Q/E + RMB look; no app hotkey uses those letters.
Round 4 (05/10/2026, NotesCLAUDE/walls_MIRROR.md section 11, vertex_choppiness_MIRROR.md section 7.4):
      B cycles only the boundaries that act here (bc_info(): the triangular 'array' has dirichlet /
      neumann = rigid ring / periodic = rhombic torus; Mur is not on the triangular lattice yet; a
      triangular web-app shape keeps its clamped ring), the domain edge (pinned / rigid) for masked
      shapes, wall types per object / brush / plate lettering, the source note, the drive-wavelength
      warning and the phase-lag readout. --version prints the version.
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
VERSION = mc.VERSION     # one version constant, in the core (05/10/2026)
TITLE = f'{mc.APP_NAME} v{VERSION}'
# the user's palette / transparency settings, kept across sessions (01/10/2026). Live runs load and
# auto-save it; --shot runs ignore it unless --settings names a file
SETTINGS_FILE = os.path.join(HERE, 'look_settingsMIRROR.json')
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
# scale mode (02/10/2026, NotesCLAUDE/scale_MIRROR.md): in 'domain' mode the playback speed is in crossings
# (r_in / c) per second at a NOMINAL frame rate (vsync); steps per frame follow N and Co so the apparent
# speed stays the same. A measured frame rate would feed back (more steps -> slower frames -> more steps).
FPS_NOMINAL = 60.0
SPF_MAX = 1024           # steps / frame cap in domain mode
AMBER = (0.95, 0.75, 0.3)
DIM = (0.55, 0.55, 0.60)
HEAD = (0.72, 0.62, 1.0)
# the core's readout levels (source_note(), phase_lag()) -> text colours (None = the normal text colour)
LEVEL_COL = {'warn': AMBER, 'dim': DIM, 'ok': None, 'info': None}
EDIT_SETTLE = 0.3        # s: readouts that cost a whole-grid pass (walls counts, the source note's clearance)
                         # wait while painting or during an object drag (two solid edits less than this
                         # apart), until the edits have rested this long; a single click edit updates them
                         # at once (05/10/2026)
# the boundary control's notes, short (the core's reasons are full sentences; the task's wording)
TRI_MUR_SHORT = 'mur: not on tri yet (absorbing-boundary round)'
TRI_MASKED_SHORT = ('the bc does not apply to this shape on the triangular lattice (its ring stays clamped): '
                    'use the edge control; the "array" shape (G) has dirichlet / neumann / periodic')
VTX_INSET_LABEL = {'wall': 'vertex inset: wall (3 sigma from both walls)',
                   'radial': "vertex inset: radial (the web app's)"}
SCALE_RANGES = {'sigma': (0.002, 0.3), 'freq': (0.005, 1.0), 'beam_width': (0.005, 1.0),
                'pk_len': (0.005, 1.0), 'pk_lambda': (0.005, 1.0)}   # domain-mode slider ranges (x r_in)
SCALE_LABELS = {'sigma': 'sigma / r_in', 'freq': 'lambda / r_in', 'beam_width': 'beam w / r_in',
                'pk_len': 'packet L / r_in', 'pk_lambda': 'packet lambda / r_in'}   # short: the panel cuts labels
N_SETTLE = 0.35         # s: the N slider applies once its value has rested this long (01/10/2026)
RESIZE_SETTLE = 0.3      # s: a new window size must hold this long before the 2D image buffer follows (01/10/2026)
TEXT_WRAP = 52           # characters per panel line (ImGui's 7 px font in a 400 px panel; 52 leaves room for the
                         # scrollbar a long panel shows -- 54 lost its last 2 characters there, 05/10/2026)
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
        self.auto_height = False                # 3D height scale / peak |u| (display only; 02/10/2026)
        self.cps = None                         # domain mode: crossings per second (None: from steps / frame)
        self._step_acc = 0.0                    # domain mode: the fractional steps carried to the next frame
        self.peak = 1.0                         # the auto gain's peak hold
        self.gain_ref = (0.0, 0)                # (largest |u| of this run, step it was updated at)
        self.gain_capped = False
        self.n_pending = None                   # (N, time) of an N slider value not applied yet
        self.sub = 4
        self.mesh_max = MESH_MAX
        self.cont_time = False                  # Co changes as a new time step (no time reflection)
        self.show_emitter = True                # the beam / packet emitter overlay (01/10/2026)
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
        self.ne = ain.NumEntry()                # typed slider values (01/10/2026)
        self.type_buttons = False               # a "^ type a value" button under every slider (else pick mode)
        self._src_note = (None, None)           # (key, note) cache of src_moved_note()
        self._edge_cache = (None, [])
        # round 4 (05/10/2026): the edge-type overlay, and readouts that are cached or wait for edits to rest
        self.show_edge = True                   # draw the rigid edge / ring line and the torus seam
        self._bnd_cache = (None, [])
        self._walls_cache = None                # (key, walls_info dict)
        self._note_last = None                  # the source note last shown
        self._note_cfg = None                   # ... and the settings it was computed for
        self._wrap_cache = {}
        self._fit_zoom = None                   # the 2D zoom fit_2d() set (the triangular array)
        self._fit_dist = None                   # the orbit distance fit_3d() set (the triangular array, 06/10/2026)
        # palette editor and settings file (01/10/2026)
        self.pal_edit = False
        self.settings_path = getattr(a, 'settings', None)
        self.settings_msg = None
        self._saved = None                      # (json text, time first seen) of the settings on disk / pending
        if self.settings_path:
            self.settings_msg = ar.load_settings(self.look, self.settings_path)
            self._saved = (self._settings_text(), 0.0)
        self.sync_domain()

    # ---- settings file --------------------------------------------------------------------------
    def _settings_text(self):
        import json
        return json.dumps(ar.look_settings(self.look), sort_keys=True)

    def autosave(self, now=None, delay=0.5):
        """Write the settings file once the palette settings have rested `delay` s after a change
        (a colour picker drag changes them every frame). Returns True when it wrote."""
        if not self.settings_path:
            return False
        now = time.perf_counter() if now is None else now
        txt = self._settings_text()
        if self._saved is not None and txt == self._saved[0] and self._saved[1] == 0.0:
            return False
        if self._saved is None or txt != self._saved[0]:
            self._saved = (txt, now)            # changed: wait for it to rest
            return False
        if now - self._saved[1] < delay:
            return False
        try:
            ar.save_settings(self.look, self.settings_path)
            self._saved = (txt, 0.0)
            self.settings_msg = None
            return True
        except OSError as ex:
            self.settings_msg = f'settings not saved ({ex})'
            self._saved = (txt, 0.0)
            return False

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
        if p.shape == 'array' and p.lattice == 'tri' and (before[0] != 'array' or before[1] != 'tri'):
            self.fit_2d()
            self.fit_3d()                       # 06/10/2026: the 3D view too
        elif before == ('array', 'tri'):
            # leaving the triangular array: back to 2D zoom 1 and the default orbit distance, each only
            # if the user has not zoomed since
            if self._fit_zoom is not None and self.v2.zoom == self._fit_zoom:
                self.v2.reset()
            if self._fit_dist is not None and self.orbit.dist == self._fit_dist:
                self.orbit.dist = acam.OrbitCam.DEFAULT['dist']
            self._fit_zoom = self._fit_dist = None

    def fit_3d(self):
        """The 3D orbit distance for the triangular 'array' (06/10/2026). The default camera frames the
        square array, [-1, 1]^2 in world units; the 60-degree rhombus is 3 wide, so at the default distance
        its near corners fell off the window and under the main panel (r4shot_layout_max_3d, 05/10). Moves
        the camera back along its current azimuth / elevation until no rhombus corner lies further from the
        window centre (in x or in y) than the square's corners do at the default distance: the rhombus
        then fills the window as the square does. No change for any other shape."""
        m, o = self.m, self.orbit
        if not (m.tri() and m.p.shape == 'array'):
            return
        asp = self.aspect()

        def reach(d, pts):
            o.dist = d
            pose = o.pose()
            rx = ry = 0.0
            for X, Z in pts:
                s = pose.world_to_screen((X, 0.0, Z), asp)
                if s is None:
                    return math.inf, math.inf
                rx, ry = max(rx, abs(2 * s[0] - 1)), max(ry, abs(2 * s[1] - 1))
            return rx, ry

        d0 = acam.OrbitCam.DEFAULT['dist']
        sx, sy = reach(d0, [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])
        pts = [acam.phys_to_world(m, x, y) for x, y in m.shape_poly()]
        d = d0
        while d < acam.DIST_MAX:
            rx, ry = reach(d, pts)
            if rx <= sx + 1e-9 and ry <= sy + 1e-9:
                break
            d *= 1.01
        o.dist = d
        o.clamp()
        self._fit_dist = o.dist

    def fit_2d(self):
        """The 2D view's zoom for the triangular 'array' (05/10/2026). Zoom 1 fits the array's height,
        but the whole 60-degree rhombus is 3c wide: 1.47 window heights, wider than the strip the side
        panels leave (1200 px of a 1600 x 1000 window), so its corners went under the main panel and
        off the window. Zooms out (centred) until it fits that strip; no change for any other shape."""
        m, v2 = self.m, self.v2
        if not (m.tri() and m.p.shape == 'array'):
            return
        W, H = self.win
        left = PANELS['main'][2] if (self.panel and self.show['main']) else 0.0
        right = PANELS['solids'][2] if (self.panel and (self.show['solids'] or self.show['align'])) else 0.0
        need = 3.0 * m.c() / (2 * m.c() * acam.HALF_FILL / H)       # its width in pixels at zoom 1
        v2.zoom = max(acam.ZOOM2D_MIN, min(1.0, 0.98 * W * (1.0 - left - right) / need))
        v2.cx = v2.cy = 0.0
        self._fit_zoom = v2.zoom

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

    # ---- boundary, edge (round 4, 05/10/2026; NotesCLAUDE/walls_MIRROR.md section 11) ------------------
    def next_bc(self, bi=None):
        """The boundary B moves to: the next of mc.BCS (in order) that ACTS here (bc_info()['options']).
        None on a triangular web-app shape, where the bc does not apply (its ring stays clamped)."""
        m, p = self.m, self.m.p
        if m.tri() and p.shape != 'array':
            return None
        opts = (bi or m.bc_info())['options']
        i = mc.BCS.index(p.bc) if p.bc in mc.BCS else -1
        for k in range(1, len(mc.BCS) + 1):
            b = mc.BCS[(i + k) % len(mc.BCS)]
            if b in opts:
                return b
        return None

    def cycle_bc(self):
        """Key B / the boundary button: the next boundary that acts here, then a reset (as before)."""
        nb = self.next_bc()
        if nb is None:
            self.flash(TRI_MASKED_SHORT)
            return
        if nb != self.m.p.bc:
            self.m.p.bc = nb
            self.reset()

    def set_edge(self, edge):
        """The domain mask's edge, pinned / rigid: the core rebuilds the link words and KEEPS the field."""
        if not self.m.bc_info()['edge_applies']:
            self.flash('edge: n/a here - the boundary (B) rules the ring')
            return
        self.m.set_edge(edge)
        self.flash(f'edge: {edge} (the field is kept)')

    def wrapped(self, text, indent='   ', width=TEXT_WRAP):
        """textwrap of a readout line (cached: the panel asks every frame), Latin-1 only (ImGui's font)."""
        key = (text, indent, width)
        hit = self._wrap_cache.get(key)
        if hit is None:
            safe = ''.join(ch if ord(ch) < 256 else ('-' if ch in '\u2013\u2014\u2212' else '?') for ch in text)
            hit = textwrap.wrap(safe, width, initial_indent=indent, subsequent_indent=indent) or [indent]
            if len(self._wrap_cache) > 512:
                self._wrap_cache.clear()
            self._wrap_cache[key] = hit
        return hit

    def lines(self, w, lines):
        """[(text, level)] from the core (source_note, phase_lag), wrapped and coloured by level."""
        for text, level in lines:
            col = LEVEL_COL.get(level)
            for piece in self.wrapped(text):
                if col is None:
                    w.text(piece)
                else:
                    w.text(piece, color=col)

    def _edit_resting(self, now=None):
        """False while painting, or while solid edits come in a burst (an object slider drag: two edits
        less than EDIT_SETTLE s apart, the second one less than EDIT_SETTLE s ago). A single click edit
        (a wall type, an added object) counts as resting, so its readouts update at once."""
        now = time.perf_counter() if now is None else now
        return not self.sol.painting and now - getattr(self.sol, 'edit_prev', -1e9) >= EDIT_SETTLE

    def walls_counts(self, now=None):
        """m.walls_info() for the walls panel (pinned / rigid cells, the walls kernel). Two whole-grid
        counts (~0.1 s at N = 8193): recomputed only when the masks, the scene, the edge or the bc
        changed, and only once painting / an object drag has rested EDIT_SETTLE s (no per-frame cost
        that scales with N)."""
        m, p = self.m, self.m.p
        sc = m.scene
        key = (m.N, getattr(m, '_mask_ver', 0), id(m.mask_np), id(m.domain_mask_np), m.n_solid, p.edge, p.bc,
               p.lattice, p.shape, p.stencil, sc.version if sc is not None else None)
        c = self._walls_cache
        # a new N / edge / bc / lattice / shape / stencil updates at once; a mask edit waits while it is a drag
        if c is not None and (c[0] == key or (c[0][5:10] == key[5:10] and c[0][0] == key[0]
                                              and not self._edit_resting(now))):
            return c[1]
        info = m.walls_info()
        self._walls_cache = (key, info)
        return info

    def source_note(self, now=None):
        """m.source_note() (the web app's srcNote: clearance, clip fraction, capped corners). The core
        caches it per reseed / mask change; its clearance search is O(clearance^2) (68 ms for a centred
        source at N = 4097), so while painting or right after an object edit the last note is shown."""
        m, p = self.m, self.m.p
        try:
            fresh = m._src_note[0] == m._src_note_key()
        except Exception:
            fresh = False
        # only a MASK change (painting, an object drag) may show the last note for a moment; any other
        # change (source, sigma, inset, shape, N, ...) recomputes at once
        cfg = (p.src, p.sigma, p.vtx_inset, p.vtx_drive, p.vtx_drive_bump, p.shape, p.lattice, m.N, p.rad, p.rot,
               p.rtheta, p.sx, p.sy, p.edge, p.bc)
        if (not fresh and self._note_last is not None and self._note_cfg == cfg
                and not self._edit_resting(now)):
            return self._note_last
        self._note_cfg = cfg
        try:
            self._note_last = m.source_note()
        except Exception as ex:                 # a readout must never break the panel
            self._note_last = {'lines': [(f'source note failed: {type(ex).__name__}: {ex}', 'warn')],
                               'driven_rest': False}
        return self._note_last

    def note_lines(self):
        """The source note's lines for the panel. A clip line that is not a warning loses its long
        parenthesis (and reads 'nothing' at exactly 0); the amber one keeps the core's full text."""
        note = self.source_note()
        out = []
        for t, lv in note.get('lines', []):
            if t.startswith('pulse cut off at t = 0') and lv != 'warn':
                t = 'pulse cut off at t = 0: nothing' if note.get('clip') == 0 else t.split(' (sum u^2')[0]
            out.append((t, lv))
        return out

    def gaussian_seeded(self):
        """A gaussian pulse is in the seed: src 'pulse', or vertex sources with their bumps."""
        p = self.m.p
        return p.src == 'pulse' or (p.src == 'vtx' and self.m.vtx_bumps())

    def boundary_segments(self):
        """The edge-type overlay (physical points): a RIGID mask edge (edge = 'rigid' on a masked
        shape) or a rigid ring (the triangular 'array' with neumann; the full square's Neumann ring,
        its mirror image, is drawn the same way) as a line in the rigid colour along the shape; a TORUS
        seam as a dashed violet line along the array edge. [] when hidden or for a pinned / clamped /
        Mur boundary."""
        if not self.show_edge:
            return []
        m, p = self.m, self.m.p
        key = (p.shape, p.rot, p.rad, p.rtheta, m.N, p.lattice, p.edge, p.bc, tuple(self.look.rigid))
        if self._bnd_cache[0] == key:
            return self._bnd_cache[1]
        segs = []
        try:
            bi = m.bc_info()
            kind = None
            if bi['edge_applies']:
                kind = 'rigid' if p.edge == 'rigid' else None
            elif bi['torus']:
                kind = 'torus'
            elif bi['effective'] == 'neumann':
                kind = 'rigid'
            if kind:
                P = m.shape_poly()
                col = tuple(self.look.rigid) if kind == 'rigid' else ar.TORUS_COL
                n = len(P)
                for k in range(n):
                    a, b = P[k], P[(k + 1) % n]
                    if kind == 'rigid':
                        segs.append((a, b, col))
                        continue
                    L = math.hypot(b[0] - a[0], b[1] - a[1])
                    nd = max(1, int(L / (m.c() / 24.0)))        # dashes of ~1/48 of the array width
                    for q in range(0, nd, 2):
                        t0, t1 = q / nd, min(1.0, (q + 1) / nd)
                        segs.append(((a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                                     (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1), col))
        except Exception as ex:                 # an overlay must never end the app
            self.flash(f'edge overlay failed: {type(ex).__name__}: {ex}')
        self._bnd_cache = (key, segs)
        return segs

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

    # ---- scale mode (02/10/2026, NotesCLAUDE/scale_MIRROR.md) -------------------------------------
    def set_scale(self, mode):
        """Switch 'cells' / 'domain' without a jump: the core converts the sizes (no reseed); the playback
        speed converts too (steps / frame <-> crossings / s at FPS_NOMINAL)."""
        m = self.m
        if mode == m.p.scale:
            return
        r, co = m.inner_R(), max(m.p.Co, 1e-9)
        if mode == 'domain':
            self.cps = self.sub * co * FPS_NOMINAL / r
            self._step_acc = 0.0
        else:
            self.sub = max(1, min(256, round(self.steps_per_frame())))
        m.set_scale(mode)

    def steps_per_frame(self):
        """Steps per frame (a float in domain mode: crossings / s x r_in / (Co x FPS_NOMINAL))."""
        m = self.m
        if m.p.scale != 'domain':
            return float(self.sub)
        if self.cps is None:
            self.cps = self.sub * max(m.p.Co, 1e-9) * FPS_NOMINAL / m.inner_R()
        return min(SPF_MAX, self.cps * m.inner_R() / (max(m.p.Co, 1e-9) * FPS_NOMINAL))

    def frame_steps(self):
        """The steps to run this frame: steps / frame in cells mode; in domain mode the fractional
        rate, with the remainder carried, so the average is exact even under 1 step / frame."""
        if self.m.p.scale != 'domain':
            return self.sub
        self._step_acc += self.steps_per_frame()
        n = int(self._step_acc)
        self._step_acc -= n
        return n

    def scale_size_slider(self, w, attr, label, lo, hi, live=True):
        """A size slider: in cells mode on the cell value, in domain mode on its fraction of r_in (the
        core derives the cells). Returns True when it moved. live=False: the caller resets."""
        m, p = self.m, self.m.p
        if p.scale != 'domain':
            v = w.slider_float(label, getattr(p, attr), lo, hi)
            if self.moved(v, getattr(p, attr)):
                setattr(p, attr, v)
                return True
            return False
        fa = next(f for a, f, _ in mc.SCALE_SIZES if a == attr)
        if getattr(p, fa) is None:
            m.fracs_from_cells(only_missing=True)
        flo, fhi = SCALE_RANGES[attr]
        v = w.slider_float(SCALE_LABELS[attr], getattr(p, fa), flo, fhi)
        if self.moved(v, getattr(p, fa)):
            setattr(p, fa, v)
            m.apply_scale()
            if attr == 'freq':
                m.harm = m.harmonics()
            return True
        return False

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
        self.ne.tick()
        if self.ne.open:
            # typed slider value (01/10/2026): every key goes to the entry, no hotkey fires
            self.ne.chars(fr.chars)
            for k in fr.presses:
                if k in ain.BUTTONS:
                    continue
                if self.ne.open:
                    _, msg = self.ne.key(k)
                    if msg:
                        self.flash(msg)
                # a key after Return / Escape in the same frame is dropped, not a hotkey
            self.mouse(fr)
            return
        if self.sol.text is not None:
            self.sol.chars(fr.chars)
        for k in fr.presses:
            if k in ain.BUTTONS:
                continue
            if self.ne.armed and k == 'Escape':             # pick mode off (typed slider values)
                self.flash(self.ne.key(k)[1])
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
            self.cycle_bc()
        elif k == 'g':
            self.cycle('shape', mc.SHAPES)
        elif k == 'm':
            self.cycle('medium', mc.MEDIA)
        elif k == 'o':
            self.cycle('src', mc.SOURCES)
        elif k == 'c':
            self.orbit.reset()
            self.v2.reset()
            self.fit_2d()                       # the triangular array: fit its width (05/10/2026)
            self.fit_3d()                       # ... and in 3D (06/10/2026)
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
        return self.cg / self.auto_peak()

    def height_scale(self):
        """The 3D height scale: the slider's value, or (auto height, 02/10/2026) divided by the same
        recent peak |u| as the auto colour gain, so the crest stays at the height of a unit pulse
        whatever N and the travelled distance (both scale modes)."""
        return self.hs / self.auto_peak() if self.auto_height else self.hs

    def auto_peak(self):
        """The peak |u| the auto gain and the auto height divide by (sets self.peak, self.gain_capped)."""
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
        return self.peak

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
        self.R.set_params(self.look, self.colour_gain(), self.height_scale(), self.v2.kernel_cx(m, *self.win), self.v2.cy, pxs)
        canvas.set_background_color(ar.SEG['bg'] if self.look.segment else self.look.bg_col())
        ring = self.brush_ring()
        edges = self.boundary_segments() + self.edge_segments()
        beam = self.beam_segments()
        if self.view3d:
            if not self.freelook:
                self.orbit.pose().apply(camera)
            scene.set_camera(camera)
            eye = self.pose().eye
            segs = []
            w = lambda p, h=0.004: (p[0] / m.c(), h, -p[1] / m.c())
            for a, b, c in edges:
                segs.append((w(a), w(b), c))
            if beam:
                # the emitter: drawn above the surface like the brush ring (each point slid along its
                # view ray, so it covers the pixels of its rest-plane position)
                hb = math.copysign(self.look.solid_h + 0.012, eye[1]) if abs(eye[1]) > 1e-6 else 0.0
                tb = hb / eye[1] if abs(eye[1]) > 1e-6 else 0.0
                lb = lambda P: (P[0] + (eye[0] - P[0]) * tb, hb, P[2] + (eye[2] - P[2]) * tb)
                segs += [(lb(w(a, 0.0)), lb(w(b, 0.0)), c) for a, b, c in beam]
            if ring:
                # painting happens on the REST plane Y = 0 (where the cursor ray meets it). The ring
                # is drawn above the surface (so solids and crests do not hide it) but each point is
                # slid along its view ray, so it covers exactly the pixels of the Y = 0 ring
                col = (1.0, 0.35, 0.3) if (self.sol.erase or self.drag_erase()) else (0.95, 0.95, 0.35)
                hr = math.copysign(self.look.solid_h + 0.01, eye[1]) if abs(eye[1]) > 1e-6 else 0.0
                t = hr / eye[1] if abs(eye[1]) > 1e-6 else 0.0
                lift = lambda P: (P[0] + (eye[0] - P[0]) * t, hr, P[2] + (eye[2] - P[2]) * t)
                segs += [(lift(w(ring[k], 0.0)), lift(w(ring[k + 1], 0.0)), col) for k in range(len(ring) - 1)]
            # GGUI draws in call order and its lines ignore alpha (measured 01/10/2026): with a
            # translucent surface the overlay goes FIRST so it shows through the surface
            early = self.look.opacity < 1.0
            if early:
                self.R.lines3d(scene, segs)
            self.R.draw3d(m, self.look, scene, eye, self.wire)
            if not early:
                self.R.lines3d(scene, segs)
            if markers:
                self.R.markers3d(scene, list(markers))
            canvas.scene(scene)
        else:
            self.R.draw2d(m, canvas)
            W, H = self.win
            s = lambda p: self.v2.phys_to_screen(m, p[0], p[1], W, H)
            segs = [(s(a), s(b), c) for a, b, c in edges + beam]
            if ring:
                col = (1.0, 0.35, 0.3) if (self.sol.erase or self.drag_erase()) else (0.95, 0.95, 0.35)
                segs += [(s(ring[k]), s(ring[k + 1]), col) for k in range(len(ring) - 1)]
            self.R.lines2d(canvas, segs)

    def beam_controls(self, w, moved):
        """The directional sources' controls (01/10/2026): angle, width, one-sided / aim (beam),
        length and wavelength (packet), and a readout of what the lattice makes of them. The beam's
        parameters apply LIVE (the core rebuilds its tables on the next step); the packet's
        relaunch it (a reset)."""
        m, p = self.m, self.m.p
        live = p.src == 'beam'
        a = w.slider_float('beam angle (deg)', p.beam_angle, -180.0, 180.0)
        if moved(a, p.beam_angle):
            p.beam_angle = a
            if not live:
                self.reset()
        if self.scale_size_slider(w, 'beam_width', 'beam width w (cells)', 1.0, 128.0) and not live:
            self.reset()
        g = w.checkbox('aim the energy (group velocity)', p.beam_group)
        if g != p.beam_group:
            p.beam_group = g
            if not live:
                self.reset()
        if live:
            o = w.checkbox('one-sided (cancel the backward lobe)', p.beam_onesided)
            if o != p.beam_onesided:
                p.beam_onesided = o
            b = m.beam or {}
            inf = b.get('info') or {}
            lam = inf.get('lam')
            if lam:
                w.text(f'   lambda {lam:.1f} cells, {len(b.get("cells", ()))} cells driven'
                       + (f', {inf.get("n_clipped")} clipped' if inf.get('n_clipped') else ''))
                w.text(f'   phase fronts at {inf.get("theta_k", p.beam_angle):.2f} deg', color=(0.55, 0.55, 0.60))
                if lam < mc.PPW_MIN:
                    w.text(f'   under {mc.PPW_MIN} cells per wavelength: strong dispersion', color=(0.95, 0.75, 0.3))
            elif inf.get('warn'):
                w.text('   ' + inf['warn'], color=(0.95, 0.75, 0.3))
        else:
            if self.scale_size_slider(w, 'pk_len', 'packet length L (cells)', 1.0, 128.0):
                self.reset()
            if self.scale_size_slider(w, 'pk_lambda', 'packet wavelength (cells)', 3.0, 128.0):
                self.reset()
            inf = m.packet_info or {}
            if inf:
                w.text(f'   group speed {math.hypot(*inf.get("vg_mean", (0, 0))):.4f} cells/step', color=(0.55, 0.55, 0.60))
        self.show_emitter = w.checkbox('draw the emitter', self.show_emitter)

    def beam_segments(self):
        """The emitter overlay: the aperture (cyan) and an arrow along the requested direction
        (orange), physical coordinates; [] when hidden or for point sources."""
        if not self.show_emitter:
            return []
        g = self.m.beam_geometry()
        if g is None:
            return []
        (x0, y0), a, b, (xe, ye) = g
        ca, cr = (0.3, 0.95, 1.0), (1.0, 0.6, 0.15)
        dx, dy = xe - x0, ye - y0
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        hl = 0.18 * L
        h1 = (xe - hl * (ux * 0.866 - uy * 0.5), ye - hl * (uy * 0.866 + ux * 0.5))
        h2 = (xe - hl * (ux * 0.866 + uy * 0.5), ye - hl * (uy * 0.866 - ux * 0.5))
        return [(a, b, ca), ((x0, y0), (xe, ye), cr), ((xe, ye), h1, cr), ((xe, ye), h2, cr)]

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
        # review v3 01/10/2026: the timed recount alone cost 36 ms at N = 8193 every 0.5 s (a visible
        # stutter, measured AppCLAUDE/probe_spike_v3_MIRROR.py); it now runs only when the masks or
        # the solid count changed (still at most twice a second while painting)
        m = self.m
        t, free, solid = self._cells[:3]
        key = (m.N, id(m.domain_mask_np), id(m.mask_np), m.n_solid)
        now = time.perf_counter()
        if (len(self._cells) < 4 or self._cells[3] != key) and now - t > 0.5:
            free, solid = m.domain_cells(), m.solid_cells()
            self._cells = (now, free, solid, key)
        return free, solid

    def gui(self, g):
        if self.show['main']:
            x, y, w_, h_ = PANELS['main']
            with g.sub_window(PANEL_TITLES['main'], x, y, w_, h_) as w:
                self.gui_main(self.typed(w, 'main', head=f'{mc.APP_NAME} v{VERSION}  ({mc.VERSION_DATE})'))
        if self.show['solids']:
            x, y, w_, h_ = PANELS['solids']
            with g.sub_window(PANEL_TITLES['solids'], x, y, w_, h_) as w:
                self.sol.gui(self.typed(w, 'solids'), self)
        if self.show['align']:
            x, y, w_, h_ = PANELS['align']
            with g.sub_window(PANEL_TITLES['align'], x, y, w_, h_) as w:
                w = self.typed(w, 'align')
                if alignment is not None and hasattr(alignment, 'gui_section'):
                    try:
                        if self.align_panel is None and hasattr(alignment, 'AlignPanel'):
                            self.align_panel = alignment.AlignPanel()
                            self.align_panel.wrap = 52     # its default 64 overflows the panel (52 since 01/10/2026)
                        alignment.gui_section(w, self)
                    except Exception as ex:
                        w.text(f'alignment panel error: {type(ex).__name__}: {ex}', color=(0.95, 0.4, 0.4))
                else:
                    w.text(f'alignment panel unavailable ({ALIGN_ERR or "no gui_section"})')
        if self.show['ledger']:
            x, y, w_, h_ = PANELS['ledger']
            with g.sub_window(PANEL_TITLES['ledger'], x, y, w_, h_) as w:
                w = self.typed(w, 'ledger')
                if ledger is not None and hasattr(ledger, 'gui_section'):
                    try:
                        ledger.gui_section(w, self)
                    except Exception as ex:
                        w.text(f'ledger panel error: {type(ex).__name__}: {ex}', color=(0.95, 0.4, 0.4))
                else:
                    w.text('ledger: not available yet (ledger_MIRROR.py ' +
                           (f'import failed: {LEDGER_ERR})' if LEDGER_ERR else 'has no gui_section)'))

    def typed(self, w, panel, head=None):
        """The panel with typed-value entries for its sliders (app_input_MIRROR.TypedGui). head: a
        first line above the pick-mode button (the main panel's version line, 05/10/2026)."""
        def start():
            self.sol.text = None                # one entry at a time
        if head:
            w.text(head, color=HEAD)
        tg = ain.TypedGui(w, self.ne, panel, self.type_buttons, self.flash, start)
        tg.header()                             # pick mode: one row per panel (01/10/2026)
        return tg

    def src_moved_note(self):
        """'source moved out of a solid to (x, y)' when the requested source point (source x/y)
        lies in a solid (or 'into the domain' when it lies outside it), else None. Prefers the core's own flag (m.src_moved) when it has one;
        otherwise the requested point's cell is looked up in solid_np. Cached per state."""
        m, p = self.m, self.m.p
        src = getattr(m, 'src', None)
        key = (p.sx, p.sy, m.N, p.lattice, p.shape, p.rad, p.rot, id(m.solid_np), m.n_solid,
               None if src is None else tuple(src[:2]))
        if self._src_note[0] == key:
            return self._src_note[1]
        note = None
        try:
            ir = m.inner_R()
            i0, j0 = m.round_cell(*m.xy_to_cell(p.sx * ir, p.sy * ir))
            in_solid = m.solid_np is not None and bool(m.solid_np[i0, j0])
            flag = getattr(m, 'src_moved', None)
            if flag is None:                # no core flag: the requested cell vs the cell used
                flag = src is not None and (int(src[0]), int(src[1])) != (i0, j0)
            if flag and src is not None and ir > 0:
                x, y = m.cell_xy(src[0], src[1])
                why = 'out of a solid' if in_solid else 'into the domain'
                note = f'   source moved {why} to ({x / ir:.3f}, {y / ir:.3f})'
        except Exception as ex:             # a readout must never break the panel
            note = f'   source check failed: {type(ex).__name__}'
        self._src_note = (key, note)
        return note

    # ---- palette editor and transparency (01/10/2026, Claude Opus 5.5) --------------------------
    def gui_palette(self, w, L):
        moved = self.moved
        if w.button(f'palette: {L.palette}  (click: next)'):
            L.cycle_palette()
        self.pal_edit = w.checkbox('edit the custom palette', self.pal_edit)
        if self.pal_edit:
            self.gui_custom(w, L)
        # transparency: real alpha in 3D (GGUI 1.7.4 blends RGBA vertex colours)
        lab = 'surface opacity' if self.view3d else 'field opacity (over the page)'
        v = w.slider_float(lab, L.opacity, 0.0, 1.0)
        if moved(v, L.opacity):
            L.opacity = v
        if self.view3d:
            v = w.slider_float('grid lines (opacity, 0 = off)', L.grid, 0.0, 1.0)
            if moved(v, L.grid):
                L.grid = v
            if L.grid > 0:
                L.grid_every = w.slider_int('grid every (mesh vertices)', L.grid_every, 1, 64)
                w.text(f'   = a line every {L.grid_every * self.R.k} cells', color=(0.7, 0.7, 0.75))
                L.grid_field = w.checkbox('grid in the field colours', L.grid_field)
                if not L.grid_field:
                    gc = w.color_edit_3('grid colour', L.grid_col)
                    L.grid_col = tuple(float(c) for c in gc)
            if L.opacity < 1 or L.grid > 0:
                w.text('   real alpha blending; grid lines behind a crest stay hidden', color=(0.7, 0.7, 0.75))
        elif L.grid > 0:
            w.text('   grid lines: 3D view only', color=(0.7, 0.7, 0.75))
        if self.settings_path:
            w.text(f'   palette saved to {os.path.basename(self.settings_path)}' if not self.settings_msg
                   else f'   {self.settings_msg}', color=(0.7, 0.7, 0.75))

    def gui_custom(self, w, L):
        """Edit the custom palette's stops: colour of each, position of the interior ones (not the
        ends and not zero), add / remove, copy the current palette in, reset. Any edit shows the
        custom palette; the zero stop's brightness is the zero level."""
        moved = self.moved
        st = L.stops() if L.palette == 'custom' else ar.clean_stops(L.custom)
        w.text('   t = -1 negative end, 0 zero, +1 positive end', color=(0.7, 0.7, 0.75))
        new, changed, remove = [], False, None
        for k, (t, c) in enumerate(st):
            lab = {-1.0: 'negative end', 0.0: 'zero', 1.0: 'positive end'}.get(t, f'stop {k}')
            cc = tuple(float(x) for x in w.color_edit_3(f'{lab} (t {t:+.2f})##pal{k}', c))
            if any(moved(a, b) for a, b in zip(cc, c)):
                c, changed = cc, True
            if t not in (-1.0, 0.0, 1.0):
                lo = st[k - 1][0] + 0.01
                hi = st[k + 1][0] - 0.01
                if hi > lo:
                    tt = w.slider_float(f'stop {k} position', t, lo, hi)
                    if moved(tt, t):
                        t, changed = float(tt), True
                if w.button(f'remove stop {k}##palrm{k}'):
                    remove = k
            new.append((t, c))
        if remove is not None:
            del new[remove]
            changed = True
        if len(new) < ar.KMAX:
            for side, lab in ((-1, 'add a stop (negative side)'), (1, 'add a stop (positive side)')):
                if w.button(lab):
                    ts = [t for t, _ in new if (t <= 0 if side < 0 else t >= 0)]
                    gaps = [(ts[i + 1] - ts[i], i) for i in range(len(ts) - 1)]
                    g, i = max(gaps)
                    tn = round((ts[i] + ts[i + 1]) / 2, 4)
                    arr = np.array([q[1] for q in new])
                    cn = tuple(float(np.interp(tn, [q[0] for q in new], arr[:, ch])) for ch in range(3))
                    new.append((tn, cn))
                    changed = True
        if w.button('copy the current palette into custom'):
            new, changed = list(L.stops()), True
        if w.button('reset the custom palette'):
            new, changed = list(ar.DEFAULT_CUSTOM), True
        if changed:
            L.custom = ar.clean_stops(new)
            L.palette = 'custom'
            L.zero = ar.zero_of(L.custom)

    def gui_boundary(self, w):
        """The boundary control (05/10/2026): the EFFECTIVE bc (bc_info()), B cycles only the bcs that act
        here, the ring in words, the unavailable ones with their reason; then the domain edge."""
        m, p = self.m, self.m.p
        bi = m.bc_info()
        w.text(f'boundary: {bi["effective"]}' + (f'   ({p.bc} n/a here)' if bi['fallback'] else ''))
        nb = self.next_bc(bi)
        if nb is None:
            if w.button('boundary  [B]: does not apply to this shape##bc'):
                self.cycle_bc()
        elif w.button(f'boundary  [B]: {p.bc}  (click: {nb})##bc'):
            self.cycle_bc()
        for piece in self.wrapped(bi['ring']):
            w.text(piece, color=DIM)
        if m.tri() and p.shape != 'array':
            for piece in self.wrapped(TRI_MASKED_SHORT):
                w.text(piece, color=DIM)
        elif 'mur' in bi['unavailable']:
            txt = TRI_MUR_SHORT + (' - falls back to the clamped ring' if p.bc == 'mur' else '')
            for piece in self.wrapped(txt):
                w.text(piece, color=AMBER if p.bc == 'mur' else DIM)
        self.gui_edge(w, bi)

    def gui_edge(self, w, bi=None):
        """The domain edge (Params.edge) for masked shapes; greyed text where the bc rules instead."""
        m, p = self.m, self.m.p
        bi = bi or m.bc_info()
        if bi['edge_applies']:
            other = 'rigid' if p.edge == 'pinned' else 'pinned'
            if w.button(f'edge: {p.edge}  (click: {other})##edge'):
                self.set_edge(other)
            w.text('   ' + ('rigid: zero slope, echo upright' if p.edge == 'rigid'
                            else 'pinned: u = 0, echo inverted') + ' (field kept)', color=DIM)
        else:
            who = 'the array' if p.shape == 'array' else 'the full square'
            for piece in self.wrapped(f'edge: n/a - {who} has no mask edge: the boundary (B) rules', indent=''):
                w.text(piece, color=DIM)

    def gui_main(self, w):
        m, p = self.m, self.m.p
        moved = self.moved
        w.text(f'{"playing" if self.playing else "paused"}   step {m.step_n}   t = {m.t:.1f}'
               + (f'   = {m.crossings():.3f} r_in/c' if p.scale == 'domain' else ''))
        if self.flash_msg and time.perf_counter() - self.flash_t < 8.0:
            for piece in textwrap.wrap(self.flash_msg, TEXT_WRAP)[:4]:
                w.text(piece, color=(0.95, 0.8, 0.35))
        tb = w.checkbox('a "type" button under every slider', self.type_buttons)
        if tb != self.type_buttons:
            self.type_buttons = tb
            if not tb:
                self.ne.cancel()
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
        if p.shape in mc.FIXED_SHAPES:
            w.text(f'domain: {p.shape} (the whole lattice array)   R = {m.dom_R():.2f}')
        else:
            w.text(f'domain: {p.shape}   radius {p.rad:.2f}   R = {m.dom_R():.2f}')
        if w.button('shape  [G]'):
            self.cycle('shape', mc.SHAPES)
        self.gui_boundary(w)
        w.text(f'medium: {p.medium}')
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
            if p.vtx_drive:
                # 05/10/2026: driven vertices start from rest (the port); on = the web app's seeded bumps
                vb = w.checkbox('seed pulses under the driven vertices (the web app)', bool(p.vtx_drive_bump))
                if vb != bool(p.vtx_drive_bump):
                    p.vtx_drive_bump = vb
                    self.reset()
            if w.button(VTX_INSET_LABEL.get(p.vtx_inset, f'vertex inset: {p.vtx_inset}')):   # click: the other
                self.cycle('vtx_inset', mc.VTX_INSETS)
            self.lines(w, self.note_lines())               # the source note, under the inset button
        mn = m.mode_note() if p.src == 'mode' else None
        if mn:
            self.lines(w, [('not an exact eigenmode here: ' + mn, 'warn')])
        if p.src in mc.BEAM_SOURCES:
            self.beam_controls(w, moved)
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
        # damping lowers the real limit: C_max(d) = C_max(0) sqrt(1 - d/2) (core co_limit, 01/10/2026)
        w.text(f'   C_max = {m.co_limit():.4f}' + (' (damped)' if p.damp > 0 else '')
               + ('   UNSTABLE' if p.Co > m.co_limit() + 1e-12 else ''))
        dmp = w.slider_float('damping', p.damp, 0.0, 0.02)
        if moved(dmp, p.damp):
            p.damp = dmp
        if w.button(f'scale: {p.scale} (sizes in ' + ('cells' if p.scale == 'cells' else 'fractions of r_in')
                    + ')  -> ' + ('domain' if p.scale == 'cells' else 'cells')):
            self.set_scale('domain' if p.scale == 'cells' else 'cells')
        if p.scale == 'domain':
            self.steps_per_frame()                     # sets cps on first use
            c = w.slider_float('crossings / s', self.cps, 0.01, 2.0)
            if moved(c, self.cps):
                self.cps = c
            w.text(f'   r_in = {m.inner_R():.1f} cells   {self.steps_per_frame():.2f} steps / frame at {FPS_NOMINAL:g} fps',
                   color=(0.55, 0.55, 0.60))
        else:
            self.sub = w.slider_int('steps / frame', self.sub, 1, 256)
        for attr, lab, lo, hi in (('rad', 'radius', 0.1, 1.5), ('sigma', 'pulse sigma', 0.8, 12.0),
                                  ('sx', 'source x', -1.0, 1.0), ('sy', 'source y', -1.0, 1.0)):
            if attr == 'sigma':                        # cells, or x r_in in domain mode (02/10/2026)
                if self.scale_size_slider(w, 'sigma', lab, lo, hi):
                    self.reset()
                continue
            if attr == 'rad' and p.shape in mc.FIXED_SHAPES:
                w.text('radius: fixed (the whole array)', color=DIM)
                continue
            v = w.slider_float(lab, getattr(p, attr), lo, hi)
            if moved(v, getattr(p, attr)):
                setattr(p, attr, v)
                self.reset()
        note = self.src_moved_note()
        if note:
            w.text(note, color=(0.95, 0.75, 0.3))      # amber: GGUI cannot recolour a slider
        if p.src != 'vtx':
            self.lines(w, self.note_lines())            # point sources: the note under source x / y
        if p.shape in mc.FIXED_SHAPES:
            w.text('rotation: fixed (the lattice array)', color=DIM)
        else:
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
        if p.src in ('cont', 'slit', 'beam') or (p.src == 'vtx' and p.vtx_drive):
            if self.scale_size_slider(w, 'freq', 'freq (cyc/step)', 0.005, 0.25):
                m.harm = m.harmonics()
            if p.scale == 'domain':
                w.text(f'   lambda {p.Co / max(p.freq, 1e-12):.1f} cells   freq {p.freq:.4f} cyc/step',
                       color=(0.55, 0.55, 0.60))
        # domain mode: under-resolved sizes; cells mode (05/10/2026): a hard drive under 8 cells per
        # wavelength (the two never both show: drive_warnings() is [] in domain mode)
        for wn in m.scale_warnings() + m.drive_warnings():
            for piece in self.wrapped(wn, indent='')[:3]:
                w.text(piece, color=AMBER)
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
        self.gui_palette(w, L)
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
        self.auto_height = w.checkbox('auto height (height scale / peak |u|)', self.auto_height)
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
        bc = w.color_edit_3('page colour (outside the array)', L.bg_col())
        if any(moved(float(a), float(b)) for a, b in zip(bc, L.bg_col())):
            L.bg_rgb = tuple(float(c) for c in bc)
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
        if self.gaussian_seeded():
            # 05/10/2026: how far the grid's fronts trail the exact ones (discrete dispersion, cached)
            try:
                self.lines(w, m.phase_lag()['lines'])
            except Exception as ex:
                w.text(f'phase lag failed: {type(ex).__name__}', color=AMBER)


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
    if any(q is None for q in scr):
        # review v3 01/10/2026: a camera that has part of the path behind the eye (e.g. --walls-test
        # with a low, close --camera) crashed with "NoneType is not subscriptable" in Frame()
        v.set_draw(was)
        v.panel = panel
        raise ValueError('programmatic stroke: part of the path is behind the camera in this view')
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
    ap = argparse.ArgumentParser(description=f'{TITLE} ({mc.VERSION_DATE})')
    ap.add_argument('--version', action='version', version=f'{TITLE} ({mc.VERSION_DATE})')
    ap.add_argument('--arch', default='cuda')
    ap.add_argument('--f32', action='store_true')
    ap.add_argument('--N', type=int, default=513)
    ap.add_argument('--shot', default=None, help='render to this PNG without a window, then exit')
    ap.add_argument('--win', default=None, help='window size W,H (e.g. 2560,1377: a maximised window)')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--view', default='3d', choices=['3d', '2d'])
    ap.add_argument('--hs', type=float, default=1.0, help='height scale')
    ap.add_argument('--cg', type=float, default=1.0, help='colour gain')
    ap.add_argument('--no-auto-gain', action='store_true', help='colour gain as given (not / peak |u|)')
    ap.add_argument('--auto-height', action='store_true', help='3D height scale / recent peak |u|')
    ap.add_argument('--scale', default=None, choices=mc.SCALES,
                    help="sizes in cells (default) or as fractions of r_in ('domain'; set fractions with "
                         "--set sigma_f=0.05 lam_f=0.1 ...)")
    ap.add_argument('--cps', type=float, default=None, help='domain mode: playback speed in crossings / s')
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
    # palette and transparency (01/10/2026)
    ap.add_argument('--palette', default=None, choices=ar.PALETTE_ORDER, help='palette preset (or custom)')
    ap.add_argument('--stops', default=None, help="custom palette stops 't:r,g,b;t:r,g,b;...' (sets --palette custom)")
    ap.add_argument('--opacity', type=float, default=None, help='3D surface opacity (2D: field over the page)')
    ap.add_argument('--grid', type=float, default=None, help='3D grid-line opacity (0 = off)')
    ap.add_argument('--grid-every', type=int, default=None, help='3D: a grid line every k mesh vertices')
    ap.add_argument('--grid-col', default=None, help='3D grid colour r,g,b')
    ap.add_argument('--grid-field', action='store_true', help='3D grid lines in the field colours')
    ap.add_argument('--bg-col', default=None, help='custom page colour r,g,b')
    ap.add_argument('--settings', default=None,
                    help='palette settings JSON (live runs default to look_settingsMIRROR.json; shots use none)')
    ap.add_argument('--no-settings', action='store_true', help='live run without loading / saving the settings file')
    # solids
    ap.add_argument('--scene', default=None, help='load a solids scene (JSON; a bare name looks in ScenesCLAUDE)')
    ap.add_argument('--plate-text', type=int, default=None, choices=[0, 1])
    ap.add_argument('--plate-screws', type=int, default=None, choices=[0, 1])
    ap.add_argument('--plate-wall', default=None, choices=list(so.WALL_TYPES),
                    help='the Unlicensed lettering and screws: pinned (default) or rigid')
    ap.add_argument('--brush-wall', default=None, choices=list(so.WALL_TYPES),
                    help='the wall type of painted strokes (--walls-test)')
    ap.add_argument('--walls-test', action='store_true',
                    help='paint a wall with a programmatic mouse stroke (in the chosen view) before stepping')
    ap.add_argument('--brush', type=float, default=None, help='brush radius in cells')
    ap.add_argument('--marker', default=None, help='3D: draw a marker particle at world x,y,z')
    ap.add_argument('--panels', default=None, help="show panels in the shot: 'main' or e.g. main,solids,align,ledger")
    # '--camera -135,60,2' would read '-135,60,2' as an option: glue such values to their flag
    argv, VALS = list(sys.argv[1:]), ('--camera', '--pan', '--marker', '--solid-col', '--stops')
    for k in range(len(argv) - 2, -1, -1):
        if argv[k] in VALS and argv[k + 1][:1] == '-' and argv[k + 1][1:2].isdigit():
            argv[k:k + 2] = [argv[k] + '=' + argv[k + 1]]
    a = ap.parse_args(argv)
    a.win = tuple(int(q) for q in parse_triple(a.win, 2)) if a.win else WIN
    if a.settings is None and not a.shot and not a.no_settings:
        a.settings = SETTINGS_FILE
    mc.init_taichi(a.arch, fp64=not a.f32, quiet=True)
    v = Viewer(a)
    p = v.m.p
    before = (p.shape, p.lattice)
    keys = set()
    if a.scale == 'domain' and a.set:
        # fractions given with --set win; the missing ones come from the cell sizes at the FINAL N
        # (the reseed below), so '--scale domain --set N=2049' keeps the default picture of N = 2049
        p.scale = 'domain'
    for kv in a.set:
        k, val = kv.split('=')
        keys.add(k[:-4] if k.endswith('_deg') else k)
        if k.endswith('_deg'):
            setattr(p, k[:-4], float(val) * D)
        else:
            cur = getattr(p, k)
            typ = float if cur is None else type(cur)   # the scale fractions start as None (02/10/2026)
            setattr(p, k, typ(val) if not isinstance(cur, bool) else val == '1')
    if a.set:
        v._fixups(before, keep=keys)                    # explicit sx/sy/rad win over the defaults
        v.reset()
    v.hs, v.cg, v.wire = a.hs, a.cg, a.wire
    v.auto_gain = not a.no_auto_gain
    v.auto_height = a.auto_height
    if a.scale:
        v.set_scale(a.scale)                    # converts the sizes at the current N (no jump)
    if a.cps is not None:
        v.set_scale('domain')
        v.cps = a.cps
    L = v.look
    if a.bg:
        L.bg = a.bg
    if a.stops:
        L.custom = ar.clean_stops([(float(t), tuple(parse_triple(c))) for t, c in
                                   (q.split(':') for q in a.stops.split(';') if q.strip())])
        L.set_palette('custom')
    elif a.palette:
        L.set_palette(a.palette)
    for attr in ('opacity', 'grid'):
        if getattr(a, attr) is not None:
            setattr(L, attr, getattr(a, attr))
    if a.grid_every:
        L.grid_every = a.grid_every
    if a.grid_col:
        L.grid_col = tuple(parse_triple(a.grid_col))
    if a.grid_field:
        L.grid_field = True
    if a.bg_col:
        L.bg_rgb = tuple(parse_triple(a.bg_col))
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
    if a.brush_wall:
        v.sol.brush_wall = a.brush_wall
    if a.scene:
        v.sol.load(v.m, a.scene)
    if a.plate_text is not None or a.plate_screws is not None or a.plate_wall:
        v.sol.set_preset(v.m, plate_text=None if a.plate_text is None else bool(a.plate_text),
                         plate_screws=None if a.plate_screws is None else bool(a.plate_screws),
                         plate_wall=a.plate_wall)
    if a.panels:
        for k in v.show:
            v.show[k] = k in [q.strip() for q in a.panels.split(',')]
    # a run that starts on the triangular 'array' (--set lattice=tri shape=array) fits it at once, as G / L
    # do (06/10/2026); a --shot refits the 2D view once its panels are laid out
    if not a.camera:
        v.fit_3d()
    if a.zoom is None and a.pan is None:
        v.fit_2d()
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
            try:
                g, scr = programmatic_stroke(v, WALLS_TEST_PATH)
            except ValueError as ex:
                g = None
                print(f'walls-test skipped: {ex}')
        if a.walls_test and g is not None:
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
        if a.zoom is None and a.pan is None:
            v.fit_2d()                      # the triangular array fits the strip the panels leave
        v.draw(window, canvas, scene, camera, markers=mk)
        v.gui(gui)
        window.save_image(a.shot)
        cnt = v.R.n_idx if v.look.floor else v.R.n_dom
        print(f'wrote {a.shot}: {TITLE}, N={v.m.N} {p.lattice} {p.shape} bc {v.m.bc_effective()} edge {p.edge} '
              f'step {v.m.step_n}, {v.ms_step:.3f} ms/step, {cnt // 3:,} triangles, solid cells {v.m.n_solid:,}')
        return
    digits = ain.DigitPoller(TITLE)
    while window.running:
        t0 = time.perf_counter()
        v.set_window_shape(window.get_window_shape(), t0)
        digits.active = v.sol.text is not None or v.ne.open
        v.handle(ain.poll_window(window, ti.ui, digits))
        if v.view3d and v.freelook and v.sol.text is None and not v.ne.open:
            camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.RMB)
        if v.playing:
            n = v.frame_steps()                     # steps / frame, or the domain mode's speed (02/10/2026)
            if n:
                v.step(n)
        v.draw(window, canvas, scene, camera)
        if v.panel:
            v.gui(gui)
        try:
            window.show()
        except RuntimeError as ex:                       # a key GGUI cannot name (see app_input)
            ain._warn_once(str(ex))
        v.autosave()
        dt = (time.perf_counter() - t0) * 1e3
        v.ms_frame = dt if v.ms_frame == 0 else 0.9 * v.ms_frame + 0.1 * dt
    v.autosave(delay=0.0)                       # a change still resting when the window closed
    v.autosave(delay=0.0)


if __name__ == '__main__':
    main()
