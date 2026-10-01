"""ZRL 2D wave membrane -- Taichi port, VIEWER: painted walls, solid objects, text entry, scenes.

The user's request (29/09/2026): "If anything outside is treated as a wall, can you implement a hand
drawing function for walls inside the domain?" -- plus objects (rect, circle, n-gon, text) and the
Unlicensed plate's toggles. All solid geometry lives in solids_MIRROR.SolidScene (grid units, see
that module); this file is the interactive layer:
  - painting: begin_stroke / extend_stroke / end_stroke, each followed by m.update_solids(box) --
    only the new segment's cell box is re-rasterised and uploaded, the field elsewhere keeps running.
    A stroke segment is a capsule between two mouse samples, so fast strokes stay continuous (the
    samples are joined, not dotted); samples closer than max(0.35, 0.25 r) cells to the last point
    are skipped (they would add segments without adding cells).
  - objects: add / select / edit / delete; every edit updates only the union of the object's old
    and new cell boxes.
  - text entry (GGUI 1.7 has no text widget): a key-capture mode -- letters, space, BackSpace,
    Return (commit), Escape (cancel); digits via app_input_MIRROR.DigitPoller.
  - scenes: save to / load from ScenesCLAUDE/ (SolidScene.save / load).
*Generated from scratch -- Claude Opus 5.5 -- 30/09/2026*
"""
import glob
import math
import os
import time

import taichi as ti

import solids_MIRROR as so

TEXT_PRESETS = ['DEATH', 'CA EXEMPT', 'ZRL', 'WAVE', 'GOVERNMENT PLATES', 'EXMILITARY', 'NO LOVE']
KIND_DEFAULTS = {'rect': dict(w=0.30, h=0.05), 'circle': dict(r=0.07), 'ngon': dict(r=0.10, n=6),
                 'text': dict(text='ZRL', height=0.16)}
DIM = (0.55, 0.55, 0.60)
GRID_CLAMP = 2.0          # stroke points are clamped to |g| <= 2 grid units (the array is |g| <= 1.73)
BRUSH_MAX = 64.0          # cells


def describe(o):
    """One panel line (<= ~50 characters) for an object."""
    if o.kind == 'rect':
        size = f'rect {o.w:.3f}x{o.h:.3f}'
    elif o.kind == 'circle':
        size = f'circle r {o.r:.3f}'
    elif o.kind == 'ngon':
        size = f'{o.n}-gon r {o.r:.3f}'
    else:
        t = o.text if len(o.text) <= 14 else o.text[:13] + '~'
        size = f'"{t}" h {o.height:.3f}'
    extra = ('' if o.anchor == 'grid' else ' dom') + ('' if o.enabled else ' off')
    return f'{size} ({o.x:+.2f},{o.y:+.2f}) {o.rot_deg:.0f}\xb0{extra}'


class SolidsUI:
    def __init__(self):
        self.draw = False             # draw-wall mode: LMB paints
        self.erase = False            # eraser (SHIFT + LMB erases too)
        self.brush = 2.0              # brush radius in cells (the core raises it to >= MIN_BRUSH)
        self.painting = False
        self.last = None              # last stroke point (grid units)
        self.sel = -1                 # selected object (index into scene.objects)
        self.text = None              # the text being typed (None: not capturing)
        self.caps = True
        self.preset_i = 0
        self.files = []
        self.file_i = 0
        self.lat_ms = []              # wall time per painted segment: extend + update + sync
        self.seg_count = 0

    # ---- painting ---------------------------------------------------------------------------------
    def spacing(self, m):
        return so.cells_to_grid(m, max(0.35, 0.25 * max(so.MIN_BRUSH, self.brush)))

    @staticmethod
    def clamp(g):
        return max(-GRID_CLAMP, min(GRID_CLAMP, g[0])), max(-GRID_CLAMP, min(GRID_CLAMP, g[1]))

    def _timed_update(self, m, bb):
        t0 = time.perf_counter()
        m.update_solids(bb)
        ti.sync()
        self.lat_ms.append((time.perf_counter() - t0) * 1e3)
        del self.lat_ms[:-200]

    def begin(self, m, g, erase=False):
        g = self.clamp(g)
        sc = m.ensure_scene()
        t0 = time.perf_counter()
        bb = sc.begin_stroke(g[0], g[1], so.cells_to_grid(m, self.brush), erase or self.erase)
        m.update_solids(bb)
        ti.sync()
        self.lat_ms.append((time.perf_counter() - t0) * 1e3)
        self.painting = True
        self.last = g
        self.seg_count = 0

    def move(self, m, g):
        """Extend the stroke to g (grid units); returns True if a segment was added."""
        if not self.painting:
            return False
        g = self.clamp(g)
        if math.hypot(g[0] - self.last[0], g[1] - self.last[1]) < self.spacing(m):
            return False
        sc = m.ensure_scene()
        t0 = time.perf_counter()
        bb = sc.extend_stroke(g[0], g[1])
        m.update_solids(bb)
        ti.sync()
        self.lat_ms.append((time.perf_counter() - t0) * 1e3)
        del self.lat_ms[:-200]
        self.last = g
        self.seg_count += 1
        return True

    def end(self, m):
        if self.painting and m.scene is not None:
            m.scene.end_stroke()
        self.painting = False
        self.last = None

    def undo(self, m):
        if m.scene is None or not m.scene.strokes:
            return 'no stroke to undo'
        self.end(m)
        self._timed_update(m, m.scene.undo_stroke())
        return f'stroke removed ({len(m.scene.strokes)} left)'

    def clear(self, m):
        if m.scene is None or not m.scene.strokes:
            return 'no painted walls'
        self.end(m)
        n = len(m.scene.strokes)
        m.scene.clear_strokes()
        self._timed_update(m, None)
        return f'{n} strokes cleared'

    # ---- objects ----------------------------------------------------------------------------------
    def add(self, m, kind):
        sc = m.ensure_scene()
        k = len(sc.objects)
        o = so.SolidObject(kind, x=-0.45 + 0.3 * (k % 4), y=0.45 - 0.3 * ((k // 4) % 3), **KIND_DEFAULTS[kind])
        idx, bb = sc.add(o)
        self._timed_update(m, bb)
        self.sel = idx
        self.text = None              # review v3 01/10/2026: the entry stayed open on the NEW object
        return f'added {describe(o)}'

    def selected(self, m):
        sc = m.scene
        if sc is None or not (0 <= self.sel < len(sc.objects)):
            return None
        return sc.objects[self.sel]

    def edit(self, m, **kw):
        if self.selected(m) is None:
            return
        self._timed_update(m, m.scene.edit(self.sel, **kw))

    def delete(self, m):
        o = self.selected(m)
        if o is None:
            return 'nothing selected'
        self.text = None
        self._timed_update(m, m.scene.remove(self.sel))
        self.sel = min(self.sel, len(m.scene.objects) - 1)
        return f'deleted {o.kind}'

    def select(self, m, d):
        n = len(m.scene.objects) if m.scene is not None else 0
        self.text = None
        self.sel = -1 if n == 0 else (self.sel + d) % n

    # ---- text entry -------------------------------------------------------------------------------
    def start_text(self, m):
        o = self.selected(m)
        if o is None or o.kind != 'text':
            return False
        self.text = o.text
        return True

    def key(self, m, k):
        """A key while the text entry is open. Returns True when consumed (always, while open:
        no hotkey may fire during typing)."""
        if self.text is None:
            return False
        if k == 'Return':
            return self.commit(m) or True
        if k == 'Escape':
            self.text = None
        elif k == 'BackSpace':
            self.text = self.text[:-1]
        elif k == ' ':
            self.text += ' '
        elif len(k) == 1 and k.isalpha():
            self.text += k.upper() if self.caps else k.lower()
        elif len(k) == 1 and (k.isdigit() or k in "-.,/+'"):
            self.text += k
        return True

    def chars(self, chars):
        if self.text is not None:
            for ch in chars:
                self.text += ch

    def commit(self, m):
        t = self.text
        self.text = None
        if t is None:
            return False
        if not t.strip():
            return False                              # an empty text keeps the old one
        o = self.selected(m)
        if o is None or o.kind != 'text':             # the selection changed under the entry
            return False
        self.edit(m, text=t)
        return True

    # ---- scenes -----------------------------------------------------------------------------------
    def refresh_files(self):
        self.files = sorted(glob.glob(os.path.join(so.SCENES_DIR, '*.json')))
        self.file_i = min(self.file_i, max(0, len(self.files) - 1))

    def save(self, m, path=None):
        return m.ensure_scene().save(path)

    def load(self, m, path):
        sc = so.SolidScene.load(path)
        self.end(m)
        self.text = None
        t0 = time.perf_counter()
        m.attach_scene(sc)
        ti.sync()
        self.lat_ms.append((time.perf_counter() - t0) * 1e3)
        self.sel = len(sc.objects) - 1
        return sc

    def set_preset(self, m, plate_text=None, plate_screws=None):
        m.ensure_scene().set_preset(plate_text=plate_text, plate_screws=plate_screws)
        self._timed_update(m, None)

    _var_cache = {}

    def variations(self, font):
        """so.font_variations, cached: it opens the font file (~ms), and the panel asks every frame."""
        if font not in self._var_cache:
            try:
                self._var_cache[font] = so.font_variations(font)
            except Exception:
                self._var_cache[font] = []
        return self._var_cache[font]

    # ---- panel ------------------------------------------------------------------------------------
    def gui(self, w, viewer):
        m = viewer.m
        p = m.p
        sc = m.scene
        moved = viewer.moved
        w.text('walls and solids', color=(0.72, 0.62, 1.0))
        d = w.checkbox('draw walls  [T]', self.draw)
        if d != self.draw:
            viewer.set_draw(d)
        w.text('   LMB paints, SHIFT+LMB erases', color=DIM)
        self.erase = w.checkbox('eraser', self.erase)
        b = w.slider_float('brush (cells)', self.brush, 1.0, BRUSH_MAX)
        if moved(b, self.brush):
            self.brush = b
        w.text(f'   radius {max(so.MIN_BRUSH, self.brush):.1f} cells = {so.cells_to_grid(m, self.brush):.4f} grid',
               color=DIM)
        w.text(f'   (>= {so.MIN_BRUSH:.0f} cell seals every lattice)', color=DIM)
        ns = len(sc.strokes) if sc is not None else 0
        npts = sum(len(s.points) for s in sc.strokes) if sc is not None else 0
        w.text(f'strokes {ns} ({npts} points)   solid cells {m.n_solid:,}')
        if self.lat_ms:
            L = sorted(self.lat_ms[-50:])
            w.text(f'   update: median {L[len(L) // 2]:.2f} ms, max {L[-1]:.2f} ms', color=DIM)
        if w.button('undo stroke  [U]'):
            viewer.flash(self.undo(m))
        if w.button('clear painted walls'):
            viewer.flash(self.clear(m))
        # the Unlicensed plate
        w.text('')
        if p.shape == 'unlicensed':
            s2 = m.ensure_scene()
            t = w.checkbox('plate lettering (CA EXEMPT / DEATH)', s2.plate_text)
            if t != s2.plate_text:
                self.set_preset(m, plate_text=t)
            s = w.checkbox('screw heads', s2.plate_screws)
            if s != s2.plate_screws:
                self.set_preset(m, plate_screws=s)
        else:
            w.text('the plate is the domain shape "unlicensed"', color=DIM)
            if w.button('switch to the Unlicensed plate'):
                viewer.set_shape('unlicensed')
        # objects
        w.text('')
        n = len(sc.objects) if sc is not None else 0
        npre = len(sc.preset_objects(m)) if sc is not None else 0
        w.text(f'objects: {n}' + (f'   (+ {npre} plate parts)' if npre else ''))
        for kind in ('rect', 'circle', 'ngon', 'text'):
            if w.button(f'add {kind}'):
                viewer.flash(self.add(m, kind))
                sc = m.scene
                n = len(sc.objects)
        if n:
            for i, o in enumerate(sc.objects[:14]):
                w.text(('> ' if i == self.sel else '  ') + f'{i}: {describe(o)}',
                       color=(0.85, 0.55, 1.0) if i == self.sel else None)
            if n > 14:
                w.text(f'  ... {n - 14} more')
            if w.button('< select'):
                self.select(m, -1)
            if w.button('select >'):
                self.select(m, +1)
        o = self.selected(m)
        if o is not None:
            self._edit_gui(w, viewer, o)
        # scenes
        w.text('')
        if w.button('save scene (ScenesCLAUDE)'):
            path = self.save(m)
            self.refresh_files()
            viewer.flash('saved ' + os.path.basename(path))
        if w.button('scene files: refresh'):
            self.refresh_files()
        if self.files:
            self.file_i = min(self.file_i, len(self.files) - 1)
            w.text(f'   {self.file_i + 1}/{len(self.files)}: {os.path.basename(self.files[self.file_i])}')
            if w.button('< file'):
                self.file_i = (self.file_i - 1) % len(self.files)
            if w.button('file >'):
                self.file_i = (self.file_i + 1) % len(self.files)
            if w.button('load this scene'):
                try:
                    self.load(m, self.files[self.file_i])
                    viewer.flash('loaded ' + os.path.basename(self.files[self.file_i]))
                except Exception as ex:                       # a bad file must not end the app
                    viewer.flash(f'load failed: {type(ex).__name__}: {ex}')

    def _edit_gui(self, w, viewer, o):
        m = viewer.m
        moved = viewer.moved
        w.text(f'selected {self.sel}: {o.kind}', color=(0.85, 0.55, 1.0))
        e = w.checkbox('enabled', o.enabled)
        if e != o.enabled:
            self.edit(m, enabled=e)
        if w.button(f'anchor: {o.anchor}'):
            self.edit(m, anchor='domain' if o.anchor == 'grid' else 'grid')
        w.text('   grid: fixed to the array; domain: turns with it', color=DIM)
        lim = 1.8
        for f, lab, lo, hi in (('x', 'x', -lim, lim), ('y', 'y', -lim, lim), ('rot_deg', 'rotation (deg)', 0.0, 360.0)):
            v = w.slider_float(lab, getattr(o, f), lo, hi)
            if moved(v, getattr(o, f)):
                self.edit(m, **{f: v})
        if o.kind == 'rect':
            for f, lab in (('w', 'width'), ('h', 'height')):
                v = w.slider_float(lab, getattr(o, f), 0.002, 2.5)
                if moved(v, getattr(o, f)):
                    self.edit(m, **{f: v})
        elif o.kind in ('circle', 'ngon'):
            v = w.slider_float('radius', o.r, 0.002, 1.5)
            if moved(v, o.r):
                self.edit(m, r=v)
            if o.kind == 'ngon':
                nn = w.slider_int('sides', o.n, 3, 16)
                if nn != o.n:
                    self.edit(m, n=nn)
        else:
            if self.text is not None:
                w.text(f'typing: {self.text}_', color=(0.4, 0.9, 0.5))
                w.text('   letters, digits, space, BackSpace', color=DIM)
                w.text('   Return applies, Escape cancels', color=DIM)
                self.caps = w.checkbox('capitals', self.caps)
                if w.button('apply text  [Return]'):
                    self.commit(m)
                if w.button('cancel  [Escape]'):
                    self.text = None
            else:
                w.text(f'text: "{o.text}"')
                if w.button('type new text  [Return]'):
                    self.start_text(m)
            if w.button(f'preset: {TEXT_PRESETS[self.preset_i]}  (apply)'):
                self.text = None
                self.edit(m, text=TEXT_PRESETS[self.preset_i])
            if w.button('next preset'):
                self.preset_i = (self.preset_i + 1) % len(TEXT_PRESETS)
            v = w.slider_float('cap height', o.height, 0.01, 1.2)
            if moved(v, o.height):
                self.edit(m, height=v)
            fw = w.slider_float('fit width (0=off)', o.fit_w, 0.0, 3.5)
            if moved(fw, o.fit_w):
                self.edit(m, fit_w=fw if fw > 1e-3 else 0.0)
            if o.fit_w <= 0:
                v = w.slider_float('stretch', o.stretch, 0.3, 3.0)
                if moved(v, o.stretch):
                    self.edit(m, stretch=v)
            v = w.slider_float('letter spacing', o.tracking, 0.0, 1.0)
            if moved(v, o.tracking):
                self.edit(m, tracking=v)
            vars_ = self.variations(o.font)
            if vars_ and w.button(f'font style: {o.variation}'):
                i = vars_.index(o.variation) if o.variation in vars_ else -1
                self.edit(m, variation=vars_[(i + 1) % len(vars_)])
        if w.button('delete selected'):
            viewer.flash(self.delete(m))
