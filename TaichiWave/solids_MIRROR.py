"""ZRL 2D wave membrane -- Taichi port: SOLID OBJECTS and PAINTED WALLS.

Every solid is PINNED: u = 0 in a solid cell, the rule the domain mask already applies (a hard,
phase-inverting Dirichlet wall). Rigid (Neumann) walls are held off by the user (29/09/2026).
The core composes the solver's mask as  M = domain AND NOT solid  (Membrane.domain_mask_np,
Membrane.solid_np); this module only decides which cells are solid.

    scene = SolidScene()
    scene.add(SolidObject('circle', x=0.3, y=0.1, r=0.05))       # grid units: base_half
    m.attach_scene(scene)                                        # rasterise + apply, no reset
    bb = scene.begin_stroke(-0.5, 0.2, radius=0.01)              # painted wall, grid units
    m.update_solids(bb)
    bb = scene.extend_stroke(-0.4, 0.25); m.update_solids(bb)    # one segment: a small patch
    scene.end_stroke()

Units ("anchor"):
  - 'grid'   : lengths in units of base_half (the array's physical half-size: (N-1)/2 cells on the
               square lattice, (N-1)/2 * sqrt(3)/2 on the triangular one), so a scene means the same
               thing at every N and on both lattices. Strokes are always in grid units.
  - 'domain' : lengths in units of dom_R (the domain's R; for 'unlicensed' the plate's HALF-WIDTH),
               and the object turns with the domain's rotation p.rot (the Unlicensed preset).
Cells are rasterised in PHYSICAL coordinates (Membrane.cell_xy), so a circle is round on the
triangular lattice too; each object visits only its own bounding box.

Walls are SEALED when the brush's physical radius r is at least MIN_BRUSH = 1.0 cell. Why: a
link between two free cells on opposite sides of a painted curve crosses the curve at a point P,
and both ends are more than r from P, so the link is longer than 2r. The longest links are 1
(sq5, tri) and sqrt(2) (sq9 diagonals), so r > 0.5 (sq5, tri) / r > 0.7071 (sq9) already seals;
1.0 keeps a margin. A ONE-CELL diagonal staircase wall, by contrast, leaks through the sq9
diagonal links and the triangular lattice's (1,-1) links (solids_tests_MIRROR.py measures it).

Text is rendered with PIL at high resolution in the object's own frame and sampled at each cell
(bilinear, threshold 50%). The enclosed counters of letters (inside D, A, P, R ...) become sealed
cavities of free cells: expected, and physical for a pinned plate.

Measured facts and dates are in NotesCLAUDE/solids_MIRROR.md.
*Generated from scratch -- Claude Opus 5.5 -- 30/09/2026*
"""
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict, astuple, fields

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES_DIR = os.path.join(HERE, 'ScenesCLAUDE')
FONT_DIR = 'C:/Windows/Fonts'
DEFAULT_FONT = 'bahnschrift.ttf'
FALLBACK_FONTS = ('ARIALNB.TTF', 'ARIALN.TTF')       # used only when the DEFAULT font is missing
DEFAULT_VARIATION = 'SemiBold SemiCondensed'
KINDS = ('rect', 'circle', 'ngon', 'text')
ANCHORS = ('grid', 'domain')
MIN_BRUSH = 1.0             # minimum physical brush radius in cells (seals sq5, sq9 and tri; see above)
EPS = 1e-9                  # the domain predicates' slack (insideNgon's +1e-9)
EMPTY_BOX = (0, 0, 0, 0)    # an update box that covers no cell (m.update_solids does nothing)
CHUNK = 1 << 22             # cells per numpy pass: bounds host memory at N = 8193
S3H = math.sqrt(3) / 2
PI = math.pi
TAU = 2 * PI

# ---- the Unlicensed preset: Death Grips, "Government Plates" (2013), the cover's plate --------
# Measured from the cover by the user's request (29/09/2026), in units of the plate's HALF-WIDTH R
# (dom_R), centred, turning with the domain. Cap heights and TOTAL ink widths are the cover's; the
# letter-spacing (tracking) is chosen so the fitted glyphs look like the cover's lettering
# (probe 30/09/2026, ScratchCLAUDE/DevCLAUDE/font_probe_MIRROR.py):
#   DEATH     : Bahnschrift 'SemiBold SemiCondensed', natural ink width 3.58 cap heights; the
#               cover's is 1.574/0.525 = 3.00 with narrow letters (~0.46 cap) and wide gaps, so
#               tracking 0.30 cap and the fitted stretch comes out at ~0.63;
#   CA EXEMPT : Bahnschrift 'SemiBold', natural 7.32 cap heights vs the cover's 1.107/0.1375 =
#               8.05, so tracking 0.09 and the stretch comes out at ~1.0.
PLATE_TEXT = (
    dict(text='CA EXEMPT', x=0.0, y=0.319, height=0.1375, fit_w=1.107, variation='SemiBold', tracking=0.09),
    dict(text='DEATH', x=0.0, y=-0.0875, height=0.525, fit_w=1.574, variation='SemiBold SemiCondensed',
         tracking=0.30),
)
PLATE_SCREWS = ((0.628, 0.4375), (-0.628, 0.4375), (-0.628, -0.4375), (0.628, -0.4375))
PLATE_SCREW_R = 0.02


@dataclass
class SolidObject:
    """One static solid. Lengths are in the anchor's unit (see the module docstring).
      rect   : w x h (FULL width and height), turned by rot_deg about (x, y)
      circle : radius r
      ngon   : regular n-gon, circumradius r, vertices at rot + 180/n + k 360/n degrees (the
               domain shapes' convention: n = 4 is axis-aligned)
      text   : cap height `height`; the glyph shapes are scaled horizontally by `stretch`, or, if
               fit_w > 0, the whole ink width is fitted to fit_w (the stretch is then solved);
               `tracking` adds letter-spacing in cap heights (before the stretch); centred on the
               ink box horizontally and on the cap box (baseline to cap line) vertically."""
    kind: str = 'rect'
    x: float = 0.0
    y: float = 0.0
    rot_deg: float = 0.0
    w: float = 0.2
    h: float = 0.1
    r: float = 0.1
    n: int = 6
    text: str = ''
    height: float = 0.1
    font: str = DEFAULT_FONT
    variation: str = DEFAULT_VARIATION
    stretch: float = 1.0
    anchor: str = 'grid'
    enabled: bool = True
    tracking: float = 0.0
    fit_w: float = 0.0
    name: str = ''

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f'SolidObject kind {self.kind!r}: must be one of {KINDS}')
        if self.anchor not in ANCHORS:
            raise ValueError(f'SolidObject anchor {self.anchor!r}: must be one of {ANCHORS}')


@dataclass
class Stroke:
    """A painted wall (erase=False) or an eraser stroke (erase=True): the union of the capsules
    of radius `radius` around its polyline. Points and radius in GRID units; the physical radius
    is never below MIN_BRUSH cells."""
    points: list = field(default_factory=list)
    radius: float = 0.01
    erase: bool = False


# ---- unit conversions (m = the core's Membrane) -----------------------------------------------
def unit_scale(m, anchor='grid'):
    """Physical cells per unit: base_half for 'grid', dom_R for 'domain'."""
    return m.dom_R() if anchor == 'domain' else m.base_half()


def grid_to_phys(m, gx, gy):
    h = m.base_half()
    return gx * h, gy * h


def phys_to_grid(m, x, y):
    h = m.base_half()
    return x / h, y / h


def grid_to_cell(m, gx, gy):
    """Fractional (i, j) of a grid-unit point (round it with m.round_cell for a cell)."""
    return m.xy_to_cell(*grid_to_phys(m, gx, gy))


def cell_to_grid(m, i, j):
    return phys_to_grid(m, *m.cell_xy(i, j))


def domain_to_phys(m, dx, dy):
    """Domain units (dom_R, turning with p.rot) -> physical."""
    R = m.dom_R()
    co, si = math.cos(m.p.rot), math.sin(m.p.rot)
    x, y = dx * R, dy * R
    return x * co - y * si, x * si + y * co


def phys_to_domain(m, x, y):
    R = m.dom_R()
    co, si = math.cos(m.p.rot), math.sin(m.p.rot)
    return (x * co + y * si) / R, (-x * si + y * co) / R


def cells_to_grid(m, cells):
    """A length in cells (physical) as grid units -- e.g. a brush size for begin_stroke."""
    return cells / m.base_half()


def brush_cells(m, radius):
    """The physical brush radius (cells) that a grid-unit radius paints with."""
    return max(MIN_BRUSH, radius * m.base_half())


def phys_box_to_cells(m, x0, x1, y0, y1, pad=1):
    """Half-open index box (i0, i1, j0, j1) holding every cell whose centre lies in the physical
    box [x0, x1] x [y0, y1] (+ pad cells), clamped to the array."""
    N = m.N
    I, J = [], []
    for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        fi, fj = m.xy_to_cell(x, y)
        I.append(fi)
        J.append(fj)
    i0 = max(0, int(math.floor(min(I))) - pad)
    i1 = min(N, int(math.ceil(max(I))) + 1 + pad)
    j0 = max(0, int(math.floor(min(J))) - pad)
    j1 = min(N, int(math.ceil(max(J))) + 1 + pad)
    return i0, max(i0, i1), j0, max(j0, j1)


def bbox_union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])


def _intersect(a, b):
    i0, i1, j0, j1 = max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])
    return None if (i1 <= i0 or j1 <= j0) else (i0, i1, j0, j1)


# ---- fonts and text rendering ----------------------------------------------------------------
_FONT_PATHS = {}
_TEXT_CACHE = {}
_TEXT_CACHE_MAX = 12


def resolve_font(name=DEFAULT_FONT):
    """The TTF path for a font name (a file in C:/Windows/Fonts or a full path). The DEFAULT font
    falls back to Arial Narrow Bold / Arial Narrow if missing; any other missing font raises
    FileNotFoundError naming what was tried."""
    name = name or DEFAULT_FONT
    if name in _FONT_PATHS:
        return _FONT_PATHS[name]
    tried = []
    cands = [name] + (list(FALLBACK_FONTS) if os.path.basename(name).lower() == DEFAULT_FONT else [])
    for c in cands:
        path = c if os.path.isabs(c) else os.path.join(FONT_DIR, c)
        tried.append(path)
        if os.path.isfile(path):
            _FONT_PATHS[name] = path
            return path
    raise FileNotFoundError(f'solids: font {name!r} not found (tried: {", ".join(tried)}); '
                            f'pass a .ttf file name from {FONT_DIR} or a full path')


def font_variations(name=DEFAULT_FONT):
    """The named variations of a variable font ([] for a static one)."""
    from PIL import ImageFont
    f = ImageFont.truetype(resolve_font(name), 64)
    try:
        return [v.decode() if isinstance(v, bytes) else v for v in f.get_variation_names()]
    except OSError:                                   # not a variable font
        return []


def _load_font(path, size, variation):
    from PIL import ImageFont
    f = ImageFont.truetype(path, size)
    if variation:
        try:
            names = [v.decode() if isinstance(v, bytes) else v for v in f.get_variation_names()]
        except OSError:
            names = None                              # a static font: the variation does not apply
        if names is not None:
            if variation not in names:
                raise ValueError(f'solids: font {os.path.basename(path)} has no variation {variation!r}; '
                                 f'it has: {", ".join(names)}')
            f.set_variation_by_name(variation)
    return f


def _cap_px_for(height_cells):
    """Render resolution: ~2 px per cell of cap height, at least 200 px (so small text keeps its
    shape), at most 1200 px (DEATH at N = 8193 is ~2150 cells tall: ~0.56 px per cell, and the
    bilinear sample + 50% threshold still places an edge to a fraction of a pixel)."""
    return int(min(1200, max(200, round(2 * height_cells))))


def text_layout(text, font=DEFAULT_FONT, variation=DEFAULT_VARIATION, tracking=0.0, cap_px=400):
    """Render `text` (glyph by glyph, pen advance + tracking * cap) into an 8-bit coverage image.
    Returns a dict: img (uint8 [rows, cols], row 0 at the top), cap (px), cx (ink-box centre,
    column coordinate), cy (cap-box centre, row coordinate), ink (c0, c1, r0, r1) in continuous
    pixel coordinates, the font path used and whether the variation applied. Cached."""
    path = resolve_font(font)
    key = (text, path, variation, round(float(tracking), 9), int(cap_px))
    if key in _TEXT_CACHE:
        return _TEXT_CACHE[key]
    from PIL import Image, ImageDraw
    # size the font so the cap height (the 'H' ink box) is cap_px
    f0 = _load_font(path, 400, variation)
    cap0 = -f0.getbbox('H', anchor='ls')[1]
    size = max(8, int(round(400 * cap_px / cap0)))
    f = _load_font(path, size, variation)
    cap = -f.getbbox('H', anchor='ls')[1]
    asc, desc = f.getmetrics()
    track = tracking * cap
    adv = [f.getlength(ch) for ch in text]
    pad = int(0.25 * cap) + 4
    width = int(math.ceil(sum(adv) + max(0, len(text) - 1) * max(track, 0) + 2 * pad + cap))
    rows = int(asc + desc + 2 * pad)
    base = pad + asc
    im = Image.new('L', (max(1, width), max(1, rows)), 0)
    d = ImageDraw.Draw(im)
    x = pad + 0.5 * cap
    for ch, a in zip(text, adv):
        d.text((x, base), ch, font=f, fill=255, anchor='ls')
        x += a + track
    img = np.asarray(im, dtype=np.uint8)
    cols = np.nonzero(img.max(axis=0) > 0)[0]
    rws = np.nonzero(img.max(axis=1) > 0)[0]
    if cols.size == 0:
        ink = (0.0, 0.0, 0.0, 0.0)
        cx = 0.0
    else:
        ink = (float(cols[0]), float(cols[-1] + 1), float(rws[0]), float(rws[-1] + 1))
        cx = 0.5 * (ink[0] + ink[1])
    try:
        f.get_variation_names()
        var_applied = bool(variation)
    except OSError:
        var_applied = False
    out = {'img': img, 'cap': float(cap), 'cx': cx, 'cy': base - 0.5 * cap, 'ink': ink,
           'font': path, 'variation_applied': var_applied}
    if len(_TEXT_CACHE) >= _TEXT_CACHE_MAX:
        _TEXT_CACHE.pop(next(iter(_TEXT_CACHE)))
    _TEXT_CACHE[key] = out
    return out


def _bilinear_u8(img, u, v):
    """Coverage in [0, 255] at continuous pixel coordinates (u = column, v = row; pixel (r, q)
    covers [q, q+1) x [r, r+1), sampled at its centre). Outside the image: 0."""
    H, W = img.shape
    q = u - 0.5
    r = v - 0.5
    q0 = np.floor(q)
    r0 = np.floor(r)
    fq = q - q0
    fr = r - r0
    q0 = q0.astype(np.int64)
    r0 = r0.astype(np.int64)
    out = np.zeros(u.shape, dtype=np.float64)
    for dq, dr, wgt in ((0, 0, (1 - fq) * (1 - fr)), (1, 0, fq * (1 - fr)), (0, 1, (1 - fq) * fr), (1, 1, fq * fr)):
        qq, rr = q0 + dq, r0 + dr
        ok = (qq >= 0) & (qq < W) & (rr >= 0) & (rr < H)
        out[ok] += wgt[ok] * img[rr[ok], qq[ok]]
    return out


# ---- object geometry -------------------------------------------------------------------------
def object_frame(m, o):
    """(ox, oy, theta, u): the object's centre (physical), its rotation (radians) and the
    physical length of one unit."""
    u = unit_scale(m, o.anchor)
    th = math.radians(o.rot_deg)
    if o.anchor == 'domain':
        rot = m.p.rot
        co, si = math.cos(rot), math.sin(rot)
        x, y = o.x * u, o.y * u
        return x * co - y * si, x * si + y * co, th + rot, u
    return o.x * u, o.y * u, th, u


def _text_params(m, o, u):
    """(layout, s = physical per pixel vertically, sx = physical per pixel horizontally)."""
    hphys = o.height * u
    lay = text_layout(o.text, o.font, o.variation, o.tracking, _cap_px_for(hphys))
    s = hphys / lay['cap']
    inkw = lay['ink'][1] - lay['ink'][0]
    if o.fit_w > 0 and inkw > 0:
        sx = o.fit_w * u / inkw
    else:
        sx = s * o.stretch
    return lay, s, sx


def text_stretch(m, o):
    """The horizontal stretch a text object is drawn with (solved from fit_w when that is set)."""
    u = unit_scale(m, o.anchor)
    lay, s, sx = _text_params(m, o, u)
    return sx / s


def object_local_extent(m, o):
    """Half-extents (hx, hy) of the object's local bounding box, physical cells."""
    u = unit_scale(m, o.anchor)
    if o.kind == 'rect':
        return 0.5 * abs(o.w) * u, 0.5 * abs(o.h) * u
    if o.kind in ('circle', 'ngon'):
        return abs(o.r) * u, abs(o.r) * u
    lay, s, sx = _text_params(m, o, u)
    c0, c1, r0, r1 = lay['ink']
    hx = max(abs(c0 - lay['cx']), abs(c1 - lay['cx'])) * sx + sx + 1
    hy = max(abs(r0 - lay['cy']), abs(r1 - lay['cy'])) * s + s + 1
    return hx, hy


def object_cell_bbox(m, o):
    """Half-open cell box (i0, i1, j0, j1) holding the object (None if it is disabled/empty)."""
    if not o.enabled:
        return None
    ox, oy, th, u = object_frame(m, o)
    hx, hy = object_local_extent(m, o)
    co, si = math.cos(th), math.sin(th)
    ex = abs(hx * co) + abs(hy * si) + EPS
    ey = abs(hx * si) + abs(hy * co) + EPS
    bb = phys_box_to_cells(m, ox - ex, ox + ex, oy - ey, oy + ey)
    return None if (bb[1] <= bb[0] or bb[3] <= bb[2]) else bb


def object_inside(m, o, X, Y):
    """Boolean mask of the physical points (X, Y) that the object covers."""
    ox, oy, th, u = object_frame(m, o)
    co, si = math.cos(th), math.sin(th)
    dx, dy = X - ox, Y - oy
    lx = dx * co + dy * si
    ly = -dx * si + dy * co
    if o.kind == 'rect':
        return (np.abs(lx) <= 0.5 * abs(o.w) * u + EPS) & (np.abs(ly) <= 0.5 * abs(o.h) * u + EPS)
    if o.kind == 'circle':
        rr = abs(o.r) * u
        return lx * lx + ly * ly <= rr * rr + EPS
    if o.kind == 'ngon':
        # the domain's insideNgon, in the object's frame
        nn = max(3, int(o.n))
        rr = abs(o.r) * u
        r = np.hypot(lx, ly)
        mth = TAU / nn
        a = np.fmod(np.fmod(np.arctan2(ly, lx) + PI / nn, mth) + mth, mth) - PI / nn
        ins = r <= rr * math.cos(PI / nn) / np.cos(a) + EPS
        ins[r < 1e-12] = True
        return ins
    lay, s, sx = _text_params(m, o, u)
    uu = lay['cx'] + lx / sx
    vv = lay['cy'] - ly / s                      # rows grow downward, y upward
    return _bilinear_u8(lay['img'], uu, vv) >= 127.5


# ---- strokes -----------------------------------------------------------------------------------
def _stroke_segments(m, st):
    """Physical segment endpoints (K, 4) array [ax, ay, bx, by] and the physical radius."""
    P = np.asarray(st.points, dtype=np.float64).reshape(-1, 2) * m.base_half()
    if len(P) == 1:
        P = np.vstack([P, P])
    seg = np.hstack([P[:-1], P[1:]])
    return seg, brush_cells(m, st.radius)


def segment_cell_bbox(m, a, b, radius_cells):
    x0, x1 = min(a[0], b[0]) - radius_cells, max(a[0], b[0]) + radius_cells
    y0, y1 = min(a[1], b[1]) - radius_cells, max(a[1], b[1]) + radius_cells
    return phys_box_to_cells(m, x0, x1, y0, y1)


def capsule_inside(X, Y, ax, ay, bx, by, r):
    """Cells within distance r (+EPS) of the segment a-b (exact point-segment distance; a mirror
    image of the segment gives the mirror image of the cell set, bit for bit)."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    px, py = X - ax, Y - ay
    if L2 > 0:
        t = np.clip((px * dx + py * dy) / L2, 0.0, 1.0)
        qx, qy = px - t * dx, py - t * dy
    else:
        qx, qy = px, py
    return qx * qx + qy * qy <= r * r + EPS


TILE = 64                   # capsule tiles: a long diagonal segment only visits the tiles it meets


def capsule_tiles(m, sub, ax, ay, bx, by, r):
    """The T x T cell tiles of the box sub = (i0, i1, j0, j1) that can hold a cell within r of the
    segment a-b, as boxes (01/10/2026, Claude Opus 5.5: a diagonal segment's bounding box is
    ~L^2/2 cells, its capsule only ~2rL). Conservative: a tile is kept when the distance from its
    physical centre to the segment is <= r + its circumradius (+1 cell slack); the cell map is
    affine, so a tile is a parallelogram and its circumradius is half its longer diagonal."""
    i0, i1, j0, j1 = sub
    ti0 = np.arange(i0, i1, TILE)
    tj0 = np.arange(j0, j1, TILE)
    TI0, TJ0 = np.meshgrid(ti0, tj0, indexing='ij')
    TI1 = np.minimum(TI0 + TILE, i1) - 1                 # last cell index of the tile
    TJ1 = np.minimum(TJ0 + TILE, j1) - 1
    c = m.c()

    def xy(I, J):
        da, db = I - c, J - c
        if m.tri():
            return da + db * 0.5, db * (math.sqrt(3) / 2)
        return da.astype(np.float64), db.astype(np.float64)

    x00, y00 = xy(TI0, TJ0)
    x11, y11 = xy(TI1, TJ1)
    x10, y10 = xy(TI1, TJ0)
    x01, y01 = xy(TI0, TJ1)
    cx, cy = (x00 + x11) * 0.5, (y00 + y11) * 0.5
    rad = 0.5 * np.maximum(np.hypot(x11 - x00, y11 - y00), np.hypot(x10 - x01, y10 - y01))
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    px, py = cx - ax, cy - ay
    t = np.clip((px * dx + py * dy) / L2, 0.0, 1.0) if L2 > 0 else np.zeros_like(px)
    d = np.hypot(px - t * dx, py - t * dy)
    hit = d <= r + rad + 1.0
    return [(int(a), int(b) + 1, int(e), int(f) + 1)
            for a, b, e, f in zip(TI0[hit], TI1[hit], TJ0[hit], TJ1[hit])]


# ---- the scene -----------------------------------------------------------------------------------
class SolidScene:
    """Static solid objects + painted walls, and the Unlicensed preset (active when the domain
    shape is 'unlicensed'). Attach to a Membrane with m.attach_scene(scene); the membrane sets
    scene.m. Methods that change the scene return the affected cell box (i0, i1, j0, j1) for
    m.update_solids(box) (None = redo the whole grid, or nothing attached)."""

    def __init__(self, objects=None, strokes=None, plate_text=True, plate_screws=True):
        self.objects = list(objects or [])
        self.strokes = list(strokes or [])
        self.plate_text = plate_text
        self.plate_screws = plate_screws
        self.version = 0                # bumped by every edit through the methods below
        self.m = None
        self._cur = None                # the stroke being painted
        self._incr = None               # the last begin/extend edit, for rasterize's incremental path

    # ---- cache signature (the core re-rasterises when it changes) --------------------------
    def signature(self):
        return (self.version, len(self.strokes), sum(len(s.points) for s in self.strokes),
                tuple(astuple(o) for o in self.objects), bool(self.plate_text), bool(self.plate_screws))

    def _bump(self):
        self.version += 1

    # ---- the Unlicensed preset --------------------------------------------------------------
    def preset_active(self, m):
        return m.p.shape == 'unlicensed' and (self.plate_text or self.plate_screws)

    def preset_objects(self, m):
        """The plate's lettering and screw heads as SolidObjects (domain-anchored), [] unless
        the domain shape is 'unlicensed'."""
        if m.p.shape != 'unlicensed':
            return []
        out = []
        if self.plate_text:
            for t in PLATE_TEXT:
                out.append(SolidObject('text', anchor='domain', font=DEFAULT_FONT, stretch=1.0,
                                       name='plate: ' + t['text'], **t))
        if self.plate_screws:
            for k, (x, y) in enumerate(PLATE_SCREWS):
                out.append(SolidObject('circle', x=x, y=y, r=PLATE_SCREW_R, anchor='domain', name=f'plate: screw {k + 1}'))
        return out

    def all_objects(self, m):
        return self.preset_objects(m) + [o for o in self.objects if o.enabled]

    # ---- rasterisation ------------------------------------------------------------------------
    def empty(self, m):
        return not self.all_objects(m) and not any(not s.erase for s in self.strokes)

    def rasterize(self, m, i0, i1, j0, j1, incremental=True):
        """The solid mask (uint8 [i1-i0, j1-j0], 1 = solid) of the half-open cell box: objects
        (union), then the strokes in order (draw = union, erase = subtract).

        Incremental path (01/10/2026, Claude Opus 5.5): right after begin_stroke/extend_stroke,
        while m's mask is still the one from just before that edit (m._solid_key equals the key
        recorded by the edit) and the scene has not changed since, the result is m.solid_np's
        box with only the new segment's capsule applied (the painted stroke is the LAST one, so
        this equals the full re-rasterisation bit for bit). A fast diagonal drag at N = 8193 no
        longer re-samples the lettering over a multi-million-cell box."""
        out = np.zeros((max(0, i1 - i0), max(0, j1 - j0)), dtype=np.uint8)
        if out.size == 0:
            return out
        reg = (i0, i1, j0, j1)
        inc = self._incr
        if (incremental and inc is not None and inc['m'] is m and self.strokes
                and self.strokes[-1] is inc['stroke'] and inc['sig'] == self.signature()
                and getattr(m, '_solid_key', None) == inc['prev_key']
                and (m.solid_np is None or m.solid_np.shape == (m.N, m.N))):
            if m.solid_np is not None:
                out[...] = m.solid_np[i0:i1, j0:j1]
            ax, ay, bx, by = inc['seg']
            self._apply_capsule(m, out, reg, (ax, ay, bx, by), inc['r'], not inc['erase'])
            return out
        for o in self.all_objects(m):
            bb = object_cell_bbox(m, o)
            sub = _intersect(bb, reg) if bb is not None else None
            if sub is not None:
                self._apply(m, out, reg, sub, lambda X, Y, o=o: object_inside(m, o, X, Y), True)
        for st in self.strokes:
            if not st.points:
                continue
            seg, r = _stroke_segments(m, st)
            # prefilter: segments whose capsule box meets the region (in physical space)
            for k in self._segments_near(m, seg, r, reg):
                self._apply_capsule(m, out, reg, tuple(seg[k]), r, not st.erase)
        return out

    def _apply_capsule(self, m, out, reg, a, r, draw):
        """One segment's capsule into out (the box reg); a large box is visited tile by tile."""
        bb = segment_cell_bbox(m, (a[0], a[1]), (a[2], a[3]), r)
        sub = _intersect(bb, reg)
        if sub is None:
            return
        pred = lambda X, Y: capsule_inside(X, Y, *a, r)
        if (sub[1] - sub[0]) * (sub[3] - sub[2]) <= 16 * TILE * TILE:
            self._apply(m, out, reg, sub, pred, draw)
            return
        for tb in capsule_tiles(m, sub, *a, r):
            self._apply(m, out, reg, tb, pred, draw)

    @staticmethod
    def _segments_near(m, seg, r, reg):
        """Indices of the segments whose capsule could touch the cell box reg (a cheap physical
        test against the box's physical extent; the exact cell box is intersected afterwards)."""
        i0, i1, j0, j1 = reg
        xs, ys = [], []
        for i, j in ((i0, j0), (i1 - 1, j0), (i0, j1 - 1), (i1 - 1, j1 - 1)):
            x, y = m.cell_xy(i, j)
            xs.append(x)
            ys.append(y)
        x0, x1, y0, y1 = min(xs) - r - 2, max(xs) + r + 2, min(ys) - r - 2, max(ys) + r + 2
        sx0 = np.minimum(seg[:, 0], seg[:, 2])
        sx1 = np.maximum(seg[:, 0], seg[:, 2])
        sy0 = np.minimum(seg[:, 1], seg[:, 3])
        sy1 = np.maximum(seg[:, 1], seg[:, 3])
        hit = (sx1 >= x0) & (sx0 <= x1) & (sy1 >= y0) & (sy0 <= y1)
        return np.nonzero(hit)[0]

    @staticmethod
    def _apply(m, out, reg, sub, pred, draw):
        """out[sub] |= pred (draw) or &= ~pred (erase), in row chunks of <= CHUNK cells."""
        i0, i1, j0, j1 = sub
        step = max(1, CHUNK // max(1, j1 - j0))
        for a in range(i0, i1, step):
            b = min(i1, a + step)
            X, Y = m._grid_xy(a, b, j0, j1)
            ins = pred(X, Y)
            view = out[a - reg[0]:b - reg[0], j0 - reg[2]:j1 - reg[2]]
            if draw:
                view |= ins.astype(np.uint8)
            else:
                view &= (~ins).astype(np.uint8)

    def rasterize_full(self, m):
        """The whole grid's solid mask, or None if the scene makes no solid cell (then the core
        runs exactly the web app's path)."""
        if self.empty(m):
            return None
        out = self.rasterize(m, 0, m.N, 0, m.N, incremental=False)
        return out if out.any() else None

    # ---- objects ------------------------------------------------------------------------------
    def _bbox(self, o):
        return object_cell_bbox(self.m, o) if self.m is not None else None

    def add(self, obj):
        """Add a SolidObject; returns (index, cell box to update)."""
        if not isinstance(obj, SolidObject):
            raise TypeError('SolidScene.add expects a SolidObject')
        self.objects.append(obj)
        self._bump()
        return len(self.objects) - 1, self._bbox(obj)

    def add_rect(self, x, y, w, h, rot_deg=0.0, anchor='grid', **kw):
        return self.add(SolidObject('rect', x=x, y=y, w=w, h=h, rot_deg=rot_deg, anchor=anchor, **kw))

    def add_circle(self, x, y, r, anchor='grid', **kw):
        return self.add(SolidObject('circle', x=x, y=y, r=r, anchor=anchor, **kw))

    def add_ngon(self, x, y, r, n, rot_deg=0.0, anchor='grid', **kw):
        return self.add(SolidObject('ngon', x=x, y=y, r=r, n=n, rot_deg=rot_deg, anchor=anchor, **kw))

    def add_text(self, text, x, y, height, rot_deg=0.0, anchor='grid', **kw):
        return self.add(SolidObject('text', x=x, y=y, text=text, height=height, rot_deg=rot_deg,
                                    anchor=anchor, **kw))

    def remove(self, idx):
        """Remove object idx; returns the cell box it covered."""
        bb = self._bbox(self.objects[idx])
        del self.objects[idx]
        self._bump()
        return bb

    def edit(self, idx, **kw):
        """Change fields of object idx (e.g. x=0.2, rot_deg=15); returns the union of its old
        and new cell boxes."""
        o = self.objects[idx]
        old = self._bbox(o)
        names = {f.name for f in fields(SolidObject)}
        for k, v in kw.items():
            if k not in names:
                raise AttributeError(f'SolidObject has no field {k!r}')
            setattr(o, k, v)
        o.__post_init__()
        self._bump()
        return bbox_union(old, self._bbox(o))

    def list_objects(self, include_preset=True):
        """[(index or 'preset', SolidObject)] -- the preset's objects are read-only."""
        out = []
        if include_preset and self.m is not None:
            out += [('preset', o) for o in self.preset_objects(self.m)]
        out += list(enumerate(self.objects))
        return out

    def set_preset(self, plate_text=None, plate_screws=None):
        """Toggle the Unlicensed preset's parts; returns None (redo the whole grid)."""
        if plate_text is not None:
            self.plate_text = bool(plate_text)
        if plate_screws is not None:
            self.plate_screws = bool(plate_screws)
        self._bump()
        return None

    # ---- painting -----------------------------------------------------------------------------
    def _seg_bbox(self, a, b, radius):
        if self.m is None:
            return None
        h = self.m.base_half()
        r = brush_cells(self.m, radius)
        return segment_cell_bbox(self.m, (a[0] * h, a[1] * h), (b[0] * h, b[1] * h), r)

    def begin_stroke(self, x, y, radius, erase=False):
        """Start a painted wall (or an eraser stroke) at grid point (x, y); radius in grid units
        (cells_to_grid converts a size in cells; below MIN_BRUSH cells it is raised to it).
        Returns the cell box of the first dot."""
        prev = self._prev_key()
        self._cur = Stroke([[float(x), float(y)]], float(radius), bool(erase))
        self.strokes.append(self._cur)
        self._bump()
        self._note_edit(prev)
        return self._seg_bbox((x, y), (x, y), radius)

    def _prev_key(self):
        m = self.m
        return m._solid_cache_key() if m is not None and m.scene is self else None

    def _note_edit(self, prev_key):
        """Record the segment just added to the current stroke (rasterize's incremental path)."""
        self._incr = None
        if prev_key is None or self._cur is None:
            return
        st = self._cur
        P = np.asarray(st.points[-2:], dtype=np.float64).reshape(-1, 2) * self.m.base_half()
        if len(P) == 1:
            P = np.vstack([P, P])
        self._incr = dict(m=self.m, stroke=st, prev_key=prev_key, sig=self.signature(),
                          seg=tuple(np.hstack([P[0], P[1]])), r=brush_cells(self.m, st.radius),
                          erase=st.erase)

    def extend_stroke(self, x, y):
        """Add a point to the current stroke; returns the cell box of the new segment. A point
        that repeats the last one changes nothing and returns the EMPTY box (0, 0, 0, 0), which
        m.update_solids treats as a no-op (None would mean "redo the whole grid")."""
        if self._cur is None:
            raise RuntimeError('extend_stroke without begin_stroke')
        last = self._cur.points[-1]
        if last[0] == x and last[1] == y:
            return EMPTY_BOX
        prev = self._prev_key()
        self._cur.points.append([float(x), float(y)])
        self._bump()
        self._note_edit(prev)
        return self._seg_bbox(last, (x, y), self._cur.radius)

    def end_stroke(self):
        self._cur = None

    def stroke_bbox(self, st):
        bb = None
        for a, b in zip(st.points, st.points[1:] or st.points):
            bb = bbox_union(bb, self._seg_bbox(a, b, st.radius))
        return bb

    def undo_stroke(self):
        """Remove the last stroke; returns its cell box."""
        if not self.strokes:
            return None
        st = self.strokes.pop()
        if st is self._cur:
            self._cur = None
        self._bump()
        return self.stroke_bbox(st)

    def clear_strokes(self):
        self.strokes = []
        self._cur = None
        self._bump()
        return None

    def add_stroke(self, points, radius, erase=False):
        """A whole polyline at once (grid units); returns its cell box."""
        st = Stroke([[float(x), float(y)] for x, y in points], float(radius), bool(erase))
        self.strokes.append(st)
        self._bump()
        return self.stroke_bbox(st)

    # ---- save / load --------------------------------------------------------------------------
    def to_dict(self):
        return {'format': 'zrl-membrane-solids', 'version': 1,
                'plate_text': self.plate_text, 'plate_screws': self.plate_screws,
                'objects': [asdict(o) for o in self.objects],
                'strokes': [{'points': s.points, 'radius': s.radius, 'erase': s.erase} for s in self.strokes]}

    @classmethod
    def from_dict(cls, d):
        if d.get('format') != 'zrl-membrane-solids':
            raise ValueError('not a ZRL membrane solids scene (format tag missing)')
        names = {f.name for f in fields(SolidObject)}
        objs = [SolidObject(**{k: v for k, v in o.items() if k in names}) for o in d.get('objects', [])]
        strokes = [Stroke([list(map(float, p)) for p in s['points']], float(s['radius']), bool(s.get('erase', False)))
                   for s in d.get('strokes', [])]
        return cls(objs, strokes, d.get('plate_text', True), d.get('plate_screws', True))

    def save(self, path=None):
        """Write the scene as JSON; the default is ScenesCLAUDE/scene_<date-time>_MIRROR.json.
        Returns the path."""
        if path is None:
            os.makedirs(SCENES_DIR, exist_ok=True)
            path = os.path.join(SCENES_DIR, time.strftime('scene_%Y%m%d_%H%M%S') + '_MIRROR.json')
        elif not os.path.isabs(path) and os.path.dirname(path) == '':
            os.makedirs(SCENES_DIR, exist_ok=True)
            path = os.path.join(SCENES_DIR, path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=1)
        return path

    @classmethod
    def load(cls, path):
        if not os.path.isabs(path) and not os.path.isfile(path):
            path = os.path.join(SCENES_DIR, path)
        with open(path, encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    def __repr__(self):
        return (f'SolidScene({len(self.objects)} objects, {len(self.strokes)} strokes, '
                f'plate_text={self.plate_text}, plate_screws={self.plate_screws})')
