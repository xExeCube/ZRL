"""ZRL 2D wave membrane -- Taichi port, VIEWER: colours, display kernels, mesh.

What the user asked for on 29/09/2026: "the field/cells are still a bit too dark against the
background". Measured before the change (30/09/2026, NotesCLAUDE/app_MIRROR.md): the domain at rest
was 0.05 grey (CIE L* ~ 4) on a (0.02, 0, 0.05) page (L* ~ 1) -- the resting membrane was nearly
the page colour. So:
  - the palette's ZERO level is a setting (default 0.22 grey, L* ~ 24); with zero 0.05, exposure 1,
    contrast 1 the palette is exactly the web app's colorOf 'div' (checked in app_tests_MIRROR.py);
  - EXPOSURE multiplies the field colours; CONTRAST bends the amplitude curve (|x|^(1/contrast));
  - four cell classes get four distinct looks: outside the array = the background (a choice of
    near-black, dark grey, dark purple), outside the domain = a darker tone (hatched in 2D, a flat
    floor in 3D), SOLID cells = light grey like the plate lettering (raised in 3D), the domain = the
    palette;
  - 3D shading: 0 = unlit (the web app's MeshBasicMaterial: the colour as is), > 0 = a HEADLIGHT at
    the camera (GGUI's two-sided Lambert, |n.l|) plus ambient 1 - 0.55 s, light 0.8 s, so a face
    seen head-on is 1 + 0.25 s and a grazing one 1 - 0.55 s of its colour.
GGUI writes colours LINEARLY to the frame (measured: 0.2 -> PNG 51, 0.5 -> 127), so every number
here is a display (sRGB-coded) value.
Round 4 (05/10/2026): solids are PINNED (off-white, as before) or RIGID (neon cyan-blue, striped in
2D); the kernels read the core's class codes m.C when it keeps them.
*Generated from scratch -- Claude Opus 5.5 -- 30/09/2026; rigid solids added by Claude Opus 5.5 -- 05/10/2026*
"""
import math
from dataclasses import dataclass, field

import numpy as np
import taichi as ti

S3H = math.sqrt(3) / 2

# background (outside the array) -> the matching outside-the-domain tone. Outside-the-domain is
# kept DARKER than the resting domain (L* ~24) and visibly apart from the background. Mean CIE L*
# measured 30/09/2026 (ScratchCLAUDE/AppCLAUDE/contrast_bg_MIRROR.py, 2D): near-black page 0.6 ->
# a lifted slate 8.4; dark-grey page 10.3 -> a recess 3.9; dark-purple page 4.2 -> a lifted muted
# violet ~13 (a darker tone was only 1.8 L* from that page).
BACKGROUNDS = {
    'near-black': ((0.012, 0.010, 0.016), (0.085, 0.075, 0.115)),
    'dark-grey': ((0.110, 0.110, 0.120), (0.045, 0.042, 0.060)),
    'dark-purple': ((0.090, 0.030, 0.150), (0.150, 0.115, 0.200)),
}
BG_ORDER = list(BACKGROUNDS)
SOLID_COL = (0.84, 0.84, 0.80)            # the plate lettering: off-white (PINNED solids)
# RIGID solids (round 4, 05/10/2026, Claude Opus 5.5; NotesCLAUDE/walls_MIRROR.md section 11): a cool
# neon cyan-blue, apart from the pinned off-white by lightness as well as hue (CIE L* ~70 vs ~85, so
# colour-blind viewers still see two tones), and in 2D also by thin dark diagonal stripes. The rigid
# domain edge / rigid ring is drawn as an overlay line in the same colour; a torus seam as a dashed
# dim violet line (TORUS_COL).
RIGID_COL = (0.22, 0.72, 1.00)
TORUS_COL = (0.62, 0.45, 0.95)
# the palette's ends (the web app's 'div': teal (-) and red (+))
POS_END = (0.88, 0.13, 0.10)
NEG_END = (0.05, 0.62, 0.52)
# segmentation colours (contrast measurement only)
SEG = {'bg': (1.0, 0.0, 0.0), 'outside': (0.0, 0.0, 1.0), 'solid': (1.0, 1.0, 0.0), 'domain': (0.0, 1.0, 0.0)}

# ---- palettes (01/10/2026, Claude Opus 5.5) ------------------------------------------------------
# A palette is a list of colour STOPS (t, (r, g, b)) over t in [-1, 1], piecewise linear between
# stops; it must hold stops at t = -1, 0 and +1. The value shown is t = sign(x) |x|^(1/contrast) with
# x = u * colour gain clamped to [-1, 1], then x exposure, clipped at 1. The ZERO LEVEL is the
# brightness (max channel) of the t = 0 stop: the zero stop keeps its hue and is scaled to the
# zero level (a grey zero stop is simply the grey `zero`), so the zero slider works on every
# palette; choosing a preset loads its own zero level. With the default 'teal-red' this is exactly
# the earlier formula Z + (end - Z)|x|^(1/contrast) (checked in app_tests_MIRROR.py).
KMAX = 17                                  # stops per palette (the kernel loops over them)
GRID_LIFT = 0.002                          # grid lines slid toward the eye by this fraction of their distance
PALETTES = {
    # the default since 30/09: the web app's teal / red ends, a grey zero (0.05 = the web app)
    'teal-red': [(-1.0, NEG_END), (0.0, (0.22, 0.22, 0.22)), (1.0, POS_END)],
    # black - grey - white: the sign by brightness alone
    'greyscale': [(-1.0, (0.0, 0.0, 0.0)), (0.0, (0.45, 0.45, 0.45)), (1.0, (1.0, 1.0, 1.0))],
    # Moreland's cool-warm (perceptually uniform diverging; matplotlib 'coolwarm' sampled at 9 points)
    'cool-warm': [(-1.0, (0.2298, 0.2987, 0.7537)), (-0.75, (0.3837, 0.5102, 0.9178)),
                  (-0.5, (0.5543, 0.6901, 0.9955)), (-0.25, (0.724, 0.8149, 0.9757)),
                  (0.0, (0.8674, 0.8644, 0.8626)), (0.25, (0.9595, 0.767, 0.6741)),
                  (0.5, (0.9567, 0.598, 0.4773)), (0.75, (0.8654, 0.3711, 0.2958)),
                  (1.0, (0.7057, 0.0156, 0.1502))],
    # cyclic, phase-friendly: matplotlib 'twilight' sampled at 17 points, x = -1 and +1 the SAME
    # colour (light), dark at 0, blue on the negative half, red on the positive half
    'cyclic': [(-1.0 + k / 8.0, c) for k, c in enumerate((
        (0.8858, 0.85, 0.888), (0.7675, 0.8098, 0.8325), (0.583, 0.7096, 0.7793), (0.448, 0.5924, 0.7557),
        (0.3841, 0.4614, 0.7309), (0.3699, 0.3164, 0.677), (0.3506, 0.166, 0.5615), (0.2701, 0.0755, 0.3606),
        (0.1849, 0.0794, 0.2131), (0.2913, 0.0749, 0.2576), (0.4538, 0.1162, 0.3097), (0.5966, 0.2072, 0.3126),
        (0.6981, 0.3383, 0.3221), (0.7626, 0.4872, 0.3868), (0.8003, 0.641, 0.5373), (0.8489, 0.7799, 0.7466),
        (0.8858, 0.85, 0.888)))],
    # hot / cold: dark zero, negative through blue and cyan to ice, positive through red and yellow to white
    'hot-cold': [(-1.0, (0.85, 1.0, 1.0)), (-0.66, (0.0, 0.78, 1.0)), (-0.33, (0.0, 0.22, 0.85)),
                 (0.0, (0.06, 0.06, 0.06)), (0.33, (0.85, 0.12, 0.0)), (0.66, (1.0, 0.75, 0.0)),
                 (1.0, (1.0, 1.0, 0.85))],
}
PALETTE_ORDER = list(PALETTES) + ['custom']
DEFAULT_CUSTOM = [(-1.0, (0.10, 0.45, 0.95)), (0.0, (0.20, 0.16, 0.26)), (1.0, (1.0, 0.55, 0.10))]


def stop_zero(stops):
    """The colour of the t = 0 stop."""
    for t, c in stops:
        if t == 0.0:
            return tuple(c)
    raise ValueError('a palette needs a stop at t = 0')


def zero_of(stops):
    """A palette's own zero level: the max channel of its zero stop."""
    return float(max(stop_zero(stops)))


def clean_stops(stops):
    """Validate / normalise a stop list: sorted, t clipped to [-1, 1], colours clipped to [0, 1],
    one stop per t, stops at -1, 0, +1 present (taken from the nearest stop if missing), <= KMAX."""
    d = {}
    for t, c in stops:
        t = round(float(min(1.0, max(-1.0, t))), 6)
        d[t] = tuple(float(min(1.0, max(0.0, x))) for x in c)
    if not d:
        d = {t: c for t, c in DEFAULT_CUSTOM}
    for need in (-1.0, 0.0, 1.0):
        if need not in d:
            near = min(d, key=lambda t: abs(t - need))
            d[need] = d[near]
    ts = sorted(d)
    if len(ts) > KMAX:                      # drop the interior stops closest to their neighbours
        keep = [-1.0, 0.0, 1.0]
        rest = [t for t in ts if t not in keep]
        ts = sorted(keep + rest[:KMAX - 3])
    return [(t, d[t]) for t in ts]


@dataclass
class Look:
    bg: str = 'near-black'
    zero: float = 0.22            # palette zero level (grey); 0.05 = the web app
    exposure: float = 1.0
    contrast: float = 1.0         # amplitude curve |x|^(1/contrast)
    solid: tuple = SOLID_COL
    rigid: tuple = RIGID_COL      # rigid solids (05/10/2026); pinned solids use `solid`
    outside: tuple = None         # None: the background's matching tone
    shading: float = 0.5          # 0 = unlit (MeshBasicMaterial); 3D only
    floor: bool = True            # 3D: draw the array outside the domain
    solid_h: float = 0.02         # 3D: height of solid cells (world units; the array half-width is 1).
                                  # Walls are painted on the rest plane, so a raised top shows
                                  # shifted by solid_h / tan(elevation): 0.026 at 38 deg (1.3% of the width)
    hatch: bool = True            # 2D: hatch outside the domain
    segment: bool = False         # debug: key colours per cell class (contrast measurement)
    # 01/10/2026 (Claude Opus 5.5): palette editor, page colour, transparency
    palette: str = 'teal-red'     # a PALETTES name or 'custom'
    custom: list = field(default_factory=lambda: list(DEFAULT_CUSTOM))   # the custom stops
    bg_rgb: tuple = None          # a custom page colour (None: the bg preset's)
    opacity: float = 1.0          # 3D: surface opacity (real alpha blending); 2D: the field mixed toward the page
    grid: float = 0.0             # 3D: grid-line opacity (0 = no grid)
    grid_every: int = 8           # 3D: a grid line every k displayed vertices
    grid_col: tuple = (0.92, 0.92, 1.0)
    grid_field: bool = False      # grid lines in the field's colours (brightened) instead of grid_col

    def stops(self):
        """The active palette's stops, the zero stop scaled to the zero level."""
        st = clean_stops(self.custom) if self.palette == 'custom' else PALETTES[self.palette]
        out = []
        for t, c in st:
            if t == 0.0:
                m = max(c)
                c = tuple(x / m * self.zero for x in c) if m > 1e-6 else (self.zero,) * 3
            out.append((t, tuple(c)))
        return out

    def stops_raw(self):
        return clean_stops(self.custom) if self.palette == 'custom' else PALETTES[self.palette]

    def set_palette(self, name):
        """Choose a palette; it brings its own zero level."""
        self.palette = name
        self.zero = zero_of(clean_stops(self.custom) if name == 'custom' else PALETTES[name])

    def cycle_palette(self):
        self.set_palette(PALETTE_ORDER[(PALETTE_ORDER.index(self.palette) + 1) % len(PALETTE_ORDER)])

    def bg_col(self):
        if self.bg_rgb is not None:
            return tuple(self.bg_rgb)
        return BACKGROUNDS[self.bg][0]

    def outside_col(self):
        return tuple(self.outside) if self.outside is not None else BACKGROUNDS[self.bg][1]

    def cycle_bg(self):
        self.bg = BG_ORDER[(BG_ORDER.index(self.bg) + 1) % len(BG_ORDER)]
        self.bg_rgb = None

    def lights(self):
        """(ambient, point-light colour) for GGUI; (1, 0) = unlit."""
        s = max(0.0, min(1.0, self.shading))
        return 1.0 - 0.55 * s, 0.8 * s


def web_palette(v):
    """The web app's colorOf, 'div' palette, cg applied by the caller (reference for the tests)."""
    x = max(-1.0, min(1.0, v))
    if x >= 0:
        return (0.05 + 0.83 * x, 0.05 + 0.08 * x, 0.05 + 0.05 * x)
    return (0.05 + 0.00 * (-x), 0.05 + 0.57 * (-x), 0.05 + 0.47 * (-x))


def palette_np(v, look):
    """The display palette on the host (numpy), identical to the kernel's palette()."""
    x = np.clip(np.asarray(v, dtype=np.float64), -1, 1)
    a = np.abs(x) ** (1.0 / max(1e-3, look.contrast))
    t = np.where(x >= 0, a, -a)
    st = look.stops()
    ts = np.array([q[0] for q in st])
    cs = np.array([q[1] for q in st], np.float64)
    out = np.stack([np.interp(t, ts, cs[:, ch]) for ch in range(3)], -1)
    return np.minimum(out * look.exposure, 1.0)


# ---- settings file (the user's palette across sessions) -------------------------------------------
SETTINGS_KEYS = ('palette', 'zero', 'custom', 'bg_rgb', 'opacity', 'grid', 'grid_every', 'grid_col', 'grid_field')


def look_settings(look):
    d = {k: getattr(look, k) for k in SETTINGS_KEYS}
    d['custom'] = [[t, list(c)] for t, c in clean_stops(look.custom)]
    for k in ('bg_rgb', 'grid_col'):
        d[k] = None if d[k] is None else [float(x) for x in d[k]]
    return {'version': 1, 'look': d}


def save_settings(look, path):
    """Write the palette settings (JSON). Written to a temp file first, then replaced."""
    import json
    import os
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(look_settings(look), f, indent=1)
    os.replace(tmp, path)


def load_settings(look, path):
    """Apply a settings file to `look`; a missing or broken file leaves `look` as it is.
    Returns None on success, else the reason (shown in the panel)."""
    import json
    import os
    if not os.path.exists(path):
        return 'no settings file yet'
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)['look']
        custom = clean_stops([(float(t), tuple(float(x) for x in c)) for t, c in d.get('custom', DEFAULT_CUSTOM)])
        pal = d.get('palette', look.palette)
        if pal not in PALETTE_ORDER:
            pal = 'teal-red'
        look.custom = custom
        look.palette = pal
        look.zero = float(min(1.0, max(0.0, d.get('zero', zero_of(look.stops_raw())))))
        bg = d.get('bg_rgb')
        look.bg_rgb = None if bg is None else tuple(float(min(1, max(0, x))) for x in bg)
        look.opacity = float(min(1.0, max(0.0, d.get('opacity', 1.0))))
        look.grid = float(min(1.0, max(0.0, d.get('grid', 0.0))))
        look.grid_every = int(min(64, max(1, d.get('grid_every', 8))))
        gc = d.get('grid_col')
        if gc is not None:
            look.grid_col = tuple(float(min(1, max(0, x))) for x in gc)
        look.grid_field = bool(d.get('grid_field', False))
        return None
    except Exception as ex:                     # a hand-edited / truncated file: keep the defaults
        return f'settings file not read ({type(ex).__name__}: {ex})'


# ---- the parameter block the kernels read (one small field, re-uploaded when it changes) -------
# 0 zero, 1 gamma, 2 exposure, 3 cg, 4 hs, 5 solid_h, 6-8 solid, 9-11 outside, 12-14 bg,
# 15 cx, 16 cy, 17 px size (2D), 18 segment, 19 hatch, 20-22 rigid solid colour (05/10/2026),
# 23 rigid stripes in 2D (1 = on), 26 stop count, 27 opacity,
# 28 grid opacity, 29-31 grid colour (or -1: the field colour), then the stops: (t, r, g, b) x KMAX at 32
LP_STOPS = 32
NLP = LP_STOPS + 4 * KMAX


@ti.func
def palette(v, LP: ti.template()):
    x = ti.max(-1.0, ti.min(1.0, v))
    a = ti.pow(ti.abs(x), LP[1])
    t = a
    if x < 0:
        t = -a
    K = ti.cast(LP[26], ti.i32)
    out = ti.Vector([LP[LP_STOPS + 1], LP[LP_STOPS + 2], LP[LP_STOPS + 3]], ti.f32)
    for k in ti.static(range(KMAX - 1)):
        if k + 1 < K:
            t0 = LP[LP_STOPS + 4 * k]
            t1 = LP[LP_STOPS + 4 * k + 4]
            if t0 <= t and t <= t1 and t1 > t0:
                c0 = ti.Vector([LP[LP_STOPS + 4 * k + 1], LP[LP_STOPS + 4 * k + 2], LP[LP_STOPS + 4 * k + 3]], ti.f32)
                c1 = ti.Vector([LP[LP_STOPS + 4 * k + 5], LP[LP_STOPS + 4 * k + 6], LP[LP_STOPS + 4 * k + 7]], ti.f32)
                out = c0 + (c1 - c0) * ((t - t0) / (t1 - t0))
    out = out * LP[2]
    return ti.min(out, 1.0)


@ti.func
def lp3(LP: ti.template(), k: ti.template()):
    return ti.Vector([LP[k], LP[k + 1], LP[k + 2]], ti.f32)


@ti.kernel
def k_verts(U: ti.types.ndarray(ndim=3), s: ti.i32, M: ti.types.ndarray(ndim=2), DOM: ti.types.ndarray(ndim=2),
            C: ti.types.ndarray(ndim=2), hasC: ti.i32,
            k: ti.i32, nd: ti.i32, tri: ti.i32, LP: ti.template(), pos: ti.template(), col: ti.template()):
    """One display vertex per k-th cell: world (x/c, height, -y/c) and the colour of its class.
    C / hasC (05/10/2026): the core's class codes (m.C, low two bits 1 pinned / 2 rigid) while it keeps
    them; hasC = 0 (a 1x1 dummy C) means every solid is pinned."""
    n = U.shape[1]
    c = (n - 1) * 0.5
    sc = 1.0 / c
    seg = LP[18] > 0.5
    for p, q in ti.ndrange(nd, nd):
        i = ti.min(p * k, n - 1)
        j = ti.min(q * k, n - 1)
        da = i - c
        db = j - c
        x = da
        y = db
        if tri:
            x = da + db * 0.5
            y = db * 0.8660254037844386
        h = ti.f32(0.0)
        cl = lp3(LP, 9)                                   # outside the domain: the floor
        al = ti.f32(1.0)
        if seg:
            cl = ti.Vector([0.0, 0.0, 1.0], ti.f32)
        if DOM[i, j] != 0:
            if M[i, j] == 0:                              # solid
                h = LP[5]
                cl = lp3(LP, 6)
                if hasC != 0:
                    if (ti.cast(C[i, j], ti.i32) & 3) == 2:   # rigid
                        cl = lp3(LP, 20)
                if seg:
                    cl = ti.Vector([1.0, 1.0, 0.0], ti.f32)
            else:
                v = ti.cast(U[s, i, j], ti.f32)
                h = v * LP[4] * 0.3
                cl = palette(v * LP[3], LP)
                al = LP[27]                               # the surface opacity (field cells only)
                if seg:
                    cl = ti.Vector([0.0, 1.0, 0.0], ti.f32)
                    al = 1.0
        pos[q * nd + p] = ti.Vector([x * sc, h, -y * sc], ti.f32)
        col[q * nd + p] = ti.Vector([cl[0], cl[1], cl[2], al], ti.f32)


@ti.kernel
def k_grid_verts(pos: ti.template(), col: ti.template(), nrm: ti.template(), vid: ti.template(), n: ti.i32,
                 ex: ti.f32, ey: ti.f32, ez: ti.f32, lift: ti.f32, LP: ti.template(),
                 gpos: ti.template(), gcol: ti.template(), gnrm: ti.template()):
    """The grid-line copy of the surface: every vertex slid toward the eye by `lift` of its distance
    (so the lines win the depth test against their own surface, at any view), coloured with the
    grid colour (or the field colour, brightened) at the grid opacity."""
    e = ti.Vector([ex, ey, ez], ti.f32)
    for i in range(n):                              # only the vertices on grid lines (gathered)
        v = vid[i]
        P = pos[v]
        gpos[i] = P + (e - P) * lift
        gnrm[i] = nrm[v]
        c = col[v]
        g = ti.Vector([LP[29], LP[30], LP[31]], ti.f32)
        if LP[29] < 0:
            g = ti.min(ti.Vector([c[0], c[1], c[2]], ti.f32) * 1.6 + 0.08, 1.0)
        gcol[i] = ti.Vector([g[0], g[1], g[2], LP[28]], ti.f32)


@ti.kernel
def k_normals(pos: ti.template(), nd: ti.i32, nrm: ti.template()):
    """Vertex normals from central differences on the display grid (GGUI's own gen_normals runs
    three atomics per triangle every frame, and turns padded (0,0,0) triangles into a NaN normal)."""
    for p, q in ti.ndrange(nd, nd):
        p0, p1 = ti.max(p - 1, 0), ti.min(p + 1, nd - 1)
        q0, q1 = ti.max(q - 1, 0), ti.min(q + 1, nd - 1)
        t1 = pos[q * nd + p1] - pos[q * nd + p0]
        t2 = pos[q1 * nd + p] - pos[q0 * nd + p]
        nv = t2.cross(t1)
        L = nv.norm()
        out = ti.Vector([0.0, 1.0, 0.0], ti.f32)
        if L > 1e-12:
            out = nv / L
        nrm[q * nd + p] = out


@ti.func
def pixel_cell(px, py, W, H, n, tri, LP: ti.template()):
    """The lattice cell a 2D-view pixel samples (the nearest in index space, as in round 1), or
    (-1, -1) outside the array. (cx, cy, px size) = LP[15..17]; see app_camera_MIRROR.View2D."""
    c = (n - 1) * 0.5
    x = LP[15] + (px + 0.5 - W * 0.5) * LP[17]
    y = LP[16] + (py + 0.5 - H * 0.5) * LP[17]
    fi = c + x
    fj = c + y
    if tri:
        db = y / 0.8660254037844386
        fi = c + x - db * 0.5
        fj = c + db
    i = ti.cast(ti.floor(fi + 0.5), ti.i32)
    j = ti.cast(ti.floor(fj + 0.5), ti.i32)
    if not (0 <= i < n and 0 <= j < n):
        i = -1
        j = -1
    return i, j


@ti.kernel
def k_image(U: ti.types.ndarray(ndim=3), s: ti.i32, M: ti.types.ndarray(ndim=2), DOM: ti.types.ndarray(ndim=2),
            C: ti.types.ndarray(ndim=2), hasC: ti.i32, tri: ti.i32, LP: ti.template(), img: ti.template()):
    """Top view in PHYSICAL coordinates: each pixel shows its nearest lattice cell (so the
    triangular lattice's rhombic array has the right shape). Rigid solids (05/10/2026): the rigid
    colour, with thin dark diagonal stripes (LP[23]) so they stay apart from pinned ones without
    colour."""
    n = U.shape[1]
    W, H = img.shape[0], img.shape[1]
    seg = LP[18] > 0.5
    for px, py in img:
        i, j = pixel_cell(px, py, W, H, n, tri, LP)
        out = lp3(LP, 12)                                            # outside the array
        if seg:
            out = ti.Vector([1.0, 0.0, 0.0], ti.f32)
        if i >= 0:
            if DOM[i, j] == 0:                                       # outside the domain
                out = lp3(LP, 9)
                if LP[19] > 0.5 and ((px + py) // 5) % 3 == 0:       # a diagonal hatch
                    out = out * 1.45 + 0.012
                if seg:
                    out = ti.Vector([0.0, 0.0, 1.0], ti.f32)
            elif M[i, j] == 0:                                       # solid
                out = lp3(LP, 6)
                if hasC != 0:
                    if (ti.cast(C[i, j], ti.i32) & 3) == 2:              # rigid: colour + stripes
                        out = lp3(LP, 20)
                        if LP[23] > 0.5 and ((px - py + 4096) // 2) % 4 == 0:   # 2 px of every 8
                            out = out * 0.55
                if seg:
                    out = ti.Vector([1.0, 1.0, 0.0], ti.f32)
            else:
                out = palette(ti.cast(U[s, i, j], ti.f32) * LP[3], LP)
                # opacity < 1: the field over the page, as the 3D surface shows over empty space
                out = lp3(LP, 12) + (out - lp3(LP, 12)) * LP[27]
                if seg:
                    out = ti.Vector([0.0, 1.0, 0.0], ti.f32)
        img[px, py] = out


@ti.kernel
def k_image_index(n: ti.i32, tri: ti.i32, LP: ti.template(), out: ti.types.ndarray(ndim=3)):
    """The (i, j) each pixel samples -- the SAME pixel_cell() the image uses (mapping tests)."""
    W, H = out.shape[0], out.shape[1]
    for px, py in ti.ndrange(W, H):
        i, j = pixel_cell(px, py, W, H, n, tri, LP)
        out[px, py, 0] = i
        out[px, py, 1] = j


# ---- mesh -------------------------------------------------------------------------------------
def mesh_indices(dom, k, tri):
    """Triangles of the displayed (strided) grid, one per LATTICE triangle.
    Returns (nd, T, n_dom): T lists first the DOMAIN triangles -- the web app's per-triangle rule
    (28/09): all three corners in the domain, so straight mask edges draw straight -- then the
    FLOOR triangles (the rest of the array: the other half of each partial quad, and every
    quad/triangle with fewer corners), so drawing T[:n_dom] is the web app's view and drawing all
    of T adds the array outside the domain. dom is the DOMAIN mask (the shape): solid cells are
    part of the domain mesh (drawn raised and grey), so painting never rebuilds the triangles."""
    sub = dom[::k, ::k].astype(bool)
    nd = sub.shape[0]
    P, Q = np.meshgrid(np.arange(nd - 1), np.arange(nd - 1), indexing='ij')
    a = Q * nd + P                                  # vertex index = q*nd + p for grid point (p, q)
    b, c2 = a + 1, a + nd
    d = c2 + 1
    A, B, C, Dm = sub[:-1, :-1], sub[1:, :-1], sub[:-1, 1:], sub[1:, 1:]
    dom_t, flo_t = [], []
    st = lambda sel, t: np.stack([t[0][sel], t[1][sel], t[2][sel]], 1)
    if tri:
        t1 = A & C & B
        t2 = B & C & Dm
        dom_t += [st(t1, (a, c2, b)), st(t2, (b, c2, d))]
        flo_t += [st(~t1, (a, c2, b)), st(~t2, (b, c2, d))]
    else:
        k4 = A.astype(np.int8) + B + C + Dm
        full = k4 == 4
        dom_t += [st(full, (a, c2, b)), st(full, (b, c2, d))]
        three = k4 == 3
        # (missing corner, the domain triangle, the floor triangle = the quad's other half)
        for miss, td, tf in ((~A, (b, c2, d), (a, c2, b)), (~Dm, (a, c2, b), (b, c2, d)),
                             (~B, (a, c2, d), (a, d, b)), (~C, (a, d, b), (a, c2, d))):
            sel = three & miss
            dom_t.append(st(sel, td))
            flo_t.append(st(sel, tf))
        few = k4 <= 2
        flo_t += [st(few, (a, c2, b)), st(few, (b, c2, d))]
    Td = np.concatenate(dom_t, 0).astype(np.int32).reshape(-1)
    Tf = np.concatenate(flo_t, 0).astype(np.int32).reshape(-1)
    return nd, np.concatenate([Td, Tf]), Td.size


def grid_indices(dom, k, every):
    """Grid lines over the displayed grid as DEGENERATE triangles (a, b, b): GGUI's wireframe pass
    draws their one edge a-b (measured 01/10/2026). Lines run along both index axes every `every`
    displayed vertices, following the surface vertex by vertex; an edge is kept when both ends are in
    the domain. Returns an int32 array (3 per edge)."""
    sub = dom[::k, ::k].astype(bool)
    nd = sub.shape[0]
    every = max(1, int(every))
    out = []
    lines = np.arange(0, nd, every)
    # lines of constant p: edges (p, q) - (p, q + 1)
    P, Q = np.meshgrid(lines, np.arange(nd - 1), indexing='ij')
    ok = sub[P, Q] & sub[P, Q + 1]
    a, b = (Q * nd + P)[ok], ((Q + 1) * nd + P)[ok]
    out.append(np.stack([a, b, b], 1))
    # lines of constant q: edges (p, q) - (p + 1, q)
    P, Q = np.meshgrid(np.arange(nd - 1), lines, indexing='ij')
    ok = sub[P, Q] & sub[P + 1, Q]
    a, b = (Q * nd + P)[ok], (Q * nd + P + 1)[ok]
    out.append(np.stack([a, b, b], 1))
    T = np.concatenate(out, 0).astype(np.int32).reshape(-1)
    # compact: only the vertices the lines use (a quarter of the mesh at every = 8), so GGUI uploads
    # those, not the whole mesh (the full copy cost ~30 ms a frame at mesh 513, measured 01/10/2026)
    verts, T = np.unique(T, return_inverse=True)
    return verts.astype(np.int32), T.astype(np.int32)


# ---- contrast measurement (sRGB -> CIE L*) ----------------------------------------------------
def srgb_to_lin(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rel_lum(rgb):
    lin = srgb_to_lin(rgb)
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def lstar(Y):
    Y = np.asarray(Y, dtype=np.float64)
    return np.where(Y > (6 / 29) ** 3, 116 * np.cbrt(Y) - 16, Y * (29 / 3) ** 3)


def classify_seg(seg_rgb):
    """Class per pixel of a SEGMENT render (uint8 RGB): 0 bg, 1 outside, 2 solid, 3 domain,
    -1 mixed (an edge pixel blended between classes; left out of every mean)."""
    s = seg_rgb.astype(np.int32)
    out = np.full(s.shape[:2], -1, np.int8)
    for cls, key in enumerate(('bg', 'outside', 'solid', 'domain')):
        ref = np.array(SEG[key]) * 255
        out[np.abs(s - ref).max(axis=2) <= 8] = cls
    return out


def contrast_stats(rgb, cls):
    """Mean L* and mean relative luminance per class; the WCAG ratio domain vs background."""
    L = rel_lum(rgb.astype(np.float64) / 255.0)
    out = {}
    for k, name in enumerate(('bg', 'outside', 'solid', 'domain')):
        sel = cls == k
        if sel.any():
            Y = float(L[sel].mean())
            out[name] = {'px': int(sel.sum()), 'Y': Y, 'Lstar': float(lstar(Y)),
                         'Lstar_mean': float(lstar(L[sel]).mean())}
    if 'domain' in out and 'bg' in out:
        out['ratio_domain_bg'] = (out['domain']['Y'] + 0.05) / (out['bg']['Y'] + 0.05)
        out['dL_domain_bg'] = out['domain']['Lstar_mean'] - out['bg']['Lstar_mean']
    if 'outside' in out and 'bg' in out:
        out['dL_outside_bg'] = out['outside']['Lstar_mean'] - out['bg']['Lstar_mean']
    if 'outside' in out and 'domain' in out:
        out['dL_domain_outside'] = out['domain']['Lstar_mean'] - out['outside']['Lstar_mean']
    return out


class Renderer:
    """The display fields (after ti.init) and the per-frame draw calls for both views."""

    def __init__(self, win_size, mask_dtype):
        self.W, self.H = win_size
        self.LP = ti.field(ti.f32, shape=NLP)
        self._lp = None
        self.img = ti.Vector.field(3, ti.f32, shape=(self.W, self.H))
        self._imgs = {}                         # other sizes' buffers (resize)
        self.mask_dtype = mask_dtype
        self.DOM = None
        self.pos = self.col = self.nrm = self.idx = None
        self.nd = 0
        self.n_idx = self.n_dom = 0
        self.k = 1
        # overlays: lines (2D in [0,1] screen units / 3D world) with per-vertex colours
        self.ov2 = ti.Vector.field(2, ti.f32, shape=4096)
        self.ov2c = ti.Vector.field(3, ti.f32, shape=4096)
        self.ov3 = ti.Vector.field(3, ti.f32, shape=4096)
        self.ov3c = ti.Vector.field(3, ti.f32, shape=4096)
        # the class codes' stand-in while the core keeps none (every solid pinned): same dtype as
        # m.C (MASK_DT), so the display kernels compile once (05/10/2026)
        self.C0 = ti.ndarray(dtype=ti.i32 if mask_dtype == np.int32 else ti.u8, shape=(1, 1))
        self.mark = ti.Vector.field(3, ti.f32, shape=16)
        self.markc = ti.Vector.field(3, ti.f32, shape=16)
        # transparency (01/10/2026): back-to-front triangle orders per view sector, grid-line mesh
        self.T_dom = self.T_floor = None
        self._sorted = {}
        self._cent = None
        self.gpos = self.gcol = None
        self.gidx = None
        self._gkey = None
        self.stats = {}                         # what the last draw3d drew (tests)

    def set_params(self, look, cg, hs, view2d_cx=0.0, view2d_cy=0.0, pxs=1.0):
        lp = np.zeros(NLP, np.float32)
        lp[0] = look.zero
        lp[1] = 1.0 / max(1e-3, look.contrast)
        lp[2] = look.exposure
        lp[3] = cg
        lp[4] = hs
        lp[5] = look.solid_h
        lp[6:9] = look.solid
        lp[9:12] = look.outside_col()
        lp[12:15] = look.bg_col()
        lp[15], lp[16], lp[17] = view2d_cx, view2d_cy, pxs
        lp[18] = 1.0 if look.segment else 0.0
        lp[19] = 1.0 if look.hatch else 0.0
        lp[20:23] = look.rigid
        lp[23] = 1.0
        st = look.stops()
        lp[26] = len(st)
        lp[27] = min(1.0, max(0.0, look.opacity))
        # each degenerate triangle (a, b, b) draws its edge TWICE (a-b and b-a), so GGUI blends a
        # grid pixel twice (measured 01/10/2026: alpha 0.6 showed as 0.84 = 1 - 0.4^2): per pass
        # alpha 1 - sqrt(1 - grid) gives the grid opacity asked for
        lp[28] = 1.0 - math.sqrt(1.0 - min(1.0, max(0.0, look.grid)))
        lp[29:32] = (-1.0, -1.0, -1.0) if look.grid_field else look.grid_col
        for k, (t, c) in enumerate(st):
            lp[LP_STOPS + 4 * k] = t
            lp[LP_STOPS + 4 * k + 1:LP_STOPS + 4 * k + 4] = c
        if self._lp is None or not np.array_equal(lp, self._lp):
            self.LP.from_numpy(lp)
            self._lp = lp

    def set_domain(self, dom, N, k, tri):
        """(Re)build the device copy of the DOMAIN mask and the mesh (on reset / new shape)."""
        if self.DOM is None or self.DOM.shape[0] != N:
            self.DOM = ti.ndarray(dtype=ti.i32 if self.mask_dtype == np.int32 else ti.u8, shape=(N, N))
        self.DOM.from_numpy(np.ascontiguousarray(dom, dtype=self.mask_dtype))
        self.k = k
        nd, T, n_dom = mesh_indices(dom, k, tri)
        if self.pos is None or self.pos.shape[0] != nd * nd:
            self.pos = ti.Vector.field(3, ti.f32, shape=nd * nd)
            # RGBA: GGUI 1.7.4 blends a vec4 per-vertex colour (src-alpha over, measured 01/10/2026)
            self.col = ti.Vector.field(4, ti.f32, shape=nd * nd)
            self.nrm = ti.Vector.field(3, ti.f32, shape=nd * nd)
            # never leave a normal at 0: GGUI's mesh draws BLACK where the normal is 0 even with no
            # point light (measured 30/09/2026: unlit frame after a new N was all black until
            # k_normals had run once)
            self.nrm.fill(ti.Vector([0.0, 1.0, 0.0]))
        # the indices stay a HOST array: GGUI 1.7's scene.mesh() calls indices.to_numpy() on EVERY
        # call when they are a ti.field (measured 30/09/2026 at N = 2049, 2.1M triangles: 12 ms for
        # that copy alone, and ~50 ms more per frame than a numpy array); a numpy array is passed on
        self.idx = np.ascontiguousarray(T if T.size else np.zeros(3, np.int32), dtype=np.int32)
        self.nd, self.n_idx, self.n_dom = nd, T.size, n_dom
        self.T_dom, self.T_floor = T[:n_dom], T[n_dom:]
        self._sorted, self._cent = {}, None
        self._dom_k = (dom, k)
        self._gkey = None

    SECTORS = 32

    def sorted_idx(self, eye):
        """The triangles for a TRANSLUCENT surface: the (opaque) floor first, then the domain
        triangles back to front along the horizontal view direction. GGUI writes depth and does not
        sort, so a near triangle drawn first would hide the far ones behind it (measured
        01/10/2026). Orders are cached per 360/32 deg sector of the view azimuth. Returns (indices,
        number of floor indices)."""
        sec = int(round(math.atan2(eye[2], eye[0]) / (2 * math.pi / self.SECTORS))) % self.SECTORS
        hit = self._sorted.get(sec)
        if hit is not None:
            return hit
        if self._cent is None:
            P = self.pos.to_numpy()
            tri = self.T_dom.reshape(-1, 3)
            self._cent = (P[tri, 0].mean(1), P[tri, 2].mean(1))
        ph = sec * 2 * math.pi / self.SECTORS
        key = self._cent[0] * math.cos(ph) + self._cent[1] * math.sin(ph)    # larger = nearer the eye
        order = np.argsort(key, kind='stable')
        dom = self.T_dom.reshape(-1, 3)[order].reshape(-1)
        out = (np.ascontiguousarray(np.concatenate([self.T_floor, dom]), dtype=np.int32), self.T_floor.size)
        self._sorted[sec] = out
        return out

    def grid_idx(self, every):
        key = (self.nd, self.k, int(every), id(self._dom_k[0]))
        if self._gkey != key:
            verts, T = grid_indices(self._dom_k[0], self.k, every)
            n = max(1, verts.size)
            if self.gpos is None or self.gpos.shape[0] < n:
                cap = 1 << (n - 1).bit_length()     # grow in powers of two: few reallocations / recompiles
                self.gpos = ti.Vector.field(3, ti.f32, shape=cap)
                self.gcol = ti.Vector.field(4, ti.f32, shape=cap)
                self.gnrm = ti.Vector.field(3, ti.f32, shape=cap)
                self.gnrm.fill(ti.Vector([0.0, 1.0, 0.0]))
                self.gvid = ti.field(ti.i32, shape=cap)
            vid = np.zeros(self.gvid.shape[0], np.int32)
            vid[:verts.size] = verts
            self.gvid.from_numpy(vid)
            self.gn = verts.size
            self.gidx = np.ascontiguousarray(T if T.size else np.zeros(3, np.int32), dtype=np.int32)
            self._gkey = key
        return self.gidx

    def classes(self, m):
        """(C, hasC) for the display kernels: the core's class codes while it keeps them (a solid
        exists or the edge is rigid), else the 1x1 stand-in (every solid pinned)."""
        C = getattr(m, 'C', None)
        return (C, 1) if C is not None else (self.C0, 0)

    def draw3d(self, m, look, scene, pose_eye, wire):
        s = (m.slot + 1) % 3
        C, hasC = self.classes(m)
        k_verts(m.U, s, m.M, self.DOM, C, hasC, self.k, self.nd, 1 if m.tri() else 0, self.LP, self.pos, self.col)
        amb, pl = look.lights()
        if look.segment:
            amb, pl = 1.0, 0.0
        scene.ambient_light((amb, amb, amb))
        if pl > 0:
            k_normals(self.pos, self.nd, self.nrm)
            e = pose_eye
            scene.point_light(pos=(e[0], e[1] + 0.25, e[2]), color=(pl, pl, pl))
        seg = look.segment
        translucent = look.opacity < 1.0 and not seg
        self.stats = {'translucent': translucent, 'grid': 0}
        if translucent and self.n_dom:
            idx, nfl = self.sorted_idx(pose_eye)
            if not look.floor:
                idx = idx[nfl:]
            cnt = idx.size
            self.stats['order'] = idx
            if cnt:
                scene.mesh(self.pos, indices=idx, normals=self.nrm, per_vertex_color=self.col,
                           two_sided=True, show_wireframe=wire, index_count=cnt)
        else:
            cnt = self.n_idx if look.floor else self.n_dom
            if cnt:
                # pass only the drawn prefix (domain triangles come first): GGUI uploads the whole array
                scene.mesh(self.pos, indices=self.idx[:cnt], normals=self.nrm, per_vertex_color=self.col,
                           two_sided=True, show_wireframe=wire, index_count=cnt)
        if look.grid > 0 and not seg and not wire and self.n_dom:
            # the grid AFTER the surface, slid toward the eye: its lines blend over the surface they
            # lie on (drawn before, a line pixel would blend over the page and the surface would fail
            # the depth test there, so a faint grid would show DARKER than the surface)
            gi = self.grid_idx(look.grid_every)
            k_grid_verts(self.pos, self.col, self.nrm, self.gvid, self.gn, float(pose_eye[0]), float(pose_eye[1]),
                         float(pose_eye[2]), GRID_LIFT, self.LP, self.gpos, self.gcol, self.gnrm)
            scene.mesh(self.gpos, indices=gi, normals=self.gnrm, per_vertex_color=self.gcol, vertex_count=self.gn,
                       two_sided=True, show_wireframe=True, index_count=gi.size)
            self.stats['grid'] = gi.size // 3

    def resize(self, W, H):
        """A new 2D image buffer size (window resized / maximised, 01/10/2026). Buffers are kept per
        size: maximise / restore swaps between two without reallocating or recompiling."""
        if (W, H) == (self.W, self.H):
            return
        self._imgs[(self.W, self.H)] = self.img
        img = self._imgs.pop((W, H), None)
        self.img = img if img is not None else ti.Vector.field(3, ti.f32, shape=(W, H))
        self.W, self.H = W, H

    def draw2d(self, m, canvas):
        s = (m.slot + 1) % 3
        C, hasC = self.classes(m)
        k_image(m.U, s, m.M, self.DOM, C, hasC, 1 if m.tri() else 0, self.LP, self.img)
        canvas.set_image(self.img)

    # overlays -----------------------------------------------------------------------------------
    def lines2d(self, canvas, segs, width=0.0022):
        """segs: [((x0, y0), (x1, y1), rgb)] in screen units [0,1] (y up). GGUI 1.7's canvas.lines
        has no vertex count: it draws the whole field, so unused slots are parked off-screen."""
        n = min(len(segs), self.ov2.shape[0] // 2)
        if n == 0:
            return
        P = np.full((self.ov2.shape[0], 2), -5.0, np.float32)
        C = np.zeros((self.ov2.shape[0], 3), np.float32)
        for q, (a, b, c) in enumerate(segs[:n]):
            P[2 * q], P[2 * q + 1], C[2 * q], C[2 * q + 1] = a, b, c, c
        self.ov2.from_numpy(P)
        self.ov2c.from_numpy(C)
        canvas.lines(self.ov2, width, per_vertex_color=self.ov2c)

    def lines3d(self, scene, segs, width=2.0):
        n = min(len(segs), self.ov3.shape[0] // 2)
        if n == 0:
            return
        P = np.zeros((self.ov3.shape[0], 3), np.float32)
        C = np.zeros((self.ov3.shape[0], 3), np.float32)
        for q, (a, b, c) in enumerate(segs[:n]):
            P[2 * q], P[2 * q + 1], C[2 * q], C[2 * q + 1] = a, b, c, c
        self.ov3.from_numpy(P)
        self.ov3c.from_numpy(C)
        scene.lines(self.ov3, width, per_vertex_color=self.ov3c, vertex_count=2 * n)

    def markers3d(self, scene, pts, radius=0.012):
        n = min(len(pts), self.mark.shape[0])
        if n == 0:
            return
        P = np.zeros((self.mark.shape[0], 3), np.float32)
        C = np.zeros((self.mark.shape[0], 3), np.float32)
        for q, (p, c) in enumerate(pts[:n]):
            P[q], C[q] = p, c
        self.mark.from_numpy(P)
        self.markc.from_numpy(C)
        scene.particles(self.mark, radius, per_vertex_color=self.markc, index_count=n)

