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
*Generated from scratch -- Claude Opus 5.5 -- 30/09/2026*
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
SOLID_COL = (0.84, 0.84, 0.80)            # the plate lettering: off-white
# the palette's ends (the web app's 'div': teal (-) and red (+))
POS_END = (0.88, 0.13, 0.10)
NEG_END = (0.05, 0.62, 0.52)
# segmentation colours (contrast measurement only)
SEG = {'bg': (1.0, 0.0, 0.0), 'outside': (0.0, 0.0, 1.0), 'solid': (1.0, 1.0, 0.0), 'domain': (0.0, 1.0, 0.0)}


@dataclass
class Look:
    bg: str = 'near-black'
    zero: float = 0.22            # palette zero level (grey); 0.05 = the web app
    exposure: float = 1.0
    contrast: float = 1.0         # amplitude curve |x|^(1/contrast)
    solid: tuple = SOLID_COL
    outside: tuple = None         # None: the background's matching tone
    shading: float = 0.5          # 0 = unlit (MeshBasicMaterial); 3D only
    floor: bool = True            # 3D: draw the array outside the domain
    solid_h: float = 0.02         # 3D: height of solid cells (world units; the array half-width is 1).
                                  # Walls are painted on the rest plane, so a raised top shows
                                  # shifted by solid_h / tan(elevation): 0.026 at 38 deg (1.3% of the width)
    hatch: bool = True            # 2D: hatch outside the domain
    segment: bool = False         # debug: key colours per cell class (contrast measurement)

    def bg_col(self):
        return BACKGROUNDS[self.bg][0]

    def outside_col(self):
        return tuple(self.outside) if self.outside is not None else BACKGROUNDS[self.bg][1]

    def cycle_bg(self):
        self.bg = BG_ORDER[(BG_ORDER.index(self.bg) + 1) % len(BG_ORDER)]

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
    z = look.zero
    end = np.where(x[..., None] >= 0, np.array(POS_END), np.array(NEG_END))
    out = z + (end - z) * a[..., None]
    return np.minimum(out * look.exposure, 1.0)


# ---- the parameter block the kernels read (one small field, re-uploaded when it changes) -------
# 0 zero, 1 gamma, 2 exposure, 3 cg, 4 hs, 5 solid_h, 6-8 solid, 9-11 outside, 12-14 bg,
# 15 cx, 16 cy, 17 px size (2D), 18 segment, 19 hatch, 20-22 pos end, 23-25 neg end
NLP = 32


@ti.func
def palette(v, LP: ti.template()):
    x = ti.max(-1.0, ti.min(1.0, v))
    a = ti.pow(ti.abs(x), LP[1])
    z = LP[0]
    zv = ti.Vector([z, z, z], ti.f32)
    end = ti.Vector([LP[20], LP[21], LP[22]], ti.f32)
    if x < 0:
        end = ti.Vector([LP[23], LP[24], LP[25]], ti.f32)
    out = (zv + (end - zv) * a) * LP[2]
    return ti.min(out, 1.0)


@ti.func
def lp3(LP: ti.template(), k: ti.template()):
    return ti.Vector([LP[k], LP[k + 1], LP[k + 2]], ti.f32)


@ti.kernel
def k_verts(U: ti.types.ndarray(ndim=3), s: ti.i32, M: ti.types.ndarray(ndim=2), DOM: ti.types.ndarray(ndim=2),
            k: ti.i32, nd: ti.i32, tri: ti.i32, LP: ti.template(), pos: ti.template(), col: ti.template()):
    """One display vertex per k-th cell: world (x/c, height, -y/c) and the colour of its class."""
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
        if seg:
            cl = ti.Vector([0.0, 0.0, 1.0], ti.f32)
        if DOM[i, j] != 0:
            if M[i, j] == 0:                              # solid
                h = LP[5]
                cl = lp3(LP, 6)
                if seg:
                    cl = ti.Vector([1.0, 1.0, 0.0], ti.f32)
            else:
                v = ti.cast(U[s, i, j], ti.f32)
                h = v * LP[4] * 0.3
                cl = palette(v * LP[3], LP)
                if seg:
                    cl = ti.Vector([0.0, 1.0, 0.0], ti.f32)
        pos[q * nd + p] = ti.Vector([x * sc, h, -y * sc], ti.f32)
        col[q * nd + p] = cl


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
            tri: ti.i32, LP: ti.template(), img: ti.template()):
    """Top view in PHYSICAL coordinates: each pixel shows its nearest lattice cell (so the
    triangular lattice's rhombic array has the right shape)."""
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
                if seg:
                    out = ti.Vector([1.0, 1.0, 0.0], ti.f32)
            else:
                out = palette(ti.cast(U[s, i, j], ti.f32) * LP[3], LP)
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
        self.mark = ti.Vector.field(3, ti.f32, shape=16)
        self.markc = ti.Vector.field(3, ti.f32, shape=16)

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
        lp[20:23] = POS_END
        lp[23:26] = NEG_END
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
            self.col = ti.Vector.field(3, ti.f32, shape=nd * nd)
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

    def draw3d(self, m, look, scene, pose_eye, wire):
        s = (m.slot + 1) % 3
        k_verts(m.U, s, m.M, self.DOM, self.k, self.nd, 1 if m.tri() else 0, self.LP, self.pos, self.col)
        amb, pl = look.lights()
        if look.segment:
            amb, pl = 1.0, 0.0
        scene.ambient_light((amb, amb, amb))
        if pl > 0:
            k_normals(self.pos, self.nd, self.nrm)
            e = pose_eye
            scene.point_light(pos=(e[0], e[1] + 0.25, e[2]), color=(pl, pl, pl))
        cnt = self.n_idx if look.floor else self.n_dom
        if cnt:
            # pass only the drawn prefix (domain triangles come first): GGUI uploads the whole array
            scene.mesh(self.pos, indices=self.idx[:cnt], normals=self.nrm, per_vertex_color=self.col,
                       two_sided=True, show_wireframe=wire, index_count=cnt)

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
        k_image(m.U, s, m.M, self.DOM, 1 if m.tri() else 0, self.LP, self.img)
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

