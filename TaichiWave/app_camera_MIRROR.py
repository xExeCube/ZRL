"""ZRL 2D wave membrane -- Taichi port, VIEWER: camera and view mapping (pure math, no Taichi).

World frame of the 3D view (membrane_app_MIRROR.k_verts): the lattice cell (i, j) with physical
offset (x, y) from the centre (core: Membrane.cell_xy) sits at world (x/c, height, -y/c), c = (N-1)/2.
So the square array spans [-1, 1] in X and Z, the membrane rests in the plane Y = 0, and the domain
centre is the world origin.

OrbitCam: the camera looks at a FIXED target (the domain centre, the origin) from azimuth az,
elevation el and distance dist:  eye = target + dist (cos el sin az, sin el, cos el cos az), up = +Y.
Elevation is clamped to [-EL_MAX, EL_MAX] (at +-90 the view direction is parallel to up and the
lookAt frame is undefined: that is the "flip"); dist to [DIST_MIN, DIST_MAX].

Projection = GGUI's, measured 30/09/2026 (ScratchCLAUDE/AppCLAUDE/colprobe_MIRROR.py): glm lookAt +
perspective with a VERTICAL fov (default 45 deg: proj[1][1] = 2.41421 = 1/tan 22.5 deg) and
aspect = window width / height. Screen coordinates here are GGUI's cursor coordinates: (sx, sy) in
[0, 1], x to the right, y UP (window.get_cursor_pos()); a saved PNG's row is (1 - sy) H.
world_to_screen / screen_to_ray are exact inverses; app_tests_MIRROR.py checks them against a marker
rendered by GGUI itself.

View2D: the top view's pixel <-> physical mapping (membrane_app_MIRROR.k_image uses the same
numbers): pixel (px, py) (py up) samples the physical point
    x = cx + (px + 0.5 - W/2) s,   y = cy + (py + 0.5 - H/2) s,   s = 2 (1.02 c_half) / (H zoom),
where c_half is the array's half-height in physical cells, (cx, cy) the pan (physical cells).
*Generated from scratch -- Claude Opus 5.5 -- 30/09/2026*
"""
import math

D = math.pi / 180
EL_MAX = 89.0            # degrees; |el| < 90 keeps the lookAt frame defined (no flip)
DIST_MIN, DIST_MAX = 0.15, 25.0
ZOOM2D_MIN, ZOOM2D_MAX = 0.25, 64.0
FOV = 45.0               # GGUI's default; set explicitly on the ti.ui.Camera so the math agrees
HALF_FILL = 1.02         # the 2D view: the array's half-height fills the window height (x 1.02)


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    L = math.sqrt(_dot(a, a))
    return (a[0] / L, a[1] / L, a[2] / L) if L > 0 else (0.0, 0.0, 0.0)


class CamPose:
    """A camera as GGUI sees it: eye, target, up, vertical fov (degrees)."""

    def __init__(self, eye, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0), fov=FOV):
        self.eye, self.target, self.up, self.fov = tuple(eye), tuple(target), tuple(up), fov

    def frame(self):
        """glm::lookAt's orthonormal frame: s (right), u (up), f (forward)."""
        f = _norm(_sub(self.target, self.eye))
        s = _norm(_cross(f, self.up))
        u = _cross(s, f)
        return s, u, f

    def world_to_screen(self, P, aspect):
        """World point -> (sx, sy) in GGUI cursor coordinates (y up), or None behind the eye."""
        s, u, f = self.frame()
        d = _sub(P, self.eye)
        xv, yv, zf = _dot(d, s), _dot(d, u), _dot(d, f)
        if zf <= 1e-12:
            return None
        t = math.tan(self.fov * D / 2)
        nx, ny = xv / (zf * t * aspect), yv / (zf * t)
        return (nx + 1) / 2, (ny + 1) / 2

    def screen_to_ray(self, sx, sy, aspect):
        """(sx, sy) in cursor coordinates -> (origin, unit direction) of the view ray."""
        s, u, f = self.frame()
        t = math.tan(self.fov * D / 2)
        nx, ny = 2 * sx - 1, 2 * sy - 1
        d = _add(_add(_mul(s, nx * t * aspect), _mul(u, ny * t)), f)
        return self.eye, _norm(d)

    def screen_to_plane(self, sx, sy, aspect, plane_y=0.0):
        """Intersection of the view ray with the horizontal plane Y = plane_y: (X, Z) or None
        (parallel ray, or the plane behind the camera)."""
        o, d = self.screen_to_ray(sx, sy, aspect)
        if abs(d[1]) < 1e-12:
            return None
        t = (plane_y - o[1]) / d[1]
        if t <= 0:
            return None
        return o[0] + t * d[0], o[2] + t * d[2]

    def apply(self, camera):
        """Write the pose into a ti.ui.Camera."""
        camera.position(*self.eye)
        camera.lookat(*self.target)
        camera.up(*self.up)
        camera.fov(self.fov)


class OrbitCam:
    """Orbit about a fixed target (the domain centre). Angles in degrees."""
    DEFAULT = dict(az=0.0, el=38.0, dist=2.9)

    def __init__(self, az=None, el=None, dist=None, fov=FOV):
        d = self.DEFAULT
        self.target = (0.0, 0.0, 0.0)
        self.fov = fov
        self.az = d['az'] if az is None else az
        self.el = d['el'] if el is None else el
        self.dist = d['dist'] if dist is None else dist
        self.clamp()

    def clamp(self):
        self.az = math.fmod(math.fmod(self.az, 360.0) + 540.0, 360.0) - 180.0    # (-180, 180]
        self.el = max(-EL_MAX, min(EL_MAX, self.el))
        self.dist = max(DIST_MIN, min(DIST_MAX, self.dist))

    def reset(self):
        self.az, self.el, self.dist = self.DEFAULT['az'], self.DEFAULT['el'], self.DEFAULT['dist']

    def orbit(self, daz, delv):
        self.az += daz
        self.el += delv
        self.clamp()

    def zoom(self, factor):
        """factor > 1 moves closer (dist / factor)."""
        self.dist /= factor
        self.clamp()

    @property
    def zoom_level(self):
        return self.DEFAULT['dist'] / self.dist

    @zoom_level.setter
    def zoom_level(self, z):
        self.dist = self.DEFAULT['dist'] / max(1e-6, z)
        self.clamp()

    def eye(self):
        a, e = self.az * D, self.el * D
        t = self.target
        return (t[0] + self.dist * math.cos(e) * math.sin(a), t[1] + self.dist * math.sin(e),
                t[2] + self.dist * math.cos(e) * math.cos(a))

    def pose(self):
        return CamPose(self.eye(), self.target, (0.0, 1.0, 0.0), self.fov)


class View2D:
    """The top view's zoom and pan. (cx, cy): the physical point (cells from the array centre)
    at the window centre; zoom 1 = the array's half-height fills the window height (x 1.02)."""

    def __init__(self):
        self.cx = self.cy = 0.0
        self.zoom = 1.0
        # (cx, cy) sits at window x = W (0.5 + ox): the viewer centres the picture in the part of the
        # window the side panels leave free (review v3 01/10/2026: at the default layout the main
        # panel hid the left ~90 px of the 2D circle)
        self.ox = 0.0

    def kernel_cx(self, m, W, H):
        """The cx the image kernel needs (it centres on W / 2)."""
        return self.cx - self.ox * W * self.px_size(m, H)

    def reset(self):
        self.cx = self.cy = 0.0
        self.zoom = 1.0

    def px_size(self, m, H):
        """Physical cells per pixel: c = (N-1)/2 (x 1.02) fills the half-height on both lattices.
        (The triangular array is c sqrt(3)/2 high but 3c wide: fitting its height would push its
        width past a 1600 x 1000 window.)"""
        return 2 * (m.c() * HALF_FILL) / (H * self.zoom)

    def pixel_to_phys(self, m, px, py, W, H):
        """Continuous pixel coordinates (0..W, 0..H, y up; a pixel's centre is +0.5) -> physical."""
        s = self.px_size(m, H)
        return self.cx + (px - W * (0.5 + self.ox)) * s, self.cy + (py - H * 0.5) * s

    def screen_to_phys(self, m, sx, sy, W, H):
        return self.pixel_to_phys(m, sx * W, sy * H, W, H)

    def phys_to_screen(self, m, x, y, W, H):
        s = self.px_size(m, H)
        return ((x - self.cx) / s + W * (0.5 + self.ox)) / W, ((y - self.cy) / s + H * 0.5) / H

    def zoom_at(self, m, factor, sx, sy, W, H):
        """Zoom by factor keeping the physical point under the cursor (sx, sy) fixed."""
        x, y = self.screen_to_phys(m, sx, sy, W, H)
        self.zoom = max(ZOOM2D_MIN, min(ZOOM2D_MAX, self.zoom * factor))
        s = self.px_size(m, H)
        self.cx = x - (sx * W - W * (0.5 + self.ox)) * s
        self.cy = y - (sy * H - H * 0.5) * s

    def pan_screen(self, m, dsx, dsy, W, H):
        """Drag by (dsx, dsy) screen units: the content follows the cursor."""
        s = self.px_size(m, H)
        self.cx -= dsx * W * s
        self.cy -= dsy * H * s


# ---- world <-> physical <-> grid units (the 3D view's frame) --------------------------------
def world_to_phys(m, X, Z):
    c = m.c()
    return X * c, -Z * c


def phys_to_world(m, x, y):
    c = m.c()
    return x / c, -y / c


def phys_to_grid(m, x, y):
    h = m.base_half()
    return x / h, y / h


def grid_to_phys(m, gx, gy):
    h = m.base_half()
    return gx * h, gy * h
