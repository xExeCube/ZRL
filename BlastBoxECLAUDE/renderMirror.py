"""BlastBox - renderer (pygame-ce), Stage 1 field view and Stage 2 gas view.

Left panel carries the live ledger in the ZRL palette (Courier, green PASS, red
FAIL, neutral n/a); the field draws on the right. This module is imported only
when a window is actually opened, so the tests never touch pygame.

The window is RESIZABLE and starts sized to the screen. The panel wraps long
lines and SCROLLS (mouse wheel over the panel, PageUp/PageDown, Home), and its
text size is adjustable with - and =, so nothing is ever cut off.

Colour is looked up from a 256-entry table rather than computed per particle.
"""
from __future__ import annotations
import colorsys
from dataclasses import replace
import numpy as np
import pygame
import configMirror as C
import gaugesMirror as G
import dynamicsMirror as D
import unitsMirror as U
from ledgerMirror import run_gas_checks

# Wall pressure and drag are counting measurements; their noise falls as
# 1/sqrt(hits). A short window is dominated by counting noise, so the live
# readouts are held over windows this long (simulation time).
WALL_WINDOW = 2.0

PROBE_STEP = 0.05           # Up/Down arrow step for the probe speed (relative)
PROBE_STEP_BIG = 0.5        # with Shift held
FONT_MIN, FONT_MAX, FONT_DEFAULT = 10, 22, 13
CONE_LENGTH = 8.0           # world units drawn for the predicted Mach-cone lines
ZOOM_MIN, ZOOM_MAX = 0.2, 200.0   # zoom limits, as multiples of the fitted scale
PANEL_MIN_W = 160           # the panel never shrinks below this, nor grows past half the window


def _hsl(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))


# 256-entry colour ramp (blue -> green -> red), built once at import.
_LUT = np.array([_hsl(210.0 * (1.0 - i / 255.0), 0.62, 0.60) for i in range(256)],
                dtype=np.uint8)
_LUT_TUPLES = [tuple(int(v) for v in _LUT[i]) for i in range(256)]


def _bins(norm):
    """Map an array in [0,1] to LUT indices."""
    return np.clip(np.asarray(norm) * 255.0, 0, 255).astype(np.uint8)


def _mono(size, bold=False):
    """A monospace font, with a fallback that never shells out to fontconfig.

    pygame's SysFont enumerates system fonts via `fc-list` on Linux; under some
    WSL / sandbox setups that call is blocked. Try SysFont for real Courier first,
    then fall back to pygame's built-in default font so the window always opens.
    """
    try:
        f = pygame.font.SysFont("couriernew,consolas,dejavusansmono,monospace",
                                size, bold=bold)
        if f is not None:
            return f
    except Exception:
        pass
    return pygame.font.Font(None, size + 2)


def _wrap(text, font, max_w, indent=""):
    """Greedy word wrap so long ledger values are never clipped off the panel.

    The text's own leading spaces are peeled off first and kept on the first
    line; splitting them as empty "words" used to collapse them, so a wrapped
    ledger value lost its indent on exactly the line that needed it.
    """
    if max_w <= 0 or font.size(text)[0] <= max_w:
        return [text]
    body = text.lstrip(" ")
    prefix = text[: len(text) - len(body)]
    lines, words = [], []
    for word in body.split(" "):
        trial = prefix + " ".join(words + [word])
        if not words or font.size(trial)[0] <= max_w:
            words.append(word)
        else:
            lines.append(prefix + " ".join(words))
            prefix, words = indent, [word]
    if words:
        lines.append(prefix + " ".join(words))
    return lines


class Camera:
    """World<->screen transform. World +y is up; screen +y is down, so we flip."""

    def __init__(self, canvas, world_size):
        self.canvas = canvas
        self.world_size = world_size
        self.fit()

    def fit(self):
        w, h = self.world_size
        cw, ch = max(1, self.canvas.width), max(1, self.canvas.height)
        self.scale = min(cw / w, ch / h) * 0.92
        self.fit_scale = self.scale
        self.ox = self.canvas.x + (cw - w * self.scale) / 2.0
        self.oy = self.canvas.y + (ch - h * self.scale) / 2.0

    def to_screen(self, pts):
        x = self.ox + pts[:, 0] * self.scale
        y = self.oy + (self.world_size[1] - pts[:, 1]) * self.scale
        return np.column_stack([x, y])

    def point_to_screen(self, p):
        return (int(self.ox + p[0] * self.scale),
                int(self.oy + (self.world_size[1] - p[1]) * self.scale))

    def box_to_screen(self, box):
        p = self.to_screen(np.array([[box.x0, box.y0], [box.x1, box.y1]]))
        x0, y0 = p[0]
        x1, y1 = p[1]
        return pygame.Rect(int(min(x0, x1)), int(min(y0, y1)),
                           int(abs(x1 - x0)), int(abs(y1 - y0)))

    def zoom_at(self, sx, sy, factor):
        """Zoom about a screen point, clamped so the field can neither shrink
        to a dot nor grow without limit (pixel coordinates are ints)."""
        new = min(max(self.scale * factor, ZOOM_MIN * self.fit_scale),
                  ZOOM_MAX * self.fit_scale)
        if new == self.scale:
            return
        wx = (sx - self.ox) / self.scale
        wy = self.world_size[1] - (sy - self.oy) / self.scale
        self.scale = new
        self.ox = sx - wx * self.scale
        self.oy = sy - (self.world_size[1] - wy) * self.scale


class _Shell:
    """Window, fonts, layout, the scrolling panel, and the controls both views
    share: resize, pan, zoom, refit, panel scroll, and panel text size."""

    def __init__(self, caption, world_size):
        pygame.init()
        info = pygame.display.Info()
        dw = info.current_w if info.current_w and info.current_w > 0 else 1280
        dh = info.current_h if info.current_h and info.current_h > 0 else 800
        size = (max(900, min(1760, int(dw * 0.9))), max(600, min(1100, int(dh * 0.88))))
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption(caption)
        # Held arrows / PgUp / PgDn / - = repeat. Toggles must not, so keys
        # currently held are tracked and the views skip repeats for those.
        pygame.key.set_repeat(300, 40)
        self.held = set()
        self.clock = pygame.time.Clock()
        self.font_size = FONT_DEFAULT
        self._make_fonts()
        self.scroll = 0
        self.content_h = 0
        self.dragging = False
        self.last = (0, 0)
        self.cam = Camera(self.canvas_rect(), world_size)

    # --- layout ------------------------------------------------------------
    def _make_fonts(self):
        self.mono = _mono(self.font_size)
        self.mono_b = _mono(self.font_size + 1, bold=True)
        self.line_h = self.font_size + 5
        self._fit_panel()

    def _fit_panel(self):
        # 27 characters wide, but never more than half the window, so the
        # field stays visible however small the window is dragged.
        w = self.size()[0]
        self.panel_w = int(min(27 * self.font_size, max(PANEL_MIN_W, w // 2)))

    def size(self):
        return self.screen.get_size()

    def canvas_rect(self):
        w, h = self.size()
        return pygame.Rect(self.panel_w, 0, max(1, w - self.panel_w), h)

    def relayout(self):
        self._fit_panel()
        self.cam.canvas = self.canvas_rect()
        self.cam.fit()

    def note_key(self, e):
        """Track held keys. Returns True if this KEYDOWN is an auto-repeat."""
        if e.type == pygame.KEYDOWN:
            repeat = e.key in self.held
            self.held.add(e.key)
            return repeat
        if e.type == pygame.KEYUP:
            self.held.discard(e.key)
        elif e.type == getattr(pygame, "WINDOWFOCUSLOST", -1):
            self.held.clear()           # a key released elsewhere never sends KEYUP
        return False

    def set_font(self, size):
        size = max(FONT_MIN, min(FONT_MAX, size))
        if size != self.font_size:
            self.font_size = size
            self._make_fonts()
            self.relayout()

    def scroll_by(self, dy):
        max_scroll = max(0, self.content_h - self.size()[1] + 12)
        self.scroll = int(max(0, min(max_scroll, self.scroll + dy)))

    # --- events ------------------------------------------------------------
    def handle(self, e):
        """Shared controls. Returns True if the event was consumed."""
        if e.type == pygame.VIDEORESIZE:
            # pygame 2 has already resized the display surface by now; just
            # pick it up and lay out again.
            self.screen = pygame.display.get_surface()
            self.relayout()
            return True
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and e.pos[0] > self.panel_w:
            self.dragging, self.last = True, e.pos
            return True
        if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.dragging = False
            return True
        if e.type == pygame.MOUSEMOTION and self.dragging:
            self.cam.ox += e.pos[0] - self.last[0]
            self.cam.oy += e.pos[1] - self.last[1]
            self.last = e.pos
            return True
        if e.type == pygame.MOUSEWHEEL:
            # precise_y carries fractional trackpad scrolls; a purely sideways
            # scroll has dy == 0 and used to read as "zoom out".
            dy = float(getattr(e, "precise_y", e.y))
            if dy == 0:
                return True
            mx, my = pygame.mouse.get_pos()
            if mx > self.panel_w:
                self.cam.zoom_at(mx, my, 1.1 ** dy)
            else:
                self.scroll_by(int(round(-dy * 3 * self.line_h)))
            return True
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_f:
                self.relayout()
                return True
            if e.key == pygame.K_PAGEDOWN:
                self.scroll_by(int(self.size()[1] * 0.8))
                return True
            if e.key == pygame.K_PAGEUP:
                self.scroll_by(-int(self.size()[1] * 0.8))
                return True
            if e.key == pygame.K_HOME:
                self.scroll = 0
                return True
            if e.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                self.set_font(self.font_size + 1)
                return True
            if e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.set_font(self.font_size - 1)
                return True
        return False

    # --- drawing -----------------------------------------------------------
    def begin_field(self):
        self.screen.fill(C.BG)
        self.screen.set_clip(self.canvas_rect())

    def end_field(self):
        self.screen.set_clip(None)

    def draw_panel(self, title, lines, checks):
        w, h = self.size()
        panel = pygame.Rect(0, 0, self.panel_w, h)
        self.screen.set_clip(panel)
        pygame.draw.rect(self.screen, C.PANEL, panel)
        x = 12
        lh = self.line_h
        max_w = self.panel_w - x - 12
        y = [12 - self.scroll]

        def put(text, color=C.TEXT, font=None, indent=""):
            f = font or self.mono
            for seg in _wrap(text, f, max_w, indent):
                if -lh < y[0] < h:
                    self.screen.blit(f.render(seg, True, color), (x, y[0]))
                y[0] += lh

        put(title, C.ACCENT, self.mono_b)
        y[0] += 4
        for label, value in lines:
            if label == "" and not value:
                y[0] += 6                              # explicit spacer
            elif label == "":
                put(f"{'':11}{value}", indent=" " * 11)
            elif value is None:
                put(label, C.ACCENT, self.mono_b)      # section header
            else:
                put(f"{label:<11}{value}", indent=" " * 11)
        y[0] += 6
        put("IDENTITY CHECKS", C.ACCENT, self.mono_b)
        for c in checks:
            colour = C.ACCENT if c.na else (C.HI if c.passed else C.LINK)
            put(f"[{c.tag}] {c.name}", colour, indent="       ")
            put(f"       {c.value}", indent="       ")
        self.content_h = y[0] + self.scroll + 12

        if self.content_h > h:                         # scroll bar
            bar_h = max(24, int(h * h / self.content_h))
            span = max(1, self.content_h - h)
            top = int((h - bar_h) * min(1.0, self.scroll / span))
            pygame.draw.rect(self.screen, C.ACCENT,
                             pygame.Rect(self.panel_w - 6, top, 4, bar_h))
        pygame.draw.line(self.screen, C.BLINE, (self.panel_w - 1, 0),
                         (self.panel_w - 1, h), 1)
        self.screen.set_clip(None)
        self.scroll_by(0)                              # re-clamp if content shrank

    def flip(self):
        pygame.display.flip()
        self.clock.tick(60)


def _draw_world(screen, cam, world):
    pygame.draw.rect(screen, C.BSHADOW, cam.box_to_screen(world.domain), width=1)
    for obs in world.obstacles:
        r = cam.box_to_screen(obs)
        pygame.draw.rect(screen, C.WALL_FILL, r)
        pygame.draw.rect(screen, C.WALL_EDGE, r, width=1)


def _draw_particles(screen, cam, pos, colour_bins, radius, flat=None):
    pts = cam.to_screen(pos).astype(int)
    if flat is not None:
        for sx, sy in pts:
            pygame.draw.circle(screen, flat, (sx, sy), radius)
    else:
        for (sx, sy), k in zip(pts, colour_bins):
            pygame.draw.circle(screen, _LUT_TUPLES[k], (sx, sy), radius)


def _draw_density(screen, cam, pos, world, cell=0.25):
    """Coarse number-density field - the view that shows a Mach cone, because
    the cone is a compression and appears as a bright wedge."""
    d = world.domain
    nx = max(1, int((d.x1 - d.x0) / cell))
    ny = max(1, int((d.y1 - d.y0) / cell))
    hist, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=(nx, ny),
                                range=[[d.x0, d.x1], [d.y0, d.y1]])
    peak = hist.max()
    if peak <= 0:
        return
    norm = hist / peak
    w = (d.x1 - d.x0) / nx
    h = (d.y1 - d.y0) / ny
    bins = _bins(norm)
    for ix in range(nx):
        for iy in range(ny):
            if norm[ix, iy] <= 0.02:
                continue
            box = C.Box(d.x0 + ix * w, d.y0 + iy * h,
                        d.x0 + (ix + 1) * w, d.y0 + (iy + 1) * h)
            pygame.draw.rect(screen, _LUT_TUPLES[bins[ix, iy]], cam.box_to_screen(box))


def _draw_cone(screen, cam, disk, mach):
    """The PREDICTED Mach cone, sin(mu) = 1/M, trailing the probe. It is drawn
    over the density view so the prediction can be compared by eye with what
    the gas actually does - at this Knudsen number, expect the gas to disagree."""
    if not np.isfinite(mach) or mach <= 1.0:
        return
    v = disk.velocity
    speed = float(np.linalg.norm(v))
    if speed <= 0:
        return
    back = -v / speed
    mu = np.arcsin(1.0 / mach)
    for sign in (1.0, -1.0):
        c, s = np.cos(sign * mu), np.sin(sign * mu)
        direction = np.array([c * back[0] - s * back[1], s * back[0] + c * back[1]])
        tip = disk.centre + direction * CONE_LENGTH
        pygame.draw.line(screen, C.ACCENT, cam.point_to_screen(disk.centre),
                         cam.point_to_screen(tip), 1)


# --- Stage 1: the static field ----------------------------------------------

def run_view(scene, world, field, neighbors, checks):
    shell = _Shell(f"BlastBox - Stage 1 - {scene.name}", scene.world_size)
    counts = neighbors.counts(scene.smoothing_length).astype(float)
    dens = counts / (counts.max() if counts.max() > 0 else 1.0)
    temp = field.temperature.astype(float)
    lo, hi = float(temp.min()), float(temp.max())
    tnorm = np.zeros_like(temp) if hi <= lo else (temp - lo) / (hi - lo)
    modes = [("density", _bins(dens)), ("temperature", _bins(tnorm)), ("flat", None)]
    mode_i = 0
    running = True

    while running:
        for e in pygame.event.get():
            repeat = shell.note_key(e)
            if e.type == pygame.QUIT:
                running = False
            elif shell.handle(e):
                continue
            elif e.type == pygame.KEYDOWN and not repeat:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    mode_i = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2}[e.key]

        shell.begin_field()
        _draw_world(shell.screen, shell.cam, world)
        name, bins = modes[mode_i]
        radius = max(1, int(0.45 * scene.spacing * shell.cam.scale))
        _draw_particles(shell.screen, shell.cam, field.pos, bins, radius,
                        flat=C.ACCENT if bins is None else None)
        shell.end_field()

        lines = [("scene", scene.name), ("particles", f"{field.n}"),
                 ("spacing", f"a = {scene.spacing:.3f}"),
                 ("h", f"{scene.smoothing_length:.3f}"),
                 ("", ""), ("COLOUR", None), ("mode", name),
                 ("", ""), ("KEYS", None),
                 ("", "1 density  2 temp  3 flat"),
                 ("", "drag pan  wheel zoom  F refit"),
                 ("", "wheel over panel / PgUp PgDn: scroll"),
                 ("", "- and = : text size   esc: quit")]
        shell.draw_panel("BLASTBOX - STAGE 1", lines, checks)
        shell.flip()
    pygame.quit()


# --- Stage 2: the molecular gas ---------------------------------------------

def run_gas_view(scene, world, state, build_state):
    """Interactive gas. `build_state` re-creates a fresh state for the reset key."""
    shell = _Shell(f"BlastBox - Stage 2 gas - {scene.name}", scene.world_size)
    gas = scene.gas
    running, paused = True, False
    modes = ["speed", "density", "flat"]
    mode_i = 0
    held = {"p_wall": float("nan"), "drag": float("nan"), "cd": float("nan")}
    typing = None            # None, or the probe-speed text being typed
    message = ""

    def current_c(st):
        return G.sound_speed(G.temperature(st.vel, st.mass),
                             float(st.mass.mean()) if st.n else 1.0)

    def references(st):
        """Energy, work and the SI calibration are all pinned to the state as
        built. The calibration declares THAT state to be air at 20 C, 1 atm;
        everything else is scaled from it by ratio, so a gas the probe has heated
        reads hotter than 293 K, as it should."""
        r0 = G.read(st, scene, world)
        cal = U.Calibration(c_ref=r0.sound_speed, t_ref=r0.temperature, p_ref=r0.p_virial)
        return st.total_energy(), st.external_work, cal

    energy_ref, work_ref, calibration = references(state)
    probe_speed = D.clamp_speed(scene, gas.disk_mach * current_c(state), state)

    def clear_held():
        for k in held:
            held[k] = float("nan")

    def apply_speed(value):
        """Set the probe speed (clamped to the contact-law cap). Returns False
        if nothing changed, so holding Up at the cap does not keep wiping the
        held drag and C_d readouts."""
        nonlocal probe_speed
        new = D.clamp_speed(scene, value, state)
        if new == probe_speed:
            return False
        probe_speed = new
        if state.disk is not None:
            D.set_disk_speed(state, scene, probe_speed)
            state.reset_window()
            clear_held()
        return True

    while running:
        for e in pygame.event.get():
            repeat = shell.note_key(e)
            if e.type == pygame.QUIT:
                running = False
                continue

            # typed speed entry captures every key until Enter or Esc
            if typing is not None and e.type == pygame.KEYDOWN:
                if repeat and e.key not in (pygame.K_BACKSPACE,):
                    continue                    # only Backspace may auto-repeat here
                if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    try:
                        apply_speed(U.parse_speed(typing, calibration, current_c(state)))
                        message = f"probe speed set: {probe_speed:+.3f} rel"
                    except ValueError as exc:
                        message = f"not set - {exc}"
                    typing = None
                elif e.key == pygame.K_ESCAPE:
                    typing, message = None, "speed entry cancelled"
                elif e.key == pygame.K_BACKSPACE:
                    typing = typing[:-1]
                elif e.unicode and e.unicode.isprintable() and len(typing) < 20:
                    typing += e.unicode
                continue

            if shell.handle(e):
                continue

            if e.type == pygame.KEYDOWN:
                shift = bool(e.mod & pygame.KMOD_SHIFT)
                if repeat and e.key not in (pygame.K_UP, pygame.K_DOWN):
                    continue                    # toggles fire once per press
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_r:
                    state = build_state()
                    if state.disk is not None:
                        # the config attached the probe at its default speed;
                        # keep the one the user chose
                        probe_speed = D.set_disk_speed(state, scene, probe_speed)
                    energy_ref, work_ref, calibration = references(state)
                    clear_held()
                    message = "reset"
                elif e.key == pygame.K_d:
                    # No energy reset needed: the ledger books the probe's work
                    # (and its insertion cost), so toggling it keeps the check live.
                    if state.disk is None:
                        D.attach_disk(state, scene, world, speed=probe_speed)
                        # the cap may have fallen since the speed was chosen
                        probe_speed = float(state.disk.velocity[0])
                    else:
                        D.detach_disk(state, scene)
                    state.reset_window()
                    clear_held()
                elif e.key in (pygame.K_UP, pygame.K_DOWN):
                    stepv = PROBE_STEP_BIG if shift else PROBE_STEP
                    changed = apply_speed(probe_speed + (stepv if e.key == pygame.K_UP else -stepv))
                    cap = D.speed_cap(scene, state)
                    message = "" if changed else f"at the contact-law cap +/-{cap:.2f} rel"
                elif e.key == pygame.K_s:
                    typing, message = "", ""
                    shell.scroll = 0            # the prompt is near the top
                elif e.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    mode_i = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2}[e.key]

        if not paused:
            D.advance(state, scene, world, gas.substeps)
        limits = D.dt_limits(scene, state)

        readout = G.read(state, scene, world)
        if state.window_time >= WALL_WINDOW:
            held["p_wall"] = readout.p_wall
            held["drag"] = readout.drag
            held["cd"] = readout.drag_coefficient
            state.reset_window()
        readout = replace(readout, p_wall=held["p_wall"], drag=held["drag"],
                          drag_coefficient=held["cd"])
        checks = run_gas_checks(scene, world, state, readout, energy_ref, limits, work_ref)

        # --- field ----------------------------------------------------------
        shell.begin_field()
        screen, cam = shell.screen, shell.cam
        mode = modes[mode_i]
        if mode == "density":
            _draw_density(screen, cam, state.pos, world)
            _draw_world(screen, cam, world)
        else:
            _draw_world(screen, cam, world)
            # True to scale the particles are sub-pixel at the default fit.
            radius = max(2, int(0.5 * gas.diameter * cam.scale))
            if mode == "flat":
                _draw_particles(screen, cam, state.pos, None, radius, flat=C.ACCENT)
            else:
                speeds = np.linalg.norm(state.vel, axis=1)
                ceiling = max(readout.rms_speed * 2.2, 1e-9)
                _draw_particles(screen, cam, state.pos, _bins(speeds / ceiling), radius)
        if state.disk is not None:
            if readout.probe_resolved:
                _draw_cone(screen, cam, state.disk, readout.mach)
            pygame.draw.circle(screen, C.LINK, cam.point_to_screen(state.disk.centre),
                               max(2, int(state.disk.radius * cam.scale)), width=2)
        shell.end_field()

        # --- panel ----------------------------------------------------------
        def si_speed(v):
            return f"  (~{calibration.speed(v):.0f} m/s)" if np.isfinite(v) else ""

        def fmt(v, spec=".4f"):
            return format(v, spec) if np.isfinite(v) else "--"

        cap = D.speed_cap(scene, state)
        lines = [
            ("scene", f"{scene.name}  N={state.n}"),
            ("state", f"{'PAUSED' if paused else 'running'}  t={state.time:.2f} rel"),
            ("", ""),
            ("PROBE", None),
            ("speed", f"{probe_speed:+.3f} rel{si_speed(abs(probe_speed))}"
                      + ("" if state.disk is not None else "  [off - D]")),
            ("cap", f"+/-{cap:.2f} rel{si_speed(cap)}  (contact law)"),
        ]
        if state.disk is not None:
            m = readout.mach
            ok = readout.probe_resolved
            ang = G.mach_angle(m) if np.isfinite(m) else float("nan")
            if not ok:
                cone = "  cone n/a"
            elif np.isfinite(ang):
                cone = f"  cone {ang:.1f} deg*"
            else:
                cone = "  subsonic: no cone"
            kn = readout.knudsen
            regime = "continuum" if kn < 0.01 else "slip" if kn < 0.1 else "rarefied"
            cd = fmt(readout.drag_coefficient, '.2f') if ok else "n/a"
            lines += [
                ("Mach", f"{fmt(m, '.2f')}{cone}"),
                ("Kn", f"{kn:.3f} ({regime})"),
                ("drag", f"{fmt(readout.drag, '.3f')}   C_d {cd}"),
                ("", f"C_d ref {G.CD_FREE_MOLECULAR_CYLINDER:.2f} = free-molecular, "
                     "very fast limit"),
                ("", "* cone angle and lines are the continuum "
                     "prediction, not a measurement"),
            ]
            if not ok:
                lines.append(("", f"! at this temperature, above {readout.probe_limit:.2f} "
                                  "rel bounced particles tunnel through each other, so "
                                  "cone and C_d are not trustworthy. Slow the probe."))
        if typing is not None:
            lines.append(("TYPE", f"{typing}_"))
            lines.append(("", "e.g. 2.5   500ms   1.6mach   (Enter / Esc)"))
        elif message:
            lines.append(("", message))

        lines += [
            ("", ""),
            ("MEASURED (relative units)", None),
            ("T", f"{readout.temperature:.4f}  (~{calibration.temperature(readout.temperature):.0f} K)"),
            ("P virial", f"{readout.p_virial:.3f}  (~{calibration.pressure(readout.p_virial) / 1000:.0f} kPa-equiv)"),
            ("P wall", f"{fmt(readout.p_wall, '.3f')}  (last {WALL_WINDOW:g}-unit window)"),
            ("v_rms = c", f"{readout.sound_speed:.4f}{si_speed(readout.sound_speed)}"),
            ("Z", f"{readout.z_measured:.4f}  (expected {readout.z_expected:.4f})"),
            ("phi_eff", f"{readout.phi_eff:.4f}  d_eff/d = {readout.d_eff_ratio:.3f}"),
            ("contacts", f"{readout.contacts}   deepest {100 * readout.max_overlap / gas.diameter:.0f}% of d"),
            ("dt", f"{state.dt:.2e} rel"
                   + (f"  (re-derived {state.dt_rederived}x as the probe heated the gas)"
                      if state.dt_rederived else "")),
            ("averaged", f"{readout.avg_time:.1f} rel time, {readout.avg_hits} wall hits"),
        ]
        if np.isfinite(readout.p_window):
            lines.append(("", f"window term {readout.p_window:+.3f} (in the virial route)"))
        lines += [
            ("", "SI figures are a calibration, not a derivation"),
            ("", ""),
            ("KEYS", None),
            ("", "space run/pause   R reset   esc quit"),
            ("", "D probe on/off   Up/Down speed   Shift x10"),
            ("", "S type an exact speed"),
            ("", "1 speed  2 density  3 flat colour"),
            ("", "drag pan   wheel zoom   F refit"),
            ("", "wheel over panel / PgUp PgDn / Home: scroll"),
            ("", "- and = : panel text size"),
        ]
        shell.draw_panel("BLASTBOX - STAGE 2", lines, checks)
        shell.flip()
    pygame.quit()
