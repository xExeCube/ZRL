"""ZRL 2D wave membrane -- Taichi port, VIEWER: one frame of user input.

The viewer never talks to the GGUI window directly for input: each frame becomes a Frame
(cursor, buttons/modifiers held, keys pressed), so the same handlers run from the live window
(poll_window) and from tests and --shot scripts (Frame(...) built by hand). Measured 30/09/2026:
a GGUI window made with show_window=False has NO OS window behind it (none for our process), so
real key and mouse events cannot be injected headless -- the fake frames are the test path.

Key names are GGUI's: letters arrive lower-case ('a'..'z'); named keys are ' ', 'Return',
'BackSpace', 'Escape', 'Tab', 'Shift', 'Control', 'Alt', 'Up', 'Down', 'Left', 'Right', 'CapsLock',
and the mouse buttons 'LMB', 'MMB', 'RMB'. GGUI 1.7.4's name table has nothing else (the binary's
string table, 30/09/2026): digits and punctuation have no name there and its key callback carries
an 'unrecognized id' error, so poll_window() swallows that error, and digits for the text entry
come from DigitPoller (Win32 GetAsyncKeyState, only while the app's own window is in front).
*Generated from scratch -- Claude Opus 5.5 -- 30/09/2026*
"""
import os
import sys

BUTTONS = ('LMB', 'MMB', 'RMB')
MODS = ('Shift', 'Control', 'Alt')


class Frame:
    """cursor = (sx, sy) in [0,1]^2, y up (GGUI's get_cursor_pos); down = held buttons and
    modifiers; presses = keys/buttons pressed since the last frame, in order; chars = extra
    characters for a text entry (digits from DigitPoller)."""

    def __init__(self, cursor=(0.5, 0.5), down=(), presses=(), chars=()):
        self.cursor = (float(cursor[0]), float(cursor[1]))
        self.down = set(down)
        self.presses = list(presses)
        self.chars = list(chars)

    def held(self, name):
        return name in self.down

    def __repr__(self):
        return f'Frame(cursor={self.cursor}, down={sorted(self.down)}, presses={self.presses})'


def poll_window(window, ti_ui, digits=None):
    """The live window's input as a Frame. Errors from a key GGUI cannot name are dropped
    (logged once per kind) instead of ending the app."""
    presses = []
    try:
        presses = [e.key for e in window.get_events(ti_ui.PRESS)]
        window.get_events(ti_ui.RELEASE)          # drain: GGUI keeps unread releases forever
    except RuntimeError as ex:
        _warn_once(str(ex))
    down = set()
    for k in BUTTONS + MODS:
        try:
            if window.is_pressed(k):
                down.add(k)
        except RuntimeError as ex:
            _warn_once(str(ex))
    cur = window.get_cursor_pos()
    chars = digits.poll() if digits is not None else []
    return Frame(cur, down, presses, chars)


_warned = set()


def _warn_once(msg):
    if msg not in _warned:
        _warned.add(msg)
        print(f'[app] input: ignored a key GGUI cannot name ({msg})', file=sys.stderr)


class DigitPoller:
    """Digits and a few symbols for the text entry, read with Win32 GetAsyncKeyState while the
    app's own window has the focus (GGUI 1.7.4 gives these keys no name). Edge-triggered: one
    character per key press. Disabled (always empty) off Windows or when the window is not found."""
    KEYS = [(0x30 + d, str(d)) for d in range(10)] + [(0x60 + d, str(d)) for d in range(10)] + \
           [(0xBD, '-'), (0x6D, '-'), (0xBE, '.'), (0x6E, '.'), (0xBC, ','), (0xBF, '/'), (0xDE, "'"),
            (0xBB, '+'), (0x6B, '+')]

    def __init__(self, title):
        self.enabled = False
        self.hwnd = None
        self.title = title
        self.prev = {}
        self.active = False           # only while a text entry is open
        if os.name != 'nt':
            return
        try:
            import ctypes
            import ctypes.wintypes as wt
            self._u32 = ctypes.windll.user32
            self._ct, self._wt = ctypes, wt
            self.enabled = True
        except Exception:
            self.enabled = False

    def _find(self):
        ct, wt, u32 = self._ct, self._wt, self._u32
        pid, found = os.getpid(), []
        proc = ct.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

        def cb(h, lp):
            p = wt.DWORD()
            u32.GetWindowThreadProcessId(h, ct.byref(p))
            if p.value == pid:
                buf = ct.create_unicode_buffer(256)
                u32.GetWindowTextW(h, buf, 256)
                if buf.value == self.title:
                    found.append(h)
            return True
        u32.EnumWindows(proc(cb), 0)
        self.hwnd = found[0] if found else None

    def poll(self):
        if not (self.enabled and self.active):
            self.prev = {}
            return []
        u32 = self._u32
        if self.hwnd is None:
            self._find()
            if self.hwnd is None:
                return []
        if u32.GetForegroundWindow() != self.hwnd:
            self.prev = {}
            return []
        out = []
        for vk, ch in self.KEYS:
            now = bool(u32.GetAsyncKeyState(vk) & 0x8000)
            if now and not self.prev.get(vk, False):
                out.append(ch)
            self.prev[vk] = now
        return out


# ---- typed slider values (user request 01/10/2026) --------------------------------------------
# GGUI 1.7.4 has no text-input widget and no same_line(), so a button per slider costs a whole
# panel row per slider (30 sliders: the panels grew by ~30 rows). The default is therefore a PICK
# mode: ONE "type a value" button at the top of each panel arms it, and the next slider you click
# (its value change is swallowed) opens the entry under that slider. The old per-slider buttons
# stay available behind a main-panel checkbox. The entry is a key-capture mode
# like the text objects' one: digits / '.' / '-' come from DigitPoller (GGUI names no digit keys),
# 'e' from GGUI (1e-4), BackSpace edits, Return applies, Escape cancels, and every key is
# consumed while it is open (no hotkey fires). The value is handed back to the panel code as the
# slider's own return value on the next panel pass, so it goes through exactly the slider's code
# path (moved() check, reset(), the N slider's settle wait, ...).
# *Generated from scratch -- Claude Opus 5.5 -- 01/10/2026*
NUM_CHARS = '0123456789.-+e'
ENTRY_COL = (0.4, 0.9, 0.5)
ENTRY_DIM = (0.55, 0.55, 0.60)


def fmt_num(x, is_int=False):
    return str(int(x)) if is_int else f'{x:.6g}'


class NumEntry:
    """The typed-value entry: at most one open at a time, keyed (panel, slider label)."""

    def __init__(self):
        self.target = None            # (panel, label) being typed; None: closed
        self.text = ''
        self.spec = None              # (lo, hi, is_int, value when opened)
        self.pending = None           # ((panel, label), value, frame) waiting for its slider
        self.frame = 0                # handle() frames (tick)
        self.msg = ''                 # last error, shown under the entry
        self.armed = False            # pick mode: the next slider clicked opens the entry

    @property
    def open(self):
        return self.target is not None

    def arm(self, on=True):
        """Pick mode on/off (closes an open entry: one entry at a time)."""
        self.cancel()
        self.armed = bool(on)

    def start(self, target, lo, hi, is_int, cur):
        self.armed = False
        self.target, self.text, self.msg = target, '', ''
        self.spec = (lo, hi, bool(is_int), cur)

    def cancel(self):
        self.target, self.text, self.spec, self.msg = None, '', None, ''

    def tick(self):
        """Once per input frame: a value whose slider was not drawn for 3 frames is dropped (its
        panel was hidden or the slider went away), so it can never apply late."""
        self.frame += 1
        if self.pending is not None and self.frame - self.pending[2] > 3:
            self.pending = None

    def chars(self, chars):
        if self.open:
            for ch in chars:
                ch = '.' if ch == ',' else ch
                if ch in NUM_CHARS and len(self.text) < 24:
                    self.text += ch

    def key(self, k):
        """A key while open (or armed): returns (consumed, message). Always consumed while open;
        while only armed, Escape disarms and other keys stay hotkeys."""
        if not self.open:
            if self.armed and k == 'Escape':
                self.armed = False
                return True, 'type a value: stopped'
            return False, None
        if k == 'Return':
            return True, self.commit()
        if k == 'Escape':
            self.cancel()
            return True, 'typed value cancelled'
        if k == 'BackSpace':
            self.text = self.text[:-1]
        elif k == 'e':
            self.chars('e')
        return True, None

    def commit(self):
        """Parse, clamp to the slider's range, queue for the slider. Returns the message to show
        (None when the value applied as typed). A bad number keeps the entry open."""
        if not self.open:
            return None
        (panel, label), (lo, hi, is_int, cur) = self.target, self.spec
        t = self.text.strip()
        if not t:
            self.cancel()
            return f'{label}: nothing typed, kept {fmt_num(cur, is_int)}'
        try:
            x = float(t)
            if x != x or x in (float('inf'), float('-inf')):
                raise ValueError
        except ValueError:
            self.msg = f'"{t}" is not a number'
            return f'{label}: {self.msg}'
        msg = None
        v = min(max(x, lo), hi)
        if is_int:
            v = int(round(v))
            v = min(max(v, int(lo)), int(hi))
        if v != x:
            if x < lo or x > hi:
                msg = (f'{label}: {t} is outside {fmt_num(lo, is_int)} .. {fmt_num(hi, is_int)}, '
                       f'clamped to {fmt_num(v, is_int)}')
            else:
                msg = f'{label}: {t} rounded to {v}'
        self.pending = ((panel, label), v, self.frame)
        self.cancel()
        return msg

    def take(self, target):
        if self.pending is not None and self.pending[0] == target:
            v = self.pending[1]
            self.pending = None
            return v
        return None


class TypedGui:
    """Wraps a GGUI sub-window: every slider_float / slider_int also gets the typed-value button
    (or, while its entry is open, the entry lines); anything else goes straight to the window."""

    def __init__(self, w, entry, panel, buttons=True, flash=None, on_start=None):
        self._w, self._e, self._panel, self._buttons, self._flash = w, entry, panel, buttons, flash
        self._on_start = on_start          # e.g. close the text objects' entry

    def __getattr__(self, name):
        return getattr(self._w, name)

    def slider_float(self, text, old_value, minimum, maximum):
        return self._slider(self._w.slider_float, text, old_value, minimum, maximum, False)

    def slider_int(self, text, old_value, minimum, maximum):
        return self._slider(self._w.slider_int, text, old_value, minimum, maximum, True)

    def header(self):
        """The panel's pick-mode button (one row per panel)."""
        e = self._e
        lab = ('>> click a slider to type its value  (here or Escape: stop)' if e.armed else
               'type a value: click here, then a slider')
        if self._w.button(f'{lab}##ne_arm/{self._panel}'):
            e.arm(not e.armed)
            if e.armed:
                if self._on_start:
                    self._on_start()
                if self._flash:
                    self._flash('type a value: click the slider you want to type (its value is kept)')

    @staticmethod
    def _changed(r, value, lo, hi, is_int):
        if is_int:
            return int(r) != int(value)
        # the slider hands back an f32 copy: ignore that rounding
        return abs(r - value) > 1e-6 * max(abs(hi - lo), abs(value), 1e-30)

    def _slider(self, fn, label, value, lo, hi, is_int):
        target = (self._panel, label)
        e = self._e
        typed = e.take(target)
        r = fn(label, value if typed is None else typed, lo, hi)
        if typed is not None:
            r = typed                      # exact (the slider would hand back an f32 copy)
        elif e.target == target:
            r = value                      # frozen while its entry is open (a drag still held)
        elif e.armed and self._changed(r, value, lo, hi, is_int):
            r = value                      # pick mode: the click picks the slider, the value stays
            e.start(target, lo, hi, is_int, value)
            if self._on_start:
                self._on_start()
            if self._flash:
                self._flash(f'typing {label}: digits . - e, Return applies, Escape cancels')
        w = self._w
        if e.target == target:
            w.text(f'   type {label}: {e.text}_', color=ENTRY_COL)
            w.text(f'   range {fmt_num(lo, is_int)} .. {fmt_num(hi, is_int)}, now {fmt_num(value, is_int)}'
                   f'{"  (whole number)" if is_int else ""}', color=ENTRY_DIM)
            if e.msg:
                w.text(f'   {e.msg}', color=(0.95, 0.45, 0.4))
            w.text('   digits . - e  BackSpace  Return  Escape', color=ENTRY_DIM)
            if w.button(f'apply  [Return]##ne_apply/{self._panel}/{label}'):
                msg = e.commit()
                if msg and self._flash:
                    self._flash(msg)
            if w.button(f'cancel  [Escape]##ne_cancel/{self._panel}/{label}'):
                e.cancel()
        elif self._buttons and w.button(f'   ^ type a value##ne/{self._panel}/{label}'):
            e.start(target, lo, hi, is_int, value)
            if self._on_start:
                self._on_start()
            if self._flash:
                self._flash(f'typing {label}: digits . - e, Return applies, Escape cancels')
        return r
