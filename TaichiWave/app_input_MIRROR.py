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
