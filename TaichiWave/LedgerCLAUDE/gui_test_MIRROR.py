"""Headless GGUI test of ledger_MIRROR.gui_section: a hidden window, the panel drawn for many frames,
auto-refresh cadence and per-frame cost measured. No window is shown (show_window=False)."""
import os, sys, time
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, HERE)
import taichi as ti
sys.stdout.reconfigure(encoding="utf-8")
import membrane_core_MIRROR as mc
import ledger_MIRROR as lg
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1025
mc.init_taichi('cuda', quiet=True)


class V:
    def __init__(self):
        self.m = mc.Membrane(mc.Params(N=N, src='pulse'))

    def reset(self):
        self.m.reseed()


v = V()
builds = [0]
_orig = lg.LedgerPanel.refresh


def counted(self, m):
    builds[0] += 1
    return _orig(self, m)


lg.LedgerPanel.refresh = counted
win = ti.ui.Window('ledger test', (1280, 800), show_window=False)
g = win.get_gui()
canvas = win.get_canvas()


class Chk:
    """Wraps the real sub_window: asserts every text line is Latin-1 (ImGui's default font)."""
    def __init__(self, w):
        self.w, self.n = w, 0

    def text(self, t, color=None):
        assert all(ord(ch) < 256 for ch in t), t
        self.n += 1
        return self.w.text(t, color=color) if color is not None else self.w.text(t)

    def __getattr__(self, k):
        return getattr(self.w, k)


def show():
    canvas.set_background_color((0.1, 0.1, 0.12))
    win.get_image_buffer_as_numpy()      # draws the frame; win.show() segfaults offscreen (30/09/2026)
frames, t_frame = 0, []
t_end = time.perf_counter() + 6.0
while time.perf_counter() < t_end:
    v.m.step(8)
    t0 = time.perf_counter()
    with g.sub_window('ledger', 0.6, 0.0, 0.4, 1.0) as w:
        lg.gui_section(Chk(w), v)
    t_frame.append((time.perf_counter() - t0) * 1e3)
    show()
    frames += 1
st = v.ledger_panel
if st.rows is None:                      # on demand above AUTO_MAX_N: what the refresh button does
    st.refresh(v.m)
st.compact = False; st.lines = None
with g.sub_window('ledger', 0.6, 0.0, 0.4, 1.0) as w:
    lg.gui_section(Chk(w), v)
show()
t_frame.sort()
print(f'N={N}: {frames} frames in 6 s, {builds[0]} ledger builds (last {st.ms:.1f} ms, cold {st.cold}, warm {st.ms_warm:.1f} ms), '
      f'panel per frame median {t_frame[len(t_frame)//2]:.2f} ms, max {t_frame[-1]:.1f} ms; '
      f'expanded lines {len(st.render_lines())}; auto={st.auto} interval={st.interval}')
print('sample lines:')
for s, c in st.render_lines()[:12]:
    print('  ', s)
