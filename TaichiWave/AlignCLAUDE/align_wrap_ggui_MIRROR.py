"""Scratch (01/10/2026, Claude Opus 5.5): draw alignment_MIRROR.gui_section in a REAL offscreen GGUI
window inside a 400 px wide sub_window (as the app does) and check that every text line it emits fits:
ImGui's default font advances 7 px per char, the window padding is 8 px a side -> <= 54 chars.
Checks the app's wrap (52) and the AlignPanel default; records the '[status] ...' ledger lines.
    python ScratchCLAUDE/AlignCLAUDE/align_wrap_ggui_MIRROR.py [--arch cuda]"""
import argparse, math, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import taichi as ti
import membrane_core_MIRROR as mc
import alignment_MIRROR as al
ap = argparse.ArgumentParser(); ap.add_argument('--arch', default='cuda'); a = ap.parse_args()
mc.init_taichi(a.arch, quiet=True)
D = math.pi / 180
W, H, PANEL = 1280, 800, 400
LIMIT = (PANEL - 16) // 7


class V:
    def __init__(self, m):
        self.m, self.align_panel = m, None
    def reset(self):
        self.m.reseed()
    def flash(self, s):
        pass


class Rec:
    def __init__(self, w):
        self.w, self.texts = w, []
    def text(self, t, color=None):
        self.texts.append(t)
        return self.w.text(t, color=color) if color is not None else self.w.text(t)
    def button(self, t):
        self.texts.append('[button] ' + t)
        return self.w.button(t)
    def checkbox(self, t, v):
        self.texts.append('[ ] ' + t)
        return self.w.checkbox(t, v)


win = ti.ui.Window('align wrap', (W, H), show_window=False)
gui = win.get_gui()
m = mc.Membrane(mc.Params(N=257, shape='unlicensed', lattice='sq'))
v = V(m)
worst, bad, ledger = 0, [], set()
for wrap in (52, None):
    for shape, lat, rot, rad in (('unlicensed', 'sq', 0, 1.0), ('unlicensed', 'tri', 30, 1.3), ('unlicensed', 'sq', 7, 0.8),
                                 ('n5', 'tri', 12, 1.0), ('rhomb', 'sq', 10, 1.0), ('circle', 'sq', 0, 1.0),
                                 ('square', 'sq', 0, 1.0), ('n12', 'sq', 5, 2.5)):
        m.p.shape, m.p.lattice, m.p.rot, m.p.rad = shape, lat, rot * D, rad
        m.reseed()
        v.align_panel = al.AlignPanel()
        if wrap:
            v.align_panel.wrap = wrap
        with gui.sub_window('alignment', 0.0, 0.0, PANEL / W, 1.0) as w:
            r = Rec(w)
            al.gui_section(r, v)
        win.get_image_buffer_as_numpy()          # the real frame (show() segfaults offscreen)
        for t in r.texts:
            if t.startswith('[') and not t.startswith('[button]') and not t.startswith('[ ]'):
                ledger.add(t)
            worst = max(worst, len(t))
            if len(t) > LIMIT:
                bad.append((wrap, shape, len(t), t))
        print(f'wrap {wrap or "default"} {shape:10s} {lat}: {len(r.texts)} lines, longest {max(map(len, r.texts))} chars')
print('ledger lines seen:'); [print('   ', t) for t in sorted(ledger)]
print(f'limit {LIMIT} chars ({PANEL} px panel); longest line {worst}; over the limit: {len(bad)}')
for b in bad[:8]:
    print('   OVER', b)
print('PASS' if not bad else 'FAIL')
