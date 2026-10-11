"""Exploration: the ledger with solids / the unlicensed plate (port-only)."""
import os, sys, time
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import membrane_core_MIRROR as mc
import solids_MIRROR as so
import ledger_MIRROR as lg
mc.init_taichi(sys.argv[1] if len(sys.argv) > 1 else 'cuda', quiet=True)
N = 161


def show(title, m, steps=1300):
    m.step(steps)
    rows = lg.build_ledger(m)
    print(f'\n===== {title} (step {m.step_n}, solids {m.n_solid}) =====')
    for r in rows:
        print(f'  [{r.badge:>8}] {r.key:10s} {r.name[:60]:60s} {r.value}')
        if r.status in ('FAIL', 'NA') or r.port:
            print(f'              note: {r.note[:220]}')
            if r.port:
                print(f'              port: {r.port[:220]}')


def sym8_bitmap(n):
    c = (n - 1) // 2
    I, J = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing='ij')
    S = np.zeros((n, n), np.uint8)
    for a, b in ((30, 0), (0, 30), (-30, 0), (0, -30), (22, 22), (-22, 22), (22, -22), (-22, -22)):
        S |= ((I - a) ** 2 + (J - b) ** 2 <= 36)
    return S


m = mc.Membrane(mc.Params(N=N, src='pulse'))
m.set_solid_np(sym8_bitmap(N)); m.reseed()
show('sq5 full pulse, 8-fold symmetric solid bitmap', m)

m = mc.Membrane(mc.Params(N=N, src='pulse'))
S = np.zeros((N, N), np.uint8); S[100:110, 40:70] = 1
m.set_solid_np(S); m.reseed()
show('sq5 full pulse, asymmetric solid block', m)

m = mc.Membrane(mc.Params(N=N, src='mode', mm=3, mn=2))
m.set_solid_np(S); m.reseed()
show('sq5 mode (3,2), asymmetric solid block', m)

m = mc.Membrane(mc.Params(N=N, lattice='tri', shape='n6', rot=30 * mc.PI / 180, src='impulse'))
sc = m.ensure_scene(); sc.add_circle(0.5, 0.0, 0.08); m.update_solids(None); m.reseed()
show('tri n6 conf impulse, one circle (scene)', m)

m = mc.Membrane(mc.Params(N=N, shape='unlicensed', src='pulse', sy=-0.9))
show('unlicensed plate (default scene) pulse sy=-0.9', m)

m = mc.Membrane(mc.Params(N=N, shape='unlicensed', src='pulse'))
m.scene.set_preset(plate_text=False, plate_screws=False); m.update_solids(None); m.reseed()
show('unlicensed plate, preset off (bare plate), centred pulse', m)

m = mc.Membrane(mc.Params(N=N, shape='unlicensed', src='vtx'))
show('unlicensed plate vtx', m)
