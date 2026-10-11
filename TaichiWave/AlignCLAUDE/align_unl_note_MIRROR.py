"""Scratch: the NOTE / straightness / ledger text for 'unlicensed' (30/09/2026)."""
import math, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import membrane_core_MIRROR as mc
import alignment_MIRROR as al
D = math.pi / 180
mc.init_taichi('cpu', quiet=True)
m = mc.Membrane(mc.Params(N=257, shape='unlicensed'))
for lat, rot, rad, cls in (('sq', 0, 0.8, 'row'), ('tri', 30, 1.0, 'smooth'), ('sq', 26.56505117707799, 1.3, 'row')):
    m.p.lattice, m.p.rot, m.p.rad = lat, rot * D, rad
    m.reseed()
    print(f'===== {lat} rot {rot} rad {rad} [{cls}]')
    print(al.al_note(m, cls))
    for s in al.al_straightness(m, cls):
        print('  ', s['text'])
    for r in al.al_ledger_rows(m):
        print('  ', r[1], r[0], '|', r[2][:300])
    print('  snap:', al.al_snap_options(m, cls)['header'], [e['label'] for e in al.al_snap_options(m, cls)['entries']])
