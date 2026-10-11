"""Scratch: are there 'unlicensed' configurations where every straight side is clipped away but
corner-arc pieces survive (geometry only, no reseed)? 30/09/2026."""
import math, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
import membrane_core_MIRROR as mc
import alignment_MIRROR as al
D = math.pi / 180
mc.init_taichi('cpu', quiet=True)
m = mc.Membrane(mc.Params(N=161, shape='unlicensed'))
hits = {}
for lat in ('sq', 'tri'):
    m.p.lattice = lat
    for rot10 in range(0, 901, 5):
        m.p.rot = rot10 / 10 * D
        for rad100 in range(100, 400, 1):
            m.p.rad = rad100 / 100
            A = al._al_edges_all(m)
            side = any(not e['grid'] and not e['arc'] for e in A)
            arc = any(e['arc'] for e in A)
            if arc and not side:
                hits.setdefault((lat, rot10 / 10), []).append(m.p.rad)
for k, v in hits.items():
    print(k, 'rad', v[0], '..', v[-1], len(v))
print(len(hits), 'rotations with arcs-only radii')
