"""Scratch: the alignment panel on the core's REAL 'unlicensed' plate (30/09/2026)."""
import math, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import membrane_core_MIRROR as mc
import alignment_MIRROR as al
D = math.pi / 180
mc.init_taichi('cpu', quiet=True)
m = mc.Membrane(mc.Params(N=161, shape='unlicensed'))
print('has_solids', m.has_solids, 'n_solid', m.n_solid, 'plate', m.plate_dims(), 'polyN', len(m.shape_poly()))
for lat, st in (('sq', 5), ('sq', 9), ('tri', 5)):
    for rot in (0, 12, 30, 45, 60):
        for rad in (0.8, 1.0, 1.3, 2.2):
            m.p.lattice, m.p.stencil, m.p.rot, m.p.rad = lat, st, rot * D, rad
            t = time.perf_counter(); m.reseed(); tr = time.perf_counter() - t
            E = al.al_edges(m); R = al.al_rep(m)['edges']; ME = al.al_measure(m)['edges']
            print(f'{lat}{st} rot {rot:>2} rad {rad} (reseed {tr:.2f}s): fits {al.fits_inside(m, m.shape_poly())} nothing {al.al_nothing_left(m)} '
                  f'list {[e["rot"] for e in al.al_list(m)]} edges {[(round(e["phi"], 3), e["cls"], e["grid"], round(e["len"], 1)) for e in E]}')
            print('    report:', [(e['cls'], e['cnt'], e['measStraight'], e['predStraight'], e['arr'], None if e['cnt'] == 0 else round(e['mean'], 3)) for e in R])
            print('    measure:', [(e['cls'], e['meas'] and round(e['meas']['d'], 3), e['pred'], e['nCells']) for e in ME])
            for row in al.al_ledger_rows(m):
                print('    ledger:', row[1], row[2][:200])
