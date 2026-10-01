"""Print the markdown tables of NotesCLAUDE/absorbing_boundaries_MIRROR.md from the result files.
    python tables_MIRROR.py > some_scratch_file.md
Reads results_1d_MIRROR.json, results_box2d_MIRROR.json, results_stability1d_MIRROR.json,
results_masked_MIRROR.json, results_masked_normal_MIRROR.json (whichever exist).
Written 30/09/2026.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ANG = (0, 15, 30, 45, 60, 75)
LAMS = (8, 16, 32)


def load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else None


def e(x):
    if x != x or x == float('inf') or x > 1e3:
        return '**blows up**'
    return '%.1e' % x if x < 0.095 else '%.2f' % x


def angle_table(title, rows):
    """rows: list of (label, dict 'lam/ang' -> R). One line per method: worst over the three
    wavelengths at each angle, then the 8/16/32-ppw values at normal incidence."""
    print('\n**%s** -- worst |R| over lambda = 8, 16, 32 cells at each angle; then normal incidence per lambda\n' % title)
    print('| method | ' + ' | '.join('%d deg' % a for a in ANG) + ' | 0 deg @8 / 16 / 32 |')
    print('|---|' + '---|' * (len(ANG) + 1))
    for label, d in rows:
        worst = [max(d['%d/%d' % (l, a)] for l in LAMS) for a in ANG]
        n = ' / '.join(e(d['%d/0' % l]) for l in LAMS)
        print('| %s | ' % label + ' | '.join(e(x) for x in worst) + ' | %s |' % n)


def main():
    r1 = load('results_1d_MIRROR.json')
    if r1:
        fx, tu, c7 = r1['Co0.5_fixed'], r1['Co0.5_tuning'], r1['Co0.7']
        local = ['dirichlet', 'mur1', 'mur2', 'higdon2(0,45)', 'higdon2(0,60)', 'higdon2(0,45)e.005',
                 'higdon3(0,30,60)', 'higdon3(0,30,60)e.005', 'higdon3(0,30,60)e.02',
                 'higdon3(0,45,70)e.005', 'liao2', 'liao3', 'liao2 direct-interp', 'liao3 direct-interp']
        angle_table('Local rules, Co = 0.5 (measured)', [(n, fx[n]) for n in local])
        angle_table('Local rules, discrete plane-wave theory R = -P(e^{+ikx})/P(e^{-ikx}), Co = 0.5',
                    [(n, r1['Co0.5_theory'][n]) for n in local if n in r1['Co0.5_theory']])
        layers = list(r1['best'].values())
        angle_table('Layers, best tuned per family and width, Co = 0.5',
                    [(n, tu[n]) for n in layers] + [('cerjan20(a=.015)', fx['cerjan20(a=.015)']),
                                                    ('cerjan40(a=.0075)', fx['cerjan40(a=.0075)'])])
        angle_table('Sponges terminated with Mur1 instead of Dirichlet, Co = 0.5',
                    list(r1['Co0.5_mur1end'].items()))
        angle_table('Co = 0.7', [(n, c7[n]) for n in ['mur1', 'mur2', 'higdon2(0,45)', 'higdon3(0,30,60)e.005',
                                                      'liao2', 'liao3'] + layers])
        print('\n**Per-wavelength detail of the layers (Co = 0.5)**\n')
        print('| method | lambda | ' + ' | '.join('%d deg' % a for a in ANG) + ' |')
        print('|---|---|' + '---|' * len(ANG))
        for n in layers:
            for l in LAMS:
                print('| %s | %d | ' % (n, l) + ' | '.join(e(tu[n]['%d/%d' % (l, a)]) for a in ANG) + ' |')
    rb = load('results_box2d_MIRROR.json')
    if rb:
        print('\n**2D box, 200 x 200 physical cells: pulse reflection sqrt(E_refl/E0)**\n')
        print('| method | Co 0.5 centre | Co 0.5 corner | Co 0.7 centre | Co 0.7 corner |')
        print('|---|---|---|---|---|')
        names = []
        for k in rb['pulse']:
            n = k.rsplit(' @', 1)[0]
            if n not in names:
                names.append(n)
        for n in names:
            c5 = rb['pulse'].get(n + ' @Co0.5')
            c7 = rb['pulse'].get(n + ' @Co0.7')
            f = lambda c, w: e(c[w]) if c else '-'
            print('| %s | %s | %s | %s | %s |' % (n, f(c5, 'centre'), f(c5, 'corner'), f(c7, 'centre'), f(c7, 'corner')))
        print('\n**2D box, white noise, 20000 steps: max|u| at 1000 -> 20000 steps (energy E/E0 at 20000)**\n')
        print('| method | Co 0.5 | Co 0.7 |')
        print('|---|---|---|')

        def nz(v):
            if not v:
                return '-'
            a = v['1000'][0]
            last = v.get('20000')
            if last is None or last[0] != last[0] or last[0] > 1e3:
                return 'max|u| %s -> **blows up**' % ('%.2g' % a if a < 1e3 else '%.1e' % a)
            return '%.2g -> %.2g (E %.1e)' % (a, last[0], last[1])
        for n in names:
            print('| %s | %s | %s |' % (n, nz(rb['noise'].get(n + ' @Co0.5')), nz(rb['noise'].get(n + ' @Co0.7'))))
    rs = load('results_stability1d_MIRROR.json')
    if rs:
        print('\n**1D-reduced growth check (20000 steps, random data): flagged (rule, Co, ky)**\n')
        for k in rs['flagged']:
            v = rs['runs'][k]
            print('- %s: max|u| %.2g (1000) -> %.2g (20000)' % (k, v['1000'], v['20000']))
    rm = load('results_masked_MIRROR.json')
    if rm:
        print('\n**Distance-to-edge sponge in masked shapes (R_phys = 100, pulse at centre, Co 0.5, app damping form)**\n')
        print('| shape | L | d 0.10 | d 0.20 | d 0.35 | d 0.50 |')
        print('|---|---|---|---|---|---|')
        for sh in ('sq circle', 'tri hex', 'tri circle'):
            print('| %s | 0 (Dirichlet) | %s | | | |' % (sh, e(rm['%s L0 (Dirichlet mask)' % sh])))
            for L in (10, 20, 40):
                print('| %s | %d | ' % (sh, L) + ' | '.join(e(rm['%s L%d d%.2f p2' % (sh, L, d)]) for d in (0.1, 0.2, 0.35, 0.5)) + ' |')
    rn = load('results_masked_normal_MIRROR.json')
    if rn:
        print('\n**Transport rule along the normal on the staircase edge (R = 100, Co 0.5)**\n')
        print('| shape | pulse | Dirichlet | T1 | T2 |')
        print('|---|---|---|---|---|')
        for sh in ('sq circle', 'tri circle', 'tri hex'):
            for sf in ('0.0', '0.5'):
                g = lambda r: e(rn['reflection']['%s src%sR %s' % (sh, sf, r)])
                print('| %s | %s R | %s | %s | %s |' % (sh, sf, g('dirichlet'), g('T1'), g('T2')))
        print('\nnoise runs (R = 40, max|u| at 1000 -> 20000):')
        for k, v in rn['noise'].items():
            vals = list(v.values())
            print('- %s: %.2g -> %s' % (k, vals[0], e(vals[-1]) if vals[-1] < 1e3 else '**blows up**'))


if __name__ == '__main__':
    main()
