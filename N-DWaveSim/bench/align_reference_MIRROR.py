"""Independent reference for grid <-> edge alignment (no code shared with the JS app).

Part A -- EXACT scan. Edge directions are rational degrees, so classification is exact
Fraction arithmetic: an edge is a ROW iff its direction mod 180 is a row angle of the
lattice, a DIAG iff it is a shortest-diagonal angle. Every slider position 0..90 in
0.05 deg steps is scanned (1801 positions) and compared with the JS module's closed-form
enumeration (align_js.json, dumped by node from the app's ALIGN section).

Part B -- MASK measurement. The domain is built from HALF-PLANES (the app uses a polar
fold -- a different predicate), boundary cells are found, and each edge's staircase
period is measured as the shortest integer lattice shift mapping its boundary cells onto
boundary cells. Run at every fully-aligned configuration Part A finds, and at a few
non-aligned controls.
"""
import json, math, sys
from fractions import Fraction as Fr
import numpy as np

HERE = sys.path[0]
ROWS = {'sq': {Fr(0), Fr(90)}, 'tri': {Fr(0), Fr(60), Fr(120)}}
DIAG = {'sq': {Fr(45), Fr(135)}, 'tri': {Fr(30), Fr(90), Fr(150)}}
DPER = {'sq': math.sqrt(2), 'tri': math.sqrt(3)}

def m180(x): return x % 180

def edge_dirs(shape, rot, theta=None):
    """directions (deg, Fraction) of every edge, parallel edges counted separately"""
    if shape == 'rhomb':
        h = Fr(theta) / 2
        return [m180(rot - h), m180(rot + h)] * 2
    n = int(shape[1:])
    return [m180(rot + 90 + Fr(360 * k, n)) for k in range(n)]

def count(shape, rot, lat, theta=None):
    d = edge_dirs(shape, rot, theta)
    return sum(x in ROWS[lat] for x in d), sum(x in DIAG[lat] for x in d), len(d)

def scan(shape, lat, theta=None):
    out = []
    for k in range(0, 1801):
        r = Fr(k, 20)
        rows, diags, n = count(shape, r, lat, theta)
        if rows + diags:
            out.append((r, rows, diags, n))
    return out

# ---------------- Part A: exact scan vs the JS enumeration -------------------------
js = json.load(open(HERE + '/align_js_MIRROR.json'))
mism = 0; checked = 0
for key, jl in js.items():
    lat, sh = key.split('/')
    theta = None
    if sh.startswith('rhomb'):
        theta = Fr(sh[5:]); shape = 'rhomb'
    else:
        shape = sh
    ref = scan(shape, lat, theta)
    refd = {float(r): (a, b) for r, a, b, n in ref}
    jsd = {round(e[0], 9): (e[1], e[2]) for e in jl}
    for r, v in jsd.items():
        checked += 1
        if refd.get(r) != v:
            mism += 1; print('MISMATCH (JS entry)', key, r, v, refd.get(r))
    for r, v in refd.items():
        if round(r, 9) not in jsd:
            mism += 1; print('MISSING in JS', key, r, v)
print(f'Part A: {len(js)} configurations, {checked} JS entries checked against the exact '
      f'0.05-deg scan: {mism} mismatches')

# full-alignment table
def full_table():
    rows_out = []
    for lat in ('sq', 'tri'):
        for sh in ('n3', 'n4', 'n5', 'n6', 'n8', 'n12'):
            S = scan(sh, lat)
            best_row = max(S, key=lambda e: (e[1], e[2]))
            best_sm = max(S, key=lambda e: (e[1] + e[2], e[1]))
            full_row = [float(e[0]) for e in S if e[1] == e[3]]
            full_sm = [float(e[0]) for e in S if e[1] + e[2] == e[3]]
            rows_out.append((lat, sh, best_row[1], best_row[3], full_row, best_sm[1] + best_sm[2], full_sm))
    return rows_out
print('\nlat  shape  max-rows  rot(all rows)            max-smooth  rot(all row|diag)')
for lat, sh, br, n, fr, bs, fs in full_table():
    print(f'{lat:4} {sh:5}  {br}/{n:<3}    {str(fr):24} {bs}/{n:<3}      {fs}')

# rhombus: which theta (0.5 grid, 20..90) allow full alignment, and at which rotations
print('\nrhombus theta with ALL 4 edges aligned (0.5-deg grid, 20..90):')
for lat in ('sq', 'tri'):
    for mode in ('row', 'smooth'):
        hits = []
        for t2 in range(40, 181):
            th = Fr(t2, 2)
            S = scan('rhomb', lat, th)
            full = [float(e[0]) for e in S if (e[1] if mode == 'row' else e[1] + e[2]) == 4]
            if full: hits.append((float(th), full))
        print(f'  {lat} {mode:6}:', '; '.join(f'theta {t} @ {r}' for t, r in hits))

# ---------------- Part B: mask measurement ----------------------------------------
def lattice_xy(lat, I, J):
    return (I + 0.5 * J, J * math.sqrt(3) / 2) if lat == 'tri' else (I.astype(float), J.astype(float))

def polygon(shape, R, rot_deg, theta=None):
    rot = math.radians(rot_deg)
    if shape == 'rhomb':
        B = R * math.tan(math.radians(theta) / 2)
        P = [(R, 0), (0, B), (-R, 0), (0, -B)]
        c, s = math.cos(rot), math.sin(rot)
        return [(x * c - y * s, x * s + y * c) for x, y in P]
    n = int(shape[1:])
    return [(R * math.cos(rot + math.pi / n + 2 * math.pi * k / n),
             R * math.sin(rot + math.pi / n + 2 * math.pi * k / n)) for k in range(n)]

def measure(shape, lat, rot_deg, theta=None, R=64.0, H=80):
    rng = np.arange(-H, H + 1)
    I, J = np.meshgrid(rng, rng, indexing='ij')
    X, Y = lattice_xy(lat, I, J)
    P = polygon(shape, R, rot_deg, theta); m = len(P)
    inside = np.ones_like(X, dtype=bool)
    for k in range(m):                                   # half-planes, CCW polygon
        (ax, ay), (bx, by) = P[k], P[(k + 1) % m]
        inside &= (bx - ax) * (Y - ay) - (by - ay) * (X - ax) >= -1e-9
    nb = [(1, 0), (-1, 0), (0, 1), (0, -1)] + ([(1, -1), (-1, 1)] if lat == 'tri' else [])
    outside_nb = np.zeros_like(inside)
    for di, dj in nb:
        outside_nb |= ~np.roll(np.roll(inside, -di, 0), -dj, 1)
    bnd = inside & outside_nb
    bset = set(zip(I[bnd].tolist(), J[bnd].tolist()))
    bx_, by_ = X[bnd], Y[bnd]; bi, bj = I[bnd], J[bnd]
    # assign boundary cells to nearest edge
    E = []
    for k in range(m):
        (ax, ay), (bx2, by2) = P[k], P[(k + 1) % m]
        L = math.hypot(bx2 - ax, by2 - ay); ux, uy = (bx2 - ax) / L, (by2 - ay) / L
        E.append((ax, ay, ux, uy, L))
    D = np.full(bx_.shape, np.inf); A = np.full(bx_.shape, -1); Sx = np.zeros_like(bx_)
    for k, (ax, ay, ux, uy, L) in enumerate(E):
        rx, ry = bx_ - ax, by_ - ay; s = rx * ux + ry * uy; t = -rx * uy + ry * ux
        d = np.where(s < 0, np.hypot(rx, ry), np.where(s > L, np.hypot(bx_ - (ax + ux * L), by_ - (ay + uy * L)), np.abs(t)))
        better = d < D; D[better] = d[better]; A[better] = k; Sx[better] = s[better]
    # candidate primitive vectors, shortest first
    cand = []
    for b in range(0, 25):
        for a in range(-24, 25):
            if (b == 0 and a <= 0) or math.gcd(a, b) != 1: continue
            x, y = (a + b / 2, b * math.sqrt(3) / 2) if lat == 'tri' else (a, b)
            cand.append((math.hypot(x, y), a, b, x, y))
    cand.sort()
    periods = []
    for k, (ax, ay, ux, uy, L) in enumerate(E):
        margin = 12.0            # >= 2.5/tan(alpha/2) for every vertex angle tested (30 deg: 9.3)
        lo, hi = margin, L - margin
        sel = (A == k) & (D < 2) & (Sx >= lo) & (Sx <= hi)
        cells = list(zip(bi[sel].tolist(), bj[sel].tolist(), Sx[sel].tolist()))
        found = None
        for ln, a, b, x, y in cand:
            if ln > (hi - lo) / 2: break
            pr = x * ux + y * uy; sg = -1 if pr < 0 else 1; ps = abs(pr)
            tested = 0; ok = True
            for i, j, s in cells:
                if s + ps > hi: continue
                if (i + sg * a, j + sg * b) not in bset: ok = False; break
                tested += 1
            if ok and tested >= 4: found = ln; break
        periods.append(found)
    return periods

print('\nPart B: boundary periods measured on an independent half-plane mask (R = 64)')
bad = 0; nconf = 0
for lat in ('sq', 'tri'):
    for sh in ('n3', 'n4', 'n5', 'n6', 'n8', 'n12'):
        for r, rows, diags, n in scan(sh, lat):
            if rows + diags != n: continue
            per = measure(sh, lat, float(r)); nconf += 1
            dirs = edge_dirs(sh, r)
            n_ = len(dirs); dirs = [dirs[(k + 1) % n_] for k in range(n_)]   # polygon edge k = dir index k+1
            exp = [1.0 if d in ROWS[lat] else DPER[lat] for d in dirs]
            ok = all(p is not None and abs(p - e) < 1e-9 for p, e in zip(per, exp))
            bad += not ok
            if not ok or float(r) in (0.0, 30.0, 45.0):
                print(f'  {lat} {sh:4} @ {float(r):5.2f}: periods {[round(p,3) if p else None for p in per]} '
                      f'{"OK" if ok else "** MISMATCH **"}')
    for th in (60, 45, 30, 90):
        for r, rows, diags, n in scan('rhomb', lat, Fr(th)):
            if rows + diags != n: continue
            per = measure('rhomb', lat, float(r), th); nconf += 1
            exp = [1.0 if d in ROWS[lat] else DPER[lat] for d in edge_dirs('rhomb', r, Fr(th))]
            ok = all(p is not None and abs(p - e) < 1e-9 for p, e in zip(per, exp)); bad += not ok
            if not ok: print(f'  {lat} rhomb({th}) @ {float(r)}: {per} ** MISMATCH **')
print(f'  {nconf} fully-aligned configurations measured, {bad} mismatches')

print('\ncontrols (not aligned) -- measured periods:')
for sh, lat, r in (('n5', 'sq', 0), ('n5', 'tri', 6), ('n8', 'tri', 0), ('n6', 'sq', 0), ('n12', 'sq', 0)):
    print(f'  {lat} {sh} @ {r}: {[round(p,3) if p else None for p in measure(sh, lat, r)]}')
