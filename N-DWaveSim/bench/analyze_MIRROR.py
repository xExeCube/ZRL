import csv, math, collections
rows = list(csv.DictReader(open(__file__.replace('analyze_MIRROR.py', 'sweep_MIRROR.csv'))))
D = collections.defaultdict(dict)
for r in rows:
    try: D[(r['impl'], int(r['stencil']))][int(r['N'])] = float(r['cups'])
    except ValueError: pass
impls = ['JS-f64', 'JS-f32', 'numpy', 'numba-1T', 'numba-4T', 'C-f64-1T', 'C-f32-1T', 'C-f64-4T', 'C-f32-4T']
Ns = [161, 321, 641, 1281, 2561, 4097]
for st in (5, 6, 9):
    print(f'\nstencil {st}: cell-updates per second (x1e6)')
    print(f'{"impl":10}' + ''.join(f'{n:>9}' for n in Ns))
    for im in impls:
        v = D[(im, st)]
        print(f'{im:10}' + ''.join(f'{v.get(n, float("nan"))/1e6:9.0f}' for n in Ns))
# what that buys, 5-point, using the throughput measured at each N
print('\n5-point: steps/s at N, the largest N for 60 fps x 2 steps (120 steps/s), and the wall time')
print('for ONE domain crossing (N/Co steps at Co = 0.5) -- the N^3 wall')
for im in impls:
    v = D[(im, 5)]
    sps = {n: v[n] / (n - 2) ** 2 for n in Ns if n in v}
    # largest N with steps/s >= 120, interpolating cups between measured N
    best = max([n for n in Ns if sps.get(n, 0) >= 120] or [0])
    cross = {n: (n / 0.5) / sps[n] for n in (1281, 4097) if n in sps}
    print(f'{im:10} steps/s @161 {sps.get(161,0):8.0f} @321 {sps.get(321,0):7.0f} @1281 {sps.get(1281,0):6.1f}'
          f' | real-time N <= {best:5d} | crossing @1281 {cross.get(1281, float("nan")):7.1f} s, @4097 {cross.get(4097, float("nan")):8.1f} s')
print('\nmemory for the fields alone (U0, U1, U2, CC float64 + mask):',
      ', '.join(f'N={n}: {n*n*33/2**20:.0f} MiB' for n in Ns))
print('the app ALSO allocates per vertex: position+colour float32 (24 B) + 6 uint32 indices (24 B) = 48 B:',
      ', '.join(f'N={n}: {n*n*48/2**20:.0f} MiB' for n in Ns))
