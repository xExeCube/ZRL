"""Throughput of the Taichi port: ms per step and ns per cell, per backend and grid size,
with the web app's own solver (run headlessly under Node) as the reference.

    python bench_MIRROR.py [--max 4097] [--js] [--record]
    python bench_MIRROR.py --rec-overhead [--sizes 161 1025 4097]

Each backend runs in its own process (Taichi initialises once per process). Times are the
FULL step as the app runs it -- interior kernel, boundary kernel, mask kernel, Python
dispatch -- synchronised at the end, so small grids show the per-step launch overhead.
The table runs the SOLVER alone: the per-step recorder (m.record, the web app's record())
is off unless --record, and there are no solids -- the configuration of the 29/09/2026 table.
--rec-overhead times the recorder itself (CUDA f64): off vs on, in long runs and in 8-step
calls with a sync each (the app's frames, where each call also copies the history out).
"""
import argparse, json, os, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [161, 513, 1025, 2049, 4097, 8193, 16385]


def worker(arch, fp, nmax, record=False):
    sys.path.insert(0, HERE)
    import taichi as ti
    import membrane_core_MIRROR as mc
    mc.init_taichi(arch, fp64=(fp == 'f64'))
    out = []
    for lat in ('sq', 'tri'):
        for N in [n for n in SIZES if n <= nmax]:
            if arch == 'cpu' and N > 4097:
                continue
            p = mc.Params(N=N, src='pulse', sigma=4, lattice=lat, shape='n6' if lat == 'tri' else 'square')
            t0 = time.perf_counter()
            m = mc.Membrane(p)
            m.record = record
            setup = time.perf_counter() - t0
            t0 = time.perf_counter()
            m.step(sum(mc.CHUNKS))         # compiles every launch size once
            ti.sync()
            warm = time.perf_counter() - t0
            K0 = mc.CHUNKS[0]
            reps = max(K0, min(4096, int(4e8 / (N * N)))) // K0 * K0
            t0 = time.perf_counter()
            m.step(reps)
            ti.sync()
            dt = (time.perf_counter() - t0) / reps
            out.append({'lat': lat, 'N': N, 'ms': dt * 1e3, 'ns_cell': dt / (N * N) * 1e9, 'setup_s': setup,
                        'warm_s': warm})
            del m
    print('RESULT ' + json.dumps(out))


def worker_rec(sizes):
    """The recorder's cost, CUDA f64: ms/step with m.record off and on."""
    sys.path.insert(0, HERE)
    import taichi as ti
    import membrane_core_MIRROR as mc
    mc.init_taichi('cuda', fp64=True)
    out = []
    for lat in ('sq', 'tri'):
        for N in sizes:
            m = mc.Membrane(mc.Params(N=N, src='pulse', sigma=4, lattice=lat, shape='n6' if lat == 'tri' else 'square'))
            row = {'lat': lat, 'N': N}
            for rec in (False, True):
                m.record = rec
                m.step(sum(mc.CHUNKS))             # compiles this variant once
                ti.sync()
                K0 = mc.CHUNKS[0]
                reps = max(K0, min(4096, int(4e8 / (N * N)))) // K0 * K0
                best = 1e9
                for _ in range(3):
                    t0 = time.perf_counter()
                    m.step(reps)
                    ti.sync()
                    best = min(best, (time.perf_counter() - t0) / reps)
                calls = max(4, reps // 8)
                bestc = 1e9
                for _ in range(3):
                    t0 = time.perf_counter()
                    for _ in range(calls):
                        m.step(8)
                        ti.sync()
                    bestc = min(bestc, (time.perf_counter() - t0) / (calls * 8))
                row['on' if rec else 'off'] = best * 1e3
                row[('on' if rec else 'off') + '_call8'] = bestc * 1e3
            out.append(row)
            del m
    print('RESULT ' + json.dumps(out))


def js_times(nmax):
    """The web app's stepOnce() under Node (V8, the browser's engine), loaded as a real ES
    module the way the page loads it -- ToolsCLAUDE/js_bench_MIRROR.mjs. Includes record()
    (energy, probes), which the app runs every step."""
    Ns = [str(n) for n in SIZES if n <= min(nmax, 2049)]
    r = subprocess.run(['node', os.path.join(HERE, 'ToolsCLAUDE', 'js_bench_MIRROR.mjs'), *Ns],
                       check=True, capture_output=True, text=True)
    res = {}
    for line in r.stdout.splitlines():
        e = json.loads(line)
        res[(e['lat'], e['N'])] = e['ms_step']
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max', type=int, default=4097)
    ap.add_argument('--js', action='store_true', help='also time the web app solver under Node')
    ap.add_argument('--record', action='store_true', help='run the per-step recorder too (as the app does by default)')
    ap.add_argument('--rec-overhead', action='store_true', help='time the recorder on vs off (CUDA f64)')
    ap.add_argument('--sizes', nargs='*', type=int, default=[161, 1025, 4097])
    ap.add_argument('--worker', nargs=2)
    ap.add_argument('--worker-rec', action='store_true')
    a = ap.parse_args()
    if a.worker:
        return worker(a.worker[0], a.worker[1], a.max, a.record)
    if a.worker_rec:
        return worker_rec(a.sizes)
    if a.rec_overhead:
        r = subprocess.run([sys.executable, __file__, '--worker-rec', '--sizes', *map(str, a.sizes)],
                           capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith('RESULT ')]
        if not line:
            print('FAILED\n' + r.stderr[-800:])
            return
        print('recorder cost, CUDA f64, ms per step (long runs | 8-step calls with a sync each, as app frames)')
        print(f'{"lattice":8}{"N":>7}{"off":>10}{"on":>10}{"extra":>10}   |{"off":>10}{"on":>10}{"extra":>10}')
        for e in json.loads(line[0][7:]):
            print(f'{e["lat"]:8}{e["N"]:>7}{e["off"]:>10.3f}{e["on"]:>10.3f}{e["on"] - e["off"]:>+10.3f}   |'
                  f'{e["off_call8"]:>10.3f}{e["on_call8"]:>10.3f}{e["on_call8"] - e["off_call8"]:>+10.3f}')
        return
    js = js_times(a.max) if a.js else {}
    rows = {}
    for arch, fp in (('cuda', 'f64'), ('cuda', 'f32'), ('vulkan', 'f32'), ('cpu', 'f64')):
        r = subprocess.run([sys.executable, __file__, '--worker', arch, fp, '--max', str(a.max)]
                           + (['--record'] if a.record else []), capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith('RESULT ')]
        if not line:
            print(f'{arch} {fp}: FAILED\n{r.stderr[-800:]}')
            continue
        for e in json.loads(line[0][7:]):
            rows.setdefault((e['lat'], e['N']), {})[f'{arch} {fp}'] = e
    cols = ['cuda f64', 'cuda f32', 'vulkan f32', 'cpu f64']
    print(f'\nms per step (ns per cell)   -- web app = its stepOnce() under Node as an ES module, incl. record(); port: recorder ' + ('ON' if a.record else 'off') + ', no solids')
    print(f'{"lattice":8}{"N":>7}' + ''.join(f'{c:>20}' for c in cols) + f'{"web app (JS)":>20}{"GPU speed-up":>14}')
    for (lat, N), r in sorted(rows.items()):
        cells = ''.join(f'{r[c]["ms"]:>10.3f} ({r[c]["ns_cell"]:6.3f})' if c in r else f'{"-":>20}' for c in cols)
        j = js.get((lat, N))
        best = min(r[c]['ms'] for c in cols if c in r and c != 'cpu f64')
        jtxt = f'{j:>10.3f} ({j * 1e6 / (N * N):6.3f})' if j else f'{"-":>20}'
        sp = f'{j / best:>13.0f}x' if j else f'{"":>14}'
        print(f'{lat:8}{N:>7}{cells}{jtxt}{sp}')
    setup = {k: v.get('cuda f64', {}).get('setup_s') for k, v in rows.items()}
    print('\nsetup (mask, sources, seed on the host; cuda f64): ' +
          ', '.join(f'{lat} {N}: {s:.2f} s' for (lat, N), s in sorted(setup.items()) if s is not None))
    warm = {k: v.get('cuda f64', {}).get('warm_s') for k, v in rows.items()}
    print('first steps incl. kernel compilation (cuda f64; cached on disk after the first run): ' +
          ', '.join(f'{lat} {N}: {s:.2f} s' for (lat, N), s in sorted(warm.items()) if s is not None))


if __name__ == '__main__':
    main()
