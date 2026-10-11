"""v3 fact-check (01/10/2026): independent rigid-by-link-removal operator (own edge list, no walls_lib
in the build), energy identity test, and an adversarial lambda_max search for the rules whose
CFL-neutrality is NOT covered by the zero-extension (principal-submatrix) argument: Pcb and R1."""
import sys, math, time
import numpy as np
sys.path.insert(0, r'F:/FDrive-Storage/Claude Local/FormConstantsCLAUDE/TaichiWaveCLAUDE/ResearchCLAUDE/WallsCLAUDE')
import walls_lib_MIRROR as W

FREE, PIN, RIG = 0, 1, 2
STEN = {
    'sq9': [((1, 0), 2/3, []), ((-1, 0), 2/3, []), ((0, 1), 2/3, []), ((0, -1), 2/3, [])] +
           [((a, b), 1/6, [(a, 0), (0, b)]) for a in (1, -1) for b in (1, -1)],
    'tri': None,
}
offs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
STEN['tri'] = [(d, 2/3, [c for c in offs if (d[0]-c[0], d[1]-c[1]) in offs]) for d in offs]
LAMO = {'sq9': 16/3, 'tri': 6.0}


def my_links(T, lat):
    """Undirected link list: ('ff', i, j, w) active free-free; ('fp', i, w) free-to-zero term."""
    n0, n1 = T.shape
    t = lambda i, j: T[i, j] if 0 <= i < n0 and 0 <= j < n1 else PIN
    out = []
    for i in range(n0):
        for j in range(n1):
            if T[i, j] != FREE:
                continue
            for (di, dj), w, com in STEN[lat]:
                tj = t(i+di, j+dj)
                if tj == RIG:
                    continue                      # rigid: link removed
                if tj == PIN:
                    out.append(('fp', i*n1+j, w)); continue
                cs = [t(i+a, j+b) for a, b in com]
                if com and cs[0] != FREE and cs[1] != FREE:   # corner-blocked
                    if PIN in cs:
                        out.append(('fp', i*n1+j, w))
                    continue
                k = (i+di)*n1 + (j+dj)
                if k > i*n1+j:
                    out.append(('ff', i*n1+j, k, w))
    return out


def my_L(T, lat):
    N = T.size
    L = np.zeros((N, N))
    for e in my_links(T, lat):
        if e[0] == 'ff':
            _, i, k, w = e
            L[i, k] += w; L[k, i] += w; L[i, i] -= w; L[k, k] -= w
        else:
            _, i, w = e
            L[i, i] -= w
    return L, my_links(T, lat)


def E_links(links, u0, u1, co):
    """E^{n+1/2} written link by link (no matrix): sum (du)^2/co^2 + sum_ff w d1 d0 + sum_fp w u1 u0."""
    du = u1 - u0
    E = float(du @ du) / co**2
    for e in links:
        if e[0] == 'ff':
            _, i, k, w = e
            E += w * (u1[i]-u1[k]) * (u0[i]-u0[k])
        else:
            _, i, w = e
            E += w * u1[i] * u0[i]
    return E


def scene(n=40):
    rng = np.random.default_rng(1)
    T = np.zeros((n, n), np.int8)
    for k in range(6, 30): T[k, k] = RIG                  # 8-connected diagonal wall
    T[20:23, 5:8] = PIN                                    # pinned block
    T[30:33, 25:36] = RIG; T[8, 25:35] = PIN               # rigid bar, pinned fin
    T[(rng.random((n, n)) < 0.04)] = RIG                  # scattered rigid cells
    T[2:6, 30:34] = RIG; T[3:5, 31:33] = FREE              # rigid ring with a 2x2 cavity
    return T


print('=== A. independent operator vs walls_lib (mixed scene, pinned + rigid) ===')
for lat in ('sq9', 'tri'):
    T = scene()
    L, links = my_L(T, lat)
    Lw = W.build_L(T, W.LATS[lat], block='both', transfer=False).toarray()
    print(lat, 'max|mine - walls_lib| =', np.abs(L - Lw).max(), ' sym err =', np.abs(L - L.T).max())

print('\n=== B. energy identity (leapfrog, f64, 20000 steps) ===')
for lat in ('sq9', 'tri'):
    T = scene(); L, links = my_L(T, lat)
    free = (T.ravel() == FREE)
    lam = np.linalg.eigvalsh(-L[np.ix_(free, free)]).max()
    rng = np.random.default_rng(2)
    for frac in (0.95, 0.999):
        co = frac * 2 / math.sqrt(lam)
        u0 = rng.standard_normal(T.size) * free; u1 = u0 + 0.3 * rng.standard_normal(T.size) * free
        E0 = E_links(links, u0, u1, co); Em = W.energy(__import__('scipy.sparse', fromlist=['x']).csr_matrix(L), u0, u1, co)
        a, b = u0.copy(), u1.copy(); dmax = 0.0; S0 = (b - a).sum()
        naive = []
        for s in range(20000):
            c = 2*b - a + co*co*(L @ b); a, b = b, c
            if s % 2000 == 1999:
                dmax = max(dmax, abs(E_links(links, a, b, co) - E0) / E0)
                naive.append(float(b @ b))
        print(f'{lat} Co={co:.4f} ({frac}*CFL of this scene, lam={lam:.4f}, lam/lam_open={lam/LAMO[lat]:.4f}): '
              f'E0 link-form={E0:.6e} matrix-form={Em:.6e}; max rel drift={dmax:.1e}; '
              f'sum(u^2) ranges {min(naive):.3e}..{max(naive):.3e} (not conserved)')

print('\n=== C. adversarial lambda_max / lambda_open search (Pcb, R1, also P/R0 as controls) ===')
t0 = time.time()
for lat in ('sq9', 'tri'):
    for rule in ('P', 'R0', 'Pcb', 'R1'):
        typ, block, tr = W.RULES[rule]
        rng = np.random.default_rng(7)
        n = 14
        def lm(S):
            T = np.where(S, typ, FREE).astype(np.int8)
            L = W.build_L(T, W.LATS[lat], block=block, transfer=tr).toarray()
            f = ~S.ravel()
            if f.sum() == 0: return 0.0
            return np.linalg.eigvalsh(-L[np.ix_(f, f)]).max() / LAMO[lat]
        best = 0.0
        for start in range(6):
            S = rng.random((n, n)) < [0.1, 0.25, 0.4, 0.5, 0.6, 0.75][start]
            cur = lm(S)
            for it in range(400):
                S2 = S.copy(); k = rng.integers(n*n, size=rng.integers(1, 3)); S2.flat[k] ^= True
                v = lm(S2)
                if v >= cur: S, cur = S2, v
            best = max(best, cur)
        print(f'{lat} {rule}: max lam/lam_open found = {best:.6f}')
print(f'(search time {time.time()-t0:.0f} s)')
