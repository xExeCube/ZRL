"""One-off patch: front_radii / cone_measure / the mode correlation through the device path."""
import os
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
f = os.path.join(HERE, 'ledger_MIRROR.py')
s = open(f, encoding='utf-8').read()


def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a[:70], s.count(a))
    s = s.replace(a, b)


# ---- front_radii: gather the two rays' cells (a few hundred) instead of copying the field ----
rep('''    ci, cj = int(m.src[0]), int(m.src[1])
    M = m.mask_np
    U = x.U()
    CLEAR = _clear(m, x.G)
    tri_fix = x.tri and x.fix('tri-diagonal')
    dstep = math.sqrt(3) if tri_fix else SQRT2
    ba, bav, bd, bdv = 0, -1.0, 0.0, -1.0
    rmax = min(math.floor(CLEAR) - 2, math.ceil(x.step * p.Co) + 6)
    for r in range(3, rmax):
        if ci + r < n and M[ci + r, cj]:
            av = abs(float(U[ci + r, cj]))
            if av > bav:
                bav, ba = av, r
        dd = jround(r / dstep)
        if ci + dd < n - 1 and cj + dd < n - 1 and M[ci + dd, cj + dd]:
            dv = abs(float(U[ci + dd, cj + dd]))
            if dv > bdv:
                bdv, bd = dv, dd * dstep''',
    '''    ci, cj = int(m.src[0]), int(m.src[1])
    M = m.mask_np
    CLEAR = _clear(m, x.G)
    tri_fix = x.tri and x.fix('tri-diagonal')
    dstep = math.sqrt(3) if tri_fix else SQRT2
    ba, bav, bd, bdv = 0, -1.0, 0.0, -1.0
    rmax = min(math.floor(CLEAR) - 2, math.ceil(x.step * p.Co) + 6)
    R = range(3, max(3, rmax))
    DD = [jround(r / dstep) for r in R]
    I = [ci + r for r in R] + [ci + dd for dd in DD]
    J = [cj] * len(R) + [cj + dd for dd in DD]
    ok = [i < n for i in I]                  # (the cells the JS reads; the rest are never used)
    V = gather(x, [i if o else 0 for i, o in zip(I, ok)], [j if o and j < n else 0 for j, o in zip(J, ok)])
    for k, r in enumerate(R):
        if ci + r < n and M[ci + r, cj]:
            av = abs(float(V[k]))
            if av > bav:
                bav, ba = av, r
        dd = DD[k]
        if ci + dd < n - 1 and cj + dd < n - 1 and M[ci + dd, cj + dd]:
            dv = abs(float(V[len(R) + k]))
            if dv > bdv:
                bdv, bd = dv, dd * dstep''')

# ---- cone_measure: the device kernel (numpy reference kept as cone_measure_np) ----
rep('''def cone_measure(x):
    """coneMeasure(): support radius''', '''def cone_measure(x):
    """coneMeasure() on the device (k_cone); cone_measure_np is the numpy reference."""
    if not DEVICE:
        return cone_measure_np(x)
    m, p = x.m, x.p
    OUT = _out_buf()
    kind = 2 if x.tri else (1 if p.stencil == 9 else 0)
    wrap = 1 if (not x.tri and p.bc == 'periodic' and m.sq_full()) else 0
    k_cone(m.U, _cur(m), int(m.src[0]), int(m.src[1]), x.step, kind, wrap, 1 if x.tri else 0,
           p.Co * x.step + 1.5, OUT)
    o = OUT.to_numpy().astype(np.float64)
    e_all, e_in = float(o[:RSLOT, 2].sum()), float(o[:RSLOT, 3].sum())
    ring = math.nan if o[:RSLOT, 5].max() > 0 else float(o[:RSLOT, 4].max())
    return {'m': x.step, 'rOut': int(o[RSLOT, 0]), 'outside': int(round(o[:RSLOT, 1].sum())),
            'frac': e_in / e_all if e_all > 0 else math.nan, 'ringMax': ring}


def cone_measure_np(x):
    """coneMeasure(): support radius''')

# ---- the eigenmode correlation ----
rep('''    if p.src == 'mode' and sq_dir and p.medium == 'uniform' and x.live and not sol:
        U = x.U()
        ii = np.arange(1, n - 1, dtype=np.float64)
        Si = np.sin(mm * PI * (ii - z) / L)
        Sj = np.sin(mn * PI * (ii - z) / L)
        A = Si[:, None] * Sj[None, :]
        B = U[1:n - 1, 1:n - 1].astype(np.float64)
        Mk = m.mask_np[1:n - 1, 1:n - 1] != 0
        num = float(np.sum(A * B, where=Mk))
        da = float(np.sum(A * A, where=Mk))
        db = float(np.sum(B * B, where=Mk))
        corr''', '''    if p.src == 'mode' and sq_dir and p.medium == 'uniform' and x.live and not sol:
        num, da, db = mode_sums(x, mm, mn, z, L)
        corr''')
rep('''def _rows_mode(x):''', '''def mode_sums(x, mm, mn, z, L):
    """(sum a b, sum a^2, sum b^2) over the masked interior, a = sin(mm pi (i-z)/L) sin(mn pi (j-z)/L)."""
    m, n = x.m, x.n
    ii = np.arange(0, n, dtype=np.float64)
    Si = np.sin(mm * PI * (ii - z) / L)
    Sj = np.sin(mn * PI * (ii - z) / L)
    if DEVICE:
        OUT = _out_buf()
        k_modecorr(m.U, _cur(m), m.M, Si.astype(m.fdt), Sj.astype(m.fdt), OUT)
        o = OUT.to_numpy()[:RSLOT].astype(np.float64)
        return float(o[:, 0].sum()), float(o[:, 1].sum()), float(o[:, 2].sum())
    U = x.U()
    A = Si[1:n - 1, None] * Sj[None, 1:n - 1]
    B = U[1:n - 1, 1:n - 1].astype(np.float64)
    Mk = m.mask_np[1:n - 1, 1:n - 1] != 0
    return (float(np.sum(A * B, where=Mk)), float(np.sum(A * A, where=Mk)), float(np.sum(B * B, where=Mk)))


def _rows_mode(x):''')
open(f, 'w', encoding='utf-8', newline='\n').write(s)
print('patched')
