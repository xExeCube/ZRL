"""One-off patch: NaN tests that also hold on Vulkan (ordered compares), ACC_DT accumulators, the mode
correlation on the host where the backend has no f64 accumulators."""
import os
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
f = os.path.join(HERE, 'ledger_MIRROR.py')
s = open(f, encoding='utf-8').read()


def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a[:70], s.count(a))
    s = s.replace(a, b)


rep("""# NaN: Math.max propagates NaN; atomic_max drops it, so a NaN is flagged in its own column.""",
    """# NaN: Math.max propagates NaN; atomic_max drops it, so a NaN is flagged in its own column. The test is
# written with ORDERED compares -- `d >= 0` for d = |a - b|, `not (v == 0)` for "v !== 0" -- because on
# Vulkan `d != d` compiled as an ordered compare and never saw a NaN (measured 30/09/2026: a diverged f32
# run read 0.0 / inf where the numpy reference read NaN).
# Sums accumulate in the core's ACC_DT (f64 on CUDA / CPU even in an f32 build; f32 on Vulkan, whose f64
# accumulators crashed on 29/09): with f32 slots the (3,2) mode's correlation read 1.00000006.""")
rep("""        b = ti.ndarray(dtype=float, shape=(RSLOT + 1, 8))""", """        b = ti.ndarray(dtype=mc.ACC_DT, shape=(RSLOT + 1, 8))""")
# orbit: NaN tests
rep("""                    d = ti.abs(a - b)
                    if d != d:
                        nan = 1.0
                    else:
                        w = ti.max(w, d)""", """                    d = ti.abs(a - b)
                    if d >= 0:
                        w = ti.max(w, d)
                    else:
                        nan = 1.0""")
rep("""                d = ti.abs(U[s, i, c + dd] - U[s, i, c - dd])
                if d != d:
                    nan = 1.0
                else:
                    w = ti.max(w, d)""", """                d = ti.abs(U[s, i, c + dd] - U[s, i, c - dd])
                if d >= 0:
                    w = ti.max(w, d)
                else:
                    nan = 1.0""")
rep("""                    d = ti.abs(U[s, c + a, c + b] - U[s, i2, c - b])
                    if d != d:
                        nan = 1.0
                    else:
                        w = ti.max(w, d)""", """                    d = ti.abs(U[s, c + a, c + b] - U[s, i2, c - b])
                    if d >= 0:
                        w = ti.max(w, d)
                    else:
                        nan = 1.0""")
rep("""                        d = ti.abs(v - U[s, i2, j2])
                        if d != d:
                            nan = 1.0
                        else:
                            w = ti.max(w, d)""", """                        d = ti.abs(v - U[s, i2, j2])
                        if d >= 0:
                            w = ti.max(w, d)
                        else:
                            nan = 1.0""")
rep("""        slot = (g * 7 + kb) % RSLOT
        ti.atomic_max(OUT[slot, 0], w)
        ti.atomic_max(OUT[slot, 1], nan)
        ti.atomic_max(OUT[slot, 2], nul)""", """        slot = (g * 7 + kb) % RSLOT
        ti.atomic_max(OUT[slot, 0], ti.cast(w, mc.ACC_DT))
        ti.atomic_max(OUT[slot, 1], ti.cast(nan, mc.ACC_DT))
        ti.atomic_max(OUT[slot, 2], ti.cast(nul, mc.ACC_DT))""")
# cone
rep("""        rmax = 0.0
        cnt = 0.0
        ea = 0.0
        ei = 0.0
        for i in range(g * gs, ti.min(n, (g + 1) * gs)):
            v = U[s, i, j]
            if v != 0:""", """        rmax = 0.0
        cnt = 0.0
        ea = ti.cast(0.0, mc.ACC_DT)
        ei = ti.cast(0.0, mc.ACC_DT)
        for i in range(g * gs, ti.min(n, (g + 1) * gs)):
            v = U[s, i, j]
            if not (v == 0):                    # NaN counts, as "v !== 0" does in the JS""")
rep("""                e = v * v
                ea += e""", """                e = ti.cast(v * v, mc.ACC_DT)
                ea += e""")
rep("""        ti.atomic_max(OUT[slot, 0], rmax)
        ti.atomic_add(OUT[slot, 1], cnt)""", """        ti.atomic_max(OUT[slot, 0], ti.cast(rmax, mc.ACC_DT))
        ti.atomic_add(OUT[slot, 1], ti.cast(cnt, mc.ACC_DT))""")
rep("""    for _ in range(1):
        r = 0.0
        for k in range(RSLOT):""", """    for _ in range(1):
        r = ti.cast(0.0, mc.ACC_DT)
        for k in range(RSLOT):""")
rep("""        rout = OUT[RSLOT, 0]
        w = 0.0""", """        rout = ti.cast(OUT[RSLOT, 0], float)
        w = 0.0""")
rep("""                a = ti.abs(U[s, i, j])
                if a != a:
                    nan = 1.0
                else:
                    w = ti.max(w, a)
        slot = (g * 7 + j) % RSLOT
        ti.atomic_max(OUT[slot, 4], w)
        ti.atomic_max(OUT[slot, 5], nan)""", """                a = ti.abs(U[s, i, j])
                if a >= 0:
                    w = ti.max(w, a)
                else:
                    nan = 1.0
        slot = (g * 7 + j) % RSLOT
        ti.atomic_max(OUT[slot, 4], ti.cast(w, mc.ACC_DT))
        ti.atomic_max(OUT[slot, 5], ti.cast(nan, mc.ACC_DT))""")
# mode correlation
rep("""        num = 0.0
        da = 0.0
        db = 0.0
        for i in range(1 + g * gs, ti.min(n - 1, 1 + (g + 1) * gs)):
            if M[i, j] != 0:
                a = Si[i] * Sj[j]
                b = U[s, i, j]
                num += a * b
                da += a * a
                db += b * b""", """        num = ti.cast(0.0, mc.ACC_DT)
        da = ti.cast(0.0, mc.ACC_DT)
        db = ti.cast(0.0, mc.ACC_DT)
        for i in range(1 + g * gs, ti.min(n - 1, 1 + (g + 1) * gs)):
            if M[i, j] != 0:
                a = ti.cast(Si[i], mc.ACC_DT) * ti.cast(Sj[j], mc.ACC_DT)
                b = ti.cast(U[s, i, j], mc.ACC_DT)
                num += a * b
                da += a * a
                db += b * b""")
rep("""    if DEVICE:
        OUT = _out_buf()
        k_modecorr(m.U, _cur(m), m.M, Si.astype(m.fdt), Sj.astype(m.fdt), OUT)""",
    """    if DEVICE and mc.ACC_DT == ti.f64:          # f32 slots: the host copy with f64 sums instead
        OUT = _out_buf()
        k_modecorr(m.U, _cur(m), m.M, Si, Sj, OUT)""")
rep("""def k_modecorr(U: ti.types.ndarray(ndim=3), s: ti.i32, M: ti.types.ndarray(ndim=2), Si: ti.types.ndarray(ndim=1),
               Sj: ti.types.ndarray(ndim=1), OUT: ti.types.ndarray(ndim=2)):""",
    """def k_modecorr(U: ti.types.ndarray(ndim=3), s: ti.i32, M: ti.types.ndarray(ndim=2), Si: ti.types.ndarray(ndim=1),
               Sj: ti.types.ndarray(ndim=1), OUT: ti.types.ndarray(ndim=2)):
    # Si / Sj are f64 host arrays (only called where ACC_DT is f64)""")
open(f, 'w', encoding='utf-8', newline='\n').write(s)
print('patched')
