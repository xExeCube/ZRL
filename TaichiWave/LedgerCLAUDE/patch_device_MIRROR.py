"""One-off patch: add the device reductions to ledger_MIRROR.py and route the symmetry rows to them."""
import os
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
f = os.path.join(HERE, 'ledger_MIRROR.py')
s = open(f, encoding='utf-8').read()


def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a[:70], s.count(a))
    s = s.replace(a, b)


rep("import numpy as np\n", "import numpy as np\nimport taichi as ti\n")
anchor = "# ---- geometry the staircase row needs -----------------------------------------------------------"
dev = open(os.path.join(os.path.dirname(__file__), 'device_block_MIRROR.txt'), encoding='utf-8').read()
rep(anchor, dev + anchor)
rep("    r = symmetry_residual(x.U()) if good else None", "    r = orbit_residual(x, 'sym8') if good else None")
rep("    r = sym_rot_residual(m, x.U(), fold) if good else None", "    r = orbit_residual(x, 'rot', fold) if good else None")
rep("    r = mirror_residual(m, x.U()) if good else None", "    r = orbit_residual(x, 'mirror') if good else None")
open(f, 'w', encoding='utf-8', newline='\n').write(s)
print('patched')
