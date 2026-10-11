"""Scratch: the 12 --ext mismatches are at rad = (nothing-left threshold) - 1e-9. How long are
the edges whose normals / windows differ? (30/09/2026)"""
import json, math, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
import membrane_core_MIRROR as mc
import alignment_MIRROR as al
mc.init_taichi('cpu', quiet=True)
names = {'rad_none_n4_sq_t60_r7.5_R1.6271323251640277': 0, 'rad_none_n6_tri_t60_r7.5_R2.194454766381376': 0,
         'rad_none_n6_tri_t60_r12_R2.259007187909826': 0, 'rad_none_n8_sq_t60_r30_R1.5162459423011818': 6,
         'rad_none_n12_sq_t60_r0_R1.4502444711519413': 1, 'rad_none_n12_sq_t60_r7.5_R1.4885588440958288': 6,
         'rad_none_n12_tri_t60_r12_R2.083063622526268': 3, 'rad_none_rhomb_sq_t60_r45_R2.511897115688806': 2,
         'rad_none_rhomb_sq_t45_r45_R3.501199877528295': 2, 'rad_none_rhomb_tri_t45_r45_R3.272271938226312': 0}
C = {c['name']: c for c in json.load(open(os.path.join(HERE, 'ScratchCLAUDE', 'align_configs_ext_MIRROR.json')))['configs']}
m = mc.Membrane(mc.Params(N=161))
for nm, k in names.items():
    for kk, v in C[nm]['p'].items():
        setattr(m.p, kk, v)
    m.reseed()
    R = al.al_rep(m)['edges']
    short = [round(e['len'], 12) for e in R if e['len'] < 1e-3]
    print(f'{nm}: edge [{k}] len {R[k]["len"]:.3e} cls {R[k]["cls"]} cnt {R[k]["cnt"]} arr {R[k]["arr"]}; '
          f'all edges < 1e-3 cells: {short}; nothing_left {al.al_nothing_left(m)}')
nm = 'rad_none_n12_sq_t60_r0_R1.4502444711519413'
for kk, v in C[nm]['p'].items():
    setattr(m.p, kk, v)
m.reseed()
print([l for l in al.al_note(m).split('\n') if 'clips' in l or 'zeroed' in l])
