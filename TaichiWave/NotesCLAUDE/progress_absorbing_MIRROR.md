# Progress log: absorbing boundaries research note (area: absorbing)

Owner files: `ResearchCLAUDE/AbsorbingBoundariesCLAUDE/`, `NotesCLAUDE/absorbing_boundaries_MIRROR.md`.
*Claude Opus 5.5, started 29/09/2026 (interrupted ~20:40), resumed 30/09/2026.*

## State found on 30/09/2026
- Done 29/09: `abc1d_MIRROR.py` (1D-reduced exact bench), `sweep1d_MIRROR.py` -> `results_1d_MIRROR.json` (ran, 139 s).
- Written but NOT run 29/09: `check2d_MIRROR.py`, `abc2d_MIRROR.py` + `box2d_runs_MIRROR.py` (no results json), `masked_MIRROR.py` (no results json), `gpu_cost_MIRROR.py`.
- Note not written. `tables_MIRROR.py` referenced by the sweep docstring does not exist.

## Milestones (30/09/2026)
- DONE check2d: 1D reduction exact vs a real 2D periodic-y grid (mur2 5e-14, higdon3 8e-13, cpml10 1e-14 abs).
- REVIEWED results_1d (29/09 run): measured |R| matches the discrete plane-wave theory to <1e-3 for Mur1/Mur2/Higdon2/Liao2;
  Higdon3 (undamped) and Liao3 deviate at lambda=32, 0 deg (0.069 / 0.11 vs theory 4.5e-5 / 2.7e-8) = low-frequency drift.
  Sponges weak at lambda=32 (gradient reflection); CPML10 <=1.2e-4, CPML20 <=1.1e-4 all angles.
- FOUND: the app's damping form -d(u-u_prev) lowers the CFL limit: Co_max = sqrt(0.5 - d/4) on sq5 (d=0.5 -> 0.612);
  the centred form keeps Co_max = 1/sqrt2. Box noise runs at Co 0.7 overflowed for app-form sponges (to confirm in json).
- NEW masked_normal_MIRROR.py: transport rule along the analytic normal (T1 = Liao-1 + interpolation, T2 = Liao-2 direct).
  Smoke (R=30, centred): T1 3.6e-4 (sq circle), 5.0e-4 (tri hex); T2 worse (1.6e-2) and blows up at Co 0.7. Full run launched.
- RUNNING: box2d_runs (pulse+noise done, cost pending), masked_MIRROR (sponge on sq circle / tri hex / tri circle).
- NEXT: stability1d_MIRROR.py (written), gpu_cost_MIRROR.py on CUDA, then the note.
- DONE box2d_runs -> results_box2d_MIRROR.json (5 min): CPML 10/20 best incl. corners; Mur2 with corners=0 BLOWS UP in the
  20000-step noise run (diag-Mur1 or avg corners stable); Higdon3/Liao3 blow up; Higdon3+eps transient 3e5; app-form sponges
  d=0.5 blow up at Co 0.7 (predicted Co_max 0.612), centred form stable. numpy cost column is meaningless (CPU was loaded) -> not used.
- DONE masked_MIRROR -> results_masked_MIRROR.json: distance sponge L40 d0.2: 1.7e-2 (sq circle, tri hex, tri circle alike).
- DONE masked_normal_MIRROR -> T1 5-7 % on staircase circle/hex, stable sq Co<=0.7071, tri hex grows at Co>=0.7; T2 unusable.
  (Smoke at R=30 was misleading: reflected pulse had left the domain twice.)
- DONE stability1d_MIRROR -> results_stability1d_MIRROR.json: the ky=0 static/drift modes (Mur2, Higdon2 plateau; Higdon3, Liao3 linear growth).
- DONE gpu_cost (interleaved rewrite + sp_ctr/sp_ctr_mul variants) -> results_gpu_cost_MIRROR.txt. f64 divide x2 at N=4097.
- DONE tables_MIRROR.py (prints all tables of the note).
- NEXT: write NotesCLAUDE/absorbing_boundaries_MIRROR.md.
- DONE cfl_damping_MIRROR.py: Co_max(d) = Co_max(0) sqrt(1 - d/2) for the app form, verified at d = 0.02/0.2/0.5 (+-0.005).
- DONE NotesCLAUDE/absorbing_boundaries_MIRROR.md (the note; every number cross-checked against the json/txt results).
- STATUS 30/09/2026: task complete. Open items are listed in section 9 of the note.
