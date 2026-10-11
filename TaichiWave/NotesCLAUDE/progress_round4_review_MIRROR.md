# Progress: round 4 adversarial review (06/10/2026)

Step: "Review" of round 4 (walls, triangular modes, vertex defaults, app v1.0). Findings: ScratchCLAUDE/ReviewCLAUDE/findings_round4_MIRROR.md. Review scripts and logs: ScratchCLAUDE/Round4CLAUDE/ReviewCLAUDE/. Final regression: ScratchCLAUDE/Round4CLAUDE/FinalCLAUDE/.

## 06/10/2026 01:47 - start
- No previous review progress file existed; starting fresh. The app step returned "done" (app_tests 109/109).
- Next: read the notes and code, then the core probes (energy, leak, parity, painting, snapping, CFL, f32).

## 06/10/2026 ~02:10 - core probes, part 1
- Done: CFL from the operator's spectrum (rv_cfl_eig, rv_cfl_climb, rv_cfl_torus): no wall pattern found that lifts lam_max(-L) above the plain lattice's band edge (1,300+ adversarial patterns, sq5/sq9/tri, torus, rings, edges) -> NOT REPRODUCED.
- Done: independent numpy reference rv_ref_lib + rv_ref_compare: operator == m.operator() exactly, kernel == reference to 7.4e-14 (CUDA, 400 configs) / 2.3e-5 (Vulkan f32, 200).
- CONFIRMED C1: sq9 Neumann ring asymmetric where a solid sits ON the ring row or a clipped shape meets the ring (incl. rigid edge): blows up (max|u| 1e12+ at 200k steps). Web-app path (clipped shape, no solids) has it too (inherited).
- Next: leak tests, parity vs BeforeCLAUDE, incremental painting, snapping, CFL time runs, then the fix of C1 (word rule for the sq9 Neumann ring: a diagonal read of a ghost is blocked when the ring common is non-free; X with a pinned edge keeps the web app).

## 06/10/2026 ~02:30 - core probes done, C1 fixed
- Leak 48/48 exact 0 (CUDA, Vulkan f32); parity 66/66 bit-identical vs BeforeCLAUDE; painting 279/279 states; snapping 5000 requests 0 bad; CFL time runs 26/26 (CUDA, Vulkan f32): all NOT REPRODUCED.
- C1 FIXED in membrane_core_MIRROR.py (_word neu rule + _nring_common; kernels take neu). After: no asymmetric pairs, spectral radius 1, dense C1 runs stable, ref/paint/parity unchanged. The web-app case (clipped shape, pinned edge) left as inherited open item.
- C4 DOC: sq9 plate lettering blocks links in 42/60 configurations (rv_plate_sq9_MIRROR.py) - walls notes §8 to correct.
- Next: app review (headless Frame harness), then docs, perf, final regression.

## 06/10/2026 ~02:50 - app, docs, performance reviewed
- App: rv_app_MIRROR.py 33/33 (version, a 19-state switching sequence against fresh Membranes, N changes with rigid ring solids, rigid strokes across both torus seams in 2D/3D, scene round trip, every new control, panel cost per frame 1.00 vs 1.10 ms at N 513/4097). rv_fit3d reproduces 3.870 / 0.749 / 3.794 / 4.149.
- Docs: D1 README s1b attribution, D2 CHANGELOG painting latency, D3 (= C4) plate blocking, D4 (= C1) Neumann mirror claims: to fix in the md files. File references and the naming rule: clean.
- Perf: bench of the 03/10 core and the current one alternated twice: within noise. Walls path cost within the stated ranges (noise +-5%).
- Running in the background: walls / tri_modes / solids (CUDA) on the fixed core (rv_pre_*_MIRROR.log).
- Next: doc fixes (walls notes, README, CHANGELOG, core docstring), then the full final regression in FinalCLAUDE.

## 06/10/2026 12:40 - resumed (previous attempt cut off after ~02:51)
- Found on resume: the doc patches rv_patch_walls_notes_MIRROR.py (walls notes 02:50) and rv_patch_readme_changelog_MIRROR.py (README, CHANGELOG 02:51) had run; the pre-run on the fixed core: walls CUDA ALL PASS (rv_pre_walls_cuda_MIRROR.log); tri_modes CUDA was cut off mid-run (no summary line); solids not started.
- Next: check the applied doc patches and the core docstring, a short extra probe of the C1 fix (ring corners, random sq9 Neumann configurations), then the full final regression in FinalCLAUDE and the README table.

## 06/10/2026 ~12:44 - extra probes of the C1 fix; naming finding
- Done: the doc patches of the first attempt are in place (walls notes §2.3/§7/§8/§9/§10 + new §14; README walls bullet, vertex citation, open items, ScratchCLAUDE row; CHANGELOG review entry). The core's _word docstring already explains the neu rule.
- Done: rv_neu_ring_climb_MIRROR.py (CPU f64, N 21; 200 random ring patterns + 2 x 500 climb steps per config): square pinned / rigid edge, circle 1.3 rigid edge: max |L-L^T| 0, max |Im eig| 0, lam_max <= 0.99663 x 16/3. rv_neu_ring_bound_MIRROR.py: the climbs' best ring patterns (a pinned run on the ring row) lift lam_max by +3.3e-4 (N 21), +2.0e-5 (41), +1.2e-6 (81), +7.4e-8 (161): no bound state above 16/3 -> NOT REPRODUCED.
- CONFIRMED D6 (naming): app_tests_MIRROR.py writes proj_marker_az-120_el+20_MIRROR.png etc. (hyphens), rewritten by every app_tests run incl. round 4. Fixed (signs as p / m; backup ReviewCLAUDE/app_tests_MIRROR_before_review_MIRROR.py); the 6 existing files renamed to the new names (no deletion).
- Next: the rest of the climb (n8, circle 1.15), then the final regression (FinalCLAUDE/final_regression_batch_MIRROR.sh).

## 06/10/2026 12:47 - climb done, final regression started
- rv_neu_ring_climb: ALL OK over 5 configurations (also octagon 1.25 and circle 1.15 with a rigid edge). Findings C9 and D6 added to findings_round4_MIRROR.md; walls notes §14.1 got a bullet on the fix's own probes.
- Started FinalCLAUDE/final_regression_batch_MIRROR.sh (20 suites; status in FinalCLAUDE/final_status_MIRROR.txt). If cut off: rerun with the names of the missing suites as arguments (the script runs only those and appends).
- Next: README table *Verification after the round-4 review*, CHANGELOG line for D6, findings summary.

## 06/10/2026 ~13:35 - final regression done; docs updated
- FinalCLAUDE/final_status_MIRROR.txt: 20/20 suites exit 0 (crosscheck_js x3 27/27; identity x2 53; solids 33; beam 34/34; scale 16/16; walls x3 64; tri_modes x3 43 + parity 40/40; defaults 16; alignment 76,552 and 44,143; ledger 3,435 rows + 26/26; app_tests 109/109; bench within noise). app_tests wrote the marker images under the new names; no hyphenated name created since 03/10.
- README: new section *Verification after the round-4 review* (rv_patch_readme_final_MIRROR.py) + a Performance line; CHANGELOG: the fix's own probes and the naming fix; findings: summary table and the final run.
- Step complete. Open for the user: the inherited sq9 + Neumann + clipped-shape instability of the web app's own path (walls_MIRROR.md §14.2).
