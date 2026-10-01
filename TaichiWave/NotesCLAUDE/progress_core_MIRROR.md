# Progress log: core + solids (area: core)

Owner files: membrane_core_MIRROR.py, solids_MIRROR.py, solids_tests_MIRROR.py, crosscheck_js_MIRROR.py,
identity_tests_MIRROR.py, ToolsCLAUDE/js_harness_MIRROR.mjs, ToolsCLAUDE/js_dump_MIRROR.mjs, bench_MIRROR.py,
NotesCLAUDE/solids_MIRROR.md, ScenesCLAUDE/, ScratchCLAUDE/DevCLAUDE/.
Work happens on copies in ScratchCLAUDE/DevCLAUDE/; each live file is replaced in ONE write when verified.

## 30/09/2026 (resumed after the 29/09 ~20:40 usage-limit cut)
- Found on disk: DevCLAUDE/membrane_core_MIRROR.py (~500 changed lines: 'unlicensed' shape, recorder with
  two-level reduction, k_stats rewrite, k_zero_masked, k_patch_mask, mask = domain AND NOT solid, slit drive
  skipping solids, nearest_free BFS, probes). Copies of crosscheck/identity/bench/harness/dump in DevCLAUDE are
  IDENTICAL to the live ones (no patches done). solids_MIRROR.py does not exist.
- Next: review the dev core; harness PATCHES + history exports; crosscheck histories; solids_MIRROR.py; tests.
- DONE (dev copies): harness PATCHES (neumann-corner; ZRL_JS_NOPATCH=1 disables; throws if a search string
  is missing or not unique) + getters ENERGY/PROBE/PROBE_B/MODEP/pA/pB/modeAntinode; js_dump writes the
  histories; crosscheck compares them (+ a 27th config sq5_mode_caps, N=49, 4100 steps: the 4000/600 caps).
  VERIFIED: dev crosscheck cpu f64 27/27 with patches, sq9_full_neu_corner same-seed 0.0, energy history
  rel diff <= 2.4e-14, probes <= 1.5e-14.
- Next: NOPATCH run (background), solids_MIRROR.py.
- VERIFIED: ZRL_JS_NOPATCH=1 cpu f64 -> 25/27; differing: sq9_full_neu_corner (field 3.2e-2 of max|u|,
  energy history 1.7e-2, probes 1.4e-5) and sq5_full_neu_impulse (field 1.3 x max|u| at the corner cell
  only; energy/probes 1e-14, the 5-point stencil never reads the corner). All other 25 unchanged.
- Font probe (DevCLAUDE/font_probe_MIRROR.py): Bahnschrift names work in PIL 12.3. DEATH: 'SemiBold
  SemiCondensed', tracking 0.30 cap, fit width -> stretch ~0.63 (letters ~0.49 cap wide, like the cover).
  CA EXEMPT: 'SemiBold', tracking 0.09 -> stretch ~1.0 (natural 7.32 cap vs target 8.05).
- Next: solids_MIRROR.py (writing in DevCLAUDE).
- DONE (dev): solids_MIRROR.py written (SolidObject/Stroke/SolidScene, conversions, PIL text with bilinear
  50% threshold, Unlicensed preset, painting API, save/load). Core: nearest_free is now a vectorised
  closed-form hop metric (same result as BFS), n_solid count (last solid gone -> exact web-app path),
  extend_stroke of a repeated point returns EMPTY_BOX (None would have meant a full re-rasterisation:
  that cost 0.55 s at N=2049, 5 s at N=8193 in the first timing).
- VERIFIED: dev identity (cuda) all pass incl. Neumann: ghost-cell energy sq9 drift 1.0e-14 (old rule numpy
  5.8e-3), sq5 2.9e-16 both; recorder bit-identical fields on/off. Symmetry test re-run at N=257/900 steps
  (at N=1025/341 steps the front never reached a corner: 0 cells differed).
- VERIFIED: dev solids tests (cuda, --quick) ALL PASS: loops sealed exactly (0.0) on sq5/sq9/tri; one-cell
  diagonal wall leaks sq9 7.5e-2, tri 2.31e-1, sq5 0.0; min brush along it 0.0; mirror symmetry exact;
  painting median 3.7 ms at N=2049; full re-rasterisation 383 ms at 2049.
- Next: bench (record flag, overhead), crosscheck cuda + vulkan f32, solids full (8193), notes, promote.
- Recorder tuning (DevCLAUDE/rec_tune_MIRROR.py, CUDA f64): extra per step +0.05 ms (1025), +0.47 (4097),
  +1.86 (8193) with REC_THREADS=65536 (was 131072: +0.14 at 1025). rec_fold carries the next row's value.
  bench --rec-overhead (before the tuning): sq 161 +0.056, 1025 +0.131, 4097 +0.590 ms/step.
- VERIFIED (dev, after the tuning): crosscheck 27/27 with patches on cuda f64 (dE <= 2.4e-14), vulkan f32
  (du <= 1.9e-5, dE <= 1.7e-6), cpu f64 (sq9_full_neu_corner same-seed 0.0), cuda f32. Solids tests
  vulkan f32 --quick ALL PASS. NotesCLAUDE/solids_MIRROR.md drafted.
- Next: bench dev vs live --max 2049 (recorder off), solids tests full (8193), promote, post-promotion checks.
- FOUND by bench dev vs live (--max 2049, recorder off): dev was ~15 us/step slower at N <= 1025
  (sq 161: 0.106 vs 0.091). Cause measured (DevCLAUDE/argcost_MIRROR.py): Taichi 1.7.4 charges ~30 us per
  ndarray arg and ~10 us per scalar arg per launch; the recorder's 3 ndarrays + 2 scalars were on every call.
  FIX: step body is a ti.func _steps; k_run keeps the original args (+ slitm template), k_run_rec adds HIST
  (ring + reduction slots merged) + PRB + h0; ng computed in-kernel. Now recorder off: sq 0.091/0.089/0.088,
  tri 0.114/0.109/0.103 ms (= live). Recorder on: +0.06-0.07 ms at N <= 1025, +0.13-0.16 at 2049.
- Next: re-run crosscheck (cpu, cuda, vulkan f32), identity, solids full, bench, then promote.
- FOUND: Vulkan still +4-8 us/step slower (dev vs live, alternated twice). Cause: by-value scalar params of the
  ti.func _steps become copy statements at the kernel's top level = one more serial task per launch. FIX: all
  _steps / rec_fold params are ti.template(). Vulkan now at parity (sq161 0.071 vs 0.070-0.073); recorder on
  (CUDA): +0.042-0.049 ms/step at N <= 1025 (was +0.06-0.07).
- Next: full re-verification of the final kernels (crosscheck cpu/cuda/vulkan, identity, solids), bench, promote.
- VERIFIED (final dev kernels, 30/09 ~01:55): crosscheck 27/27 cuda f64 / vulkan f32 / cpu f64 (same-seed
  sq9_full_neu_corner 0.0 on cpu); identity cuda ALL PASS; solids cuda full (8193) + vulkan f32 quick + cpu
  quick ALL PASS. Painting median 2.85 ms (2049), 3.14 ms (8193); full re-raster 0.2 s / 3.2 s.
- PROMOTED (ScratchCLAUDE/DevCLAUDE/promote_MIRROR.py, atomic os.replace per file): core, solids, solids
  tests, crosscheck, identity, bench, harness, dump. Live == dev (diff --strip-trailing-cr).
- Next: post-promotion checks (live crosscheck quick, app --shot headless, alignment crosscheck --quick),
  bench --max 2049 + rec overhead 4097/8193 on the live files, final notes.
- VERIFIED after promotion (live files): solids tests cuda --quick ALL PASS; app --shot headless renders the plate
  (DevCLAUDE/app_shot_unl_MIRROR.png); crosscheck_alignment --quick 27882/27882; bench --max 2049 at parity
  with 29/09 (cuda f64 sq 0.091/0.088/0.086/0.207, tri 0.113/0.107/0.105/0.160; vulkan 0.071/0.071/0.068/0.157);
  recorder +0.046 (161), +0.048 (1025), +0.475 (4097), +1.95 (8193) ms/step. Core docstring timing text fixed
  and re-promoted (text only). Moved my first-run scratch (atomic_probe*, smoke, xcheck_baseline) into DevCLAUDE.
- DONE. NotesCLAUDE/solids_MIRROR.md final. Open: README update (not my file) -> requested; app integration
  (painting UI, preset toggles, record toggle, default source for 'unlicensed') -> requested.
