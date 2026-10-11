# Progress log: round 4, "Defaults + docs" (core side)

Owner area: membrane_core_MIRROR.py (vtx_inset default, driven-vertex start, source_note(), drive_warnings()),
ToolsCLAUDE/js_harness_MIRROR.mjs (new PATCH), the cross-check / test configs that relied on the web-app
defaults, README_MIRROR.md, NotesCLAUDE/walls_MIRROR.md (section 13), NotesCLAUDE/vertex_choppiness_MIRROR.md.
Backups (taken before any edit, 05/10/2026): ScratchCLAUDE/Round4CLAUDE/DefaultsCLAUDE/*_before_MIRROR.*.
Logs: ScratchCLAUDE/Round4CLAUDE/LogsCLAUDE/def_*_MIRROR.log.

## 05/10/2026
- Started. Read README, progress_trimodes / progress_walls_build, the s13 draft, patch_readme_MIRROR.py (needs
  readme_regression_MIRROR.md, never written), vertex_choppiness_MIRROR.md, the core's vertices / seed / drive /
  scale_warnings / bc_info, the harness PATCHES, crosscheck_js / _ledger / _alignment configs, make_ref (scale).
- Backups of 10 files taken (folder above).
- Plan: (1) bench -> s13 -> walls_MIRROR.md section 13, README; (2a) Params.vtx_inset = 'wall';
  (2b) source_note(), drive_warnings(); (2c) driven-vertex dent via Params.vtx_drive_bump (False = start
  from rest, like 'cont') + harness PATCH 'vtx-drive-rest' keyed on S.vtxDriveBump (absent in the real web app);
  (3) docstring + README deviations; (4) all suites -> def_*_MIRROR.log.
- 05/10 bench (tri_modes_bench_MIRROR.py --reps 5, two runs each: cuda f64, cuda f32, vulkan f32) ->
  LogsCLAUDE/def_bench_{cuda_f64,cuda_f32,vulkan_f32}_run{1,2}_MIRROR.log. FOUND: cuda f32 run 1 crashed
  (TaichiNameError in rec_fold) because the core was edited while it ran: Taichi compiles kernels lazily
  from the source file ON DISK, so a mid-run edit hands it shifted lines (kept as
  def_bench_cuda_f32_run1_crashed_MIRROR.log; rerun clean). Rule: never edit a .py a running Taichi job imported.
  Key numbers (N 8193, ms/step, recorder off): hexagon 2.72 / array clamped 4.31-4.40 / rigid ring 3.80-4.09 /
  torus 4.41-4.43 (cuda f64); the array updates every cell, the hexagon ~56%.
- 05/10 DONE core: Params.vtx_inset default 'wall'; Params.vtx_drive_bump (False = driven vertices start from
  rest; seed() via vtx_bumps()); source_note() (cached dict + lines), clearance_from() (window doubling,
  == ledger _clear), drive_warnings(), phase_lag() (energy-weighted dispersion lag, cached). Harness PATCH
  'vtx-drive-rest' keyed on S.vtxDriveBump. Configs set the web-app values: crosscheck_js (tri_n3_conf_vtxdrive
  radial + bump, tri_square_slab_vtx radial; sq5_n8_vtxdrive_saw now tests the rest start), crosscheck_ledger
  (3 radial, the vtxdrive config bump=True to match the saved dump), crosscheck_alignment (m radial),
  make_ref (radial + bump), tri_modes test 8 (radial + bump; kwargs filtered by the 04/10 Params fields).
- 05/10 defaults_tests_MIRROR.py (new, tests 1-5). --quick on cuda: ALL PASS. Phase-lag predictor vs the lag
  measured on the lattice (N 257, 0.7 crossings, vs the exact Hankel solution): 0-2% on sq5 0/26.6/45 deg and
  tri 0/30/60/90 deg. App defaults (sq5 N 513 sigma 3): 0.60-1.79 cells per crossing, 1.7 crossings to one sigma.
- Next: the full regression ScratchCLAUDE/Round4CLAUDE/DefaultsCLAUDE/def_regression_batch_MIRROR.sh
  (status LogsCLAUDE/def_batch_status_MIRROR.txt); meanwhile docs (s13 bench table, walls section 13, README).
- 05/10 regression (def_regression_batch_MIRROR.sh; status LogsCLAUDE/def_batch_status_MIRROR.txt): defaults cuda/cuda f32/
  vulkan ALL PASS; crosscheck_js cpu/cuda/vulkan-f32 27/27 (both patches); NOPATCH cpu 24/27 (2 Neumann corners +
  sq5_n8_vtxdrive_saw, du(0) 2.5, same-seed stepping 1.5e-15); identity cuda/vulkan ALL PASS (53); scale 16/16; beam
  34/34; walls cuda ALL PASS (64); solids ALL PASS (33); crosscheck_ledger saved dump PASS (3,435 rows, 26/26 port-only),
  fresh dump with both patches PASS; crosscheck_alignment --reuse 76,552/76,552; app_tests 95/95; 4 shots render.
  FOUND: tri_modes test 8 FAILED 24/40 on cuda and vulkan -- fields bit-identical, but it compares m.vtx for EVERY source
  and the default inset changed. Fixed the test (every parity case on the 04/10 inset); rerun: 40/40, ALL PASS on both
  (failing logs kept as def_trimodes_*_run1_MIRROR.log). phase_lag text: crossings < 10 now print one decimal; defaults
  cuda rerun ALL PASS (run 1 kept).
- 05/10 phase lag at N 1025 (test 5, 0.7 crossings): 0-3% while the accumulated lag < ~0.3 sigma, 6-13% high at 0.45-0.75
  sigma (linear estimate; documented). probe_defaults_table_MIRROR.py: wall clip 0.0000-0.0053%, clearance 2.85-4.27 sigma.
- 05/10 DOCS DONE: walls_MIRROR.md section 13 appended (bench filled; contents line; section 12 row);
  vertex_choppiness_MIRROR.md section 7 (changes, measurements, UI hand-off 7.4, open items 7.5) + status line at the top;
  README (patch_readme_r4_MIRROR.py: round-4 verification table, walls/tri section, vertex-defaults section, deviations
  6-7, files, backlog: absorbing boundaries next); core docstring deviations.
- Open (not mine to wire): the panel UI for source_note / drive_warnings / phase_lag / vtx_drive_bump (hand-off 7.4).
  STEP COMPLETE.
