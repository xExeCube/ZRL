# Progress: round 4 "App + v1.0" step (05/10/2026)

*Generated from scratch -- Claude Opus 5.5 -- 05/10/2026*

The step wires the round-4 core work into the app (wall types, the domain edge, the bc control per lattice and shape, the 'array' shape, the vertex-source readouts) and tags the app v1.0 (version constant, title, CLI, README, CHANGELOG). Hand-offs: NotesCLAUDE/walls_MIRROR.md section 11, NotesCLAUDE/vertex_choppiness_MIRROR.md section 7.4.

## Milestone 0: read and back up (done)

- Read: README, walls_MIRROR.md sections 1, 5, 6, 11, 13, app_MIRROR.md, the five app files, app_tests, the core's bc_info / walls_info / set_edge / source_note / drive_warnings / phase_lag / mode_note, the solids API (wall types, plate_wall, scene v2).
- Backups in ScratchCLAUDE/Round4CLAUDE/AppCLAUDE/*_before_MIRROR.*: membrane_app, app_render, app_solids, app_tests, app_input, membrane_core, README, NotesCLAUDE/app_MIRROR.md (app_notes_before_MIRROR.md).
- Plan: (1) core VERSION constant; (2) app: title, header, --version, shot log line; (3) bc / edge / array controls in the main panel; (4) wall types in the walls panel + rigid colour and stripes in the renderer + edge overlay lines; (5) source note, drive warning, phase lag, vtx_drive_bump checkbox; (6) tests; (7) shots, look at each; (8) README, CHANGELOG, app notes.

Next: the core constant and the app edits.

## Milestone 1: core constant + app wiring (done, 05/10/2026)

- Core: `VERSION = '1.0'`, `VERSION_DATE = '05/10/2026'`, `APP_NAME = 'ZRL wave membrane'` (module constants only; no logic touched).
- membrane_app_MIRROR.py: TITLE 'ZRL wave membrane v1.0'; version line at the top of the main panel; `--version`; the --shot log line names the version, the effective bc and the edge; `--win W,H`, `--plate-wall`, `--brush-wall`.
  - boundary block (gui_boundary): effective bc, B cycles only bc_info()['options'] (next_bc / cycle_bc; a triangular web-app shape: no change + a note), the ring in words, the Mur note on the triangular array, the masked-tri note; the edge control (gui_edge: set_edge keeps the field; greyed text on 'array' / the full square).
  - 'array': radius and rotation sliders replaced by greyed text.
  - vertex inset labels, the 'seed pulses under the driven vertices' checkbox (resets), the source note (under the inset button / under source x-y), the mode note, drive_warnings() under freq, phase_lag() in the readouts while a gaussian is seeded; the old bottom clip line removed.
  - walls_counts() and source_note() wait for painting / object edits to rest 0.3 s (EDIT_SETTLE); boundary_segments(): rigid edge / ring line (rigid colour), torus seam dashed violet.
- app_solids_MIRROR.py: brush wall toggle, per-object wall toggle, plate lettering wall toggle, explainer, legend, pinned / rigid counts, walls kernel on/off, edge control, 'draw the edge type' checkbox; describe() tags RIGID; edit_t.
- app_render_MIRROR.py: Look.rigid = (0.22, 0.72, 1.00); k_image / k_verts read m.C (1x1 stand-in when None); rigid cells striped in 2D (2 px of every 8 at 0.55).
- Smoke shots (ScratchCLAUDE/Round4CLAUDE/AppCLAUDE/smoke_*_MIRROR.png) looked at: rigid edge n6, rigid vs pinned 2D/3D, tri torus, vtx radial note. Fixed: the 'edge: n/a' line overflowed the panel (now wrapped); the clip line for a centred pulse read '0.0e+00 ... (sum u^2 ...)' over 4 rows (now 'nothing'; the long parenthesis only on the amber line).

Next: tests in app_tests_MIRROR.py.

## Milestone 2: tests and shots (done, 05/10/2026)

- app_tests_MIRROR.py section 11 (t_round4, 13 checks): version (constant, title, header, --version, --shot log line); B cycling per lattice / shape; tri masked note; edge (field kept, both panels, greyed on array / full square); array in G with greyed radius / rotation and the 2D fit; wall types (brush, object, plate), legend, counts and their drag wait; scene round trip; rigid vs pinned pixels 2D / 3D; edge overlay; vertex / drive / phase-lag readouts; panel rectangles never overlap (1600x1000, 2560x1377, 1280x720) + real GGUI frames with all panels.
- Full run CUDA f64: 108/108 in 83 s (ScratchCLAUDE/Round4CLAUDE/AppCLAUDE/app_tests_cuda_MIRROR.log; 95 before + 13).
- shots_r4_MIRROR.py: 20 --shot variants, all looked at. Fixes from looking: TEXT_WRAP 54 -> 52 (the main panel's scrollbar cut the last 2 characters of wrapped lines); the vertex inset button lost its '(click: next)' (cut off); the triangular 'array' overflowed the 2D strip (1.47 window heights wide) -> fit_2d() on entering it and on C, zoom 1 restored on leaving; the source note / walls counts no longer go stale after a single click edit (only drags and painting defer them, EDIT_SETTLE 0.3 s).
- Run-only helper: ScratchCLAUDE/Round4CLAUDE/AppCLAUDE/run_r4_MIRROR.py (section 11 alone, ~15 s).

Next: the regression batch (app_regression_batch_MIRROR.sh: shots, app_tests Vulkan f32, defaults, crosscheck_js CPU, identity, walls, tri_modes, solids), then README / CHANGELOG / app notes.

## Resumed 06/10/2026 (the previous attempt was cut off on 05/10 ~19:44)

State found on disk:
- Docs were done but not logged here: README_MIRROR.md (v1.0 section, controls, Files rows), the new CHANGELOG_MIRROR.md (round 1 -> v1.0), NotesCLAUDE/app_MIRROR.md (key / panel tables, section *Round 4 / v1.0*, written by patch_app_notes_MIRROR.py).
- The regression batch (app_regression_batch_MIRROR.sh) finished 5 of 8 suites, all exit 0 (app_batch_status_MIRROR.txt): shots_r4 20/20, app_tests Vulkan f32 108/108, defaults CUDA, crosscheck_js CPU, identity CUDA. It was cut off during walls_cuda (partial log kept as app_walls_cuda_cutoff_MIRROR.log; every check in it had passed).
- No python process of this project was still running.

Next: app_regression_batch2_MIRROR.sh (walls, tri_modes, solids on CUDA; they import no app file) runs in the background; meanwhile I look at the shots again. Seen: r4shot_layout_max_3d (the triangular array at the DEFAULT 3D camera) shows the rhombus far wider than the window, its near corners off-screen and under the main panel. The 05/10 notes list this as open; it looks wrong, so it gets a 3D fit like fit_2d().

## Milestone 3: the 3D fit of the triangular array (done, 06/10/2026)

- membrane_app_MIRROR.py: `Viewer.fit_3d()`. On entering the triangular `array` (G / L), on C, and at start-up / in `--shot` when no `--camera` is given, the orbit camera moves back (angles kept) until no rhombus corner lies further from the window centre, in x or y, than the square's corners at the default distance. Leaving the array restores the default distance unless the user zoomed since (as fit_2d does for the 2D zoom). A run started with `--set lattice=tri shape=array` now also gets the 2D fit (before, only `--shot` did).
- Numbers (probe_fit3d_MIRROR.py, CPU): az 0 / el 38 -> dist 3.870 (zoom 0.749), reach x 0.710 vs the square's 0.714; az 45 / el 38 -> 3.794; az 0 / el 70 -> 4.149; az -120 / el 25 -> 3.610. The same distances at 1600x1000 and 2560x1377 (the x reach is the binding one).
- Like the square's (sx 0.143 at the default camera), the rhombus's near acute corner sits under the main panel when it is open (sx 0.145): the camera's target is the domain centre at the window centre. Left as is (the same as every shape).
- app_tests_MIRROR.py: new check (d2) in section 11 (3 camera angles, C refit, leave, zoomed-since keep, C on the square). Section 11 alone on CPU: 14/14 (run_r4_MIRROR.py).
- shots_r4_MIRROR.py: the triangular 3D shots use the default camera now; new `tri_array_3d_panel` (main panel, 1600x1000). Looked at: tri_torus_3d, tri_rigidring_3d, tri_array_3d_panel, layout_max_3d -- the whole rhombus is in the window.
- Also: app_solids_MIRROR.py describe() -- the RIGID tag line had a run of spaces from the 05/10 patch script; reformatted (no behaviour change).

Next: wait for batch 2; then app_tests CUDA and Vulkan f32 (full), all shots, look at them; docs (README verification for this step, CHANGELOG, app notes).

## Milestone 4: the rest of the regression (done, 06/10/2026)

- app_regression_batch2_MIRROR.sh (CUDA f64; logs ScratchCLAUDE/Round4CLAUDE/AppCLAUDE/app_*_MIRROR.log, exit codes in app_batch_status_MIRROR.txt): walls_tests ALL PASS (64 checks, 216 s); tri_modes_tests ALL PASS (43 checks, parity with the 04/10 core 40/40 bit-identical, 215 s); solids_tests ALL PASS (33 checks, 97 s).
- With the 05/10 part of the batch, every suite in the step's list has now run once after the step's core change (three module constants): shots, app_tests Vulkan f32, defaults, crosscheck_js CPU, identity, walls, tri_modes, solids -- all exit 0.
- Docs patched for the 3D fit (patch_docs_fit3d_MIRROR.py): README controls line, CHANGELOG (the 3D fit; the vertex-inset citation now names s1_reproduce_MIRROR.py, which measured the <= 0.012%), app notes (C row, the 3D-fit bullet, the Open item).

Next: app_regression_batch3_MIRROR.sh (app_tests CUDA f64 and Vulkan f32 with the new check, then all 21 shots) is running; then look at the shots, fill the numbers into README / CHANGELOG / app notes.

## Milestone 5: final runs and docs (done, 06/10/2026) -- the step is complete

- app_regression_batch3_MIRROR.sh: app_tests CUDA f64 **109/109** (75 s), Vulkan f32 **109/109** (62 s); shots_r4 **21/21** (120 s). Logs: app_app_tests_cuda_fit3d_MIRROR.log, app_app_tests_vulkan_f32_fit3d_MIRROR.log, shots_r4_log_MIRROR.txt.
- Every PNG of the step looked at: the 21 shots (full size for the changed 3D ones, plus contact_r4shots_MIRROR.png for all) and the 8 test frames (contact_r4tests_MIRROR.png; contact_r4_MIRROR.py builds both). Nothing else wrong found.
- Docs (patch_docs_final_MIRROR.py): README section *Verification of the v1.0 app step* (the table above plus the 05/10 / 06/10 regression); CHANGELOG v1.0 counts; app notes Files rows (54 checks -> 109; the round-4 shot folder), *Tests and shots* (14 checks, 109/109 both backends, 21 shots), header sign-offs.
- Still open (unchanged by this step): with both side panels open, full-radius shapes (square, plate) are wider than the 2D strip between them; in 3D the near corners of every full-size shape (the square's and the rhombus's) sit under the main panel at the default camera; the user still chooses the default drive frequency (0.12 warns; lambda-16 default not adopted); driven vertices are hard sources.
