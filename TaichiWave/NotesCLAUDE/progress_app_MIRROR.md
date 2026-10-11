# Progress log: app integration (area: app)

*Claude Opus 5.5, resumed run 30/09/2026 (the 29/09 run was cut off before the app agent wrote anything).*

## State at start (30/09/2026 17:23)
- membrane_app_MIRROR.py unchanged since 29/09 19:22 (the round-1 viewer). No app_*_MIRROR.py, no ScratchCLAUDE/AppCLAUDE, no NotesCLAUDE/app_MIRROR.md.
- Core + solids are live (API handed over); alignment_MIRROR.gui_section exists; ledger_MIRROR.py not on disk yet (guarded import).

## Milestones
- [x] read core/solids/alignment APIs (17:40)
- [x] GGUI probes (ScratchCLAUDE/AppCLAUDE/colprobe_MIRROR.py, keyprobe_MIRROR.py), 30/09 17:50:
  - set_image and an unlit mesh (ambient 1, no point light) write the colour LINEARLY to the PNG: c -> floor(255 c)
    (0.2 -> 51, 0.5 -> 127). So palette values are sRGB display values; no gamma step in GGUI.
  - set_image column py = 0 is the BOTTOM row of the saved PNG (same sense as get_cursor_pos y).
  - Camera: vertical fov 45 (proj[1][1] = 2.4142 = 1/tan 22.5), aspect W/H, z_near 0.1.
  - ONE GGUI window per process (a 2nd asserts in imgui_impl_vulkan).
  - show_window=False creates NO OS window (no GLFW window for our pid), so key/mouse events cannot be
    injected headless: input handling is tested through a fake-input layer.
  - GGUI key names (binary string table): letters a-z plus Shift, Alt, Control, Escape, Return, Tab, BackSpace,
    ' ', Up/Down/Left/Right, CapsLock, LMB/MMB/RMB. Digits/punctuation: 'unrecognized id' string exists -> not
    mapped (possibly an exception). Hotkeys use letters/named keys only; digits in text capture via Win32 polling.
  - GGUI scene.mesh without normals runs gen_normals (3 atomics per triangle, every frame) over the WHOLE index
    field, padding included: padded (0,0,0) triangles normalise a zero vector -> NaN normal at vertex 0.
    The app passes its own normals.
- [x] baseline shots (18:20): round-1 app copied to ScratchCLAUDE/AppCLAUDE/membrane_app_round1_MIRROR.py, before_{3d,2d}_{square,circle}_MIRROR.png. Contrast numbers: pending (segmentation render).
- [ ] visibility (palette, exposure, backgrounds, solid colour, unlit 3D)
- [ ] orbit camera + freelook
- [ ] wall painting 2D + 3D (unprojection verified)
- [ ] objects panel + text key-capture + Unlicensed toggles
- [ ] alignment + ledger sub-windows
- [ ] --shot extensions
- [ ] verification shots, contrast, unprojection error, latency
- [ ] NotesCLAUDE/app_MIRROR.md

## 30/09/2026 18:40 -- code written (first pass, smoke shots OK)
- NEW app_camera_MIRROR.py (OrbitCam, CamPose world<->screen, ray-plane, View2D zoom/pan), app_render_MIRROR.py (Look,
  palette with zero/exposure/contrast, k_verts with DOM mask + solid/floor classes, k_normals, k_image with hatch,
  k_image_index, mesh_indices domain+floor, contrast helpers), app_input_MIRROR.py (Frame, poll_window, DigitPoller),
  app_solids_MIRROR.py (SolidsUI: painting, objects, text entry, scenes, panel).
- membrane_app_MIRROR.py rewritten on top of them (all round-1 controls kept). Smoke shots 3D/2D circle look right.
- Auto colour gain (1/peak over last 256 recorded steps) added: the round-1 ring at step 300 was nearly invisible at gain 1.
- NEXT: app_tests_MIRROR.py (mapping, projection marker, fake-input painting, text capture, GUI offscreen, latency, contrast).

## 30/09/2026 19:30 -- app_tests_MIRROR.py 54/54 on CUDA (ScratchCLAUDE/AppCLAUDE/app_tests_cuda_MIRROR.txt)
- Verified: palette (web colorOf exact at zero 0.05), 2D mapping every pixel (0-1 of 1.6M differ, on cell edges),
  3D GGUI markers vs world_to_screen <= 0.17 px, unprojection 1.9e-15, orbit clamps/no flip, fake-input painting 2D+3D
  (stroke points = intended to 1e-13 cells), erase, panels, hotkeys, WASDQE inert, text entry, Unlicensed toggles,
  GUI offscreen 10 states, latency N=513 3.25 ms / N=2049 3.14 ms median per segment.
- Contrast (mean L*, domain - background): round 1 2.9 (2D) / 3.8 (3D) -> now 23.2 / 23.3.
- FIXED on the way: zero normals render BLACK even unlit (nrm filled with +Y at allocation); GGUI to_numpy()s a
  ti.field index buffer EVERY frame (now a numpy array: -50 ms/frame at N>=1025); headless tests must flush every
  frame (GGUI keeps unrendered draw calls); window.show() headless segfaults.
- 3D frame at N=2049: ~52 ms at mesh detail 1025, ~14 ms at 513 (GGUI's vertex upload dominates) -> panel slider.
- NEXT: CLI shots of every feature, vulkan/cpu test runs, NotesCLAUDE/app_MIRROR.md.

## 30/09/2026 20:10 -- shots + polish
- ScratchCLAUDE/AppCLAUDE/shots_MIRROR.py: 38 --shot variants (orbits, zooms, lit/unlit, floor, wire, 2D zoom/pan,
  3 backgrounds, web-app look, exposure/contrast, walls 2D/3D/tri, Unlicensed sq/tri 2D/3D, screws only, solid colour,
  rotated plate, scene file, freelook, marker, segment, tri n6, N=2049, panels). Fixed: '--camera -135,...' was read as
  an option (values now glued to their flag).
- Auto-gain window 256 -> 64 steps (the pulse's early peak dimmed the ring). Default 3D mesh detail 1025 -> 513
  (frame ~52 -> ~14 ms at N=2049 CUDA); slider 129..1025.
- Panels DO appear in saved images after one warm-up frame (--panels): the round-1 README's "not in --shot" was the
  first frame only. Labels shortened for the 400 px panels (ImGui slider label room ~20 chars); alignment wrap 52.
- Vulkan f32 tests 53/54 before the mesh-limit change (3D mesh 1025 frame 70 ms), CPU quick 48/48.
- NEXT: NotesCLAUDE/app_MIRROR.md; final full runs (CUDA tests, all shots).

## 30/09/2026 22:55-23:10 -- third run (after the second usage-limit cut): final runs
- Reviewed the log and the draft notes: nothing left to build; the draft NotesCLAUDE/app_MIRROR.md was complete except
  for the final numbers.
- ledger_MIRROR.py is now on disk with gui_section(w, viewer): the panel shows the real ledger (shot_panels_all_unl_3d).
- CUDA f64 full: 54/54 (38 s). Vulkan f32 full: run 1 53/54 (mesh-513 3D frame 30.7 ms vs 30 ms limit, another
  python process running), run 2 54/54 (19.8 ms). CPU quick 48/48.
- shots_MIRROR.py: 38/38 OK against the live core/solids/ledger.
- NotesCLAUDE/app_MIRROR.md updated with these numbers. DONE.

## 01/10/2026 -- review v3 (Claude Opus 5.5); findings in ScratchCLAUDE/ReviewCLAUDE/findings_app_v3_MIRROR.md
- FIXED resize/maximise: Viewer.win_real (window.get_window_shape() every frame) drives the 3D aspect; the 2D buffer
  (Viewer.win, Renderer.resize, per-size cache) follows after RESIZE_SETTLE 0.3 s. Was 11-14 cells off (N=257) in 3D
  and a 1.16x-wide 2D circle at 2560x1377; now 0.0 cells / 0.999. Test t_resize; probe AppCLAUDE/probe_resize_v3_MIRROR.py.
- FIXED auto gain on a dead field: peak floored at 1e-3 x the run's largest |u| (Viewer.gain_ref), panel shows
  "capped". Was gain 3.2e6 at step 2000 (damp 0.02). Test t_gain.
- FIXED text entry: add() closes it; commit() only edits a text object; Return also shows hidden panels. Test t_text_switch.
- FIXED default layout: the 2D picture is centred in the strip the visible side panels leave free (View2D.ox,
  Viewer.free_centre); the main panel hid the left ~90 px of the circle. t_map2d extended (ox 0.125).
- FIXED N slider drag: rebuild only after the value rests 0.35 s (was ~1 s rebuild per drag frame in 3D). Test t_nslider.
- FIXED programmatic stroke with the path behind the camera: clear ValueError (was TypeError in Frame), --walls-test
  prints "skipped". Test t_stroke_offview. GUI press probe: 917 presses, only errors were that setup crash.
- FIXED periodic stutter at N=8193: main-panel cell counts recounted only on change (was 36 ms every 0.5 s; gui max
  120 -> 1.3 ms). Test t_cells_cache. Frame medians N=8193 3D: step x4 23 + draw 9 + gui 1 ms.

## 01/10/2026 -- typed slider values + source-moved note (Claude Opus 5.5)
- BUILT (not yet tested): app_input NumEntry + TypedGui (every slider_float/int of every panel gets a
  "^ type a value" button under it; entry: digits . - e, BackSpace, Return/Escape; clamp + flash; value
  returned as the slider's own value next pass). Viewer.handle routes all keys to it while open;
  DigitPoller active and freelook off while open; checkbox to hide the buttons. Viewer.src_moved_note()
  (amber, under source x/y; uses m.src_moved if the core has it, else solid_np at the requested cell).
- Task 3: the UNSTABLE line already calls m.co_limit() -> no change.

## 01/10/2026 -- typed values, run 2 (Claude Opus 5.5): review + pick mode
- REVIEW: 30 slider call sites -> the per-slider button added up to ~30 panel rows. CHANGED to a PICK mode:
  one "type a value: click here, then a slider" button at the top of each panel (TypedGui.header) arms
  NumEntry.armed; the next slider whose value changes (f32 rounding ignored) opens the entry under it and its
  value change is swallowed; the slider stays frozen while its entry is open; Escape (or the button) disarms.
  Per-slider buttons kept behind the main-panel checkbox 'a "type" button under every slider' (default OFF).
- src_moved_note fallback generalised: requested cell (round_cell) vs the cell used (m.src); 'out of a solid'
  when the requested cell is solid, else 'into the domain'. Core m.src_moved still preferred when present
  (not in membrane_core_MIRROR.py yet at this time).
- NEXT: tests (FakeG/FakeW headless panel driver) in app_tests_MIRROR.py, notes, runs.
- TESTS ADDED + PASS (cuda, isolated runner ScratchCLAUDE/AppCLAUDE/run_typed_MIRROR.py): 10/10. t_typed (FakeG/FakeW
  drive the REAL panel code headless: rows main 67 -> 87 with per-slider buttons vs +1 in pick mode; keys . - e
  BackSpace Return Escape; clamp/round messages; 20 hotkeys swallowed while typing; int+float slider per panel;
  N settle path 300 -> 301; per-slider button; dropped pending; real GGUI pass) and t_src_note (Unlicensed (0,0)).
- NotesCLAUDE/app_MIRROR.md: section 'Typed slider values and the source-moved note' (pick mode, why, Ctrl+click unknown, note).
- FULL CUDA f64 app_tests: 71/71 in 44 s (ScratchCLAUDE/AppCLAUDE/app_tests_cuda_typed_MIRROR.txt).
- Shots: panels_all_unl_3d, panels_main_2d, panels_* , orbit_default_3d all OK (pick-mode button visible as the
  panels' first row; entry render checked in ScratchCLAUDE/AppCLAUDE/typed_entry_MIRROR.png). Entry hint line
  shortened (it overflowed the 400 px panel). DONE.
