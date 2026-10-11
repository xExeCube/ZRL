# TaichiWaveCLAUDE — the ZRL 2D membrane, ported to Python/Taichi

**This is the primary app** (decided 29/09/2026). The browser original, `FlavorRenderersCLAUDE/wave_membrane_MIRROR.html`, is frozen and serves as the reference. The solver runs on the GPU and is still checked value-for-value against the web app's own code; where the port deliberately differs, the difference is listed below and patched into the reference so the comparison stays whole-field.

Plan (agreed 29/09/2026): **port** (done) → **solid objects + walls + the "Unlicensed" plate** (done, pinned walls) → **beam source** (done 01/10/2026) → **rigid walls** (done 04/10/2026) → **triangular-lattice boundary modes** (done 05/10/2026) → absorbing boundaries → prism-like media → lattice extension. **Next up: absorbing boundaries.**

*Generated from scratch — Claude Opus 5.5 — 29/09/2026; round 2 rewrite 01/10/2026; round 4 additions (walls, triangular modes, vertex-source defaults) 05/10/2026; v1.0 section, controls and the app-wiring verification 05/10/2026; the 3D fit and the app-step verification table 06/10/2026; the round-4 review (a fix of the sq9 Neumann ring) and the final verification 06/10/2026; the user's v1.0 decisions (Deviations 8–10, files, backlog) 09/10/2026*

## v1.0 (05/10/2026)

The app is **v1.0**: `VERSION` in `membrane_core_MIRROR.py`, shown in the window title ("ZRL wave membrane v1.0"), the main panel's first line, `python membrane_app_MIRROR.py --version` and the `--shot` log line. What each round added: [CHANGELOG_MIRROR.md](CHANGELOG_MIRROR.md). v1.0 contains:
- **The port** of the web app's solver (square 5- and 9-point, triangular), cross-checked whole-field against the web app's own code under Node, on CUDA, Vulkan and CPU.
- **Solids and walls:** objects, painted walls and the Unlicensed plate, each **pinned** (u = 0, echo inverted) or **rigid** (zero slope, echo upright), with corner blocking; a pinned or rigid domain **edge**; the walls panel sets the brush's, each object's and the plate lettering's type, and rigid solids show in cyan (striped in 2D).
- **Boundaries per lattice and shape:** the `array` shape; on the triangular array a clamped ring, a rigid ring or the rhombic torus (Mur is not on the triangular lattice yet). **B** cycles only what acts on the current lattice and shape, and the panel says what runs.
- **Sources:** pulse, impulse, continuous, slit, drum mode, vertices (wall inset by default, driven vertices start from rest), the steerable one-sided **beam** and the one-way **packet**; the source note, the drive-wavelength warning and the phase-lag readout in the main panel.
- **Viewer:** 2D / 3D, an orbit camera, palettes with an editor, real transparency, scale mode (cells or fractions of the domain), typed slider values, the alignment and physics-ledger panels, and a headless `--shot` mode.

## Files

| file | what it is |
|---|---|
| `CHANGELOG_MIRROR.md` | what each round added, dated (round 1 → v1.0) |
| `membrane_core_MIRROR.py` | the solver, no GUI. It holds `VERSION` (the app's version, 1.0). Geometry, masks, radius snaps, source placement and seeding on the host (numpy, the JS's arithmetic); the time step is one fused Taichi kernel; per-step recorder (energy, peak, probes) for the ledger; `mask = domain AND NOT solid`. |
| `solids_MIRROR.py` | solid objects (rect, circle, n-gon, text), hand-painted walls (draw/erase strokes), the Unlicensed plate's lettering and screw heads, incremental painting, save/load of scenes. Every object and stroke is **pinned** (u = 0, the default) or **rigid** (zero normal slope), since 04/10/2026. |
| `beam_MIRROR.py` | the directional sources' maths: lattice symbols and discrete dispersion, energy aim, beam aperture tables, the one-way packet (see *Directional sources*). |
| `alignment_MIRROR.py` | the grid-alignment panel, ported from the web app (aligned angles, step/optimal angle/optimal flavor, straightness report). |
| `ledger_MIRROR.py` | the physics ledger, ported from the web app, with a few documented fixes (see *Deviations*). |
| `membrane_app_MIRROR.py` + `app_camera_MIRROR.py`, `app_render_MIRROR.py`, `app_input_MIRROR.py`, `app_solids_MIRROR.py` | the viewer (Taichi GGUI): orbit camera, 2D/3D views, look controls, wall painting, objects, the alignment and ledger panels, a headless `--shot` mode. |
| `crosscheck_js_MIRROR.py` | runs the **web app's own solver** headlessly under Node and compares mask, per-cell Co², seed, field after K steps, vertex cells, clip fraction, peak and the recorder histories. 28 configurations (the 28th, 09/10/2026: the sq9 Neumann ring with a clipped circle). |
| `crosscheck_alignment_MIRROR.py`, `crosscheck_ledger_MIRROR.py` | the same for the alignment functions (76,552 comparisons over 2,295 configs with `--reuse` of the full dump, 05/10/2026; 27,882 over 612 on 01/10) and the ledger (50 configs, 3,435 rows; `--jsonl` reuses a saved web-app dump). |
| `beam_tests_MIRROR.py` | directivity, beam width, steering, packet residue and speed, stability, obstacles, cost. |
| `identity_tests_MIRROR.py`, `solids_tests_MIRROR.py`, `app_tests_MIRROR.py` | exact properties, solids (sealing, rasterisation, painting), app logic (camera, painting maps, text and number entry, panels; section 11: the round-4 controls and v1.0). |
| `walls_tests_MIRROR.py`, `tri_modes_tests_MIRROR.py` | rigid walls and corner blocking (64 checks); the triangular-lattice modes (the `array` shape, the rigid ring, the rhombic torus, the Mur fallback) and the parity of every web-app configuration with the previous core. |
| `defaults_tests_MIRROR.py` | the vertex-source defaults (wall inset, driven vertices from rest) and the source helpers: `source_note()` against the ledger's clearance, `drive_warnings()`, `phase_lag()` against the lag measured on the lattice. |
| `decisions_tests_MIRROR.py` | the user's v1.0 decisions (09/10/2026): the sq9 Neumann clipped-shape fix (symmetry, spectral radius, energy, the 200,000-step reproducer, parity with v1.0), the default freq 0.03125, soft driven vertices (transparency, the soft / hard amplitude ratio, growth, the hard path bit-identical to v1.0). [NotesCLAUDE/v1_decisions_MIRROR.md](NotesCLAUDE/v1_decisions_MIRROR.md). |
| `bench_MIRROR.py` | throughput per backend and grid size, with the web app's solver as the reference. |
| `ToolsCLAUDE/` | the Node side: `js_harness_MIRROR.mjs` (loads the web app's module in a `vm`, applies the **PATCHES** list), `js_dump`, `js_align_dump`, `js_ledger_dump`, `js_bench` (times the web app as a real ES module). |
| `NotesCLAUDE/` | per-area notes (`app`, `solids`, `alignment`, `ledger`, `beam`, `scale`, `walls`), the research notes (`rigid_vs_pinned_walls`, `absorbing_boundaries`, `vertex_choppiness`), and progress logs. |
| `ResearchCLAUDE/` | prototype scripts behind the research notes. |
| `ScenesCLAUDE/` | saved scenes (objects + strokes + plate toggles), JSON. |
| `ScratchCLAUDE/` | throwaway probes, logs and screenshots (`FinalCLAUDE/` holds the 01/10 verification logs, `Round4CLAUDE/LogsCLAUDE/` the round-4 ones: `final_*` 04/10, `tri_*` and `def_*` 05/10; `Round4CLAUDE/AppCLAUDE/` the v1.0 app step's logs, shots and scripts; `Round4CLAUDE/ReviewCLAUDE/` the round-4 review's own scripts and logs (`rv_*`), `Round4CLAUDE/FinalCLAUDE/` its final regression; the findings are in `ReviewCLAUDE/findings_round4_MIRROR.md`). |

## Running it

Python 3.12 with `taichi` 1.7.4, `numpy`, `PIL` (all installed; `python` here is 3.12), and Node for the cross-checks.

```bash
python membrane_app_MIRROR.py
```

```bash
python membrane_app_MIRROR.py --N 4097 --f32
```

```bash
python crosscheck_js_MIRROR.py --arch cpu
```

```bash
python app_tests_MIRROR.py --arch cuda
```

```bash
python defaults_tests_MIRROR.py --arch cuda
```

Backends: `--arch cuda` (default, f64), `cpu`, `vulkan` (f32 only). `--f32` halves memory and roughly doubles speed at large N.

```bash
python membrane_app_MIRROR.py --version
```

**Controls** (full list: [NotesCLAUDE/app_MIRROR.md](NotesCLAUDE/app_MIRROR.md)):
- `SPACE` play/pause · `N` one step · `R` reset · `P` drop a pulse · `V` 3D/2D · `X` wireframe · `H` all panels
- `L` lattice · `B` boundary (cycles only the ones that act here; on a triangular web-app shape it changes nothing and says so) · `G` domain shape (includes `unlicensed` and `array`) · `M` medium · `O` source (includes `beam`, `packet`)
- Main panel, under the boundary: the effective bc and the ring in words, and the **edge** button (pinned / rigid, masked shapes only; it keeps the field). On `array` the radius and rotation sliders are greyed text.
- Walls panel (`I`): **brush wall**, each selected object's **wall**, the **plate lettering wall** (pinned / rigid), a legend, the pinned / rigid cell counts, the edge button, and *draw the edge type* (the rigid line, the dashed torus seam).
- Vertex sources: the **vertex inset** button (`wall` by default, or the web app's `radial`), *seed pulses under the driven vertices* (with *drive the vertices*). The source note, the drive-wavelength warning and the phase lag print in the main panel.
- `C` reset view (the triangular `array` is fitted to the window: zoomed out in 2D, the 3D camera moved back, 06/10/2026) · `F` freelook · `Z`/`SHIFT+Z` zoom · arrows orbit (3D) / pan (2D) · `Y` background
- `T` draw walls (LMB paints, SHIFT+LMB erases) · `U` undo stroke · `I` walls-and-solids panel · `J` alignment · `K` ledger
- Mouse (3D): RMB or LMB drag orbits about the domain centre; MMB or CTRL+RMB drag zooms. No app key uses W/A/S/D/Q/E (freelook's keys).
- **Typed values:** click *type a value* at the top of a panel, then a slider; type digits `. - e`; Return applies (clamped to the slider's range), Escape cancels. ImGui's own Ctrl+click on a slider also lets you type a value. Text objects: Return on the selected text object.

## Verification after the round-4 review (06/10/2026, logs in `ScratchCLAUDE/Round4CLAUDE/FinalCLAUDE/`)

*Generated from scratch — Claude Opus 5.5 — 06/10/2026.* The current state. The adversarial review of round 4 (findings: `ScratchCLAUDE/ReviewCLAUDE/findings_round4_MIRROR.md`) changed the core once: a sq9 diagonal link into the Neumann ring is blocked by a non-free ring cell beside it ([NotesCLAUDE/walls_MIRROR.md](NotesCLAUDE/walls_MIRROR.md) §14). It also renamed the 3D marker images `app_tests_MIRROR.py` writes (no hyphens). Then the whole regression ran once, in order, with `final_regression_batch_MIRROR.sh`: **20 / 20 suites exit 0** (`final_status_MIRROR.txt`: suite, exit code, duration, the log's last line).

| suite | result |
|---|---|
| `crosscheck_js_MIRROR.py` CPU f64 / CUDA f64 / Vulkan f32 | 27/27 each, the web app with PATCHES `neumann-corner` and `vtx-drive-rest`. Field after the run: CPU ≤ 2.2e-14 of max\|u\| (≤ 3.2e-15 from the web app's own seed), CUDA ≤ 3.4e-14 (1.9e-10 on the deliberately unstable Co 0.75 config), Vulkan f32 ≤ 1.9e-5. The same residuals as on 05/10 |
| `identity_tests_MIRROR.py` CUDA f64 / Vulkan f32 | ALL PASS, 53 checks each |
| `solids_tests_MIRROR.py` CUDA f64 | ALL PASS, 33 checks |
| `beam_tests_MIRROR.py` CUDA f64 | 34/34 |
| `scale_tests_MIRROR.py` CUDA f64 | 16/16 |
| `walls_tests_MIRROR.py` CUDA f64 / CUDA f32 / Vulkan f32 | ALL PASS, 64 checks each |
| `tri_modes_tests_MIRROR.py` CUDA f64 / CUDA f32 / Vulkan f32 | ALL PASS, 43 checks each; parity with the 04/10 core 40/40 bit-identical |
| `defaults_tests_MIRROR.py` CUDA f64 | ALL PASS, 16 checks |
| `crosscheck_alignment_MIRROR.py --reuse` / `--unl` | 76,552 / 76,552 comparisons over 2,295 configs / 44,143 / 44,143 plate invariants over 708 configs |
| `crosscheck_ledger_MIRROR.py --jsonl ScratchCLAUDE/LedgerCLAUDE/ledger_js_full_MIRROR.jsonl` | PASS: 3,435 / 3,435 rows agree in status and in full text (50 configs, 197 ledgers); 26/26 port-only checks |
| `app_tests_MIRROR.py` CUDA f64 | 109/109. Frame at N 2049 without the headless readback: 3D mesh 1025 ~47.7 ms, mesh 513 ~15.8 ms, 2D ~9.1 ms |
| `bench_MIRROR.py --max 4097` | ms per step, CUDA f64 / CUDA f32 / Vulkan f32: square 1025 0.089 / 0.091 / 0.067, square 4097 0.859 / 0.469 / 0.686, tri 4097 0.621 / 0.363 / 0.552. Within noise of the Performance table and of the 03/10 core (findings P1) |

The review's own probes (each with its log in `ScratchCLAUDE/Round4CLAUDE/ReviewCLAUDE/`): an independent numpy operator equal to the solver's on 600 random configurations; no leak through one-cell walls (48 cases, exactly 0); the web app's path bit-identical to the 03/10 core (66 configs); incremental painting equal to the full build (279 states); source snapping never in a solid or a sealed pocket; no wall pattern found that lowers the CFL limit; the app's round-4 controls and state switching (33 checks).

## Verification of the v1.0 app step (05-06/10/2026, logs in `ScratchCLAUDE/Round4CLAUDE/AppCLAUDE/`)

The step changed the app files and added three constants to the core (`VERSION`, `VERSION_DATE`, `APP_NAME`; no logic). The batch `app_regression_batch_MIRROR.sh` ran on 05/10 and was cut off during the walls suite; `app_regression_batch2_MIRROR.sh` ran the rest on 06/10, and `app_regression_batch3_MIRROR.sh` re-ran the app suites and the shots after the last app change (the 3D fit of the triangular `array`). Exit codes: `app_batch_status_MIRROR.txt`.

| suite | result |
|---|---|
| `app_tests_MIRROR.py` CUDA f64 / Vulkan f32 | **109/109** each (95 before the step + 14 in section 11), 06/10 (`app_app_tests_cuda_fit3d_MIRROR.log`, `app_app_tests_vulkan_f32_fit3d_MIRROR.log`); 108/108 each on 05/10 before the 3D-fit check |
| `--shot` variants (`shots_r4_MIRROR.py`) | **21/21** render (`shots_r4_log_MIRROR.txt`), each looked at (`contact_r4shots_MIRROR.png`) |
| `defaults_tests_MIRROR.py` CUDA f64 | ALL PASS (05/10) |
| `crosscheck_js_MIRROR.py` CPU f64 | 27/27, patches `neumann-corner` and `vtx-drive-rest` (05/10) |
| `identity_tests_MIRROR.py` CUDA f64 | ALL PASS (05/10) |
| `walls_tests_MIRROR.py` CUDA f64 | ALL PASS, 64 checks (06/10) |
| `tri_modes_tests_MIRROR.py` CUDA f64 | ALL PASS; parity with the 04/10 core 40/40 bit-identical (06/10) |
| `solids_tests_MIRROR.py` CUDA f64 | ALL PASS, 33 checks (06/10) |

## Verification after round 4 (05/10/2026, logs in `ScratchCLAUDE/Round4CLAUDE/LogsCLAUDE/`)

After the last core change (the vertex defaults). `def_*` logs come from `ScratchCLAUDE/Round4CLAUDE/DefaultsCLAUDE/def_regression_batch_MIRROR.sh`, with the exit codes in `def_batch_status_MIRROR.txt`. `tri_*` logs come from the triangular-modes batch earlier the same day, for suites the vertex defaults cannot reach.

| suite | result |
|---|---|
| `crosscheck_js_MIRROR.py` CPU f64 / CUDA f64 / Vulkan f32 | 27/27 each, the web app with PATCHES `neumann-corner` and `vtx-drive-rest`. CPU field ≤ 2.2e-14 of max\|u\|; from the web app's own seed the stepping agrees to ≤ 3.2e-15 (exactly for undriven sources) |
| … with `ZRL_JS_NOPATCH=1` (raw web app, CPU) | 24/27. The differing configs are exactly the two Neumann-corner ones and `sq5_n8_vtxdrive_saw` (the rest start; stepping from the same seed agrees to 1.5e-15) |
| `defaults_tests_MIRROR.py` CUDA f64 / CUDA f32 / Vulkan f32 | ALL PASS. Wall inset clip ≤ 0.0053%, the radial clip matches the diagnosis; the dent and the rest start; the source note = the ledger's clearance on 14 configs; the phase lag within 0–3% of the lattice's measured lag while it is under ~0.3σ, ≤ 13% beyond |
| `identity_tests_MIRROR.py` CUDA f64 / Vulkan f32 | ALL PASS (53 checks each) |
| `walls_tests_MIRROR.py` CUDA f64 (`def_`) / CUDA f32, Vulkan f32 (`tri_`) | ALL PASS (64 checks each) |
| `tri_modes_tests_MIRROR.py` CUDA f64, Vulkan f32 (`def_`) / CUDA f32 (`tri_`) | ALL PASS. Parity with the 04/10 core 40/40 bit-identical, with the vertex configs on the 04/10 inset |
| `solids_tests_MIRROR.py` CUDA | ALL PASS (33 checks) |
| `beam_tests_MIRROR.py` CUDA | 34/34 |
| `scale_tests_MIRROR.py` CUDA | 16/16 (cells mode bit-identical to the pre-scale references, the vertex config on the web app's defaults) |
| `crosscheck_ledger_MIRROR.py` (saved web-app dump) | PASS: 0 status and 0 text mismatches over 3,435 rows (50 configs, 197 ledgers); 26/26 port-only checks |
| … with a fresh web-app dump (both patches) | PASS, the same counts |
| `crosscheck_alignment_MIRROR.py --reuse` / `--unl` (`tri_`) | 76,552 / 76,552 comparisons over 2,295 configs / 44,143 / 44,143 plate invariants |
| `app_tests_MIRROR.py` CUDA | 95/95 |
| `--shot` | 4/4 vertex-source shots (`def_shot_*`), 4/4 triangular-mode shots (`tri_shot_*`) |

## Verification (01/10/2026, logs in `ScratchCLAUDE/FinalCLAUDE/`)

| suite | result |
|---|---|
| `crosscheck_js_MIRROR.py` CPU f64 | 27/27; field ≤ 2.2e-14 of max\|u\|; **bit-identical** from the web app's own seed; energy history ≤ 2.4e-14 |
| … CUDA f64 / Vulkan f32 | 27/27 / 27/27 (f32 ≤ 2.2e-5) |
| … with `ZRL_JS_NOPATCH=1` (raw web app) | 25/27: exactly the two Neumann-corner configs differ (3.2e-2 of max\|u\| for sq9), as documented |
| `identity_tests_MIRROR.py` CUDA f64 / Vulkan f32 | all pass (41 checks incl. light cone, CFL flip, damped CFL, symmetry incl. Neumann, energy, drum dispersion, conformance, time reflection, source snapping) |
| `solids_tests_MIRROR.py` CUDA | all pass (sealing exact on sq5/sq9/tri, rasterisation, incremental = full, painting latency) |
| `crosscheck_alignment_MIRROR.py` | 27,882 / 27,882 comparisons |
| `crosscheck_ledger_MIRROR.py` | 0 status and 0 text mismatches over 3,435 rows; 15/15 port-only checks |
| `app_tests_MIRROR.py` CUDA | 71/71 |
| `--shot` variants | 38/38 render |

The round-1 tables (29/09: per-backend residuals, the identity table, the porting faults) still hold; they are in the notes. Two porting faults were found then and fixed: Vulkan misread the last cell of an 8-bit mask (the mask is 32-bit on Vulkan), and the f32 9-point stencil was not self-consistent (written so it cancels exactly in f32).

## Directional sources: beam and packet (01/10/2026)

Two source types with no web-app counterpart, after `vtx` in the **O** cycle. Details and all the measurements are in [NotesCLAUDE/beam_MIRROR.md](NotesCLAUDE/beam_MIRROR.md).

- **beam:** a continuous Gaussian beam (waist w) from a driven aperture across the beam direction. Every cell is phased by the lattice's own discrete dispersion, so the beam steers to any angle. It is **one-sided** by default: a second aperture behind the first emits the inverted, delayed signal and cancels the backward lobe (~29–30 dB at w = 8, λ = 12; 32–38 dB at w = 16). The aperture is a **soft** source (added, not imposed), so it is transparent to returning waves and keeps the CFL limit. Angle, width and frequency apply live.
- **packet:** one wave packet (width w, length L, wavelength λ) launched one way. The previous time level is built plane wave by plane wave with the discrete ω(K), so ≤ 1e-5 of its energy goes backward; it moves at the predicted discrete group velocity (to 0.05%).
- **Aim the energy** (default): on sq5 the phase fronts and the energy direction differ off the axes (up to 2.6° at 6 cells per wavelength). The source aims the energy, and the measured direction is within 0.41° (λ = 6) and ≤ 0.21° (λ ≥ 8) of the request; sq9 and tri are within 0.03°.
- Aperture cells outside the domain or inside solids are skipped (clipped count in the panel). Pinned solids in the path reflect the beam. Cost: like `cont` (+~0.07 ms/step at N = 1025).
- Panel: beam angle, beam width, one-sided, aim, packet length and wavelength; the emitter (aperture and arrow) is drawn in 2D and 3D (*draw the emitter*).

## Scale mode: cells or fractions of the domain (02/10/2026)

This is the parked "physical scale" idea. A **scale** button in the main panel switches between two modes:
- `cells`, the default, which is unchanged and bit-identical;
- `domain`, where pulse sigma, the drive wavelength (freq = Co / λ), the beam width and the packet length and wavelength are fractions of the domain's inradius r_in.

A larger N then shows the same picture with less grid error. Switching converts the current values, so nothing jumps. Domain mode also adds:
- a time readout in crossings (r_in / c);
- a playback speed in crossings per second, with steps per frame following N;
- amber warnings for sizes under 2 cells of width or 8 cells per wavelength.

The *auto height* checkbox (3D height / recent peak) works in both modes. CLI: `--scale domain`, `--set sigma_f=0.05 lam_f=0.1`, `--cps`, `--auto-height`.

`python scale_tests_MIRROR.py` (16/16):
- **Convergence:** the field at fixed fractional positions converges at second order on sq5, sq9 and tri (differences fall 4.00–4.10× per doubling over N = 257 → 2049).
- **Amplitude:** the peak at r = 0.5 r_in is independent of N (spread 0.4–0.8%, last doubling ≤ 0.04%). Cells mode falls ~1/√2 per doubling.

The test must compare at t = (n + ½)Δt, because a run started at rest lags half a step. Details: [NotesCLAUDE/scale_MIRROR.md](NotesCLAUDE/scale_MIRROR.md).

## Walls and triangular-lattice modes (04–05/10/2026)

Details, measurements and the UI hand-off: [NotesCLAUDE/walls_MIRROR.md](NotesCLAUDE/walls_MIRROR.md) (sections 1–12 the walls, section 13 the triangular modes, section 11 the UI). The app shows all of it since v1.0 (05/10/2026): wall-type buttons in the walls panel, rigid solids in cyan (striped in 2D), a cyan line along a rigid edge or ring, a dashed violet line along a torus seam, and the effective boundary in the main panel ([NotesCLAUDE/app_MIRROR.md](NotesCLAUDE/app_MIRROR.md), *Round 4 / v1.0*).

- **Rigid walls:** every object, stroke and the plate's lettering is pinned (u = 0, echo inverted) or rigid (link removal, echo upright). `Params.edge = 'rigid'` makes a masked domain's edge rigid. **Corner blocking** seals one-cell diagonal walls on every lattice (the pinned ones leaked 2.3% on sq9, 17% on tri). Exact energy with both types; no time-step penalty. *Review fix (06/10/2026):* on the sq9 Neumann ring, a solid painted ON the ring row (or a clipped shape with a rigid edge) made the operator asymmetric and the run blew up after ~10⁵ steps; a sq9 diagonal link into the ring is now blocked by a non-free ring cell beside it ([NotesCLAUDE/walls_MIRROR.md](NotesCLAUDE/walls_MIRROR.md) §14).
- **Domain shape `array`** (port-only, last in the G cycle): the whole lattice array. On the square lattice it is exactly the full square. On the triangular lattice it is the 60° rhombus, and the **boundary control acts there**:
  - `dirichlet`: the clamped ring;
  - `neumann`: a **rigid ring** (link removal);
  - `periodic`: the **rhombic torus** (the six neighbours mod N);
  - `mur`: not available yet. The rule is unstable on the triangular lattice above Co 0.5, so it falls back to the clamped ring and `Membrane.bc_info()` says so.
- **Every web-app triangular shape** keeps the clamped ring with any `bc`, bit for bit, as in the web app.
- **The recorder** also records the triangular lattice's third link direction (`m.hist['Etri']`) and sums the whole rhombic torus. The ledger grades the triangular energy on it (FIX `tri-energy`).
- **Cost:** the `array` updates every cell, so it costs ×1.4–1.6 a radius-1 hexagon at N = 8193; the three ring modes are within −12% to +10% of each other (walls_MIRROR.md §13.7).

## Vertex sources: the port's defaults and the source readouts (05/10/2026)

The user reported choppy waves with vertex sources. The measured diagnosis is [NotesCLAUDE/vertex_choppiness_MIRROR.md](NotesCLAUDE/vertex_choppiness_MIRROR.md); its section 7 has what changed and the UI hand-off.
- **Vertex inset `wall` is the default.** The web app's default, `radial`, leaves each vertex pulse 0.7–1.7σ from the walls. The domain edge then cuts 1.2–27% of it at t = 0, and 4.5–49% of the energy becomes grid-scale chop. The `wall` inset (3σ from both walls, along the bisector) cuts at most 0.012% (the diagnosis' `s1_reproduce_MIRROR.py`, 51 configurations) and leaves 2.3–3.1% chop, the level of a centred pulse (its `s1b_invariant_all_MIRROR.py`; both N 513, σ 3; `defaults_tests_MIRROR.py` test 1 reproduces the clip column). The *vertex inset* button still offers `radial`.
- **Driven vertices start from rest.** With *drive the vertices* on, the web app also seeds a bump under each vertex, and the hard drive dents its centre to drive(0) = 0 at step 1 (a one-cell dent, 0.90 deep: `defaults_tests_MIRROR.py` 2b). The port now starts them from rest, like `cont`; `Params.vtx_drive_bump = True` restores the web app. At λ = 16 cells this lowers the chop share at 8 crossings from 1.5–3.3% to 0.4–1.3% (`s4b_driven_vertices_MIRROR.py`; the port's default run equals that variant bit for bit, test 2c).
- **New core readouts** (in the main panel since v1.0, 05/10/2026):
  - `m.source_note()`: the web app's source note — the clearance in cells and σ, the clip fraction (amber above 1%, with "the cut is a sharp edge the grid cannot resolve; try vertex inset: wall"), the capped corners. Cached per reseed and mask change.
  - `m.drive_warnings()`: in cells mode, a hard drive (`cont`, `slit`, driven vertices) under 8 cells per wavelength. The default freq 0.12 at Co 0.5 is a 4.2-cell wavelength, and 95–99.9% of the energy then sits in grid-scale waves (`s4b_driven_vertices_MIRROR.py`); freq ≤ 0.031 gives ≥ 16 cells. *09/10/2026: the default is now 0.03125 (16 cells), and driven vertices are soft by default; Deviations 9 and 10.*
  - `m.phase_lag()`: how far the grid's fronts trail the exact ones, in cells per domain crossing, from the discrete dispersion. It is within 0–3% of the lag measured on the lattice while that lag is under ~0.3σ (`defaults_tests_MIRROR.py` test 5). At the app's defaults (sq5, N 513, σ 3, Co 0.5; `ScratchCLAUDE/Round4CLAUDE/DefaultsCLAUDE/probe_defaults_table_MIRROR.py`) it reads 0.60–1.79 cells per crossing, so the fronts trail by one σ after ~1.7 crossings. Domain mode at N 2049 (σ 12 cells) gives 0.15–0.44 (27 crossings), and Co near the limit cuts it 2.7× (sq9 at 0.85) to 4.5× (tri at 0.8), as the diagnosis measured.
- What remains after these is the slow phase error, which shrinks ~4× per doubling of N in domain mode, and physics: corner diffraction and many reflections stay in a converged run.

## Absorbing boundaries (core, 09/10/2026)

*Generated from scratch — Claude Opus 5.5 — 10/10/2026.* The three boundaries the user chose, in the core; the panels are not wired yet. Details, measurements, deviations and the UI hand-off: [NotesCLAUDE/absorbing_MIRROR.md](NotesCLAUDE/absorbing_MIRROR.md); tests `absorbing_tests_MIRROR.py`.
- **`bc = 'mur2'`** (the full square, sq5 / sq9): Mur 2nd order with diagonal Mur-1 corners; 2D box with a sigma-3 pulse 8.8e-3 centre / 0.10 corner (Mur1: 2.2e-2 / 0.19).
- **`bc = 'cpml'`** (the full square, sq5 / sq9; `cpml_level` 'std' 10 cells / 'high' 20 cells): 5.3e-5 / 1.8e-4 and 2.5e-6 / 1.8e-6, also for wide pulses.
- **`bc = 'sponge'`** (the full square, the triangular `array`) and **`edge = 'open'`** (every masked shape, both lattices): a distance-ramp sponge outside the drawn edge, ~1.4-1.6e-2.
- The drawn outline is the open edge; the layer lies outside it (the full array's open domain is the array minus the frame; a masked shape at a large radius shrinks to make room). The recorder, sources and the ledger describe the open domain.
- Not on the triangular lattice: Mur, Mur2, CPML (`bc_info()` says why). Every configuration without an absorbing boundary is bit-identical to the core before.

## Deviations from the web app (all deliberate, all documented)

1. **Neumann corners.** The web app's ring loop leaves corner (0,0) with a value from an older time level (loop order); with the 9-point stencil that changed the field by 8.6% of its peak near that corner. The port sets every corner to its diagonal interior neighbour of the same step: 8-fold symmetry is exact again (sq9 5.6e-2 → 8.5e-15) and the discrete energy is conserved (drift 5.8e-3 → 1e-14). The harness applies the same rule to the web app (`PATCHES`), so the cross-check still compares whole fields.
2. **Damped stability limit.** With the damping form −d(u − u_prev) the true limit is **C_max(d) = C_max(0)·√(1 − d/2)** (0.7036 instead of 0.7071 on sq5 at d = 0.02). The web app reported runs in that gap as stable while they blew up. `co_limit()` now includes damping, the panel says *(damped)*, and the ledger's CFL row grades against it. `co_limit(damped=False)` is the web app's value.
3. **Source snapping with solids.** A source requested inside a solid goes to the nearest free cell of the domain's main open region (never into a sealed cavity such as a letter counter); the panel shows an amber *source moved out of a solid* note. Without solids the web app's rule is unchanged.
4. **Ledger fixes.** A few web-app ledger rows were wrong; the port fixes them, with a *web-app logic (no fixes)* checkbox for comparison. Among them, three false FAILs (masked n6 + Mur, tri n6 + Mur, n12 + Mur linearity, all at 1200 steps) and the damped CFL row. Details: [NotesCLAUDE/ledger_MIRROR.md](NotesCLAUDE/ledger_MIRROR.md).
   - Since 05/10/2026, also `tri-energy`. On the triangular lattice the energy row grades the triangular operator's own energy. The web app's recorded E uses only the square lattice's two difference directions, which is not conserved there. Measured: 4.0e-2 → 1.1e-3 on the rigid rhombic ring, 1.3e-1 → 3.3e-3 for a packet in the hexagon ([NotesCLAUDE/walls_MIRROR.md](NotesCLAUDE/walls_MIRROR.md) §13.6).
5. **Additions with no web-app counterpart:** solids, painted walls (pinned or rigid), corner blocking, the rigid edge, the Unlicensed plate, the `array` shape and the triangular rigid ring and rhombic torus, the beam and packet sources, the drive warning and the phase-lag readout, the Co-change *new time step* toggle (off by default, so the time-reflection effect stays), typed slider values, the orbit camera.
6. **Vertex inset default** (05/10/2026). `Params.vtx_inset` defaults to `'wall'`; the web app's default is `'radial'`. It is a setting, not a code change. Every comparison with the web app sets `vtx_inset='radial'` explicitly: `crosscheck_js`, `crosscheck_ledger`, `crosscheck_alignment`, the scale-test references and the tri-modes parity test.
7. **Driven vertices start from rest** (05/10/2026). With `vtx_drive`, the port seeds no bumps under the vertices (`Params.vtx_drive_bump = False`); the web app seeds them and its hard drive dents their centres at step 1. The harness applies the same rule to the web app as **PATCH `vtx-drive-rest`**. The patch reads `S.vtxDriveBump`, which only the harness sets, so `vtx_drive_bump=True` compares against the web app's own line. With `ZRL_JS_NOPATCH=1`, `crosscheck_js` gives 24/27: the two Neumann-corner configurations and `sq5_n8_vtxdrive_saw` differ.

8. **neumann-clip** (09/10/2026, the user's decision). With the sq9 stencil, the Neumann bc and a shape clipped by the array (a circle at radius 1.3), the web app's ring is asymmetric where the shape leaves it (an exterior ring cell holds 0 next to a ghost), and the run grows without bound (max|u| 6.7e21 after 200,000 steps). The port blocks the sq9 diagonal into such a ghost like a pinned ring solid (the round-4 rule C1 extended): the operator is exactly symmetric, the energy is conserved, the reproducer stays bounded. These configurations run the walls kernel; every other configuration is bit-identical to v1.0. The harness applies the same rule to the web app as **PATCH `neumann-clip`**; `crosscheck_js` config `sq9_circle13_neu_clip` agrees to 1.5e-13 (CPU; the walls path's self-consistent 9-point form, not bit for bit). [NotesCLAUDE/v1_decisions_MIRROR.md](NotesCLAUDE/v1_decisions_MIRROR.md) section 1, walls_MIRROR.md §14.2. With `ZRL_JS_NOPATCH=1`, `crosscheck_js` now gives 24/28: the two Neumann-corner configurations, `sq5_n8_vtxdrive_saw` and `sq9_circle13_neu_clip` differ.
9. **Default drive frequency** (09/10/2026, the user's decision). `Params.freq = 0.03125`, a 16-cell wavelength at the default Co 0.5 (a plain value, not coupled to Co); the web app's is 0.12 (4.2 cells), `mc.WEBAPP_FREQ`. A setting, not a code change: every comparison with the web app sets `freq = 0.12` explicitly (`crosscheck_js`, `crosscheck_ledger` and its saved dump, `crosscheck_alignment`, the scale-test references, the parity tests).
10. **Soft driven vertices** (09/10/2026, the user's decision: "soft as default, leave a hard toggle"). With *drive the vertices*, `Params.vtx_soft = True` adds the drive at the vertex cells (the beam's soft path), so they are transparent to returning waves; each harmonic is scaled by `soft_gain()` = Co² / |g00|, the lattice Green's function at the cell, so a lone vertex radiates the hard vertex's wave (measured soft / hard 1.0001–1.0003 in free space, 0.79–0.87 at the default square, where the hard cell also re-radiates its corner's echoes). `vtx_soft = False` is the web app's hard drive, bit-identical to v1.0; the comparisons set it (no harness patch). `vtx_drive_bump` still seeds the bumps; the soft drive adds to them (no dent).

## Performance

Measured on the user's machine (i9-9900K, RTX 5070 Ti 16 GB). Re-run 01/10/2026 with `python bench_MIRROR.py --max 4097 --js`: within noise of the 29/09 table below. Re-run 06/10/2026 after the round-4 review (`ScratchCLAUDE/Round4CLAUDE/FinalCLAUDE/final_bench_MIRROR.log`, without `--js`): within noise again, e.g. square 4097 0.859 / 0.469 / 0.686 ms per step. Milliseconds per step (nanoseconds per cell).

| lattice | N | CUDA f64 | CUDA f32 | Vulkan f32 | web app (JS) | best GPU vs JS |
|---|---|---|---|---|---|---|
| square | 1025 | 0.086 (0.082) | 0.084 (0.080) | 0.072 (0.068) | 26.5 (25.2) | 370× |
| square | 2049 | 0.220 (0.052) | 0.103 (0.025) | 0.159 (0.038) | 107 (25.4) | 1036× |
| square | 4097 | 0.911 (0.054) | 0.454 (0.027) | 0.672 (0.040) | — | — |
| square | 8193 (29/09) | 3.59 (0.053) | 1.85 (0.028) | 2.43 (0.036) | — | — |
| tri | 2049 | 0.160 (0.038) | 0.104 (0.025) | 0.134 (0.032) | 79.8 (19.0) | 765× |
| tri | 8193 (29/09) | 2.48 (0.037) | 1.30 (0.019) | 1.83 (0.027) | — | — |

- Up to N ≈ 1025 the GPU is idle (launch latency floor ~0.07–0.12 ms/step); from N ≈ 2049 it is bandwidth-bound.
- The recorder adds ~0.05 ms/step up to N = 1025, 0.48 ms at 4097, 1.95 ms at 8193 (CUDA f64, square lattice). On the triangular lattice it also sums the third link direction (05/10/2026): the recorder costs +2.8 ms at 8193 for a radius-1 hexagon (CUDA f64), +1.4 (CUDA f32), +1.2 (Vulkan) (`tri_modes_bench_MIRROR.py`). The triangular `array` modes: [NotesCLAUDE/walls_MIRROR.md](NotesCLAUDE/walls_MIRROR.md) §13.7. Painting a wall segment takes ~3–5 ms even at N = 8193; a long fast diagonal drag 30–70 ms.
- Memory: 33 bytes/cell at f64 (2.2 GB at N = 8193), 17 at f32. Out of GPU memory, the app stops with an error; the PC is not at risk.

## Time reflection (the Co-change effect)

Changing Co mid-run is a **temporal boundary**: the whole medium changes speed at one instant, the wavelength is frozen, so the frequency jumps and part of every wave reverses. Backward amplitude B = (Co₂ − Co₁)/(2Co₂) for long waves; measured 0.1432 vs 0.1432 predicted (1D, 0.5 → 0.7), 2D ring ratio 0.248 vs ~0.28 expected. The *Co change = new time step* toggle re-expresses the previous level instead (`u_prev ← u − r(u − u_prev) + ((r² − r)/2)·Co₁²·L(u)`, r = Co₂/Co₁), which removes 98–99.9% of it. Off by default.

## Research notes (fact-checked 01/10/2026; the third added 04/10/2026)

- **[Rigid vs pinned walls](NotesCLAUDE/rigid_vs_pinned_walls_MIRROR.md).** Pinned (u = 0) reflects with a sign flip; rigid (zero normal slope) without — rigid is the right model for sound hitting a solid. Recommended rule: rigid by **link removal** (drop links into the wall) plus **corner blocking** (cut diagonal links that pass between solid cells): exact energy conservation, no time-step penalty, leak-free. Corner blocking also fixes a real leak of *pinned* one-cell diagonal walls on sq9 (2.3%) and tri (17%). A single pinned cell scatters 300–450× more than a rigid one at λ = 32.
- **[Vertex sources: why the waves turn choppy](NotesCLAUDE/vertex_choppiness_MIRROR.md)** (04/10/2026, measured). The vertex chop came from pulses cut by the wall at t = 0 (the radial inset), not from grid alignment; walls cannot create short waves in a closed lossless drum, only redirect them. A slower phase error affects every source and shrinks ~4× per doubling of N in domain mode. Section 7: what the port changed (05/10/2026).
- **[Absorbing boundaries beyond Mur](NotesCLAUDE/absorbing_boundaries_MIRROR.md).** Measured worst-case reflection: Mur1 ~6%, Mur2 with diagonal corners ~1.1%, Higdon2 ~0.5%, 40-cell sponge 1.7% / 5.6% / 16% for pulse width 3 / 6 / 12, **CPML 10 cells ≤ 0.03%** (stable for 100,000 steps up to Co 0.7071). Recommended: CPML for the rectangle, a distance-ramp sponge for flavor shapes and the triangular lattice.

## Backlog and plan

Done: the port; alignment panel and ledger; static solids, painted walls, the Unlicensed plate; orbit camera, look controls, typed values; the beam and packet sources; scale mode; rigid walls and corner blocking (04/10/2026); the triangular-lattice boundary modes and the vertex-source defaults (05/10/2026); their panel wiring and v1.0 (05/10/2026); the user's v1.0 decisions in the core (09/10/2026: the sq9 Neumann clipped-shape fix, freq 0.03125, soft driven vertices; the hard / soft toggle still to wire in the panel: [NotesCLAUDE/v1_decisions_MIRROR.md](NotesCLAUDE/v1_decisions_MIRROR.md) section 5).

~~**Still to wire in the panels**~~ (done 05/10/2026, v1.0): wall types, the rigid edge, the `array` shape and the bc options; the source note, the drive warning and the phase-lag readout ([NotesCLAUDE/app_MIRROR.md](NotesCLAUDE/app_MIRROR.md), section *Round 4 / v1.0*).

Next, in order:
1. ~~Beam source~~ (done 01/10/2026).
2. ~~**Rigid walls**~~ (done 04/10/2026) and ~~the triangular-lattice modes~~ (done 05/10/2026).
3. **Absorbing boundaries:** only these three (the user's choice, 01/10/2026): **Mur 2nd order** (rectangle), **CPML** (rectangle), **sponge** (flavor shapes, tri; it also gives the triangular `array` the absorbing ring that Mur cannot). *In the core since 10/10/2026 ([NotesCLAUDE/absorbing_MIRROR.md](NotesCLAUDE/absorbing_MIRROR.md)); the panel wiring is next (its section 12).*
4. **Moving objects** as their own item (they need a coupling term to make waves; a moving window or a convected-wave "wind tunnel" needs item 3).
5. Prism-like / programmable media; then the lattice extension (separate app).

Ideas parked: ~~an N-independent "physical scale" mode~~ (done 02/10/2026: scale mode); light-cone flavors (dispersion ω = c‖k‖_p, giving square or diamond fronts); a lattice-Boltzmann "wind tunnel" app on the same solids/masks for real airfoil flow (lift, drag); an exact image-source check of the triangle's kaleidoscope patterns; 3D port and the Echo sim (06/10/2026: a separate chat and folder in the Taichi wave family, e.g. `FormConstantsCLAUDE/TaichiWave3DCLAUDE`, after v1.0 and ideally after the absorbing boundaries; BlastBox would use it for room impulse responses rather than host it); **a Taichi ray-marcher** for 3D fields (06/10/2026, after v1.0: one kernel per pixel writing into the canvas; isosurface mode with gradient shading and an emission-absorption volume mode; empty-space skipping on a coarse max|u| grid, a jittered start, clip and slice planes, solids from the mask or a distance field; to be measured against a marching-cubes mesh at 512³); sound-file sources; ZRL canon ASKs (β-normalisation fork, open).

Small open items: undoing a long stroke re-rasterises its whole box (seconds at N = 8193); the plate's corner radius measures 0.047·H on the cover vs 0.065·H used; Bahnschrift's A is wider than the cover's; ~~the default drive frequency (0.12, a 4.2-cell wavelength at Co 0.5) is unchanged and now only warned about~~ (0.03125 since 09/10/2026); the full `square`'s seed is not clamped on its ring until step 1 (inherited, 0.7% of the energy with the radial inset); the open items of the walls (walls_MIRROR.md §10, §13.9) and of the vertex sources (vertex_choppiness_MIRROR.md §7.5); ~~inherited from the web app, the sq9 stencil with the Neumann bc and a shape clipped by the array (a circle at radius 1.3) is unstable, max|u| ~1e21 after 200,000 steps~~ (fixed in the port 09/10/2026, Deviation 8).
