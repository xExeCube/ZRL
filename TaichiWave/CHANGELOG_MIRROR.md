# Changelog: the ZRL wave membrane (Taichi port)

*Generated from scratch -- Claude Opus 5.5 -- 05/10/2026. Compiled from README_MIRROR.md, the area notes in NotesCLAUDE/ and their progress logs; every number names where it was measured. Dates are DD/MM/YYYY.*

The version constant is `VERSION` in `membrane_core_MIRROR.py`; the app shows it in its window title and main panel, and `python membrane_app_MIRROR.py --version` prints it. Rounds 1-3 had no version number; v1.0 tags the end of round 4.

---

## Unreleased (v1.1)

**The user's v1.0 decisions, in the core** (09/10/2026; [NotesCLAUDE/v1_decisions_MIRROR.md](NotesCLAUDE/v1_decisions_MIRROR.md); tests `decisions_tests_MIRROR.py`; logs `ScratchCLAUDE/Round5CLAUDE/LogsCLAUDE/dec_*_MIRROR.log`)
- **sq9 Neumann ring with a clipped shape, fixed in the port** (a documented difference). With the 9-point stencil, the Neumann bc and a shape clipped by the array (a circle at radius 1.3), the web app's ring was asymmetric where the shape leaves it, and the run grew to max|u| 6.7e21 in 200,000 steps. The exterior of a pinned edge now blocks the sq9 diagonal into a ghost ring cell like a pinned ring solid (the round-4 review rule C1, extended). Every enumerated configuration has an exactly symmetric operator, the step map's spectral radius is 1, the exact energy holds to ~1e-15 over 20,000 steps, and the review's reproducer stays bounded over 200,000 steps. Every unaffected configuration is bit-identical to v1.0. Harness PATCH `neumann-clip`; new `crosscheck_js` config `sq9_circle13_neu_clip` (28 configs). `walls_info()['neu_clip']`.
- **Default drive frequency 0.03125:** a 16-cell wavelength at the default Co 0.5 (was 0.12, 4.2 cells; a plain value, not coupled to Co). `mc.WEBAPP_FREQ = 0.12`, which every comparison with the web app sets explicitly.
- **Driven vertices are soft by default, with a hard toggle** (`Params.vtx_soft`, `mc.VTX_DRIVES`). Soft: the drive is added at the vertex cells, so they are transparent to returning waves. Each harmonic is scaled by `soft_gain()` = Co² / |g00|, the lattice Green's function at the cell (`lattice_g00()`, new), so a lone vertex radiates the hard vertex's wave: soft / hard 1.0001–1.0003 in free space, 0.79–0.87 at the default square. Hard (`vtx_soft=False`) is the web app's imposed drive, bit-identical to v1.0. `vtx_drive_bump` with soft vertices seeds the bumps and adds the drive (no dent). `source_note()` says "driven soft / driven hard" (key `drive`).
- Still to wire in the app (the app agent): the hard / soft toggle and its labels (v1_decisions_MIRROR.md section 5).

**Absorbing boundaries, in the core** (09-10/10/2026, the user's choice of 01/10 and 09/10; [NotesCLAUDE/absorbing_MIRROR.md](NotesCLAUDE/absorbing_MIRROR.md); tests `absorbing_tests_MIRROR.py`; logs `ScratchCLAUDE/Round5CLAUDE/LogsCLAUDE/abs_*_MIRROR.log`)
- **Mur 2nd order** (`bc = 'mur2'`, the full square, sq5 and sq9): Engquist-Majda A2 on the ring with diagonal Mur-1 corners, one ring pass. 2D box, sigma-3 pulse: 8.8e-3 centre / 0.10 corner (Mur1 2.2e-2 / 0.19). Its first step after a seed, `set_levels`, `set_co`, a mask change or another bc is a Mur1 step: without it Mur2 kept data on the ring rows as a static offset (-0.099 for a pulse centred 8 cells from the ring; after: 1.7e-9).
- **CPML frame** (`bc = 'cpml'`, the full square, sq5 and sq9; `cpml_level` 'std' = 10 cells, 'high' = 20 cells): single pass, frame-packed memory, the sq9 operator stretched exactly. Box: 5.3e-5 / 1.8e-4 (std), 2.5e-6 / 1.8e-6 (high); both keep their level for pulses up to sigma 24 cells. 'high' uses alpha 0, not the research's 0.05 (alpha 0.05: 3.1e-2 at sigma 24).
- **Sponge** (`bc = 'sponge'` on the full square and the triangular `array`; **`edge = 'open'`** on every masked shape, both lattices): a distance-ramp layer OUTSIDE the drawn edge (the full array: the open domain is the array minus the frame; a masked shape at a large radius shrinks to make room), the centred damping form, the damping level in the mask code. 40 cells (20 under a half-size of 160), d_max 0.2 (0.35 at 20; 8 / L above 40). Box 1.4e-2 / 3.8e-2; the open edge on the sq circle / tri hexagon / tri circle at R 100: 1.6e-2. In scale mode 'domain' the width follows the scale and keeps 1.4e-2 for sigma 3-24 cells.
- Not on the triangular lattice: Mur, Mur2, CPML (`bc_info()` gives the reason). Periodic is exclusive with an open edge.
- Every configuration without an absorbing boundary is bit-identical to the core before (54 runs). Stable over 20,000 steps of noise at Co 0.5 / 0.7 (sq9 to 0.866, tri 0.8), f64 and f32; with the sponge the exact energy never rises.
- The layer is part of the stencil but not of the open domain: the recorder (energy, peak, probes), sources, vertices and the beam aperture use the open domain; a source requested in the layer snaps out.
- New core API for the UI: `edge_info()`, `absorbing()`, `abs_layer()`, `layer_mask()`, `open_mask()`, `open_domain_np()`, `sponge_width()`, `sponge_dmax()`; `bc_info()` gains `absorbing` and `layer`.
- **Ledger:** the new bcs and the open edge are handled (open-domain counts, n/a with the reason where a row assumes a closed lossless domain); a new row "energy inside the open domain does not grow" for the port's absorbing boundaries; 14 new port-only checks in `crosscheck_ledger_MIRROR.py`. The alignment panel analyses the open domain.
- `app_tests_MIRROR.py`: the B cycle now includes mur2, cpml, sponge (and sponge on the triangular array).
- Still to wire in the app (the app agent): the edge button's third state, the CPML level toggle, the sponge width / d_max, the layer tint, the info lines (absorbing_MIRROR.md section 12).

---

## v1.0 -- round 4: walls, triangular modes, vertex-source fixes, the app wiring (04-05/10/2026)

**Rigid walls and corner blocking** (04/10; [NotesCLAUDE/walls_MIRROR.md](NotesCLAUDE/walls_MIRROR.md) sections 1-12)
- Every object, painted stroke and the plate's lettering is **pinned** (u = 0, the echo comes back inverted) or **rigid** (link removal, the echo comes back upright). The default stays pinned, as in the web app.
- `Params.edge = 'rigid'` makes a masked domain's edge rigid.
- **Corner blocking** seals one-cell diagonal walls on every lattice. The pinned ones leaked 2.3% on sq9 and 17% on tri; every lattice now measures 0.0.
- Exact discrete energy with both types, no time-step penalty. With no rigid cell and no blocked link the solver runs the 03/10 kernel bit for bit.
- Scene files: format version 2 stores the wall types; version-1 files load as pinned.
- `walls_tests_MIRROR.py`: 64 checks, ALL PASS (CUDA f64, CUDA f32, Vulkan f32).

**Triangular-lattice boundary modes** (05/10; walls_MIRROR.md section 13)
- New domain shape **`array`** (last in the G cycle): the whole lattice array. On the square lattice it is exactly the full square; on the triangular lattice it is the 60-degree rhombus.
- On the triangular `array` the boundary control acts: `dirichlet` = the clamped ring, `neumann` = a **rigid ring**, `periodic` = the **rhombic torus**. `mur` is not available there yet (unstable above Co 0.5); it falls back to the clamped ring and `Membrane.bc_info()` says so.
- Every web-app triangular shape keeps its clamped ring with any bc: 40/40 configurations bit-identical to the 04/10 core (`tri_modes_tests_MIRROR.py` test 8).
- The recorder also records the triangular lattice's third link direction (`hist['Etri']`). The ledger grades the triangular energy on it (FIX `tri-energy`): 4.0e-2 -> 1.1e-3 drift on the rigid rhombic ring (`tri_energy_row_MIRROR.py`).

**Vertex sources: the defaults and the source readouts** (05/10; [NotesCLAUDE/vertex_choppiness_MIRROR.md](NotesCLAUDE/vertex_choppiness_MIRROR.md))
- The diagnosis (04/10): the vertex chop came from pulses the wall cut at t = 0 (the web app's radial inset), not from grid alignment.
- **`vtx_inset = 'wall'` is the default** (3 sigma from both walls): it cuts at most 0.012% of the pulse (`s1_reproduce_MIRROR.py`, N 513, sigma 3, 51 configurations), against 1.2-27% for `radial` (vertex_choppiness_MIRROR.md, summary item 2).
- **Driven vertices start from rest** (`vtx_drive_bump = False`); the web app's seeded bump got a one-cell dent 0.90 deep at step 1 (`defaults_tests_MIRROR.py` 2b). The cross-check harness applies the same rule to the web app as PATCH `vtx-drive-rest`.
- New core readouts: `source_note()` (clearance, clip fraction, capped corners), `drive_warnings()` (a hard drive under 8 cells per wavelength), `phase_lag()` (the fronts' lag per domain crossing; within 0-3% of the measured lag while that lag is under ~0.3 sigma).

**The app wiring and v1.0** (05/10; [NotesCLAUDE/app_MIRROR.md](NotesCLAUDE/app_MIRROR.md), section *Round 4 / v1.0*)
- **Boundary (B)** cycles only the bcs that act on the current lattice and shape, and the panel prints the effective bc and the ring in words: Mur on the triangular array reads "not on tri yet (absorbing-boundary round)"; a triangular web-app shape says the bc does not apply there (use the edge control).
- **Edge** control (pinned / rigid) for masked shapes, in the main panel and the walls panel; it keeps the field. On `array` and the full square it is greyed text ("the boundary (B) rules").
- `array`: the radius and rotation sliders turn into greyed text; in 2D the whole triangular rhombus is zoomed to fit the strip the panels leave. In 3D (added 06/10) the orbit camera moves back until the rhombus fills the window as the square does at the default camera (`Viewer.fit_3d()`).
- **Wall types in the walls panel:** the brush's type for new strokes, each object's type, the plate lettering's type, a one-line explainer, a legend, the pinned and rigid cell counts and whether the walls kernel runs.
- **Rendering:** rigid solids in neon cyan-blue (pinned stay off-white), with thin dark stripes in 2D; a rigid edge or ring is drawn as a line in that colour, a torus seam as a dashed violet line (a checkbox hides them).
- **Source readouts:** the vertex inset button (default `wall`), a checkbox to seed pulses under driven vertices (the web app's behaviour), the source note under the inset button or the source x / y sliders (the clip fraction in amber above 1%, with the hint), the drive-wavelength warning under freq, the phase lag in the readouts while a gaussian is seeded. The old bottom line "pulse clipped at t = 0" is gone.
- **Version:** `VERSION = '1.0'` in the core; the window title "ZRL wave membrane v1.0", the main panel's first line, `--version`, and the `--shot` log line. New CLI options: `--win W,H`, `--plate-wall`, `--brush-wall`.
- `app_tests_MIRROR.py`: 109/109 on CUDA f64 and on Vulkan f32 (95 before + 14 new checks in section 11; the 14th, the 3D fit, added 06/10). 21 `--shot` variants render, each looked at. The rest of the regression (walls, triangular modes, solids, defaults, identity, the web-app cross-check): all pass (README, *Verification of the v1.0 app step*).

**Review (06/10; findings `ScratchCLAUDE/ReviewCLAUDE/findings_round4_MIRROR.md`, walls_MIRROR.md section 14)**
- An adversarial review of round 4 with its own scripts (`ScratchCLAUDE/Round4CLAUDE/ReviewCLAUDE/rv_*_MIRROR.py`): an independent numpy operator, leak, parity with the 03/10 core, incremental painting, source snapping, the CFL limit from the operator's spectrum, the app's controls and state switching, the docs, the cost per step.
- **Fixed:** a solid painted ON the sq9 Neumann ring row (or a clipped shape with a rigid edge on that ring) made the operator asymmetric; the run blew up after ~10^5 steps (max|u| 9.7e12 at 200,000 steps, `rv_neu_growth_MIRROR.py`). A sq9 diagonal link into the Neumann ring is now blocked by a non-free ring cell beside it; after the fix the spectral radius is 1 and the exact energy holds to 1e-15.
- Open, inherited from the web app: the sq9 stencil with the Neumann bc and a shape clipped by the array (pinned edge, no solids) is unstable the same way (walls_MIRROR.md section 14.2).
- The fix's own new links cannot raise the time-step limit: random and hill-climbed ring patterns keep the operator symmetric with lam_max below 16/3, and the best pattern's excess over the plain operator falls as ~N^-4 (+4.6e-9 at N 321; `rv_neu_ring_climb_MIRROR.py`, `rv_neu_ring_bound_MIRROR.py`).
- Naming: `app_tests_MIRROR.py` wrote its 3D marker images with a minus sign in the name (`proj_marker_az-120_el+20_MIRROR.png`); the signs are now `p` / `m` (`proj_marker_azm120_elp20_MIRROR.png`).
- Documentation corrections: the sq9 plate's lettering does block links at other rotations and N; two source citations. The final regression: README, *Verification after the round-4 review*.

---

## Round 3: directional sources, palettes and transparency, circle fidelity, scale mode (01-02/10/2026)

**Beam and packet sources** (01/10; [NotesCLAUDE/beam_MIRROR.md](NotesCLAUDE/beam_MIRROR.md))
- `beam`: a continuous Gaussian beam, phased by the lattice's own discrete dispersion so it steers to any angle; one-sided by default (the backward lobe is cancelled by ~29-30 dB at w = 8, lambda = 12). It is a soft source, transparent to returning waves.
- `packet`: one wave packet launched one way (<= 1e-5 of its energy goes backward; it moves at the predicted discrete group velocity to 0.05%).
- "Aim the energy": the measured direction is within 0.41 degrees (lambda = 6) and <= 0.21 degrees (lambda >= 8) of the request on sq5. `beam_tests_MIRROR.py` (34/34 in the 05/10 regression).

**Palettes and transparency** (01/10; app_MIRROR.md *Palettes and transparency*)
- Six palettes (teal-red, greyscale, cool-warm, cyclic, hot-cold, custom) and a stop editor; a settings file `look_settingsMIRROR.json` keeps them between sessions.
- Real alpha blending of the 3D surface, and 3D grid lines. `app_tests_MIRROR.py`: 88/88 then.

**Circle fidelity research** (01-02/10; [NotesCLAUDE/circle_fidelity_MIRROR.md](NotesCLAUDE/circle_fidelity_MIRROR.md))
- How round a round drum stays on each lattice, measured on the real solver: after one round trip (N = 2049, sigma = R/40) tri leaves 38 ppm of the energy outside the circular shape, sq9 44 ppm, sq5 180 ppm.
- A proposed score, the circle error element **CE** (ppm after one round trip). It is a research proposal; the app does not show it.

**Scale mode** (02/10; [NotesCLAUDE/scale_MIRROR.md](NotesCLAUDE/scale_MIRROR.md))
- A `scale` button: sizes in cells (the default, bit-identical) or as fractions of the domain's inradius, so a larger N shows the same picture with less grid error.
- Domain mode adds a time readout in crossings, a playback speed in crossings per second, amber warnings for under-resolved sizes, and *auto height*.
- `scale_tests_MIRROR.py` 16/16: second-order convergence (differences fall 4.00-4.10x per doubling of N); `app_tests_MIRROR.py` 95/95.

---

## Round 2: solids, the plate, the alignment and ledger panels, the camera (30/09-01/10/2026)

Built on 30/09, reviewed and verified on 01/10 (README *Verification (01/10/2026)*; [NotesCLAUDE/app_MIRROR.md](NotesCLAUDE/app_MIRROR.md), [solids_MIRROR.md](NotesCLAUDE/solids_MIRROR.md), [alignment_MIRROR.md](NotesCLAUDE/alignment_MIRROR.md), [ledger_MIRROR.md](NotesCLAUDE/ledger_MIRROR.md)).
- **Solids:** rect, circle, n-gon and text objects, hand-painted walls (draw / erase, undo), incremental painting (2.85 ms per segment at N 2049, 3.14 ms at N 8193: solids_MIRROR.md *Painting speed*), scene save / load. All walls pinned in this round.
- **The Unlicensed plate:** the domain shape `unlicensed` (the user's request, 29/09), with lettering and screw-head toggles.
- **Viewer:** an orbit camera (no flip, clamps) with freelook, 2D zoom and pan, visibility (palette zero level, exposure, contrast, backgrounds, 3D shading), resizable / maximised window, typed slider values (pick mode).
- **Alignment panel:** the web app's grid <-> edge alignment, 27,882 / 27,882 comparisons over 612 configurations (`crosscheck_alignment_MIRROR.py`).
- **Physics ledger panel:** 0 status and 0 text mismatches over 3,435 rows against the web app (`crosscheck_ledger_MIRROR.py`), with documented fixes.
- **Deviations from the web app**, each patched into the cross-check: Neumann corners (traced 30/09), the damped stability limit (01/10), source snapping out of solids (01/10).
- Research notes: rigid vs pinned walls, absorbing boundaries beyond Mur (fact-checked 01/10).
- `app_tests_MIRROR.py` 71/71; `crosscheck_js_MIRROR.py` 27/27 on CPU, CUDA and Vulkan.

---

## Round 1: the port (29/09/2026)

- A value-for-value port of the browser app `FlavorRenderersCLAUDE/wave_membrane_MIRROR.html` to Python + Taichi: the same lattices (square 5- and 9-point, triangular 6-neighbour), domain shapes, radius snaps, media, boundaries, sources and drive waveforms, in the same update order.
- The time step is one fused Taichi kernel; backends CUDA (f64), Vulkan (f32) and CPU.
- `crosscheck_js_MIRROR.py` runs the web app's own solver under Node and compares whole fields. Two porting faults were found and fixed: Vulkan misread the last cell of an 8-bit mask (the mask is 32-bit on Vulkan), and the f32 9-point stencil was not self-consistent.
- Speed (README *Performance*, 29/09): 3.59 ms per step for the square lattice at N = 8193 on CUDA f64; up to 1036x the web app at N = 2049.
- The Taichi app became the **primary app** on 29/09; the web app is frozen.
