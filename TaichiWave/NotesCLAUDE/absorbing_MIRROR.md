# Absorbing boundaries in the core: Mur 2nd order, CPML, sponge (round 5, 09-10/10/2026)

*Generated from scratch — Claude Opus 5.5 — 09/10/2026 (design, kernels, first tests) and 10/10/2026 (the Mur2 start-up step, the CPML 'high' level, the sponge's d_max above 40 cells, the ledger, the regression, this note).*

The user chose three absorbing boundaries (01/10 and 09/10/2026): **Mur 2nd order**, a **CPML** frame and a **sponge**. The research behind them is [absorbing_boundaries_MIRROR.md](absorbing_boundaries_MIRROR.md) (its prototypes: `ResearchCLAUDE/AbsorbingBoundariesCLAUDE/`). This note says what the core now does, how, what was measured against the research's predictions, where the build deviates from the design and why, and what the app agent has to wire.

- Tests: `absorbing_tests_MIRROR.py` (new, 11 sections; `--bench` adds the cost section). Logs: `ScratchCLAUDE/Round5CLAUDE/LogsCLAUDE/abs_*_MIRROR.log` and `abs_batch_status_MIRROR.txt` (the full regression).
- Probes, prototypes and backups: `ScratchCLAUDE/Round5CLAUDE/AbsorbCLAUDE/` (`*_before_MIRROR.*` are the backups; the parity baseline is `ScratchCLAUDE/Round5CLAUDE/BeforeAbsorbCLAUDE/`, every .py after the decisions step).
- Progress log: [progress_absorbing_build_MIRROR.md](progress_absorbing_build_MIRROR.md).
- Every configuration WITHOUT an absorbing boundary is bit-identical to the baseline (test 2a: 27 configurations x recorder on / off, fields and probe / peak histories exactly).

---

## 1. Summary

| boundary | where | what you get (measured here, CUDA f64) | cost |
|---|---|---|---|
| **Mur 2nd order** (`bc = 'mur2'`) | the full square (sq5, sq9) | 2D box, sigma-3 pulse: 8.8e-3 centre / 0.10 corner (Mur1: 2.2e-2 / 0.19). Plane waves = the research's table and the discrete theory | one ring pass, as Mur1 |
| **CPML** (`bc = 'cpml'`, `cpml_level` `std` / `high`) | the full square (sq5, sq9) | std (10 cells): 5.3e-5 / 1.8e-4; high (20 cells): 2.5e-6 / 1.8e-6; both keep their level for wide pulses (sigma 24: 3.1e-5 / 2.0e-6) | see section 9 |
| **sponge frame** (`bc = 'sponge'`) | the full square and the triangular `array` | 40 cells: 1.4e-2 / 3.8e-2 (sigma 3) | ~0 |
| **open edge** (`edge = 'open'`) | every masked shape, both lattices | a sponge outside the drawn edge: sq circle 1.64e-2, tri hexagon 1.63e-2, tri circle 1.64e-2 (R 100, sigma 3) | ~0 |

- All of them are stable over 20,000 steps of white noise at Co 0.5 and 0.7 (sq9 to 0.866, tri to 0.8), in f64 and f32 (section 8.6).
- With the sponge and no drive the exact leapfrog energy never rises (to 5e-15 of E0).
- Mur, Mur2 and CPML are not available on the triangular lattice (bc_info says why); the sponge is its absorbing boundary.
- Three deviations from the design, each measured (section 10): Mur2 starts with one Mur1 step; CPML `high` uses alpha 0 (not 0.05); the sponge's auto d_max is 8 / L above 40 cells.

## 2. What acts where (bc, edge, fallbacks)

| lattice, shape | `bc` that act | `edge` | notes |
|---|---|---|---|
| square, full (`square` at rad >= 0.999, or `array`) | dirichlet, mur, **mur2**, **cpml**, **sponge**, neumann, periodic | n/a (no mask edge) | cpml needs N >= 2 L + 9 (29 / 49), else it falls back to dirichlet with a reason |
| square, masked shape | dirichlet, mur, neumann, periodic (the array ring where the shape reaches it) | pinned, rigid, **open** | mur2 / cpml / sponge fall back to dirichlet ("use edge: open"); periodic falls back with edge open (exclusive) |
| triangular `array` | dirichlet, **sponge**, neumann (rigid ring), periodic (rhombic torus) | n/a | mur, mur2, cpml fall back to the clamped ring with a reason |
| triangular web-app shape | none (the clamped ring, as the web app) | pinned, rigid, **open** | |

- `Params.bc`: `mc.BCS = ['dirichlet', 'mur', 'mur2', 'cpml', 'sponge', 'neumann', 'periodic']` (the B cycle's order), `mc.TRI_BCS` adds `sponge`. `mc.WEBAPP_BCS` are the web app's four; `mc.ABSORBING_BCS = ('mur', 'mur2', 'cpml', 'sponge')`; `mc.LAYER_BCS = ('cpml', 'sponge')`.
- `Params.edge`: `mc.EDGES = ['pinned', 'rigid', 'open']`.
- The reasons are module constants: `TRI_MUR_REASON`, `TRI_MUR2_REASON`, `TRI_CPML_REASON`, `MASKED_ABS_REASON` (formatted with the bc), `OPEN_PERIODIC_REASON`, and the too-small-array text built in `bc_info()`.
- `Membrane.bc_effective()` returns what runs; `bc_info()` says so for the UI (section 12).

## 3. Geometry: the open domain and the layer

- **The drawn domain outline is the OPEN edge: waves leave through it.** The layer lies OUTSIDE it (research section 7.2: grow the mask outward), so the visible shape keeps its size and resonances are not shifted by a layer inside it.
- **Where the array leaves no room, the open domain shrinks to make room:**
  - the full square / triangular array with a CPML or sponge frame: the open domain is the array minus the frame. `dom_R()`, `inner_R()` and `shape_poly()` describe that open square / rhombus (`_frame_Ld()` = the index depth of the first open row; CPML: Ld = L; sponge: Ld = ceil(L) on sq, ceil(L / (sqrt3/2)) on tri);
  - a masked shape with an open edge: R is capped at `base_half - 1.5 - L` (`_dom_R_cap()`), so the layer (L beyond the outline, at most L beyond the circumradius) stays 1.5 cells off the ring like every shape. A shape at radius 1 shrinks (N 161: R 78.5 -> 58.5 with the 20-cell auto width); `edge_info()` says so.
- **Layer cells are free stencil cells.** They are in `domain_mask_np` and `mask_np`; `abs_np` (uint8 [i, j]) holds their codes (>= 2; 0 elsewhere); the device mask `M` carries the codes too (`_dev_mask()`), which is how the kernel finds them without another array.
  - `open_domain_np()`: the domain without the layer (what the alignment panel analyses); `open_mask()`: `mask_np` without the layer (what sources, vertices and the beam aperture may use); `layer_mask()`: bool, the layer's cells (what the UI tints).
- **The layer's codes.** Sponge FRAME of at most 254 rows: one code per row (code 2 + its index depth), and the AUX table holds each row's exact damping d_max ((L - depth) / L)^p (the earlier 254-level quantisation of s cost up to 9-13% of |R| at some (lambda, angle); progress log 09/10). Wider frames and the OPEN EDGE: code 2 + floor(s x 254), s = the distance outside the outline / L, quantised in 254 levels. The distance is exact for every masked shape (`_sdf_out`: circle, square, regular n-gon folded into a sector, rhombus folded into a quadrant, the plate's rounded rectangle), so the ramp is smooth across the staircase. It is computed only on the band where it can be under L (every shape lies in the disk of radius `_shape_Rc()`, so a cell at radius r is at least r - Rc from it): the same codes as over the whole grid in 159 configurations (`AbsorbCLAUDE/probe_band_codes_MIRROR.log`). Reseed (host) at N 8193, CUDA f32: the open circle 4.3 s, the open tri hexagon 8.5 s (pinned 5.5 s), the open plate 8.3 s; the CPML frame 3.3 s and the sponge frame 2.5 s against 1.7 s for the Dirichlet square (`probe_build_time_MIRROR.log`).
- **What is defined relative to the domain follows the OPEN domain:** sources (`sx`, `sy` as fractions of r_in), vertex insets, the beam aperture, domain-anchored solids, the probes, the mode box, scale mode's r_in. A source requested inside a layer walks back toward the centre like one outside the domain (`src_snap == 'layer'`, test 9d). Solids may be painted anywhere, the layer included (section 6).
- **Energy, peak and ledger rows describe the open domain:** the recorder skips cells with code >= 2 (`rec_fold`'s `absx` variant: `hist['E']`, `amax`, `stats()`; test 9c: the recorded energy and peak equal numpy over the open cells to 1e-12).
- Switching `p.bc`, `p.edge` or the sponge's width between steps rebuilds the domain and keeps the field (zeroed on cells that leave the domain): `step()` compares `_layer_key()`, `set_edge()` does it at once (test 9h).

## 4. Mur 2nd order

- **The rule** (Engquist-Majda A2, the form printed in Taflove & Hagness 2005; research section 4.1), at ring cell r0 with its interior partner r1, ' = level c (new), '' = level a:
  `u0' = -u1'' + k (u1' + u0'') + g (u0 + u1) + e (Dtt u0 + Dtt u1)`, k = (Co - 1)/(Co + 1), g = 2/(Co + 1), e = Co^2 / (2 (Co + 1)), Dtt the tangential second difference at level b. Co is the ring cell's own (sqrt CC), so a slab medium's slow part is matched.
- **Corners: Mur1 along the diagonal**, k_d = (Co - sqrt2)/(Co + sqrt2) (corners held at 0 blow up: research section 1). Every read is of level a or b or the interior's level c: ONE parallel ring pass (`_mur2_side`, `_mur2_corner`; `BC_MUR2`, no new kernel argument).
- **The start-up step (NEW, 10/10/2026).** Mur2 is Mur1's residual A = u0' - u1 - k (u1' - u0) differenced in time: A^n - A^{n-1} = e (Dtt u0 + Dtt u1). So it keeps whatever residual it starts from, as a static offset that never decays (the energy cannot see a constant). A field seeded at rest with values on the ring row and its partner (the full square's seed is not clamped on its ring until step 1, inherited) starts with A != 0. Measured before the fix (`AbsorbCLAUDE/probe_mur2_dc_MIRROR.log`, N 161, 6000 steps): a sigma-3 pulse centred 8 cells from the ring left a constant -0.099 (Mur1: 4.5e-4), a spike next to the ring -0.354, the noise test -3.5. **Fix:** the first launch after `seed()`, `set_levels()`, `set_co()`, a mask change (`build_domain`, `update_solids`, `set_solid_np`) or a run with another bc is `BC_MUR2S`: its first step runs Mur1 on the sides (`_mur1_side`), which sets A = 0; Mur2 runs on from there (`Membrane._mur2_ok`). After (`probe_mur2_dc_after_MIRROR.log`): the same pulse leaves 1.7e-9, the spike 2.0e-8; test 9j (seed, drop_pulse, set_co, a bc switch): |mean| <= 6e-4.
- **What remains: corners.** Along a side the tangential sums cancel; at the corners they do not, and a source that overlaps a corner region at the seed still leaves a small static offset, as the web app's Mur1 does on sq5 (its zero-frequency rule is u0 = u1, a Neumann condition, and the sq5 stencil never reads the zeroed corners): sigma-3 pulse at (0.9, 0.9): Mur2 8.0e-4, Mur1 8.9e-4; sigma 6 there: 5.7e-2 / 1.8e-2 (`probe_mur2_corner_MIRROR.log`). Pulses that start away from the ring leave ~1e-12 or less (a centred sigma-3 pulse: 1e-14 after 9000 steps).
- **Stability:** 20,000 steps of noise bounded and falling on sq5 at Co 0.5 / 0.7 / 0.7071 and on sq9 at 0.5 / 0.7 / 0.85 / 0.866 (section 8.6). **No Co limit is needed on sq9** (the design asked to state one if sq9 were unstable at high Co: it is not).
- Not on the triangular lattice (`TRI_MUR2_REASON`).

## 5. CPML

- **Formulation** (convolutional PML, Roden & Gedney 2000; Komatitsch & Martin 2007; the research prototype `abc2d_MIRROR` 'cpml'). Along each stretched axis the face differences g get a recursive memory psi = b psi + a g at the two faces, the face fluxes (g + psi) are differenced, and a cell memory zeta = b zeta + a dq is added; b = exp(-(sigma + alpha) dt), a = sigma / (sigma + alpha) (b - 1); sigma = sigma_max s^p, sigma_max = fac (p + 1) ln(1/R0) / (2 L), R0 = 1e-6, s = depth / L.
- **Single pass, no race:** each frame cell keeps its OWN copies of its two face memories per axis and updates them in place from the same inputs its neighbour uses for the shared face (no face ownership). On sq5 this equals the research's shared-face form to rounding (3.2e-12 after 400 steps; `proto_frame_MIRROR.py --part check`).
- **sq9: the 9-point operator stretched exactly.** L9 = Dxx + Dyy + (1/6) Dxx Dyy = D~xx(u + D~yy u / 6) + D~yy u with the stretched second differences (they commute): x memory for the field u + D~yy u / 6, y memory for u. The x part needs D~yy u at the x neighbours, computed from THEIR y memory, which is therefore a ping-pong pair (the step parity rides in `s0` as 3 x parity; no extra pass). With sigma = 0 it is L9 to 3.3e-15. Box (sigma 3): 5.4e-5 / 1.8e-4 centre / corner, = sq5. The design's fallback (a sponge on sq9 if the 9-point CPML were unstable) is not needed: stable to Co 0.866 (section 8.6).
- **Memory: frame-packed**, never full-size (research section 7.2). AUX = one float ndarray (one launch argument; ~30 us per ndarray argument): [0] Lc; [AUX_INV + m], [AUX_C2 + m] the sponge tables; from AUX_TAB the 1D coefficient tables [6][2 Lc] (b, a at the cell, the inner-index face, the outer-index face); then the memory: x strips [3][2 Lc][N] and y strips [3][2 Lc][N] (sq9: a ping-pong pair of y strips). 12 Lc N values (sq5), 18 Lc N (sq9): at N 8193 and L 20, 16 MB (sq5) / 24 MB (sq9) in f64, against 1.6 GB for U[3].
- **The frame's outermost ring is Dirichlet** (the existing clamp). The memory of a cell that is not free (a solid painted in the frame) is held at 0.
- **Levels** (`Params.cpml_level`, `mc.CPML_LEVELS`): `std` L 10, p 3, fac 2, alpha 0; `high` L 20, p 3, fac 2, **alpha 0** (deviation: section 10.2). Box, sigma 3: std 5.3e-5 / 1.8e-4, high 2.5e-6 / 1.8e-6.
- **Solids in the sq9 frame act pinned** (no link removal, no corner blocking inside the frame and one row beyond: `_cpml9_Lc()`, `_class_box`): the stretched 9-point operator splits each diagonal link between the x and y memories, which link removal and blocked diagonals cannot follow -- a rigid block in the frame grew to 3.6e6 in 6000 steps (`AbsorbCLAUDE/dbg_9b_MIRROR.py`). The sq5 CPML keeps rigid solids (axis links only). Test 9b: sq9 with rigid and pinned solids in the frame and the domain, the walls kernel, 5.4e-5 after 3000 steps.
- `operator()` refuses a CPML run (its update has memory); the CPML is not on the triangular lattice (`TRI_CPML_REASON`: a PML there is open research).

## 6. Sponge (frame and open edge)

- **The centred damping form inside the layer** (research section 4.2; the app's -d (u - u_prev) form lowers the CFL limit to Co_max sqrt(1 - d/2)):
  `u+ = (2u + CC lap - d_g (u - u-)) / (1 + d_m/2) - (1 - d_m/2) / (1 + d_m/2) u-`, d_m the cell's layer damping, d_g the global damping (the app's form, unchanged). The two factors come from the 256-entry AUX tables by the cell's code: **no per-cell divide** (an f64 divide per cell doubled the step at N 4097). Cells with code < 2 compute today's update with today's expression, bit for bit (test 2b: the sponge kernel with every code cleared == the plain kernel, sq5 / sq9 / tri, 500 steps).
- **The ramp:** d = d_max s^2 (p = 2), s = the depth into the layer / L (0 at the open domain, 1 at the outer end).
- **Width** (`Params.sponge_L`, physical cells; None = auto): 40, or 20 when the array's half-size is under 160 cells (N < ~321); capped at a third of the half-size.
- **d_max** (`Params.sponge_dmax`; None = auto): 0.35 at L <= 20, 0.2 at L 40, linear between (the research's tuning); **above 40 cells 8 / L** (deviation: section 10.3).
- **Scale mode `domain`: the width follows the scale** (`sponge_Lf` x the half-size, never under 20 cells; set from the cells value at the first switch, like `sigma_f`), and with the auto d_max = 8 / L the damping per wave stays the same. Measured (section 8.5): the reflection then stays at 1.44e-2 for pulses of sigma 3 / 6 / 12 / 24 cells; a fixed 40-cell sponge rises to 0.32. The CPML frame stays 10 / 20 cells in both modes: it keeps its level for any pulse width (with alpha 0).
- Solids inside the layer are allowed (both types; test 8: a rigid and a pinned block in the open circle's layer, energy never rises, the walls kernel runs).

## 7. Kernels

- **Template specialisation per boundary kind** (a generic coefficient-table rule compiled for 37-139 s in the research): `k_run` / `k_run_rec` unchanged for every non-absorbing configuration (absk = ABS_NONE compiles the code of before); `k_run_abs` / `k_run_abs_rec` with `absk` = ABS_SPONGE or ABS_CPML run `_abs_pass` (the interior with the layer, then the clamped ring) instead of the lattice branches; Mur2 is a `bc` template value of `k_run` (`BC_MUR2`, `BC_MUR2S`).
- **Launch arguments:** + AUX and LV (the device mask with its codes) for the absorbing kernels only; CHUNKS = (8, 1) as before; the sq9 CPML parity rides in `s0`.
- **No per-cell f64 divide** anywhere (the sponge's 1/(1 + d/2) is tabulated; the CPML's b, a are 1D tables).
- **Vulkan f32:** i32 masks carry the codes; AUX is f32. The f32 9-point form is the self-consistent one (README, round 1). Vulkan's SPIR-V codegen needs a constant begin for a top-level loop (`k_aux_fill` first read `range(start, n)`: 'Assertion failure: stmt->const_begin'; now `range(n - start)`).
- **The recorder** (`rec_fold`, `k_stats`): the `absx` variant sums the open domain only (LV < 2); absx = False compiles the code of before.

## 8. Measured, against the research's predictions

All CUDA f64 unless noted; `absorbing_tests_MIRROR.py` section numbers. Every row passed.

### 8.1 The kernel == an independent numpy reference (section 3)
13 configurations (CPML sq5 / sq9, std / high, a lens medium, global damping, pinned solids in the frame; Mur2 sq5 / sq9 + slab; the sponge frame sq5 / sq9 / tri; the open edge on the sq circle, the tri hexagon, the plate, a sq9 octagon + lens), 300 steps, the four walls and corners: max difference / max |u| 8e-15 .. 4.3e-12.

### 8.2 Plane-wave |R| (section 4: the research's 1D-exact reduction of a straight wall, with the core's own coefficient tables)

| boundary | 0° | 15° | 30° | 45° | 60° | 75° | vs the research |
|---|---|---|---|---|---|---|---|
| CPML std (L 10), sq5, Co 0.5 | 9.8e-5 | 1.2e-4 | 1.2e-4 | 3.3e-5 | 2.4e-5 | 1.8e-5 | its table, 0.00% (every entry) |
| CPML high (L 20, alpha 0), sq5 | 8.1e-5 | 4.8e-6 | 3.8e-6 | 2.7e-6 | 1.6e-6 | 2.2e-5 | the research's `cpml20 p3 f2 a0.00` entries, 0.00% |
| Mur2, sq5 | 3.0e-2 | 2.7e-2 | 1.5e-2 | 2.9e-2 | 0.11 | 0.35 | its table, 0.00%; the discrete theory to 8.1e-4 |
| sponge L 40, d_max 0.2 (frame) | 1.0e-2 | 7.6e-3 | 1.0e-2 | 6.2e-2 | 0.20 | 0.49 | its table (`sponge40 p2 d0.20 ctr`), 0.0% (the exact per-row codes) |
| sponge L 20, d_max 0.35 | 0.12 | 0.13 | 0.17 | 0.26 | 0.41 | 0.65 | its table, 0.0% |
| CPML std, sq9 | 9.8e-5 | 1.2e-4 | 1.2e-4 | 3.3e-5 | 2.4e-5 | 1.8e-5 | no research table: the sq5 values |

The worst case over λ = 8 / 16 / 32 at each angle. Mur2 on sq9 equals its discrete theory to 1.2e-3 (the packet averages R over its ±5% spread in k_x).

### 8.3 The 2D box, a broadband sigma-3 pulse, the four walls and the corners (section 5)

centre / corner pulse:

| boundary | sq5 Co 0.5 | sq5 Co 0.7 | sq9 Co 0.5 | sq9 Co 0.7 | research (sq5, Co 0.5) |
|---|---|---|---|---|---|
| CPML std | 5.3e-5 / 1.8e-4 | 1.0e-4 / 2.6e-4 | 5.4e-5 / 1.8e-4 | 1.1e-4 / 2.6e-4 | 5.3e-5 / 1.8e-4 |
| CPML high | 2.5e-6 / 1.8e-6 | 2.5e-6 / 1.8e-6 | 2.5e-6 / 1.8e-6 | 2.5e-6 / 1.8e-6 | 2.5e-6 / 1.8e-6 (alpha 0.05) |
| Mur2 | 8.8e-3 / 0.10 | 5.9e-3 / 0.10 | 9.5e-3 / 9.9e-2 | 6.4e-3 / 0.10 | 8.8e-3 / 0.10 |
| sponge frame L 40 | 1.4e-2 / 3.8e-2 | 1.9e-2 / 3.1e-2 | 1.4e-2 / 3.7e-2 | 1.9e-2 / 3.1e-2 | 2.2e-2 / 4.5e-2 (p 3 d .5); the p 2 d .2 prototype 1.4e-2 / 3.8e-2 |
| Mur1 (the web app's) | 2.2e-2 / 0.19 | 2.3e-2 / 0.20 | 2.2e-2 / 0.19 | 2.3e-2 / 0.19 | 2.2e-2 / 0.19 |

### 8.4 The open edge on masked shapes (section 6)
R 100, sponge L 40, d_max 0.2, a sigma-3 pulse from the centre: sq circle **1.64e-2** (research 1.65e-2), tri hexagon **1.63e-2** (1.60e-2), tri circle **1.64e-2** (1.7e-2).

### 8.5 Long waves, scale mode (section 11; `AbsorbCLAUDE/probe_scale_refl_MIRROR.log`)
The box with the open domain 200 f + 1 and the pulse sigma 3 f at the centre (Co 0.5):

| sigma (cells) | CPML std | CPML high, alpha 0 | high with the research's alpha 0.05 | sponge 40 fixed | sponge 40 f, d_max 0.2 | sponge 40 f, d_max 8 / L (the auto rule) |
|---|---|---|---|---|---|---|
| 3 | 5.3e-5 | 2.5e-6 | 2.5e-6 | 1.44e-2 | 1.44e-2 | 1.44e-2 |
| 6 | 3.4e-5 | 2.1e-6 | 3.8e-5 | 5.1e-2 | 2.2e-2 | 1.44e-2 |
| 12 | 3.2e-5 | 2.0e-6 | 4.3e-3 | 0.15 | 3.4e-2 | 1.44e-2 |
| 24 | 3.1e-5 | 2.0e-6 | 3.1e-2 | 0.32 | 5.2e-2 | 1.45e-2 |

(The CPML high columns from `probe_cpml_high_MIRROR.log`.)

### 8.6 Stability: 20,000 steps of white noise (section 7)
max|u| from 1.0 at the start; the four marks at 5000-step intervals in the log. Bounded and falling everywhere: CPML sq5 std / high at Co 0.5 / 0.7 / 0.7071; CPML sq9 std at 0.5 / 0.7 / 0.85 / 0.866, high at 0.5 / 0.85; Mur2 sq5 at 0.5 / 0.7 / 0.7071, sq9 at 0.5 / 0.7 / 0.85 / 0.866; the sponge frame sq5 (0.5 / 0.7 / 0.7071), sq9 (0.5 / 0.85), tri (0.5 / 0.8); the open edge on the sq circle (0.5 / 0.7), the tri hexagon (0.5 / 0.8), the tri circle (0.8).
- **At the CFL limit itself** (within 0.1%: sq5 0.7071, sq9 0.866) the near-Nyquist modes have ~zero group velocity: they never reach the boundary and stay, as in a closed box. CPML sq9 at 0.866: 1.47 / 1.43 / 1.43 / 1.52 (bounded, not falling). The test asks for no growth there, and for falling below the limit.
- Mur2's noise run used to plateau (sq5 Co 0.7: 4.5, a constant): the static offset of section 4, fixed by the start-up step (0.13 after the fix, falling).

### 8.7 Energy (section 8)
With the sponge and no drive, the exact leapfrog energy (`leapfrog_energy()`, every 10 steps) never rises: max rise -1.4e-16 .. 4.9e-15 of E0 (the frame sq5 / sq9 / tri, the open sq circle, the open tri hexagon + lens, the open circle with rigid and pinned solids in the domain and in the layer). It falls to 1e-6 .. 5e-4 of E0 in 1500 steps (N 121-141). The centred form gives this exactly: E^{n+1/2} - E^{n-1/2} = -(d/2) |u+ - u-|^2 / CC <= 0.
- The **recorded** open-domain energy (the web app's functional kin/(2 Co^2) + pot/2) is not that energy: released from rest it reads half the potential and rises 10-23% within ~10 steps, in a closed drum too (`probe_eopen_MIRROR.log`: closed sq5 max / step 1 1.105). So "never above its start" is the wrong test; the ledger row and test 8 check **no growth**: the later half's maximum <= 1.05 x the earlier half's (closed drums: 0.96-1.03; with a layer: <= 3e-5 at 1500 steps).

### 8.8 Composition (section 9)
- 9a global damping 0.02 at Co 0.7 with every boundary: decays (CPML 9.0e-5, Mur2 6.6e-5, sponge ~1e-12 after 2000 steps).
- 9b solids (rigid and pinned, in the frame and the domain, sq9 CPML, the walls kernel): bounded and absorbing.
- 9c the recorder: the open domain only, = numpy over the open cells; the probes in the open domain.
- 9d sources: a source requested in the layer snaps out (`src_snap == 'layer'`); vertices and the beam aperture use open cells only.
- 9e painting across a layer: the device mask keeps the layer codes; incremental == the full rebuild (the open circle; the triangular sponge frame).
- 9f `set_co` with and without rescale on CPML and the open edge: the tables follow Co, the CPML memory is kept.
- 9g scale mode: the sponge width follows the scale (L 40 at N 513 -> 80 at N 1025, d_max 0.2 -> 0.1); the CPML frame stays 10 cells.
- 9h switching `edge` to / from open and `p.bc` between steps rebuilds the layer, the field is kept.
- 9i beam, packet, driven vertices, cont and slit with each kind of boundary: finite and bounded.
- 9j Mur2 leaves no static offset (section 4).

### 8.9 f32 and Vulkan
See the regression table (section 13): the same sections with f32 tolerances.

## 9. Cost per step

`absorbing_tests_MIRROR.py --only 10 --bench` (interleaved, best of 5; plain = the Dirichlet full square, the open circle against the pinned circle). Filled in from the regression (section 13).

## 10. Deviations from the design (each measured)

1. **Mur2's start-up step** (section 4): not in the design; without it Mur2 keeps a static offset from any data on the ring rows at a seed / set_levels / set_co / a mask change / a bc switch (-0.099 for a pulse centred 8 cells from the ring).
2. **CPML `high` = L 20 with alpha 0** (the design: alpha 0.05). The research tuned alpha on lambda 8-32 plane waves and on sigma-3 pulses, where both are equal (box 2.5e-6 / 1.8e-6). A complex-frequency shift absorbs waves with omega < alpha less: with sigma 12 / 24 pulses alpha 0.05 gives 4.3e-3 / 3.1e-2, alpha 0 2.0e-6 / 2.0e-6 (`probe_cpml_high_MIRROR.log`; alpha 0.01 / 0.02 degrade at sigma 24: 1.4e-5 / 2.1e-3). The research's 1D table has alpha 0 at 8.1e-5 worst (lambda 32, 0°) against 8.9e-6, still under `std`'s 1.2e-4; noise stable (section 8.6).
3. **The sponge's auto d_max above 40 cells is 8 / L** (the design gave L 40 / 0.2 and L 20 / 0.35 only). L d_max fixed keeps the damping per wave, so the domain-mode sponge (its width following the scale) keeps 1.44e-2 at every scale (0.2 fixed: up to 5.2e-2); in cells mode it also lowers the reflection (L 80, sigma 3: 4.8e-3 / 1.3e-2 centre / corner against 5.7e-3 / 2.3e-2; `probe_sponge_long_MIRROR.log`).
4. **Solids inside the sq9 CPML frame act pinned** (section 5): rigid link removal and corner blocking cannot follow the split diagonal links (a rigid block there grew to 3.6e6).
5. **The sponge frame's codes are one per row** (exact damping per row) instead of the 254-level ramp, where the frame has at most 254 rows (the quantisation cost up to 9-13% of |R|).
6. **The noise criterion at the CFL limit** is "no growth" instead of "falling" (section 8.6), and the recorded-energy check is "no growth" instead of "never above its start" (section 8.7): both measured properties of the scheme and of the web app's energy functional, not of the boundaries.

## 11. Ledger

- `ledger_MIRROR.py` handles every new bc / edge (`_Ctx.port_abs`: mur2, cpml, sponge, open). The rows describe the OPEN domain: the conformance and staircase counts count it (`open_domain_np`), the front clearance stops at the layer, the recorder sums it.
- n/a WITH the reason where the premise is a closed, lossless domain: the energy row ("absorbing edge removes energy by design (...)"), the drum spectrum ("an open window with no standing modes"), the eigenmode and dispersion rows. The boundary row reads "absorbing edge: little comes back" (late/early <= 0.15; n/a for an impulse, whose grid-scale tail reaches the probe late whatever the edge does -- the web app's Mur1 reads FAIL there for that reason); the wake and linearity rows accept the port's absorbing boundaries as "the absorbing boundary".
- The symmetry rows check their premise on the layer's damping map cell by cell (`_Ctx.layer_sym`); CPML sq5 measured 1.6e-15 (8-fold), Mur2 6e-17.
- **New row `eopen`** "energy inside the open domain does not grow (absorbing boundary, no drive)", only for the port's absorbing boundaries (web-app configurations keep the web app's row list): the later half's max <= 1.05 x the earlier half's of the recorded window, or the earlier half under 1e-8 of the run's peak (`Membrane.hist_emax`, new: the recorded E's maximum since the seed). n/a for a driven source, dropped pulses, a non-uniform medium, the recorder off (the history-gap rule).
- `crosscheck_ledger_MIRROR.py`: 14 new port-only checks (X1-X11: the CPML frame, the sq9 sponge frame with a mode, Mur2 impulse / pulse, the open circle, the open tri hexagon, the tri sponge frame, the CPML fallbacks, a driven CPML run, the recorder gap, an off-centre sq9 CPML pulse); no row FAILs in any of them.
- The alignment panel (`alignment_MIRROR.py`) analyses the open domain (`open_domain_np`).

## 12. UI hand-off (for the app agent)

**Params** (all port-only):
- `bc`: the B cycle follows `mc.BCS` and cycles only `bc_info()['options']` (already so in the app: on the full square dirichlet -> mur -> mur2 -> cpml -> sponge -> neumann -> periodic; on the triangular array dirichlet -> sponge -> neumann -> periodic). `app_tests_MIRROR.py` was updated for the longer cycle.
- `edge`: pinned / rigid / **open** on masked shapes (the app's edge button toggles pinned / rigid today: it needs the third state). `set_edge()` keeps the field; to / from open it rebuilds the domain (the radius may shrink).
- `cpml_level`: 'std' / 'high' (`mc.CPML_LEVEL_NAMES`): a toggle while the CPML runs.
- `sponge_L` (None = auto), `sponge_dmax` (None = auto), `sponge_Lf` (domain mode): sliders while a sponge runs; show `m.sponge_width()` and `m.sponge_dmax()` (the values in use).

**Info for the panel:**
- `m.bc_info()`: `requested`, `effective`, `fallback`, `reason`, `ring` (words, e.g. "absorbing frame (CPML, 10 cells, std; the open domain is the array minus the frame)", "absorbing ring (Mur, 2nd order; diagonal Mur-1 corners)"), `options`, `unavailable` {bc: reason}, `edge_applies`, `torus`, **`absorbing`** (the effective bc absorbs), **`layer`** (`abs_layer()` for a frame, else None).
- `m.edge_info()` (new): `requested`, `applies`, `options`, `kind`, `layer`, `R_requested`, `R_open`, `capped`, `lines` = [(text, level)] with level 'info' / 'warn' / 'dim', e.g. "open edge: a sponge 40 cells wide outside the drawn edge (d_max 0.20, centred form) - waves leave through the edge" (info), "the shape shrank from R 78.5 to 58.5 cells to leave room for the sponge inside the array" (warn), "the layer is drawn tinted; it is not part of the energy, the peak or the ledger" (dim).
- `m.absorbing()`: 'mur' / 'mur2' / 'cpml' / 'sponge' / 'open' or None. `m.abs_layer()`: kind, where ('frame' / 'edge'), L, Ld, and Lc / p / alpha / smax / level (CPML) or p / dmax / exact (sponge).
- `m.mode_note()`: n/a text for the drum-mode source with an absorbing boundary ("the domain has no standing modes").
- `m.src_snap == 'layer'`: an amber note like the solids' one ("source moved out of the absorbing layer").

**What to tint:** `m.layer_mask()` (bool [i, j]) -- the layer's cells, in 2D and 3D. They are free cells (the field runs through them), so draw the field there with a tint, not the hatch. `m.abs_np` holds the codes (>= 2) if the tint should deepen with the damping (sponge: code - 2 = the row depth for a frame of <= 254 rows, else floor(s x 254)). The open domain's outline is `m.shape_poly()` (the frame's inner edge for a full array).

**Warnings worth showing:**
- the shape shrank to make room (`edge_info()['capped']`);
- CPML on an array under N 2 L + 9 falls back (the reason);
- Mur / Mur2 / CPML on the triangular lattice fall back (the reasons);
- Mur2: a source overlapping a corner region at the seed leaves a small static offset (section 4), as Mur1 does;
- the sponge reflects long waves more: at a fixed width in cells mode, sigma 12 gives ~0.15 (use the CPML on the full square, or a wider sponge).

**Ledger panel:** the new row `eopen` appears only with an absorbing boundary.

**CFL readout:** unchanged (the centred sponge keeps the undamped limit; CPML and Mur2 are stable up to the limit).

## 13. Regression (10/10/2026)

Filled in after the batch (`AbsorbCLAUDE/abs_regression_batch_MIRROR.sh`, `LogsCLAUDE/abs_batch_status_MIRROR.txt`).

## 14. Open issues

- Mur2's corner offset (section 4) remains: a source that overlaps a corner region at the seed leaves a constant of up to ~6% of a sigma-6 pulse placed there (Mur1: 1.8%). A corner rule that damps the zero frequency (Higdon's epsilon) would remove it; not attempted (the design fixed the diag-Mur1 corner).
- CPML at Co within 0.1% of the limit keeps the near-Nyquist noise (as any boundary does: those modes do not travel). Not a CPML fault.
- The recorded energy (the web app's functional) lags the conserved energy by O(dt); the ledger's `eopen` row grades no growth on it, not monotonic decay.
- The CPML frame with an anisotropic medium (the planned prism-like media) must be re-checked (Bécache, Fauqueux & Joly 2003; research section 4.3).
- A PML on the triangular lattice: open research (the sponge serves there).

## 15. Files

| file | what |
|---|---|
| `membrane_core_MIRROR.py` | the boundaries (constants, `_mur1_side`, `_mur2_side`, `_mur2_corner`, `_cpml_lap`, `_abs_pass`, `k_run_abs(_rec)`, `k_aux_put`, `k_aux_fill`; Membrane: `bc_effective`, `bc_info`, `edge_info`, `abs_layer`, `absorbing`, `sponge_width`, `sponge_dmax`, `_sdf_out`, `_layer_rows`, `open_domain_np`, `open_mask`, `layer_mask`, `_dev_mask`, `_aux_tables`, `_aux_sync`, `_cpml9_Lc`, `hist_emax`) |
| `absorbing_tests_MIRROR.py` | the tests (11 sections + `--bench`) |
| `ledger_MIRROR.py`, `crosscheck_ledger_MIRROR.py` | the ledger's absorbing handling and row `eopen`; 14 port-only checks |
| `alignment_MIRROR.py` | the open domain |
| `app_tests_MIRROR.py` | the B-cycle expectations (the longer cycle) |
| `ScratchCLAUDE/Round5CLAUDE/AbsorbCLAUDE/` | prototypes (`proto_frame`, `proto_sq9_cpml`), probes (`probe_mur2_dc`, `probe_mur2_noise`, `probe_mur2_corner`, `probe_eopen`, `probe_scale_refl`, `probe_cpml_high`, `probe_sponge_long`), the batch script, backups |
