# Walls: rigid solids, corner blocking and the rigid edge (round 4)

Build notes for the solver core (`membrane_core_MIRROR.py`) and the solids module (`solids_MIRROR.py`). The design follows the fact-checked research note [rigid_vs_pinned_walls_MIRROR.md](rigid_vs_pinned_walls_MIRROR.md), sections 2, 4 and 5: rule **R0** (rigid = link removal) plus **corner blocking**. Rule R1 and impedance walls are not built. No app or UI work was done; section 11 is the hand-off for the agent who wires the UI.

*Generated from scratch — Claude Opus 5.5 — 04/10/2026*

**Round 4b (05/10/2026):** the triangular lattice's boundary modes — the `array` shape, the rigid ring, the rhombic torus, the Mur fallback — and the ledger's handling of rigid walls are in **section 13**; section 11 has the UI hand-off for both rounds. *Edited by Claude Opus 5.5 — 05/10/2026.*

**Contents:** 1 What is new · 2 The rule · 3 Data and kernels · 4 Readers of the mask · 5 The solids module · 6 Helpers · 7 Measured · 8 Deviations · 9 The ledger with rigid walls · 10 Open issues · 11 UI hand-off · 12 Files · 13 Triangular lattice modes (13.1 what is new, 13.2 bc and edge, 13.3 core changes, 13.4 geometry, 13.5 what else depends on the shape, 13.6 alignment and ledger, 13.7 measured and cost per step, 13.8 deviations, 13.9 open issues) · 14 Review fixes (06/10/2026). *Contents line and section 13 appended by Claude Opus 5.5 — 05/10/2026; section 14 by Claude Opus 5.5 — 06/10/2026.*

---

## 1. What is new

- **Two wall types.** Every solid object and every painted stroke is `pinned` (u = 0, the old behaviour and the default) or `rigid` (zero normal slope).
  - A pinned wall reflects with **R = −1**: a crest comes back as a trough.
  - A rigid wall reflects with **R = +1**: a crest comes back as a crest. This is a sound wave off a hard object, or a ripple off a quay.
- **Corner blocking**, for both types: a link that slips between two solid cells corner to corner is cut. This stops the leak through one-cell diagonal *pinned* walls that the research note measured: 2.3% on sq9 and 17% on tri. Every lattice now measures exactly 0.0.
- **A rigid domain edge**: `Params.edge = 'rigid'` makes the edge of any masked shape rigid, the Unlicensed plate included. On the triangular lattice it also makes the array's outer ring rigid. The default `'pinned'` is the web app's edge.
- **Nothing changes when nothing needs to.** With no rigid cell and no blocked link, the solver runs exactly the 03/10/2026 kernel. Fields are bit-identical, and the speed is the same. This holds with no solids, and with pinned solids that block no link.
- **Cost when walls are active:** +3–8% per step on CUDA f64, +10–12% on CUDA f32 at N = 8193, +0–2% on Vulkan f32. Painting still costs ~2.5 ms per segment, even at N = 8193.

---

## 2. The rule

### 2.1 Cell classes

Each cell has a class, as its neighbours see it:

| class | what it is | a link from a free cell to it |
|---|---|---|
| F | free: a stencil cell, or an array-ring cell that the bc governs (the sq Dirichlet / Mur / Neumann ring) | active: reads the neighbour |
| P | pinned solid (holds 0) | active: reads the 0. The term is w(0 − u_i), as before |
| R | rigid solid | **dropped**: the kernel reads u_i itself, so w(u_i − u_i) = 0 |
| X | domain exterior, incl. the triangular array's clamped ring | like P with `edge = 'pinned'`, like R with `edge = 'rigid'` |

A solid outside the domain counts as X: the domain wins.

### 2.2 Corner blocking

A link between two **free** cells is blocked when **both** of its common neighbours are non-free.
- On sq9 the common neighbours of a diagonal are the two edge cells it passes between.
- On tri they are the two cells that form a triangle with the link.
- sq5 has no blockable link, and the sq9 edge links are never blocked.

What a blocked link becomes:
- **pinned-blocked** if either blocker acts pinned (P, or X with a pinned edge): the term is w(0 − u_i);
- **dropped** if both blockers act rigid (R, or X with a rigid edge).

**The web-app exception.** With `edge = 'pinned'`, a blocker pair made only of X cells does **not** block. The web app has no blocking, and its own domain staircases keep every link. A pair of one solid and one X cell does block; otherwise a wall that meets the domain edge diagonally would leak at its end. With `edge = 'rigid'`, X cells are rigid blockers like any other.

On tri, all six links can be blocked. The (1,0) link is blocked by two cells √3 apart, one above and one below its midpoint. So a "dotted" vertical wall (a solid cell in every second row) seals on tri. That is the pattern behind the research note's 17% leak.

### 2.3 Ring cells

- **Full square (`sq_full()`), any bc:** the ring is the bc's.
  - `edge = 'rigid'` does **nothing** there, because there is no mask edge. `bc = 'neumann'` is the rigid box.
  - A ring cell inside the domain is F. Under Dirichlet it holds 0, so it acts pinned; under Mur it is the absorbing ring.
- **Neumann ring next to a solid (new rule, port-only).** A Neumann ring cell copies its *partner*: the interior neighbour for a side cell, the diagonal neighbour for a corner. When the partner is not free, the ring cell takes the partner's class. The ghost of a rigid cell is then rigid, so the Neumann ring stays an exact mirror where a solid touches it.
  - Measured: energy is conserved with rigid and pinned solids on a Neumann ring, to 4.8e-15 (sq9, slab medium) and 1.9e-15 (sq5, lens) over 20,000 steps (`walls_tests_MIRROR.py` test 2).
  - **09/10/2026 (section 14.2):** the same ghost rule now also applies to the exterior X with a pinned edge (a shape clipped by the array), the user's decision; the web app's ring was asymmetric there. *Edited by Claude Opus 5.5 — 09/10/2026.*
  - **Review fix (06/10/2026, section 14):** the partner rule covers a solid AT the partner (1, t), not a solid ON the ring cell (0, t). On sq9 a diagonal link from (1, t) into the ring reads the ghost of (1, t ± 1); its mirror read, from (1, t ± 1), goes through (0, t). With (0, t) solid the operator was asymmetric and the run grew without bound (max|u| 1e12 after 200,000 steps). Now a sq9 diagonal link into the Neumann ring whose common neighbour on the ring is not free is blocked like that neighbour (pinned-blocked for P; dropped for R, and for X with a rigid edge). *Edited by Claude Opus 5.5 — 06/10/2026.*
  - Mur and Dirichlet ring cells stay F.
- **Triangular array ring:** for the web app's shapes (and `unlicensed`) always clamped, so its cells are X. With `edge = 'rigid'` that ring is rigid.
  - *05/10/2026:* on the `array` shape the bc rules the ring instead: clamped (Dirichlet, X pinned), rigid (Neumann, X rigid), the rhombic torus (no ring), Mur falls back to clamped. Section 13.
- **Torus (`bc = 'periodic'`, sq):** there is no ring. Words and blocking wrap across the seam. Energy with solids across the seam: 5.2e-15 over 20,000 steps (test 2).
  - *05/10/2026:* the triangular `array` with `bc = 'periodic'` is the rhombic torus, with the same wrapping (section 13.3).

---

## 3. Data and kernels

### 3.1 Class codes `C` and link words `W` (device, `ti.ndarray`)

These arrays exist only while `walls_needed()` is true: a solid exists, or `edge = 'rigid'` on a masked domain. Otherwise both are freed (`None`).
- **`C`** (dtype `MASK_DT`: u8, or i32 on Vulkan): bits 0–1 hold the class (`CL_F`/`CL_P`/`CL_R`/`CL_X` = 0/1/2/3); bit 2 (`CL_FREE`) is the cell's own mask M.
- **`W`** (`WORD_DT`: **u16** on CUDA and CPU, **i32** on Vulkan): one word per cell.

| bits | meaning |
|---|---|
| 0 | free (= M) |
| 1 + k | link k is DROPPED (the kernel reads u_i) |
| 9 + z | link k is PINNED-BLOCKED (the kernel reads 0). z = k − 4 for the sq9 diagonals (k = 4–7); z = k on tri (k = 0–5) |
| 15 | tri only: one of links 0–3 is blocked (for `main_region`) |

Link order is the kernels' (and the JS's) summation order. sq: (−1,0) (+1,0) (0,−1) (0,+1) (−,−) (+,−) (−,+) (+,+). tri: (−1,0) (+1,0) (0,−1) (0,+1) (+1,−1) (−1,+1).

A free cell whose word is exactly 1 is **plain**: it computes today's update. A link to a pinned cell or a pinned exterior stays plain, because the neighbour holds 0.

**Why u16 on CUDA:** with an i32 word, the walls kernel cost CUDA f32 +23–27% per step at N = 8193 (`walls_bench_MIRROR.py` run 1). That case is bandwidth-bound: the 1-byte mask became 4 of the ~17 bytes a cell moves per step. With u16 it costs +10–12%. Vulkan keeps i32, because it misreads 8-bit storage (the `MASK_DT` comment) and 16-bit is untested there.

### 3.2 The step

`_steps` / `k_run` / `k_run_rec` take a new template flag `walls`.

**`walls = False`** compiles exactly the code of 03/10/2026, and `M` is the mask. The parity test (`walls_tests` "parity with the 03/10 core", 10 configurations) measures:
- fields, peak and probe histories bit-identical to the BeforeCLAUDE core;
- the energy history within 1e-14. Not bit-identical: see section 7.4.

**`walls = True`** passes `W` in `M`'s argument slot, so the argument count is the same and so is the launch cost.
- Each neighbour goes through `_wv`: the neighbour's value (active link), 0 (pinned-blocked) or u_i (dropped). The result feeds the same formula as today.
- On sq5 and tri a plain cell's arithmetic is exactly today's.
- **sq9 always uses the self-consistent form on the walls path**, (2/3)(e − 4u) + (1/6)(dg − 4u), in f64 too. The web app's order, (2/3)e + (1/6)dg − (10/3)u, leaves −4.4e-16·u on a constant field in f64. That is a tiny systematic mass term. With pinned walls it does no harm. But a closed rigid cavity has a zero mode (its mean), and that term drives it.
  - Measured (`ScratchCLAUDE/Round4CLAUDE/WallsBuildCLAUDE/cavity_dc_kernel_MIRROR.py`): the sq9 cavity's DC drifted **1.1e-9** in 5000 steps with the web app's order, and **6.9e-14** with the consistent form. sq5 and tri, which have no bias, drift 3.6e-12 and 8.1e-12.
  - Only the walls path changed. The web app's path keeps its order.
- The masked-ring pass and the slit drive read the free bit (bit 0) instead of `M`.

### 3.3 Building the words

- **Full build (`_walls_full`).** The host computes the class codes in row chunks, using `mask_np = domain AND type == 0`. It then uploads them, and **one** Taichi pass (`k_walls_full`) writes every word and the two counters.
  - Cost: 12 ms at N = 2049 and 178 ms at N = 8193 (`rebuild_time_MIRROR.py`, sq9 plate, CUDA).
  - It runs at `build_domain` (every reset and Co change: `set_co` 3.26 → 3.42 s at N = 8193, +5%), `update_solids(None)`, `set_solid_np` and `set_edge`.
  - It also runs whenever `step()` finds the words built for another lattice, stencil, bc or edge (`_walls_key`).
- **Painted box (`_walls_patch`).** One kernel, `k_walls_patch`, makes three passes:
  - it takes the old words of the box + 2 out of the counters;
  - it writes the class patch (box + 1), the device mask (its free bit) and zeros all three time levels where the cell is not free. This is k_patch_mask's job, so a segment is still **one launch and one upload**;
  - it rebuilds the words of the box + 2 and adds them back to the counters.
  - A word depends on cells within one step. A Neumann ring cell's class depends on its partner, one step away. On a torus the word box wraps and is capped at N per axis.
- **Counters** (`n_wall_cells`, `n_blk4`) are read from the device **lazily**: a painted segment only marks them dirty. The sync happens at the next `step()`, `walls_active()` or `main_region()`, not once per segment.
- `walls_active()` is `W is not None and n_wall_cells > 0`. `n_wall_cells` counts the free cells that have a dropped or pinned-blocked link.

### 3.4 Other kernels

- **Recorder (`rec_fold`, in `k_run_rec` and `k_stats`).** On the walls path it sums over the free bit. The forward differences gx, gy run along links (+1,0) and (0,+1): along a dropped link the difference counts 0, along a pinned-blocked one −u (a link to a zero).
  - So a rigid wall stores no "potential". A constant field in a closed rigid box records E = 0; the same box pinned records E = 55.5 (`walls_tests` "recorder on the walls path").
  - The web app's path keeps its arithmetic.
- **`set_co(..., rescale=True)`** uses `k_rescale_w` (the walls operator) on the walls path.

---

## 4. Readers of the mask

`mask_np` and the device `M` keep their meaning: 1 = free. A rigid cell is not free, so every host reader already treats rigid cells as solid:
- source snapping (`snap_source`: a source requested in a rigid solid moves out; `walls_tests` test 14);
- the beam and packet aperture skip;
- `pulse_clip_fraction`;
- vertex placement;
- seeding and `drop_pulse` (`U[mask == 0] = 0`);
- the renderer (`k_image`, `k_verts` read `M`);
- the ledger's `k_modecorr`.

Time levels are zeroed in every non-free cell: at seeding, by `k_zero_masked`, and by the patch kernel.

**`main_region()`** uses the walls' links when the walls kernel is active. Corner blocking can seal a letter's counter that plain 8- or 6-neighbour labelling would join to the plate.
- A link with a free common neighbour is redundant, because the two links through that neighbour are active.
- So the method labels with the 4-neighbour structure, then joins the components that an active corner link (an unblocked X–X pair) connects. This is exact for sq5 and sq9, and for tri when no tri link 0–3 is blocked.
- Otherwise (tri with `n_blk4 > 0`) it uses an exact refined-grid labelling: 4× the cells, about 1.3 GB and several seconds at N = 8193. That case is rare.
- Checked against the connected components of the operator's own links: equal on sq9 (a one-cell diagonal diamond) and tri (a dotted-wall box, the refined path), both types (`walls_tests` test 13).

---

## 5. The solids module

- **`SolidObject.wall`** and **`Stroke.wall`**: `'pinned'` (default) or `'rigid'`. An unknown value raises `ValueError`.
- `begin_stroke(x, y, radius, erase=False, wall='pinned')` and `add_stroke(points, radius, erase=False, wall='pinned')`. A stroke carries the brush's type. An eraser frees cells.
- **`SolidScene.plate_wall`** (`'pinned'` default) sets the type of the Unlicensed plate's lettering and screws. Change it with `set_preset(plate_wall=...)`, which returns `None` (redo the whole grid).
- **Rasterisation yields the type map** (uint8: 0 free, 1 pinned, 2 rigid). Objects go first, in order, then strokes, in order; each overwrites what it covers. With pinned walls only, this is the old union/subtract mask, bit for bit.
- **Scene files, format version 2**, add the objects' and strokes' `wall` and `plate_wall`. Version-1 files load with every wall pinned, which is what they meant. The round trip is checked (`walls_scene_MIRROR.json`, `solids_tests` test 10).
- `Membrane.solid_np` **is the type map**. `solid_np != 0` means solid, and `.astype(bool)` still works. `set_solid_np` accepts a type map: a bool, or any non-zero value other than 2, means pinned; 2 means rigid.

---

## 6. Helpers (core)

| function | what it does |
|---|---|
| `Params.edge`, `Membrane.set_edge(edge)` | switch the mask edge without a reset (the words are rebuilt, the field kept) |
| `walls_needed()`, `walls_active()` | are the words kept? does the step run the walls kernel? |
| `walls_info()` | dict for a panel: edge, active, non-plain cells, pinned / rigid solid cells inside the domain, blocked tri links, whether the edge applies. Two whole-grid counts, ~0.1 s at N = 8193: cache it. |
| `unknowns()` | bool [i, j]: the cells the stencil updates |
| `operator()` | the solver's L as a scipy CSR over the whole grid, built from the **device** words. Ring resolved: Dirichlet and the tri clamp read 0, Neumann reads the partner, the torus wraps. Mur raises `ValueError`. For tests, N up to ~1000. |
| `leapfrog_energy(L=None)` | the exact discrete energy of the undamped leapfrog (below) |

**The energy.** E^{n+½} = Σ_unknowns (u^{n+1} − u^n)_i² / CC_i − ⟨u^{n+1}, L u^n⟩. Derivation:
- the scheme is u⁺ − 2u + u⁻ = CC_i (L u)_i;
- multiply by (u⁺ − u⁻)_i / CC_i and sum. The per-cell weight 1/CC_i is what makes it hold in a medium;
- the left side telescopes; L = Lᵀ turns ⟨u⁻, L u⟩ into ⟨u, L u⁻⟩.

It is a norm only inside the CFL limit. It is not conserved with drives or damping.

---

## 7. Measured

### 7.1 `walls_tests_MIRROR.py` (64 checks)

**ALL PASS on CUDA f64, CUDA f32 and Vulkan f32** (logs: `ScratchCLAUDE/Round4CLAUDE/LogsCLAUDE/final_walls_*_MIRROR.log`). The numbers below are CUDA f64.

| # | test | result |
|---|---|---|
| 1 | operator == `walls_lib.build_L` on the mixed hard-case scene (left objects pinned, right rigid): sq5 / sq9 full square, sq5 / sq9 circle with rigid edge, tri circle and hexagon with rigid edge | max \|L − L_ref\| = **0.0**; after 32 steps max\|u − u_ref\|/max\|u_ref\| 1.5e-15 (sq5), 2.9e-14 (sq9), 2.3e-14 (tri). f32: within 1e-5 |
| 2 | exact energy over 20,000 steps at 0.95 C_max, pinned + rigid mixed: sq5 / sq9 Dirichlet, tri circle, sq9 Neumann + slab and sq5 Neumann + lens (solids on the ring), sq9 torus (solids across the seam), tri hexagon with rigid edge + lens | max\|E − E0\|/E0 = **1.9e-15 to 5.2e-15**. f32: 1.6e-7 to 2.2e-6. (Solids next to the ring, not on its row: a solid ON the sq9 Neumann ring row broke the energy until the review fix of 06/10/2026, section 14.) |
| 3 | leak through one-cell walls (straight, diagonal, anti-diagonal; both types; three lattices; driven cell, 1500 steps; max over the 3 levels beyond the wall) | **0.0** in all 18 cases |
| 4 | axis-wall reflection vs the image solution (plane pulse, 100 steps) | \|wall − image\| ≤ 1.2e-13. Reflected peak **+0.4954** (rigid, even mirror on the rasterisation line), **−0.4954** (pinned, odd mirror on the solid row's centres). tri ±0.4966 |
| 6 | 0.995 C_max with 25% random solids (half pinned, half rigid), noise start, 20,000 steps | stable: max\|u\| 3.7 → 4.5 (sq5), 3.7 → 5.5 (sq9), 4.8 → 6.4 (tri); energy drift ≤ 9.7e-16 |
| 7 | disc eigenfrequencies from the device operator | the eigenvalues equal `walls_lib`'s (Pcb / R0) to 1e-9. R·\|err\| inside the research note's §3.8 bands, e.g. sq5 pinned 0.27–0.37, rigid 0.008–0.140 |
| 8 | DC of a 64-cell counter inside a rigid ring, seed at rest | **6.9e-15 / 1.3e-14 / 5.6e-15** after 100 steps (sq5 / sq9 / tri); ≤ 8.3e-12 over 5000 steps. Bounds: 3e-14 and 1e-10 (section 8) |
| – | incremental painting == full rebuild: objects and strokes of both types, an eraser, a pinned → rigid edit, across the array ring, the Neumann ring, the torus seam and a rigid edge (7 cases, 12 updates each) | type map, device mask, classes, words and counters **all equal** |
| – | parity with the 03/10 core (10 configurations: four bcs, slit, lens, tri, pinned objects on sq9 and tri, the sq5 plate, `edge = 'rigid'` on the full square) | fields, peak and probe histories **bit-identical**; energy history ≤ 5.5e-16 (CUDA) |
| – | `edge = 'rigid'` on masked shapes, 20,000 steps (sq5 circle, sq9 pentagon, sq9 square r 0.7, tri circle / hexagon / triangle, sq9 plate with pinned lettering, tri plate with rigid lettering) | max\|E − E0\|/E0 **2.1e-15 to 1.3e-14**. The field differs from the pinned edge by 0.13–0.34. (All with the Dirichlet bc. A sq9 shape clipped by the array with the Neumann bc was asymmetric until the review fix of 06/10/2026, section 14.) |
| – | recorder, `main_region`, snapping, Co rescale, Mur + rigid, `set_edge` | see sections 3.4 and 4 |

### 7.2 Cost per step (`ScratchCLAUDE/Round4CLAUDE/WallsBuildCLAUDE/walls_bench_MIRROR.py`)

Setup:
- circle domain r 0.95, ~3% of the cells in separated axis-aligned rectangles;
- **base** = all rectangles pinned. No link is blocked, so this is today's masked kernel;
- **walls** = every second rectangle rigid;
- **edge** = no rectangles, `edge = 'rigid'`.

Each value is the best of 3 runs. The GPU may have been shared with another agent's jobs, so the timings are noisy. Three bench runs were made: run 1 with an i32 word (CUDA f32 at N = 8193: +23–27%, which is why the word became u16), run 2 with u16, and run 3 with u16 in reverse timing order (same within ~1%, so there is no ordering bias). Run 2, walls ÷ base:

| backend | N | sq5 | sq9 | tri | edge rigid (sq5 / sq9 / tri) |
|---|---|---|---|---|---|
| CUDA f64 | 2049 | 0.158 → 0.163 ms (×1.034) | 0.179 → 0.192 (×1.075) | 0.147 → 0.153 (×1.039) | ×1.04 / 1.07 / 1.05 |
| CUDA f64 | 8193 | 2.630 → 2.763 (×1.051) | 2.931 → 3.055 (×1.042) | 2.420 → 2.530 (×1.045) | ×1.06 / 1.05 / 1.04 |
| CUDA f32 | 2049 | ×1.007 | ×0.992 | ×0.996 (launch-bound) | ×0.99 / 0.98 / 1.00 |
| CUDA f32 | 8193 | 1.339 → 1.477 (×1.103) | 1.323 → 1.472 (×1.113) | 1.197 → 1.345 (×1.123) | ×1.11 / 1.12 / 1.13 |
| Vulkan f32 | 2049 | ×1.007 | ×0.989 | ×1.010 | ×1.01 / 0.99 / 1.01 |
| Vulkan f32 | 8193 | 1.881 → 1.882 (×1.001) | 1.870 → 1.893 (×1.012) | 1.724 → 1.726 (×1.001) | ×1.00 / 1.01 / 1.00 |

With the recorder on (N = 2049), the walls kernel was the same speed or faster than base: ×0.84–0.97 on CUDA, ×1.00–1.02 on Vulkan, in both timing orders. I did not find why.

### 7.3 Painting, rebuilds, memory

- **Painting** (`walls_bench_MIRROR.py`, CUDA f64; sq9 plate, rigid brush of 3 cells, walls kernel on): median **2.50–2.62 ms** per segment at N = 2049 and 8193 (runs 2–3). One long diagonal segment (~3700 cells at 45°) takes **69 ms** at N = 8193 and 9 ms at N = 2049.
  - `solids_tests` (sq5 plate, words kept, parity kernel): 2.88 / 2.70 ms, long segment 72 ms. The 01/10 log (`ScratchCLAUDE/FinalCLAUDE/solids_cuda.txt`) had 2.30 / 2.51 ms and 50 ms.
  - Before the patch was merged into one launch it was 3.6–3.75 ms and 114 ms.
- **Full word build:** 12 ms at N = 2049, 178 ms at N = 8193 (`rebuild_time_MIRROR.py`).
- **Memory while the words are kept:** CUDA +3 bytes per cell (C u8 + W u16): f64 33 → 36 B/cell, f32 17 → 20. That is +201 MB at N = 8193. Vulkan +8 bytes per cell (i32 + i32): +537 MB at N = 8193. Without solids and with a pinned edge, nothing is allocated.

### 7.4 The other suites (after the last code change; `LogsCLAUDE/final_*`)

- `crosscheck_js_MIRROR.py` **27/27** on CPU f64, CUDA f64 and Vulkan f32. The CPU same-seed column is identical to the 01/10 log, config by config (20 exact zeros, 7 at ~1e-15).
- `identity_tests_MIRROR.py`: ALL PASS on CUDA f64 and Vulkan f32.
- `solids_tests_MIRROR.py`: ALL PASS on CUDA. Test 3 was changed: the one-cell diagonal wall now **seals** on every lattice (0.0) instead of leaking on sq9 and tri.
- `beam_tests_MIRROR.py` 34/34, `scale_tests_MIRROR.py` 16/16, `app_tests_MIRROR.py` 95/95, `crosscheck_ledger_MIRROR.py` PASS (0 status / 0 text mismatches, 15/15).
- Headless shots render: `ScratchCLAUDE/Round4CLAUDE/WallsBuildCLAUDE/shot_scene_rigid_MIRROR.png` (a scene with rigid and pinned objects) and `shot_unlicensed_edge_rigid_MIRROR.png` (`--set shape=unlicensed edge=rigid`, 3D).

**The recorder's energy history is not run-to-run deterministic.** Its sums go through atomics, whose order varies. The same core differs from itself by up to 5.5e-16 on CUDA and CPU, and ~2e-7 on Vulkan with its f32 accumulators (`recorder_determinism_MIRROR.py`). So the parity test compares the energy history to 1e-14 (1e-6 on Vulkan) and everything else bit for bit.

---

## 8. Deviations

**From the web app** (port-only; the web app has no walls):
1. **Corner blocking changes pinned scenes that have blocked links.** Without blocking these links leaked. On tri, the Unlicensed plate's lettering has 8–20 such cells at every N from 161 to 4097, so on tri the plate now runs the walls kernel and its field differs where the leak was (`plate_blocking_MIRROR.py`). The sq9 lettering has none at rotation 0 and radius 1 for N = 161–4097, the cases tested then. *Review 06/10/2026:* at other rotations, radii or N it does: 42 of 60 configurations (N 97–1025, rot 0/7/30/45°, rad 0.8/1/1.2) block 2–51 cells and run the walls kernel, e.g. N 97 at rot 0 (`ScratchCLAUDE/Round4CLAUDE/ReviewCLAUDE/rv_plate_sq9_MIRROR.py`). That is the intended sealing; the plate has no web-app counterpart. *Edited by Claude Opus 5.5 — 06/10/2026.* A domain edge alone (`edge = 'pinned'`) never blocks.
2. The recorder's potential on the walls path skips dropped links, and counts −u along pinned-blocked ones (section 3.4).
3. The sq9 operation order on the walls path is the consistent form, also in f64 (section 3.2).
4. The Neumann ring partner rule (section 2.3), and since 06/10/2026 its ghost rule for the sq9 diagonals (section 14), extended on 09/10/2026 to the exterior of a shape clipped by the array with a pinned edge (section 14.2; harness PATCH `neumann-clip`). *Edited by Claude Opus 5.5 — 09/10/2026.*

**From the task specification / research note** (each with its measured reason):
1. **Cavity DC "to 1e-14" holds only for ~100 steps; test 8 asserts 3e-14 after 100 steps and 1e-10 over 5000.** A closed rigid cavity's mean is a zero mode, so its rounding noise is integrated twice and grows like ~n^1.5.
   - No floating-point leapfrog does better. The independent numpy reference (`walls_lib`, `cavity_dc_reference_MIRROR.py`) drifts **9.3e-13 (sq5), 1.2e-9 (sq9), 9.0e-10 (tri)** in 5000 steps on the same cavity. The kernel: 3.6e-13 / 5.7e-13 / 8.3e-12 (test 8).
   - After 100 steps the kernel measures 5.6e-15 to 1.3e-14. That is the predicted rounding-walk size, so a bound of 1e-14 sits on the noise.
   - The web app's own Neumann box shows the same behaviour on the parity path (`neumann_box_dc_MIRROR.py`, interior sum): f64 sq5 1.6e-12 and sq9 1.1e-9 in 5000 steps (the systematic sq9 order); f32 1.0e-3 (sq5) and 3.7e-3 (sq9).
2. **`W` is u16 on CUDA and CPU**, not i32 (section 3.1). It is i32 on Vulkan, as asked.
3. **The parity check compares the energy history to rounding**, not bit for bit: its atomics are not deterministic between runs, even for the old core (section 7.4).
4. **Rule R1 (test 5 of §5.5) is not built**, as scoped. On sq9 and tri, rigid axis-aligned walls therefore keep R0's tangential-moment loss: up to 72° phase error at 80° incidence, λ = 8 (research note §3.3).

---

## 9. The ledger with rigid walls (for whoever updates `ledger_MIRROR.py`)

*Done 05/10/2026 (round 4b): every point below is now in `ledger_MIRROR.py`; section 13.6 lists what changed.*

- **Energy row.**
  - A rigid wall does no work, so a closed scene with pinned and rigid solids, or a rigid edge, conserves the leapfrog energy exactly: drift ≤ 1.3e-14 over 20,000 steps (section 7.1). On the sq9 Neumann ring this needs the review fix of 06/10/2026 (section 14); the web app's own clipped shape with the Neumann bc does not conserve it (section 14.2, open).
  - The recorder's E on the walls path is the web app's formula with the dropped links left out. So it keeps the same quality as before (bounded, an O(Δt²) oscillation), and the row can stay graded.
  - The row's `port` text should say "pinned or rigid walls: both conserve the energy".
  - For an exact check, `m.leapfrog_energy()` gives the conserved quantity (host, not per step).
  - `edge = 'rigid'` with `bc = 'mur'` on a masked shape: the wall is the rigid mask edge, which is closed. `edge()` / the `edge-kind` fix should learn a rigid mask edge: it is conservative and does not absorb.
- **Boundary row ("fixed edge: the wave DOES come back").** It also holds for a rigid edge, but the echo is upright (R = +1) instead of inverted. If the row checks a sign, a rigid edge flips it.
- **Symmetry rows.** Their premise check (the free-cell mask is invariant under the orbit) must also check the **type map** (`solid_np`). A mirror-symmetric arrangement with asymmetric types (left pinned, right rigid) is not symmetric. The edge type applies everywhere, so it does not break symmetry.
- **Closed rigid cavities (letter counters with rigid lettering, a rigid edge with `src='pulse'`).** The mean of each cavity is a zero mode.
  - It is harmless for seeds at rest.
  - It random-walks with rounding: f64 ~1e-12 in 5000 steps; **f32 ~1e-3 to 1e-2 of the mean** in 5000 steps (test 8 on CUDA f32 / Vulkan).
  - A "peak" or "amplitude bounded" row in f32 over very long runs can see this as a slow offset. The web app's Neumann box already has it (section 8).
- **CFL row.** Walls never tighten the limit. That is proven for pinned and R0; for corner-blocked pinned walls it is measured (stable at 0.995 C_max with 25% random solids, test 6). `co_limit()` is unchanged.

---

## 10. Open issues

1. **Closed rigid cavities in f32:** the DC random walk above. A fix, if it ever matters: subtract the drift of each cavity's mean, using a component label pass when the solids change (research note §4.4).
2. **R1 and impedance walls** are not built. R1 needs a weight per in-wall link, which does not fit the bit word; impedance on tri is still open in the research.
3. **`main_region` on tri with blocked links 0–3** takes the refined labelling: ~1.3 GB and several seconds at N = 8193. It runs only when a source lands on a solid after the mask changed. A GPU labelling would remove that cost.
4. **Undoing a long stroke** re-rasterises its whole box, as before. The walls patch over a large box adds ~20 ms per 13M cells (the long-diagonal figures in 7.3).
5. ~~**Extension points for the triangular torus**~~ — done 05/10/2026 (section 13.3). What the list said:
   - `_word`, `_wc`, `k_walls_full` and `k_walls_patch` already wrap indices when `wrap` is set (the `_walls_key` torus flag). `_walls_key` sets that flag only for sq (`not tri and bc == 'periodic'`).
   - The tri step loop in `_steps` and the tri clamp pass do not wrap yet.
   - `_class_box` turns the tri ring into X; on a tri torus there is no ring, so that rule must be skipped.
   - `unknowns()` and `operator()` treat the tri ring as clamped.
6. `walls_info()` does two whole-grid counts per call (~0.1 s at N = 8193). The UI should cache it per mask version (`m._mask_ver`).
7. The **first** painted segment on a scene with no solids (and a pinned edge) builds every word once: ~0.18 s at N = 8193 (`rebuild_time_MIRROR.py`'s `_walls_full`). Later segments take the patch path. A device-side initialisation from `M` alone would remove that hitch.
8. ~~**Inherited from the web app (review, 06/10/2026):** the sq9 stencil with the Neumann bc and a shape CLIPPED by the array (e.g. a circle at radius 1.3) is unstable, with no solids and a pinned edge: section 14.2.~~ Fixed in the port on 09/10/2026 (the user's decision): section 14.2 and [v1_decisions_MIRROR.md](v1_decisions_MIRROR.md). *Edited by Claude Opus 5.5 — 09/10/2026.*

---

## 11. UI hand-off

**Status (05/10/2026): wired in the app (v1.0)**, as described in [app_MIRROR.md](app_MIRROR.md), section *Round 4 / v1.0*. Differences from the suggestions below: B on a triangular web-app shape changes nothing and flashes the note (it does not reset p.bc). The rigid colour is (0.22, 0.72, 1.00), with stripes in 2D. The square's Neumann ring and the square torus also get the overlay line (rigid line / dashed seam). The CLI has `--plate-wall` and `--brush-wall`. *Edited by Claude Opus 5.5 — 05/10/2026.*

**Parameters and calls**
- `m.p.edge` and `m.set_edge('pinned' | 'rigid')`: no reset needed; the field is kept.
- Objects: `so.SolidObject(..., wall='rigid')`; `sc.edit(idx, wall='rigid')` returns the box for `m.update_solids(box)`.
- Brush: `sc.begin_stroke(x, y, r, erase=False, wall=brush_wall)`. Keep a `brush_wall` setting in the walls panel, default `'pinned'`.
- Plate: `sc.set_preset(plate_wall='rigid')` then `m.update_solids(None)`.
- Readouts: `m.walls_info()` (cache it), `m.walls_active()`.
- Headless: `--set edge=rigid` already works. Suggested new CLI flags: `--brush-wall rigid` and `--plate-wall rigid`.

**What to show**
- In the walls panel:
  - an **edge** toggle (pinned / rigid). Grey it out with the note "full square: the boundary (B) rules" when `walls_info()['edge_applies']` is False;
  - a **brush type** toggle;
  - a **wall type** per object row;
  - a **plate lettering** type when the shape is `unlicensed`.
- Counts: pinned cells, rigid cells, and "walls kernel ON (n cells)" or "off (web app path)".
- A one-line explainer: *pinned: u = 0, the echo comes back inverted (a membrane held down). Rigid: no slope, the echo comes back upright (sound off a hard object).*
- The research note §5.1 suggests the default for "objects in the wave" could follow the scene: rigid for sound or water, pinned for a membrane. Today everything defaults to pinned, matching the web app.

**Colours and legend** (house style: dark purple / neon)
- On the device, the renderer can read the class from `m.C` (when it is not `None`): `(C & 3) == 1` is pinned, `== 2` is rigid, `== 3` is exterior. While `m.C` is `None`, every solid is pinned.
- Suggested colours:
  - pinned: the current solid colour (light grey; `--solid-col`);
  - rigid: a cool neon cyan-blue, or the same grey with a thin cyan outline. Keep the two distinguishable for colour-blind viewers by lightness as well as hue;
  - a rigid domain edge: a thin cyan line along the mask edge in 2D, or tinted rim vertices in 3D.
- Legend chips: "■ pinned (u = 0)", "■ rigid (∂u/∂n = 0)", "— rigid edge".
- In 3D a rigid wall is still drawn at height 0. A rigid face is an antinode (2× peak on contact), so a short note helps users who expect the surface to "stick" there.


**Triangular lattice modes (round 4b, 05/10/2026; section 13)**

*Parameters and calls*
- **Shape `array`:** `p.shape = 'array'` (it is already in `mc.SHAPES`, so G cycles to it), then reset.
  - Radius and rotation do not apply: grey their sliders out while `p.shape in mc.FIXED_SHAPES`. "snap conforming" does nothing either.
  - On the square lattice it is the full square; on the triangular one, the whole rhombic array.
- **`m.bc_info()`:** one call per panel refresh is cheap (no grid work). It returns a dict:
  - `requested`, `effective`: `p.bc`, and what runs;
  - `fallback` (bool), `reason` (text or None);
  - `ring`: a one-line description, e.g. "rigid ring (link removal, zero normal slope)", "rhombic torus (wraps along both lattice vectors)";
  - `options`: the bc values that act here;
  - `unavailable`: {bc: reason};
  - `edge_applies` (bool), `torus` (bool).
- **The bc options per lattice and shape:**

| lattice / shape | dirichlet | neumann | periodic | mur |
|---|---|---|---|---|
| square, full square or `array` | clamped ring | free ring (ghost copy, the rigid box) | torus | Mur ring |
| square, masked shape | the bc acts on the array ring only, where the shape reaches it (the web app) | same | same (wraps where the shape is clipped) | same |
| triangular, `array` | clamped ring | **rigid ring** (link removal) | **rhombic torus** | **unavailable** → clamped (`fallback`, `reason` = `mc.TRI_MUR_REASON`) |
| triangular, masked shape | clamped ring (the web app) | unavailable → clamped (`mc.TRI_MASKED_REASON`) | unavailable → clamped | unavailable → clamped |

- **`edge`:** it applies only while `bc_info()['edge_applies']` (a masked shape). On `array` the bc rules the ring.
- **Readouts:**
  - `m.walls_info()['ring']` mirrors `bc_info()['ring']`. `m.mode_note()` is None, or why the drum-mode source is not an exact eigenmode here. `m.hist['Etri']` holds the triangular energy series (triangular lattice only).
- **Headless:** `--set lattice=tri shape=array bc=periodic` (or `bc=neumann`) works today; `--scene` adds solids.

*What to show*
- **The boundary line** (`membrane_app_MIRROR.py` already prints the effective bc, and "(mur n/a here)" on a fallback). In the panel:
  - **B** should cycle only `bc_info()['options']`. Show the unavailable values greyed out, with their reason as a tooltip or an amber line. For example, on the triangular array, "Mur: not available on the triangular lattice yet (unstable above Co 0.5) — clamped ring". On a triangular masked shape: "boundary control acts on the 'array' shape only — the ring stays clamped".
  - Below it, the `ring` text, e.g. "rhombic torus (wraps along both lattice vectors)".
- **The edge toggle:** grey it out with "the array: the boundary (B) rules" while `edge_applies` is False (this replaces the full-square-only note above).
- **Colours (house style):**
  - The rigid ring takes the rigid edge's cyan line (section 11 above) along the four ring rows.
  - The rhombic torus can draw its seam as a thin dashed dim-violet line along the array edge. A short note helps: "waves leave one side and enter the opposite one".
  - The clamped ring keeps the current look.
- **The drum-mode source:** when `m.mode_note()` is not None, show it in amber under the source buttons ("not an exact eigenmode here: …"). The source still seeds.
- **The ledger panel** needs nothing new: the rows carry their reasons (n/a + text), and the `port` column says "FIX tri-energy" where the triangular functional is graded.
- **The alignment panel:** for `array` it already reads "fixed — rotation does not apply". The prev / next / optimal buttons flash that message.

---

## 12. Files

| file | what |
|---|---|
| `membrane_core_MIRROR.py` | classes, words, kernels, `edge`, helpers (sections 2–6); deviation added to the module docstring |
| `solids_MIRROR.py` | wall types, type-map rasterisation, `plate_wall`, scene format v2 |
| `walls_tests_MIRROR.py` | the tests of section 7.1 (`--arch cuda\|vulkan\|cpu --f32 --quick --only 1,2,...`) |
| `solids_tests_MIRROR.py` | test 3 updated (the one-cell diagonal wall now seals) |
| `ScratchCLAUDE/Round4CLAUDE/WallsBuildCLAUDE/` | `walls_bench_MIRROR.py` (cost per step and painting), the probes behind section 8 (`cavity_dc_reference`, `cavity_dc_kernel`, `neumann_box_dc`, `recorder_determinism`, `rebuild_time`, `plate_blocking`, all `_MIRROR.py`), the 03/10 backups `*_before_MIRROR.py`, a v2 scene file and two shots |
| `ScratchCLAUDE/Round4CLAUDE/LogsCLAUDE/` | every log quoted here (`final_*` = after the last code change; `walls_bench_run1/2/3_MIRROR.log`) |
| `NotesCLAUDE/progress_walls_build_MIRROR.md` | the progress log |
| `tri_modes_tests_MIRROR.py` | round 4b: the tests of section 13.7 (`--arch cuda\|vulkan\|cpu --f32 --quick --only 1,2,...`) |
| `ScratchCLAUDE/Round4CLAUDE/TriModesCLAUDE/` | round 4b: the 04/10 backups `*_before_MIRROR.py`, the probes behind section 13 (`torus_energy_probe`, `tri_energy_row`, `tri_modes_bench`, `smoke`, all `_MIRROR.py`), the batch script `tri_regression_batch_MIRROR.sh`, a seam scene `tri_seam_scene_MIRROR.json`, the one-off patch scripts |
| `ScratchCLAUDE/Round4CLAUDE/LogsCLAUDE/tri_*` | round 4b: every log and shot quoted in section 13 (`tri_batch_status_MIRROR.txt` = the regression's exit codes) |
| `NotesCLAUDE/progress_trimodes_MIRROR.md` | round 4b: the progress log |
| `ScratchCLAUDE/Round4CLAUDE/LogsCLAUDE/def_bench_*` | round 4b: the cost-per-step runs of section 13.7 (taken 05/10/2026 by the round 4c step, `NotesCLAUDE/progress_round4_defaults_MIRROR.md`) |

---

## 13. Triangular lattice modes (round 4b, 05/10/2026)

*Generated from scratch — Claude Opus 5.5 — 05/10/2026*

The user asked why the triangular lattice has no boundary options. In the web app, and in the port until now, the triangular array's outer ring was always clamped, whatever `bc` said. This round adds the options: a domain shape that **is** the whole array, and on it a clamped ring, a rigid ring and a rhombic torus.

### 13.1 What is new

- **Shape `array`** (appended to `SHAPES`, port-only): the whole lattice array is the domain. Rotation and radius never apply (`FIXED_SHAPES`), and domain-anchored objects do not turn with `p.rot` there (`Membrane.dom_rot()` is 0).
  - **Square lattice:** it *is* the full square. `sq_full()` is True, so it runs the same code path as `square` at radius ≥ 0.999. Fields, histories, source and vertex cells, clip fraction and link words are bit-identical (test 6, 10 configurations).
  - **Triangular lattice:** the full 60° rhombus of the array. Every cell is inside except the outer ring, which is the boundary. On the torus every cell is inside.
- **The bc control on the triangular `array`:**

| `p.bc` | what runs | the ring cells |
|---|---|---|
| `dirichlet` | the clamped ring (the web app's rule) | X, pinned |
| `neumann` | a **rigid ring** by link removal: the links into the ring are dropped | X, rigid |
| `periodic` | the **rhombic torus**: the six neighbours (i±1, j), (i, j±1), (i+1, j−1), (i−1, j+1) taken mod N | ordinary cells; words and corner blocking wrap |
| `mur` | **not available yet**: falls back to the clamped ring; `bc_info()` says so | X, pinned |

- **Mur on the triangular lattice** waits for the absorbing-boundary round: the research note measured the first-order rule unstable on the triangular lattice above Co 0.5 ([absorbing_boundaries_MIRROR.md](absorbing_boundaries_MIRROR.md)). The fallback is exact: `array` + `mur` runs the clamped ring bit for bit (test 9d).
- **Nothing the web app can make changes.** On the triangular lattice the web app ignores `bc`; so does the port for every shape except `array`. Every web-app shape keeps the clamped ring with any `bc`. Test 8 compares against the 04/10 core: every triangular configuration of `crosscheck_js` × the four bcs, the triangular plate, rigid edges, solids and square-lattice configurations. All 40 are bit-identical (fields, peak and probe histories, source cells, vertices, clip fraction).

### 13.2 How bc and edge combine (kept simple for the UI)

- **On `array`, either lattice:** `edge` does not apply, because there is no mask edge. The **bc rules the ring**, as on the full square (section 2.3). `bc_info()['edge_applies']` is False.
- **On a masked shape:** `edge` rules the mask edge, and on the triangular lattice also the clamped ring. On the triangular lattice `bc` does nothing; on the square lattice it rules only the array ring, where the shape reaches it (the web app's rule).
- **Periodic + a masked shape:**
  - On both lattices, a shape inside the array runs exactly as with Dirichlet (bit-identical). The interior update is the same, and the cells across the seam are exterior, so they hold 0.
  - The lattices differ only for a shape **clipped** by the array (radius > 1). The square lattice wraps through the clipped parts (the web app). The triangular lattice keeps the web app's clamped ring, for parity.
  - So the triangular torus is the `array` shape only. This is deliberate: the alternative would change web-app configurations such as a triangular circle at radius 1.3 with `bc = periodic`.

### 13.3 Core changes (`membrane_core_MIRROR.py`)

- **New helpers:**
  - `full_array()`: `sq_full()` or the triangular `array`.
  - `torus()`: periodic on the square lattice (any shape), or the triangular `array`. `tri_torus()`.
  - `bc_effective()`, `bc_info()` (section 11), `dom_rot()`, `mode_note()` (13.5).
- **The domain mask:** the triangular `array` has a ring of 0 unless it is a torus. `step()` notices a `p.bc` change between torus and ring made without a reseed: it rebuilds the domain and zeroes the cells that left it (test 9k).
- **The step:**
  - On the triangular lattice `step()` passes `BC_PER` for the rhombic torus and `BC_DIR` otherwise. The rigid ring is made by the words, which never read the ring.
  - `masked = not full_array() or has_solids`.
  - A new kernel branch `LAT_TRI` + `BC_PER` does one wrapped update over every cell, with the same arithmetic in the same order as the clamped branch. It has a plain and a walls form; the walls form uses `sum_tri_wi`, the neighbour sum with wrapped indices.
- **The words:**
  - The `_walls_key` torus flag is `torus()`; the rigid-exterior flag is `_erig()`: `edge == 'rigid'` on a masked shape, `bc == 'neumann'` on the triangular `array`, never on the full square.
  - `_class_box` skips the triangular ring rule on the torus.
  - `k_walls_full` and `k_walls_patch` already wrapped (section 10, item 5).
- **`set_co(rescale=True)`:** `k_rescale` gets `wrap = torus()`, and `k_rescale_w` uses the wrapped triangular sum (test 9c: 1.2e-16 against the formula, the seam cells included).
- **`unknowns()`, `operator()`, `leapfrog_energy()`:** every cell on the torus, with wrapped links. On the rigid ring the links into the ring are dropped. `operator()` equals an independent numpy operator exactly on all four bcs (test 9a).
- **The recorder** (`rec_fold` / `rec_sum`, HIST width 6 → 7). Both changes are port-only. The web app's `E` series is unchanged for every web-app configuration.
  - **Triangular lattice:** it also sums the third link direction's forward difference gz = u(i+1, j−1) − u (column 6). `m.hist['Etri']` = kin/(2Co²) + (gx² + gy² + gz²)/3 is the triangular operator's own Dirichlet form, −⟨u, L u⟩/2 for L = (2/3)Σ(u_n − u). The web app's E uses only gx and gy, the square lattice's functional. Test 9l matches numpy to 3e-16.
  - **Rhombic torus:** every cell is summed, with wrapped differences. The web app's interior-only sum left the seam rows out, and a wave crossing the seam read as a 2–3% drift: the "drift" of an impulse at N = 97 drops from 2.69e-2 to 1.84e-4 (`torus_energy_probe_MIRROR.py`). The square torus keeps the web app's interior sum, for parity.

### 13.4 Geometry of `array`

| piece | square lattice | triangular lattice |
|---|---|---|
| `dom_R()` | base_half − 1.5 (= `square` at rad 1) | base_half − 1.5 (the inradius less the square's 1.5 margin) |
| `inner_R()` (r_in: source sx/sy, scale mode, probes) | base_half = c | base_half = c·√3/2, the distance from the centre to each ring line |
| `shape_poly()` | the full square through the ring-cell centres (±c) | the 60° rhombus through the four ring-corner cell centres, CCW from cell (0, 0): edges along 0°, 60°, 180°, 240°, every edge a lattice row. It is the web app's rhombus at rtheta 60°, rot 30°, with R = c√3. |
| `vertices()` (vtx source) | 4 corners | 4 corners. Wall inset: 2·inset at the 60° corners, inset/(√3/2) at the 120° ones. Radial: inset along the ray. Measured within one snap (test 9f). |
| `clip_region()` | unchanged | unchanged (the rhombus fits inside) |
| `mode_box()` | (0, N−1) | (0, N−1), the index square, which is the rhombus |

### 13.5 Everything else that depends on the shape or the ring

- **Source placement:** sx and sy are fractions of r_in = base_half. sx = 1 lands at x = r_in; sy = 0.99 lands in the row next to the ring with no snap (test 9g).
- **`pulse_clip_fraction`:** the clamped and rigid rings cut the seed. The torus ring does not; only what leaves the array is cut. Measured with σ = 4 at sy = −0.97: torus 0.3224, rigid and clamped 0.4389 (test 9e). Seeds are not wrapped on any torus (the web app's `addBump` does not wrap the square torus either).
- **The mode source** still seeds sin·sin in index space, as the web app does on every domain. `mode_note()` gives None only where that is an exact eigenmode: the square lattice, `square` or `array`, a Dirichlet ring (full) or a pinned edge (masked), a uniform medium, no solids. On the triangular `array` it says *n/a*: the (1, −1) links couple i and j, so sin·sin is not an eigenmode of the 6-neighbour operator. The 60° rhombus has no closed-form discrete spectrum to seed instead.
- **Beam and packet:** the aperture keeps its cells inside [1, N−2] and the packet's seed is clipped at the array edge, with no wrap. A beam aimed at the seam crosses it as a wave; only the emitter is not wrapped. On the torus and on the rigid ring both run and stay finite (test 9j).
- **Solids and painting across the seam:**
  - Rasterisation does not wrap. An object or stroke beyond the array edge is clipped, not continued on the other side. This is the square torus' behaviour too.
  - The class codes, words and corner blocking do wrap.
  - Incremental painting equals the full rebuild on the torus, the rigid ring and the clamped ring: objects and strokes across both seams, both wall types, an eraser, a pinned → rigid edit (test 7).
- **`main_region()`, `nearest_free()`:** no wrap, as on the square torus. The labelling sees the array, not the torus; it only matters when a source lands on a solid.
- **`co_limit()`:** unchanged. C_max = √(2/3) for all three ring modes; stable at 0.995 C_max (test 5).

### 13.6 Alignment and ledger

- **`alignment_MIRROR.py`:**
  - Model `arr`: the edges are the array's rows, at 0/90° on the square lattice and 0/60° on the triangular one, with a fixed rotation. That gives 4/4 rows, ALL, on either lattice. On the triangular lattice it is the conformant 60° rhombus whose long diagonal lies at 30°.
  - Row 7c (the measured period): PASS, period 1 on every edge.
  - Row 7c-bis: n/a, "its walls ARE the array edge … governed by the boundary control" (+ "the rhombic torus has no wall").
  - `al_set_rot` refuses ("the lattice array is fixed"); the flavor pick is a tie (4/4 on both lattices), so the current lattice stays.
- **`ledger_MIRROR.py`:** the walls (section 9) and the new modes. Every web-app configuration keeps the web app's logic.

| row | rule |
|---|---|
| energy | Graded with rigid solids, a rigid edge and the rigid ring: they do no work. Port text: "pinned and rigid walls both conserve the energy". **FIX `tri-energy`:** on the triangular lattice the row grades `hist['Etri']`. Measured with `tri_energy_row_MIRROR.py` (1300 / 2500 steps): drift of E → Etri is 4.0e-2 → 1.1e-3 for the rigid ring (N = 97 pulse), 1.37e-1 → 9.3e-6 for the rigid-ring packet, 9.0e-3 → 1.1e-3 for the web app's rhombus 60° r 0.8 impulse, 1.28e-1 → 3.3e-3 for a packet in the hexagon. Without the fix, the rigid and clamped rhombic array FAILed at 2–4%. Symmetric shapes (n3, n6, circle) are unchanged: their fields are 6-fold symmetric, so Σgx² = Σgy² = Σgz² and the two values are equal. |
| sym8 | `array` on the square lattice is 8-fold like the full square. The premise with solids also checks the **type map** inside the domain when a solid is rigid, so left pinned / right rigid is n/a with the reason. |
| sym6 | The rhombic torus is 6-fold: its period lattice N·Z² of the triangular metric is invariant under (a, b) → (−b, a+b). The orbit is taken mod N (`k_orbit` wrap) and holds after the wave wraps: 6e-16 at 1300 steps. The clamped and rigid rhombus is only 2-fold: n/a with that reason. |
| conformance | `array`: (N−2)² lattice points (the ring excluded), N² on the torus: PASS. |
| mirror | `array` on the square lattice: mirror-symmetric. The triangular rhombus is not (n/a, "point-symmetric but not mirror-symmetric"); the rhombic torus is (graded, mod N). The premise with solids also checks the types. |
| boundary (probe) | Graded. Port text: "the edge is RIGID: the echo comes back upright (R = +1)", or "the rhombic torus: the wave comes back by wrapping". Mur on the triangular array: the edge-kind fix grades the clamped ring and names the reason. |
| wake, linearity | Mur on the triangular array is not a Mur ring (edge-kind), so wake is n/a and linearity "never settles". A rigid mask edge counts as closed. |
| staircase | Triangular `array`: exact by construction (the mask is the lattice; area = cells × √3/2). Square `array`: the full square's row. |
| light cone | The rhombic torus wraps with the hexagonal norm's nearest image (the six neighbouring images (±N, 0), (0, ±N), (±N, ∓N)). Containment PASSes after the cone meets itself (support radius 107 = 2N/3 at N = 161). |
| clearance (front rows) | The rhombic torus' seam counts like the full square's array edge: √3/2 × the index distance. |
| eigenmode, dispersion | n/a on the masked square with a **rigid edge** ("its eigenmodes are cos·cos"), and on the triangular `array` (the triangular reason). |
| spectra | `array` on the square lattice: the square's stated row. |

  `crosscheck_ledger_MIRROR.py` has 11 new port-only checks (O–W, 26/26 in all).

### 13.7 Measured

`tri_modes_tests_MIRROR.py` (full length; logs `ScratchCLAUDE/Round4CLAUDE/LogsCLAUDE/tri_trimodes_{cuda,cuda_f32,vulkan_f32}_MIRROR.log`). The values are CUDA f64; the f32 builds use f32 tolerances (energy 1e-4, symmetry 1e-4, chain 1e-5).

| # | test | CUDA f64 | CUDA f32 · Vulkan f32 |
|---|---|---|---|
| 1 | rhombic torus, exact energy, 20,000 steps at 0.95 C_max: plain / lens + pinned and rigid solids across every seam | 1.1e-15 / 1.5e-15 | 7.6e-7 / 6.1e-7 · 9.3e-7 / 4.8e-7 |
| 2 | translation invariance: seed (and solids) shifted by (37, −55), (64, 64), (−1, 1), 600 steps, plain and walls path | bit-identical (6/6) | bit-identical (6/6) · bit-identical (6/6) |
| 3 | 6-fold symmetry of a centred pulse: before the seam / after 1500 steps of wrapping | 3.0e-15 / 9.0e-15 | 1.6e-6 / 4.9e-6 · 1.8e-6 / 5.6e-6 |
| 4 | rigid ring, exact energy, 20,000 steps: corner pulse / lens + solids on the ring | 1.8e-15 / 4.3e-15 | 4.3e-7 / 4.6e-7 · 3.7e-7 / 1.6e-6 |
| 4 | plane pulse vs the 1D chain (rigid ghost = u, clamped ghost = 0), 53 columns the side rings cannot reach | 6.4e-15 both; reflected peak **+0.4966** (rigid, R = +1), **−0.4966** (clamped, R = −1) | 1.8e-6 / 2.2e-6 · 4.8e-6 / 4.8e-6; the same ±0.4966 peaks |
| 5 | 0.995 C_max, noise start, 20,000 steps: torus, rigid ring, each with 25% random solids | stable; max\|u\| 3.8 → 8.2 at most; energy drift ≤ 1.3e-15 | stable; energy drift ≤ 4.4e-7 · ≤ 2.0e-7 |
| 6 | sq `array` == `square` rad 1 (4 bcs, sq5 / sq9, mode, vertices, beam, packet, solids, rigid edge) | 10/10 bit-identical | 10/10 · 10/10 bit-identical |
| 7 | painting across the seams: incremental == full (torus, rigid ring, clamped ring) | all equal, 14 updates each | all equal · all equal |
| 8 | parity with the 04/10 core (40 configurations) | 40/40 bit-identical | 40/40 · 40/40 |
| 9 | operator == numpy (4 bcs), kernel == operator 32 steps; rescale; fallback; clip; vertices; placement; mode note; recorder; beam / packet; bc switch; Etri | exact / ≤ 1.8e-14; 1.2e-16; all pass | exact / ≤ 8.7e-7, 9.0e-8 · exact / ≤ 1.3e-6, 8.5e-8; Etri 1.4e-9 · 4.2e-8 |
| 10 | alignment 4/4 rows both lattices; ledger on 9 new-mode configurations: no FAIL | pass | pass · pass |

**Cost per step** (`ScratchCLAUDE/Round4CLAUDE/TriModesCLAUDE/tri_modes_bench_MIRROR.py --reps 5`; best of 5 timings, two runs per backend, 05/10/2026, GPU otherwise idle; logs `ScratchCLAUDE/Round4CLAUDE/LogsCLAUDE/def_bench_{cuda_f64,cuda_f32,vulkan_f32}_run{1,2}_MIRROR.log`). Milliseconds per step, recorder off / on; a range is the two runs.

| backend | N | hexagon, rot 30 (the web app's path) | `array`, clamped ring | `array`, rigid ring (walls kernel) | `array`, rhombic torus |
|---|---|---|---|---|---|
| CUDA f64 | 2049 | 0.164–0.167 / 0.347 | 0.233–0.235 / 0.405–0.427 | 0.233–0.234 / 0.444–0.446 | 0.228–0.231 / 0.399–0.433 |
| CUDA f64 | 8193 | 2.72 / 5.50–5.52 | 4.31–4.40 / 7.10–7.11 | 3.80–4.09 / 6.94–7.08 | 4.41–4.43 / 7.13–7.25 |
| CUDA f32 | 2049 | 0.111–0.112 / 0.194–0.195 | 0.095–0.104 / 0.276 | 0.108–0.109 / 0.263–0.286 | 0.112–0.113 / 0.290 |
| CUDA f32 | 8193 | 1.36–1.37 / 2.72–2.79 | 1.97–1.98 / 4.13–4.49 | 2.13–2.14 / 4.60 | 1.96–2.01 / 4.16–4.46 |
| Vulkan f32 | 2049 | 0.107 / 0.182–0.184 | 0.136 / 0.219–0.222 | 0.136–0.137 / 0.226–0.228 | 0.136–0.137 / 0.234–0.235 |
| Vulkan f32 | 8193 | 1.79–1.88 / 3.08–3.10 | 2.38–2.54 / 3.48–3.69 | 2.57–2.59 / 3.69–3.70 | 2.79–2.81 / 3.73–3.92 |

- **The `array` costs ×1.4–1.6 the hexagon at N = 8193** because it updates every cell of the array; the hexagon at radius 1 covers ~56% of it, and masked cells skip the stencil. Per updated cell the two cost the same.
- **The three ring modes** are within −12% to +10% of each other (the rigid ring through the walls kernel, the torus through the wrapped branch). At N = 2049 every configuration sits near the launch floor.
- **The recorder's third direction** (`Etri`, the hexagon with the recorder on, the 04/10 core against the current one in the same process):

| backend | N = 2049 | N = 8193 |
|---|---|---|
| CUDA f64 | 0.324–0.328 → 0.341–0.348 (+4–6%) | 4.64–4.89 → 5.53–5.54 (+13–19%) |
| CUDA f32 | 0.222–0.227 → 0.197–0.199 (−11–13%) | 3.14–3.22 → 2.76–2.81 (−12%) |
| Vulkan f32 | 0.171–0.177 → 0.184–0.185 (+4–8%) | 3.07–3.08 → 3.09–3.10 (+0–1%) |

  The CUDA f32 recorder came out faster than before in both runs. The cause was not looked into.
- One CUDA f32 run crashed and was repeated. The core had been edited while it ran: Taichi compiles each kernel lazily from the source file on disk, so it read shifted lines (`def_bench_cuda_f32_run1_crashed_MIRROR.log`). Not a solver fault.

### 13.8 Deviations

**From the web app:** none for its configurations (test 8 and `crosscheck_js`). Port-only additions:
- the `array` shape and the triangular ring modes;
- the recorder's `Etri` series, and the whole-torus sums on the rhombic torus;
- the ledger FIX `tri-energy`. Every web-app row it changes names the fix.

**From the task specification:**
1. **The triangular bc acts on `array` only.** The task also asked for every web-app triangular configuration, with any bc, to stay bit-identical. Both requirements can hold only if the web-app shapes keep their clamped ring.
2. **Periodic + a masked shape on the triangular lattice** stays clamped where the shape is clipped by the array. On the square lattice it wraps there. The two agree for every shape inside the array (section 13.2).
3. **The mode source on the triangular `array`** seeds sin·sin but is reported n/a. It is not an exact eigenmode there (13.5), and no exact alternative was built.
4. **The energy row** needed a recorder change (`Etri`, the torus sums), not only an n/a, to PASS on the new modes. The alternative would have been to excuse a row that was really failing.

### 13.9 Open issues

1. **Absorbing boundaries on the triangular lattice:** Mur is unavailable. The research note recommends a distance-ramp sponge for the triangular lattice and the flavor shapes.
2. **No wrap** in seeds (pulse, packet), the beam aperture, rasterisation, `main_region()` and `nearest_free()` on either torus. A stroke dragged off one side does not continue on the other.
3. **The square torus' recorder** keeps the web app's interior-only sum. The same whole-torus sum would remove its seam dips. That would be a ledger fix, like `tri-energy`.
4. **The square lattice's recorded E with a rigid edge or rigid solids** is still the web app's functional with dropped links (section 3.4). It is bounded and graded (PASS everywhere tested).
5. The **6-fold orbit on the rhombic torus** is graded about the centre cell only (the ledger's orbit). Any cell would do on a torus.

---

## 14. Review fixes (06/10/2026)

*Generated from scratch — Claude Opus 5.5 — 06/10/2026.* The adversarial review of round 4 (findings: `ScratchCLAUDE/ReviewCLAUDE/findings_round4_MIRROR.md`; scripts and logs: `ScratchCLAUDE/Round4CLAUDE/ReviewCLAUDE/rv_*_MIRROR.*`; progress: `NotesCLAUDE/progress_round4_review_MIRROR.md`).

### 14.1 The sq9 Neumann ring was not a mirror where a solid sits ON the ring row (fixed)

- **The fault.** A Neumann ring cell is a ghost: (0, t) holds u(1, t). Cell a = (1, t) reads the ghost of b = (1, t+1) through its diagonal link to (0, t+1), and b reads the ghost of a through (0, t). The 05/10 partner rule (section 2.3) handles a solid AT a partner. A solid painted ON the ring cell (0, t) itself, or the exterior of a shape clipped by the array with a rigid edge, gave b a 0 or nothing there, while a still read u_b: L[a, b] = 5/6, L[b, a] = 2/3. An asymmetric operator has complex eigenvalues, so the leapfrog grows at every Co.
- **Measured before the fix** (CUDA f64, review scripts in `ScratchCLAUDE/Round4CLAUDE/ReviewCLAUDE/`):
  - one pinned / rigid cell on the ring of the full square, N 41, Co 0.8: exact-energy drift 1.2e-2 / 1.8e-2 in 2000 steps (`rv_neu_ring_solid_MIRROR.py`);
  - the step map's spectral radius 1.0006–1.0007 and eigenvalues of L with |Im| up to 1.9e-3 (N 25, three ring solids, `rv_neu_growth_MIRROR.py`); a kernel run at N 41, Co 0.865: max|u| 0.3 → 25.7 at 80,000 steps → 9.7e12 at 200,000 (pinned), 3e17 (rigid);
  - dense mixed solids (40% / 55%) on the sq9 Neumann square at 0.995 C_max: 1e8 / 1e43 within 20,000 steps (`rv_cfl_run_MIRROR.py --only-c1`);
  - the rigid edge on a circle at radius 1.3, sq9, Neumann: max|u| 522 at 40,000 steps (`rv_neu_clip_growth_MIRROR.py`).
  - sq5 is not affected (no diagonals).
- **The rule** (`_word(..., neu)`, helper `_nring_common`; `k_walls_full` / `k_walls_patch` take the `neu` template = `_walls_key()[4]`): a sq9 diagonal link INTO the Neumann ring whose common neighbour ON the ring is not free is blocked like that neighbour: pinned-blocked for a pinned solid (P), dropped for a rigid solid (R) and for the exterior with a rigid edge (X, erig). The exterior with a pinned edge keeps the web app (the reasoning of the X–X exception, section 2.2). A word still reads only cells within one step, so the painted-box patch stays exact. Pinned solids on the ring row now run the walls kernel (they block a link).
- **Measured after the fix:** 0 asymmetric pairs; drift 0.0 / 6.5e-16 in 2000 steps; spectral radius 1.000000000 (pinned) and 1 + 2.1e-8 (rigid: the Neumann zero mode's double root, as without solids); 200,000-step runs stay at max|u| 0.20–0.48 (`rv_neu_growth_after_fix_MIRROR.log`); the dense runs keep the exact energy to 7.7e-16 / 4.1e-16 with max|u| 5.8 / 4.3; the rigid-edge clipped circle drifts 8.6e-16. An independent numpy operator with the same rule equals the device operator entry by entry on 600 random configurations (`rv_ref_compare_MIRROR.py`); incremental painting equals the full build (`rv_paint_MIRROR.py`, 279 states); the web app's path stays bit-identical to the 03/10 core on 66 configurations (`rv_parity_*_MIRROR.*`). No cost per step (`rv_walls_bench_MIRROR.py`: ×0.94–1.07, noise).
- **The fix attacked (review, 06/10/2026, after a resume):** the rule pinned-blocks a diagonal whose other common neighbour is free, a w u² term the corner rule never makes, so it could in principle raise the CFL limit. `rv_neu_ring_climb_MIRROR.py` (N 21; 200 random ring patterns and 2 × 500 hill-climb steps on each of: full square with a pinned and a rigid edge, circle r 1.3, octagon r 1.25, circle r 1.15 with a rigid edge): max |L − Lᵀ| 0, no complex eigenvalue, lam_max ≤ 0.99663 × 16/3. The best patterns (a run of pinned cells on the ring row) lift lam_max above the plain operator's by +3.3e-4 at N 21, +2.0e-5 at 41, +1.2e-6 at 81, +7.4e-8 at 161, +4.6e-9 at 321 (`rv_neu_ring_bound_MIRROR.py`): no bound state above 16/3, so C_max = 0.8660 holds. *Edited by Claude Opus 5.5 — 06/10/2026.*

### 14.2 The web app's own clipped shape on the sq9 Neumann ring (fixed in the port, 09/10/2026)

**Status (09/10/2026): fixed in the port, the user's decision, as a documented difference.** The exterior X of a pinned edge now blocks the sq9 diagonal into a ghost ring cell like a pinned ring solid (rule C1 extended; not for a link into an array corner, the cell's own ghost); these configurations run the walls kernel (`walls_needed()`: `_neu_clip()`). Harness PATCH `neumann-clip` and `crosscheck_js` config `sq9_circle13_neu_clip`. Measured (`decisions_tests_MIRROR.py` section 1): every enumerated clipped configuration has an exactly symmetric operator (v1.0: all the pinned-edge ones asymmetric); spectral radius 1; exact energy to ~1e-15 over 20,000 steps; the reproducer below bounded over 200,000 steps (max|u| <= 0.37 from 1.0; 3400 enumerated configurations, v1.0 1302 asymmetric); every unaffected configuration bit-identical to v1.0. Details: [v1_decisions_MIRROR.md](v1_decisions_MIRROR.md) section 1. *Edited by Claude Opus 5.5 — 09/10/2026.* What the review found (06/10/2026):


With the Neumann bc, the sq9 stencil and a shape clipped by the array (a circle at radius 1.3, an octagon at 1.25), a pinned edge and no solids, the web app's path has the same asymmetry where the shape's edge meets the ring: an exterior ring cell (0, s) holds 0 while (0, s+1) is a ghost. Measured on the port's web-app path (`rv_neu_clip_growth_MIRROR.py`, N 41, Co 0.865): max|u| 353 at 100,000 steps and 6.7e21 at 200,000 (circle); 270 at 80,000 (octagon). sq5 stays bounded (0.16–0.56 over 200,000 steps). The frozen web app does the same, so the port keeps it for parity: a fix would be a deviation with a harness patch of the JS stencil, not a value patch. It is the user's decision. The rigid edge (or the sq5 stencil) avoids it.

### 14.3 Corrections
- Section 8, deviation 1: the sq9 plate's pinned lettering does block links at other rotations, radii and N (42 of 60 configurations, `rv_plate_sq9_MIRROR.py`).
- Sections 2.3, 7.1 and 9: the Neumann-ring and rigid-edge energy statements, qualified as above.
