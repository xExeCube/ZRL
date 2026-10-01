# The physics ledger, ported (ledger_MIRROR.py)

*Generated from scratch — Claude Opus 5.5 — 30/09/2026 (three runs; the first two were cut off by usage limits, see progress_ledger_MIRROR.md)*

The web app's ledger (`buildLedger()` in `FlavorRenderersCLAUDE/wave_membrane_MIRROR.html`, lines ~1687–2492, with everything it calls) is ported value for value to `ledger_MIRROR.py`. `crosscheck_ledger_MIRROR.py` checks it against the web app's own code, run headlessly under Node. The GGUI panel is `ledger_MIRROR.gui_section(w, viewer)`.

**Status on 30/09/2026:** 50 configurations, 197 ledgers, 3435 rows. Status agreement is 100% and full-row agreement (label, status, detail text with its numbers, tier) is 100% on CPU f64, CUDA f64, CUDA f32 and Vulkan f32. The 15 port-only checks (solids, the plate, the recorder, the cache) pass on all four.

## What was ported

- **The functions:** `fmt`, `fe`, `st`, `DSG`, `symmetryResidual`, `mirrorSymmetric`, `mirrorResidual`, `symRotResidual`, `frontFree`, `frontRadii`, `srcCentred`, `coneWall`, `coneMeasure`, `energyStats` and `buildLedger`.
- **What they call:** `clearanceFrom`/`CLEAR`, `CONE_WALL`, `coneNorm`, `reachInradius`, `stencilSymbol`, `kMax`/`HARM`, `shapeArea`/`idealArea`/`circleClipArea`, the probes pA/pB, `MODEP` and `record()`. `record()` is the core's recorder: `m.hist`, `m.amax`, `m.pA()`, `m.pB()` and `m.mode_antinode()`.
- **Edge-alignment rows** (7c and 7c-bis) come from `alignment_MIRROR.al_ledger_rows`, which was checked against the JS in its own area.
- **Not in the ledger:** `sourcePPW` (~3076) feeds only the PPW render. `buildLedger` has no space-time rows; the light-cone row is the closest.
- **The Neumann corner** is fixed in the port. The JS harness applies the same fix through its `PATCHES` list (`neumann-corner`), and the cross-check prints which patches were active.

## API

```python
import ledger_MIRROR as lg
rows = lg.build_ledger(m, compat=False, profile=None)   # list of lg.Row, the web app's order
name, status, value, note, tag = rows[0][:5]
lg.summary(rows)            # {'PASS': n, 'FAIL': n, 'NA': n, 'stated': n}
lg.format_rows(rows)        # plain text (CLI, logs)
lg.gui_section(w, viewer)   # GGUI panel inside an open sub_window w; viewer has .m and .reset()
```

CLI: `python ledger_MIRROR.py --arch cuda --steps 1300 N=161 shape=n6 src=vtx [--compat] [--brief] [--f32]`.

`Row` is a NamedTuple. Its first five fields are the requested `(name, status, value, note, tag)`:

| field | meaning |
|---|---|
| `name` | the row's label, plain unicode (the web app's HTML entities decoded, tags dropped) |
| `status` | `PASS`, `FAIL`, `NA` or `stated` |
| `value` | a compact measured value (`''` when the row is n/a) |
| `note` | the web app's detail text, word for word, with this run's numbers |
| `tag` | the web app's last column (tier / reference) |
| `key` | a stable id: `cfl sym8 sym6 conform mirror energy recip bc aniso wake stair align straight cone spectra mode disp linear refraction` |
| `badge` | the web app's own status text: `PASS FAIL n/a RED OK not yet measured stated visual` |
| `dsg` | the label is a design statement (the web app's `dsg` style) |
| `num` | the primary measured number, or None |
| `port` | a port-only remark (solids, the plate, f32, the recorder, a fix) |
| `fixes` | the ids of the FIXES that changed this row |

**Badge to status:**
- `RED OK` becomes PASS: the run is past the CFL limit and did diverge.
- `not yet` becomes FAIL: the run is past the limit and is still bounded.
- `measured`, `stated` and `visual` are informational, so they become `stated`.

Other public names:
- **Status constants:** `PASS FAIL NA STATED`.
- **Tables:** `FIXES` (id → explanation), `TOL_F32`, `HIST_ROWS`, `COL`/`TAG` (the GUI colours).
- **Panel:** `LedgerPanel`, with `AUTO_MAX_N = 2049`.
- **Switches:** `DEVICE` (True: Taichi reductions; False: the numpy references on a host copy).
- **Numeric helpers:** `symmetry_residual`, `mirror_residual`, `sym_rot_residual`, `energy_stats`, `cone_measure`, `cone_measure_np`, `front_radii`, `orbit_residual`, `gather`, `mode_sums`, `stencil_symbol`, `cone_norm`.

## FIXES: faults in the web app's ledger

These are the JS bugs found while porting. They are applied by default. `build_ledger(m, compat=True)` turns them off and gives the web app's exact logic, which is what the cross-check compares. Every row a fix changes names it in `Row.fixes`. In the cross-check, 144 rows change, all attributed, and none changes without naming a fix.

| id | rows changed | what the web app does wrong |
|---|---|---|
| `edge-kind` | 22 | The BC control acts on the array ring only, but the web app graded the boundary, energy, wake and linearity rows by `S.bc` alone. Three **false FAILs** result: a masked n6 with Mur (bc row at step 1200), a tri n6 with Mur (bc at 1200), and an n12 with Mur and a continuous drive (linearity at 1200). Their walls are clamped, so the wave does come back. A closed masked cavity with Mur selected was also never energy-tested. The port uses what actually bounds the domain: `clamped`, `p.bc` on the full square, or `mixed-<bc>`. |
| `tri-diagonal` | 5 | Anisotropy on the triangular lattice: the index diagonal (1,1) is √3 long, not √2, so the ratio read √(2/3) of its value. For example, tri n6 at step 50 reads 0.8081 in the web app and 0.9897 in the port. |
| `sqfull-side` | 83 | The staircase row on the full-grid square quoted the masked square's R ("the side is 157 whole cells" at N = 161), but that domain is the whole N × N array. |
| `lens-text` | 8 | The refraction label says "the slab (c/2)" for the lens medium, which is 0.55 c. |
| `mirror-live` | 24 | Before step 6, or when the field has diverged or is zero, the mirror row said "needs a pulse, impulse or vertex source" even when the source was one. |
| `cfl-nan` | 2 | Once the field is all NaN, `record()`'s max skips NaN and `S.amax` reads 0, so a diverged run over the limit said "not yet" (red). A non-finite energy now counts as diverged. |

## Port-only decisions (solids, the 'unlicensed' plate, the recorder)

The web app has no solids and no plate. Each row was decided on its premise: a row whose premise is the pristine flavor domain reports n/a with a reason instead of a false FAIL. A row that still means something stays graded.

| row | with solids | 'unlicensed' plate |
|---|---|---|
| CFL | graded (a property of the lattice and stencil) | graded |
| 8-fold / 6-fold symmetry | Graded when the free-cell mask is invariant under the orbit (checked cell by cell, exact) and the source cell is the fixed point. Otherwise n/a, saying which failed; a snapped source is named with its cell. A symmetric arrangement of solids is still an exact solver test: 8 disks give 5.5e-16. | n/a: not 8-fold, not a lattice rotation (the JS logic already says so) |
| conformance (tri) | counts the DOMAIN mask (solids do not change how the shape is rasterised) | the JS logic |
| mirror | The premise is checked on the free-cell mask, not with `mirrorSymmetric` (a solid on the x axis keeps it; tri + circle at (0.5, 0) gives 2.0e-16). The source cells must be mirror images. | `mirrorSymmetric` has no case (it parses the shape as an n-gon), so the exact mask check is used. The bare plate gives PASS 5.5e-16; with its lettering or a rotation the row is n/a with the reason. |
| energy | graded: pinned walls conserve the same energy as a Dirichlet domain | graded |
| boundary at the probe | n/a: solids also reflect, so the edge's own reflection cannot be isolated | graded (the bare plate: "fixed edge" PASS) |
| anisotropy | graded; the clearance counts solids as walls, so the front is measured only before it reaches one | graded |
| 2D wake | n/a: solids reflect the front back to the probe | the JS logic |
| staircase | counts the DOMAIN mask | exact area in closed form, W·H − (4 − π)r²; "measured" |
| edge alignment ×2 | on the DOMAIN mask (solids are not domain edges) | the alignment module's own handling |
| light cone | graded; `CONE_WALL` counts solids as clamped cells (containment is exact whatever the walls) | graded |
| drum spectrum | n/a: the closed forms are for the empty domain | the JS logic: "no closed-form spectrum" |
| eigenmode, dispersion | n/a: sin·sin is not an eigenmode with objects inside | n/a (the JS logic: not the square) |
| linearity | graded (pinned walls are linear) | graded |
| refraction | informational, unchanged | unchanged |

**Recorder gaps** are also port-only, because the web app always records. The history rows are energy (E), boundary (pA), wake (pA), dispersion (mode) and linearity (pA). A complete history holds exactly one entry per step, capped (`len(E) == min(step, 4000)`; mode: `min(step + 1, 600)`). When the history is shorter, `m.record` was off for some steps: a graded history row becomes n/a with the reason, and an n/a row keeps its own reason and gets the remark. The panel's "reset run" button reseeds the run so that it gets a complete history.

**f32 builds** (Vulkan, `--f32`): the exact-identity tolerances are the web app's f64 ones scaled to single precision, `TOL_F32 = {'sym': 1e-4, 'disp': 1e-5}`. The largest measured PASSing residuals are well below them:

| build | sym8 | mirror | sym6 | disp |
|---|---|---|---|---|
| CUDA f32 | 4.0e-7 | 2.0e-6 | 7.2e-7 | 1.4e-8 |
| Vulkan f32 | 5.2e-7 | 3.7e-6 | 7.8e-7 | 1.75e-8 |

The cone's underflow threshold is 1e-30 in f32 builds and 1e-290 in f64.

## Device reductions (performance)

The O(N²) rows run as Taichi kernels over the current time level, with no copy of the field to the host:

| kernel | what it computes |
|---|---|
| `k_orbit` | the symmetry, mirror and 6-fold residuals, sampled exactly as the JS samples them |
| `k_cone` | support radius, cells outside the cone, Σu² and the disk's share of it, max\|u\| on the rim |
| `k_modecorr` | the three correlation sums of the eigenmode row |
| `k_gather` | the few hundred probe cells of the front rays |

How they are built:
- **Two-level reduction.** Every kernel reduces into 64 slots and the host folds the slots. This follows the core's measured rule: millions of atomics into one address are slow.
- **Accumulators** are the core's `ACC_DT`. It is f64 on CUDA and CPU, even in an f32 build: with f32 slots the (3,2) mode's correlation read 1.00000006. On Vulkan it is f32, and there the mode correlation falls back to a host copy with f64 sums.
- **NaN handling.** `Math.max` propagates NaN but `atomic_max` drops it, so a NaN is flagged in its own column. The NaN tests are written as ordered compares (`d >= 0`, `not (v == 0)`), because on Vulkan `d != d` never saw a NaN: a diverged f32 run read 0.0 or inf where the numpy reference read NaN.
- **Verified** against the numpy references with `ScratchCLAUDE/LedgerCLAUDE/dev_vs_np_MIRROR.py` (14 configurations × N 161/257 × steps 0/40/200/700): 0 differing rows on CUDA f64, CUDA f32, CPU f64 and Vulkan f32.

**Build time, CUDA f64, RTX 5070 Ti** (`perf_MIRROR.py` / `perf_first_MIRROR.py`, 30/09/2026):

| N | warm rebuild | cold first build (geometry cache) | `m.level(1)` alone, for comparison |
|---|---|---|---|
| 161 | 1.0–3.6 ms | (first-ever call: kernel compile, ~0.2 s) | 0.5 ms |
| 1025 | 0.8–4.9 ms | 7–67 ms | 3 ms |
| 2049 | 0.8–3.7 ms | 20–131 ms | 12–15 ms |
| 4097 | 0.8–5.2 ms | 73–463 ms | 43–64 ms |
| 8193 | 0.9–6.1 ms | 0.29–1.87 s | 160–200 ms |

- **The geometry cache** (`_GEO`, per Membrane) holds the boundary cells, `CLEAR`, `CONE_WALL`, the areas, the exact mask symmetry checks and the edge-alignment rows. Its key covers N, the lattice, stencil, shape, rotation, radius, BC, source parameters, mask identity, the scene version, `n_solid`, the source cell and the vertex cells. It is invalidated by a reset, a parameter change or a solid edit, including in-place `update_solids(box)`; a port-only check confirms that a cached build after a painted stroke equals a fresh one.
- **A cold build at N = 8193** spends 0.7–1.45 s in `alignment_MIRROR.al_ledger_rows` and ~250 ms in the boundary-cell scan.

**GUI refresh policy** (`LedgerPanel`):
- **Auto-refresh** (every 0.5 s by default, adjustable) is on by default up to N = 2049. Above that the ledger is built on demand ("refresh ledger"), because a cold build would stall a frame for up to ~1.9 s after every reset. The default follows N until the user touches the checkbox.
- **The interval has a floor** of 20× the last WARM build time, so the ledger takes at most ~5% of frame time. A cold build is flagged "(cold)" and does not raise the floor.
- **Headless test** (`gui_test_MIRROR.py`, hidden window):
  - N = 1025: 115 frames in 6 s, 9 builds (~1.5 per second, the first including the compile), panel median 0.36 ms per frame.
  - N = 8193: no automatic build, and one on-demand cold build of 1.2 s.
  - Every drawn line is Latin-1, ImGui's default font range. The GUI maps √ → sqrt, → → ->, and so on.

The offscreen `save_image` does not capture the imgui panel, so the panel's look is not in any PNG.

## Verification

**`ToolsCLAUDE/js_ledger_dump_MIRROR.mjs`:**
- It runs the web app's real `buildLedger` through `js_harness_MIRROR.mjs` with its PATCHES.
- It steps each configuration to its checkpoints and drops pulses where the configuration says.
- It parses the four cells of every `<tr>` that `buildLedger` writes into `#led`.
- It dumps the module state the rows read: step, `S.amax`, `S.drops`, `CLEAR`, `CONE_WALL`, `SRC` and the history lengths. Names the harness does not export are reached through the sandbox's Function constructor.

**`crosscheck_ledger_MIRROR.py`** (`--arch`, `--f32`, `--only`, `--quick`, `--jsonl <saved dump>`, `--port-only`, `--no-port`, `--verbose`):
- **50 configurations × checkpoints 0/50/300/1200.** They cover the full-grid square with Dirichlet, Neumann, Mur and periodic edges; centred and offset sources; both stencils; drum modes (1,1), (3,2), (2,5 on a 0.7 square), (7,3 damped), and modes with slab and Neumann; masked circle, n3–n12, rhomb and small squares; the slab and lens media; the pulse, impulse, cont, slit, vtx, vtx-wall, vtx-drive and mode sources; the sine, tri, square and saw waves; damping; four runs past the CFL limit, one of them a twin started from the web app's own seed; 11 triangular-lattice configurations; manual drops; and N = 257.
- **Statuses must agree exactly.** Detail texts must agree word for word. Their numbers agree if the strings are equal, or if they are within one unit of the last printed digit (fields that differ in the last bits can round to the neighbouring display value), or within 1e-6 relative (sums in another order), or if both are below 1e-13 (machine-noise residuals: only the noise floor is reproducible, because the seeds differ by one ulp, numpy exp vs V8 `Math.exp`).
- **f32 builds** use a 1e-4 floor and 1e-3 relative. Rows where the web app's `S.amax` exceeds 3.4e38 are exempt, since f32 cannot represent them (12 rows). Rows with the same status but f32-rounded numbers are listed as informational (22 on CUDA, 21 on Vulkan).
- **Runs past the CFL limit** grow a checkerboard ~6.9× per step from rounding noise, so they are compared by status only. On CPU f64 the twin started from the web app's seed must agree in full, and it does.

| run | status | full row | port-only |
|---|---|---|---|
| CPU f64, fresh Node dump (143 s) | 3435/3435 | 3435/3435 | 15/15 |
| CUDA f64 | 3435/3435 | 3435/3435 | 15/15 |
| CUDA f32 | 3435/3435 | 3435/3435 (12 exempt) | 15/15 |
| Vulkan f32 | 3435/3435 | 3435/3435 (12 exempt) | 15/15 |

A saved web-app dump, `ScratchCLAUDE/LedgerCLAUDE/ledger_js_full_MIRROR.jsonl`, skips Node (`--jsonl`). Logs are in `ScratchCLAUDE/LedgerCLAUDE/xc6..xc11_*_MIRROR.txt`.

**The port-only checks** (`crosscheck_ledger_MIRROR.py --port-only`, 15 checks):
- **Solids:**
  - 8-fold symmetric solids: the symmetry rows stay graded and PASS.
  - An asymmetric block: the symmetry rows are n/a with the reason.
  - A mode with a block inside: the mode and dispersion rows are n/a.
  - A solid over the centre: the source snaps out, and the symmetry rows say so.
  - A tri hexagon with a circle on the axis: mirror PASS, cone PASS.
  - Mur with solids: the wake and boundary rows are n/a.
  - A continuous drive with solids: no FAIL rows.
- **The plate:**
  - With its lettering, no FAIL rows.
  - The bare plate: mirror PASS, drum spectrum "stated".
  - The bare plate rotated 30°: mirror n/a.
  - The plate on the triangular lattice at r = 0.89: cone PASS.
- **Solids removed:** the ledger is identical to that of a never-solid run.
- **Cache:** the cache follows a painted stroke.
- **Recorder:** off for part of the run, and off from the start.

No row may FAIL in any of these runs unless the check names it.

## Open points

- The cold first build at N ≥ 4097 is dominated by `al_ledger_rows` (alignment module, not this area): 0.7–1.45 s at 8193.
- `Membrane` setup for n6 with vertex sources at N = 8193 took 18.8–19.9 s. That is the core, not the ledger.
- The reciprocity row is "not yet wired" in the web app, and so in the port.
- In the mirror row for the plate, the web app's detail text ("the triangle and pentagon have NO 8-fold orbit…") is kept word for word. It reads oddly for the plate but is not wrong.
