# Solids, the Unlicensed plate, the recorder and the Neumann corner fix

What round 2 of the Taichi port added to the solver (`membrane_core_MIRROR.py`) and the new module `solids_MIRROR.py`: static solid objects and hand-painted walls, the *Unlicensed* domain with its lettering, the per-step recorder the ledger reads, and the Neumann corner fix. Every number below was measured on the user's machine (i9-9900K, RTX 5070 Ti) on 30/09/2026.

*Generated from scratch — Claude Opus 5.5 — 30/09/2026*

## In one paragraph

Every solid is **pinned**: u = 0 in a solid cell, the rule the domain mask already applied. The solver's mask is now `M = domain AND NOT solid`; the kernel did not change for it. Rigid (Neumann) walls are held off, as the user asked. Walls painted with a brush of at least **1 cell** radius are sealed on all three lattices (exactly 0.0 behind them after 2000 steps); a one-cell diagonal staircase is **not** sealed on sq9 or tri. With no solids the solver runs the web app's exact path (the cross-check, 27/27).

## Files

| file | what it is |
|---|---|
| `solids_MIRROR.py` | `SolidObject`, `Stroke`, `SolidScene`: objects, painted walls, the Unlicensed preset, rasterisation, painting, save/load, unit conversions |
| `solids_tests_MIRROR.py` | 11 groups of checks, most of them exact (see *Measured*) |
| `membrane_core_MIRROR.py` | composes the mask, zeroes and patches the device mask, snaps sources out of solids, runs the recorder, the `'unlicensed'` shape, the Neumann fix |
| `ToolsCLAUDE/js_harness_MIRROR.mjs` | `PATCHES`: the documented port deviations, applied to the web app's code before it runs |
| `ScenesCLAUDE/` | saved scenes (`*_MIRROR.json`) |

## API

### Core additions (`membrane_core_MIRROR.py`)

Nothing the app or the alignment module already calls changed its signature. `SHAPES` gained `'unlicensed'` at the end.

**Masks**
- `m.domain_mask_np`: the shape alone (uint8 `[i, j]`).
- `m.solid_np`: the solids (uint8 `[i, j]`, 1 = solid), or `None` when there are none.
- `m.mask_np`: `domain AND NOT solid`, what the solver, the sources and the recorder see. It is the very same object as `domain_mask_np` when there are no solids.
- `m.has_solids`, `m.n_solid`, `m.solid_cells()` (solid cells inside the domain), `m.domain_cells()` (free cells).

**Solids**
- `Membrane(params=None, scene=None)`: a scene can be passed in; the `'unlicensed'` shape creates an empty one itself (its lettering is a preset of the scene).
- `m.ensure_scene()`: the attached `SolidScene`, created if missing.
- `m.attach_scene(scene)`: attach (or `None` to detach) and apply at once.
- `m.update_solids(bbox=None)`: apply scene edits **without a reset**.
  - `bbox = (i0, i1, j0, j1)` (half-open cells, what the scene's edit methods return): only that box is re-rasterised from the whole scene, recomposed and uploaded as one mask patch, with the three time levels zeroed where the mask became 0.
  - An empty box does nothing.
  - `None` redoes the whole grid.
  - Either way the sources are re-placed and the probes re-cached; the rest of the field and the clock are untouched.
  - When the last solid cell goes, the solver returns to the exact web-app path.
- `m.set_solid_np(mask)`: a raw bitmap of solids (tests). It survives `reseed()` and `set_co()`.

**Sources and lattice helpers**
- `m.hops(di, dj)`: lattice hop count — `|a|+|b|` (sq5), `max(|a|,|b|)` (sq9), `max(|a|,|b|,|a+b|)` (tri).
- `m.nearest_free(i, j)`: the free cell nearest in hops (ties: physical distance, then `(i, j)`).
- `m.neighbour_offsets()`: the stencil's links.

**The plate**
- `m.plate_dims()` returns `(half-width R, half-height R/2, corner radius 0.065 R)`.
- `m.plate_corners()`: the 4 corner-arc midpoints, the vertex sources.

**The recorder** (the web app's `record()`, after every step)
- `m.record` (default `True`).
- `m.hist`: a dict of lists with keys `'E'`, `'amax'`, `'pA'`, `'pB'` and `'mode'`.
  - The caps are the web app's: 4000 entries each, and 600 for `'mode'`.
  - `'mode'` gets the seeded step-0 value at seed, like `MODEP`.
- `m.amax`: the web app's `S.amax`. It is set at seed, by `drop_pulse`, and after every recorded step.
- `m.pA()`, `m.pB()`: the probe offsets. `m.probe_cells()` gives `((c+pA, c), (c, c+pB))`.
- `m.mode_antinode()`.
- `m.stats()` still returns `(E, amax)` of the current state; it now uses the same two-level reduction.

### `solids_MIRROR.py`

**Units.** Positions and sizes are in the object's `anchor` unit:
- `'grid'`: units of `base_half`, the array's physical half-size. The same scene means the same thing at every N and on both lattices.
- `'domain'`: units of `dom_R`; the object also turns with the domain's rotation (the plate preset).

Strokes are always in grid units.

**`SolidObject(kind, x, y, rot_deg, w, h, r, n, text, height, font, variation, stretch, anchor, enabled, tracking, fit_w, name)`**
- `kind` is one of:
  - `rect`: `w × h` (full sizes);
  - `circle`: radius `r`;
  - `ngon`: `n` sides, circumradius `r`, the domain shapes' vertex convention;
  - `text`: cap height `height`, horizontal scale `stretch`.
- For text, `fit_w > 0` fits the whole ink width and solves the stretch, and `tracking` adds letter-spacing in cap heights.
- `tracking`, `fit_w` and `name` are additions to the field list asked for.

**`Stroke(points, radius, erase)`**: a polyline in grid units. The physical radius is never below `MIN_BRUSH = 1.0` cell.

**`SolidScene(objects=None, strokes=None, plate_text=True, plate_screws=True)`**. Every edit returns the cell box to pass to `m.update_solids`; `None` means the whole grid.
- **Objects:** `add(obj) -> (index, box)`, plus `add_rect`, `add_circle`, `add_ngon`, `add_text`; `remove(i) -> box`; `edit(i, **fields) -> box` (old ∪ new); `list_objects()` (the preset's objects appear as `'preset'`, read-only).
- **Painting:** `begin_stroke(x, y, radius, erase=False) -> box`; `extend_stroke(x, y) -> box`, where a repeated point returns `EMPTY_BOX`; `end_stroke()`; `undo_stroke() -> box`; `add_stroke(points, radius, erase=False) -> box`; `clear_strokes() -> None`.
- **Preset:** `set_preset(plate_text=None, plate_screws=None) -> None`, `preset_objects(m)`.
- **Rasterising:** `rasterize(m, i0, i1, j0, j1)` and `rasterize_full(m)` (`None` if no solid cell).
- **Files:** `save(path=None) -> path` (default `ScenesCLAUDE/scene_<date>_<time>_MIRROR.json`), `SolidScene.load(path)`, `to_dict()`, `from_dict()`.

**Conversions:**
- `grid_to_phys`, `phys_to_grid`, `grid_to_cell`, `cell_to_grid`, `domain_to_phys`, `phys_to_domain` (all take `m` first);
- `cells_to_grid(m, cells)` (a brush size in cells);
- `brush_cells(m, radius)`, `unit_scale(m, anchor)`, `phys_box_to_cells`, `bbox_union`.

**Fonts:** `resolve_font(name)`, `font_variations(name)`, `text_layout(...)`, `text_stretch(m, obj)`.

## Geometry sources

### The Unlicensed plate

The plate is from Death Grips, *Government Plates* (2013); the cover image is in the session scratchpad as `plate_0.png`.

**Shape.** A rounded rectangle, width : height = 2 : 1, corner radius 0.065 H.
- `dom_R()` = (base_half − 1.5) × radius is the **half-width**, so W = 2R and H = R.
- `inner_R()` = min(R/2, base_half).
- `shape_poly()` samples the outline with 16 segments per corner arc (68 points, CCW); the alignment module treats its straight sides as a rectangle.
- The predicate is exact in the rotated frame (corner arcs, +1e-9).

**Lettering and screws.** Anchored to the domain, in units of R, as measured from the cover:

| item | centre | cap height | total ink width | font (Bahnschrift variation) | tracking | fitted stretch |
|---|---|---|---|---|---|---|
| CA EXEMPT | (0, +0.319) | 0.1375 | 1.107 | SemiBold | 0.09 cap | 1.011 |
| DEATH | (0, −0.0875) | 0.525 | 1.574 | SemiBold SemiCondensed | 0.30 cap | 0.651 |
| 4 screw heads | (±0.628, ±0.4375) | radius 0.02 | | | | |

**Why those font settings** (`ScratchCLAUDE/DevCLAUDE/font_probe_MIRROR.py`):
- **DEATH.** Bahnschrift *SemiBold SemiCondensed* sets "DEATH" 3.58 cap heights wide; the cover's is 3.00, with narrow letters (~0.46 cap) and wide gaps.
  - Fitting the width by stretch alone would squash the gaps too.
  - So tracking 0.30 cap is added, and the fitted stretch comes out at 0.65: letters ~0.5 cap wide, vertical stems ~0.10 cap, as on the cover.
- **CA EXEMPT.** The line is a normal-width semibold: *SemiBold* sets it 7.32 cap heights wide against the cover's 8.05, so tracking 0.09 gives a stretch of 1.01.
- **Fallbacks.** Arial Narrow Bold, then Arial Narrow, if Bahnschrift is missing. Only the default font falls back; any other missing font raises `FileNotFoundError` naming the paths tried.

A render of the mask at N = 1025 looks like the cover's plate.

**Letter counters.** The holes in D and A (and P, R in CA EXEMPT) become **sealed cavities** of free cells. That is expected: a pinned plate with cut-out letters has them too.

**Consequence for the default source.** With the source at the centre (sx = sy = 0), the centre cell is solid. The walk toward the centre fails, and the nearest free cell is inside the counter of the **A**: a 47-cell cavity at N = 161, 508 cells at N = 513. The pulse rings in the A.
- sy = −0.9 (below DEATH) or +0.85 (the band with CA EXEMPT) puts the source in the main region: 9957 of 10218 free cells at N = 161.
- This is a request to the app, not a change to `Params`.

**Triangular lattice.** The array is a rhombus in physical space, so at radius 1 it clips the plate's left and right ends, as it clips the square shape. Radius ≤ 0.89 fits the whole plate (the tests use 0.85).

### Walls: why 1 cell seals

Take a link between two free cells on opposite sides of a painted curve. It crosses the curve at some point P, and both of its ends are more than r from P. So the link is longer than 2r.

The longest links are 1 (sq5, tri) and √2 (the sq9 diagonals). A brush of radius r > 0.5 (sq5, tri) or r > 0.7071 (sq9) therefore already seals; `MIN_BRUSH = 1.0` keeps a margin.

The stroke is the exact union of capsules (point-to-segment distance), so there is no interpolation gap however fast the mouse moves.

A **one-cell staircase** along the index diagonal leaks:
- through the sq9 diagonal links;
- through the triangular lattice's (1, −1) links.

It holds only for sq5.

### Rasterisation

Cells are tested at their **physical** centres (`cell_xy`), so circles are round and text is upright on the triangular lattice too. Each object visits only its own bounding box, in row chunks of ≤ 4M cells.

**Text** is rendered with PIL glyph by glyph (pen advance + tracking; kerning is not applied), at ~2 px per cell of cap height (200–1200 px). Each cell is sampled in the object's frame, bilinearly, with a threshold at 50%. The rendered layouts are cached.

**Order:** objects (union), then the strokes in order — a draw stroke adds, an erase stroke subtracts. Erasers cut objects too.

### Sources in solids

A source cell that falls in a solid first takes the web app's walk toward the centre. If that walk finds no free cell, it goes to the **nearest free cell in lattice hops**. This is the BFS result, computed in closed form over a growing window: a Python BFS over a large solid disk would have been millions of visits.

**The slit source.** With solids present, its drive line skips masked cells, so it never writes into a solid. With no solids it drives the whole row, as the web app does.

## The recorder

It mirrors the web app's `record()` (~line 1329):
- `kin` = Σ(u_new − u_prev)², `pot` = Σ of the forward differences squared, and `amax` = max|u_new|, all over the masked interior;
- probes at `(c+pA, c)` and `(c, c+pB)` (JS `Math.round`), and the mode antinode when src = mode;
- E = kin / (2 max(Co², 1e-12)) + ½ pot.

**How it runs.** The recorder runs in its own kernel, `k_run_rec`, after the sources. `k_run_rec` shares the step body (the `ti.func` `_steps`) with the plain `k_run`. The per-step values go to a device ring of 1024 rows, which the host reads **once per `step()` call**. The reduction slots are the last 64 rows of that same buffer.

**Why two kernels.** Taichi 1.7.4 charges per kernel call, for each argument (measured in `ScratchCLAUDE/DevCLAUDE/argcost_MIRROR.py`):

| argument | cost per call |
|---|---|
| ndarray | ~30 µs |
| scalar | ~10 µs |
| template | ~2 µs |

- **The first attempt** passed the recorder's buffers to every call: 3 extra ndarrays and 2 scalars. That made even the recorder-OFF step ~15 µs slower at N ≤ 1025 (0.091 → 0.106 ms at sq 161).
- **The shared body** first took its scalars by value. A by-value `ti.func` parameter becomes a copy statement at the kernel's top level, i.e. one more serial task per launch. That cost Vulkan another 40–60 µs per launch, until every parameter of `_steps` and `rec_fold` became a template.
- **Now:** with the recorder off, the step is at parity with the 29/09 core, on CUDA and on Vulkan.

**The reduction.** It is two-level: a thread per column strip, then 64 slots, then one thread folds the slots.
- A one-address atomic reduction took 62 ms per pass at N = 4097.
- Accumulators are f64, except on Vulkan, where f64 accumulators crashed the process (29/09/2026). Vulkan uses f32 there, whose relative error measured ~2e-7 even at N = 8193.

**It reads only.** The field with the recorder on is bit-identical to the field with it off (identity test 9).

### Its cost

CUDA f64, ms per step, square lattice (`bench_MIRROR.py --rec-overhead`, final code):

| N | step, recorder off | extra per step, long runs | extra per step, 8-step calls |
|---|---|---|---|
| 161 | 0.092 | +0.046 | +0.110 |
| 1025 | 0.085 | +0.048 | +0.121 |
| 4097 | 0.864 | +0.475 | +0.465 |
| 8193 | 3.84 | +1.95 | +1.74 |

- The 8-step calls are the app's frames: each call syncs and copies the history out.
- The triangular lattice is the same within ±0.05 ms (+2.0 at N = 8193).

**At large N it costs about half a step.** It reads two f64 levels and the mask, ~17 of the step's ~41 bytes per cell. Measured in `ScratchCLAUDE/DevCLAUDE/rec_tune_MIRROR.py`:
- 131072–1048576 first-level threads and 32–256 slots were no faster at N = 4097 and 8193.
- 65536 threads were the fastest at N = 1025.

Turn it off (`m.record = False`) when the ledger is not open at large N.

## The Neumann corner fix (port and cross-check)

**The fault.** The web app copies the Neumann ring in one sequential loop, so corner (0,0) is written at i = 0 from (1,0) **before** i = 1 refreshes it. It therefore holds the rotating buffer's old contents: the time level three steps back. The other three corners already equal their diagonal interior neighbour of this step.

**The fix.** The port now sets every corner that way; `(0,0)` is copied in the final corner pass.

**The cross-check.** It compares against *the web app + documented patches*:
- `ToolsCLAUDE/js_harness_MIRROR.mjs` holds a `PATCHES` list (id, date, rationale, exact search string, replacement), applied to the module source before it runs.
- A search string that is missing or not unique throws.
- `ZRL_JS_NOPATCH=1` runs the unpatched app.

| check | result |
|---|---|
| crosscheck, patched, CPU f64 | 27/27; `sq9_full_neu_corner` bit-identical from the same seed |
| crosscheck, `ZRL_JS_NOPATCH=1`, CPU f64 | 25/27 |

The two configurations that differ unpatched:
- **`sq9_full_neu_corner`:** the field by 3.2e-2 of max\|u\|, the energy history by 1.7e-2, the probes by 1.4e-5.
- **`sq5_full_neu_impulse`:** only the corner cell (1.3 × max\|u\| there). The energy and probes agree to 1e-14, because the 5-point stencil never reads the corner.

**Identity tests (numpy reference in the web app's operation order).** Both rules were run with the same seed.

| test | new rule | old rule (numpy) |
|---|---|---|
| 8-fold symmetry of a centred pulse, sq5, N = 257, 900 steps | 5.1e-15 | 4.0e-3 (1 cell: the corner) |
| 8-fold symmetry of a centred pulse, sq9, same run | 8.5e-15 | 5.6e-2 (63673 cells differ) |
| ghost-cell leapfrog energy, sq9, pulse near (0,0), 2000 steps | drift **1.0e-14** (exactly conserved) | 5.8e-3 |
| ghost-cell leapfrog energy, sq5, same run | drift 2.9e-16 | 2.9e-16 (the 5-point stencil never reads a corner) |

For the symmetry test, the port agrees with the numpy new rule to 3.0e-14.

So energy conservation under Neumann holds exactly with the fixed corner (tested for both stencils), and never held for sq9 in the web app.

## Measured (30/09/2026)

### `solids_tests_MIRROR.py`: CUDA f64, Vulkan f32, CPU f64

All pass on all three (CUDA with the N = 8193 timings; Vulkan and CPU with `--quick`). The values below are the same on every backend unless a row says otherwise.

| check | result |
|---|---|
| no solid cell in the domain (empty scene; an object outside the domain) | field bit-identical to no scene |
| closed painted loop, minimum brush (1.000 cell), sq5 / sq9 / tri, many edge angles | max\|u\| inside = **0.0** in all 3 levels after 2000 driven steps; 0.24–0.27 just outside the wall |
| one-cell diagonal wall, driven cell of amplitude 1, 1500 steps | peak \|u\| beyond it: sq5 **0.0**, sq9 **7.5e-2**, tri **2.3e-1** |
| the same line with the minimum brush | 0.0 on all three |
| solid cells stay 0 (plate on sq9 and tri; rect / circle / pentagon / text / stroke on a full Neumann square, one rect across the array ring) | 0.0 in all 3 levels after 1000 steps |
| mirror pairs of rects (±20°), circles and strokes, on-axis source, sq5 Dirichlet | solid mask mirror-identical; max\|u − mirror(u)\| = **0.0** after 1500 steps |
| object areas, N = 1025, generic orientations | circle / 17° rect / 10° pentagon within 1e-5 – 2.2e-4 of the analytic area |
| hexagon at 0° | −2.1e-3 (sq), +5.2e-3 (tri) |
| Unlicensed plate area | sq r 1: +9.3e-4; tri r 0.85: −1.3e-3; sq r 0.6: +2.7e-3 |
| `update_solids` mid-run (circle + 12-point stroke + eraser at step 150) | new solid cells 0 in all levels; every other cell untouched (bit-exact); incremental mask = full re-rasterisation on host and device; energy finite for 500 more steps |
| sources in solids | the walk-back works; with the centre swallowed, the nearest free cell matches the brute-force minimum hop count (sq5 38, sq9 28, tri 38); driven cells never in a solid |
| save / load round trip, font errors | the dict and the rasterisation are identical; a missing font gives `FileNotFoundError`, an unknown variation `ValueError` |

Why the hexagon and the plate miss by more: their edges run **along lattice lines**. The hexagon's vertical sides at x = ±110.85 lose 0.35 cell per row per side, and the plate's sides are rows and columns. So the bound is half a line spacing per unit of boundary: error ≤ 0.5 × perimeter. At 7° or 13.3° and off-centre the hexagon lands within 1e-4.

### Painting speed

One segment is `extend_stroke` + `update_solids`, synchronised, over the lettering, brush 3 cells, 80 segments:

| N | median | p90 | full re-rasterisation | `Membrane` setup with the plate |
|---|---|---|---|---|
| 2049 | **2.85 ms** (target < 20 ms) | 3.3 ms | 0.20 s | 0.54 s |
| 8193 | **3.14 ms** | 3.7 ms | 3.2 s | 8.3 s |

The same test on other backends at N = 2049: Vulkan f32 3.1 ms, CPU 2.5 ms.

**A bug the first timing found.** Returning `None` for a repeated stroke point made the app's `update_solids(None)` redo the whole grid: 0.55 s at N = 2049, 5 s at N = 8193. It now returns `EMPTY_BOX`.

### Cross-check and other suites on the final code

**`crosscheck_js_MIRROR.py`: 27/27 on CUDA f64, Vulkan f32 and CPU f64.** 26 configurations plus `sq5_mode_caps`, the history caps: N = 49, 4100 steps, 4000 / 600 entries.

| backend | field after K steps / max\|u\| (stable configs) | energy history | probes |
|---|---|---|---|
| CUDA f64 | ≤ 3.4e-14 | ≤ 2.4e-14 | ≤ 2.3e-14 |
| CPU f64 | ≤ 2.2e-14 (`sq9_full_neu_corner`: 0.0 from the same seed) | ≤ 2.4e-14 | ≤ 1.5e-14 |
| Vulkan f32 | ≤ 2.2e-5 | ≤ 1.8e-6 | ≤ 1e-5 |

**Other suites:**
- `identity_tests_MIRROR.py` (CUDA): all pass, the Neumann and recorder tests included.
- `crosscheck_alignment_MIRROR.py --quick`, run on the promoted core and harness: 27882/27882.
- The app's `--shot` mode renders the plate headlessly (`ScratchCLAUDE/DevCLAUDE/app_shot_unl_MIRROR.png`).

### Speed with the recorder off and no solids

`bench_MIRROR.py --max 2049` matches the 29/09 table within noise, e.g. at CUDA f64:

| lattice | N = 161 | 513 | 1025 | 2049 |
|---|---|---|---|---|
| square, now | 0.091 | 0.088 | 0.086 | 0.207 |
| square, 29/09 | 0.096 | 0.093 | 0.086 | 0.207 |
| triangular, now | 0.113 | 0.107 | 0.105 | 0.160 |
| triangular, 29/09 | 0.117 | 0.111 | 0.104 | 0.153 |

Vulkan f32: 0.071 / 0.071 / 0.068 / 0.157.

### One long segment (added 01/10/2026, Claude Opus 5.5)

A fast drag in a zoomed-out view can make ONE segment span thousands of cells. Its update box is the segment's bounding box (~L²/2 cells on a diagonal), and re-rasterising that box re-sampled the lettering over millions of cells: 1.3 s (plate, sq) to 2.4 s (plate, tri) at N = 8193 for a 4000-cell diagonal.

- **Incremental path.** `begin_stroke` / `extend_stroke` record the edit (the membrane's cache key before it, the scene signature after it, the new segment). While the membrane's mask is still exactly the pre-edit one, `rasterize` returns `m.solid_np[box]` plus the new capsule only. The painted stroke is the last one in order, so this equals the full re-rasterisation bit for bit. `rasterize_full` never takes this path.
- **Tiled capsules.** Every capsule (both paths) is applied only on the 64 × 64 tiles within r + the tile's circumradius + 1 cell of the segment.
- **After:** 31 ms (plate, sq) and 47 ms (plate, tri) for the same 4000-cell diagonal; the rest is the core's numpy work over the box.
- Test: `solids_tests_MIRROR.py` section 12.

## Limitations and open points

**Wall types**
- Pinned walls only: no rigid/Neumann or impedance walls (held off), and no moving objects.

**Scene semantics**
- Erase strokes are ordered after all objects, so an eraser also cuts objects, including ones added later that overlap it.
- Kerning is not applied (glyph-by-glyph layout); with tracking it would not matter.

**Update boxes**
- `update_solids(bbox)` trusts the box to cover the change. The scene's methods return correct boxes; a hand-built box that is too small leaves stale cells until the next full update.

**Cost**
- Undoing (or `add_stroke` of) a long diagonal stroke still re-rasterises its whole bounding box, objects included: seconds at N = 8193 (only begin/extend take the incremental path). (01/10/2026, Claude Opus 5.5)
- The recorder costs ~50% of a step at N ≥ 4097.
- A full re-rasterisation at N = 8193 takes seconds (the text sampling).
