# Alignment panel port -- notes (grid <-> edge alignment)

*Generated from scratch -- Claude Opus 5.5 -- 30/09/2026*

Files (area "alignment"):
- `alignment_MIRROR.py` -- the port: a value-for-value mirror of the web app's alignment code
  (`FlavorRenderersCLAUDE/wave_membrane_MIRROR.html`, section 1c ALIGN-BEGIN..ALIGN-END, section 1d,
  alMeasure, alEdgeReport/alRep, the panel, ledger rows 7c / 7c-bis) + a GGUI panel.
- `crosscheck_alignment_MIRROR.py` -- runs the web app's OWN functions under Node and compares.
- `ToolsCLAUDE/js_align_dump_MIRROR.mjs` -- the Node side (through `js_harness_MIRROR.mjs`).
- Scratch: `ScratchCLAUDE/align_ggui_test_MIRROR.py` (real offscreen GGUI window + timings),
  `ScratchCLAUDE/AlignCLAUDE/*` (probes), `ScratchCLAUDE/align_*_MIRROR.json(l)` (config lists, JS dumps).

## How to run

    python crosscheck_alignment_MIRROR.py              # full sweep, 2295 configs (~6 min: JS 265 s + port 73 s)
    python crosscheck_alignment_MIRROR.py --quick      # 612 configs (~70 s)
    python crosscheck_alignment_MIRROR.py --ext        # extended sweep, 2370 configs (~5 min)
    python crosscheck_alignment_MIRROR.py --unl [--big] # 'unlicensed' invariants, no JS (37 s / 68 s)
    add --reuse to reuse the last JS dump, --arch cuda to run the core on the GPU (default cpu)
    python ScratchCLAUDE/align_ggui_test_MIRROR.py     # GGUI panel, offscreen, + timings (~2 min)

Exit code 0 = everything agrees.

## API (import alignment_MIRROR as al; `m` is a membrane_core_MIRROR.Membrane)

Every function reads `m.p` (shape, lattice, rot [rad], rad, rtheta [rad], stencil, Co, freq, sigma, src),
`m.N`, `m.tri()`, `m.domain_mask_np` (falls back to `m.mask_np`), `m.shape_poly()`, `m.clip_region()`,
`m.clip_convex()`, `m.dom_R()`, `m.sq_full()`, `m.co_limit()`, and for 'unlicensed' `m.plate_dims()`.
Angles are DEGREES unless noted; the `rot` of a list entry is degrees.

Pure (section 1c): `al_mod180(x)`, `al_gcd_i(a,b)`, `al_gcd_f(a,b)`, `al_edge_dirs0(shape, rtheta_rad)`,
`al_lattice_dir(phi, lat, M=24) -> {a,b,d} | None`, `al_class(phi, lat) -> {cls: row|diag|rat|irr, a, b, d}`,
`al_near_dir(phi, lat, length) -> {a,b,d,dA} | None`, `al_count(dirs0, rot_deg, lat) -> {rows, diags, n}`,
`al_angles(dirs0, lat, lo, hi) -> [{rot, rows, diags, n}]`, `al_best(list, mode)`,
`al_sym_period(shape, lat)`, `al_canon(rot, g)`. `lat` is 'sq' or 'tri'; `mode`/`cls` is 'row' or 'smooth'.

Live state (section 1d): `al_lat(m)`, `al_rot_deg(m)` (0 for the exact square), `al_dirs(m)`,
`al_list(m, cls='row', lat=None)`, `al_edges(m)` (the clipped domain edges: a, b, len, phi, grid, cls,
da, db, d, near), `al_measure(m) -> {'edges': [... + meas {a,b,d,tested} | None, pred, win, nCells]}`
(cached), `al_edge_report(m)` = `al_rep(m)` `-> {'edges': [a, b, len, nx, ny, off, normal, cls, la, lb, d,
near, arr, cnt, lo, hi, sum, mean, spread, measStraight, predStraight]}` (cached), `fits_inside(m, P)`,
`poly_area(P)`, `al_stencil_nb(m)`, `al_diag_name(lat)`.

Panel: `al_nothing_left(m)`, `al_best_of(m, lat, cls)`, `al_flavor_pick(m, cls) -> {lat, best, tie, other}`,
`al_note_lines(m, cls='row', L=None, edge_col=False) -> [(text, colour key)]`, `al_note(...) -> str`,
`al_snap_options(m, cls) -> {header, entries: [{rot, rows, diags, n, label, congruent_to}], selected}`.
Actions (they set `m.p.rot` / `m.p.lattice` / `m.p.Co`; THE CALLER RESETS):
`al_set_rot(m, deg, snap=0.05) -> (ok, msg)`, `al_step(m, +1|-1, cls, snap=0.05)`, `al_optimal(m, cls, snap)`,
`al_flavor(m, cls, snap)`, each `-> {changed, fields, msg, msgs}`. `snap=0.05` is the HTML slider (used by
the cross-check); `snap=None` sets the exact angle (the GGUI panel's choice, see Deviations).

Ledger and viewer: `al_ledger_rows(m) -> [(label, 'PASS'|'FAIL'|'n/a', text, tier)] x 2` (7c, 7c-bis; used by
ledger_MIRROR.py), `al_straightness(m, cls) -> [{i, cls, level A|B|C, aligned, pred, meas, offset, text}]`,
`al_level(cls)`, `al_edge_lines(m) -> [(a, b, rgb)]` (edge colouring, physical offsets), `AL_COL`.

GGUI: `gui_section(w, viewer)` draws the panel into an open `sub_window` `w`; `viewer` needs `.m`,
`.reset()`, optionally `.flash(msg)`; the state lives in `viewer.align_panel` (an `AlignPanel`: `.cls`,
`.edge_col`, `.snap_i`, `.wrap`; `.data(m)` = the cached panel content). `gui_text(s)` maps the text to
ImGui's Latin-1 font (em dash -> '--', lambda -> 'lambda', ...).

JS number formatting, exact: `js_round`, `js_hypot` / `js_hypot_np` (V8's formula, bitwise),
`js_num_str` (Number::toString), `js_to_fixed`, `js_to_exponential`, `fmt(x, d=6)`, `js_max`.

## What was verified, and how (30/09/2026, CPU f64 unless said)

| run | result | time |
|---|---|---|
| full sweep (9 shapes x 2 lattices x 15 rotations x 3 radii (x 3 rhombus angles) x N 161/257, + 9-point stencil, + Co / source / frequency / legend variants) | **80055/80055** comparisons over 2295 configs | JS 265 s, port 73 s |
| `--ext`: rotations -0.15..+0.15 deg in 0.05 steps around EVERY aligned angle (both classes, both lattices, 9-point stencil at 0 / +-0.05); radii at each clipping threshold (polygon stops fitting; every flavor edge clipped away) +-1e-9, +-1e-3, +-2e-2, thresholds by 60-step bisection; rhombus theta 20..90 incl. 33.33 / 71.1 / 89.95 (aligned angles off the 0.05 grid) | **82380/82380** over 2370 configs | JS 200 s, port 71 s |
| `--quick` on CUDA | 27882/27882 over 612 configs | |
| `--unl --big`: 'unlicensed' on the core's real plate, sq5 / sq9 / tri, 25 rotations (incl. +-0.05 around aligned ones and atan(1/2)-type rationals) x 8 radii 0.5..3.0, N 161 / 257 / 1025 | **51266/51266** checks over 816 configs (+3 cache checks) | 68 s |
| `js_num_str` vs Node `String(x)` (3022 numbers 1e-12..1e25, edge cases) | 3022/3022 | |
| GGUI panel in a real offscreen window, every button clicked once, 7 shape families | no error, all text Latin-1 | |

What is compared per config (full / ext): the pure functions on 1460 angles (al_class, al_lattice_dir,
al_near_dir at 4 lengths, al_angles, al_gcd_f, al_canon); alRotDeg/alLat, alDirs, alCount, alSymPeriod,
fitsInside, alNothingLeft, alEdges, alMeasure, alEdgeReport; ledger rows 7c and 7c-bis (HTML -> text);
per class (row, smooth): alList, alCanon per entry, alBestOf(sq/tri), alFlavorPick, the NOTE text, the snap
list (header, labels, selected), and the outcome of alStep(+1), alStep(-1), alOptimal, alFlavor (rotation,
lattice, Co, every flash text). Floats to 1e-9 relative, everything else exact.

Two tolerances are explained, not hidden (both printed by the cross-check):
- **normal mod 360**: 1 edge normal reads 0 in the JS and 359.9999999999996 here (the last bit of a polygon
  vertex decides the sign of a ~1e-16 component).
- **sliver edges** (`--ext` only): at rad = nothing-left threshold - 1e-9 the clipper leaves edge pieces of
  1.06e-6..1.19e-6 cells (just over the JS's 1e-6 cut). Their direction comes from two vertices 1e-6 apart
  whose coordinates carry last-bit cos/sin differences (the C runtime vs V8's fdlibm): nx/ny/off/win/near.dA
  agree to only 1e-9..4e-8. Pieces under 1e-3 cells compare float fields at 1e-13 * N / len (21 fields in
  11 configs needed it); their classes, counts (cnt 0, on the array edge) and every text agree exactly.

## 'unlicensed' (port only)

The core's plate: half-width R = dom_R(), half-height R/2, corner radius 0.065 R, 16 chords per corner
arc (68-point CCW outline), rotated by p.rot; the lettering and screws are SOLIDS (m.solid_np), not domain.

Model: aligned EXACTLY like a rectangle -- edge directions [0, 90, 0, 90] + rot; symmetry period 180
(so 90 on sq, 60 on tri, the rhombus rule); the straight sides (long 1.870 R, short 0.870 R) are the edges.
Corner-arc chords are identified geometrically (not on a side line, not on the array edge) and flagged
`arc`: they still take part in the nearest-edge assignment of boundary cells / zeroed nodes (so arc cells
never pollute a side's period or straightness), then are dropped from every list, count, note and ledger.
Aligned angles: row sq [0, 90], smooth sq [0, 45, 90]; tri [0, 30, 60, 90] for both classes.

Checked by `--unl` against the core's real implementation (shape_poly, plate_dims, dom_R, the mask,
rotation, both lattices, clipping):
- 4 straight sides of length 2(hw - rc) / 2(hh - rc), CCW outline, dirs / symmetry period as above;
- every clipped piece is a side piece (on exactly one side line, phi = side + rot to 1e-9), an array-edge
  piece, or an arc piece (endpoints on a corner circle or on the array edge); when it fits: 4 edges,
  64 arc pieces; al_nothing_left <=> no side piece survives and the plate does not fit;
- the MASK agrees with the outline (shape, size and rotation): every boundary cell and every zeroed node
  is within one lattice spacing (+ the 0.0012 rc chord sagitta) of the clipped outline;
- rule vs mask: ledger 7c and 7c-bis never FAIL; every judged edge's measured straightness equals the rule
  (rows straight; the sq 45-degree side straight with the 5-point stencil only; tri zigzags not); every
  measured period equals the predicted one;
- the scans are identical with the plate's lettering/screws attached and detached (every 7th config);
- the panel runs end to end (note, snap list, straightness, step / optimal / flavor).

Measured at N = 161 (sq): all 4 sides rows at rot 0 (zeroed nodes +0.800 / +0.400 beyond the long / short
sides at rad 0.8); at rot 45 all 4 are 45-degree diagonals, straight with the 5-point stencil and NOT with the
9-point one (as for the n-gons). Tri at rot 30: long sides zigzag (d = 1.732, not straight), short sides rows.
Clipping: arcs-only (every straight side clipped away while corner-arc pieces survive) never happens: 0 cases
over rot 0..90 step 0.5 x rad 1..4 step 0.01, both lattices, N = 161 (the corners are small: 0.065 R).
So the JS wording "every flavor edge is clipped away" stays true for the plate.

NOTE additions for 'unlicensed' (port-only lines): the sizes line and a direction rule
("the long sides lie along lattice direction alpha iff rot == alpha (mod 180), the short sides iff
rot == alpha - 90").

## Deviations from the web app (all intentional)

1. 'unlicensed' (above). Unknown shapes get empty lists, a one-line note and 'n/a' ledger rows, never an
   exception.
2. The GGUI panel calls `al_set_rot / al_step / al_optimal / al_flavor` with `snap=None`: the port's
   rotation slider is continuous, so an aligned angle off the 0.05 grid (rhombus at theta = 33.33 deg)
   lands exactly instead of "slider could not land on ... (step 0.05)". The cross-check uses snap=0.05.
3. The GGUI panel shows, in addition to the JS NOTE: a per-edge straightness list (`al_straightness`,
   levels A row / B shortest diagonal or zigzag / C other -- a port-only letter), and the two ledger
   verdicts. The snap list is a "< snap / snap > / apply snap" stepper (GGUI has no combo box).
4. Scans read `m.domain_mask_np` (the shape alone), not `m.mask_np` (which has the solids cut out):
   interior solids are not domain edges. With no solids they are the same object.
5. Caching (the JS recomputes on every render): `al_measure` / `al_edge_report` are cached per geometry
   (shape, lattice, rot, rad, rtheta, N, stencil) AND mask content -- the same array object is an
   immediate hit; a new array with the same geometry (every `set_co` rebuilds the domain) is compared with
   `np.array_equal` once for all scans; a stale mask (geometry changed before the reseed) never survives
   the reseed (tested). The panel's text is rebuilt only when (geometry, mask object, class, legend, Co,
   freq, sigma, src) changes.
6. `js_num_str` now follows ECMA Number::toString exactly (was Python repr for non-integers: '1e-05'
   instead of '0.00001', '1e-07' instead of '1e-7'). ledger_MIRROR.py uses it as `s_`; only such small or
   huge numbers print differently, now as the JS does.

## JS behaviour found (mirrored, NOT fixed -- the web app is frozen)

- **alMeasure reports the unsigned vector**: `meas = {a: v.a, b: v.b}` of the tested primitive vector,
  not the signed shift it applied (for edges whose direction projects negatively). Only the length `d`
  is used for the verdict, so no verdict is wrong; the "(a,b)" printed for irrational edges in row 7c can
  have the opposite orientation to the edge.
- **Flavor-pick text with 'smooth'**: the pick ranks (rows + diags, rows); when both lattices reach the same
  total but one has more rows, the note reads e.g. "optimal grid flavor: square @ 0.00 (4/4) vs 4/4 on the
  triangular" -- equal counts, no tie wording (seen on 'unlicensed', tri, rot 30; also n-gons).
- **Sliver edges**: a clipped piece just over 1e-6 cells counts as a surviving flavor edge, so
  alNothingLeft turns true only when the last sliver drops under the 1e-6 cut. Example (n12, sq, rot 0,
  N 161, rad 1.4502444711519413 = threshold - 1e-9): the NOTE says "the array clips this shape: 8 flavor
  edge piece(s) survive" -- all 8 are 1.11e-6 cells long. Harmless, numerically fragile.
- (Not a bug, a conditioning note: an edge's outward normal at ~0 degrees is stored mod 360, so it reads 0 or
  359.9999999999996 depending on the last bit of a vertex; nothing in the JS prints it, and the port's
  straightness text uses it mod 180.)

## GGUI (taichi 1.7.4, RTX 5070 Ti, CUDA)

- `window.show()` on a `show_window=False` window SEGFAULTS (access violation in window.show, measured
  30/09/2026); offscreen frames are drawn with `get_image_buffer_as_numpy()` (or `save_image`).
- Panel cost per frame, cached (median of 30): **0.37-0.58 ms** of Python + imgui calls (35-46 text lines),
  at N = 1025 and 4097 alike (the scans are cached; the per-frame work does not depend on N).
- First frame after a geometry change (scans + note + ledger rows): N = 1025: 0.10-0.19 s;
  N = 4097: 0.49-0.60 s (al_rep 0.16-0.19 s, al_measure 0.28-0.41 s). The reseed itself is 1.0-2.6 s there.
- After `set_co` (new mask object, same geometry): next frame +8-11 ms at N = 4097 (one content compare),
  no rescan (was a full rescan before the 30/09 cache change).
- N = 8193 not timed here (expect ~4x the N = 4097 scan times, ~2-2.5 s on a geometry change).

## Open issues

- Scans at N = 8193 are ~2 s per geometry change (numpy, host). A Taichi kernel for the boundary/zeroed-node
  scans would cut it; not needed at <= 4097.
- The panel's straightness list and level letters are port-only and have no JS reference; they are derived
  from al_rep fields that are cross-checked.
