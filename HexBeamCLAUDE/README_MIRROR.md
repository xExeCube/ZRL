# HexBeam (LatMosaic sub-project, ZRL) — 3D-flattened lattice apps

*Written by Claude Fable 5, 31/08/2026.*

Apps that hold integer 3D lattice state and flatten it to the 2D triangular-lattice
atom machinery (see `../HexMosaicCLAUDE/README_MIRROR.md` for the geometry scheme).
Build order agreed 31/08/2026: prisms first, beams second.

## hex_prism_MIRROR.html — SHIPPED 31/08/2026 (v1)

The HexMosaic floor plan extruded into a **prismatic honeycomb** (user's "Option 3").
Every tile gets an integer height; one height level shifts vertices by exactly one
lattice step along a selectable extrusion direction (6 choices), so:
- top faces are the exact 2D tiles, translated on-lattice;
- every wall is an exact 60°/120° unit-rhombus stack (ledger-sampled live);
- rendering is a **per-atom painter sweep**, provably exact: occlusion couples only
  atoms within the same lattice band along the extrusion direction (the swept strip
  of an atom stays inside its band; adjacent bands share only measure-zero lines),
  and within a band the centroid order along the direction is total. Sort all atom
  columns by centroid·d̂ descending, draw walls-then-top with full overdraw. No
  z-buffer, no 3D engine, no float geometry.
- Gaps and pit-painted cells draw nothing → you see the far walls of columns behind
  them (real-excavation look, automatic).

Features: heights (flat / random / dome / ramp + height seed, decoupled from the
floor-plan seed), raise/lower/set-h click tools, window presets + flat-window chroma
columns (whole column selectable in GIMP), wall shade sliders (two visible edge
families; the family parallel to the extrusion direction is provably never visible),
PNG/JPEG/SVG/mask exports (mask = white on every visible face of window columns,
invert + borders-in-mask options), pan/zoom, live ledger (area/occupancy bijection +
sampled wall-parallelogram exactness + face count + height range).

Verified 31/08: 200 random sample points — rendered pixel == painter-owner face
fill, mask pixel == owner's window role, 0 mismatches; determinism per seed pair
(floor seed + height seed); raise/lower/set-h clamping; all 6 directions.

Adversarial review 31/08 (9 agents, 6 confirmed, 0 refuted, all fixed; the
node-executed painter-exactness and wall-coverage checks found NO geometry
defects): heights are now a first-class edit channel — sculpted columns
(raise/lower/set-h set an hManual flag) survive *grow +25%* (all pre-existing
heights are preserved across grow; only new tiles roll) and survive non-force
re-applies; the explicit **apply heights** button is the full re-roll that clears
flags. Switching extrusion direction re-derives an applied *ramp* so the
staircase follows the new direction (sculpted tiles kept). Export filenames stamp
the seeds actually consumed, not the live inputs (re-dicing without regenerating
no longer mislabels files). Style-only changes (colors, shades, border width,
flat-windows) patch fills on existing SVG nodes instead of rebuilding ~15k
polygons per slider tick. Ledger wall-sample count fixed (reports exactly what it
checked, cap 50).

Known v1 limits (documented, deliberate):
- Tile-top borders draw after all faces, so a border can overpaint a nearer column
  where a tall front column partially covers a bordered top — border width defaults
  to 0 (the iso look is borderless).
- Same-color 0.02 overpaint stroke per face suppresses hairline seams but displaces
  color boundaries ~1% of an edge by draw order (same tradeoff as HexMosaic masks).
- No image layers yet (planned v2: textured blocks / image-mapped tops).

## hex_beam_MIRROR.html — SHIPPED 01/09/2026 (v1)

ℤ³ beam walks flattened along (1,1,1) ("Option 1"; Reutersvärd machine).

- **Projection** (corrected from the early draft; node-verified): cell (x,y,z) ↦
  lattice vertex (x−y, y−z); +x ↦ (1,0), +y ↦ (−1,1), +z ↦ (0,−1) — three unit
  lattice steps at 120°. (1,1,1) ↦ 0, so a projected column = a diagonal class;
  crossings are pure integer detection.
- Each cell contributes its 3 visible faces (+x/+y/+z) = the 3 rhombus families,
  2 atoms each (together = the hexagon at v — the canon's cube→hexagon fact).
- **Hidden surfaces per atom, exactly**: each face's depth is closed-form linear in
  the projected point (z-face on z=Z: a+2b+3Z; x-face: 3X−2a−b; y-face: 3Y+a−b);
  pairwise depth order cannot change inside an atom (face-plane intersection lines
  project onto atom-edge lines), so max-depth-at-centroid is the true visible face.
  The ledger re-verifies this on live contested atoms (×3 interior points). This
  rule alone handles runs, miters, junctions, ends — no special cases.
- **Illusion**: crossing pairs (top two beams contesting an atom) get a seeded
  per-PAIR flip (probability slider: 0% honest, 100% all inverted, between =
  coherent coin-flip). Flips swap the atom to the other beam's best face — cyclic
  occlusion renders fine because there is no global sort to break.
- **Tribar preset** = Penrose triangle: arms +x,+y,+z, 3D closure error L(1,1,1)
  projects to zero — the HONEST renderer already draws the impossible figure
  (verified visually + ledger closure line). 100% flip yields the complementary
  paradox (all crossings inverted — a different coherent impossible figure).
- **Knot walk**: axis-turning beam chain, 3D-collision truncation, crossing-bias
  weighted candidate scoring (revisit own projected columns at different depths).
- Beams click-paint: palette / window (flat chroma for GIMP masks) / custom /
  delete; mask = white where window beams own atoms; PNG/JPEG/SVG/mask exports;
  two-tier restyle; seeded determinism (figure seed + illusion seed).

Verified 01/09: knot + tribar render correctly; mask 19/19 sampled window atoms
white; deterministic; ledger green (depth-order check, 0 ties, 0 collisions).

Adversarial review 01/09 (12 agents, 8 confirmed, all fixed). The big one: the
first build's "crossings" included solid 3D joints (adjacent cells whose hexagons
share atoms), so flips could paint buried interior faces and notch elbows. Fixed
by gating both pair registration and per-atom flip application on candidate
provenance: a flippable rival must come from a cell in the SAME projected column
a nonzero multiple of (1,1,1) away — a genuine back-beam surface. Result verified:
tribar reports exactly 1 crossing pair (the closing 0|2; 100% flip yields the
canonical 3-fold-symmetric Penrose winding with elbows intact; honest mode is the
sculpture-photo closure — both are legitimate windings), knot crossings dropped
13→4 (joints excluded). Also fixed: flip coins are identity-keyed (hash of both
beams' cell content + illusion seed) so deleting a beam no longer rescrambles
unrelated crossings (verified: 3/3 surviving coins stable) and the probability
slider is monotone (higher % = superset); tribar closure is MEASURED from cells
(proj + 3D gap shown; folds into the ledger flag) — closure error is
(L−1)·(1,1,1), note corrected; depth-theorem sampling now covers crossing atoms
and junction atoms separately (labeled honestly); re-clicking the active figure
button no longer wipes paint/deletions; dead customCol live-listener removed
(custom paints snapshot at paint time, matching siblings).

v2 candidates: rhombohedron beams ("Option 2" — same lattice rotated 60°, six
beam directions, node-verified 31/08) as a beam-type toggle; manual beam drawing;
region-merged SVG output; borders.

## Necker mode (added 01/09/2026, hex_beam)

Style → Necker mode: **solid** (light-from-above prior: tops brightest),
**hollow** (brightness order reversed = relit from the opposite diagonal — biases
the concave reading), **ambiguous** (all three families equal — releases the
lighting prior; the figure goes Necker-bistable). Style-only pass, exports honor
it. Mechanism note: parallel projection discards the depth sign; shading is the
prior that picks it. Same toggle can be ported to hex_prism on request.

## Rhombohedron beams (added 01/09/2026, hex_beam)

Figure → beam family: **cubes** / **rhombohedra** / **mixed (6 dirs)**. Cells now
carry a kind; a rhombohedral cell is the parallelepiped on (1,1,0),(1,0,1),(0,1,1)
— all six faces 60° rhombi in 3D — and projects to the SAME 3-rhombus hexagon as
a cube, with beams running along the other three lattice directions. Faces are
built generically from edge vectors (far face if its outward normal faces the
viewer; 2 atoms via the short diagonal; closed-form linear depth from a 2×2
inverse) — verified to reproduce the hand-derived cube formulas exactly on 200
random cells. Rhombohedral tribar preset closes with gap (L−1)(2,2,2) → the
downward-pointing Penrose triangle. Cube-family knot walks keep the v1 rng
sequence byte-for-byte, so old seeds reproduce.

**Mixed scenes need exact splitting.** The constant-depth-order theorem holds
within a kind (pure cube, pure rhombohedron — all face-plane intersection lines
project onto lattice directions) but a cube face and a rhombohedron face can meet
along a line that projects *vertically*, off-lattice, cutting through atoms. So
buildScene resolves PIECES: an atom whose two deepest faces swap order inside it
is clipped exactly along the depth-equality line (Sutherland–Hodgman on the
triangle) and each side is resolved independently. The ledger reports split
atoms and any residual "approx pieces" (a third face crossing inside a split
piece — counted, not hidden; never observed so far). Cross-kind 3D collisions
are not detected (interpenetration allowed; same-kind collisions truncate as
before).

Review 01/09 (15 agents, 10 confirmed, 1 refuted, all fixed): origin cell is
now registered under the kind of the beam that owns it (mixed walks could
re-enter (0,0,0)); occupancy is GEOMETRIC across kinds (a cell is rejected if it
shares a base with, or has its centre strictly inside, an existing cell of any
kind — a cube 3 levels in front of a rhombohedron's base sits inside it), and the
ledger counts overlaps the same way; the crossing threshold depends on the BACK
cell's kind (|Δsum| ≥ 3 behind a cube, ≥ 6 behind a rhombohedron) in both the
generator and the resolver; a flipped crossing now draws the rival beam's own
deepest face (its real surface) rather than the same-column face, and hexagon
rim pieces adjacent to a crossing flip with it (no more single-triangle notches);
atoms are clipped recursively against every candidate whose order changes
inside them, and the 3-point honesty check runs on every contested piece; tribar
+ "mixed" coerces the family button to cubes and filenames stamp the family
actually used. Verified after fixes: cube seeds byte-identical, 120 mixed scenes
with 0 interpenetrations / 0 approximate pieces / 0 depth failures.

Presets added: **hex ring** (one false joint, sides [L,L+1,L,L,L,L]) and
**double hex** (two rings sharing a beam) — closures measured live: cubes
k = −1/+1 (depth jump ∓3), rhombohedra k = −2/+2 (∓6).

## The three closure mechanisms (settled 01/09/2026, from the user's seeds)

A beam ring can "close" in projection three ways:
1. **True closure** — end cell = start cell in 3D: a real object (regular
   hexagons: the cube's skew/Petrie hexagon).
2. **Column overlap** — end cell in the same projected column as the start cell,
   k(1,1,1) away: the tribar mechanism (a contested hexagon; flippable).
3. **False joint** — end cell *adjacent* to the start cell in projection (looks
   like a solid corner) but k(1,1,1) away in 3D: the Penrose-stairs mechanism.
   No column overlap, so the renderer treats it as an honest joint and draws a
   solid-looking corner; the ledger shows 0 crossings.
Both user hexagons (seed 29156, segs 9, len 7–8, bias 97% → beams 3–8, 44
cells; seed 52369, segs 11, len 8–9, bias 100% → every closed 6-ring) close by
mechanism 3 with k = −1 (depth jump −3): **impossible**, not valid 3D shapes.

**Tiling / sharing sides (ℤ²): yes.** Impossibility is per-loop; a shared beam
serves two rings; honeycomb vertices are valid cube corners (the three edge
directions at a vertex lift to three different axes). Constructed and verified
in-app: ring A sides [6,7,6,6,6,6] along +x,−z,+y,−x,+z,−y from (0,0,0) (end
(−1,0,−1) = +y −(1,1,1), k=−1) and ring B sharing A's first beam, then
+z,−y,+x,−z,+y with [7,6,6,6,6] (end (6,0,1) = −y +(1,1,1), k=+1): 11 beams,
0 collisions, 0 crossings, ledger green, two fused impossible hexagons.
Rule for larger tilings: at each vertex the three beam-end depths define three
pairwise jumps summing to zero; a face is impossible iff its six jumps (plus any
side-length asymmetry) don't sum to zero; Σ over faces of a patch telescopes to
the boundary — mixed-sign face gaps are unconstrained, all-same-sign patches are
bounded by their perimeter.

**ℤ³:** the lift of an impossible tiling is a *sculpture*, not a space-filler —
beams at staggered depths, disjoint at every false joint, a 2-parameter sheet
aligned to one viewpoint. "Impossible" is a property of a projection; no solid
cell of a space-filling honeycomb can carry it. The possible cousin is real: the
zero-gap beam honeycomb lifts to the skew-hexagon chicken-wire surface (the edge
graph of the cube-stack surface seen down (1,1,1)). The closest legitimate 3D
object to an "impossible hexagon" is the **helical beam ring**: a hexagonal walk
that climbs k(1,1,1) per lap — an infinite spiral staircase whose every lap
projects to the same closed hexagon.

## Closure-gap hierarchy (node-verified 01/09/2026)

Single-axis lifts of the six lattice edge directions are unique, so a beam
cycle's 3D closure gap is determined by its side lengths alone:
- triangle on (1,0),(−1,1),(0,−1): gap = (L,L,L) ALWAYS → the tribar is
  impossible at every size (why triangles are the canonical impossible figure);
- regular hexagon: gap = 0 → closes in 3D as the cube's skew (Petrie) hexagon —
  a real object that merely looks flat;
- aba hexagon (opposite sides differing by k): gap = ±k(1,1,1) → impossible,
  chirality flips under 180° rotation;
- plain rhombus 4-cycle ((1,0),(0,1),(−1,0),(0,−1)): gap never has a y-component
  → cannot be diagonal-hidden; simple rhombi can't be made impossible by length
  asymmetry (compound figures like two tribars sharing a crossbar can).

Tiling note: impossibility is per-loop (the H¹ obstruction), so 2D figures share
beams freely (the knot generator already produces shared-beam compounds).
Honest-3D constraint: face gaps telescope (sum over faces = boundary gap), so an
infinite all-same-chirality tiling is obstructed; alternating ±k is legal. aba
hexagons cannot tile alone edge-to-edge (a,b-length matching fails at degree-3
vertices — every vertex would need a hexagon with two consecutive same-length
sides); aba hexagons + side-b triangles (truncated-triangle picture) is the
candidate "impossible honeycomb", with the filler triangles themselves tribars.
Unbuilt — verify the vertex matching in-app before trusting the mixed tiling.

## hex_helix_MIRROR.html — SHIPPED 03/09/2026

The possible cousin, built: a hexagonal walk with sides (L+k, L, L+k, L, L+k, L)
along +x,−z,+y,−x,+z,−y climbs exactly k(1,1,1) per lap. Views: **down the axis**
(every lap projects to the same hexagon — the compression theorem, checked live;
a lap-shear slider explodes the stack), and **y-side / x-side** axis-aligned
elevations (the coil as a climbing zigzag; the answer to "what does the y-side
look like"). **Triangle connectors** (0–6 per lap): k×x, k×y, k×z runs joining a
corner to the same corner one lap up — the six axis orders all project to the
same closed unit triangle, and the placer tries each order against full
occupancy (rings are laid first; certain corners block certain orders because
the ring enters/leaves the target along that axis). **Multi-start** (clamped to
k): j·(1,1,1) shifts give up to k interleaved threads — k=2 renders the double
helix; color-by-start shows both threads project IDENTICALLY (the front one
hides the other in the axis view). Ledger: same-projection set equality across
laps, measured climb k(1,1,1), connector unit-step chain closure, 3D collision
count. Verified 03/09: full sweep L∈{3,6,9} × k∈{1,2,3} × starts∈{1,k} ×
conn∈{0,3,6} — zero collisions, all connectors either placed or honestly
counted as skipped; double helix k=2/starts=2/conn=3: 12 connectors, 0
collisions, ledger green.

Review 03/09 (9 agents, 7 confirmed, 0 refuted — all fixed): the y-side view
was PARITY-MIRRORED (a max-y painter with u=+x is no physical camera — the
image reversed the helix's apparent chirality; now u=−x−1, a true −y view);
the climb check was vacuous at ≤2 laps and never measured the (0,1) pair (now
measures every adjacent pair on every thread, reports the pair count, shows
n/a at 1 lap, and a rigid-displacement tamper test confirms it catches
corruption); the same-projection check now covers all (start,lap) rings, not
just thread 0; the connector chain check is pinned to the exact monotone
3k−1-step x/y/z-run family (was: any unit-step walk to the right endpoint);
PNG export bounds include the connectors' shear offset and atom-vertex extent
(sheared triangle decorations were croppable); dead code and a label-format
nit removed. Re-swept after fixes: all 27 configs green.

## Ring + rhombus tiling result (07/09/2026, node-verified on torus + plane)

User conjecture tested: "the gap left by 4 hex rings is filled by one 2-tribar
rhombus" (two tribars fused along a shared beam — outline a rhombus).
- SHIPPED hexTile layout (px=2L+5, pb=2L+6): FALSE, and not by shape — there is
  no enclosed gap at all. The inter-ring space is ONE connected channel network
  (3 cells clearance both directions; per-2-ring-period voids exactly
  [channel, interior, interior], channel = 2L²+30L+53 per ring, ODD, while every
  2-tribar footprint is EVEN — parity forbids any exact fill). Bonus: whole
  2-tribar rhombus FIGURES of every size L′=2..L+5 float inside the channel
  without touching (at L=4 an L′=9 frame fits wrapped AROUND a ring).
- TIGHT layout family px=2L+1, pb=2L+3, no row shift, flip offsets (L−1, L):
  rings tile with ZERO inter-ring gap except two enclosed diamonds per 2-ring
  period — SOLID lozenges (L−1)×(L+1) and L×(L+2), each bordered by exactly 4
  rings (torus-verified L=3..8, plane-verified 4×4 patch, 0 overlaps). So the
  conjecture is TRUE in the tight layout up to one twist: the filler is a plain
  solid lozenge. Near-miss: its atom count EQUALS the 2-tribar rhombus for L≤4
  (2(L²−1)=|R(L−1)|, 2L(L+2)=|R(L)|; 30=30, 48=48 at L=4) but the shapes differ
  (best overlap 23/30, 39/48) and counts diverge for L≥5. Impossibility note:
  any cube-cell figure projects to a union of hexagons, which cannot produce the
  lozenge's 60° corners — the exact filler can only be a drawn 2D rhombus, never
  a beam figure. 2-tribar rhombus construction detail: the two triangles close
  onto the same start column and share exactly one 3D cell (the far corner —
  a solid joint, not a collision). Search scope caveat: 2 rings per fundamental
  domain (checker + row chirality alternation); ≥4-ring bases unsearched.
  SHIPPED 07/09/2026 as the "tight tiling" preset (below).

## Tight tiling preset, SHIPPED 07/09/2026

genHexTile gained a `tight` mode (figure button "tight tiling"): a-pitch 2L+1,
b-pitch 2L+3, no row shift (rows advance diagonally — the natural rhombic
fundamental domain), flip offsets (L−1, L). The moat closes completely; the
only gaps are the two solid lozenge windows per 2-ring period, crisp rhombus
cutouts for the overlay use. The flip b-offset L is even for even L, so the
0.5-unit screen-x zigzag of flipped rings disappears at even arm lengths.
Ledger gained a live tiling audit for both layouts: atoms shared between
rings (must be 0 — loose has the moat, tight abuts edge-to-edge with zero
contested area). Exports carry the same _L/_n dims with fig name hextight.
Verified before ship: the EXACT implementation formulas re-run through the
tiling agent's atom rasterizer (scratchpad tilefit/tightpreset.js) — L=3..6,
n=4: 0 atom overlaps, all 16 ring interiors, enclosed voids exactly the
2(L²−1)- and 2L(L+2)-atom lozenges each bordered by 4 rings, nothing else;
browser: tight L=6/n=3 and L=4/n=4 ledger green (0 shared atoms, 0 crossings,
0 collisions, 0 ties), loose layout regression green with the new audit line.

Review 07/09 (diff review, 1 confirmed, 0 refuted — fixed): the cross-ring
audit derived ring membership as ⌊beamId/6⌋, which breaks after paint-delete
reindexes beam ids — the 6-beam grouping straddles real ring boundaries and
ordinary elbow-joint atoms count as cross-ring (verified: delete beam 0 in
tight L=6/n=3 → false crossRing=32, "LEDGER BROKEN" until regenerate). Fixed
by stamping .ring on each beam at generation (survives splice+reindex since
b.id = array index is restored); verified live: deleting beams 0 and 7 from
the tight tiling keeps the ledger green with 0 shared atoms. Lesson: ledger
identities must key on generation-time provenance, never on array positions
that mutate under user edits (kin to the identity-keyed flip coins of 01/09).

## Terminology (user's canon, 07/09/2026 — supersedes earlier drafts)

- **3-bar** (Penrose / impossible triangle). **4-bar** (Penrose / impossible
  rhombus); **5-bar** = double 3-bar = a 4-bar plus a diagonal bar. **6-bar**
  (impossible hexagon; at least 2 types; tiles in 1D by bar-sharing).
- **Bar**: one side of any object — an edge with thickness and 2 shading areas;
  splits into unit triangles, unit rhombohedra and/or unit cubes (= code `beam`).
  **Bar edges**: (1) *outer edge*, (2) *inner edge* (shared with the shape formed
  inside the object), (3) *shading edge* — the line along the middle of a bar
  where the shading changes (= the projected 3D edge).
- **Shared bars**: a bar belonging to two objects at once (the double hex; a
  6-bar chain shares one bar at each end). **Bar vertex**: the pseudo-3D vertex
  = a unit cube; can be shared like bars.
- **Implied-z**: all objects are 2D; the illusion is a LOCAL implied depth,
  relative to each bar (given by its two shading sides + shading edge), that
  is not required to make global sense — the user's guess, and exactly
  Penrose's cohomological reading: the ledger's closure D = e + k(1,1,1)
  measures the failure of the local depth to integrate around a loop (k ≠ 0).
- Kept from the draft: **window** (a gap; **lozenge** if its sides differ),
  **cell**, **column**, **crossing** (same column, different depth).

## Abut tiling preset (disjoint rings), 07/09/2026 — was briefly "chain tiling"

Disjoint rings joined leg-to-leg along chains (parallel ±b legs abutting outer
edge to outer edge, chirality alternating) and chains stacked so a ring's
a-bar end meets the next ring's a-bar start corner-to-corner with the bars
collinear (chirality constant). Base coordinates (flipped rings carry the
(−2, 2L+1) centre-aligning offset at odd chain positions): chain step
C = (2L+2, −(L+1)), stack step S = (0, 2L+2). Atom-verified
(tilefit/chainscan.js, chaintiling.js) L=3..8: zero overlaps, every interior
intact, windows = TRUE 60° rhombi of side L and L+1 alternating along the
chain (2L² and 2(L+1)² atoms — the wobble's long bar makes the bigger one),
one per ring, each bordered by 4 rings. Browser: L=6/n=3, L=4/n=4 green.
Renamed to "abut tiling" once the user's picture turned out to be a FUSED
structure (below) — rings that share nothing vs rings that share bars.

## Chain tiling preset = the user's 6-bar + 4-bar tiling, SHIPPED 07/09/2026

The user's construction (LatMosaicCLAUDE/doublehexbeamtiling*.png and the
annotated 4-hexagon sketch): every 6-bar is an identical copy; down a column
each 6-bar's top bar IS the bar above's bottom bar (the double-hex shared-bar
mechanism, chained — "each 2 6-bars share 1 bar, each 3 share 2"), and side by
side the columns share the side-vertex CUBE where two straight bar-lines
cross in an X; each 4-bar between columns "shares all bars with 4 6-bars"
(its bars are the 6-bars' own legs) and each 6-bar shares 2 bar vertices with
its 2 flanking 4-bars. In ℤ³ (3D model tilefit/fusedN.js, verbatim walk):
column step TV = (1, −L, L+1) [projects to (L+1, −(2L+1)), the half-unit
lean], column pitch TH = (L, −L, −L) [projects to (2L, 0)]; shared bars and
vertex cubes are created once, so the whole tiling is ONE connected cube set.
Verified L=4,6,8: zero unintended cell coincidences (shared slots exactly
n(n−1)L bars + (n−1)n cubes); every 6-bar keeps its false joint
D = (−1,0,−1) = +E1 − (1,1,1); windows are all identical LOZENGES of interior
sides L−1 and L−2 (2(L−1)(L−2) atoms; 40 at L=6, 12 at L=4), each bordered
by 4 six-bars. The 4-bar result: walking one window's boundary through the
actual cubes (27 distinct cubes at L=6, 4L+3 in general; run-lengths
+z×7, +x×1, −y×6, −z×6, [joint], +y×6) closes in 3D except for exactly ONE
break — the inherited false joint (−1,0,−1) — so the rhombus is a genuine
impossible 4-bar. Its bars are L+1, L, L, L: a wobbled rhombus, which is why this
4-cycle can be impossible while an equal-sided rhombus 4-cycle cannot
(closure hierarchy, 01/09). Chirality-alternating columns (the literal
double-hex chain) also fuse consistently (tilefit/fused.js) but give
alternating windows — a side-(L−1) rhombus and an (L−1)×(L−3) parallelogram —
so the identical-copies column is the one shipped. App: genHexChain, button
"chain tiling"; ledger re-walks every 6-bar from its stored base against the
SCENE: all six sides present, closure k = −1 measured for every complete
ring, shared bars / vertex cubes counted from cell ownership (browser:
L=6/n=3 → 48 beams, 291 cells, 9/9, 9/9, 6/6, 6/6; L=4/n=4 → 84 beams,
340 cells, 16/16, 16/16 — cell counts equal the node model's). Lesson:
"tiling" for these objects means bar/vertex SHARING, not packing — the whole
07/09 search history was packings of disjoint rings.

Review 07/09 (fused chain tiling diff, 4 agents, 2 confirmed, 1
informational-refuted — all fixed): the generator, trims, duplicate/collision/
crossing freedom and window/shared-slot claims were independently CONFIRMED;
(1) the README's "29 cubes" was an array-entry count with a repeated vertex
cube — 27 distinct (4L+3), and the model script printed a phantom D=(0,0,0)
"break" from that repeat (script fixed to collapse repeats and count distinct
cubes; also its atom-extent label used a+b instead of a+b+s for down atoms);
(2) the ledger's "false joints k=−1" was computed from the synthetic re-walk,
so it could never fail (9/9 even on an empty scene) — now gated on the ring
being complete in the scene, and the shared-bar / shared-vertex counts are
measured from cell ownership instead of quoted as constants. Lesson (again):
a ledger line must be able to go red; if the code path cannot produce a
failure, it is a label, not a check. Verified after the fix with a FORCED
synchronous restyle: delete the shared bottom bar of ring 0 → ✗ LEDGER
BROKEN, 6-bars complete 7/9, false joints 7/9, shared bars 5/6, vertices
6/6. Testing lesson: the preview pane fires requestAnimationFrame only on
demand, so a ledger refreshed through scheduleRestyle() reads STALE after a
scripted mutation — call restyleScene(true) before reading it (an earlier
"verified" read of the ring-tag fix was stale; re-verified properly 07/09).

Why the earlier "no rhombus-gap tiling" verdict missed it: that search
parameterized layouts as horizontal rows with a-pitch ≤ 2L+8; the chain
lattice ⟨(2L+2,−(L+1)), (0,2L+2)⟩ has no horizontal vector shorter than
4(L+1)², so it lay outside the box. Lesson: search reduced lattice bases, not
row/pitch boxes.

Measurement notes: the pictures were measured by a pure-node PNG decoder
(usertiles/measure.js; 20 px per lattice unit, L=6). Ring CENTRE offsets are
not base offsets — a flipped ring's centre sits (2, −(2L+1)) from a normal
ring's at the same base — an error that produced hundreds of phantom overlaps
until corrected. The picture's stack offset measured (0, 2L), which at the atom
level is NOT corner-touch but the one-shared-cell case (the two rings'
slanted legs lie on the same line a+b=2L and share their end cell) — i.e. the
hand edit sits between the user's (a) [chain tiling, S=(0,2L+2)] and (b)
[shared-cell "shifted" variant, S=(0,2L)]; S=(0,2L+1) grazes at two corners.
The scan also found tighter families with only single-atom triangular gaps
(e.g. L=6: C=(14,−7), S=(6,8), fundamental domain 154 cells/ring), backlog.

## Packed honeycomb + Sierpiński 3-bar + UI rework, SHIPPED 07/09/2026

**Packed honeycomb** (figure button): the abut chain with stack S = (L, L+2)
— the densest packing of DISJOINT wobbled 6-bars the full contact scan found.
Fundamental domain (L+1)(3L+4) cells per ring vs the zero-gap floor
(ring atoms + interior)/2 = (L+1)(3L+4) − ½: the only gaps are single atoms,
exactly ONE unit triangle per ring where three rings meet, at every L
(verified L=3..9, TileFitCLAUDE/sierp3_MIRROR.js part 1). That atom is the
wobble's irreducible residue — a regular hexagon tiles gaplessly, but a
regular 6-bar cannot be impossible (its false joints would cancel), so one
atom per ring is as close as an impossible honeycomb gets. Density ladder at
L=6 (domain cells per ring; floor 153.5): packed 154, tight 195, abut 196,
loose 306.

**Sierpiński 3-bar** (figure button; arm slider = smallest 3-bar m₀, tiling
size n = depth+1, capped at depth 4): three sub-3-bars of arm m at the corners
of a 3-bar of arm 2m−1, sharing their corner cubes pairwise (B starts on A's
+x corner, C on A's +y corner, C's +x corner is B's +y corner), recursively;
total arm 2^d(m₀−1)+1. All pieces are pure translations — no (1,1,1) lifts —
so the gasket is ONE cube set; shared corner cubes are created once. Model
(sierp3_MIRROR.js part 2) d=1..4: shared cubes exactly 3 per internal node,
zero unintended coincidences, every leaf closes by column overlap
(D = (m−1)(1,1,1)); every assembled triangle cycle is impossible too — the
outline of a depth-d gasket has 2^d column-overlap joints (one per leaf on
its hypotenuse) and a hole opened at level l has 2^(l−1), one in the smallest
holes (general walker in the script, verified m₀ ∈ {3,4,5,6,8,12}, d=1..4,
every joint k = −(m₀−1), never a ±E adjacency). The closure hierarchy's
"triangle cycles are always impossible", made fractal. Browser: m₀=3 d=2 →
27 beams, 51 cells; m₀=4 d=3 → 81 beams, 231 cells; ledger green.

**Generalized fused audit**: S.fused now carries units {base, dirs, lens,
expected closure e + k(1,1,1)} and an expected shared-slot count; the ledger
re-walks every unit against the scene (complete / closure as expected,
gated on completeness / shared cell-slots measured from ownership). Chain
tiling reads 9/9, 9/9, 42/42 at L=6 n=3 (36 bar cells + 6 vertex cubes).

**UI**: hamburger (☰) hides the controls so the render window takes the
width (re-fits the figure, clears any dragged width); the render window is
CSS-resizable at its bottom-right corner, a ResizeObserver re-syncs the
viewBox so drawing and pointer math follow the new box; every section header
collapses its controls (keyboard: Tab + Enter/Space, aria-expanded); the
ledger moved to a full-width bar under the render window; rounded corners on
panels, buttons, inputs and the board. Figure buttons relabelled to the
user's canon (3-bar, 6-bar, double 6-bar). Export names now use the
generation-time arm / tiling-size / depth (like seed and family), not the
live sliders.

Review 07/09 (this diff, 10 agents: 5 confirmed defects, 2 coverage records
of clean areas, 0 refuted — all fixed and re-verified): (1) the Sierpiński
joint counts "two on the outline, one per hole at every level" were true
only at depth 1 — the correct counts are 2^d and 2^(l−1) (the hypotenuse of
a level-l triangle is 2^l disjoint leaf +z bars, each ending one column
overlap under the next); a general outline/hole walker was added to the
script, which first reproduced the same repeated-shared-cube artefact as the
chain-tiling review (sibling edges list their shared corner cube twice —
collapse consecutive repeats) and a hypotenuse run must start at the leaf's
corner cube or the jump is not a pure (1,1,1) multiple; (2) the viewBox was
only re-synced on window resize, so the hamburger and the corner handle left
letterboxed drawing and drifting wheel-zoom/pan (measured 17–48% pointer
mismatch) — ResizeObserver + fitView; (3) resize:both only worked vertically
because a flex:1 item ignores an inline width — `.canvaswrap[style*="width"]
{flex:0 1 auto}`; (4) export filenames encoded the live sliders (a depth-2
image could be named _d4) — snapshotted at generation; (5) collapsible
headers were mouse-only — tabindex/role/aria + Enter/Space. Coverage records
confirmed: the occupancy filter only ever drops bar END cells for m₀=3..12,
d=1..4 (contiguity preserved), largest figure (m₀=12, d=4: 2634 cells,
9609 pieces) regenerates in ~250 ms, every delete path is exception-free,
the fused audit's shared counts and closure names match for all presets, and
the packed one-atom claim holds for L=3..12.

## 3-bar strip / 3-bar tiling, SHIPPED 09/09/2026

The triangular tiling of alternating up (+x,+y,+z) and down (−x,−y,−z) 3-bars
in which EVERY bar is shared by one up and one down triangle — the double-hex
mechanism applied to 3-bars. The down 3-bar D shares U(P)'s +x bar as its −x
bar (base Q = P+(L−1)x); solving the cell equations for the up 3-bars that
share D's −y and −z bars gives the up-triangle lattice T1 = (−(L−1), −L, 0),
T2 = (−(L−1), −(2L−2), −L), with D(P) shared by U(P), U(P+T1), U(P+T2) —
consistent by construction; the STRIP keeps the T1 direction only (2n
triangles), the TILING is n×n up triangles + their down partners. Model
(TileFitCLAUDE/tri3_MIRROR.js, L=4,5,7; strip both directions and tiling):
shared cell-slots exactly as designed, zero unintended 3D coincidences, every
triangle closes by column overlap (up +(L−1)(1,1,1), down −(L−1)(1,1,1)); each
triangle encloses (L−4)² atoms (solid at L=4, one-atom windows at L=5,
9-atom windows at L=7); every closing pair (a 3-bar's end cube over its own
start) is a true crossing, so the flip slider acts per triangle. App: genTri3,
buttons "3-bar strip" / "3-bar tiling"; generic fused ledger (units complete /
closures as expected / shared slots measured). Browser: tiling L=5 n=3 → 33
beams, 141 cells, 18/18, 18/18, 93/93, 18 crossing pairs; L=7 → 207 cells
(= model); strip L=6 n=3 → 25 beams, 131 cells, 12/12, 12/12, 61/61, 12
crossings all flippable.

Review 09/09 (strip/tiling diff, 2 agents, 1 confirmed text-only, 0 refuted;
fixed): the "(L−4)² atoms per triangle" formula holds for L ≥ 4 only — at the
slider minimum L=3 the triangles are solid (no interior lattice vertex), so
the three text sites now say "for arm ≥ 4 (solid at 3 and 4)". Independently
confirmed clean over 80 configs (both modes, L=3..12, n=2..5): the dedupe
drops only whole bars, shared == expected everywhere, closures e='0' with
k=±(L−1), exactly one crossing per triangle and none spurious, one connected
component, n=5/L=12 = 960 cells in ~30 ms.

## Colour by line (Style option), SHIPPED 09/09/2026

For bar-sharing lattices: every beam lies on one projected lattice line
(a-direction b=const, b-direction a=const, or (−1,1)-direction a+b=const);
`computeLines()` keys beams by that line and `beamBaseColor` paints each LINE
with a palette colour, so collinear continuations across shared bars and
vertex cubes share a colour. Single-cell beams (corner-trimmed bars) join an
existing line of a family that passes through them and touches them in 3D,
else get their own. Verified: Sierpiński m₀=4 d=3 → the 8 beams of each
outer edge share one line (three lines for the three edges, 24 lines in
all); chain tiling L=6 n=3 → every column of legs is one line (25 lines),
so each 4-bar window is bounded by continuations of its four 6-bars' legs.
Palette shuffle rotates line colours too; window/custom paints override.

Review fix 09/09: a projected line is SPLIT wherever two of its beams overlap
in projected columns — a collinear crossing at different depths, which the
knot's bias slider promotes. Keyed purely by line, the front and back beam
took one colour and the same axis shading, so the crossing cue vanished.
Continuations never overlap in columns, so the split changes nothing for the
fused presets. Verified in the browser: 40 knot seeds → 41 collinear
column-overlap pairs, 0 of them sharing a line after the split (39 sub-line
splits); gasket m₀=4 d=3 still 24 lines, chain tiling L=6 n=3 still 25.

Review fix 10/09 (second pass): the greedy first-fit split was
order-dependent — with two crossing beams X1 over Y and X2 over Z, the
order Y, X2, Z, X1 put the continuations Y and Z on different sub-lines
(knots only; no preset has collinear overlaps) — and single-cell beams
joined a sub-line by 3D adjacency without the column check, which could
re-create the merged crossing. Now the beams of one projected line are
first unioned along genuine 3D continuations (a cell of one beam
3D-adjacent to a cell of another, which on one projected line forces the
same 3D line), the unions are what the column-overlap split partitions,
and a single joins a sub-line only if its column is free there.

## 6-bar geometry facts (09/09/2026, TileFitCLAUDE/hexgeom_MIRROR.js)

Lattice unit = the projected length of a cube edge along (1,1,1); a unit cube
projects to a regular hexagon of side 1 (circumradius 1, apothem √3/2).
- The CLOSED ring with L steps on every side is the Petrie (skew) hexagon of
  the cube spanned by its corner-cube centres (true edge L, i.e. L+1 cells
  per edge): it visits the six corners that are NOT on the (1,1,1) body
  diagonal (the two missed corners differ by (L,L,L)), every side runs along
  a cube edge, and its corner-centre polygon is the regular hexagon of side
  L. Its outline is the regular hexagon of side L+1 — the shadow of that
  solid (L+1)-cell cube; a solid cube with L cells per edge casts side L
  (hexgeom: all-L walk closes with D=(0,0,0), 6 of 8 corners visited, every
  cell on a cube edge; review-fixed 09/09 — the script's "all-L" walk had a
  first side of L−1 and the "[L,…] verified" figure was the solid cube's
  outline, not the ring's). The impossible 6-bar is this Petrie hexagon with
  side 1 shortened and side 2 lengthened by one step, which displaces the
  end by (−1,0,−1): the false joint.
- Its three concentric polygons are EQUIANGULAR hexagons (all corners 120°,
  every side along a lattice direction) that are regular hexagons of side
  L+1 (outer edge), L (corner-centre / shading-edge line), L−1 (inner edge)
  perturbed by the same one-unit twist (−1, +1, 0, 0, 0, +1): boundary
  tracing measures the outer edge [L, L+2, L+1, L+1, L+1, L+2] and the inner
  edge [L−2, L, L−1, L−1, L−1, L]; the corner-centre / shading-edge line
  measures [L−1, L+1, L, L, L, L] OPEN with a one-step gap, and closes to
  [L−1, L+1, L, L, L, L+1] when the false joint is read as a corner (P sits
  one −y step past E in projection). Outer − inner = 2 in side length = the
  bar width √3 in apothem.
- The 3-bar is three edges of the L-cube from one end of the body diagonal
  to the other, seen along that diagonal (end and start in one column).
- Ratios that occur exactly: √3 (bar width / lattice unit; apothem·2/side),
  √(3/2) (true 3D cube edge / projected edge), √2 (true 3D edge / projected
  apothem = bar width / 3D edge), (L+1)/L (the wobble). The silver ratio
  1+√2 does not occur as a ratio of two straight segment lengths of the
  figure, in the plane or in 3D: every such ratio is the square root of a
  rational, and (1+√2)² = 3+2√2 is not rational. (√2 itself does occur, as
  listed; it is 1+√2 that cannot. The lattice's own irrationality is √3 —
  the Eisenstein integers ℤ[ω].) hexgeom's ratio table is a spot check over
  the listed lengths, not the proof; the degree argument is.
- The unit cube IS measurable from the image: projected side s (px) gives
  bar width √3·s and true cube edge √(3/2)·s. s is the export's px-per-edge
  setting (default 60 px per lattice unit, 4–512, independent of L) unless
  the dimensions readout says "(clamped)" — an export whose side would
  exceed 16000 px is scaled by 16000/max(W,H), so then s = ppe·16000/max(W,H),
  or simply measure s from the image (one bar width / √3); the user's
  hand-edited tiling images measured 20 px per lattice unit at L=6.
- Cube roots: the solid L-cube (L³ cells) casts a regular hexagon of side L,
  so side = ∛(cell count) — a cube root by counting, exact only for perfect
  cubes; interpolating L³ ≤ N < (L+1)³ gives the integer part. No exact ∛2:
  every segment length of the figure is the square root of a rational (the
  Eisenstein norm a²+ab+b² in the plane, the ℤ³ norm in 3D — lattice POINTS
  lie in ℚ(√−3), lattice LENGTHS such as √7 do not), so any length ratio has
  degree ≤ 2 over ℚ while ∛2 has degree 3 (Wantzel / the ZRL cube-root
  block); the lattice's real cube-root content
  is number-theoretic — ℤ[ω] is where cubic reciprocity lives (for a prime
  p ≡ 1 (mod 3), 2 is a cubic residue mod p iff p = x²+27y²; for p ≡ 2
  (mod 3) cubing is a bijection and every residue is a cube), i.e. modular
  cube roots, not the real one.

## Kagome (loose vertex) + double 6-bar honeycomb, SHIPPED 09/09/2026

**Kagome verdict.** A strict kagome of 6-bars — every ring sharing corner
cubes with six neighbours around triangular holes — does not exist. In
projection an ALL-IDENTICAL kagome fails locally (a ring presents (L−1, L, L)
to one triangle class and (L, L+1, L+1) to the other, so no hole is
equal-sided); mixing the 12 chirality/rotation/mirror variants does not
rescue it — single holes can close in projection (288 of 1728 variant
triples per class), but no assignment of variants to the 37 rings of a
radius-3 patch closes every hole (kagome_MIRROR.js DFS, deepest partial 11
of 37, geometric corner pairing) — and independently every projectively
closed triangular hole carries a 3D residue k(1,1,1), k ≠ 0, for L ≥ 3
(kagome3d_MIRROR.js, all 1728 triples per class). Periodic one- and
two-variant lattices over all corner pairings and reversed-bar contacts
reach at most 4 corner contacts per ring, never 6 (kagome_relax_MIRROR.js).
(Review 10/09 corrected the earlier wording, which asserted a LOCAL 2D
obstruction for every variant mix — false: with variants (0,2,4) round a
hole the three sides are L−1, L−1, L−1 and it closes in projection; only
the global DFS and the 3D residue carry the verdict.) The shipped relative,
button "kagome (loose)"
(genKagome): identical normal rings on T1 = (L, −L, L+1), T2 = (L, −L, −L),
sharing corner cubes in two of the three directions (our Q1 is the +T1
neighbour's Q4, our Q2 the +T2 neighbour's Q5) and loose by exactly one
column in the third. Every hole is a triangle, (L−3)² and (L−2)² atoms, one
of each per ring for L ≥ 4 (at the slider minimum L=3 the (L−3)² hole is a
solid corner and only the 1-atom hole remains — cover_MIRROR.js kagome 3),
each an impossible 3-bar whose whole residue sits at the loose vertex. Shared corner cubes 2n(n−1). fusedlat_MIRROR.js L=4..8: zero
unintended coincidences, zero crossings, zero side rims, every ring k = −1;
cover_MIRROR.js: the domain's uncovered atoms are exactly one interior plus
the 9- and 16-atom holes at L=6. Browser: L=6 n=4 → 96 beams, 568 cells,
16/16 complete, 16/16 closures, 24/24 shared, 0 crossings; L=4 n=5 → 25/25,
40/40; L=8 n=2 → 4/4, 4/4.

**Double 6-bar honeycomb** (genDblHoney): alternating-chirality double
6-bars — ring A normal from P, ring B flipped from P+(L−1)x so its first
bar is A's +x bar — on T1 = (L+4, 2−L, 3), T2 = (L−1, L+1, −(2L+2)). It is
GAPLESS: the fundamental domain holds 2|det| = 4(L+1)(3L+2) atoms and the
only uncovered ones are the two ring interiors (6L²−10L+3 each), with no
open component through the patch (cover_MIRROR.js, 5×5 at L=4, 6, 8 and 7×7
at L=6). The sharing is SPARSE, not total: per double unit the intra L−1 bar
(A0 = B0) and ONE inter-unit bar — A's +y bar A2 (L cells) is the +T2
neighbour's B −y bar B2, with B's corner at a0+(L+1)y, i.e. each ring's
corner cube sits one step past the other's bar end. Shared cell-slots in an
n×n patch: n²·L + L·n(n−1) = L·n(2n−1) (fusedlat_MIRROR.js walks the same
two full rings per unit and counts the same number as intended whole-bar
coincidences; before the 09/09 review fix it walked B from A's base, which
also printed B's closure as (L,0,1) — no k at all). Every other bar abuts a
neighbour's bar outer-edge to outer-edge with the cubes two lattice steps
apart (face-diagonal offset, same 3-colouring class, projected hexagons
sharing an edge): mid-bar √3 pairs 116 / 268 / 420 at L = 4 / 6 / 8 (5×5),
distance-1 mid-bar pairs 0 — which is why fusedlat's rim check saw nothing.
Closures A k = −1, B k = +1, zero crossings. Browser: L=6 n=3 → 93 beams,
576 cells, 18/18, 18/18, 90/90, 0 crossings; L=4 n=4 → 164 beams, 688
cells, 32/32, 112/112; L=5 n=2 → 30/30; L=8 n=2 → 48/48. Colour by line
verified on both presets.

Correction recorded: the 09/09 workflow's summary said "every bar shared,
L+1 bars pair with L+1 bars, false joints cancel in pairs". Classifying the
coincidences per bar (scratch dblshare.js) showed only A0 = B0 and
A2 = B2′, both L−1 / L bars, so the app's note paragraph, generator comment
and ledger note were rewritten, and `expShared`, which had been the
generator's own dedupe count (a ledger line that could never go red), is
now the closed form above. Two lessons: "no enclosed void" is not
"gapless" (an open channel network never encloses — the loose hex tiling),
so gaplessness is tested as atom coverage of the fundamental domain; and a
rim check at lattice distance 1 misses √3 abutments.

**Views of the tumbled double 6-bar** (user's card_spin3d screenshot,
θA = 172°, θB = 263°, θC = 360°; TileFitCLAUDE/dblviews_*, dbltile_*,
dblfuse_*, dblgallery_MIRROR.js): the pose maps to the view direction
v = (M·pre)ᵀ(0,0,1) = (0.29, −0.72, 0.63), 17° from the face-diagonal
line ±(0,1,−1) (the ray (0,−1,1); 163° from (0,1,−1) itself) and 18.5°
from the body diagonal (1,−1,1) — a generic in-between view. The "two open rings joined by the bar" is the isomer view along
(1,1,−1) / (−1,1,1), where each 6-bar reads as a bow-tie. That bow-tie pair
does not pack alone or with one companion (≥ 6 sliver atoms per copy) but
tiles by bar-sharing (hourglass chain T1 = (L, 1, −L) in the (−1,1,1)
view); the 13-view catalogue (4 body diagonals, 3 axes, 6 face diagonals)
is `TileFitCLAUDE/dblviews_gallery_MIRROR.html` (L=5, verified render).

Review 09/09 (kagome / honeycomb / colour-by-line / geometry-notes diff; 6
lenses → 3 verifiers per finding; 33 agents, of which 20 died on the session
limit — the whole kagome lens and most geometry / colour-by-line verifiers;
run `wf_ad15e3cd-a4a`, to be resumed). Honeycomb and integration lenses
clean: 80 slider configs (L=3..12, n=2..5) with closures and shared counts as
designed, a wrong-expShared simulation turns the ledger red, cube coercion
consistent with the other fused presets, 3110 cells at max sliders in
172 ms, the B ring's '−E1' matches ringClosure's U+2212. Confirmed and
fixed: (1) fusedlat walked ring B from A's base, so it printed B's closure
as (L,0,1) — no k at all — while the app comment cited it for "B k=+1"; B is
now walked in full from P+(L−1)x and closures print as e+k(1,1,1) (A +E1
k=−1, B −E1 k=+1), intended coincidences L·n(2n−1) like the app. (2) The
"measured [L−1, L+1, L, L, L, L+1]" corner-centre polygon — the script
measures the open [.., L] with a one-step gap; text now says so. (3) The
cubic-residue "iff" lacked p ≡ 1 (mod 3). Unverified by the workflow,
verified here and fixed: (4) hexgeom's "all-L" walk had a first side of
L−1, so its Petrie test failed and the README's "verified [L,…]" was the
solid cube's outline — walk fixed (closes, 6 of 8 corners, all cells on
edges, outline L+1) and the bullet rewritten; (5) "exports use s = 20 px at
L=6" was false (export scale = px-per-edge, default 60; 20 px was the
user's images) — README, context doc and memory corrected; (6) the
silver-ratio bullet contradicted itself and the ratio table carried a
dimensionless entry that made L-specific spurious hits — reworded, entry
dropped; (7) hexgeom's cube-outline caption said side L+1 beside its own
measurement of L; (8) colour-by-line merged collinear crossings in the knot
— split rule added (see "Colour by line"). Refuted (1 of 3 votes): the
view-angle "wrong sign" — the docs name unsigned lines; wording clarified
anyway (±(0,1,−1), 18.5°). Also found before the review: the kagome hole
formula needs L ≥ 4 (solid corner at L=3), now qualified everywhere.

Second pass 10/09 (resumed from cache: kagome lens live, a seventh lens
over the fix diff, three verifiers per finding judging both the original
defect and the applied fix; 31 agents, 0 failures). All eight earlier fixes
judged correct. Seven new findings confirmed (3 of 3 votes unless noted)
and fixed: (1) the kagome text asserted a LOCAL 2D obstruction ("no side of
a wobbled ring can bound an equal-sided triangle") that only holds for the
all-identical kagome; with variants (0,2,4) round a hole the sides are
L−1, L−1, L−1 and it closes in projection — the scripts establish the
GLOBAL obstruction (radius-3 DFS: 0 assignments, deepest partial 11 of
37) plus the 3D residue; reworded in the app note, generator comment,
README, context doc and memory. (2)+(3) the greedy colour split was
order-dependent and its singles rule skipped the column check — union of
continuations first, singles check added (above). (4) export scale is
px-per-edge only when the 16000-px clamp is not hit — qualified. (5) the
∛2 bullet's reason was a non sequitur (lengths are not in ℚ(√−3); the
valid argument is length² ∈ ℚ) — reworded (2 of 3 votes). (6)
cover_MIRROR.js sampled its half-open domain with float thirds, so it
printed 557/315 where the README quotes 2|det| = 560 and 2×159 = 318 —
the test is now exact integer arithmetic and the script prints the quoted
values. Refuted 3–0: "all corner pairings" conflating two searches — the
citation was nevertheless split per script in the reworded text.

## Isomer views (view-axis toggle) + depth scaling, SHIPPED 10/09/2026

**View axis.** Every figure is a set of unit cubes in ℤ³ rendered along
(1,1,1). The "view axis" row (Figure section) renders the same cube set
along another body diagonal d ∈ {(1,1,−1), (1,−1,1), (−1,1,1)}: a lattice
rotation R (signed permutation, det +1 — 90° about x for the first two,
about z for the third) with R·d = (1,1,1) is applied to every cell inside
cellsMap(), so the whole atom pipeline (faces, hidden surfaces, crossings,
coins) runs unchanged on R·S, which is the true view along d up to an
in-plane rotation of the image. Colour-by-line keys and column overlaps
follow the view frame; the identity audits (preset closures, fused units)
are measured against the canonical occupancy (cellsMap(true)), so the ledger
reports the (1,1,1)-frame closure whatever is rendered, and the
disjoint-ring overlap audit, a (1,1,1)-frame identity, is shown as
informational off-axis. Cube cells only: an off-axis click on a
rhombohedron/mixed scene coerces to cubes and regenerates (a rotated
rhombohedron is not the (1,1,1)-aligned kind the atom machinery knows).
Export names get `_v11m` / `_v1m1` / `_vm11`. Browser (L=6): 6-bar in all
four views → ledger ✓, canonical FALSE joint reported in every view, 0/1/3/1
crossing pairs; double 6-bar honeycomb, chain tiling and kagome (3×3) in
(−1,1,1) → fused audits identical to on-axis (18/18 90/90; 9/9 42/42; 9/9
12/12) with 18/19/21 view-frame crossings; 3-bar tiling 18 crossings
on-axis, 0 in (−1,1,1); rhombohedron knot + off-axis click → cube knot ✓;
the honeycomb along (−1,1,1) reads as rows of bow-tie pairs, each double
6-bar an hourglass — a different lattice from the 09/09 catalogue's
hourglass chain, whose period (L,1,−L) shares two bars and is not a
honeycomb lattice vector even modulo the view axis (review 11/09).

**Depth scaling** (the user's question 10/09; TileFitCLAUDE/depthscale_MIRROR.js).
The figures already ARE the 3D sculptures: cube sets whose (1,1,1)
projection is the impossible picture. Measured, L=6 (depth = extent along
the view axis in cube steps; a false joint's hidden offset is k·(1,1,1) =
3k in depth):

| figure | cells | impossible joints | depth (1,1,1) | depth along the other diagonals |
|---|---|---|---|---|
| closed L-ring (possible) | 36 | 0 | 6 | 18 / 18 / 18 |
| impossible 6-bar | 37 | 1 | 7 | 18 / 19 / 18 |
| 3-bar | 16 | 1 (column overlap, k=5) | 15 | 10 / 5 / 10 |
| double 6-bar | 68 | 2 | 9 | 31 / 33 / 19 |
| chain tiling n×n | — | n² | 8n−1 (general L: (L+2)n−1) | 18n / 20n−1 / 18n |
| kagome n×n | — | n² | 13n−6 ((2L+1)n−L) | 13n+5 / 25n−6 / 23n−5 |
| double 6-bar honeycomb n×n | — | 2n² | 11n−2 (11n+L−8) | 29n+2 / 33n / 23n−4 |
| 3-bar tiling n×n | — | 2n² (each k=5) | 32n−7 ((6L−4)n−L−1) | 20n−5 / 2n+3 / 12n+3 |

(All closed forms are exact for every n ≥ 1: depth is linear in the patch
indices, so its extremes sit at patch corners — slope = |T1·d| + |T2·d|,
intercept = single-unit extent minus slope; review-verified L=3..12,
n=1..8.)

**Sheet geometry (12/09, the user's follow-up: does the 3D lift get more
complex in z as the tiling grows?)** No. A fused tiling's lift is a
translation lattice in ℤ³ — every unit sits at P + i·T1 + j·T2 — so the
whole patch lies in a SLAB parallel to the plane spanned by T1 and T2,
whose thickness (extent along the normal N = T1×T2) is that of one unit
and does not change with n, while its depth along the view axis grows only
because the sheet is inclined to the picture plane by a fixed angle θ
(between N and (1,1,1)). Measured (depthscale_MIRROR.js, L=6; thickness
identical for n = 1, 2, 3, 4, 6):

| tiling | N = T1×T2 | tilt θ | slab thickness ⊥ sheet |
|---|---|---|---|
| chain tiling | (78,48,30) | 20.8° | 5.98 cubes |
| kagome (loose vertex) | (78,78,0) | 35.3° | 8.49 |
| double 6-bar honeycomb | (35,155,90) | 27.7° | 7.86 |
| 3-bar tiling | (36,−30,20) | 72.9° | 3.53 |

So a larger tiling is the same thin sheet, just wider: the only thing that
scales is the tilt times the diameter. The 3-bar tiling is the extreme
case — a sheet 3.5 cubes thick standing almost edge-on (73°) to the viewer,
which is why its depth per n is the largest (32n−7) although it is the
thinnest object. The tilt is intrinsic: bar-sharing fixes the lift up to a
global translation, so no choice of construction flattens the sheet;
viewing along N would show the sheet face-on as an ordinary mosaic of open
rings with every false joint visible as a gap — and N is not a lattice
direction, so that view is not one of the app's isomer views. (L=4 for
comparison: 22.2° / 4.28, 35.3° / 5.66, 36.3° / 5.93, 73.9° / 2.30.)

Review 11/09 (isomer view + depth-scaling diff; 5 lenses, 26 agents, 0
failures, 7 confirmed, 0 refuted; all fixed). Clean: the three rotations
(orthogonal, det +1, R·d = (1,1,1); the off-axis image matches an
independent orthonormal-basis projection along d under a proper rotation
and under no reflection, on chiral sets); ledger green in every view for
every preset, family and 40 knot seeds, closures and fused audits identical
across views, no 'r' cell survives off-axis; on-axis behaviour
byte-identical to the pre-isomer build over 384 configurations (beams,
cells, atoms, crossings, flips, lines, ledger, file names). Confirmed and
fixed: (1) the four disjoint-ring packings (hex/tight/abut/packed) lifted
each ring by (q+r·n)(6L+12) along (1,1,1) — invisible on-axis, but off-axis
the band strung the rings ~800 lattice units along a diagonal and the
ledger's informational off-axis overlap branch could never fire; the band
is removed (rings never share a column, so the on-axis render is
byte-identical — verified by the reviewers for L=3..9, n=2..4; hextight
rings now touch edge/corner-wise in 3D, never by a face), and off-axis
those packings are compact with a genuinely view-dependent overlap count.
(2) An off-axis click on a rhombohedron/mixed scene regenerated from the
live inputs, silently applying unsubmitted seed/slider edits — regenerate
now snapshots its parameters (S.gen) and the view handler regenerates from
them. (3) "kmax = L−1 is a floor no sculpture can beat" contradicted the
next sentence and ignored the bas-relief ambiguity — restated as a
lattice-model floor (above). (4)–(6) the table's isomer-depth column had
two wrong slopes and rows presented as exact that were not — replaced by
the exact forms. (7) "the honeycomb along (−1,1,1) is the catalogue's
hourglass chain" — false: that chain's period (L,1,−L) is not a honeycomb
lattice vector even modulo the view axis; the honeycomb reads as rows of
hourglasses. Side observation (not a defect): the rotated cube is placed at
base R·p rather than at its true min-corner, a uniform translation of the
whole set.

So no: the depth does not scale with the number of impossible joints. A
patch's depth grows with its DIAMETER (|T1 depth| + |T2 depth| per n) while
its joints grow with its AREA (n², 2n²); the ratio goes to zero. A single
impossible 6-bar is one step deeper than the possible closed ring (L+1
against L — its long bar), not 3 steps: the false joint is a hidden 3k jump
between two cells that are already there, it buys no depth. The one figure
whose depth IS its joint is the 3-bar: 3(L−1), exactly the column-overlap
gap, and kmax = L−1 (the largest depth separation inside one projected
column) is a floor for any UNIT-CUBE LATTICE model of that picture (every
3-bar step faces the viewer and the picture pins the front/back order at
each seam). For arbitrary sculptures no picture has a depth floor at all:
compressing every point along the view axis leaves the projection and
every front/back order unchanged, so the drawn image is identical while
the depth shrinks without limit — the bas-relief ambiguity; the classic
Penrose-triangle sculpture is not exempt, its bars just become skewed
prisms. The tilings with false joints (chain, kagome, honeycomb) have no
hidden column overlaps at all (kmax = 0): their depth comes from bar
lengths and the lattice pitch, not from the joints; the 3-bar tiling keeps
kmax = 5 at every size. Which views are impossible: the 6-bar's end offset D=(−1,0,−1)
decomposes as e + k·d in TWO of the four body diagonals — (1,1,1): e=+y,
k=−1, and (1,−1,1): e=−y, k=−1 — and is two lattice steps open in the
other two. The same sculpture is therefore impossible from two of the four
lattice directions (a hexagon from one, a self-crossing bow-tie whose ends
meet at a false joint from the other; screenshot-verified in the app) and
visibly open from the remaining two.

## Backlog / ideas

- Shifted/woven variant (user's (b)): chain legs overlapping crease-to-edge and
  stack neighbours sharing exactly one cell — needs same-depth shared cells (a
  true joint) or explicit crossings; periodicity + depth-consistency analysis.
- Other 0-gap bar-sharing lattices to model (candidates, unverified): the
  hexaflake of 6-bars sharing corner cubes, the 2×2 subdivision of wobbled
  4-bars, 3-bars sharing corner cubes on a triangular lattice (hexagonal holes
  — possible or impossible cycles?). Done 09/09: 3-bar strip / tiling, kagome
  (strict one impossible; loose-vertex relative shipped), double 6-bar
  honeycomb (gapless, sparse sharing).
- (Shipped 10/09) Isomer-view toggle — see above. Possible follow-ups: the
  three axis views and six face-diagonal views (square / rectangular
  lattices, outside the triangular atom machinery); rhombohedron cells
  off-axis (needs the rotated cell kinds).

- Multi-scale cells (user's "Option 4" insight 31/08): rhombic-dodecahedron cells
  project on-lattice at exactly 2× scale (node-verified) — more generally, mixed
  1×/2× cell scales would support masks for image stacks where one layer is
  deliberately mapped at a larger lattice size (user already does this manually in
  GIMP by resizing one mask layer).
- Textured prism blocks (image layers on tops/walls).
- Prism paradox mode (flip occlusion between adjacent towers) — deferred; reads as
  "broken" rather than "impossible" for solid terrain, revisit after beams ship.
