# Zero-Recursive Lattice — Cell-Lattice Wave App, Spec (draft: the decisions to settle)

**For:** the author first (tick the decisions in §2), then a coding agent (Claude Code).
**Extends:** the membrane solver — TaichiWave (`TaichiWave/membrane_core_MIRROR.py`, the primary app since
2026-09-29) or the frozen web app `wave_membrane_MIRROR.html` (decision D0.1).
**Companions:** `ZRL_number_domains_MIRRORv5.2.md` §13.2, §16.6, §17.2 (cells, neighbours, rings, parity);
`circle_fidelity_findings_MIRROR.md` (circle edges on grids); `R4csR2Metrics.md` (density of overlapping
circles); TaichiWave `NotesCLAUDE/walls_MIRROR.md` (pinned and rigid walls, the energy) and
`NotesCLAUDE/absorbing_MIRROR.md` (the sponge); this folder's README entries 2026-09-26 (edge sheets,
straight edges, blended grids) and 2026-09-26c (light cone, dispersion, PPW).
**Status:** DRAFT. Nothing is built. §2 lists every decision as a checkbox with a proposed default. §3 holds
the facts the decisions rest on; each number was executed before it was written down (scripts and logs in
`LatticeSpecCLAUDE/`, run log in §9). The build stages in §6 are blocked until their decisions are ticked.
**Written:** 2026-10-11 by Claude Code (cloud session), from the author's request of the same day.

---

## 0. Instructions for the receiving agent

Verify numerically before affirming; flag ambiguity rather than guessing; say "this has no established
name" when true. Five additions specific to this app:

1. **Do not build a stage whose decisions are open** (§6). Ask, with the decision's id.
2. **Words (§1.1).** A solver grid point is a **node**, never a "cell", in this app's code, UI and notes.
   A **cell** is a ZRL flavor shape.
3. **Every medium or border parameter is physical**: per unit length or area, and for a border also at a
   stated PPW. Never a raw per-node value: the same node value acts differently on another grid or line
   direction (§3.6).
4. **Lengths are tested at a generic R** (e.g. R = 1.7, or R = 37 nodes), never only at R = 1: only
   ratios are safe at R = 1 (v5.2 §16.6).
5. **Every measurement states its regime** λ / (centre spacing): the same tiling is an averaged medium, a
   crystal or a set of separate lenses depending on it (§3.8).

---

## 1. What the program is

The existing leapfrog membrane with a **ZRL cell tiling laid over its grid**. The tiling acts on waves in
two ways:

- **(a) a density map.** At each point, the number of tiling elements that cover it (D3), turned into a
  local wave speed (D4). With circumcircles as the elements, every pair of edge-neighbour cells shares a
  lens-shaped overlap (two circular arcs) straddling their shared edge. The picture is cleanest for squares
  and hexagons; the triangle's overlaps merge (§3.3). With a slower speed there, that
  overlap is a biconvex slow region: a converging lens in the optical sense (STANDARD optics; geometry in
  §3.3). This is the "shared edges act as lenses" idea of 2026-09-26.
- **(b) translucent borders.** One-node lines along the cell edges and/or the circles, which reflect part of
  every wave and pass the rest (D5). The two existing wall types are their opaque limits (§3.6).

Sources are the solver's own. Outside the cluster: background medium, then an absorbing sponge.
**Purpose (author, 2026-09-26 … 10-11):** see how the cell structure acts as a propagation medium, and
measure the space-time grid (Co, N, PPW, light cone) region by region.
**Not:** a new solver (the solver already has a per-node speed map, §3.1), and not a toggle inside the
existing app (author: a separate app, built on a copy).

### 1.1 Vocabulary (proposed; decision D0.4)

| word | meaning |
|---|---|
| node | a solver grid point, spacing h = 1. (Both apps currently call these "cells".) |
| grid | the solver lattice: sq5, sq9 (square), tri (triangular) |
| cell | one ZRL flavor shape (triangle n = 3, square n = 4, hexagon n = 6), circumradius R in nodes |
| tiling, cluster | the edge-to-edge arrangement of cells; the finite part drawn (k rings) |
| circle, incircle | a cell's circumcircle (radius R); its incircle (radius r = R cos(π/n)) |
| density m | how many tiling elements cover a point (D3) |
| node mass M, speed factor f | the solver's per-node inertia; f = local speed / background speed = 1/√M |
| border, strength s | a one-node translucent line (D5); s = 0 is invisible, s → ∞ is a wall |
| TR | tick rate (author, 2026-09-26) |

---

## 2. Decisions to settle

Format: **id**, the question, the options, **default** (proposed), why (§ of the fact), what depends on it.

### D0 — Scope

- [ ] **D0.1 Where to build.**
  - (a) TaichiWave: a new app on `membrane_core_MIRROR.py`. The README already plans "prism-like /
    programmable media; then the lattice extension (separate app)" as its last two items.
  - (b) a copy of the frozen `wave_membrane_MIRROR.html`.

  **Default (a).** It already has the per-node speed map, link removal (rigid walls) and corner blocking,
  the sponge on both grids, GPU stepping to N 8193, and the cross-check harness. The HTML is frozen and has
  none of the walls or the sponge.
  **Consequence:** the density→speed map and per-link weights go into the core, as its "programmable
  media" step (§6 stage A). The tiling, borders and panels go into the new app.
- [ ] **D0.2 What "2 Lattice" means.**
  - (a) one ZRL tiling over the solver grid: two lattices in all;
  - (b) two overlaid ZRL tilings over the grid (a flavor plus its dual, or both isomers);
  - (c) "2D lattice".

  **Default (a).** (b) is a later stage: it gives more density levels (§3.4) and doubles D1–D5.
- [ ] **D0.3 The grid-resolution symbol.**
  - N: `ZRL_number_domains_MIRRORv5.2.md` (2026-10-01) says N supersedes ρ_g.
  - ρ: the author, 2026-09-26: "ρ is grid resolution here" (README 2026-09-26c).

  **Default:** N in text, `N` in code (both apps use it). It only affects labels, but the canon should
  hold one.
- [ ] **D0.4 Names** for density, node mass, speed factor and border strength (m, M, f, s are placeholders
  here; ρ is taken). **Default:** long names in code per the author's rule (`density`, `node_mass`,
  `speed_factor`, `border_strength`). No new single letters until the canon picks them.

### D1 — Grid and tiling (author's area 1: "the rules for each lattice")

- [ ] **D1.1 Allowed pairings.** Exact pairings: square cells on sq5 / sq9, and triangle and hexagon cells
  on tri (THEOREM, §3.2). Any other pairing makes every edge a staircase. **Default:** exact pairings only;
  others greyed out with the reason.
- [ ] **D1.2 Snapping R** so that every tiling vertex is a node (§3.2 table). **Default:** snap; show R
  and the snap residual.
- [ ] **D1.3 Isomer.** ZRL isomer (a vertex on the x-axis, R⃗ along x) or grid isomer. Both are exact on
  the matched grid; they need different R snaps (§3.2). **Default:** the ZRL isomer, the canon's
  convention.
- [ ] **D1.4 Origin.** A cell centre on a node, or a vertex on a node. **Default:** centre on a node. The
  circle-fidelity work found neither centring systematically better for one circle (findings §4).
- [ ] **D1.5 Cluster.**
  - (a) k rings around a centre cell. Ring k holds n·k cells, so 1 + n·k(k+1)/2 cells within k rings
    (v5.2 §17.2).
  - (b) a periodic box holding whole tiling periods (the square torus, or the tri rhombic torus).

  **Default:** (a), with k on a slider; (b) as a measurement mode. The long-wave speed row (§5) needs (b).
- [ ] **D1.6 Outside the cluster.** Background medium, then the sponge (`edge = 'open'` works on both
  grids, TaichiWave 2026-10-10). **Default:** yes. Note that the cluster's edge is itself a density step,
  so it reflects (§3.7).
- [ ] **D1.7 Pentagon and circle flavors.** They do not tile: no lattice, no up step (v5.2 §17.2).
  **Default:** excluded from this app. A packing rule could be a later decision.
- [ ] **D1.8 Circle edges.**
  - (a) a staircase, as the solver's mask does today;
  - (b) a fitted edge, as on tri + fitted edge triangles (600–1400× better, circle-fidelity findings §5).

  **Default:** (a) in v1, (b) as stage C. On the square grid a circle keeps the grid's own cos 4θ
  anisotropy whatever the edge (findings §3).

### D2 — Cells and circles (author's area 2)

- [ ] **D2.1 Which circles exist.** Circumcircle only, or circumcircle + incircle. **Default:**
  circumcircle only. Incircles never overlap each other (§3.3), so they add a level without adding any
  overlap.
- [ ] **D2.2 A shared edge is one border** with one strength (default), not two half-borders: two one-node
  lines cannot both lie on one line.
- [ ] **D2.3 What an overlap is.**
  - just a density level (default);
  - a separate element type with its own properties.
- [ ] **D2.4 Cell colouring.**
  - All cells are equal (default).
  - Alternating: the triangle (up/down) and square (checkerboard) cell graphs are bipartite; the
    hexagon's needs three colours (v5.2 §17.2, parity).

  This only matters if cells get different densities or strengths.
- [ ] **D2.5 Vertices.** Each vertex lies exactly on q circles (q = 6 / 4 / 3) and is a node after D1.2.
  It is where borders cross (D5.6) and where point-sampled density ties (D3.3). **Default:** handled by
  D3.3 and D5.6; no separate vertex element.

### D3 — Density (author's area 3)

- [ ] **D3.1 What counts.**
  - (a) circumcircles: exactly two levels per flavor (§3.4);
  - (b) + incircles: three levels;
  - (c) + the tiles themselves: a constant +1, which carries no information;
  - (d) + a second tiling (D0.2 b).

  **Default (a).**
- [ ] **D3.2 Weights.** A plain count, or a weight per element type. **Default:** plain count.
- [ ] **D3.3 How a node reads the density.**
  - point sample at the node, with ties resolved open (d < R) or closed (d ≤ R);
  - the covered fraction of the area around the node (its square or hexagon), supersampled.

  **Default: area fraction.** Every tiling vertex is a node lying exactly on q circles, so the point
  rule's tie convention shifts the whole mean. Measured (§3.5), hexagon at R = 4: open 1.146, closed
  1.271, area 1.210 (exact 1.209). The long-wave speed moves by up to 5%.
- [ ] **D3.4 Density outside the cluster.** **Default:** 0, with the medium defined from an empty
  background (D4.2).
- [ ] **D3.5 Static or time-dependent.** **Default:** static. A moving or growing density is a moving
  boundary. It makes time reflection (TaichiWave README, *Time reflection*) and belongs with the planned
  "moving objects" item.

### D4 — How density changes the medium (author's area 4)

- [ ] **D4.1 What density changes.**
  - (a) **mass only:** the solver's existing form (§3.1). Interfaces refract and reflect, head-on
    r = (f − 1)/(f + 1) in the continuum, more on the grid at short wavelengths (§3.7).
  - (b) **stiffness only** (link weights): a different interface law and a different long-wave average.
    Not measured here.
  - (c) **both, impedance-matched:** node mass ×1/f and links ×f. The same speed change, but head-on
    reflection on the grid is 8–220× lower than (a), not zero (§3.7). This is a "clean lens".

  **Default (a); (c) as an option.**
- [ ] **D4.2 The density→mass law.**
  - (a) M = 1 + α·m: each covering element adds α node masses to an empty background, so f = 1/√(1 + α·m);
  - (b) M = m inside the cluster (the literal "density = mass"; undefined outside, where m = 0);
  - (c) a table per level.

  **Default (a), α = 1 on a slider.** The long-wave speed of a periodic tiling is then
  c_eff / c₀ = 1/√(mean M) (§3.5, measured with M = m).
- [ ] **D4.3 Time step.** With M ≥ 1 everywhere (f ≤ 1), the existing Co limit holds unchanged (THEOREM,
  §3.1). **Default:** forbid M < 1. If faster regions are ever wanted, Co ≤ C_max / f_max.
- [ ] **D4.4 Which region sets PPW.** In a slow region the local PPW is f·PPW and the local Co is f·Co
  (further from the limit, so more dispersion; README 2026-09-26c). **Default:** the PPW readout and the
  drive warning use the slowest region.
- [ ] **D4.5 Sharp or smoothed steps.** **Default:** sharp. The area rule (D3.3) already anti-aliases by
  one node.

### D5 — Translucent borders (author's area 5)

- [ ] **D5.1 Which perimeters.** Cell edges, circles, or both. **Default:** both, switched separately,
  with separate strengths.
- [ ] **D5.2 Which family** (§3.6).
  - **mass:** the opaque limit is pinned, inverted echo; reflects more at grazing incidence;
  - **link:** the opaque limit is rigid, upright echo; reflects less at grazing;
  - **both:** two knobs, setting magnitude and phase separately.

  **Default: mass for v1.** It is one CC value per node, and pinned is the apps' default wall. Link
  borders come in stage C (they need per-link weights).
- [ ] **D5.3 Units of strength.** s per unit length, entered as transmitted power at a reference PPW. For
  example "50% at PPW 16" is s = 5.03, a one-node mass line at f = 0.407 on sq5. **Default:** enter |t|²
  at PPW 16 and show s and the per-node value.
- [ ] **D5.4 Per-node value from s.** s_node = s × (node spacing along the line) / (area per node) — this
  is DERIVED. Measured on the tri grid: a row whose node value would be s on sq5 acts as s·√3/2 (§3.6).
  Along a staircase (a circle, an off-axis edge) the spacing varies. **Default:** normalise node by node.
  Accepting an angle-dependent strength is the alternative.
- [ ] **D5.5 Drawing the line.**
  - Which nodes form it: on sq5 an 8-connected line cannot be bypassed; on sq9 and tri, diagonal and
    third-direction links jump an 8-connected line.
  - For link borders, which links "cross" the line.

  This is the translucent version of the walls' corner blocking (walls_MIRROR.md). **Default:** reuse the
  walls' rasterisation and blocking rules, with weights in place of removal. **Measure the leak** (§8).
- [ ] **D5.6 Crossings.** Arcs cross arcs and edges; at a vertex q edges and q circles meet.
  - add (exact for masses; each crossing becomes a heavier point scatterer) — default;
  - cap at the largest of the crossing strengths;
  - zero at vertices.

  The vertex scattering must be measured (§8).
- [ ] **D5.7 A border on a density step.** The border node gets its area-fraction density (D3.3) plus the
  border mass on top. **Default:** yes.
- [ ] **D5.8 Negative strength** (a lighter line, −1 < s < 0). **Default:** not allowed: it can break the
  time-step limit (D4.3).
- [ ] **D5.9 The intent behind "lens-like".** A one-node border alone does not refract: the medium is the
  same on both sides (§3.6). Refraction comes from density (D4). **Default reading:** borders are partial
  mirrors, and density makes the lenses. Confirm.

### D6 — Scale regime (missing from the original list)

- [ ] **D6.1 Target regime** (§3.8).
  - λ ≫ centre spacing: one averaged medium;
  - λ ~ centre spacing: Bragg reflection, possibly band gaps — a crystal;
  - λ ≪ R: ray optics, each overlap lensing on its own.

  **Default:** all reachable through the drive wavelength; the app shows λ / spacing and names the regime.
- [ ] **D6.2 R in nodes.** **Default:** 40 (minimum 30), a compromise. The circle-fidelity study's
  error falls steeply between R ≈ 30 and 120 (findings §8). At 40, a one-node border is thin compared with
  the cell.
- [ ] **D6.3 PPW floor in the slowest region.** **Default:** warn below 8 (TaichiWave's drive warning),
  recommend 16.
- [ ] **D6.4 N** follows from D6.1–D6.3 and the cluster size (§3.8 example).

### D7 — Sources, boundary, readouts, visuals (missing from the original list)

- [ ] **D7.1 Sources.** A pulse at a cell centre, a beam or packet from outside aimed at the cluster, or a
  plane wave across the whole cluster (for transmission spectra). **Default:** packet and pulse.
- [ ] **D7.2 Outer boundary.** **Default:** the sponge (both grids). CPML exists only on the full square
  and must be re-checked once the medium is not uniform (TaichiWave absorbing_MIRROR.md, Bécache et al.
  2003).
- [ ] **D7.3 Readouts.** λ / spacing and the regime; the minimum local PPW; c_eff predicted vs measured;
  each border's |t|² at the drive frequency.
- [ ] **D7.4 Visuals.** Density as a floor tint; borders drawn by family (reuse the pinned and rigid
  colours); cell outlines and circles toggled separately.

---

## 3. Verified reference

Tags: THEOREM (proved), MEASURED (executed, §9), DERIVED (algebra, checked where it says so), STANDARD
(textbook), CODE (read from the solver).

### 3.1 The solver's medium (CODE + THEOREM)

- Step (TaichiWave `membrane_core_MIRROR.py`, the web app likewise): `u⁺ = 2u − u⁻ + CC·L(u) − d(u − u⁻)`,
  with CC = (Co·f)² per node. The slab and lens media are f = 0.5 and f = 0.55, sharp steps.
- Read as masses, with M = Co²/CC = 1/f² for each node: M (u⁺ − 2u + u⁻) = Co² L u. Density only has to
  produce M (or CC), so no new solver is needed.
- **Energy:** E^{n+½} = Σ M (u⁺ − u)² − Co²⟨u⁺, L u⟩ is conserved when L is symmetric (walls_MIRROR.md §6,
  measured to ~1e-15 with slab, lens and walls). Per-link weights keep L symmetric, so they keep this.
- **Time step (THEOREM, Rayleigh quotient):** L = −Σ w_e b_e b_eᵀ with positive weights on all three
  stencils. Weakening a link (w smaller) or adding mass (M ≥ 1) can only lower the largest eigenvalue of
  M⁻¹(−L). Neither can push the stable Co limit down.

### 3.2 Pairings and snaps (THEOREM)

- The triangle and hexagon tilings never fit ℚ² (v5.2 §16.6): no vertex-exact placement on the square grid.
- Squares never fit the triangular lattice: two perpendicular lattice vectors of equal length would need
  √3 rational.
- Hence the exact pairings: square cells ↔ sq5 / sq9; triangle and hexagon cells ↔ tri (crystallographic
  restriction; README 2026-09-26, blended grids).

| cells on grid | isomer | every vertex on a node when |
|---|---|---|
| square on sq | grid (edge on the x-axis) | side R√2 is a whole number (centres are nodes when it is even) |
| square on sq | ZRL (vertex on the x-axis) | R is a whole number; centres form the (R, R), (R, −R) checkerboard |
| hexagon on tri | ZRL (vertex on the x-axis) | R (= side) is a whole number |
| hexagon on tri | the other isomer | R/√3 is a whole number |
| triangle on tri | sides along the grid rows | side √3·R is a whole number (centres are nodes when it is a multiple of 3) |
| triangle on tri | ZRL (vertex on the x-axis) | R is a whole number (the sides run at 30°, 90°, 150°, side √3·R); centres on nodes |

- Every row checked (MEASURED, `snap_rules_MIRROR.py`): all vertices within two periods are nodes, and so
  are the centres, except where a row says otherwise.
- Circles fit no grid. On tri, the fitted edge removes the staircase noise (circle-fidelity findings §5).

### 3.3 Circles of neighbouring cells (MEASURED, `density_levels_MIRROR.py`; R = 1 and 1.7 agree)

| flavor | centre distances (d/R) × cells | circles | edge-neighbour overlap width 2R(1 − cos π/n) |
|---|---|---|---|
| triangle | 1 ×3, √3 ×6, then 2 ×3 | overlap 9 neighbours, touch 3 | 1.000 R |
| square | √2 ×4, then 2 ×4 | overlap 4, touch the 4 diagonal ones at the corner | 0.586 R |
| hexagon | √3 ×6, then 3 ×6 | overlap 6, nothing further | 0.268 R |

- Each vertex lies on exactly q = 6 / 4 / 3 circles and strictly inside none.
- The overlap width matches v5.2 §13.2 ("twice the drum echo window").
- The triangle's circles overlap more than the edge neighbours, which is why its density never drops
  below 2 (§3.4).

### 3.4 Density levels (MEASURED; Monte Carlo, 400,000 points, against the exact mean)

| flavor | circumcircle levels | area shares (exact) | mean = πR² / cell area | + incircles: levels and shares (MC) | incircles cover |
|---|---|---|---|---|---|
| triangle | 2, 3 | 0.5816, 0.4184 | 4π/(3√3) = 2.4184 | 2 / 3 / 4: 0.182, 0.612, 0.205 | π/(3√3) = 0.6046 |
| square | 1, 2 | 0.4292, 0.5708 | π/2 = 1.5708 | 1 / 2 / 3: 0.017, 0.610, 0.373 | π/4 = 0.7854 |
| hexagon | 1, 2 | 0.7908, 0.2092 | 2π/(3√3) = 1.2092 | 1 / 2 / 3: 0.028, 0.828, 0.144 | π/(2√3) = 0.9069 |

- One tiling's circumcircles give **exactly two levels**, so the exact shares follow from the mean. The
  Monte Carlo agrees to ≤ 6e-4.
- R4csR2Metrics' remark that a three-valued density needs a declared rule holds for every flavor, not only
  the square.

### 3.5 Long-wave speed through a tiling (MEASURED, `effective_speed_MIRROR.py`)

Method:
- the lowest Bloch mode of −L u = λ M u on a periodic box of whole tiling periods, against the uniform
  box, with M = m (circumcircle count);
- predicted c_eff / c₀ = 1/√(mean M): the mass-only long-wave average (STANDARD homogenisation).

| grid, tiling | R (nodes) | rule | mean m | measured (12 periods) | predicted | difference at 6 → 12 periods |
|---|---|---|---|---|---|---|
| sq5, square (grid isomer) | 8.49 | open / closed / area | 1.5347 / 1.5625 / 1.5703 | 0.80708 / 0.79985 / 0.79786 | 0.80721 / 0.80000 / 0.79801 | −5.4e-4 → −1.3e-4 |
| sq9, same | 8.49 | same | same | within 1e-5 of sq5 | same | same |
| sq5, square (ZRL isomer) | 6 | open / closed / area | 1.5139 / 1.5694 / 1.5694 | 0.81267 / 0.79815 / 0.79816 | 0.81274 / 0.79823 / 0.79823 | −2.9e-4 → −7.1e-5 |
| tri, hexagon | 4 | open / closed / area | 1.1458 / 1.2708 / 1.2096 | 0.93415 / 0.88697 / 0.90919 | 0.93420 / 0.88707 / 0.90923 | −1.7e-4 → −4.2e-5 |
| tri, triangle | 6.93 | open / closed / area | 2.2639 / 2.3472 / 2.4180 | 0.66460 / 0.65269 / 0.64307 | 0.66462 / 0.65271 / 0.64309 | −9.3e-5 → −2.2e-5 |

- The relation holds. The residual falls 4× per doubling of the box: second order in k, as the long-wave
  limit predicts.
- **The sampling rule moves the speed by up to 5%** at these small R. The area rule stays within 5e-4 of
  the exact area mean.

### 3.6 Translucent one-node borders (DERIVED and MEASURED, `border_transmission_MIRROR.py`)

Definitions:
- **mass border:** one line of nodes with mass 1 + s. A line at speed factor f has s = 1/f² − 1.
- **link border:** the links crossing the line get weight 1/(1 + s).

**1D grid, derived:** |t|² = 1/(1 + X²) with X = s·tan(k/2), k = 2π/PPW, for both families. Co drops out.
- This is README 2026-09-26's `sheet1d` result, t = 1/(1 − iμΩ²/(2 sin k)), rewritten.
- One knob: |t| = cos(arg t).
- The limits: s → ∞ on a mass border is a **pinned** wall (inverted echo); on a link border it is the
  **rigid** wall (link removal, upright echo).

| PPW | 32 | 16 | 8 | 4 |
|---|---|---|---|---|
| predicted \|t\|², s = 1 | 0.990393 | 0.961940 | 0.853553 | 0.500000 |
| direct solve, mass / link | equal to 6 digits | equal | equal | equal |
| packet, Co 0.5 and 0.9, mass / link | — | 0.9619 (r² 0.0381) | 0.8535 (r² 0.1465) | — |
| arg r, mass / link | 95.6° / −73.1° | 101.2° / −56.2° | 112.5° / −22.5° | 135° / 45° |

**2D, oblique incidence** (packets, s = 1, Co 0.5; the border along a grid row; 60° quantised to 62.7° by
the periodic strip):
- sq5 closed form, same derivation: mass X = s(2 sin²(kₙ/2) + 2 sin²(kₜ/2))/sin kₙ, link X = s·tan(kₙ/2).
- In the continuum these become s·k/(2 cos θ) and s·k·cos θ/2.

| grid, family | PPW 16: 0° / 30° / 62.7° | PPW 8: 0° / 30° / 62.7° |
|---|---|---|
| sq5 formula, mass | 0.9619 / 0.9500 / 0.8456 | 0.8536 / 0.8160 / 0.5827 |
| sq5 measured, mass | 0.9619 / 0.9499 / 0.8445 | 0.8535 / 0.8158 / 0.5811 |
| sq5 formula, link | 0.9619 / 0.9714 / 0.9919 | 0.8536 / 0.8887 / 0.9680 |
| sq5 measured, link | 0.9619 / 0.9712 / 0.9915 | 0.8535 / 0.8886 / 0.9675 |
| sq9 measured, mass / link | 0.9619 / 0.9497 / 0.8403 · 0.9619 / 0.9739 / 0.9984 | 0.8535 / 0.8138 / 0.5496 · 0.8535 / 0.8948 / 0.9926 |
| tri measured, mass / link | 0.9713 / 0.9621 / 0.8758 · 0.9713 / 0.9805 / 0.9987 | 0.8887 / 0.8569 / 0.6249 · 0.8887 / 0.9214 / 0.9953 |

- **The 2D families differ.** Mass borders reflect more toward grazing incidence; link borders reflect
  less. This is a real design choice (D5.2).
- **The per-node value is not portable.** On the tri row, the same node value acts as s·√3/2: predicted
  0.9712 vs measured 0.9713 at PPW 16, 0.8860 vs 0.8887 at PPW 8. Its area per node is √3/2 at node
  spacing 1 along the row. Hence the per-length rule D5.4.
- **A thin border does not refract.** The medium is the same on both sides, so the tangential wavenumber
  and the direction are unchanged (THEOREM, phase matching). It is a partial mirror with a phase shift,
  not a lens (D5.9). A staircase border can add grating side beams on top, as staircase walls do (README
  2026-09-26, straight edges).

### 3.7 Density steps (MEASURED; 1D exact discrete solve and packets agree to 3e-5)

Head-on reflected power, mass-only step from f = 1 to f:

| f | continuum ((f−1)/(f+1))² | grid PPW 64 / 32 / 16 / 8 | impedance-matched (mass 1/f, links f, interface link (1+f)/2), PPW 16 / 8 |
|---|---|---|---|
| 0.7071 | 0.02944 | 0.02964 / 0.03026 / 0.03289 / 0.04702 | 1.5e-4 / 2.5e-3 |
| 0.55 (the existing lens) | 0.08429 | 0.08503 / 0.08732 / 0.09739 / 0.16078 | 1.2e-3 / 1.9e-2 |

- The grid reflects more than the continuum at short wavelengths, because the slow side is resolved at
  only f·PPW.
- Matching cuts the reflection 8–220× (least for strong contrast at short wavelengths), not to zero. The interface-link rule used here is
  one choice among several (§8).

### 3.8 Scale regimes and size (STANDARD, sized with measured costs)

- **Three lengths:** node spacing h = 1, cell size R, wavelength λ.
  - λ ≫ centre spacing: the long-wave average (§3.5).
  - λ comparable to the centre spacing (nearest centres R, √2R, √3R apart for triangle, square,
    hexagon): Bragg reflection; strong contrast can open band gaps (Brillouin, *Wave Propagation in
    Periodic Structures*, 1946). Not computed for these tilings (§8).
  - λ ≪ R: ray optics. Each overlap lenses on its own. Borders reflect more as λ shrinks in nodes (X grows
    as tan(π/PPW)).
- **Example:** R = 40 nodes, hexagons, k = 5 rings.
  - The cluster is about (2k + 1)·√3R ≈ 760 nodes across; with a 40-node sponge on each side, N ≈ 850.
  - At λ = R the background PPW is 40. With α = 1 the slowest level M = 3 (f = 0.577) has a local PPW of
    23.
  - TaichiWave runs N 1025 at about 0.09 ms per step on the author's GPU (README, Performance), so this is
    cheap. Interactive N up to 4097 (0.45–0.91 ms per step) is available for larger clusters.

---

## 4. Do not implement — ruled out

| claim | why |
|---|---|
| a one-node border as a lens | no refraction across a thin sheet in one medium (§3.6) |
| a per-node border value used on another grid or line direction | the tri row acts as √3/2 of sq5 (§3.6) |
| "aligned" squares on the tri grid, or triangles or hexagons on the square grid | THEOREM (§3.2) |
| a three-level density from one tiling's circumcircles | exactly two levels (§3.4) |
| point-sampled density without a tie rule | every vertex node sits on q circles (§3.3); the mean and speed move up to 5% (§3.5) |
| a "transparency" value without a PPW | \|t\| depends on PPW (§3.6) |
| CPML around a non-uniform medium without a re-check | TaichiWave absorbing_MIRROR.md (Bécache, Fauqueux & Joly 2003) |
| pentagon or circle tilings | no lattice (v5.2 §17.2) |
| a node faster than the background without lowering Co | breaks the time-step limit (§3.1) |

---

## 5. Ledger rows (proposed)

| row | expected | tolerance, notes |
|---|---|---|
| exact energy with node masses and link weights (no drive, no damping, closed edge) | constant | ~1e-12 relative (f64); the existing energy row, generalised |
| straight axis border, normal incidence, at the drive PPW | 1/(1 + X²), §3.6 | 1e-3 on sq5; sq9 and tri: report the measured value (no closed form) |
| \|r\|² + \|t\|² for every border | 1 | 1e-3 |
| long-wave speed, periodic box (D1.5 b) | 1/√(mean M) | 2e-4 at 12 periods |
| mean density, area rule | exact πR²/(cell area) (§3.4) | 1e-3 |
| tiling symmetry: a centred pulse, field rotated by 2π/n (or 2π/q) about a centre | equal | 1e-12 on the matched grids |
| reciprocity: swap source and probe | equal | 1e-12 |
| light cone: no signal beyond the fastest local speed | exactly zero (README 2026-09-26c row) | exact |
| density-step reflection | the exact discrete formula (§3.7) | 1e-3 |
| regime label λ / spacing | reported, not graded | — |

---

## 6. Build plan (each stage blocked on its decisions)

- **Stage A — core, "programmable media"** (blocked on D0.1, D3.1–D3.4, D4.1–D4.3):
  - a node-mass map from a tiling description;
  - a per-link weight array (generalises the walls' link removal: one array per link direction, 2 / 4 / 3
    for sq5 / sq9 / tri);
  - the energy with both;
  - ledger rows 1, 3, 4, 5, 8.
- **Stage B — the new app** (blocked on D1, D2, D5.1–D5.7, D6, D7): tiling and cluster, mass borders,
  panels, readouts, the remaining rows.
- **Stage C:** link borders, the matched medium, fitted circle edges (D1.8 b).
- **Stage D:** two overlaid tilings (D0.2 b), time-dependent density (D3.5).

---

## 7. Cross-app checks

- With density off (α = 0) and no borders, the new app is bit-identical to the core's uniform medium.
- s → large on a mass border approaches the pinned wall; a link weight of 0 equals the rigid wall bit for
  bit (TaichiWave `walls_tests_MIRROR.py`).
- Density levels and shares against `LatticeSpecCLAUDE/density_levels_MIRROR.py`; long-wave speed against
  `effective_speed_MIRROR.py`; border transmission against `border_transmission_MIRROR.py`.

---

## 8. Open items (not settled by measurement)

- sq9 and tri borders have no closed form at oblique incidence (measured only, §3.6).
- The leak of borders drawn as staircases on sq9 and tri (diagonal links jumping the line): not measured
  (D5.5).
- Scattering from the added masses at vertex crossings: not measured (D5.6).
- The stiffness-only medium's long-wave average and interface law: not measured (D4.1 b).
- Band gaps for λ ~ centre spacing: not computed (§3.8).
- The interface-link rule of the matched medium: only the mean weight was tried (§3.7).
- What "2 Lattice" means (D0.2), and the N / ρ symbol (D0.3): the author's call.

---

## 9. Verification log (2026-10-11, Python 3, numpy 2.4.6, scipy 1.17.1)

| script (in `LatticeSpecCLAUDE/`) | checks | result, log |
|---|---|---|
| `snap_rules_MIRROR.py` | vertices and centres on nodes for the 6 grid / tiling / isomer rows of §3.2 (plus triangle side 4: vertices on nodes, centres not) | all as stated; `snap_rules_out_MIRROR.txt` |
| `density_levels_MIRROR.py` | neighbour shells, overlap widths, density levels and shares, incircles, vertex counts, at R = 1 and R = 1.7 | §3.3, §3.4; R = 1.7 equal to R = 1: PASS ×3; `density_levels_out_MIRROR.txt` |
| `effective_speed_MIRROR.py` | long-wave speed on periodic boxes, 5 grid/tiling pairs × 3 sampling rules × 2 box sizes | §3.5; residual ≤ 1.5e-4 at 12 periods, falling 4× per doubling; `effective_speed_out_MIRROR.txt` |
| `border_transmission_MIRROR.py` | 1D closed form vs direct solve vs packets (Co 0.5, 0.9); 2D oblique on sq5 / sq9 / tri; 1D density steps, mass-only and matched | §3.6, §3.7; the sq5 formula within 1.6e-3 at 62.7°; `border_transmission_out_MIRROR.txt` |
| `border_transmission_MIRROR.py` part 4 | exact discrete solve of the 1D mass step, PPW 64 / 32 / 16 / 8 | §3.7 grid column; agrees with the packets to 3e-5 |

The solver facts in §3.1 were read from `TaichiWave/membrane_core_MIRROR.py` and
`TaichiWave/NotesCLAUDE/walls_MIRROR.md` on origin/main (2026-10-11), not re-run here.

---

## 10. Attribution

- Mass-loaded and spring-coupled interfaces (a point mass or a spring on a string), and the impedance
  reflection r = (Z₁ − Z₂)/(Z₁ + Z₂): STANDARD (any wave-physics text).
- Long-wave average of a periodic medium (homogenisation): Bensoussan, Lions & Papanicolaou, *Asymptotic
  Analysis for Periodic Structures* (1978) — STANDARD.
- Waves in periodic structures, Bragg reflection and band gaps: Brillouin (1946) — STANDARD.
- Crystallographic restriction; Niven's theorem (Niven 1956) — STANDARD.
- PML instability in anisotropic media: Bécache, Fauqueux & Joly, J. Comput. Phys. 188 (2003) — as cited
  in TaichiWave's absorbing notes.
- Original to this project: the two-level density of flavor tilings and its use as a medium; the
  translucent-border families as the continuous versions of the pinned and rigid walls; the per-length
  rule for one-node borders across grids.
