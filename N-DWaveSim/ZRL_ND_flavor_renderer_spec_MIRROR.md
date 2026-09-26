# Zero-Recursive Lattice — N-D Flavor Renderer Spec (cube + sphere first)

**For:** a coding agent (Claude Code) building the 3D / N-D flavor renderer.
**Extends:** `ZRL_visualizer_build_spec.md` — its §0 (required behaviour), §4 (design
system) and §8 (author context) apply verbatim and are not repeated here. This file
replaces the parent's "Stage 2 — z-axis / 3D" bullet: that demo becomes tier 0 here.
**Supersedes:** the scoping reply of 2026-09-13 ("cube + sphere is a small job"). The
seven decisions it left open are locked in §4 (defaults accepted 2026-09-15).
**Status:** every formula in §2 was numerically verified on 2026-09-15 (log in §11).
The 3D flavor vocabulary is fixed in §4; the 2D vocabulary caveats of parent §6 still
apply.
**Errata to carry:** parent §2.5 (the delta-sum Σ with R⁸) is RULED OUT by the
2026-09-13 mirror. Corrected relation: Δ₁ + Δ₂ = πR²/4 − ab = R²(π/4 − 1/(2γ)).
Do not copy parent §2.5 into this app.
**Date:** 2026-09-15

---

## 0. Instructions for the receiving agent

Inherit parent §0 (verify numerically, do-not-implement list, "no established name",
flag ambiguity, works / LARP / psychosis). Two additions:

1. **Tier 0 is cube + sphere only.** The code path is generic (§2.1) and it will be
   tempting to add the other Platonic solids. Don't. Ship tier 0, then extend.
2. **The ledger (§6) is the deliverable as much as the render.** Every number in §2
   must appear in the live PASS/FAIL readout, *measured from what is drawn* (mesh
   integrals, sampled directions), not recomputed from the same formula.

---

## 1. What the program is

The Stage-1 flavor renderer lifted from regular n-gons to convex solids. One
expression drives every flavor (§2.1); tier 0 renders the cube and the sphere, plus —
for free — the p-norm sweep between them with the octahedron at the far end. A
dimension slider re-verifies the parent's ball/cube table live and renders only at
n = 2, 3.

**Medium:** one self-contained HTML file, Three.js via CDN (allowed by parent §1 for
3D), vanilla JS otherwise. Reuse the Cl(3,0) rotor code already verified in
`cut_and_project_threejs_MIRROR.html` / `card_spin3d_MIRROR.html`, including the e₃₁
sign convention recorded there — do not re-derive it.

**File name:** `flavor_renderer_3d_MIRROR.html`. Oracle: `flavor3d_oracle.py`
(pytest, every closed form in §2).

---

## 2. VERIFIED FORMULA REFERENCE

### 2.1 Unified silhouette — one expression, all convex flavors

```
r(u) = a / max_i ( u · n_i )      u unit direction, n_i outward unit face normals,
                                  a = inradius (centre-to-face distance)
```
Cube (normals ±e_i): `r(u) = a / ‖u‖_∞`. Sphere: `r(u) = R`.
This is the 3D form of parent §2.2. The 2D mod-fold closed form exists because the
n edge normals are one cyclic orbit; in 3D the normals are not, so the max is taken
explicitly. Verified: `a/max_i(u·n_i) ≡ a/‖u‖_∞` at 2·10⁵ random directions.
Convexity ⇒ hidden-surface removal is a backface test; no z-buffer tricks needed
until octants chain (tier 2).

### 2.2 The signature is a vector, not a scalar

In 2D "face" and "edge" are the same thing. In 3D they split, so the ratio r/R
becomes three radii (Coxeter's ₀R vertex = circumradius, ₁R edge midradius, ₂R face
inradius):

| flavor | ₁R/₀R (edge) | ₂R/₀R (face) | tiles ℝ³ |
|---|---|---|---|
| cube | √(2/3) = 0.816497 | 1/√3 = 0.577350 | **yes** — the unique regular honeycomb {4,3,4} |
| sphere | 1 | 1 | no — Kepler π/√18 = 0.740480, gap 25.95% |
| octahedron *(tier 1, contrast)* | √2/2 = 0.707107 | 1/√3 = 0.577350 | no |
| tetrahedron *(tier 1)* | 1/√3 = 0.577350 | 1/3 | no |

Dual pairs share the facet entry and differ in the edge entry — that is *why* the
signature must be a vector: cube and octahedron are separated only by ₁R/₀R.

N-D closed forms (k = 0 vertex … n−1 facet), verified n = 3, 4, 5 against constructed
face centroids:
```
hypercube        ₖR/₀R = √((n−k)/n)
cross-polytope   ₖR/₀R = 1/√(k+1)
simplex          ₖR/₀R = √((n−k)/(n(k+1)))
ball             all 1
```
Two identities already in canon fall out: cube ₁R/₀R = √(2/3) is the
hexagon-side/cube-edge ratio of parent §2.7 (the [111]-projected edge length), and
cube ₂R/₀R = 1/√3 is C_max at D = 3 (parent §2.9's Δf = 1/√n at n = 3). So
**cube + sphere is square + circle lifted by dimension, not by side count**; the
square sits at the intersection of the two families (cos(π/4) = 1/√2 = 2^(−1/2)).

**Flag (same number, no mechanism claimed):** √(2/3) is also the triangular-lattice
Courant limit found in the 2026-09-15 wave round. Three derivations, one number —
record, don't explain.

**Caveat:** no 3D signature is cos(π/n) for any integer n (1/√3 → n ≈ 3.29). The 2D
scalar formula does not generalise; the vector does.

### 2.3 p-norm family — the continuous knob

```
r_p(u) = a / ‖u‖_p         p = 1 octahedron,  p = 2 sphere,  p = ∞ cube
inradius/circumradius   = 3^(−|1/2 − 1/p|)       (n-D:  n^(−|1/2 − 1/p|))
```
Verified (sampled at 2·10⁵ directions vs formula): p = 1.5 → 0.832683 (0.83271),
p = 4 → 0.759836, p = 8 → 0.662342; p = 1 and p = ∞ both give 1/√3.
**Implementation gotcha (verified failure):** `|u_i|^p` underflows for p ≳ 60 — the
naive sampled ratio at p = 10⁶ returns 0. Evaluate
`‖u‖_p = ‖u‖_∞ · (Σ (|u_i|/‖u‖_∞)^p)^(1/p)` and switch to the exact max for p ≥ 64.

### 2.4 Octant pieces (tier-2 material; numbers verified now)

| piece | cube (edge 1) | sphere (R = 1) |
|---|---|---|
| octant solid | a cube at half scale — self-similar | spherical wedge, not a ball |
| boundary patch | 3 quarter-faces, area 3/4 (×8 = 6 ✓) | spherical triangle, three right angles, area π/2 (×8 = 4π ✓) |
| patch boundary loop = **the carrier** (§4.1) | 6 half-edges, length 3 | 3 quarter great circles, 3π/2 |
| free exact check | — | Gauss–Bonnet: excess 3·(π/2) − π = π/2 = area ✓ |

The sphere loop 3πR/2 is the lift of the 2D quarter-circle check πR/2 in
`lattice_pathfinder`; the check transfers verbatim.

### 2.5 Body-diagonal projection — cross-app regression

Along (1,1,1): the whole cube → regular hexagon of circumradius edge·√(2/3) split
into three 120° rhombi (magic angle arccos(1/√3) = 54.7356°, parent §2.7). The
**octant patch alone** → regular hexagon of circumradius (edge/2)·√(2/3) =
0.408248·edge with its six vertices at exactly 60° spacing. Verified. That hexagon
is the LatMosaic atom in `hex_beam_MIRROR.html` (§8).

### 2.6 Dimension-slider table (parent §2.14, extended)

V_n = π^(n/2) / Γ(n/2 + 1); ratio to the enclosing cube of edge 2 = V_n / 2ⁿ.

| n | V_n | V_n / 2ⁿ |
|---|---|---|
| 1 | 2 | 1 |
| 2 | 3.141593 | 0.785398 |
| 3 | 4.188790 | 0.523599 |
| 4 | 4.934802 | 0.308425 |
| 5 | **5.263789** | 0.164493 |
| 6 | 5.167713 | 0.080746 |
| 7 | 4.724766 | 0.036912 |
| 8 | 4.058712 | 0.015854 |
| 10 | 2.550164 | **0.002490** |

Two *distinct* statements — keep them separate in the readout: V_n peaks at n = 5;
V_n/2ⁿ decreases monotonically. The 10D value 0.00249 confirms the parent's erratum.

### 2.7 Groups and counts

- C₂³ = {±1}³, order 8, **3 bits/cell** — the octant phase group (§4.2).
- O ≅ S₄, order 24, 4.585 bits — the cube's rotation group; O_h order 48, 5.585 bits.
- T_d (order 24, ≅ S₄) **contains no central inversion** — verified: −V is not the
  tetrahedron's vertex set. So the tetrahedron admits the V₄ sub-part but not the
  full octant group, the exact twin of the 2D theorem "triangle fails V₄". Every
  other Platonic solid contains −I and passes. (The order-24 tetrahedral group *with*
  inversion is the pyritohedral T_h — not the tetrahedron's group.)

### 2.8 The Dehn wall (tier-2 material; a genuine hard wall)

Cube dihedral 90°: Dehn invariant 0. Regular tetrahedron dihedral arccos(1/3) =
70.5288° — the same angle as between adjacent cube body diagonals (canon). Since
cos = 1/3 ∉ {0, ±½, ±1}, **Niven's theorem** (already a canon tool) says the angle
is an irrational multiple of π, so the Dehn invariant is nonzero. Consequences:
no dissection connects the cube flavor and the regular-tetrahedron flavor in either
direction (Dehn 1901); volume + Dehn invariant are complete invariants (Sydler
1965), so cube partials (irregular tetrahedra, pyramids — Dehn 0) are always fine.
In 2D Wallace–Bolyai–Gerwien makes every equal-area dissection free; 3D does not.
Put it in the ledger as a stated fact with the two dihedrals, not as a computation.

### 2.9 Regular cross-sections of the cube

Exactly {triangle, square, hexagon} — the regular plane tilers {3,4,6}. No regular
pentagon: a pentagonal section meets five of the six faces, hence two parallel
face-pairs, hence two pairs of parallel section edges, and a regular pentagon has
none. The pentagon flavor needs 5-fold symmetry, i.e. ℤ⁵/ℤ⁶ → Penrose, not ℤ³.
Optional tier-1 tab: a section plane with the three regular sections as snap presets.
(Whether sections count as "partial flavors" is an interpretation — §10.)

---

## 3. DO NOT IMPLEMENT — tested, failed, or ruled out

| claim | why |
|---|---|
| signature = cos(π/n) in 3D | no integer n works; use the vector (§2.2) |
| a scalar 3D signature / "midradius = inradius" | cube and octahedron share ₂R/₀R; only ₁R/₀R separates them |
| the sphere as a 3D tiler, or as the optimal equal-volume partition | it does not tile (Kepler 0.740480); the Kelvin problem is **OPEN** — Weaire–Phelan beats Kelvin by ~0.3% and is *unproven* optimal |
| cube ↔ regular-tetrahedron dissection (equal volume) | Dehn invariant obstructs (§2.8) |
| one complex phase e^{iθ} generating 3D rotations | SO(3) is non-abelian and there is no 3D division algebra (Frobenius); use Cl(3,0) rotors e^(−Bθ/2) — three bivectors replace one i |
| T_d containing an inversion | it does not (§2.7) |
| lat–long direction sampling for the ledger | pole clustering (same pathology as the angular-CFL centre); use Fibonacci sphere / icosphere (§7) |
| `‖u‖_p` by naive powers for large p | underflow; returns 0 at p = 10⁶ (§2.3) |
| parent §2.5 delta-sum with R⁸ | ruled out 2026-09-07; corrected form in the front matter |
| "flavor reassembles under the octant group" for the tetrahedron | fails; T_d lacks −I |

---

## 4. Decisions — LOCKED 2026-09-15 (defaults accepted)

**4.1 Carrier of a 3D flavor := the boundary loop of the octant patch** (cube: 6
half-edges, length 3 at edge 1; sphere: 3 quarter great circles, 3πR/2).
Rejected: the surface patch itself (the metric becomes area / surface geodesics and
Stage 3's pathfinder does not lift); a corner-to-corner staircase (needs an
arbitrary edge choice). Consequence: Stage 3 lifts verbatim; the perimeter checks
are the 2D ones lifted.

**4.2 Phase group := C₂³** (order 8, abelian, 3 bits) — matches "bound the
fundamental domain to one octant"; frames transfer by sign flips, so a path's phase
stays a *commuting* word and the Stage-4 ledger format is unchanged.
Recorded and deferred: O ≅ S₄ is the true symmetry group; under it a path's phase is
an ordered word in a non-abelian group ("the ORDER OF MOVES becomes load-bearing").
Revisit only when Stage 4 handles non-abelian ledgers.

**4.3 Flavor set:** tier 0 {cube, sphere}. Tier 1 adds the five Platonic solids +
sphere (2D included non-tilers, so the analogy argues for six), keeping the
tetrahedron *because* it fails (§2.7, §2.8), and the space-fillers via the
truncation slider.

**4.4 Isomer := (axis class, sign).** Cube: face (4-fold) / vertex (3-fold, the
magic angle) / edge (2-fold) — the LatMosaic view classes. Sphere: one. Sign is
meaningful only for solids without inversion (tetrahedron; decide its convention
when tier 1 starts).

**4.5 Continuous knob := the p-norm slider (§2.3)** — it *is* the Finsler /
metric-ball picture already in `metrics_algebra`, now in 3D. Truncation slider
(cube → truncated cube → cuboctahedron → truncated octahedron → octahedron) optional
at tier 1; it passes through the truncated octahedron = permutohedron = BCC Voronoi
cell, Kelvin's 1887 space-filler. Dimension slider (§2.6) at tier 0 — nearly free.

**4.6 Chaining rule := transfer point and frame; new frame = old frame conjugated by
the phase element.** With C₂³ that is axis signs only — the other reason to start
abelian. (2D `P₂ = O′` transfers a point; 3D needs 3 more DOF.)

**4.7 Language := HTML + Three.js (CDN) for the renderer; pytest oracle in Python
for every closed form in §2**, run *before* the ledger tolerances are set. Python's
numpy/sympy/pytest advantages are Stage-4 / wave-engine advantages, not rendering
ones.

---

## 5. Panels and controls (tier 0)

**Left panel** (parent §4 layout, 280–300 px):
- flavor select {cube, sphere}
- p-norm slider p ∈ [1, 64] with an "∞" button that snaps to the exact cube
- isomer select {face, vertex, edge}
- rotor orbit: axis, rate, play/pause; **"snap to 54.7356°"** button (keeps the
  parent Stage-2 demo: eight vertices collapsing to six at exactly 60°)
- dimension slider n ∈ [1, 12]: renders only at n = 2, 3; table row for every n
- show/hide: octant patch, its boundary loop, [111] projection ghost, face normals
- R (scale) slider — for homogeneity checks only

**Canvas:** Three.js scene. Cube as its 6 exact quads; sphere as an icosphere
(level 4–5); p-ball meshed by radially scaling a sphere mesh by r_p(u), with the
exact vertex and edge directions inserted as p → ∞. Backface culling only.
Wireframe toggle. Use the project palette (parent §4); data colours `hsl(h,62%,60%)`.

**Readout:** live ₀R, ₁R, ₂R ratios, p and its predicted ratio, volume, area, Euler
χ, projected-hexagon circumradius, the V_n row — then the §6 ledger.

---

## 6. Identity ledger — live PASS/FAIL, tolerances stated

| check | expected | tolerance | ref |
|---|---|---|---|
| V − E + F | 2 | exact | mesh |
| volume, divergence theorem on the mesh, (1/3)Σ(c·n)A | cube 1; sphere 4π/3 = 4.188790 | cube 1e-12; icosphere relative 1e-3 at level 4 (O(h²), report the order) | mesh |
| surface area | cube 6; sphere 4π = 12.566371 | same | mesh |
| ₂R/₀R, ₁R/₀R | cube 1/√3, √(2/3); sphere 1, 1 | 1e-9 | §2.2 |
| dual check *(tier 1)* | cube & octahedron equal in ₂R/₀R, differ in ₁R/₀R | 1e-9 | §2.2 |
| max-over-normals vs mesh radial distance | max residual over Fibonacci directions | 1e-9 cube (exact quads); 1e-3 sphere | §2.1 |
| p-ball signature | 3^(−\|1/2−1/p\|) | 1e-6 at 10⁴ directions | §2.3 |
| octant volume | 1/8 of the total | 1e-12 / 1e-3 | §2.4 |
| sphere octant Gauss–Bonnet | excess = π/2 = area | 1e-6 | §2.4 |
| carrier loop length | cube 3 (edge 1); sphere 3π/2 = 4.712389 | 1e-9 | §2.4 |
| [111] projection of the octant patch | regular hexagon, circumradius (edge/2)√(2/3), vertices at 60° | 1e-9 | §2.5 |
| V_n table | peak at n = 5; V₁₀/2¹⁰ = 0.002490 | 1e-6 | §2.6 |
| −V ∉ vertex set *(tier 1, tetrahedron)* | false | exact | §2.7 |
| homogeneity: volume(2R)/volume(R), area(2R)/area(R) | 8, 4 | 1e-9 | parent standing check |

Green = `--hi`, red = `--link`, per parent §4.

---

## 7. Sampling and implementation notes

- **Directions:** Fibonacci sphere `z_i = 1 − (2i+1)/N`, `φ_i = i·π(3−√5)` (González
  2010), or a geodesic icosphere. Verified: nearest-neighbour spacing std/mean = 0.013
  at N = 2000 versus > 0.5 near the poles of a lat–long grid. **Insert the exact
  vertex directions and edge-arc samples** when checking the cube — a sampled radial
  mesh converges only O(h) at creases, and 1e-5 tolerances then fail on the cube
  while passing on the sphere.
- **Volume and area come from the mesh**, never from the formulas — that is the point.
- **Rotor:** reuse the verified Cl(3,0) code; isomer snaps are rotor presets (face:
  identity; vertex: rotate (1,1,1) onto the view axis; edge: rotate (1,1,0)).
- **Three.js via CDN only**; no build step; one file.

---

## 8. Cross-app checks

- `hex_beam_MIRROR.html`: the octant patch viewed along (1,1,1) must reproduce the
  three-rhombus atom's hexagon exactly (§2.5). This is a real regression test.
- `cut_and_project_threejs_MIRROR.html`: the "snap to 54.7356°" demo is subsumed;
  numbers must be identical.
- `metrics_algebra_MIRROR.html`: the p-norm slider is the 3D version of its
  L1 / L2 / L∞ ball tab; at p = ∞ the 2D signature is 1/√2 = cos(π/4).
- `lattice_pathfinder_MIRROR.html`: the carrier loop lengths are its quarter checks
  lifted (πR/2 → 3πR/2).

---

## 9. Tiers

| tier | scope | effort | what drives it |
|---|---|---|---|
| **0 — this spec** | cube + sphere, p-slider, isomers, rotor orbit, dimension table, full §6 ledger | one session, ~600 lines | almost all of it is check code |
| 1 | five Platonic + sphere; dual pairs; truncation slider; cross-section tab; tetrahedron failure demos (§2.7, §2.8) | +1–2 sessions | solid data tables, isomer sign convention |
| 2 | octant pieces as flavors, C₂³ phase, chaining with frame transfer, non-convex depth sorting once octants chain | code ≈ 1.5× the 2D work; spec ≈ 3–5× | depends on the Stage-4 ledger conventions; genuine depth sorting |

---

## 10. Open items — do not resolve silently

- Tetrahedron isomer sign convention (4.4) — when tier 1 starts.
- Whether cross-sections (§2.9) count as "partial flavors" — an interpretation,
  flagged, not decided.
- Kelvin problem — OPEN in the literature; never present Weaire–Phelan as optimal.
- Stage-4 ledger format for non-abelian phases — required before O replaces C₂³.
- The √(2/3) triple coincidence (§2.2) — record only.

---

## 11. Verification log (2026-09-15, Python / numpy / sympy)

| check | result |
|---|---|
| a/max_i(u·n_i) ≡ a/‖u‖_∞, 2·10⁵ random u | PASS |
| k-face radii, n = 3, 4, 5, cube / cross-polytope / simplex vs closed forms | PASS |
| L^p signature 3^(−\|1/2−1/p\|), p = 1, 1.5, 2, 3, 4, 8 | PASS; p = 10⁶ naive → 0 (underflow, §2.3) |
| octant patch areas ×8 = 6, 4π; loops 3, 3π/2; Gauss–Bonnet π/2 | PASS |
| [111] projection of the octant patch: 6 vertices, radius 0.408248, angles 60°…360° | PASS |
| V_n table and V₁₀/2¹⁰ = 0.002490 | PASS |
| −V ≠ V for the regular tetrahedron | PASS (no inversion) |
| tetrahedron dihedral arccos(1/3) = 70.528779°, cos = 1/3 rational → Niven | PASS (as stated) |
| Kepler gap 0.259520; log₂24 = 4.585 | PASS |
| Fibonacci sphere NN uniformity std/mean = 0.0134 (N = 2000) | PASS |

---

## 12. Attribution (audited 2026-09-15)

- **Coxeter**, *Regular Polytopes* (Dover 1973): the k-face radii ₖR — §2.4 (3D),
  §7.9 / §8.8 (n-D), Table I. STANDARD. The ordered tuple as a *shape signature*:
  no source found — claim as this project's framing.
- **Dehn** (Math. Ann. 55, 1901); **Sydler** (Comment. Math. Helv. 40, 1965);
  **Jessen** (Math. Scand. 22, 1968); Wallace–Bolyai–Gerwien (2D). STANDARD. The
  Niven route (Niven 1956; Aigner–Ziegler) is the standard proof; deploying it as a
  designed-in wall is this project's framing.
- **Hales** (Ann. Math. 162, 2005; Flyspeck 2017) — Kepler. **Weaire & Phelan**
  (Phil. Mag. Lett. 69, 1994); **Gabbrielli** (Phil. Mag. Lett. 89, 2009) — Kelvin
  problem, OPEN. STANDARD.
- **González**, Math. Geosci. 42, 49 (2010); Swinbank & Purser, Q.J.R. Meteorol. Soc.
  132, 1769 (2006) — Fibonacci sphere. STANDARD.
- **Barr**, IEEE CG&A 1(1), 11 (1981) — superquadrics / p-norm family. STANDARD.
- **Conway & Sloane**; **Ziegler** (permutohedron); Kelvin, Phil. Mag. 24, 503 (1887).
  STANDARD.
- **de Bruijn**, Indag. Math. 43, 39 (1981) — cut-and-project ancestor; the [111]
  projection edge factor √(2/3): KNOWN BUT SCATTERED.
- Cube regular sections {3,4,6}: KNOWN BUT SCATTERED (Steinhaus, *Mathematical
  Snapshots*; parity argument is folklore).
- Original to this project as far as searched: the ₖR signature vector; the
  Niven-route Dehn wall; the octant-patch = LatMosaic-atom regression check.
