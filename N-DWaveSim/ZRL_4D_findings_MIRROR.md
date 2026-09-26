# ZRL 4D — Findings: what tiles 4-space, the regular polytopes of 4D, and how to render one

**Project:** Zero-Recursive Lattice, sub-project `Polytopes4DCLAUDE` (new, 17/09/2026).
**Parent canon:** `ZRL md Context Files\` (v4 + general context) and
`FlavorRenderersCLAUDE\ZRL_ND_flavor_renderer_spec_MIRROR.md`. This file is the 4D
continuation of that spec's §2.2 N-D signature work; it is a **findings document**, not a
build spec. Nothing here is an app yet.
**Written by:** Claude Opus 5, 17/09/2026.
**Method:** 11-dimension research sweep, 46 agents; every claim refuted by three independent
adversarial lenses (source accuracy / derivation / overclaim), then two critics (completeness,
numbers). 465 raw claims, 2-of-3 refutation threshold, survivors only below.
**Verification status — READ THIS FIRST:** the research itself was done with **no process
executed** (your standing rule) — independent symbolic derivation and adversarial cross-check
only. `test_polytope4d_oracle_MIRROR.py` in this folder covers every closed form here and **was
run by the author on 21/09/2026: 84 passed, 1 failed.** The single failure was a **defect in this
document, not in the code** — the 16-cell's Kelvin quotient was printed as 7.228760 when the
closed form gives 2^(13/4)/3^(1/4) = 7.228816029. Corrected in §1.5; the test now asserts the
closed form. §10 states exactly what is and is not checked, and how.

---

## 0. VERDICT TABLE

| # | Your question | Verdict |
|---|---|---|
| 1 | How many shapes tile in 4D? | **Reading-dependent, and the readings differ by five orders of magnitude.** Regular honeycombs: **3**. Regular 4-polytopes that tile: **3 of 6**. Parallelotopes (translation-only, the Fedorov continuation): **52**. Convex uniform honeycombs: **143 known, completeness NOT proved**. Any convex body under any isometry: **uncountably many**, classification OPEN even in 3D. |
| 2 | What are the regular polytopes of 4D? | **Exactly 6** convex — 5-cell, tesseract, 16-cell, 24-cell, 120-cell, 600-cell — **THEOREM** (Schläfli). **16** if you allow star polytopes (+10 Schläfli–Hess). Your canon had the 6; it did not have the 10, and the 10 are the reason 4D is exceptional in a second, independent way. |
| 3a | How does the 4th dimension get represented? | Three structurally distinct routes — **orthographic**, **perspective/Schlegel**, **stereographic from S³** — plus a fourth family, **slicing**. They are not stylistic variants; they preserve disjoint things. §3.1. |
| 3b | Through time, as you imagined? | **Partly right, and the imprecision matters.** "w as time" is **two different methods** with disjoint invariants (slice-vs-time, rotate-vs-time). And the w of a 4-polytope is a **Euclidean** axis (+,+,+,+), *not* Minkowski time (−,+,+,+). Operational test: a Euclidean (x,w) rotation passes through flat and **re-inflates mirrored**; a Lorentz boost never can. §3.3. |
| 3c | How do you measure length? | **You cannot, from one still image — provably, by rank–nullity.** Every P: ℝ⁴→ℝ³ has a 1-D kernel. But it is recoverable: **two views with distinct kernel lines**, or any **rotating** sequence with vertex correspondence, determine every 4D coordinate exactly. Stereographic recovers true arc length from a single image via a stated formula. §3.4. |
| 3d | Could you tell a regular object is regular? | **Metric regularity: NO from any single projection, provably.** The tesseract's vertex-first shadow draws **all 32 edges at exactly equal length √3** — and the solid you are looking at is a **rhombic dodecahedron**, which is not regular. **Combinatorial regularity: YES**, fully readable off one Schlegel diagram. The certificate is **\|G\| = F** (flag count) on recovered coordinates. §3.5. |
| — | Bonus, and the best result in the sweep | Your Niven line is **strictly weaker than the truth**. The 4-cube's *entire* signature vector is (cos π/6, cos π/4, cos π/3), and **n = 2 and n = 4 are the only dimensions where that happens**. §2.4. |
| — | Worst news | **The carrier does not lift.** The orthant patch's boundary is S^(D−2) — a loop only at D = 3. Stage 3 does not lift to 4D as specified. §4.5. |

Status tags used throughout: THEOREM, ESTABLISHED, CONJECTURE, OPEN, CONTESTED, CONVENTION,
RULED OUT, CATEGORY ERROR, FRAMING (= this project's packaging of standard arithmetic).

---

## 1. HOW MANY SHAPES TILE IN 4D — five readings

The question has no single answer, and the house rule (split the readings, as you already do for
"the cube is the only 3D flavor") applies with force here.

| Reading | 2D | 3D | **4D** | 5D | Status |
|---|---|---|---|---|---|
| **(a) Regular honeycombs** | 3 | 1 | **3** | 1 | THEOREM (Coxeter/Schläfli) |
| **(b) Regular polytopes that tile** | 3 of ∞ | 1 of 5 | **3 of 6** | 1 of 3 | THEOREM |
| **(c) Parallelotopes** (translation only) | 2 | 5 | **52** | 110244 | THEOREM (4D); see note |
| **(d) Convex uniform honeycombs** | 11 | 28 known | **143 known** | — | **NOT PROVED COMPLETE** |
| **(e) Any convex body, any isometry** | ∞ | ∞ | **∞** | ∞ | OPEN classification |

### 1.1 (a) The three regular honeycombs — and the test that produces them

**{4,3,3,4}** tesseractic · **{3,3,4,3}** 16-cell · **{3,4,3,3}** 24-cell. **THEOREM.**
Your canon's list is exactly right.

The derivation is one line and it answers Q1 *from* Q2, which is the strongest cross-check in
the whole sweep. A regular polytope tiles Euclidean 4-space face-to-face iff an integer number
of copies close around a 2-face, i.e. **360° / dichoral angle ∈ ℤ**:

| polytope | dichoral (dihedral) angle | 360°/δ | tiles ℝ⁴ |
|---|---|---|---|
| 5-cell {3,3,3} | arccos(1/4) = 75.5224878° | 4.7667921 | no |
| **tesseract {4,3,3}** | 90° | **4** | **yes** |
| **16-cell {3,3,4}** | 120° | **3** | **yes** |
| **24-cell {3,4,3}** | 120° | **3** | **yes** |
| 120-cell {5,3,3} | 144° | 2.5 | no |
| 600-cell {3,3,5} | 164.477512° | 2.1887490 | no |

The three negatives are **closed, not open** — but the proof is *not uniform*, and that detail
was the single most-contested point in the sweep. At a generic point of a codimension-2 face,
local finiteness forces `a·δ + b·π = 2π` with a ≥ 1, b ≥ 0 integers:
- **5-cell:** cos δ = 1/4, rational and outside {0, ±½, ±1}, so **by Niven** δ/π is irrational ⇒ a = 0, contradiction. (Your own Niven tool, reused.)
- **600-cell:** 2cos δ = −(1+3√5)/4 satisfies 4x² + 2x − 11 = 0 — not monic over ℤ, so not an algebraic integer ⇒ same conclusion.
- **120-cell:** δ = 4π/5 **is** a rational multiple of π, so Niven does **not** apply. The integer step is needed: 4a/5 + b = 2 forces 5 | a, a = 5, b = −2 < 0.

Write it as three cases or it is wrong. **THEOREM**, with the non-uniformity flagged.

### 1.2 (c) 52 — the literal answer you probably want

**The number of combinatorial types of 4-dimensional parallelotopes is 52.** **THEOREM.**
Delone (1929) found 51; **Štogrin** found the missing 52nd. This is the direct continuation of
Fedorov's 5 parallelohedra in 3D and the 2 in 2D:

> **1, 2, 5, 52, 110244** for dimensions 1–5.

**Correction to a number the sweep itself got wrong three different ways:** the 5D term is
**110244** combinatorial types (Dutour Sikirić, Garber, Schürmann, Waldmann, *Acta Cryst.* A72
(2016) 673–683; arXiv:1507.00238). Engel's 2000 figures — 103769 combinatorial, 179372
contraction — are **superseded** by that paper. "Contraction type" is a *finer* invariant than
"combinatorial type"; never equate them. The same paper's contraction count is 181394.

Voronoi's conjecture (every parallelotope is affinely a lattice Voronoi cell) is a **THEOREM
through dimension 5** (Garber 2025), so in 4D the 52 parallelotopes coincide with the 52
combinatorial types of lattice Voronoi cells. **OPEN in dimension ≥ 6** — do not assume the
identification higher up.

### 1.3 (d) 143 — say "known", never "exactly"

**143 convex uniform honeycombs of E⁴ are known.** The list rests on George Olshevsky's
self-published *Uniform Panoploid Tetracombs* (2006), which is **unrefereed and not proved
complete**. **Quote as "143 known"; never as a theorem.** The 3D analogue is softer than usually
stated too: 28 known (Andreini 1905 gave 25; Johnson 1991 and Grünbaum 1994 completed the list),
completeness not proved by either.

### 1.4 (e) and the crystallographic reading nobody expects

Under "any convex body, any isometry" the answer is **uncountably many** — every prism over a
plane-filler tiles, and the parameter is continuous. The classification is **OPEN in 3D** (the
2D convex-pentagon case was only finished in 2017, at 15 types), and in 4D it is not even
approached. But it is **finite in principle**: the 4D crystallographic groups are finite —
**64 Bravais classes, 227 arithmetic crystal classes, 4783 affine space-group types**
(Brown, Bülow, Neubüser, Wondratschek, Zassenhaus, *Crystallographic Groups of Four-Dimensional
Space*, Wiley 1978) — and Delone bounds stereohedron facet counts. So the honest status is
**"finite by theorem, never enumerated"**, not "unknown".

That 4783 is the literal continuation of the 230 space groups, and is the reading a
crystallographer would assume you meant.

### 1.5 Packing, covering, and what is OPEN in 4D

| quantity | 4D value | status |
|---|---|---|
| densest **lattice** packing | **D₄**, density π²/16 = **0.616850275** | THEOREM (Korkine–Zolotareff 1872) |
| densest packing (non-lattice allowed) | — | **OPEN.** No non-lattice packing beating D₄ is known |
| best upper bound | density ≤ **0.636108** | Cohn, de Laat, Salmon, arXiv:2206.15373 — D₄ is 3.03% below it |
| kissing number τ₄ | **24** (the D₄ minimal vectors = a 24-cell) | THEOREM (Musin, *Ann. Math.* 168 (2008) 1–32) |
| uniqueness of that kissing configuration | — | **OPEN** (unlike dims 8 and 24, Bannai–Sloane) |
| thinnest lattice covering | **A₄\***, 2π²/(5√5) = **1.7655285** | Delone–Ryshkov 1963 |
| **Kelvin problem in 4D** | candidate: relaxed 24-cell honeycomb | **OPEN** — no proof of even local minimality found |

The **24-cell conjecture** (Conway & Sloane, *SPLAG*) — that the minimal Voronoi-cell volume in
any unit-sphere packing of ℝ⁴ is that of a 24-cell circumscribed about a unit sphere — is
**OPEN**, and proving it would settle the packing problem.

Flat-faced 4D Kelvin candidates, normalised S₃/V₄^(3/4) — **lower is better**, and the 4-ball is
the unattainable floor:

| candidate | exact | value | status |
|---|---|---|---|
| 4-ball (floor) | 2π²/(π²/2)^(3/4) | **5.962003** | ORACLE-VERIFIED 21/09/2026 |
| 24-cell | **2^(11/4)** | **6.727171** | ORACLE-VERIFIED |
| A₄\* permutohedron | — | 6.831200 | **UNVERIFIED** — see below |
| 16-cell | **2^(13/4)/3^(1/4)** | **7.228816029** | ORACLE-VERIFIED (**corrected**) |
| tesseract | 8 | **8** | ORACLE-VERIFIED |

**Erratum, caught by the oracle on its first run (21/09/2026):** this list originally printed the
16-cell as **7.228760**, which is wrong in the 6th significant figure. The closed form is
q⁴ = (4√2/3)⁴·216 = 1024·216/81 = **2¹³/3 = 8192/3**, so q = **2^(13/4)/3^(1/4) = 7.228816029**.
The figure came from the sweep, and the test that caught it is the one that rebuilt the quotient
from S₃ and V₄ instead of copying the number — the §10 precision warning, validated the hard way.

**Consequence: treat the A₄\* entry as suspect.** It came from the same list as the bad 16-cell
figure and is the one candidate I could not rebuild from an independent S₃ and V₄, so it is
deliberately **not** in the oracle. Do not quote 6.831200 until it is derived. The *ordering*
4-ball < 24-cell < 16-cell < tesseract is oracle-verified; where A₄\* sits in it is not.

The Weaire–Phelan precedent says the true optimum is likelier non-regular and multi-cell.

### 1.6 Aperiodicity in 4D

- Aperiodic **pairs** exist in Eⁿ for every n ≥ 3 (Goodman-Strauss, *Eur. J. Combin.* 20 (1999) 385–395). **ESTABLISHED.**
- A **strongly aperiodic monotile** in E³ or E⁴: **OPEN.** 3D has only the weakly aperiodic Schmitt–Conway–Danzer biprism and the Socolar–Taylor tile (canon already has this). 2D was settled in 2023 (the hat; the reflection-free spectre), which is *why* 3D/4D are live now.
- Minimum dimension admitting an **aperiodic translational monotile** (Greenfeld–Tao): **OPEN**; the periodic tiling conjecture is proved only for d = 1 and ℤ², false in some large unspecified dimension. All that is known is d ≥ 3, and **d = 4 is not ruled out**.

### 1.7 One structural find worth keeping

**The cross-polytope is a regular space-filler in exactly two dimensions: n = 2 and n = 4.**
**THEOREM.** The cross-polytope *is* your advection CFL cone (the weighted L1 ball). So the
object your lattice light-cone is made of tiles space only at n = 2 and n = 4.

**Handle with care.** This lands on the same {2,4} as your Niven result — but see §4.1: the
sweep's critics killed the attempt to merge it with the Huygens wake, which is an **even/odd**
fact, not a {2,4} fact. Two different families of "4 is special" are now in play and merging
them is exactly the failure mode your protocol exists to prevent. §4.6.

---

## 2. THE REGULAR POLYTOPES OF 4D

### 2.1 The six, complete

All f-vectors below were re-derived by incidence counting (cells-per-vertex = the **face** count
of the vertex figure — using its *vertex* count is the standard trap and gives 200 instead of
120 for the 600-cell), not copied from a table.

| name | Schläfli | V | E | F | C | cell | vertex figure | dual | Coxeter | \|W\| | rotations |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5-cell | {3,3,3} | 5 | 10 | 10 | 5 | tetrahedron | tetrahedron | **self-dual** | A₄ ≅ S₅ | 120 | 60 |
| tesseract | {4,3,3} | 16 | 32 | 24 | 8 | cube | tetrahedron | 16-cell | B₄ | 384 | 192 |
| 16-cell | {3,3,4} | 8 | 24 | 32 | 16 | tetrahedron | octahedron | tesseract | B₄ | 384 | 192 |
| 24-cell | {3,4,3} | 24 | 96 | 96 | 24 | octahedron | cube | **self-dual** | F₄ | 1152 | 576 |
| 120-cell | {5,3,3} | 600 | 1200 | 720 | 120 | dodecahedron | tetrahedron | 600-cell | H₄ | 14400 | 7200 |
| 600-cell | {3,3,5} | 120 | 720 | 1200 | 600 | tetrahedron | icosahedron | 120-cell | H₄ | 14400 | 7200 |

**The 4D Euler relation is V − E + F − C = 0**, not 2. General: Σ(−1)^k f_k = 1 − (−1)^n, so it
is **0 in every even dimension and 2 in every odd one**. Verified on all six:

```
5-cell     5 −   10 +   10 −   5 = 0      tesseract  16 −  32 +  24 −   8 = 0
16-cell    8 −   24 +   32 −  16 = 0      24-cell    24 −  96 +  96 −  24 = 0
120-cell 600 − 1200 +  720 − 120 = 0      600-cell  120 − 720 + 1200 − 600 = 0
```

Attribution is CONTESTED in the usual way: Schläfli stated it c.1852 without a modern proof;
**Poincaré** supplied the homological one (C. R. Acad. Sci. Paris 117 (1893) 144–145).

**Why exactly six.** The rank-4 Schläfli criterion is `cos(π/q) < sin(π/p)·sin(π/r)`, which is
positive-definiteness of the Gram matrix (det = sin²(π/p)sin²(π/r) − cos²(π/q) > 0). Eleven
ordered symbols are candidates; six pass, **{4,3,4} gives equality exactly** — that is your
cubic honeycomb sitting on the boundary — and {3,5,3}, {4,3,5}, {5,3,5} go negative into
hyperbolic space. **THEOREM** (Schläfli; independent rediscoveries include Stringham 1880).

### 2.2 **NEW CANON — the signature vectors of all six**

This is the genuinely new material: your ₖR/₀R invariant extended to 4D. All at **edge length 1**,
s_k = ₖR/₀R, vector = (s₁, s₂, s₃).

| polytope | s₁ (edge) | s₂ (face) | s₃ (facet) |
|---|---|---|---|
| **5-cell** | √6/4 = 0.612372436 | √6/6 = 0.408248290 | 1/4 = 0.250000000 |
| **tesseract** | √3/2 = 0.866025404 = **cos π/6** | √2/2 = 0.707106781 = **cos π/4** | 1/2 = 0.500000000 = **cos π/3** |
| **16-cell** | √2/2 = 0.707106781 = **cos π/4** | √3/3 = 0.577350269 | 1/2 = 0.500000000 = **cos π/3** |
| **24-cell** | √3/2 = 0.866025404 = **cos π/6** | √6/3 = 0.816496581 | √2/2 = 0.707106781 = **cos π/4** |
| **120-cell** | φ√6/4 = 0.990839415 | √((5+2√5)/10) = 0.973248990 | φ²√2/4 = 0.925614793 |
| **600-cell** | √(10+2√5)/4 = 0.951056516 = **cos π/10** | φ/√3 = 0.934172359 | φ²√2/4 = 0.925614793 |

Radii at edge 1 for the three canon families evaluate correctly from your existing closed forms
at n = 4 (hypercube √((n−k)/n), cross-polytope 1/√(k+1), simplex √((n−k)/(n(k+1)))) — **CONFIRMS
CANON**, checked against independent coordinate constructions, not by re-substituting the formula.

**The organising identity that generates the whole s₁ column** — stated nowhere in your canon and
worth adding on its own:

> **s₁ = cos(δ_edge/2)**, where **δ_edge(P) = 180° − dichoral(P\*)** is the arc an edge subtends
> at the centre of the circumsphere S³.

Edge arcs: 5-cell 104.4775122°, tesseract 60°, 16-cell 90°, 24-cell 60°, 120-cell **15.5224878°**,
600-cell 36°. Check the last: 180° − 164.477512° = 15.5224878° ✓, and cos(7.7612439°) = 0.990839415
= the 120-cell's s₁ ✓. This single line explains *why* the 600-cell's s₁ is exactly cos(π/10):
its edge arc is 36° = 2π/10, because the 120-cell's dichoral angle is 144°. No mystery left.

**Precision note:** the sweep printed 15.5219660° for that arc in one place; it is **15.5224878°**.
Wrong in the 6th significant figure, and it breaks the exact dual identity. See §10.

### 2.3 The duality law — your canon records only a special case

Your spec §2.2 says *"dual pairs share the facet entry and differ in the edge entry."* True in
3D, but it is one corollary of the general law:

> **s_k(P\*) = s_{n−1} / s_{n−1−k}(P)**   **THEOREM**, verified entry-by-entry.

Check on tesseract → 16-cell (n = 4, s₃(tess) = 1/2):
s₁(16) = s₃/s₂ = (1/2)/(√2/2) = 1/√2 ✓ · s₂(16) = s₃/s₁ = (1/2)/(√3/2) = 1/√3 ✓ ·
s₃(16) = s₃/s₀ = 1/2 ✓ (since s₀ ≡ 1).
"Duals share the facet entry" is just the k = n−1 case, where s₀ = 1. Recording only that special
case is what let one agent conclude "duals share exactly **one** component" as a theorem — which
is **false for self-dual polytopes**.

**Self-duality in the signature — the exact statement.** A self-dual regular n-polytope satisfies
**s_k · s_{n−1−k} = s_{n−1}** for every k. In 4D that is the single identity **s₁·s₂ = s₃**:

- 24-cell: (√3/2)(√6/3) = √18/6 = √2/2 = s₃ ✓
- 5-cell: (√6/4)(√6/6) = 6/24 = 1/4 = s₃ ✓
- (3D tetrahedron: s₁·s₂ = (1/√3)(1/3)… reduces to the same law at n = 3)

It is a **multiplicative palindrome, not a literal one** — s_k = s_{n−1−k} is false and would force
the ball. In log coordinates t_k = −log s_k ≥ 0 it reads t_k + t_{n−1−k} = t_{n−1}, which is a
genuine palindrome. **This is a free cross-mesh ledger row.**

### 2.4 **The result that supersedes your Niven line**

Your canon records the single-entry fact: 1/√D = cos(π/3) at D = 4. The truth is stronger and
completely explained:

> **THEOREM.** The n-cube's signature s_k = √((n−k)/n) consists *entirely* of cos(π/m) values for
> integer m **iff n = 2 or n = 4.**
>
> Proof: s_k² = (n−k)/n must lie in the Niven set of rational values of cos²(π/m), which is exactly
> {0, 1/4, 1/2, 3/4, 1}. Already at k = 1, (n−1)/n ∈ {1/4, 1/2, 3/4} forces n ∈ {2, 4}. At n = 4
> the squares are (3/4, 2/4, 1/4) — the Niven set minus its endpoints, each hit once.

So the 4-cube's vector is **(cos π/6, cos π/4, cos π/3)**, and **n = 4 is the unique dimension
above 2 in which the 2D cos(π/n) signature formula returns in full.** No residual mystery, and
**it has no established name — do not give it one.**

**Every cos(π/m) hit in 4D — 8 of the 18 components:**

| polytope | hits |
|---|---|
| tesseract | s₁ = cos π/6, s₂ = cos π/4, s₃ = cos π/3 (all three) |
| 16-cell | s₁ = cos π/4, s₃ = cos π/3 |
| 24-cell | s₁ = cos π/6, s₃ = cos π/4 |
| 600-cell | s₁ = cos π/10 |
| 5-cell, 120-cell | none |

The remaining ten are **provably not** cos(rπ) for any rational r (Niven for the rational-square
ones; the algebraic-integer test for the golden-ratio ones). The question is now **settled**, not
merely "no integer found".

**Repeat numbers worth recording, mechanism not claimed** (house rule: record, don't explain):
- 24-cell s₂ = **√6/3 = √(2/3) = 0.816497** — the *fourth* appearance of √(2/3) in this project (cube's 3D edge signature; the [111] projection edge factor; the triangular-lattice Courant limit; now the 24-cell's face signature).
- 16-cell s₂ = **1/√3** = the cube's 3D facet signature = C_max at D = 3.

### 2.5 The 16 reading — star polytopes

**There are exactly 10 regular star 4-polytopes** (Schläfli–Hess polychora) — against 4 in 3D
(Kepler–Poinsot) and **zero in every dimension above 4**. **THEOREM**; Schläfli found some,
**Hess (1883)** completed the list. All 10 share the H₄ group of order 14400. Their element counts
were re-derived independently as 14400/|stabiliser|.

- **Nine have 120 vertices**, the exception being {5/2,3,3} (600 vertices).
- **Nine have 120 cells**, the exception being {3,3,5/2} (600 cells).
- Those are **two different exceptions** (dual polytopes), not one — the sweep got this wrong twice before the critics caught it.
- Euler characteristics: 480, −480, 0, 0, 0, 0, −480, 480, 0, 0 — the four nonzero ones are a genuine star phenomenon (χ ≠ 0 cannot happen for a convex 4-polytope).

**⚠ Your invariant does not extend to them.** A self-intersecting cell has **no well-defined
inradius or midradius**, so ₖR/₀R is undefined for all ten. If the "16 regular polytopes" reading
enters canon, the signature silently acquires ten shapes it cannot measure. The star sector's
invariant is **density** (4, 4, 6, 20, 20, 66, 76, 76, 191, 191 — six distinct values), which is
literally the covering number of S³ and therefore **measurable from a render as a ray-crossing
count**. That is a ledger row nobody has built.

### 2.6 Hyperbolic and star honeycombs — where 4 stops being special

| space | compact regular | paracompact regular | regular **star** |
|---|---|---|---|
| E^n | — | — | **0 for all n** |
| H² | ∞ | ∞ | **∞** (two families {m/2,m}, {m,m/2}, odd m ≥ 7) |
| H³ | 4 | 11 | 0 |
| **H⁴** | **5** | **2** | **4** (densities 5, 5, 10, 10) |
| H⁵ | 0 | 5 | 0 |
| H^n, n ≥ 6 | 0 | 0 | 0 |

**Two corrections the sweep had to make against itself, both important:**

1. The sweep's own headline claimed *"regular star honeycombs exist in exactly one space in all of mathematics: H⁴"*, resting on H² = 0. **That is false** — H² has infinitely many. The defensible statement is: **among spaces of dimension ≥ 3, H⁴ is the only one.** This was offered as the sharpest "4D is exceptional" line for canon and would have been a real error.
2. **"Everything peaks at 4" is false.** The *compact* count peaks at 4; the *paracompact* count **dips** at 4 (11 → 2 → 5). Two distinct cutoffs: compact stops after H⁴, paracompact after H⁵ (Lännér groups exist only to rank 5). This is Coxeter's 1954 classification — **do not attribute it to Vinberg**, whose dimension-29 theorem is a different statement.

**Spherical:** the regular tessellations of S³ are exactly the six convex regular 4-polytopes,
radially projected. **THEOREM** — and it is what makes stereographic rendering (§3.1) work.

**D₄ triality does NOT cause the 24-cell's self-duality.** Self-duality follows from the
palindrome {3,4,3} alone and is order 2. Triality's genuine order-3 content is the permutation of
the **three disjoint 16-cells inscribed in the 24-cell** (matching 8v/8s/8c). Do not conflate them.

---

## 3. RENDERING A 4D OBJECT

### 3.1 The three projection families — what each preserves and destroys

| method | the map | preserves | destroys | invertible? |
|---|---|---|---|---|
| **Orthographic** | P = I − nnᵀ, rank 3, kernel = span(n) | parallelism, ratios along **parallel** lines, centroids, convexity — all **affine** structure | length, angle, volume | **No** (1-D kernel) |
| **Perspective / Schlegel** | (x,y,z,w) ↦ (d/(d−w))(x,y,z) | the **entire face lattice**, straightness, **cross-ratio** | lengths, angles, volumes | No |
| **Stereographic from S³** | σ(x,w) = x/(1−w) | **angles exactly** (conformal), circles→circles, and it is a **bijection** | nothing, for a polytope inscribed in its circumsphere | **Yes** |
| **Slicing** | {x : x·n = c}, animated in c | **exact Euclidean metric inside each frame** | everything between frames; shows a measure-zero set at each instant | per-frame yes |

A segment of direction u under orthographic projection has image length |u|·cos(angle to the
projection hyperplane). It **genuinely destroys one real number per point**.

**The Schlegel diagram is not a "limit" of perspective** — it *is* an ordinary perspective
projection from a point just beyond exactly one facet. Every cell appears as a 3D polyhedron; one
cell becomes the outer region. It is projective, so it keeps straightness and the whole
combinatorial structure.

**Stereographic is the only route that loses nothing.** Radially project the polytope to its
circumsphere S³, then σ(x,w) = x/(1−w). Edges become **circular arcs**; local magnification is
exactly **1/(1−w)**, which makes apparent size an *exact, invertible* encoding of w. True arc
length is recoverable from the picture via `ds = 2R²|dy|/(R² + |y|²)`. This is the route behind
the Leys/Ghys/Alvarez *Dimensions* renderings of the 120-cell.

**A 4D-only identity found here, no established name:** the n-cube's edge subtends an arc θ on S³
with cos θ = 1 − 2/n and **sin(θ/2) = 1/√n = the facet signature**. The condition
sin(θ/2) = cos θ holds iff 1/√n = 1/2, i.e. **exactly n = 4** — giving the tesseract a **60° edge
arc**. Your n = 4 coincidence resurfacing in a third costume. FRAMING.

**Missing from every standard taxonomy and worth building: double orthogonal projection** — two
coupled 3D views, the 4D analogue of Monge plan-and-elevation. Published and citable
(M. Zamboj, *Nexus Network Journal* 20 (2018) 267–281; arXiv:2003.09236). It keeps edges
**straight**, preserves all affine structure, and — critically — is exactly the jointly-injective
pair of §3.4, so it recovers 4D coordinates *exactly* while stereographic bends every edge into
an arc. For a project that measures what it draws, this is probably the better primary view.

### 3.2 Depth cues for w

Faithful (invertible encodings of w, with a stated map and a legend): **colour/hue**, **size**,
**opacity** under a declared ramp. Merely suggestive: fog, thickness, stereo. The distinction is
the whole game — a cue is faithful only if the viewer can *invert* it and recover w, and hence
recover true length as √(|Pu|² + (Δw)²).

### 3.3 "Through time" — you were half right, and the half that is wrong matters

**Two different methods hide under "show w as time", and their invariants are disjoint:**

| | (i) **Slicing** animated | (ii) **Rotating** animated |
|---|---|---|
| frame t shows | the slice w = t | a fixed projection of a rotated copy |
| you see the whole object? | **never** — a measure-zero set at each instant | **always**, but distorted |
| metric inside a frame | **exact** (a literal isometry) | affine only |
| carries the 4-volume? | **yes**, by Fubini | no |
| blind to | w-shears; the **sign** of w | length, angle, volume |
| its own exact identity | Σ slice volumes = the 4-volume | **Cauchy's projection formula**: mean shadow volume = (2/3π)·hypersurface content |

**The tesseract's main-diagonal slice sequence**, derived from scratch and independently matching
Banchoff's published account (including his "three-eighths of the way through"):

> point → regular tetrahedron → truncated tetrahedra → **Archimedean truncated tetrahedron at
> exactly c = 3/2** (volume 23/24) → **regular octahedron at c = 2** (volume 4/3) → the mirror
> sequence in reverse.

The 4-volume integral closes exactly at 1. Two more, both with surprises: the **16-cell's
vertex-first slices are all regular octahedra**; its cell-first family is **cuboctahedral**, not
truncated-tetrahedral — so **duality does not transport slice families**, even though the
tesseract's vertex-first axis *is* the 16-cell's cell-first axis.

**Is the 4th dimension time? No — and here is the operational test.**

The w of a 4-polytope is a **Euclidean** axis, signature (+,+,+,+); its (x,w) rotations are
**periodic** and preserve x² + w². Minkowski time is (−,+,+,+); its "rotations" are **hyperbolic
boosts** with rapidity, **non-periodic**, preserving x² − c²t².

> **The sharp test:** the Euclidean squash factor is **cos θ** — it reaches 0 and **goes negative**,
> mirroring the object. The Lorentz factor sech φ is **strictly positive forever**.
> **A 4D rotation can produce an enantiomer. A boost never can.**

That is exactly what `isomer4d_MIRROR.html` shows, and I re-derived its cos θ law independently:
a rotation in the (axis, w) plane projects to scale[axis] = cos θ. **CONFIRMS CANON.**

⚠ **Collision to flag, not resolve.** Your lattice canon adopted signature (−,+,+,+) with null =
CFL-saturating. The polytope w is (+,+,+,+). These are **different objects your project uses in
different places**, and the word "4D" now covers both. Worth one canon sentence keeping them apart.

**And a code consequence:** the same motion **under slicing would show nothing at all** — that is
the (i)/(ii) distinction at its most extreme, and it means `isomer4d` implements a **simple**
rotation only. It therefore cannot display the phenomenon unique to 4D (a double/isoclinic
rotation with no fixed direction). That is a code change, not a slider. §3.6.

### 3.4 Measuring length — the obstruction and the three fixes

**THE OBSTRUCTION (THEOREM, rank–nullity — not an engineering limitation).** Any linear
P: ℝ⁴ → ℝ³ has a 1-dimensional kernel. A single orthographic image pins only a **lower bound** on
4D length, |P⁺v|, with **every value in [|P⁺v|, ∞) attained**.

Explicit witness, in a unit tesseract: the edge (1,0,0,0) and the square-face diagonal (1,0,0,1)
have **identical images** and true lengths **1** and **√2 = 1.41421356**.

**THE FIXES — all constructive, and the sweep nearly missed every one:**

1. **Two views.** Two orthographic projections determine the full 4-vector **iff their kernel
   *lines* are distinct**: ker P₁ ∩ ker P₂ = {0}. (The tempting "rank 6 ≥ 4" argument is wrong —
   rank ≤ 4 always.) Reconstruction conditioning is **1/sin²γ** in the angle γ between kernels, so
   near-parallel kernels are numerically useless. This is exactly Zamboj's double orthogonal
   projection.
2. **Rotate.** Any rotating orthographic sequence with vertex correspondence recovers every 4D
   coordinate. A rotating renderer additionally recovers each **true edge length as the maximum of
   the drawn length over one revolution** — a one-line ledger row.
3. **Calibrate the slice parameter.** If the slice coordinate is in the same length unit as space,
   the true 4D distance between a point in frame j and a point in frame k is
   **√(|Δx₃|² + (w_k − w_j)²)** — plain Pythagoras *across frames*. Slicing then recovers the
   **full** Euclidean metric, not just the per-frame one, and the frame rate stops being a free
   parameter and becomes a calibrated instrument. **This is the direct, constructive answer to
   your "through time" guess.**
4. **Stereographic**, as above: true arc length from one image, given the formula.

**The residual ambiguity that survives everything.** For an object in w = 0, a simple (x,w)
rotation gives image scale cos θ, which is **even** in θ — so θ and −θ are byte-identical. This is
your flat-card result lifting verbatim ("cos is even"), and `isomer4d` exhibits it. More generally
**4D structure from orthographic motion is determined only up to reflection** (Ge & D'Zmura, SPIE
5016, 2003). **ESTABLISHED.**

> **The blunt ledger line:** a 4D renderer can faithfully display **combinatorics**, **affine
> ratios along parallel lines**, and (stereographically) **angles**. From a single still it can
> **never** display length. With two distinct kernels, or motion plus correspondence, it can
> display length **exactly** — up to one global reflection.

### 3.5 Can you tell a regular object is regular?

**Split the reading; the answers are opposite.**

**COMBINATORIAL regularity (flag-transitivity): YES**, fully readable off **one Schlegel diagram**,
which preserves the entire face lattice. Count cells, check every cell is the same combinatorial
polyhedron, check vertex figures.

**METRIC regularity from a single projection: NO, provably.** The counterexample is devastating
and it is your own tesseract:

> Project the tesseract along its main diagonal. **All 32 edges draw at exactly equal length √3.**
> (Vertices (±1)⁴, edge vector 2e_i, |P(2e_i)| = 2√(1−1/4) = √3 — every edge, identically.)
> The 16 vertices produce **15 distinct image points**: 1 at the centre (carrying the antipodal
> pair ±(1,1,1,1)), 8 at radius √3 = 1.73205081, 6 at radius 2. **14 of them lie on the hull, and
> the hull is a rhombic dodecahedron — which is not regular.**

Equal-looking edges, and a non-regular solid. That is the answer to your question in one picture.

There is also an explicit look-alike *family*: the 4-parameter shear v ↦ v + (⟨a,v⟩ + c)w preserves
the image of any convex polytope, and for simplicial cases a larger (V−1)-dimensional vertex-slide
family works too. So the failure is not a near-miss; it is a positive-dimensional family.

**Worse — equal edges are not enough even with perfect 4D coordinates.** The **rectified 5-cell**
and the **grand antiprism** (100 vertices, all 500 edges equal, vertex-transitive, inscribed in a
sphere) both pass "all edges equal" and are **not regular**. Any ledger row that grades regularity
on edge equality alone is vacuous, and both of those belong in it as FAIL rows.

**THE CERTIFICATE, and the test to build first.** Regularity *is* flag-transitivity. A convex
polytope's isometry group acts **freely** on flags, so:

> **|G| = F  ⟺  the action is simply transitive  ⟺  the polytope is regular.**

F = 120, 384, 384, 1152, 14400, 14400 for the six. Measure |G| by the **symmetry-snap test**:
rotate and check whether the rendered image returns **frame-identical** to itself. This is purely
visual, and the snap angles **name which polytope you are looking at**:

| polytope | Coxeter number h | snap angle 360°/h |
|---|---|---|
| 5-cell | 5 | 72° |
| tesseract / 16-cell | 8 | 45° |
| 24-cell | 12 | 30° |
| 120-cell / 600-cell | 30 | 12° |

And it plugs straight into your existing quaternion machinery, because of §3.6.

**Verdict, blunt, as asked:** **You cannot tell from a still. You can tell from motion, and the
test is cheap.** Build the snap test first.

**Human perception — what is actually known.** Ambinder, Wang, Crowell, Francis & Brinkmann,
"Human four-dimensional spatial intuition in virtual reality", *Psychonomic Bulletin & Review*
16(5) (2009) 818–823 — **the citation is real**; it found trained observers can make length and
angle judgements in 4D VR above chance. Also Wang, *Spatial Cognition & Computation* 14(2) (2014)
91–113 (hyper-volume); He et al., *Frontiers in Psychology* 14 (2023) 1180561 (rigidity
discrimination); Miwa, Sakai & Hashimoto, *IEEE TCDS* 10 (2018) 250–266 (learning 4D
representations). **Do not overstate these:** they cover *learned judgements on simple stimuli
under projection*, not "perceiving a 4-polytope as a whole". **Whether a human can integrate a
slicing animation well enough to judge anything metric: no study found — OPEN.**

### 3.6 What a 4D renderer needs to know about rotation

- **Rotation happens in a PLANE, not about an axis.** "Axis of rotation" is a 3D-only coincidence (the orthogonal complement of a plane in ℝ³ is a line). dim SO(n) = n(n−1)/2: **1, 3, 6** at n = 2, 3, 4.
- **Every rotation of ℝ⁴ has two invariant orthogonal planes with angles (θ₁, θ₂)** — a **double rotation**. Simple = one angle zero. **Isoclinic (Clifford)** = |θ₁| = |θ₂|, left or right. **THEOREM.**
- **A generic 4D rotation has no fixed direction at all** (fixed-subspace dimension is 0, 2 or 4 — never 1 or 3). Odd-dimensional rotation matrices *must* have eigenvalue +1; even-dimensional need not. This is why 4D rotation looks like nothing in 3D: the object appears to turn inside out.
- **SO(4) = (S³ × S³)/{±1}**, x ↦ q x r. **Spin(4) = SU(2) × SU(2)**, and **SO(4) is the only non-simple SO(n) for n > 2** — another genuinely-rank-4 exceptionality.
- **⚠ The Cl(3,0) habit breaks.** In Cl(4,0) there are **six** bivectors, and a generic one is **not a blade** (the Plücker condition b₁₂b₃₄ − b₁₃b₂₄ + b₁₄b₂₃ = 0 cuts out the blades). A unit non-simple bivector **does not square to −1**: ((e₁₂+e₃₄)/√2)² = −1 + e₁₂₃₄, a **zero divisor** (Z² = −2Z). What replaces "three bivectors replace one i" is: **six bivectors, splitting 3+3 under the Hodge star** (⋆² = +1 on Λ²(ℝ⁴) — a dimension-4-only fact), the two triples being the left and right 𝔰𝔲(2) of Spin(4).
- Left-multiplication by q = cos α + i sin α rotates the (1,i) plane by +α **and** the (j,k) plane by +α — like signs, hence **left-isoclinic**, with rotation angle **α, not 2α** (no half-angle doubling, unlike 3D conjugation).

**The killer application — and it is stronger than your canon states:**

> The vertex sets of the **16-cell (Q₈, order 8)**, the **24-cell (2T = the 24 Hurwitz units,
> order 24)** and the **600-cell (2I, the icosian group, order 120)** are **groups** under
> quaternion multiplication at unit-norm normalisation. Therefore **left-multiplication by any
> vertex is an isoclinic rotation that permutes the vertices** — the polytope rotates onto itself.

The Hopf ring decompositions are then just **Lagrange's theorem**: 24/6 = 4 great hexagons,
120/10 = 12 great decagons, 16-cell 8/4 = 2 great squares. This reproduces the published fibration
counts exactly, and it is the mechanism behind the snap test of §3.5.

**Two normalisation traps, both real:**
- The D₄-root coordinates (±1,±1,0,0) are also a 24-cell but are **NOT closed** under quaternion multiplication — normalised, they are the coset 2O \ 2T.
- The **tesseract's 16 vertices are not a group** (16 ∤ 24), and the 120-cell's 600 are not either.

---

## 4. BRIDGES TO EXISTING ZRL CANON — accepted and rejected

### 4.1 The "three appearances of 1/2" — collapses to ONE fact, and the mechanism runs to the 24-cell

You had: C_max = 1/√D = 1/2 at D = 4; the tesseract's facet signature 1/2; cos(π/3) = 1/2.

**(i) Two of the three are the same fact.** The CFL cone ratio is the **cross-polytope's** r/R and
the flavor signature is the **hypercube's** r/R; they are **dual**, and dual regular polytopes share
r/R. So it is one quantity in two suits — **THEOREM**, exactly as your hint suspected.

**(ii) The 1/2 is load-bearing, and its mechanism is the 24-cell, not the triangle.**
1/√D = 1/2 at D = 4 is *equivalent* to **√D/2 = 1**: the deep-hole distance of ℤ⁴ equals its
minimal distance. That single equation forces the 24 minimal vectors, forces D₄'s Voronoi cell to
be a 24-cell, and — via e₁·(1,1,1,1)/2 = 1/2 — forces the 24-cell's dichoral angle to be exactly
**120°**, hence three around a ridge, hence the honeycomb {3,4,3,3}. **ESTABLISHED.**

**(iii) The triangle bridge is RULED OUT — by mechanism, not by taste.** The structurally correct
2D analogue of the D = 4 story is the **hexagon** (Voronoi cell of A₂, R = edge, 120°,
three-around), whose signature is cos(π/6) = 0.866025 — **the wrong number**. So the cos-shaped
appearance of 1/2 is **1/√D wearing a cosine**, not the triangle's apothem ratio.

**This strengthens your 17/09 honest framing rather than weakening it.** Niven explains *where* the
two ladders meet; it is still **not** evidence that the CFL condition knows about flavors.

**The decisive experiment nobody has proposed:** change the **lattice**, not the dimension. Run the
leapfrog on **D₄** in 4D and see whether C_max is governed by the 24-cell's r/R = 1/√2 rather than
by 1/2. That would settle the question. (Do not run it on my say-so — it is a real build.)

### 4.2 Two confirmed canon errors — see §5. Both found by reading the files, both real.

### 4.3 The C₂³ phase group lifts cleanly; the carrier does not

**C₂⁴ works:** order 16, 4 bits, 16 orthants, simply transitive, abelian. The full hyperoctahedral
group is B₄ of order 2⁴·4! = **384**. Clean lift.

### 4.4 Isomers in 4D

The general theorem behind `isomer4d`: **any orientation-reversing map of ℝⁿ is realised by a
rotation in ℝ^(n+1)** — that is why a chiral 3D object reaches its enantiomer through 4D.
**THEOREM.**

But **"isomer = (axis class, sign)" does not lift.** The sign half survives in every dimension
(det = ±1); the **axis half does not**, because a generic element of SO(4) has no fixed direction
(§3.6). The proposed substitute: **the unordered pair of invariant 2-planes plus two angles**,
equivalently the quaternion pair (q, r) mod simultaneous sign. **OPEN — do not resolve silently.**

And a fact that decides where to look first: **all six regular 4-polytopes are achiral** (each
symmetry group is generated by reflections, index exactly 2). The first 4D flavor where the isomer
concept has anything at all to record is the **snub 24-cell**.

### 4.5 ⚠ THE CARRIER DOES NOT LIFT — the roadmap blocker

Spec §4.1 defines the carrier as **the boundary loop of the fundamental octant patch**, and that
definition is what lets Stage 3's pathfinder lift verbatim.

> **The boundary of the fundamental orthant patch is S^(D−2).** It is a **loop only at D = 3.**

At D = 4 it is a **2-sphere** (area 2πR² on the unit S³; area 3 on the edge-1 tesseract). Its
1-skeleton is **K₄**, all four vertices of degree 3 — **all odd**, so there is **no closed walk
covering every arc once** (no Eulerian circuit; a postman walk repeats 2 arcs, a Hamiltonian cycle
loses 2 of the 6). Worse, **π₁(S²) = 0**, which destroys the winding invariant the 3D carrier
carries.

That is precisely the case **spec §4.1 already rejected** ("the surface patch itself… Stage 3's
pathfinder does not lift"). In 4D the rejected case is **forced**.

Three repair options, all costly, none free: (a) the K₄ 1-skeleton with a chosen traversal
convention; (b) accept the 2-surface and re-found the metric on areas/geodesics; (c) restrict to a
coordinate 3-flat — which breaks B₄-equivariance and throws away the 4th bit. **Your call, and it
is a spec decision, not an implementation detail.**

### 4.6 ⚠ Two different families of "4 is special" — do not merge them

The canon block as written lists them together, which invites exactly the failure mode your
protocol exists to prevent.

| **Genuinely rank-4** | **Merely even** |
|---|---|
| F₄ and H₄ exist only at rank 4 | the Huygens wake (odd/even space dimension) |
| SO(4) is the only non-simple SO(n), n > 2 | V − E + F − C = 0 (even dims give 0, odd give 2) |
| √D/2 = 1 at D = 4; the 24-cell; D₄ | −I being a rotation |
| the 10 star polychora; ⋆² = +1 on Λ²(ℝ⁴) | |
| cross-polytope tiles at n = 2 and n = 4 | |

**Huygens is mis-binned in canon.** "4D behaves like 2D, not like 3D" is a **parity** fact about
the fundamental solution of the d'Alembertian (strong Huygens holds for odd space dimension n ≥ 3).
It is **not false** — but it does not belong beside the genuinely-rank-4 items, and see §5 for a
problem with its stated evidence.

### 4.7 Rejected bridges — logged as rejections

| bridge considered | verdict |
|---|---|
| triangle ↔ D = 4 via cos(π/3) | **RULED OUT** — the correct 2D analogue is the hexagon, wrong number (§4.1) |
| D₄ triality causes the 24-cell's self-duality | **RULED OUT** — self-duality is order 2 from the palindrome {3,4,3}; triality is order 3 and permutes the three inscribed 16-cells |
| Huygens 4D-wake ↔ the {2,4} tiling/Niven pattern | **RULED OUT** — even/odd, not {2,4} |
| signature vector extends to the star polychora | **CATEGORY ERROR** — no well-defined inradius (§2.5) |
| "duals share exactly one signature component" | **RULED OUT** — false for self-dual polytopes; the real law is §2.3 |
| D_n\* density-doubling as evidence for D = 4 | **OVERCLAIMED** — D_n\* doubles ℤⁿ's density for every n ≥ 4; it needs only √n/2 ≥ 1 |

---

## 5. CANON ERRATA — confirmed against the files, **not applied**

I have not edited your canon. Both errors below I verified by reading the exact lines. Say the
word and I will apply them with the spans marked and signed.

**E1 — `FlavorRenderersCLAUDE\ZRL_ND_flavor_renderer_spec_MIRROR.md:99` (and repeated in the
ruled-out table at :190). CONFIRMED WRONG.**

> Currently: *"**Caveat:** no 3D signature is cos(π/n) for any integer n (1/√3 → n ≈ 3.29)."*

Unscoped — and **line 75 of the same file**, 24 lines above, prints
`octahedron | ₁R/₀R = √2/2 = 0.707107`, which **is cos(π/4) exactly**. The counterexample sits
inside the very table the caveat is attached to, in a document whose entire thesis is that the
signature is a **vector**.

> **Proposed replacement:** *"no 3D **facet** signature ₂R/₀R is cos(π/n) for integer n (1/3,
> 1/√3, 0.794654 — none). The octahedron's **edge** signature ₁R/₀R is exactly cos(π/4) — an
> instance of the dimension-free fact that the n-cross-polytope has ₁R/₀R = 1/√2 for **every**
> n ≥ 2."*

**Scope note that matters:** `ZRLnFlavors$zRBridge.md:60` and `cflsqrtdlatticecone.md:51` say the
same thing but are **correctly scoped to r/R** (inradius/circumradius) and need **no change**. Only
the spec file is wrong. One reviewer checked those two files, found them fine, and wrongly
dismissed the whole finding — worth knowing about the sweep's own failure mode.

**E2 — `ZRL md Context Files\ZRLnFlavors$zRBridge.md:136`. CONFIRMED WRONG.**

> Currently: *"…plus the **24 elements (±1±e_a±e_b)/√2**"*

Those quaternions have norm **√(3/2) = 1.224745** — **not units**. The line's own parenthetical
gloss two words later, *"(the sign-permuted units of the form (±1±i)/√2 etc.)"*, is **correct**;
the displayed formula contradicts it.

> **Correct form:** **(±e_a ± e_b)/√2** with e₀ = 1, e₁ = i, e₂ = j, e₃ = k, a < b.
> C(4,2) = 6 pairs × 4 sign choices = 24, and 24 + 24 = 48 = |2O| ✓.

**E3 — same line, naming gap (not an error).** The line writes 2O as "the 24 Hurwitz units plus
24 more", which is a correct decomposition, but never names the 24 units as the **binary
tetrahedral group 2T ≅ SL(2,3)**, an index-2 subgroup of 2O. **It is 2T, not 2O, whose 24 elements
are the 24-cell's vertices.** Two agents in this sweep conflated them because of that gap. Add the
order collisions explicitly: |B₃| = 48 = |2O| but B₃ ≅ S₄ × C₂ ≇ 2O (2O has exactly one involution,
−1; S₄ × C₂ has many); |O| = |S₄| = 24 = |2T| but S₄ ≇ SL(2,3). **Any ledger keyed on group order
alone will silently confuse them.**

**E4 — `ZRLnFlavors$zRBridge.md:32`, two refinements in one sentence.**
- *"the 24-cell … is the vertex figure that makes the two extra 4-D honeycombs possible"* — it is the **vertex figure of {3,3,4,3}** and the **cell of {3,4,3,3}** (whose vertex figure is the tesseract). Four independent agents flagged this wording.
- *"It is the reason both counts (polytopes: 6; honeycombs: 3) bulge at D = 4."* True for honeycombs (1 → 3 is entirely F₄). For polytopes it is **reading-dependent**: 6 = 3 generic + 1 (F₄, the 24-cell) + 2 (H₄, the 120- and 600-cell), so deleting the 24-cell leaves **5, not 3**. It is true only of the **5 → 6 increment over 3D**, because H₃ and H₄ each contribute exactly 2. Canon should name **two** exceptional rank-4 groups doing two jobs: **F₄** drives the honeycomb bulge and the 3D→4D increment; **H₄** is solely responsible for all 10 star polychora, all 4 H⁴ star honeycombs and all 5 compact regular H⁴ honeycombs, and no Euclidean honeycomb is H₄-related.

**E5 — the Niven line should be replaced by the stronger theorem of §2.4.**

**E6 — CFL boundary case, unstated and untested.** "Stable iff Co ≤ 1/√D" holds in the von Neumann
|A| ≤ 1 sense but **not** in the uniformly-bounded sense. At **Co = 1/√D exactly**, the worst mode
(k_j h = π for all j) gives A + 1/A = −2 — a **repeated root A = −1 and a defective amplification
matrix**, so solutions grow **linearly** in step count. Many texts state the leapfrog condition as a
**strict** inequality. D = 4 puts that boundary at exactly **0.5**, a tempting round default to
ship, and the in-project 20⁴ run tested **0.49 and 0.51 and never probed 0.50**. The bound and the
1/2 are right; qualify "stable" at equality as **marginally stable**.

**E7 — the Huygens evidence is under-controlled** (the claim itself is fine). A wake on a 70⁴ grid
does not by itself separate continuum Huygens failure from **scheme dispersion**: the standard
(2D+1)-point leapfrog is dispersive for every D ≥ 2 at every Courant number and leaves a trailing
tail **in 3D too**. Missing and cheap: a **3D control run on the same code** plus a dx-halving
refinement study.

**E8 — scope warning on the flavor axiom.** Canon's operative reading is "the cube is the unique 3D
flavor" = the unique regular honeycomb. Under that same reading **4D has three flavors, not one**,
and their cells have **different** facet signatures (1/2, 1/2, √2/2). So say **"the 4D *hypercube*
flavor's facet signature is C_max"**, never "the 4D facet signature is C_max". Also worth a line:
two of the three 4D honeycombs are lattice tilings (ℤ⁴ and D₄) and one is not — the 16-cell has
**tetrahedral** facets, which are not centrally symmetric, so by Minkowski/Venkov–McMullen it is
**not a parallelotope** and {3,3,4,3} needs **two cell orientations**.

**E9 — NOT a canon error, logged so it is not "found" again.** One agent reported that canon says
"A₄ 120/240" and that this is wrong. The arithmetic is right (|W(A₄)| = 120 full, 60 rotational),
but **the string does not exist in any file under `FormConstantsCLAUDE`** — it was grepped for.
Record the correct pairs (A₄ 60/120, B₄ 192/384, F₄ 576/1152, H₄ 7200/14400) as **new** material,
with the independent check |W| = product of the degrees: 2·3·4·5, 2·4·6·8, 2·6·8·12, 2·12·20·30.

---

## 6. DO NOT — tested, failed, or ruled out

| claim | why |
|---|---|
| "the signature vector extends to the star polychora" | self-intersecting cells have no inradius — CATEGORY ERROR (§2.5) |
| "duals share exactly one signature component" | false for self-dual polytopes; use s_k(P\*) = s_{n−1}/s_{n−1−k}(P) (§2.3) |
| "regular star honeycombs exist in exactly one space" | H² has infinitely many; scope it to dimension ≥ 3 (§2.6) |
| "everything peaks at dimension 4" | the paracompact hyperbolic count **dips** at 4 (§2.6) |
| "there are exactly 143 uniform 4-honeycombs" | 143 **known**; completeness not proved (§1.3) |
| 5D parallelotopes = 103769 or 179372 | superseded; 110244 combinatorial / 181394 contraction (§1.2) |
| "equal edge lengths ⇒ regular" | rectified 5-cell and grand antiprism both pass and are not regular (§3.5) |
| "the tesseract's vertex-first shadow has 14 image points" | **15** distinct images, 14 on the hull (§3.5) |
| "the Schlegel diagram is the limit of perspective projection" | it *is* a perspective projection, from beyond one facet (§3.1) |
| "a unit bivector in Cl(4,0) squares to −1" | false for non-simple bivectors — zero divisors (§3.6) |
| "the D₄ root coordinates (±1,±1,0,0) form a quaternion group" | not closed; that set is the coset 2O \ 2T (§3.6) |
| "triality causes the 24-cell's self-duality" | order 3 vs order 2 — different statements (§2.6) |
| the 16-cell shadow-brightness range [1, 4/3] | **wrong** — b((1,1,0,0)/√2) = 2√2/3 = 0.942809 < 1, so anisotropy ≥ √2. Tag OPEN, do not ship the row (§8) |
| attributing the hyperbolic cutoff to Vinberg | it is Coxeter 1954; Vinberg's dim-29 theorem is different (§2.6) |
| "4D Kelvin is not a meaningful problem" | it is sharply posed and **OPEN** (§1.5) |

---

## 7. OPEN PROBLEMS

**In the literature:** densest non-lattice packing in ℝ⁴ · the 24-cell conjecture · uniqueness of
the optimal 4D kissing configuration · completeness of the 143 uniform honeycombs · the 4D Kelvin
and quantizer problems · strongly aperiodic monotile in E³/E⁴ · minimum dimension for an aperiodic
translational monotile · Voronoi's conjecture in dimension ≥ 6 · classification of convex bodies
tiling E⁴ (finite by theorem, never enumerated).

**In-project, and these are the ones that block things:**
1. **Does the carrier lift?** As specified, **no** (§4.5). Three repair options, all costly. **Spec decision required.**
2. **What replaces "isomer = (axis class, sign)" in 4D?** (§4.4) Untested; and all six regular 4-polytopes are achiral, so the snub 24-cell is where to look.
3. **Does the signature extend to the star sector at all?** (§2.5) Nobody computed a single star signature; density is the candidate replacement invariant.
4. **Is there any mechanism linking 1/√D to the 24-cell beyond the deep-hole restatement?** The decisive test is the D₄-lattice leapfrog run (§4.1).
5. **Can a human integrate a slicing animation well enough to judge anything metric?** No study exists (§3.5).
6. **Nets of the 24-, 120- and 600-cell.** Canon has 261 for the tesseract (cube: 11); the hypercube sequence continues 9694, 502110, 33064966 (OEIS A091159). Whether counts exist for the non-hypercube regular 4-polytopes **could not be established — tag OPEN, do not assume by analogy.**
7. **Global extrema of the shadow-brightness functions for the 16-cell and 24-cell** — a finite computation nobody has run, and one published figure is known wrong.
8. **External anchor for four numbers.** The 120-cell and 600-cell ₁R and ₂R rest on in-house derivation plus the duality identity — a real but **closed** verification loop. Coxeter, *Regular Polytopes* 3rd ed., **Table I(ii), pp. 292–293** tabulates ₀R…₃R for all sixteen regular 4-polytopes and was never consulted. **Check those four entries against it before they enter canon.**

---

## 8. LEDGER DESIGN FOR A 4D RENDERER

House rule: every row measured **from what is drawn**, never recomputed from the same formula.

| # | check | expected | notes |
|---|---|---|---|
| 1 | V − E + F − C on the built mesh | **0** | exact; the 4D Euler relation (§2.1) |
| 2 | cell / face / edge / vertex counts | per §2.1 table | exact |
| 3 | all edge lengths equal **in ℝ⁴** | 1e-12 | necessary, **not sufficient** — see row 12 |
| 4 | signature vector recomputed from the built vertex set | §2.2 table | 1e-9 |
| 5 | duality law s_k(P\*) = s₃/s₃₋ₖ(P) across the two dual meshes | exact | cross-mesh row |
| 6 | self-duality s₁·s₂ = s₃ (5-cell, 24-cell) | exact | free row (§2.3) |
| 7 | s₁ = cos(δ_edge/2) against the measured dichoral angle of the dual | 1e-9 | the organising identity (§2.2) |
| 8 | tesseract slice at \|w\| < 1/2 (edge 1) | **exactly a unit cube, volume 1** | pin the normalisation in the row |
| 9 | Σ slice volumes × Δw → 4-volume | tesseract 1; 5-cell √5/96 = 0.0232923748 | Fubini; convergence order stated |
| 10 | V₄ = (1/4)·₃R·S₃ against the meshed 4-volume | see below | independent route |
| 11 | vertex-first tesseract shadow | **15 distinct image points; 14 on the hull; radii √3 and 2; all 32 edges draw at √3** | the §3.5 row |
| 12 | **FAIL rows**: rectified 5-cell, grand antiprism | must report NOT regular despite equal edges | keeps row 3 honest |
| 13 | isoclinic snap-back: left-multiply vertices by a vertex quaternion | image frame-identical; 16-cell/24-cell/600-cell only | the regularity test (§3.5) |
| 14 | snap angle → polytope identification | 72° / 45° / 30° / 12° | §3.5 table |
| 15 | stereographic edge arcs all equal | per §2.2 edge-arc list | conformality row |
| 16 | Cauchy mean shadow = (2/3π)·S₃ | tesseract **16/(3π) = 1.697652726** | sanity: unit 4-ball gives 4π/3 = 4.18879020 = the 3-ball volume |
| 17 | two-view reconstruction residual | 0 when kernel lines distinct; report conditioning 1/sin²γ | §3.4 |
| 18 | homogeneity: V₄(2R)/V₄(R), S₃(2R)/S₃(R) | **16, 8** | the standing check, lifted |

**Surface 3-volumes at edge 1** (never assembled anywhere, and they give a free independent check
on every 4-volume via V₄ = (1/4)·₃R·S₃):
5-cell 5√2/12 = 0.589255651 · tesseract 8 · 16-cell 4√2/3 = 1.885618083 ·
24-cell 8√2 = 11.313708499 · 600-cell 50√2 = 70.710678119 · 120-cell 450 + 210√5 = 919.574275.

**Corrected numbers for rows that were wrong in the sweep** — these are the ones that would have
made a *correct* renderer fail:
- 120-cell mean shadow **195.139722** (not 195.1290); 24-cell **2.40084351**; 600-cell 15.005272.
- 120-cell 4-volume at unit circumradius = **15√5/8 = 4.192627458**.
- 5-cell 4-volume √5/96 = **0.0232923748**.
- shared 120/600-cell s₃ = **0.925614793**; 120-cell s₂ = **0.9732489895**.
- **Do not ship the 16-cell brightness row at all** — its stated range is wrong and the true extrema are unknown (§7.7).

**Precision budget — adopt one.** Nine decimals were printed throughout the sweep while hand
arccos/arcsin work is good to about 1e-6; at least eight entries were wrong in the 7th–9th
significant figure. **Every row must state which digits come from a closed form and which were
evaluated numerically, and tolerances must be relative**, across a range spanning 0.0232923748 to
787.856981.

---

## 9. ATTRIBUTION AUDIT

- **Schläfli**, *Theorie der vielfachen Kontinuität* (written 1850–52, published 1901) — the six, and the criterion. **Poincaré** (1893, 1899) — the Euler–Poincaré relation. STANDARD.
- **Coxeter**, *Regular Polytopes*, 3rd ed. Dover 1973 — Table I(i) f-vectors, **Table I(ii) pp. 292–293 the ₖR radii**, and the 1954 hyperbolic classification. STANDARD. **Not yet consulted — see §7.8.**
- **Hess** (1883) — the 10 star polychora. STANDARD.
- **Delone** (1929) + **Štogrin** — the 52 parallelotopes. **Dutour Sikirić, Garber, Schürmann, Waldmann**, *Acta Cryst.* A72 (2016) 673–683 — the corrected 5D counts. **Garber** (2025) — Voronoi's conjecture through dim 5. STANDARD.
- **Brown, Bülow, Neubüser, Wondratschek, Zassenhaus** (Wiley 1978) — 4D crystallographic groups. STANDARD.
- **Korkine & Zolotareff** (1872) — D₄ lattice-optimal. **Musin**, *Ann. Math.* 168 (2008) 1–32 — τ₄ = 24. **Cohn, de Laat, Salmon**, arXiv:2206.15373 — the 0.636108 bound. **Conway & Sloane**, *SPLAG* — the 24-cell conjecture. **Delone & Ryshkov** (1963) — A₄\* covering. STANDARD.
- **Olshevsky** (2006) — the 143. SELF-PUBLISHED, completeness unproved. Cite as such.
- **Goodman-Strauss**, *Eur. J. Combin.* 20 (1999) 385–395 — aperiodic pairs in every n ≥ 3. **Greenfeld & Tao** — periodic tiling conjecture. STANDARD.
- **Banchoff**, *Beyond the Third Dimension* (1990); **Abbott**, *Flatland* (1884); **Hinton** — the slicing tradition. STANDARD.
- **Zamboj**, *Nexus Network Journal* 20 (2018) 267–281; arXiv:2003.09236 — double orthogonal projection. STANDARD, and **under-known** — this is the method your project most wants.
- **Ge & D'Zmura**, SPIE 5016 (2003) — 4D structure from motion up to reflection. STANDARD.
- **Ambinder, Wang, Crowell, Francis & Brinkmann**, *Psychonomic Bulletin & Review* 16(5) (2009) 818–823 — **citation confirmed real.** Also Wang 2014, He et al. 2023, Miwa–Sakai–Hashimoto 2018. STANDARD, and **do not overstate** (§3.5).
- **Prior art:** projection camp — Jenn3d, Stella4D, Magic Cube 4D, Bathsheba Grossman, the *Dimensions* film (Leys/Ghys/Alvarez). Slicing camp — Miegakure and 4D Toys (Marc ten Bosch), 4D Golf (CodeParade), 4D Miner. **Games use slicing almost universally**, because a 3D slice is an ordinary 3D scene the z-buffer already handles; projection needs wireframe or transparency because projected cells overlap. **No existing 4D project has a live PASS/FAIL identity ledger** — that is the actual gap, and 4D is the dimension where it matters most, precisely because you cannot see regularity.
- **Original to this project as far as searched:** the ₖR signature vector as a shape invariant (already claimed in the N-D spec) and its 4D values; the "whole vector is cos(π/m) iff n ∈ {2,4}" packaging (**the arithmetic is a theorem; the packaging is ours and has no established name**); the signature-as-shadow-shrink-factor reading below; the ledger design of §8.

**One result that makes the signature *measurable* rather than computed** — worth its own line:

> **The hypercube's signature vector IS the vector of orthographic k-volume shrink factors under
> main-diagonal projection.** At k = 1, n = 3 that is your existing √(2/3) [111] edge factor; at
> k = 1, n = 4 it is √3/2, which is exactly why all 32 tesseract edges draw at √3 in §3.5.
> So the signature can be **read off a rendered shadow** instead of recomputed — the house
> "measure what you draw" rule, satisfied for free.

---

## 10. VERIFICATION STATUS — what was and was not checked

**Was:** every claim passed through three independent adversarial lenses at a 2-of-3 refutation
threshold, then two critics (completeness, numbers). The numbers critic re-derived by hand:
V − E + F − C on all six f-vectors ✓; all four group orders and their full-vs-rotational labelling
in all eight places they appear ✓; strict monotonicity 0 < s₃ < s₂ < s₁ < 1 on all six ✓; the
duality law on both dual pairs ✓; the self-duality identity on both self-dual cases ✓;
1/√4 = 1/2 = cos(π/3) ✓. I then re-derived the duality law, the s₁ = cos(δ_edge/2) identity, the
shadow shrink factors and the §2.2 decimals a third time while writing this file, and read both
canon-error lines in the files directly.

**Oracle run — 21/09/2026, by the author. 84 passed, 1 failed.** The failure was
`test_kelvin_candidates_normalised`, and it was **this document's error, not the oracle's**: the
16-cell Kelvin quotient was quoted as 7.228760 against a computed 7.228816028878759. Hand-checked
independently — q⁴ = 2¹³/3, so q = 2^(13/4)/3^(1/4) = 7.228816029. The doc is corrected (§1.5) and
the test now asserts the closed form rather than a decimal.

**What that one failure tells you, and it is the useful part:** the number came from the research
sweep, and it was caught by the *only* test in the file that rebuilt the quantity from its parts
(S₃ and V₄) instead of comparing a printed decimal to itself. Every row that merely re-checks a
printed decimal against its own closed form passed — as it must, and as it would have with a wrong
closed form. **The A₄\* permutohedron figure 6.831200 came from the same list and is now flagged
UNVERIFIED**, because it is the one candidate I could not rebuild from independent S₃ and V₄ and
so deliberately left out of the oracle. Do not quote it until it is derived.

**Was not:**
- **No code was executed by me.** Not python, not pytest, not node — your standing rule. The author ran the oracle.
- **Coxeter Table I(ii) was never consulted.** The 120-cell and 600-cell ₁R and ₂R rest on in-house derivation plus the duality identity — self-consistent, but a **closed loop** with no external anchor beyond ₀R and ₃R. §7.8.
- **One verifier died** (the `regularity-detection` derivation lens hit the 64k output ceiling), so that dimension was adjudicated on **two** lenses, not three. Its claims — the |G| = F certificate, the snap angles, the look-alike family — were independently corroborated by the completeness critic, but they carry **one less vote** than everything else.
- **Decimals:** anything printed to 9 places that came from a hand arccos/arcsin is good to ~1e-6, not 1e-9. The §2.2 signature values and the §2.2 angle tables are now **oracle-verified**. Figures that appear only in prose and have no oracle row — the A₄\* quotient, the covering constant, the star-polychora densities — carry the sweep's original precision and nothing more.

---

## 11. NEXT STEPS — proposed, not started

1. **Decide the carrier question (§4.5).** Everything downstream waits on it. It is a spec decision.
2. ~~Run the oracle.~~ **DONE 21/09/2026 — 84 passed, 1 failed; the failure was a doc error, now fixed (§10).** Re-run after the fix to confirm all green: `python -m pytest test_polytope4d_oracle_MIRROR.py -v`. Optional follow-up: derive the A₄\* permutohedron's S₃ and V₄ so the last Kelvin row gets an oracle test too.
3. **Apply the §5 errata** — say the word and I will edit the two files with the spans marked and signed per your Obsidian convention.
4. **Check the four unanchored radii** against Coxeter Table I(ii) pp. 292–293 if you have the book.
5. **Build the snap test first**, not a renderer: it is the only visual regularity certificate, it is cheap, and it reuses the quaternion machinery you already have.
6. **Then double orthogonal projection** (Zamboj) as the primary view — straight edges, exact reconstruction, more ledger rows than stereographic.
7. Optional and high-value: **the D₄-lattice leapfrog run (§4.1)** — the one experiment that would settle whether the CFL number knows about flavors.
