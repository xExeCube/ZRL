# Rigid vs pinned walls, and the edge rule

Research note for the Taichi port of the ZRL 2D membrane. **Analysis and prototypes only**: nothing here is wired into `membrane_core_MIRROR.py` (the user asked to hold off implementing rigid walls). Every number marked *measured* comes from a script in `ResearchCLAUDE/WallsCLAUDE/`, with its full output in `ResearchCLAUDE/WallsCLAUDE/OutCLAUDE/`. Claims marked *literature* were not checked here.

*Generated from scratch — Claude Opus 5.5 — 29/09/2026 (prototypes) and 30/09/2026 (runs, note)*

---

## 0. The short answer

- **Pinned** (u = 0, Dirichlet) is what every wall, solid and mask edge does today. It is right for a **membrane** held down by a post or a glued shape, and for acoustics with a **pressure-release** surface. It reflects with **R = −1** (the pulse comes back inverted).
- **Rigid** (∂u/∂n = 0, Neumann) is right for **sound hitting a hard object** (u = pressure), for **water ripples off a quay** (u = surface height), and for a membrane edge on a **frictionless ring**. It reflects with **R = +1** (the pulse comes back upright). "A wave bouncing off a cube / off text" in the everyday sense is this one.
- **The rule to implement later:** rigid = **link removal** (drop every stencil link that touches a rigid cell), with **corner blocking** for the diagonal links of sq9 and tri. It keeps the operator a symmetric graph Laplacian, so it:
  - conserves the discrete energy exactly (measured drift ≤ 2e-14 over 20 000 steps);
  - never tightens the CFL limit (λ_max ≤ the open-lattice value, measured);
  - leaks nothing through one-cell walls (measured: exactly 0.0).
- **One kernel handles both types:** a per-cell type byte (or a precomputed 16-bit link word) replaces the mask. Pinned links stay (the neighbour holds 0); rigid links are dropped. Measured cost: **+2 % to +20 %** per step for the link word (§3.12).
- **The price of rigid walls is staircase roughness** (§3.3, §3.11):
  - a rotated rigid square scatters **+25–33 %** more than the same square axis-aligned on sq5, +4–27 % on sq9, and within ±10 % on tri;
  - a pinned one stays within ±3 % on every lattice.
  - Up to 117° reflection phase error at grazing incidence on 45° rigid staircases (λ = 8).
- **Two things to fix for pinned walls now, before rigid walls exist** (§4.2, §5.4):
  - *Leak:* a one-cell 8-connected diagonal wall leaks through the sq9 diagonals (**2.3 %** of the energy) and through one tri direction (**17 %**). Text strokes will produce exactly such walls. Corner blocking stops it (measured 0).
  - *Offset:* a pinned wall acts at the **centres of the first solid cells**, half a cell deeper than the drawn edge. So pinned objects act ~1 cell thinner than drawn, and pinned cavities ~1 cell wider.

---

## 1. Physics

### 1.1 What u is decides the wall

| system | u is | wall | condition | reflection at normal incidence |
|---|---|---|---|---|
| drum membrane | transverse displacement | clamped rim, post, glued shape | u = 0 (Dirichlet) | −1 |
| drum membrane | displacement | edge ring free to slide on a frictionless rod | ∂u/∂n = 0 (Neumann) | +1 |
| acoustics | pressure p | sound-hard wall: rigid object in air, a room wall | ∂p/∂n = 0 (normal velocity 0) | +1 |
| acoustics | pressure p | pressure-release surface: water–air seen from the water | p = 0 | −1 |
| shallow water | surface height | vertical wall, quay | ∂η/∂n = 0 | +1 |
| 2D EM, TM (E_z) | E_z | perfect conductor | E_z = 0 | −1 |
| 2D EM, TE (H_z) | H_z | perfect conductor | ∂H_z/∂n = 0 | +1 |

*(Standard physics.)* Both walls reflect **all** the energy: |R| = 1. What differs is the phase.
- **Pinned:** the displacement must vanish at the wall, so the reflected wave cancels the incident one there. R = −1: a crest returns as a trough.
- **Rigid:** the slope must vanish, so the two waves add at the wall. R = +1: a crest returns as a crest, and the wall is a pressure (displacement) antinode, with a 2× peak on contact.

**Which one looks like "a wave bouncing off a cube / text"?** The rigid one. In air, a cube or a raised letter is sound-hard. On water, a block is a vertical wall. Both give an **upright** echo, with an antinode on the face.

The pinned wall is what a **membrane** does around a post. It also makes a **small** object look much bigger to long waves:
- a pinned point scatters a long wave strongly;
- a rigid point is almost invisible to it.

The lattice reproduces this exactly (§3.10):
- **Pinned single cell:** its scattering width *grows* with λ, like λ^0.47. It acts like a Dirichlet disc of radius 0.20 cells at every λ.
- **Rigid single cell:** its scattering width falls like λ^−3.0, the 2D Rayleigh law.

At λ = 32 cells the pinned cell scatters **304×** more energy than the rigid one.

### 1.2 Impedance (Robin) walls in between

*(Standard acoustics; literature.)* A locally reacting surface with normalised impedance ζ = Z/(ρc) reflects a plane wave at angle θ with

R(θ) = (ζ cos θ − 1) / (ζ cos θ + 1).

- ζ → ∞: rigid (R = +1).
- ζ → 0: pressure release (R = −1).
- ζ = 1: absorbs at normal incidence (R = 0). This is exactly Mur's first-order absorbing condition.

In between, the wall absorbs part of the energy. This is the natural third wall type (carpet, foam, a damped rim).

**Measured on axis-aligned walls (§3.4):** a "generalised Mur" ghost update reproduces this R(θ) closely on sq5 and sq9:
- to within 0.002 at λ = 32;
- to within 0.03 at λ = 8.

A simpler "damping on the boundary cell" form gets |R| right to ~1–2 % at λ = 32, but adds a phase error (Im R up to 0.12). On the tri lattice my ghost implementation is wrong (§3.4), so impedance walls on tri are open.

---

## 2. The edge rules

Every stencil here is a **graph Laplacian**: lap_i = Σ_links w_ij (u_j − u_i).

| stencil | links and weights | Co_max |
|---|---|---|
| sq5 | 4 edges, w = 1 | 1/√2 |
| sq9 | 4 edges w = 2/3, 4 diagonals w = 1/6 | √3/2 |
| tri | 6 neighbours, w = 2/3 | √(2/3) |

Cell types: FREE, PIN, RIG. The rules for a link from a free cell i to its neighbour j:

**(i) P — pinned, keep the link.**
- j is pinned: the term is w(0 − u_i). This is today's rule: masked cells are zeroed each step, and the stencil reads the zero.
- The operator stays symmetric.
- The wall acts on the line through the **pinned cell centres**. That is exact for flat walls on every lattice, at every angle (§3.2–3.3): odd symmetry about a lattice row is exact.

**(ii) R0 — rigid by link removal.**
- j is rigid: the term is dropped.
- The operator stays a symmetric graph Laplacian, so leapfrog conserves the discrete energy E = Σ(Δu)²/Co² − ⟨u⁺, L u⟩ exactly.
- For a flat axis-aligned wall, it equals a **ghost mirror with the wall half a cell out**, on the rasterisation line: the neighbour reads u_i itself, and w(u_i − u_i) = 0.
- Measured: exactly R = +1 with zero offset at every λ on sq5 axis walls, at every angle (§3.3).

**(ii′) R1 — link removal + tangential transfer** (sq9 and tri only).
- On sq9 and tri, link removal loses part of the stencil's **tangential** second moment at a flat wall. Measured (`checks_MIRROR.py`): the boundary row's lap of x²/2 is 0.8333 instead of 1.
- R1 gives the dropped weight back to the tangential link along the wall, then symmetrises: W = (T + Tᵀ)/2.
  - It moves τ·w of each dropped link (sq9: a dropped diagonal) onto the in-wall link between the free cell and its free common neighbour, with τ = 1 on sq9 and τ = 1/4 on tri.
  - It only applies when the dropped link's two "common neighbours" are one free and one rigid.
- **On sq9 with a straight rigid ring, R1 is identical to the ghost-copy Neumann ring with corner u(0,0) = u(1,1)** (measured 3.6e-15). That is exactly the corner fix proposed in the README's web-app finding.
- R1 makes sq9 axis walls exact at every angle, and tri row walls nearly so (§3.3). It changes nothing on 45° staircases.

**(iii) RG — ghost mirror along an estimated normal.**
- The link to a solid cell reads u at the mirror image of the solid cell's centre across the true boundary (bilinear interpolation).
- It is **not symmetric** (measured max |L − Lᵀ| = 0.6–0.96), so there is no conserved energy.
- **It blew up** in a disc at Co = 0.5 (measured: max|u| 0.27 → 1.6e3 over 20 000 steps; R0 stayed at 0.2).
- It was also *less* accurate than plain link removal on circle eigenfrequencies (+3.8 % at R = 8, first order).
- Not recommended for an explicit scheme without further stabilisation.

**(iv) Embedded-boundary / cut-cell methods for oblique walls.** *(literature, plus two sq5 prototypes measured here)*
- **Dirichlet:** Gibou, Fedkiw, Cheng & Kang (J. Comput. Phys. 2002) keep the operator symmetric by placing u = 0 at the true crossing point, at link fraction θ: the term becomes −u_i/θ. Second order.
  - Measured here (GB): j01 error 5e-4 → 9e-6 from R = 8 to 64 (≈ second order).
  - But λ_max grows like 1/θ. With θ_min = 0.001, λ_max reached **113× the open-lattice value**, which would cut the explicit time step ~10×.
  - With θ_min = 0.25: λ_max ×1.3 (Co −13 %), error 1e-4 to 1e-3.
  - With θ_min = 0.5: CFL-neutral, but back to first order.
- **Neumann:** finite-volume cut cells (face apertures + volume fractions, symmetric with respect to V).
  - Measured here (CF): j'11 error 3e-4 → 2e-5 from R = 8 to 64.
  - But the small-cell problem gives λ_max up to **12×** (Co ÷3.5).
  - Flooring V at 0.25 costs +9 % λ_max and gives back first order.
  - The literature fixes small cells by **cell merging** or **flux redistribution** (Colella and co-workers' embedded-boundary work, e.g. J. Comput. Phys. 2006).
- For the second-order wave equation specifically: Kreiss, Petersson & Yström (SIAM J. Numer. Anal. 2002 Dirichlet, 2004 Neumann) give stable embedded-boundary schemes.
- In FDTD electromagnetics, staircase error was analysed by Cangellaris & Wright (IEEE TAP 1991), and conformal FDTD (Dey & Mittra 1997) fixes it.
- None of this is needed for the first rigid-wall version. It is the upgrade path if the rigid staircase error (§3.3, §3.11) matters.

**Corner blocking** (sq9 diagonals, tri links).
- A link between two FREE cells is **blocked** when *both* of its common neighbours are solid:
  - sq9: the two edge cells the diagonal passes between;
  - tri: the two cells forming a triangle with the link.
  Such a link crosses a wall through a corner-to-corner gap.
- A blocked link is treated as a link to a pinned cell if either blocker is pinned (−w u_i), and dropped if both are rigid.
- It is symmetric by construction: both ends see the same two blockers.
- Blocking when *either* common neighbour is solid would also be symmetric, but it removes links along every wall face. Not used.

**One kernel for all of them.**

 lap_i = Σ_{links to free j, not blocked} w (u_j − u_i) + Σ_{links to pinned j, or pinned-blocked} w (0 − u_i)

Rigid links and rigid-blocked links are dropped. Because pinned cells hold u = 0, "link to pinned" is just an active link whose neighbour reads 0. Only the **pinned-blocked** diagonals need a separate diagonal-only term. §3.12 measures two encodings.

---

## 3. Measurements

All in numpy/scipy, f64, h = 1, leapfrog. The operators are built as explicit sparse matrices by `walls_lib_MIRROR.py`, so symmetry, spectrum and energy are checked directly.

### 3.1 Sanity (`checks_MIRROR.py` → `checks_out_MIRROR.txt`): 22/22 PASS

- **Rule P reproduces the port's pinned stencils** (zero padding) for sq5, sq9 and tri: max diff 1.8e-15 to 3.6e-15.
- **Every rule is exactly symmetric** (|L − Lᵀ| = 0) on a scene holding every hard case: a fin, a one-cell diagonal wall, a single cell, a 3×3 block, a closed ring with a counter, a concave notch.
- **R1 on a sq9 rigid ring equals the ghost-copy Neumann ring with corner = u(1,1):** 3.6e-15. R0 differs from it by 1.34 on random data. For sq9, link removal alone is *not* the mirror.
- **Boundary-row second moments at a flat wall:**

  | | tangential x²/2 | normal (y − y_w)²/2 | exact |
  |---|---|---|---|
  | sq5 R0 | 1.0000 | 1.0000 | 1 |
  | sq9 R0, tri R0 | **0.8333** | 1.0000 | 1 |
  | sq9 R1, tri R1 | 1.0000 | 1.0000 | 1 |

### 3.2 Reflection at normal incidence (`chain_reflection_MIRROR.py` → `chain_reflection_out_MIRROR.txt`)

**Method.** A straight wall along a lattice vector T is periodic along T, so a plane wave with a fixed tangential Bloch phase reduces **exactly** to a 1D chain. This includes the staircase's fine structure; grating orders, if any, live in the same chain. A narrow-band pulse (2–8 % bandwidth), built from the exact lattice + leapfrog dispersion, is sent in, and the reflection is Fourier-analysed.

**Reported:** δ, the effective wall position beyond the **rasterisation line** (half-way between the last free and the first solid lattice line), in physical cells. Co = 0.5.

| lattice | wall (angle) | P: δ at λ = 64 … 5 | R0: δ at λ = 64 … 5 | R1: δ at λ = 64 … 5 |
|---|---|---|---|---|
| sq5 | axis (0°) | **+0.500** (all λ) | **0.000** (all λ) | — |
| sq5 | (1,1) 45° | +0.354 | **0.000** (all λ) | — |
| sq5 | (2,1) 26.6° | +0.347 … +0.359 | 0.000 … +0.064 | — |
| sq5 | (3,1) 18.4° | +0.363 … +0.386 | 0.001 … +0.164 | — |
| sq9 | axis | +0.500 | 0.000 | 0.000 |
| sq9 | (1,1) 45° | +0.418 | 0.000 … +0.052 | same as R0 |
| sq9 | (3,2) 33.7° | +0.382 … +0.404 | 0.001 … +0.191 | 0.001 … +0.140 |
| tri | row (0°) | +0.433 (= half the row spacing) | 0.000 | 0.000 |
| tri | (1,1) 30° | +0.356 … +0.363 | 0.000 … +0.055 | 0.000 … +0.036 |
| tri | (3,1) 13.9° | +0.353 … +0.380 | 0.000 … +0.168 | 0.000 … +0.110 |

Reading it:
- **|R| = 1 everywhere**, to 0.2 % (the only deviation, |R| 1.0022, is sq5 (3,2) at λ = 5, a pulse-bandwidth effect).
- **Pinned:** the wall sits at the pinned cell centres. That is a **constant, known offset** of about half a cell, nearly independent of λ.
- **Rigid:** the wall sits *on* the rasterisation line at long λ. On staircases the offset grows like **λ^−2**: (3,1) gives 0.002, 0.009, 0.040, 0.164 at λ = 32, 16, 8, 5.

### 3.3 Reflection at oblique incidence (same script, part 2)

Walls that are lattice mirror lines (so the reflected wave is a lattice plane wave).

**Reported:** the phase error in degrees relative to an ideal wall **on the rasterisation line**. A constant position offset δ shows up as 2kδ·cos θ.

| lattice, wall | rule | λ = 32: θ = 0/30/60/80° | λ = 8: θ = 0/30/60/80° |
|---|---|---|---|
| sq5 axis | P | 11.3 / 9.7 / 5.6 / 2.0 | 45.0 / 39.0 / 22.5 / 7.8 |
| sq5 axis | R0 | **0 / 0 / 0 / 0** | **0 / 0 / 0 / 0** |
| sq5 45° | P | 8.0 / 6.9 / 4.0 / 1.4 | 31.8 / 27.6 / 15.9 / 5.5 |
| sq5 45° | R0 | 0 / 2.3 / 11.9 / **42.5** | 0 / 9.1 / 46.6 / **117** |
| sq9 axis | P | 11.3 / 9.7 / 5.6 / 2.0 | 45.0 / 39.0 / 22.5 / 7.8 |
| sq9 axis | R0 | 0 / 1.1 / 5.6 / 20.7 | 0 / 4.2 / 21.9 / 72.2 |
| sq9 axis | R1 | **0 / 0 / 0 / 0** | **0 / 0 / 0 / 0** |
| sq9 45° | P | 9.4 / 8.2 / 4.7 / 1.6 | 37.6 / 32.7 / 19.1 / 6.7 |
| sq9 45° | R0 = R1 | 0.03 / 1.6 / 8.0 / 29.1 | 1.8 / 7.8 / 33.1 / 94.9 |
| tri row | P | 9.7 / 8.4 / 4.9 / 1.7 | 39.0 / 33.8 / 19.5 / 6.8 |
| tri row | R0 | 0 / 0.9 / 4.9 / 18.0 | 0 / 3.7 / 19.5 / 65.9 |
| tri row | R1 | **0 / 0 / 0.01 / 0.04** | 0 / 0.04 / 0.58 / 2.85 |
| tri 30° | R0 | 0.02 / 1.7 / 8.5 / 30.8 | 1.7 / 8.4 / 35.4 / 98.4 |
| tri 30° | R1 | 0.02 / 1.2 / 6.4 / 23.3 | 1.2 / 6.4 / 27.6 / 83.5 |

Reading it:
- **Pinned** behaves as an exact flat wall **shifted to the solid cell centres** at every angle: the error is exactly 2kδ·cos θ.
  - Flat pinned walls are exact.
  - Pinned staircases are *nearly* a shifted flat wall.
  - Correct for it by drawing pinned objects half a cell fatter (or domains half a cell smaller).
- **Rigid on the lattice's own axes:**
  - exact with R0 on sq5;
  - exact with **R1** on sq9, nearly so on tri;
  - R0 alone gives up to 72° at grazing incidence on sq9 and tri at λ = 8.
- **Rigid on 45° / 30° staircases:** the error is angle-dependent, **not** a shift: 29–43° at 80° for λ = 32, and 95–117° for λ = 8. The stair corners act as a corrugated hard surface; R1 cannot fix this. This is the rigid-wall price. It scales roughly like 1/λ (sq5 45° wall at 60°: 11.9° at λ = 32, 46.6° at λ = 8; λ = 16 was not run, ~23° by that scaling), and it is large at grazing incidence.

In the 29/09 output, R1 looked equal to R0 at oblique incidence. That was a bug in the chain's diagonal term: it used the phased row sum, which cancelled every same-level transfer. It was fixed before this rerun, and the 29/09 file was overwritten.

### 3.4 Impedance walls (same script, part 3): axis walls, R vs ζ, θ, λ

Two forms:
- **Z (damping on the boundary cell):** the links to the wall are dropped, and u⁺ = (… + g u⁻)/(1 + g), with g = Co Σ w·(normal link length)/(2ζ).
- **ZM (generalised Mur ghost):** g^{n+1} = i^n + k (i^{n+1} − g^n), with k = (ζCo − 1)/(ζCo + 1).

Selected values (sq5; sq9 is within 0.01 of these at λ = 32):

| ζ | θ | exact | Z at λ = 32 | ZM at λ = 32 | Z at λ = 8 | ZM at λ = 8 |
|---|---|---|---|---|---|---|
| 0.25 | 0° | −0.600 | −0.588 + 0.125i | −0.599 | −0.418 + 0.461i | −0.580 |
| 1 | 0° | 0.000 | +0.005 + 0.049i | +0.002 | +0.084 + 0.180i | +0.030 |
| 1 | 60° | −0.333 | −0.331 + 0.044i | −0.333 | −0.290 + 0.168i | −0.329 |
| 2 | 30° | +0.268 | +0.271 + 0.023i | +0.269 | +0.313 + 0.085i | +0.285 |
| 5 | 60° | +0.429 | +0.430 + 0.008i | +0.429 | +0.450 + 0.030i | +0.433 |

- **ZM** is accurate on sq5 and sq9.
- **Z** has a phase error, because the damping acts at the cell centre, not at the rasterisation line.
- **ZM on tri is wrong as implemented** (e.g. ζ = 5, θ = 60°: −0.19 instead of +0.43). It reads the ghost's neighbour at the same tangential index, but tri rows are offset by half a cell, and the row spacing (√3/2) changes the effective Co. An impedance wall on tri needs its own derivation. **Open.**
- Energy with impedance walls decreases by design; their long-run stability was not measured.

### 3.5 Energy and CFL (`energy_leak_MIRROR.py` → `energy_leak_out_MIRROR.txt`, part A)

**Setup:** the hard-case scene, 96², a Gaussian pulse, Co = 0.95·Co_max, 20 000 steps.

| lattice | P | Pcb | R0nb | R0 | R1 | mixed (left half pinned, right half rigid) |
|---|---|---|---|---|---|---|
| sq5 max \|E − E0\|/E0 | 1.2e-15 | 1.2e-15 | 1.9e-15 | 1.9e-15 | 1.9e-15 | 1.8e-15 |
| sq9 | 1.8e-14 | 1.9e-14 | 1.7e-14 | 1.9e-14 | 1.7e-14 | 1.9e-14 |
| tri | 1.4e-14 | 1.6e-14 | 1.2e-14 | 1.4e-14 | 1.2e-14 | 1.6e-14 |

**λ_max(−L)/λ_open** (the CFL radius) is 0.9994–0.9997 on the scene and 0.971–0.990 on a random 15 %-solid grid. It is **never above 1**, for any rule: removing links (or adding a pinned diagonal term) cannot raise the Gershgorin bound. **Rigid walls cost no time step.**

*Compare:* the non-symmetric ghost normal (RG) grows without bound (§2 (iii)). The cut-cell and Gibou forms raise λ_max by up to 12× and 113× without floors (§2 (iv)).

### 3.6 Leaks through one-cell walls (part B)

**Reported:** the largest fraction of Σu² found on the far side over 500 steps. "0" means exactly zero (no link crosses the wall).

| wall | sq5 (every rule) | sq9 P | sq9 Pcb | sq9 R0nb | sq9 R0 / R1 | tri P | tri Pcb | tri R0nb | tri R0 / R1 |
|---|---|---|---|---|---|---|---|---|---|
| straight, 1 cell | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| array diagonal i = j (8-connected) | 0 | **2.3e-2** | 0 | **0.76** | 0 | **0.17** | 0 | **0.75** | 0 |
| anti-diagonal i + j = n − 1 | 0 | 2.3e-2 | 0 | 0.76 | 0 | 0 | 0 | 0 | 0 |

Brush-rasterised lines at 0–90° (the hand-drawn case): the only leaks are sq9 at 45° with brush radius r < √2/2 (P 9.2e-3, R0nb 0.71). At r ≥ 0.71 the rasterisation is 4-connected, and nothing leaks on any lattice or rule.

Reading it:
- A pinned one-cell diagonal wall leaks a little on sq9 (the diagonal link reads a free cell *through* the wall) and a lot on tri in one direction. The tri link (1,−1) crosses an i = j staircase; the other diagonal is a lattice row pair, so it doesn't leak.
- A rigid one without blocking is **transparent** (76 %): removing the solid cells' links leaves the diagonal links intact.
- **Corner blocking fixes every case.**

### 3.7 Enclosed cavities (part C)

- **Pinned:** a closed ring with a counter (think of the hole in "O", "A", "B") gives 2 connected components and **no zero mode**.
- **Rigid:** each closed rigid region has a **zero mode** (the constant), as a closed room with hard walls should. Measured: 2 zero modes (the domain inside a rigid ring, and the counter).

Consequence: a closed rigid cavity keeps any DC offset forever. If the initial state has a non-zero mean *velocity*, the DC part drifts linearly. The app's seeds start at rest, so the DC stays constant.

A hard (prescribed-u) source inside the cavity removes the zero mode (the source cell acts like a pinned cell).

### 3.8 Circle eigenfrequencies vs exact (`circle_eigs_MIRROR.py` → `circle_eigs_out_MIRROR.txt`)

**Setup:**
- a disc of radius R cells, free iff the cell centre is at distance < R;
- R = 8 … 128, at two centres: cell-centred, and offset (0.31, 0.17);
- the lattice's own bulk dispersion is removed to leading order with the angle-averaged k⁴ term, so what is left is the **boundary** error;
- the reported figure is R·|relative error|. For a first-order method this is constant; it is ≈ the effective radius offset.

| lattice | P (j01 = 2.404826): R·\|err\| cell-c / offset | R0 (j′11 = 1.841184) | R1 (j′11) | order |
|---|---|---|---|---|
| sq5 | 0.27–0.33 / 0.37–0.38 | 0.01–0.06 / 0.12–0.14 | — | 1 (all) |
| sq9 | 0.30–0.36 / 0.41–0.42 | 0.00–0.07 / 0.12–0.14 | 0.03–0.04 / 0.09–0.11 | 1 |
| tri | 0.31–0.35 / 0.34–0.36 | 0.04–0.11 / 0.09–0.11 | 0.00–0.07 / 0.05–0.07 | 1 |

(Mode 2 — j11 for P; j′21 = 3.054237 for rigid — behaves the same, with R·|err| up to 0.24 for R0. The j′02 = 3.831706 errors are the smallest, ≤ 5e-4 at R = 128. Per-centre fits and all raw errors are in the output.)

Reading it:
- **Every staircase rule converges at first order.** The error is a boundary offset of order one cell, as expected.
- **Pinned:** the frequency is always **too low** (one sign), by an effective radius **+0.3 to +0.42 cells** (the pinned wall sits beyond the mask edge). This is systematic, so a mask drawn ~0.35 cells smaller cancels most of it.
- **Rigid:** errors are 2–3× smaller (the wall sits on the rasterisation line) but erratic in sign at the cell-centred disc. **R1** is the most accurate.
- **Reference methods (sq5):**
  - cut cells and Gibou are second order but pay in CFL (§2 (iv));
  - the ghost normal (RG) is first order, *worse* than R0 (R·err ≈ 0.25–0.3), and unstable in time.
- A pooled log-log fit reported p = 0.15 for sq9 R0 on 29/09, because the error changes sign between radii. That figure was wrong and is replaced by this table.

### 3.10 Scattering by a single cell and a 3×3 block (`scatter_MIRROR.py` → `scatter_<lat>_out_MIRROR.txt`)

**Method** (time domain, exact difference field):
- A 10 %-bandwidth plane pulse, built from exact lattice + leapfrog plane waves, is run through the empty box and through the box with the obstacle.
- The difference is the scattered field **exactly**: dispersion and box-edge effects are identical in both runs and cancel. The box is big enough that nothing scattered reaches its edge (checked: the "edge" energy share is < 1e-6).
- Scattered energy ÷ incident energy per unit wavefront width = **scattering width σ** (physical cells).
- **References:** a continuum disc of the same area, from the exact series σ = (4/k) Σ_n |J_n(ka)/H_n(ka)|² (Dirichlet), and the same with J′/H′ (Neumann).

sq5, σ in cells:

| λ | 1 cell P | 1 cell R0 | P/R0 | 3×3 P | 3×3 R0 | P/R0 |
|---|---|---|---|---|---|---|
| 4 | 1.80 | 2.86 | 0.6 | 7.42 | 7.81 | 0.9 |
| 8 | 2.03 | 0.707 | 2.9 | 7.71 | 7.16 | 1.1 |
| 16 | 2.63 | 0.0977 | 27 | 8.91 | 2.90 | 3.1 |
| 32 | 3.64 | 0.0120 | **304** | 10.87 | 0.597 | 18 |
| long-λ law (16→32) | λ^+0.47 | λ^−3.03 | | λ^+0.29 | λ^−2.28 | |

(The 30/09 outputs print the law as "λ^−p" with p the fitted exponent, so "λ^--0.47" means σ grows like λ^+0.47. The script now prints the signed exponent.)

**Effective disc radius**, fitted to the exact continuum series at each λ (`ScratchCLAUDE/WallsCLAUDE/aeff_MIRROR.py`):

| obstacle | fitted radius | equal-area radius | what it matches |
|---|---|---|---|
| pinned single cell | Dirichlet 0.206 / 0.199 / 0.197 (λ = 8 / 16 / 32) | 0.564 | a point clamp |
| pinned 3×3 | Dirichlet 1.28–1.31 | 1.69 | the ring of solid cell *centres* (a 2×2 square) |
| rigid single cell | Neumann 0.68–0.69 | 0.564 | the faces of the solid cells |
| rigid 3×3 | Neumann 1.85–1.90 | 1.69 | the faces of the solid cells |

The fitted radius is **constant with λ**, so the lattice obstacles follow the continuum laws, not just their trend:
- pinned ~ 1/(k ln²(ka));
- rigid ~ k³a⁴.

**sq9 and tri** (σ in cells; the tri cell has area √3/2, and its "3×3" is a 3×3 rhombus of cells, area 7.79):

| | λ = 4 | λ = 8 | λ = 16 | λ = 32 | long-λ law (16→32) |
|---|---|---|---|---|---|
| sq9 cell P | 1.44 | 1.75 | 2.34 | 3.30 | λ^+0.50 |
| sq9 cell R0 (= R1) | 1.70 | 0.519 | 0.0750 | 0.00939 | λ^−3.00 |
| sq9 3×3 P | 6.76 | 7.44 | 8.75 | 10.73 | λ^+0.29 |
| sq9 3×3 R0 / R1 | 6.94 / 6.68 | 6.42 / 6.02 | 2.95 / 2.66 | 0.612 / 0.546 | λ^−2.27 / −2.29 |
| tri cell P | 1.40 | 1.75 | 2.34 | 3.30 | λ^+0.50 |
| tri cell R0 (= R1) | 1.51 | 0.430 | 0.0589 | 0.00731 | λ^−3.01 |
| tri 3×3 P | 6.08 | 7.00 | 8.48 | 10.55 | λ^+0.32 |
| tri 3×3 R0 / R1 | 7.17 / 5.50 | 4.40 / 3.92 | 2.16 / 1.94 | 0.437 / 0.388 | λ^−2.31 / −2.32 |

- At λ = 32 the pinned cell scatters **351× (sq9) and 451× (tri)** more than the rigid one.
- The laws are the same on all three lattices: log-like growth for pinned, Rayleigh λ^−3 for rigid (the 3×3 block is not yet in its long-wave limit at λ = 32: ka = 0.33).
- As predicted, **R1 equals R0 for a single cell**: no dropped link has one free and one rigid common neighbour. For the block, R1 scatters ~10 % less than R0. It restores the tangential stencil moment along the faces, so the block acts slightly smaller.

### 3.11 A rotated square vs an axis-aligned one (`scatter_MIRROR.py`, part 2)

Output: `scatter_<lat>_part2_out_MIRROR.txt`.

**Setup:**
- Scene A: an axis-aligned 11×11 square hit at −ang.
- Scene B: the same square rotated by +ang (30° or 45°), hit at 0°.
- A and B are the same physical scene rotated, so ideally σ_B = σ_A, and pattern_B(φ) = pattern_A(φ − ang) over 72 bins of 5°.
- Cell counts on sq: 121 axis / 121 at 30° / 113 at 45°. On tri: 137 / 137 / 139.

**Reference:** a disc of equal area in both setups. Its rasterisation is identical in both, so it measures only how the answer depends on the incident direction relative to the lattice.

**A first version of this test was wrong.** It used side 12 with a strict |x| < 6, so the *axis* square rasterised to 11 × 11 = 121 cells while the rotated one had 145. That 20 % area mismatch produced a spurious "+38–54 % for rigid, +13–15 % for pinned". The old part 2 is kept in `scatter_<lat>_out_MIRROR.txt`, marked SUPERSEDED.

Ratio σ_B/σ_A (rotated ÷ axis). The disc reference ratio is in brackets (only P and R0 references were run):

| lattice | λ | rot | P | R0 | R1 | disc ref P / R0 |
|---|---|---|---|---|---|---|
| sq5 | 8 | 30° | 1.034 | **1.254** | — | 1.029 / 1.245 |
| sq5 | 8 | 45° | 1.004 | **1.330** | — | 1.037 / 1.224 |
| sq5 | 16 | 30° | 1.034 | **1.253** | — | 1.007 / 0.996 |
| sq5 | 16 | 45° | 1.000 | **1.249** | — | 1.006 / 0.989 |
| sq9 | 8 | 30° | 1.024 | 1.136 | 1.253 | 1.012 / 1.220 |
| sq9 | 8 | 45° | 0.994 | 1.097 | 1.271 | 1.015 / 1.172 |
| sq9 | 16 | 30° | 1.025 | 1.118 | 1.192 | 1.000 / 0.976 |
| sq9 | 16 | 45° | 0.989 | 1.042 | 1.159 | 0.998 / 0.961 |
| tri | 8 | 30° | 0.982 | 1.101 | 1.100 | 0.997 / 0.940 |
| tri | 8 | 45° | 0.987 | 1.005 | 1.012 | 0.999 / 0.936 |
| tri | 16 | 30° | 0.985 | 0.941 | 0.941 | 1.001 / 1.053 |
| tri | 16 | 45° | 0.995 | 0.969 | 0.974 | 1.001 / 1.026 |

**Pattern difference** (L1 of 5° bins, relative):
- P: 0.02–0.06 at λ = 16 (sq5 0.04), 0.06–0.18 at λ = 8.
- R0: sq5 0.24–0.28 at both λ; sq9 0.07–0.22; tri 0.09–0.19.

Reading it:
- **Pinned is orientation-robust:** a rotated pinned square scatters within **±3 %** of the axis-aligned one on every lattice, the same size as the disc reference. The pinned staircase really is "a smooth face shifted inward" (§3.3).
- **Rigid on sq5 is not:**
  - the staircase-faced square scatters **+25 % to +33 %** more, at λ = 8 *and* λ = 16;
  - at λ = 16 the disc reference is within 1 %, so this is the staircase itself, and it does **not** shrink with resolution between these two λ.
  - At λ = 8 even the rigid *disc* depends on the incident direction by 22–24 % on sq5 and sq9 (it falls to ≤ 4 % at λ = 16).
- **sq9:** R0 +4 % to +14 %. **R1 is worse (+16 % to +27 %).** R1 makes the axis-aligned faces exact, so the axis square scatters less, but it does nothing for the stairs. The pair then disagrees more.
- **tri:** the rigid rotated/axis difference is **−6 % to +10 %**, the same size as its own disc reference (−6 % to +5 %). On tri neither square is staircase-free: tri has no vertical lattice line, so the "axis" square's vertical faces are staircases too. Both scenes carry comparable roughness.
- The rigid staircase is a **rough** hard surface: it throws energy into non-specular directions. This is the same effect as the oblique-incidence phase errors of §3.3, seen on a finite object. **It is the main accuracy cost of rigid walls on curved or oblique text outlines, worst on sq5.**

### 3.12 Kernel cost: per-cell link word vs on-the-fly types (`linkmask_probe_MIRROR.py` → `linkmask_probe_out_MIRROR.txt`)

A prototype Taichi kernel with the core's shape: K = 8 steps unrolled per launch, U[3, N, N] rotated by slot, per-cell CC, sq9. Three variants:
- **mask:** today's M byte; masked → u = 0, else the plain stencil.
- **word:** a u16 per cell (i32 on Vulkan, because of the README's u8 fault):
  - bit 15: the cell is free;
  - bits 0–7: the link is active;
  - bits 8–11: the diagonal is pinned-blocked.
  lap = Σ_k b_k w_k (u_k − u_i) − Σ_blocked w_d u_i.
- **types:** a type byte per cell (0 free, 1 pinned, 2 rigid). Each link decides itself from the neighbour's type and, for a diagonal, from the two edge cells it passes between.

**Correctness:** both variants match `walls_lib_MIRROR.build_L` on the mixed pinned/rigid hard-case scene after 32 steps at Co = 0.8:
- f64: 2.9e-14 / 3.0e-14 relative;
- f32: 5.5e-7 to 6.5e-7.

**Speed** (ms/step; RTX 5070 Ti; best of 3–5 × 10 launches; 2 % of cells solid, half pinned, half rigid). The CPU was busy with the scattering runs during these timings, and the GPU may have been shared with other agents, so the ±10 % differences are near the noise.

| backend, N | mask | word | types |
|---|---|---|---|
| CUDA f64, 4097 | 1.043 | 1.156 (+11 %) | 1.434 (+37 %) |
| CUDA f32, 4097 | 0.481 | 0.506 (+5 %) | 0.453 (−6 %) |
| CUDA f64, 8193 | 4.835 | 4.928 (+2 %) | 5.744 (+19 %) |
| CUDA f32, 8193 | 2.045 | 2.282 (+12 %) | 1.979 (−3 %) |
| Vulkan f32, 4097 | 0.944 | 1.131 (+20 %) | 1.257 (+33 %) |

- **Memory:** the word replaces M, so +1 byte/cell (17 → 18 B/cell at f32, 33 → 34 at f64; 4 bytes on Vulkan as today). The types variant adds nothing.
- **Host precompute of the word in numpy:** 3 s at N = 4097, 12–14 s at N = 8193. **Too slow:** in the real code it must be a Taichi kernel (one parallel pass, run only when solids change).
- The same layout covers sq5 (4 link bits) and tri (6 link bits + 6 blocked bits).
- **R1** needs non-uniform weights on the in-wall links. That does not fit a bit word; it needs a small float per link at boundary cells, or a sparse boundary list. It is not needed for a first version.

---

## 4. Hard cases

### 4.1 Several solid neighbours, fins, one-cell walls
- Link removal treats each link independently, so a free cell with three rigid neighbours simply keeps one edge link (plus its unblocked diagonals). There is nothing special to do; symmetry holds (§3.1).
- A **one-cell fin** reflects on both faces independently: no link crosses it.
- **Pinned vs rigid fin:** a pinned fin is a line of zeros; a rigid fin is two independent mirrors. On sq9, a rigid fin's diagonals across it are dropped (the fin cell is the endpoint), so no leak (§3.6: 0).

### 4.2 Diagonal staircase walls: the sq9 diagonals and the tri links
- An 8-connected diagonal wall lets the sq9 diagonal (and one tri link) connect free cells on the two sides **without touching a solid cell**. It crosses the wall through the corner where two solid cells meet.
  - Under **P** this is a real leak: 2.3 % on sq9, 17 % on tri, measured.
  - Under rigid without blocking it is almost transparent (76 %).
- **Blocking rule:** block a diagonal (tri: any link) between two free cells iff **both** cells it passes between are solid.
  - It is symmetric: both endpoints see the same pair.
  - It removes weight symmetrically, so energy stays exact and CFL does not tighten (measured on every scene, §3.5).
  - A pinned blocker contributes −w u_i (a link to a zero); two rigid blockers drop the link.
  - The *either*-solid variant would also be symmetric, but it removes the diagonals running along every face. Not used.
- **Text will produce these walls:** anti-aliased glyph edges thresholded to a mask give 8-connected diagonal steps on every curve. Brush radius ≥ √2/2 (4-connected strokes) is the rasterisation-side fix; corner blocking is the stencil-side fix. **Do both.**

### 4.3 Convex and concave corners
- At a **convex** rigid corner, link removal leaves the corner-adjacent free cell with its tangential links only. The corner is the diffraction point, as it should be.
- At a **concave** rigid corner (a notch), the free cell there loses two edge links; with sq9 the diagonal into the corner cell is dropped too. R1 would give the tangential moment back along each face. At the corner cell itself, R1 on a rectangular room is exactly the ghost ring with the corner = u(1,1) (§3.1).
- **Pinned corners** are exact odd reflections for axis-aligned rectangles on every lattice.

### 4.4 Enclosed cavities
- **Rigid** cavities (letter counters, a closed box) each carry a zero mode (§3.7). This is harmless for seeds at rest.
- It matters if a source with non-zero mean *velocity* is started inside, or if energy/mean readouts assume there is no DC.
- If it ever matters, subtract the mean per connected component (a label pass, when solids change).

### 4.5 sq9 on a straight rigid wall: R0 or R1
- R0 loses 1/6 of the tangential second moment on the boundary row (§3.1). That gives up to 72° phase error at 80° incidence, λ = 8 (§3.3).
- R1 restores it exactly for axis walls. Its only cost is the weight encoding (§3.12).

---

## 5. Recommendation (for later; not implemented)

### 5.1 The model
- **A wall type per object:** `pinned` (today's behaviour), `rigid`, and later `impedance(ζ)`.
- **Default for "objects in the wave" (text, cubes, painted walls):** `rigid` when the scene is sound or water, `pinned` when it is a membrane. Say so in the UI. The outer boundary keeps its own setting (Dirichlet / Neumann / Mur / periodic).
- A painted wall stroke inherits the brush's type.

### 5.2 The rule
- **R0 + corner blocking** on all three lattices. sq5 needs no blocking (it has no diagonal links).
- Symmetric, energy-exact, CFL-neutral, leak-free (§3.5–3.6). Exact for axis walls on sq5; first order on curves, 2–3× more accurate than pinned on the circle test (§3.8).
- **Its weak spot is oblique/curved rigid outlines** (§3.3, §3.11): +25–33 % scattering error for a rotated square on sq5 that did not shrink from λ = 8 to 16. Pinned outlines are within ±3 %. For rigid text, prefer tri (±10 %, at the level of its own disc reference) or sq9.
- **R1** on sq9 and tri as a second step, only if axis-aligned rigid walls (rooms, rectangular boxes) must be exact at grazing incidence. It makes axis faces exact but leaves staircases unchanged, so a scene mixing both becomes *less* consistent (§3.11: sq9 rotated/axis 1.16–1.27 with R1 vs 1.04–1.14 with R0).
- Cut cells / embedded boundaries only if curved-wall accuracy becomes the limit. They need cell merging or a θ floor to keep the explicit Co (§2 (iv)).

### 5.3 The kernel
- Replace the mask M with a **u16 link word** (i32 on Vulkan): bit 15 = free, low bits = active links, next bits = pinned-blocked diagonals.
- The update: masked → 0; else lap = Σ_k b_k w_k (u_k − u_i) − Σ_pinned-blocked w u_i. This form is also the f32-consistent one from the README (each weight multiplies u_k − u_i).
- Build the word in **one Taichi pass** from the type byte whenever the solids or the mask change. Numpy is too slow at N = 8193 (12–14 s).
- Pinned cells keep u = 0, written by the same pass that zeroes masked cells today. Rigid cells can hold 0 too, since nothing reads them.
- **Cost:** +2 to +20 % per step (§3.12). The alternative "types" kernel needs no word, and is as fast at f32 on CUDA, but +19–37 % at f64.

### 5.4 Now, before rigid walls: two changes for pinned solids

These are for the core owner (not done here; see "requests").
- **Stop the leak:** corner blocking for pinned solids on sq9 and tri, or at least 4-connected rasterisation of text and brush strokes. Measured leaks today: 2.3 % (sq9) and 17 % (tri, one direction).
- **Account for the half-cell offset:** a pinned solid acts at its cells' centres. Its effective edge is ~0.35–0.5 cells inside the drawn edge (+0.5 for sq axis walls, +0.35 (sq5) to +0.42 (sq9) for 45° staircases, +0.43 for tri rows). Either document it, or dilate solids by one cell (for thin text strokes that is a visible change, so document it first).

### 5.5 Verification tests for the implementation
1. **Operator equality:** the Taichi update equals `walls_lib_MIRROR.build_L` on the mixed hard-case scene (`energy_leak_MIRROR.mixed_types(scene())`), for every lattice. f64 ≤ 1e-13 relative after 32 steps; the prototype reached 3e-14.
2. **Energy:** drift ≤ 1e-13 over 20 000 steps (f64) with pinned + rigid objects mixed.
3. **Leak:** exactly 0.0 through one-cell straight and diagonal walls, all lattices, both types.
4. **Axis-wall reflection:** a 1D-like pulse gives the phase of R = +1 at the rasterisation line (rigid) and R = −1 at the solid centres (pinned).
5. **Rectangular rigid room on sq9 with R1:** bit-identical to the Neumann ring with the fixed corner u(0,0) = u(1,1).
6. **CFL:** stable at 0.995·Co_max with dense random solids (λ_max ≤ λ_open, §3.5).
7. **Circle eigenfrequencies** inside the §3.8 bands (a regression guard, not an accuracy target).
8. **Cavity:** a rigid ring with a counter keeps the DC of a seed at rest to 1e-14.

### 5.6 Pitfalls
- Forgetting **corner blocking**: a rigid 8-connected diagonal wall becomes 76 % transparent (§3.6).
- Making the box ring or domain edge rigid by accident when only the objects should be. My first scattering run did exactly that: a rigid ring swamped the result with its own reflection.
- **Non-symmetric ghost rules** (mirroring along an estimated normal) look more accurate, but they are unstable in explicit time stepping (§2 (iii)).
- **Rigid staircases:**
  - 30–117° phase error at 80° incidence (§3.3);
  - +25–33 % extra scattering from a rotated square on sq5, +4–27 % on sq9, within ±10 % on tri (§3.11).
  - The plane-wave phase error falls roughly like 1/λ, but the sq5 rotated-square excess did *not* fall from λ = 8 to 16 (1.25 → 1.25), so resolution alone is not a reliable fix. Prefer tri or sq9 for rigid scenes; embedded boundaries (§2 (iv)) are the real fix.
- **DC in closed rigid cavities** (§4.4).
- **Vulkan:** use a 32-bit word (the core's u8 fault applies to u16 as well, until tested).
- **Terminology:** the slit source's wall is **pinned** (u = 0), although the web app's comment (line 1316) and the port's (`membrane_core_MIRROR.py` near line 263) call it "a rigid wall". In this note, "rigid" means Neumann.

---

## 6. Measured here vs literature

**Measured here** (scripts in `ResearchCLAUDE/WallsCLAUDE/`, outputs in `OutCLAUDE/`):
- every number in §3;
- the leak, cavity, energy and CFL claims;
- that R1 equals the ghost Neumann ring with the u(1,1) corner;
- the Gibou/cut-cell accuracy vs CFL trade-off on sq5;
- the ghost-normal instability;
- the kernel costs.

**Literature** (not re-derived here):
- the continuum reflection coefficients and the impedance R(θ) formula;
- the embedded-boundary references: Gibou et al. 2002; Kreiss, Petersson & Yström 2002/2004; Colella et al. 2006 on cell merging and redistribution;
- the FDTD staircase references (Cangellaris & Wright 1991; Dey & Mittra 1997);
- Mur 1981 for the first-order ABC.

The continuum scattering series was evaluated here (scipy), but is textbook.

## 7. Files

| file | what it does |
|---|---|
| `ResearchCLAUDE/WallsCLAUDE/walls_lib_MIRROR.py` | the rules as sparse operators (P, Pcb, R0, R0nb, R1, mixed types) |
| `checks_MIRROR.py` | sanity: port equality, symmetry, R1 = ghost ring, second moments |
| `chain_reflection_MIRROR.py` | exact plane-wave reflection via Bloch reduction to a chain; impedance walls |
| `energy_leak_MIRROR.py` | energy drift, CFL radius, leaks, cavities |
| `circle_eigs_MIRROR.py` | disc eigenfrequencies vs Bessel zeros; sq5 reference methods (RG, CF, GB) |
| `scatter_MIRROR.py` | scattering widths (`python scatter_MIRROR.py sq5\|sq9\|tri [1\|2]`); rotated vs axis square (part 2 → `scatter_<lat>_part2_out_MIRROR.txt`) |
| `ScratchCLAUDE/WallsCLAUDE/aeff_MIRROR.py`, `rotcheck_MIRROR.py` | the effective-disc-radius fit, and the check that exposed the square-rasterisation mismatch |
| `NotesCLAUDE/progress_walls_MIRROR.md` | the progress log of this work |
| `linkmask_probe_MIRROR.py` | Taichi prototype kernels (mask / word / types): verification and timing |
| `OutCLAUDE/*_out_MIRROR.txt` | the outputs quoted above |
