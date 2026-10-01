# Absorbing boundaries beyond first-order Mur, for the ZRL membrane

*Generated from scratch — Claude Opus 5.5 — 29/09/2026 (measurements) and 30/09/2026 (2D, masked, GPU, stability runs; this note)*

The question was: **which absorbing boundaries could we use beyond Mur?** This note answers it for this simulator specifically:
- the scheme is second-order leapfrog for u_tt = c² ∇²u;
- the lattices are square 5-point, square 9-point, and triangular 6-neighbour;
- masked "flavor" domains have staircase walls;
- Co = c·dt/h, usually 0.5.

Every number below comes from a numpy prototype that uses the app's own update rule. The scripts are in `ResearchCLAUDE/AbsorbingBoundariesCLAUDE/`, and `python tables_MIRROR.py` reprints every table from the result files.

---

## 1. Short answer

| use case | what to build | reflection you get (measured) | cost on the GPU (measured) | effort |
|---|---|---|---|---|
| **full rectangle, sq5** | **CPML frame**: convolutional PML, 10–20 cells, cubic profile | L=10: ≤ 1.2e-4 for every angle 0–75° and λ = 8–32 cells. L=20 (α=0.05): ≤ 9e-6 up to 60°, 1.1e-4 at 75°. On a broadband 2D pulse, corners included: 1.8e-4 (L=10), 1.8e-6 (L=20). | +11–19 % per step at N=1025 (one extra pass in the prototype; zero extra passes are possible, §7). Within noise at N=4097. Memory ≈ 2 % of the field if the aux arrays cover the frame only. | ~1 day |
| cheap upgrade of the current ring | Mur2 / Engquist–Majda A2, corner = Mur1 along the diagonal. Or Higdon order 2 (0°,45°) with ε=0.005. | Mur2: 3e-2 (0°) / 2.9e-2 (45°) / 0.11 (60°) / 0.35 (75°). Higdon2: ≤ 7e-3 to 45°, 0.06 at 60°, 0.27 at 75°. The corner region stays at ~0.1. | one ring pass, as today | ½ day |
| **masked flavor shapes, any lattice** | **Distance-to-edge sponge**: grow the mask outward by L cells, damping ramps with the distance to the mask edge, centred damping form | L=40: 1.7e-2 on a broadband pulse, the same on the sq circle, the tri hexagon and the tri circle. L=20: 5e-2. The staircase does not matter. | ≈ 0 if the damping level is packed into the mask byte (a 256-entry table) | ½ day |
| masked shapes, one cell thick | transport along the analytic normal ("T1" = Liao order 1 plus interpolation, with a √((r−c·dt)/r) factor on a circle) | 5–7 % on a staircase circle or hexagon | one boundary-cell pass | ½ day. Stable on sq at Co 0.5 and 0.7 (20000 steps) and 0.7071 (5000 steps); the tri hexagon **grows at Co ≥ 0.7** |
| **triangular lattice** | the same distance sponge | 1.6e-2 at L=40 (tri hexagon) | ≈ 0 | covered by the sponge work |

**What NOT to build:**
- **Higdon order 3 and Liao MTF order 3.** In the 20000-step noise test both blow up (zero-frequency drift). With ε damping, Higdon3 survives, but only after a transient of 3e5 × the initial amplitude.
- **Mur2 with the corners held at 0**, as the app does for Mur1. It blows up in 2D: max|u| 3.7e8 after 20000 steps at Co 0.5.
- **The app's damping form −d(u − u_prev) inside a sponge.** It lowers the CFL limit to Co_max·√(1 − d/2), so at d = 0.5 the limit is 0.612. Every app-form sponge blew up at Co 0.7.

**One finding outside the question.** That same formula applies to the app's **global damping** slider. At d = 0.02 the limit is 0.7036, not 1/√2. So Co between 0.7036 and 0.7071 with full damping diverges, while the CFL row still says it is below the limit (§6, `cfl_damping_MIRROR.py`).

---

## 2. The setting

The update, as in the core (`membrane_core_MIRROR.py`) and the web app:

```
u[c] = (2 u[b] − u[a] + CC · lap(u[b])) − d (u[b] − u[a])        a = n−1, b = n, c = n+1
```

- **Today's absorbing option:** first-order Mur on the outer ring of the full square, `u0' = u1 + k (u1' − u0)`, with k = (Co−1)/(Co+1). The four corners are held at 0. The masked shapes and the triangular lattice are always Dirichlet.
- **What "reflection" means:** the amplitude ratio |R| of the wave that comes back to the incident wave, for a plane wave at angle θ from the wall normal and wavelength λ, in cells per wavelength (ppw).
- **In 2D, the energy norm:** for a pulse, the measure is √(E_reflected / E_0), where E is the leapfrog energy and E_0 is the pulse energy.
- **Continuum formulas quoted below.** For the one-way family, R(θ) = ∏ (cos α_j − cos θ)/(cos α_j + cos θ). The lattice adds a dispersion error on top of that, which is the part at normal incidence that falls like 1/λ².

## 3. How it was measured

**Plane-wave |R| (`abc1d_MIRROR.py`, `sweep1d_MIRROR.py`)**
- **The reduction.** On a grid periodic along the wall, u[i]·cos(k_y j) keeps that form under the 5-point update, with the y-part of the Laplacian replaced by m = 2cos k_y − 2. Every rule tested here is linear, shift-invariant along the wall and even in y. So a 1D run per k_y is **exact** for a straight wall, and one k_y fixes the angle exactly.
- **Check of the reduction (`check2d_MIRROR.py`).** A real 2D periodic-y grid agrees with the 1D run to rounding: 5e-14 absolute for Mur2, 8e-13 for Higdon3, 1e-14 for CPML10.
- **The packet.** A Gaussian wave packet sits on the lattice's own dispersion relation, sin²(ω/2) = Co²(sin²(k_x/2) + sin²(k_y/2)). It is 3λ/cos θ wide.
- **The reference.** The same packet runs in a domain extended far enough that nothing returns within the time window.
- **The result.** When the reflected packet is back at the launch point, |R| = ‖u_test − u_ref‖ over the physical cells / ‖incident‖.
- **The grid.** Angles 0, 15, 30, 45, 60 and 75°; λ = 8, 16 and 32 cells; Co = 0.5 and 0.7.
- **Sanity checks.**
  - Dirichlet gives |R| = 1.0000 ± 1e-4.
  - For every local rule, the measured |R| matches the discrete plane-wave theory R = −P(e^{+ik_x})/P(e^{−ik_x}) to < 1e-3. That includes Mur1 at 75° (0.589 measured vs (1−cos 75°)/(1+cos 75°) = 0.589) and Mur2 at 45° (0.0292 vs 0.0294).
  - Higdon3 and Liao3 without damping depart from theory at λ = 32 and 0° (0.069 and 0.11, against theory values of 4.5e-5 and 2.7e-8). That is the zero-frequency drift of §6 showing up inside the time window.

**2D box (`abc2d_MIRROR.py`, `box2d_runs_MIRROR.py`)**
- **Setup.** 200 × 200 physical cells plus the layer, all four walls with their corners, a reference box large enough that nothing returns, and a Gaussian pulse of σ = 3 cells seeded on both levels, as the app seeds it.
- **The two pulses.**
  - The "centre" pulse is at (100,100), measured when the front has run 230 cells.
  - The "corner" pulse is at (25,25), measured at 165 cells. It sweeps every angle up to grazing and hits the corner.
- **Stability test.** White noise, 20000 steps, Co 0.5 and 0.7.

**Masked shapes (`masked_MIRROR.py`, `masked_normal_MIRROR.py`)**
- Physical radius R = 100, the sq circle, the tri hexagon and the tri circle, a centred pulse (and one off-centre at 0.5R), the same reference method.
- The window is chosen so the reflection crosses the physical region **once**. A first smoke test at R = 30 read 3.6e-4, because the reflection had left through the absorbing edge a second time. At R = 100 the same rule reads 5.7e-2.

**Stability per wavenumber (`stability1d_MIRROR.py`)**
- Random data, 20000 steps, Co = 0.5, 0.7 and 0.7071, k_y = 0, π/4, π/2, 3π/4 and π.

**GPU cost (`gpu_cost_MIRROR.py`)**
- A self-contained copy of the core's fused K = 8 kernel shape on CUDA.
- The variants are timed **interleaved**, keeping each variant's best of 7 rounds. With other agents running, the run-to-run noise is about ±15 %. Raw output is in `results_gpu_cost_MIRROR.txt`.

---

## 4. The methods: how each works, and what it means here

### 4.1 Local one-way conditions (one boundary ring, no layer)

**Mur, first order (Mur 1981; the continuum condition is Engquist–Majda 1977's A1)**
- **How it works:** the one-way equation (∂_t − c∂_x)u = 0, discretised as a box scheme centred at (h/2, t + dt/2).
- **Order and reflection:** exact at normal incidence in the continuum. R(θ) = (1 − cos θ)/(1 + cos θ).
- **Stability and corners:** stable (§6). Needs 2 levels × 2 cells.
- **Applicability:** (a) rectangle only, as today. (b) Masked shapes have no grid-aligned normal; see T1 in §4.5. (c) The tri lattice has no axis-aligned wall except along lattice rows, and is not implemented.
- **Measured:** 3.0e-2 / 7.3e-3 / 1.8e-3 at normal incidence for λ = 8 / 16 / 32 (the lattice error), 0.17 at 45°, 0.33 at 60°, 0.59 at 75°. The box pulse gives 2.2e-2 (centre) and **0.19 (corner)**.

**Mur second order = Engquist–Majda A2 (Engquist & Majda 1977; Mur 1981; for the scalar wave equation also Clayton & Engquist 1977's A2)**
- **How it works:** the paraxial condition u_xt − u_tt/c + (c/2)u_yy = 0, centred at (h/2, n). The update is in the form printed in Taflove & Hagness (2005). Needs 3 levels × 2 cells, which U[3] already holds.
- **Reflection:** R = ((1 − cos θ)/(1 + cos θ))². Measured 1.5e-2 at 30°, 2.9e-2 at 45°, 0.11 at 60°, 0.35 at 75°. Normal incidence is identical to Mur1.
- **Corners:** the tangential term is undefined at the corner, so it needs a corner rule; the corner problem is the subject of Bamberger, Joly & Roberts 1990 (fairly sure of the reference). Two measured options:
  - "diag": Mur1 along the diagonal, with k = (Co − √2)/(Co + √2).
  - "avg": the normal rule along both edge lines, averaged.

  The two are indistinguishable: box pulse 8.8e-3 (centre) and 0.10 (corner). With the corners held at 0 the centre pulse doubles to 4.6e-2 and **the noise run blows up**.
- **Applicability:** (a) only.
- **Static mode:** at k_y = 0 it admits a static ramp. max|u| reaches a plateau of 360 in the 1D test, but it does not grow without bound.

**Clayton–Engquist (1977)**
- The paraxial family, derived for the acoustic and elastic equations, with one-sided discretisations.
- For the scalar membrane, A1 and A2 are the Mur1/Mur2 conditions above. Only the discretisation differs, so they were not measured separately.
- Its reputation for instability belongs to the elastic case (at high Poisson ratio), which does not arise here.

**Higdon (1986, 1987; Math. Comp.)**
- **How it works:** the product ∏_{j=1..p} (cos α_j ∂_t − c ∂_x) u = 0, one box-discretised factor per angle. It is exact at the p chosen angles.
  - One factor with α = 0 is Mur1.
  - Order p needs p + 1 levels and p + 1 cells, so order 2 fits U[3] and order 3 needs a fourth slot.
  - A damping term ε in each factor removes the zero-frequency solutions. I believe Higdon 1987 introduced such terms, but I am not certain.
- **Measured, order 2 (0°, 45°):** ≤ 7.2e-3 over 0–45°, 5.8e-2 at 60°, 0.27 at 75°. Box pulse 2.2e-3 (centre) and 8.3e-2 (corner).
  - Without ε, a static residue of max|u| 0.45 survives 20000 steps of noise. With ε = 0.005 it decays to 0.03.
- **Measured, order 3 (0°, 30°, 60°):** ~3e-4 over 15–30°, 3e-3 at 45°, 0.10 at 75°. Box pulse 2.7e-4 (centre) and 3.9e-2 (corner). But:
  - without ε it **blows up** (linear growth at k_y = 0, up to 2.3e6 in 1D);
  - with ε = 0.005 it survives only after a transient of **3e5 ×** in the 2D noise run.

  Not usable in an interactive app.
- **Applicability:** (a); corners as for Mur2.

**Liao multi-transmitting formula (MTF; Liao, Wong, Yang & Yuan 1984, Scientia Sinica A; fairly sure)**
- **How it works:** extrapolate along the outgoing characteristic: u_0^{n+1} = Σ_j (−1)^{j+1} C(N, j) u(j·c·dt, n + 1 − j). The off-grid values come from quadratic interpolation.
  - In the continuum this is the Higdon product with all angles 0, so R(θ) = ((1 − cos θ)/(1 + cos θ))^N. The measured values match: 5.1e-3 at 45° and 3.7e-2 at 60° for N = 3.
- **Interpolation matters:**
  - The **composed** form (the one-step interpolation applied j times, which I believe is Liao's form; not certain) is consistent.
  - **Direct** interpolation at j·c·dt is not. Measured: |R| tends to 0.14–0.19 as λ → ∞ for N = 3, and 0.10 at 8 ppw for N = 2.
- **Measured, composed:**
  - N = 2 gives excellent normal incidence: 2.2e-3 / 1.5e-4 / 9.2e-6.
  - N = 3 gives 1.1e-4 at 8 ppw.
- **Stability:** the known **drift instability** at low frequency.
  - N = 3 grows linearly at k_y = 0 (to 5e5 in 1D) and blows up in the 2D noise run.
  - N = 2 creeps: max|u| 33 → 460 at the ring and corners over 20000 steps.
  - Stabilised variants exist in the Liao literature; I cannot cite them precisely.
- **Applicability:** its natural home is unstructured and masked grids, because it needs only values along the normal. The one-cell masked-shape rule T1 in §4.5 is its N = 1 case.

**Bayliss–Turkel (1980, CPAM)**
- **How it works:** an asymptotic expansion of outgoing waves in 1/r.
  - In 2D: B1 = (∂_t/c + ∂_r + 1/(2r)).
  - B2 is the product (∂_t/c + ∂_r + 5/(2r)) · B1, which in 2D is the same thing as (∂_t/c + ∂_r + 5/(2r))(∂_t/c + ∂_r + 1/(2r)).
- **Where it fits:** exact for a radial wave from the centre of a circular boundary, so it is the natural condition for the **circle** flavor when the source is central.
- **Status here:** not measured as such. T1 with the cylindrical factor √((r − c·dt)/r) is a B1-like discrete version, and it measured 5.7e-2 (centred) and 7.0e-2 (0.5R off-centre) on a staircase circle. The staircase and the interpolation dominate, not the condition's order. A B2 would need second normal differences through the staircase, and I would not expect it to beat the ~5 % floor without a cut-cell edge.

### 4.2 Absorbing layers without matching ("sponges")

**How they work**
- Add damping σ(x)·u_t in a frame of L cells, ramped from 0 to d_max with the profile d = d_max·s^p, where s is the depth into the frame.
- Cerjan et al. (1985) do the same thing multiplicatively: both time levels are multiplied by exp(−(a(L − i))²).
- Related: Israeli & Orszag 1981; Sochacki et al. 1987 (fairly sure); and the "adiabatic absorber" view of Oskooi, Zhang & Johnson 2008 (Optics Express).
- A sponge is **not matched**. It reflects from its own gradient, and that reflection falls only as the ramp gets long compared with the wavelength. Grazing waves see a thin layer and reflect strongly.

**Measured, best tuning per width** (1D-exact, Co 0.5, geometric mean over 18 cases). At normal incidence for λ = 8 / 16 / 32:

| width | normal incidence, λ = 8 / 16 / 32 | 45° | 75° |
|---|---|---|---|
| L = 10 | 3e-2 / 6e-2 / 0.29 | up to 0.45 | 0.76 |
| L = 20 | 1.3e-3 / 4.3e-3 / 0.16 | up to 0.32 | 0.68 |
| L = 40 (p = 3) | 3e-5 / 2.7e-4 / 2.4e-2 | 0.13 | 0.55 |

- A sponge **must be several longest wavelengths thick**.
- Ending it with Mur1 instead of Dirichlet changes nothing at λ = 32, because the gradient reflection dominates.
- Cerjan with L = 40 gives ~9e-3 at 0°, 5e-2 at 45°, 0.48 at 75°.

**On the broadband 2D pulse:** sponge40 gives 2.2e-2 (centre) and 4.5e-2 (corner). A sponge has no corner problem.

**Masked shapes, the whole point of a sponge here.** The damping needs no normal direction: d(cell) = d_max·((L − dist)/L)^p, where dist is the distance to the nearest outside cell centre in physical coordinates. On a broadband pulse at R = 100:

| shape | L = 20, d_max 0.35 | L = 40, d_max 0.2 |
|---|---|---|
| sq circle | 5.3e-2 | 1.7e-2 |
| tri hexagon | 5.5e-2 | 1.6e-2 |
| tri circle | 5.3e-2 | 1.7e-2 |

The same numbers as the square box: the lattice and the staircase do not matter, because the wave has been damped before it reaches the edge.

**The damping form matters.**
- The app's form −d(u − u_prev) is a backward difference. Its amplification polynomial z² − (2 − d − Co²λ)z + (1 − d) is stable only for Co²λ_max ≤ 4 − 2d.
- The centred form, (1 + d/2)u⁺ = 2u − (1 − d/2)u⁻ + Co²·lap, keeps the undamped limit for any d ≤ 2.
- Measured: every app-form sponge with d_max = 0.5 **blew up at Co 0.7**, and the centred one did not.
- Use the centred form in the layer (§7). It needs a precomputed 1/(1 + d/2): an f64 divide in the loop cost ×2.2 at N = 4097.

### 4.3 Perfectly matched layers (PML)

**How it works**
- Analytically continue the coordinate across the layer, x → x + (i/ω)∫σ dx. This is complex coordinate stretching, the view of Chew & Weedon (1994).
- Waves decay by exp(−∫σ/c · cos θ) and are **not reflected at the interface, at any angle or frequency, in the continuum**.
- What reflects is the discretisation, plus the Dirichlet end after a round trip through the layer. Both are controlled by grading the profile (σ ∝ s^p with p = 2–3) and by the target R₀.

**Variants**
- **Split-field (Berenger 1994).** Split u into u^x + u^y, each damped by its own σ. It was designed for first-order systems (Maxwell). For our second-order scalar scheme it would mean either rewriting in velocity–pressure form or splitting u itself. The split form is only weakly well-posed. Not recommended here.
- **Unsplit / convolutional CPML (Roden & Gedney 2000; Komatitsch & Martin 2007 for seismics).** Replace ∂_x by ∂_x + ψ_x, where ψ is a recursive convolution: ψ^{n} = b·ψ^{n−1} + a·(∂_x u)^n, with b = e^{−(σ + α)dt} and a = σ/(σ + α)·(b − 1).
  - For the second-order equation it is applied twice: ψ at the faces (for ∂_x u) and ζ at the cells (for ∂_x(∂_x u + ψ)). That is two memory arrays per direction, in the frame only.
  - This is the form prototyped here (`cpml` in `abc1d`/`abc2d`/`gpu_cost`).
  - Related second-order-equation PMLs: Komatitsch & Tromp 2003 (GJI; fairly sure); Pasalic & McGarry 2010 (SEG abstract; unsure of the exact reference).
- **Complex-frequency-shifted PML (CFS; Kuzuoglu & Mittra 1996).** The stretch becomes 1 + σ/(α + iω), and optionally κ. The α > 0 shift absorbs evanescent and low-frequency content better and cures late-time growth, at the price of less absorption for very long waves.
  - Measured: α = 0.05 was the best 20-cell setting. It helped the tuning grid's geometric mean slightly, but it costs at grazing λ = 32 (1.1e-4 at 75°).
- **Second-order-in-time wave-equation PMLs with auxiliary fields (Grote & Sim 2010, "Efficient PML for the wave equation", arXiv; fairly sure).** In 1D: u_tt + σu_t = ∇·(c²∇u + φ) + ψ, with a face field φ and a cell field ψ. In 2D a vector φ and one scalar, plus a σ_xσ_y·u term.
  - Prototyped in 1D (`gs`) with a straightforward time discretisation.
  - Measured: 10 cells ≤ 6.7e-4, 20 cells ≤ 8.4e-5 up to 60°, but ~9e-4 at 75°. Worse than my CPML at the same width, probably because of my time discretisation rather than the formulation.

**Measured CPML** (1D-exact, Co 0.5; the 2D box includes the corners, where both directions overlap):

| layer | plane waves, worst over 0–60° and λ = 8–32 | at 75° | 2D pulse centre / corner | Co 0.7 |
|---|---|---|---|---|
| 5 cells, p = 2 | 5.2e-3 | 1.4e-3 | 1.8e-3 / 1.7e-3 | 3.0e-3 / 2.5e-3 |
| 10 cells, p = 3 | 1.2e-4 | 1.8e-5 | 5.3e-5 / 1.8e-4 | 1.0e-4 / 2.6e-4 |
| 20 cells, p = 3, α = 0.05 | 8.9e-6 | 1.1e-4 | 2.5e-6 / 1.8e-6 | 2.5e-6 / 1.8e-6 |

- σ_max = f·(p + 1)·ln(1/R₀)/(2L), with R₀ = 1e-6 and f = 2 for the best settings.
- **Stable** in the 20000-step noise run at Co 0.5 and 0.7 (max|u| 0.04–0.2, energy falling).
- A 10-cell CPML beats a 40-cell sponge by **~400× on the centre pulse, ~250× on the corner pulse, and ~200× on the worst normal-incidence plane wave**.

**Applicability**
- (a) **Rectangle: yes. This is the recommendation.**
- sq9: the diagonal terms of the 9-point Laplacian mix ∂_x and ∂_y, so a clean CPML needs a stretched mixed derivative. The cheap route is the 5-point stencil inside the frame only. Unmeasured.
- (b) **Masked shapes:** a PML needs a coordinate to stretch.
  - For a circle, a radial PML exists (stretch r), but on a staircase it degrades.
  - For n-gons, each face needs its own stretch, and the corners are wedges.
  - For our shapes the practical route is to **put the PML on the array frame and let the flavor shape be open**. That changes the physics: the shape is no longer a drum. Or use the sponge.
- (c) **Tri lattice:** the 6-neighbour Laplacian is (2/3)Σ_k D_k², summed over the 0°, 60° and 120° lattice directions. Stretching x mixes the directions, so a PML there needs either a Cartesian re-derivation with mixed-derivative memory terms or direction-wise stretching for faces along lattice rows. That is research-level work; I know of no published tri-lattice FD PML to point to. Use the sponge.

**Stability caveat (Bécache, Fauqueux & Joly 2003; fairly sure):** PMLs can go unstable for anisotropic media, and for "backward" waves whose group and phase velocities point opposite ways across the layer. The membrane is isotropic. But the "prism-like media" phase planned for later could make CC anisotropic in effect, so this has to be re-checked then.

### 4.4 Perfectly matched discrete layers and discrete PMLs

**PMDL (Guddati and co-workers)**
- **How it works:** discretise the PML with a few (2–5) linear finite elements, using mid-point integration in the normal direction. The discrete layer is then reflectionless for the **discretised** normal operator, and the whole thing is equivalent to a continued-fraction (rational) ABC. The continued-fraction ABCs are Guddati & Tassoulas 2000 (J. Comput. Acoust.); the extension to convex polygonal domains is Guddati & Lim 2006 (IJNME). I am fairly sure of both; which paper first used the name "PMDL" I am **unsure** of.
- **Payoff:** PML-class absorption of propagating waves with a handful of layers.

**Discrete PML derived on the lattice (Chern 2019, JCP, "A reflectionless discrete perfectly matched layer"; fairly sure)**
- The layer is derived from the lattice equations themselves, so there is no discretisation reflection at all, only the round trip.
- Attractive for a project that treats the lattice as the object. As far as I know it is derived for the square grid only.

**Here:** not prototyped. CPML already reaches 1e-4 in 10 cells and 1e-6 in 20. GPU memory is not the constraint, and the frame is 1–2 % of an N = 1025 grid. The payoff would be thinner layers, which only matters at small N. Effort: 2–3 days.

### 4.5 One cell thick on a staircase edge: transport along the normal (measured, new here)

**How it works**
- Boundary cells are the mask cells with an outside neighbour. They are not updated by the stencil. Instead, T1 sets u_b^{n+1} = f·u^n(x_b − c·dt·n̂).
  - n̂ is the analytic normal: radial for the circle, the nearest face for the hexagon. A generic mask would take it from a smoothed distance-field gradient.
  - The off-grid value is interpolated at level n: bilinear on the square lattice, barycentric on the tri triangles.
  - f = √((r − c·dt)/r) on a circle (B1-like), and 1 on flat faces.
- Only level n is read, so it is explicit and order-free: one parallel pass.

**Measured** (R = 100, Co 0.5):

| shape | centred pulse | off-centre pulse (0.5R) |
|---|---|---|
| sq circle | 5.7e-2 | 7.0e-2 |
| tri circle | 5.1e-2 | 6.3e-2 |
| tri hexagon | 5.7e-2 | 6.4e-2 |

- Stable on the sq circle: 20000 steps at Co 0.5 and 0.7, and 5000 steps at 0.7071.
- On the **tri hexagon: stable at Co 0.5 (20000 steps) and 0.6 (5000 steps), slow growth at 0.7, blow-up at 0.75 and 0.8.** Probably the corners of the hexagon, where the nearest face, and so the normal, switches.
- The second-order version (T2, Liao N = 2 with direct interpolation) is **worse (0.21–0.31) and unstable**, as the 1D inconsistency of direct interpolation predicts.

**Verdict:** a one-cell option for flavor shapes. It reflects about as much as Mur1 does on average, at no memory cost, and costs L cells less than the sponge. The sponge is 3–4× better at L = 40 and has no Co restriction.

### 4.6 High-order local ABCs: Hagstrom–Warburton and complete radiation BCs

**How they work**
- Higdon's product, rewritten with auxiliary functions φ_1..φ_P that live **on the boundary only**. Each recursion step is a first-order one-way relation, so arbitrary order costs P boundary lines of aux data and no high derivatives. References:
  - Hagstrom & Warburton 2004 (Wave Motion) covers the auxiliary-variable formulation and the **corner compatibility conditions**;
  - Givoli & Neta 2003 is a closely related scheme (fairly sure of authors and year; unsure of the venue);
  - the review is Givoli 2004 (Wave Motion).
- Complete radiation BCs (CRBC; Hagstrom & Warburton 2009, SIAM J. Numer. Anal.; Hagstrom, Warburton & Givoli 2010, J. Comput. Appl. Math.; fairly sure) add evanescent parameters. With optimal parameters, the error bound is **uniform in time** for a given order: about 1e-4 at P ≈ 5–8 over long times, by the authors' estimates, which I have not reproduced.

**Here:**
- (a) The rectangle is the ideal geometry: flat faces, the published corner conditions apply, and the memory is P·4N values.
- (b) Convex polygons are handled in the literature (as are the CFABCs above), but only with **faces along lattice rows**: the square on sq, the hexagon on tri. Rotated or staircase faces break the one-dimensional structure.
- (c) The hexagon on the tri lattice is geometrically natural, but the tri stencil's cross terms would need re-derivation.

**Not prototyped.** On the GPU it adds P ring passes per step, each costing ~20 µs of per-pass overhead at small N (§7), unless they are fused into one. At equal reflection level, CPML is simpler. Effort: 3–5 days including corners.

### 4.7 Exact non-reflecting conditions: Dirichlet-to-Neumann and boundary integrals

**How they work**
- The exact outgoing condition on a circle (Keller & Givoli 1989, JCP; time-dependent: Grote & Keller 1995, SIAM J. Appl. Math.; fairly sure of the year) is nonlocal in angle (a Fourier series) and in time (a convolution kernel per mode).
- Alpert, Greengard & Hagstrom (2000 SIAM J. Numer. Anal.; 2002 JCP) compress each mode's kernel to O(log) exponentials. The cost is then roughly O(N_b log N_b) per step for N_b boundary points, to a set tolerance.
- Kirchhoff / Green boundary integrals (e.g. Ting & Miksis 1986, JASA; fairly sure) propagate u and ∂_n u from an interior surface.

**The 2D catch:** no sharp Huygens principle. 2D waves leave a wake, so an exact condition needs the **entire time history**, or its compressed kernel. The lacuna-based truncation (Ryaben'kii, Tsynkov & Turchaninov 2001, JCP; fairly sure) relies on the lacunae of odd dimensions. I am **not sure** whether a 2D variant exists.

**Here:** the circle flavor is the only natural geometry, and it needs interpolation from the staircase to the true circle. The measured staircase floor of T1 (~5 %) suggests the edge representation, not the condition, would limit the result, unless the edge is cut-cell. Effort: 1–2 weeks. Not recommended for this app.

### 4.8 Damping ramp by distance to the mask edge (the sponge for flavor shapes)

This is the masked-shape row of §4.2, with the implementation details:

- **Grow the mask outward by L cells.** Keep the physical shape unchanged and put the ramp in the added band. If the ramp is put *inside* the shape instead, the visible shape loses L cells and the resonances change.
- **Distance.** On the host, once per reset: the physical-coordinate distance from each cell centre to the nearest outside cell centre. `scipy.spatial.cKDTree` is available (scipy 1.18.1). A lattice BFS to depth ~1.2L first limits the query to the band. It works for any mask on either lattice.
- **Profile.** p = 2, with d_max = 0.2 at L = 40 or 0.35 at L = 20. Use the centred form.
- **Reflection:** 1.7e-2 at L = 40 and 5e-2 at L = 20, the same on every shape and lattice measured.
- **Physics caveat.** An absorbing edge turns a flavor drum into an open window: there are no standing modes, and the (m,n) mode sources lose their meaning. It should be a separate option ("open") next to Dirichlet, not a replacement.

---

## 5. The measured tables

### 5.1 Local rules, plane waves, Co = 0.5 (worst over λ = 8/16/32 at each angle)

| method | 0° | 15° | 30° | 45° | 60° | 75° | 0° at λ = 8 / 16 / 32 |
|---|---|---|---|---|---|---|---|
| Dirichlet | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 |
| **Mur1 (today)** | 3.0e-02 | 1.6e-02 | 7.1e-02 | 0.17 | 0.33 | 0.59 | 3.0e-02 / 7.3e-03 / 1.8e-03 |
| Mur2 / EM-A2 | 3.0e-02 | 2.7e-02 | 1.5e-02 | 2.9e-02 | 0.11 | 0.35 | 3.0e-02 / 7.3e-03 / 1.8e-03 |
| Higdon2 (0,45) | 6.1e-03 | 2.5e-03 | 7.2e-03 | 2.2e-03 | 5.8e-02 | 0.27 | 6.1e-03 / 1.3e-03 / 3.2e-04 |
| Higdon2 (0,60) | 1.1e-02 | 5.1e-03 | 1.9e-02 | 2.9e-02 | 4.9e-03 | 0.19 | 1.1e-02 / 2.5e-03 / 6.1e-04 |
| Higdon2 (0,45) ε=.005 | 6.1e-03 | 3.2e-03 | 7.4e-03 | 3.5e-03 | 5.8e-02 | 0.27 | 6.1e-03 / 1.7e-03 / 2.2e-03 |
| Higdon3 (0,30,60) | 6.9e-02 | 2.9e-04 | 2.8e-04 | 3.0e-03 | 1.3e-03 | 0.10 | 1.1e-03 / 9.4e-04 / 6.9e-02 (drift) |
| Higdon3 (0,30,60) ε=.02 | 1.6e-03 | 1.4e-03 | 1.5e-03 | 4.1e-03 | 9.7e-03 | 0.11 | 1.2e-03 / 7.5e-04 / 1.6e-03 |
| Higdon3 (0,45,70) ε=.005 | 4.4e-03 | 1.5e-03 | 3.2e-03 | 1.2e-03 | 1.1e-02 | 4.0e-02 | 3.2e-03 / 8.7e-04 / 4.4e-03 |
| Liao2 (composed) | 2.2e-03 | 7.2e-04 | 5.0e-03 | 2.9e-02 | 0.11 | 0.35 | 2.2e-03 / 1.5e-04 / 9.2e-06 |
| Liao3 (composed) | 0.11 | 2.1e-05 | 3.6e-04 | 5.1e-03 | 3.7e-02 | 0.20 | 1.1e-04 / 1.8e-03 / 0.11 (drift) |
| Liao2, direct interp. | 0.10 | 0.10 | 7.6e-02 | 5.2e-02 | 0.11 | 0.35 | 0.10 / 5.0e-02 / 2.5e-02 |
| Liao3, direct interp. | 0.19 | 0.17 | 0.14 | 0.10 | 8.3e-02 | 0.22 | 0.18 / 0.15 / 0.19 |

At Co = 0.7 the local rules are within a few % of these values. Mur1 improves at normal incidence: 2.1e-2 / 5.0e-3 / 1.2e-3.

### 5.2 Layers, plane waves, Co = 0.5 (best tuned per family and width)

| method | 0° | 15° | 30° | 45° | 60° | 75° | 0° at λ = 8 / 16 / 32 |
|---|---|---|---|---|---|---|---|
| sponge 10, p2 d.50 | 0.29 | 0.31 | 0.36 | 0.45 | 0.58 | 0.76 | 3.1e-02 / 5.7e-02 / 0.29 |
| sponge 20, p2 d.50 (centred) | 0.16 | 0.17 | 0.22 | 0.31 | 0.45 | 0.68 | 1.2e-03 / 6.3e-03 / 0.16 |
| sponge 40, p3 d.50 (centred) | 2.1e-02 | 2.7e-02 | 5.2e-02 | 0.12 | 0.27 | 0.54 | 4.5e-05 / 1.8e-04 / 2.1e-02 |
| Cerjan 40 (a = .0075) | 9.1e-03 | 7.0e-03 | 8.1e-03 | 5.2e-02 | 0.19 | 0.48 | 6.2e-03 / 7.9e-03 / 9.1e-03 |
| **CPML 5**, p2 | 5.2e-03 | 5.1e-03 | 4.4e-03 | 1.8e-03 | 6.8e-04 | 1.4e-03 | 5.2e-03 / 1.7e-03 / 4.6e-04 |
| **CPML 10**, p3 f2 | 9.8e-05 | 1.2e-04 | 1.2e-04 | 3.3e-05 | 2.4e-05 | 1.8e-05 | 9.8e-05 / 4.3e-05 / 7.1e-05 |
| **CPML 20**, p3 f2 α.05 | 8.9e-06 | 4.8e-06 | 3.8e-06 | 2.7e-06 | 1.6e-06 | 1.1e-04 | 5.2e-06 / 3.3e-06 / 8.9e-06 |
| Grote–Sim 10 | 6.7e-04 | 6.1e-04 | 4.6e-04 | 2.9e-04 | 1.4e-04 | 9.7e-04 | 6.7e-04 / 2.7e-04 / 1.3e-04 |
| Grote–Sim 20 | 8.4e-05 | 7.6e-05 | 5.8e-05 | 3.6e-05 | 1.8e-05 | 8.7e-04 | 8.4e-05 / 3.4e-05 / 1.6e-05 |

`python tables_MIRROR.py` prints the per-wavelength detail. The sponge's reflection is almost all at λ = 32. At λ = 8, sponge40 is 2–5e-5 up to 45°.

### 5.3 2D box: broadband pulse, all four walls and the corners

| method | Co 0.5 centre | Co 0.5 corner | Co 0.7 centre | Co 0.7 corner |
|---|---|---|---|---|
| Dirichlet | 1.00 | 0.84 | 1.00 | 0.84 |
| **Mur1, corners 0 (today)** | 2.2e-02 | 0.19 | 2.3e-02 | 0.20 |
| Mur2, corners 0 | 4.6e-02 | 0.11 | 4.5e-02 | 0.11 (**blows up** in the noise run) |
| Mur2, corners diag-Mur1 | 8.8e-03 | 0.10 | 5.9e-03 | 0.10 |
| Mur2, corners avg | 9.3e-03 | 0.10 | 6.6e-03 | 0.10 |
| Higdon2 (0,45) ε.005 | 2.2e-03 | 8.2e-02 | 2.2e-03 | 8.3e-02 |
| Higdon3 (0,30,60) ε.005 | 2.7e-04 | 3.8e-02 | 2.3e-04 | 3.9e-02 |
| Higdon3 (0,45,70) ε.005 | 1.1e-03 | 2.4e-02 | 1.0e-03 | 2.5e-02 |
| Liao2 | 7.4e-04 | 0.10 | 7.6e-04 | 0.10 |
| Liao3 | 7.5e-05 | 6.2e-02 | 7.4e-05 | 6.5e-02 |
| sponge 40, app form | 2.3e-02 | 4.7e-02 | **blows up** | **blows up** |
| sponge 40, centred | 2.2e-02 | 4.5e-02 | 1.9e-02 | 3.7e-02 |
| CPML 5 | 1.8e-03 | 1.7e-03 | 3.0e-03 | 2.5e-03 |
| **CPML 10** | 5.3e-05 | 1.8e-04 | 1.0e-04 | 2.6e-04 |
| **CPML 20** | 2.5e-06 | 1.8e-06 | 2.5e-06 | 1.8e-06 |

**Corners are where local rules lose.** Every one-cell rule stays at 2–20 % on the corner pulse, whatever its straight-wall order, while CPML has no corner problem (the frame corners carry both σ's).

---

## 6. Stability (20000 steps)

**2D noise runs** (`results_box2d_MIRROR.json`: max|u| over the physical cells, starting from white noise at max 1)

| method | Co 0.5 | Co 0.7 |
|---|---|---|
| Dirichlet | 1.9 → 1.7 (energy conserved) | 3.1 → 3.1 |
| Mur1 (today) | 0.37 → 0.05 | 0.31 → 0.026 |
| Mur2, corners 0 | **blows up** (3.7e8) | **blows up** |
| Mur2, corners diag or avg | 0.32 → 0.045 | 0.29 → 0.017 |
| Higdon2 (no ε) | 0.69 → 0.45 (static residue) | 0.64 → 0.44 |
| Higdon2 ε.005 | 0.35 → 0.031 | 0.23 → 0.015 |
| Higdon3 (no ε) | **blows up** | **blows up** |
| Higdon3 ε.005 | **3.5e5** → 0.04 | **3e5** → 0.01 |
| Liao2 | 33 → **460** (ring/corners creep) | 32 → **520** |
| Liao3 | **blows up** | **blows up** |
| sponges, app form, d = 0.5 | 0.2 → 0.01–0.03 | **blow up** (Co_max = 0.612) |
| sponge 40, centred form | 0.19 → 0.015 | 0.13 → 0.006 |
| CPML 5 / 10 / 20 | 0.4 → 0.04–0.05 | 0.6–1.0 → 0.05–0.21 |

**1D per wavenumber** (`stability1d_MIRROR.py`, Co = 0.5, 0.7 and 0.7071). Every flag is at **k_y = 0**, with two exceptions at k_y = π at the CFL limit itself:

| rule | behaviour at k_y = 0 |
|---|---|
| Mur2 | plateau at 360 (the static ramp) |
| Higdon2 | plateau at 142 |
| Liao2 at Co ≥ 0.7 | plateau at 120 |
| Higdon3 | linear growth to 2–4e6 |
| Liao3 | linear growth to 5e5–5e6 |
| Higdon3 ε.02 | decays slowly, from a transient of 600 (still 22 after 20000 steps at Co 0.5) |

Mur1 is clean everywhere. The two exceptions: at Co = 0.7071 and k_y = π, Liao2 and Liao3 stay above the initial amplitude (15 → 9 and 19 → 5). They decay, but slowly. This is the classical GKS picture (Gustafsson, Kreiss & Sundström 1972; Trefethen 1982 on group velocity): **zero-frequency, zero-group-velocity modes that one-way products of order ≥ 2 do not damp.**

**The damping form and the CFL limit** (`cfl_damping_MIRROR.py`)
- Backward-difference damping gives Co_max(d) = Co_max(0)·√(1 − d/2).
- Checked at d = 0.02, 0.2 and 0.5, 0.005 below and 0.005 above the predicted limits of 0.7036, 0.6708 and 0.6124: stable below, blow-up above. The centred form stays stable up to 1/√2.
- **This applies to the app's global damping slider too** (max 0.02). The live CFL row does not account for it (request to the app/core owners).

---

## 7. Cost on the GPU, and how it fits the fused kernel

### 7.1 Measured cost

CUDA, K = 8 steps per launch, L = 20, µs per step. The noise is about ±15 %, so anything within ±15 % of "plain" is noise.

| variant | f64, N = 1025 | f64, N = 4097 | f32, N = 1025 | f32, N = 4097 |
|---|---|---|---|---|
| plain (Dirichlet ring) | 168 | 943 | 145 | 464 |
| Mur1 ring (today) | 175 | 943 | 178 | 419 |
| generic local rule (5 × 3 table + tangential term), diag corners | 190 | 783* | 176 | 474 |
| sponge, per-cell D array (+1 float read per cell) | 181 | 1031 | 147 | 538 |
| **sponge, level packed in the mask byte + 256-entry table** | 164 | 937 | 147 | 425 |
| sponge, depth computed from i,j (rectangle only) | 166 | 967 | 141 | 466 |
| sponge, centred form with an **f64 divide** | 162 | **2078** | 152 | 465 |
| **sponge, centred, divide precomputed in a 2nd table** | 161 | 940 | 155 | 467 |
| **CPML frame** (extra face pass, aux arrays full-size) | 186 | 822* | 172 | 514 |

\* below plain: noise.

- An **f64 divide per cell doubles the step at N = 4097**. This reproduced 3 times (1729, 2028 and 2078 µs). Consumer GPUs run f64 at a small fraction of the f32 rate. Precompute 1/(1 + d/2).
- The generic-rule kernel took **37–60 s to compile cold**, against ~1–3 s for the others. Specialise per rule instead of unrolling a coefficient table.
- The prototype's CPML costs one extra parallel pass per step. That is the +11–19 % at N = 1025, consistent with the core's ~20 µs per pass.

### 7.2 Fitting the fused k_run

**Sponge (all lattices, all shapes)**
- **Storage.** The core's mask is already "0 = outside, nonzero = inside" (u8, or i32 on Vulkan). Store the damping level as the mask value: 1 = plain, 2..L+1 = ramp levels. A 256-entry table holds d, and a second table holds 1/(1 + d/2). That costs no extra memory traffic, which is the measured ≈ 0 cost.
- **Update.** Branch on m ≥ 2 to the centred layer formula:

  ```
  U[c] = (2u + CC·lap)·inv[m] − (1 − d[m]/2)·inv[m]·U[a]
  ```

  Cells with m = 1 keep the existing app formula, so interior cells stay bit-identical to the web app. That is what keeps the 26/26 cross-check meaningful.
- **Geometry.** Grow the array by L on every side (or grow the mask outward by L), render only the physical region, and exclude the layer from the energy ledger rows.

**CPML (rectangle)**
- **One pass is possible.** Cell (i,j) recomputes the new face memories ψ at *both* of its x-faces, and likewise for y, from the old ψ (ping-pong buffer [2, ...], indexed by step parity) and u^n. It writes only its own face. Then it updates its cell memory ζ in place. The redundant recompute removes the race, and with it the extra pass.
- **Memory.** Per direction: ψ (2 buffers, faces) + ζ (cells), frame-packed into [2L, N] arrays. That is 12·L·N values: 7.9 MB f64 at N = 4097 and L = 20, against 403 MB for U[3]. Full-size aux arrays would cost 6·N² values (805 MB f64 at 4097), so they are fine only up to N ≈ 2049.
- **Coefficients.** 1D tables b, a at cells and faces.
- **Outer edge.** The frame's outermost ring is Dirichlet (the existing clamp).

**Local rules (Mur2 / Higdon2)**
- One ring pass after the interior pass, like today's Mur. They need 3 levels, which U[3] holds: a, b and c are distinct slots.
- Use the **diag-Mur1 corner**. The "avg" corner reads ring values written in the same pass, which is a race on the GPU. The two measured the same.
- Order 3 would need a 4th time level. Don't.

**T1 on masked shapes**
- Host-side lists of boundary cells with 3–4 interpolation indices and weights, and f. One pass after the interior. Order-free, because it reads level n only.

---

## 8. Recommended plan for the Taichi port

1. **Fix the damping CFL display first (the core/app owners' files; see requests).** Co_max(d) = Co_max(0)·√(1 − d/2) for the app's damping form.
2. **Distance-ramp sponge as a general "open edge" (½ day).**
   - Works for the rectangle, every flavor shape and both lattices.
   - Mask-code storage, centred form, L = 40 and d_max = 0.2 by default (L = 20 and d_max = 0.35 for small N).
   - Expect ~2 % reflection on the app's pulses (5 % at L = 20).
   - Tests: port `masked_MIRROR.py`'s reference-run measurement to the core, plus a noise run at Co 0.7 (tri: 0.8).
3. **CPML frame for the full rectangle, sq5 (1 day).**
   - L = 10 (p = 3, f = 2, α = 0) as the default, L = 20 (α = 0.05) as "high".
   - Expect ≤ 1e-4 (L = 10) or ~1e-6 (L = 20) on pulses, corners included.
   - Single-pass ψ ping-pong, frame-packed aux arrays. Keep Mur1 as the "1-cell" option.
   - Tests: the 1D-exact plane-wave table (port `abc1d` numbers as expectations), the 2D pulse, and 20000-step noise at Co 0.5 and 0.7, f32 and f64.
4. **Optional:**
   - Mur2 with the diag corner as a cheap improvement to the 1-cell ring (½ day). It gains ~2.5× on the centre pulse and ~2× on the corner pulse.
   - T1 for flavor shapes as the 1-cell open edge (½ day; limit Co ≤ 0.65 on the tri hexagon until the corner growth is understood).
5. **Not planned:** Higdon/Liao order 3, Hagstrom–Warburton/CRBC, PMDL, DtN or Kirchhoff. They are either unstable here or bring gains CPML already delivers, at several times the effort.
6. **Later (prism-like media phase):** re-check the PML's stability once the medium is anisotropic.

## 9. Not measured / open

- Square 9-point: every measurement here is sq5. The sponge carries over directly. The CPML needs the mixed derivative or a 5-point frame. Mur1 on sq9 exists in the app but is unmeasured here.
- T1 on the tri hexagon grows at Co ≥ 0.7. The cause is presumably the hexagon corners (the normal switches), but that is not isolated.
- CPML with the app's *global* damping d > 0 inside the frame: untested, but it should compose, since both are per-cell linear terms.
- A PML on the triangular lattice: open research here.
- The sponge profiles were tuned on a small grid (p ∈ {1, 2, 3}, 7 values of d_max). Optimised profiles might gain a factor of 2–3, not orders of magnitude.

## 10. Sources

**Cited with confidence:**
- Engquist & Majda 1977 (Math. Comp.)
- Clayton & Engquist 1977 (BSSA)
- Mur 1981 (IEEE Trans. EMC)
- Bayliss & Turkel 1980 (CPAM)
- Cerjan, Kosloff, Kosloff & Reshef 1985 (Geophysics)
- Berenger 1994 (JCP)
- Chew & Weedon 1994 (Microwave Opt. Tech. Lett.)
- Kuzuoglu & Mittra 1996 (IEEE MGWL)
- Roden & Gedney 2000 (Microwave Opt. Tech. Lett.)
- Collino & Tsogka 2001 (Geophysics)
- Komatitsch & Martin 2007 (Geophysics)
- Keller & Givoli 1989 (JCP)
- Givoli 2004 (Wave Motion, review)
- Tsynkov 1998 (Appl. Numer. Math., review)
- Gustafsson, Kreiss & Sundström 1972 (Math. Comp.)
- Trefethen 1982 (SIAM Review)
- Taflove & Hagness 2005 (Computational Electrodynamics, 3rd ed.)
- Oskooi, Zhang & Johnson 2008 (Optics Express)

**Fairly sure (author and year; venue as given):**
- Higdon 1986 and 1987 (Math. Comp.)
- Liao, Wong, Yang & Yuan 1984 (Scientia Sinica A)
- Israeli & Orszag 1981 (JCP)
- Sochacki et al. 1987 (Geophysics)
- Engquist & Majda 1979 (CPAM)
- Bamberger, Joly & Roberts 1990 (SIAM J. Numer. Anal.)
- Komatitsch & Tromp 2003 (GJI)
- Grote & Sim 2010 (arXiv)
- Bécache, Fauqueux & Joly 2003 (JCP)
- Guddati & Tassoulas 2000 (J. Comput. Acoust.)
- Guddati & Lim 2006 (IJNME)
- Chern 2019 (JCP)
- Hagstrom & Warburton 2004 (Wave Motion) and 2009 (SIAM J. Numer. Anal.)
- Hagstrom, Warburton & Givoli 2010 (J. Comput. Appl. Math.)
- Grote & Keller 1995 (SIAM J. Appl. Math.)
- Alpert, Greengard & Hagstrom 2000 and 2002
- Ting & Miksis 1986 (JASA)
- Ryaben'kii, Tsynkov & Turchaninov 2001 (JCP)

**Unsure:**
- Givoli & Neta 2003: the venue.
- Pasalic & McGarry 2010 (SEG): the exact reference.
- Which Guddati paper coined "PMDL".
- Whether Higdon introduced the ε terms.
- Whether "composed" interpolation is exactly Liao's published form.
- The stabilised-MTF literature.
- Whether a 2D lacuna method exists.

## 11. Files (all in `ResearchCLAUDE/AbsorbingBoundariesCLAUDE/`)

| file | what | run time |
|---|---|---|
| `abc1d_MIRROR.py` | 1D-reduced exact bench; every rule as a polynomial in shift/delay; sponge, Cerjan, CPML, Grote–Sim | — |
| `sweep1d_MIRROR.py` → `results_1d_MIRROR.json` | the plane-wave sweep, tuning, Co 0.7 | 139 s, 8 processes |
| `check2d_MIRROR.py` | proves the 1D reduction on a real 2D grid | 2 s |
| `abc2d_MIRROR.py`, `box2d_runs_MIRROR.py` → `results_box2d_MIRROR.json` | 2D box: pulse (centre/corner), 20000-step noise. The json's numpy cost column is noise; do not use it | 5 min |
| `stability1d_MIRROR.py` → `results_stability1d_MIRROR.json` | per-wavenumber growth | 16 s |
| `masked_MIRROR.py` → `results_masked_MIRROR.json` | distance sponge on sq circle / tri hexagon / tri circle | 6 min |
| `masked_normal_MIRROR.py` → `results_masked_normal_MIRROR.json` | T1/T2 transport rule on the staircase edge | 4 min |
| `cfl_damping_MIRROR.py` | the damping-form CFL check | 5 s |
| `gpu_cost_MIRROR.py` → `results_gpu_cost_MIRROR.txt` | fused-kernel cost on CUDA | ~2 min per precision |
| `tables_MIRROR.py` | prints every table above from the json files | — |
