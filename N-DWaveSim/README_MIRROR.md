# FlavorRenderersCLAUDE — ZRL flavor renderers beyond Stage 1

Parent project: `FormConstantsCLAUDE\ZeroRecursiveLatticeCLAUDE` (Stage-1 `flavor_renderer_MIRROR.html`
is the 2D ancestor). Specs live here; the parent build spec's §0 / §4 / §8 apply verbatim.

| file | what |
|---|---|
| `ZRL_ND_flavor_renderer_spec_MIRROR.md` | spec for the N-D renderer (cube + sphere first) — the contract for `flavor_renderer_3d_MIRROR.html` |
| `ZRL_circle_flavor_renderer_spec_MIRROR.md` | spec for the circle-mapped (rapidity) renderer |
| `circle_flavor_MIRROR.html` | **built 2026-09-16** — Stage A + B of that spec (Jacobi–Anger tab included); Canvas 2D, no CDN; 14–15-row ledger read back from drawn pixel coordinates |
| `flavor_renderer_3d_MIRROR.html` | **tier 0 build, 2026-09-15**, extended 2026-09-16 — cube + sphere, p-norm sweep, isomers, rotor orbit, dimension table, the three signature spheres, vertex map, 2D unfolder; 24-row live ledger + per-frame view rows |
| `test_flavor3d_oracle_MIRROR.py` | pytest oracle: every closed form in spec §2 + a numpy port of the app's mesh |
| `wave_idecomp_MIRROR.html` | **built 2026-09-16, v2 2026-09-17** — 1D complex field drawn in 3D (x = propagation, y = Re, z = Im); Schrödinger / EM, plane + packet sources, **audio (wav/mp3) + synth input**, **speed v with β/γ/ψ**, **δ polarisation slider + quarter-wave ghost**, **editable cyclic gradients**, **WebM export**; 25-row live ledger. Three.js via CDN. |
| `ZRL_wave_idecomp_spec_MIRROR.md` | the build contract for that app (§2 verified formulas, §3 do-not-implement, §9 roadmap) |
| `ZRL_wave_findings_MIRROR.md` | the verified canon additions behind it (ς Fork 1, i^n j, ℤ₂⊂ℤ₄⊂U(1), conj = reciprocal, the Co ≤ 1/√D proof, the Huygens deep dive, the polarisation ellipse, paraxial ≡ Schrödinger) |
| `ZRL_laser_project_sketch_MIRROR.md` | sketch (not a spec) for the laser/lens project |
| `wave_membrane_MIRROR.html` | **built 2026-09-17, extended 2026-09-19 … 2026-09-26b** — 2D real membrane, leapfrog FDTD, Three.js surface. **The app where the CFL row is LIVE.** Boundaries (Dirichlet / Mur-absorbing / Neumann / periodic), 9 flavor domain shapes (incl. rotatable square), uniform/slab/lens media, 5- and 9-point stencils, square + triangular lattices, **domain radius slider**, **grid ↔ edge alignment block** (every aligned angle, optimal angle, optimal grid flavor, edge colouring, measured ledger row), vertex sources with a 3σ wall inset, full screen / photo mode, typed number boxes, **editable diverging gradient**. |

## 2026-09-26c — the space-time grid: light cone, dispersion, points per wavelength, N to 2048

**Author decisions (26/09), binding for code:** the tick-rate symbol is **TR** (v4's `h` is
retired for it; **h stays the grid spacing** in code, and long descriptive names are
preferred to single letters). **ρ = grid resolution** in these apps (not v4's radial ρ).
"C_max = c = 1" in the ℝ⁴ list was a 1D statement — its N-D generalisation is below.

### The N-D generalisation of "C_max = c = 1" (measured; `bench/spacetime_limits_MIRROR.py`)

One tick moves information one **bond**. After m ticks an impulse fills exactly the stencil's
**reach polytope** of radius m (convex hull of the neighbour offsets) — the lattice's own
light cone, speed **u₀ = h·F** (one bond per tick; F = 1/Δt). A wave of speed c fills a ball of
radius c·t, which must fit inside: **c ≤ u₀ · inradius(reach polytope)** — necessary (CFL 1928).
The sharp limit is von Neumann's. So the canon's "C_max = c = 1" becomes **c = Co·u₀ with
Co ≤ C_max ≤ inradius**, and D = 1 (segment, inradius 1) is the one case where u₀ = c_max:

| stencil | reach polytope | inradius (necessary) | sharp (von Neumann) | |
|---|---|---|---|---|
| 1D 3-point | segment | 1 | 1 | coincide |
| 2D square 5-point | diamond | 0.707107 | 0.707107 | coincide — the canon's r/R "signature" |
| 2D square 9-point | square | 1 | 0.866025 | **gap 0.134** |
| 2D triangular | hexagon | 0.866025 | 0.816497 | **gap 0.050** |
| 3D SC 7 / BCC 8 / FCC 12 | octahedron / cube / cuboctahedron | 0.5774 / 0.5774 / 0.7071 | 0.5774 / 0.5774 / 0.7071 | coincide |
| 4D tesseractic 9-point | 16-cell | 0.5 | 0.5 | coincide |

(Von Neumann limits from a Brillouin-zone scan polished by a local minimiser — the raw scan
misses the irrational extremal k of BCC and of the triangular K point.) Where they coincide a
mode exists that puts every neighbour in antiphase at once; the triangular lattice's best
is the three-colouring (cos k·d = −1/2), and the 9-point's diagonals carry only 1/6 weight, so
their cones overstate the speed. **It matters here directly:** it is the Co limit, and the
app now measures the cone (below) and draws it.

### New in `wave_membrane_MIRROR.html`
- **Space-time grid (ZRL) readout.** With the Listener's room scale (L metres over 2R cells,
  c = 343 m/s): ρ = 2R/L cells/m, h, Δt = Co·h/c, F = 1/Δt, the floor **F_min = cρ/C_max**,
  Co/C_max, u₀ = h·F and c = Co·u₀, reach polytope inradius vs sharp limit, cost = cells × F
  (∝ ρ³), and the MEASURED cost on this machine (ms/step, ms/draw, fps, simulated time per
  second). The N note turns the measurement into "~N for 60 fps / 30 fps here".
- **Ledger 7d — the lattice light cone, measured (exact).** Impulse source: after m ticks the
  field is bitwise zero outside the reach polytope of radius m and the support radius equals
  m while the cone is clear of the walls (tip underflow below 1e-290 is reported, not failed);
  also the share of Σu² inside the wave's disk Co·m + 1.5 (99.8 % at m = 40). Optional overlay
  draws the polytope and the disk around the source.
- **Accuracy — points per wavelength.** PPW of the active source (drive Co/f; pulse 2.07σ =
  the shortest λ with ≥ 1 % spectral amplitude; impulse → grid limit; mode 2L/√(m²+n²)), the
  dispersion relation's mean / worst phase error, anisotropy and group error for the live
  lattice + stencil + Co, the lag after one domain crossing, a drawing (continuum wave vs the
  grid's, sampled at the PPW points), and an overlay line.
- **Drum mode (m, n)** (was (1,1) only) and **ledger 9c — dispersion, measured:** an exact
  eigenmode obeys u(t+1) + u(t−1) = 2 cos(ωΔt) u(t), so the field's own history at an
  antinode gives ω with no fit; compared with the dispersion relation (must agree to rounding)
  and with the continuum (their ratio IS the phase-speed error at that PPW).
- **N slider 48 … 2048** (was 320 — a UI choice from the first build, no technical limit),
  typed mesh index (a plain array of 25 M indices at N = 2049 was the reseed bottleneck).

### Points per wavelength: how far, and which configuration (dispersion relation, measured live)

PPW = λ·ρ is not a property of the grid alone: at fixed N, PPW × (wavelengths across the
domain) ≈ N per axis, so "more PPW" costs N³ in 2D like everything else. What the
configuration controls is the accuracy per PPW. Phase-speed error falls as 1/PPW², and — the
leapfrog's time error having the opposite sign to the space error — it also **falls as Co
approaches its limit**. At PPW 8:

| stencil, Co/limit | phase error mean / worst | anisotropy | group error mean | PPW for 1 % / 0.1 % (worst dir.) |
|---|---|---|---|---|
| square 5-point, 0.5 (app default) | 1.61 % / 2.25 % | 1.3e-2 | 4.8 % | 12.0 / 37.9 |
| square 5-point, 0.95 | 0.79 % / 1.44 % | 1.3e-2 | 2.4 % | 9.6 / 30.1 |
| square 9-point, 0.95 | 0.85 % / 0.86 % | 2.8e-4 | 2.6 % | 7.5 / 23.1 |
| **triangular, 0.95** | **0.39 % / 0.40 %** | **6.9e-5** | **1.2 %** | **5.2 / 15.7** |
| 4th-order square (2,4), 0.25 — *not in the app* | 0.07 % / 0.14 % | 1.5e-3 | 0.4 % | 5.1 / 8.6 |

Measured live (ledger 9c, drum mode (40,40), PPW ≈ 5.6): −1.32 % at Co = 0.5, **−0.13 %** at
0.976 of the limit — the prediction to 2e-16. Cost of equal accuracy (∝ PPW³ · work / Co at
equal point density): the triangular lattice near its limit is ~13× cheaper than the square
5-point at Co = 0.5 for a 1 % target. **Best in the app: triangular lattice, Co ≈ 0.95 of its
limit** (0.776). Beyond it: a 4th-order stencil for sub-0.1 % work (its error *grows* with Co —
the time error no longer cancels). Mode sweep: 84 configurations (2 stencils × 2 radii ×
Co 0.2/0.5/0.69 × modes up to (64,64) ≈ PPW 2): all PASS, max |Δcos(ωΔt)| = 2.2e-16.

### Languages, and the GPU (`bench/`)

Same kernel, all implementations checked to 13 digits (float64). 5-point, M cell-updates/s:

| N | JS (app) | C++ 1T | C# safe | C# unsafe | C# 4 threads | C 4T | C float32 4T |
|---|---|---|---|---|---|---|---|
| 1281 | 193 | 449 | 277 | 360 | 899 | 1721 | 4792 |
| 4097 | 192 | 403 | 241 | 331 | 1016 | 1174 | 2965 |

C++ = C (same compiler). C# (.NET 8) is 1.3–1.9× the JS single-threaded and 3–6× with
`Parallel.For` (`sweep_lang_MIRROR.csv`). Unity's Burst compiler (LLVM, SIMD) should land near C — not measurable here.
**Correction:** the earlier C float32 rows for the 9-point and triangular stencils were
double arithmetic in disguise (`2.0/3` is a double literal); fixed with `(real)` casts and
re-timed (1T 500–820 M, 4T 1.3–1.8 G; `sweep_MIRROR.csv` updated).

**GPU:** `bench/gpu_bench_MIRROR.html` — the app's update in WebGL2 fragment shaders (R32F
textures, ping-pong, the field never leaves the GPU) next to the app's JS path, on the
viewer's machine. Validated here on a software GPU (SwiftShader) for **correctness only**: the
sum of u² after 200 steps at N = 101 is 15.40773704 / 18.56107341 / 15.32615048 for 5-point /
triangular / 9-point — identical to strict float32 arithmetic on the CPU
(`f32check_MIRROR.mjs`), i.e. 3.9e-7 / 4.3e-7 / **1.9e-5** from float64. The 9-point stencil
loses ~40× more in float32 (its weights 2/3, 1/6, −10/3 cancel), so a GPU build favours the
triangular lattice on precision as well. The exact-symmetry rows would need float32-scaled
tolerances on a GPU; the light cone stays exact (zero is zero in any precision).

### Verification of this round
Regression **1170 configurations** (the 780 above + the impulse source): 0 FAIL on any row,
no JS errors; the light-cone row PASSes in all 390 impulse configurations (support radius = m
exactly, 0 cells outside). Mode sweep 84/84. GPU page: correctness as above, benchmark path
exercised end to end.

## 2026-09-26b — grid ↔ edge alignment (wired and measured), six fixes, and how far N goes

**Provenance.** The local session hit its usage limit halfway through this build; it was
finished in a cloud session from the uploaded snapshot. At that point the panel controls and
the pure `ALIGN-BEGIN…END` section existed, nothing was wired and nothing had been run. The
slab/lens bug recorded as "not fixed" in the 26/09 entry below **had** been fixed in code by
the local session; re-verified here (see fixes).

**The block (Domain → grid ↔ edge alignment).**
- *aligned =* a straight lattice row, or a row **or** the shortest diagonal ("acts smooth":
  period √2 / √3, silent for λ > 2d at every incidence — measured on the triangular lattice
  26/09; the square lattice's 45° case follows from the same grating rule, unmeasured).
- *dropdown:* every rotation in 0–90° with ≥ 1 aligned edge — the exact enumeration, not a
  scan — with row/diag counts, an ALL marker, and `≅ x°` on rotations that give the same mask
  (lattice turn × shape turn, and mirror; e.g. every pentagon alignment on the triangular
  lattice is ONE configuration, since gcd(72, 60) = 12 and 6 = 12 − 6).
- ‹ prev / next › and **A** (shift = back); **optimal angle**; **optimal grid flavor** and
  **G** — the lattice + angle aligning the most edges, the current lattice kept on a tie.
- per-edge note: class, period d, silence threshold λ > 2d against the drive λ, the rule
  `edge k lies along lattice direction α iff rot ≡ α − 90° − 360°k/n (mod 180°)`.
- *colour the domain edges:* row green, diag blue, rational yellow, irrational red, drawn on
  top of the surface at the clamped rim (both caps in prism mode).
- *vertex source inset → 3σ from BOTH walls*, along each vertex's bisector (distance
  3σ / sin(α/2)), plus a readout of how much of the seeded pulse the mask cut at t = 0.

**The answer: which shapes can be fully grid-aligned.** Exact, from a Fraction scan of all
1801 slider positions, confirmed on two independent masks:

| lattice | every edge a straight row | every edge a row or shortest zigzag ("acts smooth") | never fully (best) |
|---|---|---|---|
| square | square @ 0°, 90° · rhombus(90°) @ 45° (= the square) | + octagon @ 0°, 45°, 90° (4 rows + 4 × 45°) · square @ 45° and rhombus(90°) @ 0°, 90° (the diamond, all diag) · rhombus(45°) @ 22.5°, 67.5° (2 + 2) | triangle 1/3 · pentagon 1/5 · hexagon 2/6 · dodecagon 4/12 |
| triangular | triangle, hexagon @ 30°, 90° · rhombus(60°) @ 30°, 90° | + triangle, hexagon, rhombus(60°) @ 0°, 60° (all zigzag) · square @ 0°, 30°, 60°, 90° (2 + 2) · **dodecagon @ 0°, 30°, 60°, 90° (6 + 6)** · rhombus(30°), rhombus(90°) @ 15°, 45°, 75° | pentagon 1/5 (@ 6° + 12°k, all congruent) · octagon 2/8 rows, 4/8 smooth |

The circle has no edges. "Straight row" means the direction; **exact conformance** (the mask
IS the shape's lattice points) additionally needs the radius snap: hexagon integer side,
triangle side a multiple of 3, rhombus(60°) an **even** side (fix 5 below), square
half-integer R. The pentagon is the one polygon here that no lattice, no rotation and no
definition rescues: its edge directions differ by multiples of 36°, which never equal a
difference between two lattice row/diagonal directions (45°, 90°, 135° square; 30°…150° in
steps of 30° triangular).

**Verification.**
- *Ledger row 7c* measures each edge's period from the mask — the shortest integer lattice
  shift that maps its boundary cells onto boundary cells (exact, no tolerance) — against
  the class prediction. Regression **780 configurations** (2 lattices × 9 shapes incl.
  rhombus 30/45/60/90 × every listed angle + an off-angle × radius 1 / 1.5 / 0.5 × pulse and
  wall-inset vertex sources × 60 steps): **686 PASS, 94 n/a, 0 FAIL; 0 FAIL on any row; no
  JS errors.** Mirror 237 PASS, conformance 26, 8-fold 45, 6-fold 32.
- *Independent Python reference* (`align_reference.py`, shares no code with the app):
  Part A — all **1476** entries of the JS enumeration over 294 configurations (incl.
  rhombus θ = 20…90° in 0.5° steps) against an exact Fraction scan: **0 mismatches**. Part B —
  a half-plane mask (not the app's polar fold) at all **37** fully-aligned configurations:
  every edge's measured period is 1 or √2 / √3 as predicted, **0 mismatches**.
- *3σ wall inset*, σ = 3: the mask's cut of the pulse at t = 0 falls from 25.8–28.4 %
  (triangle), 14.7–15.8 % (rhombus 60°), 5.6–5.7 % (pentagon), 2.3–7.0 % (hexagon) with
  the radial inset to **≤ 0.004 %** for every shape, clearance 2.85–3.18σ. The radial
  figures reproduce experiment B's 23–28 % / ~4 %.

**Fixes, each measured before and after.**
1. *(introduced and caught in this build)* `alClass` returns the lattice direction as
   `{a, b}`, which overwrote the edge ENDPOINTS `a, b` — every non-irrational edge lost its
   geometry (NaN outline, every period "too short"). Renamed `(da, db)`.
2. Start-up overlay read "diverged (max|u| = —)": `S.amax` was undefined until the first step
   (and stale after a reset). `seed()` now sets it.
3. `'square'` ignores the rotation slider, but the 8-fold, staircase and (1,1)-mode rows
   tested rot = 0, so an exact square with the slider moved was labelled "staircase". Now
   rotation-independent. The 8-fold orbit also now covers n = 4, 8, 12 at every multiple of
   180°/n (the local session's rotatable square was reported "not 8-fold" at 0°).
4. The conformance row accepted 30° only, while `domR()` snaps at 30° and 90°. Now
   `atConfRot()` in both.
5. Rhombus(60°) conformance needs an **even** side — its vertices ±(s/2)(e₁ ± e₂) are lattice
   points only then — but the snap allowed odd s: mask s² against the formula (s+1)² at
   radius 0.5 and 1.5 (s = 39, 117). Snap now even; conformance 26 PASS, 0 FAIL.
6. Mirror row with vertex sources (pre-existing: the radial inset already failed on the tri
   hexagon at radius 0.5, 7.16e-2). A source ON the mirror axis lands on a rounding tie
   (x = 0 in an odd triangular row, or a half-integer offset), and float noise —
   cos 90° = +4.8e-15 but cos 270° = −1.4e-14, plus the `c + d − c` round trip in
   `snapInside` — broke the tie one way for +y and the other for −y. Quantising to 1e-9 in
   `vertices()` and `snapInside()` fixed every case.
- Slab/lens (the 26/09 bug, fixed locally before the snapshot): **verified** — the slab now
  covers 42.2 / 41.8 / 42.6 % of the square and 41.1 / 40.7 / 40.1 % of the hexagon at radius
  1 / 0.5 / 0.3 (expected ≈ 42 %), and the lens lies inside the domain at every radius.
- The `≅` congruence tags: 122 tagged pairs checked by comparing the multiset of squared
  distances to the centre node (invariant under rotation and mirror): 0 not congruent.

### Porting, and how far N goes (measured; scripts in `bench/`)

**The kernel** is the app's own update — mask, per-cell Co² (`CC`), damping — for the 5-point,
9-point and triangular stencils, in JS (V8, Chrome's engine), numpy, numba and C (gcc -O3,
OpenMP). All six implementations give the same field to 13 significant digits after 200 steps
(float32: 4e-7 relative). Cell-updates per second, 5-point, this container (Xeon 2.1 GHz,
4 vCPU; note its 260 MB L3 flatters the large-N rows compared with a desktop):

| | N = 161 | 321 | 1281 | 4097 |
|---|---|---|---|---|
| **the app's JS kernel** | 208 M | 202 M | 193 M | 192 M |
| numpy (vectorised) | 59 M | 69 M | 66 M | 34 M |
| numba, 1 thread | 1246 M | 798 M | 470 M | 388 M |
| C float64, 1 thread | 698 M | 667 M | 467 M | 371 M |
| numba, 4 threads | 2036 M | 3317 M | 1639 M | 1293 M |
| C float64, 4 threads | 1815 M | 1805 M | 1721 M | 1174 M |
| C float32, 4 threads | 3515 M | 4950 M | 4792 M | 2965 M |

The triangular and 9-point stencils mostly run at 80–90 % and 60–80 % of the 5-point rate
(outliers both ways; full table: `bench/sweep_MIRROR.csv`). JS float32 is *not* faster —
V8 computes in doubles.

**Verdict.** Plain numpy is a *regression*: 3–6× slower than the app's JS (the 3D figures of
2026-09-21 said the same: 26.2 M vs 34.2 M). A compiled port — numba or C — buys ~2× on one
core at large N (the kernel is memory-bound there), 6–9× on four cores, and 15–38× in
float32 on four cores.
**But the app is not running at the kernel's speed:** its own `stepOnce` measures **50–68 M**
cell-updates/s (N = 161 … 2049), 3–4× below the bare kernel. Profiled at N = 641 / 1281:
`record()` (energy, gradients and peak over the whole grid, every step) is **40 % / 47 %** of
the step, the second full-grid mask pass 3 % / 7 %, and the update loop itself still runs
below the bare kernel's rate (not profiled further). `updateMesh` costs about one more step
per frame (0.47 / 1.65 / 6.9 / 26.8 / 73.8 ms at N = 161 / 321 / 641 / 1281 / 2049, with one
`colorOf` array allocation per vertex). Frame budget at 2 steps/frame: N = 641 ≈ 23 ms
(~43 fps), N = 1281 ≈ 92 ms (~11 fps) — so **~550 (60 fps) to ~770 (30 fps) is the
interactive ceiling as written** (headless CPU time here; the slider stops at 320). Moving
`record()` off the per-step path would recover about half of the step before any port.
Beyond that the win is the GPU (field kept in textures, no per-frame upload of N² vertices);
not measurable here (this container has no GPU) — the order of magnitude is memory bandwidth
over ~20 bytes per float32 update, i.e. ~10 G updates/s at 200 GB/s, **an estimate, unmeasured**.

**Memory.** Fields (U0, U1, U2, CC float64 + mask): 52 MiB at N = 1281, 528 MiB at 4097. The
app's mesh adds ~48 B per vertex (768 MiB at 4097), so the render mesh, not the solver, is
the first memory wall in the browser.

### The ZRL reading: Frame Rate, Δt, ρ — why N is the expensive knob

With the canon's definitions (context transfer v3/v4, `Zero-Recursive Lattice.md`):
**F = 1/Δt**, **C_max** (v4: Δf merged into it), and **ρ_g = 1/h** (v3: grid resolution,
"purely visual for now… revisit-flagged"). In these apps Δt = Co·h/c, so

    F = 1/Δt = c·ρ_g / Co        and stability (Co ≤ C_max) gives   F ≥ F_min = c·ρ_g / C_max

**The frame rate is not a free parameter: the resolution sets its floor.** C_max belongs to
the lattice + stencil, not only to D: 1/√2 (square 5-point), √3/2 (9-point), √(2/3)
(triangular); 1/√3 (SC, BCC), 1/√2 (FCC) in 3D. Simulating a time T on a domain of size L
costs cells × frames = (L ρ_g)^D · T c ρ_g / Co ∝ **ρ_g^(D+1)** — N³ in 2D, N⁴ in 3D:
doubling N costs 8× (16× in 3D). Measured wall time for ONE domain crossing (N/Co steps,
Co = 0.5) at N = 4097: 714 s JS kernel, 117 s C 4-thread, 46 s C float32 4-thread.
**Accuracy is set by points per wavelength (PPW = λ·ρ_g), not by Co** (2026-09-17: sweeping
Co 0.1→0.7 moves anisotropy by 5e-5; coarsening moves it by 1e-1) — so the N you need is
≈ PPW × (domain width in wavelengths), and ρ_g is no longer visual: it sets dispersion, the
staircase, and the frame-rate floor at once. That is the canon's revisit, answered.

**Grid flavor, at EQUAL point density** (`bench/dispersion_MIRROR.py`; triangular spacing
√(2/√3) = 1.0746 so both lattices hold one point per unit area; Co at half of each limit):

| PPW = 8 | mean phase error | anisotropy | Δt_max (h/c) | steps / unit time |
|---|---|---|---|---|
| square 5-point | 1.61e-2 | 1.31e-2 | 0.7071 | 1.414 |
| square 9-point | 2.09e-2 | 2.70e-4 | 0.8660 | 1.155 |
| triangular | 1.74e-2 | **9.07e-5** | **0.8774** | **1.140** |

The triangular lattice takes 19 % fewer steps per unit time and is 145× more isotropic than
the square 5-point grid at the same point count (its leading error term is isotropic:
−L/k² spread 3e-9 at k = 0.05, against 1e-4 for 5-point); with the measured per-update cost
(JS: 170 M vs 193 M) it is ~9 % cheaper per simulated second. So for a domain with no edges to
align (the circle), the optimal grid flavor is the triangular lattice on cost AND isotropy.

**Canon flags (not resolved — the author's call).**
1. *Symbol clash:* v4 renames the tick-rate to **h**, but h is the grid spacing in every wave
   app here and in the CFL literature (Co = c·Δt/h). Suggest keeping h = spacing in code.
2. *ρ has two meanings:* v3's ρ_g (grid resolution, used above) and v4's ρ (the
   Schwarzschild-style radial coordinate, g = diag(−v₀²f, 1/f, ρ², ρ² sin²θ)). If the latter
   was meant, the connection is the angular-CFL singularity (Δt ∝ ρ·Δθ → 0 at the centre).
3. *"C_max = c = 1" (the ℝ⁴ list in `Zero-Recursive Lattice.md`)* is the 1D / advection
   statement. On a D-lattice the lattice's own causal speed is u₀ = h·F (one cell per tick) and
   the wave's speed is c = Co·u₀ ≤ C_max·u₀ — so the CFL-saturating ("null") wave runs at
   u₀/√D, and u₀ = c ⇔ D = 1. Consistent with v4's "C_max is scheme-dependent".
4. *Local frame rate:* a slow region (the slab's c/2) only needs half the global frame rate.
   Giving each refinement level its own Δt is standard (Berger & Oliger 1984, adaptive mesh
   refinement with time subcycling) and is the discrete analogue of the canon's lapse
   t₁ = t₀√f. It is the principled way past the ρ_g^(D+1) wall: refine only where PPW demands.

**Open / next.** The lattice segment → a separate app copied from this one (the author's
decision): edge-sharing domain cells whose shared edges act as thin partial mirrors — the
`sheet1d.py` mass-loaded sheet (t = 1/(1 − iμΩ²/(2 sin k)), one knob; thickness for a second).
Cheap speed-ups before any port: run `record()` every k steps (it is 40–47 % of a step), drop
the second mask pass, write colours straight into the buffer. GPU solver: not started.

## 2026-09-26 — jagged edges, blended grids, edge-sheets (measured; NO app code changed)

Question round on the triangular lattice. Two measurement agents (experiments B and C, run
2026-09-26) plus my own checks. Scripts are in the session scratchpad: `straight_edges.py`,
`blended2.py`, `sheet1d.py`, `medium_geom.py`, and the agents' `wallcore.py` / `refine.py` family.

**Straight edges (exact rule, every integer rotation checked).** On the tri lattice, edge k of a
regular n-gon lies on a lattice row iff `rot + k·360/n ≡ 30 (mod 60)`. Pentagon: at most 1 edge,
every 12° from 6° (42° is one of them). Hexagon/triangle: all edges at 30°/90°. Octagon 2, dodecagon 6.

**What a staircase wall does to a wave (experiment C, single half-plane wall, one reflection).**
- A wall along lattice direction (a,b) repeats every `√(a²+ab+b²)` cells (the Loeschian norm): it
  is a shallow sawtooth grating, one row high. Checked for 9 directions.
- Reflected field = mirror reflection + side beams at `k_t + 2πm/d`, nothing else (≥ 99.98 %).
- **Side beam closed** (`λ > d(1+|sin θ|)`): the wall reflects totally (|R| = 1.0000) exactly like a
  smooth wall shifted ~0.35–0.45 cells outward (= the static extrapolation length). Spurious
  energy < 1e-7. Zigzag walls (30° to the rows, d = √3) are silent for every resolved λ (≥ 4).
- **Side beam open**: a real stray beam at the grating angle, strength
  `η_m ≈ 3 cos θ_i cos θ_m / (m² λ²)` (λ in cells) — measured/predicted median 1.07 over 39 cases.
- Grazing: an OPEN beam weakens toward grazing (∝ cos θ_i, not Rayleigh's cos²), but grazing also
  OPENS beams that are closed head-on — (4,1) wall at λ = 8: 1.7e-8 at 0°, 9.3e-3 at 75°.
- Irrational wall angles add weak long-period components that radiate ~1e-3 even at λ = 14.

**Why vertex sources make straightness look important (experiment B, 4-level refinement).**
Refuted: "vertex sources are more sensitive" (jagged/aligned error ratio is SMALLER for them) and
"grazing along-wall travel". Actual causes: (1) timing — they touch walls at t ≈ 0; (2) the source
inset `max(2, ceil(1.5σ))` is RADIAL, so the true wall distance is ×cos(π/n): 0.87 (hex), 0.5
(triangle). The mask clips 4 % (hex) / 23–28 % (triangle) of the pulse at t = 0, filling the field
with grid-scale ripple (1–32 % of energy vs 0.013 % uncut) that does not converge — on straight
walls too. Same amount either way; straight walls line it up in tidy fringes, jagged ones don't.
76–97 % of the jagged-wall error is the coherent outward wall shift, which converges at 1st order.

**Blended grids (`blended2.py`).** No periodic mesh, whatever tiles it blends, can make every edge
of a regular n-gon straight at arbitrary size unless n ∈ {3, 4, 6}. Straight mesh lines must run in
lattice directions (pentagon fails: the cross-ratio of its directions is φ), and node spacings along
them must be commensurate (octagon fails by √2, dodecagon by √3). Tilings with octagons/dodecagons
(4.8.8, 4.6.12, 3.12.12) have no straight lines at all; kagome (hexagons + triangles) has only the
three row directions. What does work: domain-fitted meshes — pie slices (any n; exact n-fold
symmetry, one element shape, pentagon element 54-54-72, c·dt ≤ 0.79 edge) or zonogon rhombus
tilings (even n: octagon squares + 45° rhombi, decagon Penrose rhombi, dodecagon squares + 30°/60°).
Both need an FEM (cotangent) Laplacian, which reduces EXACTLY to the app's `(2/3)(Σ−6u)` on the
equilateral lattice (checked). Rhombus meshes cost timestep: c·dt ≤ 0.69 / 0.58 / 0.50 edge.

**Edge as a partial mirror (`sheet1d.py`).** One row of nodes with `CC /= (1+μ)` — the app's own
per-cell c², i.e. Berglund's `tcc` table — is a mass-loaded sheet: `t = 1/(1 − iμΩ²/(2 sin k))`,
Ω² = 4 sin²(k/2), matched to 2.3e-4; |r|²+|t|² = 1; r and t always 90° apart; |t| = cos(phase t).
A single sheet has ONE knob; a separate transmitted-phase knob needs thickness (delay).

**BUG FOUND, not fixed (question round):** `buildDomain` compares the array ROW INDEX `j` with the
physical `ir`-scaled `slabJ` / `lensJ`. Measured: at radius 1 the slab covers 92 % of the domain and
the lens sits at the bottom edge; at radius ≤ 0.5 the slab fills 100 % (no interface at all) and the
lens is entirely outside the domain. Fix: use the physical `dy` from `cellXY` in both lines.

## 2026-09-22 — VOXEL FLAVORS in the 3D app (SC / BCC / FCC)

**I had the CFL limits wrong, and shipping them would have blown the app up.** Last turn I
quoted BCC Co_max = 0.7334 and FCC = 0.7218 from my own scan. A verification workflow refuted
both; I then reproduced the refutation independently before touching any code.

Root cause, now precise: I sampled **physical k over [−π,π]³**. The Brillouin zone is
**κⱼ = k·aⱼ over [−π,π]³**, and for BCC/FCC the reciprocal lattice is *larger*, so a physical-k
box of half-width π misses the minimum. Re-running my old method reproduces 0.7334 / 0.7218
exactly; the zone scan gives λ_min = −12 / −8.

| | Z | coef 2D/Z | λ_min | **Co_max** | my wrong claim | at that Co |
|---|---|---|---|---|---|---|
| SC | 6 | 1 | −12 | **1/√3 = 0.5773503** | 0.577389 | ok |
| BCC | 8 | 3/4 | −12 | **1/√3 = 0.5773503** | ~~0.733423~~ | **diverges** |
| FCC | 12 | 1/2 | −8 | **1/√2 = 0.7071068** | ~~0.721791~~ | **diverges** |

BCC reaches −12 because **a₁+a₂+a₃ is itself a nearest-neighbour vector**, so all four distinct
cosines hit −1 together at κ = (π,π,π). Confirmed three ways: closed form, a dense zone scan,
and empirically (stable at 0.98×, divergent at 1.02×, on the actual `np.roll`-indexed array).

**The honest cost picture, which reverses what I told the author last turn.** BCC buys **no
timestep at all** over SC and costs 1.73× the work per simulated second; FCC gains √(3/2) but
does 2× the neighbour work, for 2.31×. **Neither is a speed win.** Their payoff is *isotropy*,
and that is real: directional spread of the numerical wave speed at 5 points per wavelength is
4.53% (SC), 2.98% (BCC), **1.14% (FCC)**. The panel says this rather than implying a free win.

**Implementation.** Same cubic index array; only the index→position map and the neighbour
offset list change — exactly as the 2D triangular lattice did. Bases normalised to
nearest-neighbour distance 1, so all primitive vectors are unit length and only the angles
differ (90° / 109.4712° / 60°). Verified |aᵢ| = 1 to 1e-16 and Σ dₘ⊗dₘ = (Z/3)I to 9e-16.
Array footprint (largest sphere per unit N): 0.5 for SC, **1/√6** for both BCC and FCC.

**Three traps the verification caught that I would otherwise have hit:**
1. **FCC basis ordering is handedness-sensitive.** Written as a₁=(1,1,0), a₂=(1,0,1), a₃=(0,1,1)
   the basis is **left-handed** (det = −1/√2), which mirrors every marched triangle and renders
   the surface inside-out under normal shading. The order used — a₁=(0,1,1), a₂=(1,0,1),
   a₃=(1,1,0) — is right-handed (det = +1/√2), checked at startup.
2. **Areas and normals are NOT affine-invariant** — only the combinatorics and volume *ratios*
   are. An index-space normal on FCC is off by 159° (essentially reversed). The marching code
   computes positions through the basis and takes normals from the resulting **world-space**
   triangles, so it was already correct, but it is now commented so nobody "optimises" it back.
3. **Marching tetrahedra survives the shear**: each of the 6 tets is exactly 1/6 of the
   parallelepiped on all three lattices, and the area error stays second order (SC C = 0.2603,
   BCC 0.1840, FCC 0.1750 — BCC/FCC are ~15–17% *more* accurate at equal point density).

**The octahedral orbit works on all three.** BCC and FCC are still cubic Bravais lattices with
the full Oh point group, but in the primitive index basis the 48 operations are **M = B⁻¹RB**,
which must come out integral — that integrality is *checked at build*, not assumed, and the row
disables itself with a reason if it ever fails. Measured **1.94e-16 / 2.21e-16 / 3.33e-16**.

One more gate the sweep exposed: the BCC/FCC **array is a rhombohedron**, not octahedrally
symmetric, so an inscribed cube whose corners reach past it is **clipped unevenly** and the
field is then correctly not symmetric. Same class as the 2D rhombic array breaking the mirror
row. The row now tests **mask symmetry directly** under the 48 orbit rather than inferring it
from the shape name.

Regression: **48 configurations** (3 lattices × 4 shapes × 2 radii × 2 sources), zero failures,
orbit passing 32, clipped case gated with its reason in 4. Surfaces render on all three.

## 2026-09-21c — W / D / B shortcuts

`W` toggles the wireframe, `D` cycles the domain shape, `B` cycles the boundary mode, in
**both** the membrane and the 3D app. `Shift` reverses the cycle. Each fires a short overlay
flash naming what changed, so a keypress in photo mode (where the panel is hidden) still tells
you what it did.

Implemented as `cycleSelect(id, back)`, which advances a `<select>` and dispatches `change` —
so the existing handler runs and there is exactly one code path per control, no duplicated
state updates. Disabled options are skipped and the wrap is modulo, so it cannot land out of
range. Verified: membrane square→circle→n3 with Shift+D returning to circle, boundary
dirichlet→mur, wireframe on with the flash; 3D cube→sphere→tet likewise. Also verified that a
keypress **inside a number box does not hijack** — the handler already guards on
INPUT/SELECT/TEXTAREA, and the guard was re-tested here.

The **1D app does not get D or B**: it has neither a domain shape nor a boundary mode. It has
the F/H/O/space/R set from the previous pass.

## 2026-09-21b — 3D cube bug fixed, wireframe decoupled, 1D presentation

**The 3D container cube was genuinely wrong, and the author's report was precise.** The cube
was built as a Platonic solid of *circumradius* R (half-width R/√3) while `cubeFull()`
overrode R to the grid half-width — two meanings of R in one variable. Above radius 0.999 the
drawn hull came out **√3 = 1.732× smaller than the cube being simulated**, which is exactly
"the cube shown is not the cube being simulated, and it works at smaller R". A second fault
rode along: crossing the 0.999 cutoff resized the *simulated* cube by **1.84×**.

Fixed by giving the cube **half-width semantics**, matching the 2D app's square. Verified: R
now equals the measured half-width at every radius (15.3→15, 24.4→24, 32→32), and the cutoff
jump is 2 cells instead of 1.84×. The hull is drawn from the same half-width.

**Membrane wireframe decoupled from the grid.** `material.wireframe` draws every triangle
edge, so its density *is* the grid's — there is no knob. Replaced with an independent
lattice-line overlay: every k-th row and column, as segments between adjacent in-mask cells,
following the surface and respecting the mask. Measured 7,582 / 1,914 / 652 segments at steps
1 / 4 / 12, independent of N. A "filled surface" checkbox allows wireframe-only.

**1D app** gained the same presentation contract as the other two: full screen (with the
timer-armed photo-mode fallback), photo mode, overlay toggle, and F/H/O/space/R hotkeys.

### Answers recorded (questions, not builds)

**Are all square/saw/triangle waves equal? No.** The Gibbs overshoot is **8.949% of the jump**
(= 17.898% above the plateau for a ±1 square) and it **does not decay**: 17.947% at 25
harmonics, 17.898% at 2001. More circles never remove a corner. But the severity scales with
the order of the discontinuity: saw and square jump in *value* (harmonics ~1/k, Gibbs), the
**triangle is continuous and jumps only in slope** (~1/k², measured overshoot −0.020% at 2001
harmonics — no Gibbs). So triangle waves really are more "equal" to each other than squares
and saws are. *(Correction recorded: I first compared the 17.9% measurement against the 8.949%
constant and called it a failure — two different normalisations, jump-relative vs
amplitude-relative.)*

**Can a 2D front be square?** Not from the medium alone: a homogeneous second-order
anisotropic wave equation has a *quadratic* slowness surface, so the front is always a smooth
convex **ellipse** — no choice of coefficients gives a corner. What the app *can* already show
is the **grid's own anisotropy**: measured group-velocity diagonal/axial ratio 1.0015 at 40
points per wavelength, 1.041 at 8, **1.211 at 4**. Verified with a direct FDTD run and
sub-cell interpolation that the **diagonal runs ahead**, so the front bulges toward the
diagonals. *(Two corrections: phase velocity gives the opposite sign to group velocity — the
front is carried by the group velocity; and my hypothesis that the app's integer diagonal
sampling flips the sign was wrong — integer 1.0260 vs interpolated 1.0255, they agree.)*

**Can any domain be made grid-aligned?** Not by phase — translation does not change edge
directions. As a *lattice* symmetry the crystallographic restriction allows only n ∈ {1,2,3,4,6}.
**But the author's "composed grid" intuition is right and buys more than the lattice does:**
an n-gon can be a single *tile* of a periodic tiling whose lattice is only 3-, 4- or 6-fold.
Of the 11 edge-to-edge regular/Archimedean tilings, the polygons that occur are **{3, 4, 6, 8,
12}** — so the **octagon is exact in 4.8.8** and the **dodecagon in 3.12.12 and 4.6.12**, even
though 8- and 12-fold are forbidden as lattice symmetries. The **pentagon is genuinely
impossible**: 5 and 10 occur in none of the 11. *(Correction: 5.5.10 does satisfy the 360°
angle condition, so my "no valid vertex figure" claim was wrong — the angle test is necessary,
not sufficient, and no tiling of the plane realises it.)*

## wave_3d_MIRROR.html — NEW APP, 3D isosurface (2026-09-21)

Tooling first, because it was asked and it is measurable. **Three.js stays.** numpy's 3D
FDTD runs at **26.2M** cell-updates/s against the browser's measured **34.2M** — the browser
is *faster*, because plain numpy is memory-bandwidth bound (seven full array passes per
step). And matplotlib is disqualified twice: **2.97 fps** at 20k triangles, and
`mpl_toolkits.mplot3d` has **no depth buffer** — it sorts polygons by centroid, so two
nested isosurfaces render in the wrong order. That is a correctness failure, not a speed one.
PyVista/VTK is the real Python contender; it still costs the ledger harness, the domain
predicates and the house UI for no measured gain.

**Marching tetrahedra, not marching cubes.** Marching cubes needs a 256-entry table that
cannot be checked by inspection and whose ambiguous faces can open holes. Six tets per cube
gives 16 derivable cases. Verified: each tet is exactly 1/6 of the cube, they sum to 1, and a
Monte-Carlo sweep puts every interior point in **exactly one** of them — a true partition.
The marched level set of a radial field converges to 4πr² (9.1e-3 → 2.3e-3 over a 2×
refinement).

**Domains** are the Platonic solids as intersections of half-spaces, with face planes derived
from each solid's own vertices — never from the dual's coordinates, and inward normals
**flipped rather than dropped**, both lessons carried over from the tier-1 renderer. Measured
inradius/circumradius: tet 0.351 (1/3), oct 0.589 (1/√3), **dodecahedron 0.796 and
icosahedron 0.800 — the dual pair sharing its face entry**, exactly as the closed-form table
says.

**Live rows:** CFL at **1/√3 = 0.577350** (so Co ≤ 1/√D now has two live measurements, not a
proof with one); the full **octahedral group, order 48**, measured at **1.62e-16**; energy;
the isosurface; the (1,1,1) cavity mode; and two that had to be rebuilt.

**Four of my own errors, each caught by a row rather than by inspection:**
1. Every shape was **inscribed**, so the outer grid faces sat outside the mask and were zeroed
   after the boundary update — the **absorbing boundary silently never acted**, and the
   Huygens row depends on it. The cube at full radius is now the whole grid, as the 2D app's
   square is.
2. My `innerR` cube branch returned `R` via a no-op `R/√3*√3`. The generic
   `R*min(face offset)` already gives R/√3 correctly; the special case was both wrong and
   unnecessary.
3. **The front-speed row tested the wrong model.** It expected the shell peak at ct and read
   0.8125 on a correct solver. For spherical symmetry `w = ru` obeys the 1D wave equation, so
   `u(r,t) = [(r+ct)f(r+ct) + (r−ct)f(r−ct)]/2r` exactly — and the outgoing term carries a
   factor **(r−ct) that vanishes at the front**, putting the peak near ct−σ. Only
   asymptotically, though (at σ=4 the offset is still 1.39 cells off at ct=12), so the row now
   scans the closed form rather than using the approximation. It passes at σ = 1.5/2.5/4 with
   deviations of 0.43/0.21/0.39 cells. That (r−ct) factor *is* the sharp front — the same fact
   the Huygens row measures, seen analytically.
4. **The Huygens row was wrong twice.** Its window ran 90 steps past the first-order Mur
   leakage returning from the wall, so it measured leakage (3.67e-2, a mere 4× below 2D) — in
   2D the real wake swamps that leakage, but in 3D the true residual is tiny so the leakage was
   all of it. Capping the window helped (1.19e-2), but refinement then showed only a 7% drop
   where dispersion would give 36%. The reason: **the analytic tail at this window is already
   1.64e-3**, so the 3e-5 of the offline run — which used a far longer window — was never the
   right expectation here. The row now **correlates the probe trace against the exact
   solution**, which is self-calibrating and tests arrival time, front shape and tail at once:
   **0.997688**. The Huygens content is now a comparison of two measured numbers — the exact
   3D answer 1.64e-3 against the 2D app's 1.35e-1, ~82× — rather than a guessed threshold.

Six shapes × two radii, zero failures. **Environment note:** the desktop preview pane
throttles `requestAnimationFrame` to ~1 fps, so the play loop cannot be timed there; stepping
synchronously gives 135 steps in 1.1 s at N=65. Same class as the fullscreen block — it is the
pane, not the app.

## wave_membrane_MIRROR.html — tier C, waveforms, reverb v1 (2026-09-20b)

**Tier C: the triangular lattice is not a new data structure.** It is the same N×N
`Float64Array` with a different neighbour list — (±1,0), (0,±1), (1,−1), (−1,1), all at
distance exactly 1 — and a different index→position map, e₁ = (1,0), e₂ = (½, √3/2).
Σₘ dₘ⊗dₘ = 3I, so `lap = (2/3h²)Σ(uₘ−u₀)`, verified second order (error falls 4.00× per
halving). **Co ≤ 2/√6 = √(2/3) = 0.816497 — less restrictive than the 5-point square grid's
0.707**, verified stable at 0.98× and divergent at 1.02×.

Two consequences that had to be threaded through everything: the **cell area is √3/2**, not 1
(without the factor the staircase row reads 13.4% low and blames the staircase for it), and
the N×N array is a **rhombus** with inradius (√3/2)c, so the same N holds a 13.4% smaller
domain. My first arithmetic said 0.75c; that was wrong and the test caught it.

**Which flavor on which lattice is a theorem.** The crystallographic restriction (2cos(2π/n)
an integer) allows **n ∈ {1,2,3,4,6}** — the same set as Niven — so triangle, hexagon and
rhombus(60°) conform to the triangular lattice, square to the square, and **pentagon,
octagon, dodecagon and circle conform to neither, ever**. Non-conforming domains use (a),
staircase, as instructed.

**All three tilers conform at rotation 30°**, with one extra condition the test found: the
triangle also needs side L ≡ 0 (mod 3), or its centroid is not a lattice site (L = 12 and 30
conform, L = 20 does not). `domR()` snaps to the conforming radius there, mirroring the
square's half-integer snap, because the slider's 0.005 step lands ~0.13 cells off — enough to
flip boundary lattice points and break the exact count. The ledger **measures** the result:
hexagon 14077 cells = 3L²+3L+1 at L = 68, triangle 406 = (L+1)(L+2)/2 at L = 27.

**New exact invariant: the triangular lattice's own 6-fold orbit.** A 60° rotation is
(a,b) → (−b, a+b) in lattice coordinates — integer, order exactly 6, and it permutes the six
neighbours in one cycle. Measured **3.1e-16 to 4.2e-16**. A triangle gets the 3-fold orbit
(the map applied twice); a square or rhombus gets neither, since 4- and 2-fold are not
lattice rotations, and the row says so. **Checked and NOT true:** this does not detect
conformance — a regular hexagon is 6-fold symmetric at any rotation, so the residual is ~1e-16
at rot 0 and rot 30 alike. Conformance is the cell count, a separate measurement.

**Four bugs the new invariants caught, all mine:**
1. `addBump` bounded its gaussian by an **index-space box**, which is a rhombus in physical
   space — so it truncated the tail further along one axis than another. The 6-fold residual
   read **3.80e-6 = exp(−25/2)**, exactly the 5σ cut. Seeding is O(N²) on reset, not per
   step, so the box is gone entirely.
2. `symRound` is a **square-lattice** trick. The triangular mirror is (a,b) → (a+b,−b), so
   rounding the two indices independently put mirror partners on non-mirror cells and the
   dodecagon's vertex sources broke the mirror row. Fixed exactly: round b symmetrically,
   then a = round(x − b/2) from the physical x — the mirror keeps x and sends b → −b, and
   round(x + b/2) = round(x − b/2) + b identically because b is an integer.
3. `insideNgon` had **no epsilon**. On a conforming lattice the shape's vertices land exactly
   *on* lattice points, and the bound goes through `atan2` and the fold, so two points of one
   orbit straddle the comparison by an ulp. Both the triangle's 3-fold row and the conforming
   count failed while the geometry was right.
4. Four rows were **square-lattice constructions applied to a triangular lattice**: the 8-fold
   octant orbit, "grid-aligned exactly", the (1,1) `sin·sin` eigenmode, and the mirror row on
   a shape the **rhombic array clips** (the rhombus is point-symmetric but *not*
   mirror-symmetric, so a clipped square keeps its bottom-left corner and loses its top-left).
   All four now gate with the reason.

**Band-limited drive waveforms.** k_max = Co/(8f) harmonics, because below ~8 cells per
wavelength a harmonic travels at the wrong speed. At the app's **default f = 0.12 that is
k_max = 1** — not even a second harmonic, so a "square wave" there is a sine in all but name,
and the panel says so. The ledger row is the sharp one: **a linear scheme cannot create a
harmonic that was not injected.** It needed two fixes of its own — an undamped closed cavity
*never* settles (its own modes ring for ever at frequencies unrelated to the drive, which read
as 3.3e-2 out-of-band for a pure **sine**, which no linear scheme could produce by mixing), so
the row requires an absorbing edge or damping; and an absolute tolerance put all four
waveforms within a factor of 3 of it, the signature of measuring a floor. It now measures the
ratio over **two successive windows** and passes when it is *falling* — a decaying transient
falls, genuine mixing would not.

**Reverb v1.** IR capture plus `ConvolverNode`, on the verified LTI result (superposition
2e-15, IR convolution reproduces a driven run to 1.25e-15). The time scale is the one real
decision: dt = Co·h/c with h = L/(2R), so **fs = 2Rc/(Co·L)** — 10770 Hz for a 10 m room at
R = 78.5, shown live because it decides whether the capture is a usable audio rate. Capture
runs chunked through rAF with progress, normalises on the peak, resamples to the context rate,
and gives a stereo IR from the two probes plus an RT60 fit. Load a file, convolve, wet/dry,
record to .webm. **Stated in the panel, not buried: this is a drumhead, not a room** — 2D, so
Huygens fails, and there is no stiffness term.

Regression: **864 configurations** (2 lattices × 6 shapes × 2 rotations × 3 radii × 4 sources
× 3 waveforms), **zero failures**; 6-fold row passes 48, mirror 258, conformance 84,
partial-flavor 276.

### wave_idecomp_MIRROR.html — live monitoring (2026-09-20)

The visual playhead is driven **from the audio clock**, not the frame clock. Advancing
`S.aud.pos` by dt per frame alongside a playing buffer drifts, because rAF is not locked to
the audio hardware. Reading `ctx.currentTime` makes picture and sound agree by construction at
any rate. A BufferSource cannot run backwards, so it is gated on rate > 0 and says so rather
than playing forwards under a reversing picture. Verified: playhead advances off the audio
clock, 25 ledger rows, zero failures.

## wave_membrane_MIRROR.html — radius > 1, partial flavors (2026-09-20)

The radius cap was 1.00 for one reason: past it the flavor shape stops fitting the grid, so
part of its boundary is replaced by the grid edge — and the ledger would have gone on
comparing the mask against a shape **that is no longer there**, charging the missing corners
to staircase error. That is the domain-limit-as-physics-failure bug in its purest form, so
the cap was the cheap fix rather than the right one.

Now raised to **2.00**, with the geometry taken on the *clipped* shape. Exact areas come
from **Sutherland–Hodgman clipping + shoelace** for the polygons and a **closed-form
circle∩square integral** for the disk. Below radius 1 the clip is a no-op and the shoelace
reproduces the old closed forms *exactly* (verified: circle πR² = 19359.3, n3 8005.0,
n5 14651.6, n6 16010.0, n8 17429.5, n12 18486.8, rhombus 7115.6 — all unchanged), so this is
a clean refactor, not a new formula. Past it, every clipped area was checked against a fine
numerical integration (**rel dev ≤ 2e-4**) and every value the app printed was independently
recomputed (**dev ≤ 1.4e-6**).

**The payoff is that R > 1 gives you partial flavors for free** — the domain becomes
*shape ∩ square*, which is the intersection family. The staircase row relabels itself
"partial flavor", reports the clipped percentage on its own line, and keeps measuring
staircase error against the clipped shape. `vertices()` and the prism outline both use the
clipped polygon, so a hexagon at radius 1.5 correctly grows from 6 vertex sources to **8**
(verified against the same count computed in Python: n3 3→6, n6 6→8, n5 5→8, and 6→4 once
the grid is wholly inside the shape). `innerR()` now clamps to the grid, because past radius
1 the **grid edge is the nearest wall**.

Degenerate checks that come out right on their own: circle at radius 1.8 has R ≥ h√2, so the
whole square is inside the disk — exact area **4h² = N² = 25921**, staircase error **exactly
0**, clipped fraction 58.7%.

Regression: **360 configurations** across 5 radii including 1.25/1.6/2.0, zero failures;
mirror row PASSes in 135; the partial-flavor row fires in 180.

## wave_membrane_MIRROR.html — source placement + prism (2026-09-19b)

Four source additions and the prism render. **The headline is a new exact invariant that
the movable source made available, and which immediately caught a bug in the feature that
motivated it.**

**New: MIRROR symmetry about the x-axis.** A reflection in y is a *row swap* on a square
grid — no interpolation — so it is exact to machine precision, and unlike the 8-fold octant
orbit it survives **odd n**. That matters: before this row the triangle and pentagon had no
exact solver test at all. It also gives the source-position slider an invariant, since a
source moved along x alone must preserve it. Measured **1.5e-16 to 5.3e-16** across
triangle, pentagon, hexagon, octagon, dodecagon, square and rhombus, at every radius and
offset tried. The n-gon mask is mirror-symmetric about x exactly when `rot` is a multiple
of π/n (the fold in `insideNgon` is symmetric in θ and cos is even, so the radial bound is
mirror-symmetric even where the half-open fold interval is not) — the row gates on that and
reports rot as the reason when it does not hold.

**The bug it caught, immediately: `Math.round` is not symmetric about a centre.** It breaks
ties toward +∞, so `c+73.5 → 154` while `c−73.5 → 7`, and 154/7 are not mirror images (6
is). The hexagon's vertices at ±90° land on *exactly* half-integer offsets, so with vertex
sources n6 read a mirror residual of **7.70e-2** while every other shape read ~3e-16.
Fixed with `symRound(c,d) = c + sign(d)·round(|d|)`, which rounds the magnitude and
re-applies the sign; n6 now reads 3.89e-16 like the rest. Verified across n = 3,4,5,6,8,12
at two rotations: **0 broken mirror pairs**.

**Source placement is a fraction of the INRADIUS, and is pushed back inside rather than
trusted.** Along an axis |s| = 1 lands exactly on the wall; a diagonal offset of the same
magnitude does not. A source outside the mask does not merely move — Dirichlet clamps it to
zero every step and it **vanishes silently**, so `snapInside` walks any out-of-range
placement back toward the centre and the note says it did.

**Vertex sources** fire in phase at every vertex of the flavor shape. Vertices sit *exactly
on* the boundary (verified: inside at 0.999R, outside at 1.001R), so each is inset by
`max(2, ⌈1.5σ⌉)` cells first — otherwise the mask clamps them to zero. Counts verified
3/5/6/8/12 for the n-gons, 4 for the square and the rhombus; the circle correctly reports
**0 with "the ∞ flavor has no vertices"** rather than inventing an arbitrary count.

**Drop pulse (P)** adds a gaussian at the live source position to **both** U0 and U1, so
the new bump starts at rest while the existing field's velocity (U1−U0) is untouched —
superposition, not a reset. It also recomputes `S.amax` directly, because that is otherwise
only refreshed inside `record()` on a step, so a drop looked like it did nothing while
paused (measured: max|u| 0.101 → 0.994 after the fix). The energy row now gates on the drop
count with the reason: each drop *adds* energy, so drift there is correct.

**Everything that assumed a centred source was re-derived from the actual source cell:**
`frontFree` now tests against the source's own scanned **clearance** (the inradius is
measured from the centre and is simply the wrong bound once the source moves), `frontRadii`
measures from the source, and the Huygens window uses the true source-to-probe distance.
The 8-fold row gates to n/a with "the source is offset to (x, y)" rather than failing.

**The prism** draws the domain as both caps of a prism, mirrored about the mid-plane: the
bottom at −H/2 + u, the top at +H/2 − u, so the prism thins where u > 0 and can never cross.
The **uniform** height (every side face a square) is the cap's edge length: **2R for the
axis-aligned square — which makes it literally a cube** (verified: all 8 vertices carry the
cube distance signature to **4.44e-16**), **2R sin(π/n)** for a regular n-gon (**R exactly**
for the hexagon), **R/cos(θ/2)** for the rhombus, and **2R** for the circle (equilateral
cylinder — it has no edge to match). The walls are drawn from the *ideal* outline, not the
staircased mask, so you can see what the mask is trying to be. Cost is **nil**: 658 vs 659
steps/s, because the solver dominates and the second cap shares the colour buffer.

It must not be oversold, and the in-file comment says so: **this is a rendering.** The two
membranes are not coupled — nothing propagates between them — which is exactly why H is a
free parameter. The moment H carries physics it stops being adjustable and becomes a third
grid dimension with its own cell count and its own CFL.

Regression: **480 configurations** (prism on/off × 5 shapes × 2 radii × 4 sources × 3 offsets
× 2 boundaries), zero failures; the mirror row PASSes in 88 of them rather than sitting
vacuously n/a. Six repeated drops mid-run: clean.

### Assessed, not built — a beam from a tunnel in the wall

The geometry is a mask **union** (polygon OR rotated rectangle), about ten lines. Two things
make it more than trivial, and only one of them is serious.

**1. A tunnel is a waveguide, so it has a cutoff.** For a hard-walled channel whose clamped
walls are W cells apart, propagation needs `f > f_c = Co/(2W)`; below that the mode is
evanescent and *nothing comes out*. Measured with a lock-in detector: the evanescent decay
rate matches `κ = √(k_y² − k²)` to **0.6–3.8%** (residual is grid dispersion), and above
cutoff the decay is **+6e-5 per cell**, i.e. none. Two false starts of my own worth
recording: a max-envelope detector measures the **switch-on transient**, which is broadband
and so contains components above cutoff that do propagate; and measuring far down the
channel reads the **numerical floor** rather than the tail, so the decay must be sampled
near the source. *Practically, though, this is a caveat rather than a blocker*: at the app's
default Co = 0.5 and f = 0.12 only a **1-cell** tunnel is below cutoff (w = 2 already gives
f_c = 0.0833 < 0.12). It needs stating and a ledger row, not a redesign.

**2. The junction with a staircased wall is the awkward part.** Nudging the wall half a cell
moves **0** mask cells on the axis-aligned square but **295** on a hexagon, **202** on a
rotated hexagon and **155** on a pentagon. A clean rectangular tunnel meets an axis-aligned
wall cleanly and a slanted one raggedly, so the feature wants either the square domain or
the conforming lattice of tier C.

## wave_membrane_MIRROR.html — radius slider + presentation pass (2026-09-19)

Four additions, and the radius slider turned out to reach much further into the solver than expected.

**The radius slider is a resolution control in disguise.** The grid stays N×N, so shrinking the domain
does not make the run cheaper — it makes it *coarser*. R falls in cells, staircase error grows, and the
drive wavelength λ = Co/f spans fewer cells. Both numbers are now live under the slider, because hiding
that coupling would be the same class of mistake as a row reporting a domain limit as physics.

**Three things in the solver were pinned to the GRID when they should have been pinned to the DOMAIN**,
and all three were latent bugs the slider exposed rather than caused:

1. **Probe and source placement.** Probes at `0.22N`/`0.30N` and the continuous/two-slit drivers at
   fixed grid fractions fall *outside* a shrunk domain, where the mask zeroes them every step — the
   source silently does nothing. Measured: **105 of 160** grid-pinned placements land outside the mask
   across 8 shapes × 5 radii. Now placed as fractions of the **inradius**, which guarantees they are
   inside for any shape; at radius 1.00 on the full square they reproduce the old absolute offsets
   (35, 48, 19, 6) *exactly*, so nothing moved for the existing configurations.
2. **`frontFree()` compared against the grid half-width**, which equals the inradius only for the full
   square. On a triangle the wall is at `R·cos(π/3) = R/2`, so the anisotropy row kept measuring long
   after the front had hit it. Now against `innerR()`.
3. **The (1,1) eigenmode was hard-coded to `sin(πi/(N−1))`** — the mode of the *grid*, not of the
   domain. On a shrunk square that profile is not an eigenmode at all: correlation falls to **0.761**
   after 600 steps. Built from the clamped box instead, it holds at **1.000000000000** at every radius.

**The square is snapped to whole cells, deliberately.** It is this app's one exactly-representable
domain and every other row is calibrated against it. A masked square holds `2·floor(R)+1` cells per
side, so its area `(2R)²` is exact only when `2R` is an odd integer — which, with a 0.005 slider step,
happens at radius 1.00 and essentially nowhere else. Unsnapped, the reference domain would quietly stop
being exact the moment the slider moved. `domR()` now snaps R to the half-integer; the ledger still
**measures** the exactness rather than assuming it, so a broken snap turns the row red. Verified
**48/48** square configurations exact.

**The custom gradient forbids exactly what the 1D app forces.** There the quantity is a phase, which
wraps, so the editor makes stop 360 equal stop 0 and a hand-made ramp can never open a seam. Here it is
a *signed amplitude*, which does not wrap: identical ends make the map two-to-one and +u becomes
indistinguishable from −u. So this editor clamps instead of wrapping and measures three things —
sign separation, luminance reversals, zero contrast — rather than repairing any of them. A ramp built
to the 1D app's own rule scores **sign separation 0.000** here and is flagged.

**A metric I wrote was mis-specified, and the test caught it.** Sign separation was first defined as the
minimum of |colour(+u) − colour(−u)| over *all* u. That scores the house diverging ramp at **0.035**,
because near zero a diverging map *must* have both sides meeting at the neutral — the metric penalised
the very property that makes it diverging. Re-specified over |u| ≥ ½: house **0.555** (1.110 at the
ends), cyclic **0.000**, constant **0.000**, and a non-monotone ramp still caught at **60** reversals.
Thresholds were read off those measurements rather than guessed.

**A comment claimed a bug that cannot happen.** I wrote that changing the mask predicate from
`shape !== 'square'` to `!sqFull()` fixed a leak, where the BC block writes live Mur/Neumann values into
the outer ring and the j=1 row reads them back in. Measured: it cannot. The ring is read only by row 1,
and row 1 is inside the mask only when R ≥ 79, which needs radius ≥ 1.006 — above the slider maximum.
Both predicates measure **0.000e+00** outside the mask for neumann and mur from radius 0.995 to 0.3.
The change is correctness of the predicate, not a fix, and the comment now says so.

**Full screen cannot be verified from the desktop preview pane, and the fallback is measured, not
assumed.** The pane serves the file from a `data:` URL — an opaque origin. `document.fullscreenEnabled`
reads **true**, yet a call *without* a user gesture throws `TypeError: Permissions check failed` and a
call *with* a trusted click leaves the promise **unsettled forever**, neither resolved nor rejected.
A `.catch()` fallback therefore never fires. The fallback is armed on a 500 ms timer as well, and drops
into photo mode saying so, rather than leaving a button that does nothing. Opening the file directly
(`file://`) should be unaffected; **real fullscreen remains unverified**.

Regression: **288 configurations** (6 shapes × 4 radii × 4 sources × 3 boundaries), zero false
failures. Exact 8-fold symmetry still holds on the new masked geometry at **6.11e-16** (periodic) and
**4.09e-16** (9-point).

## wave_membrane_MIRROR.html (2026-09-17)

Everywhere else in this project `Co ≤ 1/√D` has been a *stated* fact. **Here you can drag the slider
past the line and watch the scheme diverge**, exactly as the von Neumann proof says it must:
5-point at Co = 0.705 stays bounded at 0.108; at Co = 0.720 it reaches **1.4×10⁷⁴**.

**The sharpest check in the app is an exact symmetry invariant.** A centred source on a symmetric
domain must preserve 8-fold symmetry to machine precision. It reads **2.26e-16** — and it *caught a
real bug*: the periodic boundary was a convoluted edge patch-up rather than a genuine wrapped update.
Nothing else in the ledger would have found it. Worth copying into any future solver.

Other measured rows: the (1,1) eigenmode keeps its shape (correlation **1.000000000** after 832 steps);
Mur absorbing returns 5.22e-4 of the outgoing amplitude against Dirichlet's 9.43e-1 (**1800×**); the 2D
Huygens wake reads **1.35e-1** just after the front, matching the offline figure; energy drift 5.65e-5
with ±6.6% oscillation, which is the expected O(Δt²) behaviour of a leapfrog conserving a *modified*
energy — not an error.

### Two results that correct earlier canon

**1. The CFL limit belongs to the STENCIL, not only to the dimension.** `Co ≤ 1/√D` is the 5-point
result. The 9-point isotropic Laplacian (2/3 edge, 1/6 diagonal, −10/3 centre) has |λ|max = 16/3
instead of 8, so its limit is **√3/2 = 0.866025** — verified analytically and numerically, and shown
live: **at the same Co = 0.75 the 5-point diverges to 10¹⁴⁴ while the 9-point stays bounded at 0.109.**

**2. Grid anisotropy does NOT improve with Courant number** (an earlier note in this app said it did).
The Co² term cancels to leading order, leaving (kh)²/48. Measured with sub-cell peak interpolation:
sweeping **Co from 0.1 to 0.7 moves the anisotropy ratio by 5.1e-5**, while coarsening the grid moves
it by 1e-1. It is driven by **points per wavelength**. Budget resolution first, Courant second. The
9-point stencil additionally buys **145–2500×** isotropy.

### Berglund reconnaissance — errata for the parent canon

Read by four parallel agents over the local clone. The load-bearing correction: **the solver is not in
`wave_common.c`.** The parent canon (v3 §6) says `evolve_wave_half … (wave_common.c, ~200 lines)`; it is
actually in **`wave_billiard.c:302-631`**, and `wave_common.c` holds only initial conditions and
diagnostics. Also: his update is byte-for-byte this one plus `-KAPPA*u` (a Klein-Gordon restoring term,
not damping) and `-tgamma*(u-u_prev)` (damping on the velocity, which is what this app does); his
`evolve_wave` runs **three** timesteps per call via a 3-buffer rotation, so ported source frequencies
are 3× off if you miss it; his `BC_DIRICHLET` is **misnamed** (uses −3.0·u, a free/reflecting edge, not
a clamped drumhead); his absorbing BC is **not Mur** but a first-order upwind relaxation, exact only as
Co → 1; and **he never runs near the CFL limit** — Co ≈ 0.08–0.12, max 0.25 across ~1300 recorded runs,
because dispersion rather than stability is his binding constraint. His source contains no CFL check at
all, plus three real bugs (a live heap overflow at `wave_billiard.c:589`, a wrong malloc dimension, and
an accumulating damping factor) that should not be transcribed.

**Ledger-design lessons, repeated from the v2 review:** five rows initially failed for reasons that were
*validity limits, not physics* — anisotropy measured after the front had reflected (no single front to
find), energy graded on a variable medium (the functional assumes uniform c), energy graded on a driven
source (which injects energy by design), energy graded over a window too short to separate the
standing-wave beat from drift, and the Huygens wake sampled so late the t⁻² tail had decayed. All now
report `n/a` with the reason. This keeps being the dominant failure mode; it is worth checking for
before writing any new row.

## wave_idecomp_MIRROR.html — v1 (2026-09-16)

No spec file yet (built from the 8 decisions the author gave on 2026-09-16); this section is the
interim contract. **Analytic only — there is no solver and therefore no CFL condition in this build**;
`Co ≤ 1/√D` is carried as a *stated* row against the day the FDTD mode lands.

**Model.** ψ(x,t) = Σ sources, evaluated in closed form at any t.
Schrödinger ω = k²/2 (ℏ = m = 1); EM vacuum ω = c|k| so sign(k) sets the direction.
Plane: `A e^{i(kx+φ−ωt)}`. Packet: `A(σ/√a)exp(−(x−x₀−v_g t)²/4a)e^{i(k(x−x₀)+φ−ωt)}` with
**a = σ² + it/2 for Schrödinger** (complex ⇒ spreads) and **a = σ² for EM** (real ⇒ does not).
Everything is computed in the `e^{i(kx−ωt)}` convention and conjugated once at the end, which is
what makes "flipping the convention conjugates the field" exactly true (measured dev 0.00e+0).

**Why 1D.** A complex field on a D-dimensional domain has a graph in D+2 dimensions. D = 1 → 3, so the
helix fits exactly; 2D and 3D domains must *encode* phase rather than draw it. Stated on screen.

**Ledger.** 19 rows. Two are render-fidelity rows read back from the GPU buffers (Float32 vs the
Float64 field, and the two shadow lines vs the helix); the rest are computed from the Float64 array the
draw was built from — same split as the 3D file's deviation 16. The two PDE rows use O(h²) stencils, so
their tolerance is **stated in the step sizes** (`(dx²/12)k⁴|ψ|` etc.), not a magic constant — the
deviation-4 pattern. One row is red by design: *"Re alone CANNOT satisfy it (it cannot)"*.

**Verified in the pane 2026-09-16** with real dispatched clicks and slider input: all 5 presets, both
equations, both conventions, palette switch, all 4 camera snaps, play/pause. Zero FAILs. Highlights —
phase slope 2.000000000 vs k, flipping the convention gives −2.000000000 (the helix handedness reverses)
with conjugation dev 0.00e+0; standing wave `max|Im(ψ/ψ_ref)| = 0.00e+0`, **signs seen {−1, +1}** (ℤ₂);
two-source rhombus identity dev 4.4e-16; conj(ψ) = 1/ψ dev 2.2e-16; packet at t = 6 σ 2.500000 vs
2.500000, centre 12.0000 vs 12.0000, norm 5.013257 vs 5.013257; EM packet σ 2.000000 unchanged.

**Bugs found and fixed during the build** (recorded so they are not re-introduced):
1. `buildAxes` indexed `seg.length/6` where the segment stride is 12 → read past the array and wrote
   NaN into the frame buffer. Now `seg.length/12`.
2. The packet width/norm rows FAILED at t = 6 because the packet had run off the x window (v_g = k = 3
   in Schrödinger, three times the EM speed) — truncation narrows σ and drags the centroid inward. That
   is a measurement artifact, not physics, so both rows now gate on a ±3σ containment test and report
   `n/a` with the reason. **A ledger row must not report a domain error as a physics failure.**
3. The quadrature row is only meaningful over a whole number of wavelengths; the window is trimmed to
   one (corr 0.0016 over 7λ) instead of using a loose tolerance.

### v2 additions (2026-09-17)

| feature | notes |
|---|---|
| **audio / synth input** | `decodeAudioData` handles WAV (raw PCM) and MP3 (MDCT frames) identically, so the format difference vanishes at one call. The PCM is lifted to its **analytic signal** by a hand-rolled radix-2 FFT: helix radius = instantaneous envelope, winding rate = instantaneous frequency. A 2× window is taken and the middle half kept (Hilbert edge artefacts). Five built-in synths make the path testable with no file. |
| **speed v, β, γ, ψ** | `S.v` is the EM medium speed; **v = v₀ is the null case** (β = 1, ψ = 90°, γ → ∞), and β = v/v₀ is exactly `circle_flavor`'s β. γ is guarded: **β > 1 reads "undefined", never a number** (1 − β² < 0). In Schrödinger mode β comes from v_g = k instead, and in audio mode there is no k at all, so it reads "— (no k)". |
| **δ relative phase** | display transform y = \|ψ\|cos φ, z = \|ψ\|cos(φ − δ). **δ = 90° is the true field**; 0°/180° collapse to a line. This is the polarisation ellipse, and it answers "is the helix the same as E-vs-B?" — it is at δ = 0, which is where E and B actually live. |
| **quarter-wave ghost** | Re(x − λ/4) drawn on the Im plane, where it must lie *on* the Im shadow. The 90° shift, made literal. Measured separation 7.7e-4. |
| **editable gradients** | stops → LUT. **Cyclicity is forced by construction** (entry 360 = entry 0), and the **luminance ratio is shown live** so the banding a ramp introduces is visible rather than silent (built-in corrected ramp 1.00×, raw hue 4.49×). |
| **WebM export** | `captureStream` + `MediaRecorder`, no library. Duration = one common period when the ω are commensurate, else the hard cap (default 180 s). Verified end-to-end: 1.95 MB file, correct name, auto-stop. |

**v2 verified in the pane** with real dispatched input: δ sweep (90° → 0.997939, 45° → 0.414144 vs
tan 22.5° = 0.414214, 0°/180° → 0.000000), ghost separation 7.69e-4, v = 1 → NULL row, v = 0.5 →
γ 1.154701 / ψ 30° / n 2, synth tone → **219.5 Hz** against a true 220, **chirp at t = 0.5 s → 200.1 Hz
against a true 110 + 180t = 200**, AM envelope 0.552–0.635, recorder 1.95 MB. Zero FAILs in every mode.

**v2 bugs found and fixed during the build:**
1. `C_LIGHT` left dangling after renaming it to `S.v` — threw inside `buildLedger`, so the ledger
   rendered **zero rows** while the rest of the page looked fine. A silent empty ledger is worse than
   a red one; worth a guard.
2. `recInfo()` never re-ran on source changes, so the planned export duration was a **stale startup
   value** for every configuration. Fixed by calling it from `refresh()`.
3. The polarisation thickness row read 0.983 instead of 1.000 at δ = 90° — partial-wavelength bias,
   the *same trap* the quadrature row already had. Fixed by trimming to whole wavelengths and
   subtracting the means.
4. β leaked the source-list `k` into audio mode, where there is no k. Now `NaN` → "— (no k)".
5. "incommensurate ω" overclaimed: the search is bounded (denominators ≤ 64, T ≤ cap), so the
   message now says *no common period found* within that, which is what was actually tested.

### Adversarial review, 2026-09-17 (126 agents, 6 lenses × 3 refuters)

40 raw findings, **28 survived** three independent refuters each. All 28 addressed. They fell into
five classes, and the classes are more useful than the individual items:

1. **Vacuous passes on a degenerate field.** An identically-zero field satisfies every linear
   identity, so a row graded on it glows green while testing nothing. Hit the equation residual,
   |ψ| constant, homogeneity, standing wave, norm, packet σ, and γ. **Fixed with one guard**
   (`LIVE = peak > 1e-12 && sources exist`) applied to every numeric row. The conjugation row had
   already been guarded this way earlier in the build — the guard simply had not been generalised.
2. **The quarter-wave ghost was the most broken feature.** The shift is **signed** — d = s·π/2k,
   so it flips for k < 0 *and* with the time convention — and it was hard-coded to +λ/4; it was
   compared against the *displayed* Im (so δ ≠ 90° turned a correct field red) instead of Im ψ;
   and it printed a PASS from an empty loop when λ/4 exceeded the window.
3. **Audio rows could not fail.** Both were graded on "is the number finite and positive", so
   conjugating the analytic signal, or doubling the DC/Nyquist bins, left them green. **Now graded
   against ground truth** the synths provide exactly — including the chirp's *moving* instantaneous
   frequency 110 + 180t and FM's 220 + 20cos(2π·5t). A loaded file has no ground truth, so those
   rows read n/a with the measurement shown rather than a meaningless PASS.
4. **Tolerances that encoded an assumption.** `tolX` modelled |ψ⁗| as k⁴|ψ| — carrier only — so a
   **k = 0 packet got a bound of exactly zero and failed** despite being exactly right; the envelope
   terms 6k²/σ² + 3/σ⁴ are now included. The polarisation window still **duplicated one endpoint**
   (m wavelengths span M−1 intervals, not M), leaving an O(1/M) bias that failed at N ≤ 220 — the
   same partial-period trap as before, one layer deeper.
5. **Domain and resolution limits reported as physics failures.** A grid violating Nyquist, a packet
   with no mass in the window, a window past the end of the audio, and a by-design-red row whose
   truncation error was not far below the quantity it asserts — all now read **n/a with the reason**.
   That last one became a three-state row: assert, cannot-tell, or genuinely unexpected.

Also fixed: the audio window began *at* the playhead and ran forward (leaving the file a full `need`
samples early) — now centred; the Hz readout divided by `need` instead of the true sample span,
biasing every reading low by (need−1)/need; `rate = 0` made the export announce 3,141,592 s.

**Re-verified after the fixes: 121 configurations** (5 presets × 2 equations × 2 conventions × 4 N,
plus δ × window sweeps, 5 synths × 4 playheads, and 3 speeds) — **zero FAILs, zero unexpected reds,
no console errors.** Spot values: synth tone 219.9 vs known 220.0; chirp at t = 2.0 s **470.0 vs known
470.0**; polarisation at N = 120 now 0.999628 (was 0.991362, a FAIL); ghost 7.69e-4 against an
interpolation bound of 2.41e-3, with the shift flipping sign correctly for k < 0 and for the "+"
convention; audio at playhead 0 correctly reports "256/512 kept samples past the end".

### Audio in the WebM export (2026-09-17)

In audio mode the clip now carries the sound. A `MediaStreamAudioDestinationNode` is mixed into the
canvas stream, so one `MediaRecorder` writes VP9 + **Opus**. The source starts at the **current
playhead**, so the audio lines up with what is drawn. Two honest consequences, both surfaced in the UI:
`playbackRate` follows the rate slider (so a non-unity rate shifts pitch — that is what scrubbing
faster actually means), and a *negative* rate cannot play audio backwards, so it runs forward and says
so. The synth path renders its own buffer, so it records too. The decoded AudioBuffer is now kept alive
(the context is no longer closed after decoding) because the export needs it.

Verified end to end: the recorded blob was fetched back and `decodeAudioData`'d — **4.98 s, 48 kHz,
stereo, RMS 0.245, peak 0.604, non-silent.** Not just a track that claims to exist.

**Open / next:** MP4 (needs a muxer — WebM is native); 2D Schrödinger in the membrane app (different
integrator, first order in time); live tab/system audio capture (assessed: easy via `getDisplayMedia`,
deferred by the author).

## Run

Open `flavor_renderer_3d_MIRROR.html` in a browser (needs internet once: three.js 0.160 via jsdelivr importmap).

```
pytest test_flavor3d_oracle_MIRROR.py -v
```
(numpy + pytest only; no scipy, no sympy.)

## Status — 2026-09-15

Built and **browser-verified in the Claude Browser pane** (real dispatched clicks / slider inputs;
ledger read from the DOM after a forced frame): every row that has a closed form at that p is green
for cube (N = 32, 64), sphere (N = 8, 32, 64, and R×2), octahedron p = 1, and p = 4; rows that are
cube-only (§2.1 identity, both [111] rows, the camera collapse) or sphere-only (Gauss–Bonnet) read
`n/a` with the measured value elsewhere; vertex and edge isomers + snap give 54.7356°, 0.0000° to the
diagonal and the 8 → 6 collapse through the camera (7 clusters, six at 0.816497, gaps 60.00000°);
no console errors. Then reviewed by a 6-lens / 3-refuter adversarial workflow (2026-09-15); every
surviving finding is applied (see deviations 8–13) and the changed paths re-driven.
Measured sphere numbers at N = 32, a = 1: V 4.184060 (rel 1.13e-3, order 1.98), A 12.559017
(rel 5.85e-4), ray-cast residual 9.55e-4 (bound a/N² = 9.77e-4), arc loop 4.712389 (dev 7e-14),
Gauss–Bonnet 1.570796 both ways. **The pytest oracle has not been run** (the author runs scripts
himself) — run it before trusting the tolerances on another machine.

## Deviations from the spec (flagged, not silent)

1. **Mesh = cube-sphere, not icosphere** (spec §5). Six N×N grids on the cube, welded, each vertex
   pushed to r(u)·u. Reason: the axis, edge-midpoint and body-diagonal directions must be exact mesh
   vertices for the min/max-radius rows to pass at 1e-9 — an icosphere has the axis directions (from
   level 1) but never the edge-midpoint or body-diagonal ones. One topology
   serves the whole p-sweep, the (+,+,+) octant is an exact sub-mesh, and ray-cast lookup is O(1).
   Quads are split on the diagonal with the larger midpoint radius (ties → corner-ward diagonal), which
   is sign-flip invariant, so the eight octants are exact mirror images (octant volume = ⅛ to 1e-12).
2. **Spec §2.4 "6 half-edges"** — the carrier loop of the cube octant is the skew hexagon of six
   face-midline halves, face-centre ↔ edge-midpoint; the cube's own half-edges are interior spokes of
   the patch. Length 3 and the [111] hexagon are unchanged. Drawn that way.
3. **Spec §2.3 `a`** is the axis-direction radius (inradius for p ≥ 2, circumradius for p ≤ 2).
   Labelled `a = r(axis)` in the readout.
4. **Curved tolerances are explicit in N**: volume/area relative 3/N² (chordal, O(h²)); ray-cast
   residual 1.5·a/N² (sag bound a/N²). Planar flavors (cube, p = 1): volume/area 1e-12, ray-cast
   1e-9 (spec §6). Off p ∈ {1, 2, ∞} the ray-cast row is `n/a` with the residual shown — chordal error
   there is O(h^min(p,2)) near vertices. The spec's icosphere-level-4 figure of 1e-3 is **not** met by
   the cube-sphere at the default N = 32 (sphere volume rel 1.13e-3) and is met from N = 64 (2.8e-4);
   the row prints "spec 1e-3 met / NOT met at N" beside its own tolerance.
5. **Sphere loop and Gauss–Bonnet are measured as geodesic quantities on the drawn vertices** (arc
   sum, Van Oosterom–Strackee excess sum) — exact at any N — with the chord sum shown beside.
6. **₁R/₀R for the p-ball** = (2/3)^(½−1/p) for p ≥ 2, 2^(½−1/p) for p ≤ 2 — derived here (not in the
   spec's §2.3), checked live and in the oracle.
7. **Dimension slider**: n = 3 full 3D; n = 2 draws the equatorial section z = 0 (which *is* the 2D
   p-ball of the same a) over a faint 3D wire; n = 1 the segment [−a, a]; n ≥ 4 table only with an
   on-canvas notice, the solid reduced to the same faint wire. The 3D ledger rows always measure the
   (possibly faint) 3D mesh.
8. **Snap** sets the camera on a body diagonal with an up-vector ⊥ the view, so OrbitControls does
   not clamp it off the pole; `reset view` restores z-up; panning is disabled (every view readout is
   taken about the orbit target). The 8 → 6 collapse row is evaluated only for the cube and only while
   the view is within 1e-6° of a body diagonal (atan2 form, exact after snap; otherwise `n/a`).
   Choosing an isomer also puts the camera on the aligned axis (spec §7: "rotate onto the view axis"),
   so `vertex` shows the hexagon immediately.
9. **p = 1 closed forms** (octahedron: V = 4a³/3, A = 4√3a², patch √3a²/2, loop 3√2a, radii a, a/√2,
   a/√3) were added as ledger targets so the far end of the sweep is checkable; oracle-covered by
   `test_mesh_octahedron_is_exact`.
10. **p slider is log-scaled** (p = 2^s, s ∈ [0, 6], step 0.02) so the octahedron end has room; spec
    §2.3's sample p = 1.5 is reachable through the `1.5` button or the exact numeric `p` input.
11. **[111] rows** judge the length residual (units of a) and the angular residual (degrees)
    separately, each at the spec's 1e-9.
12. **Two "stated fact" rows** carry spec §2.7 (phase-group bits 3 / 4.585 / 5.585) and §2.8 (the
    Dehn wall via Niven on arccos ⅓ = 70.5288°) — computed where computable, labelled stated, not
    mesh measurements. The homogeneity row is exact by construction (×2 is a power-of-two rescale) and
    says so; real rounding at non-power-of-two scale is exercised by the R-scale slider against the
    closed forms.
13. **The `[111] projection ghost` checkbox** hides both the inset and the 3D rings.
14. **The p-ball signature row is the spec's sampled form** — min/max of the *ray-cast* mesh radius over
    the 4122 sampled directions (26 extremal included) — a measurement distinct from the vertex-radius
    signature row. Its tolerance is 1e-9 for planar flavors and 1.5/N² otherwise: between vertices an
    inscribed chordal mesh sits *below* the surface by the sag, so the sampled minimum radius is low by
    ~1/N²; the spec's "1e-6 at 10⁴ directions" holds for the formula sampled, not for a mesh. The §2.1
    row is labelled what it is: a code-path identity, exact by construction (both sides reduce to
    max|uᵢ|); the mesh measurement of §2.1 is the ray-cast row.
15. **Oracle file name** is `test_flavor3d_oracle_MIRROR.py` (spec §1 says `flavor3d_oracle.py`):
    `test_` prefix for pytest discovery, `MIRROR` suffix for the house naming rule.
16. The ledger integrates the Float64 vertex array; the GPU draws its Float32 copy (≤ 6e-8 relative).

## Spec errata found while building (report back into the spec)

- **§2.3, p = 8 sample:** the spec prints 0.662342; the closed form 3^(−|½−1/8|) = 3^(−3/8) =
  **0.662338** (residual 4.2e-6). The printed figure is presumably the sampled ratio. The p = 1.5 and
  p = 4 samples (0.832683, 0.759836) are correct to 1e-6. Both the app row and the oracle use the closed
  form.
- **§2.4 "6 half-edges"** — see deviation 2 (the loop is the face-midline hexagon).
- **§2.3 `a`** — see deviation 3 (axis-direction radius, not always the inradius).

## Ledger rows (all measured from the drawn mesh / drawn coordinates)

χ = V − E + F · volume (divergence theorem) · area · signature (₁R/₀R, ₂R/₀R) from vertex radii ·
p-ball ₂R/₀R = 3^(−|½−1/p|) · §2.1 a/max(u·nᵢ) ≡ a/‖u‖∞ · r(u) formula vs ray-cast mesh · octant
volume = ⅛ · octant patch area ×8 and closed form · sphere Gauss–Bonnet (corner excess = Σ spherical
excess = πa²/2) · carrier loop length + closure · [111] octant hexagon (radius a√(2/3), 60°) ·
[111] whole cube (2 → centre, 6 at 2a√(2/3)) · V_n peak at 5 and V₁₀/2¹⁰ · Dehn wall (stated) ·
phase-group bits (stated) · homogeneity 8 / 4 · rotor = Rodrigues = quaternion + isomer axis → +z ·
n = 2 section signature 2^(−|½−1/p|).
Per-frame: view ∠ nearest face normal (54.7356°), ∠ nearest body diagonal, 8 → 6 collapse through
the camera.

## 2026-09-16 additions

**circle_flavor_MIRROR.html** — the circle-mapped renderer, Stage A + B. Every ledger row is computed
from the drawn pixel positions mapped back to world coordinates (`toPx` → `toWorld`), so it tests the
renderer, not the slider. Flags from the build: (1) the spec overloads R — the whiteboard scale R (§2.2)
and the unit-modulus bridge R = e^{iθ} (§2.3) are kept apart as R and 𝐑; (2) the Bondi-k row uses a
*relative* 1e-12 (k ≈ 115 at ψ = 89°, so an absolute 1e-12 fails on conditioning alone); (3) the
Pythagorean list is every coprime (m, n) with n < m ≤ 7 (17 pairs; the spec's table is a selection),
non-primitive both-odd pairs shown reduced; (4) H is `n/a` when it leaves the hyperbola panel at the
locked scale (ψ ≳ 78°). Verified in the pane at ψ = 30°, 36.87° ((2,1) snap: k = 2, γ = 5/4), 89°,
with R = 2, C₄ on, Jacobi–Anger residual 3.7e-16; no console errors. No separate oracle: the spec's
§9 log covers the closed forms and the ledger measures the rest.

**3D renderer additions.**
- *The three spheres* (circum ₀R, mid ₁R, in ₂R, drawn at the measured radii) with an "outermost as
  wireframe" toggle: the circumsphere goes to wire if shown, else the crystal itself ghosts so an
  inscribed sphere shows through. Ledger: canonical directions on each sphere = 8 / 12 / 6 for the cube
  (V, E, F), 6 / 12 / 8 for the octahedron, 26 / 26 / 26 for the sphere; the two gaps ₀R−₁R, ₁R−₂R are
  the 2D "R − a" gap lifted. Counted over the 26 direction classes, not every grid vertex: at p = 1 the
  midsphere meets each face in its incircle and the N = 32 grid has 9 rational points per face on it
  (an all-vertex count reads 84) — reported alongside, not hidden.
- *Vertex map*: for the flavor's canonical vertices, the C₂³ sign bits, parity (the 4 + 4 split into
  the two tetrahedra of the stella octangula), the phase element (identity / reflection / π-rotation /
  inversion, with det), the minimal rotor from the (+,+,+) reference (axis, angle, landing dev) and the
  pure-quaternion form u = e^{(π/2)u}. Measured: the C₂³ orbit of the reference has 8 points (cube:
  simply transitive, vertex ↔ 3-bit phase) and only 2 for the octahedron (C₂³ is not transitive there —
  the transitive group is the signed permutations). Generic in the vertex list for tier-1 solids.
- *Unfolder*: edge nets for polyhedral flavors by a random spanning tree of the face graph (cube 6 faces,
  octahedron 8; overlap test on interiors; Σ net area = Σ face area), and for the sphere the gores /
  sinusoidal / Mollweide projections with the equal-area check Σ = 4πa² (240 samples per edge; measured
  rel 2e-5 gores, 8e-5 Mollweide at 200) and the gore-edge shear at 60° = atan((π/12) sin 60°) = 12.77°,
  expectation evaluated at the chord's actual midpoint latitude (at 200 samples the nearest sample was
  60.3°, which read 12.81° against a 60° expectation — a real FAIL caught in the pane and fixed by
  sampling at a multiple of 6). Stated walls on the ledger: Theorema Egregium (K ≠ 0 ⇒ no isometric
  net) and Dürer's problem (OPEN). N ≥ 4 nets (tesseract: 8 cubes, 261 nets) are tier 1.

## Next

Tier 1 per spec §9 (five Platonic solids, dual pairs, truncation slider, cross-section tab,
tetrahedron failure demos). Open items are spec §10 — do not resolve silently.
