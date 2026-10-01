# TaichiWaveCLAUDE — the ZRL 2D membrane, ported to Python/Taichi

Phase 1 of the plan agreed 29/09/2026: **port first**, then solid objects, then prism-like media, then the lattice extension. This folder holds the port of `FlavorRenderersCLAUDE/wave_membrane_MIRROR.html` (the merged build of 28/09/2026). The solver runs on the GPU and is checked value-for-value against the web app's own code.

*Generated from scratch — Claude Opus 5.5 — 29/09/2026*

## Files

| file | what it is |
|---|---|
| `membrane_core_MIRROR.py` | the solver, no GUI. Geometry, masks, radius snaps, source placement and seeding run on the host (numpy, the same arithmetic as the JS). The time step is one fused Taichi kernel. |
| `membrane_app_MIRROR.py` | the viewer (Taichi GGUI): a 3D heightfield or 2D top view, a control panel, hotkeys, and a headless `--shot` mode that renders a PNG. |
| `crosscheck_js_MIRROR.py` | runs the **web app's own solver** headlessly under Node, runs the port with the same settings, and compares mask, per-cell Co², seed, field after K steps, vertex cells, clip fraction and peak. 26 configurations. |
| `identity_tests_MIRROR.py` | exact properties at sizes the browser can't reach: light cone, CFL flip, 8-/6-fold symmetry, energy conservation, drum dispersion, conformance counts, time reflection. |
| `bench_MIRROR.py` | throughput per backend and grid size, with the web app's solver as the reference. |
| `ToolsCLAUDE/js_harness_MIRROR.mjs` | loads the web app's module script in a Node `vm` with DOM/THREE stubs, exposing its S, stepOnce, reseed, … (used by the cross-check). |
| `ToolsCLAUDE/js_dump_MIRROR.mjs` | writes the web app's fields for a list of configurations (used by the cross-check). |
| `ToolsCLAUDE/js_bench_MIRROR.mjs` | times the web app's stepOnce with the app loaded as a real ES module, the way the browser loads it. |

## Running it

Needs Python 3.12 with `taichi` 1.7.4 and `numpy` (both installed; `python` here is 3.12), and Node for the cross-check.

```bash
python membrane_app_MIRROR.py
```

```bash
python membrane_app_MIRROR.py --N 2049 --f32 --arch vulkan
```

```bash
python crosscheck_js_MIRROR.py --arch cpu
```

```bash
python identity_tests_MIRROR.py
```

```bash
python bench_MIRROR.py --js
```

Backends: `--arch cuda` (default, f64), `cpu`, `vulkan` (f32 only: Taichi's Vulkan backend has no 64-bit `sin`). `--f32` halves the memory and roughly doubles the speed at large N; see *Precision* for what it costs.

**App keys:**
- `SPACE` play/pause · `N` one step · `R` reset · `P` drop a pulse
- `V` 3D/2D · `X` wireframe · `H` hide the panel
- `L` lattice · `B` boundary · `G` domain shape · `M` medium · `O` source
- right-drag orbits the camera; W/A/S/D/Q/E move it (GGUI's own keys, which is why the app avoids them)

## Verification (29/09/2026)

### Against the web app: `crosscheck_js_MIRROR.py`

26 configurations cover:
- all four boundaries on the full square;
- sq5/sq9 periodic;
- (m,n) modes on the full and the shrunk square;
- every source type (impulse, pulse, continuous, slit, mode, vertices radial/wall, driven vertices);
- every waveform, slab and lens media, damping;
- masked n3/n4/n5/n6/n8/n12/circle/rhombus/square on both lattices, including the conforming 30° snaps;
- the 9-point stencil with Neumann and Mur boundaries (the only case where the ring corners feed back, through the diagonal);
- N = 257, and one deliberately unstable run (Co 0.75 > 1/√2).

| backend | result | field after 300 steps vs the web app |
|---|---|---|
| CPU f64 | 26/26 | ≤ 2.2e-14 of max\|u\|; **bit-identical** (0.0) when started from the web app's own seed, even in the unstable run |
| CUDA f64 | 26/26 | ≤ 3.4e-14 |
| CUDA f32 | 26/26 | ≤ 1.5e-5 |
| Vulkan f32 | 26/26 | ≤ 1.5e-5 |

Masks, per-cell Co², vertex-source cells and pulse-clip fractions agree **exactly** in every configuration. The seeds differ by ≤ 2.2e-16, from numpy's `exp` vs V8's `Math.exp`. The driven sources differ by ~1e-15, from Python's `sin` vs `Math.sin`.

What "agree" means for the unstable run: past the CFL limit, the checkerboard mode grows ~6.9× per step **from rounding noise**, so any last-bit difference is amplified without bound.
- **CPU f64:** must be bit-identical from the same seed, and is.
- **CUDA f64:** must blow up by the same factor, and does (within 1%). CUDA fuses `a*b + c` into one rounding (FMA) even with `fast_math` off.
- **f32:** must blow up *sooner* (its noise floor is ~1e9 times higher), and does.

### Exact properties: `identity_tests_MIRROR.py` (N = 1025)

| test | f64 (CUDA, CPU) | Vulkan f32 |
|---|---|---|
| light cone: exactly 0 outside the reach polytope (diamond / square / hexagon), nonzero on its rim | 0.0 outside after 200 steps | 0.0 outside after 20 steps¹ |
| CFL flip at C_max (1/√2, √3/2, √(2/3)) | stable at 0.995, overflows at 1.01 | same |
| 8-fold symmetry (square) / 6-fold symmetry (tri hexagon at 30°) | 4.2e-15 / 5.4e-15 | 2.5e-6 / 2.6e-6 |
| discrete energy Σ(Δu)²/Co_k² − ⟨u⁺, L u⟩ over 2000 steps (uniform, lens, tri) | drift ≤ 1e-14 | ≤ 5e-6 |
| drum dispersion: (1,1), (7,3), (40,25) modes vs the closed form, sq5 and sq9, 400 steps | ≤ 9e-12 | ≤ 4.3e-5² |
| conformance: tri n6 / n3 / 60° rhombus at 30° hold exactly 3s²+3s+1 / (s+1)(s+2)/2 / (s+1)² cells | exact (6/6) | exact |
| time reflection: Co 0.5 → 0.7 mid-run makes a backward ring; the continuous-time rescale removes it | ratio 0.248; rescale leaves 1.8% | same |

¹ The leading edge after m steps is ~Co^(2m): 1e-62 at m = 200, below f32's smallest number, so in f32 the rim itself underflows to 0. The web app's underflow row describes the same effect.
² The (1,1) mode's curvature is 1.9e-5 of its amplitude at N = 1025, so each step's rounding is ~1% of the signal, and for a smooth field it adds up coherently.

### Faults found while porting (both fixed here; neither affects the web app)

1. **Vulkan: the last cell of an 8-bit mask read as 0.**
   - *Found by:* the cross-check — N = 161 periodic, cell (160,160), invisible until a wave reached that corner at step 291.
   - *Cause:* Vulkan emulates 8-bit storage inside 32-bit words, and 161² is not a multiple of 4.
   - *Fix:* the mask is 32-bit on Vulkan and 8-bit elsewhere.
2. **f32: the 9-point stencil was not self-consistent.**
   - *Cause:* (2/3)·4 + (1/6)·4 − 10/3 cancels exactly in f64 but not in f32, and the residue acts as a small mass term.
   - *Effect:* the (3,2) mode was off by 5e-4 after 300 steps, identically on CUDA and Vulkan.
   - *Fix:* in f32 the stencil is written as (2/3)(e − 4u) + (1/6)(dg − 4u), which cancels by construction (5e-4 → 1.5e-5). f64 keeps the web app's exact operation order.

### A web-app finding (not fixed; the port mirrors it)

**Neumann boundary, corner (0,0):** the web app copies the ring in one sequential loop. Traced through that loop, corner (0,0) is written at i = 0 from cell (1,0) before iteration i = 1 refreshes it. So it gets the value that buffer held **two steps earlier**, while the other three corners get this step's values.

It reaches the field only through the 9-point stencil's diagonal. Measured with a pulse near that corner, sq9 Neumann, 400 steps:
- **9-point:** the field differs by **8.6% of its peak** from a fresh-corner version.
- **5-point:** only the corner cell itself differs (7e-5), since the stencil never reads it.

The port reproduces the value exactly (CPU bit-identical), and runs in parallel. **Fix it in both apps at once**, so they keep agreeing: the physical Neumann corner is the diagonal neighbour, u(0,0) = u(1,1), or the mean of its two side neighbours.

**Panel in `--shot` images:** GGUI's panel does not appear in offscreen `save_image` captures. Heights, colours and the mesh do. The panel shows in the live window.

### A measurement fault avoided

Timing the web app inside Node's `vm` sandbox (which the cross-check uses) gave ~120 ns/cell. That is ~5× slower than the same code as a real ES module (~25 ns/cell): in a `vm` context every top-level variable access goes through an interceptor. The benchmark uses the module figure, so the port isn't flattered.

## Performance

Measured 29/09/2026 on the user's machine (i9-9900K, RTX 5070 Ti 16 GB, Windows 11), `python bench_MIRROR.py --max 8193 --js`.

The figures are milliseconds per step, with nanoseconds per cell in brackets:
- **Port:** the full step as the app runs it (interior, boundary, mask ring, sources, dispatch).
- **Web app:** its `stepOnce()` under Node's V8 (the browser's engine), loaded as an ES module the way the page loads it. This includes the per-step `record()`.

| lattice | N | CUDA f64 | CUDA f32 | Vulkan f32 | CPU f64 (8 cores) | web app (JS) | best GPU vs JS |
|---|---|---|---|---|---|---|---|
| square | 161 | 0.096 (3.7) | 0.093 (3.6) | 0.069 (2.7) | 0.147 (5.7) | 0.634 (24.4) | 9× |
| square | 513 | 0.093 (0.35) | 0.092 (0.35) | 0.067 (0.26) | 0.562 (2.1) | 7.20 (27.4) | 107× |
| square | 1025 | 0.086 (0.082) | 0.092 (0.087) | 0.072 (0.069) | 2.19 (2.1) | 27.0 (25.7) | 374× |
| square | 2049 | 0.207 (0.049) | 0.155 (0.037) | 0.174 (0.042) | 9.18 (2.2) | 105 (25.1) | 682× |
| square | 4097 | 0.841 (0.050) | 0.448 (0.027) | 0.695 (0.041) | 35.3 (2.1) | — | — |
| square | 8193 | 3.59 (0.053) | 1.85 (0.028) | 2.43 (0.036) | — | — | — |
| tri | 161 | 0.117 (4.5) | 0.117 (4.5) | 0.078 (3.0) | 0.164 (6.3) | 0.484 (18.7) | 6× |
| tri | 513 | 0.111 (0.42) | 0.124 (0.47) | 0.075 (0.29) | 0.455 (1.7) | 5.79 (22.0) | 77× |
| tri | 1025 | 0.104 (0.099) | 0.106 (0.10) | 0.077 (0.074) | 1.67 (1.6) | 20.2 (19.3) | 262× |
| tri | 2049 | 0.153 (0.037) | 0.103 (0.024) | 0.126 (0.030) | 6.82 (1.6) | 78.9 (18.8) | 768× |
| tri | 4097 | 0.598 (0.036) | 0.346 (0.021) | 0.531 (0.032) | 25.0 (1.5) | — | — |
| tri | 8193 | 2.48 (0.037) | 1.30 (0.019) | 1.83 (0.027) | — | — | — |

Reading it:
- **Up to N ≈ 1025 the GPU is idle.** The floor of ~0.07–0.12 ms/step is launch latency: ~20 µs per parallel pass on Windows (WDDM), with 3–5 passes per step. That is why the speed-up grows with N.
- **From N ≈ 2049 the GPU is bandwidth-bound:** ~0.02–0.05 ns/cell. Two-thirds of the per-step cost is f64 traffic, so f32 is ~2× faster there.
- **Real-time budget:** N = 8193 (67 M cells) at 60 fps takes ~9 steps/frame at f32, ~4.6 at f64. The web app at N = 1025 managed ~0.6 steps/frame at 60 fps.
- **Setup** (mask, sources and seed on the host): 0.1 s at N = 2049, 1.5 s (square) / 5.2 s (tri n6) at N = 8193.
- **Compilation:** the first run of each lattice/boundary/source combination compiles for ~2 s, then Taichi's disk cache serves it (0.0 s afterwards, at every N).
- **Memory:** 3 levels + Co² + mask = 33 bytes/cell at f64 (2.2 GB at N = 8193), 17 at f32. So N = 16385 needs 4.6 GB at f32 on the GPU, but the host-side setup at that size is untested.

## What the port does differently

- **Co changed mid-run.** The web app keeps both stored time levels, so a change is a *temporal boundary* and every wave splits (see below). The port does the same by default. The panel checkbox *"Co change = new time step"* instead reads the change as a new Δt for the same medium, and re-expresses the previous level to second order:
  `u_prev ← u − r(u − u_prev) + ((r² − r)/2)·Co₁²·L(u)`, with `r = Co₂/Co₁`.
- **Not ported (yet):**
  - the ledger, alignment panel, prism/3D-extrusion view, room IR/audio, probes and energy plot;
  - palettes other than the diverging one;
  - surface opacity (GGUI meshes have no alpha).

  The solver is complete; these are readouts and views.
- **N up to 8193 in the app** (16385 fits on the 16 GB GPU at f32). Past N = 1025 the displayed mesh is strided (every k-th vertex). The solver always runs the full grid.
- **Steps are batched.** A kernel call from Python costs ~170–265 µs here, more than a whole step at N ≤ 2049. So 8 steps are unrolled into one call.

  Measured trade-off (N = 161, Mur):

  | steps per call | compile | time per step |
  |---|---|---|
  | 1 | 0.25 s | 480 µs |
  | 8 | 1.8 s | 115 µs |
  | 16 | 4.8 s | 92 µs |
  | 32 | 14.7 s | 73 µs |

  8 keeps the first run of a new configuration to ~2 s.
- **Mur and Neumann rings run in parallel,** with the web app's values. Serial loops ran on one GPU thread (~130 µs/step). Mur's ring cells read only interior cells, so order doesn't matter. Neumann's order matters only at the corners, and those are reproduced exactly (see the finding below).
- **Storage is `ti.ndarray`,** so a new N or a reset never recompiles.

## Time reflection: what happens when Co changes during a run

It isn't a sonic boom: nothing moves faster than the waves. It's a **temporal boundary**, the time-domain mirror image of a wave hitting a change of medium.
- **Across a spatial interface,** frequency is conserved; the wavelength changes, and part of the wave reflects.
- **Across a temporal interface,** where the whole medium changes speed at one instant, the *wavelength* is conserved (the field's shape is frozen at that instant). So the frequency must change, and part of the wave reverses direction.

Leapfrog stores two time levels, so it carries both the field and its rate of change across the switch. Those two, re-read at the new speed, no longer describe a purely forward wave. The remainder is a backward one:
- amplitude **B = (Co₂ − Co₁)/(2Co₂)** in the long-wave limit;
- the lattice-exact value for wavelength λ follows from `e^{iθ}` with `θ = 2 asin(Co sin(π/λ))`.

Measured 29/09:
- **1D:** 0.1432 against a predicted 0.1432 (Co 0.5 → 0.7).
- **2D:** backward/forward ring ratio 0.248. The 1D value is B/F = 0.167, and a converging ring gains amplitude while a diverging one loses it, so ~0.28 is expected.

With the rescale the backward wave drops to 0.07–0.26% (1D) and 1.8% (2D, where the residue is the Co-dependence of the grid's own dispersion).

**Which reading is right for the app?** It depends on what Co means:
- If Co is a property of the medium (c changes, Δt and h fixed), the reflection is real physics, the same effect as in time-varying metamaterials.
- If Co is just the time step (c and h fixed, Δt changes), the reflection is a numerical artifact, and the rescale removes it.

The web app's clock (`t += Co`) treats Co as Δt, which is the second reading.

## Backlog and plan

In the order agreed 29/09/2026:

1. **Taichi port.** Done: solver, viewer, verification. Next: the ledger rows that matter at large N, the energy plot, and probes.
2. **Solid objects inside the domain** (and the *"Unlicensed"* domain). Design:
   - Objects are masks composed onto the domain mask: `M = domain AND NOT objects`. Each object is a shape with a position (x, y), rotation and scale.
   - **Text** is rasterised on the host (PIL, any TTF), then placed as an object mask.
   - **Wall type per object:**
     - Dirichlet (u = 0, the current mask rule: a hard, phase-inverting wall);
     - Neumann (∂u/∂n = 0, a rigid wall, no phase inversion: the mirrored-neighbour rule on the object's rim);
     - later, impedance / partial reflection.
   - **Moving objects:** re-mask each frame. Cells that become solid are zeroed; cells that become fluid start at 0. A moving Dirichlet wall does work on the field, as a real piston does. Wall speed must stay below Co cells/step.
   - **"Unlicensed":** a rounded rectangle at the plate's aspect ratio; *CA EXEMPT* and *DEATH* as text objects. The plate's screw holes and the letter counters (the holes in A, D, P, R) are small scatterers of their own, and a good test of sub-wavelength scattering.
   - **Effort:** a few hundred lines, mostly host-side mask composition. The solver doesn't change: it already honours any mask.
3. **Prism-like media and areas.** A programmable per-cell speed map (CC is already per cell): polygons/text as regions of a given index, gradient-index lenses, and the 1D prism/lens.
4. **Lattice extension.** Edge-sharing domains as partial mirrors or lenses: a separate app, per the 27/09 decision.

Also pending (from earlier rounds; not forgotten):
- **Sources:** a beam source (directed, width and angle); sound-file sources.
- **Boundaries:** PML (a true absorbing boundary; Mur is first-order only).
- **The web app:** the Neumann corner fix (above, in both apps); a Co-change mode toggle (continuous time); per-step `record()` off by default, the `colorOf` allocation, and the domain-cell cache; the reciprocity row; a wall-offset R snap.
- **3D:** the port (volume rendering, marching-cubes speed), then the Echo sim.
- **Other apps and experiments:** a melting/viscous sim; 2D Schrödinger; a quantitative Snell check; measured drum spectra.
- **ZRL canon:** the ASKs, including the β-normalisation fork (open).
