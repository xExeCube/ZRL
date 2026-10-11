# Vertex sources: why the waves turn choppy, and what fixes it

*Generated from scratch — Claude Opus 5.5 — 04/10/2026*

All measurements use the frozen snapshot `ScratchCLAUDE/Round4CLAUDE/BeforeCLAUDE/`, the code the user runs today. No project source file was changed. Scripts, outputs and pictures are in `ScratchCLAUDE/ChoppyCLAUDE/` (`OutCLAUDE/`, `PicsCLAUDE/`). Every number below names the script that produced it. The progress log is `NotesCLAUDE/progress_choppy_MIRROR.md`.

**Round 4c (05/10/2026):** recommendations 1–5 of section 5 are now in the port's core. The port defaults to the wall inset, and driven vertices start from rest. New readouts: the source note, the drive warning and a validated phase-lag estimate. Section 7 has the changes, the measurements and the UI hand-off. *Edited by Claude Opus 5.5 — 05/10/2026.*

## 1. The short answer

1. It is not a code error. It is a numerical artefact, and the part that is specific to vertex sources can be fixed completely.
2. **The vertex chop comes from the starting pulses being cut by the wall.** The default `vertex inset: radial` moves each pulse only max(2, ⌈1.5σ⌉) cells inward along the radius. Its true distance to the walls is then only 0.7–1.7σ (0.67–1.15σ for triangles and rhombi). So the domain edge cuts off 1.2–27% of each pulse at t = 0, and the cut is a sharp step at the wall. A step contains every wavelength down to the grid spacing. The result: 4.5–49% of the wave energy sits at wavelengths under 8 cells, against 2.5% for a centred pulse.
3. This has nothing to do with grid alignment. Across all 51 configurations the chop follows the clip fraction (correlation 0.97), and fully aligned shapes are still affected: the triangle on the triangular lattice at 30°, every edge a lattice row, has 23% chop with the radial inset (49% at 7°, because more of the pulse is clipped there). Alignment only changes how the chop looks: tidy fringes on straight walls, random speckle on staircase walls.
4. **The JavaScript version was never "fixed" by default.** On 27/09 it gained an option, *3σ from BOTH walls*, and its default stayed `radial`. The port has the same option: the `vertex inset` button. Set to `wall`, it cuts the clip to ≤ 0.012% and the chop to 2.3–3.1%, which is the level of a centred pulse. It also converges to zero with resolution, while the radial chop grows with resolution.
5. **Different wall types will not fix it.** With rigid domain edges, the radial inset still leaves 5–26% chop: the cut becomes a kink instead of a step. With the wall inset, pinned and rigid edges are equally clean (2.0–3.1%).
6. **There is a second, slower effect that every source shows, the centred pulse included.** After about 4–8 domain crossings at N = 513 the field turns into blotchy speckle. This is the grid's phase error: it carries a 3-cell-wide pulse 0.1–0.35% too slowly, by different amounts in different directions on the square lattice. The error builds up until the field has lost its shape: correlation with the true solution is 0.90 after 1 crossing, 0.52 after 4 and 0.40 after 8. It is also numerical. In domain mode it falls about 3–4× per doubling of N: at N = 2049 the correlation is still 0.97 after 8 crossings.
7. **Driven vertices are a separate case.** The default drive frequency (0.12 at Co 0.5) is a 4.2-cell wavelength, half the 8-cell minimum. So 95–99.9% of the energy is grid-scale whatever the inset. Use a wavelength of at least 16 cells (freq ≤ 0.03125 at Co 0.5).
8. **Is it fixable for good?** The vertex chop: yes, use the wall inset. The slow phase error never reaches zero on any grid, but it can be pushed out as far as you like (domain mode at higher N, a wider pulse, or Co nearer the limit). The intricate kaleidoscope pattern that remains in a fully converged run is real physics, not a bug: corner diffraction at 108/120/135/150° corners and the interference of many reflections.

## 2. The user's questions, one by one

**"How was this fixed in the JavaScript version?"** It was made optional, not fixed. The 27/09 build of `wave_membrane_MIRROR.html` added *vertex source inset: 3σ from BOTH walls*. It moves each source along its corner's bisector by 3σ / sin(α/2), so it sits 3σ from both walls. The build also added a readout of the clip fraction, with the hint *"the cut is a sharp edge the grid cannot resolve; try 3σ from BOTH walls"*. The default stayed `vtxInset:'radial'` (web app line 494; checked read-only). The port copied both the option (`Params.vtx_inset`, the *vertex inset* button) and the radial default.

The port's `wall` inset gives the web app's numbers:
- clip fraction ≤ 0.012% at N = 513 over 51 configurations, most 0.001–0.005% (`s1_reproduce_MIRROR.py`); the web app reported ≤ 0.004% for its set;
- clearance 2.67–4.33σ.

The port's app is missing three parts of the web app's source note:
- the clearance readout;
- the coloured clip fraction with that hint;
- the *inset capped* note.

The port shows only a plain "pulse clipped at t = 0: x%" line at the bottom of the panel.

**"Is it ever truly fixable so long as the shape isn't grid aligned (though it seems to happen when it is aligned, also)?"** Yes. The user's observation is right: alignment is not the cause.
- With the wall inset, unaligned minus aligned is −0.40 to +0.41 percentage points of chop (mean −0.06) over every shape and lattice (`s1b_invariant_all_MIRROR.py`).
- With the radial inset, the large differences between rotations come from different clip fractions (the vertex snaps differently), not from the walls.

The deeper reason: the domain is a closed, lossless, time-invariant system, so the energy in each frequency band is fixed at t = 0 (§6.1). Walls, aligned or staircase, cannot create short waves. They can only send the existing ones in other directions. Measured: the energy behind the chop budget stays constant to ≤ 6.4e-15 over 4 crossings on the port's own run (`s0b_invariant_check_MIRROR.py`), and the budget is unchanged to all printed digits over 8 crossings for pinned and for rigid walls (`s3_rigid_vs_pinned_MIRROR.py`). So the amount of chop is decided entirely by the starting field (and by a drive, if there is one).

**"Will different wall types fix this?"** No. The test used the research operators of `ResearchCLAUDE/WallsCLAUDE/walls_lib_MIRROR.py` on the snapshot's own masks. Rule P reproduces the port to ≤ 2.7e-14. Rule R0 is rigid: link removal plus corner blocking.

| starting field | pinned (the port) | rigid R0 |
|---|---|---|
| radial inset | 4.5–48.7% | 5.3–26.1% (lower in only 12 of 51) |
| wall inset | 2.3–3.1% | 2.0–3.0% |

N = 513, chop budget hfE (§6.1), `s1b_invariant_all_MIRROR.py`.

The theory checks out:
- At a pinned wall, the cut pulse acts like its odd mirror image, which has a **step** at the wall.
- At a rigid wall it acts like its even mirror image, which has only a **kink**.

So rigid edges do make the clipping chop converge with resolution, where pinned edges do not (§4.4). But at the app's σ = 3 cells the kink is still strong, because the pulse sits only ~1σ from the wall. Rigid walls also change the physics: reflections come back upright, and a rigid cavity keeps the seed's mean as a constant offset (visible in p7). The wall inset is the fix for both wall types.

**"Is it even a bug?"** No code error was found. The solver matches the web app and the walls_lib reference to round-off. It is three numerical effects plus one physical one:
1. a starting field with a step in it (the radial inset);
2. a pulse only 3 cells wide, close to the grid scale (2.5% of its energy is under 8 cells);
3. a drive at 4.2 cells per wavelength;
4. the physical part: the slowly growing complexity of a closed drum.

Two code details make it worse, and both are inherited from the web app:
- the radial inset default;
- a hard-driven vertex overwrites its own seeded bump with drive(0) = 0 at step 1, so the vertex cell goes from 1.0000 to 0.0000 while its neighbour reads 0.8976 (checked on sq5 n6 rot 7, λ 16). That leaves a one-cell dent, which radiates grid-scale waves.

## 3. The causes and their measured shares

The **chop budget** is hfE: the share of the energy in eigenmodes above the 8-cells-per-wavelength frequency. It is exact and constant in time (§6.1). Values are at the app's defaults (N 513, σ 3, Co 0.5) unless stated.

| cause | what it is | share at the defaults | with resolution (domain mode) | kind | fix |
|---|---|---|---|---|---|
| (a) clipping at t = 0 | the radial inset puts the pulse 0.8–1.2σ from the walls; the mask cuts 1.2–27% of it; a step at the wall | **+2 to +46 points** of chop over the pulse's own 2.5% (4.5–48.7% in total; n3 23–49, rhombus 17–39, n4 13–25, square 14, n5 10–16, n6 6–15, n8 6–8, n12 5–7, plate 4.5–9) | **does not converge; grows**: sq5 n3 rot 7 46 → 47 → 54% at N 513 → 1025 → 2049 | numerical: initial data with a step | `vertex inset: wall` (→ 2.3–3.1%, then 0.02–0.1% at σ 6–12) |
| (b1) the pulse's own short waves | a σ = 3-cell gaussian has 2.4–2.5% of its energy below 8 cells (hfE 2.47 on sq5 and tri, 2.44 on sq9; continuum 2.54) | 2.4–2.7% for every pulse source | falls fast: 2.3e-5% at σ = 6 | numerical: pulse too narrow for the grid | wider σ, or domain mode at larger N |
| (b2) phase error over time | the lattice carries the pulse's main wavelengths (~19 cells) 0.12–0.35% too slowly on sq5 (direction-dependent), 0.35% on sq9, 0.23% on tri; its 8-cell part 0.65–1.95% too slowly; staircase walls act ~0.4 cells outside the drawn edge | correlation with the converged field: 0.90 (T 1), 0.52 (T 4), 0.40 (T 8), −0.18 (T 16) | ~3–4× smaller per doubling: N 2049 0.97 at T 8, 0.89 at T 16 | numerical: dispersion and wall offset | domain mode at larger N (error ÷3–4 per doubling); Co near the limit (tri 0.8: 4.5× less; sq9 0.85: 2.7×); wider σ |
| (c) staircase scattering | a staircase wall sends waves into other directions at the same frequency | **0 change in the budget** (unaligned − aligned −0.40 to +0.41 points at the wall inset); changes the look (fringes vs speckle) | converges | numerical, redistribution only | none needed |
| (d) corner diffraction | 108/120/135/150° corners have no exact mirror-image system, so each corner sends out a new circular wave every time a front passes; 60/90° corners have exact images and send none | 0 change in the budget (it keeps frequency) | stays: it is in the converged N = 4097 field | **physics** | none (it is the right answer) |
| (e) long-time mixing | a closed drum's field becomes a complex interference pattern | 0 change in the budget | stays (p10 right panel, N = 2049) | **physics** | none |
| (f) driven vertices | default freq 0.12 at Co 0.5 = 4.17-cell wavelength (3.80–4.06 on the grid); hard point sources; the seeded bump dented to 0 at step 1 | **95–99.9%** of the energy under 8 cells from T 0.5 on, any inset; λ 16: 1.5–3.3% at T 8 (0.4–1.3% without the seeded bump) | n/a | numerical: wavelength under 8 cells | freq ≤ 0.03125 (λ ≥ 16); a warning in cells mode |

The table gives "shares" in two units:
- (a), (b1), (c) and (f) are shares of the energy budget;
- (b2) is a loss of correlation over time;
- (d) and (e) are what remains when everything numerical has converged.

## 4. Evidence

### 4.1 The metric is exact (`s0_sanity_MIRROR.py`, `s0b_invariant_check_MIRROR.py`)

- **FFT split, free space.** In free space the FFT split f8E is constant to round-off (energy drift ≤ 1.6e-15 over 0.9 crossings on sq5, sq9 and tri).
- **Free gaussian content.** f8E of a free gaussian at rest:

  | σ (cells) | 1.5 | 2 | 3 | 4 | 6 |
  |---|---|---|---|---|---|
  | f8E | 58.3% | 28.6% | 2.47% | 0.055% | 5e-7% |

  The continuum formula (1 + x)e^(−x), x = (2πσ/8)², gives 59.6% / 29.4% / 2.54% / 0.056% / 5.3e-7%.
- **hfE is exactly conserved.** The chop budget hfE (§6.1) is constant from T 0 to T 4 (energy drift ≤ 6.4e-15) on the port's own Taichi run. The one exception is the full `square`: its seed is not clamped on the array ring until step 1 (§4.9). hfE and the FFT split agree within ~0.3 points while the fronts are away from the walls. When a front lies on a wall, the FFT reads up to 1.3 points high.
- **Chebyshev filter vs direct eigendecomposition**, N 101, sq5 n3 rot 7, radial:

  | rule | exact | K = 600 | K = 1200 |
  |---|---|---|---|
  | pinned | 40.84% | 41.34% | 41.00% |
  | rigid | 23.67% | 23.68% | 23.68% |

  So K = 600 (used throughout) is within 0.5 points.
- **The FFT split is wrong at t = 0 near walls.** There it reads 51.5% for the full square with the radial inset, where the exact budget is 14.1%. The masked "zero outside" picture labels the step as one-cell sharp, while the pinned dynamics see a two-cell odd ramp. That is why hfE is the primary number.

### 4.2 Every shape at the app's defaults

Table 1 below is the exact chop budget hfE (% of the energy) for pinned walls (the port), N 513, σ 3, Co 0.5, from `s1b_invariant_all_MIRROR.py` via `s6_tables_MIRROR.py`. "rot (r+d/n)" means: rotation in degrees, then edges along a lattice row (r) or diagonal (d), out of n edges (the port's alignment model). Clip = % of Σu² the mask cuts at t = 0 with the radial inset.

| lattice | shape | rot (r+d/n) | clip | **radial** | **wall** | centred pulse | unaligned rot | clip | **radial** | **wall** | centred pulse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sq5 | square | 0 (4+0/4) | 9.4 | **14.1** | **2.7** | 2.5 | - | - | - | - | - |
| sq5 | n3 | 0 (1+0/3) | 23.7 | **40.6** | **2.9** | 2.5 | 7 (0/3) | 25.6 | **46.4** | **3.0** | 2.5 |
| sq5 | n4 | 0 (4+0/4) | 9.4 | **13.6** | **2.7** | 2.5 | 7 (0/4) | 13.5 | **25.4** | **2.7** | 2.5 |
| sq5 | n5 | 0 (1+0/5) | 6.6 | **15.5** | **3.1** | 2.5 | 7 (0/5) | 5.1 | **13.2** | **2.9** | 2.5 |
| sq5 | n6 | 0 (2+0/6) | 4.5 | **12.6** | **3.1** | 2.5 | 7 (0/6) | 4.3 | **11.4** | **3.0** | 2.5 |
| sq5 | n8 | 0 (4+4/8) | 2.6 | **7.1** | **2.9** | 2.5 | 7 (0/8) | 2.2 | **7.2** | **2.7** | 2.5 |
| sq5 | n12 | 0 (4+0/12) | 1.7 | **5.5** | **2.8** | 2.5 | 7 (0/12) | 2.0 | **7.0** | **2.8** | 2.5 |
| sq5 | rhomb | 30 (2+0/4) | 12.8 | **26.2** | **3.0** | 2.5 | 7 (0/4) | 16.1 | **38.5** | **2.8** | 2.5 |
| sq5 | unlicensed | 0 (4+0/4) | 1.2 | **5.7** | **2.6** | 2.6 | 7 (0/4) | 3.8 | **9.1** | **2.3** | 2.6 |
| sq9 | square | 0 (4+0/4) | 9.4 | **13.9** | **2.6** | 2.4 | - | - | - | - | - |
| sq9 | n3 | 0 (1+0/3) | 23.7 | **36.4** | **2.9** | 2.4 | 7 (0/3) | 25.6 | **41.8** | **3.0** | 2.4 |
| sq9 | n4 | 0 (4+0/4) | 9.4 | **13.4** | **2.6** | 2.4 | 7 (0/4) | 13.5 | **23.0** | **2.6** | 2.4 |
| sq9 | n5 | 0 (1+0/5) | 6.6 | **12.9** | **3.0** | 2.4 | 7 (0/5) | 5.1 | **10.3** | **2.9** | 2.4 |
| sq9 | n6 | 0 (2+0/6) | 4.5 | **10.2** | **3.0** | 2.4 | 7 (0/6) | 4.3 | **9.2** | **3.0** | 2.4 |
| sq9 | n8 | 0 (4+4/8) | 2.6 | **5.7** | **2.9** | 2.4 | 7 (0/8) | 2.2 | **5.5** | **2.7** | 2.4 |
| sq9 | n12 | 0 (4+0/12) | 1.7 | **4.7** | **2.8** | 2.4 | 7 (0/12) | 2.0 | **5.8** | **2.8** | 2.4 |
| sq9 | rhomb | 30 (2+0/4) | 12.8 | **23.4** | **3.0** | 2.4 | 7 (0/4) | 16.1 | **33.6** | **2.8** | 2.4 |
| sq9 | unlicensed | 0 (4+0/4) | 1.2 | **4.5** | **2.6** | 2.5 | 7 (0/4) | 3.8 | **8.5** | **2.3** | 2.6 |
| tri | square | 0 (2+2/4) | 5.6 | **14.3** | **2.7** | 2.5 | - | - | - | - | - |
| tri | n3 | 30 (3+0/3) | 15.0 | **23.2** | **2.6** | 2.5 | 7 (0/3) | 27.1 | **48.7** | **3.0** | 2.5 |
| tri | n4 | 0 (2+2/4) | 11.0 | **22.8** | **2.8** | 2.5 | 7 (0/4) | 8.7 | **19.5** | **2.8** | 2.5 |
| tri | n5 | 6 (1+0/5) | 4.9 | **12.0** | **2.9** | 2.5 | 7 (0/5) | 4.8 | **12.2** | **3.1** | 2.5 |
| tri | n6 | 30 (6+0/6) | 2.2 | **5.6** | **3.1** | 2.5 | 7 (0/6) | 5.4 | **15.4** | **2.7** | 2.5 |
| tri | n8 | 0 (2+2/8) | 2.9 | **8.0** | **2.6** | 2.5 | 7 (0/8) | 1.8 | **6.1** | **2.8** | 2.5 |
| tri | n12 | 0 (6+6/12) | 1.8 | **6.0** | **2.8** | 2.5 | 7 (0/12) | 1.8 | **6.5** | **2.7** | 2.5 |
| tri | rhomb | 30 (4+0/4) | 8.6 | **17.0** | **2.8** | 2.5 | 7 (0/4) | 15.9 | **37.8** | **2.9** | 2.5 |
| tri | unlicensed | 0 (2+2/4) | 1.7 | **6.1** | **2.8** | 2.7 | 7 (0/4) | 1.9 | **6.2** | **2.4** | 2.5 |

Notes:
- Ranges over 51 configurations each: radial 4.5–48.7%, wall 2.3–3.1%, centred pulse 2.4–2.7%.
- Across the radial rows hfE tracks the clip fraction (correlation 0.97).
- The plate's centred pulse is at sy = −0.9, the app's own placement. The core alone leaves it in the lettering (0.33 cells from a solid, 37–48% clipped).

**The FFT view over time** (`s1_reproduce_MIRROR.py`, T = 0, 0.5, 1, 2, 4, 8, 16 crossings) shows the same thing. The f8E share is flat from T 1 to T 16: its spread is at most 1.8 points (median 0.7) over the radial configurations; for example sq5 n3 rot 7 radial reads 46.4 / 46.6 / 46.6% at T 1 / 4 / 16.

**What the eye sees**, from the same script, as medians over 48 configurations:

| source | RMS chop / max\|u\| at T 4 | at T 16 | peak chop / max\|u\| at T 16 |
|---|---|---|---|
| vertex, radial inset | 0.023 (max 0.064) | 0.028 (max 0.098) | 0.14 (max 0.45) |
| vertex, wall inset | 0.012 | 0.017 | 0.08 |
| centred pulse | 0.009 | 0.015 | 0.07 |

The chop gets more visible over time while its energy stays the same, for two reasons:
- the coherent fronts spread out, so the app's auto gain (colour / peak |u|) rises;
- staircase walls scatter the short waves into every direction.

That is the "becoming choppy" the user sees (p9).

### 4.3 Aligned vs unaligned

`s1b_invariant_all_MIRROR.py`, same lattice, shape and inset, unaligned minus aligned:

| inset | difference (points) | mean |
|---|---|---|
| wall | −0.40 to +0.41 | −0.06 |
| radial | −3.3 to +25.5 | +4.6 |

The radial differences come from the clip fraction, not from alignment:
- tri n3: 15.0% clipped at 30° vs 27.1% at 7°;
- sq5 n4: 9.4% vs 13.5%.

The vertex cell snaps differently relative to the staircase at each rotation. p3 shows that the amount looks alike for n4 at 0° and 7°; only the arrangement differs.

### 4.4 Wall types and resolution (`s3_rigid_vs_pinned_MIRROR.py`)

**Part A** (N 257, σ 3; walls_lib rule P vs the Taichi run after 300 steps: max difference 5e-16 to 2.7e-14):
- hfE is constant from T 0 to T 8 for both rules.
- Radial, pinned → rigid: sq5 n3 rot 7 45.7 → 23.1%; n4 rot 0 13.6 → 21.8%; n6 rot 7 11.1 → 14.6%; tri n3 rot 30 23.2 → 26.1%.
- Wall, pinned → rigid: 2.6–3.1 → 1.9–2.6%.
- Rigid lowers the absolute chop energy in most cases (×0.07–1.15 here; ×0.06–1.23 over the 51 configurations of `s1b_invariant_all_MIRROR.py`). It also removes the step's own large energy, so the total falls (sq5 n4: E 11.6 → 4.2), and the share does not drop.

**Part B** keeps the same fractions while refining (σ = 1.5, 3, 6, 12 cells at N = 129, 257, 513, 1025):

| case | rule | σ 1.5 | σ 3 | σ 6 | σ 12 | trend |
|---|---|---|---|---|---|---|
| sq5 n3 rot 7, radial | pinned | 91.2% | 45.7% | 49.9% | **54.4%** | grows; chop energy 6.6 → 13.5 → 26.4 (×2 per doubling: a step) |
| | rigid | 62.5% | 23.1% | 8.0% | **3.2%** | converges; chop energy ÷2–3 per doubling (a kink) |
| sq5 n4 rot 0, radial | pinned | 47.8% | 13.6% | 34.4% | **39.1%** | grows |
| | rigid | 80.3% | 21.8% | 4.8% | **1.9%** | converges |
| tri n3 rot 30, radial | pinned | 66.1% | 23.2% | 43.7% | **50.0%** | grows |
| | rigid | 73.4% | 26.1% | 6.8% | **2.9%** | converges |
| sq5 n3 rot 7, wall | pinned | 63.4% | 3.08% | 0.038% | **0.10%** | the residual 3σ step (1.1% of the peak) grows ~σ |
| | rigid | 61.6% | 2.58% | 0.0074% | **0.0042%** | converges |
| tri n3 rot 30, wall | pinned / rigid | 63.2 / 57.9% | 2.60 / 2.29% | 0.014 / 0.0040% | **0.071 / 0.0037%** | |

At σ 1.5 every case is dominated by the under-resolved pulse itself (58.8% for an unclipped one).

### 4.5 Domain mode, N = 257 → 2049 (`s2_domain_series_MIRROR.py`)

σ_f is set so that σ = 3 cells at N 513, so σ = 1.5 / 3 / 6 / 12 cells. Chop budget hfE at N 257 / 513 / 1025, then the FFT share at T 1 at N 2049:

| case | radial inset | wall inset | centred pulse |
|---|---|---|---|
| sq5 n4 rot 0 | 47.6 / 13.6 / 29.4 / **37.6%** | 65.2 / 2.68 / 0.034 / 0.05% | 58.8 / 2.47 / 2e-5 % |
| sq5 n3 rot 7 | 92.4 / 46.4 / 47.5 / **53.7%** | 65.0 / 3.05 / 0.041 / 0.11% | 58.7 / 2.47 / 2e-5 % |
| sq5 n6 rot 7 | 53.7 / 11.4 / 25.3 / **38.9%** | 61.5 / 2.99 / 0.038 / 0.09% | 58.9 / 2.47 / 2e-5 % |
| tri n3 rot 30 | 65.8 / 23.2 / 39.5 / **49.4%** | 63.3 / 2.61 / 0.016 / 0.07% | 58.9 / 2.47 / 2e-5 % |
| tri n6 rot 30 | 45.0 / 5.6 / 18.7 / **34.4%** | 64.5 / 3.07 / 0.016 / 0.06% | 59.0 / 2.47 / 2e-5 % |
| sq9 n6 rot 7 | 52.4 / 9.2 / 22.1 / **35.0%** | 61.4 / 2.96 / 0.030 / 0.08% | 58.5 / 2.44 / 2e-5 % |

The FFT figures at N 2049 for the wall inset include a ~0.05–0.1 point near-wall bias of the FFT split (§4.1).

The field itself was sampled at fixed fractional positions, with relative L2 differences taken between N and 2N:

| case | T 1: 257→513 | 513→1025 | 1025→2049 | T 4: 513→1025 | 1025→2049 |
|---|---|---|---|---|---|
| sq5 n3 rot 7, radial | 1.15 | 0.73 | 0.50 | 1.42 | **0.96** |
| sq5 n3 rot 7, wall | 0.64 | 0.27 | 0.11 | 0.63 | **0.26** |
| sq5 n3 rot 7, centred pulse | 0.65 | 0.21 | 0.054 | 0.71 | **0.27** |
| tri n3 rot 30, radial | 1.14 | 0.79 | 0.57 | 1.53 | **1.02** |
| tri n3 rot 30, wall | 0.72 | 0.38 | 0.16 | 0.82 | **0.38** |

- The wall-inset and centred-pulse fields converge: ratios 1.9–3.9 per doubling at T 1.
- The radial ones barely do: 1.4–1.6.
- p4 shows this directly. The wall frames at N 513 / 1025 / 2049 are nearly identical. The radial frames keep sharp step fronts with grid-scale ripple at every N.

### 4.6 Long-time phase error (`s2b_long_time_MIRROR.py`, `s4c_phase_error_MIRROR.py`)

Domain mode, σ = 3 cells at N 513. Each N is compared with N = 4097 at fixed fractional positions (relative L2 / correlation):

| case | N (σ cells) | T 1 | T 2 | T 4 | T 8 | T 16 |
|---|---|---|---|---|---|---|
| sq5 n6 rot 7, wall | 513 (3) | 0.45 / 0.90 | 0.72 / 0.74 | 0.98 / 0.52 | 1.12 / 0.40 | 1.53 / −0.18 |
| | 1025 (6) | 0.15 / 0.99 | 0.26 / 0.97 | 0.50 / 0.88 | 0.86 / 0.64 | 1.06 / 0.41 |
| | 2049 (12) | 0.05 / 1.00 | 0.07 / 1.00 | 0.14 / 0.99 | 0.25 / 0.97 | 0.46 / 0.89 |
| sq5 n6 rot 7, centred pulse | 513 | 0.44 / 0.91 | 0.68 / 0.77 | 0.97 / 0.53 | 1.07 / 0.44 | 1.28 / 0.13 |
| | 2049 | 0.03 / 1.00 | 0.06 / 1.00 | 0.12 / 0.99 | 0.25 / 0.97 | 0.45 / 0.90 |
| tri n6 rot 30, wall | 513 | 0.51 / 0.87 | 0.82 / 0.66 | 1.15 / 0.32 | 1.41 / −0.06 | 1.67 / −0.33 |
| | 2049 | 0.06 / 1.00 | 0.09 / 1.00 | 0.20 / 0.98 | 0.41 / 0.92 | 0.76 / 0.70 |
| sq9 n6 rot 7, wall | 513 | 0.56 / 0.84 | 0.86 / 0.63 | 1.03 / 0.48 | 1.23 / 0.25 | 1.51 / −0.18 |
| | 2049 | 0.06 / 1.00 | 0.09 / 1.00 | 0.18 / 0.98 | 0.32 / 0.95 | 0.57 / 0.83 |

What this shows:
- **Not a vertex effect.** The centred pulse decorrelates exactly like the vertex sources.
- **Time to lose the shape grows ~4× per doubling.** Correlation 0.9 lasts about 1 crossing at N 513, 4 at N 1025 and 16 at N 2049.
- **sq9 and tri are not better at Co 0.5.**

`s4c_phase_error_MIRROR.py` explains the last point. Phase-speed error of the σ = 3 pulse's main wavelength (2πσ ≈ 19 cells), in % (lag after one N-513 crossing of 220 cells, in cells):

| lattice | Co 0.5 | Co near the limit |
|---|---|---|
| sq5 | −0.12 to −0.35 (0.26–0.77), direction-dependent | −0.01 to −0.24 at Co 0.693 |
| sq9 | −0.35 (0.77) | −0.13 at Co 0.849 |
| tri | −0.23 (0.51) | −0.05 at Co 0.800 |

sq9 and tri make the error the same in every direction, but at Co 0.5 not smaller. The 8-cell part of the pulse lags 1.4–4.3 cells per crossing. The pinned staircase walls add a first-order position error of ~0.4 cells per wall (`NotesCLAUDE/rigid_vs_pinned_walls_MIRROR.md` §0; web-app README, experiment B).

p10 compares T 8 frames:
- wall inset at N 513: correlation 0.40;
- centred pulse at N 513: 0.44;
- wall inset in domain mode at N 2049: 0.97. This is what the true pattern looks like: still intricate (corners, many reflections), but ordered and 6-fold.

### 4.7 Driven vertices (`s4a_drive_dispersion_MIRROR.py`, `s4b_driven_vertices_MIRROR.py`)

The default freq 0.12 at Co 0.5 is a continuum wavelength of 4.17 cells. On the grid:

| lattice | wavelength (cells) | group speed (c) |
|---|---|---|
| sq5 | 3.80–4.06 (6.9% spread by direction) | 0.73–0.92 |
| sq9 | 3.80–3.82 | 0.73–0.76 |
| tri | 3.94 | 0.83 |

f8E (FFT) at T 8 crossings, N 513:

| variant | sq5 n4 rot 0 | sq5 n6 rot 7 | tri n3 rot 30 |
|---|---|---|---|
| app default (freq 0.12, radial) | 99.8% | 99.8% | 99.6% |
| freq 0.12, wall | 99.9% | 99.9% | 99.7% |
| λ 8 (freq 0.0625), wall | 87.7% | 88.6% | 80.6% (the drive itself sits at the cut: 7.84–7.95 cells on the grid) |
| λ 16, wall | 3.3% | 3.1% | 1.5% |
| λ 32, wall | 3.1% | 1.2% | 2.3% |
| λ 16, wall, seeded bump removed | 0.87% | 1.32% | 0.37% |
| λ 16, radial | 1.7% | 0.87% | 5.3% |

- The app shows the amber "under 8 cells per wavelength" warning only for the beam (cells mode) and in domain mode (`scale_warnings`). Driven vertices, `cont` and `slit` get no warning in cells mode.
- Driven cells are HARD sources (`U = DRV[q]`). They also reflect returning waves like pinned points. The beam source is additive (soft) by contrast.

### 4.8 When the wall inset is capped (`s5_capping_MIRROR.py`)

The cap (m.vtx_capped) engages when 3σ / sin(α/2) exceeds 0.95 of the vertex's distance from the centre:
- at N 513 only for σ ≥ 35 (triangle, rhombus on tri) up to σ ≥ 100 (square);
- at σ 3 only on tiny grids: from N ≤ 41 (sq triangle and rhombus), N ≤ 45 (tri triangle and rhombus), N ≤ 35 (tri n4, tri plate) or N ≤ 31 (tri n5); never down to N = 31 for the other shapes.

Once capped, the clip stays small (0.002–0.07%), except on the plate. There 3σ no longer fits in its half-height, and the clip jumps to 30–33%. At the app's defaults the cap never engages: 0 capped corners in all 51 configurations (`s1_reproduce_MIRROR.py`).

### 4.9 Side findings (inherited from the web app; small)

- **Full `square` (array ring as the wall).** The seed is masked by `mask_np`, which is all ones, so the ring keeps gaussian values at t = 0. Step 1 clamps them. With the radial inset this changes the energy by 0.7% at step 1; with the wall inset by 1.7e-4 (`s0b_invariant_check_MIRROR.py`).
- **Driven vertex bump.** The hard drive overwrites the seeded bump with drive(0) = 0 at step 1: a one-cell dent of depth 1.0.
- **Plate's centred pulse.** In the core alone it sits inside the lettering; the app moves it to sy = −0.9 (`_fixups`).

## 5. Recommendations for the UI agent, ranked by effect

1. **Make `vtx_inset = 'wall'` the port's default** (a documented deviation from the web app).
   - Effect: the chop budget at the defaults falls from 4.5–48.7% to 2.3–3.1%, the level of a centred pulse. It converges to zero with resolution instead of growing.
   - Cost: the vertex sources move inward along the bisector: 3σ / sin(α/2), i.e. 18 cells for a triangle at σ 3, 12.7 for a square, 10.4 for a hexagon.
   - Risk: anything that compares vertex sources with the web app must set `vtx_inset='radial'` explicitly. That covers the `crosscheck_js` configs with `src='vtx'`, the ledger sweep / `crosscheck_ledger` configs, and the bit-identity references of `scale_tests_MIRROR.py` (`ref_cells_*_MIRROR.npz` include a vtx-driven config).
2. **Warn about under-resolved drives in cells mode.** Show an amber line when Co / freq < `PPW_MIN` for every hard-driven source (`cont`, `slit`, vertices with *drive the vertices*), as the beam already does: "λ 4.2 cells < 8: grid-scale waves".
   - Optionally give driven vertices their own default of λ 16 (freq 0.03125 at Co 0.5).
   - Effect for driven vertices: 99.6–99.9% chop at T 8 → 1.2–3.3% at λ 16–32.
   - Risk: none for a warning; a default change needs the same crosscheck care as item 1.
3. **Port the web app's source note.**
   - Clearance in cells and σ.
   - The clip fraction in amber above 1%, with "the cut is a sharp edge the grid cannot resolve; try vertex inset: wall".
   - "3σ inset capped at k corner(s)".
   - Place it next to the *vertex inset* button, not at the bottom of the panel.
   - Effect: the cause becomes visible. This is parity with the web app, not a deviation.
4. **Give the long-time phase error a hint.** For example, a one-line readout "phase lag ≈ x cells per crossing", computed from the lattice symbol at k = 1/σ, plus a pointer to domain mode / larger N.
   - Effect: the user knows the fronts scramble after ~1–4 crossings at σ = 3, N = 513, and why.
   - Co near the limit cuts the phase error 2.7× (sq9) to 4.5× (tri at 0.8). Offer it as advice, not as a default: Co changes the time-reflection behaviour and every crosscheck.
5. **Driven vertices: do not seed a bump under a hard-driven cell, or start the drive at the cell's seeded value.**
   - Effect: 1.5–3.3% → 0.4–1.3% at λ 16 (T 8).
   - Risk: a web-app deviation; the crosscheck's vtx-driven config needs a patch.
6. **Optional: a 4σ inset choice for very large N in domain mode.** The residual 3σ step grows like σ (0.04% at σ 6, 0.10% at σ 12, an estimated ~0.4% at σ 48 / N 8193). A 4σ inset makes the step 3e-4 of the peak instead of 1.1e-2. Low priority.
7. **Rigid walls (this round's stage): do not present them as a fix for this.**
   - If the domain edge can be made rigid, a seed cut by a rigid wall converges with resolution where a pinned one does not, but the wall inset is better with either.
   - A rigid cavity keeps the seed's mean as a constant offset (p7). The ledger's energy row and the auto gain should not be surprised by it.
8. **Clamp the array ring in `seed()` for the full `square`.** 0.7% energy at step 1 with the radial inset; inherited; trivial. Lowest priority.

## 6. Method

### 6.1 Metrics
- **hfE, the chop budget (primary).** The share of the conserved leapfrog energy held by eigenmodes of the actual masked operator whose eigenvalue μ (of −L, i.e. ω² of the semi-discrete system) exceeds μ_c. μ_c is the lattice symbol at |k| = 2π/8, averaged over directions: sq5 0.5935, sq9 0.5859, tri 0.5935.
  - Per mode, E_j = ½[(c1_j − c0_j)²/Co² + μ_j c1_j c0_j] is exactly conserved, so Σ h(μ_j)E_j is conserved for any filter h.
  - Without a drive, no wall of any type, aligned or not, can change hfE after t = 0. That is the proof behind "walls cannot create chop".
  - h is a step, built as a Jackson-damped Chebyshev series of degree K = 600 in A = −L on the free cells.
  - The operator is `walls_lib_MIRROR.build_L`: rule P (= the port: links kept, outside = 0, no corner blocking; checked to ≤ 2.7e-14 against the Taichi run) or rule R0 (rigid: link removal + corner blocking).
  - Why 8 cells: it is the app's own `PPW_MIN`, and below it the lattice's phase speed is off by > 0.6–1.9% (`s4c_phase_error_MIRROR.py`).
- **f8E (FFT).** The same energy split in wavenumber: per Fourier mode E_k = (|U1 − U0|²/Co² + s(k) Re(U1 Ū0)) / 2N², with s the lattice symbol. It sums exactly to the discrete energy because u = 0 outside the mask. It is conserved exactly in free space, close to hfE away from walls, and biased right at a wall at t = 0 (§4.1). Used for time series and at N 2049.
- **Visibility.**
  - V8 = RMS over the domain of the λ < 8 part of u, divided by max|u|. P8 = the peak of that part, divided by max|u|. The app's colours are auto-gained by the recent peak.
  - For rigid walls, Vhp uses a normalised-convolution high-pass instead (u − G*(uM)/G*M, gaussian s = 1.5 cells), which treats both wall types alike.
- **Convergence.** In domain mode, fields are sampled at fixed fractional positions (cubic spline in space, cubic Lagrange in time, the half-step offset of `scale_tests_MIRROR.py`). Reported as relative L2, max and correlation, against 2N or against N = 4097.

### 6.2 References and settings

- **App defaults** (`membrane_app_MIRROR.py` `main()`, `Params`): N 513, sq5, Co 0.5, σ 3, Dirichlet, rad 1, `vtx_inset 'radial'`, freq 0.12, 4 steps per frame.
- **Rotations.** "Aligned" = the rotation with the most edges along lattice rows or diagonals, from the port's `alignment_MIRROR.al_angles`. "Unaligned" = 7° (0 aligned edges for every shape tested). The plate on tri uses rad 0.89, as the app does.
- **References.** A centred pulse in the same domain (and at the app's sy = −0.9 for the plate). An unclipped gaussian in free space (the pulse's own content).
- **Timing.** No timing claims are made. The GPU was shared with other agents; the "s" figures in the outputs are wall-clock times of these scripts only.

### 6.3 Scripts (`ScratchCLAUDE/ChoppyCLAUDE/`) and outputs (`OutCLAUDE/`)

| script | what | output |
|---|---|---|
| `choppy_lib_MIRROR.py` | shared helpers: snapshot on sys.path, metrics, hfE, operators, clearance | - |
| `s0_sanity_MIRROR.py` | free-space conservation of f8E; free-gaussian content per σ | `s0_sanity_MIRROR.txt` |
| `s0b_invariant_check_MIRROR.py` | hfE conservation in time; Chebyshev vs eigh | `s0b_invariant_check_MIRROR.txt` |
| `s1_reproduce_MIRROR.py` (+ `s1_rotations_MIRROR.py`, its import-safe rotation helper) | 51 configs × radial / wall / pulse at the defaults, FFT time series, visibility | `s1_reproduce_MIRROR.txt/.json` |
| `s1b_invariant_all_MIRROR.py` | hfE for the same configs, pinned and rigid R0 | `s1b_invariant_all_MIRROR.txt/.json` |
| `s2_domain_series_MIRROR.py` | domain mode N 257–2049, 8 cases | `s2_domain_series_MIRROR.txt/.json`, `s2_samples_*.npz` |
| `s2b_long_time_MIRROR.py` | domain mode N 513–4097, T up to 16 | `s2b_long_time_MIRROR.txt`, `s2b_samples_*.npz` |
| `s3_rigid_vs_pinned_MIRROR.py` | A: time series pinned / rigid; B: resolution at fixed fractions | `s3_rigid_vs_pinned_A/B_MIRROR.txt`, `s3_fields_MIRROR.npz` |
| `s4a_drive_dispersion_MIRROR.py` | grid wavelength / group speed of the drive | `s4a_drive_dispersion_MIRROR.txt` |
| `s4b_driven_vertices_MIRROR.py` | driven-vertex variants | `s4b_driven_vertices_MIRROR.txt/.json` |
| `s4c_phase_error_MIRROR.py` | phase-speed error vs lattice, Co, direction | `s4c_phase_error_MIRROR.txt` |
| `s5_capping_MIRROR.py` | when the wall inset is capped | `s5_capping_MIRROR.txt` |
| `s6_tables_MIRROR.py` | the tables of §4.2–4.3 from the JSON | `s6_tables_MIRROR.md` |
| `p_pictures_MIRROR.py` | headless `--shot` of the snapshot app, labelled composites | `PicsCLAUDE/*.png` |

### 6.4 Pictures (`ScratchCLAUDE/ChoppyCLAUDE/PicsCLAUDE/`, 9 PNGs, 2D unless noted)

All are headless shots of the snapshot app with its default look, except p7. `p_pictures_MIRROR.py` also writes the single panels to `PicsCLAUDE/RawCLAUDE/`; they were deleted afterwards to keep the count at 9 (rerun the script to get them back). p5 was merged into p4.

1. `p1_sq5_n3_rot7_T4_radial_wall_pulse_MIRROR.png`: the same moment (T 4) for the radial inset, the wall inset and a centred pulse.
2. `p2_tri_n3_rot30_aligned_T4_radial_wall_MIRROR.png`: a fully aligned triangle (3/3 rows) still chops with the radial inset.
3. `p3_sq5_n4_aligned_vs_unaligned_T8_MIRROR.png`: n4 at 0° (aligned) and 7°, radial and wall, T 8.
4. `p4_domain_sq5_n3_rot7_N513_1025_2049_T4_zoom_MIRROR.png`: domain mode N 513 / 1025 / 2049 at 3× zoom. The wall frames converge; the radial frames do not.
5. `p6_driven_sq5_n6_rot7_T4_default_vs_lam16_MIRROR.png`: the default drive (λ 4.2 cells: grid-scale grey moiré) vs λ 16.
6. `p7_rigid_vs_pinned_sq5_n3_rot7_N257_T4_MIRROR.png`: walls_lib leapfrog rendered with numpy, pinned vs rigid, radial vs wall. Note the rigid cavity's offset.
7. `p8_3d_sq5_n3_rot7_T4_radial_wall_MIRROR.png`: the 3D view, radial vs wall.
8. `p9_sq5_n6_rot7_radial_T05_2_8_MIRROR.png`: "becoming choppy" with the radial inset at T 0.5 / 2 / 8. The chop budget is the same 11% throughout.
9. `p10_sq5_n6_rot7_T8_N513_vs_N2049_MIRROR.png`: T 8, wall inset at N 513 and a centred pulse at N 513 (both scrambled by phase error), vs domain mode N 2049 (the real pattern).

## 7. What changed in the port (round 4c, 05/10/2026)

**Status (09/10/2026):** two items this section left open are now done, by the user's decisions: the default drive frequency is 0.03125 (λ 16 at Co 0.5; 7.3) and driven vertices are soft by default with a hard toggle (7.5, item 2). The tests of 7.1 ("driven vertices from rest") now set the hard drive explicitly (`defaults_tests_MIRROR.py` section 2, `vtx_soft=False`). [v1_decisions_MIRROR.md](v1_decisions_MIRROR.md). *Edited by Claude Opus 5.5 — 09/10/2026.*

*Added by Claude Opus 5.5 — 05/10/2026.* Recommendations 1–5 of section 5 are now in the core, `membrane_core_MIRROR.py`. Wiring them into the app panel is the next step (7.4). Tests: `defaults_tests_MIRROR.py`; logs `ScratchCLAUDE/Round4CLAUDE/LogsCLAUDE/def_defaults_{cuda,cuda_f32,vulkan_f32}_MIRROR.log` (ALL PASS on all three). Progress log: `NotesCLAUDE/progress_round4_defaults_MIRROR.md`.

### 7.1 Two new defaults

| setting | web app | port, since 05/10/2026 | how the comparisons with the web app stay whole |
|---|---|---|---|
| `Params.vtx_inset` | `'radial'` | **`'wall'`** | Every comparison sets `vtx_inset='radial'` explicitly: `crosscheck_js` (`tri_n3_conf_vtxdrive`, `tri_square_slab_vtx`), `crosscheck_ledger` (3 configurations), `crosscheck_alignment`, the scale tests' references (`ScratchCLAUDE/ScaleCLAUDE/make_ref_MIRROR.py`) and the tri-modes parity test (test 8). |
| driven vertices (`vtx_drive`) | a bump under each vertex, dented to 0 at step 1 | **start from rest**, like `cont` (`Params.vtx_drive_bump = False`) | Harness **PATCH `vtx-drive-rest`** (`ToolsCLAUDE/js_harness_MIRROR.mjs`). `vtx_drive_bump=True` gives the web app's behaviour. |

**The wall inset.** `probe_defaults_table_MIRROR.py` (in `ScratchCLAUDE/Round4CLAUDE/DefaultsCLAUDE/`) and test 1b ran the diagnosis' 12 configurations at the app's defaults (N 513, σ 3, Co 0.5):
- the radial inset reproduces the diagnosis' clip column to 0.05 points;
- the wall inset (the default) cuts **0.0000–0.0053%** of the vertex pulses, where the radial one cuts 1.22–27.12%;
- the wall inset's clearance is **2.85–4.27σ** (8.5–12.8 cells), against 0.67–1.73σ for the radial one;
- no corner is capped (log `LogsCLAUDE/def_probe_defaults_table_MIRROR.log`).

**Driven vertices from rest** (test 2, n6 rot 7, N 257, freq 0.03125):
- The seed is all zero. The clip fraction is still computed, because the web app computes it; the source note does not show it.
- With `vtx_drive_bump=True` the seed equals the undriven vertex seed exactly, and step 1 dents every bump centre by 0.8976 (neighbour mean − centre).
- The default run equals the diagnosis' variant "seeded bump removed" bit for bit (600 steps, field and probe histories). So that variant's measured effect applies (`s4b_driven_vertices_MIRROR.py`, λ 16, wall inset, chop share at T 8):

  | configuration | bump seeded (web app) | from rest (port) |
  |---|---|---|
  | sq5 n4 rot 0 | 3.3% | 0.87% |
  | sq5 n6 rot 7 | 3.1% | 1.32% |
  | tri n3 rot 30 | 1.5% | 0.37% |

**The patch.** It changes one line of the web app's `seed()`:
`if(!(S.vtxDrive && S.vtxDriveBump===false)) for(const v of VTX) addBump(v.i,v.j,1);`
- Only the harness sets `S.vtxDriveBump`. `crosscheck_js` and `crosscheck_ledger` send every `Params` key, and `vtx_drive_bump` is renamed to `vtxDriveBump`.
- The real web app never defines the key, so its own line runs there.
- In `crosscheck_js`, `sq5_n8_vtxdrive_saw` now runs the rest start against the patched web app. `tri_n3_conf_vtxdrive` keeps the web app's bump and radial inset. Both paths stay compared.
- **Results** (`LogsCLAUDE/def_crosscheck_js_*_MIRROR.log`):
  - `crosscheck_js` 27/27 on CPU f64, CUDA f64 and Vulkan f32. `sq5_n8_vtxdrive_saw` matches with du(0) = 0 and du(300) = 1.2e-15 of max|u| (CPU).
  - Against the unpatched web app (`ZRL_JS_NOPATCH=1`, CPU): 24/27. The differences are exactly the two Neumann-corner configurations and `sq5_n8_vtxdrive_saw`. That last one has du(0) = 2.5, the bumps, but its stepping from the same seed agrees to 1.5e-15: only the seed differs.

### 7.2 Three core readouts

All three are cheap and safe to call every frame. Exact return formats are in 7.4.

- **`m.source_note()`: the web app's source note (`srcNote`).**
  - The clearance from the source to the nearest wall: the nearest of the vertex sources, in cells and in σ. It equals the ledger's `CLEAR` on 14 configurations, solids included (test 3a); `crosscheck_ledger` compares that `CLEAR` with the web app's own.
  - The clip fraction, amber above 1%. The amber line carries the hint "the cut is a sharp edge the grid cannot resolve", plus "; try vertex inset: wall" for vertex sources on the radial inset.
  - The number of capped corners, and the inset line.
  - Cached on its inputs (the source, the mask version, σ, the inset, the drive settings). Test 3i at N 4097 (cuda f64 / cuda f32 / vulkan f32):
    - a repeat call costs 1.5–1.7 µs;
    - the first call after a reseed costs 0.7–0.8 ms for vertex sources;
    - and 67–73 ms for a centred pulse, whose search window grows to the whole array (17 ms at N 2049 in the quick run).
  - The clearance search (`m.clearance_from(i, j)`) is a window that doubles until it holds the nearest wall, so it costs O(clearance²).
- **`m.drive_warnings()`: hard drives under 8 cells per wavelength.**
  - Applies to `cont`, `slit` and driven vertices, in cells mode. Domain mode returns `[]`, because `scale_warnings()` already covers it; the beam keeps its own check.
  - Default text: "continuous source: lambda 4.2 cells < 8 (Co / freq): grid-scale waves - freq <= 0.0312 gives lambda >= 16 cells" (test 4).
- **`m.phase_lag()`: how far the fronts trail the exact solution, in cells per domain crossing.**
  - **Definition.** Per wavevector K the lag is r_in (1 − v_p/c), with v_p = ω(K)/|K| from the leapfrog's discrete dispersion 4 sin²(ω/2) = Co² Λ(K) (`beam_MIRROR.omega_of`) and c = Co. The readout averages that lag over the pulse's 2D energy spectrum, k³ e^(−k²σ²) per shell, for each direction. For a small lag this average is exactly the radial shift that best aligns the lattice's front with the exact one.
  - The energy peaks at k_dom = √(3/2)/σ, and the average sits at ≈ √2/σ. The diagnosis' k = 1/σ (section 4.6) reads ~2× lower.
  - **Validated** against that shift measured on the lattice (test 5). A centred pulse ran on the torus, and each ray was fitted to the exact continuum solution (the 2D Hankel transform, at t = (n + ½)Co). Results:

| case (N 1025 unless noted, about 0.7 crossings) | measured | predicted (cells per crossing) |
|---|---|---|
| sq5 Co 0.5 σ 3: 0° / 26.6° / 45° | 3.14 / 1.92 / 1.16 | 3.57 / 2.05 / 1.19 |
| sq5 Co 0.69 σ 3: 0° / 45° | 2.23 / 0.118 | 2.52 / 0.115 |
| sq9 Co 0.5 σ 3: 0° / 45° | 3.11 / 3.08 | 3.57 / 3.55 |
| tri Co 0.5 σ 3: 0° / 30° / 60° / 90° | 1.91 / 1.92 / 1.91 / 1.92 | 2.06 / 2.07 / 2.06 / 2.07 |
| tri Co 0.8 σ 3: 0° / 30° | 0.460 / 0.468 | 0.455 / 0.461 |
| sq5 Co 0.5 σ 6: 0° / 45° | 0.889 / 0.301 | 0.890 / 0.297 |
| N 257 (quick run), sq5 Co 0.5 σ 3 and tri Co 0.5 σ 3, 7 directions | 0.31–0.88 | 0–2% apart |

  - Within 0–3% while the accumulated lag is under ~0.3σ.
  - 6–13% high at 0.45–0.75σ, where the dispersion reshapes the front (7.5, item 3).
  - It reproduces the diagnosis' ratios: Co near the limit cuts the lag 2.7× (sq9 at 0.85) and 4.5× (tri at 0.8).
  - **Its readings** (`probe_defaults_table_MIRROR.py`): 

  | lattice, Co | N 513, σ 3 | N 2049, σ 3 | N 2049, σ 12 (domain mode's picture of N 513, σ 3) |
  |---|---|---|---|
  | sq5, 0.5 (the app's default) | 0.60–1.79 (1.7 crossings to one σ) | 2.39–7.15 (0.4) | 0.15–0.44 (27) |
  | sq9, 0.5 | 1.77–1.79 (1.7) | 7.10–7.15 (0.4) | 0.44 (27) |
  | tri, 0.5 | 1.02 (2.9) | 4.11–4.12 (0.7) | 0.26 (47) |
  | sq5, 0.69 | 0.06–1.26 (2.4) | 0.23–5.04 (0.6) | 0.01–0.31 (39) |
  | sq9, 0.85 | 0.66–0.67 (4.5) | 2.64–2.69 (1.1) | 0.16 (73) |
  | tri, 0.8 | 0.23 (13) | 0.91–0.92 (3.3) | 0.06 (213) |

  The values are cells per crossing over the directions, with the crossings to one σ in brackets.
  - **`crossings_to_sigma` = σ / lag_max**, the crossings until the slowest direction trails by one σ. It is a rough scale for when the fronts lose their shape. Checked against the correlation table of section 4.6: about where the correlation with the converged field falls to ~0.7–0.75. At N 513, σ 3 it reads ~2 crossings, and the correlation measured 0.74 at T 2. At N 2049, σ 12 it reads ~31 crossings, and the correlation measured 0.89 at T 16.
  - **The lines** turn amber (with the hint "domain mode / larger N / Co nearer the limit") when that is under 4 crossings, which includes the app's defaults.

### 7.3 Unchanged, by decision

- ~~**The default drive frequency stays at 0.12** (λ 4.2 cells at Co 0.5) for every driven source. The warning now names the problem. Changing a default would move every `cont` / `slit` comparison too. Recommendation 2's optional λ-16 default for driven vertices is left to the user.~~
  - **Status (09/10/2026): changed, the user's decision.** `Params.freq = 0.03125` for every driven source: λ 16 cells at the default Co 0.5 (a plain value, not coupled to Co). Every comparison with the web app sets `freq = mc.WEBAPP_FREQ` (0.12) explicitly. [v1_decisions_MIRROR.md](v1_decisions_MIRROR.md) section 2. *Edited by Claude Opus 5.5 — 09/10/2026.*
- **Recommendation 6** (a 4σ inset) and **recommendation 8** (clamping the full square's ring in `seed()`) are not done; both are low priority.
- **The colour threshold for the clearance** stays at the web app's 2.9σ. After snapping, the wall inset can leave a source at 2.67–2.9σ while cutting only ~1e-5 of it. So the clearance line is amber only when the clip is over 1% as well; otherwise it shows in the normal colour.

### 7.4 UI hand-off (for the agent who wires the panel)

**Status (05/10/2026): wired in the app (v1.0)**, as described in [app_MIRROR.md](app_MIRROR.md), section *Round 4 / v1.0*. One change from the plan below: a clip line that is not a warning is printed short (no parenthesis; *nothing* at exactly 0). The source note and the walls counts stay as they were while painting or during an object drag, and update once the edits rest. *Edited by Claude Opus 5.5 — 05/10/2026.*

**Calls** — all on the `Membrane` `m`, cheap, cached; call them per frame:

1. **`m.source_note()` → dict:**
   - `src`, `vtx` (bool), `n_sources`;
   - `clearance` (cells, or None), `clearance_sigma` (or None), `clearance_ok` (≥ 2.9σ, or None);
   - `inset` (`'wall'` / `'radial'`, or None), `capped` (int), `clip` (fraction 0–1, or None), `clip_warn` (bool), `driven_rest` (bool);
   - `lines`: a list of `(text, level)`, with level `'ok'` (green or the normal text), `'info'` (normal), `'warn'` (amber, `AMBER` in the app) or `'dim'` (grey `(0.55, 0.55, 0.60)`). The texts are plain ASCII ("sigma", "pi", "u^2"), like the panel's other texts ("lambda").
   - Print `lines` as they are, wrapped with `textwrap.wrap(t, TEXT_WRAP)` like `scale_warnings`. Place them **directly under the vertex-inset button** for vertex sources, and under the source x/y sliders for point sources.
   - **Replace** the panel's bottom line "pulse clipped at t = 0: x%" (`membrane_app_MIRROR.py` ~line 1159); the note's clip line supersedes it.
2. **`m.drive_warnings()` → list of str** (empty, or one line). Print it in amber under the freq slider. That is the slider shown for `cont` / `slit` / `beam` / driven vertices; the same spot as the domain-mode `scale_warnings()` loop, which it complements (they never both return text).
3. **`m.phase_lag()` → dict:**
   - `lag` (the worst direction unless `theta_deg` is given), `lag_min`, `lag_max`, `dir_min`, `dir_max` (degrees);
   - `sigma`, `k_dom`, `r_in`, `per_sigma`, `crossings_to_sigma`;
   - `lines`: `[(text, level)]`, with the same levels as above.
   - Show `lines` in the main panel's readout block (next to "E = …, max|u| = …") while a gaussian is seeded: `pulse`, and `vtx` unless `source_note()['driven_rest']`. A drive has no σ; for drives `drive_warnings()` is the relevant line. The first line example: "phase lag ~0.60-1.79 cells per crossing at sigma 3 (0.596 sigma): the fronts trail by one sigma after ~1.7 crossing(s)". The second, amber case: "the grid carries the pulse too slowly; it scrambles over time - domain mode / larger N / Co nearer the limit".
   - `m.phase_lag(sigma=…)` evaluates another width; the app's domain mode can show the gain from a larger N.

**Settings**
- **Vertex inset:** the *vertex inset* button already cycles `mc.VTX_INSETS` (`['radial', 'wall']`). It now starts on `wall`. Suggested button text: "vertex inset: wall (3 sigma from both walls)" / "vertex inset: radial (the web app's)".
- **Driven vertices:** add a checkbox under *drive the vertices*, shown only while it is on: "seed pulses under the driven vertices (the web app)" ↔ `p.vtx_drive_bump`. Changing it needs a reset.
- **CLI:** `--set vtx_inset=radial` and `--set vtx_drive_bump=1` work today.

**Do not**
- Do not recompute any of these per frame by hand. In particular, never call `clearance_from()` per frame for a centred source at large N: the first call is O(N²).

### 7.5 Still open

1. **The slow phase error** (cause b2) is only made visible: the readout above. Domain mode at a larger N is the remedy.
2. ~~**Driven vertices remain hard sources:** they reflect returning waves like pinned points. A soft (added) drive like the beam's would change the physics and every comparison; not attempted.~~
   - **Status (09/10/2026): done, the user's decision ("soft as default, leave a hard toggle").** `Params.vtx_soft = True`: the drive is added at the vertex cells (the beam's kernel path), so they are transparent; each harmonic is scaled by the lattice Green's function at the cell, so a lone vertex radiates the hard one's wave (measured 1.000 in free space, 0.79–0.87 at the default square). `vtx_soft = False` is the web app's hard drive, bit for bit; the comparisons set it. With `vtx_drive_bump` the bumps are seeded and the soft drive adds to them (no dent). [v1_decisions_MIRROR.md](v1_decisions_MIRROR.md) section 3. *Edited by Claude Opus 5.5 — 09/10/2026.*
3. **`phase_lag()` assumes a uniform medium** (Co = `p.Co`) and no damping. It is a linear (first-order) estimate.
   - It stays within 0–3% of the measured lag while the accumulated lag is under ~0.3σ.
   - Past that the dispersion reshapes the front, and the estimate over-reads by 6–13% at 0.45–0.75σ. Test 5 at N 1025 over 0.7 crossings: sq5 0°, 3.14 measured vs 3.57 predicted cells per crossing.
   - So it is accurate where it matters most: the early warning.
