# Circle fidelity of the membrane lattices: how round does a round drum stay?

*Claude Opus 5.5, 02/10/2026. Research code, data and figures: `ResearchCLAUDE/CircleFidelityCLAUDE/`
(`circle_lib_MIRROR.py`, `sweep_MIRROR.py`, `jitter_MIRROR.py`, `freespace_MIRROR.py`, `tables_MIRROR.py`,
`figures_MIRROR.py`; raw results in `OutCLAUDE/`, full tables in `OutCLAUDE/tables_MIRROR.txt`, figures `fig1..fig5_*_MIRROR.png`).
Every number below comes from runs of the real solver (`membrane_core`, Taichi CUDA, float64).*

## 1. Short answer

- **Ranking: tri best, sq9 close behind on the first wall hit, sq5 far behind.** With a pulse that
  scales with the drum (sigma = R/40), N = 2049, after 1 round trip: tri 38 ppm, sq9 44 ppm, sq5 180 ppm
  of the energy has left the circular shape. After 10 trips: tri 280 ppm, sq9 1 200 ppm (4.4x tri), sq5 9 500 ppm (34x tri).
- **What breaks the circle differs.** On sq5 it is mostly the lattice itself (its waves travel at
  different speeds in different directions). On sq9 and tri it is almost all the **staircase wall**:
  their bulk anisotropy is 10^4 to 10^8 times smaller than what the wall does.
- **The time step (Co) does not matter for roundness** (changes mostly < 2 %, at most 11 %). It does matter a little for the waveform timing (dispersion).
- **Proposed score, the "circle error element" CE:** energy fraction outside the radial mode after one
  round trip, centred pulse sigma = R/40, in ppm. Details in section 8.

## 2. The method (why no reference is needed)

A pulse that starts exactly at the centre of a perfect circle stays perfectly round forever: in polar
coordinates the exact solution has only angular mode m = 0. So I resample the grid field on polar rings
(quintic spline, about one sample per cell of arc), Fourier-transform each ring in angle, and split the
energy (kinetic + gradient) by angular mode m.

- **eps_ang** = energy in all modes m != 0 / total energy. Exactly 0 for the true solution at every time,
  so any value is grid error, and no reference solution is needed. Measurement floor (the seeded pulse at t = 0): 2-4e-10.
- **eps_0** = energy of (the m = 0 part minus the exact answer) / total energy. The exact answer is a
  Fourier-Bessel series of the same Gaussian in the same circle, compared **at equal times**. This catches
  timing / waveform errors (dispersion) that leave the pulse round but late or smeared. eps_0 near 2 means
  "completely out of step"; it cannot detect shape errors and eps_ang cannot detect timing errors, so the two are complementary.

Sweep: sq5 / sq9 / tri x N = 257, 513, 1025, 2049 (radius R = 0.433 N, the same R in lattice spacings for
all three) x Co = 0.5, 0.9 Cmax, 0.99 Cmax x sigma = 3 cells (app default) or R/40, 10 round trips, 4 samples
per trip. 72 runs, all finished (N = 2049 took about 50 s per run, so no extrapolation was needed).
Extra runs: 8 nearby radii at N = 513 ("staircase luck"), and a free-space control (section 6).

## 3. Which angular modes appear (checked)

- **Square lattices (sq5, sq9): only m = 4, 8, 12, ...** Every other mode sits at 1e-16 to 1e-29 (rounding).
  m = 4 is the biggest (N 2049, R/40, trip 1: sq5 1.5e-4, sq9 2.0e-5), then a comb of multiples of 4 up to m = 64 and beyond.
- **Triangular lattice: only m = 6, 12, 18, ...** (m = 4, 8, 28 at 1e-15 to 1e-17). m = 6 and m = 24 lead (1.3e-5, 6e-6).
- This holds for the wall as much as for the bulk: the staircase drawn on a lattice has the same 4- or
  6-fold symmetry as the lattice. So **a big m = 4 is not proof of bulk anisotropy**: on sq9 the m = 4
  energy is 4 million times larger than the bulk part (section 6) and comes from the wall.
- Beyond m = 8, the sq5 and sq9 spectra are nearly identical (fig3): same mask, same staircase.

## 4. Ranking and scaling with N

Trip-1 / trip-10 eps_ang, Co = 0.5:

| | N 257 | N 513 | N 1025 | N 2049 | slope vs N (513-2049), trip 1 |
|---|---|---|---|---|---|
| sigma R/40: sq5 | 1.3e-1 / 2.7e-1 | 1.6e-2 / 2.2e-1 | 1.1e-3 / 5.3e-2 | 1.8e-4 / 9.5e-3 | N^-3.2 |
| sigma R/40: sq9 | 1.1e-2 / 6.5e-2 | 2.6e-3 / 2.7e-2 | 2.4e-4 / 2.2e-3 | 4.4e-5 / 1.2e-3 | N^-2.9 |
| sigma R/40: tri | 1.1e-2 / 7.9e-2 | 1.8e-3 / 8.0e-3 | 2.5e-4 / 1.9e-3 | 3.8e-5 / 2.8e-4 | N^-2.8 |
| sigma 3: sq5 | 8.9e-2 / 0.21 | 0.22 / 0.47 | 0.40 / 0.69 | 0.60 / 0.83 | N^+0.7 |
| sigma 3: sq9 | 8.5e-3 / 4.7e-2 | 1.3e-2 / 0.14 | 1.0e-2 / 0.13 | 1.5e-2 / 0.26 | N^+0.1 |
| sigma 3: tri | 8.9e-3 / 6.7e-2 | 9.0e-3 / 7.0e-2 | 9.5e-3 / 0.14 | 9.0e-3 / 0.14 | N^0.0 |

- **Pulse scaled with the drum (R/40):** everything converges, about 7x less per doubling of N (N^-2.8 to
  N^-3.2). tri and sq9 are tied on the first wall hit (ratio 1.0-1.4, inside the staircase luck), tri pulls
  ahead with time (sq9/tri at trip 10: 0.8, 3.4, 1.2, 4.4). sq5/tri: 5-12x at trip 1, 27-34x at trip 10 (N >= 513).
- **Pulse fixed at 3 cells (app default):** a bigger drum does NOT help. For sq9/tri each wall hit scatters
  about 1 % of the energy whatever N (the stair steps look the same to a 3-cell pulse). sq5 gets **worse**
  with N (9 % to 60 % at trip 1), because its bulk error builds up with the distance travelled, and a bigger drum is a longer path.
- **Staircase luck** (8 radii at N = 513, trip 1 medians [min..max]): sigma 3: sq5 0.21 [0.20..0.22],
  sq9 9.2e-3 [5.5e-3..1.3e-2], tri 6.9e-3 [4.8e-3..9.0e-3]; sigma 5.5: sq5 1.3e-2, sq9 1.3e-3 [6.8e-4..2.6e-3],
  tri 1.2e-3 [7.7e-4..1.8e-3]. A 2-3x spread inside one lattice: **sq9 vs tri cannot be separated at
  one radius on trip 1**; tri's lead shows over several trips and in the spread (tri's worst ~ sq9's median).
- Cost note: at equal R, tri has 15 % more nodes (2.84 M vs 2.46 M at N 2049); Cmax 0.707 / 0.866 / 0.816 (sq5 / sq9 / tri).

## 5. Time step (Co) and dispersion

- **eps_ang is independent of Co:** Co 0.5 vs 0.99 Cmax changes it by < 2 % in most cases, at most 11 % (sq9, N 257, trip 10).
  Roundness is a space-discretisation property.
- **eps_0 (timing) does depend on Co**, mildly, and better near the limit: tri N 2049 R/40 trip 10: 0.35 (Co 0.5) → 0.23 (0.99 Cmax); sq5 0.36 → 0.29; sq9 0.50 → 0.34.
- **Dispersion is the bigger error overall.** Before the wall (t = T/4) eps_0 falls 16x per doubling
  (2nd order: 4.5e-3, 2.8e-4, 1.7e-5 for tri / sq5 at N 513..2049), sq5 and tri equal, sq9 2.2x worse.
  After the first wall hit it falls only 6-8x per doubling. Likely cause (not verified): the staircase wall
  sits at a slightly different effective radius from the nominal R, which shifts the echo time by a fixed amount per trip.
- At sigma = 3 cells eps_0 is 0.8-2 already after one trip at every N: the default pulse is too narrow to
  keep its waveform; it is round-ish but its timing and shape are gone. This is the main reason the default pulse looks "wrong" after a few trips.

## 6. Staircase vs lattice anisotropy (free-space control)

Same pulse in a much larger drum, travelling the same distance as one round trip (2R) with the wall out of
reach: its m != 0 energy is the lattice's own anisotropy only.

| N (path 2R) | sigma | sq5 free / circle trip 1 | sq9 free / circle | tri free / circle |
|---|---|---|---|---|
| 513 (440 cells) | 3 | 0.19 / 0.22 (**85 % bulk**) | 1.4e-4 / 1.3e-2 (1 %) | 9.3e-6 / 9.0e-3 (0.1 %) |
| 1025 (884) | 3 | 0.38 / 0.40 (**95 %**) | 5.7e-4 / 1.0e-2 (6 %) | 3.7e-5 / 9.5e-3 (0.4 %) |
| 513 | 5.5 (R/40) | 8.9e-3 / 1.6e-2 (**55 %**) | 3.3e-7 / 2.6e-3 (1e-4) | 2.1e-8 / 1.8e-3 (1e-5) |
| 1025 | 11 (R/40) | 5.6e-4 / 1.1e-3 (**50 %**) | 1.3e-9 / 2.4e-4 (5e-6) | 7.9e-11 / 2.5e-4 (3e-7) |

- Free-space anisotropy grows with the **square of the path** (x4 per doubling at fixed sigma; the T/4
  samples in the sweep times 16 reproduce the 2R values exactly).
- With sigma = R/40 it falls **N^-4 on sq5** (16x per doubling) and **N^-8 on sq9 and tri** (260-270x):
  sq5's direction error is 2nd order, sq9's and tri's are 4th order. **tri's bulk anisotropy is exactly
  16x below sq9's** at every N (energy; 4x in amplitude).
- The wall falls only ~N^-3, so sq5's bulk share shrinks with N (55 % → 50 % → ~20 % at N 2049 from the T/4 sample), but it is still sq5's handicap.
- For sq9 and tri, **fixing the wall is the only lever left**; a better interior stencil would change nothing visible.

## 7. Growth per round trip

Log-log slope of eps_ang(trip k), k = 1..10 (Co 0.5, sigma R/40, N 1025 / 2049): sq5 1.70 / 1.74,
sq9 0.94 / 1.47, tri 0.90 / 0.88.
- **Slope ~1 (tri): the wall adds about the same amount every hit, with random phases**, so energy adds
  up linearly: tri N 2049 gains +3.8 to 5.6e-5 per trip for trips 1-5, then levels off at ~2.8e-4 (some
  energy goes back into m = 0; eps_ang need not grow forever).
- **Slope ~2 (sq5): a steady, coherent direction-dependent drift**: the m = 4 amplitude grows with
  distance, energy with its square. Per-trip gain rises from 1.8e-4 to 1.3e-3 by trip 7.
- sq9 is in between (the wall dominates, but the wall-made m = 4 is partly coherent from trip to trip).
- At sigma 3, sq5 saturates (0.6-0.9 of the energy off-circle; it even drops after trip 5), sq9/tri grow with slope 1.0-1.25.

## 8. The proposed score: "circle error element" (CE)

**Definition.** Circle domain, Dirichlet wall, no damping, Gaussian pulse at rest exactly at the centre,
width sigma = R/40. Run one round trip (t = 2R/c). CE = energy in angular modes m != 0 divided by total
energy, from the polar-FFT split above. **Units: dimensionless, shown in ppm** (parts per million of energy).
Companions: (a) growth exponent p (slope over trips 1-10), (b) bulk share b = free-space value / CE.

**How to read it.** CE = 38 ppm (tri, N 2049) means 0.0038 % of the energy has left the round shape per
round trip; as an amplitude, the non-round ripple is about sqrt(CE) = 0.6 % of the pulse. 1 % of energy
(10 000 ppm) is a ~10 % ripple, visible by eye. p near 1 says "wall-made, random"; p near 2 says
"lattice-made, systematic". b near 1 says the lattice is to blame, b near 0 the staircase.

**Why sigma = R/40 and not the default 3 cells:** with a pulse that scales with the drum, CE converges with
N and ranks lattices fairly. With sigma fixed in cells, CE says more about the pulse than about the grid
(section 4). Values at the app default are useful as a warning label (N 513: sq5 22 %, sq9 1.3 %, tri 0.9 %).
**Report the median of 3-4 nearby radii** (or say the spread is 2-3x).

**In the app (proposal only, not built).**
- **Ledger row** `circle error element`: per lattice and N, CE in ppm at trip 1 and trip 10, with p and the
  bulk/wall split, run headless like the other ledger checks (seconds on CUDA up to N 1025).
- **Live readout** when the domain is a circle and the source is centred: compute eps_ang once per round
  trip (one spline prefilter + polar FFT, cheap compared to a frame) and show "circle error: 38 ppm / trip
  (wall)". Show eps_0 next to it as "timing error" so a user does not read a round-but-late pulse as good.

## 9. What could beat both lattices, and what the score would need

- **Higher-order isotropic stencils** (4th-order 13-point square, 4th-order triangular): cut dispersion
  (eps_0) a lot and sq5-type bulk anisotropy, but for sq9 and tri the bulk is already 10^4-10^8 below the
  wall, so **CE would not move** with a staircase wall. Wider stencils also make the wall harder (more ghost nodes).
- **Cut cells / embedded boundary** (keep the lattice, correct the weights in the wall band): attacks
  exactly the dominant term. The earlier study's tri + fitted edge was 600-1400x better than staircased tri.
  Cost: tiny cut cells can lower the stable time step unless merged.
- **Conforming unstructured mesh**: exact wall, but irregular nodes scatter into all m (earlier: ~100x
  behind tri + edge, still ahead of staircased lattices).
- **Polar mesh / hex shells bent into circles**: exactly round about the centre, so a centred pulse
  stays m = 0 **by construction** and CE would read ~0 while the scheme still has a centre singularity and stretched cells.
- **10-fold quasicrystal (Penrose) points with cotangent (Laplace-Beltrami) weights**: 10-fold on average,
  so the lowest symmetric anisotropy would be m = 10 and above, but varying local neighbourhoods scatter
  into every m (earlier: Penrose + edge 15-35x behind tri + edge).

What CE would need to compare those fairly:
1. an **off-centre pulse** (and several positions), so rotationally symmetric meshes are not rewarded for
   free; then the exact answer is no longer pure m = 0 and a reference is needed (Bessel series via the
   addition theorem, or a much finer run);
2. **cost normalisation**: CE at equal work (nodes x steps, with each scheme's own Cmax), not equal N;
3. **eps_0 next to CE**: edge fixes remove shape noise, not timing errors;
4. a **disorder check**: energy outside the lattice's own mode family (m not a multiple of 4 or 6) is 0 on
   lattices; on quasicrystals and unstructured meshes it measures scatter and is worth reporting separately;
5. the **median over radii**, and sigma tied to R.

## 10. Agreement with the earlier findings file

`ResearchCLAUDE/CircleFidelityCLAUDE/circle_fidelity_findings_MIRROR.md` (01/10/2026: a numpy
re-implementation of the stencils, radial spline fit instead of angular FFT, R = 60, sigma 6 mostly).

**Agrees:**
- tri is the best lattice; the 2D circle cannot be judged by "cycle n = cycle 1" but by roundness.
- sq5's noise is mostly its own bulk (earlier ~90 % at sigma 3, ~half at sigma 6; here 85-95 % and ~50 %);
  for sq9 and tri almost all of it is the staircase.
- Free-space values match after scaling by path squared: earlier sq9 1.0e-5 and tri 6.5e-7 over 120 cells
  give 1.3e-4 and 8.7e-6 over 440 cells; measured here 1.4e-4 and 9.3e-6. Two independent codes and two different measures agree.
- Staircase luck 2-3x; sigma 3 ruins the waveform (eps_tot / eps_0 ~ 1-2 within a few trips); sq9's
  dispersion ~2x worse than sq5/tri (here 2.2x).

**Differs:**
- Earlier (R 60, sigma 6) sq5, sq9 and tri were level at trip 1 (8e-4 / 1.3e-3 / 1.0e-3) and sq5 ≈ sq9 at
  trip 10. Here sq5 is 5-12x worse at trip 1 and ~30x at trip 10. Not a contradiction: sq5's bulk term grows
  as (path / sigma)^2 x (1/sigma)^4, which is about 20x larger in this study's geometry (path 440, sigma 5.5) than in the earlier one (path 120, sigma 6).
- Earlier N-scaling (sigma R/10, N 64-244) was too noisy to give slopes; here, with 4 sizes up to N 2049,
  trip 1 falls cleanly as N^-2.8 to N^-3.2.
- Earlier eps_tot fell 13-14x per doubling; here eps_0 falls 16x before the wall but only 6-8x after it
  (probably the effective-radius offset; the earlier file measured R_eff - R ≈ +0.31..0.35 for staircases).
- Earlier open item "sq9 + fitted edge": supported indirectly; sq9's bulk is only 16x noisier than tri's and
  both are tiny (1.3e-9 and 7.9e-11 at N 1025), so sq9 + edge should land close to tri + edge. Still not built.

## 11. Caveats

- One centred pulse shape (Gaussian at rest); m != 0 cannot arise from the initial data, only from the grid.
- Measured energy is the continuum energy of the spline-interpolated field, not the scheme's own discrete
  energy; for sq9 at sigma 3 they differ by 0.25 % (free-space E 1.567 vs pi/2), at sigma >= 5.5 by < 0.07 %.
- sq5's bulk share at N 2049 (~20 %) comes from the T/4 sample x 16, not from a separate free-space run.
- eps_ang is not monotone in time (sq5 at sigma 3 falls after trip 5; tri levels off), so the score uses trip 1, plus trip 10 for growth.
