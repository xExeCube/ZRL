# Scale mode: sizes in cells or as fractions of the domain

*Generated from scratch — Claude Opus 5.5 — 02/10/2026*

The user asked for a "finer grid" variant: "pulse width and wavelength would be given as fractions of the domain. Higher N then shows the same picture with less grid error, and the amplitude stops depending on N."

Until now every size was in **cells**, so a larger N was a bigger world with the same small pulse. At the same relative position, the ring's amplitude then fell like 1/√N. There is now a scale-mode toggle:

| mode | sizes | what a larger N does |
|---|---|---|
| `cells` (**default**, the web app's) | in cells | a bigger world, same small pulse (unchanged, bit-identical) |
| `domain` | fractions of the domain's inradius r_in | the same picture with less grid error |

## 1. What scales

r_in is `Membrane.inner_R()`, the domain's inradius in physical cells: the half-width of the square, the circle's radius, R·cos(π/n) for an n-gon, and the half-height of the plate. Source positions (sx, sy), the slit and the mode indices were already fractions of the domain, so they did not change.

| size | cells field | fraction field | in domain mode |
|---|---|---|---|
| pulse sigma | `sigma` | `sigma_f` | sigma = sigma_f · r_in |
| drive wavelength | `freq` (cycles/step) | `lam_f` | λ = lam_f · r_in, **freq = Co / λ** |
| beam width | `beam_width` | `bw_f` | w = bw_f · r_in |
| packet length | `pk_len` | `pkl_f` | L = pkl_f · r_in |
| packet wavelength | `pk_lambda` | `pklam_f` | λ = pklam_f · r_in |

The vertex insets follow sigma, as before. The impulse (one cell) and the wall brush stay in cells.

**Core** (`membrane_core_MIRROR.py`): `Params.scale` (`'cells'` | `'domain'`) and the five fraction fields (None until set).
- `apply_scale()` derives the cell fields from the fractions. `reseed()` (so on every N, shape and radius change) and `set_co()` call it. In `cells` mode it does nothing.
- So in domain mode a **Co change keeps the wavelength**: freq follows Co. The time-reflection effect is unchanged.
- `set_scale(mode)` switches without a jump. Entering `domain` turns the current cell sizes into fractions at the current N; the cell values are kept exactly and the field is not touched (tested). Leaving `domain` keeps the derived cell values.
- A fraction that is None is taken from the cell value at the first apply. So `--set scale=domain N=2049` keeps the default look of N = 2049.
- `crossings()` returns t / r_in, where t = Σ Co is the web app's clock.
- `scale_warnings()` lists the sizes that the current source uses and that fall under 2 cells of width or under `PPW_MIN` = 8 cells per wavelength at this N. It only reports in domain mode.

## 2. The app

| control | where | what |
|---|---|---|
| `scale: cells … -> domain` button | main panel, above the steps slider | toggles the mode (converts, no reset) |
| `crossings / s` slider (domain) | replaces `steps / frame` | playback speed in r_in / c per second at a nominal 60 fps; the readout below shows r_in and the steps/frame. The fractional remainder is carried, so the average is exact even below 1 step/frame (tested: 127 / 255 steps in 60 frames for 128 / 256 wanted at N = 513 / 1025). Steps/frame are capped at 1024. |
| `sigma / r_in`, `lambda / r_in`, `beam w / r_in`, `packet L / r_in`, `packet lambda / r_in` | in place of the cell sliders (domain) | ranges 0.002–0.3 (sigma) and 0.005–1.0; λ also shows the cells and freq it gives |
| amber notes | under the source sliders (domain) | e.g. `pulse sigma 0.48 cells < 2 at N = 49: under ~2 cells wide` |
| status line (domain) | top | `t = …  = 1.000 r_in/c` |
| `auto height (height scale / peak |u|)` | look section, **both modes**, off by default | the 3D height scale divided by the same recent peak as the auto colour gain, so the crest stays at the height of a unit pulse |

The nominal frame rate is used instead of the measured one, because a measured rate would feed back: more steps make slower frames, which ask for more steps.

**CLI options:**
- `--scale domain` converts at the starting N. With `--set`, the given fractions win, and the missing ones are taken at the final N.
- `--cps 0.5` sets the speed (it implies domain mode).
- `--auto-height` turns on the auto height.
- `--set sigma_f=0.05 lam_f=0.1 …` sets the fractions.

Example: `python membrane_app_MIRROR.py --N 2049 --set scale=domain sigma_f=0.05 --auto-height`.

## 3. Verification (`python scale_tests_MIRROR.py`, CUDA f64, 16/16, 82 s)

**Cells mode unchanged.** The 15 configurations (sq5, sq9 and tri × pulse, cont, vtx-driven, beam and packet; N = 161, 300 steps, both time levels) are **bit-identical** to the core before the change. The reference fields are `ScratchCLAUDE/ScaleCLAUDE/ref_cells_{cuda,cpu}_MIRROR.npz`, made by `make_ref_MIRROR.py` from the pre-edit copy `membrane_core_before_MIRROR.py`. The fraction fields are ignored in cells mode (tested). The JS cross-check (27/27) and the ledger cross-check also pass unchanged (§5).

**Convergence.** A centred pulse with sigma = 0.05 r_in, Co = 0.5, Dirichlet square, at N = 257, 513, 1025 and 2049.
- The field is taken at T = 0.5 crossings, before the ring reaches a wall.
- It is sampled at the same **fractional** positions, a 181 × 181 grid on [−0.9, 0.9]², using cubic-spline resampling in space and cubic Lagrange in time.
- On tri, r_in = c·√3/2 − 1.5 is not exactly proportional to N, so the grids do not nest; that is why the field is resampled rather than injected.
- The table gives the maximum difference between successive N, divided by the peak:

| lattice | 257→513 | 513→1025 | 1025→2049 | ratios (max) | ratios (RMS) |
|---|---|---|---|---|---|
| sq5 | 1.60e-2 | 3.98e-3 | 9.94e-4 | 4.02, 4.01 | 4.01, 4.00 |
| sq9 | 1.71e-2 | 4.23e-3 | 1.06e-3 | 4.03, 4.01 | 4.01, 4.00 |
| tri | 1.58e-2 | 3.84e-3 | 9.51e-4 | 4.10, 4.04 | 4.08, 4.04 |

That is clean second order: 4× per doubling.

**A half step matters.** A run started at rest (u_prev = u, the web app's seed) is the continuum solution that is at rest **half a step earlier**: u(−Δt) = u(0) is the time symmetry about t = −Δt/2. So level n samples t = (n + ½)Δt. Comparing at t = nΔt instead mixes in a first-order error, and the ratios drop to **1.55–1.87**. The test uses the offset. The app's readout keeps the web app's clock t = nΔt; the difference is ½ step, or 0.001 crossings at N = 257.

**Amplitude independent of N.** The test records the peak over time of u at the fractional point r = 0.5 r_in (0° and 45°), sampled every step and refined with a parabola:

| lattice | N = 257 | 513 | 1025 | 2049 | spread | last doubling |
|---|---|---|---|---|---|---|
| sq5, 0° | 0.11695 | 0.11765 | 0.11783 | 0.11787 | 0.79% | 0.037% |
| sq5, 45° | 0.11810 | 0.11794 | 0.11790 | 0.11789 | 0.18% | — |
| sq9, 0° / 45° | 0.11727 | 0.11774 | 0.11785 | 0.11788 | 0.52% | 0.024% |
| tri, 0° / 45° | 0.11737 | 0.11777 | 0.11786 | 0.11788 | 0.43% | 0.019% |

The peak converges to 0.1179 at second order: the differences fall 4× per doubling. The sq5 axis/diagonal gap at N = 257 (1%) is that lattice's anisotropy, and it closes to 0.02% at 2049. For contrast, **cells mode** (sigma = 3 cells) at the same fractional point gives 0.0750, 0.0494, 0.0310 and 0.0208: ratios of 0.66, 0.63 and 0.67 per doubling, about 1/√2 as expected.

**Logic tests:**
- The mode switch keeps sizes and field.
- An N change from 257 to 1025 keeps every fraction to 1e-14.
- A Co change keeps λ.
- Back in cells mode, an N change no longer scales the sizes.
- The warnings appear (sigma, beam λ and width) only in domain mode.
- `crossings()` = steps · Co / r_in.

**App** (`app_tests_MIRROR.py`, 7 new checks, 95/95 in all):
- The default is cells mode.
- The button converts without a jump.
- A fraction slider resets with sigma = f·r_in.
- The N slider keeps the fraction.
- The crossings/s playback works.
- The amber warning appears.
- The auto height is 1.0 at the start and 10.6 after one crossing, and renders in 3D.
- Back to cells gives steps/frame from the speed.

## 4. Caveats and open items

- **Walls.** The convergence test stops before the ring reaches a wall. A staircased curved or tilted wall (circle, n-gons, the tri square) is a first-order boundary error, so after reflections expect less than 4×. The full Dirichlet square on sq is exact (the wall is the array edge).
- **Auto window in steps.** The auto colour gain and the auto height use the peak of the last 64 recorded steps. In domain mode at large N, that is a smaller fraction of a crossing. It is still adequate (64 steps ≥ 0.008 crossings at N = 8193), but the window could follow N.
- **Brush.** The wall brush (and the brush ring) stay in cells. Objects are already in grid units, so they scale.
- **Resolution warnings.** These are rules of thumb (8 points per wavelength, 2 cells of width). For a pulse, sigma = 2 cells still leaves a visible dispersion tail on sq5. The convergence table starts at sigma = 6.4 cells.
- **Pre-edit copy.** `ScratchCLAUDE/ScaleCLAUDE/membrane_core_before_MIRROR.py` is kept only so the reference fields can be remade; the test needs only the `.npz` files.
