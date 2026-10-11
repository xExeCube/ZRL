# Directional sources: the beam and the packet

*Generated from scratch — Claude Opus 5.5 — 01/10/2026*

The user asked for "a beam - like a directed wave, as a source option, then having its width and direction be configurable". There are two new source types. Neither has a web-app counterpart, and no existing configuration changes (the JS cross-check stays 27/27).

| source | what it is | parameters |
|---|---|---|
| `beam` | a continuous, steered Gaussian beam from a driven aperture, **one-sided** by default | angle, width w, position (source x/y), freq + waveform, one-sided, energy aim |
| `packet` | a single wave packet launched one way (initial condition) | angle, width w, length L, wavelength, position, energy aim |

Code: `beam_MIRROR.py` holds the numpy maths (lattice symbols, dispersion, aim, aperture tables, packet levels). `membrane_core_MIRROR.py` has `build_beam`, `_add_packet`, `beam_geometry` and the `SRC_BEAM` kernel branch. The app is `membrane_app_MIRROR.py` (`beam_controls`, `beam_segments`). Measurements come from `beam_tests_MIRROR.py`.

## 1. The lattice's discrete dispersion (what everything is built on)

For a plane wave with physical wavevector K, the core's Laplacians give −Λ(K):

- sq5: Λ = 4 sin²(kx/2) + 4 sin²(ky/2)
- sq9: Λ = 10/3 − (4/3)(cos kx + cos ky) − (2/3) cos kx cos ky
- tri: Λ = (2/3)(6 − 2 Σ cos(K·e)), with e = (1,0), (½, √3/2), (−½, √3/2)

Leapfrog, undamped: **4 sin²(ω/2) = Co² Λ(K)** (ω in rad/step; Co² is the per-cell CC at the source, so media are respected). Group velocity: **∇ω = Co² ∇Λ / (2 sin ω)**. All three small-k limits give Λ = k².

**Aim.** On an anisotropic lattice, energy (group velocity) and phase fronts point in different directions, except on symmetry axes. With *aim the energy* on (the default), the angle you set is where the energy goes. The wavevector angle θ_k that achieves this is solved by Newton's method, and the panel shows it as *phase fronts at …*. With it off, the angle sets the phase fronts and the beam walks off.

## 2. The beam

**Aperture.** The front aperture is the slab of cells with ζ ∈ [−d/2, d/2) and |s| ≤ 2.5 w. Here ζ is the coordinate along the beam (n̂ at θ_k), s the coordinate across it, and d = −n̂·v, where v is the nearest-neighbour vector most opposed to n̂. On sq, d ≥ 0.707; on tri, d ≥ 0.866. Shifting this slab by v tiles the lattice, so it is exactly one cell thick in the lattice's own sense. Slab-edge ties are quantised (1e-9): on the axes, whole rows of cells sit exactly on an edge. Without the quantisation, cos/sin round-off sorted them at random, and tri at 180° fell from 29.6 to 18.4 dB.

**Drive (per cell, per harmonic h of the waveform).** Each cell is driven with `a_h · S_h · exp(−(s/w)²) · sin(hωn − k_h ζ)`. Here k_h is the discrete wavenumber of hω along n̂, so each cell samples the plane wave that should leave the aperture. The phase delay k_h ζ is what steers the beam and corrects the staircase of the rasterised line. The sheet gain is **S = Co² (dΛ/dk) / d**, which gives a unit plane wave on each side. It is exact for an infinite sheet normal to an axis (1D check: U = S / (2 Co² sin k)). Measured: the near-field peak is 1.017–1.019 on all three lattices. The beam switches on with a raised-cosine ramp over 4 periods. Harmonics past the cut-off in that direction are dropped; if the fundamental is past it, the panel warns and nothing is driven.

**Soft (added), not hard (imposed).** The kernel does `U_new[cell] += drive`, after the boundary pass, like the other sources.

- An imposed line would be a **mirror**: every wave returning from a wall would reflect off the aperture, building a cavity between the aperture and the walls. A soft aperture is transparent.
- Added forcing does not change the update's amplification matrix, so the CFL limit is unchanged. Measured: Co = 0.99 C_max on sq5, sq9 and tri, with Dirichlet and Mur, for 3000 steps: finite and bounded.
- The existing `cont` source stays hard. It is one cell, and the web app defines it that way.
- In a closed undamped domain, a continuous soft drive pumps energy, as `cont` does. Use Mur or damping for a steady picture.

**One-sided (default).** A second, *back* aperture sits a distance d behind the front one. It is its own slab, with ζ ∈ [−3d/2, −d/2), and every cell is phased by its own position. It emits the **inverted** front signal **delayed by the travel time d/v_p**: its lag is k ζ + 2kd + π. Backward, the two sheets cancel; forward, they add with gain |1 − e^{2ikd}| = 2|sin kd|, which is divided out so the forward amplitude stays 1. This is a total-field/scattered-field-style pair built from the discrete k, so it works on sq5, sq9 and tri alike.

- **Why its own slab and not the front cells shifted by v** (`BACK_MODE = 'shift'`, kept for comparison): off the axes, v also has a component *across* the beam. That leaves a residue of first order in the backward lobe's angle α: 17.6 dB at 45° on sq. With the slab, the sheets are exactly d apart *along* the beam, and the residue is second order, ≈ (kd/2)² ⟨α⁴⟩. For w0 = 8, λ = 12 that predicts −32 dB; ~29–30 dB is measured.
- Two-sided (checkbox off) drives only the front aperture: a symmetric pair of beams (measured 0.0–0.4 dB forward/backward).

**Walls and solids.** Aperture cells that fall outside the domain, inside a solid, or on the array ring are skipped (counted as *clipped* in the panel). Nothing crashes, and nothing is ever written into a solid; solid cells stay exactly 0 (tested). A clipped aperture loses the cancellation locally. Expect a weak backward/sideways leak where the aperture meets a wall, and a narrower effective aperture. A source requested inside a solid is moved by the existing snapping rule. Pinned solids in the path reflect the beam: backward energy ×187 (sq5), ×198 (sq9), ×299 (tri) with a 10×100-cell slab ahead.

**Live changes.** Angle, width, one-sided, aim, frequency, waveform and Co rebuild the tables on the next step (key-checked in `step()`), without a reset, so you can steer while it runs. The phase stays continuous in time, but not in space, so a quick swing leaves a short transient.

## 3. The packet

u⁰ = exp(−(ζ/L)² − (s/w)²) · cos(kζ), with k = 2π/λ. The previous level u⁻¹ is built **mode by mode**:

1. FFT the packet on a local box (index space; on tri the index wavenumbers map to K = (a, (2b − a)/√3)).
2. Keep only the plane waves with K·n̂ > 0, doubled. The K·n̂ = 0 line has weight 1, within a 1e-7 tolerance so mirror symmetry stays exact. The real part of the sum is still u⁰.
3. Move each wave back by its **own** discrete ω(K).

The packet then travels one way under the lattice's own dispersion. Measured backward residue: **≤ 9e-6 of its energy** on all three lattices. The same packet started at rest leaves 0.50 behind. **P** launches another packet.

The envelope's speed equals the packet's **spectrum-weighted** discrete group velocity to within 0.05%. The central-K v_g is 1.5% faster, because a finite-width packet's off-axis plane waves go partly sideways (⟨θ²⟩ ≈ (λ/2πw)²). The panel shows the weighted value. Direction error: ≤ 0.02°.

## 4. Measurements (`python beam_tests_MIRROR.py`, CUDA f64, Co = 0.5)

**Directivity**, forward/backward energy of the time-averaged field (|ζ| > 3), λ = 12, 520 steps:

| lattice | 0° | 20° | 30° | 45° | 60° | two-sided | one-sided, w0 = 16 |
|---|---|---|---|---|---|---|---|
| sq5 | 29.2 | 30.7 | 28.7 | 29.8 | 28.7 dB | 0.0–0.3 dB | 34.9–37.9 dB |
| sq9 | 29.4 | 30.1 | 29.4 | 29.6 | 29.4 dB | 0.0–0.3 dB | 32.2–37.7 dB |
| tri | 29.6 | 29.9 | 29.5 | 29.9 | 29.6 dB | 0.0–0.4 dB | 33.1–37.8 dB |

(The one-sided columns from 0° to 60° are w0 = 8.) Wider apertures cancel better, since their backward lobe is narrower. Tri at 60° was 26.5–27.2 dB until the aim stopped iterating on a symmetry axis: a 2e-9 rad Newton residue was enough to break the slab-edge ties again (the aim tolerance is now 1e-9).

**Width** vs the Gaussian-beam law w(z) = w0 √(1 + (z/z_R)²), z_R = π w0²/λ, with w0 = 16, λ = 12 (z_R = 67, half-angle λ/(π w0) = 0.237 rad). The measured 1/e² intensity half-width (2× the RMS) is within **1–7% for z = 100–250** on all lattices and at 0°/30°/45°. The far-field half-angle measures 0.235–0.255 against 0.237–0.238. One near-field exception: tri at 45°, z = 50 is 17% wide. At w0 = 8 (half-angle 0.47 rad) the beam is no longer paraxial and the law does not apply (the measured far angle is 0.29–0.31).

**Steering**, measured energy direction minus requested, from the centroid on an annulus of 100–200 cells, with w0 = 1.5 λ:

| lattice | λ = 6 | λ = 8 | λ = 12 | λ = 24 | phase aim, λ = 6 / 12 |
|---|---|---|---|---|---|
| sq5 | ≤ 0.41° | ≤ 0.21° | ≤ 0.09° | ≤ 0.02° | ≤ 2.61° / ≤ 0.58° |
| sq9 | ≤ 0.03° | ≤ 0.01° | ≤ 0.00° | ≤ 0.01° | ≤ 0.12° / 0.00° |
| tri | ≤ 0.03° | ≤ 0.01° | ≤ 0.03° | ≤ 0.02° | ≤ 0.05° / ≤ 0.01° |

All angles 0–90° in steps of 10–15°; errors vanish on the symmetry axes. sq5 is the anisotropic one: its phase-aimed beam bends toward the diagonals by up to 2.6° at 6 cells per wavelength. The energy aim takes most of that out; the remainder is the beam's spread of wavevectors.

**Cost** (ms/step, recorder off / on). The host computes the drive (K steps × cells × harmonics) and uploads it once per launch, like `cont`.

| N | sq5 pulse | sq5 cont | sq5 beam w 8 (82 cells) | sq5 beam w 64 (642) | tri beam w 8 / 64 |
|---|---|---|---|---|---|
| 1025 | 0.097 / 0.138 | 0.165 / 0.252 | 0.172 / 0.232 | 0.211 / 0.267 | 0.221 / 0.206 |
| 2049 | 0.218 / 0.345 | 0.288 / 0.431 | 0.299 / 0.439 | 0.299 / 0.434 | 0.277 / 0.275 |

The packet costs nothing per step (its set-up is one FFT of a local box, milliseconds).

## 5. App

- **O** cycles … vtx → beam → packet → impulse.
- Main panel, under *source*: *beam angle (deg)* (−180…180, counter-clockwise from +x), *beam width w (cells)* (the waist: amplitude exp(−(s/w)²)), *aim the energy (group velocity)*. For the beam, also *one-sided* and the freq slider, plus a readout: λ in cells, cells driven / clipped, the phase-front angle, and an amber warning under 8 cells per wavelength. For the packet: *packet length L* and *packet wavelength*, plus the group speed.
- **Emitter overlay** (toggle *draw the emitter*): the aperture in cyan and an orange arrow along the requested direction, in 2D and in 3D. In 3D it is lifted above the surface and slid along the view ray, like the brush ring.
- Pick-mode typing and Ctrl+click work on the new sliders unchanged (tested: typed angle 37.5 → θ_k 37.18 on sq5, applied live; a typed wavelength relaunches the packet).
- `--shot` example: `python membrane_app_MIRROR.py --shot b.png --view 2d --steps 700 --set src=beam freq=0.04 beam_angle=30 beam_width=10 sx=-0.5 sy=-0.4`

## 6. Ledger

- `beam` counts as a driven source (like `cont`/`slit`). Energy, edge-kind and linearity rows treat it as open.
- Rows that need a point, pulse or impulse source were already n/a for anything else.
- New: the **energy row is n/a for a packet on the triangular lattice**. The recorder's energy is the square lattice's (forward differences on the index axes). On tri it is not the conserved quantity, and for a *directed* wave it changes whenever the packet turns at a wall: 5.4% drift on n6 is a false FAIL, while the solver is exact. A pulse, being isotropic, stays at 0.7%. On sq5/sq9 the packet's energy row genuinely applies and PASSes (drift 2–7e-3).
- Probe over beam/packet × {sq square Dirichlet, sq circle Mur, tri n6}: 0 FAIL; CFL, staircase and alignment rows grade as before.

## 7. Limitations

- **Second-order cancellation only.** The backward lobe is cancelled exactly straight back and to second order around it (~30 dB at w0/λ ≈ 0.7, ~35 dB at 1.3). A third sheet could push it further.
- **Undamped dispersion.** k comes from the undamped dispersion. With damping or a strongly varying medium at the aperture (it uses CC at the source cell), the design is slightly off.
- **Phase jump on a fast change.** A fast angle/width change mid-run is applied instantly (no re-ramp), so it leaves a transient.
- **Mur interaction.** Mur 1 on sq is still the only absorber; a beam hitting a Mur edge obliquely reflects ~6% (see the absorbing-boundaries note).
- **Clipping.** Clipped apertures lose the one-sided cancellation where they are cut. The packet's FFT box is clipped by the array edges (its `in_array` share is in `m.packet_info`).
- **High harmonics.** The square/saw waveforms' harmonics each get their own k and gain. Harmonics above the cut-off in that direction are dropped, which rounds the waveform.
