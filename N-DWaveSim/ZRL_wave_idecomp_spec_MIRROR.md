# Zero-Recursive Lattice — Wave i-Decomposition Renderer, Spec

**For:** a coding agent (Claude Code) working on `wave_idecomp_MIRROR.html`.
**Extends:** `ZRL_visualizer_build_spec.md` — its §0 (required behaviour), §4 (design system)
and §8 (author context) apply verbatim and are not repeated.
**Companion:** `ZRL_wave_findings_MIRROR.md` — the verified math this spec builds on
(ς Fork 1; i^n j; the ℤ₂/ℤ₄/U(1) ladder; conj = reciprocal; the Co ≤ 1/√D proof; the Huygens
deep dive; the polarisation ellipse; paraxial = Schrödinger). **Read it before §2 here.**
**Status:** v1 SHIPPED 2026-09-16 and browser-verified. This spec was written after the build
(the build ran from eight decisions given in conversation); it is now the contract for v2+.
**Date:** 2026-09-17

---

## 0. Instructions for the receiving agent

Inherit parent §0 (verify numerically before affirming; §3 here is a do-not-implement list;
say "this has no established name" when true; flag ambiguity rather than guessing;
works / LARP / psychosis). Three additions specific to this app:

1. **State which regime is live.** Whether the imaginary part is *forced* or *declared* depends
   on the equation (§2.1). An app that shows an imaginary part without saying which case it is
   in is making a claim it has not earned.
2. **A ledger row must never report a domain error as a physics failure.** Gate every
   moment-based measurement on containment. This cost a real debugging cycle — findings §9.2.
3. **Assert the wrap horizon** in any periodic-domain measurement, and subtract the source
   width from it. This cost three debugging cycles — findings §9.3.

---

## 1. What the program is

**A general 1D wave visualiser**, whose distinguishing feature is that it draws the *complex*
field: a curve in 3D whose axes are space, Re ψ and Im ψ. Intended uses, in priority order:

1. **Visualising real signals** — audio from `.wav`/`.mp3`, or from a synth (§9.1). This is the
   primary purpose. The app is an instrument for *seeing* a waveform's amplitude and phase
   structure, not a physics engine.
2. Teaching/【checking】the complex structure of wave equations (the i-decomposition).
3. Interference of a small number of coherent sources.

**Not** a light simulation (that is the separate laser project — §9.5), **not** a room-acoustics
model (that is the separate Echo app — and see the hard constraint in §9.6).

**Scope of "wave":** a disturbance travelling at speed **v ≤ v₀**, with **v = v₀ included** as a
first-class case, not an edge case (§2.5). v₀ is the lattice/medium speed limit.

**Medium:** one self-contained HTML file, Three.js 0.160 via CDN importmap, vanilla JS
otherwise. **File name:** `wave_idecomp_MIRROR.html`.

---

## 2. VERIFIED FORMULA REFERENCE

Every item was executed before being written. Sources: `ZRL_wave_findings_MIRROR.md` §1–8.

### 2.1 Is i forced or declared? — the regime rule

| equation | Re and Im | status of the imaginary part |
|---|---|---|
| **Schrödinger** i ψ_t = −½ ψ_xx | **coupled**: ∂_t Re = −½ ∂_xx Im (verified) | **forced by the equation** |
| **wave / EM** ψ_tt = c² ψ_xx | **decoupled**: each satisfies it alone (verified) | **a declared lift**; conj(ψ) has the same Re, so even the sign of Im is a convention |

Verified residuals: Schrödinger complex field 1.8e-5, Re alone **2.645** (fails); wave equation
Re alone passes. **The app must display which case is live.** v1 does this in the equation panel
and in the ledger.

### 2.2 The dimension wall

A complex field on a D-dimensional domain has a graph in **D + 2** dimensions.

```
D = 1  →  R³   fits exactly — this is WHY the helix works
D = 2  →  R⁴   must encode one axis
D = 3  →  R⁵   must encode two
```

Re and Im can be **drawn** only for a 1D domain. For 2D/3D domains the phase must be *encoded*
(domain colouring), never drawn. This is a theorem, and belongs on screen as a stated wall.

### 2.3 The field

Computed always in the `e^{i(kx−ωt)}` convention and **conjugated once at the end** if the other
convention is selected. This is what makes "flipping the convention conjugates the field" exactly
true (measured deviation 0.00e+0). Do not restructure this.

```
plane    ψ = A e^{i(kx + φ − ωt)}
packet   ψ = A (σ/√a) exp(−(x − x₀ − v_g t)² / 4a) · e^{i(k(x−x₀) + φ − ωt)}
             a = σ² + i t/2   (Schrödinger — complex ⇒ SPREADS)
             a = σ²           (EM vacuum   — real    ⇒ does NOT spread)
```

Dispersion and velocities:

```
Schrödinger   ω = k²/2  (ℏ = m = 1)     v_p = k/2    v_g = k       ⇒ v_g = 2 v_p
EM vacuum     ω = v|k|                  v_p = v·sgn k v_g = v·sgn k ⇒ v_g = v_p = v
```

Verified: σ(t) = σ₀√(1 + (t/2σ₀²)²) to 2e-3 in-app and 1e-15 offline; EM σ constant; both
velocity relations exact.

### 2.4 The bridges are Re and Im (findings §4)

```
|ψ| = 1  ⟹  conj(ψ) = 1/ψ        (verified 2.2e-16 in-app)
Re = (ψ + 1/ψ)/2 = x-bridge       Im = (ψ − 1/ψ)/2i = y-bridge
```

The convention toggle **is** the swap ψ ↔ 1/ψ. Looking down the propagation axis at a plane wave
reproduces `circle_flavor_MIRROR.html`'s canvas exactly — a real cross-app regression target.

### 2.5 Wave speed v, and the v = v₀ case

**v is a first-class control, not a constant.** Define v₀ = the medium/lattice speed limit and
**β := v/v₀ ∈ [0, 1]**.

- **β = v/v₀ is exactly the circle app's β.** The speed slider IS `circle_flavor`'s ψ slider:
  β = sin ψ, γ = sec ψ, and **v = v₀ is β = 1, ψ = 90°, γ → ∞ — the null case**, which canon
  already defines as the CFL-saturating path. The two apps share one parameter.
- Refractive index n = v₀/v = 1/β ≥ 1. Two regions with different v is refraction, and is
  Berglund's two-Courant-number mechanism.
- **Once FDTD lands, v ≤ v₀ becomes literally the CFL condition.** With v₀ = Δx/Δt, Co = v/v₀,
  so v ≤ v₀ ⟺ Co ≤ 1 — the D = 1 stability limit exactly. In D dimensions the bound tightens to
  **v/v₀ ≤ 1/√D** (findings §5). The author's design constraint and the stability condition are
  the same statement.

**Do not clamp v_phase to v₀.** Phase velocity may legitimately exceed the medium speed (it
carries no information); it is the group/signal velocity that is bounded. Expose v as the medium
speed, show v_p as a derived readout, and label it.

### 2.6 The polarisation ellipse (findings §7)

Two co-propagating real components at relative phase δ trace: **line at δ = 0 and 180°**
(linear — this is E-vs-B in vacuum, which is locked at δ = 0 forever, correlation +1.000000000000),
**circle at δ = 90°** (circular — the Re/Im helix, correlation 0.00e+0), **ellipse between**.
The helix is physical for EM as **circular polarisation** — the two transverse components of
**E alone**, never E-vs-B.

### 2.7 Stated facts carried in the ledger

- **Co ≤ 1/√D for every D** (proved, findings §5): 1.000000 / 0.707107 / 0.577350 / **0.500000** /
  0.447214 / 0.353553 at D = 1,2,3,4,5,8. No ceiling at D = 3. 1/√3 is also the cube's face
  signature ₂R/₀R — third appearance, recorded without a mechanism claim.
- **Huygens** (findings §6): 3D sharp, 2D and 4D leave a wake; measured 2D/3D ratio **177,586×**
  at ct = 40 on an identical physical problem, and the 2D wake is **converged to 4%** under a 2×
  grid refinement while the 3D residual **collapses** (median fine/coarse 0.960 vs 0.222).

---

## 3. DO NOT IMPLEMENT

| claim | why |
|---|---|
| an imaginary part for a classical wave presented as physically present | Re and Im are decoupled there; it is a declared lift (§2.1) |
| drawing Re and Im as geometry for a 2D or 3D domain | needs D+2 = 4 or 5 axes; encode phase instead (§2.2) |
| the Re-vs-Im helix labelled as E-vs-B | E and B are in phase (δ = 0, correlation +1); the helix is δ = 90° (§2.6) |
| a single time-delay to produce the quadrature partner of a broadband signal | works only monochromatically; use the Hilbert transform (§9.1) |
| clamping phase velocity to v₀ | v_p may exceed v₀ legitimately; only v_g is bounded (§2.5) |
| a non-cyclic phase colormap | phase wraps; a non-cyclic map draws a false seam (§4.8) |
| a "cyclic" map made by going out and back (blue→red→blue) | cyclic but not injective — two phases share a colour |
| GIF for the phase field | 256 colours destroys a smooth cyclic ramp (§9.3) |
| a 2D model of room reverberation in the Echo app | 2D has a physical wake that 3D does not — wrong impulse response (findings §6.6) |
| `k = ±1` as a phase label | ruled out in canon; use the i^n j orbit (findings §2) |
| resolving the ς exponent fork or the α-nesting convention silently | ς is Fork 1 (settled); α-nesting is still OPEN — ask |
| a fixed camera position | it crops; fit from the two half-FOVs (§7) |

---

## 4. Decisions — LOCKED 2026-09-16 (author's answers)

1. **Equations:** Schrödinger **and** EM vacuum. Others later.
2. **Domain:** 1D now. Higher D later, and then phase must be encoded (§2.2).
3. **Solver:** analytic (closed form) for v1; FDTD later — at which point the CFL row goes live.
4. **i-decomposition regime:** Schrödinger (where i is forced). Other modes added later.
5. **Time convention:** a **toggle**, default `−` i.e. `e^{i(kx−ωt)}`.
6. **Sources:** beam (Gaussian packet) is the headline; plane wave retained as the degenerate
   reference. Point source is *not* a separate 1D type — a 1D point source is two
   counter-propagating plane waves, which is what the `standing` preset builds.
7. **Time:** animated **and** a frozen-t slider. Both.
8. **Palette:** cyclic, luminance-corrected by default; raw hue retained so the banding it
   causes is visible; user-editable gradients are a v2 item (§9.3).

**Added 2026-09-17 (v2 decisions):**

9. **Wave speed v is a control** (§2.5). v₀ = 1 fixed; the slider is the **EM medium speed**.
   Schrödinger genuinely has no medium speed, so there β is taken from v_g = k and the slider
   is disabled with that stated — flagging rather than inventing a meaning. In audio mode
   there is no k at all and β reads "— (no k)".
   **γ is guarded:** β = 1 is the NULL case (γ → ∞) and β > 1 is outside the v ≤ v₀ constraint
   entirely, where 1 − β² < 0 and γ is **not a real number** — it must read "undefined", never
   a large number and never ∞. (In ℏ = m = 1 units any |k| > 1 lands there; that is a units
   artefact, not a superluminal claim, and the row says so.)
10. **δ is a DISPLAY transform, not a field transform.** `FR/FI` hold the true field; `DR/DI`
    hold what is drawn. Physics rows must read `FR/FI`; only render-fidelity rows read the GPU
    buffer. δ = 90° must reproduce the field exactly, and a ledger row asserts it.
11. **Export is WebM only.** MP4 needs a muxer; GIF is forbidden (§3). When no common period
    is found the hard cap applies — and the message says *no common period **found*** (the
    search is bounded: denominators ≤ 64, T ≤ cap), not "incommensurate", which would claim
    more than was tested.

---

## 5. Panels and controls

Parent §4 layout (280–300 px left panel, canvas flexes right, Courier numerics).
**v1 shipped:** equation toggle, convention toggle, five presets (plane / standing / beats /
packet / collision), three sources each with type / A / k / phase / x₀ / σ, play-pause + rate +
frozen t, x half-width, sample count, amplitude scale, six draw toggles, spoke spacing, palette
select, four camera snaps (3/4, Re, Im, complex-plane).

**Scene convention (fixed):** `x` = propagation, `y` = Re, `z` = Im. Re shadow on the floor
plane, Im shadow on the wall plane, phasor spokes drawing ψ(x) as a vector. Camera z-up, set
**before** any `lookAt` (three.js lesson from the 3D file).

**v2 additions** in §9.

---

## 6. Identity ledger — 19 rows live

Two classes, and the split matters:
- **Render-fidelity rows** read back the **GPU Float32 buffers** (the vertex buffer vs the
  Float64 field; the two shadow lines vs the helix).
- **Math rows** use the **Float64 array the draw was built from** — the 3D file's deviation-16
  split (ledger integrates Float64, GPU draws its Float32 copy, ≤ 6e-8 relative).

**Tolerances are stated in the step sizes, never as magic constants** (deviation-4 pattern). The
two PDE rows use O(h²) stencils, so their bound is `3·(½·(dx²/12)k⁴|ψ| + (ht²/6)ω³|ψ|)` and the
row prints the bound beside the residual.

**One row is red by design:** *"Re alone CANNOT satisfy it (it cannot)"* — the label carries the
"(it cannot)" marker so red reads as theorem, not bug (house pattern).

Verified live 2026-09-16, zero FAILs: phase slope 2.000000000 vs k; convention flip →
−2.000000000 with conjugation dev 0.00e+0; standing wave max|Im(ψ/ψ_ref)| = 0.00e+0 with signs
`{−1, +1}`; rhombus identity dev 4.4e-16; conj = 1/ψ dev 2.2e-16; packet at t = 6 σ 2.500000 vs
2.500000, centre 12.0000 vs 12.0000, norm 5.013257 vs 5.013257; EM packet σ unchanged 2.000000;
quadrature corr 0.0016 over 7λ.

---

## 7. Implementation notes

- **Camera fitting:** derive the distance from both half-FOVs
  (`dH = L/tan(hf)`, `dV = A/tan(vf)`, take the max × 1.12) against the *measured* peak |ψ|.
  A fixed position cropped the standing wave, whose peak is 2A.
- **Palette LUT:** 361-entry lightness table solved by bisection at load so relative luminance is
  constant (raw hue swings **4.5×**: 0.144 at h = 240° to 0.646 at h = 60°).
- **Ledger cadence:** recomputed on a 500 ms timer while playing so it never stalls the frame loop.
- **Containment gates** on every moment-based row (§0.2).

---

## 8. Cross-app checks

- `circle_flavor_MIRROR.html` — looking down +x at a plane wave must reproduce its canvas
  (constant-radius phasor fan, full hue wheel). The x/y-bridges must agree to 1e-12 (§2.4).
- `metrics_algebra_MIRROR.html` — shares the split-complex/Cl(1,1) material behind findings §2.
- `flavor_renderer_3d_MIRROR.html` — source of the Three.js scaffolding, the rotor block, the
  ledger harness and the Float64/Float32 split. **Its oracle now passes: 37/37 under
  `python -m pytest` (2026-09-17).**

---

## 9. Roadmap — the author's priority order

> **STATUS 2026-09-17: items 9.1, 9.2, 9.3 and 9.4 are BUILT and pane-verified (v2).**
> The descriptions below stand as the contract; what shipped matches them with these
> decisions taken: audio mapping = **(a) analytic signal** (the author's choice); export =
> **WebM only**, with a **hard cap** (default 180 s, adjustable 5–300) as the fallback when no
> common period exists; gradients ship with **forced cyclicity + a live luminance ratio**.
> Two features were added beyond the list, both to answer "show the 90° shift visually":
> a **δ relative-phase slider** (the polarisation ellipse — §2.6) and a **quarter-wave ghost**
> (Re(x − λ/4) drawn on the Im plane, where it must coincide with the Im shadow).
> Remaining from the roadmap: FDTD (9.x), MP4, 2D domains, audio playback sync.

### 9.1 `.wav` / `.mp3` input — **BUILT (v2)**

**How the formats store audio.**
- **WAV** is *uncompressed PCM*: a RIFF header then raw samples — literally amplitude versus
  time, typically 16-bit signed integers at 44.1 kHz. It **is** a waveform on disk.
- **MP3** is *lossy and frequency-domain*: audio is cut into frames, each transformed by an MDCT,
  and the coefficients quantised according to a psychoacoustic model that discards what the ear
  will not miss. There is **no waveform in the file** — it must be decoded back to samples.

**Practical consequence: one API handles both.** `AudioContext.decodeAudioData(arrayBuffer)`
returns Float32 PCM for WAV *and* MP3 (and FLAC/OGG/M4A), in every major browser, with no
library and no CDN. Roughly 15 lines. The format question, for implementation purposes,
disappears.

**The real design question is the mapping**, because audio is a function of *time* and this app
draws a function of *space*. Three honest options:

| mapping | meaning | verdict |
|---|---|---|
| **(a) analytic signal** — Hilbert transform gives Im from the real samples | helix radius = **instantaneous envelope**, winding rate = **instantaneous frequency** | **the right default** — it *is* the i-decomposition, and it makes the helix directly meaningful for audio |
| (b) buffer as a spatial snapshot | exact for non-dispersive travel: ψ(x,t) = f(x − vt), so a recording at one point is the spatial profile reversed and scaled by v | valid for EM/sound, **invalid under Schrödinger** (dispersive) — gate it |
| (c) FFT into k-modes, feed the existing multi-source machinery | generalises the source list | good later; heaviest |

**Effort:** decode ~15 lines; radix-2 FFT for the Hilbert transform ~60 lines (no dependency);
windowing/decimation for display ~40; transport controls ~40. **~200 lines, one session.** The
Hilbert transform is already verified exact for a pure tone (5e-15) and for a two-tone sum
(3e-14) where a single time-delay fails (dev 2.0) — findings via the quadrature work.

### 9.2 GIF / MP4 export — **PRIORITY 2**

- **WebM is free:** `canvas.captureStream()` + `MediaRecorder` — no library, ~30 lines,
  native in Chrome/Edge/Firefox.
- **MP4** specifically needs a muxer (`mp4-muxer` via CDN) or ffmpeg.wasm. Safari's
  MediaRecorder emits MP4 natively; Chrome historically does not.
- **GIF is a poor fit here and is in §3:** 256 colours will destroy a smooth cyclic phase ramp.
  If a GIF is genuinely required, quantise deliberately and say so.
- **Recommended:** WebM via MediaRecorder, plus a **PNG frame-sequence export** (a robust
  fallback that converts to anything with ffmpeg and never loses colour).
- **"One t-cycle"** needs care: the period is 2π/ω for a single source, but **several sources
  with incommensurate ω have no common period** — detect it and fall back to a user-set duration.
  For an audio-driven render the duration is the file's.
- **Effort:** WebM ~30 lines; PNG sequence ~40; period detection ~30. Half a session.

### 9.3 Editable gradient colours — **PRIORITY 3**

Straightforward (~120 lines: stop list, colour inputs, drag to reposition, live strip) with
**two non-negotiable instruments**, because a hand-made gradient will usually break both
properties the current palette guarantees:
1. **Force cyclicity by construction** — the last stop wraps to the first; never let the user
   create a seam by accident.
2. **Show a live luminance readout** (max/min ratio) so the banding being introduced is visible.
   The built-in corrected ramp reads 1.0000×; raw hue reads 4.49×.

Ship the two verified ramps as presets so there is always a known-good baseline.

### 9.4 Wave speed v as a control — cheap, high value

~5 lines for the parameter plus a readout; the payoff is §2.5 (β = v/v₀ shared with the circle
app; v = v₀ as the null case). Do this alongside 9.1 — it costs almost nothing and it connects
two apps.

### 9.5 Laser / lens project — separate app, **same solver**

Beam divergence through lenses at range. **The paraxial wave equation IS Schrödinger with z in
place of t** (findings §8, verified to 1.6e-16), so w(z) = w₀√(1+(z/z_R)²) and
σ(t) = σ₀√(1+(t/2σ₀²)²) are one law. Reuse the packet code unchanged. New material needed:
ABCD ray matrices for lens trains (q' = (Aq+B)/(Cq+D)), the M² beam-quality factor, and
aberration — spherical aberration is the first genuinely non-paraxial effect and marks where
this equation stops being sufficient. **Verdict: an excellent concrete light project, and
cheaper than it looks because the hard part already exists.**

### 9.6 Echo app — one hard constraint up front

**Build it in 3D, or with an image-source/ray method that assumes sharp arrivals.** A 2D room
model is *not* a cheap approximation of a 3D one: in 2D every reflection leaves a physical wake
(findings §6), so the impulse response is qualitatively wrong — a hand clap returns smeared
instead of sharp. Measured 2D/3D ratio 177,586× at ct = 40, with the 2D wake converged under
grid refinement and the 3D residual collapsing.

---

## 10. Open items — do not resolve silently

- **α-nesting convention** ("one nesting level = one δθ advance") — still OPEN; blocks any
  per-tick-spectrum tab here, as it already blocks circle spec Stage C.
- Whether the audio mapping default should be (a) or (b) in §9.1 — recommended (a); author's call.
- MP4 vs WebM (§9.2) — recommended WebM + PNG sequence.
- Whether to expose the polarisation-ellipse δ slider (§2.6) — free, not yet requested.
- Higher-D domains and the phase-encoding scheme they force (§2.2).

---

## 11. Verification log

| check | result |
|---|---|
| Schrödinger coupled / wave equation decoupled | PASS (Re-only residual 2.645 vs 1.8e-5) |
| helix projections exactly cos/sin; \|ψ\| const; pitch = λ | PASS (ptp 4e-16) |
| traveling phase slope 1.000000000; standing phase ∈ {0, π} | PASS |
| convention flip = conjugation; conj = 1/ψ on \|ψ\|=1 | PASS (0.00e+0; 2.2e-16) |
| x/y-bridge = Re/Im | PASS (1.1e-16) |
| two-source = 2cos(δ/2)e^{iδ/2} | PASS (4.4e-16) |
| packet σ(t) Schrödinger spreads / EM does not | PASS (2.500000 vs 2.500000; 2.000000) |
| v_g = 2v_p (Schrödinger); v_g = v_p (EM) | PASS |
| norm conservation | PASS (5.013257 vs 5.013257) |
| Co ≤ 1/√D at D = 1…8 + independent 20⁴ FDTD at D = 4 | PASS (stable 0.49, unstable 0.51) |
| Huygens 2D/3D/4D + resolution-scaling discriminator | PASS (177,586×; 0.960 vs 0.222) |
| polarisation ellipse at δ = 0/45/90/135/180 | PASS |
| paraxial = Schrödinger, Gaussian beam law | PASS (1.6e-16) |
| Hilbert quadrature exact for a tone, delay fails broadband | PASS (5e-15; dev 2.0) |
| `flavor3d` oracle under `python -m pytest` | **37/37 PASS** (2026-09-17) |

---

## 12. Attribution

- **Hadamard**, *Lectures on Cauchy's Problem* (1923) — method of descent, Huygens' principle in
  odd dimensions. STANDARD.
- **Courant, Friedrichs & Lewy** (Math. Ann. 100, 32, 1928); **von Neumann** stability analysis —
  the CFL condition. STANDARD. *The general-D form Co ≤ 1/√D is textbook-derivable but was not
  found stated for arbitrary D in the sources searched; the proof in findings §5 is written out
  because the author had been unable to locate one.*
- **Gabor**, J. IEE 93, 429 (1946) — the analytic signal. STANDARD.
- **Jones** (1941) / **Stokes** (1852) — polarisation ellipse. STANDARD.
- **Siegman**, *Lasers* (1986) — Gaussian beams, ABCD matrices, M². STANDARD.
- **Cl(1,1) / split-quaternions**: Sobczyk, College Math. J. 26, 268 (1995). STANDARD.
- Original to this project as far as searched: the i-decomposition helix as a *flavor* object
  with the bridges as its shadows; the ℤ₂ ⊂ ℤ₄ ⊂ U(1) phase ladder tied to the flavor index;
  the i^n j orbit as the replacement for `k = ±1`.
