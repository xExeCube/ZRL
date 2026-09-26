# Zero-Recursive Lattice — Circle-Mapped Flavor Renderer Spec (the rapidity circle)

**For:** a coding agent (Claude Code).
**Extends:** `ZRL_visualizer_build_spec.md` — §0, §4, §8 apply verbatim. Sits beside
`metrics_algebra_MIRROR.html` (whose circle and hyperbola panels this app slaves
together) and `lattice_pathfinder_MIRROR.html` (whose quarter-circle check it reuses).
**Supersedes:** nothing — new app. It consolidates the 2026-09-13/15 results: Bondi's
k, the γ/β dictionary, and the R-to-cos / R-to-sin bridges.
**Status:** every formula in §2 was numerically verified on 2026-09-15 (log in §9).
There are no open definitions. The one ambiguity in the request ("circle-mapped") is
resolved as: *one point on the unit circle carries (1/γ, β), and every canon object
is placed on that circle or on its hyperbola twin.* If a vertices-as-roots-of-unity
app was meant instead, that is a different, smaller file.
**Declared geometry, not physics:** β, γ, f, ρ/r_s are *labels* for sin, sec, cos²
and 1/sin² of one angle. Where a lapse or shell radius is named it is bookkeeping;
reading "the hexagon lives at 4r_s" as physics is LARP by the canon's own ruling.
**Date:** 2026-09-15

---

## 0. Instructions for the receiving agent

Inherit parent §0. Two additions:

1. **β and γ are subscripted everywhere in the readout** (β_esc, γ_esc; β_D, γ_D).
   The unsubscripted conflation is in the canon's ruled-out table (§3).
2. This app *draws identities*. Every identity drawn is checked from the drawn
   coordinates (the pixel-space points, not the input slider), so the ledger tests
   the renderer, not the formula.

---

## 1. What the program is

One slider, ψ ∈ [0°, 90°). **Left canvas:** the Q1 quarter of the unit circle with
the point **P = (cos ψ, sin ψ) = (1/γ, β)**. **Right canvas:** the hyperbola
x² − y² = 1 with the point **H = (cosh φ, sinh φ) = (γ, γβ)**, φ = gd⁻¹(ψ), the two
panels locked by the Gudermannian. Overlays: the whiteboard half-angle triangle at
θ = π/4 + ψ/2 with k = tan θ; the bridge parallelogram of R = e^{iθ} and 1/R (their
sum 2cos θ on the real axis, their difference 2i sin θ on the imaginary axis); fixed
markers for the four flavors at ψ_n = π/n, the silver point, and the Pythagorean
snaps. A readout of every quantity in §2, then the §5 ledger.

**Medium:** one self-contained HTML file, vanilla JS + Canvas 2D, no framework.
**File name:** `circle_flavor_MIRROR.html`. About 500 lines; most of it is checks.

---

## 2. VERIFIED FORMULA REFERENCE

### 2.1 The dictionary — one angle, three faces

| circle (ψ) | hyperbola (φ) | Bondi (k) |
|---|---|---|
| cos ψ = 1/γ | cosh φ = γ | (k + 1/k)/2 = γ |
| sin ψ = β | sinh φ = γβ | (k − 1/k)/2 = γβ |
| tan ψ = γβ | tanh φ = β | k = e^φ = γ(1 + β) |

```
ψ = gd(φ)   = 2·arctan(e^φ) − π/2
φ = gd⁻¹(ψ) = ln tan(π/4 + ψ/2) = artanh(sin ψ) = arsinh(tan ψ)
k = √((1+β)/(1−β)) = tan(π/4 + ψ/2)
θ := π/4 + ψ/2   (the whiteboard half-angle),   k = tan θ,   (cos 2θ, sin 2θ) = (−β, 1/γ)
```
The √ in Bondi's k is "√ halves the angle": k² = (1+β)/(1−β) = (1 + sin ψ)/(1 − sin ψ)
= tan²θ lives at 2θ, k at θ.
Check at β = 3/5: ψ = 36.870°, φ = ln 2 = 0.693147, θ = 63.435°, k = 2, γ = 5/4.

### 2.2 The whiteboard triangle

```
(a, b) = R(cos θ, sin θ),  κ := tan θ = b/a = k
ab = R²/(2γ)                       ⇒   γ = R²/(2ab)
𝒮_Δ = Δ₁ + Δ₂ = πR²/4 − ab = R²(π/4 − 1/(2γ))
```
γ = 1 on the diagonal (θ = 45°, ψ = 0 — the maximal inscribed rectangle); γ → ∞ as
the rectangle collapses onto an axis (the null directions). Degree-2 homogeneity:
𝒮_Δ(2R) = 4·𝒮_Δ(R). (Parent §2.5's Σ with R⁸ is ruled out; this is the replacement.)

### 2.3 The bridges on the circle (the R-to-cos and R-to-sin bridges)

```
R = e^{iθ}:   (R + 1/R)/2 = cos θ        (R − 1/R)/(2i) = sin θ          (Euler; Joukowski z + 1/z)
V₁² = R² + R⁻² = 2 cos 2θ     V₂² = R² − R⁻² = 2i sin 2θ     V₁⁴ − V₂⁴ = 4cos²2θ + 4sin²2θ = 4
R = e^{w/2} real:  V₁² = 2 cosh w,  V₂² = 2 sinh w,  w = 2 ln R  (double rapidity, k_bridge = R²)
```
The circle and real-line readings are the same invariant 4 — cos² + sin² = 1 versus
cosh² − sinh² = 1 — mapped into each other by gd. On the circle the y-bridge is
purely imaginary.
Nulls: V₂ = 0 at R⁴ = 1 (C₄: θ = 0°, 90°, 180°, 270° — the quadrant map); V₁ = 0 at
R⁴ = −1 (θ = 45°, 135°, 225°, 315°); together R⁸ = 1, hence V₁V₂ = √(R⁸ − 1)/R².
Chebyshev: Rⁿ + R⁻ⁿ = 2·T_n(cos θ) — the bridge turns the power ladder into the
three-term recurrence T_{n+1} = 2x·T_n − T_{n−1}.

### 2.4 Flavor points and shells

| n | ψ_n | cos ψ_n = signature = x-bridge at √ζ_n | sin ψ_n | f_n = cos²ψ_n | ρ_n/r_s = 1/sin²ψ_n |
|---|---|---|---|---|---|
| 3 | 60° | 0.500000 | 0.866025 | 0.250000 | 4/3 |
| 4 | 45° | 0.707107 | 0.707107 | 0.500000 | 2 |
| 5 | 36° | 0.809017 = φ_golden/2 | 0.587785 | 0.654508 = φ_golden²/4 | 2 + 2/√5 = 2.894427 |
| 6 | 30° | 0.866025 | 0.500000 | 0.750000 | 4 |
| ∞ | 0° | 1 | 0 | 1 | ∞ |

The signature is the x-bridge ½(R + 1/R) evaluated at R = e^{iπ/n} = √ζ_n, the square
root of the flavor generator ζ_n = e^{2πi/n} — verified for n = 3…6. The shell
radii are bookkeeping labels (canon: "flavor shells — TRUE, exactly as deep as
sin² + cos² = 1"). No flavor point is rational (Niven), so none is a Pythagorean snap.

### 2.5 Lapse dictionary (labels only)

```
1/γ_esc = √f = cos ψ      β_esc = sin ψ      f = cos²ψ      γ_esc = sec ψ      γ_esc² = 1/f
β_D = (1−f)/(1+f) = sin²ψ / (1 + cos²ψ)      γ_D = (1+f)/(2√f) = (1 + cos²ψ)/(2 cos ψ)
k(β_D) = γ_esc   exactly
```
Check ψ = 30°: β_D = 1/7 = 0.142857, k(β_D) = 1.154701 = γ_esc, γ_D = 1.010363 ≠ γ_esc.
Both pairs are shown so the conflation cannot come back.

### 2.6 Rational boosts — the Pythagorean snaps

(m, n) ↦ β = (m²−n²)/(m²+n²), 1/γ = 2mn/(m²+n²), k = m/n. The circle point is
(m + ni)²/|m + ni|² up to the axis swap: **squaring the Gaussian integer doubles the
angle** — the square flavor's own arithmetic. Integer identity exact (BigInt check):

| (m,n) | β | 1/γ | k = tan(π/4 + ψ/2) |
|---|---|---|---|
| (2,1) | 3/5 | 4/5 | 2 |
| (3,1) | 4/5 | 3/5 | 3 |
| (3,2) | 5/13 | 12/13 | 3/2 |
| (4,1) | 15/17 | 8/17 | 4 |
| (4,3) | 7/25 | 24/25 | 4/3 |
| (5,2) | 21/29 | 20/29 | 5/2 |
| (5,4) | 9/41 | 40/41 | 5/4 |
| (6,1) | 35/37 | 12/37 | 6 |
| (7,4) | 33/65 | 56/65 | 7/4 |

### 2.7 The silver point

φ = arsinh 1 = ln(1 + √2) = 0.881374; β = 1/√2; γ = √2; γβ = 1; ψ = 45°; θ = 67.5°;
k = 1 + √2 = 2.414214; gd(φ) = π/4. It coincides with the square's flavor point
(ψ₄ = 45°) — the 45° cone appearing twice, not a derivation (canon).

### 2.8 Optional tab — Jacobi–Anger (the per-tick-turn spectrum)

```
e^{ix sin θ} = Σ_n J_n(x) e^{inθ}        e^{x cos θ} = Σ_n I_n(x) e^{inθ}
```
At the flavor's per-tick turn θ = 2π/n the coefficients J_n(x) form the harmonic
spectrum; the flavor index and the Bessel order are the same integer (canon). Needs a
J_n implementation (Miller backward recurrence or the power series; |n| ≤ 30 suffices
for x ≤ 5; J_{−n} = (−1)ⁿ J_n). Verified: x = 1.7, θ = 0.6, partial sum matches
e^{ix sin θ} to 1e-15.

---

## 3. DO NOT IMPLEMENT — ruled out

| claim | why |
|---|---|
| √f = k (Bondi) as a general identity | √f = 1/γ(β_esc); k only in Rindler / near-horizon (ruled-out table) |
| f = k² as a radar factor | ruled out |
| a single unsubscripted β or γ | β_esc ≠ β_D, γ_esc ≠ γ_D (§2.5) |
| the additive lapse (√f − 1) anywhere | pole at f = 1; standing rule: any denominator vanishing at f = 1 is the additive lapse in disguise |
| α = sin θ as a Bessel order (§2.8 tab) | ruled out; on a Zₙ cone the order is n·m |
| flavor shells as physics | bookkeeping only |
| "60° Wick rotation" | Wick is 90°; here the circle↔hyperbola map is gd, not a rotation |
| parent §2.5 delta-sum with R⁸ | ruled out; §2.2 is the replacement |
| the app's split-quaternion generator named k | k is Bondi's here; the unipotent is j (2026-09-13 mirror) |

---

## 4. Panels and controls

**Left panel** (parent §4 layout):
- ψ slider 0–89.9°, default 30° (the hexagon); input modes ψ / β / φ / k, each
  recomputing the others through §2.1
- snap buttons: △ □ ⬠ ⬡ (ψ_n), silver, and a Pythagorean list (m ≤ 7, §2.6)
- toggles: whiteboard triangle, bridge parallelogram, C₄ null points, hyperbola
  panel, Jacobi–Anger tab
- R slider (scale only; drives the homogeneity check)

**Left canvas — the circle:** Q1 unit circle; P and its radius vector; the (1/γ, β)
right triangle (legs drawn, hypotenuse 1); the half-angle point Q = (cos θ, sin θ)
with the whiteboard rectangle a × b shaded; R and 1/R as vectors with their sum on
the real axis and difference on the imaginary axis; flavor markers with n labels;
the silver marker; the C₄ images of P when toggled.

**Right canvas — the hyperbola:** branch x² − y² = 1, asymptotes drawn dashed (the
null cone — same convention as `metrics_algebra`), H = (γ, γβ), the connector
labelled "gd"; scale locked to the circle panel so γ reads as the same length.

**Readout** (monospace, right-floated values in `--accent`): ψ, φ, β_esc, γ_esc, γβ,
k, 1/k, θ, f, ρ/r_s, β_D, γ_D, V₁², V₂² (complex), the J_n spectrum when the tab is
on — then §5.

---

## 5. Identity ledger — live PASS/FAIL

| check (from drawn coordinates) | expected | tol |
|---|---|---|
| β² + 1/γ² | 1 | 1e-12 |
| γ² − (γβ)² | 1 | 1e-12 |
| k · (1/k) | 1 | 1e-12 |
| √((1+β)/(1−β)) − tan(π/4 + ψ/2) | 0 | 1e-12 |
| gd(gd⁻¹(ψ)) − ψ | 0 | 1e-12 |
| γ − R²/(2ab) | 0 | 1e-12 |
| 𝒮_Δ(2R) / 𝒮_Δ(R) | 4 | 1e-12 |
| V₁⁴ − V₂⁴ at R = e^{iθ} | 4 | 1e-12 |
| ½(√ζ_n + 1/√ζ_n) − cos(π/n), n = 3…6 | 0 | 1e-12 |
| ρ_n/r_s − 1/sin²(π/n), n = 3…6 | 0 | 1e-12 |
| k(β_D) − γ_esc | 0 | 1e-12 |
| (m²−n²)² + (2mn)² − (m²+n²)² (BigInt) | 0 | exact |
| gd(ln(1+√2)) − π/4 | 0 | 1e-12 |
| Σ_{\|n\|≤30} J_n(x)e^{inθ} − e^{ix sin θ} *(tab)* | 0 | 1e-10 |

Green = `--hi`, red = `--link`.

---

## 6. Build plan

- **Stage A (one session, ~500 lines):** both canvases, all controls except the
  Bessel tab, full §5 ledger.
- **Stage B (optional):** Jacobi–Anger tab (§2.8).
- **Stage C — BLOCKED:** a "per-tick-turn spectrum" as a *flavor property* needs the
  α-nesting convention ("one nesting level = one δθ advance"), which the canon lists
  as OPEN. Do not build until it is declared.

---

## 7. Cross-app checks

- `metrics_algebra_MIRROR.html`: its e^{iθ} circle and e^{jφ} hyperbola panels are
  this app's two canvases; the same φ must land on the same H. Its generator is
  renamed j; k is reserved for Bondi.
- `lattice_pathfinder_MIRROR.html`: the quarter-arc check πR/2 is reused for the Q1
  arc.
- `flavor_renderer_MIRROR.html`: the four markers use its signature table (parent
  §2.1); values must agree to 1e-12.

---

## 8. Open items

- Full circle with the C₄ images versus Q1 only — default Q1 (canon: space lives in
  Q1); the toggle draws the images of P under {1, i, −1, −i}.
- Whether θ (half-angle) should own the slider instead of ψ — cosmetic; ψ chosen
  because P is what the flavors mark.
- Stage C (above).

---

## 9. Verification log (2026-09-15, Python / numpy / scipy)

| check | result |
|---|---|
| dictionary at β = 3/5: ψ, φ, θ, k = 2, γ = 5/4 | PASS |
| flavor points n = 3…6: cos, sin, f, ρ/r_s, and ½(√ζ_n + 1/√ζ_n) = cos(π/n) | PASS |
| silver point: φ = 0.881374, β = 0.707107, γ = 1.414214, γβ = 1, ψ = 45°, θ = 67.5°, k = 2.414214 | PASS |
| β_D, γ_D at ψ = 30°: 1/7, 1.010363; k(β_D) = γ_esc = 1.154701 | PASS |
| Pythagorean table (m,n) ≤ (7,4): integer identity and k = tan(π/4 + ψ/2) | PASS |
| V₁⁴ − V₂⁴ = 4 at R = 0.9, 1.2, 2, 5 and on the circle | PASS |
| bridge nulls: R⁴ − 1 → C₄; R⁴ + 1 → 45° + 90°k | PASS |
| Jacobi–Anger at x = 1.7, θ = 0.6 (J and I versions) | PASS (1e-15) |

---

## 10. Attribution (audited 2026-09-15)

- Gudermannian: Lambert (1760s), Gudermann (1830) — STANDARD.
- Bondi's k: H. Bondi, *Relativity and Common Sense* (1964) — STANDARD; the
  half-angle form k = tan(π/4 + ψ/2) is a classical identity, KNOWN BUT SCATTERED
  (expository notes, not a headline in the texts).
- River / lapse reading √f = 1/γ(β_esc): Hamilton & Lisle, Am. J. Phys. 76, 519
  (2008) — STANDARD.
- Cl(1,1) idempotents and e^{jφ}: Sobczyk, College Math. J. 26, 268 (1995) — STANDARD.
- Joukowski map z + 1/z; Chebyshev T_n (Abramowitz & Stegun ch. 22; Mason &
  Handscomb 2003) — STANDARD.
- Jacobi–Anger: DLMF §10.12.2–10.12.3, §10.35.1–10.35.2 — STANDARD.
- Rational points on the circle / Pythagorean triples (Euclid, Diophantus); Niven's
  theorem (Niven 1956) — STANDARD.
- Original to this project as far as searched: the single combined circle↔hyperbola
  gd diagram with the half-angle k; γ = R²/(2ab) as the whiteboard reading of the
  Lorentz factor; the flavor points as bookkeeping shells.
