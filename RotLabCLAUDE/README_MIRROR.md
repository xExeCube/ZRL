# RotLab (LatMosaic sub-project, ZRL) — rotation & dimension-borrowing apps

*Written by Claude Fable 5.1, 02/09/2026.*

Three single-file apps exploring one idea at three dimensions: **dimension n's
unreachable orientation class is dimension n+1's rotation** (the V₄ element you
cannot reach becomes a path; the projection quotients out the borrowed axis).

## card_flip_MIRROR.html — image ↔ mirror through the 3rd dimension

2D canvas, no libraries. Load any image (or the chiral demo pattern). A mirror
flip (det −1 in the plane) is realized as R(180°) about an in-plane axis;
orthographic projection collapses the whole rotation to u ↦ u·cos θ. Variants:
**mirror in place** (axis through the centre, same frame) and **hinge on edge**
(page-turn; frame = 2× the image on that axis — the user's variant-2 frame
arithmetic). Perspective mode renders the keystone via vertical slicing.
Oscillate play, θ scrub, PNG frame export. A frozen mid-flip frame is
depth-sign ambiguous — a controlled spinning dancer.

## tribar_spin_MIRROR.html — the always-impossible rotation

The one rigid motion an impossible figure survives: rotation about the view
axis. Embeds a minimal honest-occlusion cube-beam core (tribar / hex ring /
double hex presets from hex_beam). **Rigid** mode spins the shaded image;
**relight every 60°** re-derives the face-family shading at each lattice
coincidence (the three rhombus-orientation families permute cyclically per
60°), so the light stays fixed in the world and the figure reads as a rotating
object that is impossible in all six rest orientations. Per Penrose's cocycle
argument the implied 3D object cannot stay rigid across frames — here it never
existed: the paradox lives in the rendering.

## isomer4d_MIRROR.html — chiral 3D object → its enantiomer through ℤ⁴

three.js (module importmap, three@0.160.0, same CDN pattern as
cut_and_project_threejs; annotated for learning). Chiral shapes — screw
tetracube, 6-cube spiral staircase, the **open tribar** (its two windings are
4D-rotations of each other), torus knot. 4D rotation in the (axis, w) plane
projects to scale[axis] = cos θ: the object squashes, passes through a flat
state at 90° (inside the 4th dimension), and re-inflates as its mirror isomer.
Ghost wireframes show the θ=0 and θ=180 states; orbit the camera during the
flip. Möbius 1827; the card flip one dimension up.

## Review 02/09 (14 agents, 9 confirmed, 0 refuted — all fixed)

card_flip: frame now clips the perspective near-edge (the frame is the
projection window); strips overdraw 1px toward their successor (hairline seams
killed); stop cancels the pending animation frame. tribar_spin: the reviewers
CONFIRMED the embedded core is byte-faithful to hex_beam (1929 faces × 6 points,
0 mismatches; preset winners identical) and the relight permutation direction is
correct by shape congruence; wording updated to "relight (nearest 60°)" to match
the round() behavior (the switch lands midway, where the figure is maximally
off-lattice — deliberately kept). isomer4d: rebuild() disposes
geometries/materials (torus-knot leak was MB-scale per click); dead exact-flat
guard removed (cos(90°) ≈ 6e-17 renders fine); the CDN warning retracts on a
slow-but-successful load and names the addon; resize() re-reads devicePixelRatio
(browser zoom / monitor DPR). hex_beam hextile: rhombohedron family is coerced
to cubes (its layout constants come from the cube walk's transposed footprint —
a rhomb tiling needs its own measured pitches, backlog); preset filenames now
carry _L<arm> and _n<tilen>; comments corrected (real pitches 2L+5/2L+6, and
flipped rings sit an irremovable cosmetic 0.5 lattice units off-column).

## Silhouette mode + GIF export (added 03/09/2026)

card_flip gained **silhouette mode** (the spinning-dancer condition): the image
renders as a flat single-color cutout — a transparent PNG cuts by alpha, an
opaque image by luminance threshold. A cue-free flat shape mid-flip is exactly
depth-sign ambiguous: cos is even, so the θ(t) and −θ(t) silhouette sequences
are IDENTICAL — asymmetric features (e.g. a side ponytail) cannot break the
ambiguity, they only make the two readings (object spinning CW vs its mirror
spinning CCW) more vividly different objects. Both card_flip and tribar_spin
gained **gif loop** export: a self-contained GIF89a encoder (exact palette when
≤255 colors — always true for the flat beam renders — else a 6×7×6 quantization
cube; LZW with variable code width and 4096 reset; Netscape loop-forever;
optional transparency via disposal-2 frames). card_flip's GIF sweeps θ through
a full 360° so the loop closes seamlessly; the transparent-gif option captures
the card alone (no frame/axis decorations) for overlay use. Verified 03/09:
byte-level structure (header/trailer/frame count), browser round-trip decode,
pixel-exact 2-color round trip, 40 KB per 12-frame 320px capture.

Why the user's mirror-flip observation holds (impossible figures staying
impossible through the card flip): the flip is a path from a figure to its
MIRROR, and both windings of an impossible figure are legitimate impossible
figures; every intermediate frame is an anamorphically squashed one. The
card-flip of an impossible figure is the 2D shadow of what isomer4d does to the
open tribar in 4D — the same chirality-connecting path, one dimension down.

GIF/silhouette review 03/09 (10 agents, 7 confirmed, 1 refuted — all fixed):
the encoder itself was VERIFIED CORRECT by an independently written from-spec
GIF decoder — 610 directed assertions + 300 fuzz round-trips, zero byte or
pixel mismatches, including 22-reset LZW streams, the quantized path, the
transparency slot, and 1×1/edge dims. Fixed: capture-while-playing now resumes
the animation (rAF re-armed in a finally, which also re-enables the button and
restores θ/ρ on error); a CAPTURING lock ignores control input mid-capture
(frames can no longer be corrupted by clicks); exact-palette cap is 256 when no
transparency slot is needed (256-color captures now lossless, verified);
silhouette alpha-mode detection requires >0.5% of pixels below the solidity
cut (a stray watermark can't blank the silhouette); tribar_spin got the missing
.row style and a comment-stripped-copy note on its encoder.

## card_spin3d_MIRROR.html — the three.js port, SHIPPED 04/09/2026

card_flip promoted to three.js (annotated; same importmap/CDN as
cut_and_project_threejs), with the four requested upgrades:
- **axis tilt** — axis A at any in-plane angle (slider −90°..90°);
- **movable axis** — position slider scaled by the card rect's support function
  so ±1 is exactly hinge-on-edge (presets: center/left/right/top/bottom);
- **dual-axis spin** — optional axis B (own tilt/position), each axis with its
  own signed velocity; both axes world-fixed, composed as conjugated rotation
  matrices (M = T(p_B)R_B T(−p_B) · T(p_A)R_A T(−p_A)): a tumble. Rational
  speed ratios are periodic — the gif exporter computes the least common period
  360/gcd(|ω_A|,|ω_B|) and closes the loop exactly (verified: ω=(60,90) → 12.0 s,
  144 frames); incommensurate ratios are quasi-periodic and never repeat (the
  temporal cousin of the quasilattice — noted in-app);
- **visible spin axes** — colored lines + pivot dots (A teal, B violet) rendered
  in the VIEW only; every export hides them.
The card is a double-sided unlit plane: the back face IS the mirrored image
(three.js provides the flip's endpoint physically), no lighting so the front
view stays cue-free (drag to orbit = reveal; front-view button = ambiguity
back). Silhouette mode and the proven GIF encoder carried over. Verified: exact
hinge math (180° about the left edge pins that edge, far edge lands at −3w/2),
finite poses across the parameter space, axes excluded from exports and
restored after.

card_spin3d review 04/09 (9 agents, 7 confirmed, 0 refuted — all fixed): the
top/bottom presets were SWAPPED (sign error in the perp convention; left/right
were correct — verified corrected: top preset pivot at +y edge); axis geometry
is now SNAPSHOTTED at gif-capture start (axisSpec read the DOM live, so a
mid-capture slider move silently changed the rotation axis for later frames —
verified: a mid-capture tilt attack no longer affects the export); the camera
is frozen during capture (controls disabled, rAF loop yields the renderer to
the exporter — kills both damping drift at the loop seam and preview flicker);
window resizes mid-capture are deferred to capture end; ortho↔perspective
toggle now matches apparent size across the swap (persp distance set to
VIEWH/tan(fov/2) ≈ 13.74; verified equal half-heights both ways); the gif
status reports the ENCODED duration (frames × delay) rather than the
mathematical period (centisecond rounding differs by a few percent); front-view
button uses the projection-correct distance and is capture-guarded.

## card_spin3d: 3D beam subjects + axis C, SHIPPED 07/09/2026

"What if we spin a 3D object rather than its image": the tribar / hex ring /
double hex are now REAL geometry in card_spin3d — hex_beam's exact cell walks
(verified: zero cell collisions L=3..9, every beam a contiguous collinear run,
so one stretched box per beam is exact), pre-rotated so the magic (1,1,1)
direction faces the camera. Front view IS the illusion (the tribar gap
(L−1)(1,1,1) projects to exactly (0,0) — column overlap physically realized);
a 180° vertical spin is the animation between the two magic viewpoints:
illusion → gap reveal → mirror-winding illusion (−(1,1,1) shows the
enantiomer — the isomer4d connection). The "object or camera?" question
dissolves: only relative orientation reaches the projection, and rotating the
object keeps all axis/gif machinery unchanged. The hexring's 90° side view
shows the truth: a crinkled non-planar walk — the hexagon exists only in
projection. Beams are Lambert-lit (one key light + load-bearing 0.45 ambient:
the third face family is edge-on to the key; levels 1.41/1.73/0.45) — a real
object relights as it turns, which tribar_spin could only fake. Plus AXIS C:
three simultaneous world-fixed axes, M = M_C·M_B·M_A, gif period = 360/gcd of
all enabled speeds. Silhouette mode flattens beams to one unlit color — a
volume whose outline genuinely changes, yet direction stays undecidable.

Review 07/09 round 1 (24 agents, 16 confirmed → 9 distinct, 0 refuted — all
fixed): near-plane clipping (edge-pivoted triple tumbles swing to z≈23, whole
subjects vanished from view AND exports); capture-window races (mid-capture
control changes were suppressed but never reconciled: stale axis lines, lying
silhouette checkbox, '_sil' filename on textured frames, stop silently
overridden, image decodes dropped); bPNG unguarded + async restore flash;
ortho scroll-zoom ignored by front-view/camera-swap; false comments ("camera
forced front", "one fixed world light").

## card_spin3d: back image + thickness slab, SHIPPED 07/09/2026

Two user features: (1) BACK IMAGE — a loadable second face (chiral demo back
included); the card becomes two FrontSide planes, back rotated 180° so its
image reads unmirrored when facing you (a physical card; deliberately trades
the dancer ambiguity for a playing-card). Different aspect = centre-cropped to
the front's (letterboxing would alphaTest away). (2) THICKNESS — extrudes the
card into a slab; the four walls are the image's border pixels stacked through
the depth (the user's "stack of whatever the last pixel is"), via 1-px
edgeStrip textures on a 6-material box (UV layout verified against real
three@0.160 BoxGeometry). Math: front view unchanged (τ projects to 0); for a
centred axis the slab silhouette outline w|cosθ|+τ|sinθ| is EVEN in θ — the
dancer ambiguity survives thickness exactly; but a constant τ cannot match a
drawn figure's per-beam depths, so mid-turn an impossible-figure image reads
honestly as picture-on-a-slab (the depth-true object is the 3D beam subject).

Review 07/09 round 2 (15 agents, 10 confirmed → 7 distinct, 2 refuted — all
fixed): silhouette slab walls vanished (cutout borders are alpha-0, alphaTest
discarded every wall fragment → floating sheet with ODD τ/2·sinθ offset that
would have made asymmetric silhouettes direction-decidable — walls are now
solid sil-color, restoring the even outline); constant camera planes could not
survive wide images (swing bound 6·r₀ = 3·hypot(w,h) GROWS with aspect) — the
ortho depth window and perspective fov are now derived per subject in
updateCameraBounds(); unbounded ortho zoom-out carried through the swap past
the persp far plane (blank stage) — far raised, controls capped, carry
clamped; thickness drags re-uploaded full-res textures per tick (now a
geometry-only fast path); back image was silently stretched (now cropped);
mid-capture play-after-stop lost (STOP flag → last-click-wins tri-state);
deferred alerts popped over raw export state (flush moved after restore).
Refuted (both rounds, same pattern): "restorePose paints a wrong-pose frame" —
the rAF loop always re-applies the pose before the compositor samples; renders
into the drawing buffer that are never presented are not defects.

Review 07/09 round 3 (5 agents, on the round-2 fix implementations themselves;
3 confirmed, 1 refuted — all fixed): the swing bound 6·r₀ = 3·hypot(w,h) was
sharp only for the subject's CENTER — three 180° point-reflections at offsets
−1/+1/−1 send the far CORNER to 7·r₀ = 3.5·hypot(w,h) (verified: a 5:1 image
reaches 107.08 vs my claimed 97.8 — my own round-2 "clears the bound" check
was measuring the wrong point; crossover at aspect √3, which is why the 1.5:1
demo card passed by 0.55 and hid it); minZoom 0.01 was inconsistent with
maxDistance 9000 at fov 12 (the ortho-entering swap assigned zoom 0.0053 and
r160 OrbitControls re-clamps zoom EVERY update() — a ~2× snap one frame after
the toggle; now 0.005 ≤ VIEWH/(maxDist·tan6°)); the fov-rescale in
updateCameraBounds could place the camera past maxDistance and get yanked back
next frame (now clamped same-frame). Verified after fix: 5:1 fov 5.019°,
distance 114.08 > corner swing 107.08. Lesson: when one fix writes camera
state and another fix imposes controls limits, check the pair for mutual
consistency — and swing bounds must be taken over the whole body, not its
centre. Refuted: the unpresented-render claim, a third time.

## Backlog

- Helical beam ring app (the "possible cousin"): hexagonal walk climbing
  k(1,1,1) per lap. Verified 02/09: inter-lap connectors project to CLOSED unit
  triangles (any lap-to-lap path has edge-sum k(1,1,1) → projects closed; the
  minimal x,y,z staircase connector is literally a unit triangle in projection —
  the user's "triangle-flavored" guess exact); a k-climb helix admits exactly
  k−1 interleaved disjoint copies via j·(1,1,1) shifts (multi-start screw
  threads: k=2 double helix, k=3 triple; k=1 none); pitch per lap = k√3 equals
  the area-balanced hexagonal prism height (√3/2)ℓ exactly when k = ℓ/2
  (algebra verified; meaning open — do not oversell).
- Spinning-dancer replica (three.js, silhouette renderer with optional
  splitting lines; "less weird shape" — e.g. a rotating chiral polycube
  silhouette).
- Sierpiński tribar (3 sub-tribars of size L/2 at the corners of a size-L
  triangle, vertex-sharing corner cubes; every triangle cycle is forced
  impossible at every scale, so the recursion works at all levels with
  L divisible by 2^depth).
