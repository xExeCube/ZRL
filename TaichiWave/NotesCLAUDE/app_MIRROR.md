# The Taichi app: controls, what changed in round 2, and how it was verified

*Generated from scratch — Claude Opus 5.5 — 30/09/2026*

`membrane_app_MIRROR.py` is the primary app since 29/09/2026 (the web app is frozen). Round 2 does the user's feedback of 29/09/2026:
- visibility;
- an orbit camera with zoom;
- hand-drawn walls;
- solid objects and the Unlicensed plate;
- the alignment and ledger panels.

Every number below was measured on 30/09/2026 on the user's machine (i9-9900K, RTX 5070 Ti, Windows 11, taichi 1.7.4).

## Files

| file | what it holds |
|---|---|
| `membrane_app_MIRROR.py` | The viewer: the main loop, input dispatch, the main panel, the four sub-windows, the `--shot` CLI and `programmatic_stroke()`. |
| `app_render_MIRROR.py` | Look settings (backgrounds, palette zero level, exposure, contrast, solid and outside colours, shading) and the display kernels (`k_verts`, `k_normals`, `k_image`, `k_image_index`). Also the mesh (domain + floor triangles), the overlays, and the contrast helpers (sRGB → L*, segmentation). |
| `app_camera_MIRROR.py` | `OrbitCam` (fixed target, clamps) and `CamPose` (GGUI's lookAt + perspective, `world_to_screen`, `screen_to_ray`, `screen_to_plane`). Also `View2D` (2D zoom/pan, pixel ↔ physical) and the world ↔ physical ↔ grid conversions. Pure math: no Taichi. |
| `app_input_MIRROR.py` | `Frame` (one frame of input), `poll_window()` (the live GGUI window → Frame) and `DigitPoller` (Win32 digits for the text entry). |
| `app_solids_MIRROR.py` | `SolidsUI`: painting (begin / extend / end), objects (add / select / edit / delete), text entry, scene files, the Unlicensed toggles, and the walls-and-solids panel. |
| `app_tests_MIRROR.py` | Headless tests (plain script): 54 checks on CUDA. |
| `ScratchCLAUDE/AppCLAUDE/` | Shots, probes and logs. `shots_MIRROR.py` renders every `--shot` variant (38). |

## Controls

### Keys

Letters only: GGUI 1.7.4 names no digit or punctuation key (see *GGUI facts*).

| key | action |
|---|---|
| SPACE | play / pause |
| N | one step |
| R | reset |
| P | drop a pulse |
| V | 3D / 2D |
| X | wireframe |
| H | hide / show all panels |
| L | lattice |
| B | boundary |
| G | domain shape (the cycle includes `unlicensed`) |
| M | medium |
| O | source |
| C | reset the view (orbit, and the 2D zoom/pan) |
| F | freelook on / off |
| Z / SHIFT+Z | zoom in / out |
| arrows | 3D: orbit 5° per press · 2D: pan |
| Y | cycle the background |
| T | draw-walls mode |
| U | undo the last stroke |
| I | walls-and-solids panel |
| J | alignment panel |
| K | ledger panel |
| Return | type the selected text object's text |

All round-1 keys are kept with their meaning. The new ones use letters that were free. **No app key uses W/A/S/D/Q/E**: those belong to GGUI's freelook (a test presses them and checks that no app state changes).

**Text entry.** Return opens it on the selected text object (walls-and-solids panel).
- Letters, space and BackSpace type. Letters are capitals by default; a *capitals* checkbox turns that off.
- Digits and `- . , / + '` also type. They arrive through Win32 `GetAsyncKeyState`, and only while the app's window is in front.
- Return applies the text; Escape cancels.
- While typing, no hotkey fires and freelook is suspended, so W/A/S/D type letters instead of moving the camera.
- An empty text keeps the old one.

### Mouse

| view | button | action |
|---|---|---|
| 3D, orbit | RMB drag | orbit about the domain centre (180° per window height) |
| | MMB drag, or CTRL+RMB drag (vertical) | zoom (×e^(3·dy)) |
| | LMB drag, draw mode off | orbit |
| 3D, freelook (F) | | GGUI's own camera: W/A/S/D/Q/E move, RMB held + mouse looks |
| 2D | RMB or LMB drag | pan (the content follows the cursor) |
| | MMB drag, or CTRL+RMB drag | zoom about the cursor |
| draw mode (T), both views | LMB | paints a wall |
| | SHIFT+LMB, or the *eraser* checkbox | erases (red brush ring) |

**Orbit.** The target is fixed at the domain centre (the world origin). Elevation is clamped to ±89°, distance to [0.15, 25], and azimuth wraps to (−180°, 180°].

**Freelook.** It starts from the current orbit view. The orbit state is kept untouched and comes back when F is pressed again.

**Presses on a panel.** A press inside a visible panel's rectangle never paints, orbits or pans.
- The rectangles are the panels' default places.
- GGUI places a sub-window once per session, so if a panel is dragged elsewhere, its *old* rectangle is what counts.

### Panels

| panel | where | contents |
|---|---|---|
| main (left, 25%) | always shown | every round-1 control; a **view** section (3D, wireframe, mesh detail, freelook, zoom, azimuth, elevation, 2D zoom, reset view); a **look** section; panel toggles; readouts (cells, solid cells, E, max\|u\|, ms/step, frame ms, 3D mesh size) |
| walls and solids [I] | right, top | see below |
| alignment [J] | right, bottom | `alignment_MIRROR.gui_section` |
| ledger [K] | bottom centre | `ledger_MIRROR.gui_section` |

**Look section:**
- background;
- exposure, zero level, contrast;
- auto colour gain, colour gain, height scale;
- 3D shading, the array outside the domain, solid height;
- 2D hatch;
- solid colour and outside colour.

**Walls-and-solids panel:**
- draw mode, eraser, brush (cells);
- stroke count and update latency;
- undo, clear painted walls;
- the Unlicensed plate's lettering and screw-head toggles, or a *switch to the plate* button;
- add rect / circle / n-gon / text;
- the object list, select < >, and editing of the selected object: enabled, anchor, x, y, rotation, and its size (w, h / r / n / cap height, fit width, stretch, letter spacing, font style);
- text entry and text presets;
- delete;
- save scene, and the scene files list with load.

**Alignment panel.** Its *colour the domain edges* flag also draws the coloured edges in both views.

**Ledger panel.** It calls `ledger_MIRROR.gui_section(w, viewer)`; the ledger agent's module is on disk and the panel renders its rows (final `--panels` shot, 30/09/2026 late run). If the module is missing or has no `gui_section`, a one-line placeholder is shown, and the import is retried when the panel is opened.

The "Co change = new time step" toggle is kept, off by default. So a Co change mid-run makes the time reflection, which the user likes.

## What changed

### 1. Visibility

**Measured before** (`ScratchCLAUDE/AppCLAUDE/before_*`, rendered by a copy of the round-1 app):
- The domain at rest was 0.05 grey (CIE L* ≈ 4) on a (0.02, 0, 0.05) page (L* ≈ 0.5).
- At colour gain 1, a pulse ring after 300 steps was nearly the rest colour: the pulse is down to ~0.1 of its start by then.

**Now:**
- **Palette.** `colour = (Z + (end − Z)·|x|^(1/contrast))·exposure`, with Z the zero level (default 0.22 grey) and the web app's teal/red ends. With zero 0.05, exposure 1 and contrast 1 it is exactly the web app's `colorOf('div')`: a test checks 601 values to 1e-12 (host) and 7e-8 (kernel). The default look keeps luminance rising with |u| on both signs; the sign separation at |u| ≥ ½ is 0.526 (the web app's house ramp: 0.555).
- **Auto colour gain** (default on; display only): the gain divides by the peak |u| of the last 64 recorded steps. A 256-step window still held the pulse's early, 2–3× larger peak. `--no-auto-gain --zero 0.05 --no-hatch` gives the round-1 / web-app look.
- **Four cell classes, four looks:**
  - outside the array: the background, one of near-black (default), dark-grey, dark-purple;
  - outside the domain: a darker tone matched to the background, hatched in 2D, a flat floor in 3D (toggle);
  - solid cells: off-white (0.84, 0.84, 0.80) like the plate lettering, raised by `solid_h` (default 0.02) in 3D;
  - the domain: the palette.
- **3D shading.** 0 = unlit (the web app's MeshBasicMaterial). Above 0, a headlight at the camera: ambient 1 − 0.55 s, light 0.8 s, GGUI's two-sided Lambert. Default 0.5.

**Contrast**, mean CIE L* per pixel class. The pixel classes come from a segmentation render of the same frame; edge pixels blended between classes are left out. Circle and square at N = 513 after 300 steps; round 1 is its own PNGs, segmented at its camera.

| view | look | background | outside the domain | domain | domain − background | WCAG ratio |
|---|---|---|---|---|---|---|
| 2D | round 1 | 0.5 | 6.3 | 3.4 | **2.9** | 1.06 |
| 2D | now | 0.6 | 8.4 | 24.6 | **24.0** | 1.86 |
| 3D | round 1 | 0.6 | — | 4.4 | **3.8** | 1.08 |
| 3D | now | 0.6 | 6.7 | 24.6 | **24.0** | 1.86 |

Per background, on the Unlicensed plate (2D / 3D; `ScratchCLAUDE/AppCLAUDE/contrast_bg_MIRROR.txt`):

| background | background L* | outside L* | domain L* | solid L* |
|---|---|---|---|---|
| near-black | 0.6 / 0.6 | 8.4 / 6.7 | 24.3 / 24.6 | 85.4 / 84.0 |
| dark-grey | 10.3 / 10.4 | 3.9 / 3.1 | 24.3 / 24.6 | 85.4 / 84.0 |
| dark-purple | 4.2 / 4.5 | 15.6 / 13.1 | 24.3 / 24.6 | 85.4 / 84.0 |

The dark-purple page's outside tone was first darker than the page: only 1.8 L* apart from it. It is now a lifted muted violet.

The WCAG ratio stays small for any dark scene (it adds 0.05 to both luminances), so the L* difference is the meaningful number. HDR/AutoHDR on the display changes the absolute brightness; the exposure slider is there for that.

### 2. Camera

The orbit camera and freelook are described under *Controls*.

**Projection.** The projection is GGUI's own: glm lookAt, vertical fov 45° (`proj[1][1] = 2.41421` measured), aspect = window width / height. `CamPose` reproduces it.

Markers that GGUI rendered at 4 known world points, under 6 poses, land within **≤ 0.17 px** of `world_to_screen`:
- az 0 / 35 / −120 / 170 / 60 / −45;
- el 38 / 60 / 20 / 85 / −30 / 45;
- dist 0.9–3.5.

`screen_to_plane(world_to_screen(P)) = P` to **1.9e-15** world units (1200 points).

**No flip.** For every elevation in [−89, 89] and 5 azimuths (1785 poses), screen-up has +Y and the eye is at the set distance from the fixed target.

### 3. Hand-drawn walls

**Mapping the cursor.**
- **2D:** the cursor maps through `View2D`, the exact inverse of the image mapping. For every pixel of the 1600×1000 window, the cell the image SHOWS (`k_image_index`, the same `pixel_cell()` the image kernel uses) equals the cell the cursor at that pixel's centre maps to:

  | lattice, N, zoom, pan | pixels that differ |
  |---|---|
  | sq 513, 1.0 | 0 |
  | sq 161, 12.5 | 0 |
  | sq 2049, 0.6 | 0 |
  | tri 257, 3.7 | 1 |
  | tri 1025, 1.0 | 8 |

  Every mismatch lies within 1.1e-5 cells of a cell edge (f32 in the kernel, f64 on the host).
- **3D:** the cursor ray is intersected with the rest plane Y = 0.

**Painting.**
- Each mouse sample calls `extend_stroke`, then `update_solids(box)`.
- A segment is a capsule between two samples, so fast strokes stay continuous.
- Samples closer than max(0.35, 0.25 r) cells to the last point are skipped.
- Stroke points are clamped to |g| ≤ 2 grid units: a 3D ray near the horizon would otherwise make a huge box.

**The brush ring** shows the brush at the cursor.
- In 3D it is drawn above the surface, so solids and crests cannot hide it. Each point is slid along its own view ray, so the ring covers exactly the pixels of the Y = 0 ring the brush paints.
- **Known parallax:** a raised solid's top face shows shifted by `solid_h / tan(elevation)`: 0.026 of the array's half-width at the default 38°. Painting is on the rest plane.

**Through the real input path** (fake mouse frames into `Viewer.handle`), the stroke points equal the intended grid points to ≤ 2.4e-13 cells, and every intended path cell is solid. This holds in 2D, and in 3D at (az 30, el 50, dist 2.4) and (az −140, el 25, dist 3.1). In `--walls-test` shots, the cursor round trip is ≤ 1.6e-13 cells on sq and tri.

**Latency.** One segment, meaning `extend_stroke` + `update_solids` + a GPU sync, brush 2 cells, 199 segments:

| N | CUDA f64: median / p95 / max | Vulkan f32: median / max |
|---|---|---|
| 513 | 3.05 / 3.72 / 4.51 ms | 3.38 / 4.77 ms (run 2: 4.81 / 8.25 ms) |
| 2049 | 3.13 / 3.83 / 5.13 ms | 3.28 / 4.92 ms (run 2: 3.51 / 8.77 ms) |

Final runs of 30/09/2026 (late). An earlier run gave 2.8-3.7 ms medians and Vulkan maxima up to 15.6 ms; all are far under the 20 ms target.

**Undo, clear, scenes.**
- U undoes the last stroke.
- *Clear painted walls* uses `clear_strokes()`, which triggers a full re-rasterisation: ~3 s at N = 8193 with the plate lettering, per the core.
- Scenes save to `ScenesCLAUDE/scene_<date-time>_MIRROR.json` and load from a list of files in that folder. A save/load round trip gives the same 3,706 solid cells.

### 4. Objects and the Unlicensed plate

**Objects** (rect, circle, n-gon, text) have their add / select / edit / delete controls listed under *Controls*.
- Positions and sizes are in grid units: the array's half-size, the same at every N and on both lattices.
- `anchor = domain` makes an object turn with the domain.
- An edit re-rasterises only the union of the object's old and new cell boxes.
- Text presets: DEATH, CA EXEMPT, ZRL, WAVE, GOVERNMENT PLATES, EXMILITARY, NO LOVE.

**The Unlicensed plate.**
- The shape cycle includes `unlicensed`.
- On entering it (key G, the panel button, or `--set shape=unlicensed`), the default source (sx = sy = 0) moves to **sy = −0.9**. The centre cell is solid, and the nearest free cell is inside the A's counter, a sealed cavity.
- On the triangular lattice the radius drops to **0.89**: at radius 1 the rhombic array clips the plate.
- Both defaults are flashed in the main panel. An explicit `sx`, `sy` or `rad` in `--set` wins over them.
- The lettering and screw-head toggles work: 22,578 solid cells with both, 324 with the screws only, 0 with neither (N = 513). With neither, the solver returns to the no-solids path.

### 5. Alignment and ledger panels

Both are separate sub-windows, toggled with J and K and listed under *Controls*. Their errors are caught and shown in the panel; they never end the app.

The alignment module's own text wrap (64) overflows the 400 px panel. The app creates its `AlignPanel` with `wrap = 52`.

### 6. Kept from round 1

- `--shot`, the hotkeys, the Co-change toggle, and `Viewer.moved`'s relative tolerance. The tolerance is now used by every slider, and a test checks that the f32 round trip of 10 typical values is not a change.
- `--set` Params overrides.

## `--shot` options

The command is `python membrane_app_MIRROR.py --shot out.png [options]`.

| option | effect |
|---|---|
| `--view 3d/2d` | the view |
| `--N` | grid size |
| `--steps` | steps before the shot |
| `--set key=value ...` | Params overrides (`rot_deg=30` sets `rot` in degrees) |
| `--hs`, `--cg` | height scale, colour gain |
| `--no-auto-gain` | use the colour gain as given |
| `--wire` | wireframe |
| `--camera az,el,dist` | orbit camera. A negative first value is fine: `--camera -135,60,2` |
| `--zoom Z`, `--pan x,y` | 2D zoom; pan in grid units |
| `--freelook` | start in freelook |
| `--bg near-black/dark-grey/dark-purple` | background |
| `--zero`, `--exposure`, `--contrast` | the palette |
| `--shading s` | 3D shading (0 = unlit) |
| `--no-floor`, `--no-hatch` | hide the 3D floor / the 2D hatch |
| `--solid-h`, `--solid-col r,g,b` | solid height and colour |
| `--segment` | the segmentation colours used by the contrast tests |
| `--scene file.json` | load a scene; a bare name looks in ScenesCLAUDE |
| `--plate-text 0/1`, `--plate-screws 0/1` | the Unlicensed toggles |
| `--walls-test` | paint a zig-zag wall through the real input path in the chosen view, and print the cursor round-trip error and the update latency |
| `--brush cells` | brush size |
| `--marker x,y,z` | a magenta particle at a world point |
| `--panels main,solids,align,ledger` | show those panels in the image. One warm-up frame is rendered first, so ImGui has laid them out. |

`ScratchCLAUDE/AppCLAUDE/shots_MIRROR.py` renders all 38 variants: **38/38 OK** (final re-run 30/09/2026 late, against the live core, solids and ledger modules) (`shots_log_MIRROR.txt`, `shot_*_MIRROR.png`, `contact_sheet_MIRROR.png`). They cover:
- 3D orbits: az 45 / el 25, az −135 / el 60, top at el 89, below at el −25;
- zoom out and in;
- lit, unlit, full shade, no floor, wireframe;
- 2D zoom 3 with pan;
- the three backgrounds in 2D, and dark-purple in 3D;
- the web-app look, exposure and contrast;
- walls in 2D, 3D and 3D-tri;
- Unlicensed on sq and tri, 2D and 3D, screws only, a custom solid colour, rotated 20°;
- a scene file, freelook, a marker, a segmentation render, tri n6 at 30°, N = 2049, and panels.

A plain shot hides the panels.

## Tests: `python app_tests_MIRROR.py [--arch cuda|vulkan|cpu] [--f32] [--quick]`

| run | result |
|---|---|
| CUDA f64 (full) | **54/54** in 38 s (`ScratchCLAUDE/AppCLAUDE/app_tests_cuda_MIRROR.txt`) |
| Vulkan f32 (full), run 1 | 53/54 (`app_tests_vulkan_f32_MIRROR.txt`): the 3D frame at mesh 513 took 30.7 ms against its 30 ms limit, while another agent's Python process was on the machine |
| Vulkan f32 (full), run 2 | **54/54** in 38 s (`app_tests_vulkan_f32_run2_MIRROR.txt`): the same frame 19.8 ms |
| CPU f64 (quick) | **48/48** in 28 s (`app_tests_cpu_quick_MIRROR.txt`) |

The frame-time limits (85 ms at mesh 1025, 30 ms at 513 and in 2D) are regression guards, not user targets; the Vulkan 513 frame sits close to its limit and is noisy under load (19.8-30.7 ms in two runs).

What the tests cover: sections 1–9 of the file's docstring, plus the numbers above.

**Frame cost at N = 2049, every panel open.** The frame is 4 steps + draw + panels + render. The numbers exclude the ~51 ms headless readback that the tests need to flush a frame:

| 3D mesh detail | CUDA | Vulkan f32 |
|---|---|---|
| 1025 (2.1M triangles) | ~51 ms | ~61-70 ms |
| 513 (default) | ~16-20 ms | ~20-31 ms |
| 2D view | ~9-11 ms | ~11-22 ms |

The 3D cost is mostly GGUI's own per-frame vertex upload. At 1025, split: its `copy_all_to_vbo` kernel ~15 ms, the C++ mesh call ~20 ms; the app's kernels are 1.4 ms. So the default mesh detail is now **513** (round 1: 1025); the panel slider goes 129–1025. The 2D view always shows every cell.

## GGUI 1.7.4 facts found on the way

Each is measured, with its probe in ScratchCLAUDE/AppCLAUDE.

**Rendering**
- **Colours are written linearly:** 0.2 → PNG 51, 0.5 → 127. Palette numbers are display values. set_image column py = 0 is the image's bottom row.
- **A mesh vertex whose normal is 0 renders BLACK**, even with no point light. An unlit frame right after a new N was all black. The app fills the normals with +Y on allocation.
- **`scene.mesh()` with a ti.field index buffer calls `to_numpy()` on it every call:** 12 ms, and ~50 ms per frame more than a numpy array at 2.1M triangles. The app passes a numpy prefix of exactly the drawn length. Round 1 paid this cost.
- **GGUI's `gen_normals`**, used when no normals are passed, runs 3 atomics per triangle every frame over the whole index field. A padded (0,0,0) triangle gives vertex 0 a NaN normal. The app computes its own normals (0.4 ms).

**Windows and frames**
- **One GGUI window per process.** A second window asserts in imgui_impl_vulkan.
- **`show_window=False` creates no OS window**, so keys and mouse cannot be injected headless; the tests drive `Frame`s. **`window.show()` on such a window segfaults**, and the tests flush with a readback instead.
- **Headless frames must each be rendered.** GGUI keeps the draw calls of unrendered frames: after 18 draws without a render, the next `save_image` showed a stale 2D image over the scene.
- **Panels do show in saved images once ImGui has had one frame** (`--panels`). The round-1 README's "not in `--shot` images" is true only of the first frame.
- **Sub-windows are placed once per session.** The app's panel rectangles are the defaults.

**Keys**
- **Key names:** a–z plus Shift, Control, Alt, Escape, Return, Tab, BackSpace, space, the arrows, CapsLock, LMB/MMB/RMB, from the binary's string table.
- **Digits and punctuation have no name** (the binary has an "unrecognized id" error). `poll_window()` and `show()` swallow that error. Whether GGUI raises it in a live window is **not verified**: no keyboard exists in headless mode. Digits for the text entry come from `DigitPoller` instead.

## Open issues

- **Not tested live.** The real keyboard and mouse in a live window cannot be tested headless: no OS window, and the rules say no visible windows. This covers the digit path (`DigitPoller`: Win32, foreground check) and GGUI's behaviour on digit keys. The whole input logic is tested through `Frame`.
- **Moved panels.** Painting and orbiting are blocked in the panels' default rectangles. After the user drags a panel, the old rectangle still blocks, and clicks on the panel's new place act on the view. GGUI 1.7 exposes neither ImGui's `WantCaptureMouse` nor the window rectangles.
- **Parallax on raised solids.** A raised solid's top face shows offset by `solid_h / tan(el)`, because painting is on the rest plane. Set solid height 0 for exact 3D painting over existing walls.
- **3D frame cost.** At N ≥ 1025 with mesh detail 1025, the 3D frame is ~50–70 ms. This is GGUI's vertex upload (CUDA → Vulkan interop, and its own copy kernel), not app work.
- **Slow full updates.**
  - *Clear painted walls*, the plate toggles and scene loads do a full re-rasterisation: ~3 s at N = 8193 with the lettering.
  - `m.domain_cells()` is a full-grid sum: 2.3 ms at N = 2049. The panel recounts at most twice a second.
- **The recorder feeds the auto gain.** The auto colour gain reads the recorder's history (`m.record`, default on). With the recorder off it falls back to `m.amax`, which only changes at seed and drop.
