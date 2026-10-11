# The Taichi app: controls, what changed in round 2, and how it was verified

*Generated from scratch — Claude Opus 5.5 — 30/09/2026. Round 4 / v1.0 (05/10/2026): the key and panel tables below are updated, and the last section, *Round 4 / v1.0*, describes the new controls. Edited by Claude Opus 5.5 — 05/10/2026; the 3D fit of the triangular array and the final test counts 06/10/2026.*

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
| `app_tests_MIRROR.py` | Headless tests (plain script): 54 checks on CUDA on 30/09; 109 since v1.0 (06/10/2026). |
| `ScratchCLAUDE/AppCLAUDE/` | Shots, probes and logs. `shots_MIRROR.py` renders every `--shot` variant (38). The v1.0 step's shots and logs: `ScratchCLAUDE/Round4CLAUDE/AppCLAUDE/` (`shots_r4_MIRROR.py`, 21 variants). |

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
| B | boundary: the next one that acts on this lattice and shape (05/10/2026). The triangular `array`: dirichlet, neumann, periodic (Mur skipped). A triangular web-app shape: no change, a note says why |
| G | domain shape (the cycle includes `unlicensed` and `array`) |
| M | medium |
| O | source |
| C | reset the view (orbit, and the 2D zoom/pan; the triangular `array` is zoomed to fit the panel-free strip in 2D, and in 3D the camera moves back so it fills the window as the square does, 06/10/2026) |
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
| main (left, 25%) | always shown | the version line (`ZRL wave membrane v1.0 (05/10/2026)`); every round-1 control; the boundary block and the **edge** button (05/10/2026); the source note, the drive warning, the phase lag; a **view** section (3D, wireframe, mesh detail, freelook, zoom, azimuth, elevation, 2D zoom, reset view); a **look** section; panel toggles; readouts (cells, solid cells, E, max\|u\|, ms/step, frame ms, 3D mesh size) |
| walls and solids [I] | right, top | see below |
| alignment [J] | right, bottom | `alignment_MIRROR.gui_section` |
| ledger [K] | bottom centre | `ledger_MIRROR.gui_section` |

**Look section:**
- background; palette (click: next preset), *edit the custom palette*, surface opacity, grid lines (3D) — see *Palettes and transparency* below;
- exposure, zero level, contrast;
- auto colour gain, colour gain, height scale;
- 3D shading, the array outside the domain, solid height;
- 2D hatch;
- page colour (outside the array; picking one overrides the background preset until Y cycles it), solid colour and outside colour.

**Walls-and-solids panel:**
- draw mode, eraser, **brush wall** (pinned / rigid, 05/10/2026), brush (cells);
- stroke count and update latency;
- undo, clear painted walls;
- **wall types** (05/10/2026): the explainer, the legend, the pinned / rigid cell counts, *walls kernel ON (n cells)* or *off*, the **edge** button, *draw the edge type*;
- the Unlicensed plate's lettering and screw-head toggles and the **plate lettering wall**, or a *switch to the plate* button;
- add rect / circle / n-gon / text;
- the object list (a rigid object is tagged RIGID), select < >, and editing of the selected object: enabled, **wall**, anchor, x, y, rotation, and its size (w, h / r / n / cap height, fit width, stretch, letter spacing, font style);
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
| `--plate-wall pinned/rigid` | the plate lettering's wall type (05/10/2026) |
| `--brush-wall pinned/rigid` | the wall type of `--walls-test`'s stroke (05/10/2026) |
| `--win W,H` | the window size, e.g. `2560,1377` for a maximised window (05/10/2026) |
| `--version` | print `ZRL wave membrane v1.0 (05/10/2026)` and exit (05/10/2026) |
| `--walls-test` | paint a zig-zag wall through the real input path in the chosen view, and print the cursor round-trip error and the update latency |
| `--brush cells` | brush size |
| `--marker x,y,z` | a magenta particle at a world point |
| `--panels main,solids,align,ledger` | show those panels in the image. One warm-up frame is rendered first, so ImGui has laid them out. |
| `--palette NAME` | teal-red / greyscale / cool-warm / cyclic / hot-cold / custom |
| `--stops "t:r,g,b;..."` | a custom palette (sets `--palette custom`), e.g. `--stops "-1:0.1,0.45,0.95;0:0.2,0.16,0.26;1:1,0.55,0.1"` |
| `--opacity a` | surface opacity (3D real alpha; 2D: the field mixed toward the page) |
| `--grid g`, `--grid-every k`, `--grid-col r,g,b`, `--grid-field` | 3D grid lines: opacity, a line every k mesh vertices, colour, or the field's colours |
| `--bg-col r,g,b` | a custom page colour |
| `--settings file.json` | load a palette settings file (shots ignore `look_settingsMIRROR.json` otherwise) |

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
- **Alpha (01/10/2026, `ScratchCLAUDE/VisualsCLAUDE/*probe*MIRROR.py`).**
  - `scene.mesh` takes a vec4 `per_vertex_color` and blends it (source-alpha over; green A 0.5 over red gave (0.50, 0.50, 0)). Depth is still written and nothing is sorted: drawn front first, a translucent quad hides the one behind it.
  - `show_wireframe=True` draws only the triangle edges (no fill), in the vertex colours, alpha-blended. A degenerate triangle (a, b, b) draws its edge a-b, but twice: alpha 0.6 showed as 0.84 = 1 - 0.4^2.
  - `scene.particles` blends. `scene.lines`, `canvas.lines` and `canvas.triangles` ignore alpha (opaque).
  - Calls draw in call order across kinds: a `scene.lines` call made before a translucent mesh shows through it. A point light keeps the alpha.
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

## Review v3 changes (Claude Opus 5.5, 01/10/2026)
- **Resized / maximised window.** `Viewer.win_real` is read from `window.get_window_shape()` every frame and drives
  the 3D aspect (cursor ray, orbit drag). `Viewer.win` is the 2D image buffer; it follows the window once the new
  size has held 0.3 s (`RESIZE_SETTLE`; `Renderer.resize` keeps one buffer per size). Until then GGUI stretches
  the old buffer, and the 2D mapping (screen fractions of the buffer) stays exact. Before: 11-14 cells (N = 257)
  paint error in 3D and a 1.16x-wide 2D circle at 2560 x 1377.
- **Auto colour gain cap.** The peak is floored at 1e-3 x the run's largest |u| (`AUTO_REL_FLOOR`), so a field that
  has died away fades out instead of showing amplified round-off. When the cap applies, the panel says "capped at 1000x".
- **2D picture centred in the free strip** between the visible side panels (`View2D.ox`, `Viewer.free_centre`).
  The picture shifts when panels are toggled; with H (no panels) it is centred in the window.
- **N slider** applies its value once it has rested 0.35 s (`N_SETTLE`). Each N change costs a rebuild of ~1 s in 3D.
- **Typing:** adding an object closes the text entry, Return never writes text onto a non-text object, and Return
  brings hidden panels back.
- **Cell counts** in the main panel are recounted only when the masks or the solid count change. Before, a 36 ms
  sum every 0.5 s at N = 8193 caused a periodic stutter.
- **`--walls-test`** with the path behind the camera prints "walls-test skipped: ..." instead of crashing.
- **imgui.ini** cannot move or resize panels (GGUI re-applies the coded rect every frame), but it does keep a
  panel's collapsed state between sessions, and that also applies to `--shot --panels` images.
- Tests: CUDA f64 **61/61** (`ScratchCLAUDE/AppCLAUDE/app_tests_cuda_v3_MIRROR.txt`). Shots: **38/38 OK**
  (`shots_v3_stdout_MIRROR.txt`).

## Typed slider values and the source-moved note (Claude Opus 5.5, 01/10/2026)
- **Typing a slider's value (pick mode).** Each panel's first row is a button *type a value: click here, then a
  slider*. After it, the next slider you click opens an entry under that slider; the click itself does not change
  the value. The entry takes digits `. - e` (`1e-3` works), BackSpace, Return (apply) and Escape (cancel). The value
  is clamped to the slider's range (the panel flashes "outside lo .. hi, clamped to ..."); int sliders round. A
  bad number ("1e") keeps the entry open with a red message. While the entry is open every key goes to it (no
  hotkey fires, freelook's W/A/S/D/Q/E are off, the Win32 digit poller is on). While only armed, hotkeys still work
  and Escape (or the button again) stops pick mode. The value goes back through the slider's own code path, so
  N still waits for `N_SETTLE` and the sliders that restart the run still restart it.
- **Why pick mode.** GGUI 1.7.4 has no `same_line()` and no text-input widget, so a button cannot share a slider's
  row. A button under every slider (the first build) made the panels much longer: the main panel went from 67 to
  87 rows, the solids panel from 37 to 43. Pick mode costs one row per panel. The per-slider buttons are still
  there behind the main-panel checkbox *a "type" button under every slider* (off by default).
- **ImGui's own Ctrl+click on a slider** (type into the slider in place) may or may not work in GGUI 1.7.4. GGUI
  has no names for digit keys in its own key table (see *GGUI facts*), but Dear ImGui may get characters
  through its own GLFW callbacks, so this is unknown and cannot be tested headless (no OS window). The user is
  testing it. If it works, pick mode is a fallback; the two do not interfere while pick mode is off (a Ctrl+click changes no value
  until the in-place entry is applied, and then the slider just reports the new value).
- **Source-moved note.** Under source x / y, an amber line "source moved out of a solid to (x, y)" appears when the
  requested source point lies in a solid (for example (0, 0) on the Unlicensed plate's A counter). It says "into
  the domain" when the point was outside the domain. It uses the core's `m.src_moved` when the core has it; until
  then it compares the requested cell (`round_cell`) with the cell actually used (`m.src`).
- Tests: `t_typed`, `t_src_note` (headless `FakeG` / `FakeW` drive the real panel code; scripted slider clicks and
  `Frame` key events). Runner for just these: `ScratchCLAUDE/AppCLAUDE/run_typed_MIRROR.py`.

## Directional sources in the panel (01/10/2026, Claude Opus 5.5)

With `beam` or `packet` selected (key O), the source section shows *beam angle (deg)*, *beam width w (cells)* and *aim the energy*. For the beam, it also shows *one-sided*, the freq slider and a readout (λ, cells driven/clipped, phase-front angle, a warning under 8 cells per wavelength). For the packet, it shows *packet length L* and *packet wavelength*. The beam's parameters apply live; the packet's relaunch it. *draw the emitter* toggles the overlay (aperture in cyan, arrow in orange) in both views. Pick-mode typing works on all the new sliders (app_tests `t_beam`). Design and measurements: [beam_MIRROR.md](beam_MIRROR.md).

## Palettes and transparency (01/10/2026, Claude Opus 5.5)

User request 01/10/2026: customizable gradient colours, and grid/mesh transparency like the web app's surface-opacity slider.

### Palettes

A palette is a list of colour **stops** `(t, rgb)` over t in [-1, 1], linear between stops, with stops at -1, 0 and +1 (at most 17). The colour of a field value u is the palette at `t = sign(x)|x|^(1/contrast)` with `x = u * colour gain` clamped to [-1, 1], times exposure, clipped at 1. Host (`palette_np`) and kernel (`palette`, a loop over the stops in the parameter block) are the same function. 2D and 3D both use it.

| preset | stops | zero level |
|---|---|---|
| teal-red (default) | the web app's teal and red ends, grey zero | 0.22 |
| greyscale | black, grey, white | 0.45 |
| cool-warm | Moreland's perceptually uniform diverging map (matplotlib `coolwarm`, 9 samples) | 0.87 |
| cyclic | matplotlib `twilight` (17 samples): x = -1 and +1 are the same colour, dark at 0, blue on the negative half, red on the positive half | 0.21 |
| hot-cold | dark zero; negative blue -> cyan -> ice; positive red -> yellow -> white | 0.06 |
| custom | the user's stops (default blue / dark violet / orange) | from its zero stop |

- **Zero level:** the brightness (max channel) of the zero stop. The slider rescales the zero stop and keeps its hue, so it works on every palette. Choosing a preset loads that preset's own zero level. With teal-red this is exactly the 30/09 formula `Z + (end - Z)|x|^(1/contrast)`: 1.7e-16 on the host. The web-app check (zero 0.05) still passes.
- **Contrast is the curve** (a gamma on |x|). Exposure, auto gain and colour gain apply on top, unchanged.
- **Editor** (look section -> *edit the custom palette*):
  - one colour picker per stop, and a position slider for each interior stop (typed values work);
  - *remove stop k*, *add a stop* on the negative or positive side (it starts at the midpoint of the widest gap, in the colour already shown there);
  - *copy the current palette into custom*, *reset the custom palette*.
  - Any edit switches the view to the custom palette. Picking the zero colour sets the zero level.
- **Page, solid and outside colours** are each separately editable. Solid and outside existed already. The page colour is new (*page colour (outside the array)*); Y (the background preset) clears it.

### Settings file

`look_settingsMIRROR.json`, in the project folder. It holds the palette name, the custom stops, the zero level, opacity, grid settings and the page colour.
- A live run loads it at start and saves it once a change has rested 0.5 s; a colour-picker drag does not write every frame. It also saves on close.
- A broken or missing file keeps the defaults; the panel shows why.
- `--shot` runs ignore it, so shots stay reproducible, unless `--settings` names a file. `--no-settings` turns it off for a live run.

### Transparency: real alpha, measured

GGUI 1.7.4 blends RGBA vertex colours (see *GGUI 1.7.4 facts*), so this is real blending. Screen-door and fake blends were not needed.
- **Surface opacity** (3D): the field cells' vertices carry alpha = opacity; solids and the floor stay opaque.
  - GGUI does not sort, so with opacity < 1 the triangles are drawn floor first, then domain triangles back to front. The order is the triangle centroids projected on the horizontal view direction, cached per 1/32 of a turn of azimuth. A new sector sorts in ~20 ms at mesh 513 (2 x 512^2 triangles).
  - The overlay lines (domain edges, beam emitter, brush ring) go before the surface, so they show through it.
  - Through a translucent crest you see the far side of the ring.
- **2D:** the same slider mixes the field toward the page colour. That is what the 3D surface shows over empty space. The measured pixels are identical in the two views.
- **Grid lines** (3D, *grid lines (opacity, 0 = off)*):
  - a second wireframe pass of degenerate triangles along both lattice index axes, a line every k mesh vertices (the panel shows it in cells), following the surface vertex by vertex;
  - colour: the picked grid colour, or the field's own colours brightened;
  - slid 0.2 % of their distance toward the eye, so they win the depth test against their own surface.
  - Each edge is drawn twice, so the per-pass alpha is `1 - sqrt(1 - g)`.
  - The grid is drawn after the surface. Drawn first, a faint line would blend over the page and show darker than the surface. So grid lines behind a crest stay hidden; the panel says so.
  - Only the vertices on grid lines are uploaded: +4 ms. A full-mesh copy cost +30 ms at mesh 513.
- **X (wireframe)** is unchanged: GGUI's own edges of the surface in the field colours, now with the surface alpha.

Measured (`app_tests_MIRROR.py` section 10, unlit top view, 8-bit):
- surface: pixel = a * surface + (1 - a) * page within 0.6 at a = 1, 0.5, 0, in 3D and 2D;
- grid: line pixel = g * colour + (1 - g) * surface, (got, wanted) = (116, 115.8), (175, 175.4), (255, 255) at g = 0.3, 0.6, 1;
- back-to-front order: correlation of draw rank with -(eye distance) 0.987-0.990 at 4 views;
- frame at N = 1025, mesh 513, including the ~50 ms readback: opaque 69.5 ms, opacity 0.5 67.4 ms, with the grid 73.3 ms (full run; the sector sort 19 ms).

**Limits:**
- Sorting is per triangle along the horizontal view, not per pixel, so two translucent sheets crossing within one triangle can blend in the wrong order. On a height field this is rare.
- Grid lines are 3D only, and they also cross raised solids.
- Line width is GGUI's (1 px).

**Shots:** `ScratchCLAUDE/VisualsCLAUDE/shots_visualsMIRROR.py` renders 24 variants, **24/24 OK**:
- every preset and two custom palettes, each in 2D and 3D (`sheet_palettes_MIRROR.png`);
- opacity 1, 0.6 and 0.3 in 2D and 3D, plus grid combinations and the plate at opacity 0.5 (`sheet_opacity_MIRROR.png`).

**Tests:** `app_tests_MIRROR.py --arch cuda` gives **88/88**: 71 before, plus section 10 (`ScratchCLAUDE/VisualsCLAUDE/app_tests_cuda_MIRROR.txt`).

## Scale mode (02/10/2026)

*Added by Claude Opus 5.5, 02/10/2026.* Full description: [scale_MIRROR.md](scale_MIRROR.md).

- **Scale button.** `scale: cells … -> domain` (main panel, above the speed slider) switches the size units without a reset. `Viewer.set_scale()` converts steps/frame ↔ crossings/s at `FPS_NOMINAL` = 60.
- **Domain-mode sliders.** `crossings / s` replaces `steps / frame`. `Viewer.frame_steps()` carries the fractional steps, and the main loop calls it. The fraction sliders `sigma / r_in`, `lambda / r_in`, `beam w / r_in`, `packet L / r_in` and `packet lambda / r_in` come from `Viewer.scale_size_slider()`.
- **Readout and warnings.** The status line shows the crossings, and amber notes come from `m.scale_warnings()`.
- **Auto height.** In the look section, off by default, both modes. `height_scale()` = hs / `auto_peak()`, the same peak as the auto colour gain; `colour_gain()` now calls `auto_peak()`, with no behaviour change.
- **CLI:** `--scale`, `--cps`, `--auto-height`. `--set` accepts the None-valued fraction fields (as floats).
- **Tests:** `app_tests_MIRROR.py` `t_scale`, 7 checks (95/95).


## Round 4 / v1.0 (05/10/2026)

*Added by Claude Opus 5.5, 05/10/2026.* The panel wiring of the round-4 core work (hand-offs: [walls_MIRROR.md](walls_MIRROR.md) section 11, [vertex_choppiness_MIRROR.md](vertex_choppiness_MIRROR.md) section 7.4) and the version. Progress log: [progress_round4_app_MIRROR.md](progress_round4_app_MIRROR.md).

### Version
- `VERSION = '1.0'`, `VERSION_DATE`, `APP_NAME` in `membrane_core_MIRROR.py`; the viewer imports them. Window title `ZRL wave membrane v1.0`; the main panel's first line; `--version`; the `--shot` log line (`wrote out.png: ZRL wave membrane v1.0, N=... bc <effective> edge <edge> ...`).

### Boundary, edge, `array` (main panel)
- **boundary:** the effective bc (`bc_info()`), plus `(<requested> n/a here)` on a fallback; the button `boundary [B]: <bc> (click: <next>)`; the ring in words (grey), e.g. *rigid ring (link removal, zero normal slope)*, *rhombic torus (wraps along both lattice vectors)*.
  - The triangular `array`: a grey line *mur: not on tri yet (absorbing-boundary round)*; amber, with *falls back to the clamped ring*, while mur is requested.
  - A triangular web-app shape: the button reads *does not apply to this shape*, and a grey note says to use the edge control (the `array` shape has the three bcs). B flashes the same note and changes nothing.
  - Changing the bc resets the run, as before.
- **edge:** `edge: <pinned|rigid> (click: <other>)` for masked shapes, with a grey line (*pinned: u = 0, echo inverted* / *rigid: zero slope, echo upright*). It calls `Membrane.set_edge()`: the link words are rebuilt and the **field is kept** (no reset; checked in app_tests). On `array` and the full square: grey text *edge: n/a - the array / the full square has no mask edge: the boundary (B) rules*. The same control sits in the walls panel.
- **`array`:** the radius slider and the rotation slider and button become grey text. In 2D the whole triangular rhombus (3c wide: 1.47 window heights at zoom 1) is zoomed to fit the strip the panels leave (`Viewer.fit_2d()`: on entering it, on C, and in `--shot`; leaving it restores zoom 1 unless the user zoomed since).
- **The 3D fit** (06/10/2026, `Viewer.fit_3d()`): the default camera frames the square ([-1, 1]^2 in world units); the rhombus is 3 wide, so its near corners fell off the window (seen in `r4shot_layout_max_3d`). On entering the triangular `array`, on C, and at start-up / in `--shot` without `--camera`, the orbit camera moves back along its azimuth and elevation until no rhombus corner lies further from the window centre (in x or y) than the square's corners do at the default distance. At the default angles that is distance 3.870 (the zoom slider reads 0.749); az 45: 3.794; el 70: 4.149 (`ScratchCLAUDE/Round4CLAUDE/AppCLAUDE/probe_fit3d_MIRROR.py`). Leaving the array restores the default distance unless the user zoomed since. A run started on the array (`--set lattice=tri shape=array`) now gets both fits at once.

### Wall types (walls panel, renderer)
- **brush wall**, the selected object's **wall**, the **plate lettering wall** (Unlicensed only): pinned <-> rigid buttons. An object edit re-rasterises its box; the plate toggle redoes the grid.
- Explainer: *pinned = held at zero, reflects inverted (a membrane post). rigid = zero slope, reflects upright (sound off a hard object, water off a quay).*
- Legend: *pinned solid (u = 0)* in the solid colour, *rigid solid (du/dn = 0), striped in 2D* in the rigid colour, *rigid edge: line; torus seam: dashed*. The legend uses coloured words: ImGui's default font has no box glyph.
- Counts: `walls_info()` (two whole-grid counts, ~0.1 s at N = 8193 per walls_MIRROR.md section 6) is cached on the masks, the scene, the edge and the bc. While painting, or during an object drag (two solid edits less than `EDIT_SETTLE` = 0.3 s apart), the last counts stay shown until the edits rest; a single click edit updates them at once.
- **Renderer:** `Look.rigid` = (0.22, 0.72, 1.00), a neon cyan-blue: CIE L* 71.0 against the pinned off-white's 85.4 (`app_render_MIRROR.lstar` of the two colours), and 52/255 apart by luma on the rendered 8-bit pixels (app_tests section 11), so the two differ in lightness too. `k_image` / `k_verts` read the core's class codes `m.C` (low two bits: 1 pinned, 2 rigid); a 1x1 stand-in with `hasC = 0` while the core keeps none (every solid pinned). In 2D rigid cells carry dark diagonal stripes (2 px of every 8 at 0.55). Measured in app_tests (8-bit, unlit): pinned (214, 214, 204), rigid (56, 183-184, 255), stripes on 45 of 169 pixels of a patch.
- **Edge overlay** (*draw the edge type*, on by default): a rigid edge (masked shape) and a rigid ring (the triangular `array` with neumann; the square's Neumann ring too) as a line in the rigid colour along `shape_poly()`; a torus seam (square periodic, the rhombic torus) as a dashed violet line (`ar.TORUS_COL`). Cached per shape / bc / edge.

### Vertex sources and the drive (main panel)
- **vertex inset** button: *vertex inset: wall (3 sigma from both walls)* / *vertex inset: radial (the web app's)*; it cycles `mc.VTX_INSETS` and resets. `wall` is the default.
- *seed pulses under the driven vertices (the web app)*: shown only while *drive the vertices* is on; bound to `p.vtx_drive_bump`; it resets.
- **Source note** (`m.source_note()['lines']`): under the inset button for vertex sources, under source x / y for point sources. Levels: warn = amber, dim = grey, ok / info = the normal colour. A clip line that is not a warning drops its long parenthesis (and reads *nothing* at exactly 0); the amber one keeps the core's full text with *try vertex inset: wall*. The old bottom line *pulse clipped at t = 0* is gone. The core caches the note; while painting or during an object drag the last note stays shown (its clearance search is O(clearance^2)); any other change recomputes at once.
- *not an exact eigenmode here: ...* (`m.mode_note()`) in amber under the source buttons for the drum-mode source.
- **Drive warning** (`m.drive_warnings()`): amber under the freq slider, where the domain-mode `scale_warnings()` print (the two never both show).
- **Phase lag** (`m.phase_lag()['lines']`): in the readouts while a gaussian is seeded (src `pulse`, or `vtx` with its bumps).

### Text width
- `TEXT_WRAP` 54 -> **52**: a long panel shows ImGui's scrollbar, and wrapped lines of 54 characters lost their last two characters under it (seen in the shots). The alignment panel keeps its wrap of 52.

### Tests and shots
- `app_tests_MIRROR.py` section 11 (`t_round4`, 14 checks): **109/109 on CUDA f64 and on Vulkan f32** (95 + 14; 06/10/2026, `ScratchCLAUDE/Round4CLAUDE/AppCLAUDE/app_app_tests_cuda_fit3d_MIRROR.log` and `..._vulkan_f32_fit3d_MIRROR.log`; 108/108 on 05/10 before the 3D-fit check). It covers the version (incl. `--version` and a `--shot` subprocess), B per lattice and shape, the edge (field kept), `array` (greyed sliders, the 2D fit, the 3D fit at three camera angles), wall types and the plate, the legend and counts (and their drag wait), the scene round trip with wall types, rigid vs pinned pixels in 2D and 3D, the edge overlay, the vertex / drive / phase-lag readouts, and the panel rectangles (no overlap and inside the window at 1600x1000, 2560x1377 and 1280x720) plus real GGUI frames with all four panels.
- `ScratchCLAUDE/Round4CLAUDE/AppCLAUDE/shots_r4_MIRROR.py`: 21 `--shot` variants, 21/21 OK on 06/10 (`r4shot_*_MIRROR.png`, log `shots_r4_log_MIRROR.txt`, contact sheet `contact_r4shots_MIRROR.png`), each looked at: rigid vs pinned objects (2D, 3D, 3D lit), a rigid and a pinned n6 edge (the echo upright vs inverted), the triangular torus and rigid ring (2D, 3D at the default, fitted camera; the 3D array with the main panel) and the Mur fallback, the source note with the radial and the wall inset, driven vertices, the drive warning, the rigid plate lettering, a rigid `--walls-test` stroke, and the layout at 1600x1000 and 2560x1377 (2D, 3D) with all four panels.
- `ScratchCLAUDE/Round4CLAUDE/AppCLAUDE/run_r4_MIRROR.py` runs section 11 alone (~15 s).

### Open
- ~~The 3D default camera shows the triangular `array` wider than the window~~ fixed 06/10/2026 (`fit_3d()`, above). As with the square (its near corners at x = 0.143 of the window at the default camera), the rhombus's near acute corner (x = 0.145) lies under the main panel while that panel is open: the orbit target is the domain centre, drawn at the window centre.
- With both side panels open, a full-radius shape (square, plate) is wider than the 800 px strip between them in 2D, and its sides go under the panels. That was so before v1.0 (zoom 1 fits the array's height); only the triangular `array` gets a fit.
