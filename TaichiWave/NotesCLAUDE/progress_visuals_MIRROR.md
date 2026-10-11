# Progress: app visuals (palette editor, mesh transparency) — 01/10/2026

*Claude Opus 5.5, 01/10/2026*

- [ ] read render/app code
- [ ] GGUI alpha probe
- [ ] palette module + editor + persistence
- [ ] transparency control
- [ ] tests + shots
- [ ] notes

## 01/10/2026 14:10 — GGUI 1.7.4 alpha measured (ScratchCLAUDE/VisualsCLAUDE/*probe*MIRROR.py)
- scene.mesh accepts a vec4 per_vertex_color and BLENDS (src-alpha over): green A=0.5 over red -> (0.50, 0.50, 0). Depth is still written, so draw order matters (front-first hides the back quad: green A=0.5 over the blue page instead).
- show_wireframe=True draws ONLY the triangle edges (no fill), in the vertex colours, alpha-blended. Degenerate triangles (a,b,b) draw the edge a-b (412 px for 2 edges vs 618 for a triangle) -> a grid of lines with real alpha.
- scene.particles blends; scene.lines, canvas.lines and canvas.triangles IGNORE alpha (opaque).
- Draw order across types is call order: a scene.lines call made before a translucent mesh shows through it.
- Point light keeps the vertex alpha.
=> real alpha exists: surface opacity + grid-line opacity are real blending (no screen-door needed).

## 01/10/2026 ~15:00 — implemented (app_render, membrane_app, app_tests)
- Palettes: stops model (KMAX 17) in app_render: teal-red (default, == 30/09 formula to 1.7e-16), greyscale, cool-warm (matplotlib coolwarm 9 pts), cyclic (twilight 17 pts), hot-cold, custom. Zero level = max channel of the zero stop. Kernel loops over the stops (LP block grown to 100).
- Settings: look_settingsMIRROR.json (live runs load + auto-save after 0.5 s rest; shots ignore unless --settings).
- Transparency: RGBA vertex colours (field cells only; solids/floor opaque), back-to-front order per 1/32 azimuth sector (sort 19-22 ms at N=1025 mesh 513), overlay lines drawn first when translucent. Grid: degenerate-triangle wireframe pass, lifted 0.2% toward the eye, alpha compensated for the double-drawn edge (1 - sqrt(1 - g)), compact vertex gather (+4 ms; the full-mesh copy cost +30 ms).
- Verified: run_visuals_onlyMIRROR.py 18/18 (sections 1 + 10); shots_visualsMIRROR.py 24/24 (sheets in ScratchCLAUDE/VisualsCLAUDE).

## 01/10/2026 ~15:30 — done
- Full app_tests_MIRROR.py --arch cuda: 88/88 in 70 s (ScratchCLAUDE/VisualsCLAUDE/app_tests_cuda_MIRROR.txt); 71 earlier checks all still pass.
- Regression shots (segment, wire, all panels, 2D tri walls-test, plate + opacity + grid + panel): 5/5 render (reg_*_MIRROR.png).
- Notes: NotesCLAUDE/app_MIRROR.md section "Palettes and transparency", --shot table, GGUI alpha facts.
- Not done / open: README_MIRROR.md not touched (not owned); grid lines are 3D only and cross solids; live-window behaviour of the colour pickers is untested (headless only).
