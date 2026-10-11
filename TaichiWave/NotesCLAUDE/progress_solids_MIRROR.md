# Solids progress log

## 01/10/2026 review v3 (Claude Opus 5.5)
Findings with evidence: ScratchCLAUDE/ReviewCLAUDE/findings_solids_v3_MIRROR.md.

- FIXED: one long painted segment re-rasterised its whole bounding box (1.3-2.4 s at N = 8193 for a 4000-cell diagonal over the plate). solids_MIRROR.py: incremental rasterise after begin/extend_stroke (old mask + new capsule, bit-exact) and tile-culled capsules (capsule_tiles, TILE = 64). Now 31-56 ms. Test: solids_tests section 12 (bit-equality vs rasterize_full on sq/tri, plate/n6/circle, draw + eraser; < 250 ms timing).
- PASS: sealing (closed loops 0/203 leaks over two seeds; straight walls ending on the visible domain edge 0/176), text size/orientation on sq and tri, font fallback, anchors, JSON round trip, sources never in a solid.
- NOT VERIFIABLE: plate vs the cover, the cover image file is gone.
- Suites: solids_tests cuda ALL PASS (55 s), vulkan --f32 --quick ALL PASS (45 s), crosscheck_js --arch cpu 27/27.
- Backup of the pre-fix module: ScratchCLAUDE/SolidsCLAUDE/solids_before_v3_MIRROR.py.

## 01/10/2026 -- plate vs cover (user's cover ScratchCLAUDE/GovernmentPlates.jpg, 300x300)
- Measured with ScratchCLAUDE/SolidsCLAUDE/plate_cover_MIRROR.py: homography from the inner light area's
  4 corners (px TL 49,104.7 TR 250,100.6 BR 250,199.2 BL 49,194.7) to (+-1, +-0.5) R; 1 cover px ~ 0.01 R.
  Corner jitter +-0.5 px (--sens): SD <= 0.004 R on every number. Letter edges are crisp (emboss shade is
  inside the strokes), so the ink boxes are the ink.
- Cover (rectified) vs preset BEFORE: CA EXEMPT cx -0.0075/0, cy 0.3375/0.320, h 0.145/0.14, w 1.105/1.11;
  DEATH cx -0.0425/0, cy -0.0875/-0.0875, h 0.545/0.525, w 1.595/1.57, letters 0.235 x5 vs
  0.20/0.185/0.25/0.21/0.20, gaps .13/.105/.075/.10 vs .165/.125/.105/.13, stem .065 vs .050;
  screws |x| 0.642/0.628, |y| 0.476/0.4375; corner r 0.047+-0.004 vs 0.065 (core's, < 1 px on the diagonal).
- DEATH sweep (sweep_death_MIRROR.py): SemiBold SemiCondensed tracking 0.20 gives letters .225/.21/.28/.23/.225,
  gaps .14/.10/.08/.105, stem .060 -- best one-knob match (Bold variants: stem .07-.085, bar too thick).
- DONE: preset changed in solids_MIRROR.py (PLATE_TEXT/PLATE_SCREWS, dated comment): CA EXEMPT y 0.319->0.3375;
  DEATH x 0->-0.0425, height 0.525->0.545, fit_w 1.574->1.595, tracking 0.30->0.20; screws y 0.4375->0.476.
  After: mask boxes = cover boxes to 0.005 R (CA cy .3375 h .145 w 1.11; DEATH cx -.0425 h .545 w 1.595;
  screws (+-.6286, +-.4759)). Side-by-side: ScratchCLAUDE/SolidsCLAUDE/plate_vs_cover_MIRROR.png
  (before: plate_vs_cover_before_MIRROR.png). Kept: corner r (core), screw x, DEATH y, CA x/h/w.
- Notes: NotesCLAUDE/solids_MIRROR.md preset table + 'Plate vs cover, 01/10/2026' section updated.
- solids_tests_MIRROR.py --arch cuda after the preset change: ALL PASS (33 checks, 51.0 s), no expected values needed changing. Log: ScratchCLAUDE/SolidsCLAUDE/tests_cuda_0110_MIRROR.txt
