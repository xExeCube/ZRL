# Solids progress log

## 01/10/2026 review v3 (Claude Opus 5.5)
Findings with evidence: ScratchCLAUDE/ReviewCLAUDE/findings_solids_v3_MIRROR.md.

- FIXED: one long painted segment re-rasterised its whole bounding box (1.3-2.4 s at N = 8193 for a 4000-cell diagonal over the plate). solids_MIRROR.py: incremental rasterise after begin/extend_stroke (old mask + new capsule, bit-exact) and tile-culled capsules (capsule_tiles, TILE = 64). Now 31-56 ms. Test: solids_tests section 12 (bit-equality vs rasterize_full on sq/tri, plate/n6/circle, draw + eraser; < 250 ms timing).
- PASS: sealing (closed loops 0/203 leaks over two seeds; straight walls ending on the visible domain edge 0/176), text size/orientation on sq and tri, font fallback, anchors, JSON round trip, sources never in a solid.
- NOT VERIFIABLE: plate vs the cover, the cover image file is gone.
- Suites: solids_tests cuda ALL PASS (55 s), vulkan --f32 --quick ALL PASS (45 s), crosscheck_js --arch cpu 27/27.
- Backup of the pre-fix module: ScratchCLAUDE/SolidsCLAUDE/solids_before_v3_MIRROR.py.
