# Progress log -- alignment panel port (area: alignment)

Owner files: alignment_MIRROR.py, crosscheck_alignment_MIRROR.py, ToolsCLAUDE/js_align_dump_MIRROR.mjs,
NotesCLAUDE/alignment_MIRROR.md, ScratchCLAUDE/align* scratch.

## 30/09/2026 -- resumed (second interruption recovery)
- State found: port exists; `--quick --reuse` = 27882/27882 over 612 configs (lead, 30/09, after the
  core promotion: ScratchCLAUDE/DevCLAUDE/align_quick_after_promote.txt). No notes, no progress log.
- TO DO: (1) full sweep + extended angle/rad/rhombus sweep; (2) 'unlicensed' vs the core's real
  shape; (3) gui_section in a real offscreen GGUI window + timings N=1025/4097; (4) notes.

## 30/09/2026 -- milestone 1: FULL sweep (no --quick) DONE
- `python crosscheck_alignment_MIRROR.py` (CPU f64): TOTAL 80055/80055 over 2295 configs (+ pure sweep);
  JS dump 265 s, port 73 s. Log: ScratchCLAUDE/align_full_run_MIRROR.txt.
- js_num_str rewritten to the ECMA Number::toString layout (Python repr writes 1e-05 / 1e-07 differently);
  3022/3022 vs Node String(x) (ScratchCLAUDE/AlignCLAUDE/align_numstr_MIRROR.py).
- Added `--ext` to crosscheck_alignment_MIRROR.py (near-aligned +-0.05..0.15 deg, clipping-threshold radii
  +-1e-9..2e-2 found by bisection, rhombus theta 20..90). Next: run it.
- 'unlicensed' probe on the REAL core (ScratchCLAUDE/AlignCLAUDE/align_unl_probe_MIRROR.py): sides
  1.87R x 0.87R as expected, all ledgers PASS/n-a, no DISAGREE. Arcs-only clipping (all sides gone,
  arc pieces left): 0 cases, N=161, rot 0..90 step 0.5, rad 1..4 step 0.01, both lattices.

## 30/09/2026 -- milestones 2-4 DONE (all four TO DO items)
- --ext sweep: 82380/82380 over 2370 configs (12 first-run mismatches = sliver edges 1.1e-6 cells at the
  nothing-left radius - 1e-9, last-bit cos/sin conditioning; now compared at 1e-13*N/len, reported).
- --unl --big: 51266/51266 invariant checks over 816 configs on the core's REAL plate (+3 cache checks).
  NOTE for 'unlicensed' gained sizes + direction-rule lines.
- Scan cache now keyed on geometry + mask CONTENT (set_co no longer rescans; stale masks never survive).
- GGUI: real offscreen window, every button clicked; show() segfaults offscreen -> get_image_buffer_as_numpy.
  Panel 0.37-0.58 ms/frame cached; first frame 0.1-0.2 s (N=1025), 0.5-0.6 s (N=4097).
- Final re-runs after all edits: full --reuse 80055/80055, --ext --reuse 82380/82380, --unl 44143/44143,
  --quick --reuse --arch cuda 27882/27882.
- NotesCLAUDE/alignment_MIRROR.md written (API, verification, deviations, JS quirks, GGUI timings).
- Next (optional): N=8193 timing; a Taichi kernel for the scans if 8193 matters.

## 01/10/2026 -- panel wrap (Claude Opus 5.5)
- FIXED: AlignPanel._lines printed the '[PASS]/[n/a] ...' ledger claims and the 56-char 'straightness, per edge
  (...)' header unwrapped; both now go through textwrap.wrap(..., self.wrap) (claims indent 2 on continuation).
  AlignPanel default wrap 64 -> 52 (the app already overrides to TEXT_WRAP - 2 = 52; 64 x 7 px overflowed 400 px).
- VERIFIED in a real offscreen GGUI window (400 px sub_window, 8 shapes incl. 3 unlicensed, wrap 52 and default):
  ScratchCLAUDE/AlignCLAUDE/align_wrap_ggui_MIRROR.py -> longest line 52 chars <= 54 (= (400 - 16) / 7), PASS.
  Before the header fix: 10 lines over (all the 56-char header); the claims were already split by the fix.
- crosscheck_alignment_MIRROR.py --quick --reuse after the edit: TOTAL 27882/27882 over 612 configs (ScratchCLAUDE/AlignCLAUDE/align_quick_0110_MIRROR.txt). Notes: alignment_MIRROR.md GGUI paragraph.
