# Progress log: ledger port (area: ledger)

Owner files: ledger_MIRROR.py, crosscheck_ledger_MIRROR.py, ToolsCLAUDE/js_ledger_dump_MIRROR.mjs,
NotesCLAUDE/ledger_MIRROR.md, ScratchCLAUDE/LedgerCLAUDE/. Read-only: everything else.

## 30/09/2026 (resumed run; the 29/09 cut left NO ledger files on disk -- started from scratch)
- Read: web app ledger (lines 1687-2487: fmt/fe/st/NA/DSG, symmetryResidual, mirrorSymmetric,
  mirrorResidual, symRotResidual, frontFree, frontRadii, srcCentred, coneWall, coneMeasure, energyStats,
  buildLedger) and its helpers (coLimit, reachInradius/reachName/coneNorm, stencilSymbol, kMax/HARM,
  domR/innerR/atConfRot, shapeArea/idealArea/circleClipArea/clipRegion/fitsInside/polyArea,
  clearanceFrom/CLEAR, modeBox/modeMN/modeAntinode, record()). sourcePPW (3076) is NOT used by the
  ledger (ppwRender only); no space-time rows are in buildLedger (the light-cone row 7d is).
- The JS writes the rows to $('led').innerHTML (<tr><td>label<td class=s>status<td>detail<td class=r>tier).
- Plan: ledger_MIRROR.py (rows as NamedTuple (name, status, value, note, tag, + key/badge/dsg/num/port)),
  js_ledger_dump_MIRROR.mjs (checkpoints per config, parse led.innerHTML), crosscheck (>= 30 configs,
  statuses exact, numbers within a display unit / noise floor), GUI, notes.
- Next: write ledger_MIRROR.py.
- DONE: ledger_MIRROR.py v1 (all rows; solids / unlicensed / f32 adaptations; FIXES with compat flag;
  GUI panel; CLI `python ledger_MIRROR.py --arch cpu --steps 300 N=161 src=impulse`). Smoke run OK (cpu).
- DONE: ToolsCLAUDE/js_ledger_dump_MIRROR.mjs (checkpoints per config, drops, parses #led rows, dumps
  CLEAR/CONE_WALL/SRC/amax). Tested on one config.
- DONE: crosscheck_ledger_MIRROR.py (49 configs x checkpoints 0/50/300/1200). First full run (cpu) started.
- Next: triage the cross-check diffs; performance at N = 1025..8193; GUI headless test; notes.
- VERIFIED 30/09 (~18:40): crosscheck_ledger 50 configs / 197 ledgers / 3435 rows, web-app dump saved as
  ScratchCLAUDE/LedgerCLAUDE/ledger_js_full_MIRROR.jsonl (reuse: --jsonl <it>):
  cpu f64 PASS 100% status, 100% text (the unstable run from its own seed: status only; its _js_seed twin exact);
  cuda f64 PASS 100/100; cuda f32 PASS (12 rows exempt: web-app amax > 3.4e38 at step 300; 22 f32 text diffs).
  FIXES (compat=False) change 144 rows, all attributed: mirror-live 24, sqfull-side 83, lens-text 8,
  edge-kind 22 (3 false FAILs of the web app: masked n6 + Mur bc@1200, tri n6 + Mur bc@1200,
  n12 + Mur cont linearity@1200), cfl-nan 2, tri-diagonal 5. Largest f32 residuals: sym8 4e-7, mirror 2e-6,
  sym6 7e-7, disp 1.4e-8 (TOL_F32 sym 1e-4, disp 1e-5).
- Next: performance vs N; vulkan f32 run; port-only checks (solids, unlicensed); notes.
- (found on disk, not logged before the 2nd cut) 30/09 ~18:06-18:10: device reductions patched into
  ledger_MIRROR.py (k_orbit / k_cone / k_modecorr / k_gather, DEVICE flag; ScratchCLAUDE/LedgerCLAUDE/
  patch_device*_MIRROR.py), test ScratchCLAUDE/LedgerCLAUDE/dev_vs_np_MIRROR.py. NOT yet verified.

## 30/09/2026 (third run, resumed ~23:00 after the 2nd usage-limit cut)
- Plan: verify the device path (dev_vs_np cuda f64/f32, cpu, vulkan); re-run the cross-check from the saved
  jsonl on the device path; perf vs N; vulkan f32 cross-check; port-only checks (solids, unlicensed);
  GUI headless test; finish notes.
- VERIFIED (third run): device path == numpy references, 0 differing rows (dev_vs_np: 14 configs x N 161/257
  x steps 0/40/200/700) on cuda f64, cuda f32, cpu f64, vulkan f32. Added np.errstate to the numpy references.
- VERIFIED: crosscheck from the saved jsonl on the DEVICE path: cuda f64 / cpu f64 / cuda f32 / vulkan f32 all
  RESULT PASS, 3435/3435 status and text (f32: 12 rows exempt beyond f32 range; vulkan 21 informational f32 text
  diffs). Logs ScratchCLAUDE/LedgerCLAUDE/xc6..xc9_*_MIRROR.txt. Vulkan f32 residuals: sym8 5.2e-7, mirror 3.7e-6,
  sym6 7.8e-7, disp 1.75e-8 (TOL_F32 1e-4 / 1e-5).
- NEW: recorder-gap rule: history rows (energy, bc, wake, disp, linear) are n/a when len(hist) != one entry per
  step (m.record was off for some steps); replaces the old "recorder is off" remarks.
- Port-only checks drafted (ScratchCLAUDE/LedgerCLAUDE/port_checks_draft_MIRROR.py): 14/15 before the gap rule.
- DONE + VERIFIED: port-only checks moved into crosscheck_ledger_MIRROR.py (--port-only / --no-port; run by
  default after the comparison): 15/15 on cuda f64, cuda f32, vulkan f32, cpu f64.
- VERIFIED: full cross-check after the gap rule: cuda f64 (saved jsonl) and cpu f64 with a FRESH Node dump
  (143 s): 3435/3435 status + text, port-only 15/15, RESULT PASS (xc10, xc11 logs).
- Next: performance vs N (perf_MIRROR.py), GUI headless test, notes.
- MEASURED perf (cuda f64, perf_MIRROR.py): WARM rebuild 0.8-6.1 ms at every N 161..8193 (device reductions);
  COLD first build (geometry cache) <= 131 ms at 2049, 0.1-0.46 s at 4097, 0.29-1.87 s at 8193, dominated by
  al_ledger_rows (0.7-1.45 s at 8193) and the boundary-cell scan (~250 ms). Membrane setup n6 vtx 8193: 19.9 s (core).
- DONE: LedgerPanel: auto default follows N (<= 2049; AUTO_MAX_N) until the user touches it; above it the
  ledger is built on demand (hint line); interval floor = 20x the last WARM build; cold builds flagged;
  'reset run' button (viewer.reset()).
- VERIFIED GUI headless (gui_test_MIRROR.py, hidden window): N=1025 115 frames / 6 s, 9 builds (~1.5/s),
  panel median 0.36 ms/frame; N=8193 no auto build, on-demand cold build 1.2 s; all text Latin-1.
- Next: notes (NotesCLAUDE/ledger_MIRROR.md), final re-run, result.
- DONE: NotesCLAUDE/ledger_MIRROR.md complete (API, FIXES = JS bugs, port-only decisions row by row, device
  reductions, perf table, GUI policy, verification table). Module docstring: recorder-gap rule.
- VERIFIED final: cuda f64 crosscheck (saved jsonl) 3435/3435 + port-only 15/15, RESULT PASS (xc12 log).
- STATUS: task complete. Open (not this area): cold build dominated by al_ledger_rows at N >= 4097; core
  Membrane setup n6 vtx N=8193 ~19 s.
