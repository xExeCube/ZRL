# Progress: round 5, the user's v1.0 decisions (09/10/2026)

*Generated from scratch -- Claude Opus 5.5 -- 09/10/2026.* Step "Decisions" of the round-5 workflow. Backups: `ScratchCLAUDE/Round5CLAUDE/DecisionsCLAUDE/*_before_MIRROR.*`; logs: `ScratchCLAUDE/Round5CLAUDE/LogsCLAUDE/dec_*_MIRROR.log`.

The three items: (1) the sq9 Neumann clipped-shape instability fixed in the port; (2) default freq 0.03125 (16-cell wavelength at Co 0.5); (3) soft driven vertices by default, with a hard toggle.

## 09/10/2026 -- start
- Read README, CHANGELOG, walls_MIRROR.md sections 2-3, 8-10, 13-14, the core (words, step, operator, sources), the harness, the cross-check configs.
- No web-app comparison config hits item 1 (sq9 + Neumann + a clipped masked shape): crosscheck_js and crosscheck_ledger have none; alignment does not step.
- Next: item 1 in the core.

## 09/10/2026 -- core edits (items 1-3), not yet regression-tested
- Item 1 (sq9 Neumann clip): `_nring_common` returns 3 for an exterior X ring cell with a pinned edge; `_word` turns it into a pinned block (r 1) except for a link into an array corner (the cell's own ghost: a self-read, never asymmetric). `walls_needed()` also holds for `_neu_clip()` (sq9, bc neumann, masked shape that reaches the ring; `_dom_on_ring` set in build_domain). Probe `ScratchCLAUDE/Round5CLAUDE/DecisionsCLAUDE/dec_neu_enum_MIRROR.py --quick --N 41 --eig` (CUDA f64, 172 configs reaching the ring): v1.0 66 asymmetric (all pinned edge; max|L - L^T| 0.167, max |Im eig| 2.2e-3); after the fix 0 asymmetric, lam_max(-L) <= 0.999188 x 16/3.
- Item 2: `Params.freq = 0.03125`.
- Item 3: `Params.vtx_soft = True`; `lattice_g00(kind, E)` (1D reduction + Gauss-Legendre with a cosine substitution; 0.42 ms per call; agrees with scipy quad to <= 1.5e-9, `dec_g00_check_MIRROR.py`), `soft_gain(kind, co2, omega) = co2 / |g00|` (Co 0.5, lambda 16: sq5 0.4987, sq9 0.4690, tri 0.5446); `Membrane.vtx_soft_on()`, `soft_tables()`, `soft_drive()`; step() runs soft vertices on the SRC_BEAM kernel path (U += per-cell drive), no new kernel variant; `source_note()` head "driven soft / driven hard" and key `drive`.
- Next: smoke + amplitude-ratio probe, then the comparisons' explicit freq=0.12 / vtx_soft=False, then decisions_tests_MIRROR.py.

## 09/10/2026 -- tests and comparison configs
- New suite `decisions_tests_MIRROR.py` (sections 1-3; loads the v1.0 snapshot as a second module). `--quick` ALL PASS on CUDA f64 (216 s, after two test fixes: E0 after 2 steps -- the seed's tail on the ring cells gave every sq9 config, the full square too, ~1e-9 from step 0, `dec_energy_co_probe_MIRROR.py`; and freq set for both cores in 3b) and on Vulkan f32 (252 s, `ScratchCLAUDE/Round5CLAUDE/DecisionsCLAUDE/dev_decisions_vulkan_quick_MIRROR.log`).
- Key numbers (quick, CUDA f64): 1a 136 configs, v1.0 52 asymmetric (all pinned edge), now 0; the 84 v1.0-symmetric operators unchanged entry for entry. 1b spectral radius 1.000000000 (rigid edge 1 + 2.5e-8, the zero mode); v1.0 up to 1.000732. 1c energy 1e-15 (v1.0 2.4e-3 in 4000 steps). 1d bounded (max|u| <= 0.49 over 40,000 steps). 1e 13/13 unaffected bit-identical. 3c soft at zero drive: 0.0 difference; hard 0.28-0.39 of the peak. 3d free space soft/hard 1.0001-1.0003 (phase lag 19.3 / 17.5 / 17.5 deg = arg g00 - omega); at the defaults 0.83 / 0.85 / 0.79 (3 lambda from a vertex), 0.82 / 0.84 / 0.87 (centre).
- Core: `WEBAPP_FREQ = 0.12`; module docstring deviations (neumann-clip, default freq, soft vertices).
- Harness: PATCH `neumann-clip` (the dg sum of the cells next to the ring, Neumann, masked shapes). crosscheck_js: base `Params(freq=WEBAPP_FREQ, vtx_soft=False)`, new config `sq9_circle13_neu_clip`. crosscheck_ledger: `web_base()` the same. crosscheck_alignment: the panel Membrane gets freq=WEBAPP_FREQ. make_ref (scale): freq=0.12, vtx_soft=False (npz unchanged). defaults_tests: section 2 hard (vtx_soft=False), section 4 freq 0.12 explicit + 2 new cases. tri_modes_tests t8: freq 0.12 + vtx_soft False for the current core.
- Next: run each comparison, then the full regression batch, then docs.

## 09/10/2026 -- regression batch running (`ScratchCLAUDE/Round5CLAUDE/DecisionsCLAUDE/dec_regression_batch_MIRROR.sh`)
- Docs written before the batch finished: `NotesCLAUDE/v1_decisions_MIRROR.md`, walls_MIRROR.md §14.2 (+ §2.3, §8 item 4, §10 item 8), vertex_choppiness_MIRROR.md §7 (status line, 7.3, 7.5 item 2), README (files table, Deviations 8-10, backlog, small open items), CHANGELOG "Unreleased (v1.1)". Backups in DecisionsCLAUDE.
- First results: decisions cuda f64 (full, --long) ALL PASS 377 s: 1a 3400 configs (N 41 + 97), v1.0 1302 asymmetric (all pinned edge), now 0; 2098 v1.0-symmetric operators unchanged. 1c 20,000 steps 0.9-2.6e-15 (v1.0 2.4e-3 to 1.4e-2). 1d 200,000 steps max|u| <= 0.37. 3d lambda 8 too: 1.0001 / 1.0001 / 1.0012. 3e 20,000 steps 0.5-1.6e-14. decisions cuda f32 ALL PASS 490 s, vulkan f32 ALL PASS 344 s (1c 0.4-1.6e-6).
- crosscheck_js cpu 28/28; ZRL_JS_NOPATCH=1 24/28 (sq5_full_neu_impulse, sq9_full_neu_corner, sq5_n8_vtxdrive_saw, sq9_circle13_neu_clip: du(300) 2.5e-2 unpatched).
