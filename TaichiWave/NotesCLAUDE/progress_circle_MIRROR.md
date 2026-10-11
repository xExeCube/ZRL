# Progress — circle fidelity research (area: circle)

*Claude Opus 5.5 — 01/10/2026*

Owns: `ResearchCLAUDE/CircleFidelityCLAUDE/` (new scripts `*_MIRROR.py`), `NotesCLAUDE/circle_fidelity_MIRROR.md`, this log.

- 01/10/2026 start. Found prior work already in `ResearchCLAUDE/CircleFidelityCLAUDE/` (findings + `code/`, numpy re-implementation of the stencils, spline-based eps_ang, Bessel reference, R = 60). Kept untouched; this round adds the angular-FFT measure on the REAL solver (membrane_core, CUDA) and the N/Co/sigma sweep.
- 01/10/2026 `circle_lib_MIRROR.py` built: real Membrane (CUDA f64), quintic-B-spline polar resampling on the GPU, rFFT in angle, energy split by m; Bessel-series reference (energy-norm m = 0 error). Verified: reference at t = 0 equals the Gaussian to 4e-16; metric floor on the seeded Gaussian 2.4e-10 (sq) / 3.6e-10 (tri); energy pi/2 conserved to 1e-4 in view. Smoke test N = 257, sigma 3, 2 trips: eps_ang after trip 1 sq5 8.9e-2, sq9 8.5e-3, tri 8.9e-3.
- 01/10/2026 `sweep_MIRROR.py` launched (72 runs: sq5/sq9/tri x N 257..2049 x Co 0.5/0.9/0.99 Cmax x sigma 3 / R/40, 10 trips, 4 samples per trip) -> `OutCLAUDE/sweep_MIRROR.json`, log `OutCLAUDE/sweep_log_MIRROR.txt`.
- 01/10/2026 sweep done through N = 1025 (log). Added `jitter_MIRROR.py` (8 radii at N = 513, staircase luck), `freespace_MIRROR.py` (pulse travels 2R with the wall out of reach: bulk anisotropy alone), `tables_MIRROR.py`, `figures_MIRROR.py`. Early reading: Co hardly changes eps_ang (m != 0) but strongly changes eps_0 (dispersion); sq5 at sigma 3 is 10-40x worse than sq9/tri.
- 01/10/2026 jitter done (`OutCLAUDE/jitter_MIRROR.json`): over 8 radii at N = 513, trip-1 eps_ang medians sigma 3: sq5 2.1e-1, sq9 9.2e-3, tri 6.9e-3; sigma 5.5: 1.3e-2, 1.3e-3, 1.2e-3. Spread 2-3x within a lattice.
- 02/10/2026 resumed. Found freespace already done (`OutCLAUDE/freespace_MIRROR.json`, N_small 513/1025) and the N = 2049 sweep done for sigma 3 (all Co) + sigma R/40 Co 0.5; relaunched `sweep_MIRROR.py --N 2049` for the last 6 runs (~50 s each; skips finished ones).
- 02/10/2026 sweep complete (72/72 runs incl. N = 2049, ~50 s each).
- 02/10/2026 tables (OutCLAUDE/tables_MIRROR.txt) + fig1-5 rendered (fig5 = staircase vs bulk split added; figures now at N = 2049).
- 02/10/2026 DONE: `NotesCLAUDE/circle_fidelity_MIRROR.md` written (method, modes m%4 / m%6 confirmed to rounding, ranking, N/Co/sigma scaling, growth, staircase-vs-bulk split, CE score proposal, beyond-lattice discussion, reconciliation with the earlier findings file). Open: sq9 + fitted edge not built; effective-radius explanation of eps_0's post-wall scaling not verified; off-centre CE variant not built.
