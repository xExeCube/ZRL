# The v1.0 decisions (round 5, 09/10/2026)

*Generated from scratch — Claude Opus 5.5 — 09/10/2026.* On 09/10/2026 the user decided three items left open at v1.0. This note says what changed in the core, why, what was measured, and what the app agent still has to wire. Tests: `decisions_tests_MIRROR.py` (new). Probes and backups: `ScratchCLAUDE/Round5CLAUDE/DecisionsCLAUDE/`. Logs: `ScratchCLAUDE/Round5CLAUDE/LogsCLAUDE/dec_*_MIRROR.log`, with `dec_batch_status_MIRROR.txt`. Progress log: `NotesCLAUDE/progress_round5_decisions_MIRROR.md`.

| # | the user's decision | in the port |
|---|---|---|
| 1 | "fix it in the port as a documented difference" | the sq9 Neumann ring with a clipped shape is symmetric again; harness PATCH `neumann-clip` |
| 2 | yes to a 16-cell default wavelength | `Params.freq = 0.03125` (the web app's 0.12 is `mc.WEBAPP_FREQ`) |
| 3 | "Change to soft as default, leave a hard toggle" | `Params.vtx_soft = True`; `vtx_soft=False` is the web app's hard drive, bit for bit |

---

## 1. The sq9 Neumann ring with a shape clipped by the array

**The fault** (inherited from the web app; [walls_MIRROR.md](walls_MIRROR.md) §14.2). With the 9-point stencil, the Neumann bc and a masked shape clipped by the array (a circle at radius 1.3), the ring has two kinds of cell where the shape leaves it: an exterior cell (0, t) that holds 0, and a ghost (0, t+1) that holds u(1, t+1). Cell (1, t) reads the ghost through its diagonal (weight 1/6). Cell (1, t+1) reads the 0 at (0, t). So L[a, b] = 5/6 but L[b, a] = 2/3. An asymmetric operator has complex eigenvalues, and the leapfrog grows at every Co: max|u| 6.7e21 after 200,000 steps (the review's `rv_neu_clip_growth_MIRROR.py`).

**The rule.** It extends the round-4 review fix C1 (walls_MIRROR.md §14.1) to the exterior with a pinned edge:
- A sq9 diagonal from an interior cell into a ghost ring cell is blocked when its common neighbour ON the ring is not a ghost. For the exterior X with a pinned edge it is now pinned-blocked (it reads 0), like a pinned solid on the ring row. Before, only P, R and X with a rigid edge blocked.
- Not for a link into an array corner. From (1, 1) the corner (0, 0) is the cell's own ghost: a self-read, which is symmetric anyway. So no configuration that was symmetric before changes.
- Code: `_nring_common` returns 3 for X with a pinned edge; `_word` turns 3 into a pinned block (or nothing for a corner target). `walls_needed()` is also true for `_neu_clip()`: sq9, `bc = 'neumann'`, a masked shape that reaches the ring (`_dom_on_ring`, set in `build_domain`). So these configurations run the walls kernel. Where nothing is blocked, the words stay idle and the web app's path runs bit for bit.
- `walls_info()['neu_clip']` says when the rule applies (for the UI).

**The web app comparison.** Harness **PATCH `neumann-clip`** (`ToolsCLAUDE/js_harness_MIRROR.mjs`) applies the same rule to the web app's stencil. It touches only the diagonal sum of the cells next to the ring, with the Neumann bc and a masked domain, in the same summation order. New `crosscheck_js` config `sq9_circle13_neu_clip` (circle r 1.3, N 161, pulse at (0.8, 0.5)). It agrees to 1.5e-13 of max|u| on CPU, not bit for bit. The reason: the port's walls path always uses the self-consistent 9-point form (walls_MIRROR.md §3.2), and the web app's order differs in the last bits. No other `crosscheck_js` or `crosscheck_ledger` config reaches this case; `crosscheck_alignment` does not step.

**Measured** (`decisions_tests_MIRROR.py` section 1, CUDA f64 unless noted; the full run's numbers are in the table at the end):
- **1a, symmetry.** Enumerated: **3400 configurations** reaching the ring (9 shapes clipped at radii 1.02–2.0 and rotations 0–45°, both edges, solids none / pinned / rigid / mixed, at N 41 and 97). Every operator is now exactly symmetric (max|L − Lᵀ| = 0). The v1.0 core had **1302 asymmetric**, all with the pinned edge (max|L − Lᵀ| = 1/6). The **2098** that v1.0 had symmetric keep their operator entry for entry.
- **1b, the step map's spectral radius** (dense, N 25, Co 0.5 and 0.865): 1.000000000 with the pinned edge. With the rigid edge it is 1 + 2.5e-8: that domain is fully Neumann, so its zero mode is a double root, as before. v1.0: up to 1.000732.
- **1c, exact leapfrog energy over 20,000 steps** (N 61, Co 0.8–0.865, 5 configs): 0.9–2.6e-15 (CUDA f64), ≤ 1.6e-6 (CUDA f32, Vulkan f32). v1.0: 2.4e-3 to 1.4e-2.
  - E0 is taken after 2 steps. The seed's gaussian tail sits on the ring cells, which hold a ghost (or 0) only from step 1 (inherited). Measured from step 0, every sq9 configuration here, the unaffected full square too, drifts ~1e-9 (`dec_energy_co_probe_MIRROR.py`).
- **1d, the review's reproducer** (N 41, Co 0.865, circle r 1.3 and octagon r 1.25, 200,000 steps, CUDA f64): max|u| stays ≤ 0.37 (from 1.0 at the seed; 40,000 steps in f32: ≤ 0.49). v1.0: 6.7e21.
- **1e, unaffected configurations bit-identical to v1.0** (13 configs, 400 steps: sq9 Neumann full square, inside shapes, Dirichlet / Mur / periodic clipped shapes, sq5 Neumann clipped, the rigid edge with and without solids, ring solids, tri). The affected circle runs the walls kernel (8 blocked cells at N 161) and differs by 1.5e-2 of max|u| after 400 steps.
- **CFL:** the largest eigenvalue of −L over the enumeration stays below the plain sq9 bound: lam_max ≤ 0.999188 × 16/3 (`dec_neu_enum_MIRROR.py --eig`, N 41). So C_max = 0.866 holds.
- **Cost:** these configurations now run the walls kernel: +3–8% per step on CUDA f64 (walls_MIRROR.md §7.2).

## 2. The default drive frequency: 16 cells per wavelength

- **`Params.freq = 0.03125`** cycles per step (was 0.12). λ = Co / freq, so this is a 16-cell wavelength at the default Co 0.5. The web app's 0.12 is 4.2 cells: 95–99.9% of the energy then sits in grid-scale waves (vertex_choppiness_MIRROR.md §4.7).
- **It is a plain default value, not coupled to Co.** At Co 0.7 the same freq is a 22.4-cell wavelength. Domain mode is the way to fix the wavelength itself: `lam_f` is taken from the cell values at the first apply, so at the defaults it is 16 / r_in at the starting N (test 2d).
- **Band limit:** `harmonics()` keeps k ≤ floor(Co / (8 freq)), which is 2 at the default (a square or saw wave gets harmonics 1 and 2).
- **No drive warning at the defaults** (cont, slit, soft and hard driven vertices); at freq 0.12 it comes back (test 2a).
- **The comparisons set the web app's value explicitly**, `mc.WEBAPP_FREQ = 0.12`:
  - `crosscheck_js`: the base `Params(freq=WEBAPP_FREQ, vtx_soft=False)`, so every config is v1.0's;
  - `crosscheck_ledger`: `web_base()` the same, for the configs and for the saved dump `ScratchCLAUDE/LedgerCLAUDE/ledger_js_full_MIRROR.jsonl`;
  - `crosscheck_alignment`: the panel Membrane (its own configs already set 0.12);
  - the scale tests' references: `ScratchCLAUDE/ScaleCLAUDE/make_ref_MIRROR.py` sets `freq=0.12, vtx_soft=False` (the npz files are unchanged);
  - `defaults_tests` section 4 (the warning cases named "freq 0.12");
  - `tri_modes_tests` test 8 (the parity with the 04/10 core).
  - The other suites set freq wherever they drive.
- **The default is only a value** (test 2c): a default run equals v1.0 with freq 0.03125 set, and `freq=0.12` equals v1.0's default run, bit for bit (cont, slit, beam, tri, sq9 hard vertices).
- **The app** builds `Params(N=..., src='pulse')`, so it starts with the new default. Its freq slider (0.005–0.25) holds it.

## 3. Soft driven vertices, with a hard toggle

**What.** With *drive the vertices* (`vtx_drive`), `Params.vtx_soft = True` (the port's default) makes each vertex a **soft** source: the drive is ADDED to the new level at the vertex cells. This is the beam aperture's kernel path (`SRC_BEAM`, `U += DRV[q·n + k]`), so no new kernel variant compiles. The cells are transparent to returning waves and start from rest. `vtx_soft = False` is the web app's **hard** drive, u = D(n) imposed on the cells, bit-identical to v1.0. `mc.VTX_DRIVES = ['soft', 'hard']`.

**The amplitude rule (derived).** A soft source adds s(n) at a cell. At frequency ω (per step) its steady field at the cell is U = (S / CC) g00(E), with E = 4 sin²(ω/2) / CC and g00 the lattice Green's function at the source cell:

  g00(E) = ∫_BZ dK/(2π)² 1 / (Λ(K) − E − i0)

A hard source of amplitude A forces U = A at the cell. On the infinite lattice that is exactly a soft source of strength A·CC / g00. So a soft source of amplitude **A·CC / |g00(E)|** radiates the hard source's wave: the same amplitude everywhere, a constant arg(g00) − ω behind in phase (the hard drive writes the next level).
- `mc.soft_gain(kind, co2, ω) = co2 / |g00|`, with `mc.lattice_g00(kind, E)`.
- g00 is computed by a 1D reduction: the inner Brillouin-zone integral is closed-form for all three lattices, the outer one uses Gauss-Legendre with a cosine substitution that cancels the 1/√ singularities. It costs 0.42 ms per call and is cached.
- It agrees with scipy's `quad` to ≤ 1.5e-9 over E in [1e-4, 8.5] (`dec_g00_check_MIRROR.py`). It matches the small-E limits: Im g00 → 1/4 (sq5, sq9), √3/8 (tri); Re g00 → (1/4π) ln(32/E) on sq5 (`dec_green_probe_MIRROR.py`).
- Each harmonic of the drive (the hard drive's band limit is kept) gets its own gain at its own frequency, from the Co² of its vertex cell (two values in a slab medium): `Membrane.soft_tables()`, `soft_drive(n)`.
- At Co 0.5, λ 16: gain 0.4987 (sq5), 0.4690 (sq9), 0.5446 (tri); |g00| = 0.5013 / 0.5331 / 0.4591.
- The soft drive has the hard drive's phase, sin(2π f h n) from n = 0, with no ramp. A phase-matched soft drive would start with a jump, which radiates a broadband burst.

**Measured soft / hard amplitude ratio** (lock-in amplitude at the drive frequency; `decisions_tests` 3d and `dec_soft_ratio_MIRROR.py`, CUDA f64; Vulkan f32 the same to 4 digits):

| setting | sq5 | sq9 | tri |
|---|---|---|---|
| free space: one driven cell, N 1025, λ 16, r = 3λ = 48 cells, along an axis / the diagonal (tri: 60°) | 1.0001 / 1.0002 | 1.0001 / 1.0001 | 1.0003 / 1.0003 |
| the same at λ 8 (r = 24) | 1.0001 / 1.0001 | 1.0001 / 1.0001 | 1.0012 / 1.0012 |
| phase, soft − hard (= −(arg g00 − ω)) | −19.3° | −17.5° | −17.5° |
| the defaults: the square, N 513, Co 0.5, λ 16, wall inset 9 cells from both walls; 3λ from a vertex toward the centre | 0.83 | 0.85 | 0.79 |
| the same, at the centre (the first arrivals of all vertices) | 0.82 | 0.84 | 0.87 |

The derived rule makes a lone vertex radiate exactly the hard vertex's wave (free space). At the defaults the soft vertices come out 13–21% weaker. Each vertex sits 9 cells from two pinned walls. The hard cell is a reflecting point: it re-radiates the echoes of its own corner. The soft cell lets them pass. The ratio depends on the corner's geometry, so the rule stays the geometry-free one. A gain of ~1.2 would match the default square, but it would then be wrong in free space.

**Transparency** (test 3c; a centred pulse σ 3 crossing four driven vertices at ZERO drive amplitude, N 161, 420 steps): soft vertices change nothing, bit for bit. Hard ones reflect it: 0.63–0.82 of the field's peak (sq5 0.71, sq9 0.82, tri 0.63).

**No spurious growth** (test 3e):
- soft vertices at zero drive keep the exact leapfrog energy to 0.5–1.6e-14 over 20,000 steps (CUDA f64; ≤ 5.3e-6 in f32);
- a damped driven run (d 0.01) reaches a steady amplitude (0.0% between windows);
- driven with Mur on the full square at Co 0.7 it stays bounded.
- The soft path does not change the operator, so the CFL limit is the plain one.

**`vtx_drive_bump` with soft vertices** (test 3f): the flag still seeds the web app's bumps under the driven vertices, and the soft drive adds to them. There is no dent: drive(0) = 0, so after step 1 the field equals the undriven run's exactly. The hard drive dents each bump centre by 0.95 at step 1.

**Readouts:**
- `source_note()`: the head line says "N vertex sources, driven soft" / "driven hard", and the new key `drive` is `'soft'`, `'hard'` or None.
- `drive_warnings()` covers soft vertices too: a soft point radiates the same wavelength.
- `vtx_soft_on()` is True while the soft path runs.

**Comparisons with the web app set `vtx_soft=False`:** `crosscheck_js` and `crosscheck_ledger` (in the base), the scale references, `defaults_tests` section 2 (the 05/10 hard-drive rules), `tri_modes_tests` test 8. The web app has no soft vertices, so there is no harness patch.

## 4. Deviations (from the task text)

1. **The soft gain is free-space exact, not matched at the defaults.** The task asked for a radiated wave "comparable to the hard source's at the default settings". The derived rule gives 1.000 in free space and 0.79–0.87 at the default square, because the hard vertex also re-radiates its corner's echoes (section 3).
2. **The corner exclusion** of item 1: a link into an array corner is never blocked by the new rule (a self-read; the configurations it would change were never asymmetric).
3. **`crosscheck_js` gained one config** (`sq9_circle13_neu_clip`, 28 in all). It agrees to rounding (1.5e-13 on CPU), not bit for bit, for the walls path's self-consistent 9-point form.

## 5. UI hand-off (for the app agent)

1. **Hard / soft toggle** under *drive the vertices*, shown while it is on: `p.vtx_soft` (cycle `mc.VTX_DRIVES`, `['soft', 'hard']`).
   - Suggested labels: "vertex drive: soft (added, transparent; the port's)" / "vertex drive: hard (imposed, reflects; the web app's)".
   - The step reads it every launch, so it can switch live, no reset. A reset is harmless.
   - The source note already follows it (`source_note()['drive']`, the head line).
2. **The bump checkbox** ("seed pulses under the driven vertices (the web app)") keeps its meaning for both drives. Suggested hint for soft: "the soft drive adds to them; no dent".
3. **Optional readout:** `m.soft_tables()['gain']` (per vertex and harmonic; cheap, cached), e.g. "soft gain 0.499 = Co^2 / |g00| (radiates the hard vertex's wave)".
4. **The freq slider** now starts at 0.0312. In cells mode a "lambda = Co / freq cells" text next to it, as domain mode already shows, would make the 16 cells visible.
5. **The walls panel:** `walls_info()['neu_clip']` is True for sq9 + Neumann + a clipped shape. The walls kernel can then run with no solid, and `wall_cells` counts the blocked ring diagonals. Suggested line: "sq9 Neumann ring: diagonals blocked where the shape leaves the ring (port fix, 09/10/2026)".
6. **CLI** (works today): `--set vtx_soft=0` (hard), `--set freq=0.12` (the web app's drive).
7. **App tests to add:** the toggle, its label, the live switch, the bump checkbox under both drives.

## 6. Files

- `membrane_core_MIRROR.py`: `WEBAPP_FREQ`, `VTX_DRIVES`, `lattice_g00`, `soft_gain`; `Params.freq` 0.03125, `Params.vtx_soft`; `Membrane._neu_clip`, `vtx_soft_on`, `soft_tables`, `soft_drive`, `drive_cells`; the step's soft path; `_nring_common` / `_word` (rule 3); `walls_info()['neu_clip']`; `source_note()['drive']`; the module docstring's deviations.
- `ToolsCLAUDE/js_harness_MIRROR.mjs`: PATCH `neumann-clip`.
- `decisions_tests_MIRROR.py` (new).
- Comparison configs: `crosscheck_js`, `crosscheck_ledger`, `crosscheck_alignment`, `ScratchCLAUDE/ScaleCLAUDE/make_ref_MIRROR.py`, `defaults_tests`, `tri_modes_tests`.
- Probes: `ScratchCLAUDE/Round5CLAUDE/DecisionsCLAUDE/dec_*_MIRROR.py`.
