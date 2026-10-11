# Progress log: triangular-lattice modes (round 4, phase 1b) + full regression

Owner area: membrane_core_MIRROR.py (shape 'array', tri bc neumann/periodic, mur fallback), solids_MIRROR.py,
alignment_MIRROR.py, ledger_MIRROR.py, crosscheck_ledger_MIRROR.py, tri_modes_tests_MIRROR.py,
NotesCLAUDE/walls_MIRROR.md (new section + UI hand-off). Backups (copied before any edit, 05/10/2026):
ScratchCLAUDE/Round4CLAUDE/TriModesCLAUDE/*_before_MIRROR.py. Logs: ScratchCLAUDE/Round4CLAUDE/LogsCLAUDE/tri_*.

## 05/10/2026
- Started. Read README, walls_MIRROR.md, progress_walls_build_MIRROR.md, the core (whole file).
- Backups of 18 modules taken (see folder above).
- Read: walls notes, ledger (all rows), alignment (al_* model), crosscheck_ledger port checks, solids frame,
  app shape/bc cycle (G cycles mc.SHAPES, so 'array' shows up there; B cycles bc on both lattices).
- DESIGN (decided 05/10, to implement):
  * SHAPES += 'array'. sq: sq_full() is True for 'array' (same path, mask all ones, bc rules, edge n/a).
    tri: domain mask = the interior (ring 0) unless the torus (all ones). rot and rad never apply.
    dom_R = base_half - 1.5, inner_R = base_half, shape_poly = the 4 ring-corner cell centres (CCW from
    cell (0,0) on tri; the sq_full square on sq), mode_box (0, N-1), dom_rot() = 0 (solids' domain frame).
  * full_array() = sq_full() or tri 'array'; torus() = bc periodic and (sq, or tri 'array').
  * Tri bc acts on 'array' ONLY: dirichlet = clamped ring (X pinned), neumann = rigid ring (X rigid via
    the words' erig flag), periodic = rhombic torus (wrapped kernel branch, words wrap), mur = fallback to
    dirichlet + bc_info() flag/reason. Every web-app tri shape keeps the clamped ring whatever bc says
    (bit-identity demanded; the web app ignores bc on tri). On 'array' (both lattices) edge does not
    apply (like sq_full): bc rules the ring. erig = edge=='rigid' on masked shapes; tri array: bc==neumann.
  * Masked tri + periodic == masked tri + dirichlet (consistent with sq periodic + masked shape whenever
    the shape stays inside the array; a shape clipped by the array (rad > 1) wraps on sq, stays clamped on
    tri = web-app parity). Documented as a deliberate difference.
- 05/10 DONE core: SHAPES += 'array', FIXED_SHAPES, TRI_BCS, reasons; full_array(), torus(), tri_torus(),
  bc_effective(), bc_info(), dom_rot(), _erig(), _ring_key() (step() rebuilds the tri array's domain if bc
  moved between torus and ring without a reseed), mode_note(); kernel: LAT_TRI + BC_PER wrapped branch
  (plain + walls), sum_tri_wi (wrapped indices; k_rescale_w uses it); _walls_key torus = torus(), erig =
  _erig(); _class_box skips the tri ring rule on the torus; unknowns/operator/k_rescale/pulse_clip use
  torus(). solids: domain frame uses m.dom_rot(). Backups were taken before these edits.
- Smoke (ScratchCLAUDE/Round4CLAUDE/TriModesCLAUDE/smoke_MIRROR.py, cuda): all 16 array modes run; tri mur
  falls back (flag); tri neumann runs the walls kernel; sq9 'array' == 'square' rad 1 bit-identical, 4 bcs.
- Next: tri_modes_tests_MIRROR.py, then alignment + ledger, then the regression.
- 05/10 tri_modes_tests_MIRROR.py written (tests 1-10). --quick --only 1-9 on cuda f64: ALL PASS (142 s):
  torus energy 5.0e-16 / 3.6e-16 (4000 steps), translation bit-identical (plain + walls), 6-fold 3.0e-15
  before the seam, 5.3e-15 after; rigid ring energy 1.5e-15 / 2.4e-15; plane pulse vs 1D chain 6.4e-15,
  peaks +0.4966 rigid / -0.4966 clamped; 0.995 C_max stable x4; sq array == square rad 1 x10 bit-identical;
  painting incremental == full x3; parity with the 04/10 core 26/26; helpers 9a-9k pass.
- Next: alignment ('array' model), ledger (array / tri bcs / rigid edge / rigid solids), crosscheck_ledger
  port-only configs, then full-length tests on 3 backends, then the regression.
- 05/10 alignment: model 'arr' for 'array' (dirs sq 0/90, tri 0/60; fixed rotation; 4/4 rows on both lattices;
  7c PASS period 1, 7c-bis n/a "its walls ARE the array edge"; al_set_rot refuses).
- 05/10 ledger patched (ScratchCLAUDE/Round4CLAUDE/TriModesCLAUDE/patch_ledger_MIRROR.py, patch_ledger_energy_MIRROR.py):
  _Ctx flags arr/full/torus/tri_torus/rigid_solids/edge_rigid/rigid_ring; edge() learns 'rigid' and the tri array;
  type map checked in the symmetry premises; wrap on the rhombic torus for mirror/6-fold/cone/clearance;
  conformance (N-2)^2 / N^2; staircase exact for the tri array; mode rows n/a for a rigid masked square;
  Mur fallback port texts. FOUND: the energy row FAILed on the tri array (2-4% "drift") -- the recorder's E
  is the square lattice's functional (gx, gy only). FIX 'tri-energy': the recorder also sums gz^2 (3rd tri
  direction, HIST column 6) -> m.hist['Etri'] = kin/(2Co^2) + (gx^2+gy^2+gz^2)/3; the ledger grades it on tri.
  FOUND 2: on the rhombic torus the interior-only sum read 2.7e-2 drift; the recorder now sums the whole
  torus with wrapped differences there only (torus_energy_probe_MIRROR.py: 2.69e-2 -> 1.84e-4, N=97 impulse).
  tri_energy_row_MIRROR.py (cuda, log LogsCLAUDE/tri_energy_row_MIRROR.log): every tri config PASS at 1300 and
  2500 steps; e.g. rigid packet E 1.37e-1 -> Etri 9.3e-6; rhomb60 r0.8 impulse 9.0e-3 -> 1.1e-3; n3/n6/circle
  equal (6-fold symmetric fields).
- Next: tri_modes test 10, crosscheck_ledger port-only configs, full-length test runs, regression.
- 05/10 crosscheck_ledger port-only checks + 11 (O-W: torus pulse/impulse, rigid ring, Mur fallback + bc_info,
  sq array == square, rigid edge + Mur, rigid masked square mode, asymmetric pinned/rigid types, all-rigid
  8-fold, tri hexagon rigid edge): 26/26 PASS (cuda). app: the boundary label shows bc_info()'s effective bc.
  tri_modes test 9l (Etri == numpy, rel 3.2e-16 torus / 1.5e-16 clamped).
- 05/10 FULL REGRESSION launched: ScratchCLAUDE/Round4CLAUDE/TriModesCLAUDE/tri_regression_batch_MIRROR.sh
  (status: LogsCLAUDE/tri_batch_status_MIRROR.txt, logs tri_*_MIRROR.log, shots tri_shot_*_MIRROR.png).
  If cut off: rerun the script (or the lines not in the status file).
- Next while it runs: walls_MIRROR.md section 13 + section 11 hand-off, README rows.
- 05/10 regression: trimodes cuda f64 ALL PASS (195 s; full length: torus energy 1.1e-15/1.5e-15, rigid ring
  1.8e-15/4.3e-15, 6-fold 3.0e-15 / 9.0e-15 after 1500 steps, parity 40/40). Notes: walls_MIRROR.md header,
  2.3, 9, 10.5, 11 (tri hand-off inserted), 12 edited; section 13 drafted in
  ScratchCLAUDE/Round4CLAUDE/TriModesCLAUDE/walls_notes_s13_MIRROR.md (@@ placeholders for f32 numbers + bench),
  to append once the batch is done.
- 05/10 trimodes cuda f32 ALL PASS (323 s), vulkan f32 ALL PASS (325 s): torus energy 7.6e-7/6.1e-7 (cuda f32),
  9.3e-7/4.8e-7 (vk); translation bit-identical on both; parity 40/40 on both. Numbers filled into the s13 draft.
- 05/10 walls tests cuda f64 / cuda f32 / vulkan f32 ALL PASS (229 / 245 / ~200 s). Notes: ledger_MIRROR.md and
  alignment_MIRROR.md got short dated sections (backups *_notes_before_MIRROR.md in TriModesCLAUDE). README patch
  prepared (patch_readme_MIRROR.py; needs readme_regression_MIRROR.md with the final table).
- 05/10 crosscheck_js cpu/cuda/vulkan-f32 27/27 (every line identical to the 04/10 final_ logs except the energy
  history dE, the atomics' order); identity cuda + vulkan ALL PASS; solids ALL PASS; beam 34/34; scale 16/16.
  Remaining: crosscheck_alignment (+ --unl), crosscheck_ledger, app_tests, shots.
- 05/10 (by the round 4c "Defaults + docs" step, NotesCLAUDE/progress_round4_defaults_MIRROR.md): the bench was run
  (LogsCLAUDE/def_bench_*), section 13 appended to walls_MIRROR.md, the README updated. The regression's remaining
  suites all passed (tri_batch_status_MIRROR.txt: DONE).
