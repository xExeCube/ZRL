# Progress log: walls build (round 4: rigid walls + corner blocking in the core)

Owner area: membrane_core_MIRROR.py (walls path), solids_MIRROR.py (wall types), walls_tests_MIRROR.py,
NotesCLAUDE/walls_MIRROR.md. Backups: ScratchCLAUDE/Round4CLAUDE/WallsBuildCLAUDE/*_before_MIRROR.py
(identical to the BeforeCLAUDE snapshot, checked with cmp 04/10/2026). Logs: ScratchCLAUDE/Round4CLAUDE/LogsCLAUDE/.

## 04/10/2026
- Read: README, research note sections 0/2/4/5, walls_lib, linkmask_probe, core, solids, tests.
- Design decided (details go to NotesCLAUDE/walls_MIRROR.md):
  - solid_np becomes the TYPE map (0 free, 1 pinned, 2 rigid); mask_np / M keep their meaning (1 = free),
    so every host reader of the mask (snapping, beam, clip fraction, vertices, renderer) already treats
    rigid cells as solid.
  - per-cell class C on the device (free bit + neighbour class F/P/R/X) and a per-cell link word W
    (bit 0 free, bits 1-8 "self" = dropped link, bits 9-14 "zero" = pinned-blocked, bit 15 = a blocked
    4-type link on tri), built by one Taichi pass; painting patches C and rebuilds W in box + 2.
  - the walls path is taken only when some free cell has a non-plain word (counter kept on the device);
    otherwise today's kernel runs unchanged (template flag).
- Next: implement core, then solids, then walls_tests, then the suites.
- 04/10 DONE core: classes C + words W (k_walls_full / k_walls_patch, counters), walls template in
  _steps / k_run / k_run_rec / k_stats, k_rescale_w, recorder forward differences through the words,
  main_region on the walls path (4-structure + corner merge; tri refined grid when n_blk4 > 0),
  set_edge, walls_info, operator(), leapfrog_energy(), unknowns(). solids: wall field on objects and
  strokes, plate_wall, type-map rasterisation (overwrite in order), scene format v2.
- 04/10 walls_tests_MIRROR.py written. Results so far (cuda f64, walls_tests): 1 operator == walls_lib
  exact (max|L-L_ref| 0, 32 steps <= 4.6e-14); 2 energy 1.9e-15..6.6e-15 over 20000 steps; 3 leak 0.0
  everywhere; 6 CFL stable, drift <= 1.9e-15; 7 disc bands ok.
- FOUND: cavity DC to 1e-14 is not attainable over 5000 steps by any float leapfrog (zero mode
  integrates rounding twice): walls_lib numpy reference drifts 9.3e-13 sq5 / 1.2e-9 sq9 / 9.0e-10 tri
  (scratchpad dc_probe). The sq9 f64 web-app order leaves -4.4e-16 u on a constant field (systematic):
  the walls path now uses the consistent sq9 form in f64 too -> sq9 cavity drift 1.1e-9 -> 6.9e-14.
  Test 8 now asserts 1e-14 at 100 steps and 1e-10 at 5000 steps.
- FOUND: test 4 tri image mismatch 2e-12 was the image pulse's tail on the free side (rows 0.866
  apart), not the solver; fixed in the test.
- 04/10 suites (logs in ScratchCLAUDE/Round4CLAUDE/LogsCLAUDE/): crosscheck_js cpu/cuda/vulkan-f32 27/27
  (cpu same-seed values identical to FinalCLAUDE/xc_cpu.txt); identity cuda + vulkan f32 ALL PASS;
  solids cuda ALL PASS (test 3 now expects the one-cell diagonal wall to SEAL); beam 34/34; scale
  16/16; app_tests 95/95; crosscheck_ledger PASS.
- FOUND: the recorder's energy history is not run-to-run deterministic (atomic order): the same core
  differs from itself by 5.5e-16 (cuda, cpu); 2e-7 on Vulkan (f32 accumulators). Parity test compares
  fields / peak / probes bit for bit and E to 1e-14 (1e-6 Vulkan).
- Painting: k_patch_mask merged into k_walls_patch (one launch, one upload), class code in fewer
  numpy passes, counters read lazily: 2.5-2.6 ms / segment at 2049 and 8193, long diagonal 69 ms.
- Bench (walls_bench_MIRROR.py, 3 runs): i32 word cost CUDA f32 +23-27% at 8193 -> WORD_DT u16 on
  CUDA/CPU (i32 Vulkan): CUDA f64 +3..8%, CUDA f32 +10..12% (8193), Vulkan ~+0..2%; reversed order
  (run 3) same within 1%.
- Next: rerun walls tests after the u16 switch (running), write NotesCLAUDE/walls_MIRROR.md, final logs.
- 04/10 u16 word (CUDA/CPU), explicit casts (no Taichi precision warnings); walls tests ALL PASS on
  cuda f64, cuda f32, vulkan f32 (64 checks each). NotesCLAUDE/walls_MIRROR.md written (design,
  numbers, deviations, ledger guidance, open issues, UI hand-off). Probe scripts copied to
  ScratchCLAUDE/Round4CLAUDE/WallsBuildCLAUDE/*_MIRROR.py. FOUND: tri plate lettering has 8-20
  corner-blocked cells at N 161-4097 (walls kernel ON on the tri plate); sq9 plate has none.
- Final validation batch running -> LogsCLAUDE/final_*_MIRROR.log, status in final_batch_status_MIRROR.txt.
