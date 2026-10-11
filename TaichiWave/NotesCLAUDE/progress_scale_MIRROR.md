# Progress: scale mode ('cells' / 'domain')

*Generated from scratch — Claude Opus 5.5 — 02/10/2026*

- 02/10/2026: started; read README, app and beam notes.
- 02/10/2026: core done (Params.scale + fractions, apply_scale/set_scale/crossings/scale_warnings; reseed + set_co apply). Cells mode bit-identical on 15 ref configs (refs made from the pre-edit core, ScratchCLAUDE/ScaleCLAUDE/ref_cells_*_MIRROR.npz).
- 02/10/2026: scale_tests_MIRROR.py 16/16 (CUDA, N 257-2049, 82 s): max-norm diff ratios 4.01-4.10 per doubling on sq5/sq9/tri (with the half-step rest-start offset; without it 1.55-1.87); domain peak at r=0.5 spread 0.43-0.79% (last doubling <=0.037%); cells peak ratio 0.63-0.67 per doubling. Log ScratchCLAUDE/ScaleCLAUDE/scale_tests_run1_MIRROR.log. Next: app.
- 02/10/2026: app done (scale button, crossings readout, crossings/s speed with fractional-step carry, fraction sliders for sigma/lambda/beam w/packet L/lambda, amber warnings, auto height in both modes, --scale/--cps/--auto-height). Shot OK. Next: app tests.
- 02/10/2026: app_tests 95/95 (88 + 7 scale). Running all suites.
- 02/10/2026: NotesCLAUDE/scale_MIRROR.md written; README section + app_MIRROR.md section added.
- 02/10/2026: FINAL all green: crosscheck_js cpu 27/27, identity ALL PASS, solids ALL PASS, beam exit 0, ledger PASS, app 95/95, scale 16/16 (cuda, N to 2049) + 16/16 (cpu quick). Logs ScratchCLAUDE/ScaleCLAUDE/final_*_MIRROR.log.
