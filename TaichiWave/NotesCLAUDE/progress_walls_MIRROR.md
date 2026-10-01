# Progress log: walls research (rigid vs pinned)

Owner area: ResearchCLAUDE/WallsCLAUDE/ + NotesCLAUDE/rigid_vs_pinned_walls_MIRROR.md

## 30/09/2026 (resumed after the 29/09 usage-limit cut)
- Found on disk: walls_lib, checks, chain_reflection (+ old output), energy_leak (+ output), circle_eigs (never run to an output file).
- Fixed: chain_reflection_MIRROR.py had an unterminated string (a literal newline inside print('\n=== 3...')) -- it would not even compile.
- Next: run checks, rerun chain_reflection, run circle_eigs, write scattering + rotated-square script, Taichi link-mask cost probe, then the note.
- 30/09 done: checks_MIRROR.py 22/22 PASS (P == port stencils; all rules symmetric; R1 sq9 == ghost-copy Neumann ring with corner u(1,1)) -> OutCLAUDE/checks_out_MIRROR.txt
- 30/09 done: circle_eigs_MIRROR.py ran (36 s) -> OutCLAUDE/circle_eigs_out_MIRROR.txt. All staircase rules first order; the pooled log-log fit is misleading when errors change sign -> report R*err (effective offset) instead.
- 30/09 done: chain_reflection rerun (the fixed R1 now gives 0.00 deg on the sq9/tri axis wall at every angle; the 29/09 output predated that fix).
- 30/09 in progress: scatter_MIRROR.py (single cell, 3x3, rotated 12x12 square) running for sq5/sq9/tri. Bug found+fixed in it: op_for(solid|ring) made the BOX ring rigid under R0.
- Next: taichi_linkmask_probe_MIRROR.py (kernel cost of a per-cell link word vs on-the-fly types), then the note.
- 30/09 done: linkmask_probe_MIRROR.py (Taichi prototype, sq9): link-word and on-the-fly-type kernels verified vs walls_lib on a mixed pinned/rigid scene (f64 3e-14, f32 6e-7); timings in OutCLAUDE/linkmask_probe_out_MIRROR.txt (word +2..20%, types -6..+38% vs the plain mask kernel; CPU was loaded by the scatter runs).
- 30/09 done: circle_eigs rerun with a per-centre convergence summary (all rules first order; R*|err| ~0.35 P, ~0.14-0.24 R0, ~0.06-0.19 R1).
- Heads-up for whoever continues: Bash heredocs here collapse double backslashes, so python-in-heredoc edits containing \n silently fail; use the Edit tool.
- 30/09 done: note drafted (NotesCLAUDE/rigid_vs_pinned_walls_MIRROR.md) with sq5 scattering + rotated square; sq9/tri scatter still running -> add 3.10b/3.11b when they finish, then rerun energy_leak on the final lib (reproducibility) and finish.
- 30/09 FOUND + FIXED: scatter part 2 compared an 11x11-cell axis square (strict |x|<6 with side 12) against a 145-cell rotated one; the 20% area mismatch explained most of the "rigid rotated square scatters 38-54% more". Old part 2 marked SUPERSEDED in the outputs; part 2 rerunning with side 11 (121 vs ~121 cells) + 45 deg -> OutCLAUDE/scatter_<lat>_part2_out_MIRROR.txt. Note section 3.11 must be rewritten from those.
- 30/09 verified: energy_leak rerun on the final lib is identical to the 29/09 output (except timings).
- 30/09 done: part 2 reruns (matched rasterisation, 30 and 45 deg) -> pinned within +-3%, rigid sq5 +25..33% (not shrinking lam 8->16), sq9 +4..27% (R1 worse than R0), tri within +-10% (= its disc reference). Note sections 0, 3.10 (sq9/tri table), 3.11, 5.2, 5.6 updated.
- STATUS: note complete. Remaining open items are listed in the note (impedance on tri, R1 encoding, cut cells).
