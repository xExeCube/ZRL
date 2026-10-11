# Progress: directional beam source (area: beam)

*Claude Opus 5.5 -- started 01/10/2026*

## 01/10/2026
- Started. Reading core source handling.
- DESIGN fixed: new beam_MIRROR.py (lattice symbols sq5/sq9/tri, discrete dispersion, group-velocity aim,
  slab aperture + back aperture shifted by the most-opposed neighbour vector, sheet gain Co^2 dLambda/dk / d,
  one-sided gain /|2 sin kd|; packet u_prev by FFT mode-by-mode (K.n > 0 half, each mode moved by its omega)).
- CORE (membrane_core_MIRROR.py replaced from ScratchCLAUDE/BeamCLAUDE/membrane_core_work_MIRROR.py; backup
  membrane_core_backup_MIRROR.py): SOURCES += beam, packet; Params += beam_angle (deg), beam_width,
  beam_onesided, beam_group, pk_len, pk_lambda; SRC_BEAM kernel branch (soft, += DRV[q*nsrc+k]);
  build_beam (key-checked in step, so Co/freq changes rebuild), _add_packet (seed + P), beam_geometry().
- SMOKE (ScratchCLAUDE/BeamCLAUDE/smoke_MIRROR.py, cuda): sq5/sq9/tri beam + packet at 30 deg, 200 steps:
  finite; beam near-field peak 1.017-1.019 (normalised to 1), packet in-array 1.0. aim moves sq5 by 0.55 deg.
- DIRECTIVITY: back aperture 'shift' (front translated by a neighbour vector) left a first-order residue
  off-axis (sq 45 deg 17.6 dB); 'slab' (own slab d behind, own phases) gives 28-30 dB everywhere at
  w0 8 / lambda 12 (limit ~ (kd/2)^2 <alpha^4> of the backward lobe, ~-32 dB predicted). Default 'slab';
  ramp 4 periods. Slab-edge ties quantised (tri 180 deg was 18.4 dB from cos/sin round-off -> 29.6).
- WIDTH (w0 16, lambda 12): w(z) within 1-7% of the Gaussian-beam law for z >= 100 (tri 45 deg z=50: 17%).
- PACKET: residue ~8e-6 of energy (at-rest start: 0.50), speed = spectrum-weighted discrete v_g within 0.05%
  (central-K v_g is 1.5% faster: off-axis plane waves), direction 0.02 deg.
- STABILITY Co 0.99 Cmax all lattices/bc: finite, bounded. OBSTACLES: pinned slab reflects (x187-299 backward),
  aperture clipped at circle wall and through a solid, solids stay 0.
- STEERING (energy aim): worst 0.41 deg (sq5 lambda 6), <=0.21 deg lambda>=8, sq9/tri <=0.03 deg; phase aim on sq5
  up to 2.6 deg (lambda 6) / 0.58 deg (lambda 12). COST: beam +0.07 ms/step at N 1025 (like 'cont'), ~0 at 2049.
- APP: beam_controls (angle, width, energy-aim, one-sided, lambda readout, packet L/lambda), emitter overlay
  (aperture cyan + arrow orange) 2D + 3D, 'draw the emitter' toggle; freq slider shown for beam. Shots OK.
- LEDGER: 'beam' counted as driven (3 places); energy row n/a for packet on tri (square-lattice energy formula,
  5.4% false FAIL measured). Probe: 0 FAIL over beam/packet x sq-square/circle-mur/tri-n6.
- TESTS: app_tests t_beam 4/4 (isolated); identity: 12 mirror-symmetry checks (fixed aim tolerance + K.n=0 split
  tolerance: sq9 packet 45 deg was 5e-3 asymmetric) ALL PASS.
- FINAL RUNS 01/10/2026 (logs ScratchCLAUDE/BeamCLAUDE/final_*_MIRROR.txt): crosscheck_js cpu 27/27; identity ALL PASS
  (incl. 12 new mirror checks); solids ALL PASS; app_tests 75/75; crosscheck_ledger 0/0 mismatches, 15/15;
  beam_tests 34/34 (tri 60 deg now 29.6 dB after the aim tolerance fix). Shots: beam_sq_2d, beam_tri_2d,
  beam_sq_3d, packet_tri_3d (+ contact_MIRROR.png), app_tests beam_overlay_2d/3d. Notes beam_MIRROR.md + README. DONE.
