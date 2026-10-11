# Circle fidelity of grid flavors — the membrane's error element

**Date:** 2026-10-01
**Status:** every number below was executed (numpy/scipy, float64) before being written down.
The lattice runs use the app's exact stencils and mask rule: cross-checked against
`membrane_core_MIRROR.py` itself (Taichi, CPU f64, circle, N = 101, 300 steps): max |difference|
4.2e-16 (sq5), 3.9e-14 (sq9), 2.5e-14 (tri), identical node counts.
**Code:** `code/` (run order in `code/README_MIRROR.md`). **Figure:** `circle_grid_fidelity_MIRROR.png`.

Setup unless stated: circle of radius R = 60 cells (N ≈ 124), centred Gaussian pulse σ = 6 cells
(σ = 3 is the app default and is reported too), clamped wall, no damping, Co = 0.5, exact
2nd-order start at rest. A "cycle" is one round trip, centre → wall → centre (t = 2R).

---

## 1. The premise — RULED OUT in 2D, TRUE in 1D and 3D

"With no absorption or damping, cycle n should equal cycle 1" fails for a perfect continuum circle.
Exact Bessel series, ‖u(2mR) − u(0)‖ / ‖u(0)‖:

| cycle m | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 2D disc | 0.98 | 2.00 | 1.06 | 0.015 | 0.90 | 1.99 | 1.15 | 0.059 |
| 3D ball | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 (machine precision) |

Mechanism: each cycle adds 180° at the wall plus the focus (Gouy) shift: 90° in 2D, 180° in 3D. So 2D
returns only every 4th cycle (inverted at cycle 2), and even then drifts, because the drum's radial
frequencies (Bessel zeros) are not evenly spaced. Radial-mode spacing / π, first five gaps:
D = 2: 0.9916 … 0.9995; **D = 3: 1.00000 exactly**; D = 4: 1.0135 …; D = 5: 1.0287 …
Evenly spaced (perfect refocusing) only for D = 1 and 3: another "dimension controls what exists"
entry, next to strong Huygens.

**Consequence:** comparing cycle 1 with cycle N measures physics plus grid error. Measure instead what a
true circle keeps at every cycle: radial symmetry.

## 2. The error element — ε_ang

ε_ang(t) = energy of the field's non-radial part / total energy, where the radial part is a
least-squares cubic-spline fit u ≈ f(r) (knots every 0.5 cell). It is exactly 0 for a true circle at all
times, needs no reference solution, and its growth per cycle is the grid's circle error.
Metric floor (the exact solution sampled on each grid): 1e-10 (σ = 3), 3e-12 (σ = 6).
Companion: ε_tot = energy of (u − exact Bessel solution) / total energy. It also contains dispersion.

Lattice-only shortcut (bias-free, no spline): every lattice node has an integer radius², i² + j² (square)
or a² + ab + b² (triangular). Nodes sharing it must agree in a true circle, so the spread within each
group is pure angular error.

## 3. Results (R = 60, σ = 6)

| grid | ε_ang cycle 1 | cycle 2 | cycle 5 | cycle 10 | ε_tot cycle 10 | nodes | max stable Δt |
|---|---|---|---|---|---|---|---|
| sq5 [app] | 8.0e-4 | 2.5e-3 | 7.3e-3 | 1.8e-2 | 0.26 | 11 289 | 0.707 |
| sq9 [app] | 1.3e-3 | 4.6e-3 | 7.2e-3 | 1.5e-2 | 0.48 | 11 289 | 0.866 |
| tri [app] | 1.0e-3 | 1.3e-3 | 1.3e-3 | 1.8e-3 | 0.27 | 13 057 | 0.817 |
| sq + edge triangles | 6.4e-4 | 2.4e-3 | 4.3e-3 | 1.4e-3 | 0.25 | 11 133 | 0.707 |
| **tri + edge triangles** | **4.9e-7** | **1.0e-6** | **1.4e-6** | **2.1e-6** | 0.27 | 12 823 | 0.760 |
| hex shells → circles | 1.1e-6 | 1.7e-6 | 1.2e-6 | 1.2e-6 | 0.47 | 10 621 | 0.738 |
| Penrose + edge triangles | 1.8e-5 | 3.1e-5 | 2.3e-5 | 3.1e-5 | 0.27 | 13 682 | 0.633 |
| unstructured (Triangle, ≥30°) | 8.8e-5 | 1.7e-4 | 2.3e-4 | 5.5e-4 | 0.31 | 13 052 | 0.577 |
| graded 2:1 toward the edge | 8.2e-5 | 1.5e-4 | 1.5e-4 | 2.8e-4 | 0.35 | 21 230 | 0.321 |

σ = 3 (app default), ε_ang at cycle 1 / 10: sq5 2.7e-2 / 0.10 · sq9 8.7e-3 / 9.0e-2 · tri 9.0e-3 /
3.4e-2 · sq+edge 2.4e-2 / 4.0e-2 · tri+edge 5.1e-5 / 3.4e-4 · hex shells 2.4e-5 / 1.8e-5. At σ = 3,
ε_tot passes 1 by cycle 5 on every grid: the waveform's timing has drifted (dispersion), whatever the
boundary.

**Staircases depend on luck.** Moving R by single cells (55…65), time-averaged ε_ang over cycles 1–5
(σ = 6): sq5 median 5.0e-3 (2.4e-3…1.0e-2), sq9 4.5e-3 (3.5e-3…7.3e-3), tri 1.5e-3 (0.9e-3…3.0e-3),
tri + edge 2.5e-6 (1.1e-6…7.5e-6). Compare flavors over several N, never at one.

**Where the noise comes from** (free-space control: same pulse, wall out of reach, ε_ang after one
cycle's travel distance, 120 cells): sq5 2.4e-2 (σ 3) / 4.3e-4 (σ 6); sq9 1.0e-5 / 1.1e-8;
tri 6.5e-7 / 6.6e-10. So the square lattice makes about 90 % (σ 3) or about half (σ 6) of its own
circle noise with no wall at all (its leading error has a cos 4θ term). For sq9 and tri, almost all of
the noise is the staircase wall.

## 4. Hex lattice: does it have 2 phases?

- **Pointy-top vs flat-top: the same thing.** It is a 30° rotation, and the circle has one isomer.
  Measured: identical to 1e-10 (relative), same 13 057 nodes. For polygon domains the lattice angle
  matters only relative to the domain, which the app's isomer rotation already covers.
- **What does differ is centring**: on a node (6-fold symmetric staircase) or on a triangle's centre
  (3-fold). Time-averaged ε_ang over cycles 1–10: node 1.6e-3, hole 2.8e-3 (σ 6); 2.4e-2 vs 2.2e-2
  (σ 3). Not systematically better. Square lattice, cell-centred vs node-centred: worse (1.4e-2 vs 8.1e-3).
- Ambiguity flagged: a "hexagon grid" with nodes at hexagon corners (honeycomb, 3 neighbours) *does* have
  2 inequivalent node types (A/B, as in graphene). That is not the app's triangular lattice.

## 5. Square grid with triangles only at the edge — WORKS, but on the wrong lattice

Built as mass-lumped P1 elements: lattice nodes deeper than half a cell, plus nodes on the circle, all
triangulated. Away from the edge the rows are **exactly the app's stencils** (0 deviant rows); only the
edge band differs (~420 of 12 823 nodes for tri). Wall position becomes exact (R_eff − R = +0.003
vs +0.31…+0.35 for staircases).
- On the square lattice: about 2× on average (time-averaged 2.3e-3 vs 4.5e-3 at N ≈ 124). Limited because
  the square lattice's own anisotropy stays.
- On the triangular lattice: about **600–1400×** (2.5e-6 vs 1.5e-3 jitter medians; 1.1e-6 vs 1.5e-3 at N ≈ 124).
- Cost: stable Co 0.817 → 0.760 (tri), 0.707 → 0.707 (square). The app's Co = 0.5 is unaffected.
- What it does not fix: ε_tot (dispersion) is unchanged: 0.27 at cycle 10 with or without the edge
  fix. That needs a higher-order stencil or a larger N.

## 6. Cells smaller toward the edge — no gain for the cost

With an exact edge, 2:1 grading changed ε_ang at cycle 10 from 5.5e-4 (uniform unstructured) to 2.8e-4.
It cost 1.6× the nodes and ~1.8× the steps, because the explicit time step is set by the smallest cell
everywhere. tri + edge was 100–250× better than either. With a staircase edge, refinement shrinks the stair
steps, but uniform 2× refinement of sq5 buys 6.7× (4.5e-3 → 6.8e-4) for 8× the work. Fitting the edge
beats refining around it.

## 7. Best grid flavor for the circle — and does it have to be one of the two?

- Among lattices: **the triangular one is the best possible.** Crystallographic restriction: a 2D
  lattice can have only 2-, 3-, 4- or 6-fold symmetry, and 6 is the maximum. Its leading error is
  isotropic, and the first anisotropy is cos 6θ at the next order.
- Off-lattice is allowed and helps the shape. **Hex shells bent into circles** (ring k carries 6k nodes:
  the hexagon flavor morphed into the circle flavor) has the smallest and flattest ε_ang. It is
  coarser radially, though, so ε_tot is 2× worse. **Penrose + edge** (a ZRL flavor, 10-fold on
  average) is 15–35× behind tri + edge on shape and level with it on ε_tot: its local vertex
  neighbourhoods vary, which scatters. **Unstructured** meshes are worse still.
- Best overall at this N: **triangular lattice inside + exact circular edge.**

## 8. Convergence with N (same drum, σ = R/10; ε_ang averaged over cycles 1–5)

| grid | N ≈ 64 | N ≈ 124 | N ≈ 244 |
|---|---|---|---|
| sq5 [app] | 1.5e-2 | 4.5e-3 | 6.8e-4 |
| sq9 [app] | 3.5e-2 | 4.8e-3 | 9.7e-4 |
| tri [app] | 1.2e-3 | 1.5e-3 | 2.9e-4 |
| sq + edge | 2.2e-2 | 2.3e-3 | 1.8e-4 |
| tri + edge | 1.3e-4 | 1.1e-6 | 1.8e-7 |
| hex shells | 1.9e-5 | 9.4e-7 | 6.1e-8 |
| Penrose + edge | 5.2e-4 | 2.8e-5 | 4.8e-6 |

ε_tot falls ~13–14× per doubling of N on every grid (second-order dispersion). sq9 and hex shells sit
about 2× above the rest. Single-N slopes of ε_ang are noisy because of the staircase luck in §3.

## 9. Proposed ledger rows for the app (not built)

- `circle fidelity ε_ang per cycle`: centred pulse, circle, no damping. Pass if it stays ≤ a declared
  budget; report the per-cycle growth.
- `free-space anisotropy`: the same pulse with the wall out of reach. This separates the lattice's own
  noise from the staircase's.
- Optional boundary mode `tri + fitted edge`: precomputed weights for the edge band only; the interior
  kernel is unchanged.

## 10. Caveats

- The finite-element grids use mixed-Voronoi lumping. The first attempt with 1/3-per-triangle lumping
  turned Qhull's arbitrary square diagonals into a random medium (node mass 0.67–1.33) and is
  discarded. The home-made DistMesh meshes (slivers to 0.1°) are discarded too.
- `sq9 + fitted edge` was not built (P1 elements give the 5-point stencil inside). Given sq9's quiet
  bulk (1e-8), it would likely land near tri + edge. **OPEN.**
- All grids are 2nd-order schemes; the edge fix removes noise, not dispersion.
