# BlastBox

*Written by Claude Fable 5, 04/09/2026.*

A blast-wave / sound-wave simulation in a 2D particle field. The learning goals are
the behaviours of pressure, heat, fluid flow, energy and sound: shocks, breaking
the sound barrier, echo, propagation, interference, and how all of it changes
across different environments and inside multi-room structures.

This file doubles as the ZRL-to-physics transfer note requested on 30/08/2026:
the second half maps Zero-Recursive Lattice machinery and notation onto this and
any other physics sim.

---

## Stack

First project of this kind, so the stack is deliberately small with room to grow.

| Piece | Library | Why |
|---|---|---|
| Engine | **numpy** | every particle property is one flat array (struct-of-arrays) |
| Neighbours | **scipy** `cKDTree` | one call, exact, O(N); swap for a hash grid / Taichi later |
| Window | **pygame-ce** (`import pygame`) | one dependency: draw + input + frame loop |
| Checks | **pytest** | the ZRL verify-before-affirm rule: every claim gets a test |
| Exact math | **sympy** | symbolic identities in tests (Sod/Sedov later); optional |
| Plots | **matplotlib** | validation plots; optional |

`simpy` (installed) is a discrete-event queue library, unrelated to physics. It is
**not used** here; the physics dependency you want is `sympy`.

Expansion without a rewrite: **Taichi** for GPU-scale particle counts, and a
grid-based FDTD companion (ported from Nils Berglund's CC0 wave code) for clean
linear acoustics.

---

## Run it

From this folder (`...\PhysicsSimE\BlastBoxECLAUDE\BlastBoxECLAUDE\`):

```bash
python BlastBoxECLAUDE.py
```

- `python BlastBoxECLAUDE.py open_box` - single open room
- `python BlastBoxECLAUDE.py two_rooms` - two rooms + doorway (default)
- `python BlastBoxECLAUDE.py --checks` - print the ledger and exit, no window
  (relaxes for 2 and measures for 4 time units: a couple of minutes)

Controls: drag = pan, wheel = zoom, `1` density / `2` temperature / `3` flat,
`space` = refit, `esc` = quit.

Tests:

```bash
python -m pytest
```

If the bare `pytest` command "is not recognised", that is the `--user` PATH
warning you saw: the launcher script is not on PATH. `python -m pytest` sidesteps
it. To fix the PATH itself, add the reported `Scripts` (Windows) / `.local/bin`
(Linux) directory to PATH. Note the warning showed a Linux path
(`/home/execube/.local/bin`) - if the packages went into WSL, the Windows
interpreter that Visual Studio uses will not see them. Verify with the interpreter
you actually run BlastBox from:

```bash
python -c "import numpy, scipy, pygame; print('ok', numpy.__version__)"
```

If that errors, install into that interpreter (or make a project venv:
`uv venv` then `uv pip install numpy scipy pygame-ce pytest sympy matplotlib`).

---

## File map

| File | Role |
|---|---|
| `BlastBoxECLAUDE.py` | entry point: build the scene, run the ledger, open the window |
| `configMirror.py` | all constants, the ZRL spec-4 palette, `Scene`/`Box`/`Zone`/`GasConfig`, presets |
| `spaceMirror.py` | the space: walls/rooms as a signed-distance field |
| `fieldMirror.py` | the particle field: struct-of-arrays + jittered hex placement |
| `neighborsMirror.py` | neighbour search (cKDTree) behind a swappable interface |
| `dynamicsMirror.py` | **Stage 2**: contact forces, walls, the moving probe, velocity Verlet |
| `gaugesMirror.py` | **Stage 2**: temperature, pressure (two routes), speed moments, Knudsen |
| `unitsMirror.py` | **Stage 2**: relative units and the ratio calibration to SI |
| `ledgerMirror.py` | the live PASS/FAIL identity checks for both stages |
| `renderMirror.py` | pygame views + ledger panel (imported only when a window opens) |
| `testsCLAUDE/` | pytest suite (folder carries the CLAUDE marker; files keep the `test_` name pytest requires) |

Naming: generated modules take the `Mirror` suffix; the entry point keeps the
project name `BlastBoxECLAUDE.py` so it matches the Visual Studio startup file,
and test files keep pytest's `test_` prefix inside the CLAUDE-marked folder.

---

## Staging

Each stage ships a runnable app with its own readout and tests.

1. **Field & space (this stage).** Particle arrays, jittered hex placement, walls
   as an SDF, neighbour search, pygame view, live ledger.
2. **Molecular gas.** Elastic/soft-sphere collisions, a piston, a dragged disk.
   Pressure and temperature *emerge*; checks: 2D ideal-gas law, Maxwell speed
   distribution, measured sound speed, Mach-cone angle sin(mu) = 1/M.
3. **Compressible SPH.** Kernel density, ideal-gas EOS (gamma = 1.4), pressure
   force, Monaghan artificial viscosity, energy equation, CFL-limited timestep
   with a Co readout. Oracles: Sod shock tube, Sedov blast radius (2D: R proportional
   to t^(1/2)), energy ledger, time-reversal residual.
4. **Environments & instruments.** Material regions, water via a stiff EOS,
   absorbing boundaries, microphones (echo = 2d/c), schlieren view, multi-room
   presets, isotropy and interference checks.

Later: Berglund FDTD companion, Taichi GPU port, Kingery-Bulmash calibration
(free-air TNT only, never a room model), 3D.

---

## Stage 1 identity checks

The ledger (bottom of the panel, green PASS / red FAIL) proves the build rather
than asserting it:

1. **placement inside fluid** - no particle sits in a wall (min clearance > 0)
2. **within domain bounds** - all positions inside the world box
3. **mass ledger** - sum of masses finite and positive
4. **mean NN spacing ~ a** - jittered lattice still near its target constant
5. **number density (sanity band)** - measured density vs 2/(sqrt3 * a^2)
6. **kd-tree == brute force** - neighbour counts match an exact count on a sample

Checks 4-5 are approximate sanity bands (jitter plus disc-count discretization),
with deliberately loose tolerances; checks 1-3 and 6 are exact.

---

## Stage 2 - the molecular gas (built)

Particles fly freely and repel on contact through a linear spring,
`|F| = k(d - r)` for `r < d`. Velocity Verlet integrates them at a **frozen**
timestep; walls reflect them specularly off the signed-distance field. Press `D`
for a rigid disk dragged through the gas at Mach 1.6.

**Nothing sets pressure or temperature.** Both are measured:

- **Temperature** from 2D equipartition, `<KE> per particle = (D/2)kT = kT`,
  on *peculiar* velocities - bulk drift is flow, not heat.
- **Pressure two independent ways**: the internal virial
  `P = (NkT + (1/D) sum r.F)/A`, and momentum delivered to the walls per unit
  time per unit *length* (2D pressure is force per length). Their agreement is
  the stage's headline claim, graded against the interaction term rather than
  against the mean, so a band wider than the effect cannot hide a dropped term.
- **The 2D Maxwell speed ratio** `<v>^2/<v^2> = pi/4`, exact and parameter-free.
  The run deliberately starts NON-Maxwellian (every particle at the same speed,
  random direction, ratio 1.0) so this row measures relaxation rather than
  restating the initial draw.
- **Equation of state** against the Barker-Henderson closed form for this force
  law: a soft sphere acts as a smaller hard disc, `d_eff = d - sqrt(pi*T/2k)`,
  about 0.93 d at the shipped stiffness `k = 2e5`.

**Why k = 2e5.** Two soft spheres meeting head-on pass straight *through* each
other once they close faster than `u* = d*sqrt(2k/m)`. At the earlier `k = 2e4`
that was 8.4, and particles bounced off a Mach 1.6 probe (which leave with up to
`2V` extra) already crossed it in the fast tail. At `2e5` it is 26.6. The price is
a step about half as long, so each frame covers about half the simulated time;
raise `substeps` in `configMirror.py` if the frame rate has room.

Timestep, and what it is *not*: the lattice CFL bound (`1/sqrt(D)`, or
`sqrt(2/3)` on a triangular lattice) governs a wave solver on a grid and does not
apply here. A particle integrator is limited by the contact-spring period
`2*pi*sqrt(m/2k)` and by the closing-speed travel limit. Courant returns properly
in Stage 3 as `dt <= Co*h/(c_s + v_max)`. dt is frozen because velocity Verlet is
symplectic only at constant dt. It is re-derived only on an event: reset, probe
on/off, a speed change, or - with the probe running - the gas heating past the
speed dt was chosen for (a probe in a closed room heats the gas on every pass; the
panel counts these). The speed bound uses the *hottest local* temperature, so a
hot pocket's own thermal tail cannot outrun it, plus `2|V|` for probe bounces.

**The probe** is a rigid body with prescribed motion - infinitely massive, as if
pulled by a motor - not a soft or hard sphere. Particles bounce off it instantly,
like the walls. Set its speed with `Up`/`Down` (`Shift` x10) or press `S` and type
`2.5`, `500ms` or `1.6mach`; negative reverses it. The panel shows the speed in
relative units and its approximate m/s, the Mach number, the Knudsen number, the
drag and drag coefficient `C_d`, and draws the *predicted* Mach cone over the field.
Speed is capped by the contact law at `(u* - 6 sigma)/2` (about 10 at T = 1): a
bounced particle meeting a thermal one head-on must not reach `u*`. The cap falls
as the gas heats; if the probe ends up above it, the panel marks the cone and `C_d`
as not trustworthy. The probe wraps round *before* its leading edge reaches the
end wall, and re-enters from outside the far wall, so no gas is ever trapped
between it and a wall.

**Energy ledger.** Energy is not conserved while the probe runs, so the ledger
checks `E(t) - E(0) = work booked` instead. Each bounce off the probe changes a
particle's energy by exactly `V . (m dv)`, walls book `0.5 m (e^2 - 1) v_n^2`, and
inserting the probe books the energy of the gas it displaces. The error is graded
against the larger of the starting and current energy.

**Pressure comparison.** Both routes are averaged over the same undisturbed
interval and graded against the wall gauge's own counting noise,
`2/sqrt(pi*hits)`. The row reads "keep averaging" until that noise band is smaller
than the interaction term (measured by the *wall*, as wall minus ideal - not by the
route under test), then goes live and sharpens with time. The virial route carries
the exact **finite-window term** of the virial theorem: with
`G = sum m (r - origin) . v`, `dG/dt = 2K + sum r.F - P*D*A`, so over a window
`P = (2<K> + <sum r.F> - dG/tau) / (D*A)`. The `dG/tau` piece vanishes in a long
equilibrium average but not while the hot pocket's blast is still ringing round
the room, and it is included rather than waited out. Settling is set in time
(`equilibrate_time = 2.0`), not steps.

**Window.** Resizable; the panel wraps and scrolls (wheel over the panel,
`PgUp`/`PgDn`, `Home`) and never takes more than half the window; `-` and `=`
change the panel text size. Held arrow keys repeat; toggles do not.

Two honesty notes shown in the panel: the Mach cone angle is *asserted* from
`sin(mu) = 1/M`, which is a continuum result, and the Knudsen readout says the
shipped gas is rarefied - so expect a diffuse wake rather than a crisp wedge
until density rises. And the SI pressure is labelled `kPa-equiv`, because a 2D
line pressure and a 3D pressure are not the same dimension.

## ZRL -> BlastBox: notation and concept map

The 30/08 request was to carry what is useful from the Zero-Recursive Lattice into
physics-sim work, noting where ZRL notation differs from standard notation. In code
the convention is long names; the symbols below are for the docs and the readouts.

### Notation

| ZRL symbol | ZRL meaning | Standard / BlastBox name | Note |
|---|---|---|---|
| Co | CFL number | Courant number `C` | timestep stability ratio |
| C_max | max stable Co | `C_max` | scheme-dependent (see below) |
| h | tick-rate | timestep / smoothing length | **clash**: SPH `h` is the kernel radius; keep tick-rate as `dt` |
| D | spatial dimension count | `dim` (= 2) | flavour order stays `n` |
| v | CFL velocity | flow speed `u` or `v` | |
| rho (radial coord) | radial coordinate | **BlastBox uses rho for density** | resolve the clash: radial coord -> `r`, density -> `rho` |
| a_n | descent sequence | - | not used here |
| nabla^2 | Laplacian | `laplacian` | FDTD 5-point stencil |
| r_0 | escape saturates CFL | sonic horizon `|v| = c` | the Unruh acoustic-metric twin |

Also colliding once physics starts: `k` (Boltzmann k_B / wavenumber / loop index),
`S` (entropy), `T` (temperature vs period `P`), `n` (refractive index vs count).
Decide the name at first use; do not silently overload.

### What carries over

| ZRL machinery | Lands in |
|---|---|
| CFL canon: advection sum-form <= 1; **wave C_max = 1/sqrt(D)** (verified flip) | Stage 3 timestep + Co readout; Stage 1 uses none yet |
| Lattice units dx_0 = u_0 = 1 | nondimensionalisation; keep ambient c_0 distinct from the lattice speed limit |
| L1-diamond light cone (anisotropy) | Stage 4 isotropy check; the hex initial lattice reduces grid-aligned artefacts |
| r_0 = sonic horizon (Unruh) | Stage 4 "acoustic horizon" contour where flow speed = sound speed |
| Reversibility ledger (uncompute, never erase) | Stage 3 time-reversal residual + energy ledger - **undamped runs only**; shocks are irreversible (entropy must increase) |
| "Product invariants are blind to f<->1/f" (30/08) | pair every conserved-quantity check with a sign / monotonicity check |
| Live PASS/FAIL readout; verify-before-affirm | every stage; this is `ledgerMirror.py` |
| Cell record {position, crystal, phase, frozen} | material record {id, c, rho, absorption, wall} |
| Triangular-lattice / hex axial coordinates | Stage 1 particle placement (isotropic 2D init) |

### What does NOT carry over

Kept out on purpose: the varsigma-calculus, flavours, phases, isomers, Klein-group
elements, and the Schwarzschild metric factor f(rho) = 1 - r_s/rho. None has a
blast-sim meaning, and the ZRL canon says never resolve its open forks silently.
Kingery-Bulmash scaling, when it arrives, is free-air TNT data - a calibration
oracle, never the room model.

### Open, user-owned (do not pre-design)

- **Density as circle-cell overlap** (29/08 note): the geometric density encoding is
  tied to ZRL cell-geometry issue #2, which is still yours to define. Stage 1 uses a
  conventional per-particle density with a hook for it.
- One vault note (`Zero-Recursive Lattice.md`, 01/09) still carries the mixed-sign
  metric diagonal flagged 30/08 as the inverted form, and files "C_max = 1/sqrt(n)"
  under Advection (canon: advection = sum form, 1/sqrt(D) = wave). Reconcile the note
  with the canon before any acoustic-metric work in Stage 4.
