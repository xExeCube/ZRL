"""BlastBox - the live identity ledger (the anti-sycophancy readout in code form).

Every ZRL app proves its own invariants on screen rather than asserting them; the
same rule carries here. Stage 1 has no dynamics yet, so the checks are the things
that must be true of a freshly built field: particles are inside the fluid, the
mass ledger is finite, the lattice spacing and number density match the triangular
lattice, and the kd-tree neighbour counts agree exactly with a brute-force count.

Each check returns PASS/FAIL; the renderer paints them green/red.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Check:
    name: str
    value: str
    passed: bool
    detail: str = ""
    na: bool = False        # measured nothing: a domain/resolution limit, not a failure

    @property
    def tag(self) -> str:
        if self.na:
            return "n/a "
        return "PASS" if self.passed else "FAIL"


def skipped(name, reason):
    """A row that could not be measured. Per the ZRL ledger rules, a domain or
    resolution limit must NEVER be reported as a physics failure - it reports
    itself, with the reason."""
    return Check(name, reason, True, reason, na=True)


def run_checks(scene, world, field, neighbors):
    checks = []
    if field.n == 0:
        return [Check("field non-empty", "0 particles placed", False,
                      "every candidate was rejected by the world")]
    h = scene.smoothing_length
    pos = field.pos
    phi = world.clearance(pos)

    # 1. every particle sits in open fluid (placement invariant)
    ok = bool(np.all(phi > 0.0))
    checks.append(Check("placement inside fluid",
                        f"min clearance = {phi.min():.3f} m", ok,
                        "no particle inside a wall"))

    # 2. every particle is within the domain box
    w, h_dom = scene.world_size
    inb = bool(np.all((pos[:, 0] >= 0) & (pos[:, 0] <= w) &
                      (pos[:, 1] >= 0) & (pos[:, 1] <= h_dom)))
    checks.append(Check("within domain bounds", "yes" if inb else "no", inb))

    # 3. mass ledger is finite and positive
    msum = float(field.mass.sum())
    checks.append(Check("mass ledger", f"sum(m) = {msum:.3f}",
                        bool(np.isfinite(msum) and msum > 0.0),
                        f"{field.n} particles"))

    # 4. mean nearest-neighbour spacing ~ target lattice constant
    if field.n < 2:
        checks.append(skipped("mean NN spacing ~ a",
                              "fewer than two particles - no bond to measure"))
    else:
        nn = neighbors.nearest_dist()
        mean_nn = float(nn.mean())
        rel = abs(mean_nn - scene.spacing) / scene.spacing
        checks.append(Check("mean NN spacing ~ a",
                            f"{mean_nn:.4f} vs a={scene.spacing:.4f}  ({rel * 100:.1f}%)",
                            rel < 0.30,
                            "min-bond mean is biased below a under jitter"))

    # 5. measured number density ~ triangular-lattice value 2/(sqrt3 a^2)
    counts = neighbors.counts(h)
    interior = phi > h                          # far enough from a wall for a full disc
    dens_expect = 2.0 / (np.sqrt(3.0) * scene.spacing ** 2)
    if int(interior.sum()) == 0:
        # A resolution limit, not a physics failure: substituting a sentinel that
        # forces FAIL would print a red row reading "nan", which is exactly the
        # antipattern skipped() exists to prevent.
        checks.append(skipped("number density (sanity band)",
                              "no particle clears a full disc of radius h"))
    else:
        dens_meas = counts[interior].mean() / (np.pi * h * h)
        reld = abs(dens_meas - dens_expect) / dens_expect
        checks.append(Check("number density (sanity band)",
                            f"{dens_meas:.1f} vs {dens_expect:.1f} /m^2  ({reld * 100:.1f}%)",
                            reld < 0.25,
                            "disc-count band, not a strict density validation"))

    # 6. kd-tree neighbour counts == brute force on a random sample
    ok_tree, maxdiff, k = _verify_tree(pos, counts, h, seed=scene.seed)
    checks.append(Check("kd-tree == brute force",
                        f"max count difference = {maxdiff}", ok_tree,
                        f"radius neighbours, sample of {k}"))
    return checks


PRESSURE_ROW = "pressure: virial vs wall"
GAS_EQUILIBRIUM_ROWS = ("velocity isotropy", "Maxwell 2D speed ratio",
                        PRESSURE_ROW, "equation of state")
ENERGY_ROW = "energy ledger (dE = work)"
DT_ROW = "frozen dt still within limits"
NOISE_SIGMAS = 4.0          # pressure agreement band, in units of counting noise


def run_gas_checks(scene, world, state, readout, energy_ref=None, dt_limits=None,
                   work_ref=0.0):
    """Stage-2 ledger: what must be true of a molecular gas.

    Gating rules, all of them learned the hard way in the ZRL wave thread:
      * a row must never report a domain or resolution limit as a FAILURE;
      * an identically-zero field satisfies every linear identity, so rows that
        would pass on a dead simulation are gated on there being something to
        measure;
      * rows are graded against closed forms, not against bands picked to clear
        a known bias.
    """
    import numpy as np
    import gaugesMirror as G
    checks = []
    gas = scene.gas

    # --- forces -------------------------------------------------------------
    if readout.contacts == 0:
        checks.append(skipped("pair forces cancel (Newton 3)",
                              "no contacts this frame - nothing to measure"))
    else:
        net = state.mass[:, None] * state.accel
        net_sum = float(np.linalg.norm(net.sum(axis=0)))
        net_scale = float(np.abs(net).sum()) + 1e-30
        ratio = net_sum / net_scale
        checks.append(Check("pair forces cancel (Newton 3)",
                            f"|sum F| / sum|F| = {ratio:.2e}", ratio < 1e-9,
                            "exact: every contact contributes +f and -f"))
    # There is deliberately no "contacts stay shallow" row. Graded as "overlap
    # below one diameter" it can never fail (overlap = d - r with r > 0), and any
    # tighter band grades a stiffness choice: at equilibrium the deepest of ~N*n*d
    # contacts sits near half a diameter by pure statistics. The real failure - a
    # pair tunnelling through each other - dumps energy, which the energy ledger
    # catches. Max overlap is shown as a plain readout on the panel instead.

    # --- timestep -----------------------------------------------------------
    # The frozen dt is graded against the LIVE limits, so if the gas heats or the
    # probe speeds up past what dt was chosen for, this row says so. Comparing
    # min(a,b) to its own inputs, as an earlier version did, could never fail.
    # With the probe running the gas heats without limit, and dynamicsMirror.
    # advance re-derives dt whenever the bound is crossed - so the row could not
    # fail, and says so instead of printing a PASS that measured nothing.
    if dt_limits is not None:
        dt_contact, dt_travel = dt_limits
        limit = min(dt_contact, dt_travel)
        binding = "contact spring" if dt_contact <= dt_travel else "travel limit"
        if state.disk is not None:
            checks.append(skipped(DT_ROW,
                                  f"probe is heating the gas: dt re-derived "
                                  f"{state.dt_rederived}x, now {state.dt:.2e}"))
        else:
            checks.append(Check(DT_ROW,
                                f"dt={state.dt:.2e} vs live limit {limit:.2e} ({binding})",
                                state.dt <= limit,
                                "dt is fixed for symplecticity; this watches the live bound"))

    live = state.n > 0 and readout.temperature > 1e-12

    # --- energy ledger ------------------------------------------------------
    # Valid with or without the probe. Energy is not conserved while the probe
    # runs, but every joule it adds is booked exactly (V . momentum delivered),
    # as is anything an inelastic wall removes. So the check is not "E stays
    # constant" but "E changed by exactly the work that was done" - which
    # reduces to plain conservation when nothing external acts.
    if not live:
        checks.append(skipped(ENERGY_ROW, "gas is not live (T = 0 or no particles)"))
    elif energy_ref is None or abs(energy_ref) < 1e-12:
        checks.append(skipped(ENERGY_ROW, "no reference energy captured yet"))
    else:
        change = readout.total_energy - energy_ref
        work = readout.external_work - work_ref
        # Scaled by the LARGER of the start and current energy: once the probe
        # has pumped the gas to several times its starting energy, integration
        # error grows with it, and grading against E(0) alone would tighten the
        # band exactly as the numbers it grades get bigger.
        scale = max(abs(energy_ref), abs(readout.total_energy))
        err = abs(change - work) / scale
        checks.append(Check(ENERGY_ROW,
                            f"dE={change:+.2f}  work={work:+.2f}  (err {err * 100:.3f}%)",
                            err < 0.02,
                            "leapfrog conserves a MODIFIED energy: bounded wobble is correct"))

    # --- gates for the equilibrium identities -------------------------------
    if not live:
        for nm in GAS_EQUILIBRIUM_ROWS:
            checks.append(skipped(nm, "gas is not live (T = 0 or no particles)"))
        return checks
    if state.disk is not None:
        for nm in GAS_EQUILIBRIUM_ROWS:
            checks.append(skipped(nm, "driven by the probe - equilibrium "
                                      "identities do not apply"))
        return checks
    if state.time < state.settle_until:
        to_go = state.settle_until - state.time
        for nm in GAS_EQUILIBRIUM_ROWS:
            checks.append(skipped(nm, f"relaxing: {to_go:.2f} time units to go"))
        return checks

    # --- statistical mechanics ---------------------------------------------
    t_mean = 0.5 * (readout.temp_x + readout.temp_y)
    aniso = abs(readout.temp_x - readout.temp_y) / t_mean if t_mean > 0 else 1.0
    checks.append(Check("velocity isotropy",
                        f"Tx={readout.temp_x:.4f} Ty={readout.temp_y:.4f} ({aniso * 100:.1f}%)",
                        aniso < 0.10,
                        "no axis is special; a split means the lattice is showing through"))

    dev = abs(readout.maxwell_ratio - G.MAXWELL_2D_RATIO)
    checks.append(Check("Maxwell 2D speed ratio",
                        f"<v>^2/<v^2> = {readout.maxwell_ratio:.4f} vs pi/4 = "
                        f"{G.MAXWELL_2D_RATIO:.4f}",
                        dev < 0.02,
                        "started at 1.000 by construction; this measures relaxation"))

    # --- pressure, two independent ways ------------------------------------
    # Both routes are time-averaged over the SAME undisturbed interval, both
    # carry the exact finite-window term (gaugesMirror.read), and the agreement
    # band is set by the wall gauge's own counting noise rather than by hand.
    # Two conditions, because agreement alone proves little:
    #   resolvable - the noise band must be SMALLER than the interaction term,
    #                or the row could not tell a real gas from one whose virial
    #                term had been dropped. The term is measured by the WALL
    #                (wall minus ideal), not by the route under test: sized from
    #                the virial itself, a dropped virial would shrink the
    #                "effect" to zero and hold the row at n/a for ever.
    #   agree      - the two routes then differ by less than the noise band.
    # The band shrinks as 1/sqrt(hits), so this row gets sharper the longer the
    # gas is left undisturbed.
    p_vir, p_wall, p_id = readout.p_virial_avg, readout.p_wall_avg, readout.p_ideal_avg
    if readout.avg_hits <= 0 or not (np.isfinite(p_vir) and np.isfinite(p_id)):
        checks.append(skipped(PRESSURE_ROW, "averaging has not started"))
    elif not np.isfinite(p_wall) or p_wall <= 0:
        checks.append(Check(PRESSURE_ROW,
                            f"wall pressure {p_wall:.3f} from {readout.avg_hits} hits",
                            False, "hits were counted but no push was recorded: "
                                   "the wall gauge itself is broken"))
    else:
        effect = abs(p_wall - p_id)
        band = NOISE_SIGMAS * readout.wall_noise * p_wall
        diff = abs(p_vir - p_wall)
        summary = (f"{p_vir:.3f} vs {p_wall:.3f}  diff {diff:.3f}  "
                   f"band {band:.3f}  effect {effect:.3f}")
        if band >= effect:
            checks.append(skipped(PRESSURE_ROW,
                                  f"averaging ({readout.avg_hits} hits): noise band "
                                  f"{band:.2f} still exceeds effect {effect:.2f}"))
        else:
            checks.append(Check(PRESSURE_ROW, summary, diff < band,
                                "independent routes, graded against their own noise"))

    # --- equation of state --------------------------------------------------
    # Graded against the Barker-Henderson closed form for THIS force law, with a
    # band sized to the non-ideal effect, plus a floor so a non-interacting gas
    # (Z = 1 exactly) cannot pass.
    expected_excess = readout.z_expected - 1.0
    measured_excess = readout.z_measured - 1.0
    if expected_excess <= 0 or not np.isfinite(readout.z_measured):
        checks.append(skipped("equation of state", "no measurable excess pressure"))
    else:
        close = abs(measured_excess - expected_excess) < 0.35 * expected_excess
        real = measured_excess > 0.5 * expected_excess
        checks.append(Check("equation of state",
                            f"Z={readout.z_measured:.4f} vs hard-disc(d_eff) "
                            f"{readout.z_expected:.4f} (phi_eff={readout.phi_eff:.4f})",
                            close and real,
                            "closed form for a linear spring, not a hand-picked band"))
    return checks



def _verify_tree(pos, tree_counts, radius, seed=0, k=64):
    rng = np.random.default_rng(seed + 1)
    n = pos.shape[0]
    k = min(k, n)
    idx = rng.choice(n, size=k, replace=False)
    maxdiff = 0
    for i in idx:
        d = np.hypot(pos[:, 0] - pos[i, 0], pos[:, 1] - pos[i, 1])
        brute = int(np.count_nonzero(d <= radius + 1e-12)) - 1  # exclude self; eps guards ties
        maxdiff = max(maxdiff, abs(brute - int(tree_counts[i])))
    return maxdiff == 0, maxdiff, k
