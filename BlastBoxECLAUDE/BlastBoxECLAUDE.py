"""BlastBox - entry point.

Usage:
    python BlastBoxECLAUDE.py                 # Stage 2: the molecular gas (default)
    python BlastBoxECLAUDE.py open_box        # a named scene
    python BlastBoxECLAUDE.py --stage1        # Stage 1: the static particle field
    python BlastBoxECLAUDE.py --checks        # headless: print both ledgers, no window

Run the test suite from this folder:
    python -m pytest          (or: uv run python -m pytest)
"""
from __future__ import annotations
import sys
import configMirror as C
import dynamicsMirror as D
import gaugesMirror as G
from fieldMirror import build_field
from neighborsMirror import NeighborSearch
from ledgerMirror import run_checks, run_gas_checks

FLAGS = {"--stage1", "--checks"}
# Length of the measurement window after relaxation, in simulation TIME (a step
# count would change meaning whenever dt does). The walls take ~1400 hits per
# time unit, so 4.0 gives ~5600 hits: the 4-sigma noise band is then ~6% of
# the pressure against an interaction term of ~14% - resolved with a margin of
# about 2.3x, not the 1.3x the earlier 6000-step window had. Expect the whole
# headless run to take a couple of minutes.
MEASURE_TIME = 4.0


def build_stage1(scene):
    """Build the world, place the field, index neighbours, run the Stage-1 ledger."""
    world = scene.make_world()
    field, info = build_field(scene, world)
    neighbors = NeighborSearch(field.pos)
    checks = run_checks(scene, world, field, neighbors)
    return world, field, neighbors, checks, info


def build_stage2(scene, world, field):
    """Promote the placed field into a gas state with its initial velocities."""
    return D.make_state(scene, world, field)


def _print(title, checks):
    print(f"\n{title}")
    for c in checks:
        print(f"  [{c.tag}] {c.name}: {c.value}")
    # n/a rows are not failures - they report that nothing was measured.
    return all(c.passed for c in checks)


def main(argv):
    names = [a for a in argv[1:] if not a.startswith("-")]
    flags = {a for a in argv[1:] if a.startswith("-")}

    unknown_flags = flags - FLAGS
    if unknown_flags:
        print(f"[BlastBox] unknown flag(s): {', '.join(sorted(unknown_flags))}")
        print(f"           known flags: {', '.join(sorted(FLAGS))}")
        return 2
    if names and names[0] not in C.SCENES:
        print(f"[BlastBox] unknown scene '{names[0]}'")
        print(f"           known scenes: {', '.join(sorted(C.SCENES))}")
        return 2

    scene = C.SCENES[names[0] if names else C.DEFAULT_SCENE]()

    world, field, neighbors, checks, info = build_stage1(scene)
    print(f"[BlastBox] scene={scene.name}  requested={info['requested']}  "
          f"placed={info['placed']}  rejected={info['rejected']}")
    ok = _print("Stage 1 - field & space", checks)

    if "--stage1" in flags:
        if "--checks" in flags:
            return 0 if ok else 1
        try:
            from renderMirror import run_view
        except Exception as exc:
            print(f"[BlastBox] window unavailable ({exc}). Ran checks only.")
            return 0 if ok else 1
        run_view(scene, world, field, neighbors, checks)
        return 0

    state = build_stage2(scene, world, field)

    if "--checks" in flags:
        # References are taken at t = 0, BEFORE anything runs, and graded at the
        # end: the ledger then covers the whole run. (Taking them from the final
        # readout would grade a readout against itself, which can never fail.)
        energy_ref = state.total_energy()
        work_ref = state.external_work
        D.advance_time(state, scene, world, scene.gas.equilibrate_time)
        state.reset_window()
        D.advance_time(state, scene, world, MEASURE_TIME)

        readout = G.read(state, scene, world)
        limits = D.dt_limits(scene, state)
        gas_checks = run_gas_checks(scene, world, state, readout, energy_ref, limits,
                                    work_ref)
        ok2 = _print("Stage 2 - molecular gas", gas_checks)
        print(f"\n  T={readout.temperature:.4f}  Z={readout.z_measured:.4f} "
              f"(expected {readout.z_expected:.4f})  phi_eff={readout.phi_eff:.4f}  "
              f"c={readout.sound_speed:.4f}  Kn={readout.knudsen:.3f}")
        print(f"  averaged over {readout.avg_time:.2f} time units, {readout.avg_hits} "
              f"wall hits: P_virial={readout.p_virial_avg:.4f}  "
              f"P_wall={readout.p_wall_avg:.4f}  P_ideal={readout.p_ideal_avg:.4f}  "
              f"window term={readout.p_window:+.4f}  "
              f"noise={readout.wall_noise * 100:.2f}%   [relative units]")
        print(f"  {state.steps} steps at dt={state.dt:.2e}")
        return 0 if (ok and ok2) else 1

    try:
        from renderMirror import run_gas_view
    except Exception as exc:
        print(f"[BlastBox] window unavailable ({exc}). Ran Stage-1 checks only.")
        return 0 if ok else 1
    run_gas_view(scene, world, state, lambda: build_stage2(scene, world, field))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
