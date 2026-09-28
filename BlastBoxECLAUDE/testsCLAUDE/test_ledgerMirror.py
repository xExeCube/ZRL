from dataclasses import replace
import configMirror as C
import dynamicsMirror as D
import gaugesMirror as G
from configMirror import open_box, two_rooms
from fieldMirror import build_field
from neighborsMirror import NeighborSearch
from ledgerMirror import run_checks, run_gas_checks, PRESSURE_ROW, DT_ROW, ENERGY_ROW


def _checks_for(scene_builder):
    s = scene_builder()
    w = s.make_world()
    field, _ = build_field(s, w)
    ns = NeighborSearch(field.pos)
    return run_checks(s, w, field, ns)


def test_all_checks_pass_open_box():
    failed = [c.name for c in _checks_for(open_box) if not c.passed]
    assert failed == [], f"unexpected failures: {failed}"


def test_all_checks_pass_two_rooms():
    failed = [c.name for c in _checks_for(two_rooms) if not c.passed]
    assert failed == [], f"unexpected failures: {failed}"


# --- Stage 2 rows, driven with synthetic readouts ----------------------------
# Each branch of the pressure row is exercised on purpose, including the ones a
# healthy run never reaches: a row that has never been seen to FAIL has not
# been shown to be able to.

def _settled_small_gas():
    scene = C.Scene(name="small", world_size=(3.0, 2.0))
    world = scene.make_world()
    field, _ = build_field(scene, world)
    state = D.make_state(scene, world, field)
    state.settle_until = -1.0          # past the settling gate: the rows below are synthetic
    return scene, world, state


def _row(rows, name):
    return next(c for c in rows if c.name == name)


def _pressure_row(p_ideal, p_vir, p_wall, hits):
    scene, world, state = _settled_small_gas()
    base = G.read(state, scene, world)
    readout = replace(base, p_ideal_avg=p_ideal, p_virial_avg=p_vir, p_wall_avg=p_wall,
                      avg_hits=hits, wall_noise=G.wall_noise(hits))
    return _row(run_gas_checks(scene, world, state, readout), PRESSURE_ROW)


def test_pressure_row_waits_until_averaging_starts():
    row = _pressure_row(float("nan"), float("nan"), float("nan"), 0)
    assert row.na


def test_pressure_row_waits_while_noise_hides_the_effect():
    # 100 hits: the 4-sigma band is ~45% of P, far wider than a 10% effect
    row = _pressure_row(60.0, 66.0, 66.3, 100)
    assert row.na and "exceeds" in row.value


def test_pressure_row_passes_when_resolved_and_agreeing():
    # 1e5 hits: band ~0.95 against an effect of 6.3; the routes differ by 0.3
    row = _pressure_row(60.0, 66.0, 66.3, 100_000)
    assert not row.na and row.passed


def test_pressure_row_fails_when_the_virial_term_is_dropped():
    """The case the independent gate exists for. With the virial route reading
    only the ideal-gas part, sizing the effect from that route would give zero
    and hold the row at n/a for ever. Sized from the wall, it is caught."""
    row = _pressure_row(60.0, 60.0, 66.3, 100_000)
    assert not row.na and not row.passed


def test_pressure_row_fails_when_hits_deliver_no_push():
    row = _pressure_row(60.0, 66.0, 0.0, 5_000)
    assert not row.na and not row.passed


def test_dt_row_is_live_without_the_probe_and_na_with_it():
    scene, world, state = _settled_small_gas()
    rows = run_gas_checks(scene, world, state, G.read(state, scene, world),
                          dt_limits=D.dt_limits(scene, state))
    row = _row(rows, DT_ROW)
    assert not row.na and row.passed
    D.attach_disk(state, scene, world, speed=2.0)
    rows = run_gas_checks(scene, world, state, G.read(state, scene, world),
                          dt_limits=D.dt_limits(scene, state))
    assert _row(rows, DT_ROW).na


def test_dt_row_fails_when_the_gas_outruns_the_frozen_dt():
    scene, world, state = _settled_small_gas()
    state.vel[0] = [50.0, 0.0]          # one particle far beyond the bound dt was frozen for
    rows = run_gas_checks(scene, world, state, G.read(state, scene, world),
                          dt_limits=D.dt_limits(scene, state))
    row = _row(rows, DT_ROW)
    assert not row.na and not row.passed


def test_energy_row_is_graded_against_the_larger_energy():
    """Once the probe has pumped the gas to several times its starting energy,
    a fixed error should not read as proportionally worse just because E(0)
    was small."""
    scene, world, state = _settled_small_gas()
    base = G.read(state, scene, world)
    e0 = base.total_energy
    # the gas now holds 5x its starting energy, all but 1.5% of it booked
    readout = replace(base, total_energy=5.0 * e0, external_work=4.0 * e0 - 0.075 * e0)
    row = _row(run_gas_checks(scene, world, state, readout, energy_ref=e0), ENERGY_ROW)
    assert not row.na and row.passed     # 0.075 e0 over 5 e0 = 1.5%, not 7.5%
