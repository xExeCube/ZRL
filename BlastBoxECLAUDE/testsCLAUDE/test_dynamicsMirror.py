import numpy as np
import pytest
import dynamicsMirror as D
import gaugesMirror as G
import configMirror as C
from configMirror import open_box, Box
from spaceMirror import World
from fieldMirror import build_field


def _gas(scene=None):
    scene = scene or open_box()
    world = scene.make_world()
    field, _ = build_field(scene, world)
    return scene, world, D.make_state(scene, world, field)


def _small():
    """A 3 x 2 box of ~350 particles and no hot pocket: big enough for the
    probe (radius 0.45, on the mid-line) and cheap enough to run for a while."""
    return _gas(C.Scene(name="small", world_size=(3.0, 2.0)))


# --- initial conditions ------------------------------------------------------

def test_maxwell_hits_the_requested_temperature():
    rng = np.random.default_rng(0)
    n = 4000
    mass = np.ones(n)
    vel = D.maxwell_velocities(n, temperature=2.5, mass=mass, rng=rng)
    assert abs(D.temperature_of(vel, mass) - 2.5) < 1e-9


def test_uniform_speed_ic_starts_far_from_maxwell():
    """The default IC: <v>^2/<v^2> starts at 1, not pi/4, so the ledger's Maxwell
    row measures relaxation instead of the initial draw."""
    import gaugesMirror as G
    rng = np.random.default_rng(2)
    n = 5000
    mass = np.ones(n)
    vel = D.uniform_speed_velocities(n, temperature=1.0, mass=mass, rng=rng)
    _, _, ratio = G.speed_moments(vel, mass)
    assert ratio > 0.99
    assert abs(ratio - G.MAXWELL_2D_RATIO) > 0.2
    assert abs(D.temperature_of(vel, mass) - 1.0) < 1e-9


def test_per_particle_temperature_is_honoured():
    rng = np.random.default_rng(5)
    n = 4000
    mass = np.ones(n)
    temps = np.where(np.arange(n) < n // 2, 1.0, 4.0)
    vel = D.uniform_speed_velocities(n, temps, mass, rng)
    cold = D.temperature_of(vel[: n // 2], mass[: n // 2])
    hot = D.temperature_of(vel[n // 2:], mass[n // 2:])
    assert hot > 2.5 * cold


def test_maxwell_has_no_net_momentum():
    rng = np.random.default_rng(1)
    n = 2000
    mass = np.full(n, 1.5)
    vel = D.maxwell_velocities(n, temperature=1.0, mass=mass, rng=rng)
    assert np.linalg.norm((mass[:, None] * vel).sum(axis=0)) < 1e-9


def test_temperature_ignores_bulk_flow():
    rng = np.random.default_rng(4)
    n = 3000
    mass = np.ones(n)
    vel = D.maxwell_velocities(n, temperature=1.7, mass=mass, rng=rng)
    t0 = D.temperature_of(vel, mass)
    t1 = D.temperature_of(vel + np.array([3.0, -1.0]), mass)
    assert abs(t1 - t0) < 1e-9


# --- forces ------------------------------------------------------------------

def test_pair_force_is_repulsive_and_cancels():
    pos = np.array([[0.0, 0.0], [0.03, 0.0]])
    acc, pe, virial, contacts, overlap = D.pair_forces(pos, np.ones(2), 0.05, 1000.0)
    assert contacts == 1
    assert acc[0, 0] < 0 and acc[1, 0] > 0
    assert np.allclose(acc.sum(axis=0), 0.0, atol=1e-12)
    assert pe > 0 and virial > 0
    assert abs(overlap - 0.02) < 1e-12


def test_no_force_beyond_the_cutoff():
    pos = np.array([[0.0, 0.0], [1.0, 0.0]])
    acc, pe, virial, contacts, overlap = D.pair_forces(pos, np.ones(2), 0.05, 1000.0)
    assert contacts == 0 and pe == 0.0 and virial == 0.0
    assert np.allclose(acc, 0.0)


# --- walls -------------------------------------------------------------------

def test_wall_reflection_mirrors_the_penetration_depth():
    world = World(Box(0.0, 0.0, 10.0, 10.0))
    pos = np.array([[0.005, 5.0]])          # 0.015 inside the reflecting surface
    vel = np.array([[-2.0, 1.0]])
    impulse, energy, hits = D.reflect_walls(pos, vel, np.ones(1), world, radius=0.02)
    assert vel[0, 0] > 0 and abs(vel[0, 1] - 1.0) < 1e-12
    assert impulse > 0 and hits == 1
    assert energy == 0.0                     # elastic: no energy removed
    assert abs(pos[0, 0] - 0.035) < 1e-9     # mirrored, not snapped


def test_wall_leaves_outgoing_particles_completely_alone():
    world = World(Box(0.0, 0.0, 10.0, 10.0))
    pos = np.array([[0.005, 5.0]])
    vel = np.array([[+2.0, 0.0]])
    pos_before, vel_before = pos.copy(), vel.copy()
    impulse, energy, hits = D.reflect_walls(pos, vel, np.ones(1), world, radius=0.02)
    assert impulse == 0.0 and energy == 0.0 and hits == 0
    assert np.allclose(vel, vel_before) and np.allclose(pos, pos_before)


def test_inelastic_wall_reports_exactly_the_energy_it_removes():
    """The ledger books wall losses; the booked number must be the real one."""
    world = World(Box(0.0, 0.0, 10.0, 10.0))
    pos = np.array([[0.005, 5.0]])
    vel = np.array([[-2.0, 0.7]])
    mass = np.array([1.3])
    ke_before = 0.5 * mass[0] * float(vel[0] @ vel[0])
    _, energy, _ = D.reflect_walls(pos, vel, mass, world, radius=0.02, restitution=0.5)
    ke_after = 0.5 * mass[0] * float(vel[0] @ vel[0])
    assert energy < 0
    assert abs((ke_after - ke_before) - energy) < 1e-12


# --- the probe ---------------------------------------------------------------

def test_disk_reflects_in_its_own_rest_frame():
    disk = D.MovingDisk(centre=np.array([0.0, 0.0]), radius=1.0,
                        velocity=np.array([1.0, 0.0]))
    pos = np.array([[1.005, 0.0]])
    vel = np.array([[-1.0, 0.0]])
    delivered = disk.collide(pos, vel, np.ones(1), particle_radius=0.01)
    # disk frame: arrives at -2, leaves at +2  ->  lab frame: leaves at +3
    assert abs(vel[0, 0] - 3.0) < 1e-9
    assert np.allclose(delivered, [4.0, 0.0])


def test_probe_work_identity_is_exact():
    """Each bounce preserves speed in the disk's frame, so the particle's lab KE
    changes by exactly V . (m dv). This is what makes the energy ledger exact."""
    rng = np.random.default_rng(9)
    disk = D.MovingDisk(centre=np.array([0.0, 0.0]), radius=1.0,
                        velocity=np.array([1.7, -0.4]))
    n = 200
    ang = rng.uniform(0, 2 * np.pi, n)
    pos = np.column_stack([np.cos(ang), np.sin(ang)]) * 1.004
    vel = rng.normal(0, 1.0, (n, 2))
    mass = rng.uniform(0.5, 2.0, n)
    ke0 = 0.5 * float(np.sum(mass * np.sum(vel * vel, axis=1)))
    delivered = disk.collide(pos, vel, mass, particle_radius=0.01)
    ke1 = 0.5 * float(np.sum(mass * np.sum(vel * vel, axis=1)))
    assert np.linalg.norm(delivered) > 0
    assert abs((ke1 - ke0) - float(delivered @ disk.velocity)) < 1e-9


def test_attach_relocates_the_gas_it_lands_on_and_books_the_cost():
    scene, world, state = _gas()
    gas, dom = scene.gas, world.domain
    centre = np.array([dom.x0 + 1.5 * gas.disk_radius, 0.5 * (dom.y0 + dom.y1)])
    reach = gas.disk_radius + 0.5 * gas.diameter
    inside = np.flatnonzero(np.linalg.norm(state.pos - centre, axis=1) < reach)
    assert inside.size > 10                                   # there really was gas there
    e0, w0 = state.total_energy(), state.external_work

    D.attach_disk(state, scene, world, speed=1.5)
    assert np.allclose(state.disk.centre, centre)
    r = np.linalg.norm(state.pos - state.disk.centre, axis=1)
    assert np.all(r >= reach - 1e-12)                         # nobody left inside
    band = max(gas.diameter, scene.spacing)
    assert np.all(r[inside] <= reach + band + 1e-12)          # set just outside, not flung
    booked = state.external_work - w0
    assert booked != 0.0                                      # moving gas has a cost...
    assert abs((state.total_energy() - e0) - booked) < 1e-9   # ...and it is the real one


def test_speed_control_is_capped_by_the_contact_law():
    scene, world, state = _gas()
    D.attach_disk(state, scene, world, speed=1.0)
    dt_slow = state.dt
    cap = D.speed_cap(scene, state)
    v_star = G.tunnel_speed(scene.gas.diameter, scene.gas.stiffness,
                            float(state.mass.max()))
    assert 0.0 < cap < 0.5 * v_star      # a bounce adds up to 2V, so 2*cap stays under v*
    applied = D.set_disk_speed(state, scene, 100.0)
    assert applied == cap                                     # clamped
    assert state.disk.velocity[0] == applied
    assert state.dt < dt_slow                                 # faster probe, smaller dt
    assert D.set_disk_speed(state, scene, -3.0) == -3.0       # sign reverses direction


def test_clamp_rejects_non_finite_speeds():
    scene = open_box()
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            D.clamp_speed(scene, bad)


def test_tunnel_speed_separates_bounce_from_pass_through():
    """Graded against the integrator itself: two particles meeting head-on just
    below u* = d*sqrt(2k/m) must bounce, and just above it must pass through."""
    scene = _tiny_scene()
    world = scene.make_world()
    u_star = G.tunnel_speed(scene.gas.diameter, scene.gas.stiffness, 1.0)
    for factor, bounces in ((0.9, True), (1.1, False)):
        u = factor * u_star
        pos = np.array([[4.7, 5.0], [5.3, 5.0]])
        vel = np.array([[0.5 * u, 0.0], [-0.5 * u, 0.0]])
        state = D.GasState(pos=pos, vel=vel, mass=np.ones(2), accel=np.zeros_like(pos))
        D.refresh_forces(state, scene)
        D.freeze_dt(state, scene)
        D.advance(state, scene, world, int(round(0.6 / (u * state.dt))) + 200)
        assert state.contacts == 0                            # the encounter is over
        assert (state.vel[0, 0] < 0) == bounces, f"u = {factor} u*"


def test_probe_wraps_before_its_leading_edge_reaches_the_end_wall():
    """Regression: the probe used to wrap only once its TRAILING edge had left
    the room, ploughing gas into the end wall on the way."""
    world = World(Box(0.0, 0.0, 10.0, 4.0))
    margin, reach = 0.2, 0.55
    for vx in (+1.0, -1.0):
        disk = D.MovingDisk(centre=np.array([5.0, 2.0]), radius=0.5,
                            velocity=np.array([vx, 0.0]))
        closest = np.inf
        for _ in range(2000):
            if disk.advance(0.01, world, margin=margin, reach=reach):
                break
            if vx > 0:
                closest = min(closest, world.domain.x1 - (disk.centre[0] + disk.radius))
            else:
                closest = min(closest, (disk.centre[0] - disk.radius) - world.domain.x0)
        assert disk.wraps == 1
        assert closest > margin - 1e-9                        # never nearer than the gap
        expected = world.domain.x0 - reach if vx > 0 else world.domain.x1 + reach
        assert disk.centre[0] == pytest.approx(expected)      # re-enters from outside


def _ledger_error(state, e0, w0):
    work = state.external_work - w0
    change = state.total_energy() - e0
    return work, abs(change - work)


def test_energy_ledger_balances_through_a_full_pass_and_wrap():
    """E(t) - E(0) must equal the work booked, graded against the WORK (not the
    starting energy, which would let a small error hide in a big number), over
    a run long enough for the probe to reach the end wall and wrap."""
    scene, world, state = _small()
    e0, w0 = state.total_energy(), state.external_work        # includes insertion
    D.attach_disk(state, scene, world, speed=3.0)
    batches = 0
    while state.disk.wraps == 0 and batches < 2000:
        D.advance(state, scene, world, 12)
        batches += 1
    assert state.disk.wraps == 1, "the probe never reached the end wall"
    for _ in range(40):                                       # and re-enters the room
        D.advance(state, scene, world, 12)
    work, err = _ledger_error(state, e0, w0)
    assert work > 0.5 * e0             # the probe really did work on the gas
    assert err < 0.02 * work
    assert np.all(np.isfinite(state.pos)) and np.all(np.isfinite(state.vel))
    assert np.all(world.clearance(state.pos) > 0.0)           # nobody pushed into a wall


def test_energy_ledger_survives_reversal_and_speed_changes():
    scene, world, state = _small()
    e0, w0 = state.total_energy(), state.external_work
    D.attach_disk(state, scene, world, speed=2.0)
    # forward, then reversed back through its own wake and out through the near
    # wall (it starts only ~0.2 from the wrap point), then fast the other way
    for speed, batches in ((2.0, 40), (-2.5, 120), (3.5, 60)):
        D.set_disk_speed(state, scene, speed)
        for _ in range(batches):
            D.advance(state, scene, world, 12)
    work, err = _ledger_error(state, e0, w0)
    assert state.disk.wraps >= 1                # the reversal carried it through a wall
    assert work > 0.3 * e0
    assert err < 0.02 * work


def test_detaching_the_probe_restarts_the_settling_clock():
    scene, world, state = _small()
    D.attach_disk(state, scene, world, speed=2.0)
    D.advance(state, scene, world, 24)
    assert state.avg_time == 0.0                              # driven: nothing averaged
    D.detach_disk(state, scene)
    assert state.settle_until == pytest.approx(state.time + scene.gas.equilibrate_time)
    D.advance(state, scene, world, 24)
    assert state.avg_time == 0.0                              # still settling
    state.settle_until = state.time + 0.5 * state.dt          # cut the wait short
    D.advance(state, scene, world, 24)
    assert state.avg_time == pytest.approx(23 * state.dt)     # 1 step opens the window


def test_frozen_dt_anticipates_the_probe_bounce():
    """Regression: dt was frozen for the thermal speed plus V, but a bounce off
    the probe adds up to 2V, so the live check failed as soon as the first
    particles bounced. Plain steps here, so nothing is re-derived behind it."""
    scene, world, state = _gas()
    D.attach_disk(state, scene, world, speed=4.0)
    for _ in range(120):
        D.step(state, scene, world)
    assert state.drag_impulse > 0                             # particles did bounce
    assert state.dt <= min(D.dt_limits(scene, state))


def test_running_probe_rederives_dt_only_as_an_event():
    scene, world, state = _small()
    D.attach_disk(state, scene, world, speed=3.0)
    for _ in range(100):
        dt_before = state.dt
        D.advance(state, scene, world, 12)
        if state.dt != dt_before:
            assert state.dt < dt_before                       # only ever tightens
        assert state.dt <= min(D.dt_limits(scene, state))     # and is never left stale


# --- integration -------------------------------------------------------------

def _tiny_scene():
    scene = C.Scene(name="tiny", world_size=(10.0, 10.0))
    scene.gas = C.GasConfig(diameter=0.05, stiffness=2000.0, temperature=1.0)
    return scene


def _pair(scene, offset):
    """Two particles 0.4 apart closing head-on at 0.8, each pulled back by `offset`."""
    pos = np.array([[4.8 - offset, 5.0], [5.2 + offset, 5.0]])
    vel = np.array([[0.4, 0.0], [-0.4, 0.0]])
    state = D.GasState(pos=pos, vel=vel, mass=np.ones(2), accel=np.zeros_like(pos))
    D.refresh_forces(state, scene)
    D.freeze_dt(state, scene)
    return state


def _collision_energy_error(scene, world, dt, offset):
    state = _pair(scene, offset)
    state.dt = dt
    e0 = state.total_energy()
    D.advance(state, scene, world, int(round(1.0 / dt)))
    assert state.contacts == 0      # separated again: all the energy is kinetic
    return abs(state.total_energy() - e0) / abs(e0)


def test_collision_energy_error_shrinks_with_the_timestep():
    """Graded as a refinement trend: velocity Verlet is second order, so halving
    dt should cut the error by about four. A leak would refuse to shrink.

    The error of one collision depends on WHERE in a step contact begins, so a
    single run can land on a lucky phase. Each dt is run from six starting
    offsets spread across one step of closing travel, and the worst case (the
    envelope) is what gets compared. Its scale is (omega*dt)^2, omega being the
    pair's contact frequency."""
    scene = _tiny_scene()
    world = scene.make_world()
    dt = _pair(scene, 0.0).dt
    omega = np.sqrt(2.0 * scene.gas.stiffness / 1.0)
    offsets = [0.4 * dt * k / 6.0 for k in range(6)]
    coarse = max(_collision_energy_error(scene, world, dt, s) for s in offsets)
    fine = max(_collision_energy_error(scene, world, 0.5 * dt, s) for s in offsets)
    assert coarse < (omega * dt) ** 2, f"per-collision error {coarse:.2e} is larger than expected"
    assert fine < 0.5 * coarse, f"halving dt only improved {coarse:.2e} -> {fine:.2e}"


def test_energy_bounded_over_many_rebatched_calls():
    scene, world, state = _gas()
    e0, dt0 = state.total_energy(), state.dt
    for _ in range(40):
        D.advance(state, scene, world, 12)
    assert state.dt == dt0, "dt must stay frozen across batches"
    assert abs(state.total_energy() - e0) / abs(e0) < 0.02


def test_frozen_dt_respects_both_limits():
    scene, world, state = _gas()
    dt_contact, dt_travel = D.dt_limits(scene, state)
    assert 0 < state.dt <= dt_contact + 1e-18
    assert state.dt <= dt_travel + 1e-18


def test_averages_accumulate_only_once_settled():
    scene, world, state = _gas()
    # settle mid-way between steps 50 and 51, so no float tie decides anything
    state.settle_until = state.time + 50.5 * state.dt
    D.advance(state, scene, world, 40)
    assert state.avg_time == 0.0 and state.avg_g0 is None    # still settling
    D.advance(state, scene, world, 160)
    # step 51 opens the window and records G; steps 52..200 are averaged
    assert state.avg_g0 is not None
    assert state.avg_time == pytest.approx(149 * state.dt)
    assert state.avg_wall_hits > 0 and state.avg_wall_impulse > 0
    assert state.avg_kinetic == pytest.approx(state.kinetic_energy() * state.avg_time,
                                              rel=0.05)   # energy is conserved, so ~flat


def test_finite_window_term_is_exact_in_free_flight():
    """Closed form for the virial theorem's finite-window term. Particles that
    never touch anything have dG/dt = 2K exactly, so over any window the term
    must cancel the kinetic part completely: ideal-gas pressure 0, matching a
    wall that has felt nothing."""
    scene = _tiny_scene()
    world = scene.make_world()
    pos = np.array([[4.0, 4.0], [6.0, 4.0], [4.0, 6.0], [6.0, 6.0]])
    vel = 0.5 * (pos - 5.0) / np.linalg.norm(pos - 5.0, axis=1)[:, None]  # outward
    state = D.GasState(pos=pos, vel=vel, mass=np.ones(4), accel=np.zeros_like(pos))
    D.refresh_forces(state, scene)
    D.freeze_dt(state, scene)
    state.settle_until = -1.0                                 # average from the start
    D.advance(state, scene, world, 200)
    r = G.read(state, scene, world)
    kinetic_part = 2.0 * state.kinetic_energy() / (G.DIM * r.area)
    assert r.avg_hits == 0 and r.p_wall_avg == 0.0            # nothing reached a wall
    assert r.p_window == pytest.approx(kinetic_part, rel=1e-9)
    assert abs(r.p_ideal_avg) < 1e-9 * kinetic_part
