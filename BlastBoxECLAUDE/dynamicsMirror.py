"""BlastBox Stage 2 - molecular gas: contact forces, walls, the probe, the integrator.

The model is a SOFT-SPHERE gas. Particles fly freely and, when two of them
overlap, push apart with a linear spring:

    |F| = k (d - r)      for r < d,    0 otherwise
    U(r) = 0.5 k (d - r)^2

The force goes to zero continuously at the cutoff, so there is no impulse
discontinuity for the integrator to trip over, and the pair potential is exact,
which lets the ledger check energy.

Nothing here knows about pressure or temperature. Those are MEASURED from the
motion in gaugesMirror.py, which is the entire point of this stage.

Integration is velocity Verlet (leapfrog): symplectic and time-reversible, so
the energy oscillates within a bound rather than drifting away. Two properties
that bound depends on, and that this module protects:
  * dt is FIXED. Velocity Verlet is symplectic only at constant dt, so dt is
    frozen on the state and re-derived only on an explicit event (reset, probe
    attached/detached, probe speed changed, or a running probe having heated
    the gas past the speed dt was frozen for) - never routinely per batch.
  * reflection is a TRUE MIRROR. Snapping a particle to a surface throws away
    its penetration depth, which is irreversible. The mirror's residual error
    depends on where in the step the crossing happened, so it is zero-mean.

ENERGY LEDGER. Energy is not always conserved - the probe does work, and an
inelastic wall removes it. Rather than switching the energy check off whenever
that happens, every external energy flow is booked as it occurs:

    E(t) - E(0) = (work done by the probe) + (energy the walls absorbed)

and the ledger checks that identity. Both terms are exact per collision; see
reflect_walls and MovingDisk.collide.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

SPEED_SIGMAS = 5.0          # thermal speed bound for the frozen dt, in sigma = sqrt(T/m)
HOT_CELL = 0.5              # cell size for the hottest-local-temperature estimate
HOT_CELL_MIN = 10           # a cell needs this many particles to count
# The probe vanishes and re-enters from the far side BEFORE its leading edge
# reaches the end wall, leaving at least this many particle diameters of gap.
# Letting it plough into the wall trapped gas between a moving surface and a
# fixed one, where each bounce adds 2V: an unbounded energy pump.
WRAP_GAP_DIAMETERS = 4.0


# --- initial conditions ------------------------------------------------------

def _finish_ic(vel, temperature, mass):
    """Remove net momentum, then pin the realised temperature exactly."""
    total_mass = float(mass.sum())
    if total_mass > 0:
        vel -= (mass[:, None] * vel).sum(axis=0) / total_mass
    t_now = temperature_of(vel, mass)
    t_target = float(np.mean(np.atleast_1d(temperature)))
    if t_now > 0 and t_target > 0:
        vel *= np.sqrt(t_target / t_now)
    return vel


def maxwell_velocities(n, temperature, mass, rng):
    """Draw 2D Maxwell-Boltzmann velocities (k_B = 1).

    Each Cartesian component is Gaussian with variance kT/m, which makes the
    SPEED distribution Rayleigh - the 2D Maxwell distribution. `temperature` may
    be a scalar or a per-particle array.

    Because this draw is already exactly Maxwellian, the ledger's pi/4 row passes
    from step 0 when started from it, testing numpy's Gaussian generator rather
    than the gas. The default initial condition is `uniform_speed_velocities`.
    """
    mass = np.asarray(mass, dtype=float).reshape(-1)
    temp = np.broadcast_to(np.asarray(temperature, dtype=float).reshape(-1), (n,))
    sigma = np.sqrt(temp / mass)
    vel = rng.normal(0.0, 1.0, size=(n, 2)) * sigma[:, None]
    return _finish_ic(vel, temp, mass)


def uniform_speed_velocities(n, temperature, mass, rng):
    """Every particle at the same speed sqrt(2kT/m), directions uniform.

    Deliberately NOT the equilibrium distribution: <v>^2/<v^2> starts at exactly
    1.0 and must relax to pi/4 = 0.785 through collisions alone.
    """
    mass = np.asarray(mass, dtype=float).reshape(-1)
    temp = np.broadcast_to(np.asarray(temperature, dtype=float).reshape(-1), (n,))
    speed = np.sqrt(2.0 * temp / mass)
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
    vel = np.column_stack([speed * np.cos(angle), speed * np.sin(angle)])
    return _finish_ic(vel, temp, mass)


IC_MODES = {"uniform_speed": uniform_speed_velocities, "maxwell": maxwell_velocities}


def temperature_of(vel, mass):
    """2D equipartition with k_B = 1: <KE> per particle = (D/2)kT = kT.

    Measured on PECULIAR velocities - the bulk drift is flow, not heat.
    """
    n = vel.shape[0]
    if n == 0:
        return 0.0
    total_mass = float(mass.sum())
    drift = (mass[:, None] * vel).sum(axis=0) / total_mass if total_mass > 0 else 0.0
    w = vel - drift
    ke = 0.5 * float(np.sum(mass * np.sum(w * w, axis=1)))
    return ke / n


def hottest_cell_temperature(pos, vel, mass, cell=HOT_CELL, min_count=HOT_CELL_MIN):
    """Temperature of the hottest coarse cell (lab-frame KE per particle).

    A global temperature hides a hot pocket: at T = 4 inside a room at T = 1,
    the pocket's own thermal tail runs far past 5 sigma of the room average.
    Cells with too few particles are skipped as too noisy; returns 0.0 when no
    cell qualifies, which leaves the caller on the global figure.
    """
    if pos.shape[0] == 0:
        return 0.0
    ij = np.floor(pos / cell).astype(np.int64)
    ij -= ij.min(axis=0)
    key = ij[:, 0] * (int(ij[:, 1].max()) + 1) + ij[:, 1]
    ke = 0.5 * mass * np.sum(vel * vel, axis=1)
    counts = np.bincount(key)
    sums = np.bincount(key, weights=ke)
    ok = counts >= min_count
    if not np.any(ok):
        return 0.0
    return float(np.max(sums[ok] / counts[ok]))


# --- forces ------------------------------------------------------------------

def pair_forces(pos, mass, diameter, stiffness, tree=None):
    """Soft-sphere contact forces over every overlapping pair.

    Returns (accel, potential_energy, virial, n_contacts, max_overlap) where
    `virial` is sum over pairs of r_ij . F_ij - the quantity the pressure gauge
    needs. For a purely repulsive force it is positive, which is why a real gas
    pushes harder than an ideal one at the same temperature.
    """
    n = pos.shape[0]
    accel = np.zeros((n, 2), dtype=float)
    if n == 0:
        return accel, 0.0, 0.0, 0, 0.0

    if tree is None:
        tree = cKDTree(pos)
    pairs = tree.query_pairs(r=diameter, output_type="ndarray")
    if pairs.shape[0] == 0:
        return accel, 0.0, 0.0, 0, 0.0

    i, j = pairs[:, 0], pairs[:, 1]
    delta = pos[i] - pos[j]
    r = np.linalg.norm(delta, axis=1)

    # Coincident particles have no defined direction; guard against a silent NaN.
    good = r > 1e-12
    if not np.any(good):
        return accel, 0.0, 0.0, 0, 0.0
    i, j, delta, r = i[good], j[good], delta[good], r[good]

    overlap = diameter - r
    f_mag = stiffness * overlap                      # >= 0, repulsive
    f_vec = (f_mag / r)[:, None] * delta             # on i, pointing away from j

    np.add.at(accel, i, f_vec / mass[i][:, None])
    np.add.at(accel, j, -f_vec / mass[j][:, None])

    potential = 0.5 * stiffness * float(np.sum(overlap ** 2))
    virial = float(np.sum(f_mag * r))                # r_ij . F_ij, forces are radial
    return accel, potential, virial, int(r.size), float(overlap.max())


# --- boundaries --------------------------------------------------------------

def reflect_walls(pos, vel, mass, world, radius, restitution=1.0):
    """Specular reflection off the signed-distance walls, in place.

    Returns (impulse, energy_change, hits):
      impulse        total |momentum| delivered to the walls (pressure gauge)
      energy_change  kinetic energy the walls removed: the normal component goes
                     from v_n to -e*v_n, a change of 0.5*m*(e^2 - 1)*v_n^2 per
                     bounce - exactly zero when e = 1
      hits           number of bounces, which sets the gauge's counting noise

    Only particles moving INTO the wall are touched; one that has already bounced
    and is still inside the band must not be displaced again. The penetration
    depth is MIRRORED, not discarded.
    """
    if pos.shape[0] == 0:
        return 0.0, 0.0, 0
    phi = world.clearance(pos)
    idx = np.flatnonzero(phi < radius)
    if idx.size == 0:
        return 0.0, 0.0, 0

    normal = world.normal(pos[idx])                  # unit, points into the fluid
    v_normal = np.sum(vel[idx] * normal, axis=1)
    incoming = v_normal < 0.0
    if not np.any(incoming):
        return 0.0, 0.0, 0

    sel = idx[incoming]
    n_in = normal[incoming]
    v_in = v_normal[incoming]
    depth = radius - phi[sel]
    pos[sel] += 2.0 * depth[:, None] * n_in          # mirror, do not snap
    vel[sel] -= (1.0 + restitution) * v_in[:, None] * n_in
    impulse = float(np.sum(mass[sel] * (1.0 + restitution) * np.abs(v_in)))
    energy = float(np.sum(0.5 * mass[sel] * (restitution ** 2 - 1.0) * v_in ** 2))
    return impulse, energy, int(sel.size)


@dataclass
class MovingDisk:
    """The probe: a disk dragged through the gas.

    It is neither a soft sphere nor a hard sphere in the particle sense. It is a
    RIGID body with PRESCRIBED motion - effectively infinitely massive, as if
    pulled by a motor that supplies whatever force it takes to hold the speed.
    The gas therefore cannot slow it down. Particles bounce off it instantly and
    specularly in its own rest frame, exactly like the walls; the soft springs
    act only between particles.

    Because each bounce preserves the particle's speed in the disk's frame, the
    particle's kinetic energy in the lab frame changes by exactly V . (m dv).
    Summed, the motor's work is V . (momentum delivered to the gas) - exact, not
    estimated, which is what lets the energy ledger stay live with the probe on.
    """
    centre: np.ndarray
    radius: float
    velocity: np.ndarray
    wraps: int = 0

    def advance(self, dt, world=None, margin=0.0, reach=None):
        """Move, and wrap so the probe can be watched repeatedly.

        The wrap fires while the LEADING edge is still `margin` short of the end
        wall, and the probe re-enters from the far side wholly outside the fluid
        (centre `reach` beyond the wall), so it emerges from the wall moving
        away from it. Gas is never trapped between the probe and a wall.
        Returns True on the step it wrapped.
        """
        self.centre = self.centre + self.velocity * dt
        if world is None:
            return False
        d = world.domain
        reach = self.radius if reach is None else reach
        vx = float(self.velocity[0])
        if vx > 0 and self.centre[0] + self.radius + margin >= d.x1:
            self.centre[0] = d.x0 - reach
        elif vx < 0 and self.centre[0] - self.radius - margin <= d.x0:
            self.centre[0] = d.x1 + reach
        else:
            return False
        self.wraps += 1
        return True

    def collide(self, pos, vel, mass, particle_radius):
        """Reflect particles off the disk. Returns the momentum delivered to the
        gas as a 2-vector (zero if nothing was hit)."""
        nothing = np.zeros(2)
        if pos.shape[0] == 0:
            return nothing
        reach = self.radius + particle_radius
        delta = pos - self.centre
        r = np.linalg.norm(delta, axis=1)
        idx = np.flatnonzero(r < reach)
        if idx.size == 0:
            return nothing
        rr = r[idx]
        safe = rr > 1e-12
        idx, rr = idx[safe], rr[safe]
        if idx.size == 0:
            return nothing

        normal = delta[idx] / rr[:, None]                  # outward from the disk
        rel = vel[idx] - self.velocity                     # disk rest frame
        v_normal = np.sum(rel * normal, axis=1)
        incoming = v_normal < 0.0
        if not np.any(incoming):
            return nothing
        sel = idx[incoming]
        n_in = normal[incoming]
        v_in = v_normal[incoming]
        depth = reach - rr[incoming]
        dv = -2.0 * v_in[:, None] * n_in
        pos[sel] += 2.0 * depth[:, None] * n_in
        vel[sel] += dv
        return (mass[sel][:, None] * dv).sum(axis=0)


# --- state -------------------------------------------------------------------

@dataclass
class GasState:
    pos: np.ndarray
    vel: np.ndarray
    mass: np.ndarray
    accel: np.ndarray
    dt: float = 0.0                # FROZEN step actually integrated with
    time: float = 0.0
    steps: int = 0
    potential: float = 0.0
    virial: float = 0.0
    contacts: int = 0
    max_overlap: float = 0.0
    disk: MovingDisk | None = None

    # short window, for the live panel readouts
    wall_impulse: float = 0.0
    drag_impulse: float = 0.0      # momentum given to the gas along the probe's motion
    window_time: float = 0.0

    # cumulative energy ledger - never reset by a window
    probe_work: float = 0.0
    wall_work: float = 0.0

    # times dt had to be re-derived because the running probe heated the gas
    dt_rederived: int = 0

    # long equilibrium averages for the pressure comparison. They accumulate
    # only while the gas is undisturbed (no probe, past `settle_until`, which is
    # a TIME, not a step count) and are cleared by any disturbance, so the
    # comparison gets sharper the longer the gas is left alone.
    settle_until: float = 0.0
    avg_time: float = 0.0
    avg_wall_impulse: float = 0.0
    avg_wall_hits: int = 0
    avg_virial: float = 0.0
    avg_kinetic: float = 0.0       # LAB-frame kinetic energy, as the virial theorem needs
    # G = sum m (r - origin) . v when the window opened. Its change over the
    # window is the finite-window term of the virial theorem; see gaugesMirror.
    avg_g0: float | None = None
    avg_origin: np.ndarray | None = None

    @property
    def n(self) -> int:
        return self.pos.shape[0]

    @property
    def external_work(self) -> float:
        return self.probe_work + self.wall_work

    def reset_window(self):
        self.wall_impulse = 0.0
        self.drag_impulse = 0.0
        self.window_time = 0.0

    def reset_averages(self):
        self.avg_time = 0.0
        self.avg_wall_impulse = 0.0
        self.avg_wall_hits = 0
        self.avg_virial = 0.0
        self.avg_kinetic = 0.0
        self.avg_g0 = None
        self.avg_origin = None

    def disturb(self, relax_time):
        """Something pushed the gas out of equilibrium: wait, then re-average."""
        self.settle_until = self.time + float(relax_time)
        self.reset_averages()

    def virial_moment(self, origin) -> float:
        """G = sum m (r - origin) . v. Its time derivative is 2K + sum r . F."""
        if self.n == 0:
            return 0.0
        rel = self.pos - np.asarray(origin, dtype=float)
        return float(np.sum(self.mass * np.sum(rel * self.vel, axis=1)))

    def kinetic_energy(self) -> float:
        return 0.5 * float(np.sum(self.mass * np.sum(self.vel * self.vel, axis=1)))

    def total_energy(self) -> float:
        return self.kinetic_energy() + self.potential

    def momentum(self) -> np.ndarray:
        return (self.mass[:, None] * self.vel).sum(axis=0)

    def drift_velocity(self) -> np.ndarray:
        total = float(self.mass.sum())
        return self.momentum() / total if total > 0 else np.zeros(2)


def refresh_forces(state, scene):
    """Recompute forces, potential and virial for the current positions."""
    gas = scene.gas
    (state.accel, state.potential, state.virial,
     state.contacts, state.max_overlap) = pair_forces(
        state.pos, state.mass, gas.diameter, gas.stiffness)


def make_state(scene, world, field_obj):
    """Promote a Stage-1 ParticleField into a Stage-2 gas state.

    Stage 1 paints zones onto `field.temperature`; they are honoured here, so a
    hot pocket really is a hot pocket.
    """
    gas = scene.gas
    rng = np.random.default_rng(gas.seed)
    pos = np.array(field_obj.pos, dtype=float, copy=True)
    mass = np.array(field_obj.mass, dtype=float, copy=True)
    temps = np.asarray(field_obj.temperature, dtype=float).copy()

    draw = IC_MODES.get(gas.ic_mode, uniform_speed_velocities)
    vel = draw(pos.shape[0], temps, mass, rng)

    state = GasState(pos=pos, vel=vel, mass=mass, accel=np.zeros_like(pos),
                     settle_until=gas.equilibrate_time)
    refresh_forces(state, scene)
    if gas.disk_enabled:
        attach_disk(state, scene, world)
    freeze_dt(state, scene)
    return state


# --- the probe ---------------------------------------------------------------

def speed_cap(scene, state=None):
    """Largest probe speed the contact law can represent (relative units).

    A fixed `disk_speed_max` in the config wins. Otherwise it is derived: a
    particle bounced off the probe leaves with up to 2V more speed, and if it
    then meets another particle head-on faster than the tunnelling speed
    d*sqrt(2k/m), the two pass straight through each other. See
    gaugesMirror.probe_speed_limit. Uses the LIVE temperature when a state is
    given, so the cap falls as the probe heats the gas.
    """
    import gaugesMirror as G
    gas = scene.gas
    if gas.disk_speed_max is not None:
        return float(gas.disk_speed_max)
    if state is not None and state.n:
        temp = temperature_of(state.vel, state.mass)
        mass = float(state.mass.max())
    else:
        temp, mass = gas.temperature, scene.particle_mass
    return G.probe_speed_limit(gas.diameter, gas.stiffness, mass, temp)


def clamp_speed(scene, speed, state=None):
    """Clamp a requested probe speed to +/- speed_cap. Rejects NaN and infinity,
    which would otherwise put NaN into every particle the probe touched."""
    speed = float(speed)
    if not np.isfinite(speed):
        raise ValueError("speed must be a finite number")
    cap = speed_cap(scene, state)
    return float(np.clip(speed, -cap, cap))


def attach_disk(state, scene, world, speed=None):
    """Insert the probe, moving along x at `speed` (relative units; sign = direction).

    Inserting a body displaces the gas it lands on. Particles found inside the
    probe are moved radially outward into a band one lattice spacing wide just
    outside its surface, forces are recomputed, and the resulting energy change
    is booked as probe work. Insertion is a move with a cost; the ledger records
    it instead of letting it look like a leak. (A band only d/2 wide crammed
    ~40 particles into a thin shell, and with a stiff spring the overlap energy
    alone was a small explosion.)
    """
    import gaugesMirror as G
    gas = scene.gas
    if speed is None:
        temp = temperature_of(state.vel, state.mass)
        mass_mean = float(state.mass.mean()) if state.n else 1.0
        speed = gas.disk_mach * G.sound_speed(temp, mass_mean)
    speed = clamp_speed(scene, speed, state)

    d = world.domain
    x = d.x0 + 1.5 * gas.disk_radius if speed >= 0 else d.x1 - 1.5 * gas.disk_radius
    disk = MovingDisk(centre=np.array([x, 0.5 * (d.y0 + d.y1)]),
                      radius=gas.disk_radius, velocity=np.array([speed, 0.0]))

    if state.n:
        reach = disk.radius + 0.5 * gas.diameter
        delta = state.pos - disk.centre
        r = np.linalg.norm(delta, axis=1)
        inside = np.flatnonzero(r < reach)
        if inside.size:
            e_before = state.total_energy()
            rng = np.random.default_rng(gas.seed + state.steps + 1)
            rr = r[inside]
            angles = rng.uniform(0.0, 2.0 * np.pi, size=inside.size)
            fallback = np.column_stack([np.cos(angles), np.sin(angles)])
            dirs = np.where((rr > 1e-12)[:, None],
                            delta[inside] / np.maximum(rr, 1e-12)[:, None], fallback)
            band = max(gas.diameter, scene.spacing)
            new_r = reach + rng.uniform(0.0, band, size=inside.size)
            state.pos[inside] = disk.centre + dirs * new_r[:, None]
            refresh_forces(state, scene)
            state.probe_work += state.total_energy() - e_before

    state.disk = disk
    state.reset_averages()
    freeze_dt(state, scene)
    return disk


def set_disk_speed(state, scene, speed):
    """Change the probe's speed in place. Returns the (clamped) speed applied."""
    speed = clamp_speed(scene, speed, state)
    if state.disk is not None:
        state.disk.velocity = np.array([speed, 0.0])
        freeze_dt(state, scene)
    return speed


def detach_disk(state, scene):
    """Remove the probe. The gas it leaves behind is heated and still moving, so
    the equilibrium checks wait for it to settle again."""
    state.disk = None
    state.disturb(scene.gas.equilibrate_time)
    freeze_dt(state, scene)


# --- timestep ----------------------------------------------------------------

def dt_limits(scene, state, v_fast=None):
    """The two real constraints on the step, reported separately.

    This is NOT a Courant condition. The lattice CFL bound (1/sqrt(D), or
    sqrt(2/3) on a triangular lattice) governs a wave solver on a grid; a particle
    integrator is limited by:

      contact:  two particles of mass m on a spring of stiffness k oscillate with
                reduced mass m/2, so the period is 2*pi*sqrt(m/(2k)). Step a
                small fraction of it or a collision becomes one wild kick.
      travel:   two particles must not CLOSE by more than a small fraction of the
                interaction diameter in one step. Closing speed is up to 2v
                head-on, where v is the fastest particle.

    `v_fast` is that fastest speed. Left as None it is MEASURED - the live
    check - and then already includes any particle the probe has bounced.
    freeze_dt passes a bound instead, with the probe's boost added in advance.

    Courant returns properly in Stage 3, as dt <= Co * h / (c_s + v_max).
    """
    gas = scene.gas
    m_min = float(np.min(state.mass)) if state.n else 1.0
    period = 2.0 * np.pi * np.sqrt(m_min / (2.0 * gas.stiffness))
    dt_contact = gas.dt_safety_contact * period

    if v_fast is None:
        v_fast = float(np.linalg.norm(state.vel, axis=1).max()) if state.n else 0.0
    closing = 2.0 * v_fast
    dt_travel = (gas.dt_safety_travel * gas.diameter / closing) if closing > 0 else np.inf
    return dt_contact, dt_travel


def thermal_speed_bound(state, scene):
    """SPEED_SIGMAS sigma at the HOTTEST temperature in the gas, or the fastest
    particle now, whichever is larger.

    Using the room-average temperature here let a hot pocket's own thermal tail
    outrun the bound, and the ledger's dt row flickered red on a correct run.
    """
    gas = scene.gas
    if not state.n:
        return 1.0
    t_hot = max(temperature_of(state.vel, state.mass), gas.temperature,
                hottest_cell_temperature(state.pos, state.vel, state.mass))
    m_min = float(np.min(state.mass))
    v_now = float(np.linalg.norm(state.vel, axis=1).max())
    return max(SPEED_SIGMAS * np.sqrt(t_hot / m_min), v_now)


def freeze_dt(state, scene):
    """Choose dt once from a conservative speed bound and hold it.

    The bound is thermal_speed_bound plus, with the probe attached, 2|V|: in
    the probe's frame a bounce preserves speed, so in the lab a particle can
    leave with up to 2|V| more than it arrived with. (Adding only |V|, as an
    earlier version did, made the live check fail as soon as the first
    particles bounced.) The ledger grades the frozen dt against the live limits
    every frame.
    """
    v_fast = thermal_speed_bound(state, scene)
    if state.disk is not None:
        v_fast += 2.0 * float(np.linalg.norm(state.disk.velocity))
    dt_contact, dt_travel = dt_limits(scene, state, v_fast=v_fast)
    state.dt = min(dt_contact, dt_travel)
    return state.dt


# --- stepping ----------------------------------------------------------------

def step(state, scene, world, dt=None):
    """One velocity-Verlet step, mutating `state` in place."""
    gas = scene.gas
    radius = 0.5 * gas.diameter
    dt = state.dt if dt is None else dt

    # half kick, drift
    state.vel += 0.5 * dt * state.accel
    state.pos += dt * state.vel

    # boundaries act on the drifted positions
    if state.disk is not None:
        state.disk.advance(dt, world, margin=WRAP_GAP_DIAMETERS * gas.diameter,
                           reach=state.disk.radius + radius)
        delivered = state.disk.collide(state.pos, state.vel, state.mass, radius)
        v_probe = state.disk.velocity
        speed = float(np.linalg.norm(v_probe))
        if speed > 0:
            state.drag_impulse += float(delivered @ v_probe) / speed
        state.probe_work += float(delivered @ v_probe)

    impulse, energy, hits = reflect_walls(state.pos, state.vel, state.mass, world,
                                          radius, gas.restitution)
    state.wall_impulse += impulse
    state.wall_work += energy

    # new forces, half kick
    refresh_forces(state, scene)
    state.vel += 0.5 * dt * state.accel

    state.time += dt
    state.window_time += dt
    state.steps += 1

    if state.disk is None and state.time > state.settle_until:
        if state.avg_g0 is None:
            # The window opens at the END of this step: record G there, and
            # count only what happens from here on.
            d = world.domain
            state.avg_origin = np.array([0.5 * (d.x0 + d.x1), 0.5 * (d.y0 + d.y1)])
            state.avg_g0 = state.virial_moment(state.avg_origin)
        else:
            state.avg_time += dt
            state.avg_wall_impulse += impulse
            state.avg_wall_hits += hits
            state.avg_virial += state.virial * dt
            state.avg_kinetic += state.kinetic_energy() * dt
    return state


def advance(state, scene, world, n_steps):
    """Run `n_steps` steps at the frozen dt.

    With the probe attached the gas heats without limit - the probe does work
    on it every step - so the speed bound dt was frozen for goes stale. Crossing
    it is treated as an EVENT, like the others: dt is re-derived once, counted,
    and held again. Without the probe nothing heats the gas, so nothing is
    re-derived and the ledger's dt row stays a real check.
    """
    if state.dt <= 0:
        freeze_dt(state, scene)
    for _ in range(n_steps):
        step(state, scene, world, state.dt)
    if state.disk is not None:
        if state.dt > min(dt_limits(scene, state)):
            freeze_dt(state, scene)
            state.dt_rederived += 1
    return state.dt


def advance_time(state, scene, world, duration, batch=None):
    """Advance by at least `duration` simulation time, in batches."""
    target = state.time + float(duration)
    batch = batch or scene.gas.substeps
    while state.time < target:
        if state.dt <= 0:
            freeze_dt(state, scene)
        remaining = int(np.ceil((target - state.time) / state.dt))
        advance(state, scene, world, max(1, min(batch, remaining)))
    return state.dt
