"""BlastBox Stage 2 - gauges: reading thermodynamics out of the motion.

Nothing in dynamicsMirror.py knows what pressure or temperature are. They are
MEASURED here from particle motion, which is the whole lesson of this stage:
give particles positions, velocities and a repulsive contact force, and the bulk
quantities appear on their own.

Two independent pressure routes are provided on purpose. They rest on different
physics - one counts momentum delivered to the walls, the other sums the internal
virial - so agreement between them is a real check rather than a restatement.

Three measurement subtleties this module is careful about:
  * Temperature is measured on PECULIAR velocities. Bulk flow is flow, not heat;
    counting it inflates T, and through c = sqrt(2T) it inflates the reported
    Mach number too - all self-consistently, which is what makes it dangerous.
  * The container available to particle CENTRES is the geometry ERODED by the
    particle radius. Using the raw geometry biases the two pressure routes
    against each other by about a percent - invisible under a loose band, and
    the dominant error under a tight one.
  * A soft sphere is not a hard disc of the same size. Under a linear spring its
    Barker-Henderson effective diameter is smaller, and the equation-of-state row
    grades against that, not against the nominal diameter.

All values are in simulation units (k_B = 1). See unitsMirror.py.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

DIM = 2
MAXWELL_2D_RATIO = np.pi / 4.0      # 0.7853981633974483
HARD_DISC_B3 = 3.128                # 2D hard-disc third virial coefficient


def drift_velocity(vel, mass):
    """Mass-weighted mean velocity: the bulk flow."""
    if vel.shape[0] == 0:
        return np.zeros(2)
    total = float(mass.sum())
    return (mass[:, None] * vel).sum(axis=0) / total if total > 0 else np.zeros(2)


def peculiar(vel, mass):
    """Velocities with the bulk flow removed - the thermal part."""
    return vel - drift_velocity(vel, mass)


def temperature(vel, mass):
    """2D equipartition, k_B = 1: <KE> per particle = (D/2)kT = kT."""
    n = vel.shape[0]
    if n == 0:
        return 0.0
    w = peculiar(vel, mass)
    ke = 0.5 * float(np.sum(mass * np.sum(w * w, axis=1)))
    return ke / n


def component_temperatures(vel, mass):
    """Temperature from x and y separately. Equal, for an isotropic gas."""
    n = vel.shape[0]
    if n == 0:
        return 0.0, 0.0
    w = peculiar(vel, mass)
    tx = float(np.sum(mass * w[:, 0] ** 2)) / n
    ty = float(np.sum(mass * w[:, 1] ** 2)) / n
    return tx, ty   # each equals kT: (1/2)m<vx^2> = kT/2


def speed_moments(vel, mass=None):
    """Return (mean speed, rms speed, <v>^2 / <v^2>) on peculiar velocities.

    That last ratio is the sharpest check in the stage. For the 2D Maxwell
    (Rayleigh) speed distribution, <v> = sigma*sqrt(pi/2) and <v^2> = 2*sigma^2,
    so the ratio is exactly pi/4 - independent of temperature, of mass, and of
    every other parameter. It tests the SHAPE of the distribution with nothing to
    tune, which is only meaningful if the run did not START Maxwellian; see
    dynamicsMirror.uniform_speed_velocities.
    """
    if vel.shape[0] == 0:
        return 0.0, 0.0, 0.0
    w = peculiar(vel, mass) if mass is not None else vel
    speeds = np.linalg.norm(w, axis=1)
    mean = float(speeds.mean())
    mean_sq = float(np.mean(speeds ** 2))
    ratio = (mean * mean / mean_sq) if mean_sq > 0 else 0.0
    return mean, float(np.sqrt(mean_sq)), ratio


def pressure_virial(n, area, temp, virial):
    """P = (N kT + (1/D) * sum r_ij . F_ij) / A.

    The first term is ideal-gas momentum transport; the second is the extra push
    from particles repelling each other, positive for a repulsive force.
    """
    if area <= 0:
        return float("nan")
    return (n * temp + virial / DIM) / area


def pressure_ideal(n, area, temp):
    """The ideal-gas part alone - the baseline the interaction term sits on."""
    if area <= 0:
        return float("nan")
    return n * temp / area


def pressure_wall(impulse, window_time, perimeter):
    """P = impulse delivered per unit time per unit wall length.

    In 2D pressure is force per unit LENGTH, so the denominator is the wetted
    perimeter, not an area.
    """
    if window_time <= 0 or perimeter <= 0:
        return float("nan")
    return impulse / (window_time * perimeter)


def effective_diameter(diameter, stiffness, temp):
    """Barker-Henderson effective hard-disc diameter of the soft sphere.

    d_eff = integral_0^inf [1 - exp(-U(r)/T)] dr. For the linear spring
    U(x) = 0.5 k x^2 measured inward from contact, that is
    d - sqrt(pi*T/(2k)) once the spring is stiff enough to cut off inside d.

    A soft particle is squeezable, so it behaves as a SMALLER hard disc than its
    nominal size. At the shipped stiffness d_eff is about 0.93 d (0.79 d at the
    softer k = 2e4 used earlier). Even 0.93 d shifts the predicted EXCESS
    pressure by about 14%, so it cannot be waved away.
    """
    if temp <= 0 or stiffness <= 0:
        return diameter
    return max(0.0, diameter - np.sqrt(np.pi * temp / (2.0 * stiffness)))


def area_fraction(n, diameter, area):
    """Fraction of the domain covered by discs of the given diameter."""
    if area <= 0:
        return 0.0
    return n * np.pi * (0.5 * diameter) ** 2 / area


def compressibility(pressure, n, area, temp):
    """Z = PA / (N kT). Exactly 1 for an ideal gas."""
    if n == 0 or temp <= 0 or area <= 0:
        return float("nan")
    return pressure * area / (n * temp)


def compressibility_hard_disc(phi_eff):
    """2D hard-disc virial series: Z = 1 + 2*phi + 3.128*phi^2 + ...

    Fed the EFFECTIVE packing fraction, this is a closed form to grade against,
    not a hand-waved band.
    """
    return 1.0 + 2.0 * phi_eff + HARD_DISC_B3 * phi_eff ** 2


def sound_speed(temp, mass_mean):
    """Ideal monatomic gas: c^2 = gamma kT / m with gamma = (D+2)/D.

    In 2D gamma = 2, so c = sqrt(2kT/m) - which is exactly v_rms. That identity
    is a consequence of the definitions, not an independent measurement, so the
    readout labels it as an identity and never grades it as a test. A genuine
    sound-speed measurement needs a pulse and a timed arrival; that is Stage 2b.
    """
    if mass_mean <= 0:
        return float("nan")
    gamma = (DIM + 2.0) / DIM
    return float(np.sqrt(gamma * temp / mass_mean))


def mean_free_path(n, area, diameter):
    """2D mean free path: lambda = 1 / (sqrt(2) * number density * diameter)."""
    if area <= 0 or diameter <= 0 or n == 0:
        return float("inf")
    density = n / area
    return 1.0 / (np.sqrt(2.0) * density * diameter)


def knudsen(n, area, diameter, body_size):
    """Kn = mean free path / body size.

    Continuum flow needs Kn < 0.01. Above about 0.1 the gas is rarefied: a shock
    is several mean free paths thick, so it is wider than the body and no sharp
    Mach cone can form. This is the number that says whether the cone angle
    printed next to it means anything.
    """
    if body_size <= 0:
        return float("inf")
    return mean_free_path(n, area, diameter) / body_size


PROBE_TAIL_SIGMAS = 6.0     # thermal allowance in probe_speed_limit: 3 sigma per particle


def tunnel_speed(diameter, stiffness, mass):
    """Closing speed above which two equal soft spheres pass through each other.

    Head-on, the pair's relative kinetic energy is (1/2)(m/2)u^2 = (1/4)m u^2.
    The spring can store at most (1/2)k d^2, reached at full overlap. Below
    u* = d*sqrt(2k/m) they bounce; above it they sail through, which no hard
    particle would do. Past this speed the gas is no longer the model it claims
    to be, whatever the timestep.
    """
    if mass <= 0 or stiffness <= 0:
        return float("inf")
    return float(diameter * np.sqrt(2.0 * stiffness / mass))


def probe_speed_limit(diameter, stiffness, mass, temp):
    """Fastest probe speed V whose bounces stay below the tunnelling speed.

    A particle arriving at speed v leaves the probe at up to v + 2V. Meeting a
    thermal particle head-on at v', the pair closes at 2V + v + v'. Allowing
    3 sigma for each thermal speed gives V_max = (u* - 6 sigma) / 2, with
    sigma = sqrt(T/m). It falls as the gas heats.
    """
    sigma = np.sqrt(max(temp, 0.0) / mass) if mass > 0 else 0.0
    return max(0.0, 0.5 * (tunnel_speed(diameter, stiffness, mass)
                           - PROBE_TAIL_SIGMAS * sigma))


def mach_angle(mach):
    """Mach cone half-angle: sin(mu) = 1/M. Undefined at or below Mach 1."""
    if mach <= 1.0:
        return float("nan")
    return float(np.degrees(np.arcsin(1.0 / mach)))


def wall_noise(hits):
    """Relative counting noise of the wall-pressure gauge after `hits` bounces.

    The gauge sums a random number of random impulses 2*m*v_n. For a Poisson
    count the relative spread is sqrt(1 + cv^2)/sqrt(hits), where cv is the
    coefficient of variation of v_n among particles that HIT the wall. That
    flux-weighted distribution is Rayleigh, with <v^2>/<v>^2 = 4/pi, so

        relative noise = sqrt(4 / (pi * hits)) = 2 / sqrt(pi * hits)

    It is the number the pressure comparison is graded against: a band chosen
    by hand was, in the first version, narrower than this noise and would have
    flickered red roughly one window in eight on a correct simulation.

    It treats hits as independent. In a closed box the particle number and total
    energy are fixed, which should suppress the slow fluctuations, so this
    figure is expected to OVERSTATE the true spread: conservative for a band,
    not a tight estimate of it.
    """
    if hits <= 0:
        return float("inf")
    return 2.0 / np.sqrt(np.pi * hits)


# Free-molecular, hypersonic drag of a cylinder with specular reflection, as a
# REFERENCE (not a check). A surface element at angle t from the stagnation line
# receives particles at normal speed U*cos(t) and returns each with 2*m*U*cos(t)
# of normal momentum, cos(t) of which lies along the flow. Integrating
# 2*rho*U^2*R*cos^3(t) over -pi/2..pi/2 gives (8/3)*rho*U^2*R, and dividing by
# 0.5*rho*U^2*(2R) leaves 8/3. It holds only when the gas is collisionless
# (Kn >> 1) and the body far outruns the thermal motion. At finite speed the
# thermal particles add impacts from every side, so C_d sits ABOVE 8/3 and falls
# toward it as the probe speeds up. The same argument gives 4 for a flat plate
# broadside and 4*sin^2(delta) for a wedge of half-angle delta.
CD_FREE_MOLECULAR_CYLINDER = 8.0 / 3.0


def drag_force(drag_impulse, window_time):
    """Mean force the probe exerts on the gas along its motion."""
    if window_time <= 0:
        return float("nan")
    return drag_impulse / window_time


def drag_coefficient(drag, rho_mass, speed, width):
    """C_d = D / (0.5 * rho * U^2 * width). Dimensionless, so it needs no SI map.

    `width` is the probe's diameter as the particle CENTRES see it,
    2 * (R + d/2) - the same erosion logic the pressure gauges use.
    """
    if (not np.isfinite(drag)) or rho_mass <= 0 or width <= 0 or speed == 0:
        return float("nan")
    return drag / (0.5 * rho_mass * speed ** 2 * width)


@dataclass
class GasReadout:
    """Everything the panel and the ledger need, measured once per refresh."""
    n: int
    area: float
    perimeter: float
    temperature: float
    temp_x: float
    temp_y: float
    drift: float
    mean_speed: float
    rms_speed: float
    maxwell_ratio: float
    p_virial: float
    p_ideal: float
    p_wall: float
    phi: float
    phi_eff: float
    d_eff_ratio: float
    z_measured: float
    z_expected: float
    sound_speed: float
    knudsen: float
    kinetic: float
    potential: float
    total_energy: float
    momentum: float
    contacts: int
    max_overlap: float
    external_work: float
    # long equilibrium averages (the pressure comparison)
    avg_time: float
    avg_hits: int
    p_virial_avg: float
    p_ideal_avg: float
    p_wall_avg: float
    p_window: float          # finite-window term, already inside the two above
    wall_noise: float
    # the probe
    probe_speed: float
    probe_limit: float       # fastest speed the contact law can represent now
    probe_resolved: bool     # False: bounces tunnel; cone and C_d are not trustworthy
    mach: float
    drag: float
    drag_coefficient: float


def read(state, scene, world):
    """Take a full set of measurements from the current state.

    Geometry is eroded by the particle radius: reflection happens when a
    particle's CENTRE comes within d/2 of a wall, so that is the container the
    centres actually live in, and it is the one both pressure routes must use.
    """
    gas = scene.gas
    inset = 0.5 * gas.diameter
    area = world.fluid_area(inset=inset)
    perimeter = world.wall_perimeter(inset=inset)

    temp = temperature(state.vel, state.mass)
    tx, ty = component_temperatures(state.vel, state.mass)
    mean_v, rms_v, ratio = speed_moments(state.vel, state.mass)
    p_vir = pressure_virial(state.n, area, temp, state.virial)
    p_id = pressure_ideal(state.n, area, temp)
    p_wall = pressure_wall(state.wall_impulse, state.window_time, perimeter)

    d_eff = effective_diameter(gas.diameter, gas.stiffness, temp)
    phi = area_fraction(state.n, gas.diameter, area)
    phi_eff = area_fraction(state.n, d_eff, area)
    mass_mean = float(state.mass.mean()) if state.n else 1.0
    c = sound_speed(temp, mass_mean)
    disk = state.disk
    body = 2.0 * (disk.radius if disk is not None else gas.disk_radius)

    # Equilibrium averages: both pressure routes over the SAME interval.
    #
    # The virial theorem is a statement about TIME AVERAGES. With
    # G = sum m (r - origin) . v, exactly
    #     dG/dt = 2K + sum r_ij . F_ij + (wall term),
    # and for a uniform wall pressure the wall term is -P * D * A. Averaged over
    # a window of length tau:
    #     P = (2<K> + <sum r.F> - (G(tau) - G(0)) / tau) / (D A)
    # The last piece is the finite-window term. It vanishes for a long window in
    # equilibrium, but not while a pulse is still ringing round the room: an
    # expanding blast raises G before the walls have felt anything. It is
    # computed exactly and included, rather than waited out.
    # K is the LAB-frame kinetic energy here, as the identity requires.
    p_window = float("nan")
    if state.avg_time > 0 and state.avg_g0 is not None and area > 0:
        tau = state.avg_time
        k_avg = state.avg_kinetic / tau
        vir_avg = state.avg_virial / tau
        dg = state.virial_moment(state.avg_origin) - state.avg_g0
        p_window = dg / (DIM * area * tau)
        p_id_avg = 2.0 * k_avg / (DIM * area) - p_window
        p_vir_avg = p_id_avg + vir_avg / (DIM * area)
        p_wall_avg = pressure_wall(state.avg_wall_impulse, tau, perimeter)
    else:
        p_vir_avg = p_id_avg = p_wall_avg = float("nan")

    # the probe
    probe_speed = float(np.linalg.norm(disk.velocity)) if disk is not None else 0.0
    m_max = float(state.mass.max()) if state.n else scene.particle_mass
    probe_limit = probe_speed_limit(gas.diameter, gas.stiffness, m_max, temp)
    probe_resolved = disk is None or probe_speed <= probe_limit
    mach = (probe_speed / c) if (disk is not None and c > 0) else float("nan")
    if disk is not None and state.window_time > 0:
        drag = drag_force(state.drag_impulse, state.window_time)
        rho_mass = float(state.mass.sum()) / area if area > 0 else 0.0
        cd = drag_coefficient(drag, rho_mass, probe_speed, 2.0 * (disk.radius + inset))
    else:
        drag, cd = float("nan"), float("nan")

    return GasReadout(
        n=state.n,
        area=area,
        perimeter=perimeter,
        temperature=temp,
        temp_x=tx,
        temp_y=ty,
        drift=float(np.linalg.norm(drift_velocity(state.vel, state.mass))),
        mean_speed=mean_v,
        rms_speed=rms_v,
        maxwell_ratio=ratio,
        p_virial=p_vir,
        p_ideal=p_id,
        p_wall=p_wall,
        phi=phi,
        phi_eff=phi_eff,
        d_eff_ratio=(d_eff / gas.diameter) if gas.diameter > 0 else 0.0,
        z_measured=compressibility(p_vir, state.n, area, temp),
        z_expected=compressibility_hard_disc(phi_eff),
        sound_speed=c,
        knudsen=knudsen(state.n, area, gas.diameter, body),
        kinetic=state.kinetic_energy(),
        potential=state.potential,
        total_energy=state.total_energy(),
        momentum=float(np.linalg.norm(state.momentum())),
        contacts=state.contacts,
        max_overlap=state.max_overlap,
        external_work=state.external_work,
        avg_time=state.avg_time,
        avg_hits=state.avg_wall_hits,
        p_virial_avg=p_vir_avg,
        p_ideal_avg=p_id_avg,
        p_wall_avg=p_wall_avg,
        p_window=p_window,
        wall_noise=wall_noise(state.avg_wall_hits),
        probe_speed=probe_speed,
        probe_limit=probe_limit,
        probe_resolved=probe_resolved,
        mach=mach,
        drag=drag,
        drag_coefficient=cd,
    )
