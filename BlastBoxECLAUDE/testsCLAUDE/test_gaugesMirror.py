import numpy as np
import gaugesMirror as G
import dynamicsMirror as D


def test_temperature_matches_2d_equipartition():
    """In 2D, <KE> per particle = (D/2)kT = kT, so T = <KE> with k_B = 1."""
    vel = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    mass = np.ones(4)
    assert abs(G.temperature(vel, mass) - 0.5) < 1e-12


def test_temperature_and_components_are_measured_on_peculiar_velocities():
    """Bulk flow is flow, not heat. Adding a uniform drift must change neither
    the total nor the per-axis temperature."""
    rng = np.random.default_rng(11)
    n = 20_000
    mass = np.ones(n)
    vel = D.maxwell_velocities(n, temperature=1.0, mass=mass, rng=rng)
    t0 = G.temperature(vel, mass)
    tx0, ty0 = G.component_temperatures(vel, mass)
    moved = vel + np.array([2.5, -1.5])
    assert abs(G.temperature(moved, mass) - t0) < 1e-9
    tx1, ty1 = G.component_temperatures(moved, mass)
    assert abs(tx1 - tx0) < 1e-9 and abs(ty1 - ty0) < 1e-9
    assert abs(G.drift_velocity(moved, mass)[0] - 2.5) < 1e-9


def test_maxwell_speed_ratio_is_pi_over_four():
    """The sharpest check in the stage: <v>^2/<v^2> = pi/4 for the 2D Maxwell
    (Rayleigh) speed distribution, independent of temperature and mass."""
    rng = np.random.default_rng(7)
    n = 200_000
    mass = np.full(n, 2.1)
    vel = D.maxwell_velocities(n, temperature=3.7, mass=mass, rng=rng)
    _, _, ratio = G.speed_moments(vel, mass)
    assert abs(ratio - G.MAXWELL_2D_RATIO) < 5e-3


def test_maxwell_ratio_constant_against_an_independent_derivation():
    """Derived from the Rayleigh moments rather than restating np.pi/4:
    <v> = sigma*sqrt(pi/2), <v^2> = 2*sigma^2, so the ratio is pi/4."""
    sigma = 1.7
    mean = sigma * np.sqrt(np.pi / 2.0)
    mean_sq = 2.0 * sigma ** 2
    assert abs(G.MAXWELL_2D_RATIO - mean * mean / mean_sq) < 1e-15


def test_virial_pressure_reduces_to_ideal_without_contacts():
    p = G.pressure_virial(n=100, area=10.0, temp=2.0, virial=0.0)
    assert abs(p - 100 * 2.0 / 10.0) < 1e-12
    assert abs(p - G.pressure_ideal(100, 10.0, 2.0)) < 1e-12


def test_repulsion_pushes_pressure_above_ideal():
    ideal = G.pressure_ideal(100, 10.0, 2.0)
    real = G.pressure_virial(100, 10.0, 2.0, 50.0)
    assert real > ideal
    assert abs((real - ideal) - 50.0 / (G.DIM * 10.0)) < 1e-12


def test_wall_pressure_is_force_per_unit_length():
    p = G.pressure_wall(impulse=12.0, window_time=2.0, perimeter=3.0)
    assert abs(p - 2.0) < 1e-12
    assert not np.isfinite(G.pressure_wall(1.0, 0.0, 3.0))   # no window yet


def test_sound_speed_equals_rms_speed_in_2d():
    """gamma = (D+2)/D = 2 in 2D, so c = sqrt(2kT/m), which IS v_rms.
    An identity, which is why the ledger states it rather than grading it."""
    rng = np.random.default_rng(3)
    n = 50_000
    mass = np.full(n, 1.3)
    vel = D.maxwell_velocities(n, temperature=0.9, mass=mass, rng=rng)
    temp = G.temperature(vel, mass)
    _, rms, _ = G.speed_moments(vel, mass)
    assert abs(G.sound_speed(temp, 1.3) - rms) < 1e-9


def test_effective_diameter_is_smaller_than_nominal_and_shrinks_when_softer():
    """A soft particle is squeezable, so it acts as a smaller hard disc."""
    d, k, t = 0.042, 20000.0, 1.0
    d_eff = G.effective_diameter(d, k, t)
    assert 0 < d_eff < d
    assert abs(d_eff - (d - np.sqrt(np.pi * t / (2 * k)))) < 1e-15
    assert G.effective_diameter(d, k / 10.0, t) < d_eff   # softer -> smaller still


def test_softness_deficit_scales_as_one_over_root_stiffness():
    """d - d_eff = sqrt(pi*T/2k), so a hundredfold stiffer particle is ten times
    closer to its nominal size. Tested as a scaling law rather than against an
    arbitrary 'close enough to rigid' epsilon."""
    d, t = 0.042, 1.0
    deficit = lambda k: d - G.effective_diameter(d, k, t)
    assert abs(deficit(1e8) / deficit(1e12) - 100.0) < 1e-6
    assert abs(deficit(1e10) / deficit(1e12) - 10.0) < 1e-6
    assert deficit(1e12) < 1e-5          # effectively rigid at this stiffness


def test_hard_disc_series_grows_with_packing():
    z1 = G.compressibility_hard_disc(0.02)
    z2 = G.compressibility_hard_disc(0.08)
    assert 1.0 < z1 < z2
    # second-order term must actually contribute
    assert abs(z2 - (1 + 2 * 0.08)) > 1e-6


def test_compressibility_detects_a_non_ideal_gas():
    """Z = 1 exactly without interactions, and above 1 with repulsion."""
    area, n, temp = 4.0, 50, 1.5
    p_ideal = G.pressure_virial(n, area, temp, 0.0)
    assert abs(G.compressibility(p_ideal, n, area, temp) - 1.0) < 1e-12
    p_real = G.pressure_virial(n, area, temp, 40.0)
    assert G.compressibility(p_real, n, area, temp) > 1.05


def test_area_fraction():
    phi = G.area_fraction(n=1000, diameter=0.1, area=100.0)
    assert abs(phi - 1000 * np.pi * 0.0025 / 100.0) < 1e-12


def test_knudsen_flags_a_rarefied_gas():
    """Kn = mean free path / body size; continuum needs Kn << 1."""
    lam = G.mean_free_path(n=6300, area=110.0, diameter=0.042)
    assert abs(G.knudsen(6300, 110.0, 0.042, 1.2) - lam / 1.2) < 1e-12
    assert G.knudsen(6300, 110.0, 0.042, 1.2) > 0.1        # shipped defaults: rarefied
    assert G.knudsen(6300, 110.0, 0.042, 1000.0) < 0.01    # huge body: continuum


def test_wall_noise_formula_against_a_monte_carlo():
    """The pressure row's band rests on relative noise = 2/sqrt(pi*hits). Check
    it independently: simulate a Poisson number of bounces, each delivering
    2*m*v_n with v_n drawn from the flux-weighted (Rayleigh) distribution."""
    rng = np.random.default_rng(21)
    lam, trials, sigma = 400, 6000, 1.3
    hits = rng.poisson(lam, trials)
    sums = np.array([np.sum(2.0 * rng.rayleigh(sigma, k)) for k in hits])
    measured = sums.std() / sums.mean()
    predicted = G.wall_noise(lam)
    assert abs(predicted - 2.0 / np.sqrt(np.pi * lam)) < 1e-15
    assert abs(measured - predicted) / predicted < 0.08
    assert not np.isfinite(G.wall_noise(0))


def test_free_molecular_cylinder_constant_by_integration():
    """8/3 derived independently: drag per rho*U^2*R is 2 * integral of cos^3."""
    from scipy.integrate import quad
    integral, _ = quad(lambda t: np.cos(t) ** 3, -np.pi / 2, np.pi / 2)
    force_per_rho_u2_r = 2.0 * integral
    cd = force_per_rho_u2_r / (0.5 * 2.0)     # divided by 0.5 * rho * U^2 * (2R)
    assert abs(cd - G.CD_FREE_MOLECULAR_CYLINDER) < 1e-12


def test_drag_coefficient_formula():
    cd = G.drag_coefficient(drag=3.0, rho_mass=2.0, speed=1.5, width=0.8)
    assert abs(cd - 3.0 / (0.5 * 2.0 * 1.5 ** 2 * 0.8)) < 1e-12
    assert not np.isfinite(G.drag_coefficient(float("nan"), 2.0, 1.5, 0.8))
    assert not np.isfinite(G.drag_coefficient(3.0, 2.0, 0.0, 0.8))   # stationary


def test_mach_angle():
    assert abs(G.mach_angle(2.0) - 30.0) < 1e-9        # sin(mu) = 1/2
    assert not np.isfinite(G.mach_angle(0.8))          # subsonic: no cone
    assert not np.isfinite(G.mach_angle(1.0))


def test_tunnel_speed_is_where_kinetic_energy_meets_the_barrier():
    """Derived independently: head-on, (1/4) m u^2 = (1/2) k d^2 at u = u*."""
    d, k, m = 0.042, 2.0e5, 1.0
    u = G.tunnel_speed(d, k, m)
    assert abs(0.25 * m * u * u - 0.5 * k * d * d) < 1e-9 * k * d * d
    assert abs(u - 26.563) < 1e-3                      # the shipped defaults


def test_probe_speed_limit_leaves_room_for_the_thermal_tail():
    d, k, m = 0.042, 2.0e5, 1.0
    u = G.tunnel_speed(d, k, m)
    assert abs(G.probe_speed_limit(d, k, m, 0.0) - 0.5 * u) < 1e-12   # cold gas: u*/2
    cap = G.probe_speed_limit(d, k, m, 1.0)
    assert abs((2 * cap + 6.0) - u) < 1e-12            # 2V + 3 sigma + 3 sigma = u*
    assert G.probe_speed_limit(d, k, m, 100.0) == 0.0  # too hot for any probe: clamps at 0
