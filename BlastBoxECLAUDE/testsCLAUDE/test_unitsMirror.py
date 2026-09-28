import numpy as np
import unitsMirror as U


def _cal():
    return U.Calibration(c_ref=1.4142135, t_ref=1.0, p_ref=57.3)


def test_reference_state_maps_onto_air():
    c = _cal()
    assert abs(c.speed(c.c_ref) - U.C_AIR) < 1e-6
    assert abs(c.temperature(c.t_ref) - U.T_AIR) < 1e-9
    assert abs(c.pressure(c.p_ref) - U.P_AIR) < 1e-6


def test_speed_scales_linearly():
    c = _cal()
    assert abs(c.speed(2 * c.c_ref) - 2 * U.C_AIR) < 1e-6


def test_time_unit_is_length_over_velocity():
    """Derived independently: one length unit is 1 m and the reference speed maps
    to 343 m/s, so a unit of simulation time is 1/343 s divided by c_ref."""
    c = _cal()
    expected = 1.0 / (U.C_AIR / c.c_ref)
    assert abs(c.seconds_per_unit - expected) < 1e-15
    # crossing the 14-unit room at the reference speed takes 14/c_ref time units
    crossing = (14.0 / c.c_ref) * c.seconds_per_unit
    assert abs(crossing - 14.0 / U.C_AIR) < 1e-12


def test_temperature_scaling_is_consistent_with_the_speed_map():
    """c ~ sqrt(T), so doubling T must multiply the mapped speed by sqrt(2)."""
    c = _cal()
    t2 = 2.0 * c.t_ref
    assert abs(c.temperature(t2) - 2 * U.T_AIR) < 1e-9
    c_at_t2 = c.c_ref * np.sqrt(2.0)
    assert abs(c.speed(c_at_t2) - U.C_AIR * np.sqrt(2.0)) < 1e-6


def test_degenerate_reference_does_not_raise():
    c = U.Calibration(c_ref=0.0, t_ref=0.0, p_ref=0.0)
    assert not np.isfinite(c.metres_per_second_per_unit)
    assert not np.isfinite(c.temperature(1.0))
    assert not np.isfinite(c.pressure(1.0))


def test_parse_speed_relative_mach_and_si():
    cal = _cal()
    c = 1.4142135
    assert U.parse_speed("2.5") == 2.5
    assert U.parse_speed(" 2.5 rel ") == 2.5
    assert U.parse_speed("-1") == -1.0                      # reverse direction
    assert abs(U.parse_speed("1.6mach", cal, c) - 1.6 * c) < 1e-12
    # 343 m/s is the reference speed, which maps back to c_ref exactly
    assert abs(U.parse_speed("343ms", cal, c) - cal.c_ref) < 1e-9
    assert abs(U.parse_speed("343 m/s", cal, c) - cal.c_ref) < 1e-9


def test_parse_speed_rejects_nonsense_with_a_reason():
    import pytest
    with pytest.raises(ValueError):
        U.parse_speed("")
    with pytest.raises(ValueError):
        U.parse_speed("fast")
    with pytest.raises(ValueError):
        U.parse_speed("500ms", calibration=None)            # no SI map yet
    with pytest.raises(ValueError):
        U.parse_speed("1.2mach", sound_speed=None)


def test_parse_speed_rejects_non_finite_numbers():
    """float() accepts all of these; any of them would put NaN into every
    particle the probe touched."""
    import pytest
    cal = _cal()
    for text in ("nan", "-nan", "inf", "-inf", "infinity", "nanms", "infmach"):
        with pytest.raises(ValueError):
            U.parse_speed(text, cal, 1.4142135)


def test_format_pair_labels_relative_and_si():
    s = U.format_pair(1.4142, 343.0, "m/s")
    assert "rel" in s and "~343" in s
