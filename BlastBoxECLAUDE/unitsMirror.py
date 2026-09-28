"""BlastBox - relative (simulation) units, and an approximate map to SI.

BlastBox runs in its OWN units. Three choices fix them:

    length unit  = 1.0          (so the world really is 14 x 8 "metres")
    particle mass = 1.0         (every particle, unless a zone overrides it)
    Boltzmann constant k_B = 1.0

Everything the ledger reports is in those units and is labelled `rel`. That is
deliberate: 6299 particles in a 14 x 8 m room is not air - real air holds about
1e25 molecules per cubic metre - so the simulation's temperature and pressure are
its own, not nature's.

The conversion below is a CALIBRATION, not a derivation. We take one reference
state (the equilibrated gas at startup), declare it to be "air at 20 C, 1 atm,
sound speed 343 m/s", and scale everything else from there by ratio. That keeps
the numbers honest: they answer "if this gas were air, what would this be?" and
nothing stronger. Any SI figure printed from here is prefixed with `~`.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

# SI reference state: dry air at 20 C, 1 atm.
C_AIR = 343.0          # m/s, speed of sound
T_AIR = 293.15         # K
P_AIR = 101325.0       # Pa
RHO_AIR = 1.204        # kg/m^3
METRES_PER_UNIT = 1.0  # declared: one simulation length unit is one metre


@dataclass(frozen=True)
class Calibration:
    """Ratio calibration pinned to one reference state of the simulation."""
    c_ref: float          # measured sound speed in the reference state (rel)
    t_ref: float          # temperature there (rel)
    p_ref: float          # pressure there (rel)

    @property
    def metres_per_second_per_unit(self) -> float:
        """SI speed of one simulation velocity unit."""
        return C_AIR / self.c_ref if self.c_ref > 0 else float("nan")

    @property
    def seconds_per_unit(self) -> float:
        """SI duration of one simulation time unit: length / velocity."""
        v = self.metres_per_second_per_unit
        return METRES_PER_UNIT / v if v > 0 else float("nan")

    def speed(self, v_rel: float) -> float:
        return v_rel * self.metres_per_second_per_unit

    def time(self, t_rel: float) -> float:
        return t_rel * self.seconds_per_unit

    def length(self, x_rel: float) -> float:
        return x_rel * METRES_PER_UNIT

    def temperature(self, t_rel: float) -> float:
        """Ratio scaling. Consistent with the speed map because c ~ sqrt(T)."""
        return T_AIR * (t_rel / self.t_ref) if self.t_ref > 0 else float("nan")

    def pressure(self, p_rel: float) -> float:
        return P_AIR * (p_rel / self.p_ref) if self.p_ref > 0 else float("nan")

    def frequency(self, f_rel: float) -> float:
        s = self.seconds_per_unit
        return f_rel / s if s > 0 else float("nan")


def format_pair(value_rel: float, value_si: float, unit: str, rel_fmt: str = "{:.4g}",
                si_fmt: str = "{:.4g}") -> str:
    """'1.414 rel  (~343 m/s)' - the house format for a calibrated quantity."""
    return f"{rel_fmt.format(value_rel)} rel  (~{si_fmt.format(value_si)} {unit})"


# Suffixes accepted by parse_speed, longest first so "m/s" is not read as "s".
_SPEED_SUFFIXES = (("m/s", "si"), ("mps", "si"), ("ms", "si"),
                   ("mach", "mach"), ("rel", "rel"))


def parse_speed(text, calibration=None, sound_speed=None):
    """Turn a typed probe speed into RELATIVE units.

        "2.5"  or "2.5rel"      2.5 relative units
        "500ms" or "500 m/s"    500 m/s-equivalent, through the calibration
        "1.6mach"               1.6 times the current sound speed

    Negative values reverse the direction. Raises ValueError with a message short
    enough to show on the panel.
    """
    s = str(text).strip().lower().replace(" ", "")
    if not s:
        raise ValueError("nothing typed")
    kind, number = "rel", s
    for suffix, k in _SPEED_SUFFIXES:
        if s.endswith(suffix):
            kind, number = k, s[: -len(suffix)]
            break
    try:
        value = float(number)
    except ValueError:
        raise ValueError(f"not a number: '{number}'") from None
    # float() happily accepts 'nan' and 'inf'. Either would put NaN into every
    # particle the probe touched.
    if not math.isfinite(value):
        raise ValueError(f"not a finite number: '{number}'")

    if kind == "rel":
        return value
    if kind == "mach":
        if sound_speed is None or not (sound_speed > 0):
            raise ValueError("sound speed unknown")
        return value * sound_speed
    if calibration is None:
        raise ValueError("no SI calibration yet - use rel or mach")
    per_unit = calibration.metres_per_second_per_unit
    if not (per_unit > 0):
        raise ValueError("calibration is degenerate")
    return value / per_unit
