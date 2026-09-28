"""BlastBox - central configuration, scene presets, and the ZRL spec-4 palette.

Everything tunable lives here so the rest of the package carries no magic numbers.
This module imports nothing heavy (no numpy, no pygame), so it stays cheap to
import from tests and headless tools.

UNITS: simulation units throughout - length unit 1, particle mass 1, k_B = 1.
See unitsMirror.py for the approximate map to SI.
"""
from __future__ import annotations
from dataclasses import dataclass, field

# --- ZRL spec-4 palette (non-negotiable for this project) --------------------
# RGB tuples, 0-255. Names mirror the ZRL build-spec CSS variables.
BG        = (0x00, 0x00, 0x00)   # page background
PANEL     = (0x13, 0x13, 0x13)   # side panel
BSHADOW   = (0x2F, 0x05, 0x42)   # subtle borders / shadow
BLINE     = (0x50, 0x2F, 0xA5)   # active borders
ACCENT    = (0x48, 0x5F, 0xC7)   # accent, headers, control hints
TEXT      = (0xFF, 0xFF, 0xFF)
HI        = (0x00, 0x9E, 0x86)   # highlight / PASS
LINK      = (0xE2, 0x22, 0x11)   # links / FAIL
WALL_FILL = BSHADOW              # obstacle interior (on-palette dark purple)
WALL_EDGE = BLINE                # obstacle outline


@dataclass(frozen=True)
class Box:
    """Axis-aligned rectangle [x0,x1] x [y0,y1] in world units (metres)."""
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class Zone:
    """A rectangular region whose particles get property overrides at build time.

    This is the Stage-1 hook for "different environments" (a hot pocket, a dense
    region). A field left as None keeps the scene's ambient default.
    """
    box: Box
    temperature: float | None = None   # relative units (k_B = 1)
    mass: float | None = None          # per-particle mass


@dataclass
class GasConfig:
    """Stage 2 - molecular gas.

    The contact law is a linear (Hookean) repulsive spring acting only while two
    particles overlap:  |F| = stiffness * (diameter - r)  for r < diameter.
    `diameter` is therefore both the particle size and the force cutoff.

    `diameter` is deliberately much smaller than the lattice spacing (0.3x by
    default): particles must fly freely and collide occasionally, or the system
    is a solid, not a gas. At these defaults the area fraction is about 8%.
    """
    diameter: float = 0.042            # collision diameter d, also the force cutoff
    # Stiff enough that particles do not tunnel through each other. Two equal
    # soft spheres meeting head-on pass straight through once their closing speed
    # exceeds d*sqrt(2k/m): 8.4 at the earlier k = 2e4, which the fast tail of
    # particles bounced by a Mach 1.6 probe already crossed, and 26.6 at 2e5.
    # The cost: the contact period is sqrt(10) shorter, so dt is roughly half
    # what it was (~2.0e-4 with the probe off) and a frame covers about half the
    # simulated time. Raise `substeps` if the frame rate has room. It also makes
    # the gas closer to hard discs (d_eff/d ~ 0.93 rather than 0.79).
    stiffness: float = 200000.0        # contact spring constant k
    temperature: float = 1.0           # ambient initial temperature (k_B = 1)
    restitution: float = 1.0           # wall restitution; 1.0 = perfectly elastic
    dt_safety_contact: float = 0.02    # dt as a fraction of the contact period
    dt_safety_travel: float = 0.10     # dt as a fraction of diameter / closing speed
    substeps: int = 12                 # physics steps per rendered frame
    seed: int = 20260923

    # "uniform_speed" starts every particle at the same speed with a random
    # direction, so <v>^2/<v^2> begins at exactly 1.0 and must RELAX to pi/4
    # through collisions. "maxwell" starts already at equilibrium, which makes
    # that ledger row a restatement of the initial draw rather than a measurement.
    ic_mode: str = "uniform_speed"

    # Settling is set in TIME, not steps: dt changes whenever it is re-frozen, so
    # a step count would mean a different physical wait each time. The collision
    # time here is about 0.22, so 2.0 is roughly nine collisions per particle -
    # enough for the speed distribution to relax. It is NOT enough for pressure
    # pulses (e.g. from the hot pocket) to die away; the pressure row accounts
    # for that transient explicitly rather than waiting it out.
    equilibrate_time: float = 2.0

    # The supersonic probe: a rigid disk dragged through the gas. Above Mach 1 it
    # cannot push the gas out of the way in time and a Mach cone forms.
    disk_enabled: bool = False
    # 0.45, not 0.6: the probe runs along the room's mid-line, straight through
    # the two_rooms doorway, which is 1.2 tall. A 0.6 disk is exactly 1.2 across
    # and grazes both wall slabs with ZERO clearance, pinching any particle caught
    # there between a moving surface and a fixed one. 0.45 leaves room for about
    # two particle diameters on each side.
    disk_radius: float = 0.45
    disk_mach: float = 1.6             # default speed as a multiple of sound speed
    # Cap on the manual speed control. None = derive it from the contact law: a
    # particle bounced off the probe leaves at up to 2V, and above the tunnelling
    # speed d*sqrt(2k/m) it passes through whatever it meets next, so no shock
    # can form. The derived cap keeps 2V plus the thermal tail below that limit.
    disk_speed_max: float | None = None


@dataclass
class Scene:
    """A complete setup: the world box, its obstacles, the particle lattice
    parameters, any property zones, and the Stage-2 gas parameters."""
    name: str
    world_size: tuple = (14.0, 8.0)     # (width, height)
    spacing: float = 0.14               # target inter-particle spacing a
    jitter: float = 0.20                # lattice jitter as a fraction of spacing
    particle_radius: float = 0.05       # keep-away from walls at placement time
    particle_mass: float = 1.0
    ambient_temperature: float = 1.0    # RELATIVE units (k_B = 1), not kelvin
    smoothing_length: float = 0.30      # h: Stage-1 neighbour radius (~2 x spacing)
    seed: int = 20260904
    obstacles: tuple = ()               # tuple[Box, ...]
    zones: tuple = ()                   # tuple[Zone, ...]
    gas: GasConfig = field(default_factory=GasConfig)

    def make_world(self):
        # imported lazily to avoid a config <-> space import cycle
        from spaceMirror import World
        w, h = self.world_size
        return World(Box(0.0, 0.0, w, h), self.obstacles)


# --- scene presets -----------------------------------------------------------

def open_box() -> Scene:
    """A single open room. The cleanest scene for the gas gauges, because with no
    obstacles the wall-pressure and virial-pressure routes are both exact."""
    return Scene(
        name="open_box",
        world_size=(14.0, 8.0),
        # 4x ambient: a hot pocket, in relative units. Stage 2 honours this, so
        # the pocket really does expand into the rest of the room.
        zones=(Zone(Box(6.4, 3.4, 7.6, 4.6), temperature=4.0),),
    )


def two_rooms() -> Scene:
    """Two rooms split by a wall with a central doorway - the multi-room hook
    from the second reference video."""
    w, h = 14.0, 8.0
    t = 0.30                       # wall thickness
    midx = w / 2.0
    door_lo, door_hi = 3.4, 4.6    # doorway gap in y
    walls = (
        Box(midx - t / 2, 0.0, midx + t / 2, door_lo),    # wall below the door
        Box(midx - t / 2, door_hi, midx + t / 2, h),      # wall above the door
    )
    return Scene(
        name="two_rooms",
        world_size=(w, h),
        obstacles=walls,
        zones=(Zone(Box(2.6, 3.0, 4.2, 5.0), temperature=4.0),),
    )


# name -> builder (call to get a fresh Scene)
SCENES = {"open_box": open_box, "two_rooms": two_rooms}
DEFAULT_SCENE = "two_rooms"
