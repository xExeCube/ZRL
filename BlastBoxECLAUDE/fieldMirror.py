"""BlastBox - the particle field: a struct-of-arrays layout and its builder.

Struct-of-arrays (separate flat numpy arrays per property, not a list of Particle
objects) is what keeps the sim fast and, later, portable to a GPU backend. Stage 1
fills position, velocity, mass and temperature; later stages add density, pressure
and internal energy as more arrays alongside these.

Particles start on a jittered triangular (hexagonal-close-packed) lattice: a
perfect grid gives artificial crystalline artefacts, pure random gives clumps, so
we jitter a triangular lattice - the standard isotropic 2D initial condition.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class ParticleField:
    pos: np.ndarray          # (N,2) positions, metres
    vel: np.ndarray          # (N,2) velocities, m/s  (zero in Stage 1)
    mass: np.ndarray         # (N,)  per-particle mass
    temperature: np.ndarray  # (N,)  per-particle temperature, RELATIVE units (k_B = 1)

    @property
    def n(self) -> int:
        return self.pos.shape[0]


def triangular_lattice(size, spacing):
    """Points of a triangular lattice covering [0,size_x] x [0,size_y].

    Rows are spaced by a*sqrt(3)/2 and every other row is offset by a/2, so each
    interior point has six equidistant neighbours at distance `spacing`.
    """
    dx = spacing
    dy = spacing * math.sqrt(3.0) / 2.0
    n_rows = int(math.floor(size[1] / dy)) + 1
    rows = []
    for j in range(n_rows):
        y = j * dy
        offset = 0.5 * dx if (j % 2) else 0.0
        n_cols = int(math.floor((size[0] - offset) / dx)) + 1
        x = offset + np.arange(n_cols) * dx
        x = x[x <= size[0]]
        rows.append(np.column_stack([x, np.full(x.shape[0], y)]))
    return np.vstack(rows)


def build_field(scene, world):
    """Place particles for `scene` inside `world` and apply zone overrides.

    Returns (ParticleField, info dict).
    """
    lattice = triangular_lattice(scene.world_size, scene.spacing)
    requested = lattice.shape[0]

    rng = np.random.default_rng(scene.seed)
    jitter = (rng.random(lattice.shape) * 2.0 - 1.0) * (scene.jitter * scene.spacing)
    pts = lattice + jitter

    keep = world.inside(pts, margin=scene.particle_radius)
    pts = pts[keep]
    n = pts.shape[0]

    vel = np.zeros((n, 2), dtype=float)
    mass = np.full(n, scene.particle_mass, dtype=float)
    temperature = np.full(n, scene.ambient_temperature, dtype=float)

    for zone in scene.zones:
        b = zone.box
        sel = ((pts[:, 0] >= b.x0) & (pts[:, 0] <= b.x1) &
               (pts[:, 1] >= b.y0) & (pts[:, 1] <= b.y1))
        if zone.temperature is not None:
            temperature[sel] = zone.temperature
        if zone.mass is not None:
            mass[sel] = zone.mass

    info = {"requested": requested, "placed": n, "rejected": requested - n}
    return ParticleField(pts, vel, mass, temperature), info
