"""BlastBox - neighbour search.

Every particle method in Stage 2+ (density, pressure force, viscosity) is a sum
over nearby particles. A uniform spatial structure turns the naive O(N^2)
all-pairs scan into ~O(N). Stage 1 uses scipy's cKDTree (already installed, one
call, exact); a hand-rolled uniform hash grid or a Taichi backend can replace this
later behind the same interface without touching the physics.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


class NeighborSearch:
    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=float)
        self.tree = cKDTree(self.pos)

    def counts(self, radius):
        """Neighbours within `radius` of each particle (excluding self)."""
        groups = self.tree.query_ball_point(self.pos, radius)
        return np.array([len(g) - 1 for g in groups], dtype=int)

    def pairs_within(self, radius):
        """List (length N) of neighbour-index arrays per particle (excludes self)."""
        groups = self.tree.query_ball_point(self.pos, radius)
        out = []
        for i, g in enumerate(groups):
            out.append(np.array([j for j in g if j != i], dtype=int))
        return out

    def nearest_dist(self):
        """Distance to each particle's single nearest neighbour (NaN if < 2 points)."""
        if self.pos.shape[0] < 2:
            return np.full(self.pos.shape[0], np.nan)
        d, _ = self.tree.query(self.pos, k=2)   # k=1 is the point itself
        return d[:, 1]
