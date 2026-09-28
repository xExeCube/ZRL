"""BlastBox - the space: world bounds and obstacles as a signed-distance field.

A point's "clearance" is its distance to the nearest solid boundary: positive in
open fluid, negative inside a wall or outside the domain. This one scalar field
does three jobs - reject particles that would sit in a wall (Stage 1), draw the
walls, and reflect particles off them (Stage 2) using the gradient as a normal.

The box distance is the standard 2D box SDF (Inigo Quilez), vectorised over an
(N,2) array of points with numpy.
"""
from __future__ import annotations
import numpy as np
from configMirror import Box


def sdf_box(points, box):
    """Signed distance from each point to an axis-aligned box.

    Negative inside, positive outside, zero on the boundary. points: (N,2).
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)   # (0,2) for empty input
    cx = 0.5 * (box.x0 + box.x1)
    cy = 0.5 * (box.y0 + box.y1)
    hx = 0.5 * (box.x1 - box.x0)
    hy = 0.5 * (box.y1 - box.y0)
    dx = np.abs(points[:, 0] - cx) - hx
    dy = np.abs(points[:, 1] - cy) - hy
    outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    inside = np.minimum(np.maximum(dx, dy), 0.0)
    return outside + inside


class World:
    """A rectangular domain with rectangular obstacles."""

    def __init__(self, domain, obstacles=()):
        self.domain = domain
        self.obstacles = list(obstacles)

    def clearance(self, points):
        """Distance to the nearest solid boundary. Positive in open fluid."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        phi = -sdf_box(points, self.domain)               # + inside the domain
        for obs in self.obstacles:
            phi = np.minimum(phi, sdf_box(points, obs))    # + outside each obstacle
        return phi

    def inside(self, points, margin=0.0):
        """Boolean mask: points sitting in open fluid by more than `margin`."""
        return self.clearance(points) > margin

    def normal(self, points, eps=1e-4):
        """Unit vector pointing into the fluid (up the clearance gradient).

        On an obstacle's medial axis the central differences are bit-identical
        and the gradient is exactly zero. That is measure-zero for continuously
        jittered particles, but a zero "normal" would silently disable both the
        lift and the reflection and let a particle walk through a wall, so
        degenerate rows retry at a coarser epsilon and then fall back to a
        radial direction from the domain centre. Every returned row is unit
        length.
        """
        p = np.asarray(points, dtype=float).reshape(-1, 2)
        if p.shape[0] == 0:
            return np.zeros((0, 2))

        def grad(step):
            ex = np.array([step, 0.0])
            ey = np.array([0.0, step])
            gx = self.clearance(p + ex) - self.clearance(p - ex)
            gy = self.clearance(p + ey) - self.clearance(p - ey)
            return np.column_stack([gx, gy])

        g = grad(eps)
        mag = np.linalg.norm(g, axis=1)
        degenerate = mag < 1e-12
        if np.any(degenerate):
            g2 = grad(eps * 100.0)
            g[degenerate] = g2[degenerate]
            mag = np.linalg.norm(g, axis=1)
            still = mag < 1e-12
            if np.any(still):
                d = self.domain
                centre = np.array([0.5 * (d.x0 + d.x1), 0.5 * (d.y0 + d.y1)])
                radial = p[still] - centre
                rmag = np.linalg.norm(radial, axis=1, keepdims=True)
                radial = np.where(rmag > 1e-12, radial / np.maximum(rmag, 1e-12),
                                  np.array([1.0, 0.0]))
                g[still] = radial
                mag = np.linalg.norm(g, axis=1)
        return g / np.maximum(mag, 1e-30)[:, None]

    # --- measures used by the Stage-2 gauges ---------------------------------
    # Both take an `inset`: reflection happens when a particle CENTRE comes
    # within d/2 of a wall, so the container the centres actually occupy is the
    # geometry eroded by that radius. Using the raw geometry biases the two
    # pressure routes against each other by about a percent - invisible under a
    # loose tolerance and dominant under a tight one.
    #
    # Both assume axis-aligned obstacles that do not overlap each other. That
    # holds for every shipped scene; a scene with overlapping obstacles must
    # revisit these. Dilated obstacle corners are treated as square rather than
    # rounded, a small overestimate local to each corner.

    def _inset_geometry(self, inset):
        d = self.domain
        dom = Box(d.x0 + inset, d.y0 + inset, d.x1 - inset, d.y1 - inset)
        obs = []
        for o in self.obstacles:
            x0 = max(o.x0 - inset, dom.x0)
            y0 = max(o.y0 - inset, dom.y0)
            x1 = min(o.x1 + inset, dom.x1)
            y1 = min(o.y1 + inset, dom.y1)
            if x1 > x0 and y1 > y0:
                obs.append(Box(x0, y0, x1, y1))
        return dom, obs

    def fluid_area(self, inset=0.0):
        """Area available to particle centres: domain eroded, obstacles dilated."""
        dom, obs = self._inset_geometry(inset)
        area = max(0.0, (dom.x1 - dom.x0) * (dom.y1 - dom.y0))
        for o in obs:
            area -= (o.x1 - o.x0) * (o.y1 - o.y0)
        return max(0.0, area)

    def wall_perimeter(self, inset=0.0, tol=1e-9):
        """Total wetted wall length, for the wall-momentum-flux pressure gauge.

        An obstacle edge flush against the domain boundary is not a second wall -
        it replaces that stretch of the domain edge. Counting both would inflate
        the perimeter and make the two pressure routes disagree for no physical
        reason, so flush stretches are removed from the domain edge and the
        obstacle edge is skipped.
        """
        dom, obs = self._inset_geometry(inset)
        dom_edges = [
            (1, dom.y0, dom.x0, dom.x1),   # bottom
            (1, dom.y1, dom.x0, dom.x1),   # top
            (0, dom.x0, dom.y0, dom.y1),   # left
            (0, dom.x1, dom.y0, dom.y1),   # right
        ]
        covered = [0.0] * len(dom_edges)
        total = 0.0
        for o in obs:
            for (ax, val, lo, hi) in (
                (1, o.y0, o.x0, o.x1),
                (1, o.y1, o.x0, o.x1),
                (0, o.x0, o.y0, o.y1),
                (0, o.x1, o.y0, o.y1),
            ):
                flush = False
                for k, (dax, dval, dlo, dhi) in enumerate(dom_edges):
                    if ax == dax and abs(val - dval) < tol:
                        overlap = min(hi, dhi) - max(lo, dlo)
                        if overlap > tol:
                            covered[k] += overlap
                            flush = True
                        break
                if not flush:
                    total += (hi - lo)
        for k, (ax, val, lo, hi) in enumerate(dom_edges):
            total += max(0.0, (hi - lo) - covered[k])
        return total
