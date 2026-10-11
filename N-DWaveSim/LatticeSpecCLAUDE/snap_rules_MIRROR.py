"""Snap rules of spec §3.2: for each tiling / grid / isomer, are all vertices and centres within two periods nodes?

sq: nodes at integer (x, y). tri: nodes at x = a + b/2, y = b*sqrt(3)/2 for integer a, b.
Run: python snap_rules_MIRROR.py      (numpy only, instant)
"""
import numpy as np
S3 = np.sqrt(3)
def on_sq(p): return np.allclose(p, np.round(p), atol=1e-9)
def on_tri(p):  # x = a + b/2, y = b*sqrt3/2
    b = p[..., 1] * 2 / S3; a = p[..., 0] - b / 2
    return np.allclose(b, np.round(b), atol=1e-9) and np.allclose(a, np.round(a), atol=1e-9)
def cells(n, R, rot, A1, A2, C):
    """vertices and centres of the cells within 2 periods"""
    V, P = [], []
    for c in C:
        for i in range(-2, 3):
            for j in range(-2, 3):
                ctr = c + i * A1 + j * A2; P.append(ctr)
                sgn = 1 if len(C) == 1 or np.allclose(c, C[0]) else -1   # down triangles are flipped
                for k in range(n):
                    a = rot + 2 * np.pi * k / n + (np.pi if sgn < 0 else 0)
                    V.append(ctr + R * np.array([np.cos(a), np.sin(a)]))
    return np.array(V), np.array(P)
rows = []
# square grid isomer: side s whole (s = 6), R = s/sqrt2, edges axis-aligned -> vertices at 45 deg
s = 6; R = s / np.sqrt(2); V, P = cells(4, R, np.pi / 4, np.array([s, 0.]), np.array([0., s]), [np.array([s / 2, s / 2])])
rows.append(('square/sq grid isomer, side 6', on_sq(V), on_sq(P)))
# square ZRL isomer: R whole (5), vertex on x-axis; centres (R,R),(R,-R)
R = 5; V, P = cells(4, R, 0, np.array([R, R + 0.]), np.array([R, -R + 0.]), [np.zeros(2)])
rows.append(('square/sq ZRL isomer, R 5', on_sq(V), on_sq(P)))
# hexagon ZRL isomer on tri: R whole (4), centres R(e1+e2), R(2e2-e1)
e1, e2 = np.array([1, 0.]), np.array([.5, S3 / 2]); R = 4
V, P = cells(6, R, 0, R * (e1 + e2), R * (2 * e2 - e1), [np.zeros(2)])
rows.append(('hexagon/tri ZRL isomer, R 4', on_tri(V), on_tri(P)))
# hexagon other isomer: R = sqrt3 * 2, rotated 30 deg; centres R*sqrt3 apart along 0 and 60 deg
R = 2 * S3; V, P = cells(6, R, np.pi / 6, R * S3 * e1, R * S3 * e2, [np.zeros(2)])
rows.append(('hexagon/tri other isomer, R = 2 sqrt3', on_tri(V), on_tri(P)))
# triangle, sides along rows: side s = 6 (multiple of 3) and 4 (not); up triangle vertex at 90 deg
for s in (6, 4):
    R = s / S3; up = (s * e1 + s * e2) / 3
    V, P = cells(3, R, np.pi / 2, s * e1, s * e2, [up, 2 * up])
    rows.append((f'triangle/tri rows, side {s}', on_tri(V), on_tri(P)))
# triangle ZRL isomer: R whole (3), vertex on x-axis; up/down cells
R = 3; A1, A2 = R * S3 * np.array([np.cos(np.pi/6), np.sin(np.pi/6)]), R * S3 * np.array([0, 1.])
# up triangle (vertex at 0 deg) at origin; down neighbour across the edge opposite that vertex at (-R, 0)
V, P = cells(3, R, 0, A1 * 1, A2 * 1, [np.zeros(2), np.array([-R, 0.])])
rows.append(('triangle/tri ZRL isomer, R 3', on_tri(V), on_tri(P)))
for name, v, p in rows: print(f'{name:<40} vertices on nodes: {v}   centres on nodes: {p}')
