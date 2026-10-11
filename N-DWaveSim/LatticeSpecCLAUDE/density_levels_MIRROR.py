"""Circle overlap and density levels of the three flavor tilings (spec §3.3, §3.4).

For each tiling (triangle, square, hexagon; cells of circumradius R) it prints:
  - the neighbour shells: centre distance / R, how many cells, and whether the circumcircles overlap,
    touch or miss (d < 2R, d = 2R, d > 2R);
  - the edge-neighbour overlap width 2R(1 - cos(pi/n));
  - the density levels (how many circumcircle discs cover a point) and their area shares, Monte Carlo
    against the exact mean (total disc area / cell area);
  - the same with the incircles added (they never overlap each other);
  - the counts at the tiling's vertices, which lie exactly on circles (the point-sampling tie).
Every quantity is a ratio (degree 0), so it runs at R = 1 and at R = 1.7 and checks they agree.

Run: python density_levels_MIRROR.py      (numpy only, ~10 s)
"""
import numpy as np

def tiling(n, R):
    """Period vectors a1, a2, the cell centres in one period, and the vertices in one period."""
    if n == 4:                                  # squares, grid isomer: side R*sqrt2
        s = R * np.sqrt(2); a1, a2 = np.array([s, 0.0]), np.array([0.0, s])
        C = [np.array([s / 2, s / 2])]; V = [np.zeros(2)]
    elif n == 6:                                # hexagons, side R: centres on a triangular lattice, spacing R*sqrt3
        s = R * np.sqrt(3); a1, a2 = np.array([s, 0.0]), np.array([s / 2, s * np.sqrt(3) / 2])
        C = [np.zeros(2)]
        V = [R * np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)]), R * np.array([0.0, 1.0])]
    else:                                       # triangles, side R*sqrt3: up and down cells per rhombus
        s = R * np.sqrt(3); a1, a2 = np.array([s, 0.0]), np.array([s / 2, s * np.sqrt(3) / 2])
        C = [(a1 + a2) / 3, 2 * (a1 + a2) / 3]; V = [np.zeros(2)]
    return a1, a2, C, V

def centres_near(a1, a2, C, reach=4):
    return np.array([c + i * a1 + j * a2 for c in C for i in range(-reach, reach + 1) for j in range(-reach, reach + 1)])

def report(n, R, rng, samples=400_000):
    a1, a2, C, V = tiling(n, R)
    P = centres_near(a1, a2, C)
    out = {}
    # neighbour shells of the first centre
    d = np.sort(np.linalg.norm(P - C[0], axis=1))[1:]
    shells = []
    for x in np.unique(np.round(d / R, 9)):
        if x > 3.01: break
        k = int((np.abs(d / R - x) < 1e-6).sum())
        shells.append((x, k, 'overlap' if x < 2 - 1e-9 else ('touch' if x < 2 + 1e-9 else 'miss')))
    out['shells'] = shells
    out['lens'] = 2 * (1 - np.cos(np.pi / n))
    # Monte Carlo over one period
    u = rng.random((samples, 2)); X = u[:, :1] * a1 + u[:, 1:] * a2
    d2 = ((X[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    m_circ = (d2 < R * R).sum(1)
    m_both = m_circ + (d2 < (R * np.cos(np.pi / n)) ** 2).sum(1)
    cell_area = abs(a1[0] * a2[1] - a1[1] * a2[0]) / len(C)
    out['exact_mean'] = np.pi * R * R / cell_area
    out['inc_cover'] = np.pi * (R * np.cos(np.pi / n)) ** 2 / cell_area
    for key, m in (('circ', m_circ), ('both', m_both)):
        v, c = np.unique(m, return_counts=True)
        out[key] = (dict(zip(v.tolist(), (c / samples).round(4).tolist())), m.mean())
    # vertices: how many circles pass exactly through, how many strictly contain
    dv = np.linalg.norm(P - V[0], axis=1) / R
    out['vertex'] = (int((np.abs(dv - 1) < 1e-9).sum()), int((dv < 1 - 1e-9).sum()))
    return out

if __name__ == '__main__':
    rng = np.random.default_rng(20261011)
    names = {3: 'triangle', 4: 'square', 6: 'hexagon'}
    for n in (3, 4, 6):
        a, b = report(n, 1.0, rng), report(n, 1.7, rng)
        print(f'== {names[n]} (n = {n})')
        print('   neighbour shells (d/R, cells, circles):', ', '.join(f'{x:.6f} x{k} {s}' for x, k, s in a['shells']))
        print(f'   edge-neighbour overlap width 2R(1 - cos(pi/n)) = {a["lens"]:.6f} R')
        lv, mean = a['circ']
        lo = int(np.floor(a['exact_mean'])); share_hi = a['exact_mean'] - lo
        print(f'   circumcircle levels (MC): {lv}  mean {mean:.4f}   exact mean {a["exact_mean"]:.6f} '
              f'-> exact shares {lo}: {1 - share_hi:.6f}, {lo + 1}: {share_hi:.6f}')
        print(f'   + incircles (levels, MC): {a["both"][0]}  mean {a["both"][1]:.4f}   '
              f'incircles cover {a["inc_cover"]:.6f} of the plane, never overlap')
        print(f'   a vertex lies ON {a["vertex"][0]} circles and strictly inside {a["vertex"][1]}')
        same = (a['shells'] == b['shells'] and abs(a['exact_mean'] - b['exact_mean']) < 1e-12
                and a['vertex'] == b['vertex'] and abs(a['circ'][1] - b['circ'][1]) < 0.01)
        print(f'   R = 1.7 gives the same ratios: {"PASS" if same else "FAIL"}')
