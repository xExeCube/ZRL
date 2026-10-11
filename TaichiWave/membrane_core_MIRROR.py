"""ZRL 2D wave membrane -- Taichi port, CORE (no GUI).

A value-for-value mirror of FormConstantsCLAUDE/FlavorRenderersCLAUDE/wave_membrane_MIRROR.html
(the merged build of 28/09/2026):
  - the same lattices (square, triangular), stencils (5-, 9-point; 6-neighbour triangular),
    domain shapes, radius snaps, media, boundaries, sources and drive waveforms;
  - the same update ORDER, down to the order of the neighbour sums, so a field computed here
    agrees with the web app's to floating-point rounding (checked by crosscheck_js_MIRROR.py).

Split of work: everything that runs once per reset (masks, per-cell Courant numbers, source
placement, seeding) is host-side numpy, written with the same arithmetic as the JS; everything
that runs every step is a Taichi kernel on the GPU (or CPU).

Array convention: fields are indexed [i, j] with i = column (x), j = row (y). The JS app stores
the same grid flat as k = j*N + i, so a JS array reshaped to (N, N) is [j, i] -- transpose it.

Documented deviations from the web app (29/09/2026; the cross-check applies the same changes to
the web app's code as PATCHES in ToolsCLAUDE/js_harness_MIRROR.mjs, so it still compares
value for value):
  - neumann-corner: every Neumann corner equals its diagonal interior neighbour of THIS step.
    The web app's sequential ring loop left (0,0) holding (1,0) from the rotating buffer's old
    contents -- the time level three steps back (traced 30/09/2026).
  - damped CFL limit (01/10/2026): co_limit() includes the damping, C_max(d) = C_max(0) sqrt(1 - d/2);
    the web app showed C_max(0) whatever the damping (Co = 0.705 at d = 0.02 on sq5 blew up while
    it read "stable"). co_limit(damped=False) is the web app's value. No field changes.
  - source snapping with solids (01/10/2026, the user's decision): a source requested on a SOLID
    cell goes to the nearest free cell of the main free region (never a letter's sealed counter);
    Membrane.src_moved says so. Without solids the web app's walk toward the centre is kept exactly.
  - walls (round 4, 04/10/2026, NotesCLAUDE/walls_MIRROR.md): solids are PINNED (u = 0) or RIGID
    (link removal, R0), and the sq9 diagonals and the triangular links between two free cells are
    CORNER-BLOCKED when both of their common neighbours are non-free (pinned blocker: the link acts
    as a link to a pinned cell, -w u_i; both blockers rigid: dropped). This also stops the leak of
    one-cell diagonal PINNED walls (2.3% sq9, 17% tri). Params.edge = 'rigid' makes the domain
    mask's edge (and the triangular array's ring) rigid. With edge = 'pinned' (the default) a
    blocker pair made only of domain-exterior cells does NOT block (the web app has no blocking),
    so without solids, or with pinned solids that block no link, the solver runs the web app's
    exact path (bit-identical; the cross-check is unaffected).
  - triangular-lattice modes (round 4b, 05/10/2026, NotesCLAUDE/walls_MIRROR.md section 13): the
    domain shape 'array' (the whole lattice array; on the square lattice exactly the full square).
    On the triangular 'array' the bc control acts: dirichlet = the clamped ring, neumann = a RIGID
    ring (link removal), periodic = the rhombic torus (the six neighbours taken mod N), mur = not
    available yet (falls back to the clamped ring; bc_info() says why). Every web-app shape on the
    triangular lattice keeps the web app's clamped ring whatever bc says (bit-identical).
  - vertex-source defaults (round 4c, 05/10/2026, NotesCLAUDE/vertex_choppiness_MIRROR.md section 7):
    Params.vtx_inset defaults to 'wall' (the web app's default is 'radial', whose pulses the domain edge
    cuts by 1.2-27% at t = 0: 4.5-49% of the energy turns into grid-scale chop; 'wall' cuts <= 0.012%).
    A setting, not a code change: every comparison with the web app sets vtx_inset='radial' explicitly.
  - vtx-drive-rest (05/10/2026): with vtx_drive, the vertices start from rest like 'cont' (no seeded
    bumps; Params.vtx_drive_bump = True restores the web app, whose hard drive dents each bump centre to
    drive(0) = 0 at step 1). The harness applies the same rule to the web app (PATCH 'vtx-drive-rest',
    keyed on S.vtxDriveBump, which only the harness sets).
  - neumann-clip (09/10/2026, the user's decision; NotesCLAUDE/v1_decisions_MIRROR.md, walls_MIRROR.md 14.2): the
    sq9 stencil with the Neumann bc and a shape CLIPPED by the array (a circle at radius 1.3) had an asymmetric
    operator where the shape leaves the ring (an exterior ring cell holds 0 next to a ghost) and the run grew
    without bound (max|u| 6.7e21 after 200,000 steps). A sq9 diagonal into a ghost ring cell whose common
    neighbour on the ring is exterior (pinned edge) now reads 0, like a pinned ring solid (the round-4 rule C1,
    _word / _nring_common; not for a link into an array corner, the cell's own ghost). The words exist for
    these configurations (walls_needed: _neu_clip()), so they run the walls kernel. The harness applies the
    same rule to the web app (PATCH 'neumann-clip').
  - default drive frequency (09/10/2026, the user's decision): Params.freq = 0.03125, a 16-cell wavelength at
    the default Co 0.5 (a plain value, not coupled to Co); the web app's is 0.12 (WEBAPP_FREQ, 4.2 cells). A
    setting, not a code change: every comparison with the web app sets freq=WEBAPP_FREQ explicitly.
  - soft driven vertices (09/10/2026, the user's decision "soft as default, leave a hard toggle"): with vtx_drive,
    Params.vtx_soft = True ADDS the drive at the vertex cells (the beam's soft kernel path), so they are
    transparent to returning waves; each harmonic is scaled by soft_gain() (the lattice Green's function at the
    cell) so a lone vertex radiates the hard vertex's wave (measured 1.000 in free space; 0.79-0.87 at the
    defaults, where the hard cell also re-radiates its corner's echoes). vtx_soft=False is the web app's hard
    drive, bit for bit; every comparison with the web app sets it. vtx_drive_bump with soft vertices seeds the
    bumps as before and the drive adds to them (no dent).
  - source helpers with no web-app counterpart in the core (the web app has them in its panel):
    source_note() (its srcNote: clearance, clip fraction, the hint, capped corners), drive_warnings()
    (hard drives under PPW_MIN cells per wavelength, cells mode) and phase_lag() (the grid's phase lag
    per domain crossing, port-only).
Additions with no web-app counterpart: the 'unlicensed' domain (a rounded 2:1 plate), solid
objects / painted walls (solids_MIRROR.py; the mask becomes domain AND NOT solid), and the
per-step recorder inside the fused kernel (the web app's record(): energy, peak, probes).
"""
import math
import os
import sys
from dataclasses import dataclass, asdict, replace
import numpy as np
import taichi as ti

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beam_MIRROR as bm                # directional sources (01/10/2026)

# The app's version (05/10/2026, Claude Opus 5.5): v1.0 = rounds 1-4 (port, solids and the plate, the
# beam / packet and scale mode, walls and the triangular modes). The viewer imports it for its window
# title, its main panel and --version; CHANGELOG_MIRROR.md lists what each round added.
VERSION = '1.0'
VERSION_DATE = '05/10/2026'
APP_NAME = 'ZRL wave membrane'

PI = math.pi
TAU = 2 * PI
S3H = math.sqrt(3) / 2
CO_LIMIT_5 = 1 / math.sqrt(2)          # square, 5-point
CO_LIMIT_9 = math.sqrt(3) / 2          # square, 9-point isotropic
CO_LIMIT_T = math.sqrt(2 / 3)          # triangular, 6-neighbour
PPW_MIN = 8

# 'unlicensed' (appended 29/09/2026, the user's request): the license plate of Death Grips'
# "Government Plates" cover -- a rounded rectangle, width:height = 2:1, corner radius 0.065 H,
# measured from the cover. dom_R() is its HALF-WIDTH R, so W = 2R and H = R.
# 'array' (appended 05/10/2026, port-only): the WHOLE lattice array is the domain. Square lattice:
# exactly the full-grid square ('square' at rad >= 0.999: the same code path, bit-identical).
# Triangular lattice: the full 60-degree rhombus of the array, every cell inside except the outer
# ring, which is the boundary the bc control governs (clamped / rigid / the rhombic torus, where the
# ring cells are ordinary cells). Rotation and radius do not apply.
SHAPES = ['square', 'n4', 'circle', 'n3', 'n5', 'n6', 'n8', 'n12', 'rhomb', 'unlicensed', 'array']   # the web app's order + 2
FIXED_SHAPES = ('array',)              # shapes that ignore p.rot and p.rad entirely
PLATE_ASPECT = 2.0                     # W / H
PLATE_CORNER = 0.065                   # corner radius / H
PLATE_ARC_SEG = 16                     # segments per corner arc in shape_poly()
# 09/10/2026 (absorbing boundaries, NotesCLAUDE/absorbing_MIRROR.md): mur2, cpml and sponge are port-only. They act
# on the FULL array only (sq_full() / the 'array' shape); a masked shape gets its absorbing boundary from
# Params.edge = 'open' (a sponge outside the drawn edge). The order is the B cycle's.
BCS = ['dirichlet', 'mur', 'mur2', 'cpml', 'sponge', 'neumann', 'periodic']
WEBAPP_BCS = ('dirichlet', 'mur', 'neumann', 'periodic')
ABSORBING_BCS = ('mur', 'mur2', 'cpml', 'sponge')
LAYER_BCS = ('cpml', 'sponge')          # the bcs with a layer (a frame inside the array)
# the triangular lattice's boundaries (05/10/2026): they act on the 'array' shape only. Mur is not
# available there yet (NotesCLAUDE/absorbing_boundaries_MIRROR.md: the T1 rule is unstable on the
# triangular lattice above Co 0.5); a request falls back to the clamped ring (bc_info() says so).
# 09/10/2026: the sponge frame acts there too (the triangular array's absorbing boundary).
TRI_BCS = ['dirichlet', 'neumann', 'periodic', 'sponge']
TRI_MUR_REASON = ('Mur is not available on the triangular lattice yet: its first-order rule is unstable there '
                  'above Co 0.5 (absorbing-boundary round) -- the ring stays clamped (Dirichlet)')
TRI_MUR2_REASON = ('Mur 2nd order is not available on the triangular lattice: the one-way rules need a lattice axis '
                   'normal to the wall, and the first-order rule is already unstable there above Co 0.5 -- the ring '
                   'stays clamped (Dirichlet); use the sponge')
TRI_CPML_REASON = ('a PML on the triangular lattice is open research (its 6-neighbour Laplacian mixes the stretched '
                   'directions) -- the ring stays clamped (Dirichlet); use the sponge')
TRI_MASKED_REASON = ('on the triangular lattice the boundary control acts on the "array" shape only: every other '
                     'shape keeps the web app\'s clamped ring (its own edge is set by "edge": pinned / rigid / open)')
MASKED_ABS_REASON = ('{bc} acts on the full array only (the square at radius 1, or "array"): for this shape use '
                     'edge: open (a sponge outside its edge) -- the array ring stays clamped (Dirichlet)')
OPEN_PERIODIC_REASON = ('periodic is exclusive with an absorbing edge (edge: open): the array ring stays clamped '
                        '(Dirichlet); the layer keeps the shape off the ring anyway')
# the absorbing layers' parameters (NotesCLAUDE/absorbing_boundaries_MIRROR.md sections 4.2, 4.3, 4.8, 5)
# 'high' (10/10/2026): alpha 0, not the research's 0.05 -- a complex-frequency shift absorbs waves with omega < alpha less:
# in the 2D box a sigma 24 pulse came back at 3.1e-2 with alpha 0.05, 2.0e-6 with alpha 0, which matches alpha 0.05 on the
# research's sigma-3 box (2.5e-6 / 1.8e-6; ScratchCLAUDE/Round5CLAUDE/AbsorbCLAUDE/probe_cpml_high_MIRROR.log)
CPML_LEVELS = {'std': dict(L=10, p=3, fac=2.0, alpha=0.0), 'high': dict(L=20, p=3, fac=2.0, alpha=0.0)}
CPML_R0 = 1e-6                          # the target reflection in sigma_max = fac (p + 1) ln(1/R0) / (2 L)
SPONGE_P = 2                            # the ramp's power: d = d_max s^p, s = depth into the layer / L
SPONGE_L_MIN = 20.0                     # the small default width (cells), and domain mode's minimum
SPONGE_LEVELS = 254                     # damping levels packed in the mask value: codes 2..255
MEDIA = ['uniform', 'slab', 'lens']
SOURCES = ['impulse', 'pulse', 'cont', 'slit', 'mode', 'vtx', 'beam', 'packet']
# 'beam' and 'packet' (01/10/2026) have no web-app counterpart: directional sources, beam_MIRROR.py
BEAM_SOURCES = ('beam', 'packet')
WAVES = ['sine', 'square', 'saw', 'tri']
# Scale modes (02/10/2026, NotesCLAUDE/scale_MIRROR.md). 'cells' (the default, the web app's): every
# size is in cells, so a larger N is a bigger world with the same small pulse. 'domain': the sizes
# are fractions of the domain's inradius r_in (inner_R()), so a larger N shows the SAME picture with
# less grid error. Each entry: (the cell-valued Params field, its fraction field, kind); 'freq' is
# special: lambda = lam_f * r_in, freq = Co / lambda cycles per step.
SCALES = ['cells', 'domain']
SCALE_SIZES = (('sigma', 'sigma_f', 'width'), ('beam_width', 'bw_f', 'width'),
               ('pk_len', 'pkl_f', 'width'), ('pk_lambda', 'pklam_f', 'wave'), ('freq', 'lam_f', 'wave'))
SCALE_MIN_WIDTH = 2.0                  # cells: the amber warning under this width


# ---- JavaScript arithmetic, reproduced exactly --------------------------------------------
def js_round(x):
    """Math.round: halves go toward +infinity (Python's round() is banker's rounding)."""
    return math.floor(x + 0.5)


def js_sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def sym_round(c, d):
    """Round an offset from the centre SYMMETRICALLY, so +d and -d land on mirrored cells."""
    return c + (0 if d == 0 else js_sign(d) * js_round(abs(d)))


def snap_q(v):
    """Quantise to 1e-9 so floating-point noise cannot break a rounding tie differently for
    mirror partners (the web app's snapQ)."""
    return js_round(v * 1e9) / 1e9


def js_fmod(a, m):
    return math.fmod(a, m)


@dataclass
class Params:
    """Mirrors the web app's S object (angles in RADIANS, as S.rot / S.rtheta are)."""
    N: int = 161
    Co: float = 0.5
    damp: float = 0.0
    shape: str = 'square'
    bc: str = 'dirichlet'
    medium: str = 'uniform'
    src: str = 'pulse'
    sigma: float = 3.0
    # 09/10/2026 (the user's decision): the default drive is a 16-cell wavelength at the default Co 0.5
    # (lambda = Co / freq cells; freq is in cycles per step). A plain default VALUE, not coupled to Co: at
    # another Co the same freq is another wavelength. The web app's default is 0.12 (a 4.2-cell wavelength,
    # grid-scale: drive_warnings()); every comparison with the web app sets freq=0.12 explicitly.
    freq: float = 0.03125
    stencil: int = 5
    rtheta: float = PI / 3
    rot: float = 0.0
    rad: float = 1.0
    sx: float = 0.0
    sy: float = 0.0
    vtx_drive: bool = False
    lattice: str = 'sq'
    wave: str = 'sine'
    # 05/10/2026 (NotesCLAUDE/vertex_choppiness_MIRROR.md): the port's default is 'wall' (3 sigma from
    # BOTH walls, along the corner's bisector); the web app's is 'radial', which leaves the vertex pulses
    # 0.7-1.7 sigma from the walls, cuts 1.2-27% of them off at t = 0 and fills 4.5-49% of the energy
    # with grid-scale chop. Comparisons with the web app set vtx_inset='radial' explicitly.
    vtx_inset: str = 'wall'
    mm: int = 1
    mn: int = 1
    # directional sources (01/10/2026, no web-app counterpart; NotesCLAUDE/beam_MIRROR.md)
    beam_angle: float = 0.0           # DEGREES from +x (counter-clockwise), the energy direction
    beam_width: float = 8.0           # w (cells): amplitude exp(-(s/w)^2) across the beam (the waist)
    beam_onesided: bool = True        # cancel the backward lobe with a second, delayed, inverted aperture
    beam_group: bool = True           # aim the energy (group velocity); False: the phase fronts
    pk_len: float = 10.0              # packet envelope length L (cells): exp(-(zeta/L)^2)
    pk_lambda: float = 16.0           # packet carrier wavelength (cells)
    # scale mode (02/10/2026, NotesCLAUDE/scale_MIRROR.md): 'cells' (default) or 'domain'. In 'domain'
    # the fractions below (of r_in) are the truth and the cell fields above are DERIVED from them by
    # Membrane.apply_scale() at every reseed / Co change. None = not set yet: taken from the cell
    # values at the first apply (so '--set scale=domain' keeps the picture of the starting N).
    scale: str = 'cells'
    sigma_f: float = None             # pulse sigma / r_in
    lam_f: float = None               # drive wavelength / r_in (freq = Co / (lam_f r_in))
    bw_f: float = None                # beam width / r_in
    pkl_f: float = None               # packet length / r_in
    pklam_f: float = None             # packet wavelength / r_in
    # walls (04/10/2026, NotesCLAUDE/walls_MIRROR.md): the domain mask's edge. 'pinned' = the web
    # app (u = 0 outside the shape); 'rigid' = link removal at the shape's edge (and the triangular
    # array's clamped ring). On the full square array (sq_full()) there is no mask edge: bc rules.
    edge: str = 'pinned'
    # driven vertices (05/10/2026, NotesCLAUDE/vertex_choppiness_MIRROR.md section 7): False (the port's
    # default) = with vtx_drive on, the vertices are point drives that start from rest, like 'cont' (no
    # seeded pulses). True = the web app: each driven vertex ALSO gets the gaussian bump, whose centre
    # cell the hard drive overwrites with drive(0) = 0 at step 1 -- a one-cell dent of depth 1 that
    # radiates grid-scale waves. No effect unless src == 'vtx' and vtx_drive.
    # With vtx_soft (below) the bumps are seeded the same way, but the soft drive adds to them: no dent.
    vtx_drive_bump: bool = False
    # driven vertices (09/10/2026, the user's decision "soft as default, leave a hard toggle"): True (the
    # port's default) = a SOFT source: the drive is ADDED to the new level at the vertex cells (the beam
    # aperture's kernel path), so they are transparent to returning waves; each harmonic is scaled by
    # soft_gain() (the lattice Green's function at the source cell) so the radiated wave matches the hard
    # source's. False = HARD (the web app): the cells are set to the drive, u = D(n), and reflect like pinned
    # points. No effect unless src == 'vtx' and vtx_drive. Comparisons with the web app set vtx_soft=False.
    vtx_soft: bool = True
    # absorbing boundaries (09/10/2026, NotesCLAUDE/absorbing_MIRROR.md). bc 'cpml': the frame's level, 'std' (L 10
    # cells, p 3, alpha 0) or 'high' (L 20, alpha 0.05). The sponge (bc 'sponge' on the full array, edge 'open' on a
    # masked shape): sponge_L = its width in PHYSICAL cells (None = auto: 40, or 20 when the array's half-size is
    # under 160 cells), sponge_dmax = the damping per step at its outer end (None = auto: 0.2 at L 40, 0.35 at L 20,
    # linear between), sponge_Lf = scale mode 'domain': the width as a fraction of the array's half-size base_half()
    # (None = taken from the cells value at the first use, like sigma_f), never under SPONGE_L_MIN cells.
    cpml_level: str = 'std'
    sponge_L: float = None
    sponge_dmax: float = None
    sponge_Lf: float = None


# the web app's own default drive frequency (cycles per step; the port's Params.freq default is 0.03125 since
# 09/10/2026): every comparison with the web app sets it explicitly (crosscheck_js / _ledger / _alignment, the
# scale-test references, the parity tests against older cores)
WEBAPP_FREQ = 0.12
# 'open' (09/10/2026, port-only): a masked shape's edge absorbs -- a distance-ramp sponge OUTSIDE the drawn edge
EDGES = ['pinned', 'rigid', 'open']
CPML_LEVEL_NAMES = ['std', 'high']
VTX_DRIVES = ['soft', 'hard']          # Params.vtx_soft True / False, for the UI (09/10/2026)
VTX_INSETS = ['radial', 'wall']        # the 'vertex inset' button's cycle (the web app's order)


# ---- soft point sources: the lattice Green's function at the source cell (09/10/2026) ----------------
# A SOFT source adds s(n) at a cell: u+ = 2u - u- + CC L u + s delta. At frequency omega (per step, time
# dependence e^{-i omega n}) the steady field is U = (S / CC) g00(E) at the cell, E = 4 sin^2(omega/2) / CC,
#     g00(E) = int_BZ dK / (2 pi)^2  1 / (Lambda(K) - E - i0)        (the outgoing / causal branch)
# and Lambda the lattice symbol (beam_MIRROR.symbol; index wavevectors on [-pi, pi]^2, one site per cell).
# A HARD source of amplitude A imposes U = A at the cell; on the infinite lattice that is exactly a soft
# source of strength A CC / g00, so a soft source of amplitude A * CC / |g00| radiates the hard source's
# wave: the same amplitude everywhere, a constant phase arg(g00) - omega behind it (the hard drive writes
# the next level). Measured (dec_soft_ratio_MIRROR.py, lambda 16, Co 0.5, 3 lambda away, CUDA f64): amplitude
# soft / hard 1.0001 (sq5, sq9), 1.0003 (tri); lag 19.3 / 17.5 / 17.5 degrees = arg g00 - omega. The inner BZ integral is
# closed-form: Lambda - E = A(k) - B(k) cos k' with
#     sq5: A = 4 - 2 cos k - E,                    B = 2
#     sq9: A = 10/3 - (4/3) cos k - E,             B = 4/3 + (2/3) cos k
#     tri: A = (2/3)(6 - 2 cos k) - E,             B = (8/3) |cos(k/2)|   (2 cos k' + 2 cos(k' - k) = B' cos(.))
# and int dk' / (A - i0 - B cos k') over a period = 2 pi / (sqrt(A - B - i0) sqrt(A + B - i0)) (principal
# square roots: each factor is -i sqrt(|x|) where it is negative), so g00 = (1/pi) int_0^pi dk / (...).
# Its 1/sqrt singularities sit where A = +-B: the integral is split there and each piece is taken with
# k = a + (b - a)(1 - cos t)/2 (which cancels them) and Gauss-Legendre in t. Checked against scipy's
# quad, a brute-force 2D sum with a small imaginary shift and the small-E limits (Im g00 -> 1/4 on sq5
# and sq9, sqrt(3)/8 on tri; Re g00 -> (1/4pi) ln(32/E) on sq5): ScratchCLAUDE/Round5CLAUDE/DecisionsCLAUDE/
# dec_green_probe_MIRROR.py. At Co 0.5 and lambda 16 cells: |g00| = 0.5013 (sq5), 0.5331 (sq9), 0.4591 (tri).
_G00_CACHE = {}
_G00_NODES = np.polynomial.legendre.leggauss(64)


def _g00_ab(kind, k, E):
    c = np.cos(k)
    if kind == 'sq5':
        return 4.0 - 2.0 * c - E, np.full_like(c, 2.0)
    if kind == 'sq9':
        return 10.0 / 3.0 - (4.0 / 3.0) * c - E, 4.0 / 3.0 + (2.0 / 3.0) * c
    return (2.0 / 3.0) * (6.0 - 2.0 * c) - E, (8.0 / 3.0) * np.abs(np.cos(k / 2))


def lattice_g00(kind, E):
    """The lattice Green's function at the source cell, g00(E) (complex; see above), kind 'sq5' | 'sq9' |
    'tri'. Cached. Above the band (E > max Lambda) it is real: the frequency does not propagate."""
    key = (kind, float(E))
    g = _G00_CACHE.get(key)
    if g is not None:
        return g
    ks = np.linspace(0.0, PI, 2049)
    A, B = _g00_ab(kind, ks, E)
    pts = {0.0, PI}
    for sgn in (-1.0, 1.0):
        gk = A + sgn * B
        pts.update(float(v) for v in ks[gk == 0])
        for q in np.nonzero(gk[:-1] * gk[1:] < 0)[0]:
            lo, hi = float(ks[q]), float(ks[q + 1])
            flo = float(A[q] + sgn * B[q])
            for _ in range(64):
                mid = 0.5 * (lo + hi)
                a_, b_ = _g00_ab(kind, np.array([mid]), E)
                fm = float(a_[0] + sgn * b_[0])
                if (fm < 0) == (flo < 0):
                    lo, flo = mid, fm
                else:
                    hi = mid
            pts.add(0.5 * (lo + hi))
    pts = sorted(pts)
    x, w = _G00_NODES
    t = (x + 1.0) * (PI / 2)
    wt = w * (PI / 2)
    tot = 0j
    for a, b in zip(pts[:-1], pts[1:]):
        if b - a < 1e-13:
            continue
        k = a + (b - a) * (1.0 - np.cos(t)) / 2
        jac = (b - a) / 2 * np.sin(t)
        A, B = _g00_ab(kind, k, E)
        s1 = np.where(A - B > 0, np.sqrt(np.abs(A - B)) + 0j, -1j * np.sqrt(np.abs(A - B)))
        s2 = np.where(A + B > 0, np.sqrt(np.abs(A + B)) + 0j, -1j * np.sqrt(np.abs(A + B)))
        den = s1 * s2
        ok = den != 0
        tot += complex(np.sum((wt * jac)[ok] / den[ok]))
    g = tot / PI
    if len(_G00_CACHE) > 4096:
        _G00_CACHE.clear()
    _G00_CACHE[key] = g
    return g


def soft_gain(kind, co2, omega):
    """The amplitude factor of a SOFT point source (added each step) whose radiated wave equals that of a
    HARD point source (imposed) of amplitude 1 at angular frequency omega (rad per step), on lattice kind
    with Co^2 = co2 at the cell: co2 / |g00(4 sin^2(omega/2) / co2)| (see lattice_g00). 0 for omega <= 0.
    At Co 0.5 and lambda 16 cells: 0.4987 (sq5), 0.4690 (sq9), 0.5446 (tri)."""
    if co2 <= 0 or omega <= 0:
        return 0.0
    E = 4.0 * math.sin(omega / 2) ** 2 / co2
    g = abs(lattice_g00(kind, E))
    return co2 / g if g > 0 else 0.0
CLEAR_OK_SIGMA = 2.9                   # the web app's source note: clearance >= 2.9 sigma reads green
CLIP_WARN = 0.01                       # ... and a clip fraction above 1% reads amber, with the hint

MASK_DT = ti.u8
FP64 = True
# the recorder's accumulators: f64 wherever the backend has it. Vulkan f64 accumulators
# crashed the process (segfault, measured 29/09/2026), so f32 there: the two-level sum below
# keeps its relative error at ~2e-7 even at N = 8193 (a naive serial f32 sum was off by 18%).
ACC_DT = ti.f64
# The 9-point weights (2/3)*4 + (1/6)*4 - 10/3 cancel exactly in f64 but NOT in f32: the
# residue acts as a small mass term and shifts low modes (measured 29/09: the (3,2) mode off
# by 5e-4 of its amplitude after 300 steps, on CUDA and Vulkan alike). In f32 the stencil is
# written in the form that cancels by construction, (2/3)(e - 4u) + (1/6)(dg - 4u); in f64 it
# keeps the web app's exact operation order, so the two stay bit-comparable.
SQ9_CONSISTENT = False
# the walls path's per-cell link word (04/10/2026; layout at W_B4: bits 0-15). 32-bit on Vulkan
# (it misreads 8-bit storage, see MASK_DT; 16-bit is untested there), 16-bit elsewhere: with i32
# the walls kernel cost CUDA f32 +23-27% per step at N = 8193 (the mask's 1 byte became 4 of ~17
# per cell; walls_bench_MIRROR.py run 1); init_taichi sets it.
WORD_DT = ti.u16


def init_taichi(arch='cuda', fp64=True, fast_math=False, quiet=False):
    """fast_math is OFF by default: it lets the compiler fuse and reorder float operations,
    which is fine for speed but breaks agreement with the web app at the last bits.
    (CUDA still fuses a*b + c into one FMA rounding regardless; see README, cross-check.)"""
    global MASK_DT, SQ9_CONSISTENT, FP64, ACC_DT, WORD_DT
    SQ9_CONSISTENT = not fp64
    FP64 = fp64
    archs = {'cuda': ti.cuda, 'vulkan': ti.vulkan, 'cpu': ti.cpu, 'gpu': ti.gpu}
    ti.init(arch=archs[arch], default_fp=ti.f64 if fp64 else ti.f32, fast_math=fast_math,
            log_level=ti.ERROR if quiet else ti.WARN)
    # Vulkan emulates 8-bit storage inside 32-bit words, and with N*N not a multiple of 4
    # the LAST cell of a u8 mask read as 0 (measured 29/09: N=161 periodic, cell (160,160)
    # clamped from step 1 -- invisible until a wave reached that corner at step 291).
    # A 32-bit mask on Vulkan; u8 elsewhere (a quarter of the memory traffic).
    vk = ti.lang.impl.current_cfg().arch == ti.vulkan
    MASK_DT = ti.i32 if vk else ti.u8
    ACC_DT = ti.f32 if vk else ti.f64
    WORD_DT = ti.i32 if vk else ti.u16


def mask_np_dtype():
    return np.uint8 if MASK_DT == ti.u8 else np.int32


# ---- the time step: ONE fused kernel for K steps --------------------------------------------
# Taichi's launch overhead on this machine is ~170-265 us per kernel call (measured 29/09) --
# more than a whole step costs the GPU below N ~ 2000. So K steps are unrolled into ONE launch
# (ti.static): each step's loops are still separate parallel passes run in order, with the
# web app's per-cell arithmetic. The three time levels live in one array U[3, N, N] and rotate
# by slot: step q reads slot (s+q)%3 (previous) and (s+q+1)%3 (current), writes (s+q+2)%3.
# Storage is ti.ndarray, not ti.field: a field is baked into the compiled kernel, so every new
# grid (every reset at a new N, every Membrane) recompiled everything; an ndarray is not.
LAT_SQ, LAT_TRI = 0, 1
BC_DIR, BC_MUR, BC_NEU, BC_PER = 0, 1, 2, 3
BC_MUR2 = 4                             # 09/10/2026: Mur 2nd order (Engquist-Majda A2), diag-Mur1 corners
BC_MUR2S = 5                            # 10/10/2026: Mur2's start-up launch (its first step Mur1 on the sides; step())
# cpml and sponge clamp the ring (the frame's outer end is Dirichlet); their layer is the absorbing kernel's (absk)
BC_CODE = {'dirichlet': BC_DIR, 'mur': BC_MUR, 'neumann': BC_NEU, 'periodic': BC_PER, 'mur2': BC_MUR2,
           'cpml': BC_DIR, 'sponge': BC_DIR}
# the absorbing kernel variants (09/10/2026; k_run_abs / k_run_abs_rec, template absk): none (k_run), the sponge
# (centred damping where the mask code is >= 2), the CPML frame (square lattice, the full array)
ABS_NONE, ABS_SPONGE, ABS_CPML = 0, 1, 2
# AUX (one float ndarray per absorbing configuration; the launch costs ~30 us per ndarray argument, so everything
# the layer needs is packed in it): [0] the CPML frame width Lc (cells); [AUX_INV + m] = 1 / (1 + d_m / 2) and
# [AUX_C2 + m] = (1 - d_m / 2) / (1 + d_m / 2) for the sponge's mask codes m = 2..255; from AUX_TAB the CPML's 1D
# coefficient tables [6][2 Lc] (b, a at the cell, the inner-index face, the outer-index face), then its memory:
# the x strips [3][2 Lc][N] (psi at the cell's -x face, psi at its +x face, zeta at the cell) and the y strips.
AUX_INV = 16
AUX_C2 = AUX_INV + 256
AUX_TAB = AUX_C2 + 256
SRC_NONE, SRC_CELLS, SRC_SLIT, SRC_BEAM = 0, 1, 2, 3
# steps per launch; a run of n steps is split greedily. Measured 29/09 (CUDA, N = 161, Mur):
#   K =  1: compile 0.25 s,  480 us/step     K = 16: compile  4.8 s,  92 us/step
#   K =  8: compile 1.8 s,   115 us/step     K = 32: compile 14.7 s,  73 us/step
# The residue is ~20 us per parallel pass inside the kernel (Windows' WDDM launch latency);
# 8 keeps the first run of a new lattice/boundary/source combination to ~2 s.
CHUNKS = (8, 1)
KMAX = CHUNKS[0]

# ---- the recorder (the web app's record(), run after every step) -----------------------------
# kin = sum (u_new - u_prev)^2, pot = sum of forward differences squared, amax = max |u_new|,
# all over the masked interior, plus three single-cell probes. A per-cell atomic into ONE
# address does not scale: measured 29/09/2026 (CUDA f64, one reduction pass): 3.8 ms at N=1025,
# 62 ms at N=4097, 247 ms at N=8193 (Taichi applies no thread-local reduction to an ndarray
# target; a 0-D field gets one, but then costs a flat ~1.4 ms per launch from ~290k per-thread
# atomics). So TWO levels: thread (g, j) sums a run of ~n/ng rows in column j serially
# (neighbouring threads read neighbouring memory), adds its partial into one of NSLOT slots,
# and one thread folds the slots. Measured, one pass: 0.34 ms at N=161..2049 (launch-bound),
# 0.58 ms at 4097, 2.05 ms at 8193 -- the cost of one copy pass -- relative error 3e-16 (f64).
# Inside k_run, per step (30/09/2026, CUDA f64, ScratchCLAUDE/DevCLAUDE/rec_tune_MIRROR.py):
# +0.05 ms at N=1025, +0.47 ms at 4097, +1.86 ms at 8193 (~50% of a step: it reads two f64
# levels and the mask, ~17 of the step's ~41 bytes per cell) with 65536 first-level threads;
# 131072-1048576 threads and 32-256 slots were no faster at 4097/8193 and slower at 1025.
NSLOT = 64
REC_THREADS = 65536                   # ~ng * n threads in the first level
HCAP = 1024                           # rows of the device-side history ring (6 values each)
# Launch cost (30/09/2026, Taichi 1.7.4, CUDA, ScratchCLAUDE/DevCLAUDE/argcost_MIRROR.py): ~30 us
# per ndarray argument and ~10 us per scalar argument, per kernel call. The recorder's buffers
# as three extra ndarrays + two scalars made even the recorder-OFF step ~15 us slower at N <= 1025
# (0.091 -> 0.106 ms/step). So the recorder has its own kernel (k_run_rec) sharing the step body
# (_steps) with the plain k_run, whose argument list is the original one; the reduction slots
# live in the history buffer's last NSLOT rows, and the group count is computed in the kernel.


def rec_groups(n):
    """First-level groups of the recorder's reduction (the kernels compute the same in-kernel)."""
    return max(1, min(n - 2, REC_THREADS // n))


# ---- walls: cell classes and link words (round 4, 04/10/2026, NotesCLAUDE/walls_MIRROR.md) ----
# A cell's CLASS as its neighbours see it (the low 2 bits of the class code C; bit 2 = the cell's
# own mask M, 1 = free):
#   F  free: a stencil cell, or an array-ring cell that the bc governs (sq: Dirichlet/Mur/Neumann ring)
#   P  pinned solid (holds 0; the link is kept and reads the 0)
#   R  rigid solid (the link is dropped: link removal, rule R0 of the research note)
#   X  domain exterior, incl. the triangular array's clamped ring: P with edge = 'pinned', R with
#      edge = 'rigid'.
# Corner blocking: a link between two FREE cells whose two common neighbours are both non-free is
# blocked -- pinned-blocked (reads 0) if either blocker acts pinned, dropped if both act rigid --
# EXCEPT, with edge = 'pinned', when both blockers are X (the web app has no blocking; its own
# domain staircases keep every link). sq5 has no blockable link; sq9 only its diagonals.
CL_F, CL_P, CL_R, CL_X = 0, 1, 2, 3
CL_FREE = 4
# the stencils' links, in the kernels' (the JS's) summation order
SQ_LINKS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))
TRI_LINKS = ((-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (-1, 1))
# The link WORD of a cell (WORD_DT; built by k_walls_full / k_walls_patch from C):
#   bit 0        free (= M)
#   bit 1 + k    link k is DROPPED: the kernel reads u_i itself, so w (u_i - u_i) = 0 (a rigid
#                neighbour, a rigid exterior, a rigid-blocked link)
#   bit 9 + z    link k is PINNED-BLOCKED: the kernel reads 0 (z = k - 4 for the sq9 diagonals,
#                z = k on tri; sq edge links are never blocked)
#   bit 15       tri only: one of links 0-3 ((+-1,0), (0,+-1)) is blocked (main_region's exact path)
# A free cell whose word is exactly 1 is PLAIN: it computes today's update to the bit. A link to a
# pinned cell or a pinned exterior stays plain (the neighbour holds 0).
W_B4 = 15
W_LINKBITS = 0x7FFE                    # bits 1-14: any dropped or pinned-blocked link


def lat_links(lat, nine):
    return TRI_LINKS if lat == LAT_TRI else (SQ_LINKS if nine else SQ_LINKS[:4])


def w_sbit(k):
    """The word's DROPPED bit of link k."""
    return 1 + k


def w_zbit(lat, nine, k):
    """The word's PINNED-BLOCKED bit of link k, or -1 if link k is never blocked."""
    if lat == LAT_TRI:
        return 9 + k
    return 9 + (k - 4) if (nine and k >= 4) else -1


def link_commons(lat, k):
    """The two common neighbours of link k (offsets from the cell): sq9 diagonal (a, b) -> (a, 0)
    and (0, b); tri -> the two cells that form a triangle with the link (walls_lib's rule)."""
    if lat == LAT_TRI:
        d = TRI_LINKS[k]
        return tuple(c for c in TRI_LINKS if (d[0] - c[0], d[1] - c[1]) in TRI_LINKS)
    a, b = SQ_LINKS[k]
    return ((a, 0), (0, b))


@ti.func
def _wc(C: ti.template(), i, j, n, wrap: ti.template()):
    """The class (low 2 bits of C) of cell (i, j); on a torus the indices wrap."""
    ii, jj = i, j
    if ti.static(wrap):
        ii = (i + n) % n
        jj = (j + n) % n
    return ti.cast(C[ii, jj], ti.i32) & 3


@ti.func
def _nring_common(qi, qj, n, cq, erig: ti.template()):
    """The square Neumann ring's ghost rule (review fix, 06/10/2026, Claude Opus 5.5): how a sq9 diagonal
    link INTO the ring is blocked by its common neighbour (qi, qj) when that neighbour is a ring cell that
    is not free -- 0 no, 1 pinned (a solid P on the ring row), 2 dropped (R, or X with a rigid edge),
    3 pinned for X with a PINNED edge (09/10/2026, the user's decision: the exterior of a shape clipped by
    the array; _word drops it for a link into an array corner, which is the cell's own ghost)."""
    r = 0
    if qi == 0 or qj == 0 or qi == n - 1 or qj == n - 1:
        if cq == CL_P:
            r = 1
        elif cq == CL_R:
            r = 2
        elif cq == CL_X:
            if ti.static(erig):
                r = 2
            else:
                r = 3
    return r


@ti.func
def _word(C: ti.template(), i, j, n, lat: ti.template(), nine: ti.template(), erig: ti.template(),
          wrap: ti.template(), neu: ti.template()):
    """The link word of cell (i, j) from the classes of its neighbours and their common neighbours
    (the rule of walls_lib_MIRROR.build_L, block = 'both', no transfer; plus the X-X exception).
    neu (review fix, 06/10/2026): the square lattice's Neumann ring. Its cells are ghosts: cell (1, t) reads
    the ghost of (1, t+1) through its diagonal link to (0, t+1), and (1, t+1) reads the ghost of (1, t)
    through (0, t). When (0, t) is not a ghost (a solid painted ON the ring row, or the exterior of a clipped
    shape with a rigid edge), the second read gets 0 or nothing while the first still read u(1, t+1): the
    operator was ASYMMETRIC and the run grew without bound (spectral radius 1.0007 per step; max|u| 1e12
    after 200,000 steps, ScratchCLAUDE/Round4CLAUDE/ReviewCLAUDE/rv_neu_growth_MIRROR.py). So a sq9
    diagonal link into the Neumann ring whose common neighbour on the ring is not free is blocked like
    that neighbour: pinned-blocked for P, dropped for R (and X with a rigid edge)."""
    w = (ti.cast(C[i, j], ti.i32) >> 2) & 1
    inner = 1
    if ti.static(not wrap):
        if i <= 0 or j <= 0 or i >= n - 1 or j >= n - 1:
            inner = 0                       # a ring cell: the bc / the clamp updates it, not the stencil
    if w == 1 and inner == 1:
        for k in ti.static(range(len(lat_links(lat, nine)))):
            d = ti.static(lat_links(lat, nine)[k])
            cn = _wc(C, i + d[0], j + d[1], n, wrap)
            if cn == CL_R:
                w |= ti.static(1 << w_sbit(k))
            if ti.static(erig):
                if cn == CL_X:
                    w |= ti.static(1 << w_sbit(k))
            if ti.static(w_zbit(lat, nine, k) >= 0):
                if cn == CL_F:
                    cm = ti.static(link_commons(lat, k))
                    c1 = _wc(C, i + cm[0][0], j + cm[0][1], n, wrap)
                    c2 = _wc(C, i + cm[1][0], j + cm[1][1], n, wrap)
                    blk = 0
                    if c1 != CL_F and c2 != CL_F:
                        blk = 1
                        if ti.static(not erig):
                            if c1 == CL_X and c2 == CL_X:
                                blk = 0             # web-app parity: the domain's own staircase never blocks
                    rpin = -1                       # the Neumann ring's ghost rule (neu): -1 not used
                    if ti.static(neu and nine and k >= 4):
                        if blk == 0:
                            ti_, tj_ = i + d[0], j + d[1]
                            if ti_ == 0 or tj_ == 0 or ti_ == n - 1 or tj_ == n - 1:
                                r1 = _nring_common(i + cm[0][0], j + cm[0][1], n, c1, erig)
                                r2 = _nring_common(i + cm[1][0], j + cm[1][1], n, c2, erig)
                                # 09/10/2026 (the user's decision): the exterior X of a shape clipped by the
                                # array, with a PINNED edge, blocks like a pinned ring solid (r 3 -> 1), the
                                # web app's own asymmetric ring (walls_MIRROR.md 14.2). Not for a link into an
                                # array CORNER: from (1, 1) that is the cell's own ghost (a self-read, already
                                # symmetric), so the configurations it would change were never asymmetric.
                                x3 = 1
                                if ti_ == 0 or ti_ == n - 1:
                                    if tj_ == 0 or tj_ == n - 1:
                                        x3 = 0
                                if r1 == 3:
                                    r1 = x3
                                if r2 == 3:
                                    r2 = x3
                                if r1 != 0 or r2 != 0:
                                    blk = 1
                                    rpin = 0
                                    if r1 == 1 or r2 == 1:
                                        rpin = 1
                    if blk == 1:
                        pin = 0
                        if c1 == CL_P or c2 == CL_P:
                            pin = 1
                        if ti.static(not erig):
                            if c1 == CL_X or c2 == CL_X:
                                pin = 1
                        if rpin >= 0:
                            pin = rpin
                        if pin == 1:
                            w |= ti.static(1 << w_zbit(lat, nine, k))
                        else:
                            w |= ti.static(1 << w_sbit(k))
                        if ti.static(lat == LAT_TRI and k < 4):
                            w |= ti.static(1 << W_B4)
    return w


@ti.func
def _wcount(CNT: ti.template(), w, sgn):
    if (w & 1) == 1 and (w & W_LINKBITS) != 0:
        ti.atomic_add(CNT[0], sgn)
    if ((w >> W_B4) & 1) == 1:
        ti.atomic_add(CNT[1], sgn)


@ti.kernel
def k_walls_full(C: ti.types.ndarray(ndim=2), W: ti.types.ndarray(ndim=2), CNT: ti.types.ndarray(ndim=1),
                 lat: ti.template(), nine: ti.template(), erig: ti.template(), wrap: ti.template(),
                 neu: ti.template()):
    # every cell's word from the class codes C (one parallel pass); CNT[0] = free cells with a
    # dropped or pinned-blocked link (0: the solver may take the web app's path), CNT[1] = cells
    # with a blocked tri link 0-3
    n = C.shape[0]
    for t in range(1):
        CNT[0] = 0
        CNT[1] = 0
    for i, j in ti.ndrange(n, n):
        w = _word(C, i, j, n, lat, nine, erig, wrap, neu)
        W[i, j] = ti.cast(w, WORD_DT)
        _wcount(CNT, w, 1)


@ti.kernel
def k_walls_patch(U: ti.types.ndarray(ndim=3), M: ti.types.ndarray(ndim=2), C: ti.types.ndarray(ndim=2),
                  W: ti.types.ndarray(ndim=2), Q: ti.types.ndarray(ndim=2),
                  CNT: ti.types.ndarray(ndim=1), qi0: ti.i32, qj0: ti.i32, i0: ti.i32, j0: ti.i32, ni: ti.i32,
                  nj: ti.i32, lat: ti.template(), nine: ti.template(), erig: ti.template(), wrap: ti.template(),
                  neu: ti.template()):
    # a painted box, ONE launch and ONE upload (k_patch_mask's job included): the class patch Q
    # goes in at (qi0, qj0), with the mask M = its free bit and all three time levels zeroed where
    # that is 0; the words of the box [i0, i0+ni) x [j0, j0+nj) (the patch + the stencil's reach; it
    # wraps on a torus, the host caps it at n per axis) are rebuilt, the counters kept up to date
    n = C.shape[0]
    for a, b in ti.ndrange(ni, nj):
        i = (i0 + a + n) % n
        j = (j0 + b + n) % n
        _wcount(CNT, ti.cast(W[i, j], ti.i32), -1)
    for a, b in ti.ndrange(Q.shape[0], Q.shape[1]):
        q = Q[a, b]
        C[qi0 + a, qj0 + b] = q
        f = (ti.cast(q, ti.i32) >> 2) & 1
        M[qi0 + a, qj0 + b] = ti.cast(f, MASK_DT)
        if f == 0:
            U[0, qi0 + a, qj0 + b] = 0.0
            U[1, qi0 + a, qj0 + b] = 0.0
            U[2, qi0 + a, qj0 + b] = 0.0
    for a, b in ti.ndrange(ni, nj):
        i = (i0 + a + n) % n
        j = (j0 + b + n) % n
        w = _word(C, i, j, n, lat, nine, erig, wrap, neu)
        W[i, j] = ti.cast(w, WORD_DT)
        _wcount(CNT, w, 1)


@ti.func
def _wv(U: ti.template(), b, ii, jj, u, w, sb: ti.template(), zb: ti.template()):
    """The value link k contributes: the neighbour (active), 0 (pinned-blocked) or u_i (dropped)."""
    v = U[b, ii, jj]
    if ti.static(zb >= 0):
        if ((w >> zb) & 1) == 1:
            v = 0.0
    if ((w >> sb) & 1) == 1:
        v = u
    return v


@ti.func
def lap_sq_w(U: ti.template(), b, i, j, im, ip, jm, jp, nine: ti.template(), w):
    # lap_sq with each neighbour replaced by _wv. sq5: the same arithmetic as lap_sq. sq9: ALWAYS
    # the self-consistent form (2/3)(e - 4u) + (1/6)(dg - 4u), also in f64 (04/10/2026): the web
    # app's order (2/3)e + (1/6)dg - (10/3)u leaves -4.4e-16 u on a constant field in f64, a tiny
    # systematic mass term -- harmless with pinned walls, but a closed RIGID cavity has a zero mode
    # (its mean), which that term slowly drives: its DC drifted 1.1e-9 in 5000 steps (vs ~1e-13 for
    # the unbiased rounding of sq5 and tri). Only the walls path uses this; the web app's path keeps
    # its order.
    u = U[b, i, j]
    lap = 0.0
    if ti.static(nine):
        e = (_wv(U, b, im, j, u, w, 1, -1) + _wv(U, b, ip, j, u, w, 2, -1) + _wv(U, b, i, jm, u, w, 3, -1)
             + _wv(U, b, i, jp, u, w, 4, -1))
        dg = (_wv(U, b, im, jm, u, w, 5, 9) + _wv(U, b, ip, jm, u, w, 6, 10) + _wv(U, b, im, jp, u, w, 7, 11)
              + _wv(U, b, ip, jp, u, w, 8, 12))
        lap = (2.0 / 3.0) * (e - 4 * u) + (1.0 / 6.0) * (dg - 4 * u)
    else:
        lap = (_wv(U, b, im, j, u, w, 1, -1) + _wv(U, b, ip, j, u, w, 2, -1) + _wv(U, b, i, jm, u, w, 3, -1)
               + _wv(U, b, i, jp, u, w, 4, -1) - 4 * u)
    return lap


@ti.func
def sum_tri_wi(U: ti.template(), b, i, j, im, ip, jm, jp, u, w):
    # the triangular neighbour sum, JS order, each neighbour through _wv, with the neighbour indices
    # given (im = i - 1, ip = i + 1, ... -- wrapped mod n on the rhombic torus, 05/10/2026)
    return (_wv(U, b, im, j, u, w, 1, 9) + _wv(U, b, ip, j, u, w, 2, 10) + _wv(U, b, i, jm, u, w, 3, 11)
            + _wv(U, b, i, jp, u, w, 4, 12) + _wv(U, b, ip, jm, u, w, 5, 13)
            + _wv(U, b, im, jp, u, w, 6, 14))


@ti.func
def sum_tri_w(U: ti.template(), b, i, j, u, w):
    # the triangular neighbour sum, JS order, each neighbour through _wv
    return sum_tri_wi(U, b, i, j, i - 1, i + 1, j - 1, j + 1, u, w)


@ti.func
def lap_sq(U: ti.template(), b, i, j, im, ip, jm, jp, nine: ti.template()):
    # sum order = the JS order: (i-1,j) (i+1,j) (i,j-1) (i,j+1); diagonals (-,-) (+,-) (-,+) (+,+)
    u = U[b, i, j]
    lap = 0.0
    if ti.static(nine):
        e = U[b, im, j] + U[b, ip, j] + U[b, i, jm] + U[b, i, jp]
        dg = U[b, im, jm] + U[b, ip, jm] + U[b, im, jp] + U[b, ip, jp]
        if ti.static(SQ9_CONSISTENT):
            lap = (2.0 / 3.0) * (e - 4 * u) + (1.0 / 6.0) * (dg - 4 * u)
        else:
            lap = (2.0 / 3.0) * e + (1.0 / 6.0) * dg - (10.0 / 3.0) * u
    else:
        lap = U[b, im, j] + U[b, ip, j] + U[b, i, jm] + U[b, i, jp] - 4 * u
    return lap


@ti.func
def rec_fold(U: ti.template(), M: ti.template(), PS: ti.template(), p0: ti.template(), b: ti.template(),
             c: ti.template(), walls: ti.template(), lat: ti.template(), wrap: ti.template(), LV: ti.template(),
             absx: ti.template()):
    """First level of the recorder's reduction (see NSLOT): thread (g, j) walks rows
    [1 + g*gsz, 1 + (g+1)*gsz) of column j. Slot s (row p0 + s of PS) collects the partials of
    ~ng*n/NSLOT threads. Template parameters: no copy statements at the kernel's top level (see
    _steps). walls (04/10/2026): M is the link word; a forward difference along a DROPPED link
    (rigid wall or edge) counts 0, along a pinned-blocked one -u (a link to a zero), so a rigid
    wall stores no 'potential' (the web app's formula, with the walls' links). Triangular lattice
    (05/10/2026, port-only): also the third link direction's forward difference gz = u(i+1, j-1) - u
    into slot column 3 -- the web app's E uses only gx, gy (the square lattice's functional), which is
    not the conserved quantity there; (1/3)(gx^2 + gy^2 + gz^2) is the triangular operator's own
    Dirichlet form (m.hist['Etri'], the ledger's 'tri-energy' fix). wrap (05/10/2026: the rhombic
    torus only): every cell is summed, with wrapped differences -- the web app's interior-only sum
    leaves the seam rows out, and their share of a wave crossing the seam read as a 2-3% 'drift'
    (ScratchCLAUDE/Round4CLAUDE/TriModesCLAUDE/torus_energy_probe_MIRROR.py). The square lattice's
    torus keeps the web app's interior sum (parity). absx (09/10/2026, absorbing layers): only the OPEN
    domain is summed -- a cell whose mask code LV is >= 2 (a sponge or CPML layer cell) is left out, so the
    energy, the peak and the ledger's rows describe the drawn domain (LV is the device mask M, which holds the
    codes; on the walls path M is the word array and LV the mask). absx = False compiles the code of before."""
    n = U.shape[1]
    lo = ti.static(0 if wrap else 1)
    nr = n - 2
    if ti.static(wrap):
        nr = n
    ng = ti.max(1, ti.min(nr, REC_THREADS // n))
    gsz = (nr + ng - 1) // ng
    for g, j in ti.ndrange(ng, (lo, lo + nr)):
        ka = ti.cast(0.0, ACC_DT)
        pa = ti.cast(0.0, ACC_DT)
        mx = ti.cast(0.0, ACC_DT)
        za = ti.cast(0.0, ACC_DT)
        i0 = lo + g * gsz
        i1 = ti.min(lo + nr, i0 + gsz)
        u = U[c, ti.min(i0, n - 1), j]
        jp = j + 1
        jm = j - 1
        if ti.static(wrap):
            jp = (j + 1) % n
            jm = (j - 1 + n) % n
        for i in range(i0, i1):
            ip = i + 1
            if ti.static(wrap):
                ip = (i + 1) % n
            un = U[c, ip, j]                # the next row's value: this row's gx, the next row's u
            if ti.static(absx):
                # the absorbing layers' variant: the same sums over the open domain only (LV < 2)
                inl = 1
                if ti.static(walls):
                    w = ti.cast(M[i, j], ti.i32)
                    if (w & 1) == 0:
                        inl = 0
                else:
                    if M[i, j] == 0:
                        inl = 0
                if ti.cast(LV[i, j], ti.i32) >= 2:
                    inl = 0
                if inl != 0:
                    dv = u - U[b, i, j]
                    gx = un - u
                    gy = U[c, i, jp] - u
                    if ti.static(walls):
                        w = ti.cast(M[i, j], ti.i32)
                        gx = _wv(U, c, ip, j, u, w, 2, ti.static(w_zbit(lat, False, 1))) - u
                        gy = _wv(U, c, i, jp, u, w, 4, ti.static(w_zbit(lat, False, 3))) - u
                    ka += ti.cast(dv * dv, ACC_DT)
                    pa += ti.cast(gx * gx + gy * gy, ACC_DT)
                    mx = ti.max(mx, ti.cast(ti.abs(u), ACC_DT))
                    if ti.static(lat == LAT_TRI):
                        gz = U[c, ip, jm] - u
                        if ti.static(walls):
                            w = ti.cast(M[i, j], ti.i32)
                            gz = _wv(U, c, ip, jm, u, w, 5, 13) - u
                        za += ti.cast(gz * gz, ACC_DT)
            elif ti.static(walls):
                w = ti.cast(M[i, j], ti.i32)
                if (w & 1) != 0:
                    dv = u - U[b, i, j]
                    gx = _wv(U, c, ip, j, u, w, 2, ti.static(w_zbit(lat, False, 1))) - u
                    gy = _wv(U, c, i, jp, u, w, 4, ti.static(w_zbit(lat, False, 3))) - u
                    ka += ti.cast(dv * dv, ACC_DT)
                    pa += ti.cast(gx * gx + gy * gy, ACC_DT)
                    mx = ti.max(mx, ti.cast(ti.abs(u), ACC_DT))
                    if ti.static(lat == LAT_TRI):
                        gz = _wv(U, c, ip, jm, u, w, 5, 13) - u
                        za += ti.cast(gz * gz, ACC_DT)
            else:
                if M[i, j] != 0:
                    dv = u - U[b, i, j]
                    gx = un - u
                    gy = U[c, i, jp] - u
                    ka += ti.cast(dv * dv, ACC_DT)
                    pa += ti.cast(gx * gx + gy * gy, ACC_DT)
                    mx = ti.max(mx, ti.cast(ti.abs(u), ACC_DT))
                    if ti.static(lat == LAT_TRI):
                        gz = U[c, ip, jm] - u
                        za += ti.cast(gz * gz, ACC_DT)
            u = un
        s = p0 + (g * 7 + j) % NSLOT
        ti.atomic_add(PS[s, 0], ka)
        ti.atomic_add(PS[s, 1], pa)
        ti.atomic_max(PS[s, 2], mx)
        if ti.static(lat == LAT_TRI):
            ti.atomic_add(PS[s, 3], za)


@ti.func
def rec_sum(PS: ti.template(), p0, H: ti.template(), row):
    """Second level: ONE thread folds the NSLOT slots (rows p0.. of PS) into H[row, 0:3] and
    re-zeroes them, so the slots are always zero between uses (the buffers are zero-filled at
    allocation)."""
    ka = ti.cast(0.0, ACC_DT)
    pa = ti.cast(0.0, ACC_DT)
    mx = ti.cast(0.0, ACC_DT)
    za = ti.cast(0.0, ACC_DT)
    for s in range(p0, p0 + NSLOT):
        ka += PS[s, 0]
        pa += PS[s, 1]
        mx = ti.max(mx, PS[s, 2])
        za += PS[s, 3]
        PS[s, 0] = 0.0
        PS[s, 1] = 0.0
        PS[s, 2] = 0.0
        PS[s, 3] = 0.0
    H[row, 0] = ka
    H[row, 1] = pa
    H[row, 2] = mx
    H[row, 6] = za                      # the triangular lattice's third direction (0 on the square one)


# ---- absorbing boundaries (09/10/2026, Claude Opus 5.5; NotesCLAUDE/absorbing_MIRROR.md) ---------------------
@ti.func
def _mur2_side(U: ti.template(), CC: ti.template(), a, b, c, r0i, r0j, r1i, r1j, ti_, tj_):
    """Mur 2nd order at the ring cell r0 with its interior partner r1 (the wall runs along (ti_, tj_)):
        u0' = -u1'' + k (u1' + u0'') + g (u0 + u1) + e (Dtt u0 + Dtt u1),   ' = level c, '' = level a
    k = (Co - 1)/(Co + 1), g = 2/(Co + 1), e = Co^2/(2 (Co + 1)), Co from CC at the ring cell."""
    cc = CC[r0i, r0j]
    co = ti.sqrt(cc)
    k = (co - 1) / (co + 1)
    g = 2 / (co + 1)
    e = cc / (2 * (co + 1))
    d0 = U[b, r0i - ti_, r0j - tj_] - 2 * U[b, r0i, r0j] + U[b, r0i + ti_, r0j + tj_]
    d1 = U[b, r1i - ti_, r1j - tj_] - 2 * U[b, r1i, r1j] + U[b, r1i + ti_, r1j + tj_]
    U[c, r0i, r0j] = (-U[a, r1i, r1j] + k * (U[c, r1i, r1j] + U[a, r0i, r0j]) + g * (U[b, r0i, r0j] + U[b, r1i, r1j])
                      + e * (d0 + d1))


@ti.func
def _mur1_side(U: ti.template(), CC: ti.template(), b, c, r0i, r0j, r1i, r1j):
    """Mur 1st order at the ring cell r0 with its partner r1, k from the ring cell's own Co: Mur2's START-UP step
    (10/10/2026). Mur2 is Mur1's residual A^n = u0' - u1 - k (u1' - u0) differenced in time,
    A^n - A^{n-1} = e (Dtt u0 + Dtt u1), so it keeps whatever A it starts from; a field seeded at rest with values
    on the ring row and its partner row has A != 0 there, which then stays forever as a static offset (a pulse
    7 cells from the ring left -0.099, a spike next to the ring -0.35 of its peak: ScratchCLAUDE/Round5CLAUDE/
    AbsorbCLAUDE/probe_mur2_dc_MIRROR.log). One Mur1 step sets A = 0; Mur2 runs on from there."""
    co = ti.sqrt(CC[r0i, r0j])
    k = (co - 1) / (co + 1)
    U[c, r0i, r0j] = U[b, r1i, r1j] + k * (U[c, r1i, r1j] - U[b, r0i, r0j])


@ti.func
def _mur2_corner(U: ti.template(), CC: ti.template(), b, c, ci, cj, si, sj):
    """Mur2's corner: Mur1 along the diagonal, u0' = u1 + kd (u1' - u0), kd = (Co - sqrt2)/(Co + sqrt2)."""
    co = ti.sqrt(CC[ci, cj])
    kd = (co - 1.4142135623730951) / (co + 1.4142135623730951)
    U[c, ci, cj] = U[b, ci + si, cj + sj] + kd * (U[c, ci + si, cj + sj] - U[b, ci, cj])


@ti.func
def _nb(U: ti.template(), b, ii, jj, u, w, sb: ti.template(), zb: ti.template(), walls: ti.template()):
    """A neighbour's value: the plain read, or through the link word (_wv) on the walls path."""
    v = U[b, ii, jj]
    if ti.static(walls):
        v = _wv(U, b, ii, jj, u, w, sb, zb)
    return v


@ti.func
def _fidx(i, n, Lc):
    """The CPML frame index of row / column i: 0..Lc-1 for depth 1..Lc on the low side, Lc..2Lc-1 for depth
    Lc..1 on the high side, -1 outside the frame (depth > Lc). Depth Lc is an open-domain cell whose outer face
    lies in the layer, so it carries the face's memory too."""
    s = -1
    if i <= Lc:
        s = i - 1
    elif i >= n - 1 - Lc:
        s = Lc + (i - (n - 1 - Lc))
    return s


@ti.func
def _cpml_lap(U: ti.template(), AUX: ti.template(), b, i, j, n, u, w, sx, sy, Lc, par, nine: ti.template(),
              walls: ti.template()):
    """The CPML Laplacian of a frame cell (convolutional PML, Roden & Gedney 2000 / Komatitsch & Martin 2007; the
    research prototype abc2d_MIRROR 'cpml'): along each stretched axis the face differences g get a recursive memory
    psi = b psi + a g, the face fluxes (g + psi) are differenced, and the cell memory zeta = b zeta + a dq is added.
    SINGLE PASS: each frame cell keeps its OWN copies of its two face memories per axis and updates them in place
    from the same inputs its neighbour uses for the shared face, so there is no face ownership and no race; on sq5
    it equals the research's shared-face form to rounding (ScratchCLAUDE/Round5CLAUDE/AbsorbCLAUDE/
    proto_frame_MIRROR.py --part check: 3e-12 after 400 steps).
    sq9: the isotropic 9-point operator L9 = Dxx + Dyy + (1/6) Dxx Dyy is stretched EXACTLY, as
        L~ = D~xx(u + D~yy u / 6) + D~yy u
    (D~ the stretched second differences; they commute): x memory for the field u + D~yy u / 6, y memory for u.
    The x part needs D~yy u at the x neighbours (i +- 1, j): computed here from THEIR y memory, which is therefore a
    ping-pong pair (par = this step's parity: read buffer par, the cell writes its own new values to 1 - par),
    3x the y recursion in the y frame rows, no extra pass. Outside the y frame D~yy u = Dyy u (raw), so the x strip
    runs D~xx((1 + Dyy/6) u) + Dyy u. With sigma = 0 it is L9 exactly. The neighbours go through the link words on
    the walls path; inside the sq9 frame no link is dropped or blocked (solids there act pinned: _class_box)."""
    L2 = 2 * Lc
    tb = AUX_TAB
    xm = _nb(U, b, i - 1, j, u, w, 1, -1, walls)
    xp = _nb(U, b, i + 1, j, u, w, 2, -1, walls)
    ym = _nb(U, b, i, j - 1, u, w, 3, -1, walls)
    yp = _nb(U, b, i, j + 1, u, w, 4, -1, walls)
    lx = 0.0
    ly = 0.0
    OMX = tb + 6 * L2
    if ti.static(nine):
        dmm = _nb(U, b, i - 1, j - 1, u, w, 5, 9, walls)
        dpm = _nb(U, b, i + 1, j - 1, u, w, 6, 10, walls)
        dmp = _nb(U, b, i - 1, j + 1, u, w, 7, 11, walls)
        dpp = _nb(U, b, i + 1, j + 1, u, w, 8, 12, walls)
        # D~yy u at (i - 1, j), (i, j), (i + 1, j)
        dym = dmp - 2.0 * xm + dmm
        dy0 = yp - 2.0 * u + ym
        dyp = dpp - 2.0 * xp + dpm
        if sy >= 0:
            Ln = L2 * n
            OMY = OMX + 3 * Ln
            ob = OMY + par * (3 * Ln)
            nb_ = OMY + (1 - par) * (3 * Ln)
            bm, am = AUX[tb + 2 * L2 + sy], AUX[tb + 3 * L2 + sy]
            bp, ap = AUX[tb + 4 * L2 + sy], AUX[tb + 5 * L2 + sy]
            bc, ac = AUX[tb + sy], AUX[tb + L2 + sy]
            k0 = ob + sy * n + i
            # the cell itself: its new y memory goes to the other buffer
            gp = yp - u
            gm = u - ym
            pm = bm * AUX[k0 - 0] + am * gm
            pp = bp * AUX[k0 + Ln] + ap * gp
            dq = (gp + pp) - (gm + pm)
            z = bc * AUX[k0 + 2 * Ln] + ac * dq
            kn = nb_ + sy * n + i
            AUX[kn] = pm
            AUX[kn + Ln] = pp
            AUX[kn + 2 * Ln] = z
            dy0 = dq + z
            # the x neighbours, from their OLD memory (read only)
            gp = dmp - xm
            gm = xm - dmm
            pm = bm * AUX[k0 - 1] + am * gm
            pp = bp * AUX[k0 - 1 + Ln] + ap * gp
            dq = (gp + pp) - (gm + pm)
            dym = dq + (bc * AUX[k0 - 1 + 2 * Ln] + ac * dq)
            gp = dpp - xp
            gm = xp - dpm
            pm = bm * AUX[k0 + 1] + am * gm
            pp = bp * AUX[k0 + 1 + Ln] + ap * gp
            dq = (gp + pp) - (gm + pm)
            dyp = dq + (bc * AUX[k0 + 1 + 2 * Ln] + ac * dq)
        t6 = 1.0 / 6.0
        wm = xm + dym * t6
        w0 = u + dy0 * t6
        wp = xp + dyp * t6
        gxp = wp - w0
        gxm = w0 - wm
        lx = gxp - gxm
        if sx >= 0:
            k0 = OMX + sx * n + j
            km = k0
            kp = k0 + L2 * n
            kz = k0 + 2 * L2 * n
            pm = AUX[tb + 2 * L2 + sx] * AUX[km] + AUX[tb + 3 * L2 + sx] * gxm
            pp = AUX[tb + 4 * L2 + sx] * AUX[kp] + AUX[tb + 5 * L2 + sx] * gxp
            AUX[km] = pm
            AUX[kp] = pp
            dq = (gxp + pp) - (gxm + pm)
            z = AUX[tb + sx] * AUX[kz] + AUX[tb + L2 + sx] * dq
            AUX[kz] = z
            lx = dq + z
        ly = dy0
    else:
        gxp = xp - u
        gxm = u - xm
        gyp = yp - u
        gym = u - ym
        lx = gxp - gxm
        ly = gyp - gym
        if sx >= 0:
            k0 = OMX + sx * n + j
            km = k0
            kp = k0 + L2 * n
            kz = k0 + 2 * L2 * n
            pm = AUX[tb + 2 * L2 + sx] * AUX[km] + AUX[tb + 3 * L2 + sx] * gxm
            pp = AUX[tb + 4 * L2 + sx] * AUX[kp] + AUX[tb + 5 * L2 + sx] * gxp
            AUX[km] = pm
            AUX[kp] = pp
            dq = (gxp + pp) - (gxm + pm)
            z = AUX[tb + sx] * AUX[kz] + AUX[tb + L2 + sx] * dq
            AUX[kz] = z
            lx = dq + z
        if sy >= 0:
            k0 = OMX + 3 * L2 * n + sy * n + i
            km = k0
            kp = k0 + L2 * n
            kz = k0 + 2 * L2 * n
            pm = AUX[tb + 2 * L2 + sy] * AUX[km] + AUX[tb + 3 * L2 + sy] * gym
            pp = AUX[tb + 4 * L2 + sy] * AUX[kp] + AUX[tb + 5 * L2 + sy] * gyp
            AUX[km] = pm
            AUX[kp] = pp
            dq = (gyp + pp) - (gym + pm)
            z = AUX[tb + sy] * AUX[kz] + AUX[tb + L2 + sy] * dq
            AUX[kz] = z
            ly = dq + z
    return lx + ly


@ti.func
def _abs_pass(U: ti.template(), M: ti.template(), CC: ti.template(), AUX: ti.template(), LV: ti.template(),
              s0: ti.template(), q: ti.template(), lat: ti.template(), nine: ti.template(), d: ti.template(),
              walls: ti.template(), absk: ti.template()):
    """One step's interior pass with an absorbing layer, then the clamped ring (the layer's outer end, and the
    ring of every masked shape with edge 'open', which the layer keeps off the ring). Template-specialised per
    kind (a generic coefficient-table rule compiled for 37-139 s in the research). Every cell outside the layer
    computes TODAY's update with TODAY's expression (sq: lap_sq / lap_sq_w then the app's formula; tri: the
    JS-order sum and the app's formula), so the open domain is bit for bit the plain solver's update.
      ABS_SPONGE: a cell whose mask code m = LV[i, j] is >= 2 takes the CENTRED damping form (the research's
        section 4.2: it keeps the undamped CFL limit; the app's -d(u - u_prev) form lowers it),
            u+ = (2u + CC lap - d_g (u - u-)) / (1 + d_m/2) - (1 - d_m/2) / (1 + d_m/2) u-
        with the two factors read from the AUX tables (no per-cell divide: an f64 divide per cell doubled the step
        at N 4097); d_g is the global damping (the app's form, unchanged).
      ABS_CPML: a cell within Lc + 1 of the array ring (square lattice) takes _cpml_lap; its memory is zeroed while
        the cell is not free (a solid painted in the frame). s0 carries 3 x the sq9 ping-pong parity on top of the
        slot (s0 % 3 is the slot as everywhere; (s0 // 3 + q) % 2 the parity of step q)."""
    n = U.shape[1]
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        a, b, c = (s0 + q) % 3, (s0 + q + 1) % 3, (s0 + q + 2) % 3
        par = (s0 // 3 + q) % 2                 # the sq9 CPML's ping-pong parity (packed into s0 by step())
        w = 0
        free = 1
        if ti.static(walls):
            w = ti.cast(M[i, j], ti.i32)
            free = w & 1
        else:
            if M[i, j] == 0:
                free = 0
        if free == 0:
            U[c, i, j] = 0.0
            if ti.static(absk == ABS_CPML):
                Lc = ti.cast(AUX[0], ti.i32)
                L2 = 2 * Lc
                OM = AUX_TAB + 6 * L2
                sx = _fidx(i, n, Lc)
                sy = _fidx(j, n, Lc)
                if sx >= 0:
                    for k in ti.static(range(3)):
                        AUX[OM + (k * L2 + sx) * n + j] = 0.0
                if sy >= 0:
                    # sq5: the y memory (3); sq9: both ping-pong buffers of it (6)
                    for k in ti.static(range(6 if nine else 3)):
                        AUX[OM + 3 * L2 * n + (k * L2 + sy) * n + i] = 0.0
        else:
            u = U[b, i, j]
            if ti.static(lat == LAT_TRI):
                s = 0.0
                if ti.static(walls):
                    s = sum_tri_w(U, b, i, j, u, w)
                else:
                    s = (U[b, i - 1, j] + U[b, i + 1, j] + U[b, i, j - 1] + U[b, i, j + 1]
                         + U[b, i + 1, j - 1] + U[b, i - 1, j + 1])
                if ti.static(absk == ABS_SPONGE):
                    m = ti.cast(LV[i, j], ti.i32)
                    if m >= 2:
                        ua = U[a, i, j]
                        U[c, i, j] = ((2 * u + CC[i, j] * (2.0 / 3.0) * (s - 6 * u) - d * (u - ua)) * AUX[AUX_INV + m]
                                      - AUX[AUX_C2 + m] * ua)
                    else:
                        U[c, i, j] = (2 * u - U[a, i, j] + CC[i, j] * (2.0 / 3.0) * (s - 6 * u)) - d * (u - U[a, i, j])
                else:
                    U[c, i, j] = (2 * u - U[a, i, j] + CC[i, j] * (2.0 / 3.0) * (s - 6 * u)) - d * (u - U[a, i, j])
            else:
                lap = 0.0
                if ti.static(absk == ABS_CPML):
                    Lc = ti.cast(AUX[0], ti.i32)
                    sx = _fidx(i, n, Lc)
                    sy = _fidx(j, n, Lc)
                    if sx >= 0 or sy >= 0:
                        lap = _cpml_lap(U, AUX, b, i, j, n, u, w, sx, sy, Lc, par, nine, walls)
                    else:
                        if ti.static(walls):
                            lap = lap_sq_w(U, b, i, j, i - 1, i + 1, j - 1, j + 1, nine, w)
                        else:
                            lap = lap_sq(U, b, i, j, i - 1, i + 1, j - 1, j + 1, nine)
                else:
                    if ti.static(walls):
                        lap = lap_sq_w(U, b, i, j, i - 1, i + 1, j - 1, j + 1, nine, w)
                    else:
                        lap = lap_sq(U, b, i, j, i - 1, i + 1, j - 1, j + 1, nine)
                if ti.static(absk == ABS_SPONGE):
                    m = ti.cast(LV[i, j], ti.i32)
                    if m >= 2:
                        ua = U[a, i, j]
                        U[c, i, j] = (2 * u + CC[i, j] * lap - d * (u - ua)) * AUX[AUX_INV + m] - AUX[AUX_C2 + m] * ua
                    else:
                        U[c, i, j] = (2 * u - U[a, i, j] + CC[i, j] * lap) - d * (u - U[a, i, j])
                else:
                    U[c, i, j] = (2 * u - U[a, i, j] + CC[i, j] * lap) - d * (u - U[a, i, j])
    # the ring: clamped (the layer's Dirichlet end; a masked open shape never reaches it)
    for t in range(n):
        c = (s0 + q + 2) % 3
        U[c, t, 0] = 0.0
        U[c, t, n - 1] = 0.0
        U[c, 0, t] = 0.0
        U[c, n - 1, t] = 0.0


@ti.func
def _steps(U: ti.template(), M: ti.template(), CC: ti.template(), DRV: ti.template(), SC: ti.template(),
           HIST: ti.template(), PRB: ti.template(), s0: ti.template(), K: ti.template(), lat: ti.template(),
           nine: ti.template(), bc: ti.template(), masked: ti.template(), srck: ti.template(),
           nsrc: ti.template(), d: ti.template(), kmur: ti.template(), jw: ti.template(), jd: ti.template(),
           cA: ti.template(), cB: ti.template(), half: ti.template(), rec: ti.template(),
           slitm: ti.template(), h0: ti.template(), walls: ti.template(), AUX: ti.template(), LV: ti.template(),
           absk: ti.template()):
    """The K steps (inlined into k_run / k_run_rec, so its loops are the kernel's top-level,
    parallel loops). HIST and PRB are read only when rec is on. EVERY parameter is a template
    (passed by reference): a by-value scalar parameter becomes a copy statement at the top of the
    kernel, i.e. one more serial task per launch -- measured 30/09/2026 on Vulkan: +40-60 us per
    launch (+5-8 us/step at N <= 1025) until the scalars were templates; CUDA showed nothing.
    walls (04/10/2026): M is the LINK WORD array (see W_B4) instead of the mask; every neighbour
    goes through _wv (the neighbour, 0, or u_i itself) and then into the same formula. walls =
    False compiles exactly the code of 03/10/2026 (the web app's path).
    absk (09/10/2026): ABS_NONE compiles exactly the code of before (k_run, k_run_rec pass dummies for AUX and LV);
    ABS_SPONGE / ABS_CPML run _abs_pass (the interior with the layer, then the clamped ring) instead of the lattice
    branches below (never on a torus: periodic is exclusive with an absorbing layer)."""
    n = U.shape[1]
    for q in ti.static(range(K)):
        # -- interior -------------------------------------------------------------------------
        if ti.static(absk != ABS_NONE):
            _abs_pass(U, M, CC, AUX, LV, s0, q, lat, nine, d, walls, absk)
        elif ti.static(lat == LAT_TRI and bc == BC_PER):
            # the RHOMBIC TORUS (05/10/2026, port-only: the triangular 'array' with bc = periodic):
            # ONE wrapped update over every cell, the six neighbours taken mod n -- the same
            # arithmetic, in the same order, as the clamped branch below; no ring
            if ti.static(walls):
                for i, j in ti.ndrange(n, n):
                    a, b, c = (s0 + q) % 3, (s0 + q + 1) % 3, (s0 + q + 2) % 3
                    w = ti.cast(M[i, j], ti.i32)
                    if (w & 1) == 0:
                        U[c, i, j] = 0.0
                    else:
                        u = U[b, i, j]
                        s = sum_tri_wi(U, b, i, j, (i - 1 + n) % n, (i + 1) % n, (j - 1 + n) % n, (j + 1) % n, u, w)
                        U[c, i, j] = (2 * u - U[a, i, j] + CC[i, j] * (2.0 / 3.0) * (s - 6 * u)) - d * (u - U[a, i, j])
            else:
                for i, j in ti.ndrange(n, n):
                    a, b, c = (s0 + q) % 3, (s0 + q + 1) % 3, (s0 + q + 2) % 3
                    if M[i, j] == 0:
                        U[c, i, j] = 0.0
                    else:
                        im, ip, jm, jp = (i - 1 + n) % n, (i + 1) % n, (j - 1 + n) % n, (j + 1) % n
                        u = U[b, i, j]
                        s = (U[b, im, j] + U[b, ip, j] + U[b, i, jm] + U[b, i, jp]
                             + U[b, ip, jm] + U[b, im, jp])
                        U[c, i, j] = (2 * u - U[a, i, j] + CC[i, j] * (2.0 / 3.0) * (s - 6 * u)) - d * (u - U[a, i, j])
        elif ti.static(lat == LAT_TRI):
            # 6 neighbours at distance 1: lap = (2/3)(sum u_m - 6 u_0), JS order
            # (i-1,j) (i+1,j) (i,j-1) (i,j+1) (i+1,j-1) (i-1,j+1)
            if ti.static(walls):
                for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
                    a, b, c = (s0 + q) % 3, (s0 + q + 1) % 3, (s0 + q + 2) % 3
                    w = ti.cast(M[i, j], ti.i32)
                    if (w & 1) == 0:
                        U[c, i, j] = 0.0
                    else:
                        u = U[b, i, j]
                        s = sum_tri_w(U, b, i, j, u, w)
                        U[c, i, j] = (2 * u - U[a, i, j] + CC[i, j] * (2.0 / 3.0) * (s - 6 * u)) - d * (u - U[a, i, j])
            else:
                for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
                    a, b, c = (s0 + q) % 3, (s0 + q + 1) % 3, (s0 + q + 2) % 3
                    if M[i, j] == 0:
                        U[c, i, j] = 0.0
                    else:
                        u = U[b, i, j]
                        s = (U[b, i - 1, j] + U[b, i + 1, j] + U[b, i, j - 1] + U[b, i, j + 1]
                             + U[b, i + 1, j - 1] + U[b, i - 1, j + 1])
                        U[c, i, j] = (2 * u - U[a, i, j] + CC[i, j] * (2.0 / 3.0) * (s - 6 * u)) - d * (u - U[a, i, j])
            # the triangular array edge is clamped (the web app's shapes always; the 'array' shape with
            # bc dirichlet, neumann or the mur fallback -- a RIGID ring is made by the link words,
            # which never read the ring, 05/10/2026)
            for t in range(n):
                c = (s0 + q + 2) % 3
                U[c, t, 0] = 0.0
                U[c, t, n - 1] = 0.0
                U[c, 0, t] = 0.0
                U[c, n - 1, t] = 0.0
        elif ti.static(bc == BC_PER):
            # a torus has no boundary: ONE wrapped update over every cell
            if ti.static(walls):
                for i, j in ti.ndrange(n, n):
                    a, b, c = (s0 + q) % 3, (s0 + q + 1) % 3, (s0 + q + 2) % 3
                    w = ti.cast(M[i, j], ti.i32)
                    if (w & 1) == 0:
                        U[c, i, j] = 0.0
                    else:
                        lap = lap_sq_w(U, b, i, j, (i - 1 + n) % n, (i + 1) % n, (j - 1 + n) % n, (j + 1) % n,
                                       nine, w)
                        u = U[b, i, j]
                        U[c, i, j] = (2 * u - U[a, i, j] + CC[i, j] * lap) - d * (u - U[a, i, j])
            else:
                for i, j in ti.ndrange(n, n):
                    a, b, c = (s0 + q) % 3, (s0 + q + 1) % 3, (s0 + q + 2) % 3
                    if M[i, j] == 0:
                        U[c, i, j] = 0.0
                    else:
                        lap = lap_sq(U, b, i, j, (i - 1 + n) % n, (i + 1) % n, (j - 1 + n) % n, (j + 1) % n, nine)
                        u = U[b, i, j]
                        U[c, i, j] = (2 * u - U[a, i, j] + CC[i, j] * lap) - d * (u - U[a, i, j])
        else:
            if ti.static(walls):
                for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
                    a, b, c = (s0 + q) % 3, (s0 + q + 1) % 3, (s0 + q + 2) % 3
                    w = ti.cast(M[i, j], ti.i32)
                    if (w & 1) == 0:
                        U[c, i, j] = 0.0
                    else:
                        lap = lap_sq_w(U, b, i, j, i - 1, i + 1, j - 1, j + 1, nine, w)
                        u = U[b, i, j]
                        U[c, i, j] = (2 * u - U[a, i, j] + CC[i, j] * lap) - d * (u - U[a, i, j])
            else:
                for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
                    a, b, c = (s0 + q) % 3, (s0 + q + 1) % 3, (s0 + q + 2) % 3
                    if M[i, j] == 0:
                        U[c, i, j] = 0.0
                    else:
                        lap = lap_sq(U, b, i, j, i - 1, i + 1, j - 1, j + 1, nine)
                        u = U[b, i, j]
                        U[c, i, j] = (2 * u - U[a, i, j] + CC[i, j] * lap) - d * (u - U[a, i, j])
            # -- boundary of the array ------------------------------------------------------
            if ti.static(bc == BC_DIR):
                for t in range(n):
                    c = (s0 + q + 2) % 3
                    U[c, t, 0] = 0.0
                    U[c, t, n - 1] = 0.0
                    U[c, 0, t] = 0.0
                    U[c, n - 1, t] = 0.0
            elif ti.static(bc == BC_NEU):
                # Sides first (each reads only interior cells: order-free, parallel), then the
                # corners, each = its diagonal interior neighbour of THIS step, read through the
                # side cell that already holds it: (0,0)=(1,1) (0,n-1)=(1,n-2) (n-1,0)=(n-2,1)
                # (n-1,n-1)=(n-2,n-2). That is the ghost-cell rule for which the 9-point operator
                # is symmetric, so the discrete energy is conserved (identity_tests_MIRROR.py).
                # PORT FIX 29/09/2026: the web app's ONE sequential ring loop writes (0,0) at
                # i = 0 from (1,0) BEFORE i = 1 refreshes it, i.e. from the rotating buffer's old
                # contents, the level three steps back (the other three corners were right). Through the
                # 9-point stencil's diagonal that changed the field by 8.6% of its peak near the
                # corner (pulse at the corner, 400 steps); it also broke the 8-fold symmetry and
                # energy conservation. The cross-check patches the web app the same way.
                for t in range(1, n - 1):
                    c = (s0 + q + 2) % 3
                    U[c, t, 0] = U[c, t, 1]
                    U[c, t, n - 1] = U[c, t, n - 2]
                    U[c, 0, t] = U[c, 1, t]
                    U[c, n - 1, t] = U[c, n - 2, t]
                for t in range(1):
                    c = (s0 + q + 2) % 3
                    U[c, 0, 0] = U[c, 1, 0]
                    U[c, 0, n - 1] = U[c, 1, n - 1]
                    U[c, n - 1, 0] = U[c, n - 1, 1]
                    U[c, n - 1, n - 1] = U[c, n - 2, n - 1]
            elif ti.static(bc == BC_MUR):
                # 1st-order Mur: u[0]^{n+1} = u[1]^n + k (u[1]^{n+1} - u[0]^n), k = (Co-1)/(Co+1).
                # Every ring cell reads interior cells only, so this is order-free: parallel.
                for t in range(1, n - 1):
                    b, c = (s0 + q + 1) % 3, (s0 + q + 2) % 3
                    U[c, t, 0] = U[b, t, 1] + kmur * (U[c, t, 1] - U[b, t, 0])
                    U[c, t, n - 1] = U[b, t, n - 2] + kmur * (U[c, t, n - 2] - U[b, t, n - 1])
                    U[c, 0, t] = U[b, 1, t] + kmur * (U[c, 1, t] - U[b, 0, t])
                    U[c, n - 1, t] = U[b, n - 2, t] + kmur * (U[c, n - 2, t] - U[b, n - 1, t])
                for t in range(1):
                    c = (s0 + q + 2) % 3
                    U[c, 0, 0] = 0.0
                    U[c, n - 1, 0] = 0.0
                    U[c, 0, n - 1] = 0.0
                    U[c, n - 1, n - 1] = 0.0
            elif ti.static(bc == BC_MUR2 or bc == BC_MUR2S):
                # Mur 2nd order (09/10/2026; Engquist-Majda A2, u_xt - u_tt/c + (c/2) u_yy = 0 at (h/2, n), the form
                # printed in Taflove & Hagness 2005; NotesCLAUDE/absorbing_boundaries_MIRROR.md section 4.1) on the
                # four sides, reading levels a, b, c of the ring cell and its interior partner and the tangential
                # second differences at level b; the CORNERS take Mur1 along the diagonal (diag-Mur1, k = (Co - sqrt2)
                # / (Co + sqrt2)): corners held at 0 blow up (research section 1). Every read is of level a or b or of
                # the interior's level c, so ONE parallel pass (t = 0: the four corners; t >= 1: the four sides).
                # The coefficients use the ring cell's own Co (sqrt CC): a slab medium's slow part is matched too.
                # BC_MUR2S (10/10/2026): the launch after a seed / set_levels / set_co / a mask change / a bc switch;
                # its FIRST step runs Mur1 on the sides (_mur1_side: zeroes the residual Mur2 would keep as a static
                # offset), the corners as always.
                for t in range(n - 1):
                    a, b, c = (s0 + q) % 3, (s0 + q + 1) % 3, (s0 + q + 2) % 3
                    if t == 0:
                        _mur2_corner(U, CC, b, c, 0, 0, 1, 1)
                        _mur2_corner(U, CC, b, c, n - 1, 0, -1, 1)
                        _mur2_corner(U, CC, b, c, 0, n - 1, 1, -1)
                        _mur2_corner(U, CC, b, c, n - 1, n - 1, -1, -1)
                    else:
                        if ti.static(bc == BC_MUR2S and q == 0):
                            _mur1_side(U, CC, b, c, t, 0, t, 1)
                            _mur1_side(U, CC, b, c, t, n - 1, t, n - 2)
                            _mur1_side(U, CC, b, c, 0, t, 1, t)
                            _mur1_side(U, CC, b, c, n - 1, t, n - 2, t)
                        else:
                            _mur2_side(U, CC, a, b, c, t, 0, t, 1, 1, 0)
                            _mur2_side(U, CC, a, b, c, t, n - 1, t, n - 2, 1, 0)
                            _mur2_side(U, CC, a, b, c, 0, t, 1, t, 0, 1)
                            _mur2_side(U, CC, a, b, c, n - 1, t, n - 2, t, 0, 1)
        # -- any masked domain is Dirichlet on the mask edge (the interior pass already zeroed
        #    the interior cells outside it; this is the array ring) --------------------------
        if ti.static(masked):
            if ti.static(walls):
                for t in range(n):
                    c = (s0 + q + 2) % 3
                    if (ti.cast(M[t, 0], ti.i32) & 1) == 0:
                        U[c, t, 0] = 0.0
                    if (ti.cast(M[t, n - 1], ti.i32) & 1) == 0:
                        U[c, t, n - 1] = 0.0
                    if (ti.cast(M[0, t], ti.i32) & 1) == 0:
                        U[c, 0, t] = 0.0
                    if (ti.cast(M[n - 1, t], ti.i32) & 1) == 0:
                        U[c, n - 1, t] = 0.0
            else:
                for t in range(n):
                    c = (s0 + q + 2) % 3
                    if M[t, 0] == 0:
                        U[c, t, 0] = 0.0
                    if M[t, n - 1] == 0:
                        U[c, t, n - 1] = 0.0
                    if M[0, t] == 0:
                        U[c, 0, t] = 0.0
                    if M[n - 1, t] == 0:
                        U[c, n - 1, t] = 0.0
        # -- driven sources, AFTER the boundary, as in the JS ---------------------------------
        if ti.static(srck == SRC_CELLS):
            for k in range(nsrc):
                c = (s0 + q + 2) % 3
                U[c, SC[k, 0], SC[k, 1]] = DRV[q]
        elif ti.static(srck == SRC_SLIT):
            for i in range(n):                    # a rigid wall with two openings ...
                c = (s0 + q + 2) % 3
                if jw > 0 and jw < n:
                    inA = ti.abs(i - cA) < half
                    inB = ti.abs(i - cB) < half
                    if not inA and not inB:
                        U[c, i, jw] = 0.0
            for i in range(n):                    # ... driven from below
                c = (s0 + q + 2) % 3
                if jd > 0 and jd < n:
                    if ti.static(slitm):
                        # with solids present the drive line skips masked cells, so it never
                        # writes into a solid (the web app, which has no solids, drives the
                        # whole row, outside the domain too; slitm = 0 keeps that exactly)
                        if ti.static(walls):
                            if (ti.cast(M[i, jd], ti.i32) & 1) != 0:
                                U[c, i, jd] = DRV[q]
                        else:
                            if M[i, jd] != 0:
                                U[c, i, jd] = DRV[q]
                    else:
                        U[c, i, jd] = DRV[q]
        elif ti.static(srck == SRC_BEAM):
            # the directional beam (01/10/2026): a SOFT source -- each driven cell gets its own
            # value ADDED to the new level (DRV is (K, nsrc) flattened, q-major). Added, not
            # imposed, so the aperture is transparent to waves crossing it (an imposed line would
            # be a mirror); the cells are distinct, so the parallel += never races.
            for k in range(nsrc):
                c = (s0 + q + 2) % 3
                U[c, SC[k, 0], SC[k, 1]] += DRV[q * nsrc + k]
        # -- the recorder: the web app's record(), after the step (reads only) ----------------
        if ti.static(rec):
            rec_fold(U, M, HIST, HCAP, (s0 + q + 1) % 3, (s0 + q + 2) % 3, walls, lat,
                     ti.static(lat == LAT_TRI and bc == BC_PER), LV, ti.static(absk != ABS_NONE))
            for t in range(1):
                c = (s0 + q + 2) % 3
                row = (h0 + q) % HCAP
                rec_sum(HIST, HCAP, HIST, row)
                HIST[row, 3] = ti.cast(U[c, PRB[0], PRB[1]], ACC_DT)
                HIST[row, 4] = ti.cast(U[c, PRB[2], PRB[3]], ACC_DT)
                HIST[row, 5] = ti.cast(U[c, PRB[4], PRB[5]], ACC_DT)


@ti.kernel
def k_run(U: ti.types.ndarray(ndim=3), M: ti.types.ndarray(ndim=2), CC: ti.types.ndarray(ndim=2),
          DRV: ti.types.ndarray(ndim=1), SC: ti.types.ndarray(ndim=2),
          s0: ti.i32, K: ti.template(), lat: ti.template(), nine: ti.template(), bc: ti.template(),
          masked: ti.template(), srck: ti.template(), nsrc: ti.i32, d: float, kmur: float,
          jw: ti.i32, jd: ti.i32, cA: ti.i32, cB: ti.i32, half: ti.i32, slitm: ti.template(),
          walls: ti.template()):
    # the solver alone: the argument list of 29/09 (+ the slitm and walls templates, ~free).
    # walls = True: M is the link word array (Membrane.W), not the mask
    _steps(U, M, CC, DRV, SC, U, DRV, s0, K, lat, nine, bc, masked, srck, nsrc, d, kmur,
           jw, jd, cA, cB, half, False, slitm, 0, walls, DRV, M, ABS_NONE)


@ti.kernel
def k_run_rec(U: ti.types.ndarray(ndim=3), M: ti.types.ndarray(ndim=2), CC: ti.types.ndarray(ndim=2),
              DRV: ti.types.ndarray(ndim=1), SC: ti.types.ndarray(ndim=2), HIST: ti.types.ndarray(ndim=2),
              PRB: ti.types.ndarray(ndim=1),
              s0: ti.i32, K: ti.template(), lat: ti.template(), nine: ti.template(), bc: ti.template(),
              masked: ti.template(), srck: ti.template(), nsrc: ti.i32, d: float, kmur: float,
              jw: ti.i32, jd: ti.i32, cA: ti.i32, cB: ti.i32, half: ti.i32, slitm: ti.template(), h0: ti.i32,
              walls: ti.template()):
    # the solver + the recorder after every step: HIST rows [0, HCAP) are the history ring (kin,
    # pot, amax, probe A, probe B, mode probe), rows [HCAP, HCAP + NSLOT) the reduction slots
    _steps(U, M, CC, DRV, SC, HIST, PRB, s0, K, lat, nine, bc, masked, srck, nsrc, d, kmur,
           jw, jd, cA, cB, half, True, slitm, h0, walls, DRV, M, ABS_NONE)


@ti.kernel
def k_run_abs(U: ti.types.ndarray(ndim=3), M: ti.types.ndarray(ndim=2), CC: ti.types.ndarray(ndim=2),
              DRV: ti.types.ndarray(ndim=1), SC: ti.types.ndarray(ndim=2), AUX: ti.types.ndarray(ndim=1),
              LV: ti.types.ndarray(ndim=2),
              s0: ti.i32, K: ti.template(), lat: ti.template(), nine: ti.template(), bc: ti.template(),
              masked: ti.template(), srck: ti.template(), nsrc: ti.i32, d: float, kmur: float,
              jw: ti.i32, jd: ti.i32, cA: ti.i32, cB: ti.i32, half: ti.i32, slitm: ti.template(),
              walls: ti.template(), absk: ti.template()):
    # 09/10/2026: k_run with an absorbing layer (absk = ABS_SPONGE / ABS_CPML): + AUX (the layer's tables and the
    # CPML memory, packed) and LV (the device mask M with its layer codes; on the walls path M here is the word array)
    _steps(U, M, CC, DRV, SC, U, DRV, s0, K, lat, nine, bc, masked, srck, nsrc, d, kmur,
           jw, jd, cA, cB, half, False, slitm, 0, walls, AUX, LV, absk)


@ti.kernel
def k_run_abs_rec(U: ti.types.ndarray(ndim=3), M: ti.types.ndarray(ndim=2), CC: ti.types.ndarray(ndim=2),
                  DRV: ti.types.ndarray(ndim=1), SC: ti.types.ndarray(ndim=2), HIST: ti.types.ndarray(ndim=2),
                  PRB: ti.types.ndarray(ndim=1), AUX: ti.types.ndarray(ndim=1), LV: ti.types.ndarray(ndim=2),
                  s0: ti.i32, K: ti.template(), lat: ti.template(), nine: ti.template(), bc: ti.template(),
                  masked: ti.template(), srck: ti.template(), nsrc: ti.i32, d: float, kmur: float,
                  jw: ti.i32, jd: ti.i32, cA: ti.i32, cB: ti.i32, half: ti.i32, slitm: ti.template(), h0: ti.i32,
                  walls: ti.template(), absk: ti.template()):
    # k_run_rec with an absorbing layer: the recorder sums the open domain only (LV < 2)
    _steps(U, M, CC, DRV, SC, HIST, PRB, s0, K, lat, nine, bc, masked, srck, nsrc, d, kmur,
           jw, jd, cA, cB, half, True, slitm, h0, walls, AUX, LV, absk)


@ti.kernel
def k_rescale(U: ti.types.ndarray(ndim=3), M: ti.types.ndarray(ndim=2), CC: ti.types.ndarray(ndim=2),
              a: ti.i32, b: ti.i32, r: float, tri: ti.i32, nine: ti.i32, wrap: ti.i32):
    # set_co(rescale=True): re-express the previous time level (slot a) at the new time step.
    # CC still holds the OLD Co^2 here (set_co rebuilds it afterwards).
    n = U.shape[1]
    for i, j in ti.ndrange(n, n):
        interior = i > 0 and j > 0 and i < n - 1 and j < n - 1
        if M[i, j] != 0 and (interior or wrap):
            im = (i - 1 + n) % n
            ip = (i + 1) % n
            jm = (j - 1 + n) % n
            jp = (j + 1) % n
            u = U[b, i, j]
            lap = 0.0
            if tri:
                lap = (2.0 / 3.0) * (U[b, im, j] + U[b, ip, j] + U[b, i, jm] + U[b, i, jp]
                                     + U[b, ip, jm] + U[b, im, jp] - 6 * u)
            elif nine:
                lap = ((2.0 / 3.0) * (U[b, im, j] + U[b, ip, j] + U[b, i, jm] + U[b, i, jp] - 4 * u)
                       + (1.0 / 6.0) * (U[b, im, jm] + U[b, ip, jm] + U[b, im, jp] + U[b, ip, jp] - 4 * u))
            else:
                lap = U[b, im, j] + U[b, ip, j] + U[b, i, jm] + U[b, i, jp] - 4 * u
            U[a, i, j] = u - r * (u - U[a, i, j]) + 0.5 * (r * r - r) * CC[i, j] * lap


@ti.kernel
def k_rescale_w(U: ti.types.ndarray(ndim=3), W: ti.types.ndarray(ndim=2), CC: ti.types.ndarray(ndim=2),
                a: ti.i32, b: ti.i32, r: float, lat: ti.template(), nine: ti.template(), wrap: ti.template()):
    # k_rescale on the walls path (04/10/2026): the same re-expression with the walls' operator
    n = U.shape[1]
    for i, j in ti.ndrange(n, n):
        w = ti.cast(W[i, j], ti.i32)
        inner = 1
        if ti.static(not wrap):
            if i <= 0 or j <= 0 or i >= n - 1 or j >= n - 1:
                inner = 0
        if (w & 1) == 1 and inner == 1:
            im = (i - 1 + n) % n
            ip = (i + 1) % n
            jm = (j - 1 + n) % n
            jp = (j + 1) % n
            u = U[b, i, j]
            lap = 0.0
            if ti.static(lat == LAT_TRI):
                # wrapped indices: on the rhombic torus (wrap) every cell is inner; elsewhere the
                # inner cells' neighbours never wrap, so this is sum_tri_w exactly
                lap = (2.0 / 3.0) * (sum_tri_wi(U, b, i, j, im, ip, jm, jp, u, w) - 6 * u)
            else:
                lap = lap_sq_w(U, b, i, j, im, ip, jm, jp, nine, w)
            U[a, i, j] = u - r * (u - U[a, i, j]) + 0.5 * (r * r - r) * CC[i, j] * lap


@ti.kernel
def k_stats(U: ti.types.ndarray(ndim=3), M: ti.types.ndarray(ndim=2), acc: ti.types.ndarray(ndim=2),
            a: ti.i32, b: ti.i32, walls: ti.template(), lat: ti.template(), wrap: ti.template(),
            LV: ti.types.ndarray(ndim=2), absx: ti.template()):
    # the web app's record() sums (kinetic, gradient, peak |u| over the masked interior) of the
    # CURRENT state, on demand. Same two-level reduction as the recorder: the one-address
    # atomic version this replaces took 62 ms at N = 4097 and 247 ms at N = 8193 (29/09/2026).
    # acc row 0 = the result, rows 1..NSLOT = the slots. walls: M is the link word (rec_fold). absx (09/10/2026):
    # the open domain only (LV = the device mask with its layer codes)
    rec_fold(U, M, acc, 1, a, b, walls, lat, wrap, LV, absx)
    for t in range(1):
        rec_sum(acc, 1, acc, 0)


@ti.kernel
def k_zero_masked(U: ti.types.ndarray(ndim=3), M: ti.types.ndarray(ndim=2)):
    # every time level, wherever the mask is 0 (a cell that just became solid holds no wave)
    for i, j in ti.ndrange(U.shape[1], U.shape[2]):
        if M[i, j] == 0:
            U[0, i, j] = 0.0
            U[1, i, j] = 0.0
            U[2, i, j] = 0.0


@ti.kernel
def k_patch_mask(U: ti.types.ndarray(ndim=3), M: ti.types.ndarray(ndim=2), P: ti.types.ndarray(ndim=2),
                 i0: ti.i32, j0: ti.i32):
    # write a mask patch at (i0, j0) and zero all three time levels where it is 0: one launch
    # (and one small host->device copy of P) per painted segment, not a whole-grid upload
    for a, b in ti.ndrange(P.shape[0], P.shape[1]):
        v = P[a, b]
        M[i0 + a, j0 + b] = v
        if v == 0:
            U[0, i0 + a, j0 + b] = 0.0
            U[1, i0 + a, j0 + b] = 0.0
            U[2, i0 + a, j0 + b] = 0.0


@ti.kernel
def k_load(U: ti.types.ndarray(ndim=3), s: ti.i32, src: ti.types.ndarray(ndim=2)):
    for i, j in ti.ndrange(U.shape[1], U.shape[2]):
        U[s, i, j] = src[i, j]


@ti.kernel
def k_store(U: ti.types.ndarray(ndim=3), s: ti.i32, dst: ti.types.ndarray(ndim=2)):
    for i, j in ti.ndrange(U.shape[1], U.shape[2]):
        dst[i, j] = U[s, i, j]


@ti.kernel
def k_zero(U: ti.types.ndarray(ndim=3), s: ti.i32):
    for i, j in ti.ndrange(U.shape[1], U.shape[2]):
        U[s, i, j] = 0.0


@ti.kernel
def k_aux_put(AUX: ti.types.ndarray(ndim=1), T: ti.types.ndarray(ndim=1)):
    # 09/10/2026: the absorbing layer's tables into the head of AUX (the CPML memory behind them is kept)
    for k in range(T.shape[0]):
        AUX[k] = T[k]


@ti.kernel
def k_aux_fill(AUX: ti.types.ndarray(ndim=1), start: ti.i32, v: float):
    # 09/10/2026: zero the CPML memory (AUX[start:]) at a reseed. 10/10/2026: the range starts at 0 (Vulkan's SPIR-V
    # codegen needs a constant begin for a top-level loop: 'Assertion failure: stmt->const_begin')
    for k in range(AUX.shape[0] - start):
        AUX[start + k] = v


# the web app's caps; 'Etri' (05/10/2026, port-only, triangular lattice only): the energy with the
# triangular operator's own potential (1/3)(gx^2 + gy^2 + gz^2), see rec_fold
HIST_CAP = {'E': 4000, 'amax': 4000, 'pA': 4000, 'pB': 4000, 'mode': 600, 'Etri': 4000}
HCOLS = 7                              # kin, pot, amax, probe A, probe B, mode probe, pot of the 3rd tri direction


class Membrane:
    """The solver state. Construct AFTER init_taichi().

    Masks: domain_mask_np (the shape) and solid_np (solids_MIRROR.py objects and painted walls,
    None when there are none; since 04/10/2026 the TYPE map: 0 free, 1 pinned, 2 rigid) compose to
    mask_np = domain AND solid_np == 0, which is what the sources, snapping and the recorder see
    (and what M holds on the device). While walls_needed(): C (class codes, see CL_F) and W (link
    words, see W_B4) on the device; the step reads W instead of M while walls_active()."""

    def __init__(self, params=None, scene=None):
        self.p = params or Params()
        self.N = 0
        self.U = self.M = self.CC = None
        self.slot = 0                   # U[slot] = previous level, U[slot+1] = current (mod 3)
        self.fdt = np.float64 if FP64 else np.float32
        self.acc = ti.ndarray(dtype=ACC_DT, shape=(1 + NSLOT, HCOLS))   # stats(): result row + slots
        self.acc.fill(0)
        self.drv = ti.ndarray(dtype=float, shape=KMAX)
        self.vdrv = None                # soft driven vertices (09/10/2026): the (K, nsrc) drive, per launch
        self._soft = (None, None)       # soft_tables()' cache: (key, tables)
        self.drive_cells = []           # the driven cells (cont: the source; driven vertices), host
        self.beam = None                # the beam's tables (beam_MIRROR.beam_tables) + its key
        self.beam_key = None
        self.nbeam = 0
        self.packet_info = None
        self.harm = [(1, 1.0)]
        self.step_n = 0
        self.t = 0.0
        self.drops = 0
        self.vtx_capped = 0
        self.clip_frac = None
        self._src_note = (None, None)   # source_note()'s cache: (key, note)
        # recorder: device ring HIST (HCAP rows: kin, pot, amax, probe A, probe B, mode) and the
        # reduction slots PS; the host history m.hist mirrors the web app's arrays
        self.record = True
        self.hist_dev = ti.ndarray(dtype=ACC_DT, shape=(HCAP + NSLOT, HCOLS))   # ring + reduction slots
        self.hist_dev.fill(0)           # rec_sum re-zeroes the slots after use; they must START at 0
        self.prb = ti.ndarray(dtype=ti.i32, shape=6)
        self.hrow = 0                   # the next HIST row the kernel writes (a ring of HCAP rows)
        self.hist = {k: [] for k in HIST_CAP}
        self.hist_emax = 0.0                        # 10/10/2026: the recorded E's maximum since the seed (ledger 'eopen')
        self.amax = 0.0
        # solids
        self.scene = scene
        if scene is not None:
            scene.m = self
        self.solid_np = None
        self.domain_mask_np = None
        self.has_solids = False
        self.n_solid = 0
        self._solid_key = None
        self._mask_ver = 0              # bumped on every mask change (the main-region cache key)
        self._main_cache = None         # (key, bool [i, j] or None, exact): the main region's free cells
        self.src_moved = False          # the requested source cell is SOLID, so the source was moved
        self.src_snap = 'none'          # how it was placed: none | walk | solid | nearest-main | centre
        # walls (04/10/2026): class codes C and link words W on the device, allocated only while
        # walls_needed(); n_wall_cells = free cells with a dropped or pinned-blocked link (0: the
        # web app's path runs), n_blk4 = tri cells with a blocked link 0-3 (main_region)
        self.C = self.W = None
        self.wcnt = ti.ndarray(dtype=ti.i32, shape=4)
        self.wcnt.fill(0)
        self._wn = (0, 0)               # (n_wall_cells, n_blk4) as last read from wcnt
        self._wdirty = False            # wcnt changed on the device since (a painted segment)
        self._wkey = None
        self._dom_ring_key = None       # _ring_key() the domain mask was built for (05/10/2026)
        # absorbing layers (09/10/2026): abs_np = the layer's mask codes (uint8 [i, j], >= 2 on layer cells) or None;
        # aux = the absorbing kernels' packed float array (tables + CPML memory) or None
        self.abs_np = None
        self._open_cache = None
        self.aux = None
        self._aux_key = None
        self._aux_tkey = None
        self._dom_layer_key = None      # _layer_key() the domain mask was built for
        self._cpml_par = 0              # the sq9 CPML's ping-pong parity (the y memory buffer read next)
        self._mur2_ok = False           # 10/10/2026: False = Mur2's next launch starts with a Mur1 step (step())
        self.reseed()

    # ---- geometry, exactly as the web app computes it ------------------------------------
    def tri(self):
        return self.p.lattice == 'tri'

    def c(self):
        return (self.N - 1) / 2

    def cell_xy(self, i, j):
        c = self.c()
        da, db = i - c, j - c
        return (da + db * 0.5, db * S3H) if self.tri() else (da, db)

    def xy_to_cell(self, x, y):
        c = self.c()
        if not self.tri():
            return c + x, c + y
        db = y / S3H
        da = x - db * 0.5
        return c + da, c + db

    def co_limit(self, damped=True):
        """The CFL limit C_max of the lattice + stencil, and (damped=True, the default) of the
        damping too. PORT DEVIATION (01/10/2026, NotesCLAUDE/absorbing_boundaries_MIRROR.md §6):
        the update u+ = 2u - u- + Co^2 L u - d (u - u-) has, per Laplacian eigenvalue -lam, the
        amplification roots of g^2 - (2 - d - Co^2 lam) g + (1 - d); |g| <= 1 iff
        Co^2 lam <= 4 - 2d, so C_max(d) = C_max(0) * sqrt(1 - d/2) (0 <= d < 2). The web app shows
        C_max(0) whatever the damping: d = 0.02 gives 0.7036 on sq5, and Co = 0.705 blew up while
        the readout said stable. damped=False is the web app's undamped value (the ledger's
        compat mode; the light-cone geometry)."""
        L = CO_LIMIT_T if self.tri() else (CO_LIMIT_9 if self.p.stencil == 9 else CO_LIMIT_5)
        d = self.p.damp
        if damped and d > 0:
            L *= math.sqrt(max(0.0, 1.0 - d / 2.0))
        return L

    def sq_full(self):
        """The full-grid square: 'square' at rad >= 0.999, or 'array' (05/10/2026), on the square
        lattice. Its walls ARE the array ring, governed by the bc control."""
        return (self.p.shape == 'array' or (self.p.shape == 'square' and self.p.rad >= 0.999)) and not self.tri()

    def full_array(self):
        """The whole lattice array is the domain (05/10/2026): sq_full(), or the triangular 'array'.
        The bc control governs its ring; Params.edge does not apply (there is no mask edge)."""
        return self.sq_full() or self.p.shape == 'array'

    def torus(self):
        """Every cell is a cell of a torus (the step wraps): bc = periodic on the square lattice (any
        shape, as the web app), or on the triangular 'array' (the rhombic torus, 05/10/2026). A
        web-app shape on the triangular lattice keeps its clamped ring whatever bc says.
        09/10/2026: the effective bc (periodic falls back with edge 'open': exclusive with an absorbing edge)."""
        return self.bc_effective() == 'periodic' and (not self.tri() or self.p.shape == 'array')

    def tri_torus(self):
        return self.tri() and self.torus()

    def _cpml_fits(self):
        """The CPML frame leaves an open domain at this N (at least ~7 cells across)."""
        lv = CPML_LEVELS.get(self.p.cpml_level)
        if lv is None:
            raise ValueError(f'cpml_level {self.p.cpml_level!r}: one of {CPML_LEVEL_NAMES}')
        return self.N >= 2 * lv['L'] + 9

    def bc_effective(self):
        """The boundary the solver applies to the array ring: p.bc on the square lattice; on the
        triangular lattice p.bc for the 'array' shape except 'mur' (not available yet: falls back to
        'dirichlet'), and 'dirichlet' (the web app's clamped ring) for every other shape.
        09/10/2026 (absorbing boundaries): mur2 / cpml / sponge act on the FULL array only, so on a masked shape
        they fall back to 'dirichlet' (the shape's own absorbing boundary is edge 'open'); periodic with edge
        'open' falls back to 'dirichlet' (exclusive); on the triangular 'array' mur, mur2 and cpml fall back to
        'dirichlet' (the sponge acts there); cpml on an array too small for its frame falls back to 'dirichlet'."""
        p = self.p
        if not self.tri():
            if self.full_array():
                if p.bc == 'cpml' and not self._cpml_fits():
                    return 'dirichlet'
                return p.bc
            if p.bc in ('mur2', 'cpml', 'sponge'):
                return 'dirichlet'
            if p.bc == 'periodic' and p.edge == 'open':
                return 'dirichlet'
            return p.bc
        if p.shape != 'array':
            return 'dirichlet'
        return p.bc if p.bc in TRI_BCS else 'dirichlet'

    def bc_info(self):
        """For the UI and the ledger (05/10/2026): what the boundary control does here.
        requested / effective (bc_effective()); fallback = the request is not what runs, with the
        reason (Mur on the triangular array; any bc but dirichlet on a triangular web-app shape);
        ring = what bounds the array, in words; options = the bc values that act here (others are
        listed in unavailable with their reason); edge_applies = Params.edge means something here
        (a masked domain has a mask edge; the full array does not).
        09/10/2026 (absorbing boundaries): absorbing = the effective bc absorbs (mur, mur2, cpml, sponge on the
        full array); layer = abs_layer() when the bc's frame runs (cpml, sponge), else None; the new bcs'
        reasons (TRI_MUR2_REASON, TRI_CPML_REASON, MASKED_ABS_REASON, OPEN_PERIODIC_REASON, a too-small array)."""
        p, tri = self.p, self.tri()
        eff = self.bc_effective()
        unavailable = {}
        if tri:
            unavailable['mur'] = TRI_MUR_REASON
            unavailable['mur2'] = TRI_MUR2_REASON
            unavailable['cpml'] = TRI_CPML_REASON
            if p.shape != 'array':
                for b in ('neumann', 'periodic', 'sponge'):
                    unavailable[b] = TRI_MASKED_REASON
        elif not self.full_array():
            for b in ('mur2', 'cpml', 'sponge'):
                unavailable[b] = MASKED_ABS_REASON.format(bc=b)
            if p.edge == 'open':
                unavailable['periodic'] = OPEN_PERIODIC_REASON
        elif not self._cpml_fits():
            Lc = CPML_LEVELS[p.cpml_level]['L']
            unavailable['cpml'] = (f'the CPML frame ({Lc} cells, {p.cpml_level}) needs N >= {2 * Lc + 9}: the ring '
                                   'stays clamped (Dirichlet); use the sponge or a larger N')
        reason = unavailable.get(p.bc) if eff != p.bc else None
        lay = self.abs_layer() if (self.full_array() and eff in LAYER_BCS) else None
        if eff == 'cpml':
            frame = (f'absorbing frame (CPML, {lay["Lc"]} cells, {p.cpml_level}; the open domain is the array '
                     'minus the frame)')
        elif eff == 'sponge':
            frame = (f'absorbing frame (sponge, {lay["L"]:.0f} cells, d_max {lay["dmax"]:.3g}; the open domain '
                     'is the array minus the frame)')
        else:
            frame = ''
        ring = {'dirichlet': 'clamped ring (Dirichlet, u = 0)', 'mur': 'absorbing ring (Mur, 1st order)',
                'mur2': 'absorbing ring (Mur, 2nd order; diagonal Mur-1 corners)',
                'cpml': frame, 'sponge': frame,
                'neumann': ('rigid ring (link removal, zero normal slope)' if tri
                            else 'free ring (Neumann, ghost copy)'),
                'periodic': 'rhombic torus (wraps along both lattice vectors)' if tri else 'torus (wraps)'}[eff]
        if not self.full_array():
            ring += ', where the shape reaches the array edge' if not tri else ' (outside the shape)'
        return dict(requested=p.bc, effective=eff, fallback=eff != p.bc, reason=reason, ring=ring,
                    options=[b for b in BCS if b not in unavailable], unavailable=unavailable,
                    edge_applies=not self.full_array(), torus=self.torus(),
                    absorbing=self.full_array() and eff in ABSORBING_BCS, layer=lay)

    # ---- absorbing layers (09/10/2026, NotesCLAUDE/absorbing_MIRROR.md) ----------------------------------
    def _open_edge(self):
        """A masked shape with edge 'open': a sponge outside the drawn edge."""
        return self.p.edge == 'open' and not self.full_array()

    def _sponge_cells(self):
        """The sponge's width in cells mode (physical cells): Params.sponge_L, or the auto value."""
        p = self.p
        if p.sponge_L is not None:
            return float(p.sponge_L)
        return 40.0 if self.base_half() >= 160 else SPONGE_L_MIN

    def sponge_width(self):
        """The sponge's width L in PHYSICAL cells. Cells mode: Params.sponge_L or the auto value (40, or 20 when the
        array's half-size is under 160). Domain mode: sponge_Lf x base_half(), never under SPONGE_L_MIN (the
        width follows the scale: a sponge reflects long waves, so a fixed width in cells would get thinner per
        wavelength as N grows; NotesCLAUDE/absorbing_MIRROR.md, scale). Capped at a third of the half-size."""
        p, bh = self.p, self.base_half()
        if p.scale == 'domain':
            lf = p.sponge_Lf if p.sponge_Lf is not None else self._sponge_cells() / bh
            L = max(SPONGE_L_MIN, lf * bh)
        else:
            L = self._sponge_cells()
        return float(min(L, max(2.0, bh / 3.0)))

    def sponge_dmax(self, L=None):
        """The damping per step at the sponge's outer end: Params.sponge_dmax, or auto (the research's tuning,
        section 4.8: 0.2 at L 40, 0.35 at L 20, linear between, 0.35 below 20). Above 40 cells (10/10/2026) 8 / L, i.e.
        L d_max fixed: the same damping per wave when the width follows the scale (domain mode), which keeps the box
        reflection at 1.44e-2 for sigma 3 / 6 / 12 / 24 with L 40 / 80 / 160 / 320 (0.2 fixed: up to 5.2e-2;
        ScratchCLAUDE/Round5CLAUDE/AbsorbCLAUDE/probe_scale_refl_MIRROR.log), and in cells mode lowers it too (L 80,
        sigma 3: 4.8e-3 / 1.3e-2 centre / corner against 5.7e-3 / 2.3e-2 at 0.2; probe_sponge_long_MIRROR.log)."""
        p = self.p
        if p.sponge_dmax is not None:
            return float(p.sponge_dmax)
        L = self.sponge_width() if L is None else L
        if L >= 40.0:
            return float(8.0 / L)
        if L <= 20.0:
            return 0.35
        return float(0.35 - 0.15 * (L - 20.0) / 20.0)

    def abs_layer(self):
        """The absorbing LAYER in use, or None (no layer: also Mur and Mur2, which act on the ring only):
          kind   'cpml' | 'sponge'
          where  'frame' (the full array's bc: the open domain is the array minus the frame) | 'edge' (a masked
                 shape's edge 'open': the layer lies OUTSIDE the drawn edge, the shape is capped so it fits)
          L      the width in physical cells; Ld (frame) the index depth of the first open cell
          cpml:   Lc (= L = Ld, cells), p, fac, alpha, smax (sigma_max, per unit time)
          sponge: p (SPONGE_P), dmax"""
        p = self.p
        if self.full_array():
            eff = self.bc_effective()
            if eff == 'cpml':
                lv = CPML_LEVELS[p.cpml_level]
                Lc = int(lv['L'])
                smax = lv['fac'] * (lv['p'] + 1) * math.log(1.0 / CPML_R0) / (2.0 * Lc)
                return dict(kind='cpml', where='frame', L=float(Lc), Ld=Lc, Lc=Lc, p=lv['p'], fac=lv['fac'],
                            alpha=lv['alpha'], smax=smax, level=p.cpml_level)
            if eff == 'sponge':
                L = self.sponge_width()
                Ld = int(math.ceil(L / S3H - 1e-9)) if self.tri() else int(math.ceil(L - 1e-9))
                # exact: one mask code per frame row (up to SPONGE_LEVELS rows), else the ramp quantised in s
                return dict(kind='sponge', where='frame', L=L, Ld=Ld, p=SPONGE_P, dmax=self.sponge_dmax(L),
                            exact=Ld <= SPONGE_LEVELS)
            return None
        if p.edge == 'open':
            L = self.sponge_width()
            return dict(kind='sponge', where='edge', L=L, Ld=0, p=SPONGE_P, dmax=self.sponge_dmax(L))
        return None

    def _layer_key(self):
        """What the domain's layer codes depend on beyond the shape (step() rebuilds the domain when it changed)."""
        lay = self.abs_layer()
        if lay is None:
            return None
        return (lay['kind'], lay['where'], round(lay['L'], 12), lay['Ld'], round(lay.get('dmax', 0.0), 12),
                lay.get('level'))

    def _frame_Ld(self):
        lay = self.abs_layer()
        return lay['Ld'] if (lay is not None and lay['where'] == 'frame') else 0

    def absorbing(self):
        """What absorbs here (09/10/2026), or None: the full array's bc 'mur' / 'mur2' / 'cpml' / 'sponge', or 'open'
        (a masked shape's sponge edge). A masked square-lattice shape that reaches a Mur ring is partly absorbing:
        None here (the ledger calls it 'mixed-mur')."""
        if self.full_array():
            eff = self.bc_effective()
            return eff if eff in ABSORBING_BCS else None
        return 'open' if self.p.edge == 'open' else None

    def edge_info(self):
        """For the UI (09/10/2026): what the domain EDGE (Params.edge) does here.
          requested   p.edge
          applies     a masked shape (the full array has no mask edge: the bc rules its ring / frame)
          options     EDGES where it applies, else []
          kind        'pinned' | 'rigid' | 'open' as it runs, None where it does not apply
          layer       abs_layer() for an open edge (the sponge outside the drawn edge), else None
          R_requested / R_open / capped   the shape's radius before / after the cap that makes room for the layer
                      (open edge only: the layer must fit inside the array ring)
          lines       [(text, level)] for the panel, level 'info' | 'warn' | 'dim'."""
        p = self.p
        applies = not self.full_array()
        lines = []
        out = dict(requested=p.edge, applies=applies, options=list(EDGES) if applies else [],
                   kind=p.edge if applies else None, layer=None, R_requested=None, R_open=None, capped=False,
                   lines=lines)
        if not applies:
            lines.append(('edge: n/a on the full array - the boundary (B) rules its ring'
                          + (' (cpml / sponge make it absorbing)' if not self.tri() else ' (sponge makes it absorbing)'),
                          'dim'))
            return out
        if p.edge == 'open':
            lay = self.abs_layer()
            Rq = (self.base_half() - 1.5) * p.rad
            Ro = self.dom_R()
            capped = Rq > self._dom_R_cap() + 1e-9
            out.update(layer=lay, R_requested=Rq, R_open=Ro, capped=capped)
            lines.append((f'open edge: a sponge {lay["L"]:.0f} cells wide outside the drawn edge (d_max '
                          f'{lay["dmax"]:.3g}, centred form) - waves leave through the edge', 'info'))
            if capped:
                lines.append((f'the shape shrank from R {Rq:.1f} to {Ro:.1f} cells to leave room for the sponge '
                              'inside the array', 'warn'))
            lines.append(('the layer is drawn tinted; it is not part of the energy, the peak or the ledger', 'dim'))
        elif p.edge == 'rigid':
            lines.append(('rigid edge: zero slope, the echo comes back upright', 'dim'))
        else:
            lines.append(('pinned edge: u = 0, the echo comes back inverted', 'dim'))
        return out

    def dom_rot(self):
        """The domain's rotation as its objects see it (solids' 'domain' anchor): p.rot, but 0 for a
        shape that ignores rotation ('array')."""
        return 0.0 if self.p.shape in FIXED_SHAPES else self.p.rot

    def base_half(self):
        return self.c() * (S3H if self.tri() else 1)

    def at_conf_rot(self):
        d = js_fmod(js_fmod(self.p.rot * 180 / PI - 30, 60) + 60, 60)
        return min(d, 60 - d) < 1e-6

    def dom_R(self):
        p = self.p
        # 09/10/2026: an absorbing frame (cpml / sponge on the full array) shrinks the OPEN domain by its width, so
        # what is defined relative to the domain (domain-anchored solids, the radius readouts) follows it
        fr = self._frame_Ld() * (S3H if self.tri() else 1.0) if self.full_array() else 0.0
        if p.shape == 'array':
            # the square's full-grid value at rad 1 (radius does not apply): base_half - 1.5
            return self.base_half() - 1.5 - fr
        R = (self.base_half() - 1.5) * p.rad - fr
        if self._open_edge():
            # 09/10/2026: an open edge's sponge lies OUTSIDE the drawn shape and must fit inside the array ring, so
            # R is capped (edge_info() says so); the snaps below apply to the capped radius
            R = min(R, self._dom_R_cap())
        if self.tri():
            if self.at_conf_rot():
                if p.shape == 'n6':
                    return max(2, js_round(R))
                if p.shape == 'rhomb' and abs(p.rtheta - PI / 3) < 1e-6:
                    return max(2, 2 * js_round(R / math.sqrt(3))) * math.sqrt(3) / 2
                if p.shape == 'n3':
                    L = js_round(R * math.sqrt(3))
                    L = max(3, js_round(L / 3) * 3)
                    return L / math.sqrt(3)
            return R
        return math.floor(R) + 0.5 if (p.shape == 'square' and not self.sq_full()) else R

    def _dom_R_cap(self):
        """The largest radius that leaves room for an open edge's sponge inside the array (09/10/2026): the layer
        reaches L beyond the shape's circumradius at most, and must stay 1.5 cells off the ring like the shapes."""
        return max(2.0, self.base_half() - 1.5 - self.sponge_width())

    def inner_R(self):
        p = self.p
        R, h = self.dom_R(), self.base_half()
        Ld = self._frame_Ld()
        if Ld:
            # 09/10/2026: an absorbing frame (cpml / sponge on the full array): the OPEN domain is the array minus the
            # frame, its edge the first open row (index depth Ld)
            h = h - Ld * (S3H if self.tri() else 1.0)
        if p.shape == 'array':
            ir = h                                    # the array's inradius: the ring lines (c on sq, c s3/2 on tri)
        elif p.shape == 'square':
            ir = h if self.sq_full() else R
        elif p.shape == 'circle':
            ir = R
        elif p.shape == 'rhomb':
            B = R * math.tan(p.rtheta / 2)
            ir = R * B / math.hypot(R, B)
        elif p.shape == 'unlicensed':
            ir = R / PLATE_ASPECT                     # the plate's half-height
        else:
            ir = R * math.cos(PI / int(p.shape[1:]))
        return min(ir, h)

    # ---- scale mode (02/10/2026, NotesCLAUDE/scale_MIRROR.md) ----------------------------
    def size_cells(self, attr):
        """A SCALE_SIZES length in cells as the solver uses it now (freq -> its wavelength Co/freq)."""
        p = self.p
        return p.Co / max(p.freq, 1e-12) if attr == 'freq' else getattr(p, attr)

    def fracs_from_cells(self, only_missing=False):
        """Set every fraction from the current cell values at the current N (the no-jump rule)."""
        p, r = self.p, self.inner_R()
        for attr, fa, _ in SCALE_SIZES:
            if not only_missing or getattr(p, fa) is None:
                setattr(p, fa, self.size_cells(attr) / r)
        # 09/10/2026: the sponge's width as a fraction of the array's half-size (sponge_width(); no jump)
        if not only_missing or p.sponge_Lf is None:
            p.sponge_Lf = self._sponge_cells() / self.base_half()

    def apply_scale(self):
        """'domain' mode: derive the cell sizes from the fractions (sigma = sigma_f r_in, freq =
        Co / (lam_f r_in), ...). 'cells' mode: nothing (bit-identical to before). Called by reseed()
        and set_co(); the app also calls it after a fraction slider moves."""
        p = self.p
        if p.scale != 'domain':
            return
        self.fracs_from_cells(only_missing=True)
        r = self.inner_R()
        for attr, fa, _ in SCALE_SIZES:
            v = getattr(p, fa) * r
            setattr(p, attr, p.Co / v if attr == 'freq' else v)

    def set_scale(self, mode):
        """Switch the scale mode WITHOUT a jump: entering 'domain' converts the current cell sizes
        into fractions of r_in (the cell values themselves are kept exactly); leaving it keeps the
        derived cell values. No reseed."""
        if mode not in SCALES:
            raise ValueError(f'scale mode {mode!r}: one of {SCALES}')
        if mode == 'domain' and self.p.scale != 'domain':
            self.fracs_from_cells()
        self.p.scale = mode

    def crossings(self):
        """Elapsed time in units of r_in / c (the web app's clock t = sum Co, divided by r_in)."""
        return self.t / max(self.inner_R(), 1e-12)

    def scale_warnings(self, mode=None):
        """Amber notes for sizes the grid resolves badly at this N: a width under SCALE_MIN_WIDTH
        cells, a wavelength under PPW_MIN cells. Only the sizes the current source uses; only in
        'domain' mode unless mode is given."""
        p = self.p
        if (mode or p.scale) != 'domain':
            return []
        used = {'sigma': p.src in ('pulse', 'vtx'),
                'freq': p.src in ('cont', 'slit', 'beam') or (p.src == 'vtx' and p.vtx_drive),
                'beam_width': p.src in BEAM_SOURCES, 'pk_len': p.src == 'packet', 'pk_lambda': p.src == 'packet'}
        names = {'sigma': 'pulse sigma', 'freq': 'wavelength', 'beam_width': 'beam width',
                 'pk_len': 'packet length', 'pk_lambda': 'packet wavelength'}
        out = []
        for attr, _, kind in SCALE_SIZES:
            if not used[attr]:
                continue
            v = self.size_cells(attr)
            lim = PPW_MIN if kind == 'wave' else SCALE_MIN_WIDTH
            if v < lim:
                out.append(f'{names[attr]} {v:.2f} cells < {lim:g} at N = {self.N}: '
                           + ('under-resolved, strong dispersion' if kind == 'wave' else 'under ~2 cells wide'))
        return out

    def plate_dims(self):
        """'unlicensed': (half-width, half-height, corner radius) in physical cells."""
        R = self.dom_R()
        H = 2 * R / PLATE_ASPECT
        return R, H / 2, PLATE_CORNER * H

    def plate_corners(self):
        """The 4 corner-arc MIDPOINTS (rotated): the plate's 'vertices' for the vertex sources,
        in the order of shape_poly()'s corners (bottom-right, top-right, top-left, bottom-left)."""
        hw, hh, rc = self.plate_dims()
        co, si = math.cos(self.p.rot), math.sin(self.p.rot)
        d = rc * math.sqrt(0.5)
        out = []
        for sx, sy in ((1, -1), (1, 1), (-1, 1), (-1, -1)):
            x, y = sx * (hw - rc + d), sy * (hh - rc + d)
            out.append((x * co - y * si, x * si + y * co))
        return out

    def shape_poly(self):
        p = self.p
        R, P = self.dom_R(), []
        if p.shape == 'unlicensed':
            # the rounded rectangle, PLATE_ARC_SEG segments per corner (4 x 17 = 68 points, CCW)
            hw, hh, rc = self.plate_dims()
            co, si = math.cos(p.rot), math.sin(p.rot)
            for k, (sx, sy) in enumerate(((1, -1), (1, 1), (-1, 1), (-1, -1))):
                cx, cy = sx * (hw - rc), sy * (hh - rc)
                a0 = -PI / 2 + k * PI / 2
                for s in range(PLATE_ARC_SEG + 1):
                    a = a0 + (PI / 2) * s / PLATE_ARC_SEG
                    x, y = cx + rc * math.cos(a), cy + rc * math.sin(a)
                    P.append((x * co - y * si, x * si + y * co))
            return P
        if p.shape == 'array' and self.tri():
            # the 60-degree rhombus through the four ring-corner cell centres, CCW from cell (0, 0)
            # (an acute corner): edges along 0, 60, 180, 240 degrees -- every edge a lattice row. It is
            # the web app's rhombus at rtheta 60, rot 30 with R = c sqrt(3) (corners at the ring, as the
            # full-grid square's are)
            n1, Ld = self.N - 1, self._frame_Ld()     # 09/10/2026: a sponge frame's open rhombus (Ld = 0 without)
            return [self.cell_xy(i, j) for i, j in ((Ld, Ld), (n1 - Ld, Ld), (n1 - Ld, n1 - Ld), (Ld, n1 - Ld))]
        if p.shape in ('square', 'array'):
            r = (self.c() - self._frame_Ld()) if self.sq_full() else R
            P = [(-r, -r), (r, -r), (r, r), (-r, r)]
        elif p.shape == 'circle':
            P = [(R * math.cos(TAU * k / 256), R * math.sin(TAU * k / 256)) for k in range(256)]
        elif p.shape == 'rhomb':
            B = R * math.tan(p.rtheta / 2)
            co, si = math.cos(p.rot), math.sin(p.rot)
            P = [(x * co - y * si, x * si + y * co) for x, y in ((R, 0), (0, B), (-R, 0), (0, -B))]
        else:
            nn = int(p.shape[1:])
            for k in range(nn):
                t = p.rot + PI / nn + k * TAU / nn
                P.append((R * math.cos(t), R * math.sin(t)))
        return P

    def clip_region(self):
        N = self.N
        if not self.tri():
            h = N / 2
            return [(-h, -h), (h, -h), (h, h), (-h, h)]
        s = N / (N - 1)
        out = []
        for i, j in ((0, 0), (N - 1, 0), (N - 1, N - 1), (0, N - 1)):
            x, y = self.cell_xy(i, j)
            out.append((x * s, y * s))
        return out

    @staticmethod
    def clip_convex(P, C):
        """Sutherland-Hodgman against a CCW convex polygon (the web app's clipConvex)."""
        side = lambda p, a, nx, ny: nx * (p[0] - a[0]) + ny * (p[1] - a[1])
        out = list(P)
        for e in range(len(C)):
            a, b = C[e], C[(e + 1) % len(C)]
            nx, ny = -(b[1] - a[1]), (b[0] - a[0])
            src, out = out, []
            for k in range(len(src)):
                p, q = src[(k + len(src) - 1) % len(src)], src[k]
                sp, sq = side(p, a, nx, ny), side(q, a, nx, ny)
                def cut():
                    t = sp / (sp - sq)
                    return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
                if sq >= 0:
                    if sp < 0:
                        out.append(cut())
                    out.append(q)
                elif sp >= 0:
                    out.append(cut())
            if not out:
                return []
        return out

    # ---- absorbing layers: geometry, mask codes, AUX (09/10/2026, NotesCLAUDE/absorbing_MIRROR.md) ------------
    def _sdf_out(self, X, Y):
        """The distance (physical cells) from points (X, Y) OUTSIDE the drawn shape to its outline (<= 0 or
        meaningless inside): exact for every masked shape (circle; the square; a regular n-gon folded into one
        sector; the rhombus folded into one quadrant; the plate's rounded rectangle). The open edge's sponge ramps
        with it, so the damping is smooth across the staircase."""
        p = self.p
        R = self.dom_R()
        if p.shape == 'circle':
            return np.hypot(X, Y) - R
        if p.shape == 'square':
            return np.hypot(np.maximum(np.abs(X) - R, 0.0), np.maximum(np.abs(Y) - R, 0.0))
        if p.shape == 'unlicensed':
            hw, hh, rc = self.plate_dims()
            co, si = math.cos(p.rot), math.sin(p.rot)
            qx = np.abs(X * co + Y * si) - (hw - rc)
            qy = np.abs(-X * si + Y * co) - (hh - rc)
            return (np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) + np.minimum(np.maximum(qx, qy), 0.0) - rc)
        if p.shape == 'rhomb':
            co, si = math.cos(p.rot), math.sin(p.rot)
            ax = np.abs(X * co + Y * si)
            ay = np.abs(-X * si + Y * co)
            B = max(R * math.tan(p.rtheta / 2), 1e-9)
            ex, ey = -R, B                                  # the edge from (R, 0) to (0, B)
            t = np.clip(((ax - R) * ex + ay * ey) / (ex * ex + ey * ey), 0.0, 1.0)
            return np.hypot(ax - (R + t * ex), ay - t * ey)
        nn = int(p.shape[1:])
        m = TAU / nn
        r = np.hypot(X, Y)
        th = np.arctan2(Y, X) - p.rot
        a = np.fmod(np.fmod(th + PI / nn, m) + m, m) - PI / nn
        xf, yf = r * np.cos(a), r * np.sin(a)
        ap, he = R * math.cos(PI / nn), R * math.sin(PI / nn)
        return np.hypot(np.maximum(xf - ap, 0.0), np.maximum(np.abs(yf) - he, 0.0))

    def _shape_Rc(self):
        """The radius of a disk about the centre that holds the drawn shape (the open edge's band test; 10/10/2026)."""
        p, R = self.p, self.dom_R()
        if p.shape == 'square':
            return R * math.sqrt(2.0)
        if p.shape == 'rhomb':
            return max(R, R * math.tan(p.rtheta / 2))
        if p.shape == 'unlicensed':
            hw, hh, rc = self.plate_dims()
            return math.hypot(hw, hh)
        return R                                    # the circle; a regular n-gon's circumradius

    def _layer_rows(self, i0, i1, inside, lay):
        """Rows [i0, i1) of the layer: (the grown domain, the mask codes). A layer cell is a free stencil cell with
        code 2 + floor(s x SPONGE_LEVELS), s in (0, 1] its depth into the layer (0 at the open domain, 1 at the outer
        end); every other cell has code 0. Frame (the full array): s = (L - depth) / L with depth the physical
        distance to the array ring (sq: the index depth; tri: sqrt(3)/2 x the index depth); the ring itself is the
        clamped outer end. Edge (masked, open): s = the distance outside the drawn outline / L, the cells with s < 1
        (and off the array ring) join the domain."""
        N = self.N
        I = np.arange(i0, i1)[:, None]
        J = np.arange(N)[None, :]
        if lay['where'] == 'frame':
            qd = np.broadcast_to(np.minimum(np.minimum(I, J), np.minimum(N - 1 - I, N - 1 - J)), inside.shape)
            dep = qd * (S3H if self.tri() else 1.0)
            L = float(lay['L'])
            lm = inside & (dep < L - 1e-9)
            if lay.get('exact'):
                # a frame of at most SPONGE_LEVELS rows: one code per row (code 2 + its index depth), so the table
                # holds each row's EXACT damping (_aux_tables)
                code = np.zeros(inside.shape, dtype=np.uint8)
                code[lm] = (2 + qd[lm]).astype(np.uint8)
                return inside, code
            s = (L - dep) / L
            grown = inside
        else:
            X, Y = self._grid_xy(i0, i1)
            L = float(lay['L'])
            ring = np.broadcast_to((I == 0) | (I == N - 1) | (J == 0) | (J == N - 1), inside.shape)
            # 10/10/2026: the distance only where it can be < L. Every shape lies inside the disk of radius
            # _shape_Rc(), so a cell at radius r is at least r - Rc from it: the n-gon's trigonometry then runs on
            # that band, not on the whole grid (a tri hexagon's reseed at N 8193: 11.7 s before; the codes are the same)
            Rc = self._shape_Rc()
            cand = ~inside & ~ring & (X * X + Y * Y < (Rc + L + 1.0) ** 2)
            sd = np.full(inside.shape, np.inf)
            if cand.any():
                sd[cand] = self._sdf_out(X[cand], Y[cand])
            lm = cand & (sd < L)
            s = sd / L
            grown = inside | lm
        code = np.zeros(inside.shape, dtype=np.uint8)
        q = np.clip(np.floor(np.clip(s, 0.0, 1.0) * SPONGE_LEVELS), 0, SPONGE_LEVELS - 1)
        code[lm] = (2 + q[lm]).astype(np.uint8)
        return grown, code

    def open_domain_np(self):
        """uint8 [i, j]: the OPEN domain (the drawn shape, or the array minus an absorbing frame), without the layer;
        domain_mask_np itself (the same object) without a layer. What the alignment panel analyses."""
        if self.abs_np is None:
            return self.domain_mask_np
        key = ('dom', id(self.domain_mask_np), id(self.abs_np))
        oc = getattr(self, '_open_dom_cache', None)
        if oc is None or oc[0] != key:
            oc = (key, self.domain_mask_np & (self.abs_np < 2).view(np.uint8))
            self._open_dom_cache = oc
        return oc[1]

    def layer_mask(self):
        """bool [i, j]: the absorbing layer's cells (free stencil cells outside the open domain; the UI tints them),
        or None without a layer."""
        return None if self.abs_np is None else (self.abs_np >= 2)

    def open_mask(self):
        """mask_np without the layer's cells: the free cells of the OPEN domain (what sources, vertices and the beam
        aperture may use). mask_np itself without a layer. Cached on the mask version."""
        if self.abs_np is None:
            return self.mask_np
        key = (self._mask_ver, id(self.mask_np), id(self.abs_np))
        oc = self._open_cache
        if oc is None or oc[0] != key:
            oc = (key, self.mask_np & (self.abs_np < 2).view(np.uint8))
            self._open_cache = oc
        return oc[1]

    def _dev_mask(self, i0=None, i1=None, j0=None, j1=None, patch=None):
        """The device mask M for the whole grid (or the box, with patch = its 0/1 mask): mask_np with the layer's
        codes (>= 2) on its free layer cells. Without a layer: mask_np (patch) itself, unchanged."""
        if patch is None:
            m = self.mask_np
            if self.abs_np is not None:
                m = np.where((self.abs_np >= 2) & (m != 0), self.abs_np, m)
            return m.astype(mask_np_dtype())
        if self.abs_np is None:
            return patch
        a = self.abs_np[i0:i1, j0:j1]
        return np.ascontiguousarray(np.where((a >= 2) & (patch != 0), a, patch), dtype=mask_np_dtype())

    def _upload_mask(self):
        self.M.from_numpy(self._dev_mask())

    def _aux_tables(self, lay):
        """The host part of AUX (header, the sponge tables, the CPML coefficient tables), float64."""
        Lc = int(lay['Lc']) if lay['kind'] == 'cpml' else 0
        T = np.zeros(AUX_TAB + 12 * Lc)
        T[AUX_INV:AUX_INV + 256] = 1.0
        T[AUX_C2:AUX_C2 + 256] = 1.0
        if lay['kind'] == 'sponge':
            m = np.arange(2, 256, dtype=np.float64)
            if lay.get('exact'):
                # code 2 + q = the frame row at index depth q: d = d_max ((L - q h) / L)^p, h = 1 (sq), sqrt3/2 (tri)
                dep = (m - 2) * (S3H if self.tri() else 1.0)
                d = lay['dmax'] * np.clip((lay['L'] - dep) / lay['L'], 0.0, 1.0) ** lay['p']
            else:
                d = lay['dmax'] * ((m - 1.5) / SPONGE_LEVELS) ** lay['p']
            inv = 1.0 / (1.0 + 0.5 * d)
            T[AUX_INV + 2:AUX_INV + 256] = inv
            T[AUX_C2 + 2:AUX_C2 + 256] = (1.0 - 0.5 * d) * inv
        else:
            T[0] = Lc
            L2 = 2 * Lc
            dt = float(self.p.Co)
            s = np.arange(L2)
            low = s < Lc
            dd = np.where(low, s + 1, L2 - s).astype(np.float64)       # the cell's depth from its own ring
            # the -x face is the outer one on the low side, the inner one on the high side; +x the opposite
            dm = np.where(low, dd - 0.5, dd + 0.5)
            dp = np.where(low, dd + 0.5, dd - 0.5)

            def ba(depth):
                sn = np.clip((Lc - depth) / Lc, 0.0, 1.0)
                sig = lay['smax'] * sn ** lay['p']
                b = np.exp(-(sig + lay['alpha']) * dt)
                a = np.where(sig > 0, sig / np.maximum(sig + lay['alpha'], 1e-300) * (b - 1.0), 0.0)
                return b, a
            for k, depth in enumerate((dd, dm, dp)):
                b, a = ba(depth)
                T[AUX_TAB + (2 * k) * L2:AUX_TAB + (2 * k + 1) * L2] = b
                T[AUX_TAB + (2 * k + 1) * L2:AUX_TAB + (2 * k + 2) * L2] = a
        return T

    def _aux_sync(self, zero_memory=False):
        """Keep AUX in step with the layer: (re)allocate it when the layer's kind, N or frame width change (the CPML
        memory starts at 0), rewrite the tables when their inputs change (Co, d_max ...: the CPML memory is KEPT, as
        set_co keeps the field), and zero the memory on request (seed). Returns AUX or None."""
        lay = self.abs_layer()
        if lay is None:
            self.aux, self._aux_key, self._aux_tkey = None, None, None
            return None
        Lc = int(lay['Lc']) if lay['kind'] == 'cpml' else 0
        nine = (not self.tri()) and self.p.stencil == 9
        # the CPML memory: x strips [3][2 Lc][N]; y strips [3][2 Lc][N], on sq9 a ping-pong pair of them
        key = (self.N, lay['kind'], Lc, nine and Lc > 0)
        tkey = (key, float(self.p.Co), self._layer_key(), bool(FP64))
        if self.aux is None or self._aux_key != key:
            self.aux = ti.ndarray(dtype=float, shape=(AUX_TAB + 12 * Lc + (18 if nine else 12) * Lc * self.N,))
            self.aux.fill(0)
            self._aux_key, self._aux_tkey = key, None
            self._cpml_par = 0
        elif zero_memory and Lc:
            k_aux_fill(self.aux, AUX_TAB + 12 * Lc, 0.0)
            self._cpml_par = 0
        if self._aux_tkey != tkey:
            T = self._aux_tables(lay)
            Td = ti.ndarray(dtype=float, shape=(len(T),))
            Td.from_numpy(T.astype(self.fdt))
            k_aux_put(self.aux, Td)
            self._aux_tkey = tkey
        return self.aux

    # ---- allocation, mask and per-cell Courant numbers ----------------------------------
    def _alloc(self, N):
        if N % 2 == 0:
            N += 1                                    # odd: a true centre cell
        self.N = N
        self.U = ti.ndarray(dtype=float, shape=(3, N, N))
        self.M = ti.ndarray(dtype=MASK_DT, shape=(N, N))
        self.CC = ti.ndarray(dtype=float, shape=(N, N))

    def _ring_key(self):
        """What the domain mask's ring depends on beyond the geometry: the triangular 'array' holds its
        ring only on the torus (05/10/2026). None for every other shape."""
        return self.tri_torus() if (self.tri() and self.p.shape == 'array') else None

    def _grid_xy(self, i0=0, i1=None, j0=0, j1=None):
        """Physical offsets of the cells in the index window [i0,i1) x [j0,j1), as (x, y)
        arrays indexed [i, j] -- the web app's cellXY, vectorised."""
        c = self.c()
        i1 = self.N if i1 is None else i1
        j1 = self.N if j1 is None else j1
        I, J = np.meshgrid(np.arange(i0, i1, dtype=np.float64), np.arange(j0, j1, dtype=np.float64),
                           indexing='ij')
        da, db = I - c, J - c
        if self.tri():
            return da + db * 0.5, db * S3H
        return da, db

    def build_domain(self):
        """Mask + per-cell Co^2, in row chunks: at N = 8193 a whole-grid pass held ~3 GB of
        float64 temporaries on the host; a chunk of 512 rows holds a few hundred MB.
        The solids (if any) are composed in: mask_np = domain_mask_np AND NOT solid_np."""
        self._mur2_ok = False                       # 10/10/2026: Mur2 restarts with a Mur1 step (step())
        N = self.N
        dom = np.empty((N, N), dtype=np.uint8)
        self.cc_np = np.empty((N, N), dtype=np.float64)
        lay = self.abs_layer()                      # 09/10/2026: an absorbing layer's cells and codes
        code = np.zeros((N, N), dtype=np.uint8) if lay is not None else None
        CH = max(1, (1 << 22) // N)
        for i0 in range(0, N, CH):
            i1 = min(N, i0 + CH)
            ins, cc = self._domain_rows(i0, i1)
            if lay is not None:
                ins, code[i0:i1] = self._layer_rows(i0, i1, ins, lay)
            dom[i0:i1] = ins
            self.cc_np[i0:i1] = cc
        self.domain_mask_np = dom
        self.abs_np = code
        self._open_cache = None
        self._dom_layer_key = self._layer_key()
        # the shape reaches the array ring (a shape clipped by the array): _neu_clip() (09/10/2026)
        self._dom_on_ring = bool(dom[0, :].any() or dom[-1, :].any() or dom[:, 0].any() or dom[:, -1].any())
        self._dom_ring_key = self._ring_key()
        if self.scene is None and self.p.shape == 'unlicensed':
            self.ensure_scene()                     # the plate's lettering is a scene preset
        self._rasterize_solids()
        self._compose()
        self._upload_mask()
        self.CC.from_numpy(self.cc_np.astype(self.fdt))
        self._walls_full()
        self._aux_sync()

    # ---- solids (solids_MIRROR.py supplies the scene; the core only composes) -------------
    def ensure_scene(self):
        """The attached SolidScene, creating an empty one (imported lazily: solids_MIRROR
        imports this module) if there is none. Does not rebuild anything."""
        if self.scene is None:
            here = os.path.dirname(os.path.abspath(__file__))
            if here not in sys.path:
                sys.path.insert(0, here)
            import solids_MIRROR
            self.scene = solids_MIRROR.SolidScene()
            self.scene.m = self
        return self.scene

    def attach_scene(self, scene):
        """Attach (or with None, detach) a SolidScene and apply it at once (update_solids)."""
        self.scene = scene
        if scene is not None:
            scene.m = self
        self._solid_key = None
        self.update_solids()

    def _solid_cache_key(self):
        p = self.p
        sig = self.scene.signature() if self.scene is not None else None
        return (id(self.scene), sig, self.N, p.lattice, p.shape, p.rot, p.rad, p.rtheta)

    def _rasterize_solids(self, force=False):
        """solid_np for the whole grid (None if the scene makes no solid cell). Cached on the
        geometry + scene signature, so set_co (which rebuilds the domain) does not re-render.
        solid_np is the TYPE map (04/10/2026): 0 free, 1 pinned, 2 rigid (uint8 [i, j])."""
        key = self._solid_cache_key()
        if not force and key == self._solid_key and (self.solid_np is None or self.solid_np.shape[0] == self.N):
            return
        self.solid_np = self.scene.rasterize_full(self) if self.scene is not None else None
        self._solid_key = key

    def _compose(self):
        self._mask_ver += 1
        if self.solid_np is not None and not self.solid_np.any():
            self.solid_np = None                        # nothing solid: the web app's exact path
        if self.solid_np is None:
            self.mask_np = self.domain_mask_np          # the same object: no solids, no copy
            self.has_solids = False
            self.n_solid = 0
        else:
            # solid_np is the type map (0 free, 1 pinned, 2 rigid): every non-zero type is solid
            self.mask_np = self.domain_mask_np & (self.solid_np == 0).view(np.uint8)
            self.has_solids = True
            self.n_solid = int(np.count_nonzero(self.solid_np))

    def update_solids(self, bbox=None):
        """Apply the scene's solids to the running simulation WITHOUT a reset.

        bbox = (i0, i1, j0, j1), a half-open cell box (what SolidScene.begin_stroke /
        extend_stroke / add / edit / remove return): only that box is re-rasterised (from the
        whole scene: objects, then every stroke in order), recomposed, uploaded (ONE kernel with
        a patch of M, which also zeroes the three time levels there) -- measured 30/09/2026:
        ~3 ms per painted segment at N = 2049 and at N = 8193. The box must cover every cell
        the edit can change; an empty box does nothing. bbox = None redoes the whole grid
        (N = 8193 with the Unlicensed lettering: ~3 s). Either way every time level is zeroed
        where the new mask is 0, the sources are re-placed (they snap OUT of solids) and the
        probes re-cached; the field elsewhere and the clock are untouched. When the last solid
        cell goes, the solver returns to the web app's exact path (has_solids False)."""
        self._mur2_ok = False                       # 10/10/2026: Mur2 restarts with a Mur1 step (step())
        if bbox is None:
            self._rasterize_solids(force=True)
            self._compose()
            self._upload_mask()
            k_zero_masked(self.U, self.M)
            self._walls_full()
        else:
            N = self.N
            i0, i1, j0, j1 = (max(0, int(bbox[0])), min(N, int(bbox[1])),
                              max(0, int(bbox[2])), min(N, int(bbox[3])))
            if i1 <= i0 or j1 <= j0:
                return
            reg = (self.scene.rasterize(self, i0, i1, j0, j1) if self.scene is not None
                   else np.zeros((i1 - i0, j1 - j0), dtype=np.uint8))
            if self.solid_np is None:
                if not reg.any():
                    self._solid_key = self._solid_cache_key()
                    return
                self.solid_np = np.zeros((N, N), dtype=np.uint8)
                self.n_solid = 0
            old_key = self._main_key()
            self._mask_ver += 1
            self.n_solid += int(np.count_nonzero(reg)) - int(np.count_nonzero(self.solid_np[i0:i1, j0:j1]))
            self.solid_np[i0:i1, j0:j1] = reg
            patch = self.domain_mask_np[i0:i1, j0:j1] & (reg == 0).view(np.uint8)
            if self.n_solid == 0:
                self.solid_np = None
                self.mask_np = self.domain_mask_np
                self.has_solids = False
            else:
                if self.mask_np is self.domain_mask_np:
                    self.mask_np = self.domain_mask_np.copy()
                self.mask_np[i0:i1, j0:j1] = patch
                self.has_solids = True
            self._solid_key = self._solid_cache_key()
            # the device mask patch (and the zeroed levels), plus the walls' classes and words when
            # they are kept: one launch either way
            self._walls_patch(i0, i1, j0, j1, np.ascontiguousarray(patch, dtype=mask_np_dtype()))
            mc_ = self._main_cache
            if self.has_solids and mc_ is not None and mc_[0] == old_key and mc_[1] is not None:
                # INCREMENTAL main region (a stroke segment costs O(box), not a relabel of the
                # grid: ~0.7 s at N = 8193): the box's newly solid cells leave it; cells an eraser
                # frees and a stroke that SEALS a region off are not seen until the next exact
                # relabel (any recache_source outside a stroke, e.g. a source move or reset).
                main = mc_[1]
                main[i0:i1, j0:j1] &= patch.astype(bool)
                self._main_cache = (self._main_key(), main, False)
            else:
                self._main_cache = None
            self.recache_source(approx_ok=True)
            return
        self.recache_source()

    def set_solid_np(self, solid):
        """A raw solid map applied like update_solids(None): bool / uint8 [i, j], 0 free, 1 (or
        any non-zero value other than 2, or True) pinned, 2 rigid. For tests and bitmaps. It
        survives reseed() and set_co(); a new N, lattice, shape, rotation or radius, or any
        scene edit, replaces it with the scene's rasterisation."""
        self._mur2_ok = False                       # 10/10/2026: Mur2 restarts with a Mur1 step (step())
        if solid is None:
            self.solid_np = None
        else:
            a = np.asarray(solid)
            self.solid_np = np.where(a == 2, 2, a != 0).astype(np.uint8) if a.dtype != bool else a.astype(np.uint8)
        self._solid_key = self._solid_cache_key()
        self._compose()
        self._upload_mask()
        k_zero_masked(self.U, self.M)
        self._walls_full()
        self.recache_source()

    # ---- walls: classes, link words, the edge (04/10/2026, NotesCLAUDE/walls_MIRROR.md) -------
    def walls_needed(self):
        """The class codes and link words live on the device while a solid exists or the domain
        mask's edge is rigid (edge = 'rigid' on a masked domain). Otherwise they are freed and the
        solver is exactly the web app's."""
        if self.p.edge not in EDGES:
            raise ValueError(f'edge {self.p.edge!r}: one of {EDGES}')
        return self.has_solids or self._erig() or self._neu_clip()

    def _neu_clip(self):
        """The sq9 stencil, the Neumann bc and a masked shape that reaches the array ring (clipped by the
        array; 09/10/2026, the user's decision). The web app's ring is asymmetric there (an exterior ring cell
        holds 0 next to a ghost: walls_MIRROR.md 14.2) and the run grows without bound; the port blocks the
        sq9 diagonals into the ring beside an exterior ring cell (_word's ghost rule), so the link words must
        exist. They stay idle (the web app's path, bit for bit) where nothing is blocked."""
        p = self.p
        return (not self.tri() and p.stencil == 9 and p.bc == 'neumann' and not self.full_array()
                and bool(getattr(self, '_dom_on_ring', False)))

    def _erig(self):
        """Do the domain-exterior cells (class X) act RIGID? A masked domain: edge = 'rigid'. The full
        array has no mask edge, so Params.edge does not apply there: on the square lattice there is
        no X cell at all; on the triangular 'array' the X cells are its ring, rigid for bc = neumann
        (05/10/2026: the rigid ring by link removal)."""
        if self.full_array():
            return self.tri() and self.bc_effective() == 'neumann'
        return self.p.edge == 'rigid'

    def walls_active(self):
        """True while the step runs the walls kernel: some free cell has a dropped or a
        pinned-blocked link. False = the web app's exact path -- also with pinned solids that
        block no link (a pinned neighbour already holds 0)."""
        return self.W is not None and self.n_wall_cells > 0

    def _walls_key(self):
        """What the words depend on besides the cells: (N, tri, sq9, torus, Neumann ring, rigid
        exterior). torus: the sq torus or the triangular 'array' torus (05/10/2026); rigid exterior:
        _erig() (edge = 'rigid' on a masked domain, or the triangular array's rigid ring)."""
        p, tri = self.p, self.tri()
        return (self.N, tri, (not tri) and p.stencil == 9, self.torus(),
                (not tri) and p.bc == 'neumann', self._erig(), self._cpml9_Lc())

    def _cpml9_Lc(self):
        """The sq9 CPML frame's width Lc, else 0 (09/10/2026). Inside that frame (index depth <= Lc + 1) every solid
        acts PINNED with no corner blocking (_class_box maps P / R there to X, which with the full array's pinned
        exterior never blocks among itself): the stretched 9-point operator splits each diagonal link between the
        x and y memories, which link removal and blocked diagonals cannot follow -- a rigid block in the frame grew
        to max|u| 3.6e6 in 6000 steps (ScratchCLAUDE/Round5CLAUDE/AbsorbCLAUDE/dbg_9b_MIRROR.py), pinned ones decay.
        The sq5 CPML keeps rigid solids (its links are axis links: a dropped one carries no flux)."""
        if self.tri() or self.p.stencil != 9 or not self.full_array() or self.bc_effective() != 'cpml':
            return 0
        return int(CPML_LEVELS[self.p.cpml_level]['L'])

    def _cls_raw(self, M, sel):
        """Classes (CL_*) of the cells sel (a box slice or an index pair) whose masks are M (bool):
        free -> F; a solid inside the domain -> its type (P 1 / R 2); else (exterior) -> X."""
        cls = np.full(M.shape, CL_X, dtype=np.uint8)
        cls[M] = CL_F
        if self.solid_np is not None:
            T = self.solid_np[sel]
            sol = ~M & (self.domain_mask_np[sel] != 0) & (T != 0)
            cls[sol] = T[sol]
        return cls

    def _class_box(self, i0, i1, j0, j1):
        """The class codes (CL_* | CL_FREE if the cell's own mask is 1) of the half-open box, from
        mask_np, domain_mask_np and solid_np. Ring rules: the triangular array's ring is clamped, so
        X; a Neumann ring cell copies its partner (a side's interior neighbour, a corner's diagonal
        one), so behind a solid or the exterior it takes that cell's class (the mirror image of a
        rigid cell is rigid)."""
        N = self.N
        sel = (slice(i0, i1), slice(j0, j1))
        # mask = domain AND type == 0, so in one go: inside the domain the code is the type (P 1,
        # R 2) or F | CL_FREE (4) for a free cell; outside it is X (3), not free
        D = self.domain_mask_np[sel] != 0
        if self.solid_np is None:
            code = np.where(D, np.uint8(CL_F | CL_FREE), np.uint8(CL_X))
        else:
            T = self.solid_np[sel]
            code = T + (T == 0).view(np.uint8) * np.uint8(CL_F | CL_FREE)
            code[~D] = CL_X
        key = self._walls_key()
        Lc9 = key[6]
        if Lc9:
            # 09/10/2026: the sq9 CPML frame (and one row beyond): solids act pinned, never block (X)
            I = np.arange(i0, i1)[:, None]
            J = np.arange(j0, j1)[None, :]
            dep = np.minimum(np.minimum(I, J), np.minimum(N - 1 - I, N - 1 - J))
            fr = np.broadcast_to(dep <= Lc9 + 1, code.shape) & ((code & 3) != CL_F)
            code[fr] = (code[fr] & CL_FREE) | CL_X
        tri = key[1] and not key[3]                 # the clamped triangular ring (not on the rhombic torus)
        neu = key[4]
        if (tri or neu) and (i0 == 0 or j0 == 0 or i1 == N or j1 == N):
            I = np.arange(i0, i1)[:, None]
            J = np.arange(j0, j1)[None, :]
            ring = np.broadcast_to((I == 0) | (I == N - 1) | (J == 0) | (J == N - 1), code.shape) & ((code & 3) == CL_F)
            if ring.any():
                if tri:
                    code[ring] = (code[ring] & CL_FREE) | CL_X
                else:
                    ri, rj = np.nonzero(ring)
                    pi, pj = np.clip(ri + i0, 1, N - 2), np.clip(rj + j0, 1, N - 2)
                    code[ri, rj] = (code[ri, rj] & CL_FREE) | self._cls_raw(self.mask_np[pi, pj] != 0, (pi, pj))
        return code

    def _walls_read(self):
        c = self.wcnt.to_numpy()
        self._wn = (int(c[0]), int(c[1]))
        self._wdirty = False

    @property
    def n_wall_cells(self):
        """Free cells with a dropped or pinned-blocked link (0: the web app's path). Read from the
        device lazily: a painted segment only marks it dirty, so painting costs no extra sync."""
        if self._wdirty:
            self._walls_read()
        return self._wn[0]

    @property
    def n_blk4(self):
        """Triangular cells with a blocked link 0-3 (main_region's exact refined path when > 0)."""
        if self._wdirty:
            self._walls_read()
        return self._wn[1]

    def _walls_free(self):
        self.C = self.W = None
        self._wn = (0, 0)
        self._wdirty = False
        self._wkey = None

    def _walls_full(self):
        """(Re)build every class code and link word (one host pass for the classes, in row chunks;
        ONE Taichi pass for the words), or free them when no longer needed."""
        if not self.walls_needed():
            self._walls_free()
            return
        N = self.N
        if self.C is None or self.C.shape[0] != N:
            self.C = ti.ndarray(dtype=MASK_DT, shape=(N, N))
            self.W = ti.ndarray(dtype=WORD_DT, shape=(N, N))
        code = np.empty((N, N), dtype=mask_np_dtype())
        CH = max(1, (1 << 22) // N)
        for i0 in range(0, N, CH):
            i1 = min(N, i0 + CH)
            code[i0:i1] = self._class_box(i0, i1, 0, N)
        self.C.from_numpy(code)
        key = self._walls_key()
        k_walls_full(self.C, self.W, self.wcnt, LAT_TRI if key[1] else LAT_SQ, key[2], key[5], key[3], key[4])
        self._wkey = key
        self._walls_read()

    def _walls_patch(self, i0, i1, j0, j1, patch):
        """After a painted box [i0, i1) x [j0, j1) changed (patch = its new mask, host): the device
        mask (and zeroed levels), the classes of the box + 1 (a Neumann ring cell follows its
        partner) and the words of the box + 2 (a word reads its neighbours and their common
        neighbours, all within one cell) -- ONE kernel launch and ONE upload, as k_patch_mask alone
        did before; the counters are read lazily. Never a whole-grid pass (except the first time
        the words are needed, or after the lattice / stencil / bc / edge changed)."""
        dpatch = self._dev_mask(i0, i1, j0, j1, patch)     # 09/10/2026: the layer's codes (patch itself without)
        if not self.walls_needed():
            self._walls_free()
            k_patch_mask(self.U, self.M, dpatch, i0, j0)
            return
        if self.W is None or self._wkey != self._walls_key():
            k_patch_mask(self.U, self.M, dpatch, i0, j0)
            self._walls_full()
            return
        N, key = self.N, self._wkey
        ci0, ci1, cj0, cj1 = max(0, i0 - 1), min(N, i1 + 1), max(0, j0 - 1), min(N, j1 + 1)
        Q = np.ascontiguousarray(self._class_box(ci0, ci1, cj0, cj1), dtype=mask_np_dtype())
        wi0, wi1, wj0, wj1 = ci0 - 1, ci1 + 1, cj0 - 1, cj1 + 1
        if key[3]:                                  # a torus: the box wraps, at most N per axis
            if wi1 - wi0 > N:
                wi0, wi1 = 0, N
            if wj1 - wj0 > N:
                wj0, wj1 = 0, N
        else:
            wi0, wi1, wj0, wj1 = max(0, wi0), min(N, wi1), max(0, wj0), min(N, wj1)
        k_walls_patch(self.U, self.M, self.C, self.W, Q, self.wcnt, ci0, cj0, wi0, wj0, wi1 - wi0, wj1 - wj0,
                      LAT_TRI if key[1] else LAT_SQ, key[2], key[5], key[3], key[4])
        if self.abs_np is not None:
            # k_walls_patch wrote M = the free bit over the class box: put the layer's codes back there
            k_patch_mask(self.U, self.M, self._dev_mask(ci0, ci1, cj0, cj1, (self.mask_np[ci0:ci1, cj0:cj1] != 0)
                                                        .astype(mask_np_dtype())), ci0, cj0)
        self._wdirty = True

    def _walls_sync(self):
        """Before a step: the words must match the lattice / stencil / bc / edge they were built for."""
        if self.walls_needed():
            if self.W is None or self._wkey != self._walls_key():
                self._walls_full()
        elif self.W is not None:
            self._walls_free()

    def set_edge(self, edge):
        """The domain mask's edge, 'pinned' (the web app) or 'rigid', switched WITHOUT a reset: the
        link words are rebuilt and the field is kept (no cell changes between free and solid)."""
        if edge not in EDGES:
            raise ValueError(f'edge {edge!r}: one of {EDGES}')
        self.p.edge = edge
        if self._dom_layer_key != self._layer_key():
            # 09/10/2026: to / from 'open' on a masked shape: the domain itself changes (the layer outside the
            # edge, the radius cap) -- rebuilt, the field kept and zeroed on the cells that leave the domain
            self.build_domain()
            k_zero_masked(self.U, self.M)
            self.recache_source()
            return
        self._mask_ver += 1                         # the main region's links can change
        self._walls_full()
        self.recache_source()

    def walls_info(self):
        """For the panel: the edge, whether the walls kernel runs, the counts (pinned / rigid
        solid cells inside the domain: two whole-grid counts, ~0.1 s at N = 8193 -- cache it)."""
        sol, dom = self.solid_np, self.domain_mask_np != 0
        npin = int(np.count_nonzero((sol == 1) & dom)) if sol is not None else 0
        nrig = int(np.count_nonzero((sol == 2) & dom)) if sol is not None else 0
        # neu_clip (09/10/2026): the sq9 Neumann ring with a clipped shape, where the port blocks the diagonals
        # beside an exterior ring cell (the walls kernel can run with no solid at all)
        return dict(edge=self.p.edge, active=self.walls_active(), wall_cells=self.n_wall_cells,
                    pinned_cells=npin, rigid_cells=nrig, blocked_tri4=self.n_blk4,
                    edge_applies=not self.full_array(), ring=self.bc_info()['ring'], neu_clip=self._neu_clip())

    def unknowns(self):
        """bool [i, j]: the cells the stencil updates (free; inside the ring unless on a torus)."""
        unk = self.mask_np != 0
        if not self.torus():
            unk = unk.copy()
            unk[0, :] = unk[-1, :] = unk[:, 0] = unk[:, -1] = False
        return unk

    def operator(self):
        """The solver's spatial operator L (u+ = 2u - u- + CC (L u) - d (u - u-)) as a scipy CSR
        matrix over the whole grid, flat index i * N + j, built from the DEVICE link words (or the
        mask on the web app's path), so it is what the kernel applies. Rows exist for the unknowns
        only. Ring cells are resolved: Dirichlet and the triangular clamp read 0 (a rigid triangular
        ring is dropped by the words), a Neumann ring cell reads its partner, a torus wraps (the
        rhombic one too); Mur's ring is not a function of the current level (ValueError). Host
        numpy, for tests (N up to ~1000)."""
        import scipy.sparse as sp
        N, p, tri = self.N, self.p, self.tri()
        nine = (not tri) and p.stencil == 9
        wrap = self.torus()
        neu = (not tri) and p.bc == 'neumann'
        mur = (not tri) and self.bc_effective() in ('mur', 'mur2')
        if (not tri) and self.bc_effective() == 'cpml':
            raise ValueError('operator(): the CPML frame has memory (its update is not a function of the current level)')
        lat = LAT_TRI if tri else LAT_SQ
        links = lat_links(lat, nine)
        wts = [2.0 / 3.0] * 6 if tri else ([2.0 / 3.0] * 4 + [1.0 / 6.0] * 4 if nine else [1.0] * 4)
        M = self.mask_np != 0
        self._walls_sync()
        Wh = self.W.to_numpy().astype(np.int64) if self.walls_active() else M.astype(np.int64)
        I, J = np.nonzero(self.unknowns())
        w = Wh[I, J]
        me = I * N + J
        unk_flat = self.unknowns().ravel()
        rows, cols, vals = [], [], []
        diag = np.zeros(N * N)
        for k, (di, dj) in enumerate(links):
            wk = wts[k]
            sb = (w >> w_sbit(k)) & 1
            zi = w_zbit(lat, nine, k)
            zb = ((w >> zi) & 1) if zi >= 0 else np.zeros_like(w)
            diag[me[zb == 1]] -= wk                     # pinned-blocked: a link to a zero
            act = (sb == 0) & (zb == 0)
            src = me[act]
            ni, nj = I[act] + di, J[act] + dj
            diag[src] -= wk
            if wrap:
                ni, nj = ni % N, nj % N
                tgt = np.where(M[ni, nj], ni * N + nj, -1)
            else:
                ring = (ni == 0) | (nj == 0) | (ni == N - 1) | (nj == N - 1)
                tgt = np.where(M[ni, nj], ni * N + nj, -1)
                if tri or self.bc_effective() in ('dirichlet', 'sponge'):
                    tgt[ring] = -1
                elif mur:
                    if (ring & M[ni, nj]).any():
                        raise ValueError('operator(): a Mur ring cell is not a function of the current level')
                elif neu:
                    r = ring & M[ni, nj]
                    pi, pj = np.clip(ni[r], 1, N - 2), np.clip(nj[r], 1, N - 2)
                    pt = pi * N + pj
                    tgt[r] = np.where(unk_flat[pt], pt, -1)
            ok_ = tgt >= 0
            rows.append(src[ok_])
            cols.append(tgt[ok_])
            vals.append(np.full(int(ok_.sum()), wk))
        L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N * N, N * N))
        L = (L + sp.diags(diag)).tocsr()
        L.eliminate_zeros()
        return L

    def leapfrog_energy(self, L=None):
        """The exact discrete energy of the undamped leapfrog for the current operator:
            E^{n+1/2} = sum_unknowns (u^{n+1} - u^n)_i^2 / CC_i  -  <u^{n+1}, L u^n>.
        Derivation: u+ - 2u + u- = CC_i (L u)_i; multiply by (u+ - u-)_i / CC_i (the per-cell
        weight 1/CC_i is what makes it work in a medium) and sum: the left side telescopes to
        sum (u+ - u)^2/CC - sum (u - u-)^2/CC, the right is <u+, L u> - <u-, L u>, and L = L^T
        turns <u-, L u> into <u, L u->. So E is the same at every step (drives and damping
        excepted). A norm only inside the CFL limit. u^n = level(0), u^{n+1} = level(1)."""
        if L is None:
            L = self.operator()
        u0 = self.level(0).astype(np.float64).ravel()
        u1 = self.level(1).astype(np.float64).ravel()
        unk = self.unknowns().ravel()
        du = (u1 - u0)[unk]
        return float((du * du / self.cc_np.ravel()[unk]).sum() - u1 @ (L @ u0))

    def _domain_rows(self, i0, i1):
        p = self.p
        R, ir = self.dom_R(), self.inner_R()
        X, Y = self._grid_xy(i0, i1)
        if p.shape == 'array' and self.tri():
            # the triangular array: every cell but the outer ring (the boundary); on the rhombic torus
            # the ring cells are ordinary cells, so every cell is inside
            inside = np.ones_like(X, dtype=bool)
            if not self.torus():
                N = self.N
                I = np.arange(i0, i1)[:, None]
                J = np.arange(N)[None, :]
                inside &= (I > 0) & (I < N - 1) & (J > 0) & (J < N - 1)
        elif p.shape == 'unlicensed':
            # rounded rectangle in the rotated frame, the same +1e-9 slack as insideNgon
            hw, hh, rc = self.plate_dims()
            co, si = math.cos(p.rot), math.sin(p.rot)
            x = np.abs(X * co + Y * si)
            y = np.abs(-X * si + Y * co)
            ax, ay = x - (hw - rc), y - (hh - rc)
            corner = (ax > 0) & (ay > 0)
            inside = (x <= hw + 1e-9) & (y <= hh + 1e-9) & ~(corner & (np.hypot(ax, ay) > rc + 1e-9))
        elif p.shape == 'circle':
            inside = (X * X + Y * Y) <= R * R
        elif p.shape == 'rhomb':
            co, si = math.cos(p.rot), math.sin(p.rot)
            x = X * co + Y * si
            y = -X * si + Y * co
            B = R * math.tan(p.rtheta / 2)
            inside = np.abs(x) / R + np.abs(y) / max(B, 1e-9) <= 1 + 1e-9
        elif p.shape not in ('square', 'array'):     # ('array' on sq is sq_full: the last branch)
            nn = int(p.shape[1:])
            r = np.hypot(X, Y)
            m = TAU / nn
            th = np.arctan2(Y, X) - p.rot
            a = np.fmod(np.fmod(th + PI / nn, m) + m, m) - PI / nn
            inside = r <= R * math.cos(PI / nn) / np.cos(a) + 1e-9
            inside[r < 1e-12] = True
        elif not self.sq_full():
            inside = np.maximum(np.abs(X), np.abs(Y)) <= R
        else:
            inside = np.ones_like(X, dtype=bool)
        f = np.ones_like(X)
        slabJ, lensJ, lensR = ir * 0.16, ir * 0.19, ir * 0.36
        if p.medium == 'slab':
            f[Y > slabJ] = 0.5
        elif p.medium == 'lens':
            ddy = Y - lensJ
            f[np.sqrt(X * X + ddy * ddy) < lensR] = 0.55
        co = p.Co * f
        return inside, co * co

    def set_co(self, co, rescale=False):
        """Change Co WITHOUT a reset.

        rescale=False is the web app's slider: only the per-cell Courant numbers change, and
        the two stored time levels are kept. The medium's wave speed then jumps at one instant
        -- a TEMPORAL BOUNDARY -- and every wave splits into a forward part and a smaller
        backward ("time-reflected") part, amplitude B = (Co2 - Co1)/(2 Co2) in the long-wave
        limit (README: time reflection).

        rescale=True reads the change as a new TIME STEP for the same medium (Co = c dt/h, so
        at fixed c and h a new Co IS a new dt): the previous time level is re-expressed at the
        new spacing, to second order,
            u_prev <- u - r (u - u_prev) + ((r^2 - r)/2) Co1^2 L(u),   r = Co2/Co1,
        so the velocity AND acceleration carry over and nothing is reflected (measured in 1D:
        the backward wave drops to 0.07-0.26% of the unrescaled one)."""
        self._mur2_ok = False                       # 10/10/2026: Mur2 restarts with a Mur1 step (step())
        r = co / self.p.Co if self.p.Co > 0 else 1.0
        if rescale and abs(r - 1) > 1e-15:
            self._walls_sync()
            if self.walls_active():                     # the walls' operator (04/10/2026)
                key = self._wkey
                k_rescale_w(self.U, self.W, self.CC, self.slot, (self.slot + 1) % 3, r,
                            LAT_TRI if key[1] else LAT_SQ, key[2], key[3])
            else:
                k_rescale(self.U, self.M, self.CC, self.slot, (self.slot + 1) % 3, r, 1 if self.tri() else 0,
                          1 if self.p.stencil == 9 else 0, 1 if self.torus() else 0)
        self.p.Co = co
        self.apply_scale()                               # 'domain': freq = Co / lambda keeps lambda
        self.harm = self.harmonics()
        self.build_domain()

    # ---- source placement ---------------------------------------------------------------
    def round_cell(self, fi, fj):
        c, N = self.c(), self.N
        oi, oj = snap_q(fi - c), snap_q(fj - c)
        if self.tri():
            db = oj
            x = snap_q((fi - c) + (fj - c) * 0.5)
            B = sym_round(0, db)
            A = js_round(x - B * 0.5)
            return max(1, min(N - 2, int(c + A))), max(1, min(N - 2, int(c + B)))
        return (max(1, min(N - 2, int(sym_round(c, oi)))), max(1, min(N - 2, int(sym_round(c, oj)))))

    def snap_inside(self, fi, fj, approx_ok=False):
        """(i, j, moved): the cell a source requested at fractional cell (fi, fj) goes to."""
        return self.snap_source(fi, fj, approx_ok)[:3]

    def snap_source(self, fi, fj, approx_ok=False):
        """(i, j, moved, how, solid_req) for a source requested at fractional cell (fi, fj);
        solid_req = the requested (rounded) cell is a solid cell of the domain -- what
        Membrane.src_moved reports for the source.

        Without solids: the web app exactly -- the rounded cell if it is in the domain, else a
        walk back toward the centre in physical space (how = 'walk'), else the centre ('centre').

        With solids (PORT DEVIATION, the user's decision 01/10/2026):
        - a requested cell that is FREE is kept, even inside a sealed cavity (a click inside the
          D's counter puts the source there on purpose);
        - a requested cell that is SOLID goes to the NEAREST FREE CELL OF THE MAIN REGION
          ('nearest-main'): nearest in lattice hops, ties by physical distance, then (i, j).
          The main region is the largest connected component of free cells under the stencil's
          own links (4 / 8 / 6 neighbours) -- the plate, as opposed to the letters' sealed
          counters (inside of the D, A, O ...). A sealed cavity is never chosen, even when one of
          its cells is strictly nearer: a source there could not reach the plate, and the user
          clicked on the letter, not into its counter. (The web app's walk toward the centre
          could cross a stroke and drop the source into a counter.)
        - a requested cell OUTSIDE the domain keeps the web app's walk; if the walk ends in a
          sealed cavity or finds nothing free, the nearest free cell of the main region instead.
        - fallback if no cell is free at all: the centre (the web app's own fallback).
        approx_ok=True (inside a paint stroke) accepts the incrementally updated main region."""
        i0, j0 = self.round_cell(fi, fj)
        # 09/10/2026: with an absorbing layer only the OPEN domain's cells are allowed; a cell requested in the layer
        # walks back toward the centre like a cell outside the domain (how = 'layer')
        msk = self.open_mask()
        if msk[i0, j0]:
            return (i0, j0, False, 'none', False)
        in_layer = self.abs_np is not None and self.abs_np[i0, j0] >= 2 and bool(self.mask_np[i0, j0])
        solid_req = (self.has_solids and self.solid_np is not None and bool(self.solid_np[i0, j0])
                     and bool(self.domain_mask_np[i0, j0]))
        if not solid_req:
            x, y = self.cell_xy(i0, j0)
            t = 0.96
            while t >= 0:                           # walk back in PHYSICAL space
                ii, jj = self.round_cell(*self.xy_to_cell(x * t, y * t))
                if msk[ii, jj]:
                    if not self.has_solids or self.main_region(approx_ok)[ii, jj]:
                        return (ii, jj, True, 'layer' if in_layer else 'walk', False)
                    break                           # the walk ended in a sealed cavity
                t -= 0.02
        if self.has_solids:
            main = self.main_region(approx_ok)
            if main is not None and self.abs_np is not None:
                main = main & (self.abs_np < 2)
            f = self.nearest_free(i0, j0, main) if main is not None else None
            if f is not None:
                return (f[0], f[1], True, 'solid' if solid_req else 'nearest-main', solid_req)
        c = int(self.c())
        return (c, c, True, 'centre', solid_req)

    def stencil_structure(self):
        """The 3x3 connectivity of the stencil's links (a scipy.ndimage.label structure)."""
        st = np.zeros((3, 3), dtype=np.uint8)
        st[1, 1] = 1
        for di, dj in self.neighbour_offsets():
            st[1 + di, 1 + dj] = 1
        return st

    def _main_key(self):
        return (self._mask_ver, id(self.mask_np), self.N, self.tri(), self.p.stencil)

    def main_region(self, approx_ok=False):
        """bool [i, j]: the free cells (mask 1, inside [1, N-2]^2) of the largest connected
        component under the stencil's links; None if no cell is free. Cached on the mask version
        and the stencil. Cost (scipy.ndimage.label): ~0.7 s at N = 8193 (measured 01/10/2026),
        paid only when a source lands on a solid / outside cell after the mask changed."""
        key = self._main_key()
        mc_ = self._main_cache
        if mc_ is not None and mc_[0] == key and (mc_[2] or approx_ok):
            return mc_[1]
        from scipy import ndimage
        free = np.zeros(self.mask_np.shape, dtype=np.uint8)
        free[1:-1, 1:-1] = self.mask_np[1:-1, 1:-1]
        if self.walls_active():
            main = self._walls_main(free.astype(bool))
        else:
            lab, n = ndimage.label(free, structure=self.stencil_structure())
            if n == 0:
                main = None
            else:
                cnt = np.bincount(lab.ravel())
                cnt[0] = 0
                main = lab == int(np.argmax(cnt))
            del lab
        self._main_cache = (key, main, True)
        return main

    def _walls_main(self, fb):
        """main_region() on the walls path (04/10/2026): the components under the ACTIVE links only
        (corner blocking can seal a counter that the plain 8-/6-neighbour labelling would join).
        A link with a free common neighbour is redundant (the two links through that neighbour are
        active), so: label the 4-neighbour structure, then join the components that an active
        CORNER link (both common neighbours non-free: an unblocked X-X pair) connects -- exact for
        sq5 and sq9 (sq edge links are never blocked) and for tri when no tri link 0-3 is blocked.
        Otherwise (tri, n_blk4 > 0) the exact refined-grid labelling: cells at (2i, 2j), each
        active link (1,0) / (0,1) / (1,-1) as a node between its ends, 8-connected (a link node
        also touches the link's common neighbours, which are connected to both ends whenever they
        are free). ~4x the cells: at N = 8193 ~1.3 GB and several seconds (rare: tri + blocked
        links + a source landing on a solid)."""
        from scipy import ndimage
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components
        N, tri = self.N, self.tri()
        nine = (not tri) and self.p.stencil == 9
        lat = LAT_TRI if tri else LAT_SQ
        Wh = self.W.to_numpy().astype(np.int32)

        def inactive(w, k):
            z = w_zbit(lat, nine, k)
            bits = (1 << w_sbit(k)) | ((1 << z) if z >= 0 else 0)
            return (w & bits) != 0
        if tri and self.n_blk4 > 0:
            R = np.zeros((2 * N - 1, 2 * N - 1), dtype=np.uint8)
            R[0::2, 0::2] = fb
            R[1::2, 0::2] = fb[:-1, :] & fb[1:, :] & ~inactive(Wh[:-1, :], 1)          # (+1, 0)
            R[0::2, 1::2] = fb[:, :-1] & fb[:, 1:] & ~inactive(Wh[:, :-1], 3)          # (0, +1)
            R[1::2, 1::2] = fb[:-1, 1:] & fb[1:, :-1] & ~inactive(Wh[:-1, 1:], 4)      # (+1, -1) from (i, j)
            labr, n = ndimage.label(R, structure=np.ones((3, 3), dtype=np.uint8))
            del R
            lab = np.ascontiguousarray(labr[0::2, 0::2])
            del labr
            comp = np.arange(n + 1)
        else:
            cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
            lab, n = ndimage.label(fb, structure=cross)
            pa, pb = [], []
            if nine:            # (+1, +1) from (i, j): commons (i+1, j), (i, j+1); link 7
                A = fb[:-1, :-1] & fb[1:, 1:] & ~fb[1:, :-1] & ~fb[:-1, 1:] & ~inactive(Wh[:-1, :-1], 7)
                pa.append(lab[:-1, :-1][A]); pb.append(lab[1:, 1:][A])
            if nine or tri:     # (+1, -1) from (i, j): commons (i+1, j), (i, j-1); sq9 link 5, tri link 4
                k = 4 if tri else 5
                A = fb[:-1, 1:] & fb[1:, :-1] & ~fb[1:, 1:] & ~fb[:-1, :-1] & ~inactive(Wh[:-1, 1:], k)
                pa.append(lab[:-1, 1:][A]); pb.append(lab[1:, :-1][A])
            a = np.concatenate(pa) if pa else np.zeros(0, np.int64)
            b = np.concatenate(pb) if pb else np.zeros(0, np.int64)
            if a.size:
                g = sp.coo_matrix((np.ones(a.size, np.int8), (a, b)), shape=(n + 1, n + 1))
                _, comp = connected_components(g, directed=False)
            else:
                comp = np.arange(n + 1)
        if n == 0:
            return None
        cnt = np.bincount(lab.ravel(), minlength=n + 1).astype(np.float64)
        cnt[0] = 0.0
        ccnt = np.bincount(comp, weights=cnt)
        best = int(np.argmax(ccnt))
        if ccnt[best] <= 0:
            return None
        sel = comp == best
        sel[0] = False
        return sel[lab]

    def neighbour_offsets(self):
        """The stencil's links: 4 (sq5), 8 (sq9) or 6 (tri) index offsets."""
        if self.tri():
            return ((-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (-1, 1))
        if self.p.stencil == 9:
            return ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))
        return ((-1, 0), (1, 0), (0, -1), (0, 1))

    def hops(self, di, dj):
        """Lattice hop count of an index offset: the breadth-first distance in the stencil's
        neighbour graph -- |a|+|b| (sq5), max(|a|,|b|) (sq9), max(|a|,|b|,|a+b|) (tri, whose
        links are (+-1,0), (0,+-1), (1,-1), (-1,1)). Shortest paths can be taken monotone in
        both indices, so the array's edges never lengthen them."""
        a, b = np.abs(di), np.abs(dj)
        if self.tri():
            return np.maximum(np.maximum(a, b), np.abs(np.asarray(di) + np.asarray(dj)))
        if self.p.stencil == 9:
            return np.maximum(a, b)
        return a + b

    def nearest_free(self, i0, j0, allowed=None):
        """The free cell (mask 1, inside [1, N-2]^2) nearest to (i0, j0) in lattice hops -- what
        a breadth-first search over the neighbour graph finds -- with ties at the same hop count
        broken by the smallest physical distance, then the smallest (i, j). None if there is no
        free cell. allowed (bool [i, j], default mask_np) restricts the candidates (main_region()).
        Vectorised over a growing window instead of a Python BFS (a solid disk of
        radius 1000 cells would have been ~3M Python visits): every cell within h hops lies
        within h of (i0, j0) in each index, so a window of half-size h that holds a free cell
        at <= h hops holds the nearest one."""
        N, msk = self.N, (self.mask_np if allowed is None else allowed)
        x0, y0 = self.cell_xy(i0, j0)
        h = 4
        while True:
            a0, a1 = max(1, i0 - h), min(N - 1, i0 + h + 1)
            b0, b1 = max(1, j0 - h), min(N - 1, j0 + h + 1)
            whole = a0 == 1 and b0 == 1 and a1 == N - 1 and b1 == N - 1
            if a1 > a0 and b1 > b0:
                I, J = np.nonzero(msk[a0:a1, b0:b1])
                if I.size:
                    I, J = I + a0, J + b0
                    hp = self.hops(I - i0, J - j0)
                    hmin = int(hp.min())
                    if hmin <= h or whole:
                        sel = hp == hmin
                        I, J = I[sel], J[sel]
                        da, db = (I - self.c()).astype(np.float64), (J - self.c()).astype(np.float64)
                        X, Y = (da + db * 0.5, db * S3H) if self.tri() else (da, db)
                        d2 = (X - x0) ** 2 + (Y - y0) ** 2
                        k = np.lexsort((J, I, d2))[0]
                        return int(I[k]), int(J[k])
            if whole:
                return None
            h *= 2

    def src_ij(self):
        ir = self.inner_R()
        return self.snap_inside(*self.xy_to_cell(self.p.sx * ir, self.p.sy * ir))

    def vertices(self, inset, mode):
        if self.p.shape == 'circle':
            return []
        # the plate's corners are arcs: its 4 arc midpoints are the vertex set, not the 68
        # points of the sampled outline (which would put a source on every arc segment)
        poly = self.plate_corners() if self.p.shape == 'unlicensed' else self.shape_poly()
        Q = self.clip_convex(poly, self.clip_region())
        m, out = len(Q), []
        for k in range(m):
            x, y = Q[k]
            L = math.hypot(x, y) or 1
            if mode == 'wall':
                a, b = Q[(k + m - 1) % m], Q[(k + 1) % m]
                ux, uy, vx, vy = a[0] - x, a[1] - y, b[0] - x, b[1] - y
                lu, lv = math.hypot(ux, uy) or 1, math.hypot(vx, vy) or 1
                ux, uy, vx, vy = ux / lu, uy / lu, vx / lv, vy / lv
                bx, by = ux + vx, uy + vy
                lb = math.hypot(bx, by)
                sh = math.sqrt(max(0, (1 - (ux * vx + uy * vy)) / 2))
                want = inset / sh if (lb > 1e-12 and sh > 1e-9) else inset
                if want > 0.95 * L:
                    self.vtx_capped += 1
                d = min(want, 0.95 * L)
                dx, dy = (bx / lb, by / lb) if lb > 1e-12 else (-x / L, -y / L)
                px, py = x + dx * d, y + dy * d
            else:
                f = max(0, (L - inset)) / L
                px, py = x * f, y * f
            px, py = js_round(px * 1e9) / 1e9, js_round(py * 1e9) / 1e9
            out.append(self.snap_inside(*self.xy_to_cell(px, py), getattr(self, '_snap_approx', False)))
        seen, uniq = set(), []
        for v in out:
            if (v[0], v[1]) not in seen:
                seen.add((v[0], v[1]))
                uniq.append(v)
        return uniq

    def recache_source(self, approx_ok=False):
        p = self.p
        ir = self.inner_R()
        sn = self.snap_source(*self.xy_to_cell(p.sx * ir, p.sy * ir), approx_ok)
        self.src = sn[:3]
        self.src_snap, self.src_moved = sn[3], bool(sn[4])
        self._snap_approx = approx_ok
        self.vtx_capped = 0
        self.vtx = (self.vertices(3 * p.sigma, 'wall') if p.vtx_inset == 'wall'
                    else self.vertices(max(2, math.ceil(1.5 * p.sigma)), 'radial'))
        # the driven cells, on the device: the continuous source, or every vertex in phase
        cells = ([self.src[:2]] if p.src == 'cont'
                 else [v[:2] for v in self.vtx] if (p.src == 'vtx' and p.vtx_drive) else [])
        self.drive_cells = [(int(c[0]), int(c[1])) for c in cells]
        self.nsrc = len(cells)
        self.sc = ti.ndarray(dtype=ti.i32, shape=(max(1, len(cells)), 2))
        if cells:
            self.sc.from_numpy(np.array(cells, dtype=np.int32))
        self.clip_frac = self.pulse_clip_fraction()
        self.beam_key = None
        if p.src == 'beam':
            self.build_beam()
        # the recorder's three probe cells, clamped into the array (the web app's own indices
        # always lie inside it for N >= 11; a clamp only keeps a GPU read in bounds)
        (ai, aj), (bi, bj) = self.probe_cells()
        mi, mj = self.mode_antinode() if p.src == 'mode' else (ai, aj)
        cl = lambda v: max(0, min(self.N - 1, int(v)))
        self.prb.from_numpy(np.array([cl(ai), cl(aj), cl(bi), cl(bj), cl(mi), cl(mj)], dtype=np.int32))

    # ---- the web app's probe placements (record(), the ledger) ---------------------------
    def pA(self):
        """Offset of probe A along +x (index units): max(3, round(innerR * 0.44))."""
        return max(3, js_round(self.inner_R() * 0.44))

    def pB(self):
        """Offset of probe B along +y (index units): max(4, round(innerR * 0.60))."""
        return max(4, js_round(self.inner_R() * 0.60))

    def probe_cells(self):
        """((i, j) of probe A, (i, j) of probe B): (c + pA, c) and (c, c + pB), c = round((N-1)/2)."""
        c = js_round((self.N - 1) / 2)
        return (c + self.pA(), c), (c, c + self.pB())

    def mode_antinode(self):
        """The (m, n) mode's antinode cell [round(z + L/(2m)), round(z + L/(2n))]."""
        z, L = self.mode_box()
        mm, mn = self.mode_mn()
        return js_round(z + L / (2 * mm)), js_round(z + L / (2 * mn))

    def pulse_clip_fraction(self):
        """Share of the seeded gaussian(s), by u^2, cut off at t = 0 (outside the mask, off
        the array, or on the clamped ring) -- the web app's pulseClipFraction."""
        p = self.p
        pts = self.vtx if p.src == 'vtx' else ([self.src] if p.src == 'pulse' else [])
        if not pts:
            return None
        s2 = 2 * p.sigma * p.sigma
        Mw = math.ceil(4 * p.sigma / (S3H if self.tri() else 1)) + 2
        N, c = self.N, self.c()
        # the web app sums over the array extended by Mw, but only where r^2 < 64 s2: the
        # union of the sources' windows of that radius is the same sum
        boxes = [self._window(pi, pj, math.sqrt(64 * s2), pad=Mw) for (pi, pj, _) in pts]
        i0, i1 = min(b[0] for b in boxes), max(b[1] for b in boxes)
        j0, j1 = min(b[2] for b in boxes), max(b[3] for b in boxes)
        I, J = np.meshgrid(np.arange(i0, i1, dtype=np.float64), np.arange(j0, j1, dtype=np.float64),
                           indexing='ij')
        da, db = I - c, J - c
        X, Y = (da + db * 0.5, db * S3H) if self.tri() else (da, db)
        v = np.zeros_like(X)
        for (pi, pj, _) in pts:
            cx, cy = self.cell_xy(pi, pj)
            r2 = (X - cx) ** 2 + (Y - cy) ** 2
            v += np.where(r2 < 64 * s2, np.exp(-r2 / s2), 0.0)
        w = v * v
        off = (I < 0) | (J < 0) | (I >= N) | (J >= N)
        cut = off.copy()
        inner = ~off
        ii, jj = I[inner].astype(int), J[inner].astype(int)
        ring = (ii == 0) | (jj == 0) | (ii == N - 1) | (jj == N - 1)
        # the ring is cut where it is clamped: the square lattice's Dirichlet ring, the triangular ring
        # (always, except on the rhombic torus, where the ring cells are ordinary cells)
        ring_cut = (self.tri() and not self.torus()) or (not self.tri() and self.bc_effective() in ('dirichlet', 'cpml',
                                                                                                    'sponge'))
        cut[inner] = (self.mask_np[ii, jj] == 0) | (ring & ring_cut)
        tot = w.sum()
        return float(w[cut].sum() / tot) if tot > 0 else None

    # ---- the source note and the drive warning (05/10/2026, NotesCLAUDE/vertex_choppiness_MIRROR.md) ----
    def vtx_bumps(self):
        """True when the vertex source seeds its gaussian bumps: always for undriven vertices; for driven
        ones only with p.vtx_drive_bump (the web app's behaviour)."""
        p = self.p
        return p.src == 'vtx' and (not p.vtx_drive or bool(p.vtx_drive_bump))

    def clearance_from(self, i, j):
        """The web app's clearanceFrom(i, j): the PHYSICAL distance from cell (i, j) to the nearest
        masked-out cell (solids count as walls), the array edge too on the full square (index distance,
        as the web app), the seam on the rhombic torus (S3H x the index distance, as ledger_MIRROR's
        _clearance); base_half() if there is no wall at all. Agrees with ledger_MIRROR._clearance.
        Cost: a search window around the cell that doubles until it holds the nearest wall, so
        O(clearance^2): ~0 for vertex sources, one pass over the array for a centred source."""
        N, msk = self.N, self.mask_np
        if msk[i, j] == 0:
            return 0.0
        best = math.inf
        if self.sq_full():
            best = float(min(i, j, N - 1 - i, N - 1 - j))
        elif self.tri_torus():
            best = S3H * min(i, j, N - 1 - i, N - 1 - j)
        px, py = self.cell_xy(i, j)
        r = 8.0
        while True:
            i0, i1, j0, j1 = self._window(i, j, r)
            free = msk[i0:i1, j0:j1] != 0
            if not free.all():
                # only masked cells with a free 8-neighbour can be the nearest (ledger_MIRROR's
                # _boundary_cells); a nearest one within r has that neighbour inside the window too
                nb = np.zeros_like(free)
                h, w = free.shape
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di or dj:
                            a0, a1, b0, b1 = max(0, di), h + min(0, di), max(0, dj), w + min(0, dj)
                            nb[a0 - di:a1 - di, b0 - dj:b1 - dj] |= free[a0:a1, b0:b1]
                I, J = np.nonzero(nb & ~free)
                if I.size:
                    X, Y = self._grid_xy_cells(I + i0, J + j0)
                    best = min(best, float(np.hypot(X - px, Y - py).min()))
            whole = i0 <= 0 and j0 <= 0 and i1 >= N and j1 >= N
            if best <= r or whole:
                break
            r *= 2
        return best if math.isfinite(best) else self.base_half()

    def _grid_xy_cells(self, I, J):
        """Physical (x, y) of index arrays I, J (cell_xy, vectorised)."""
        c = self.c()
        da, db = I.astype(np.float64) - c, J.astype(np.float64) - c
        return (da + db * 0.5, db * S3H) if self.tri() else (da, db)

    def _src_note_key(self):
        p = self.p
        return (p.src, p.sigma, p.vtx_inset, p.vtx_drive, p.vtx_drive_bump, p.vtx_soft, p.shape, p.lattice, self.N,
                self._mask_ver, id(self.mask_np), tuple(self.src[:2]), tuple(tuple(v[:2]) for v in self.vtx),
                self.clip_frac, self.vtx_capped)

    def source_note(self):
        """The web app's source note (srcNote), for the UI: where the source sits and what the mask
        cuts from it at t = 0. Cached: recomputed only when the source, the mask or the inputs change
        (a reseed, painting), never per frame -- call it every frame freely. Returns a dict:
          src            p.src
          vtx            True for vertex sources
          n_sources      the number of vertex sources (1 for a point source; 0: the circle has none)
          clearance      physical distance (cells) from the source -- the nearest of the vertex
                         sources -- to the nearest wall (clearance_from), None for sources with no
                         point (mode, slit, beam, packet: the beam and packet have their own readouts)
          clearance_sigma  clearance / sigma when a gaussian is seeded (pulse, vertex bumps), else None
          clearance_ok   clearance >= CLEAR_OK_SIGMA sigma (the web app's green), None when n/a
          inset          p.vtx_inset for vertex sources, else None
          capped         vertex sources whose 3-sigma wall inset was capped (0 unless inset 'wall')
          clip           share of the seeded gaussian(s) cut at t = 0 (m.clip_frac) when one is seeded,
                         else None (an undriven point or driven vertices that start from rest)
          clip_warn      clip > CLIP_WARN (1%)
          driven_rest    driven vertices with no seeded bumps (p.vtx_drive_bump False)
          drive          'soft' / 'hard' for driven vertices (p.vtx_soft, 09/10/2026), else None
          lines          [(text, level)] ready to print, level 'info' | 'ok' | 'warn' | 'dim'
                         (warn = amber, ok = green / the normal text, info = the normal text, dim = grey).
                         The clearance line is ok from 2.9 sigma (the web app's green), amber below that
                         only when the clip is over 1% too (the wall inset can snap to 2.67 sigma with
                         ~1e-5 cut), info otherwise."""
        key = self._src_note_key()
        if self._src_note[0] == key:
            return self._src_note[1]
        p = self.p
        vtx = p.src == 'vtx'
        point = vtx or p.src in ('impulse', 'pulse', 'cont')
        bumps = p.src == 'pulse' or (vtx and self.vtx_bumps())
        pts = self.vtx if vtx else ([self.src] if point else [])
        clear = min(self.clearance_from(int(v[0]), int(v[1])) for v in pts) if pts else None
        sig = max(float(p.sigma), 1e-12)
        csig = clear / sig if (clear is not None and bumps) else None
        cok = (clear >= CLEAR_OK_SIGMA * sig) if csig is not None else None
        clip = self.clip_frac if bumps else None
        cwarn = clip is not None and clip > CLIP_WARN
        capped = self.vtx_capped if (vtx and p.vtx_inset == 'wall') else 0
        lines = []
        if vtx:
            if self.vtx:
                head = f'{len(self.vtx)} vertex sources' + ((', driven soft' if p.vtx_soft else ', driven hard')
                                                             if p.vtx_drive else '')
                if csig is not None:
                    # the web app colours the sigma figure green from 2.9 sigma; snapping can leave the wall
                    # inset's sources at 2.67-2.9 sigma with nothing cut, so amber only when the clip is too
                    lines.append((f'{head} - clearance {clear:.1f} cells = {csig:.2f} sigma to the nearest wall',
                                  'ok' if cok else ('warn' if cwarn else 'info')))
                else:
                    lines.append((f'{head} - clearance {clear:.1f} cells to the nearest wall', 'info'))
                if bumps:
                    lines.append(('inset 3 sigma from BOTH walls (along the bisector)' if p.vtx_inset == 'wall'
                                  else f'radial inset {max(2, math.ceil(1.5 * p.sigma))} cells - the wall distance is '
                                       'only x cos(pi/n) of it, so the pulse is clipped at t = 0 on sharp vertices',
                                  'dim'))
                else:
                    lines.append(('driven vertices start from rest: no seeded pulses (vtx_drive_bump off)', 'dim'))
            else:
                lines.append(('the circle has no vertices - pick a polygon or the rhombus', 'warn'))
        elif point:
            c = (self.N - 1) / 2
            off = math.hypot(self.src[0] - c, self.src[1] - c)
            lines.append((f'source cell ({self.src[0]}, {self.src[1]}) - {off:.1f} cells off centre - '
                          f'clearance {clear:.1f} cells', 'info'))
        if capped:
            lines.append((f'3 sigma inset capped at {capped} corner(s) - the domain is too small to keep 3 sigma '
                          'from both walls; those sources sit closer', 'warn'))
        if clip is not None:
            amt = f'{clip:.1e}' if clip < 1e-4 else f'{100 * clip:.2f}%'
            txt = (f'pulse cut off at t = 0: {amt} of the whole gaussian (sum u^2: outside the mask, off the '
                   'array, or on the clamped ring)')
            if cwarn:
                txt += ' - the cut is a sharp edge the grid cannot resolve'
                if vtx and p.vtx_inset != 'wall':
                    txt += '; try vertex inset: wall'
            lines.append((txt, 'warn' if cwarn else 'ok'))
        note = dict(src=p.src, vtx=vtx, n_sources=len(pts), clearance=clear, clearance_sigma=csig,
                    clearance_ok=cok, inset=p.vtx_inset if vtx else None, capped=capped, clip=clip,
                    clip_warn=cwarn, driven_rest=bool(vtx and p.vtx_drive and not self.vtx_bumps()),
                    drive=(('soft' if p.vtx_soft else 'hard') if (vtx and p.vtx_drive) else None), lines=lines)
        self._src_note = (key, note)
        return note

    def drive_warnings(self, mode=None):
        """Cells mode (05/10/2026): an amber note when a point-driven source -- cont, slit, or vertices with
        vtx_drive (hard, or soft since 09/10/2026: a soft point radiates the same wavelength) -- has a drive
        wavelength lambda = Co / freq under PPW_MIN cells (the beam has its own check; domain mode has
        scale_warnings(), so this returns [] there). An under-resolved drive fills the domain with grid-scale
        waves: the web app's default freq 0.12 at Co 0.5 (lambda 4.2 cells) puts 95-99.9% of the energy under
        8 cells per wavelength (vertex_choppiness_MIRROR.md section 4.7); the port's default since 09/10/2026,
        0.03125, is 16 cells. Cheap: no grid work."""
        p = self.p
        if (mode or p.scale) == 'domain':
            return []
        if not (p.src in ('cont', 'slit') or (p.src == 'vtx' and p.vtx_drive)):
            return []
        lam = p.Co / max(p.freq, 1e-12)
        if lam >= PPW_MIN:
            return []
        who = {'cont': 'continuous source', 'slit': 'slit drive', 'vtx': 'driven vertices'}[p.src]
        return [f'{who}: lambda {lam:.1f} cells < {PPW_MIN} (Co / freq): grid-scale waves - '
                f'freq <= {p.Co / (2 * PPW_MIN):.4f} gives lambda >= {2 * PPW_MIN} cells']

    def phase_lag(self, sigma=None, theta_deg=None, ndir=36):
        """The grid's phase error for a gaussian pulse of width sigma (cells; default p.sigma), as the lag of
        its fronts behind the exact (continuum) solution in CELLS PER DOMAIN CROSSING (r_in of travel).
        From the lattice's discrete dispersion 4 sin^2(w/2) = Co^2 Lambda(K) (beam_MIRROR.omega_of; uniform
        medium, Co = p.Co): per wavenumber K the lag is r_in (1 - v_p/c), v_p = w/|K| cells per step,
        c = Co; the front's lag is its mean over the pulse's 2D energy spectrum, k^3 exp(-k^2 sigma^2) per
        shell -- the radial shift that best aligns the lattice's front with the exact one (validated
        against that measured shift: defaults_tests_MIRROR.py test 5). The energy peaks at k_dom =
        sqrt(3/2) / sigma. On sq5 the lag depends on the direction; sq9 and tri are nearly isotropic.
        The lag grows linearly with the distance and falls ~4x per halving of k (doubling sigma); in domain
        mode a doubling of N halves the lag per crossing and quarters it in units of sigma.
        Returns dict(lag = at theta_deg, or the largest over directions; lag_min, lag_max; dir_min,
        dir_max (deg); sigma; k_dom; r_in; per_sigma = lag_max / sigma; crossings_to_sigma = crossings
        until the slowest direction trails by one sigma (a rough scale for when the fronts lose their
        shape: NotesCLAUDE/vertex_choppiness_MIRROR.md section 7); lines = [(text, level)]).
        ~5 ms (37 directions x 2000 wavenumbers); cached on its inputs, so a per-frame call is free."""
        p = self.p
        s = float(p.sigma if sigma is None else sigma)
        s = max(s, 0.5)
        kind, co = self.lattice_kind(), float(p.Co)
        r_in = self.inner_R()
        key = (s, kind, co, r_in, theta_deg, ndir)
        hit = getattr(self, '_phase_lag', None)
        if hit is not None and hit[0] == key:
            return hit[1]
        kmax = min(9.0 / s, PI)
        k = np.linspace(kmax / 2000, kmax, 2000)
        w = k ** 3 * np.exp(-(k * s) ** 2)

        def lag_at(th):
            om = bm.omega_of(kind, co * co, k * math.cos(th), k * math.sin(th))
            d = 1.0 - om / (k * co)
            ok_ = np.isfinite(d)
            return r_in * float((w[ok_] * d[ok_]).sum() / max(w[ok_].sum(), 1e-300))
        span = 90.0 if kind != 'tri' else 60.0
        dirs = np.linspace(0.0, span, ndir + 1)
        lags = np.array([lag_at(math.radians(t)) for t in dirs])
        lo, hi = float(lags.min()), float(lags.max())
        dlo, dhi = float(dirs[int(lags.argmin())]), float(dirs[int(lags.argmax())])
        lag = lag_at(math.radians(theta_deg)) if theta_deg is not None else hi
        cross = s / hi if hi > 0 else math.inf
        rng = f'{lo:.2f}' if abs(hi - lo) < 0.005 else f'{lo:.2f}-{hi:.2f}'
        cx = f'{cross:.1f}' if cross < 10 else f'{cross:.0f}'
        lines = [(f'phase lag ~{rng} cells per crossing at sigma {s:g} ({hi / s:.3f} sigma): the fronts trail '
                  f'by one sigma after ~{cx} crossing(s)' if math.isfinite(cross) else
                  f'phase lag ~{rng} cells per crossing at sigma {s:g}', 'warn' if cross < 4 else 'dim')]
        if cross < 4:
            lines.append(('the grid carries the pulse too slowly; it scrambles over time - '
                          'domain mode / larger N / Co nearer the limit', 'dim'))
        out = dict(lag=lag, lag_min=lo, lag_max=hi, dir_min=dlo, dir_max=dhi, sigma=s,
                   k_dom=math.sqrt(1.5) / s, r_in=r_in, per_sigma=hi / s, crossings_to_sigma=cross, lines=lines)
        self._phase_lag = (key, out)
        return out

    # ---- directional sources: beam and packet (01/10/2026, beam_MIRROR.py) ----------------
    def lattice_kind(self):
        return 'tri' if self.tri() else ('sq9' if self.p.stencil == 9 else 'sq5')

    def src_co2(self):
        """Co^2 at the source cell (the medium there): what the beam's and packet's discrete
        dispersion is computed with."""
        i, j = self.src[:2]
        cc = float(self.cc_np[i, j])
        return cc if cc > 0 else self.p.Co ** 2

    def _beam_key(self):
        p = self.p
        return (p.src, p.Co, p.freq, p.wave, p.beam_angle, p.beam_width, p.beam_onesided, p.beam_group,
                p.lattice, p.stencil, self.N, tuple(self.src[:2]), self._mask_ver, id(self.mask_np),
                id(self.cc_np), p.medium)

    def build_beam(self):
        """The beam's driven cells (front aperture + back aperture) and per-harmonic amplitude
        and phase lag, on the host; uploaded as SC (cells) and, per launch, the (K, n) drive."""
        p = self.p
        kind, co2 = self.lattice_kind(), self.src_co2()
        omega = TAU * p.freq
        th = math.radians(p.beam_angle)
        k1 = bm.k_along(kind, co2, omega, th)
        th_k = bm.aim(kind, co2, th, omega=omega, group=p.beam_group) if k1 else th
        w = max(0.5, float(p.beam_width))
        si, sj = self.src[:2]
        x0, y0 = self.cell_xy(si, sj)
        i0, i1, j0, j1 = self._window(si, sj, bm.APERTURE_TRUNC * w + 3)
        I, J = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing='ij')
        X, Y = self._grid_xy(i0, i1, j0, j1)
        I, J, X, Y = I.ravel(), J.ravel(), X.ravel(), Y.ravel()
        N = self.N
        msk = self.open_mask()                      # 09/10/2026: the aperture skips an absorbing layer's cells

        def ok(ii, jj):
            inside = (ii >= 1) & (jj >= 1) & (ii <= N - 2) & (jj <= N - 2)
            out = np.zeros(ii.shape, dtype=bool)
            out[inside] = msk[ii[inside], jj[inside]] != 0
            return out
        ok_a = ok(I, J)
        vx, vy, _ = bm.back_shift(kind, th_k)
        # the shift v in index units: sq (vx, vy); tri x = di + dj/2, y = dj s3/2
        dj = js_round(vy / S3H) if self.tri() else js_round(vy)
        di = js_round(vx - dj * 0.5) if self.tri() else js_round(vx)

        def ok_b(front):
            bi, bj = I[front] + di, J[front] + dj
            return np.arange(len(front)), ok(bi, bj)
        harm = list(self.harm)
        T = bm.beam_tables(kind, co2, omega, harm, th_k, X, Y, x0, y0, w, bool(p.beam_onesided), ok_a, ok_b)
        f, bk = T['front'], T['back']
        cells = [np.stack([I[f], J[f]], 1)]
        if len(bk) and T['info']['back_mode'] == 'slab':
            cells.append(np.stack([I[bk], J[bk]], 1))
        elif len(bk):
            cells.append(np.stack([I[f][bk] + di, J[f][bk] + dj], 1))
        cells = np.concatenate(cells).astype(np.int32) if len(f) else np.zeros((0, 2), np.int32)
        keep = np.isfinite(T['amp']).all(0) & (np.abs(T['amp']).max(0) > 0) if len(cells) else np.zeros(0, bool)
        if not T['info']['ok']:
            keep[:] = False
        cells = cells[keep]
        info = T['info']
        info.update(theta_k=math.degrees(th_k), co2=co2, omega=omega, ppw=(info['lam'] or 0.0),
                    n_cells=len(cells))
        self.beam = dict(cells=cells, amp=T['amp'][:, keep], lag=T['lag'][:, keep],
                         hh=[h for h, _ in harm], omega=omega, info=info,
                         ramp=bm.RAMP_PERIODS / max(p.freq, 1e-9))
        self.nbeam = len(cells)
        self.sc = ti.ndarray(dtype=ti.i32, shape=(max(1, self.nbeam), 2))
        if self.nbeam:
            self.sc.from_numpy(cells)
        self.bdrv = ti.ndarray(dtype=float, shape=KMAX * max(1, self.nbeam))
        self.beam_key = self._beam_key()
        return info

    def _add_packet(self, U0, Up):
        """Add the one-way packet (beam_MIRROR.packet_levels) at the source cell to the two
        levels U0 (current) and Up (previous). The FFT box holds the envelope to e^-16."""
        p = self.p
        kind, co2 = self.lattice_kind(), self.src_co2()
        lam = max(2.0, float(p.pk_lambda))
        k = TAU / lam
        th = math.radians(p.beam_angle)
        th_k = bm.aim(kind, co2, th, k=k, group=p.beam_group)
        w, L = max(0.5, float(p.beam_width)), max(0.5, float(p.pk_len))
        si, sj = self.src[:2]
        x0, y0 = self.cell_xy(si, sj)
        R = 4.0 * max(w, L) + 4
        hj = math.ceil(R / S3H) + 2 if self.tri() else math.ceil(R) + 2
        hi = math.ceil(R + hj * 0.5) + 2 if self.tri() else hj
        bi0, bj0 = si - hi, sj - hj
        X, Y = self._grid_xy(bi0, si + hi + 1, bj0, sj + hj + 1)
        u0, um, vmean = bm.packet_levels(kind, co2, x0, y0, th_k, k, w, L, X, Y, self.tri(), p.beam_group)
        N = self.N
        a0, a1 = max(0, bi0), min(N, si + hi + 1)
        b0, b1 = max(0, bj0), min(N, sj + hj + 1)
        U0[a0:a1, b0:b1] += u0[a0 - bi0:a1 - bi0, b0 - bj0:b1 - bj0]
        Up[a0:a1, b0:b1] += um[a0 - bi0:a1 - bi0, b0 - bj0:b1 - bj0]
        om = float(bm.omega_of(kind, co2, k * math.cos(th_k), k * math.sin(th_k)))
        vg = bm.group_velocity(kind, co2, k * math.cos(th_k), k * math.sin(th_k))
        kept = float((u0[a0 - bi0:a1 - bi0, b0 - bj0:b1 - bj0] ** 2).sum() / max((u0 ** 2).sum(), 1e-300))
        self.packet_info = dict(theta_k=math.degrees(th_k), k=k, lam=lam, omega=om,
                                vg=(float(vg[0]), float(vg[1])), vg_mean=vmean, co2=co2, in_array=kept)
        return self.packet_info

    def beam_geometry(self):
        """For the overlays: ((x0, y0), (xa, ya), (xb, yb), (xe, ye)) in PHYSICAL coordinates --
        the source point, the two ends of the aperture (perpendicular to the phase direction,
        +-2.5 w for the beam, +-2 w for the packet) and the tip of the arrow along the requested
        (energy) direction. None for other sources."""
        p = self.p
        if p.src not in BEAM_SOURCES:
            return None
        x0, y0 = self.cell_xy(*self.src[:2])
        th = math.radians(p.beam_angle)
        if p.src == 'beam':
            info = (self.beam or {}).get('info') or {}
            th_k = math.radians(info.get('theta_k', p.beam_angle))
            half = bm.APERTURE_TRUNC * p.beam_width
            lam = info.get('lam') or 10.0
        else:
            th_k = math.radians((self.packet_info or {}).get('theta_k', p.beam_angle))
            half = 2.0 * p.beam_width
            lam = p.pk_lambda
        tx, ty = -math.sin(th_k), math.cos(th_k)
        ln = max(3.0 * p.beam_width, 3.0 * lam, 0.15 * self.inner_R())
        return ((x0, y0), (x0 - half * tx, y0 - half * ty), (x0 + half * tx, y0 + half * ty),
                (x0 + ln * math.cos(th), y0 + ln * math.sin(th)))

    # ---- drive --------------------------------------------------------------------------
    def k_max(self):
        return max(1, math.floor(self.p.Co / (PPW_MIN * max(self.p.freq, 1e-9))))

    def harmonics(self):
        K, out, w = self.k_max(), [], self.p.wave
        if w == 'sine':
            return [(1, 1.0)]
        for k in range(1, K + 1):
            if w == 'square':
                if k % 2:
                    out.append((k, 4 / (PI * k)))
            elif w == 'saw':
                out.append((k, (2 if k % 2 else -2) / (PI * k)))
            elif w == 'tri':
                if k % 2:
                    out.append((k, 8 * (-1 if ((k - 1) / 2) % 2 else 1) / (PI * PI * k * k)))
        if not out:
            out.append((1, 8 / (PI * PI) if w == 'tri' else 4 / PI))
        return out

    def drive(self, step):
        s = 0.0
        for k, a in self.harm:
            s += a * math.sin(TAU * self.p.freq * k * step)
        return s

    # ---- soft driven vertices (09/10/2026, the user's decision; NotesCLAUDE/v1_decisions_MIRROR.md) ----
    def vtx_soft_on(self):
        """True while the driven vertices are SOFT sources (Params.vtx_soft, the port's default): the step
        ADDS the drive at the vertex cells instead of imposing it."""
        p = self.p
        return p.src == 'vtx' and bool(p.vtx_drive) and bool(p.vtx_soft) and self.nsrc > 0

    def soft_tables(self):
        """The soft vertices' drive, per vertex and harmonic (cached on the lattice, freq, harmonics and the
        cells' Co^2): hh (harmonic numbers), amp[v, h] = a_h * soft_gain(lattice, CC_v, 2 pi freq h), gain[v, h],
        g00[v][h] (the lattice Green's function used), co2[v]. The hard drive is sum_h a_h sin(2 pi freq h n);
        the soft one sum_h amp[v, h] sin(2 pi freq h n): every harmonic the hard drive has (the band limit of
        harmonics() is kept), each scaled so that a lone vertex radiates the hard vertex's wave."""
        p = self.p
        cells = self.drive_cells
        co2 = tuple(float(self.cc_np[i, j]) for i, j in cells)
        kind = self.lattice_kind()
        key = (kind, float(p.freq), tuple(self.harm), co2)
        if self._soft[0] == key:
            return self._soft[1]
        hh = np.array([h for h, _ in self.harm], dtype=np.float64)
        ah = np.array([a for _, a in self.harm], dtype=np.float64)
        gain = np.zeros((len(cells), len(hh)))
        g00 = []
        for v, c2 in enumerate(co2):
            row = []
            for q, h in enumerate(hh):
                om = TAU * p.freq * h
                gain[v, q] = soft_gain(kind, c2, om)
                row.append(lattice_g00(kind, 4.0 * math.sin(om / 2) ** 2 / c2) if c2 > 0 and om > 0 else None)
            g00.append(row)
        tab = dict(hh=hh, amp=gain * ah[None, :], gain=gain, g00=g00, co2=co2, kind=kind)
        self._soft = (key, tab)
        return tab

    def soft_drive(self, step):
        """The soft drive added at each driven vertex at step n (host, per vertex): sum_h amp[v, h]
        sin(2 pi freq h n)."""
        t = self.soft_tables()
        return t['amp'] @ np.sin(TAU * self.p.freq * t['hh'] * step)

    # ---- seeding ------------------------------------------------------------------------
    def mode_box(self):
        if self.full_array():
            return 0, self.N - 1                        # the whole index square (the tri 'array': its rhombus)
        hw = math.floor(self.dom_R())
        return self.c() - hw - 1, 2 * hw + 2

    def mode_note(self):
        """None when the 'mode' source's sin.sin seed is an EXACT eigenmode of the solver here (the
        square lattice, a square domain -- the full square / 'array' with the Dirichlet ring, or the
        masked square with a pinned edge --, a uniform medium, no solids); otherwise why not. The
        source still seeds sin.sin in index space (the web app does so on every domain)."""
        p = self.p
        ab = self.absorbing()
        if ab is not None and ab != 'mur':
            # 09/10/2026: an absorbing boundary (mur2, cpml, sponge, an open edge): the domain is an open window
            return (f'an absorbing boundary ({ab}): the domain has no standing modes -- n/a (the seeded sin.sin '
                    'leaves through the open edge)')
        if self.tri():
            return ('sin.sin in index space is not an eigenmode of the triangular Laplacian (its (1,-1) links '
                    'couple i and j)' + (' -- n/a on the rhombic array' if p.shape == 'array' else ''))
        if p.shape not in ('square', 'array'):
            return 'sin.sin is the eigenmode of a square domain only'
        if self.sq_full() and p.bc != 'dirichlet':
            return f'the full square with a {p.bc} ring: sin.sin is the Dirichlet eigenmode'
        if not self.sq_full() and p.edge == 'rigid':
            return 'the masked square with a RIGID edge: its eigenmodes are cos.cos, not sin.sin'
        if p.medium != 'uniform':
            return 'a non-uniform medium'
        if self.has_solids:
            return 'solids inside the domain'
        return None

    def mode_mn(self):
        L = self.mode_box()[1]
        return min(self.p.mm, L - 1), min(self.p.mn, L - 1)

    def _window(self, bi, bj, rphys, pad=0):
        """Index box holding the physical disk of radius rphys about cell (bi, bj). On the
        triangular lattice one row is S3H apart and a row shift adds up to half a column."""
        hj = math.ceil(rphys / S3H) + 2 if self.tri() else math.ceil(rphys) + 2
        hi = math.ceil(rphys + hj * 0.5) + 2 if self.tri() else hj
        lo, hi_ = -pad, self.N + pad
        return (max(lo, bi - hi), min(hi_, bi + hi + 1), max(lo, bj - hj), min(hi_, bj + hj + 1))

    def _bump(self, U, bi, bj, amp):
        """The web app sweeps the whole grid (addBump). Past r = 38.7 sigma, exp(-r^2/2s^2)
        underflows to EXACTLY 0 in f64 (e^-745), so a window of that radius adds the very same
        values -- and at N = 8193 it is ~10^3 times less work and memory."""
        s = self.p.sigma
        i0, i1, j0, j1 = self._window(bi, bj, math.sqrt(2 * 746) * s)
        bx, by = self.cell_xy(bi, bj)
        X, Y = self._grid_xy(i0, i1, j0, j1)
        dx, dy = X - bx, Y - by
        U[i0:i1, j0:j1] += amp * np.exp(-(dx * dx + dy * dy) / (2 * s * s))

    def seed(self):
        self._mur2_ok = False                       # 10/10/2026: Mur2 restarts with a Mur1 step (step())
        p, N = self.p, self.N
        U = np.zeros((N, N))
        self.drops = 0
        if p.src == 'impulse':
            U[self.src[0], self.src[1]] = 1.0
        elif p.src == 'pulse':
            self._bump(U, self.src[0], self.src[1], 1.0)
        elif p.src == 'vtx':
            # driven vertices start from rest unless vtx_drive_bump (05/10/2026; the web app seeds the
            # bumps and the hard drive dents their centres at step 1 -- harness PATCH 'vtx-drive-rest')
            if self.vtx_bumps():
                for v in self.vtx:                      # sequential, as the JS accumulates
                    self._bump(U, v[0], v[1], 1.0)
        elif p.src == 'packet':
            Up = np.zeros((N, N))
            self._add_packet(U, Up)
        elif p.src == 'mode':
            z, L = self.mode_box()
            mm, mn = self.mode_mn()
            I, J = np.meshgrid(np.arange(N, dtype=np.float64), np.arange(N, dtype=np.float64), indexing='ij')
            U = np.sin(mm * PI * (I - z) / L) * np.sin(mn * PI * (J - z) / L)
        U[self.mask_np == 0] = 0.0                      # applyMask
        self.slot = 0
        if p.src == 'packet':                           # a moving packet: two different levels
            Up[self.mask_np == 0] = 0.0
            self.set_levels(Up, U)
        else:
            self.set_levels(U, U)                       # starts at rest
        k_zero(self.U, 2)
        if self.abs_np is not None:
            self._aux_sync(zero_memory=True)            # 09/10/2026: the CPML memory starts at 0
        self.step_n = 0
        self.t = 0.0
        # clearField: the histories restart; the mode probe gets the seeded value (step 0)
        self.hist = {k: [] for k in HIST_CAP}
        self.hist_emax = 0.0                        # 10/10/2026: the recorded E's maximum since the seed (ledger 'eopen')
        if self.record and p.src == 'mode':
            ai, aj = self.mode_antinode()
            if 0 <= ai < N and 0 <= aj < N:
                self.hist['mode'].append(float(self.fdt(U[ai, aj])))
        self.amax = float(np.abs(U).max()) if N else 0.0   # the web app sets S.amax at seed

    def reseed(self):
        """Everything the web app's reseed() does that affects the field."""
        if self.p.N % 2 == 0:
            self.p.N += 1                                # the web app's N slider: odd -> true centre
        if self.N != self.p.N:
            self._alloc(self.p.N)
        self.apply_scale()                               # 'domain' mode only (02/10/2026)
        self.harm = self.harmonics()
        self.build_domain()
        self.recache_source()
        self.seed()

    def drop_pulse(self):
        """Superpose a gaussian at the source cell onto the running field (both time levels,
        so the new bump starts at rest), then applyMask -- the web app's dropPulse."""
        s = self.src_ij()
        add = np.zeros((self.N, self.N))
        if self.p.src == 'packet':                      # P launches another packet
            addp = np.zeros((self.N, self.N))
            self._add_packet(add, addp)
        else:
            self._bump(add, s[0], s[1], 1.0)
            addp = add
        out = self.mask_np == 0
        u0, u1 = self.level(0) + addp, self.level(1) + add
        u0[out] = 0.0
        u1[out] = 0.0
        self.set_levels(u0, u1)
        self.drops += 1
        self.amax = float(np.abs(u1).max())             # dropPulse refreshes S.amax too

    # ---- the time step --------------------------------------------------------------------
    def step(self, n=1):
        """n steps, in launches of CHUNKS steps each (one kernel call per launch)."""
        p = self.p
        if self._dom_ring_key != self._ring_key() or self._dom_layer_key != self._layer_key():
            # the triangular 'array' changed between the torus and a ring (p.bc set without a reseed):
            # its domain mask includes the ring only on the torus, so rebuild it (the field is kept,
            # and zeroed on the cells that leave the domain). 09/10/2026: the same when an absorbing layer
            # appeared, went or changed (p.bc / p.edge / the sponge's width set without a reseed)
            self.build_domain()
            k_zero_masked(self.U, self.M)
            self.recache_source()
        lat = LAT_TRI if self.tri() else LAT_SQ
        # triangular lattice: the clamped branch, except the rhombic torus (05/10/2026); the rigid ring
        # is made by the link words. Square lattice: the effective bc (09/10/2026: p.bc, except the absorbing
        # bcs' fallbacks on masked shapes; cpml and sponge clamp the ring, their layer is absk's)
        bc = (BC_PER if self.tri_torus() else BC_DIR) if lat == LAT_TRI else BC_CODE[self.bc_effective()]
        lay = self.abs_layer()
        absk = ABS_NONE if lay is None else (ABS_CPML if lay['kind'] == 'cpml' else ABS_SPONGE)
        aux = self._aux_sync() if absk != ABS_NONE else None
        nine = lat == LAT_SQ and p.stencil == 9
        # a solid can touch the array ring, which only the masked pass clamps
        masked = not self.full_array() or self.has_solids
        slitm = self.has_solids
        # walls (04/10/2026): the link words replace the mask in the kernel only while some free
        # cell has a dropped or pinned-blocked link; otherwise the 03/10 kernel runs unchanged
        self._walls_sync()
        walls = self.walls_active()
        Marr = self.W if walls else self.M
        rec = bool(self.record)
        srck = SRC_NONE
        jw = jd = cA = cB = half = 0
        if p.src == 'slit':
            srck = SRC_SLIT
            c = js_round(self.c())
            ir = self.inner_R()
            ox, oy = js_round(p.sx * ir), js_round(p.sy * ir)
            jw, jd = c + oy - js_round(ir * 0.40), c + oy - js_round(ir * 0.76)
            half, off = max(2, js_round(ir * 0.07)), js_round(ir * 0.24)
            cA, cB = c + ox - off, c + ox + off
        elif self.nsrc:
            srck = SRC_CELLS
        drv, nsrc = self.drv, self.nsrc
        soft = None
        if srck == SRC_CELLS and self.vtx_soft_on():
            # soft driven vertices (09/10/2026): the beam's kernel path, U += DRV[q * nsrc + k] -- added, not
            # imposed, so the cells are transparent; the per-vertex drive comes from soft_tables()
            soft = self.soft_tables()
            if self.vdrv is None or self.vdrv.shape[0] != KMAX * nsrc:
                self.vdrv = ti.ndarray(dtype=float, shape=KMAX * nsrc)
            srck, drv = SRC_BEAM, self.vdrv
        if p.src == 'beam':
            if self.beam_key != self._beam_key():
                self.build_beam()
            srck, drv, nsrc = (SRC_BEAM, self.bdrv, self.nbeam) if self.nbeam else (SRC_NONE, drv, 0)
        kmur = (p.Co - 1) / (p.Co + 1)
        start, pending = self.hrow, 0
        # Mur 2nd order (10/10/2026): the first launch after anything that set the levels or the mask, changed Co, or
        # ran another bc starts with one Mur1 step on the sides (BC_MUR2S; _mur1_side says why)
        mur2s = bc == BC_MUR2 and not self._mur2_ok
        if n > 0:
            self._mur2_ok = bc == BC_MUR2
        while n > 0:
            K = next(k for k in CHUNKS if k <= n)
            bcl = bc
            if mur2s:
                bcl, mur2s = BC_MUR2S, False
            if rec and pending + K > HCAP:
                self._flush_hist(start, pending)
                start, pending = self.hrow, 0
            if soft is not None:
                ph = (TAU * p.freq) * np.outer(self.step_n + np.arange(K, dtype=np.float64), soft['hh'])
                dv = np.zeros(KMAX * nsrc, dtype=self.fdt)
                dv[:K * nsrc] = (np.sin(ph) @ soft['amp'].T).ravel()      # (K, nsrc), q-major
                self.vdrv.from_numpy(dv)
            elif srck == SRC_BEAM:
                b = self.beam
                dv = np.zeros(KMAX * nsrc, dtype=self.fdt)
                dv[:K * nsrc] = bm.beam_drive(b['amp'], b['lag'], b['hh'], b['omega'], self.step_n, K,
                                              b['ramp']).ravel()
                self.bdrv.from_numpy(dv)
            elif srck != SRC_NONE:
                # the drive, host-side with the same sin() as before, for the K steps ahead
                dv = np.zeros(KMAX, dtype=self.fdt)
                for q in range(K):
                    dv[q] = self.drive(self.step_n + q)
                self.drv.from_numpy(dv)
            if absk != ABS_NONE:
                # 09/10/2026: an absorbing layer (sponge / CPML): + AUX and the coded mask LV; s0 also carries the sq9
                # CPML's ping-pong parity (3 x par: s0 % 3 stays the slot)
                s0p = self.slot + 3 * self._cpml_par
                if rec:
                    k_run_abs_rec(self.U, Marr, self.CC, drv, self.sc, self.hist_dev, self.prb, aux, self.M,
                                  s0p, K, lat, nine, bc, masked, srck,
                                  nsrc, p.damp, kmur, jw, jd, cA, cB, half, slitm, self.hrow, walls, absk)
                else:
                    k_run_abs(self.U, Marr, self.CC, drv, self.sc, aux, self.M, s0p, K, lat, nine, bc, masked,
                              srck, nsrc, p.damp, kmur, jw, jd, cA, cB, half, slitm, walls, absk)
                self._cpml_par = (self._cpml_par + K) % 2
            elif rec:
                k_run_rec(self.U, Marr, self.CC, drv, self.sc, self.hist_dev, self.prb,
                          self.slot, K, lat, nine, bcl, masked, srck,
                          nsrc, p.damp, kmur, jw, jd, cA, cB, half, slitm, self.hrow, walls)
            else:
                k_run(self.U, Marr, self.CC, drv, self.sc, self.slot, K, lat, nine, bcl, masked, srck,
                      nsrc, p.damp, kmur, jw, jd, cA, cB, half, slitm, walls)
            self.slot = (self.slot + K) % 3
            for _ in range(K):                   # the web app's clock: S.t += S.Co per step
                self.t += p.Co
            self.step_n += K
            n -= K
            if rec:
                self.hrow = (self.hrow + K) % HCAP
                pending += K
        if rec and pending:
            self._flush_hist(start, pending)

    def _flush_hist(self, start, cnt):
        """Append the device ring's rows [start, start+cnt) to m.hist: ONE device->host copy
        per step() call (not per launch), since each copy is a sync (~0.4 ms, 29/09/2026)."""
        H = self.hist_dev.to_numpy()[:HCAP]
        rows = H[(start + np.arange(cnt)) % HCAP].astype(np.float64)
        co = self.p.Co
        E = rows[:, 0] / (2 * max(co * co, 1e-12)) + 0.5 * rows[:, 1]    # the web app's formula
        h = self.hist
        h['E'].extend(E.tolist())
        em = float(np.max(E))
        if not (em <= self.hist_emax):              # a NaN propagates (a diverged run)
            self.hist_emax = em
        if self.tri():
            # the triangular operator's own potential: -<u, L u> = (2/3) sum over the 3 link directions
            h['Etri'].extend((rows[:, 0] / (2 * max(co * co, 1e-12)) + (rows[:, 1] + rows[:, 6]) / 3.0).tolist())
        h['amax'].extend(rows[:, 2].tolist())
        h['pA'].extend(rows[:, 3].tolist())
        h['pB'].extend(rows[:, 4].tolist())
        if self.p.src == 'mode':
            h['mode'].extend(rows[:, 5].tolist())
        for k, cap in HIST_CAP.items():            # ENERGY.shift() etc.: keep the newest
            if len(h[k]) > cap:
                del h[k][:len(h[k]) - cap]
        self.amax = float(rows[-1, 2])

    # ---- readouts ---------------------------------------------------------------------------
    def level(self, k):
        """A time level as a numpy array [i, j]: k = 1 the current one, k = 0 the previous."""
        out = np.empty((self.N, self.N), dtype=self.fdt)
        k_store(self.U, (self.slot + k) % 3, out)
        return out

    def set_levels(self, u_prev, u_cur):
        self._mur2_ok = False                       # 10/10/2026: Mur2 restarts with a Mur1 step (step())
        k_load(self.U, self.slot, np.ascontiguousarray(u_prev, dtype=self.fdt))
        k_load(self.U, (self.slot + 1) % 3, np.ascontiguousarray(u_cur, dtype=self.fdt))

    def value(self, i, j):
        """One cell of the current level (a device read: fine for probes, not for sweeps)."""
        return float(self.U[(self.slot + 1) % 3, i, j])

    def stats(self):
        """(energy, peak |u|) of the CURRENT state with the web app's record() formula (one
        kernel + one sync). While the recorder runs, m.hist['E'][-1] / m.amax hold the same
        values for the last step without a kernel."""
        self._walls_sync()
        walls = self.walls_active()
        k_stats(self.U, self.W if walls else self.M, self.acc, self.slot, (self.slot + 1) % 3, walls,
                LAT_TRI if self.tri() else LAT_SQ, self.tri_torus(), self.M, self.abs_np is not None)
        a = self.acc.to_numpy()[0].astype(np.float64)
        E = a[0] / (2 * max(self.p.Co * self.p.Co, 1e-12)) + 0.5 * a[1]
        # the triangular operator's own energy (m.hist['Etri']'s formula), None on the square lattice
        self.stats_Etri = (float(a[0] / (2 * max(self.p.Co * self.p.Co, 1e-12)) + (a[1] + a[6]) / 3.0)
                           if self.tri() else None)
        return float(E), float(a[2])

    def field(self):
        return self.level(1)

    def domain_cells(self):
        """Free (wave-carrying) cells: the domain minus the solids."""
        return int(self.mask_np.sum())

    def solid_cells(self):
        """Solid cells inside the domain (solid cells outside it change nothing)."""
        if self.solid_np is None:
            return 0
        return int(np.count_nonzero((self.domain_mask_np != 0) & (self.solid_np != 0)))
