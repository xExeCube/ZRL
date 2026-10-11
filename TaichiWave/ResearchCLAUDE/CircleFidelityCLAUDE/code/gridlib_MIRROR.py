"""Benchmark grids for the circle-fidelity test. Every grid is (pos, L, w): node positions,
a sparse discrete Laplacian on the FREE nodes (Dirichlet u = 0 outside), node area weights."""
import math, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
from scipy.spatial import Delaunay
S3H = math.sqrt(3)/2

class Grid:
    def __init__(s, name, pos, L, w, R_geom, kind):
        s.name, s.pos, s.L, s.w, s.R_geom, s.kind = name, pos, L.tocsr(), w, R_geom, kind
        s.r = np.hypot(pos[:,0], pos[:,1]); s.n = len(pos)
    def eig_extremes(s):
        # -L is symmetric w.r.t. the weights: A = W^1/2 (-L) W^-1/2 is symmetric
        Wh = sp.diags(np.sqrt(s.w)); Wih = sp.diags(1/np.sqrt(s.w))
        A = (Wh @ (-s.L) @ Wih); A = ((A + A.T)/2).tocsc()
        lmax = sla.eigsh(A, k=1, which='LA', return_eigenvectors=False)[0]
        lmin = sla.eigsh(A, k=1, sigma=0, which='LM', return_eigenvectors=False)[0]
        s.lmax, s.lmin = lmax, lmin
        s.dt_max = 2/math.sqrt(lmax)
        s.R_eff = 2.404825557695773/math.sqrt(lmin)      # radius of the continuous drum with the same fundamental
        return s


def mixed_area(P, tri):
    """Per-vertex 'mixed Voronoi' area (Meyer et al. 2003): circumcentric dual area for non-obtuse
    triangles, area/2 to the obtuse vertex and area/4 to the others otherwise. On any square or
    triangular lattice this equals the lattice cell area whatever diagonals the triangulation picked."""
    n = len(P); A = np.zeros(n)
    a = P[tri[:,1]]-P[tri[:,0]]; b = P[tri[:,2]]-P[tri[:,0]]
    area = np.abs(0.5*(a[:,0]*b[:,1]-a[:,1]*b[:,0]))
    cots, obt = [], np.zeros(len(tri), int) - 1
    for k in range(3):
        i, j, l = tri[:,k], tri[:,(k+1)%3], tri[:,(k+2)%3]
        u, v = P[j]-P[i], P[l]-P[i]
        dot = (u*v).sum(1); cr = np.abs(u[:,0]*v[:,1]-u[:,1]*v[:,0])
        cots.append(dot/cr); obt[dot < -1e-12] = k          # angle at vertex k is obtuse
    for k in range(3):
        i, j, l = tri[:,k], tri[:,(k+1)%3], tri[:,(k+2)%3]
        e_ij = ((P[j]-P[i])**2).sum(1); e_il = ((P[l]-P[i])**2).sum(1)
        vor = (e_ij*cots[(k+2)%3] + e_il*cots[(k+1)%3])/8   # cot of angle opposite each edge
        contrib = np.where(obt < 0, vor, np.where(obt == k, area/2, area/4))
        np.add.at(A, i, contrib)
    return A

def lattice(kind, R, rot=0.0, shift=(0.0,0.0)):
    """The app's stencils on a masked lattice, nodes x = a e1 + b e2 (rotated by rot, shifted)."""
    if kind == 'tri': e1, e2 = np.array([1.0,0.0]), np.array([0.5,S3H])
    else:             e1, e2 = np.array([1.0,0.0]), np.array([0.0,1.0])
    c, s_ = math.cos(rot), math.sin(rot); Rm = np.array([[c,-s_],[s_,c]])
    e1, e2 = Rm@e1, Rm@e2
    M = int(R/ (S3H if kind=='tri' else 1)) + 4
    A, B = np.meshgrid(np.arange(-M, M+1), np.arange(-M, M+1), indexing='ij')
    A, B = A.ravel(), B.ravel()
    P = A[:,None]*e1 + B[:,None]*e2 + np.array(shift)
    ins = (P**2).sum(1) <= R*R + 1e-9
    idx = -np.ones(len(A), int); idx[ins] = np.arange(ins.sum())
    key = {(a,b): i for a,b,i in zip(A[ins], B[ins], idx[ins])}
    if kind == 'sq5': nb = [((1,0),1.0),((-1,0),1.0),((0,1),1.0),((0,-1),1.0)]
    elif kind == 'sq9': nb = [((1,0),2/3),((-1,0),2/3),((0,1),2/3),((0,-1),2/3),
                              ((1,1),1/6),((-1,1),1/6),((1,-1),1/6),((-1,-1),1/6)]
    elif kind == 'tri': nb = [((1,0),2/3),((-1,0),2/3),((0,1),2/3),((0,-1),2/3),((1,-1),2/3),((-1,1),2/3)]
    diag = -sum(wt for _,wt in nb)
    rows, cols, vals = [], [], []
    for (a,b), i in key.items():
        rows.append(i); cols.append(i); vals.append(diag)
        for (da,db), wt in nb:
            j = key.get((a+da, b+db))
            if j is not None: rows.append(i); cols.append(j); vals.append(wt)
    n = ins.sum(); L = sp.csr_matrix((vals,(rows,cols)), shape=(n,n))
    w = np.full(n, S3H if kind=='tri' else 1.0)
    return Grid(f"{kind}", P[ins], L, w, R, 'lattice')

def p1_fem(points, is_bnd, name, R):
    """Mass-lumped P1 finite elements (cotangent Laplacian) on the Delaunay triangulation;
    nodes flagged is_bnd are Dirichlet (u = 0)."""
    tri = Delaunay(points).simplices
    P = points
    # drop degenerate triangles (area ~ 0)
    a = P[tri[:,1]]-P[tri[:,0]]; b = P[tri[:,2]]-P[tri[:,0]]
    area = 0.5*(a[:,0]*b[:,1]-a[:,1]*b[:,0])
    keep = np.abs(area) > 1e-10; tri = tri[keep]; area = np.abs(area[keep])
    n = len(P); I, J, V = [], [], []
    Mass = np.zeros(n)
    for k in range(3):
        i, j, l = tri[:,k], tri[:,(k+1)%3], tri[:,(k+2)%3]
        # cot of the angle at l, opposite edge (i,j)
        u, v = P[i]-P[l], P[j]-P[l]
        cot = (u*v).sum(1)/np.abs(u[:,0]*v[:,1]-u[:,1]*v[:,0])
        I += [i, j, i, j]; J += [j, i, i, j]; V += [-cot/2, -cot/2, cot/2, cot/2]
    Mass = mixed_area(P, tri)
    K = sp.csr_matrix((np.concatenate(V),(np.concatenate(I),np.concatenate(J))), shape=(n,n))
    free = ~is_bnd; fi = np.where(free)[0]
    Kff = K[fi][:,fi]
    L = -sp.diags(1/Mass[fi]) @ Kff
    g = Grid(name, P[fi], L, Mass[fi], R, 'fem')
    # mesh quality
    ang = []
    for k in range(3):
        i, j, l = tri[:,k], tri[:,(k+1)%3], tri[:,(k+2)%3]
        u, v = P[i]-P[l], P[j]-P[l]
        ang.append(np.degrees(np.arccos(np.clip((u*v).sum(1)/np.linalg.norm(u,axis=1)/np.linalg.norm(v,axis=1),-1,1))))
    ang = np.array(ang); g.min_angle, g.max_angle = ang.min(), ang.max(); g.ntri = len(tri)
    return g

def circle_nodes(R, h):
    nb = int(round(2*math.pi*R/h))
    th = 2*math.pi*np.arange(nb)/nb
    return np.c_[R*np.cos(th), R*np.sin(th)]

def sq_fitted(R, h=1.0, gap=0.5):
    """Square lattice inside, triangles only at the edge: lattice nodes deeper than `gap` cells,
    plus nodes ON the circle (spacing ~h), Delaunay-triangulated."""
    M = int(R/h)+2
    A, B = np.meshgrid(np.arange(-M,M+1), np.arange(-M,M+1), indexing='ij')
    P = np.c_[A.ravel(), B.ravel()]*h
    P = P[np.hypot(P[:,0],P[:,1]) < R - gap*h]
    C = circle_nodes(R, h)
    pts = np.r_[P, C]; bnd = np.r_[np.zeros(len(P),bool), np.ones(len(C),bool)]
    return p1_fem(pts, bnd, f"sq + edge triangles (gap {gap})", R)

def tri_fitted(R, h=1.0, gap=0.5):
    M = int(R/h/S3H)+3
    A, B = np.meshgrid(np.arange(-M,M+1), np.arange(-M,M+1), indexing='ij')
    P = (A.ravel()[:,None]*np.array([1,0]) + B.ravel()[:,None]*np.array([0.5,S3H]))*h
    P = P[np.hypot(P[:,0],P[:,1]) < R - gap*h]
    C = circle_nodes(R, h)
    pts = np.r_[P, C]; bnd = np.r_[np.zeros(len(P),bool), np.ones(len(C),bool)]
    return p1_fem(pts, bnd, f"tri + edge triangles (gap {gap})", R)

def hex_rings(R, h=1.0):
    """Hexagonal shells bent into circles: ring k at radius k h carries 6k nodes."""
    K = int(round(R/h)); pts = [np.zeros((1,2))]
    for k in range(1, K+1):
        n = 6*k; th = 2*math.pi*(np.arange(n) + 0.5*(k % 2))/n
        pts.append(np.c_[k*h*np.cos(th), k*h*np.sin(th)])
    pts = np.vstack(pts); bnd = np.zeros(len(pts), bool); bnd[-6*K:] = True
    return p1_fem(pts, bnd, "hex shells bent to circles", K*h)

def distmesh_circle(R, hfun, h0, iters=120, seed=0, name="unstructured"):
    """Compact DistMesh (Persson & Strang 2004) for the disc with size function hfun(r)."""
    rng = np.random.default_rng(seed)
    # initial points: triangular lattice with spacing h0, rejection by (hmin/h)^2
    M = int(R/h0)+2
    A, B = np.meshgrid(np.arange(-M,M+1), np.arange(-M,M+1), indexing='ij')
    P = (A.ravel()[:,None]*np.array([1,0]) + B.ravel()[:,None]*np.array([0.5,S3H]))*h0
    P = P + rng.normal(0, 1e-3*h0, P.shape)
    P = P[np.hypot(P[:,0],P[:,1]) < R]
    hp = hfun(np.hypot(P[:,0],P[:,1]))
    P = P[rng.random(len(P)) < (hp.min()/hp)**2]
    nb = int(round(2*math.pi*R/hfun(np.array([R]))[0]))
    fix = np.c_[R*np.cos(2*math.pi*np.arange(nb)/nb), R*np.sin(2*math.pi*np.arange(nb)/nb)]
    P = P[np.hypot(P[:,0],P[:,1]) < R - 0.5*hfun(np.array([R]))[0]]
    nf = len(fix)
    for it in range(iters):
        X = np.r_[fix, P]
        T = Delaunay(X).simplices
        bars = np.unique(np.sort(np.r_[T[:,[0,1]], T[:,[1,2]], T[:,[0,2]]], axis=1), axis=0)
        bv = X[bars[:,0]] - X[bars[:,1]]; Lb = np.linalg.norm(bv, axis=1)
        mid = (X[bars[:,0]] + X[bars[:,1]])/2; hb = hfun(np.hypot(mid[:,0], mid[:,1]))
        L0 = hb*1.2*math.sqrt((Lb**2).sum()/(hb**2).sum())
        F = np.maximum(L0 - Lb, 0); Fv = (F/Lb)[:,None]*bv
        Ftot = np.zeros_like(X); np.add.at(Ftot, bars[:,0], Fv); np.add.at(Ftot, bars[:,1], -Fv)
        P = P + 0.2*Ftot[nf:]
        r = np.hypot(P[:,0], P[:,1]); out = r > R - 0.3*hfun(np.array([R]))[0]
        P[out] *= ((R - 0.3*hfun(np.array([R]))[0])/r[out])[:,None]
    X = np.r_[fix, P]; bnd = np.r_[np.ones(nf,bool), np.zeros(len(P),bool)]
    return p1_fem(X, bnd, name, R)

def distmesh2(R, hfun, hmin, iters=400, seed=0, name="unstructured"):
    """Standard DistMesh for the disc (Persson & Strang 2004): free points, outside points
    projected back onto the circle, retriangulate on large moves; boundary = points on the circle."""
    rng = np.random.default_rng(seed)
    M = int(R/hmin)+2
    A, B = np.meshgrid(np.arange(-M,M+1), np.arange(-M,M+1), indexing='ij')
    P = (A.ravel()[:,None]*np.array([1,0]) + B.ravel()[:,None]*np.array([0.5,S3H]))*hmin
    P = P[np.hypot(P[:,0],P[:,1]) < R]
    hp = hfun(np.hypot(P[:,0],P[:,1])); P = P[rng.random(len(P)) < (hmin/hp)**2]
    Pold = np.full_like(P, np.inf); ttol, Fscale, dt_, geps = 0.1, 1.2, 0.2, 1e-3*hmin
    for it in range(iters):
        if np.max(np.hypot(*(P-Pold).T))/hmin > ttol:
            Pold = P.copy(); T = Delaunay(P).simplices
            bars = np.unique(np.sort(np.r_[T[:,[0,1]], T[:,[1,2]], T[:,[0,2]]], axis=1), axis=0)
        bv = P[bars[:,0]] - P[bars[:,1]]; Lb = np.linalg.norm(bv, axis=1)
        mid = (P[bars[:,0]] + P[bars[:,1]])/2; hb = hfun(np.hypot(mid[:,0], mid[:,1]))
        L0 = hb*Fscale*math.sqrt((Lb**2).sum()/(hb**2).sum())
        F = np.maximum(L0 - Lb, 0); Fv = (F/Lb)[:,None]*bv
        Ft = np.zeros_like(P); np.add.at(Ft, bars[:,0], Fv); np.add.at(Ft, bars[:,1], -Fv)
        P = P + dt_*Ft
        r = np.hypot(P[:,0], P[:,1]); out = r > R; P[out] *= (R/r[out])[:,None]
        r = np.hypot(P[:,0], P[:,1]); inn = r < R - geps
        if np.max(np.hypot(*(dt_*Ft[inn]).T))/hmin < 1e-3: break
    r = np.hypot(P[:,0], P[:,1]); bnd = r > R - 1e-6*hmin
    P[bnd] *= (R/r[bnd])[:,None]
    g = p1_fem(P, bnd, name, R); g.iters = it
    return g

def p1_fem_tris(P, tri, is_bnd, name, R):
    """Same as p1_fem but with a given triangulation."""
    import scipy.spatial
    class _D: pass
    a = P[tri[:,1]]-P[tri[:,0]]; b = P[tri[:,2]]-P[tri[:,0]]
    area = np.abs(0.5*(a[:,0]*b[:,1]-a[:,1]*b[:,0]))
    n = len(P); I, J, V = [], [], []; Mass = np.zeros(n); ang = []
    for k in range(3):
        i, j, l = tri[:,k], tri[:,(k+1)%3], tri[:,(k+2)%3]
        u, v = P[i]-P[l], P[j]-P[l]
        cot = (u*v).sum(1)/np.abs(u[:,0]*v[:,1]-u[:,1]*v[:,0])
        I += [i, j, i, j]; J += [j, i, i, j]; V += [-cot/2, -cot/2, cot/2, cot/2]
        ang.append(np.degrees(np.arccos(np.clip((u*v).sum(1)/np.linalg.norm(u,axis=1)/np.linalg.norm(v,axis=1),-1,1))))
    Mass = mixed_area(P, tri)
    K = sp.csr_matrix((np.concatenate(V),(np.concatenate(I),np.concatenate(J))), shape=(n,n))
    fi = np.where(~is_bnd)[0]
    L = -sp.diags(1/Mass[fi]) @ K[fi][:,fi]
    g = Grid(name, P[fi], L, Mass[fi], R, 'fem'); ang = np.array(ang)
    g.min_angle, g.max_angle, g.ntri = ang.min(), ang.max(), len(tri)
    return g

def triangle_mesh(R, hfun, h_edge, name, minangle=30, rounds=6):
    """Shewchuk's Triangle: quality mesh of the disc, boundary polygon ON the circle (spacing h_edge),
    max triangle area (sqrt3/4) h(r)^2 enforced by repeated refinement."""
    import triangle as tr
    nb = int(round(2*math.pi*R/h_edge)); th = 2*math.pi*np.arange(nb)/nb
    V = np.c_[R*np.cos(th), R*np.sin(th)]; S = np.c_[np.arange(nb), (np.arange(nb)+1) % nb]
    m = tr.triangulate(dict(vertices=V, segments=S), f'pq{minangle}a{(math.sqrt(3)/4)*hfun(np.array([0.0]))[0]**2:.6f}')
    for _ in range(rounds):
        P, T = m['vertices'], m['triangles']; c = P[T].mean(1)
        amax = (math.sqrt(3)/4)*hfun(np.hypot(c[:,0], c[:,1]))**2
        a = P[T[:,1]]-P[T[:,0]]; b = P[T[:,2]]-P[T[:,0]]; area = np.abs(0.5*(a[:,0]*b[:,1]-a[:,1]*b[:,0]))
        if np.all(area <= amax*1.0001): break
        m['triangle_max_area'] = amax
        m = tr.triangulate(m, f'rpq{minangle}a')
    P, T = m['vertices'], m['triangles']
    bnd = m['vertex_markers'].ravel() == 1
    r = np.hypot(P[:,0], P[:,1]); P[bnd] *= (R/r[bnd])[:,None]
    return p1_fem_tris(P, T, bnd, name, R)

def penrose_vertices(Rv, gam=(0.2, 0.1, -0.15, -0.05, -0.1)):
    """de Bruijn pentagrid dual (canon: V = sum_{l != j,k} ceil(p.e_l - g_l) e_l). Edge length 1."""
    e = np.array([[math.cos(2*math.pi*k/5), math.sin(2*math.pi*k/5)] for k in range(5)])
    g = np.array(gam); nmax = int(Rv/2.5) + 4; V = []
    for j in range(5):
        for k in range(j+1, 5):
            A = np.array([e[j], e[k]])
            nj, nk = np.meshgrid(np.arange(-nmax, nmax+1), np.arange(-nmax, nmax+1), indexing='ij')
            rhs = np.c_[nj.ravel() + g[j], nk.ravel() + g[k]]
            p = np.linalg.solve(A, rhs.T).T                       # intersection points
            base = np.zeros_like(p)
            for l in range(5):
                if l in (j, k): continue
                base += np.ceil(p @ e[l] - g[l])[:,None]*e[l]
            for a in (0, 1):
                for b in (0, 1):
                    V.append(base + (nj.ravel()[:,None] + a)*e[j] + (nk.ravel()[:,None] + b)*e[k])
    V = np.vstack(V); V = V[np.hypot(V[:,0], V[:,1]) < Rv]
    V = np.unique(np.round(V, 9), axis=0)
    return V

def penrose_fitted(R, gap=0.5):
    V = penrose_vertices(R + 3)
    P = V[np.hypot(V[:,0], V[:,1]) < R - gap]
    C = circle_nodes(R, 1.0)
    pts = np.r_[P, C]; bnd = np.r_[np.zeros(len(P),bool), np.ones(len(C),bool)]
    return p1_fem(pts, bnd, "Penrose + edge triangles", R)
