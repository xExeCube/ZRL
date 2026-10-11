# R != 1 sweep of the whiteboard-derived relationships (ZRL docs), 2026-10-01
import math, numpy as np
from scipy.integrate import quad
rng = np.random.default_rng(7)
Rs = [0.6, 2**-0.25, 1.0, 1.6, 2.9]
pi = math.pi
def euler_deg(Q, R, h=1e-6):            # R dQ/dR / Q
    return R*(Q(R*(1+h)) - Q(R*(1-h)))/(2*R*h)/Q(R)
def joint_deg(F, R, R0, lam=1.37):       # log_lam F(lam R, lam R0)/F(R, R0)
    a, b = F(lam*R, lam*R0), F(R, R0)
    return math.log(abs(a/b))/math.log(lam)
rows = []
def row(name, src, res, deg, verdict): rows.append((name, src, res, deg, verdict))

# ---- 1. delta-sum family vs the actual geometry (quarter disc minus the a x b rectangle)
def true_area(R, k):
    a, b = R/math.sqrt(1+k*k), k*R/math.sqrt(1+k*k)
    d1 = quad(lambda x: math.sqrt(max(R*R-x*x,0)) - b, 0, a)[0]        # above the rectangle
    d2 = quad(lambda x: math.sqrt(max(R*R-x*x,0)), a, R)[0]            # right of it
    return d1 + d2
S_good = lambda R,k: R*R*(pi/4 - k/(1+k*k))
S_bad  = lambda R,k: pi*R**2/4 - k*(R**8-1)/(2*R**4)
S_v1   = lambda R,k: (4*pi*R**6 - 2*k*R**8 + 2*k)/(4*R**4)
S_slip = lambda R,k: (4*pi*R**6 - 2*k*R**8 + 2*k)/4
for nm, f in [("S_good", S_good), ("S_bad", S_bad), ("S_v1", S_v1), ("S_slip", S_slip)]:
    errs = {R: max(abs(f(R,k)-true_area(R,k)) for k in (0.3, 1.0, 2.5)) for R in Rs}
    degs = [euler_deg(lambda r: f(r, 0.7), R) for R in (0.6, 1.6, 2.9)]
    print(f"{nm:7s} max|err vs geometry|: " + "  ".join(f"R={R:.3f}:{e:.2e}" for R,e in errs.items())
          + f"   Euler degree @0.6/1.6/2.9: " + "/".join(f"{d:+.3f}" for d in degs))
# structural identity: the bad correction term is the bridge product
V1sq = lambda R: R*R + R**-2; V2sq = lambda R: R*R - R**-2
print("S_bad == pi R^2/4 - (k/2) V1^2 V2^2 :", max(abs(S_bad(R,0.7) - (pi*R*R/4 - 0.35*V1sq(R)*V2sq(R))) for R in Rs))
print("S_bad(1, k) for k = 0.3, 1, 2.5:", [round(S_bad(1,k),12) for k in (0.3,1,2.5)], " pi/4 =", round(pi/4,12))
# k-inverse round trip and its R = 1 behaviour
kinv = lambda R,S: R**4*(pi*R*R - 4*S)/(2*(R**8-1))
print("k-inverse round trip (k=0.7):", {R: (round(kinv(R, S_bad(R,0.7)),12) if abs(R-1)>1e-12 else '0/0') for R in Rs})
quart = lambda R,k: (lambda u,S: 2*k*u**4 - pi*u**3 + 4*S*u*u - 2*k)(R*R, S_bad(R,k))
print("quartic residual (k=0.7):", max(abs(quart(R,0.7)) for R in Rs))

# ---- 2. reciprocal inversion bridge (anchored: the '1' is R0^4)
for R in (1.6, 2.9):
    y = math.sqrt(R**4-1)/R; x = math.sqrt(R**4+1)/R
    assert abs(y*y + R**-2 - R*R) < 1e-12 and abs(x*x - R**-2 - R*R) < 1e-12
Fy = lambda R,R0: (math.sqrt(R**4-R0**4)/R)**2 + (R0**2/R)**2 - R*R + R*R  # = R^2 when the identity holds
print("inversion bridge y^2 + (R0^2/R)^2 = R^2: joint degree", round(joint_deg(Fy, 1.6, 1.0),6),
      "| R-only Euler degree of R^-2 term:", round(euler_deg(lambda r: r**-2, 1.6),6))
# ---- 3. bridge invariants
inv = max(abs(V1sq(R)**2 - V2sq(R)**2 - 4) for R in Rs)
prod = max(abs(math.sqrt(V1sq(R)*V2sq(R)) - math.sqrt(R**8-1)/R**2) for R in Rs if R>=1)
ch = max(abs(V1sq(R) - 2*math.cosh(2*math.log(R))) + abs(V2sq(R) - 2*math.sinh(2*math.log(R))) for R in Rs)
print(f"V1^4-V2^4=4 max dev {inv:.1e}; V1V2=sqrt(R^8-1)/R^2 max dev {prod:.1e}; cosh/sinh form max dev {ch:.1e}")
print("  with R0 restored V1^4-V2^4 = 4R0^4: joint degree",
      round(joint_deg(lambda R,R0: (R*R+R0**4/R**2)**2-(R*R-R0**4/R**2)**2, 1.6, 1.0),6))
# ---- 4. degree-1/2 geometry (should pass at every R)
def silver(R):
    A,B,C = math.sqrt(2)*R/2, (2-math.sqrt(2))*R/2, (2*math.sqrt(2)-2)*R/2
    return max(abs(A+B-R), abs(math.sqrt(2)*B-C), abs(B+C-A))
print("silver descent max dev:", max(silver(R) for R in Rs))
def rhombus(R, th):           # long diagonal 2R along x, acute angle th at the long tips
    B = R*math.tan(th/2); P = np.array([[R,0],[0,B],[-R,0],[0,-B]])
    s = np.linalg.norm(P[0]-P[1]); ang = math.acos(np.dot(P[1]-P[0], P[3]-P[0])/s/s)
    r_in = R*B/math.hypot(R,B)
    return abs(s - R/math.cos(th/2)), abs(ang-th), abs(r_in/R - math.sin(th/2))
print("rhombus (side, tip angle, r/R) max dev:", max(max(rhombus(R, t)) for R in Rs for t in (0.6, 1.2, 2.0)))
wick = max(abs(quad(lambda y: math.sqrt(R*R+y*y), 0, R)[0] - R*R/2*(math.sqrt(2)+math.log(1+math.sqrt(2)))) for R in Rs)
print("Wick quarter integral max dev:", wick)
seg = lambda R,th: R*R/2*(th-math.sin(th))
dseg = max(abs((seg(R+1e-6,1.1)-seg(R-1e-6,1.1))/2e-6 - (R*1.1 - R*math.sin(1.1))) for R in Rs)
print("d(segment)/dR at fixed theta = s - R sin(s/R) max dev:", dseg)
lens = max(abs(2*seg(R, pi/3) - R*R*(pi/3-math.sqrt(3)/2)) for R in Rs)
hexsil = max(abs(quad(lambda x: min(math.sqrt(3)*R/2, math.sqrt(3)*(R-abs(x))), -R, R, points=[-R/2,R/2])[0] - 3*math.sqrt(3)*R*R/4) for R in Rs)
print("hexagon lens 2*seg(60deg) = R^2(pi/3 - sqrt3/2) max dev:", lens, "| hexagon silhouette integral max dev:", hexsil)
# ---- 5. the whiteboard tab's anisotropy readout  D_i = R(s_i - sqrt(s_i))
D = lambda R,s: R*(s - math.sqrt(s))
for sx, sy in [(1.0,1.0), (1.3,1.3), (0.7,0.7), (1.3,0.8)]:
    print(f"  tab: s=({sx},{sy})  SumDelta/R = {(D(1,sx)+D(1,sy)):+.4f}   squeeze rapidity 0.5ln(sx/sy) = {0.5*math.log(sx/sy):+.4f}"
          f"   conformal factor sqrt(sx sy) = {math.sqrt(sx*sy):.4f}")
# ---- 6. precedent artifacts at other 'units'
print("(2^n)^4 vs 2^n equal only at n=0:", [n for n in range(-3,4) if 2**(4*n) == 2**n])
