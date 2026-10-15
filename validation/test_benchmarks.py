"""
Independent checks of the two benchmark solvers of the paper: the exact Rayleigh solver (scattering.rayleigh_exact) and the C-method (cmethod.c_method).

No third solver is used; the checks are physical identities and internal consistency of each solver:
  B1  flat interface (Fresnel written independently), C-method.
  B3  reciprocity of the reflected efficiencies, e_n(alpha) = e_n(-alpha_n), for the C-method and the Rayleigh solver.
  B4  boundary conditions satisfied by the C-method amplitudes at off-grid points of the surface (where the Rayleigh hypothesis holds), with a sensitivity test.
  B5  convergence of the C-method with the number of modes M and its insensitivity to the tiny loss used to sort the modes.
  B6  base case of the paper: Rayleigh solver against the C-method.
  B7  profiles outside the Rayleigh hypothesis (max slope 1.2 ... 1.7): difference between the two solvers and truncation of the C-method (M = r_max against M = 2 r_max).
      This is NOT an independent confirmation of the C-method there (no third solver): it documents the size of the effects.
usage: python test_benchmarks.py            (~1 min)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scattering import *
from cmethod import c_method, spectral_derivative

WL, EPS = 0.1, 2.25
results = []
L1 = lambda a, b, m: np.abs(a - b)[m].sum() / b[m].sum()


def check(name, value, tol, detail="", greater=False):
    ok = value > tol if greater else value < tol; results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {value:.2e} ({'>' if greater else '<'} {tol:.0e}) {detail}")


def fresnel_ref(pol, cos_l):
    s = np.sqrt(EPS - (1 - cos_l ** 2) + 0j)
    return (cos_l - s) / (cos_l + s) if pol == "TE" else (EPS * cos_l - s) / (EPS * cos_l + s)


def profile(a, kb, Lam, k):
    return lambda x: a * np.cos(2 * np.pi * x) + kb / k * np.cos(Lam * 2 * np.pi * x)


def solve_c(pol, th, Ff, M, delta=1e-10):
    p = Problem(1.0, WL, th, EPS, pol, M, N=4096); F = Ff(p.x)
    B = c_method(p, F, spectral_derivative(p, F, 1), spectral_derivative(p, F, 2), M=M, delta=delta)
    return p, [np.nan_to_num(b) for b in B]


def solve_r(pol, th, Ff, rmax):
    p = Problem(1.0, WL, th, EPS, pol, rmax, N=4096); F = Ff(p.x)
    return p, rayleigh_exact(p, F, spectral_derivative(p, F, 1))


def restrict(p, B, e, m):
    """efficiencies of the problem p restricted to |r| <= m (to compare solvers with different truncations)"""
    i = np.abs(p.r) <= m; return e[0][i], e[1][i], p.prop_p[i], p.prop_m[i]


def compare(pa, Ba, pb, Bb):
    """L1 error (reflected, transmitted) of solver a against solver b on the orders |r| <= min(rmax_a, rmax_b)"""
    m = min(pa.rmax, pb.rmax); ea = efficiencies(pa, *Ba); eb = efficiencies(pb, *Bb)
    ra, ta, _, _ = restrict(pa, Ba, ea, m); rb, tb, mp, mm = restrict(pb, Bb, eb, m)
    return L1(ra, rb, mp), L1(ta, tb, mm)


# ------------------------------------------------------------------------------------------------------------------ B1
print("B1 flat interface")
for pol in ("TE", "TM"):
    th = np.deg2rad(20); R = fresnel_ref(pol, np.cos(th)); flat = lambda x: 0 * x
    pc, Bc = solve_c(pol, th, flat, 10)
    check(f"{pol} C-method |B+_0 - R| + |B-_0 - (1+R)|", abs(Bc[0][10] - R) + abs(Bc[1][10] - (1 + R)), 1e-9)

# ------------------------------------------------------------------------------------------------------------------ B3
print("B3 reciprocity of the reflected efficiencies, e_n(alpha) = e_n(-alpha_n), profile of the base case")
for pol in ("TE", "TM"):
    th = np.arcsin(0.35); k = 2 * np.pi / WL; Ff = profile(0.03, 0.25, 15, k); K = 2 * np.pi
    for name, solver in (("C-method", lambda t: solve_c(pol, t, Ff, 57)), ("Rayleigh", lambda t: solve_r(pol, t, Ff, 57))):
        worst = 0.0
        p, B = solver(th); e = efficiencies(p, *B)
        for nn in (-5, -2, 1, 3):
            alpha_n = k * 0.35 + nn * K; th2 = np.arcsin(-alpha_n / k)
            p2, B2 = solver(th2); e2 = efficiencies(p2, *B2)
            a, b = e[0][p.rmax + nn], e2[0][p2.rmax + nn]
            worst = max(worst, abs(a - b) / a)
        check(f"{pol} {name}: max relative difference over n = -5, -2, 1, 3", worst, {"Rayleigh": 5e-3, "C-method": 1e-5}[name], "(Rayleigh: truncation at r_max = 57 limits the weak orders)")

# ------------------------------------------------------------------------------------------------------------------ B4
print("B4 boundary conditions satisfied by the C-method amplitudes at off-grid surface points (profile of the Rayleigh regime), summing the Rayleigh series up to |r| <= R")
rng = np.random.default_rng(1); xs = rng.uniform(0, 1, 300)
for pol in ("TE", "TM"):
    th = np.deg2rad(20); k = 2 * np.pi / WL; af, bz, Lam = 0.03, 0.1 / k, 5; Ff = profile(af, 0.1, Lam, k)
    p, B = solve_c(pol, th, Ff, 80)
    K, al, be, ar = p.K, p.alpha, p.beta, p.alpha_r
    Fx = Ff(xs); Fp = lambda x: -af * K * np.sin(K * x) - bz * Lam * K * np.sin(Lam * K * x)
    prev = None; res2 = []
    for R in (30, 40, 50, 60):
        sel = np.abs(p.r) <= R
        ua = lambda x, z: np.exp(1j * (al * x - be * z)) + (B[0][sel][None, :] * np.exp(1j * (ar[sel][None, :] * x[:, None] + p.bp[sel][None, :] * z[:, None]))).sum(1)
        ub = lambda x, z: (B[1][sel][None, :] * np.exp(1j * (ar[sel][None, :] * x[:, None] - p.bm[sel][None, :] * z[:, None]))).sum(1)
        h = 1e-6; dn = lambda u: -Fp(xs) * (u(xs + h, Fx) - u(xs - h, Fx)) / (2 * h) + (u(xs, Fx + h) - u(xs, Fx - h)) / (2 * h)
        r1 = np.abs(ua(xs, Fx) - ub(xs, Fx)).max() / np.abs(ua(xs, Fx)).max()
        r2 = np.abs(dn(ua) - p.nu2 * dn(ub)).max() / (k * np.abs(ua(xs, Fx)).max()); res2.append((r1, r2))
        if R == 60: r2w = np.abs(dn(ua) - (1.0 if pol == "TM" else 1 / EPS) * dn(ub)).max() / (k * np.abs(ua(xs, Fx)).max())
    print(f"        {pol}: residual of the field / of nu^2 d_n u for R = 30, 40, 50, 60: " + "  ".join(f"{a:.1e}/{b:.1e}" for a, b in res2))
    results.append(all(res2[i + 1][1] < res2[i][1] / 2 for i in range(3)))
    check(f"{pol} residual of the second condition at R = 60 (it falls by more than 2 per 10 orders: the tail of the series)", res2[-1][1], 1e-4)
    print(f"        sensitivity: with the wrong nu^2 the residual is {r2w:.1e}"); results.append(r2w > 1e-3)

# ------------------------------------------------------------------------------------------------------------------ B5
print("B5 C-method: convergence with the number of modes and insensitivity to the loss delta (base case)")
for pol in ("TE", "TM"):
    th = np.arcsin(0.35); Ff = profile(0.03, 0.25, 15, 2 * np.pi / WL)
    pref, Bref = solve_c(pol, th, Ff, 120)
    for M in (40, 57, 80):
        p, B = solve_c(pol, th, Ff, M); r, t = compare(p, B, pref, Bref); e = efficiencies(p, *B)
        print(f"        {pol} M={M:3d}: reflected {r:.1e}, transmitted {t:.1e} against M=120; R+T-1 = {e[0].sum() + e[1].sum() - 1:.1e}")
        if M == 57: results.append(r < 1e-5)
    for d in (1e-8, 1e-12):
        p, B = solve_c(pol, th, Ff, 57, delta=d); p0, B0 = solve_c(pol, th, Ff, 57); r, t = compare(p, B, p0, B0)
        check(f"{pol} loss delta = {d:.0e} against 1e-10 (M = 57)", max(r, t), 1e-6)

# ------------------------------------------------------------------------------------------------------------------ B6
print("B6 base case (Lambda = 15, kb = 0.25, L = 10 lambda, sin = 0.35): Rayleigh solver against the C-method (M = 57)")
for pol in ("TE", "TM"):
    th = np.arcsin(0.35); Ff = profile(0.03, 0.25, 15, 2 * np.pi / WL); pc, Bc = solve_c(pol, th, Ff, 57); pr, Br = solve_r(pol, th, Ff, 57)
    check(f"{pol} Rayleigh against the C-method (reflected L1)", compare(pr, Br, pc, Bc)[0], 2e-4)

# ------------------------------------------------------------------------------------------------------------------ B7
print("B7 outside the Rayleigh hypothesis (no third solver: these numbers document the size of the effects)")
for pol, Lam, kb in (("TE", 10, 1.0), ("TE", 15, 1.0), ("TM", 10, 1.0)):
    th = np.arcsin(0.35); k = 2 * np.pi / WL; Ff = profile(0.03, kb, Lam, k)
    p0 = Problem(1.0, WL, th, EPS, pol, 30); rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30, 3 * Lam + 12)
    pc2, Bc2 = solve_c(pol, th, Ff, 2 * rmax); pc1, Bc1 = solve_c(pol, th, Ff, rmax); pr, Br = solve_r(pol, th, Ff, rmax)
    F = Ff(pr.x); slope = np.abs(spectral_derivative(pr, F, 1)).max()
    rc, _ = compare(pr, Br, pc2, Bc2); rtrunc, _ = compare(pc1, Bc1, pc2, Bc2); e2 = efficiencies(pc2, *Bc2); e1 = efficiencies(pc1, *Bc1)
    print(f"      {pol} Lambda={Lam} kb={kb} (max slope {slope:.2f}, r_max={rmax}): Rayleigh against C-method {rc:.1e}; C-method M=r_max against M=2 r_max {rtrunc:.1e}; "
          f"energy R+T-1 of the C-method: {e1[0].sum() + e1[1].sum() - 1:.1e} (M=r_max), {e2[0].sum() + e2[1].sum() - 1:.1e} (M=2 r_max)")
    results.append(abs(e2[0].sum() + e2[1].sum() - 1) < 1e-6)

print(f"\n{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
