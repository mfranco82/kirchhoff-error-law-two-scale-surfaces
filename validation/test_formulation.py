"""
Independent checks of the code that implements the formulation of the paper (scattering.py): KA amplitudes, system matrix, Rayleigh solver, recursion of the boundary variations.

Each test avoids the code path it checks (the older test_scattering.py compares the recursion with rayleigh_exact, which shares system_matrix with it, so a convention error common to both would pass):
  T1  flat interface: Fresnel coefficients written here (angles, no use of the code's formulas), TE and TM; energy balance of the code's efficiencies.
  T2  KA amplitudes, Eqs. (11)-(12): adaptive quadrature (scipy.integrate.quad) of the integrals with the local Fresnel coefficient written here, against the FFT/mean of ka_amplitudes.
  T3  boundary conditions: the Rayleigh solution of a two-scale profile is substituted in the two boundary conditions at points of the surface that are not on the FFT grid; the normal
      derivatives are finite differences of the fields (not the analytical derivatives of the code). A copy of the solution with a wrong nu^2 must fail the test (sensitivity).
  T4  recursion: the Taylor coefficients in xi of the Rayleigh solution of f + xi*psi are obtained by a Cauchy integral on a circle of complex xi (the full-profile system built
      anew for every xi) and compared with the coefficients d_n of mvb_coefficients, all Floquet orders, n = 0..5.
  T5  first order about a flat surface derived by hand (2x2 system for each r = +-Lambda) and, for TE, the textbook closed form
      B1+_r = 2 i beta k^2 (eps-1) psi_r / ((beta+beta-_0)(beta+_r+beta-_r))   (Rice's first order), against mvb_coefficients with f = 0, n = 1.
  T6  the matrices of the code (FFT of exp(+-i beta_r f)) against the matrices of the paper, sum_s c_{s,q} (+-i beta_r)^s with c_{s,q} the Fourier coefficients of f^s/s! (the entries of the systems (23) and (26) of Sec. 3).
  T8  the equations of the paper written literally (Eqs. (22) and (25) for Omega and Upsilon with the sums over s, r, q', p and the Fourier coefficients c_{s,q} of f^s/s! and Gamma_{n,q'} of psi^n/n!)
      are evaluated with the coefficients d_n of the code: the residual of both boundary conditions must vanish for n = 1, 2.
  T9  numerical convergence of the implementation: coefficients d_n (n <= 3, base case of the paper) with the grid of 4096 points against 16384, and with the factorization of SmoothOperator reused
      against recomputed.
  T7  KA tends to the exact solution when the surface becomes smooth (convergence rate printed).
usage: python test_formulation.py          (~1 min)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.integrate import quad
from math import factorial
from scattering import *

WL, TH, EPS = 0.1, 20.0, 2.25
results = []


def check(name, value, tol, detail=""):
    ok = value < tol; results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {value:.2e} (tolerance {tol:.0e}) {detail}")


def prob(pol, rmax=40, th=TH, eps=EPS):
    return Problem(1.0, WL, np.deg2rad(th), eps, pol, rmax, N=4096)


def fresnel_ref(pol, cos_l, eps):
    """Fresnel reflection coefficient of the amplitude of E_y (TE) or H_y (TM), from the local cosine of the incidence angle."""
    s = np.sqrt(eps - (1 - cos_l ** 2) + 0j)
    return (cos_l - s) / (cos_l + s) if pol == "TE" else (eps * cos_l - s) / (eps * cos_l + s)


# ------------------------------------------------------------------------------------------------------------------ T1
print("T1 flat interface")
for pol in ("TE", "TM"):
    p = prob(pol, rmax=3); z0 = np.zeros(p.N)
    R = fresnel_ref(pol, np.cos(p.theta_i), EPS)
    B_ex, B_ka = rayleigh_exact(p, z0, z0), ka_amplitudes(p, z0, z0)
    for lab, B in (("Rayleigh", B_ex), ("KA", B_ka)):
        others = np.abs(np.delete(B[0], p.rmax)).max() + np.abs(np.delete(B[1], p.rmax)).max()
        check(f"{pol} {lab}: |B+_0 - R| + |B-_0 - (1+R)| + other orders", abs(B[0][p.rmax] - R) + abs(B[1][p.rmax] - (1 + R)) + others, 1e-12)
    e = efficiencies(p, *B_ex); check(f"{pol} energy balance R + T - 1 (flat)", abs(e[0].sum() + e[1].sum() - 1), 1e-12)
    check(f"{pol} reflectivity from the efficiencies against |R|^2", abs(e[0].sum() - abs(R) ** 2), 1e-12)

# ------------------------------------------------------------------------------------------------------------------ T2
print("T2 KA amplitudes, Eqs. (11)-(12), adaptive quadrature")
a = 0.03
for pol in ("TE", "TM"):
    p = prob(pol, rmax=60); k, K, al, be = p.k, p.K, p.alpha, p.beta; nu2 = p.nu2
    f, f1 = cos_surface(p, a); Bp, Bm = ka_amplitudes(p, f, f1)
    fx = lambda x: a * np.cos(K * x); fpx = lambda x: -a * K * np.sin(K * x)

    def integrand(x, r, sign):
        cl = (np.cos(p.theta_i) + fpx(x) * np.sin(p.theta_i)) / np.sqrt(1 + fpx(x) ** 2); R = fresnel_ref(pol, cl, EPS)
        ar = al + r * K; bpr = np.sqrt(k ** 2 - ar ** 2 + 0j); bmr = np.sqrt(EPS * k ** 2 - ar ** 2 + 0j)
        if sign > 0:
            return np.exp(1j * ((al - ar) * x - (be + bpr) * fx(x))) * ((bpr - ar * fpx(x)) * (1 + R) - (be + al * fpx(x)) * (1 - R)) / (2 * bpr)
        return np.exp(1j * ((al - ar) * x + (bmr - be) * fx(x))) * ((be + al * fpx(x)) * (1 - R) / nu2 + (bmr + ar * fpx(x)) * (1 + R)) / (2 * bmr)

    # the amplitudes of the high evanescent orders are 1e-13 ... 1e-17 (round-off level of the quadrature itself): they are compared to the largest amplitude,
    # and the orders with |B| > 1e-6 are also compared one by one
    err_norm = {1: 0.0, -1: 0.0}; err_rel = 0.0
    for sign, B in ((1, Bp), (-1, Bm)):
        for r in (-12, -3, -1, 0, 1, 2, 5, 9, 20, -20, 40):
            re = quad(lambda x: integrand(x, r, sign).real, 0, 1, limit=800, epsabs=1e-13, epsrel=1e-13)[0]
            im = quad(lambda x: integrand(x, r, sign).imag, 0, 1, limit=800, epsabs=1e-13, epsrel=1e-13)[0]
            d = abs(B[p.rmax + r] - (re + 1j * im)); err_norm[sign] = max(err_norm[sign], d / np.abs(B).max())
            if abs(B[p.rmax + r]) > 1e-6: err_rel = max(err_rel, d / abs(B[p.rmax + r]))
    check(f"{pol} ka_amplitudes against quad, 11 orders, difference relative to the largest amplitude (reflected, transmitted)", max(err_norm.values()), 1e-10)
    check(f"{pol} same, order by order for the orders with |B| > 1e-6", err_rel, 1e-8)

# ------------------------------------------------------------------------------------------------------------------ T3
print("T3 boundary conditions of the Rayleigh solution at off-grid points of the surface (finite-difference normal derivatives)")


def bc_residuals(p, B, af, bz, Lam, nu2_test, xs):
    k, K, al, be = p.k, p.K, p.alpha, p.beta; ar = p.alpha_r; r = p.r
    Fx = lambda x: af * np.cos(K * x) + bz * np.cos(Lam * K * x); Fpx = lambda x: -af * K * np.sin(K * x) - bz * Lam * K * np.sin(Lam * K * x)
    ua = lambda x, z: np.exp(1j * (al * x - be * z)) + (B[0][None, :] * np.exp(1j * (ar[None, :] * x[:, None] + p.bp[None, :] * z[:, None]))).sum(1)
    ub = lambda x, z: (B[1][None, :] * np.exp(1j * (ar[None, :] * x[:, None] - p.bm[None, :] * z[:, None]))).sum(1)
    z = Fx(xs); h = 1e-6
    dn = lambda u: (-Fpx(xs) * (u(xs + h, z) - u(xs - h, z)) / (2 * h) + (u(xs, z + h) - u(xs, z - h)) / (2 * h))
    r1 = np.abs(ua(xs, z) - ub(xs, z)).max() / np.abs(ua(xs, z)).max()
    r2 = np.abs(dn(ua) - nu2_test * dn(ub)).max() / (k * np.abs(ua(xs, z)).max())
    return r1, r2


rng = np.random.default_rng(1); xs = rng.uniform(0, 1, 400)
for pol in ("TE", "TM"):
    p = prob(pol, rmax=40); af, bz, Lam = 0.03, 0.1 / p.k, 5
    f, f1 = cos_surface(p, af); z, z1 = cos_surface(p, bz, Lam)
    B = rayleigh_exact(p, f + z, f1 + z1)
    r1, r2 = bc_residuals(p, B, af, bz, Lam, p.nu2, xs)
    check(f"{pol} continuity of the field", r1, 1e-6); check(f"{pol} continuity of nu^2-weighted normal derivative", r2, 1e-5)
    r1w, r2w = bc_residuals(p, B, af, bz, Lam, 1.0 if pol == "TM" else 1 / EPS, xs)      # wrong nu^2 (TE <-> TM)
    print(f"        sensitivity: with the wrong nu^2 the second residual is {r2w:.1e} (must be large)"); results.append(r2w > 1e-3)

# ------------------------------------------------------------------------------------------------------------------ T4
print("T4 recursion against the Cauchy integral of the full-profile Rayleigh solution in complex xi")
for pol, Lam, kb in (("TE", 5, 0.15), ("TM", 5, 0.15), ("TE", 7, 0.1), ("TM", 3, 0.2)):
    p = prob(pol, rmax=40); af = 0.03
    f, f1 = cos_surface(p, af); z, z1 = cos_surface(p, kb / p.k, Lam)
    dp, dm = mvb_coefficients(p, f, f1, z, z1, rayleigh_exact(p, f, f1), 5)
    M, rho = 32, 1.0; xis = rho * np.exp(2j * np.pi * np.arange(M) / M)
    Bs = [rayleigh_exact(p, f + w * z, f1 + w * z1) for w in xis]
    worst = 0.0
    for n in range(6):
        cp = sum(B[0] * w ** (-n) for B, w in zip(Bs, xis)) / M; cm = sum(B[1] * w ** (-n) for B, w in zip(Bs, xis)) / M
        worst = max(worst, np.abs(cp - dp[n]).max() / np.abs(cp).max(), np.abs(cm - dm[n]).max() / np.abs(cm).max())
    check(f"{pol} Lambda={Lam} kb={kb}: max over n=0..5 and all {len(p.r)} orders of the relative difference between d_n and the Cauchy coefficients", worst, 1e-8)

# ------------------------------------------------------------------------------------------------------------------ T5
print("T5 first order about a flat surface (hand derivation)")
for pol in ("TE", "TM"):
    p = prob(pol, rmax=30); Lam, b = 5, 0.003; K = p.K; nu2 = p.nu2
    f = np.zeros(p.N); z, z1 = cos_surface(p, b, Lam)
    B0 = rayleigh_exact(p, f, f); dp, dm = mvb_coefficients(p, f, f, z, z1, B0, 1)
    R, T = B0[0][p.rmax], B0[1][p.rmax]; be, al, bm0 = p.beta, p.alpha, p.bm[p.rmax]
    worst = 0.0; worst_te = 0.0
    for r in (Lam, -Lam):
        zr = b / 2                                           # Fourier coefficient of b cos(Lambda K x)
        rK = r * K                                           # r = +-Lambda
        # order-1 equations: B1+ - B1- = s1 ; beta+_r B1+ + nu2 beta-_r B1- = s2
        s1 = -1j * zr * (be * R + bm0 * T - be)
        s2 = -1j * zr * (be ** 2 * (R + 1) - nu2 * bm0 ** 2 * T) - 1j * al * rK * zr * (-R + nu2 * T - 1)
        # (the derivative psi' has Fourier coefficient i r K psi_r)
        A = np.array([[1, -1], [p.bp[p.rmax + r], nu2 * p.bm[p.rmax + r]]], complex)
        sol = np.linalg.solve(A, np.array([s1, s2]))
        worst = max(worst, abs(sol[0] - dp[1][p.rmax + r]) / abs(sol[0]), abs(sol[1] - dm[1][p.rmax + r]) / abs(sol[1]))
        if pol == "TE":
            cf = 2j * be * p.k ** 2 * (EPS - 1) * zr / ((be + bm0) * (p.bp[p.rmax + r] + p.bm[p.rmax + r]))
            worst_te = max(worst_te, abs(cf - dp[1][p.rmax + r]) / abs(cf))
    check(f"{pol} B1+_(+-Lambda), B1-_(+-Lambda) against the 2x2 hand derivation", worst, 1e-10)
    if pol == "TE": check("TE B1+ against Rice's closed form", worst_te, 1e-10)
    other = np.abs(np.delete(dp[1], [p.rmax + Lam, p.rmax - Lam])).max()
    check(f"{pol} first order has no other orders (flat reference)", other, 1e-12)

# ------------------------------------------------------------------------------------------------------------------ T6
print("T6 matrices of the code against the series of the paper, sum_s c_{s,q} (+-i beta_r)^s")
for pol in ("TE", "TM"):
    p = prob(pol, rmax=30); af = 0.03
    f, f1 = cos_surface(p, af); S = 90
    c = [np.fft.fft(f ** s / factorial(s)) / p.N for s in range(S)]               # c[s][q mod N]: Fourier coefficient of f^s/s!
    M = system_matrix(p, f, f1); n = len(p.r)
    Pp, Pm, Qp, Qm = M[:n, :n], -M[:n, n:], M[n:, :n], M[n:, n:] / p.nu2
    worst = 0.0
    for ip_, ir_ in ((p.rmax, p.rmax), (p.rmax + 3, p.rmax), (p.rmax, p.rmax + 5), (3, 40), (55, 7), (20, 21), (45, 10), (0, 60), (60, 0)):
        q = p.r[ip_] - p.r[ir_]; cs = np.array([c[s][q % p.N] for s in range(S)])
        sp = sum(cs[s] * (1j * p.bp[ir_]) ** s for s in range(S)); sm = sum(cs[s] * (-1j * p.bm[ir_]) ** s for s in range(S))
        ar_ = p.alpha_r[ir_]
        paper = (sp, sm, (p.bp[ir_] - ar_ * q * p.K / p.bp[ir_]) * sp, (p.bm[ir_] - ar_ * q * p.K / p.bm[ir_]) * sm)
        code = (Pp[ip_, ir_], Pm[ip_, ir_], Qp[ip_, ir_], Qm[ip_, ir_])
        for x, y in zip(paper, code): worst = max(worst, abs(x - y) / max(abs(x), abs(y), 1e-30) if max(abs(x), abs(y)) > 1e-12 * np.abs(Pp).max() else 0)
    check(f"{pol} entries (P+, P-, Q+, Q-) of nine (r', r) pairs, s <= {S - 1}: max relative difference", worst, 1e-8)

# ------------------------------------------------------------------------------------------------------------------ T8
print("T8 the printed equations of Sec. 3 (Omega, Upsilon) evaluated literally with the coefficients of the code")
for pol, Lam in (("TE", 4), ("TM", 4), ("TM", 3)):
    p = prob(pol, rmax=30); af, kb = 0.01, 0.2; K = p.K; nu2 = p.nu2; S = 40; N = p.N
    f, f1 = cos_surface(p, af); z, z1 = cos_surface(p, kb / p.k, Lam)
    dp, dm = mvb_coefficients(p, f, f1, z, z1, rayleigh_exact(p, f, f1), 2)
    cache_c = {}
    def cc(s_, q):
        if s_ not in cache_c: cache_c[s_] = np.fft.fft(f ** s_ / factorial(s_)) / N
        return cache_c[s_][q % N]
    cache_g = {}
    def Gam(m, qp):
        if m not in cache_g: cache_g[m] = np.fft.fft(z ** m / factorial(m)) / N
        return cache_g[m][(qp * Lam) % N]
    al, be = p.alpha, p.beta
    for n in (1, 2):
        worst1 = worst2 = scale1 = scale2 = 0.0
        for rp in range(-6, 7):
            tot1 = tot2 = 0j
            for s_ in range(S):
                for ir, r in enumerate(p.r):
                    q_list = range(-n, n + 1)
                    for qp in q_list:
                        tau = rp - r - Lam * qp; cs = cc(s_, tau)
                        if abs(cs) < 1e-300: continue
                        bp_, bm_, ar_ = p.bp[ir], p.bm[ir], p.alpha_r[ir]
                        om = 0j; up = 0j
                        if qp == 0:
                            om += dp[n][ir] * (1j * bp_) ** s_ - dm[n][ir] * (-1j * bm_) ** s_
                            up += dp[n][ir] * (bp_ * (1j * bp_) ** s_ - ar_ / (1j * bp_) * (1j * bp_) ** s_ * 1j * tau * K) \
                                  + nu2 * dm[n][ir] * (bm_ * (-1j * bm_) ** s_ - ar_ / (1j * bm_) * (-1j * bm_) ** s_ * 1j * tau * K)
                        for pp in range(n):
                            m = n - pp; g = Gam(m, qp)
                            om += g * (dp[pp][ir] * (1j * bp_) ** (s_ + m) - dm[pp][ir] * (-1j * bm_) ** (s_ + m))
                            qq = (tau + Lam * qp) * K
                            up += g * (dp[pp][ir] * (bp_ * (1j * bp_) ** (s_ + m) - ar_ * (1j * bp_) ** (s_ + m - 1) * 1j * qq)
                                       + nu2 * dm[pp][ir] * (bm_ * (-1j * bm_) ** (s_ + m) + ar_ * (-1j * bm_) ** (s_ + m - 1) * 1j * qq))
                        if r == 0:
                            g = Gam(n, qp); qq = (tau + Lam * qp) * K
                            om += g * (-1j * be) ** (s_ + n)
                            up -= g * (-1j * be) ** (s_ + n) * (be - al / (1j * be) * 1j * qq)
                        tot1 += cs * om; tot2 += cs * up; scale1 = max(scale1, abs(cs * om)); scale2 = max(scale2, abs(cs * up))
            worst1 = max(worst1, abs(tot1)); worst2 = max(worst2, abs(tot2))
        check(f"{pol} Lambda={Lam}, n={n}: residual of Eq. (Omega) and of Eq. (Upsilon) relative to the largest term", max(worst1 / scale1, worst2 / scale2), 1e-9, f"(Omega {worst1/scale1:.1e}, Upsilon {worst2/scale2:.1e})")

# ------------------------------------------------------------------------------------------------------------------ T9
print("T9 grid convergence and reuse of the factorization (base case of the paper: Lambda = 15, kb = 0.25, r_max = 57)")
for pol in ("TE", "TM"):
    out = {}
    for N in (4096, 16384):
        p = Problem(1.0, WL, np.arcsin(0.35), EPS, pol, 57, N=N)
        f, f1 = cos_surface(p, 0.03); z, z1 = cos_surface(p, 0.25 / p.k, 15)
        out[N] = mvb_coefficients(p, f, f1, z, z1, rayleigh_exact(p, f, f1), 3)
        if N == 4096:
            reuse = mvb_coefficients(p, f, f1, z, z1, rayleigh_exact(p, f, f1), 3, smooth=SmoothOperator(p, f, f1))
            check(f"{pol} SmoothOperator reuse against recomputation (max over n <= 3)", max(np.abs(reuse[0][n] - out[N][0][n]).max() / np.abs(out[N][0][n]).max() for n in range(4)), 1e-12)
    pm = Problem(1.0, WL, np.arcsin(0.35), EPS, pol, 57, N=4096).prop_p
    d = max(np.abs(out[4096][0][n] - out[16384][0][n])[pm].max() / np.abs(out[16384][0][n])[pm].max() for n in range(4))
    check(f"{pol} N = 4096 against 16384, propagating reflected orders, n <= 3", d, 1e-8, "(cond of the matrix ~1e11)")

# ------------------------------------------------------------------------------------------------------------------ T7
print("T7 KA tends to the exact solution when the surface becomes smooth")
for pol in ("TE", "TM"):
    errs = []
    for af in (0.0025, 0.005, 0.01, 0.02):
        p = prob(pol, rmax=30); f, f1 = cos_surface(p, af)
        Be, Bk = rayleigh_exact(p, f, f1), ka_amplitudes(p, f, f1); m = p.prop_p
        errs.append(np.abs(Bk[0] - Be[0])[m].max() / np.abs(Be[0])[m].max())
    sl = np.polyfit(np.log([0.0025, 0.005, 0.01, 0.02]), np.log(errs), 1)[0]
    print(f"      {pol}: max relative amplitude difference KA vs exact for a/L = 0.0025 ... 0.02: " + " ".join(f"{e:.1e}" for e in errs) + f"; log-log slope {sl:.2f}")
    results.append(errs[0] < errs[-1] and errs[0] < 1e-2)

print(f"\n{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
