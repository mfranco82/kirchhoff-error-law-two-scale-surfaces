"""
Kirchhoff amplitudes of the evanescent orders in the zeroth order of the series (Sec. 3 and Sec. 4, stability).

KA is a far-field approximation, but Eqs. (11)-(12) can be evaluated at imaginary beta_r.  For the profile z = a cos(Kx) + b cos(Lambda K x)
(L = 10 lambda, Lambda = 5, kb = 0.25, sin(theta_i) = 0.35, eps = 2.25, TE) this script compares, for aK = 0.19 and 0.38 and for several r_max, the error (L1, reflected
efficiencies, against the C-method with M = r_max) of
  * KA on the full profile,
  * the smooth profile alone (exact zeroth order, n = 0: roughness ignored),
  * the series n <= 3 with the Kirchhoff zeroth order with the evanescent amplitudes KEPT,
  * the series n <= 3 with the Kirchhoff zeroth order of the propagating orders only (evanescent amplitudes set to zero: 'KA zeroth order' of the paper),
  * the series n <= 3 with the exact zeroth order,
together with the condition number of the matrix of the smooth profile.  It also prints, for the base case (Lambda = 15, a/L = 0.03), the same errors for TE and TM.
usage: python evanescent_zeroth_order.py          (~1 min)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scattering.py, cmethod.py live in the parent folder
import numpy as np
from scattering import *
from cmethod import c_method, spectral_derivative

WL, SIN, EPS, KB = 0.1, 0.35, 2.25, 0.25
def l1(e, er, m): return np.abs(e - er)[m].sum() / er[m].sum()


def errors(pol, lam, a, rmax):
    p = Problem(1.0, WL, np.arcsin(SIN), EPS, pol, rmax, N=4096)
    f, f1 = cos_surface(p, a); z, z1 = cos_surface(p, KB / p.k, lam); F, F1 = f + z, f1 + z1
    ref = efficiencies(p, *[np.nan_to_num(x) for x in c_method(p, F, F1, spectral_derivative(p, F, 2), M=rmax, delta=1e-8)])[0]
    err = lambda B: 100 * l1(efficiencies(p, *B)[0], ref, p.prop_p)
    op = SmoothOperator(p, f, f1); Bex = rayleigh_exact(p, f, f1)
    out = dict(KA_full=err(ka_amplitudes(p, F, F1)), smooth_alone=err(Bex))
    for lab, B0 in (("series_KA0_evanescent_kept", ka_amplitudes(p, f, f1)), ("series_KA0_propagating", ka_amplitudes_propagating(p, f, f1)), ("series_exact0", Bex)):
        dp, dm = mvb_coefficients(p, f, f1, z, z1, B0, 3, smooth=op); out[lab] = err(mvb_sum(dp, dm, 1, 3))
    out["cond"] = np.linalg.cond(system_matrix(p, f, f1))
    return out


if __name__ == "__main__":
    print("errors in %, TE, Lambda=5, kb=0.25 (rows: aK, r_max)")
    for a in (0.03, 0.06):
        for rmax in (30, 45, 60):
            o = errors("TE", 5, a, rmax)
            print(f"  aK={2 * np.pi * a:.2f} r_max={rmax}: " + ", ".join(f"{k} {v:.3g}" for k, v in o.items()))
    print("base case (Lambda=15, kb=0.25, a/L=0.03, r_max=57)")
    for pol in ("TE", "TM"):
        o = errors(pol, 15, 0.03, 57)
        print(f"  {pol}: " + ", ".join(f"{k} {v:.3g}" for k, v in o.items()))
    # first-order coefficients d+_1 of the base case (exact zeroth order, TE): norm over the evanescent and over the propagating orders (Sec. 5, first order and split of the spectrum)
    p = Problem(1.0, WL, np.arcsin(SIN), EPS, "TE", 57, N=4096)
    f, f1 = cos_surface(p, 0.03); z, z1 = cos_surface(p, KB / p.k, 15)
    dp, dm = mvb_coefficients(p, f, f1, z, z1, rayleigh_exact(p, f, f1), 2, smooth=SmoothOperator(p, f, f1)); d1 = np.array(dp[1])
    print(f"base case, TE, |d+_1|: evanescent orders {np.linalg.norm(d1[~p.prop_p]):.2f}, propagating orders {np.linalg.norm(d1[p.prop_p]):.2f}")
