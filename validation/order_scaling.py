"""
Order-by-order checks of the recursion.

1. Scaling: with the exact zeroth order the error of the partial sum up to order n against the exact (Rayleigh, same truncation) solution
   of the full profile must scale as b^(n+1) or faster (parity can raise the slope of the efficiencies).  Printed: log-log slopes of the error versus kb (kb = 0.02 ... 0.16) for n = 1..5,
   for a = 0.03 L (smooth profile present) and for a = 0 (flat base: the classical method of variation of boundaries), TE and TM.
2. Convergence with n at larger kb (Lambda = 8, lambda_s = 1.25 lambda), against the C-method: error of the partial sums n = 0..8 for
   kb = 0.25, 0.5, 0.7, 1.0, with the exact and the Kirchhoff zeroth order.
usage: python order_scaling.py            (~1 min)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scattering.py, cmethod.py live in the parent folder
import numpy as np
from scattering import *
from cmethod import c_method, spectral_derivative

WL, SIN, EPS = 0.1, 0.35, 2.25


def l1(e, er, m): return np.abs(e - er)[m].sum() / er[m].sum()


print("1. slopes d log(error) / d log(kb), exact zeroth order, kb = 0.02, 0.04, 0.08, 0.16; expected n+1 = 2 ... 6")
for a in (0.03, 0.0):
    for pol in ("TE", "TM"):
        for Lam in (5, 10):
            rmax = 5 * Lam + 12
            p = Problem(1.0, WL, np.arcsin(SIN), EPS, pol, rmax, N=4096)
            f, f1 = cos_surface(p, a); op = SmoothOperator(p, f, f1); B0 = rayleigh_exact(p, f, f1)
            kbs = np.array([0.02, 0.04, 0.08, 0.16]); errs = np.zeros((len(kbs), 5))
            for i, kb in enumerate(kbs):
                z, z1 = cos_surface(p, kb / p.k, Lam)
                ref = efficiencies(p, *rayleigh_exact(p, f + z, f1 + z1))
                dp, dm = mvb_coefficients(p, f, f1, z, z1, B0, 5, smooth=op)
                for n in range(1, 6): errs[i, n - 1] = l1(efficiencies(p, *mvb_sum(dp, dm, 1, n))[0], ref[0], p.prop_p)
            sl = [np.polyfit(np.log(kbs), np.log(errs[:, n]), 1)[0] for n in range(5)]
            print(f"   a/L={a:4.2f} {pol} Lambda={Lam:2d}: slopes n=1..5 = {np.round(sl, 2)}")

print("\n2. convergence with n, Lambda = 8, TE: L1 error (%) of the partial sum n = 0..8 against the C-method")
Lam = 8; rmax = 5 * Lam + 12
for pol in ("TE", "TM"):
    p = Problem(1.0, WL, np.arcsin(SIN), EPS, pol, rmax, N=4096)
    f, f1 = cos_surface(p, 0.03); op = SmoothOperator(p, f, f1)
    print(f"   {pol}: r_max={rmax}, cond={np.linalg.cond(system_matrix(p, f, f1)):.1e}")
    for kb in (0.25, 0.5, 0.7, 1.0):
        z, z1 = cos_surface(p, kb / p.k, Lam); F, F1 = f + z, f1 + z1
        ref = efficiencies(p, *[np.nan_to_num(x) for x in c_method(p, F, F1, spectral_derivative(p, F, 2), M=rmax, delta=1e-8)])
        for lab, B0 in (("exact zeroth", rayleigh_exact(p, f, f1)), ("KA zeroth   ", ka_amplitudes(p, f, f1))):
            dp, dm = mvb_coefficients(p, f, f1, z, z1, B0, 8, smooth=op)
            print(f"      kb={kb:4.2f} {lab}: " + " ".join(f"{100 * l1(efficiencies(p, *mvb_sum(dp, dm, 1, n))[0], ref[0], p.prop_p):8.3g}" for n in range(9)))
