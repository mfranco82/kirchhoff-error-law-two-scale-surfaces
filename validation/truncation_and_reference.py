"""
Truncation, accuracy of the references and energy balance for the base-case profile (TE and TM).

Profile: z = a cos(Kx) + b cos(Lambda K x), a = 0.03 L, Lambda = 15, kb = 0.25, L = 10 lambda, sin(theta_i) = 0.35, eps = 2.25.
Prints
  1. C-method: energy balance and convergence with the number of modes M (against M = 120);
  2. Rayleigh solution of the full profile against the C-method (M = 120) for r_max = 45, 57, 70;
  3. error of the series (n = 1..5, both zeroth orders) against the C-method (M = 120) for several r_max
     (the paper uses r_max = max(30, 3 Lambda + 12, ...) = 57; larger r_max makes the Rayleigh matrix ill conditioned);
  4. energy balance R + T - 1 of every model.
usage: python truncation_and_reference.py            (~1 min)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scattering.py, cmethod.py live in the parent folder
import numpy as np
from scattering import *
from cmethod import c_method, spectral_derivative

WL, SIN, EPS, A, LAM, KB = 0.1, 0.35, 2.25, 0.03, 15, 0.25
M_REF = 120


def l1(e, er, m): return np.abs(e - er)[m].sum() / er[m].sum()


def build(pol, rmax):
    p = Problem(1.0, WL, np.arcsin(SIN), EPS, pol, rmax, N=4096)
    f, f1 = cos_surface(p, A); z, z1 = cos_surface(p, KB / p.k, LAM)
    return p, f, f1, z, z1, f + z, f1 + z1


def cmethod_eff(pol, M):
    p, f, f1, z, z1, F, F1 = build(pol, M)
    return p, efficiencies(p, *[np.nan_to_num(x) for x in c_method(p, F, F1, spectral_derivative(p, F, 2), M=M, delta=1e-8)])


def on(p_small, p_big, e):
    """restrict the efficiencies of the big problem to the orders of the small one"""
    idx = np.where(np.isin(p_big.r, p_small.r))[0]
    return e[0][idx], e[1][idx]


for pol in ("TE", "TM"):
    print(f"==================== {pol}, Lambda={LAM}, kb={KB}")
    pref, eref = cmethod_eff(pol, M_REF)
    print(f"1. C-method: R+T-1 = {eref[0].sum() + eref[1].sum() - 1:.1e} (M={M_REF})")
    for M in (57, 80):
        p, e = cmethod_eff(pol, M); er = on(p, pref, eref)
        print(f"   M={M:3d}: R+T-1 = {e[0].sum() + e[1].sum() - 1:.1e}; L1 distance to M={M_REF}: reflected {l1(e[0], er[0], p.prop_p):.1e}, transmitted {l1(e[1], er[1], p.prop_m):.1e}")
    print("2. Rayleigh solution of the full profile against the C-method (M=120)")
    for rmax in (45, 57, 70):
        p, f, f1, z, z1, F, F1 = build(pol, rmax); e = efficiencies(p, *rayleigh_exact(p, F, F1)); er = on(p, pref, eref)
        print(f"   r_max={rmax}: reflected {l1(e[0], er[0], p.prop_p):.1e}, transmitted {l1(e[1], er[1], p.prop_m):.1e}, R+T-1 = {e[0].sum() + e[1].sum() - 1:.1e}")
    print("3. series error (%, reflected, L1) against the C-method (M=120);  columns n = 1..5;  cond = condition number of the smooth-profile matrix")
    for rmax in (30, 45, 57, 70, 90):
        p, f, f1, z, z1, F, F1 = build(pol, rmax); er = on(p, pref, eref); op = SmoothOperator(p, f, f1)
        line = f"   r_max={rmax:3d} cond={np.linalg.cond(system_matrix(p, f, f1)):.0e}"
        for lab, B0 in (("exact zeroth", rayleigh_exact(p, f, f1)), ("KA zeroth", ka_amplitudes(p, f, f1))):
            dp, dm = mvb_coefficients(p, f, f1, z, z1, B0, 5, smooth=op)
            line += f" | {lab}: " + " ".join(f"{100 * l1(efficiencies(p, *mvb_sum(dp, dm, 1, n))[0], er[0], p.prop_p):8.4g}" for n in range(1, 6))
        print(line)
    print("4. energy balance R+T-1 at r_max=57")
    p, f, f1, z, z1, F, F1 = build(pol, 57); op = SmoothOperator(p, f, f1)
    rt = lambda B: (lambda e: e[0].sum() + e[1].sum() - 1)(efficiencies(p, *B))
    print(f"   KA full profile {rt(ka_amplitudes(p, F, F1)):.1e} | KA smooth profile {rt(ka_amplitudes(p, f, f1)):.1e} | Rayleigh smooth {rt(rayleigh_exact(p, f, f1)):.1e} | Rayleigh full {rt(rayleigh_exact(p, F, F1)):.1e}")
    for lab, B0 in (("KA zeroth", ka_amplitudes(p, f, f1)), ("exact zeroth", rayleigh_exact(p, f, f1))):
        dp, dm = mvb_coefficients(p, f, f1, z, z1, B0, 4, smooth=op)
        print(f"   series, {lab}: " + "  ".join(f"n<={n}: {rt(mvb_sum(dp, dm, 1, n)):.1e}" for n in range(5)))
