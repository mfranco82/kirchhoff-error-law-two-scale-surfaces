"""
First-order small-slope approximation (SSA-1, Voronovich) for a 1D periodic dielectric surface z = h(x),
in the conventions of scattering.py:

  incident    exp(i(alpha x - beta z))
  reflected   sum_r A_r exp(i(alpha_r x + beta+_r z))
  transmitted sum_r T_r exp(i(alpha_r x - beta-_r z))
  BC on z = h:  u+ + phi_i - u- = 0 ;  d_n(u+ + phi_i) - nu^2 d_n u- = 0   (nu^2 = 1 TE, 1/eps TM)

First-order small-perturbation (SPM-1) kernels, from the boundary conditions expanded to first order in h:
  A^(1)_r = B+(alpha_r) h_r ,  T^(1)_r = B-(alpha_r) h_r ,  h_r = (1/L) int h exp(-i r K x) dx
SSA-1 (deterministic, periodic):
  A_r = B+(alpha_r) / (-i Q+_r) * (1/L) int exp(-i r K x) exp(-i Q+_r h(x)) dx ,   Q+_r = beta+_r + beta
  T_r = B-(alpha_r) / (-i Q-_r) * (1/L) int exp(-i r K x) exp(-i Q-_r h(x)) dx ,   Q-_r = beta  - beta-_r
It reduces to SPM-1 for small h and is exact for a flat surface displaced by a constant (B+(alpha) = -i Q+_0 R,
B-(alpha) = -i Q-_0 T, checked in validation/ssa_checks.py).
"""
import numpy as np


def fresnel(prob):
    """Flat-interface reflection and transmission coefficients (R, T = 1 + R) for the incident wave."""
    be, ga, nu2 = prob.beta, np.sqrt(prob.eps * prob.k ** 2 - prob.alpha ** 2 + 0j), prob.nu2
    R = (be - nu2 * ga) / (be + nu2 * ga)
    return R, 1 + R, ga


def spm1_kernels(prob):
    """SPM-1 kernels B+(alpha_r), B-(alpha_r) for all retained orders r."""
    R, T, ga = fresnel(prob)
    be, al, nu2, K = prob.beta, prob.alpha, prob.nu2, prob.K
    bp, bm, r = prob.bp, prob.bm, prob.r
    S1 = -1j * (be * R - be + ga * T)                                     # BC1:  a_r - t_r = S1 h_r
    S2 = -(-be ** 2 * (1 + R) + nu2 * ga ** 2 * T + r * K * al * (1 + R - nu2 * T))   # BC2
    Bp = (S2 + 1j * nu2 * bm * S1) / (1j * (bp + nu2 * bm))
    Bm = Bp - S1
    return Bp, Bm


def ssa1_amplitudes(prob, h):
    """SSA-1 reflected and transmitted amplitudes A_r, T_r of the surface h (values on prob.x)."""
    Bp, Bm = spm1_kernels(prob)
    Qp = prob.bp + prob.beta
    Qm = prob.beta - prob.bm
    ph = np.exp(-1j * prob.r[None, :] * prob.K * prob.x[:, None])
    Ip = (ph * np.exp(-1j * Qp[None, :] * h[:, None])).mean(axis=0)
    Im = (ph * np.exp(-1j * Qm[None, :] * h[:, None])).mean(axis=0)
    return Bp / (-1j * Qp) * Ip, Bm / (-1j * Qm) * Im


def ssa1_propagating(prob, h):
    """SSA-1 amplitudes of the propagating orders only (the efficiencies use only these)."""
    A, T = ssa1_amplitudes(prob, h)
    return np.where(prob.prop_p, A, 0), np.where(prob.prop_m, T, 0)
