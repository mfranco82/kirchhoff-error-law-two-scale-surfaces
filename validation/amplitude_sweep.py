"""
Cost of a sweep over the roughness amplitude (Sec. 5 of the paper).

The Taylor coefficients d_n are polynomials in the amplitude of the small scale, so after one factorization and one recursion (n <= 3) the
spectrum for any kb is a polynomial evaluation.  Printed for M = 10, 100, 1000 values of kb in [0.02, 0.4]: time of the series (set-up once +
evaluation), of M Rayleigh solutions and of M C-method solutions (the last two timed on a few amplitudes and multiplied by M), and the
difference between a swept evaluation and a direct recursion at kb = 0.2.  Base-case profile (TE, Lambda = 15, 115 orders), single thread.
usage: python amplitude_sweep.py          (forces single-thread BLAS; run it on an otherwise idle machine)
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"          # single-core timings (set before importing numpy)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scattering.py, cmethod.py live in the parent folder
import numpy as np
from scattering import *
from cmethod import c_method, spectral_derivative

LAM = 15
p = Problem(1.0, 0.1, np.arcsin(0.35), 2.25, "TE", 57, N=4096)
f, f1 = cos_surface(p, 0.03); z, z1 = cos_surface(p, 1.0 / p.k, LAM)          # roughness of unit kb: xi = kb
t0 = time.perf_counter()
op = SmoothOperator(p, f, f1); B0 = rayleigh_exact(p, f, f1); dp, dm = mvb_coefficients(p, f, f1, z, z1, B0, 3, smooth=op)
t_setup = time.perf_counter() - t0
print(f"base-case profile: {2 * p.rmax + 1} orders, N=4096; series set-up (factorization + zeroth order + recursion n<=3): {t_setup:.2f} s")


def best(fn, rep):
    t = []
    for _ in range(rep):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return min(t)


def ray(kb):
    zz, zz1 = cos_surface(p, kb / p.k, LAM); rayleigh_exact(p, f + zz, f1 + zz1)


def cm(kb):
    zz, zz1 = cos_surface(p, kb / p.k, LAM); F, F1 = f + zz, f1 + zz1
    c_method(p, F, F1, spectral_derivative(p, F, 2), M=p.rmax, delta=1e-8)


t_ray, t_c = best(lambda: ray(0.2), 3), best(lambda: cm(0.2), 2)
for M in (10, 100, 1000):
    kbs = np.linspace(0.02, 0.4, M)
    t0 = time.perf_counter()
    for kb in kbs: efficiencies(p, *mvb_sum(dp, dm, kb, 3))
    t_eval = time.perf_counter() - t0
    print(f"  M={M:5d}: series {t_setup + t_eval:6.2f} s (evaluation {t_eval:.3f} s) | Rayleigh {M * t_ray:7.1f} s | C-method {M * t_c:7.1f} s")
print(f"  break-even with Rayleigh at about {t_setup / t_ray:.0f} amplitudes, with the C-method at about {t_setup / t_c:.0f}")
kb = 0.2; zz, zz1 = cos_surface(p, kb / p.k, LAM)
a = efficiencies(p, *mvb_sum(dp, dm, kb, 3)); dpp, dmm = mvb_coefficients(p, f, f1, zz, zz1, B0, 3, smooth=op); b = efficiencies(p, *mvb_sum(dpp, dmm, 1, 3))
print(f"swept evaluation vs direct recursion at kb={kb}: max |difference of e_r| = {np.abs(a[0] - b[0]).max():.1e}")
