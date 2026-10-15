"""
Cost against the size of the problem (cost paragraph of Sec. 5 of the paper): KA applied to the full profile versus the Rayleigh method,
the C-method and the series (KA zeroth order, n <= 3; 'series' = zeroth order + factorization + recursion, with the Kirchhoff (KA0) or the exact (ex0) zeroth order), for L/lambda = 5 ... 163 (L/lambda = 10 is the base case; one core; run it on an otherwise idle machine).

Profile: z = a cos(Kx) + b cos(Lambda K x), a = 0.03 L, Lambda = 15, kb = 0.25, sin(theta_i) = 0.35, eps = 2.25, TE.  KA only needs the propagating
orders (the transmitted ones are the most numerous: |r| up to ~1.5 L/lambda), the exact solvers need the evanescent orders as well (r_max = highest
propagating transmitted order + 12); the series needs the factorization of the dense matrix of the smooth profile, and so costs more than the Rayleigh
solution of the full profile.  Values of L/lambda that put an order exactly at grazing (e.g. 20, 40, 80 for sin = 0.35) are avoided.
usage: python cost_scaling.py [L/lambda ...]      (default 5 10 21 41 83 163; about 2 min)
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"          # single-core timings (set before importing numpy)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scattering import *
from cmethod import c_method, spectral_derivative

SIN, EPS, LAM, KB, A = 0.35, 2.25, 15, 0.25, 0.03


def best(fn, rep):
    t = []
    for _ in range(rep):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return min(t)


if __name__ == "__main__":
    lrels = [int(a) for a in sys.argv[1:]] or [5, 10, 21, 41, 83, 163]
    print(f"{'L/lam':>6} {'orders(KA)':>10} {'orders(exact)':>13} {'N':>5} | {'KA':>8} {'Rayleigh':>9} {'C-method':>9} {'KA zeroth':>9} {'ex. zeroth':>9} {'factoriz.':>11} {'recursion':>11} {'series(KA0)':>9} {'series(ex0)':>9}   [s, 1 thread]", flush=True)
    for Lrel in lrels:
        wl = 1.0 / Lrel; th = np.arcsin(SIN)
        pbig = Problem(1.0, wl, th, EPS, "TE", int(3 * Lrel) + 40, N=2048)
        rtr = int(np.abs(pbig.r[pbig.prop_m]).max())                      # highest propagating transmitted order
        rmax = max(rtr + 12, 3 * LAM + 12)
        N = 4096                                                           # as in benchmark_cost.py
        while N < 6 * rmax: N *= 2
        p_ka = Problem(1.0, wl, th, EPS, "TE", rtr, N=N)
        p = Problem(1.0, wl, th, EPS, "TE", rmax, N=N)
        f, f1 = cos_surface(p, A); z, z1 = cos_surface(p, KB / p.k, LAM)
        F, F1 = f + z, f1 + z1; F2 = spectral_derivative(p, F, 2)
        fk, fk1 = cos_surface(p_ka, A); zk, zk1 = cos_surface(p_ka, KB / p_ka.k, LAM)
        rep = 3 if Lrel <= 21 else 1
        t_ka = best(lambda: ka_amplitudes(p_ka, fk + zk, fk1 + zk1), rep)
        t_ray = best(lambda: rayleigh_exact(p, F, F1), rep)
        t_c = best(lambda: c_method(p, F, F1, F2, M=rmax, delta=1e-8), rep)
        t_prep = best(lambda: SmoothOperator(p, f, f1), rep)
        op = SmoothOperator(p, f, f1); B0 = ka_amplitudes(p, f, f1)
        t_ser = best(lambda: mvb_coefficients(p, f, f1, z, z1, B0, 3, smooth=op), rep)
        t_ka0 = best(lambda: ka_amplitudes(p, f, f1), rep)                      # zeroth orders of the series (all the retained orders)
        t_ex0 = best(lambda: rayleigh_exact(p, f, f1), rep)
        print(f"{Lrel:6d} {2 * rtr + 1:10d} {2 * rmax + 1:13d} {N:5d} | {t_ka:8.3f} {t_ray:9.3f} {t_c:9.3f} {t_ka0:9.3f} {t_ex0:9.3f} {t_prep:11.3f} {t_ser:11.3f} {t_ka0 + t_prep + t_ser:9.3f} {t_ex0 + t_prep + t_ser:9.3f}", flush=True)
