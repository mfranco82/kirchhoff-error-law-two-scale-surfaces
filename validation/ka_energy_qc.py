"""
Quality control of KA without a rigorous reference: its own energy balance.  (Sec. 4 of the paper.)

Cosine profiles z = a cos(Kx) + b cos(Lambda K x), a = 0.03 L, L = 10 lambda, eps = 2.25, TE and TM,
Lambda = 3..25, kb = 0.05..1, sin(theta_i) = 0.25, 0.35, 0.55 (594 cases).  For each one: the validity parameter C of the full profile,
the error of KA applied to the full profile against the C-method (reflected orders, Eq. (27)) and the residuals R+T-1 of KA and of the C-method.
Cases for which the C-method itself violates energy by more than 1e-4 (all with a small-scale slope >= 0.8, outside the domain of the paper)
are discarded.  Then it prints: rank correlation and scatter of log(error) with C and with |R+T-1|, the ratio |R+T-1|/error per bin of C, and how
many of the profiles with an error above 5 / 10 % are flagged by a threshold on |R+T-1|.
usage: python ka_energy_qc.py [--cached]       (about 1 min on 8 cores; the results are stored in ../data/ka_energy_qc.npz)
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
import numpy as np
from multiprocessing import Pool
from scipy.stats import spearmanr
from scattering import *
from cmethod import c_method, spectral_derivative

WL, EPS, A = 0.1, 2.25, 0.03
LAMS = [3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 25]
KBS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
SINS = [0.25, 0.35, 0.55]          # none of them puts an order at grazing for L = 10 lambda (neither for the reflected nor for the transmitted field)
FILE = os.path.join(PARENT, "data", "ka_energy_qc.npz")


def job(a):
    sin, pol, Lam, kb = a
    th = np.arcsin(sin)
    p0 = Problem(1.0, WL, th, EPS, pol, 30, N=4096)
    rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30, 3 * Lam + 12)
    p = Problem(1.0, WL, th, EPS, pol, rmax, N=4096)
    f, f1 = cos_surface(p, A); z, z1 = cos_surface(p, kb / p.k, Lam)
    F, F1 = f + z, f1 + z1
    ref = efficiencies(p, *[np.nan_to_num(x) for x in c_method(p, F, F1, spectral_derivative(p, F, 2), M=rmax, delta=1e-8)])
    with np.errstate(all="ignore"):
        ka = efficiencies(p, *ka_amplitudes(p, F, F1))
    mp, mm = p.prop_p, p.prop_m
    eps_p = np.abs(ka[0] - ref[0])[mp].sum() / ref[0][mp].sum()
    eps_m = np.abs(ka[1] - ref[1])[mm].sum() / ref[1][mm].sum()
    cs = np.cos(th) ** 3
    C = np.pi * A * WL / cs + np.pi * (kb * WL / (2 * np.pi)) * Lam ** 2 * WL / cs
    return (sin, 0 if pol == "TE" else 1, Lam, kb, C, eps_p, eps_m, ka[0].sum() + ka[1].sum() - 1, ref[0].sum() + ref[1].sum() - 1)


def compute():
    tasks = [(s, pol, L, kb) for s in SINS for pol in ("TE", "TM") for L in LAMS for kb in KBS]
    with Pool(os.cpu_count()) as pool:
        D = np.array(pool.map(job, tasks, chunksize=4), float)
    os.makedirs(os.path.dirname(FILE), exist_ok=True)
    np.savez(FILE, D=D)
    return D


if __name__ == "__main__":
    D = np.load(FILE)["D"] if ("--cached" in sys.argv and os.path.exists(FILE)) else compute()
    sin, pol, Lam, kb, C, ep, em, rK, rR = D.T
    slope_small = kb * Lam * WL                      # slope of the small scale, b Lambda K
    bad = np.abs(rR) > 1e-4
    print(f"{len(D)} cases; discarded {bad.sum()} whose C-method reference violates energy by more than 1e-4 "
          f"(small-scale slope {slope_small[bad].min():.2f}-{slope_small[bad].max():.2f}; kept: <= {slope_small[~bad].max():.2f}); kept {(~bad).sum()}")
    ok = ~bad & (ep > 0)
    print("\nlog10(error of KA) against C and against |R+T-1| of KA (kept cases)")
    for name, sel in (("TE", pol == 0), ("TM", pol == 1), ("both", pol >= 0)):
        s = ok & sel
        lc, le, lr = np.log10(C[s]), np.log10(ep[s]), np.log10(np.abs(rK[s]))
        sC = np.std(le - np.polyval(np.polyfit(lc, le, 1), lc)); sE = np.std(le - np.polyval(np.polyfit(lr, le, 1), lr))
        X = np.c_[np.ones(s.sum()), lc, lr]; coef, *_ = np.linalg.lstsq(X, le, rcond=None); sB = np.std(le - X @ coef)
        print(f"  {name:4s} n={s.sum():3d}  Spearman with C {spearmanr(lc, le)[0]:.3f}, with |R+T-1| {spearmanr(lr, le)[0]:.3f} | scatter (decades): "
              f"C {sC:.2f}, |R+T-1| {sE:.2f}, both {sB:.2f}")
    print(f"  fraction of the KA residuals R+T-1 that are negative: {np.mean(rK[ok] < 0):.2f}")
    print("\nper bin of C (both polarisations): median error of KA | median |R+T-1| of KA | median ratio |R+T-1|/error")
    edges = [0.03, 0.1, 0.2, 0.4, 0.8, 1.6, 5]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = ok & (C >= lo) & (C < hi)
        if s.sum() > 3:
            print(f"  C {lo:4.2f}-{hi:4.2f}  n={s.sum():3d}  {100 * np.median(ep[s]):6.2f} % | {100 * np.median(np.abs(rK[s])):7.3f} % | {np.median(np.abs(rK[s]) / ep[s]):.3f}")
    for tol in (0.05, 0.10):
        badk = ep[ok] > tol
        print(f"\nprofiles with an error of KA above {100 * tol:.0f} %: {badk.sum()} of {ok.sum()}")
        for thr in (1e-3, 3e-3, 1e-2):
            pred = np.abs(rK[ok]) > thr
            print(f"  alarm |R+T-1| > {thr:g}: flags {(pred & badk).sum()} of them, {(pred & ~badk).sum()} false positives")
