"""
Is the 2-norm condition number of the (unscaled) Rayleigh matrix of the smooth profile a meaningful stability criterion?  (Sec. 4, stability.)

The columns of the matrix M of Sec. 3 contain exp(+i beta+_r f) and exp(-i beta-_r f), which grow like exp(|beta_r| f) for the evanescent orders, so cond_2(M) can be
huge for a perfectly well-posed problem.  For every distinct smooth-profile matrix of the parametric study (configurations of robustness.py without grazing incidence, aK = 0.38 included, and the Lambda-kb grid
of phase_map.py; r_max as in the other scripts) with cond_2(M) above 1e12, and for the base case, this script prints
  cond2        : 2-norm condition number of M
  cond2_eq     : after Ruiz equilibration (rows and columns scaled to unit infinity norm, 30 sweeps)
  Skeel        : componentwise condition number  || |M^-1||M||x| ||_inf / ||x||_inf  of the zeroth-order solution x  (invariant under row/column scaling)
  Skeel_prop   : the same restricted to the propagating reflected orders (the ones that enter the efficiencies)
  bw           : componentwise relative backward error  max_i |M x - b|_i / (|M||x| + |b|)_i  of the solution found by LU (relative residual of the zeroth-order boundary conditions)
  dE           : change of the reflected efficiencies (L1 relative to their sum) when every entry of M is perturbed by a random relative error of 1e-13 (median of 3 trials)
usage: python conditioning_diagnostics.py      (~2 min)
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np
import robustness as R, phase_map as P
from scattering import *
import scattering as S

rng = np.random.default_rng(7)


def ruiz(A, it=30):
    n = A.shape[0]; r = np.ones(n); c = np.ones(n); B = A.copy()
    for _ in range(it):
        rs = np.sqrt(np.abs(B).max(axis=1)); B = B / rs[:, None]; r *= rs
        cs = np.sqrt(np.abs(B).max(axis=0)); B = B / cs[None, :]; c *= cs
    return B


def diagnose(p, f, f1):
    M = system_matrix(p, f, f1); b = S._incident_rhs_exact(p, f, f1); n = len(p.r)
    c2 = np.linalg.cond(M); c2e = np.linalg.cond(ruiz(M))
    x = np.linalg.solve(M, b); Mi = np.linalg.inv(M)
    sk = (np.abs(Mi) @ (np.abs(M) @ np.abs(x))); skeel = sk.max() / np.abs(x).max()
    prop = np.zeros(2 * n, bool); prop[:n] = p.prop_p
    skeel_p = sk[prop].max() / np.abs(x[prop]).max()
    res = np.abs(M @ x - b); bw = (res / (np.abs(M) @ np.abs(x) + np.abs(b))).max()
    e0 = efficiencies(p, x[:n], x[n:])[0]; ds = []
    for _ in range(3):
        Mp = M * (1 + 1e-13 * (rng.standard_normal(M.shape) + 1j * rng.standard_normal(M.shape)))
        xp = np.linalg.solve(Mp, b); ep = efficiencies(p, xp[:n], xp[n:])[0]; ds.append(np.abs(ep - e0)[p.prop_p].sum() / e0[p.prop_p].sum())
    return c2, c2e, skeel, skeel_p, bw, np.median(ds)


def cases():
    seen = set()
    d = np.load(R.FILE)["data"]
    for ic, name in enumerate(R.NAMES):
        if name in ("sin=0.75", "sin=0.85"): continue
        c = R.CONF[name]; Lr = c["L"]
        for iL, Lam in enumerate(R.LAMS[Lr]):
            for ip, pol in enumerate(R.POLS):
                cond = d[ic, ip, iL, 0][6]
                key = (name, Lam, pol)
                if (cond > 1e12 or (name == "base" and Lam == 15)) and key not in seen:
                    seen.add(key); yield name, Lam, pol, dict(wl=1.0 / Lr, sin=c["sin"], eps=c["eps"], a=c["a"], Lr=Lr), cond
    pm = np.load(P.FILE)["data"]
    for ip, pol in enumerate(P.POLS):
        for iL, Lam in enumerate(P.LAMS):
            cond = pm[ip, iL, 0][4]
            if cond > 1e12: yield "grid20", Lam, pol, dict(wl=P.WLr, sin=np.sin(np.deg2rad(P.TH)), eps=P.EPS, a=P.A_L, Lr=1 / P.WLr), cond


if __name__ == "__main__":
    allv = []
    print(f"{'config':>8} {'Lam':>3} {'pol':>3} {'rmax':>4} | {'cond2':>8} {'cond2_eq':>8} {'Skeel':>8} {'Skeel_prop':>10} {'bw':>8} {'dE':>8}")
    for name, Lam, pol, c, _ in cases():
        th = np.arcsin(c["sin"])
        p0 = Problem(1.0, c["wl"], th, c["eps"], pol, 30, N=4096)
        rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30, 3 * Lam + 12)
        p = Problem(1.0, c["wl"], th, c["eps"], pol, rmax, N=4096)
        f, f1 = cos_surface(p, c["a"])
        v = diagnose(p, f, f1)
        print(f"{name:>8} {Lam:3d} {pol:>3} {rmax:4d} | {v[0]:8.1e} {v[1]:8.1e} {v[2]:8.1e} {v[3]:10.1e} {v[4]:8.1e} {v[5]:8.1e}", flush=True); allv.append(v)
    a = np.array(allv); big = a[:, 0] > 1e12
    print(f"\nmatrices with cond_2 > 1e12: {big.sum()} (max {a[:, 0].max():.1e}); equilibration lowers cond_2 by a factor {np.min(a[big, 0] / a[big, 1]):.0f} to {np.max(a[big, 0] / a[big, 1]):.0f}")
    print(f"  Skeel condition number of the propagating orders <= 35 in {int(np.sum(a[big, 3] <= 35))}; backward error <= 4e-10 in {int(np.sum(a[big, 4] <= 4e-10))} (cond_2 up to {a[big & (a[:, 4] <= 4e-10), 0].max():.1e}); perturbation of 1e-13: efficiencies change by at most {a[big & (a[:, 4] <= 4e-10), 5].max():.1e}")
