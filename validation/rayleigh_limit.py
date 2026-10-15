"""
Is the range of the series set by the Rayleigh hypothesis of the FULL profile, or by the truncation of the series?  (Sec. 4 of the paper.)

With the exact zeroth order the coefficients d_n of the recursion are the Taylor coefficients in xi of the Rayleigh solution of the full profile f + xi*psi
(test_formulation.py, T4), so the partial sums of the series converge, at best, to that Rayleigh solution and the series cannot be more accurate than the full-profile
Rayleigh method.  This script separates the two sources of error for the 742 cosine profiles of Fig. 2 (the configurations of robustness.py without grazing incidence
and aK = 0.38, and the (Lambda, kb) grid of phase_map.py), TE and TM:
  * e_ray      : L1 error (Eq. (27), reflected orders) of the Rayleigh solution of the full profile against the C-method (M = r_max),
  * e_n(C)     : the same for the partial sum of order n of the series (exact zeroth order) against the C-method,
  * e_n(ray)   : the same for the partial sum of order n against the Rayleigh solution of the full profile: the truncation error of the series alone.
Printed: medians by group of aK + b Lambda K (the Rayleigh-hypothesis indicator of the paper, threshold 0.6) and by slope s = b Lambda K of the small scale at fixed group,
the fraction of profiles in which the series (n <= 3) is more accurate than the full-profile Rayleigh solution, and the convergence of e_n(ray) with n.
usage: python rayleigh_limit.py compute | summary      (compute: ~4 min on 6-8 cores; stores ../data/rayleigh_limit.npz)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import robustness as R, phase_map as P
import error_metrics as EM
from scattering import *
from cmethod import c_method, spectral_derivative

FILE = os.path.join(EM.HERE, "data", "rayleigh_limit.npz")
NMAX = 5
L1 = lambda a, b, m: np.abs(a - b)[m].sum() / b[m].sum()


def cell(args):
    kind, ic, ip, iL, ik = args
    if kind == "rob":
        c = R.CONF[R.NAMES[ic]]; Lrel = c["L"]; Lam = R.LAMS[Lrel][iL]; kb = R.KBS[ik]; pol = R.POLS[ip]; wl = 1.0 / Lrel; th = np.arcsin(c["sin"]); eps = c["eps"]; a = c["a"]
    else:
        pol = P.POLS[ip]; Lam = P.LAMS[iL]; kb = P.KBS[ik]; wl = P.WLr; th = np.deg2rad(P.TH); eps = P.EPS; a = P.A_L
    out = np.full(3 + 1 + 2 * NMAX, np.nan)     # aK, s, kb | e_ray | e_n(C), n = 1..NMAX | e_n(ray), n = 1..NMAX
    out[0], out[1], out[2] = 2 * np.pi * a, kb * Lam * wl, kb
    try:
        with np.errstate(all="ignore"):
            b = kb * wl / (2 * np.pi)
            p0 = Problem(1.0, wl, th, eps, pol, 30, N=4096)
            rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30, 3 * Lam + 12)
            p = Problem(1.0, wl, th, eps, pol, rmax, N=4096)
            f, f1 = cos_surface(p, a); z, z1 = cos_surface(p, b, Lam); F, F1 = f + z, f1 + z1
            m = p.prop_p
            ref = efficiencies(p, *[np.nan_to_num(x) for x in c_method(p, F, F1, spectral_derivative(p, F, 2), M=rmax, delta=1e-8)])[0]
            ray = efficiencies(p, *rayleigh_exact(p, F, F1))[0]
            out[3] = L1(ray, ref, m)
            op = SmoothOperator(p, f, f1)
            dp, dm = mvb_coefficients(p, f, f1, z, z1, rayleigh_exact(p, f, f1), NMAX, smooth=op)
            for n in range(1, NMAX + 1):
                e = efficiencies(p, *mvb_sum(dp, dm, 1, n))[0]
                out[3 + n] = L1(e, ref, m); out[3 + NMAX + n] = L1(e, ray, m)
    except Exception:
        pass
    return out


def compute():
    tasks = EM.kinds(); t0 = time.time(); res = []
    with Pool(8) as pool:
        for k, out in enumerate(pool.imap(cell, tasks, chunksize=2)):
            res.append(out)
            if k % 100 == 0: print(f"{k}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    np.savez(FILE, data=np.array(res), C=np.array([EM.cvalue(a) for a in tasks]), pol=np.array([(R.POLS if a[0] == "rob" else P.POLS)[a[2]] for a in tasks]))
    print(f"saved in {time.time()-t0:.0f} s; failed {int(np.isnan(np.array(res)[:, 3]).sum())}")


def pct(x, q): return np.percentile(x, q) if len(x) else np.nan


def summary():
    d = np.load(FILE); X = d["data"]; C = d["C"]; pol = d["pol"]; ok = np.isfinite(X).all(1); X, C, pol = X[ok], C[ok], pol[ok]
    aK, s, kb, eray = X[:, 0], X[:, 1], X[:, 2], X[:, 3]; eC = X[:, 4:4 + NMAX]; eR = X[:, 4 + NMAX:]
    ind = aK + s
    print(f"{len(X)} cosine cases (profile x polarization); indicator aK + b Lambda K: {(ind < 0.6).sum()} below 0.6, {(ind >= 0.6).sum()} above")
    for lab, mk in (("all", np.ones(len(X), bool)), ("kb <= 0.5", kb <= 0.5), ("C < 0.8", C < 0.8)):
        print(f"\n--- {lab}: median [p90] error (%), n <= 3, exact zeroth order")
        for gl, g in (("aK+s < 0.6 ", ind < 0.6), ("aK+s >= 0.6", ind >= 0.6)):
            m = mk & g
            print(f"  {gl}: n={m.sum():4d} | Rayleigh(full) vs C-method {100*np.median(eray[m]):8.3g} [{100*pct(eray[m],90):8.3g}] | series vs C-method {100*np.median(eC[m,2]):8.3g} [{100*pct(eC[m,2],90):8.3g}]"
                  f" | series vs Rayleigh(full) = truncation {100*np.median(eR[m,2]):8.3g} [{100*pct(eR[m,2],90):8.3g}] | series more accurate than Rayleigh(full) in {100*np.mean(eC[m,2] < eray[m]):.0f} %")
    print("\n--- by slope s of the small scale, at fixed Rayleigh indicator (median error %, n <= 3): series vs C-method | truncation (series vs Rayleigh full) | Rayleigh(full) vs C-method")
    for gl, g in (("aK+s < 0.6 ", ind < 0.6), ("aK+s >= 0.6", ind >= 0.6)):
        for lo, hi in ((0, 0.15), (0.15, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.9), (0.9, 1.4), (1.4, 3)):
            m = g & (s >= lo) & (s < hi)
            if m.sum() >= 3: print(f"  {gl} s {lo:4.2f}-{hi:4.2f}: n={m.sum():4d} | {100*np.median(eC[m,2]):9.3g} | {100*np.median(eR[m,2]):9.3g} | {100*np.median(eray[m]):9.3g}")
    print("\n--- convergence of the partial sums to the Rayleigh solution of the full profile: median e_n(ray) (%) for n = 1..5, and to the C-method, e_n(C)")
    for gl, g in (("aK+s < 0.6 ", ind < 0.6), ("aK+s >= 0.6", ind >= 0.6)):
        print(f"  {gl}: to Rayleigh(full): " + " ".join(f"{100*np.median(eR[g,n]):9.3g}" for n in range(NMAX)) + "   | to C-method: " + " ".join(f"{100*np.median(eC[g,n]):9.3g}" for n in range(NMAX)))
    m = (ind >= 0.6) & (C < 0.8)
    print(f"\n--- aK+s >= 0.6 and C < 0.8: n={m.sum()}; Rayleigh(full) vs C-method above 1e-3 in {100*np.mean(eray[m] > 1e-3):.0f} %, above 1e-2 in {100*np.mean(eray[m] > 1e-2):.0f} %; series (n<=3) vs C-method above 1e-2 in {100*np.mean(eC[m,2] > 1e-2):.0f} %; truncation alone above 1e-2 in {100*np.mean(eR[m,2] > 1e-2):.0f} %")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("compute", "summary"): sys.exit("usage: python rayleigh_limit.py compute | summary")
    {"compute": compute, "summary": summary}[sys.argv[1]]()
