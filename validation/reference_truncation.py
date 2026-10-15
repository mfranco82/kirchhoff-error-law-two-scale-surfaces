"""
How much does the truncation of the reference (C-method with M = r_max modes, the value used for all the figures) change the errors quoted in the paper?

For the 742 cosine profiles of Fig. 2 the C-method is run with M = r_max and with M = 2 r_max and the reflected-orders L1 errors (Eq. (27)) of KA on the full profile, of the series with the Kirchhoff
zeroth order (propagating orders) and with the exact zeroth order are computed against both.  Printed: the L1 distance between the two references and the relative change of each error.
usage: python reference_truncation.py compute | summary          (compute: ~6 min on 8 cores; stores ../data/reference_truncation.npz)
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

FILE = os.path.join(EM.HERE, "data", "reference_truncation.npz")
L1 = lambda a, b, m: np.abs(a - b)[m].sum() / b[m].sum()


def cell(args):
    kind, ic, ip, iL, ik = args
    if kind == "rob":
        name = R.NAMES[ic]; c = R.CONF[name]; Lrel = c["L"]; Lam = R.LAMS[Lrel][iL]; kb = R.KBS[ik]; pol = R.POLS[ip]; wl = 1.0 / Lrel; th = np.arcsin(c["sin"]); eps = c["eps"]; a = c["a"]
    else:
        pol = P.POLS[ip]; Lam = P.LAMS[iL]; kb = P.KBS[ik]; wl = P.WLr; th = np.deg2rad(P.TH); eps = P.EPS; a = P.A_L
    out = np.full(7, np.nan)
    try:
        with np.errstate(all="ignore"):
            b = kb * wl / (2 * np.pi)
            p0 = Problem(1.0, wl, th, eps, pol, 30, N=4096)
            rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30, 3 * Lam + 12)
            p = Problem(1.0, wl, th, eps, pol, rmax, N=4096); p2 = Problem(1.0, wl, th, eps, pol, 2 * rmax, N=8192)
            f, f1 = cos_surface(p, a); z, z1 = cos_surface(p, b, Lam); F, F1 = f + z, f1 + z1
            ref = efficiencies(p, *[np.nan_to_num(x) for x in c_method(p, F, F1, spectral_derivative(p, F, 2), M=rmax, delta=1e-8)])[0]
            g, g1 = cos_surface(p2, a); h, h1 = cos_surface(p2, b, Lam); G, G1 = g + h, g1 + h1
            ref2 = efficiencies(p2, *[np.nan_to_num(x) for x in c_method(p2, G, G1, spectral_derivative(p2, G, 2), M=2 * rmax, delta=1e-8)])[0]
            i = np.abs(p2.r) <= rmax; ref2 = ref2[i]; m = p.prop_p
            op = SmoothOperator(p, f, f1); Bex = rayleigh_exact(p, f, f1)
            dpk, dmk = mvb_coefficients(p, f, f1, z, z1, ka_amplitudes_propagating(p, f, f1), 3, smooth=op)
            dpx, dmx = mvb_coefficients(p, f, f1, z, z1, Bex, 3, smooth=op)
            e = [efficiencies(p, *B)[0] for B in (ka_amplitudes(p, F, F1), mvb_sum(dpk, dmk, 1, 3), mvb_sum(dpx, dmx, 1, 3))]
            out[0] = L1(ref, ref2, m)
            for j in range(3): out[1 + j] = L1(e[j], ref, m); out[4 + j] = L1(e[j], ref2, m)
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
    print(f"saved in {time.time()-t0:.0f} s; failed {int(np.isnan(np.array(res)[:, 0]).sum())}")


def summary():
    d = np.load(FILE); X = d["data"]; C = d["C"]; ok = np.isfinite(X).all(1); X = X[ok]; C = C[ok]
    print(f"{ok.sum()} profiles.  distance between the references M = r_max and M = 2 r_max (reflected L1): median {np.median(X[:, 0]):.1e}, 95th percentile {np.percentile(X[:, 0], 95):.1e}, max {X[:, 0].max():.1e}")
    names = ("KA, full profile", "series, KA zeroth", "series, exact zeroth")
    for j, nm in enumerate(names):
        e1, e2 = X[:, 1 + j], X[:, 4 + j]; ch = np.abs(e1 - e2) / e2
        print(f"  {nm:22s}: relative change of the error with the better reference: median {np.median(ch):.1e}, 95th percentile {np.percentile(ch, 95):.1e}, max {ch.max():.1e};  profiles with a change above 1 %: {np.sum(ch > 0.01)}, above 10 %: {np.sum(ch > 0.1)}")
    print("by bin of C (series, exact zeroth): median error with M = r_max, median relative change")
    for lo, hi in zip((0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6), (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 4)):
        m = (C >= lo) & (C < hi)
        if m.sum(): print(f"  C {lo:4.2f}-{hi:4.2f}: n={m.sum():3d}  e_exact {100*np.median(X[m, 3]):.3g} %  change {np.median(np.abs(X[m, 3] - X[m, 6]) / X[m, 6]):.1e} (max {np.max(np.abs(X[m, 3] - X[m, 6]) / X[m, 6]):.1e});  KA full change max {np.max(np.abs(X[m, 1] - X[m, 4]) / X[m, 4]):.1e}")


if __name__ == "__main__":
    {"compute": compute, "summary": summary}[sys.argv[1]]()
