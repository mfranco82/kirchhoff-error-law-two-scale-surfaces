"""
Other error metrics for the cosine profiles of Fig. 2 (Sec. 4): non-specular orders, transmitted orders and complex amplitudes (phase included).

Same 742 profiles as Fig. 2 (the configurations of robustness.py without grazing incidence and aK = 0.38, and the (Lambda, kb) grid of phase_map.py), reference: C-method
(its amplitudes and those of the models share the phase convention: the Rayleigh solution of the full profile agrees with the C-method amplitudes to 1e-5).
Models: KA on the full profile, the smooth profile alone (exact zeroth order, n = 0), and the series n <= 3 with the Kirchhoff zeroth order (propagating orders) and with the exact zeroth order.
Metrics (sums over the propagating orders):
  e+   : L1 error of the reflected efficiencies relative to their sum (Eq. (27) of the paper)      e-   : the same for the transmitted orders
  ens  : the same as e+ restricted to the non-specular reflected orders r != 0 (the roughness signature)
  A+   : complex-amplitude error  sqrt( sum w |B - B_ref|^2 / sum w |B_ref|^2 ),  w = beta_r/beta,  reflected orders      A- : transmitted orders (w = nu^2 beta^-_r/beta)
  Ans  : A+ restricted to r != 0         dphi : |arg(B_0/B_0^ref)| of the specular reflected amplitude (rad)
usage: python error_metrics.py compute | summary [--all-kb]     (compute: ~8 min on 8 cores; stores ../data/error_metrics.npz; summary uses the profiles with kb <= 0.5 like Table 2, --all-kb every profile)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Pool
import robustness as R, phase_map as P
from scattering import *
from cmethod import c_method, spectral_derivative

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE = os.path.join(HERE, "data", "error_metrics.npz")
EXCL = ("sin=0.75", "sin=0.85", "aK=0.38")
MODELS = ("KA", "smooth", "KA0", "ex0")
METRICS = ("e+", "e-", "ens", "A+", "A-", "Ans", "dphi")


def metrics(p, B, Bref):
    """The 7 metrics of the amplitudes B = (B+, B-) against Bref."""
    e, er = efficiencies(p, *B), efficiencies(p, *Bref)
    mp, mm = p.prop_p, p.prop_m; ns = mp & (p.r != 0)
    L1 = lambda a, b, m: np.abs(a - b)[m].sum() / b[m].sum()
    wp = (p.bp.real / p.beta); wm = p.nu2 * p.bm.real / p.beta
    A = lambda x, y, w, m: np.sqrt((w * np.abs(x - y) ** 2)[m].sum() / (w * np.abs(y) ** 2)[m].sum())
    r0 = p.rmax
    return [L1(e[0], er[0], mp), L1(e[1], er[1], mm), L1(e[0], er[0], ns), A(B[0], Bref[0], wp, mp), A(B[1], Bref[1], wm, mm), A(B[0], Bref[0], wp, ns), abs(np.angle(B[0][r0] / Bref[0][r0]))]


def cell(args):
    kind, ic, ip, iL, ik = args
    if kind == "rob":
        name = R.NAMES[ic]; c = R.CONF[name]; Lrel = c["L"]; Lam = R.LAMS[Lrel][iL]; kb = R.KBS[ik]; pol = R.POLS[ip]; wl = 1.0 / Lrel; th = np.arcsin(c["sin"]); eps = c["eps"]; a = c["a"]
    else:
        pol = P.POLS[ip]; Lam = P.LAMS[iL]; kb = P.KBS[ik]; wl = P.WLr; th = np.deg2rad(P.TH); eps = P.EPS; a = P.A_L
    out = np.full(len(MODELS) * len(METRICS), np.nan)
    try:
        with np.errstate(all="ignore"):
            b = kb * wl / (2 * np.pi)
            p0 = Problem(1.0, wl, th, eps, pol, 30, N=4096)
            rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30, 3 * Lam + 12)
            p = Problem(1.0, wl, th, eps, pol, rmax, N=4096)
            f, f1 = cos_surface(p, a); z, z1 = cos_surface(p, b, Lam); F, F1 = f + z, f1 + z1
            Bref = [np.nan_to_num(x) for x in c_method(p, F, F1, spectral_derivative(p, F, 2), M=rmax, delta=1e-8)]
            op = SmoothOperator(p, f, f1); Bex = rayleigh_exact(p, f, f1)
            dpk, dmk = mvb_coefficients(p, f, f1, z, z1, ka_amplitudes_propagating(p, f, f1), 3, smooth=op)
            dpx, dmx = mvb_coefficients(p, f, f1, z, z1, Bex, 3, smooth=op)
            for j, B in enumerate((ka_amplitudes(p, F, F1), Bex, mvb_sum(dpk, dmk, 1, 3), mvb_sum(dpx, dmx, 1, 3))):
                out[j * len(METRICS):(j + 1) * len(METRICS)] = metrics(p, B, Bref)
    except Exception:
        pass
    return args, out


def kinds():
    t = [("rob", ic, ip, iL, ik) for ic, nm in enumerate(R.NAMES) if nm not in EXCL for ip in range(2) for iL in range(7) for ik in range(len(R.KBS))]
    return t + [("grid", 0, ip, iL, ik) for ip in range(2) for iL in range(len(P.LAMS)) for ik in range(len(P.KBS))]


def cvalue(args):
    """Validity parameter C of the full profile (as in make_results_figures.gather) and the small-scale slope."""
    kind, ic, ip, iL, ik = args
    if kind == "rob":
        c = R.CONF[R.NAMES[ic]]; Lr = c["L"]; cs = np.sqrt(1 - c["sin"] ** 2) ** 3; Lam = R.LAMS[Lr][iL]; kb = R.KBS[ik]
        return np.pi * c["a"] / Lr / cs + np.pi * kb / (2 * np.pi * Lr) * Lam ** 2 / Lr / cs
    cs = np.cos(np.deg2rad(P.TH)) ** 3; Lam = P.LAMS[iL]; kb = P.KBS[ik]
    return np.pi * P.A_L * P.WLr / cs + np.pi * (kb * P.WLr / (2 * np.pi)) * Lam ** 2 * P.WLr / cs


def compute():
    tasks = kinds(); res = []; t0 = time.time()
    with Pool(8) as pool:
        for k, (a, out) in enumerate(pool.imap(cell, tasks, chunksize=4)):
            res.append(out)
            if k % 100 == 0: print(f"{k}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    pol = np.array([(R.POLS if a[0] == "rob" else P.POLS)[a[2]] for a in tasks]); C = np.array([cvalue(a) for a in tasks])
    np.savez(FILE, data=np.array(res), C=C, pol=pol)
    print(f"saved {FILE} in {time.time()-t0:.0f} s; failed {int(np.isnan(np.array(res)[:, 0]).sum())}")


BINS = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 4.0])


def summary():
    d = np.load(FILE); X = d["data"].reshape(len(d["C"]), len(MODELS), len(METRICS)); C = d["C"]; pol = d["pol"]
    kb = np.array([(R.KBS if a[0] == "rob" else P.KBS)[a[4]] for a in kinds()])
    ok = np.isfinite(X).all(axis=(1, 2)) & (kb <= (np.inf if "--all-kb" in sys.argv else 0.5)); print(f"{ok.sum()} of {len(C)} profiles (kb <= {np.inf if '--all-kb' in sys.argv else 0.5})")
    for met in ("e+", "ens", "A+", "Ans", "e-", "A-", "dphi"):
        j = METRICS.index(met); print(f"\nmedian {met} (% ; rad for dphi) by bin of C, TE | TM   [KA | smooth alone | series KA0 | series exact0]")
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            line = f"  C {lo:4.2f}-{hi:4.2f}: "
            for pl in ("TE", "TM"):
                m = ok & (pol == pl) & (C >= lo) & (C < hi)
                line += " ".join(f"{(1 if met == 'dphi' else 100)*np.median(X[m, k, j]):8.3g}" for k in range(len(MODELS))) + "  |  "
            print(line + f"n={m.sum()}")
    print("\nmedian ratio metric / e+ (same model and profile), 0.1<C<0.8: ens/e+, A+/e+, A-/e-")
    m = ok & (C >= 0.1) & (C < 0.8)
    for k, mod in enumerate(MODELS):
        print(f"  {mod:7s}: ens/e+ {np.median(X[m, k, 2] / X[m, k, 0]):6.2f}   A+/e+ {np.median(X[m, k, 3] / X[m, k, 0]):6.2f}   A-/e- {np.median(X[m, k, 4] / X[m, k, 1]):6.2f}")
    print("\nC at which the median reaches 5 % (first upward crossing of the binned curve, interpolated), TE / TM, for each metric and model (0 = above 5 % already in the lowest bin)")
    mid = np.sqrt(BINS[:-1] * BINS[1:])
    def cross(v, tol):
        if v[0] > tol: return 0.0
        for i in range(len(v) - 1):
            if np.isfinite(v[i]) and np.isfinite(v[i + 1]) and v[i] <= tol < v[i + 1]:
                t = (np.log(tol) - np.log(v[i])) / (np.log(v[i + 1]) - np.log(v[i])); return float(np.exp(np.log(mid[i]) + t * (np.log(mid[i + 1]) - np.log(mid[i]))))
        return np.inf
    for met in ("e+", "ens", "A+", "Ans", "e-", "A-"):
        j = METRICS.index(met); row = f"  {met:4s}: "
        for k, mod in enumerate(MODELS):
            if mod == "smooth": continue
            lims = []
            for pl in ("TE", "TM"):
                sel = [ok & (pol == pl) & (C >= lo) & (C < hi) for lo, hi in zip(BINS[:-1], BINS[1:])]   # bins with 3 profiles or fewer are left out, as in tolerance_Cmax.py
                v = np.array([np.median(X[s_, k, j]) if s_.sum() > 3 else np.nan for s_ in sel]); lims.append(cross(v, 0.05))
            row += f"{mod}: {lims[0]:.2f}/{lims[1]:.2f}   "
        print(row)


if __name__ == "__main__":
    {"compute": compute, "summary": summary}[sys.argv[1]]()
