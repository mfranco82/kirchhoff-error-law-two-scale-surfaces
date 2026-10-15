"""
How far can KA be used for a given error tolerance?  (Table 2 of the paper, Sec. 4.)

For the cosine profiles of Fig. 2 (the cases of robustness.npz and phase_map.npz, read through make_results_figures.gather) it prints, in the
same logarithmic bins of the KA validity parameter C as the figure, the median and the 90th percentile of the error of
  * KA applied to the full profile,
  * the series KA + MVB (n <= 3) with the Kirchhoff zeroth order and with the exact zeroth order,
and the value of C at which each model reaches a tolerance of 1, 2, 5 and 10 %: first upward crossing of the binned curve, log-log interpolation
between the geometric centres of the bins.  Where the crossing falls in the last bins (C > 0.8, 17 and 4 profiles per polarisation) the value is
only indicative.  A value 0 means that the model is already above the tolerance in the lowest bin (C = 0.02-0.05); the median of the series in TM stays at
about 1 % for 0.2 < C < 0.8, so the 1 % crossing is not sharp there.
The limits are computed for the cosine profiles with kb <= 0.5 (the range of the series claimed in the paper); python tolerance_Cmax.py --all-kb uses every profile (kb up to 1).  A limit that is
not reached in the range of the bins prints inf (">1.1" in the table).  A second block gives the limits in the slope s = kb lambda/lambda_s of the small scale (bins of s below).
usage: python tolerance_Cmax.py [--all-kb]          (reads the caches in ../data; a few seconds)
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # make_results_figures.py and the caches live in the parent folder
warnings.filterwarnings("ignore")
import numpy as np
import make_results_figures as M

BINS = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 4.0])
MID = np.sqrt(BINS[:-1] * BINS[1:])
TOLS = (1, 2, 5, 10)
KBMAX = np.inf if "--all-kb" in sys.argv else 0.5
BINS_S = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.9, 1.4, 3.0])
MID_S = np.sqrt(BINS_S[:-1] * BINS_S[1:])


def binned(pol, fam, C, e, pl, q):
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        s = (pol == pl) & (fam == "cos") & (C >= lo) & (C < hi)
        out.append(np.percentile(e[s], q) if s.sum() > 3 else np.nan)
    return np.array(out)


def binned_s(pol, fam, s, e, pl, q):
    out = []
    for lo, hi in zip(BINS_S[:-1], BINS_S[1:]):
        m = (pol == pl) & (fam == "cos") & (s >= lo) & (s < hi)
        out.append(np.percentile(e[m], q) if m.sum() > 3 else np.nan)
    return np.array(out)


def crossing(v, tol, mid=MID):
    if v[0] > tol: return 0.0
    for i in range(len(v) - 1):
        if np.isfinite(v[i]) and np.isfinite(v[i + 1]) and v[i] <= tol < v[i + 1]:
            t = (np.log(tol) - np.log(v[i])) / (np.log(v[i + 1]) - np.log(v[i]))
            return float(np.exp(np.log(mid[i]) + t * (np.log(mid[i + 1]) - np.log(mid[i]))))
    return np.inf


if __name__ == "__main__":
    pol, fam, X, n_excl, n_all = M.gather()
    keep = (fam != "cos") | (X[:, 10] <= KBMAX); pol, fam, X = pol[keep], fam[keep], X[keep]
    print(f"cosine profiles with kb <= {KBMAX:g}: {int(np.sum(fam == 'cos') // 2)} per polarisation")
    C, eK, eM, eX = X[:, 0], 100 * X[:, 2], 100 * X[:, 3], 100 * X[:, 4]
    s_ = X[:, 9]
    models = (("KA, full profile", eK), ("series, KA zeroth order", eM), ("series, exact zeroth order", eX), ("smooth profile alone", 100 * X[:, 7]))
    print("bins of C:", " ".join(f"{lo:g}-{hi:g}" for lo, hi in zip(BINS[:-1], BINS[1:])))
    res = {}
    for q, lab in ((50, "median"), (90, "90th percentile")):
        print(f"\nerror (%) per bin, {lab}")
        for pl in ("TE", "TM"):
            for name, e in models:
                v = binned(pol, fam, C, e, pl, q); res[(q, pl, name)] = v
                print(f"  {pl} {name:27s}", " ".join(f"{x:7.2f}" for x in v))
    for q, lab in ((50, "median"), (90, "90th percentile")):
        print(f"\nC at which the {lab} error reaches the tolerance (TE / TM)")
        print(f"{'tolerance':>10} | " + " | ".join(f"{n:>26s}" for n, _ in models) + " | gain in C, series (KA zeroth) / KA")
        for tol in TOLS:
            cells, cv = [], {}
            for name, _ in models:
                a = [crossing(res[(q, pl, name)], tol) for pl in ("TE", "TM")]; cv[name] = a
                cells.append(f"{a[0]:10.2f} / {a[1]:<10.2f}".rjust(26))
            g = [cv["series, KA zeroth order"][i] / cv["KA, full profile"][i] if cv["KA, full profile"][i] > 0 else np.nan for i in (0, 1)]
            g2 = [cv["series, exact zeroth order"][i] / cv["KA, full profile"][i] if cv["KA, full profile"][i] > 0 else np.nan for i in (0, 1)]
            print(f"{tol:9d}% | " + " | ".join(cells) + f" | {g[0]:.1f} / {g[1]:.1f}   (exact zeroth: {g2[0]:.1f} / {g2[1]:.1f}; limits unrounded {cv['KA, full profile'][0]:.3f} {cv['KA, full profile'][1]:.3f})")
    print("\nfor comparison, inversion of the fit of Eq. (29) (C alone, all profiles):", {pl: {t: round((t / 100 / a) ** (1 / b), 3) for t in (1, 5, 10, 30)}
          for pl, (a, b) in (("TE", (0.62, 1.43)), ("TM", (0.44, 1.27)))})
    print("\n=== limits in the slope s of the small scale (same profiles; bins of s: " + " ".join(f"{lo:g}-{hi:g}" for lo, hi in zip(BINS_S[:-1], BINS_S[1:])) + ")")
    for q, lab in ((50, "median"), (90, "90th percentile")):
        print(f"\ns at which the {lab} error reaches the tolerance (TE / TM)")
        print(f"{'tolerance':>10} | " + " | ".join(f"{n:>26s}" for n, _ in models[:3]))
        for tol in TOLS:
            cells = []
            for name, e in models[:3]:
                a = [crossing(binned_s(pol, fam, s_, e, pl, q), tol, MID_S) for pl in ("TE", "TM")]
                cells.append(f"{a[0]:10.2f} / {a[1]:<10.2f}".rjust(26))
            print(f"{tol:9d}% | " + " | ".join(cells))
